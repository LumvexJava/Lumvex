package dev.lumvex.texture;

import dev.lumvex.vk.Buffer;
import dev.lumvex.vk.ShaderCompiler;
import dev.lumvex.vk.VkContext;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.stream.IntStream;

import static dev.lumvex.texture.TextureSet.*;
import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
import static org.lwjgl.vulkan.VK13.*;

/**
 * Sparse virtual texturing for the ray tracer, with block compression and tile reuse.
 *
 * <p><b>Tiles.</b> Every texture set is split into 128x128 tiles per mip level. The GPU holds two fixed pools of
 * tile slots, so VRAM use is independent of the number of textures:
 * <pre>
 *   near pool (mips 0-1): albedo BC7 sRGB, normal BC5, ORM BC7    8 bpp
 *   far pool  (mips 2+):  albedo BC1 sRGB, normal BC5, ORM BC1    4 / 8 / 4 bpp
 * </pre>
 * Color depth thus drops with distance along with resolution. With {@code compress = false} everything is RGBA8,
 * for comparisons. Until a tile is resident, the shader falls back to the next coarser resident mip.
 *
 * <p><b>Feedback.</b> The shaders record, per tile, the smallest camera distance at which it was needed. The host
 * uploads missing tiles (coarse mips first, with a per-frame budget), evicts least recently used ones and drops
 * tiles that have not been needed for a while. Everything stays available in RAM.
 *
 * <p><b>Reuse.</b> Every tile has a small fingerprint that is compared with all tiles of the same mip level. The
 * farther away a tile is needed, the more a stand-in may differ, and a tile whose stand-in is good enough at its
 * distance is never uploaded. When the camera comes closer, the real tile is loaded and blended in over a distance
 * band to avoid popping.
 */
public final class VirtualTextures implements AutoCloseable {
    public static final int EVICT_AFTER_FRAMES = 240;
    public static final int NEAR = 0, FAR = 1, POOLS = 2;
    /** Mips below this level use the near pool. */
    public static final int FAR_MIP = 2;
    private static final int CANDIDATES = 4;

    private final VkContext ctx;
    public final List<TextureSet> sets;
    public final boolean compress;
    public final int slotsPerRow;
    public final int slotCount;
    public final int atlasSize;
    public int uploadBudget = 48;
    /** Scales the stand-in tolerance: 0 disables reuse, 2 is twice as tolerant as the default. */
    public float similarityScale = 1f;

    private final long[] atlasImage = new long[POOLS * CHANNELS];
    private final long[] atlasMemory = new long[POOLS * CHANNELS];
    public final long[] atlasView = new long[POOLS * CHANNELS];
    private final int[] atlasFormat = new int[POOLS * CHANNELS];
    private final BlockCompression.Encoder[] encoder = new BlockCompression.Encoder[POOLS * CHANNELS];
    public final long sampler;

    public final Buffer pageTable;   // uvec4 per tile: own entry, stand-in entry, stand-in distance, upload frame
    public final Buffer feedback;    // uint per tile: smallest distance in cm it was needed at, ~0 if not needed
    private final Buffer feedbackRead; // CPU copy of the last traced frame's feedback, in cached memory
    public final Buffer setInfo;     // uvec4(slotsPerRow, atlasSize, frame, 0), then uvec4(size, mips, firstTile, 0) per set
    /** A stand-in is kept while it stays within this factor of the tolerance, so small moves do not flip it. */
    private static final float STAND_IN_HYSTERESIS = 1.25f;
    private final Buffer staging;

    private final int totalTiles;
    private final int[] tileSet, tileMip, tileX, tileY;
    // Residency
    private final int[] tileSlot;                   // tile -> slot in its pool, or -1
    private final int[][] slotTile = new int[POOLS][];
    private final long[][] slotLastUsed = new long[POOLS][];
    private final boolean[][] slotPinned = new boolean[POOLS][];
    private long frame;
    // CPU block compression, on demand
    private final AtomicReferenceArray<ByteBuffer> encoded;   // per tile: all channels, compressed
    private final boolean[] encodePending;
    private final ExecutorService workers;
    private final int workerCount;
    private final AtomicLong encodeNanos = new AtomicLong(), encodeCount = new AtomicLong();
    private final ConcurrentLinkedQueue<ByteBuffer> freed = new ConcurrentLinkedQueue<>();
    // Similarity
    private volatile int[][] candTile;              // per tile: the most similar tiles of the same mip
    private volatile float[][] candDist;
    private final int[] aliasOf;                    // tile -> stand-in currently in the page table, or -1
    private volatile boolean similarityReady;
    private volatile String similarityStatus = "computing...";

    // Statistics of the last processed frame
    public int statRequested, statUploads, statEvictions, statMissing, statAliased, statEncodeQueue;
    public final int[] statResident = new int[POOLS];
    public long statTotalUploads, statUploadsSaved;

    /** Encode tiles on the GPU instead of on CPU worker threads. */
    public static boolean gpuEncode = true;
    private GpuTileEncoder gpuEncoder;

    /**
     * Sets up the tile directory, atlases and buffers. The coarsest mip of every set is uploaded right away and
     * pinned as a permanent fallback; the similarity search starts in the background.
     */
    public VirtualTextures(VkContext ctx, List<TextureSet> sets, int slotsPerRow, boolean compress, ShaderCompiler compiler) {
        this.ctx = ctx;
        this.sets = sets;
        this.compress = compress && ctx.textureCompressionBC;
        this.slotsPerRow = slotsPerRow;
        this.slotCount = slotsPerRow * slotsPerRow;
        this.atlasSize = slotsPerRow * SLOT;

        int n = 0;
        int[] firstTile = new int[sets.size()];
        for (int s = 0; s < sets.size(); s++) { firstTile[s] = n; n += sets.get(s).tileCount(); }
        totalTiles = Math.max(n, 1);
        tileSet = new int[totalTiles]; tileMip = new int[totalTiles]; tileX = new int[totalTiles]; tileY = new int[totalTiles];
        int id = 0;
        for (int s = 0; s < sets.size(); s++) {
            TextureSet set = sets.get(s);
            for (int m = 0; m < set.mipCount; m++) {
                int tps = set.tilesPerSide(m);
                for (int y = 0; y < tps; y++) for (int x = 0; x < tps; x++) {
                    tileSet[id] = s; tileMip[id] = m; tileX[id] = x; tileY[id] = y; id++;
                }
            }
        }
        tileSlot = new int[totalTiles];
        Arrays.fill(tileSlot, -1);
        aliasOf = new int[totalTiles];
        Arrays.fill(aliasOf, -1);
        encoded = new AtomicReferenceArray<>(totalTiles);
        encodePending = new boolean[totalTiles];
        for (int p = 0; p < POOLS; p++) {
            slotTile[p] = new int[slotCount];
            Arrays.fill(slotTile[p], -1);
            slotLastUsed[p] = new long[slotCount];
            slotPinned[p] = new boolean[slotCount];
        }

        int[] rawFmt = {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
        int[][] bcFmt = {{VK_FORMAT_BC7_SRGB_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC7_UNORM_BLOCK},
                         {VK_FORMAT_BC1_RGB_SRGB_BLOCK, VK_FORMAT_BC5_UNORM_BLOCK, VK_FORMAT_BC1_RGB_UNORM_BLOCK}};
        BlockCompression.Encoder[][] bcEnc = {{BlockCompression.BC7, BlockCompression.BC5, BlockCompression.BC7},
                                              {BlockCompression.BC1, BlockCompression.BC5, BlockCompression.BC1}};
        for (int p = 0; p < POOLS; p++) {
            for (int c = 0; c < CHANNELS; c++) {
                int a = p * CHANNELS + c;
                atlasFormat[a] = this.compress ? bcFmt[p][c] : rawFmt[c];
                encoder[a] = this.compress ? bcEnc[p][c] : null;
                createAtlas(a, atlasFormat[a]);
            }
        }
        try (MemoryStack stack = stackPush()) {
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateSampler(ctx.device, VkSamplerCreateInfo.calloc(stack)
                    .sType$Default()
                    .magFilter(VK_FILTER_LINEAR).minFilter(VK_FILTER_LINEAR)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .maxLod(0f), null, lp), "vkCreateSampler");
            sampler = lp.get(0);
        }
        pageTable = new Buffer(ctx, totalTiles * 16L, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        memSet(pageTable.map(), 0);
        // Every texture lookup updates the feedback with an atomic, so it lives in VRAM and the CPU reads a copy.
        feedback = new Buffer(ctx, totalTiles * 4L, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, Buffer.DEVICE);
        ctx.submitImmediate(c -> vkCmdFillBuffer(c, feedback.handle, 0, VK_WHOLE_SIZE, -1));
        feedbackRead = new Buffer(ctx, totalTiles * 4L, VK_BUFFER_USAGE_TRANSFER_DST_BIT, Buffer.READBACK);
        memSet(feedbackRead.map(), 0xFF);
        setInfo = new Buffer(ctx, 16L * (1 + Math.max(1, sets.size())), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        ByteBuffer info = setInfo.map();
        info.putInt(0, slotsPerRow).putInt(4, atlasSize);
        for (int s = 0; s < sets.size(); s++) {
            info.putInt(16 + s * 16, sets.get(s).size).putInt(20 + s * 16, sets.get(s).mipCount)
                    .putInt(24 + s * 16, firstTile[s]).putInt(28 + s * 16, 0);
        }
        boolean gpu = this.compress && gpuEncode && compiler != null;
        if (gpu) uploadBudget = 64;     // with GPU encoding only the raw copy (~220 KB per tile) limits uploads
        // The encoder reads the raw tiles from a compute shader, so the upload buffer is put in VRAM through the BAR.
        staging = new Buffer(ctx, (long) uploadBudget * rawTileBytes(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | (gpu ? VK_BUFFER_USAGE_STORAGE_BUFFER_BIT : 0), gpu ? Buffer.UPLOAD : Buffer.HOST);
        if (gpu) gpuEncoder = new GpuTileEncoder(ctx, compiler, staging, uploadBudget * Math.max(tileBytes(NEAR), tileBytes(FAR)),
                uploadBudget * CHANNELS);

        // A few workers suffice (BC7 takes ~5 ms per tile); more would compete with the render thread.
        workerCount = Math.clamp(Runtime.getRuntime().availableProcessors() / 4, 1, 8);
        workers = Executors.newFixedThreadPool(workerCount, r -> {
            Thread t = new Thread(r, "vt-worker");
            t.setDaemon(true);
            t.setPriority(Thread.NORM_PRIORITY - 1);
            return t;
        });

        List<Integer> pins = new ArrayList<>();
        for (int s = 0; s < sets.size(); s++) pins.add(firstTile[s] + sets.get(s).tileCount() - 1);
        if (gpuEncoder == null) for (int t : pins) if (this.compress) encoded.set(t, encodeTile(t));
        ctx.submitImmediate(cmd -> {
            try (MemoryStack stack = stackPush()) {
                upload(cmd, stack, pins, true);
            }
        });

        if (!sets.isEmpty()) workers.submit(this::computeSimilarity);

        System.out.printf("[vt] %d sets, %d tiles (%.0f MB in RAM), %s, pools 2 x %d slots = %.0f MB VRAM, %s%n",
                sets.size(), totalTiles, ramBytes() / 1048576.0, this.compress ? "BC7/BC5 near + BC1/BC5 far" : "RGBA8 (uncompressed)",
                slotCount, poolBytes() / 1048576.0, gpuEncoder != null ? "encoded on the GPU" : workerCount + " encode workers");
    }

    private static long rawTileBytes() { return (long) SLOT * SLOT * 4 * CHANNELS; }

    private long channelBytes(int pool, int channel) {
        BlockCompression.Encoder e = encoder[pool * CHANNELS + channel];
        return e == null ? (long) SLOT * SLOT * 4 : (long) (SLOT / 4) * (SLOT / 4) * e.bytesPerBlock();
    }

    private long tileBytes(int pool) {
        long b = 0;
        for (int c = 0; c < CHANNELS; c++) b += channelBytes(pool, c);
        return b;
    }

    /** VRAM used by both pools. */
    public long poolBytes() { return slotCount * (tileBytes(NEAR) + tileBytes(FAR)); }

    /** RAM used by all texture sets. */
    public long ramBytes() { return sets.stream().mapToLong(TextureSet::bytes).sum(); }

    public int workerCount() { return workerCount; }

    public double averageEncodeMs() {
        long c = encodeCount.get();
        return c == 0 ? 0 : encodeNanos.get() / 1e6 / c;
    }

    public long encodedTiles() { return encodeCount.get(); }

    /** Status line of the similarity search for the overlay. */
    public String similarityStatus() { return similarityStatus; }

    private static int poolOf(int mip) { return mip < FAR_MIP ? NEAR : FAR; }

    /** Creates one atlas (pool and channel) in the GENERAL layout. */
    private void createAtlas(int a, int format) {
        try (MemoryStack stack = stackPush()) {
            VkImageCreateInfo info = VkImageCreateInfo.calloc(stack)
                    .sType$Default()
                    .imageType(VK_IMAGE_TYPE_2D)
                    .format(format)
                    .mipLevels(1)
                    .arrayLayers(1)
                    .samples(VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
            info.extent().set(atlasSize, atlasSize, 1);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateImage(ctx.device, info, null, lp), "vkCreateImage(atlas)");
            atlasImage[a] = lp.get(0);
            VkMemoryRequirements req = VkMemoryRequirements.malloc(stack);
            vkGetImageMemoryRequirements(ctx.device, atlasImage[a], req);
            check(vkAllocateMemory(ctx.device, VkMemoryAllocateInfo.calloc(stack)
                    .sType$Default()
                    .allocationSize(req.size())
                    .memoryTypeIndex(ctx.findMemoryType(req.memoryTypeBits(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), null, lp),
                    "vkAllocateMemory(atlas)");
            atlasMemory[a] = lp.get(0);
            check(vkBindImageMemory(ctx.device, atlasImage[a], atlasMemory[a], 0), "vkBindImageMemory(atlas)");
            VkImageViewCreateInfo view = VkImageViewCreateInfo.calloc(stack)
                    .sType$Default().image(atlasImage[a]).viewType(VK_IMAGE_VIEW_TYPE_2D).format(format);
            view.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
            check(vkCreateImageView(ctx.device, view, null, lp), "vkCreateImageView(atlas)");
            atlasView[a] = lp.get(0);
        }
        ctx.submitImmediate(cmd -> {
            try (MemoryStack stack = stackPush()) {
                VkImageMemoryBarrier.Buffer b = VkImageMemoryBarrier.calloc(1, stack);
                b.get(0).sType$Default()
                        .srcAccessMask(0).dstAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT)
                        .oldLayout(VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK_IMAGE_LAYOUT_GENERAL)
                        .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .image(atlasImage[a])
                        .subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, null, null, b);
            }
        });
    }

    /** Block-compresses all channels of a tile, border included. */
    private ByteBuffer encodeTile(int tile) {
        long t0 = System.nanoTime();
        int pool = poolOf(tileMip[tile]);
        TextureSet set = sets.get(tileSet[tile]);
        ByteBuffer raw = memAlloc(SLOT * SLOT * 4);
        ByteBuffer out = memAlloc((int) tileBytes(pool));
        int o = 0;
        for (int c = 0; c < CHANNELS; c++) {
            set.extractTile(c, tileMip[tile], tileX[tile], tileY[tile], raw, 0);
            encoder[pool * CHANNELS + c].encode(raw, 0, SLOT, SLOT, out, o);
            o += (int) channelBytes(pool, c);
        }
        memFree(raw);
        encodeNanos.addAndGet(System.nanoTime() - t0);
        encodeCount.incrementAndGet();
        return out;
    }

    private void requestEncode(int tile) {
        if (encodePending[tile] || encoded.get(tile) != null) return;
        encodePending[tile] = true;
        statEncodeQueue++;
        workers.submit(() -> encoded.set(tile, encodeTile(tile)));
    }

    // Similarity

    private static final int FP_ALBEDO = 8, FP_SMALL = 4;
    private static final int FP_LEN = FP_ALBEDO * FP_ALBEDO * 3 + FP_SMALL * FP_SMALL * 2 + FP_SMALL * FP_SMALL * 3;

    /** Fingerprint of a tile's core: 8x8 albedo, 4x4 normal xy and 4x4 ORM. */
    private float[] fingerprint(int tile) {
        TextureSet set = sets.get(tileSet[tile]);
        int mip = tileMip[tile], dim = set.size >> mip;
        float[] fp = new float[FP_LEN];
        int o = 0;
        int[][] layout = {{ALBEDO, FP_ALBEDO, 3}, {NORMAL, FP_SMALL, 2}, {ORM, FP_SMALL, 3}};
        for (int[] l : layout) {
            ByteBuffer src = set.level(l[0], mip);
            int cells = l[1], step = TILE / cells, comps = l[2];
            for (int cy = 0; cy < cells; cy++) for (int cx = 0; cx < cells; cx++) {
                float[] acc = new float[3];
                for (int y = 0; y < step; y += 2) for (int x = 0; x < step; x += 2) {   // every other texel suffices
                    int px = tileX[tile] * TILE + cx * step + x, py = tileY[tile] * TILE + cy * step + y;
                    int i = (py * dim + px) * 4;
                    for (int c = 0; c < comps; c++) acc[c] += (src.get(i + c) & 0xFF) / 255f;
                }
                float inv = 1f / ((step / 2) * (step / 2));
                for (int c = 0; c < comps; c++) fp[o++] = acc[c] * inv * (l[0] == NORMAL ? 2f : 1f); // normals span [-1, 1]
            }
        }
        return fp;
    }

    /** RMS difference with albedo weighted 1, normals and ORM 0.5. */
    private static float distance(float[] a, float[] b) {
        float sa = 0, sn = 0, so = 0;
        int na = FP_ALBEDO * FP_ALBEDO * 3, nn = FP_SMALL * FP_SMALL * 2;
        for (int i = 0; i < na; i++) { float d = a[i] - b[i]; sa += d * d; }
        for (int i = na; i < na + nn; i++) { float d = a[i] - b[i]; sn += d * d; }
        for (int i = na + nn; i < FP_LEN; i++) { float d = a[i] - b[i]; so += d * d; }
        return (float) Math.sqrt(sa / na + 0.5f * sn / nn + 0.5f * so / (FP_LEN - na - nn));
    }

    /** Finds the most similar tiles of the same mip level for every tile. Runs on a worker thread. */
    private void computeSimilarity() {
        long t0 = System.nanoTime();
        float[][] fp = new float[totalTiles][];
        IntStream.range(0, totalTiles).parallel().forEach(t -> fp[t] = fingerprint(t));
        int[][] ct = new int[totalTiles][];
        float[][] cd = new float[totalTiles][];
        // Only tiles of the same mip are compared, so a stand-in never has less resolution or color depth.
        IntStream.range(0, totalTiles).parallel().forEach(a -> {
            int[] best = new int[CANDIDATES];
            float[] bestD = new float[CANDIDATES];
            Arrays.fill(best, -1);
            Arrays.fill(bestD, Float.MAX_VALUE);
            for (int b = 0; b < totalTiles; b++) {
                if (b == a || tileMip[b] != tileMip[a]) continue;
                float d = distance(fp[a], fp[b]);
                if (d >= bestD[CANDIDATES - 1]) continue;
                int k = CANDIDATES - 1;
                while (k > 0 && bestD[k - 1] > d) { bestD[k] = bestD[k - 1]; best[k] = best[k - 1]; k--; }
                bestD[k] = d;
                best[k] = b;
            }
            ct[a] = best;
            cd[a] = bestD;
        });
        candTile = ct;
        candDist = cd;
        similarityReady = true;
        int close = 0;
        for (float[] d : cd) if (d[0] < 0.04f) close++;
        similarityStatus = String.format("%d/%d tiles have a stand-in < 0.04 (%.1f s)", close, totalTiles, (System.nanoTime() - t0) / 1e9);
        System.out.println("[vt] similarity: " + similarityStatus);
    }

    /**
     * Maximum fingerprint difference of a stand-in for a tile needed at {@code dist} meters: none below 3 m, then
     * growing to 0.04 at 12 m and 0.08 at 30 m, capped at 0.1.
     */
    private float tolerance(float dist) {
        float t = 0.04f * Math.clamp((dist - 3f) / 9f, 0f, 1f) + 0.04f * Math.clamp((dist - 12f) / 18f, 0f, 1f)
                + 0.02f * Math.clamp((dist - 30f) / 30f, 0f, 1f);
        return t * similarityScale;
    }

    /** Inverse of {@link #tolerance}: the distance from which a stand-in with difference {@code d} is acceptable. */
    private float standInDistance(float d) {
        float s = Math.max(similarityScale, 1e-6f);
        d /= s;
        if (d <= 0.04f) return 3f + 9f * d / 0.04f;
        if (d <= 0.08f) return 12f + 18f * (d - 0.04f) / 0.04f;
        if (d <= 0.10f) return 30f + 30f * (d - 0.08f) / 0.02f;
        return Float.MAX_VALUE;
    }

    /** Page table entry: {@code pool << 30 | (slot + 1)}, or 0 for none. */
    private static int entry(int pool, int slot) { return slot < 0 ? 0 : (pool << 30) | (slot + 1); }

    /**
     * Processes the previous frame's feedback and records this frame's uploads. Call before tracing, after waiting
     * for the previous frame's fence.
     */
    public void update(VkCommandBuffer cmd, MemoryStack stack) {
        frame++;
        setInfo.map().putInt(8, (int) frame);        // used to fade in freshly uploaded tiles
        ByteBuffer fb = feedbackRead.map();
        ByteBuffer pt = pageTable.map();
        List<Integer> missing = new ArrayList<>();
        int requested = 0, aliased = 0;
        boolean reuse = similarityReady && similarityScale > 0;
        int[][] ct = candTile;
        float[][] cd = candDist;
        for (int tile = 0; tile < totalTiles; tile++) {
            int dcm = fb.getInt(tile * 4);
            if (dcm == -1) continue;                    // not needed this frame
            fb.putInt(tile * 4, -1);                    // mark as consumed
            requested++;
            float dist = Integer.toUnsignedLong(dcm) / 100f;
            int pool = poolOf(tileMip[tile]);

            // Prefer resident stand-ins. A non-resident one is only worth loading if it is good enough for every
            // pixel of this tile, since one upload can then serve many tiles.
            int stand = -1;
            float standDist = 0f;
            if (reuse) {
                float tol = tolerance(dist);
                // Hysteresis: keep last frame's stand-in while it is still good enough.
                int prev = aliasOf[tile];
                if (prev >= 0 && tileSlot[prev] >= 0) {
                    for (int k = 0; k < CANDIDATES && ct[tile][k] >= 0; k++) {
                        if (ct[tile][k] == prev && cd[tile][k] <= tol * STAND_IN_HYSTERESIS) {
                            stand = prev;
                            standDist = standInDistance(cd[tile][k]);
                            break;
                        }
                    }
                }
                for (int k = 0; k < CANDIDATES && stand < 0; k++) {
                    int b = ct[tile][k];
                    if (b < 0 || cd[tile][k] > tol) break;
                    if (tileSlot[b] >= 0) { stand = b; standDist = standInDistance(cd[tile][k]); break; }
                }
                if (stand < 0 && ct[tile][0] >= 0 && cd[tile][0] <= tol) {
                    stand = ct[tile][0];
                    standDist = standInDistance(cd[tile][0]);
                    if (tileSlot[stand] < 0) missing.add(stand);
                }
            }
            boolean ownNeeded = stand < 0 || dist < standDist;
            int own = tileSlot[tile];
            if (own >= 0) slotLastUsed[pool][own] = frame;
            if (stand >= 0 && tileSlot[stand] >= 0) slotLastUsed[pool][tileSlot[stand]] = frame;
            if (ownNeeded && own < 0) missing.add(tile);
            if (!ownNeeded) { aliased++; if (own < 0) statUploadsSaved++; }

            int standEntry = stand >= 0 && tileSlot[stand] >= 0 ? entry(pool, tileSlot[stand]) : 0;
            aliasOf[tile] = standEntry != 0 ? stand : -1;
            pt.putInt(tile * 16 + 4, standEntry);
            pt.putFloat(tile * 16 + 8, standEntry != 0 ? standDist : 0f);
        }
        // Evict tiles that have not been needed for a while.
        int evicted = 0;
        for (int p = 0; p < POOLS; p++) {
            for (int s = 0; s < slotCount; s++) {
                if (slotTile[p][s] >= 0 && !slotPinned[p][s] && frame - slotLastUsed[p][s] > EVICT_AFTER_FRAMES) {
                    evict(p, s, pt);
                    evicted++;
                }
            }
        }
        // Coarse mips first: they cover more of the screen and are the fallback for finer ones.
        missing = new ArrayList<>(new LinkedHashSet<>(missing));
        missing.sort((a, b) -> Integer.compare(tileMip[b], tileMip[a]));
        statMissing = missing.size();
        statEncodeQueue = 0;
        List<Integer> ready = new ArrayList<>();
        for (int t : missing) {
            if (!compress || gpuEncoder != null || encoded.get(t) != null) ready.add(t);
            else requestEncode(t);                   // the coarser mip is shown until the worker is done
            if (ready.size() >= uploadBudget) break;
        }
        int before = evictionsByUpload;
        int uploads = upload(cmd, stack, ready, false);
        statRequested = requested;
        statAliased = aliased;
        statUploads = uploads;
        statEvictions = evicted + (evictionsByUpload - before);
        statTotalUploads += uploads;
        for (int p = 0; p < POOLS; p++) {
            int r = 0;
            for (int t : slotTile[p]) if (t >= 0) r++;
            statResident[p] = r;
        }
        int pendingNow = 0;
        for (int t = 0; t < totalTiles; t++) if (encodePending[t] && encoded.get(t) == null) pendingNow++;
        statEncodeQueue = pendingNow;
    }

    /** Records the copy of this frame's feedback for {@link #update} and clears it for the next frame. */
    public void recordFeedbackReadback(VkCommandBuffer cmd, MemoryStack stack) {
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, mb, null, null);
        vkCmdCopyBuffer(cmd, feedback.handle, feedbackRead.handle, VkBufferCopy.calloc(1, stack).size(feedback.size));
        VkMemoryBarrier.Buffer cb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_TRANSFER_READ_BIT).dstAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, cb, null, null);
        vkCmdFillBuffer(cmd, feedback.handle, 0, VK_WHOLE_SIZE, -1);
        VkMemoryBarrier.Buffer hb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_HOST_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR, 0, hb, null, null);
    }

    private static int bcCode(BlockCompression.Encoder e) {
        return e == BlockCompression.BC1 ? 0 : e == BlockCompression.BC5 ? 1 : 2;
    }

    private int evictionsByUpload;

    /** Frees a slot and clears the page table entries of its tile and of all tiles using it as stand-in. */
    private void evict(int pool, int slot, ByteBuffer pt) {
        int old = slotTile[pool][slot];
        if (old >= 0) {
            tileSlot[old] = -1;
            pt.putInt(old * 16, 0);
            for (int t = 0; t < totalTiles; t++) {
                if (aliasOf[t] == old) { aliasOf[t] = -1; pt.putInt(t * 16 + 4, 0); }
            }
        }
        slotTile[pool][slot] = -1;
    }

    /** A free slot, else the least recently used unpinned slot not needed this frame; -1 if there is none. */
    private int findSlot(int pool) {
        int best = -1;
        long bestStamp = Long.MAX_VALUE;
        for (int s = 0; s < slotCount; s++) {
            if (slotTile[pool][s] < 0) return s;
            if (slotPinned[pool][s] || slotLastUsed[pool][s] >= frame) continue;
            if (slotLastUsed[pool][s] < bestStamp) { bestStamp = slotLastUsed[pool][s]; best = s; }
        }
        return best;
    }

    /** Uploads up to {@code uploadBudget} tiles and returns how many were uploaded. */
    private int upload(VkCommandBuffer cmd, MemoryStack stack, List<Integer> tiles, boolean pin) {
        if (tiles.isEmpty()) return 0;
        ByteBuffer st = staging.map();
        ByteBuffer pt = pageTable.map();
        long offset = 0, encOffset = 0;
        int count = 0;
        VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack);
        List<long[]> gpuCopies = new ArrayList<>();      // {encoded offset, atlas, x, y}, copied after encoding
        if (gpuEncoder != null) gpuEncoder.begin();
        for (int tile : tiles) {
            if (count >= uploadBudget) break;
            if (tileSlot[tile] >= 0) continue;              // requested twice
            int pool = poolOf(tileMip[tile]);
            int slot = findSlot(pool);
            if (slot < 0) continue;                         // pool full of tiles needed this frame
            if (slotTile[pool][slot] >= 0) { evict(pool, slot, pt); evictionsByUpload++; }
            int sx = (slot % slotsPerRow) * SLOT, sy = (slot / slotsPerRow) * SLOT;
            ByteBuffer enc = compress ? encoded.get(tile) : null;
            long encOff = 0;
            for (int c = 0; c < CHANNELS; c++) {
                if (compress && enc == null && gpuEncoder != null) {
                    // Raw channel into the upload buffer; it is encoded on the GPU below.
                    sets.get(tileSet[tile]).extractTile(c, tileMip[tile], tileX[tile], tileY[tile], st, (int) offset);
                    gpuEncoder.add(offset, encOffset, bcCode(encoder[pool * CHANNELS + c]), SLOT);
                    gpuCopies.add(new long[]{encOffset, pool * CHANNELS + c, sx, sy});
                    encOffset += channelBytes(pool, c);
                    offset += (long) SLOT * SLOT * 4;
                    continue;
                }
                long bytes = channelBytes(pool, c);
                if (compress) {
                    memCopy(memAddress(enc) + encOff, memAddress(st) + offset, bytes);
                    encOff += bytes;
                } else {
                    sets.get(tileSet[tile]).extractTile(c, tileMip[tile], tileX[tile], tileY[tile], st, (int) offset);
                }
                region.get(0).bufferOffset(offset).bufferRowLength(0).bufferImageHeight(0);
                region.get(0).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
                region.get(0).imageOffset().set(sx, sy, 0);
                region.get(0).imageExtent().set(SLOT, SLOT, 1);
                vkCmdCopyBufferToImage(cmd, staging.handle, atlasImage[pool * CHANNELS + c], VK_IMAGE_LAYOUT_GENERAL, region);
                offset += bytes;
            }
            slotTile[pool][slot] = tile;
            tileSlot[tile] = slot;
            slotLastUsed[pool][slot] = frame;
            slotPinned[pool][slot] = pin;
            pt.putInt(tile * 16, entry(pool, slot));
            pt.putInt(tile * 16 + 12, (int) frame);     // upload frame, for the fade-in
            count++;
        }
        if (!gpuCopies.isEmpty()) {
            gpuEncoder.record(cmd, stack);
            for (long[] g : gpuCopies) {
                region.get(0).bufferOffset(g[0]).bufferRowLength(0).bufferImageHeight(0);
                region.get(0).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(0).baseArrayLayer(0).layerCount(1);
                region.get(0).imageOffset().set((int) g[2], (int) g[3], 0);
                region.get(0).imageExtent().set(SLOT, SLOT, 1);
                vkCmdCopyBufferToImage(cmd, gpuEncoder.encoded.handle, atlasImage[(int) g[1]], VK_IMAGE_LAYOUT_GENERAL, region);
            }
        }
        if (count > 0) {
            VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK_ACCESS_SHADER_READ_BIT);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR,
                    0, mb, null, null);
        }
        return count;
    }

    /** Frees the GPU resources and encoded tiles. The texture sets belong to the caller and are not freed. */
    @Override
    public void close() {
        workers.shutdownNow();
        if (gpuEncoder != null) gpuEncoder.close();
        staging.close();
        setInfo.close();
        feedback.close();
        feedbackRead.close();
        pageTable.close();
        vkDestroySampler(ctx.device, sampler, null);
        for (int a = 0; a < POOLS * CHANNELS; a++) {
            vkDestroyImageView(ctx.device, atlasView[a], null);
            vkDestroyImage(ctx.device, atlasImage[a], null);
            vkFreeMemory(ctx.device, atlasMemory[a], null);
        }
        for (int t = 0; t < totalTiles; t++) {
            ByteBuffer b = encoded.get(t);
            if (b != null) memFree(b);
        }
    }
}
