package dev.lumvex.render;

import dev.lumvex.render.scene.Camera;
import dev.lumvex.render.scene.Material;
import dev.lumvex.render.scene.Mesh;
import dev.lumvex.render.scene.Scene;
import dev.lumvex.texture.TextureSet;
import dev.lumvex.texture.VirtualTextures;
import dev.lumvex.vk.*;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.*;
import static org.lwjgl.vulkan.KHRAccelerationStructure.*;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK13.*;

/**
 * The hardware path tracer. Owns all GPU state: Vulkan context, pipelines, scene buffers and acceleration
 * structures, the per-pixel history, virtual textures and the radiance cache.
 *
 * <p>A frame applies pending scene changes, traces, denoises, composites and presents. Pixel history survives camera
 * motion through reprojection, and moving objects only restart the image inside their influence zones. Once every
 * pixel reaches {@link #sampleLimit}, tracing stops and only compositing runs.
 */
public final class Renderer implements AutoCloseable {

    private record GpuMesh(Buffer vertices, Buffer indices, AccelerationStructure blas) {}

    private static final int TLAS_INSTANCE_SIZE = 64;
    private static final int INSTANCE_INFO_SIZE = 24;
    private static final int RT_STAGES = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
            | VK_SHADER_STAGE_MISS_BIT_KHR;

    // Settings; written by the render thread between frames.
    public int maxBounces = 8;
    public int samplesPerFrame = 1;
    public float exposure = 1.0f;
    /** Lens flare intensity; 0 disables them. */
    public float flareStrength = 1.0f;
    private static final float SUN_FLARE = 1.0f;
    public int debugMode = 0;
    public boolean vsync;
    public boolean denoise = !Boolean.getBoolean("lumvex.nodenoise");
    public int denoiseIterations = 4;

    private static final int TS_START = 0, TS_TRACE_BEGIN = 1, TS_TRAIN_END = 2, TS_TRACE_END = 3, TS_DENOISE_END = 4, TS_END = 5,
            TS_COUNT = 6;

    private final RenderSurface window;
    private final VkContext ctx;
    private final Scene scene;
    private final ShaderCompiler compiler;
    private final RtPipeline pipeline;
    private final PostProcess post;
    private final long queryPool;
    private boolean timestampsPending;
    private double gpuTraceMs, gpuTrainMs, gpuDenoiseMs, gpuTotalMs;
    /** Raw GPU times in ms of the last completed frame: trace (including training), denoise, total, training. */
    public final double[] lastGpuMs = new double[4];

    private final List<GpuMesh> gpuMeshes = new ArrayList<>();
    private Buffer instanceInfoBuffer;
    private Buffer materialBuffer;
    private Buffer tlasInstanceBuffer;
    private AccelerationStructure tlas;
    private LensFlare flare;
    /** Whether the GPU traces rays at all; without it everything below that needs ray tracing is left out. */
    private final boolean rt;
    /** The raster render mode's shading pass, see {@link RasterShade}. */
    private final RasterShade rasterShade;
    /** An empty flare list for the composite pass on GPUs without ray tracing (lens flares need ray queries). */
    private Buffer noFlares;
    private Camera currentCamera;
    private Buffer tlasScratch;
    /** Instances in the last TLAS build and refits since, see recordFrame. */
    private int tlasBuiltCount = -1, tlasRefits;
    private static final int REFIT_LIMIT = 120;
    private long tlasScratchAddress;

    private final RadianceCache cache;
    /** Push constant flag: objects moved this frame (FLAG_ANIMATED in common.glsl). */
    private static final int FLAG_ANIMATED = 8;
    /** Push constant flag: the camera moved this frame (FLAG_CAMERA_MOVED in common.glsl). */
    private static final int FLAG_CAMERA_MOVED = 2;
    private VirtualTextures vt;

    /** Moving objects whose influence zones the shader tests; must match MAX_DYNAMIC in raygen.rgen. */
    private static final int MAX_DYNAMIC = 64;
    /**
     * uvec4 count, then per object the swept AABB as two vec4: min.w holds the influence radius, max.w the owning
     * instance + 1 (0.5 for none), negated for emitters; then vec4(power, 0, 0, 0), where an emitter's irradiance at
     * distance d is about power / d^2.
     */
    private final Buffer dynamicBuffer;
    private static final int DYNAMIC_STRIDE = 48;
    /** Light positions of the previous frame; a light that moved gets an influence zone of its own. */
    private final List<Vector3f> previousLightPositions = new ArrayList<>();
    /** uvec4 header, then per light vec4(position, radius) and vec4(radiance, 0), then the light grid. */
    private Buffer lightBuffer;

    // Light transport options (OPT_* in raygen.rgen), passed in the light buffer header every frame.
    /**
     * Pick lights by estimated contribution instead of uniformly. Off by default: with a dozen nearby candles it
     * brought no visible gain and cost 4.5 ms.
     */
    public boolean lightImportance = false;
    /** Pick shadow-ray lights from the light grid, see {@link LightGrid}. */
    public boolean lightGrid = !Boolean.getBoolean("lumvex.nolightgrid");
    /**
     * Seed newly visible pixels with the radiance cache's estimate, so disocclusions do not start black. Off by
     * default: with {@link #catchUpSamples} of 8 it brought no visible gain, while the barely trained 8 cm cells of the
     * first seconds showed as colored blocks.
     */
    public boolean cachePrior = Boolean.getBoolean("lumvex.prior");
    private final LightGrid grid = new LightGrid();
    /**
     * Terminate paths in the radiance cache at bounce {@link #cacheBounce}; {@link #trainRate} of the pixels trace
     * full paths to train it. Ending at bounce 1 measured the same error as full paths at half the trace time.
     */
    public boolean cachePaths = true;
    public float trainRate = 1f / 16f;
    public int cacheBounce = 1;
    /** Compute indirect light on 1 / 2^indirectShift of the pixels per frame, weighted accordingly. */
    public int indirectShift = 0;
    /** ReSTIR DI for direct light. Off by default: in the test scenes it was noisier than plain NEE and cost 5.7 ms. */
    public boolean restir = false;
    /** How a scene is rendered, see {@code SceneDoc.Settings.renderMode} and docs/HYBRID_RENDERING.md. */
    public enum RenderMode {
        /** Rasterized primary visibility, path traced light. */
        HYBRID,
        /** Everything path traced, the renderer of Lumvex 1.0. */
        PATH,
        /** Rasterization only: textured surfaces lit by the sun and the sky, no ray tracing. */
        RASTER;

        /** The mode named {@code "hybrid"}, {@code "path"} or {@code "raster"}; anything else is hybrid. */
        public static RenderMode parse(String name) {
            if (name == null) return HYBRID;
            return switch (name.trim().toLowerCase(java.util.Locale.ROOT)) {
                case "path", "pathtracing", "path_tracing" -> PATH;
                case "raster", "rasterization" -> RASTER;
                default -> HYBRID;
            };
        }
    }

    private RenderMode renderMode = RenderMode.HYBRID;

    /** The mode in use: what the scene asked for, or {@link RenderMode#RASTER} on GPUs without ray tracing. */
    public RenderMode renderMode() { return renderMode; }

    /** Sets the render mode; GPUs without hardware ray tracing stay in {@link RenderMode#RASTER}. */
    public void setRenderMode(RenderMode mode) {
        RenderMode m = ctx.rayTracingSupported ? mode : RenderMode.RASTER;
        if (m != renderMode) resetAccumulation();
        renderMode = m;
        rasterPrimary = m != RenderMode.PATH;
    }

    /**
     * Take the primary surface of every pixel from the raster pre-pass instead of tracing a ray through its center
     * (hybrid rendering, see {@link RasterPrepass}); follows the render mode.
     */
    public boolean rasterPrimary = true;
    private final RasterPrepass raster;
    /** Per instance: the object-to-world matrix as mat4x3 (48 bytes), for hits rebuilt from the visibility buffer. */
    private Buffer instanceTransformBuffer;
    private static final int INSTANCE_TRANSFORM_SIZE = 48;
    /** Whether any instance is alpha-tested; the raster pass leaves those out and raygen tests them by ray. */
    private boolean hasCutouts;
    /** Reversed infinite-Z projection of the raster pass, and its product with the view and the overscan scale. */
    private final Matrix4f rasterProj = new Matrix4f(), rasterViewProj = new Matrix4f();
    /** Debug view that compares the raster pass with traced primary rays, see raygen.rgen. */
    private static final int DEBUG_RASTER = 8;
    /** Sets the strength of the denoiser's disocclusion fix, see {@link PostProcess#disocclusionFix}; 0 disables it. */
    public void setDisocclusionFix(float strength) { post.disocclusionFix = Math.max(0f, strength); }
    /** Shader execution reordering for bounce rays, where supported and compiled in. */
    public boolean ser = true;
    /**
     * Samples a newly visible pixel takes in its first frame. Below 8, walls revealed next to a bright lamp sparkle
     * while the camera moves; only disoccluded pixels pay for it.
     */
    public int catchUpSamples = Integer.getInteger("lumvex.catchup", 8);
    /** Extend the training pass about 25% beyond the view, shifted towards where the camera is turning. */
    public boolean prefetch = true;
    private static final float PREFETCH_GUARD = 1.25f;
    /** Predicted view shift in NDC: last frame's rotation times PREFETCH_FRAMES. */
    private final float[] prefetchShift = new float[2];
    private static final float PREFETCH_FRAMES = 8f;
    private final Vector3f prevForward = new Vector3f(), curForward = new Vector3f();
    private boolean hasPrevForward;
    /**
     * Whether to compile the SER code path at all; set before construction. Off by default because on an RTX 4060
     * the reorder point, which has to spill the shader's live state, cost more than it gained.
     */
    public static boolean compileSer = false;
    /** Collect ray counters; cheap thanks to subgroup aggregation. */
    public boolean rayStats = true;
    private final Buffer statsBuffer;
    /** Counters of the last traced frame, see STAT_* in raygen.rgen. */
    public final long[] lastStats = new long[32];

    private Buffer pickBuffer;
    // Game UI image, premultiplied ARGB at output size; uploaded only when it changes.
    private Buffer uiBuffer;
    private volatile int[] uiPending;
    private volatile int uiPendingW, uiPendingH;
    private volatile boolean uiChanged;
    // requested pixel, copied by the next recorded frame
    private int[] pickPending;
    // pixel whose value is in pickBuffer once that frame completes
    private int[] pickCopied;
    private Consumer<float[]> pickCallback;

    /** Sets the UI overlay as premultiplied ARGB pixels, or null for none. Thread-safe; applied before the next frame. */
    public void setUiImage(int[] argbPre, int w, int h) {
        uiPending = argbPre;
        uiPendingW = w;
        uiPendingH = h;
        uiChanged = true;
    }

    // Grows the UI buffer if needed, which requires waiting for the GPU and rebinding it.
    private void applyUi() {
        if (!uiChanged) return;
        uiChanged = false;
        int[] px = uiPending;
        int w = uiPendingW, h = uiPendingH;
        long need = 16 + 4L * (px == null ? 0 : (long) w * h);
        if (need > uiBuffer.size) {
            vkDeviceWaitIdle(ctx.device);
            uiBuffer.close();
            uiBuffer = new Buffer(ctx, need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
            post.updateUiBuffer(uiBuffer);
        }
        ByteBuffer b = uiBuffer.map();
        if (px == null) {
            b.putInt(8, 0);
            return;
        }
        b.putInt(0, w).putInt(4, h).putInt(8, 1).putInt(12, 0);
        b.position(16);
        b.asIntBuffer().put(px, 0, w * h);
        b.position(0);
    }

    /**
     * Requests the render instance and view depth at an image pixel. One or two frames later the callback receives
     * {instance + 1, depth}, where 0 means sky.
     */
    public void requestPick(int x, int y, Consumer<float[]> callback) {
        pickPending = new int[]{x, y};
        pickCallback = callback;
    }

    private void readPick() {
        if (pickCopied == null || pickCallback == null) return;
        ByteBuffer b = pickBuffer.map();
        // meta.z: instance + 1, meta.y: view depth
        float[] r = {b.getFloat(8), b.getFloat(4)};
        pickCopied = null;
        Consumer<float[]> cb = pickCallback;
        pickCallback = null;
        cb.accept(r);
    }

    /** Records the copy of the requested pick pixel of the meta image, offset by the overscan border, to the host. */
    private void recordPickCopy(MemoryStack stack, int shown) {
        if (pickPending == null) return;
        int x = marginX + Math.clamp((long) ((pickPending[0] + 0.5f) * vW / fbW), 0, vW - 1);
        int y = marginY + Math.clamp((long) ((pickPending[1] + 0.5f) * vH / fbH), 0, vH - 1);
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(cmd, ctx.rtStage | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, mb, null, null);
        VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack);
        region.get(0).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
        region.get(0).imageOffset().set(x, y, 0);
        region.get(0).imageExtent().set(1, 1, 1);
        vkCmdCopyImageToBuffer(cmd, history[shown].meta().image, VK_IMAGE_LAYOUT_GENERAL, pickBuffer.handle, region);
        pickCopied = pickPending;
        pickPending = null;
    }
    private final StorageImage[] reservoir = new StorageImage[2];
    private EnvironmentTexture envTexture;
    /** Per instance: mat4 from the current to the previous world position, for object motion vectors. */
    private Buffer deltaBuffer;
    private final List<float[]> meshBounds = new ArrayList<>();
    // Scene change tracking. Buffers grow by doubling; nothing that was uploaded is thrown away.
    private int instanceCapacity, materialCapacity, lightCapacity;
    private int seenStructure, seenTransforms, seenLights, seenEnvironment, seenTextures, seenLightMoves;
    private boolean tlasDirty;
    private static final int GEOMETRY_USAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    /** Keep the history when objects move and restart only their influence zones. */
    public boolean dynamicHistory = true;

    private long descriptorPool;
    /** {@code descriptorSets[p]} writes {@code history[p]} and reads {@code history[1 - p]}. */
    private final long[] descriptorSets = new long[2];

    private Swapchain swapchain;
    /** Double-buffered per-pixel history. */
    private final PostProcess.HistoryImages[] history = new PostProcess.HistoryImages[2];
    private int parity;
    private StorageImage pingA, pingB;
    /** Ping-pong images of the specular filter. */
    private StorageImage specPingA, specPingB;
    /** Output of the specular temporal stage, double-buffered like the history. */
    private final StorageImage[] specFiltered = new StorageImage[2];
    /** Output of the temporal filter, double-buffered like the history. */
    private final StorageImage[] filtered = new StorageImage[2];
    private StorageImage outputImage;
    private Buffer readback;

    private final VkCommandBuffer cmd;
    private final long imageAvailable;
    private final long inFlight;
    private final ByteBuffer pushConstants = memAlloc(RtPipeline.PUSH_CONSTANT_SIZE);
    private final Matrix4f tmp = new Matrix4f();

    /** Frames since the camera last moved or the history was reset. */
    private int accumulatedFrames;
    /** Monotonic frame counter for RNG seeds. */
    private int frameCounter;
    private boolean historyValid;
    private final Matrix4f viewProj = new Matrix4f();
    private final Matrix4f prevViewProj = new Matrix4f();
    private boolean hasPrevViewProj;
    /** Reuse samples across camera motion instead of restarting on every move. */
    public boolean reprojection = true;
    private Path pendingScreenshot;

    /**
     * Creates the Vulkan context on the best GPU (see {@link VkContext}), compiles the shaders, uploads the scene and
     * builds its acceleration structures. Without hardware ray tracing only what the raster render mode needs is
     * created. The environment variable {@code LUMVEX_DEFINES=NAME,NAME2} adds shader defines for experiments.
     */
    public Renderer(RenderSurface window, Scene scene, boolean validation, boolean vsync) {
        this.window = window;
        this.scene = scene;
        this.vsync = vsync;
        ctx = new VkContext(window, validation);
        rt = ctx.rayTracingSupported;
        System.out.println("[lumvex] GPU: " + ctx.deviceName + (rt
                ? " | SBT handle " + ctx.shaderGroupHandleSize + "B | max recursion " + ctx.maxRayRecursionDepth
                : " | no hardware ray tracing: rasterization only"));
        if (!rt) renderMode = RenderMode.RASTER;

        compiler = new ShaderCompiler();
        int expectedShaderCount = (rt ? 19 : 4) + (scene.textureCompression && ctx.textureCompressionBC && VirtualTextures.gpuEncode
                && ctx.shaderInt64Supported ? 1 : 0);
        compiler.setProgressListener(expectedShaderCount, (shader, percent) -> window.shaderCompilationProgress(shader, percent));
        if (ctx.atomicFloatAddSupported) compiler.define("LUMVEX_CACHE", "1");
        else System.err.println("[lumvex] no float atomics on this GPU: speculative rendering disabled");
        if (ctx.invocationReorderSupported && compileSer) compiler.define("LUMVEX_SER", "1");
        else ser = false;
        if (System.getenv("LUMVEX_DEFINES") != null) {
            for (String d : System.getenv("LUMVEX_DEFINES").split(",")) if (!d.isBlank()) compiler.define(d.trim(), "1");
        }
        pipeline = rt ? new RtPipeline(ctx, compiler) : null;
        raster = new RasterPrepass(ctx, compiler);
        rasterShade = new RasterShade(ctx, compiler, RtPipeline.PUSH_CONSTANT_SIZE);

        vt = new VirtualTextures(ctx, scene.textureSets,
                scene.textureSets.isEmpty() ? 1 : texturePoolSlots(), scene.textureCompression, compiler);
        vt.similarityScale = scene.textureSimilarity;

        cache = rt ? new RadianceCache(ctx, compiler) : null;
        post = new PostProcess(ctx, compiler);

        uiBuffer = new Buffer(ctx, 16 + 4 * 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        memSet(uiBuffer.map(), 0);
        pickBuffer = new Buffer(ctx, 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, Buffer.READBACK);

        try (MemoryStack stack = stackPush()) {
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateQueryPool(ctx.device, VkQueryPoolCreateInfo.calloc(stack)
                    .sType$Default()
                    .queryType(VK_QUERY_TYPE_TIMESTAMP)
                    .queryCount(TS_COUNT), null, lp), "vkCreateQueryPool");
            queryPool = lp.get(0);
        }

        long t0 = System.nanoTime();
        long triangles = uploadNewMeshes();
        dynamicBuffer = new Buffer(ctx, 16 + (long) MAX_DYNAMIC * DYNAMIC_STRIDE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        statsBuffer = new Buffer(ctx, 4L * lastStats.length, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        memSet(statsBuffer.map(), 0);
        exposure = scene.exposure;
        envTexture = new EnvironmentTexture(ctx, scene.environment);
        ensureCapacities();
        writeInstanceInfo();
        writeMaterials();
        writeLights();
        updateDynamics(false);
        writeTlasInstances();
        writeRasterDraws();
        if (rt) ctx.submitImmediate(c -> tlas.recordTopLevelBuild(c, tlasInstanceBuffer, scene.instances.size(), tlasScratchAddress, false));
        tlasBuiltCount = scene.instances.size();
        tlasRefits = 0;
        markSceneSeen();
        System.out.printf("[lumvex] %d meshes, %d instances, %,d triangles, " + (rt ? "BLAS+TLAS built" : "uploaded") + " in %.1f ms%n",
                scene.meshes.size(), scene.instances.size(), triangles, (System.nanoTime() - t0) / 1e6);

        if (rt) try (MemoryStack stack = stackPush()) {
            VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(4, stack);
            sizes.get(3).type(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(2 * 7);
            sizes.get(0).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(2);
            sizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(2 * 16);
            sizes.get(2).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(2 * 16);
            LongBuffer lp = stack.mallocLong(2);
            check(vkCreateDescriptorPool(ctx.device, VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(2).pPoolSizes(sizes), null, lp), "vkCreateDescriptorPool");
            descriptorPool = lp.get(0);
            check(vkAllocateDescriptorSets(ctx.device, VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default()
                    .descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(pipeline.descriptorSetLayout, pipeline.descriptorSetLayout)), lp),
                    "vkAllocateDescriptorSets");
            descriptorSets[0] = lp.get(0);
            descriptorSets[1] = lp.get(1);
        }
        writeSceneDescriptors();
        try (MemoryStack stack = stackPush()) {
            LongBuffer lp = stack.mallocLong(1);

            PointerBuffer pp = stack.mallocPointer(1);
            check(vkAllocateCommandBuffers(ctx.device, VkCommandBufferAllocateInfo.calloc(stack)
                    .sType$Default()
                    .commandPool(ctx.commandPool)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY)
                    .commandBufferCount(1), pp), "vkAllocateCommandBuffers");
            cmd = new VkCommandBuffer(pp.get(0), ctx.device);
            check(vkCreateSemaphore(ctx.device, VkSemaphoreCreateInfo.calloc(stack).sType$Default(), null, lp), "vkCreateSemaphore");
            imageAvailable = lp.get(0);
            check(vkCreateFence(ctx.device, VkFenceCreateInfo.calloc(stack).sType$Default()
                    .flags(VK_FENCE_CREATE_SIGNALED_BIT), null, lp), "vkCreateFence");
            inFlight = lp.get(0);
        }

        createSizeDependent(VK_NULL_HANDLE);
    }

    public String deviceName() { return ctx.deviceName; }
    /** Samples gained since the camera stopped; reprojected pixels may hold more. */
    public int accumulatedSamples() { return accumulatedFrames * samplesPerFrame; }
    public int width() { return fbW; }
    public int height() { return fbH; }

    /** One training path per trainBlock x trainBlock pixels; a train rate of 1/16 gives 4. */
    private int trainBlock() { return Math.max(1, Math.round((float) Math.sqrt(1.0 / Math.max(trainRate, 1e-4f)))); }

    /**
     * Writes the per-frame part of the light buffer header: the OPT_* bits of raygen.rgen, the training block, the
     * cache bounce and indirect shift, the prefetch guard band and shift, and the catch-up samples.
     */
    private void writeLightOptions() {
        int bits = (lightImportance ? 1 : 0) | (cachePaths && ctx.atomicFloatAddSupported ? 2 : 0) | (restir ? 4 : 0)
                | (ser && ctx.invocationReorderSupported ? 8 : 0) | (rayStats ? 16 : 0) | (lightGrid ? 32 : 0) | (cachePrior ? 64 : 0)
                | (rasterPrimary ? 128 : 0) | (hasCutouts ? 256 : 0);
        lightBuffer.map().putInt(4, bits).putInt(8, trainBlock()).putInt(12, cacheBounce | indirectShift << 8)
                .putFloat(16, prefetch ? PREFETCH_GUARD : 1f).putFloat(20, prefetch ? prefetchShift[0] : 0f)
                .putFloat(24, prefetch ? prefetchShift[1] : 0f).putInt(28, catchUpSamples);
    }

    private boolean statsPending;
    /** Must match STAT_WARP_RATE in raygen.rgen. */
    private static final int STAT_WARP_RATE = 16;

    /** Reads and clears the ray counters of the last traced frame. */
    private void readRayStats() {
        if (!statsPending) return;
        ByteBuffer b = statsBuffer.map();
        for (int i = 0; i < lastStats.length; i++) {
            // only every 16th warp counts
            lastStats[i] = Integer.toUnsignedLong(b.getInt(i * 4)) * STAT_WARP_RATE;
            b.putInt(i * 4, 0);
        }
        statsPending = false;
    }

    /** The radiance cache that paths end in; its fields tune how fast it follows changing light. */
    public RadianceCache radianceCache() { return cache; }

    /**
     * Discards the pixel history and clears the radiance cache, whose light reflects the old settings. Not needed for
     * camera motion, which reprojects the history.
     */
    public void resetAccumulation() {
        accumulatedFrames = 0;
        historyValid = false;
        if (cache != null) cache.clear();
    }

    /** Whether every pixel has reached the sample limit. */
    public boolean idle() { return sampleLimitReached(); }

    /** Samples per pixel after which a pixel stops tracing; 0 means unlimited. */
    public int sampleLimit = 100;

    public boolean sampleLimitReached() {
        return sampleLimit > 0 && accumulatedFrames > 0 && accumulatedFrames >= framesToLimit();
    }

    /** Frames needed to reach the sample limit with a still camera, counting the catch-up frame. */
    public int framesToLimit() {
        double spp = samplesPerFrame, n = sampleLimit, first = Math.max(spp, catchUpSamples);
        return n <= first ? 1 : 1 + (int) Math.ceil((n - first) / spp);
    }
    /** Saves the next frame as PNG. */
    public void requestScreenshot(Path file) { pendingScreenshot = file; }

    /** Shifts the training guard band towards the side where new content is entering the view. */
    private void updatePrefetchShift(Camera camera) {
        camera.forward(curForward);
        prefetchShift[0] = prefetchShift[1] = 0f;
        if (hasPrevForward) {
            Vector4f c = new Vector4f(prevForward, 0f).mul(viewProj);
            if (c.w > 1e-4f) {
                float sx = -c.x / c.w * PREFETCH_FRAMES, sy = -c.y / c.w * PREFETCH_FRAMES;
                prefetchShift[0] = Math.max(-0.5f, Math.min(0.5f, sx));
                prefetchShift[1] = Math.max(-0.5f, Math.min(0.5f, sy));
            }
        }
        prevForward.set(curForward);
        hasPrevForward = true;
    }

    /**
     * Uploads meshes added since the last call and builds their BLAS. Meshes replaced in place get new buffers once
     * the GPU is idle; replaced meshes that were never uploaded are handled as new ones. Returns the number of new
     * triangles.
     */
    private long uploadNewMeshes() {
        long triangles = 0;
        if (!scene.replacedMeshes.isEmpty()) {
            vkDeviceWaitIdle(ctx.device);
            for (int i : scene.replacedMeshes) {
                if (i >= gpuMeshes.size()) continue;
                GpuMesh old = gpuMeshes.get(i);
                if (old.blas != null) old.blas.close();
                old.vertices.close();
                old.indices.close();
                Mesh mesh = scene.meshes.get(i);
                gpuMeshes.set(i, upload(mesh));
                meshBounds.set(i, mesh.bounds());
                triangles += mesh.triangleCount();
            }
            scene.replacedMeshes.clear();
        }
        for (int i = gpuMeshes.size(); i < scene.meshes.size(); i++) {
            Mesh mesh = scene.meshes.get(i);
            gpuMeshes.add(upload(mesh));
            meshBounds.add(mesh.bounds());
            triangles += mesh.triangleCount();
        }
        return triangles;
    }

    /** Uploads a mesh's vertices and indices to device memory and, with ray tracing, builds its BLAS. */
    private GpuMesh upload(Mesh mesh) {
        ByteBuffer vb = memAlloc(mesh.vertices.length * Float.BYTES);
        vb.asFloatBuffer().put(mesh.vertices);
        ByteBuffer ib = memAlloc(mesh.indices.length * Integer.BYTES);
        ib.asIntBuffer().put(mesh.indices);
        Buffer vertices = Buffer.deviceLocal(ctx, vb, GEOMETRY_USAGE);
        Buffer indices = Buffer.deviceLocal(ctx, ib, GEOMETRY_USAGE);
        memFree(vb);
        memFree(ib);
        AccelerationStructure blas = rt ? AccelerationStructure.buildBottomLevel(ctx, vertices, mesh.vertexCount(),
                Mesh.STRIDE, indices, mesh.triangleCount()) : null;
        return new GpuMesh(vertices, indices, blas);
    }

    private static int grow(int needed, int minimum) {
        int c = minimum;
        while (c < needed) c *= 2;
        return c;
    }

    /**
     * Grows the instance, material and light buffers and the TLAS to the scene's size, doubling, and recreates the
     * sun's flare pass, which binds the TLAS. Without ray tracing the composite pass gets an empty flare list instead.
     * Returns true if the descriptors must be rewritten.
     */
    private boolean ensureCapacities() {
        boolean realloc = false;
        int n = scene.instances.size();
        if (n > instanceCapacity || instanceInfoBuffer == null) {
            if (instanceInfoBuffer != null) {
                vkDeviceWaitIdle(ctx.device);
                if (tlas != null) { tlas.close(); tlasScratch.close(); tlasInstanceBuffer.close(); }
                instanceInfoBuffer.close(); deltaBuffer.close();
                instanceTransformBuffer.close();
            }
            instanceCapacity = grow(n, 256);
            instanceInfoBuffer = new Buffer(ctx, (long) instanceCapacity * INSTANCE_INFO_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
            deltaBuffer = new Buffer(ctx, (long) instanceCapacity * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
            instanceTransformBuffer = new Buffer(ctx, (long) instanceCapacity * INSTANCE_TRANSFORM_SIZE,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
            raster.setInstanceBuffers(instanceInfoBuffer, instanceTransformBuffer);
            movedLastUpdate = null;
            if (rt) {
                tlasInstanceBuffer = new Buffer(ctx, (long) instanceCapacity * TLAS_INSTANCE_SIZE,
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        Buffer.UPLOAD);
                long[] scratchSize = new long[1];
                tlas = AccelerationStructure.createTopLevel(ctx, tlasInstanceBuffer, instanceCapacity, scratchSize);
                tlasBuiltCount = -1;
                long align = ctx.minScratchOffsetAlignment;
                tlasScratch = new Buffer(ctx, scratchSize[0] + align,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, Buffer.DEVICE);
                tlasScratchAddress = VkUtil.align(tlasScratch.address, align);
            }
            realloc = true;
        }
        int m = scene.materials.size();
        if (m > materialCapacity || materialBuffer == null) {
            if (materialBuffer != null) { vkDeviceWaitIdle(ctx.device); materialBuffer.close(); }
            materialCapacity = grow(m, 64);
            materialBuffer = new Buffer(ctx, (long) materialCapacity * Material.GPU_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
            realloc = true;
        }
        int l = scene.lights.size();
        if (l > lightCapacity || lightBuffer == null) {
            if (lightBuffer != null) { vkDeviceWaitIdle(ctx.device); lightBuffer.close(); }
            lightCapacity = grow(l, 32);
            lightBuffer = new Buffer(ctx, 32 + 32L * (lightCapacity + LightGrid.entries()), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
            memSet(lightBuffer.map(), 0);
            realloc = true;
        }
        if (realloc && rt) {
            if (flare != null) flare.close();
            flare = new LensFlare(ctx, compiler, tlas.handle, 1);
            if (descriptorSets[0] != 0) post.updateFlareBuffer(flare.buffer);
        }
        if (!rt && noFlares == null) {
            noFlares = new Buffer(ctx, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
            memSet(noFlares.map(), 0);
        }
        return realloc;
    }

    /**
     * Writes per instance the vertex and index buffer addresses, the material and the dynamic flag. The motion
     * matrices and flags of all instances are rewritten on the next {@link #updateDynamics}.
     */
    private void writeInstanceInfo() {
        movedLastUpdate = null;
        ByteBuffer info = instanceInfoBuffer.map();
        for (int i = 0; i < scene.instances.size(); i++) {
            Scene.Instance inst = scene.instances.get(i);
            GpuMesh gm = gpuMeshes.get(inst.mesh);
            info.putLong(i * INSTANCE_INFO_SIZE, gm.vertices.address)
                    .putLong(i * INSTANCE_INFO_SIZE + 8, gm.indices.address)
                    .putInt(i * INSTANCE_INFO_SIZE + 16, inst.material)
                    .putInt(i * INSTANCE_INFO_SIZE + 20, inst.isDynamic() ? 1 : 0);
        }
    }

    private void writeMaterials() {
        ByteBuffer mats = materialBuffer.map();
        for (int i = 0; i < scene.materials.size(); i++) scene.materials.get(i).write(mats, i * Material.GPU_SIZE);
    }

    private void writeLights() { writeLights(true); }

    /** Writes the lights and, if requested, rebuilds the light grid. The header options are written per frame. */
    private void writeLights(boolean rebuildGrid) {
        ByteBuffer lb = lightBuffer.map();
        lb.putInt(0, scene.lights.size());
        for (int i = 0; i < scene.lights.size(); i++) {
            Scene.SphereLight l = scene.lights.get(i);
            int o = 32 + i * 32;
            lb.putFloat(o, l.position().x).putFloat(o + 4, l.position().y).putFloat(o + 8, l.position().z).putFloat(o + 12, l.radius())
                    .putFloat(o + 16, l.radiance().x).putFloat(o + 20, l.radiance().y).putFloat(o + 24, l.radiance().z).putFloat(o + 28, l.shape().ordinal());
        }
        if (rebuildGrid) grid.write(scene.lights, lb, 32 + scene.lights.size() * 32);
    }

    private void markSceneSeen() {
        seenStructure = scene.structureRevision;
        seenTransforms = scene.transformRevision;
        seenLights = scene.lightRevision;
        seenLightMoves = scene.lightMoveRevision;
        seenEnvironment = scene.environmentRevision;
        seenTextures = scene.textureRevision;
    }

    /**
     * Uploads scene changes since the last frame. Must run after the previous frame has finished. Restarts the
     * image, except for soft changes (objects added, removed or recolored while playing), which only restart the
     * influence zones of the changed objects. Lights that only moved keep the light grid: its lists only steer the
     * choice, the exact weight is computed per shading point, and rebuilding thousands of cells per frame is costly.
     */
    private void applySceneChanges() {
        boolean structure = scene.structureRevision != seenStructure, transforms = scene.transformRevision != seenTransforms,
                lights = scene.lightRevision != seenLights, env = scene.environmentRevision != seenEnvironment,
                textures = scene.textureRevision != seenTextures;
        boolean lightMoves = scene.lightMoveRevision != seenLightMoves;
        if (!(structure || transforms || lights || env || textures)) {
            if (lightMoves) {
                writeLights(false);
                seenLightMoves = scene.lightMoveRevision;
            }
            return;
        }
        boolean realloc = false;
        if (textures) {
            vkDeviceWaitIdle(ctx.device);
            uploadAlphaMasks();
            vt.close();
            vt = new VirtualTextures(ctx, scene.textureSets,
                    scene.textureSets.isEmpty() ? 1 : texturePoolSlots(), scene.textureCompression, compiler);
            vt.similarityScale = scene.textureSimilarity;
            realloc = true;
        }
        if (env) {
            vkDeviceWaitIdle(ctx.device);
            envTexture.close();
            envTexture = new EnvironmentTexture(ctx, scene.environment);
            exposure = scene.exposure;
            realloc = true;
        }
        if (structure) uploadNewMeshes();
        realloc |= ensureCapacities();
        if (structure) {
            writeInstanceInfo();
            writeMaterials();
            writeRasterDraws();
        }
        if (structure || lights) writeLights();
        if (structure || transforms) {
            for (Scene.Instance i : scene.instances) if (!i.scripted) i.prevTransform.set(i.transform);
            writeTlasInstances();
            tlasDirty = true;
        }
        if (realloc) {
            vkDeviceWaitIdle(ctx.device);
            writeSceneDescriptors();
        }
        markSceneSeen();
        boolean soft = scene.softChange && !textures && !env && !realloc;
        scene.softChange = false;
        if (!soft) {
            scene.changedBoxes.clear();
            resetAccumulation();
        }
    }

    /**
     * Rewrites the descriptors of all scene resources that can be replaced at runtime, for both parities of the ray
     * tracing sets, and passes them to the raster passes.
     */
    private void writeSceneDescriptors() {
        if (alphaMaskBuffer == null) uploadAlphaMasks();
        rasterShade.writeScene(instanceInfoBuffer, materialBuffer, instanceTransformBuffer, vt, envTexture);
        raster.setAlphaBuffers(materialBuffer, alphaMaskBuffer);
        if (!rt) return;
        try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSetAccelerationStructureKHR asWrite = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack)
                    .sType$Default()
                    .pAccelerationStructures(stack.longs(tlas.handle));
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(30, stack);
            VkDescriptorImageInfo.Buffer atlasInfo = VkDescriptorImageInfo.calloc(6, stack);
            for (int c = 0; c < 6; c++) atlasInfo.get(c).sampler(vt.sampler).imageView(vt.atlasView[c]).imageLayout(VK_IMAGE_LAYOUT_GENERAL);
            Buffer[] vtBuffers = {vt.pageTable, vt.feedback, vt.setInfo};
            for (int p = 0; p < 2; p++) {
                long set = descriptorSets[p];
                writes.get(28 + p).sType$Default().dstSet(set).dstBinding(31).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(instanceTransformBuffer.handle).range(VK_WHOLE_SIZE));
                writes.get(26 + p).sType$Default().dstSet(set).dstBinding(25).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(alphaMaskBuffer.handle).range(VK_WHOLE_SIZE));
                writes.get(24 + p).sType$Default().dstSet(set).dstBinding(22).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(statsBuffer.handle).range(VK_WHOLE_SIZE));
                writes.get(22 + p).sType$Default().dstSet(set).dstBinding(21).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(VkDescriptorImageInfo.calloc(1, stack).sampler(envTexture.sampler).imageView(envTexture.view)
                                .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
                writes.get(20 + p).sType$Default().dstSet(set).dstBinding(20).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(lightBuffer.handle).range(VK_WHOLE_SIZE));
                writes.get(12 + p * 4).sType$Default().dstSet(set).dstBinding(16).descriptorCount(6)
                        .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(atlasInfo);
                for (int k = 0; k < 3; k++) {
                    writes.get(13 + p * 4 + k).sType$Default().dstSet(set).dstBinding(17 + k).descriptorCount(1)
                            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                            .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(vtBuffers[k].handle).range(VK_WHOLE_SIZE));
                }
                writes.get(8 + p * 2).sType$Default().dstSet(set).dstBinding(14).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(dynamicBuffer.handle).range(VK_WHOLE_SIZE));
                writes.get(9 + p * 2).sType$Default().dstSet(set).dstBinding(15).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(deltaBuffer.handle).range(VK_WHOLE_SIZE));
                writes.get(6 + p).sType$Default().dstSet(set).dstBinding(11).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(cache.buffer.handle).range(VK_WHOLE_SIZE));
                writes.get(p * 3).sType$Default().pNext(asWrite.address()).dstSet(set).dstBinding(0)
                        .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
                writes.get(p * 3 + 1).sType$Default().dstSet(set).dstBinding(3).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(instanceInfoBuffer.handle).range(VK_WHOLE_SIZE));
                writes.get(p * 3 + 2).sType$Default().dstSet(set).dstBinding(4).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(materialBuffer.handle).range(VK_WHOLE_SIZE));
            }
            vkUpdateDescriptorSets(ctx.device, writes, null);
        }
    }

    /** Alpha masks of all texture sets: per set (word offset, side) or (~0, 0), followed by the mask bits. */
    private Buffer alphaMaskBuffer;

    private void uploadAlphaMasks() {
        int sets = scene.textureSets.size();
        long words = 2L * Math.max(sets, 1);
        for (TextureSet t : scene.textureSets) if (t.alphaMask != null) words += t.alphaMask.length;
        if (alphaMaskBuffer != null) alphaMaskBuffer.close();
        alphaMaskBuffer = new Buffer(ctx, words * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        ByteBuffer b = alphaMaskBuffer.map();
        int at = 2 * Math.max(sets, 1);
        int masks = 0;
        for (int i = 0; i < sets; i++) {
            TextureSet t = scene.textureSets.get(i);
            if (t.alphaMask == null) {
                b.putInt(i * 8, -1).putInt(i * 8 + 4, 0);
                continue;
            }
            b.putInt(i * 8, at).putInt(i * 8 + 4, t.alphaSize);
            for (int w : t.alphaMask) { b.putInt(at * 4, w); at++; }
            masks++;
        }
        if (sets == 0) b.putInt(0, -1).putInt(4, 0);
        if (masks > 0) System.out.printf("[lumvex] alpha masks: %d texture sets with cut-outs (%.1f MB)%n", masks, words * 4 / 1048576.0);
    }

    /** Writes the TLAS instances and the matching object-to-world matrices for the raster primary surface. */
    private void writeTlasInstances() { writeTlasInstances(false); }

    /**
     * TLAS mask of an instance: 0x01 opaque, 0x02 glass, 0x04 light proxies, 0x10 alpha-tested (leaves), plus 0x08
     * for the raster pass's cutout rays. Shadow rays test 0x11; rays that skip leaves 0x01.
     */
    private static int instanceMask(Material mat) {
        return mat.isLightProxy() ? 0x04 : mat.transmission() > 0.5f ? 0x02 : mat.isAlphaTested() ? 0x18 : 0x01;
    }

    /**
     * Writes the object-to-world matrices (mat4x3, column-major) and, with ray tracing, the TLAS instances; with
     * {@code movingOnly} only those of moving instances; the others are unchanged since the last full write. Only
     * alpha-tested instances run the any-hit shader; all others are forced opaque.
     */
    private void writeTlasInstances(boolean movingOnly) {
        int n = scene.instances.size();
        VkAccelerationStructureInstanceKHR.Buffer insts = rt
                ? VkAccelerationStructureInstanceKHR.create(memAddress(tlasInstanceBuffer.map()), n) : null;
        ByteBuffer transforms = instanceTransformBuffer.map();
        boolean cutouts = hasCutouts && movingOnly;
        for (int i = 0; i < n; i++) {
            Scene.Instance inst = scene.instances.get(i);
            if (movingOnly && !inst.isDynamic()) continue;
            for (int c = 0; c < 4; c++) {
                for (int r = 0; r < 3; r++) {
                    transforms.putFloat(i * INSTANCE_TRANSFORM_SIZE + (c * 3 + r) * 4, inst.transform.getRowColumn(r, c));
                }
            }
            Material mat = scene.materials.get(inst.material);
            cutouts |= mat.isAlphaTested();
            if (!rt) continue;
            VkAccelerationStructureInstanceKHR vi = insts.get(i);
            VkTransformMatrixKHR t = vi.transform();
            for (int r = 0; r < 3; r++) {
                for (int c = 0; c < 4; c++) t.matrix(r * 4 + c, inst.transform.getRowColumn(r, c));
            }
            int mask = instanceMask(mat);
            int flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR
                    | (mat.isAlphaTested() ? VK_GEOMETRY_INSTANCE_FORCE_NO_OPAQUE_BIT_KHR : VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR);
            vi.instanceCustomIndex(i)
                    .mask(mask)
                    .instanceShaderBindingTableRecordOffset(0)
                    .flags(flags)
                    .accelerationStructureReference(gpuMeshes.get(inst.mesh).blas.address);
        }
        hasCutouts = cutouts;
    }

    /**
     * Instances whose motion matrix and animated flag were written as moving in the last update; the others hold the
     * identity and 0, so they are only rewritten when they start or stop moving. Reset to "unknown" (null) whenever the
     * buffers are reallocated or the scene's structure changes.
     */
    private boolean[] movedLastUpdate;

    /**
     * Writes the per-instance motion matrices and the influence zones of what changed this frame (see
     * {@link #dynamicBuffer}), at most {@link #MAX_DYNAMIC}:
     * <ul>
     * <li>moving instances: their swept box, padded a little for soft shadows. An emitter is approximated by a
     * sphere of the same surface as its box (the half diagonal would turn a neon tube into a huge ball of light).
     * Untextured instances only reproject their translation: they look the same wherever they turn, so a rolling ball
     * keeps the history of the same lighting instead of dragging it along with the rotation;</li>
     * <li>lights that moved, with the light's power;</li>
     * <li>objects removed or changed by a soft rebuild, once at their old place; a removed emitter's power is
     * unknown, so its whole zone restarts (power 0).</li>
     * </ul>
     */
    private void updateDynamics(boolean animated) {
        ByteBuffer dyn = dynamicBuffer.map();
        ByteBuffer delta = deltaBuffer.map();
        ByteBuffer info = instanceInfoBuffer.map();
        Matrix4f inv = new Matrix4f(), d = new Matrix4f();
        int count = 0;
        int n = scene.instances.size();
        boolean all = movedLastUpdate == null || movedLastUpdate.length < n;
        if (all) movedLastUpdate = new boolean[Math.max(n, 1)];
        for (int i = 0; i < n; i++) {
            Scene.Instance inst = scene.instances.get(i);
            boolean moving = animated && inst.isDynamic();
            if (!all && !moving && !movedLastUpdate[i]) continue;
            movedLastUpdate[i] = moving;
            if (animated && inst.isDynamic() && scene.materials.get(inst.material).textureSet() < 0) {
                Vector3f now = inst.transform.getTranslation(new Vector3f());
                Vector3f before = inst.prevTransform.getTranslation(new Vector3f());
                d.translation(before.sub(now));
            } else if (animated && inst.isDynamic()) {
                inst.prevTransform.mul(inst.transform.invert(inv), d);
            } else {
                d.identity();
            }
            d.get(i * 64, delta);
            info.putInt(i * INSTANCE_INFO_SIZE + 20, animated && inst.isDynamic() ? 1 : 0);
            if (!animated || !inst.isDynamic() || count >= MAX_DYNAMIC) continue;
            float[] b = meshBounds.get(inst.mesh);
            float[] box = {Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE};
            for (Matrix4f m : new Matrix4f[]{inst.transform, inst.prevTransform}) {
                for (int c = 0; c < 8; c++) {
                    Vector3f p = m.transformPosition(new Vector3f(
                            (c & 1) == 0 ? b[0] : b[3], (c & 2) == 0 ? b[1] : b[4], (c & 4) == 0 ? b[2] : b[5]));
                    box[0] = Math.min(box[0], p.x); box[1] = Math.min(box[1], p.y); box[2] = Math.min(box[2], p.z);
                    box[3] = Math.max(box[3], p.x); box[4] = Math.max(box[4], p.y); box[5] = Math.max(box[5], p.z);
                }
            }
            float halfDiag = 0.5f * (float) Math.sqrt(sq(box[3] - box[0]) + sq(box[4] - box[1]) + sq(box[5] - box[2]));
            Material mat = scene.materials.get(inst.material);
            float emission = luminance(mat.emission());
            float power = 0;
            if (emission > 0) {
                float sx = box[3] - box[0], sy = box[4] - box[1], sz = box[5] - box[2];
                power = (float) Math.PI * emission * (float) ((sx * sy + sy * sz + sx * sz) / (2 * Math.PI));
            }
            putZone(dyn, count++, box, 0.05f, zoneRadius(halfDiag, power), (power > 0 ? -1f : 1f) * (i + 1), power);
        }
        for (int i = 0; i < scene.lights.size(); i++) {
            Scene.SphereLight l = scene.lights.get(i);
            Vector3f now = l.position();
            if (i >= previousLightPositions.size()) previousLightPositions.add(new Vector3f(now));
            Vector3f before = previousLightPositions.get(i);
            boolean moved = !before.equals(now, 1e-5f);
            if (animated && moved && count < MAX_DYNAMIC) {
                float r = l.radius();
                float[] box = {Math.min(before.x, now.x) - r, Math.min(before.y, now.y) - r, Math.min(before.z, now.z) - r,
                        Math.max(before.x, now.x) + r, Math.max(before.y, now.y) + r, Math.max(before.z, now.z) + r};
                float power = (float) Math.PI * luminance(l.radiance()) * r * r;
                putZone(dyn, count++, box, 0.02f, zoneRadius(r, power), -0.5f, power);
            }
            before.set(now);
        }
        while (previousLightPositions.size() > scene.lights.size()) previousLightPositions.remove(previousLightPositions.size() - 1);
        for (float[] b : scene.changedBoxes) {
            if (!animated || count >= MAX_DYNAMIC) break;
            float halfDiag = 0.5f * (float) Math.sqrt(sq(b[3] - b[0]) + sq(b[4] - b[1]) + sq(b[5] - b[2]));
            putZone(dyn, count++, b, 0.05f, Math.min(12f, b[6] > 0 ? 4f * halfDiag + 1f : 1.5f * halfDiag),
                    b[6] > 0 ? -0.5f : 0.5f, 0);
        }
        if (animated) scene.changedBoxes.clear();
        dyn.putInt(0, count);
    }

    /**
     * Influence radius of a moving object. Indirect light and contact shadows reach about 1.5x its size. An emitter
     * reaches as far as its irradiance (power / d^2) is a few percent of the sun's; in a dark scene that is up to the
     * 12 m limit, and the shader then decides per pixel whether the change is visible there (see dynamicInfluence).
     */
    private float zoneRadius(float halfDiag, float power) {
        float radius = 1.5f * halfDiag;
        if (power > 0) radius = Math.max(radius, (float) Math.sqrt(power / (0.03f * Math.max(scene.sunIntensity, 0.1f))));
        return Math.min(radius, 12f);
    }

    /** Writes one influence zone, see {@link #dynamicBuffer}. */
    private static void putZone(ByteBuffer dyn, int index, float[] box, float pad, float radius, float owner, float power) {
        int o = 16 + index * DYNAMIC_STRIDE;
        dyn.putFloat(o, box[0] - pad).putFloat(o + 4, box[1] - pad).putFloat(o + 8, box[2] - pad).putFloat(o + 12, radius)
                .putFloat(o + 16, box[3] + pad).putFloat(o + 20, box[4] + pad).putFloat(o + 24, box[5] + pad)
                .putFloat(o + 28, owner).putFloat(o + 32, power).putFloat(o + 36, 0).putFloat(o + 40, 0).putFloat(o + 44, 0);
    }

    private static float luminance(Vector3f c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }

    private static float sq(float v) { return v * v; }

    /** Size of the visible image. */
    private int fbW, fbH;
    /**
     * Fraction of extra border traced around the visible image, split between both sides. Pixels that rotate into
     * view then already have history, and the denoiser has neighbors at the screen edges. 0.1 costs 21% more pixels.
     */
    private float overscan = 0.10f;
    /** Overscan or render scale changed; the traced images are recreated before the next frame. */
    private boolean tracedSizeChanged;
    /**
     * Requested fraction of the window's pixels traced per axis, 0.25 to 1, or 0 for automatic, see
     * {@link #setRenderScale}.
     */
    private float renderScale = Float.parseFloat(System.getProperty("lumvex.renderscale", "0"));
    /** Traced pixels of the visible image that the automatic render scale aims for. */
    private static final double AUTO_PIXEL_BUDGET = 1920.0 * 1080.0;
    /** Size of the traced visible image, before the border is added. */
    private int vW, vH;
    /** Size of the traced image including the border, and the border in pixels. */
    private int rW, rH, marginX, marginY;
    /** Clip-space scale from the visible to the traced frustum, and its inverse. */
    private final Matrix4f overscanScale = new Matrix4f(), overscanScaleInv = new Matrix4f();
    /** View-projection of the visible image; lens flares are placed with it. */
    private final Matrix4f viewProjShown = new Matrix4f();

    public float overscan() { return overscan; }

    /** Sets the overscan fraction (0 to 0.5); applied before the next frame. */
    public void setOverscan(float fraction) {
        float f = Math.max(0f, Math.min(0.5f, fraction));
        if (f != overscan) {
            overscan = f;
            tracedSizeChanged = true;
        }
    }

    /**
     * Sets the fraction of the window's pixels that is path traced, per axis (0.25 to 1); the composite pass scales
     * the denoised image up to the window. 0 chooses it automatically: windows with more than 1920 x 1080 pixels,
     * such as on high-DPI displays, are traced at about that many pixels. Applied before the next frame.
     */
    public void setRenderScale(float scale) {
        float s = scale <= 0 ? 0 : Math.clamp(scale, 0.25f, 1f);
        if (s != renderScale) {
            renderScale = s;
            tracedSizeChanged = true;
        }
    }

    /** The render scale in use, see {@link #setRenderScale}. */
    public float renderScale() { return fbW == 0 ? 1 : (float) vW / fbW; }

    private float effectiveRenderScale(int width, int height) {
        if (renderScale > 0) return renderScale;
        return (float) Math.min(1, Math.sqrt(AUTO_PIXEL_BUDGET / ((double) width * height)));
    }

    /**
     * (Re)creates the swapchain and all images that depend on the window size (history, filter, output and
     * visibility images, sized for the render scale plus the overscan border) and rebinds them. The filtered
     * histories are cleared, alpha 0 marking "no history yet". Restarts the image.
     */
    private void createSizeDependent(long oldSwapchain) {
        int[] fb = window.framebufferSize();
        swapchain = window.headless() ? null : new Swapchain(ctx, fb[0], fb[1], vsync, oldSwapchain);
        fbW = swapchain != null ? swapchain.width : fb[0];
        fbH = swapchain != null ? swapchain.height : fb[1];
        float scale = effectiveRenderScale(fbW, fbH);
        vW = Math.max(1, Math.round(fbW * scale));
        vH = Math.max(1, Math.round(fbH * scale));
        marginX = Math.round(vW * overscan / 2);
        marginY = Math.round(vH * overscan / 2);

        rW = vW + 2 * marginX;
        rH = vH + 2 * marginY;
        overscanScale.scaling((float) vW / rW, (float) vH / rH, 1f);
        overscanScaleInv.scaling((float) rW / vW, (float) rH / vH, 1f);
        tracedSizeChanged = false;
        if (vW != fbW) System.out.printf("[lumvex] tracing %d x %d for a %d x %d window%n", vW, vH, fbW, fbH);
        int w = rW, h = rH;
        for (int p = 0; p < 2; p++) {
            history[p] = new PostProcess.HistoryImages(
                    new StorageImage(ctx, w, h, VK_FORMAT_R32G32B32A32_SFLOAT),
                    new StorageImage(ctx, w, h, VK_FORMAT_R32G32B32A32_SFLOAT),
                    new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT),
                    new StorageImage(ctx, w, h, VK_FORMAT_R32G32B32A32_SFLOAT),
                    new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT),
                    new StorageImage(ctx, w, h, VK_FORMAT_R32G32B32A32_SFLOAT),
                    new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT));
        }
        for (int p = 0; p < 2; p++) reservoir[p] = new StorageImage(ctx, w, h, VK_FORMAT_R32G32B32A32_SFLOAT);
        for (int p = 0; p < 2; p++) filtered[p] = new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT);
        raster.resize(w, h);
        for (int p = 0; p < 2; p++) specFiltered[p] = new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT);
        ctx.submitImmediate(c -> {
            try (MemoryStack st = stackPush()) {
                VkClearColorValue zero = VkClearColorValue.calloc(st);
                VkImageSubresourceRange range = VkImageSubresourceRange.calloc(st).aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
                for (StorageImage f : filtered) vkCmdClearColorImage(c, f.image, VK_IMAGE_LAYOUT_GENERAL, zero, range);
                for (StorageImage f : specFiltered) vkCmdClearColorImage(c, f.image, VK_IMAGE_LAYOUT_GENERAL, zero, range);
            }
        });
        pingA = new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT);
        pingB = new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT);
        specPingA = new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT);
        specPingB = new StorageImage(ctx, w, h, VK_FORMAT_R16G16B16A16_SFLOAT);
        outputImage = new StorageImage(ctx, fbW, fbH, VK_FORMAT_R8G8B8A8_UNORM);
        readback = new Buffer(ctx, (long) fbW * fbH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, Buffer.READBACK);

        if (rt) try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(32, stack);
            int i = 0;
            for (int p = 0; p < 2; p++) {
                PostProcess.HistoryImages cur = history[p], prev = history[1 - p];
                StorageImage[] images = {cur.color(), cur.guideNormalDepth(), cur.guideAlbedo(), cur.meta(),
                        prev.color(), prev.guideNormalDepth(), prev.guideAlbedo(), prev.meta(), cur.motion(),
                        reservoir[p], reservoir[1 - p], cur.specular(), prev.specular(), cur.specGuide(), prev.specGuide(),
                        raster.visibility};
                int[] bindings = {1, 2, 5, 6, 7, 8, 9, 10, 12, 23, 24, 26, 27, 28, 29, 30};
                for (int k = 0; k < images.length; k++) {
                    writes.get(i++).sType$Default().dstSet(descriptorSets[p]).dstBinding(bindings[k]).descriptorCount(1)
                            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                            .pImageInfo(VkDescriptorImageInfo.calloc(1, stack).imageView(images[k].view).imageLayout(VK_IMAGE_LAYOUT_GENERAL));
                }
            }
            vkUpdateDescriptorSets(ctx.device, writes, null);
        }
        post.updateDescriptors(history, pingA, pingB, outputImage, rt ? flare.buffer : noFlares, filtered, uiBuffer,
                specPingA, specPingB, specFiltered);
        rasterShade.writeImages(history, raster.visibility);
        if (rt) flare.resetHistory();
        resetAccumulation();
    }

    private void destroySizeDependent(boolean destroySwapchain) {
        readback.close();
        for (StorageImage r : reservoir) r.close();
        for (StorageImage f : filtered) f.close();
        for (StorageImage f : specFiltered) f.close();
        outputImage.close();
        specPingB.close();
        specPingA.close();
        pingB.close();
        pingA.close();
        for (PostProcess.HistoryImages h : history) {
            h.color().close();
            h.guideNormalDepth().close();
            h.guideAlbedo().close();
            h.meta().close();
            h.motion().close();
            h.specular().close();
            h.specGuide().close();
        }
        if (destroySwapchain && swapchain != null) swapchain.close();
    }

    /**
     * Denoiser iterations for the current sample count. The filter strength follows the noise anyway, so it is
     * reduced after 64 frames and skipped after 256, unless a sample limit is set.
     */
    public int effectiveDenoiseIterations() {
        return effectiveDenoiseIterations(Math.max(1, sampleLimitReached() ? accumulatedFrames : accumulatedFrames + 1));
    }

    private int effectiveDenoiseIterations(int frames) {
        // With a sample limit the filter keeps running; switching it off would make the final image visibly pop.
        if (debugMode == 3 || sampleLimit > 0) return denoiseIterations;
        if (frames >= 256) return 0;
        if (frames >= 64) return Math.max(1, denoiseIterations - 2);
        return denoiseIterations;
    }

    // Smoothed GPU times in ms.
    public double gpuTraceMs() { return gpuTraceMs; }
    public double gpuDenoiseMs() { return gpuDenoiseMs; }
    public double gpuTotalMs() { return gpuTotalMs; }

    /** Smoothed GPU time of the radiance cache training pass, part of the trace time. */
    public double gpuTrainMs() { return gpuTrainMs; }
    private int tlasBuilds;
    /** Number of TLAS builds; moving instances cause one per frame. */
    public int tlasBuilds() { return tlasBuilds; }

    /**
     * Reads the GPU timestamps of the last frame into {@link #lastGpuMs} and the smoothed times, and prints them every
     * 30 frames with {@code -Dlumvex.gputimes=true}.
     */
    private void readTimestamps(MemoryStack stack) {
        if (!timestampsPending) return;
        timestampsPending = false;
        LongBuffer ts = stack.mallocLong(TS_COUNT);
        if (vkGetQueryPoolResults(ctx.device, queryPool, 0, TS_COUNT, ts, 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
        double p = ctx.timestampPeriodNs / 1e6;
        lastGpuMs[0] = (ts.get(TS_TRACE_END) - ts.get(TS_TRACE_BEGIN)) * p;
        lastGpuMs[1] = (ts.get(TS_DENOISE_END) - ts.get(TS_TRACE_END)) * p;
        lastGpuMs[2] = (ts.get(TS_END) - ts.get(TS_START)) * p;
        lastGpuMs[3] = (ts.get(TS_TRAIN_END) - ts.get(TS_TRACE_BEGIN)) * p;
        gpuTraceMs = smooth(gpuTraceMs, lastGpuMs[0]);
        gpuTrainMs = smooth(gpuTrainMs, lastGpuMs[3]);
        gpuDenoiseMs = smooth(gpuDenoiseMs, lastGpuMs[1]);
        gpuTotalMs = smooth(gpuTotalMs, lastGpuMs[2]);
        if (GPU_TIMES && frameCounter % 30 == 0) {
            System.out.printf("[gpu] raster+train %.1f ms, trace %.1f ms, denoise %.1f ms, total %.1f ms%n",
                    gpuTrainMs, gpuTraceMs - gpuTrainMs, gpuDenoiseMs, gpuTotalMs);
        }
    }

    /** {@code -Dlumvex.gputimes=true} prints the smoothed GPU times of the passes every 30 frames. */
    private static final boolean GPU_TIMES = Boolean.getBoolean("lumvex.gputimes");

    private static double smooth(double old, double value) {
        return old == 0 ? value : old * 0.9 + value * 0.1;
    }

    private void timestamp(int stage, int index) {
        if (ctx.timestampsSupported) vkCmdWriteTimestamp(cmd, stage, queryPool, index);
    }

    /** CPU times in ms of the last frame: fence wait, acquire, present, dynamics, VT update, recording. */
    public final double[] lastCpuMs = new double[6];
    public int swapchainRecreations;
    public int lastPresentResult;

    /** Handles a resize; blocks while the window is minimized. */
    public void recreateSwapchain() {
        swapchainRecreations++;
        int[] fb = window.framebufferSize();
        while (fb[0] == 0 || fb[1] == 0) {
            window.waitWhileEmpty();
            fb = window.framebufferSize();
        }
        vkDeviceWaitIdle(ctx.device);
        Swapchain old = swapchain;
        destroySizeDependent(false);
        createSizeDependent(old.handle);
        old.close();
    }

    private void applyTracedSizeChange() {
        vkDeviceWaitIdle(ctx.device);
        if (swapchain != null) {
            recreateSwapchain();
        } else {
            destroySizeDependent(false);
            createSizeDependent(VK_NULL_HANDLE);
        }
    }

    /**
     * Renders and presents one frame: waits for the previous one, uploads scene and UI changes, updates motion and
     * influence zones, records and submits the frame and presents it. With {@code animated} objects moved this frame;
     * their history is kept and only their influence zones restart. Camera motion keeps the history through
     * reprojection and only restarts the still-camera counter. At the sample limit nothing is traced and only the
     * denoiser and the composite run, which keeps the UI live.
     */
    public void render(Camera camera, boolean animated) {
        try (MemoryStack stack = stackPush()) {
            long s0 = System.nanoTime();
            vkWaitForFences(ctx.device, inFlight, true, Long.MAX_VALUE);
            long s1 = System.nanoTime();

            if (tracedSizeChanged) {
                applyTracedSizeChange();
                return;
            }
            IntBuffer ip = stack.mallocInt(1);
            int acquire = VK_SUCCESS;
            if (swapchain != null) acquire = vkAcquireNextImageKHR(ctx.device, swapchain.handle, Long.MAX_VALUE, imageAvailable, VK_NULL_HANDLE, ip);
            else ip.put(0, 0);
            long s2 = System.nanoTime();
            lastCpuMs[0] = (s1 - s0) / 1e6;
            lastCpuMs[1] = (s2 - s1) / 1e6;
            if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapchain();
                return;
            }
            if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) check(acquire, "vkAcquireNextImageKHR");
            int imageIndex = ip.get(0);
            vkResetFences(ctx.device, inFlight);
            readTimestamps(stack);
            applySceneChanges();
            applyUi();
            readPick();
            readRayStats();
            writeLightOptions();

            if (rasterDrawsMode != renderMode) {
                vkDeviceWaitIdle(ctx.device);
                writeRasterDraws();
            }
            if (animated) {
                writeTlasInstances(true);
                if (dynamicHistory) accumulatedFrames = 0;
                else resetAccumulation();
            }
            long d0 = System.nanoTime();
            updateDynamics(animated && dynamicHistory);
            lastCpuMs[3] = (System.nanoTime() - d0) / 1e6;

            camera.viewProjection((float) fbW / fbH, viewProjShown);
            overscanScale.mul(viewProjShown, viewProj);
            currentCamera = camera;
            boolean moved = hasPrevViewProj && !viewProj.equals(prevViewProj, 1e-6f);
            if (moved) {
                if (reprojection) accumulatedFrames = 0;
                else resetAccumulation();
            }

            boolean trace = !idle();
            updatePrefetchShift(camera);
            int flags = (historyValid ? 1 : 0) | (moved ? 2 : 0)
                    | (animated && dynamicHistory ? FLAG_ANIMATED : 0) | (scene.night && scene.environment == null ? 16 : 0) | (scene.environment != null ? 32 : 0);
            Path screenshot = pendingScreenshot;
            pendingScreenshot = null;

            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack).sType$Default()
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT));
            int shown = trace ? parity : 1 - parity;
            long r0 = System.nanoTime();
            if (animated || tlasDirty) tlasBuilds++;
            recordFrame(stack, imageIndex, animated || tlasDirty, trace, shown, flags, screenshot != null, camera);
            tlasDirty = false;
            lastCpuMs[5] = (System.nanoTime() - r0) / 1e6;
            check(vkEndCommandBuffer(cmd), "vkEndCommandBuffer");

            VkSubmitInfo submit = VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(stack.pointers(cmd));
            if (swapchain != null) {
                submit.waitSemaphoreCount(1)
                        .pWaitSemaphores(stack.longs(imageAvailable))
                        .pWaitDstStageMask(stack.ints(VK_PIPELINE_STAGE_TRANSFER_BIT))
                        .pSignalSemaphores(stack.longs(swapchain.renderFinished[imageIndex]));
            }
            check(vkQueueSubmit(ctx.queue, submit, inFlight), "vkQueueSubmit");
            timestampsPending = ctx.timestampsSupported;
            statsPending = trace;

            long s3 = System.nanoTime();
            int pr = VK_SUCCESS;
            if (swapchain != null) {
                VkPresentInfoKHR present = VkPresentInfoKHR.calloc(stack)
                        .sType$Default()
                        .pWaitSemaphores(stack.longs(swapchain.renderFinished[imageIndex]))
                        .swapchainCount(1)
                        .pSwapchains(stack.longs(swapchain.handle))
                        .pImageIndices(stack.ints(imageIndex));
                pr = vkQueuePresentKHR(ctx.queue, present);
                if (pr == VK_SUCCESS) window.framePresented();
            }
            lastCpuMs[2] = (System.nanoTime() - s3) / 1e6;
            if (pr != VK_SUCCESS) lastPresentResult = pr;
            if (trace) {
                accumulatedFrames++;
                frameCounter++;
                historyValid = true;
                parity ^= 1;
                prevViewProj.set(viewProj);
                hasPrevViewProj = true;
            }

            if (screenshot != null) {
                vkWaitForFences(ctx.device, inFlight, true, Long.MAX_VALUE);
                saveScreenshot(screenshot);
            }

            if (swapchain != null && (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR || window.consumeResized())) {
                recreateSwapchain();
            } else if (pr != VK_SUCCESS) {
                check(pr, "vkQueuePresentKHR");
            }
        }
    }

    /**
     * Records a frame into the command buffer:
     * <ol>
     * <li>the TLAS build, or a refit when only transforms changed (a full build after a structure change and every
     * {@link #REFIT_LIMIT} refits);</li>
     * <li>texture streaming, then the raster pre-pass in the hybrid and raster modes;</li>
     * <li>the raster mode's shading pass, or the radiance cache training pass (a few full paths, whose camera rays
     * also cover a guard band that prefetches texture tiles around the view), the cache resolve and the path
     * tracer;</li>
     * <li>the denoiser and, on traced frames, which write motion vectors, the temporal filters; the specular filter by
     * reflection footprint;</li>
     * <li>pick copy, lens flares, composite, the blit to the swapchain image and the screenshot copy.</li>
     * </ol>
     * A traced frame writes {@code history[parity]}; {@code shown} is the history displayed.
     */
    private void recordFrame(MemoryStack stack, int imageIndex, boolean rebuildTlas, boolean trace, int shown, int flags,
                             boolean screenshot, Camera camera) {
        long swapImage = swapchain != null ? swapchain.images[imageIndex] : VK_NULL_HANDLE;
        int w = rW, h = rH;

        vkCmdResetQueryPool(cmd, queryPool, 0, TS_COUNT);
        timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, TS_START);

        boolean rasterOnly = renderMode == RenderMode.RASTER;
        if (trace && cache != null) cache.recordClear(cmd, stack);

        if (rebuildTlas && rt) {
            boolean refit = !tlasDirty && tlasBuiltCount == scene.instances.size() && tlasRefits < REFIT_LIMIT;
            tlas.recordTopLevelBuild(cmd, tlasInstanceBuffer, scene.instances.size(), tlasScratchAddress, refit);
            tlasRefits = refit ? tlasRefits + 1 : 0;
            tlasBuiltCount = scene.instances.size();
            VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR)
                    .dstAccessMask(VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                    ctx.rtStage | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, mb, null, null);
        }

        VkMemoryBarrier.Buffer frameBarrier = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd,
                ctx.rtStage | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                ctx.rtStage | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, frameBarrier, null, null);

        long v0 = System.nanoTime();
        if (trace) vt.update(cmd, stack);
        lastCpuMs[4] = (System.nanoTime() - v0) / 1e6;

        timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, TS_TRACE_BEGIN);
        if (trace && (rasterPrimary || debugMode == DEBUG_RASTER)) recordRaster(stack, camera);
        if (trace && rasterOnly) {
            writePushConstants(camera, (float) fbW / fbH, flags);
            timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, TS_TRAIN_END);
            rasterShade.record(cmd, stack, parity, w, h, pushConstants);
            vt.recordFeedbackReadback(cmd, stack);
        } else if (trace) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline.pipelineLayout, 0,
                    stack.longs(descriptorSets[parity]), null);
            writePushConstants(camera, (float) fbW / fbH, flags);
            vkCmdPushConstants(cmd, pipeline.pipelineLayout, RT_STAGES, 0, pushConstants);
            if ((cachePaths || prefetch) && ctx.atomicFloatAddSupported) {
                int b = trainBlock();
                float g = prefetch ? PREFETCH_GUARD : 1f;
                vkCmdTraceRaysKHR(cmd, pipeline.trainRegion, pipeline.missRegion, pipeline.hitRegion, pipeline.callableRegion,
                        (int) Math.ceil(w * g / b), (int) Math.ceil(h * g / b), 1);
                VkMemoryBarrier.Buffer tb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                        .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT).dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                vkCmdPipelineBarrier(cmd, ctx.rtStage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0, tb, null, null);
                cache.recordResolve(cmd, stack, (flags & FLAG_ANIMATED) != 0);
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, ctx.rtStage,
                        0, tb, null, null);
            }
            timestamp(ctx.rtStage, TS_TRAIN_END);
            vkCmdTraceRaysKHR(cmd, pipeline.raygenRegion, pipeline.missRegion, pipeline.hitRegion, pipeline.callableRegion, w, h, 1);
            vt.recordFeedbackReadback(cmd, stack);
        } else {
            timestamp(VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, TS_TRAIN_END);
        }
        timestamp(ctx.rtStage, TS_TRACE_END);

        VkMemoryBarrier.Buffer rtToCompute = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, ctx.rtStage, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, rtToCompute, null, null);
        int source = 0;
        int framesInImage = Math.max(1, trace ? accumulatedFrames + 1 : accumulatedFrames);
        int iterations = effectiveDenoiseIterations(framesInImage);
        if (((denoise && debugMode == 0) || debugMode == 3) && iterations > 0 && !rasterOnly) {
            source = post.recordDenoise(cmd, stack, shown, w, h, sampleLimit, iterations, frameCounter);
            PostProcess.computeBarrier(cmd, stack);
            if (post.temporal && debugMode == 0 && trace) {
                source = post.recordTemporal(cmd, stack, shown, w, h, source);
                PostProcess.computeBarrier(cmd, stack);
            }
        }
        int specularSource = 0;
        if (post.specularFilter && denoise && debugMode == 0 && iterations > 0 && !rasterOnly) {
            float pixelAngle = 2f * (float) Math.tan(Math.toRadians(camera.fovYDegrees) / 2) / vH;
            specularSource = post.recordSpecular(cmd, stack, shown, w, h, iterations, pixelAngle, (flags & FLAG_CAMERA_MOVED) != 0);
            PostProcess.computeBarrier(cmd, stack);
            if (post.temporal && trace) {
                specularSource = post.recordSpecularTemporal(cmd, stack, shown, w, h, specularSource);
                PostProcess.computeBarrier(cmd, stack);
            }
        }
        timestamp(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, TS_DENOISE_END);
        recordPickCopy(stack, shown);
        recordFlares(stack);
        post.recordComposite(cmd, stack, shown, fbW, fbH, (float) vW / fbW, (float) vH / fbH, marginX, marginY, exposure, debugMode, source,
                specularSource, sampleLimit, debugMode == 0 ? flareStrength : 0f, hazeConstants());
        w = fbW;
        h = fbH;

        imageBarrier(stack, outputImage.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
        if (swapImage != VK_NULL_HANDLE) {
            imageBarrier(stack, swapImage, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, VK_ACCESS_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);
            VkImageBlit.Buffer blit = VkImageBlit.calloc(1, stack);
            VkImageBlit b = blit.get(0);
            b.srcSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
            b.srcOffsets(1).set(w, h, 1);
            b.dstSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
            b.dstOffsets(1).set(w, h, 1);
            vkCmdBlitImage(cmd, outputImage.image, VK_IMAGE_LAYOUT_GENERAL, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    blit, VK_FILTER_NEAREST);
        }

        if (screenshot) {
            VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack);
            region.get(0).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
            region.get(0).imageExtent().set(w, h, 1);
            vkCmdCopyImageToBuffer(cmd, outputImage.image, VK_IMAGE_LAYOUT_GENERAL, readback.handle, region);
            VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                    .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK_ACCESS_HOST_READ_BIT);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, mb, null, null);
        }

        if (swapImage != VK_NULL_HANDLE) {
            imageBarrier(stack, swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        }
        timestamp(VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, TS_END);
    }

    /**
     * Rebuilds the raster pass's draw list for the current render mode: instances grouped by mesh, one indirect draw
     * per group, opaque groups first. The raster mode adds the alpha-tested instances as groups of their own, drawn
     * with the alpha test, and leaves out glass, which it cannot look through; the other modes leave cutouts to the
     * ray tracer. Runs when the structure or the render mode changes; moved instances only need their transforms.
     */
    private void writeRasterDraws() {
        rasterDrawsMode = renderMode;
        boolean rasterOnly = renderMode == RenderMode.RASTER;
        int n = scene.instances.size(), meshes = scene.meshes.size();
        int[] slotOf = new int[n];
        int[] perSlot = new int[2 * meshes + 1];
        for (int i = 0; i < n; i++) {
            Scene.Instance inst = scene.instances.get(i);
            Material mat = scene.materials.get(inst.material);
            int slot = mat.isAlphaTested() ? (rasterOnly ? meshes + inst.mesh : -1)
                     : rasterOnly && mat.transmission() > 0.5f ? -1 : inst.mesh;
            slotOf[i] = slot;
            if (slot >= 0) perSlot[slot + 1]++;
        }
        int[] start = new int[2 * meshes + 1];
        List<MeshLod> groupLods = new ArrayList<>();
        float[] bounds = new float[6 * Math.max(2 * meshes, 1)];
        int groups = 0, offset = 0, opaqueGroups = 0;
        int[] groupOf = new int[2 * meshes];
        for (int s = 0; s < 2 * meshes; s++) {
            if (s == meshes) opaqueGroups = groups;
            if (perSlot[s + 1] == 0) { groupOf[s] = -1; continue; }
            int m = s % meshes;
            groupOf[s] = groups;
            start[groups] = offset;
            groupLods.add(MeshLod.of(scene.meshes.get(m)));
            System.arraycopy(meshBounds.get(m), 0, bounds, 6 * groups, 6);
            offset += perSlot[s + 1];
            groups++;
        }
        if (meshes == 0) opaqueGroups = groups;
        start[groups] = offset;
        int[] list = new int[Math.max(offset, 1)], fill = java.util.Arrays.copyOf(start, Math.max(groups, 1));
        for (int i = 0; i < n; i++) if (slotOf[i] >= 0) list[fill[groupOf[slotOf[i]]]++] = i;
        MeshLod.retain(scene.meshes);
        raster.setDraws(list, offset, start, groupLods, bounds, groups, opaqueGroups);
    }

    /** The render mode the raster draw list was built for; it is rebuilt when the mode changes. */
    private RenderMode rasterDrawsMode;

    /**
     * Records the raster pre-pass into the visibility buffer, with the traced image's frustum (including the overscan
     * border) and reversed infinite depth. Passes the screen pixels per radian at the image center, by which the
     * culling pass chooses levels of detail.
     */
    private void recordRaster(MemoryStack stack, Camera camera) {
        rasterProj.setPerspective((float) Math.toRadians(camera.fovYDegrees), (float) fbW / fbH, Float.POSITIVE_INFINITY, 0.01f, true);
        overscanScale.mul(rasterProj, rasterViewProj).mul(camera.view());
        float pixelsPerRadian = 0.5f * rH * rasterProj.m11() * (float) vH / rH;
        raster.record(cmd, stack, rasterViewProj, camera.position, pixelsPerRadian);
    }

    private final float[] haze = new float[20];
    private final Vector3f hazeTmp = new Vector3f();

    /** Push constants of the aerial perspective: haze color and density, sun, camera height and basis. */
    private float[] hazeConstants() {
        Scene s = scene;
        float sun = s.sunDirection.y > -0.1f ? s.sunIntensity * s.hazeSun : 0f;
        float[] v = {s.hazeColor.x, s.hazeColor.y, s.hazeColor.z, s.hazeDensity,
                s.sunColor.x * sun, s.sunColor.y * sun, s.sunColor.z * sun, s.hazeFalloff,
                s.sunDirection.x, s.sunDirection.y, s.sunDirection.z, currentCamera.position.y};
        System.arraycopy(v, 0, haze, 0, 12);
        currentCamera.forward(hazeTmp);
        haze[12] = hazeTmp.x; haze[13] = hazeTmp.y; haze[14] = hazeTmp.z; haze[15] = 0f;
        float tanX = (float) Math.tan(Math.toRadians(currentCamera.fovYDegrees) / 2) * fbW / Math.max(fbH, 1);
        currentCamera.right(hazeTmp).normalize().mul(tanX);
        haze[16] = hazeTmp.x; haze[17] = hazeTmp.y; haze[18] = hazeTmp.z; haze[19] = 0f;
        return haze;
    }

    /** VRAM for the virtual texture pools in MB when the scene leaves their size to the engine. */
    private static final long TEXTURE_POOL_MB = Long.getLong("lumvex.texturepool", 3072);

    /** Slots per row of each virtual texture pool: the scene's choice, or as many as fit the VRAM budget. */
    private int texturePoolSlots() {
        if (scene.texturePoolSlotsPerRow > 0) return scene.texturePoolSlotsPerRow;
        int slots = VirtualTextures.slotsForBudget(TEXTURE_POOL_MB << 20, scene.textureCompression && ctx.textureCompressionBC,
                ctx.maxImageDimension2D);
        System.out.printf("[lumvex] texture pools: %d x %d slots per pool%n", slots, slots);
        return slots;
    }

    /** Adds the sun as flare source and records its visibility pass. Point lights have no flares, see LensFlare. */
    private void recordFlares(MemoryStack stack) {
        if (!rt) return;
        flare.begin();
        if (flareStrength > 0 && debugMode == 0) {
            if (scene.sunIntensity > 0 && scene.sunDirection.y > -0.02f) {
                flare.addDirectional(scene.sunDirection, 0.0047f,
                        new Vector3f(scene.sunColor).mul(scene.sunIntensity * SUN_FLARE));
            }
        }
        flare.record(cmd, viewProjShown, currentCamera.position, fbW, fbH, frameCounter);
        if (System.getenv("LUMVEX_FLARE_DEBUG") != null && frameCounter % 30 == 0) flare.dump();
        PostProcess.computeBarrier(cmd, stack);
    }

    private void writePushConstants(Camera camera, float aspect, int flags) {
        ByteBuffer pc = pushConstants;
        camera.inverseView(tmp).get(0, pc);
        camera.inverseProjection(aspect, tmp).mul(overscanScaleInv).get(64, pc);
        pc.putInt(128, frameCounter)
                .putInt(132, maxBounces)
                .putInt(136, samplesPerFrame)
                .putInt(140, sampleLimit)
                .putFloat(144, scene.sunDirection.x).putFloat(148, scene.sunDirection.y).putFloat(152, scene.sunDirection.z)
                .putFloat(156, scene.sunIntensity)
                .putFloat(160, scene.sunColor.x).putFloat(164, scene.sunColor.y).putFloat(168, scene.sunColor.z)
                .putFloat(172, scene.skyIntensity)
                .putInt(176, debugMode)
                .putFloat(180, camera.aperture)
                .putFloat(184, camera.focusDistance)
                .putInt(188, flags);
        prevViewProj.get(192, pc);
    }

    private void imageBarrier(MemoryStack stack, long image, int oldLayout, int newLayout,
                              int srcAccess, int dstAccess, int srcStage, int dstStage) {
        VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack);
        barrier.get(0)
                .sType$Default()
                .srcAccessMask(srcAccess)
                .dstAccessMask(dstAccess)
                .oldLayout(oldLayout)
                .newLayout(newLayout)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
        vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, null, null, barrier);
    }

    private void saveScreenshot(Path file) {
        int w = fbW, h = fbH;
        ByteBuffer px = readback.map();
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        int[] row = new int[w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int o = (y * w + x) * 4;
                row[x] = ((px.get(o) & 0xFF) << 16) | ((px.get(o + 1) & 0xFF) << 8) | (px.get(o + 2) & 0xFF);
            }
            img.setRGB(0, y, w, 1, row, 0, w);
        }
        try {
            Path abs = file.toAbsolutePath();
            if (abs.getParent() != null) Files.createDirectories(abs.getParent());
            ImageIO.write(img, "png", abs.toFile());
            System.out.println("[lumvex] screenshot saved: " + abs + " (" + accumulatedSamples() + " spp)");
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public void close() {
        vkDeviceWaitIdle(ctx.device);
        destroySizeDependent(true);
        vkDestroyFence(ctx.device, inFlight, null);
        vkDestroySemaphore(ctx.device, imageAvailable, null);
        if (rt) {
            vkDestroyDescriptorPool(ctx.device, descriptorPool, null);
            tlas.close();
            tlasScratch.close();
            tlasInstanceBuffer.close();
        }
        materialBuffer.close();
        instanceInfoBuffer.close();
        instanceTransformBuffer.close();
        for (GpuMesh m : gpuMeshes) {
            if (m.blas != null) m.blas.close();
            m.vertices.close();
            m.indices.close();
        }
        vkDestroyQueryPool(ctx.device, queryPool, null);
        if (cache != null) cache.close();
        vt.close();
        dynamicBuffer.close();
        lightBuffer.close();
        if (alphaMaskBuffer != null) alphaMaskBuffer.close();
        statsBuffer.close();
        envTexture.close();
        deltaBuffer.close();
        uiBuffer.close();
        pickBuffer.close();
        if (flare != null) flare.close();
        if (noFlares != null) noFlares.close();
        post.close();
        rasterShade.close();
        raster.close();
        if (pipeline != null) pipeline.close();
        compiler.close();
        memFree(pushConstants);
        ctx.close();
    }
}
