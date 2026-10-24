package dev.lumvex.texture;

import java.nio.ByteBuffer;
import java.util.stream.IntStream;

import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.system.MemoryUtil.memAlloc;
import static org.lwjgl.system.MemoryUtil.memCopy;
import static org.lwjgl.system.MemoryUtil.memFree;

/**
 * The textures of a material in system memory: albedo (sRGB), tangent-space normal map and ORM (occlusion,
 * roughness, metallic). All maps are square RGBA8 with a power-of-two size and a mip chain down to one tile. The GPU
 * only ever holds individual tiles, see {@link VirtualTextures}.
 */
public final class TextureSet implements AutoCloseable {
    public static final int TILE = 128;
    public static final int BORDER = 4;                    // filter margin around each tile in the atlas
    public static final int SLOT = TILE + 2 * BORDER;      // 136
    public static final int CHANNELS = 3;                  // albedo, normal, orm
    public static final int ALBEDO = 0, NORMAL = 1, ORM = 2;

    public final String name;
    public final int size;
    public final int mipCount;
    /** {@code maps[channel][mip]}, row-major RGBA8. */
    private final ByteBuffer[][] maps;

    /** Maximum side length of the alpha mask. */
    public static final int MASK_MAX = 2048;
    /**
     * One bit per texel, set where the albedo is opaque; null if the albedo has no holes. It is kept outside the
     * virtual texture because the far mips use BC1 without alpha and the any-hit shader needs it on every ray.
     */
    public final int[] alphaMask;
    public final int alphaSize;

    private TextureSet(String name, int size, ByteBuffer[] level0) {
        if (Integer.bitCount(size) != 1 || size < TILE) throw new IllegalArgumentException("size must be a power of two >= " + TILE);
        this.name = name;
        this.size = size;
        int[] mask = null;
        int ms = Math.min(size, MASK_MAX);
        ByteBuffer alb = level0[ALBEDO];
        long holes = 0, total = (long) size * size;
        for (int i = 0; i < total; i++) if ((alb.get(i * 4 + 3) & 0xFF) < 128) holes++;
        // Some holes, but not nearly everything: exporters sometimes leave the alpha channel at zero.
        if (holes > total / 1000 && holes < total * 98 / 100) {
            mask = new int[(ms * ms + 31) / 32];
            int step = size / ms;
            for (int y = 0; y < ms; y++) {
                for (int x = 0; x < ms; x++) {
                    int src = (y * step + step / 2) * size + x * step + step / 2;
                    if ((alb.get(src * 4 + 3) & 0xFF) >= 128) {
                        int bit = y * ms + x;
                        mask[bit >> 5] |= 1 << (bit & 31);
                    }
                }
            }
        }
        this.alphaMask = mask;
        this.alphaSize = mask == null ? 0 : ms;
        this.mipCount = Integer.numberOfTrailingZeros(size / TILE) + 1;
        maps = new ByteBuffer[CHANNELS][mipCount];
        for (int c = 0; c < CHANNELS; c++) {
            maps[c][0] = level0[c];
            for (int m = 1; m < mipCount; m++) maps[c][m] = downsample(maps[c][m - 1], size >> (m - 1), c);
        }
    }

    /** Takes ownership of the three level-0 buffers and builds the mip chain. */
    public static TextureSet of(String name, int size, ByteBuffer albedo, ByteBuffer normal, ByteBuffer orm) {
        return new TextureSet(name, size, new ByteBuffer[]{albedo, normal, orm});
    }

    public int tilesPerSide(int mip) { return Math.max(1, (size >> mip) / TILE); }

    public ByteBuffer level(int channel, int mip) { return maps[channel][mip]; }

    /** Tiles per channel over all mip levels. */
    public int tileCount() {
        int n = 0;
        for (int m = 0; m < mipCount; m++) n += tilesPerSide(m) * tilesPerSide(m);
        return n;
    }

    public long bytes() {
        long b = 0;
        for (ByteBuffer[] ch : maps) for (ByteBuffer mip : ch) b += mip.capacity();
        return b;
    }

    /** Copies a tile with its border into {@code dst} as SLOT x SLOT RGBA8. The border wraps around. */
    public void extractTile(int channel, int mip, int tx, int ty, ByteBuffer dst, int dstOffset) {
        ByteBuffer src = maps[channel][mip];
        long srcAddr = memAddress(src), dstAddr = memAddress(dst) + dstOffset;
        int dim = size >> mip;
        int x0 = Math.floorMod(tx * TILE - BORDER, dim), y0 = ty * TILE - BORDER;
        // Each slot row is at most two contiguous runs of the wrapping source row.
        int first = Math.min(SLOT, dim - x0);
        for (int y = 0; y < SLOT; y++) {
            long row = srcAddr + (long) Math.floorMod(y0 + y, dim) * dim * 4;
            long out = dstAddr + (long) y * SLOT * 4;
            memCopy(row + x0 * 4L, out, first * 4L);
            for (int done = first; done < SLOT; ) {          // wrapped part; tiny mips may wrap several times
                int n = Math.min(SLOT - done, dim);
                memCopy(row, out + done * 4L, n * 4L);
                done += n;
            }
        }
    }

    private static final float[] SRGB_TO_LINEAR = new float[256];
    static {
        for (int i = 0; i < 256; i++) {
            double c = i / 255.0;
            SRGB_TO_LINEAR[i] = (float) (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
        }
    }

    static int linearToSrgb8(float l) {
        double c = l <= 0.0031308 ? l * 12.92 : 1.055 * Math.pow(l, 1 / 2.4) - 0.055;
        return (int) Math.round(Math.clamp(c, 0.0, 1.0) * 255);
    }

    /** 2x2 box filter. Albedo is averaged in linear space and normals are renormalized. */
    private static ByteBuffer downsample(ByteBuffer src, int srcDim, int channel) {
        int dim = srcDim / 2;
        ByteBuffer dst = memAlloc(dim * dim * 4);
        IntStream.range(0, dim).parallel().forEach(y -> {
            for (int x = 0; x < dim; x++) {
                float[] acc = new float[4];
                for (int k = 0; k < 4; k++) {
                    int o = (((y * 2 + (k >> 1)) * srcDim) + x * 2 + (k & 1)) * 4;
                    for (int c = 0; c < 4; c++) {
                        int v = src.get(o + c) & 0xFF;
                        acc[c] += (channel == ALBEDO && c < 3) ? SRGB_TO_LINEAR[v] : v / 255f;
                    }
                }
                int o = (y * dim + x) * 4;
                if (channel == NORMAL) {
                    float nx = acc[0] / 2 - 1, ny = acc[1] / 2 - 1, nz = acc[2] / 2 - 1; // mean * 2 - 1
                    float len = (float) Math.sqrt(nx * nx + ny * ny + nz * nz);
                    if (len < 1e-6f) { nx = 0; ny = 0; nz = 1; len = 1; }
                    dst.put(o, (byte) Math.round((nx / len * 0.5f + 0.5f) * 255));
                    dst.put(o + 1, (byte) Math.round((ny / len * 0.5f + 0.5f) * 255));
                    dst.put(o + 2, (byte) Math.round((nz / len * 0.5f + 0.5f) * 255));
                    dst.put(o + 3, (byte) Math.round(acc[3] / 4 * 255));
                } else {
                    for (int c = 0; c < 4; c++) {
                        float avg = acc[c] / 4;
                        dst.put(o + c, (byte) ((channel == ALBEDO && c < 3) ? linearToSrgb8(avg) : Math.round(avg * 255)));
                    }
                }
            }
        });
        return dst;
    }

    @Override
    public void close() {
        for (ByteBuffer[] ch : maps) for (ByteBuffer b : ch) memFree(b);
    }
}
