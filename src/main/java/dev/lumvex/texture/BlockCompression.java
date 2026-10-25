package dev.lumvex.texture;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * CPU block compression for 4x4 blocks of RGBA8 data, plus decoders for quality checks. It is the reference for
 * {@code bc_encode.comp}, which produces the same bitstream on the GPU.
 *
 * <pre>
 *   BC1         4 bpp  RGB, two 565 endpoints, 4 colors           far mips
 *   BC4 / BC5   4/8 bpp one or two channels, 8 values             normal maps (x, y)
 *   BC7 mode 6  8 bpp  RGBA, 7-bit endpoints + p-bit, 16 colors    near mips
 * </pre>
 *
 * Endpoints are fitted along the block's principal axis and refined once by least squares. A full BC7 encoder
 * would search 8 modes and 64 partitions; mode 6 alone is the usual real-time choice for opaque textures.
 */
public final class BlockCompression {
    private BlockCompression() {}

    /** Encodes a row-major RGBA8 image whose dimensions are multiples of 4. */
    public interface Encoder {
        void encode(ByteBuffer src, int srcOff, int w, int h, ByteBuffer dst, int dstOff);
        int bytesPerBlock();
    }

    public static final Encoder BC1 = new Encoder() {
        public void encode(ByteBuffer s, int so, int w, int h, ByteBuffer d, int dOff) { encodeImage(s, so, w, h, d, dOff, 8, BlockCompression::bc1Block); }
        public int bytesPerBlock() { return 8; }
    };
    public static final Encoder BC5 = new Encoder() {
        public void encode(ByteBuffer s, int so, int w, int h, ByteBuffer d, int dOff) { encodeImage(s, so, w, h, d, dOff, 16, BlockCompression::bc5Block); }
        public int bytesPerBlock() { return 16; }
    };
    public static final Encoder BC7 = new Encoder() {
        public void encode(ByteBuffer s, int so, int w, int h, ByteBuffer d, int dOff) { encodeImage(s, so, w, h, d, dOff, 16, BlockCompression::bc7Mode6Block); }
        public int bytesPerBlock() { return 16; }
    };

    private interface BlockFn { void block(int[] rgba, ByteBuffer dst, int off); }

    private static void encodeImage(ByteBuffer src, int srcOff, int w, int h, ByteBuffer dst, int dstOff, int bpb, BlockFn fn) {
        int[] px = new int[64];
        int bw = w / 4;
        for (int by = 0; by < h / 4; by++) {
            for (int bx = 0; bx < bw; bx++) {
                for (int y = 0; y < 4; y++) {
                    for (int x = 0; x < 4; x++) {
                        int o = srcOff + ((by * 4 + y) * w + bx * 4 + x) * 4;
                        int i = (y * 4 + x) * 4;
                        px[i] = src.get(o) & 0xFF; px[i + 1] = src.get(o + 1) & 0xFF;
                        px[i + 2] = src.get(o + 2) & 0xFF; px[i + 3] = src.get(o + 3) & 0xFF;
                    }
                }
                fn.block(px, dst, dstOff + (by * bw + bx) * bpb);
            }
        }
    }

    /** Projects the block onto its principal axis and returns the extreme points in {@code lo} and {@code hi}. */
    private static void fitLine(int[] px, int channels, float[] lo, float[] hi) {
        float[] mean = new float[4];
        for (int i = 0; i < 16; i++) for (int c = 0; c < channels; c++) mean[c] += px[i * 4 + c];
        for (int c = 0; c < channels; c++) mean[c] /= 16f;
        float[][] cov = new float[4][4];
        for (int i = 0; i < 16; i++) {
            for (int a = 0; a < channels; a++) {
                float da = px[i * 4 + a] - mean[a];
                for (int b = a; b < channels; b++) cov[a][b] += da * (px[i * 4 + b] - mean[b]);
            }
        }
        for (int a = 0; a < channels; a++) for (int b = 0; b < a; b++) cov[a][b] = cov[b][a];
        float[] axis = {1, 1, 1, 1};
        // power iteration for the dominant eigenvector
        for (int it = 0; it < 6; it++) {
            float[] n = new float[4];
            for (int a = 0; a < channels; a++) for (int b = 0; b < channels; b++) n[a] += cov[a][b] * axis[b];
            float len = 0;
            for (int a = 0; a < channels; a++) len += n[a] * n[a];
            if (len < 1e-12f) break;
            len = (float) Math.sqrt(len);
            for (int a = 0; a < channels; a++) axis[a] = n[a] / len;
        }
        float tMin = Float.MAX_VALUE, tMax = -Float.MAX_VALUE;
        for (int i = 0; i < 16; i++) {
            float t = 0;
            for (int c = 0; c < channels; c++) t += (px[i * 4 + c] - mean[c]) * axis[c];
            tMin = Math.min(tMin, t);
            tMax = Math.max(tMax, t);
        }
        for (int c = 0; c < channels; c++) {
            lo[c] = clamp255(mean[c] + axis[c] * tMin);
            hi[c] = clamp255(mean[c] + axis[c] * tMax);
        }
    }

    private static float clamp255(float v) { return Math.max(0f, Math.min(255f, v)); }

    // BC1

    private static int to565(float r, float g, float b) {
        int r5 = Math.round(r * 31f / 255f), g6 = Math.round(g * 63f / 255f), b5 = Math.round(b * 31f / 255f);
        return (r5 << 11) | (g6 << 5) | b5;
    }

    // Expands RGB565 to 8 bits per channel by bit replication.
    private static void from565(int c, int[] out, int o) {
        int r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;
        out[o] = (r << 3) | (r >> 2); out[o + 1] = (g << 2) | (g >> 4); out[o + 2] = (b << 3) | (b >> 2);
    }

    static void bc1Block(int[] px, ByteBuffer dst, int off) {
        float[] lo = new float[4], hi = new float[4];
        fitLine(px, 3, lo, hi);
        int c0 = to565(hi[0], hi[1], hi[2]), c1 = to565(lo[0], lo[1], lo[2]);
        int indices = 0;
        if (c0 != c1) {
            // c0 > c1 selects 4-color mode
            if (c0 < c1) { int t = c0; c0 = c1; c1 = t; }
            int[] pal = new int[12];
            from565(c0, pal, 0);
            from565(c1, pal, 3);
            for (int c = 0; c < 3; c++) {
                pal[6 + c] = (2 * pal[c] + pal[3 + c]) / 3;
                pal[9 + c] = (pal[c] + 2 * pal[3 + c]) / 3;
            }
            for (int i = 0; i < 16; i++) {
                int best = 0, bestErr = Integer.MAX_VALUE;
                for (int k = 0; k < 4; k++) {
                    int e = 0;
                    for (int c = 0; c < 3; c++) { int d = px[i * 4 + c] - pal[k * 3 + c]; e += d * d; }
                    if (e < bestErr) { bestErr = e; best = k; }
                }
                indices |= best << (i * 2);
            }
        }
        dst.order(ByteOrder.LITTLE_ENDIAN);
        dst.putShort(off, (short) c0).putShort(off + 2, (short) c1).putInt(off + 4, indices);
    }

    /** Decodes a BC1 block in 4-color or 3-color mode. */
    static void bc1Decode(ByteBuffer src, int off, int[] out) {
        src.order(ByteOrder.LITTLE_ENDIAN);
        int c0 = src.getShort(off) & 0xFFFF, c1 = src.getShort(off + 2) & 0xFFFF, idx = src.getInt(off + 4);
        int[] pal = new int[12];
        from565(c0, pal, 0);
        from565(c1, pal, 3);
        for (int c = 0; c < 3; c++) {
            pal[6 + c] = c0 > c1 ? (2 * pal[c] + pal[3 + c]) / 3 : (pal[c] + pal[3 + c]) / 2;
            pal[9 + c] = c0 > c1 ? (pal[c] + 2 * pal[3 + c]) / 3 : 0;
        }
        for (int i = 0; i < 16; i++) {
            int k = (idx >>> (i * 2)) & 3;
            for (int c = 0; c < 3; c++) out[i * 4 + c] = pal[k * 3 + c];
            out[i * 4 + 3] = 255;
        }
    }

    // BC4 / BC5

    /** One channel in 8-value mode with min and max as endpoints. */
    private static void bc4Block(int[] px, int channel, ByteBuffer dst, int off) {
        int mn = 255, mx = 0;
        for (int i = 0; i < 16; i++) { int v = px[i * 4 + channel]; mn = Math.min(mn, v); mx = Math.max(mx, v); }
        long bits = 0;
        if (mx > mn) {
            for (int i = 0; i < 16; i++) {
                // 0 = min, 7 = max
                int j = Math.round((px[i * 4 + channel] - mn) * 7f / (mx - mn));
                // BC4 order: e0 = max, e1 = min, then max to min
                int index = j == 7 ? 0 : j == 0 ? 1 : 8 - j;
                bits |= (long) index << (i * 3);
            }
        }
        dst.put(off, (byte) mx).put(off + 1, (byte) mn);
        for (int b = 0; b < 6; b++) dst.put(off + 2 + b, (byte) (bits >>> (b * 8)));
    }

    /** BC4 on the red and green channels, i.e. normal x and y. */
    static void bc5Block(int[] px, ByteBuffer dst, int off) {
        bc4Block(px, 0, dst, off);
        bc4Block(px, 1, dst, off + 8);
    }

    static void bc4Decode(ByteBuffer src, int off, int[] out, int channel) {
        int e0 = src.get(off) & 0xFF, e1 = src.get(off + 1) & 0xFF;
        long bits = 0;
        for (int b = 0; b < 6; b++) bits |= (long) (src.get(off + 2 + b) & 0xFF) << (b * 8);
        int[] pal = new int[8];
        pal[0] = e0; pal[1] = e1;
        for (int k = 2; k < 8; k++) {
            pal[k] = e0 > e1 ? ((8 - k) * e0 + (k - 1) * e1) / 7 : (k < 6 ? ((6 - k) * e0 + (k - 1) * e1) / 5 : (k == 6 ? 0 : 255));
        }
        for (int i = 0; i < 16; i++) out[i * 4 + channel] = pal[(int) ((bits >>> (i * 3)) & 7)];
    }

    // BC7 mode 6

    private static final int[] W4 = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

    /** Quantizes an endpoint to 7 bits per channel plus a shared p-bit: {r7, g7, b7, a7, p}. */
    private static int[] quantize76(float[] e) {
        int[] best = null;
        float bestErr = Float.MAX_VALUE;
        for (int p = 0; p < 2; p++) {
            int[] q = new int[5];
            float err = 0;
            for (int c = 0; c < 4; c++) {
                q[c] = Math.max(0, Math.min(127, Math.round((e[c] - p) / 2f)));
                float d = q[c] * 2 + p - e[c];
                err += d * d;
            }
            q[4] = p;
            if (err < bestErr) { bestErr = err; best = q; }
        }
        return best;
    }

    /** Assigns each texel the nearest of the 16 interpolated colors and returns the total squared error. */
    private static int bc7Indices(int[] px, int[] e0, int[] e1, int[] idx) {
        int[] pal = new int[64];
        for (int k = 0; k < 16; k++) {
            for (int c = 0; c < 4; c++) pal[k * 4 + c] = (e0[c] * (64 - W4[k]) + e1[c] * W4[k] + 32) >> 6;
        }
        int total = 0;
        for (int i = 0; i < 16; i++) {
            int best = 0, bestErr = Integer.MAX_VALUE;
            for (int k = 0; k < 16; k++) {
                int e = 0;
                for (int c = 0; c < 4; c++) { int d = px[i * 4 + c] - pal[k * 4 + c]; e += d * d; }
                if (e < bestErr) { bestErr = e; best = k; }
            }
            idx[i] = best;
            total += bestErr;
        }
        return total;
    }

    static void bc7Mode6Block(int[] px, ByteBuffer dst, int off) {
        float[] lo = new float[4], hi = new float[4];
        fitLine(px, 4, lo, hi);
        int[] q0 = quantize76(lo), q1 = quantize76(hi);
        int[] e0 = new int[4], e1 = new int[4], idx = new int[16];
        for (int c = 0; c < 4; c++) { e0[c] = q0[c] * 2 + q0[4]; e1[c] = q1[c] * 2 + q1[4]; }
        int err = bc7Indices(px, e0, e1, idx);

        // One least-squares refinement of the endpoints for the chosen indices; kept only if the error drops.
        float aa = 0, ab = 0, bb = 0;
        float[] ax = new float[4], bx = new float[4];
        for (int i = 0; i < 16; i++) {
            float t = W4[idx[i]] / 64f, s = 1 - t;
            aa += s * s; ab += s * t; bb += t * t;
            for (int c = 0; c < 4; c++) { ax[c] += s * px[i * 4 + c]; bx[c] += t * px[i * 4 + c]; }
        }
        float det = aa * bb - ab * ab;
        if (Math.abs(det) > 1e-6f) {
            float[] n0 = new float[4], n1 = new float[4];
            for (int c = 0; c < 4; c++) {
                n0[c] = clamp255((ax[c] * bb - bx[c] * ab) / det);
                n1[c] = clamp255((bx[c] * aa - ax[c] * ab) / det);
            }
            int[] r0 = quantize76(n0), r1 = quantize76(n1);
            int[] f0 = new int[4], f1 = new int[4], idx2 = new int[16];
            for (int c = 0; c < 4; c++) { f0[c] = r0[c] * 2 + r0[4]; f1[c] = r1[c] * 2 + r1[4]; }
            int err2 = bc7Indices(px, f0, f1, idx2);
            if (err2 < err) { q0 = r0; q1 = r1; idx = idx2; }
        }
        // The anchor index is stored with 3 bits, so its top bit must be 0; swap the endpoints if needed.
        if (idx[0] >= 8) {
            int[] t = q0; q0 = q1; q1 = t;
            for (int i = 0; i < 16; i++) idx[i] = 15 - idx[i];
        }
        long lo64 = 0, hi64 = 0;
        int pos = 0;
        long[] w = {0, 0};
        // mode 6
        pos = put(w, pos, 1 << 6, 7);
        for (int c = 0; c < 4; c++) { pos = put(w, pos, q0[c], 7); pos = put(w, pos, q1[c], 7); }
        pos = put(w, pos, q0[4], 1);
        pos = put(w, pos, q1[4], 1);
        pos = put(w, pos, idx[0], 3);
        for (int i = 1; i < 16; i++) pos = put(w, pos, idx[i], 4);
        lo64 = w[0];
        hi64 = w[1];
        dst.order(ByteOrder.LITTLE_ENDIAN);
        dst.putLong(off, lo64).putLong(off + 8, hi64);
    }

    // Bit packing within a 128-bit block stored as two longs.
    private static int put(long[] w, int pos, int value, int bits) {
        for (int b = 0; b < bits; b++, pos++) {
            if (((value >> b) & 1) != 0) w[pos >> 6] |= 1L << (pos & 63);
        }
        return pos;
    }

    private static int get(long lo, long hi, int pos, int bits) {
        int v = 0;
        for (int b = 0; b < bits; b++, pos++) {
            long word = pos < 64 ? lo : hi;
            if (((word >>> (pos & 63)) & 1) != 0) v |= 1 << b;
        }
        return v;
    }

    static void bc7Mode6Decode(ByteBuffer src, int off, int[] out) {
        src.order(ByteOrder.LITTLE_ENDIAN);
        long lo = src.getLong(off), hi = src.getLong(off + 8);
        int pos = 7;
        int[] q0 = new int[4], q1 = new int[4];
        for (int c = 0; c < 4; c++) { q0[c] = get(lo, hi, pos, 7); pos += 7; q1[c] = get(lo, hi, pos, 7); pos += 7; }
        int p0 = get(lo, hi, pos++, 1), p1 = get(lo, hi, pos++, 1);
        int[] e0 = new int[4], e1 = new int[4];
        for (int c = 0; c < 4; c++) { e0[c] = q0[c] * 2 + p0; e1[c] = q1[c] * 2 + p1; }
        for (int i = 0; i < 16; i++) {
            int bits = i == 0 ? 3 : 4;
            int k = get(lo, hi, pos, bits);
            pos += bits;
            for (int c = 0; c < 4; c++) out[i * 4 + c] = (e0[c] * (64 - W4[k]) + e1[c] * W4[k] + 32) >> 6;
        }
    }

    /** Round-trip PSNR in dB over the first {@code channels} channels; used to validate the encoders. */
    public static double psnr(Encoder enc, ByteBuffer img, int w, int h, int channels) {
        ByteBuffer comp = ByteBuffer.allocateDirect((w / 4) * (h / 4) * enc.bytesPerBlock()).order(ByteOrder.LITTLE_ENDIAN);
        enc.encode(img, 0, w, h, comp, 0);
        int[] px = new int[64];
        double se = 0;
        long n = 0;
        int bw = w / 4;
        for (int by = 0; by < h / 4; by++) {
            for (int bx = 0; bx < bw; bx++) {
                int o = (by * bw + bx) * enc.bytesPerBlock();
                if (enc == BC1) bc1Decode(comp, o, px);
                else if (enc == BC7) bc7Mode6Decode(comp, o, px);
                else { bc4Decode(comp, o, px, 0); bc4Decode(comp, o + 8, px, 1); }
                for (int y = 0; y < 4; y++) for (int x = 0; x < 4; x++) {
                    int s = ((by * 4 + y) * w + bx * 4 + x) * 4;
                    for (int c = 0; c < channels; c++) {
                        double d = (img.get(s + c) & 0xFF) - px[(y * 4 + x) * 4 + c];
                        se += d * d;
                        n++;
                    }
                }
            }
        }
        double mse = se / n;
        return mse == 0 ? 99 : 10 * Math.log10(255.0 * 255.0 / mse);
    }
}
