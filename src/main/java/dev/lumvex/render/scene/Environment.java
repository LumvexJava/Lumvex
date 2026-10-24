package dev.lumvex.render.scene;

import org.joml.Vector3f;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * An equirectangular HDR environment loaded from a Radiance {@code .hdr} file.
 *
 * <p>The sun disk in such a panorama is tiny and extremely bright; paths would only hit it by chance and produce
 * fireflies. {@link #extractSun} therefore clamps it and turns its energy into a directional sun that is sampled
 * explicitly.
 *
 * <p>The mapping must match {@code environment()} in {@code raygen.rgen}:
 * {@code u = 0.5 + atan(dir.x, -dir.z) / 2pi}, {@code v = acos(dir.y) / pi}.
 */
public final class Environment {
    public final int width, height;
    public final float[] rgb;               // linear radiance, row-major, 3 floats per texel
    public final Vector3f sunDirection = new Vector3f(0, 1, 0);
    public final Vector3f sunColor = new Vector3f(1);
    /** Irradiance of the extracted sun at normal incidence. */
    public float sunIrradiance;

    private Environment(int w, int h, float[] rgb) {
        this.width = w;
        this.height = h;
        this.rgb = rgb;
    }

    /** Reads an RGBE file with flat or run-length encoded scanlines in "-Y h +X w" orientation. */
    public static Environment load(Path hdrFile) throws IOException {
        long t0 = System.nanoTime();
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(hdrFile));
        String line;
        int w = 0, h = 0;
        while (!(line = readLine(b)).isEmpty() || w == 0) {
            if (line.startsWith("-Y ")) {
                String[] t = line.split("\\s+");
                h = Integer.parseInt(t[1]);
                w = Integer.parseInt(t[3]);
                break;
            }
        }
        if (w == 0) throw new IOException("unsupported .hdr resolution line in " + hdrFile);
        float[] out = new float[w * h * 3];
        byte[] scan = new byte[w * 4];
        for (int y = 0; y < h; y++) {
            int p = b.position();
            if (w >= 8 && w < 32768 && b.get(p) == 2 && b.get(p + 1) == 2 && ((b.get(p + 2) << 8 | (b.get(p + 3) & 0xFF)) == w)) {
                b.position(p + 4);
                for (int c = 0; c < 4; c++) {                 // new-style RLE, one component at a time
                    int x = 0;
                    while (x < w) {
                        int count = b.get() & 0xFF;
                        if (count > 128) {
                            count -= 128;
                            byte v = b.get();
                            for (int i = 0; i < count; i++) scan[(x++) * 4 + c] = v;
                        } else {
                            for (int i = 0; i < count; i++) scan[(x++) * 4 + c] = b.get();
                        }
                    }
                }
            } else {
                b.get(scan, 0, w * 4);                        // flat RGBE
            }
            for (int x = 0; x < w; x++) {
                int e = scan[x * 4 + 3] & 0xFF;
                float f = e == 0 ? 0 : (float) Math.scalb(1.0, e - 136);  // 2^(e-128) / 256
                int o = (y * w + x) * 3;
                out[o] = ((scan[x * 4] & 0xFF) + 0.5f) * f;
                out[o + 1] = ((scan[x * 4 + 1] & 0xFF) + 0.5f) * f;
                out[o + 2] = ((scan[x * 4 + 2] & 0xFF) + 0.5f) * f;
            }
        }
        Environment env = new Environment(w, h, out);
        System.out.printf("[env] %s %dx%d loaded in %.1f s%n", hdrFile.getFileName(), w, h, (System.nanoTime() - t0) / 1e9);
        return env;
    }

    private static String readLine(ByteBuffer b) {
        StringBuilder sb = new StringBuilder();
        while (b.hasRemaining()) {
            char c = (char) (b.get() & 0xFF);
            if (c == '\n') break;
            sb.append(c);
        }
        return sb.toString();
    }

    // Rec. 709 luminance
    private static float lum(float r, float g, float b) { return 0.2126f * r + 0.7152f * g + 0.0722f * b; }

    /** World direction of a texel center, using the shader's mapping. */
    private Vector3f dir(float x, float y) {
        double u = (x + 0.5) / width, v = (y + 0.5) / height;
        double phi = (u - 0.5) * 2 * Math.PI, theta = v * Math.PI;
        // Inverse of u = 0.5 + atan(dx, -dz) / 2pi: dx = sin(phi) sin(theta), dz = -cos(phi) sin(theta)
        return new Vector3f((float) (Math.sin(phi) * Math.sin(theta)), (float) Math.cos(theta), (float) (-Math.cos(phi) * Math.sin(theta)));
    }

    /**
     * Clamps every texel brighter than {@code factor} times the mean sky luminance. The clamped energy, weighted by
     * solid angle, defines {@link #sunDirection}, {@link #sunColor} and {@link #sunIrradiance}.
     */
    public void extractSun(float factor) {
        double mean = 0, wsum = 0;
        for (int y = 0; y < height; y++) {
            double sinT = Math.sin((y + 0.5) / height * Math.PI);
            for (int x = 0; x < width; x++) {
                int o = (y * width + x) * 3;
                mean += lum(rgb[o], rgb[o + 1], rgb[o + 2]) * sinT;
                wsum += sinT;
            }
        }
        mean /= wsum;
        float threshold = (float) (mean * factor);
        double dOmega = (2 * Math.PI / width) * (Math.PI / height);
        Vector3f dirSum = new Vector3f(), colorSum = new Vector3f();
        double energy = 0;
        for (int y = 0; y < height; y++) {
            double sinT = Math.sin((y + 0.5) / height * Math.PI);
            for (int x = 0; x < width; x++) {
                int o = (y * width + x) * 3;
                float l = lum(rgb[o], rgb[o + 1], rgb[o + 2]);
                if (l <= threshold) continue;
                float keep = threshold / l;
                float w = (float) ((l - threshold) * sinT * dOmega);
                colorSum.add(rgb[o] * (1 - keep) * (float) (sinT * dOmega), rgb[o + 1] * (1 - keep) * (float) (sinT * dOmega),
                        rgb[o + 2] * (1 - keep) * (float) (sinT * dOmega));
                dirSum.fma(w, dir(x, y));
                energy += w;
                rgb[o] *= keep; rgb[o + 1] *= keep; rgb[o + 2] *= keep;
            }
        }
        if (energy <= 0) return;
        sunDirection.set(dirSum).normalize();
        sunIrradiance = (float) energy;
        float cl = lum(colorSum.x, colorSum.y, colorSum.z);
        sunColor.set(colorSum).div(cl);
        System.out.printf("[env] sun: elevation %.1f deg, irradiance %.1f (sky mean %.2f), color %.2f %.2f %.2f%n",
                Math.toDegrees(Math.asin(sunDirection.y)), sunIrradiance, mean, sunColor.x, sunColor.y, sunColor.z);
    }

    /** Rotates the panorama about the vertical axis, in whole texels, so the sun ends up at {@code targetU}. */
    public void rotateSunTo(float targetU) {
        double sunU = 0.5 + Math.atan2(sunDirection.x, -sunDirection.z) / (2 * Math.PI);
        int shift = (int) Math.round((targetU - sunU) * width);
        float[] row = new float[width * 3];
        for (int y = 0; y < height; y++) {
            System.arraycopy(rgb, y * width * 3, row, 0, width * 3);
            for (int x = 0; x < width; x++) {
                int sx = Math.floorMod(x - shift, width);
                System.arraycopy(row, sx * 3, rgb, (y * width + x) * 3, 3);
            }
        }
        double phi = (targetU - 0.5) * 2 * Math.PI;
        double horiz = Math.sqrt(1 - sunDirection.y * sunDirection.y);
        sunDirection.set((float) (Math.sin(phi) * horiz), sunDirection.y, (float) (-Math.cos(phi) * horiz)).normalize();
    }
}
