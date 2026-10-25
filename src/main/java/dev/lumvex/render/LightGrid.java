package dev.lumvex.render;

import dev.lumvex.render.scene.Scene;
import org.joml.Vector3f;

import java.nio.ByteBuffer;
import java.util.List;

/**
 * Spatial grid for sampling many small lights. Each cell lists the {@link #K} lights with the largest potential
 * contribution (power over distance to the cell). The path tracer picks its shadow-ray light from the cell list in
 * proportion to the exact weight at the shading point, and with a small probability uniformly from all lights, so
 * the estimator stays unbiased.
 *
 * <p>Without the grid, a scene with about a hundred lights spread over many rooms mostly samples lights in other
 * rooms, and the rare sample of the nearby bulb gets a huge weight, which shows up as speckles in shadowed areas.
 *
 * <p>Layout: the grid follows the lights in the light buffer, in 32-byte entries. The first entry is the header
 * (origin xyz, cell size, dims xyz, uniform share). Each cell takes two entries: 31 16-bit light indices
 * (0xFFFF = empty) and the list length in the 32nd slot.
 */
final class LightGrid {
    /** Lights per cell. The 32nd 16-bit slot holds the list length. */
    static final int K = 31;
    static final int MAX_CELLS = 4096;
    static final int ENTRIES_PER_CELL = 2;            // 32 x 16 bit = 16 words = 2 entries of 8 words
    /** Fraction of shadow rays that pick a light uniformly, so lights outside the cell lists stay reachable. */
    static final float UNIFORM_SHARE = 0.1f;
    /** Margin around the lights' bounding box in meters. */
    private static final float MARGIN = 3f;

    /** Size of the grid in light buffer entries, header included. */
    static int entries() { return 1 + MAX_CELLS * ENTRIES_PER_CELL; }

    private float[] scores = new float[0];
    private int[] order = new int[0];

    /** Builds the grid for the given lights and writes it at byte offset {@code base}. */
    void write(List<Scene.SphereLight> lights, ByteBuffer lb, int base) {
        int n = lights.size();
        if (n == 0) {
            lb.putFloat(base + 12, 0f);               // cell size 0 disables the grid
            return;
        }
        Vector3f lo = new Vector3f(Float.MAX_VALUE), hi = new Vector3f(-Float.MAX_VALUE);
        for (Scene.SphereLight l : lights) {
            lo.min(l.position());
            hi.max(l.position());
        }
        lo.sub(MARGIN, MARGIN, MARGIN);
        hi.add(MARGIN, MARGIN, MARGIN);
        Vector3f ext = new Vector3f(hi).sub(lo);
        float cell = (float) Math.max(0.75, Math.cbrt(ext.x * ext.y * ext.z / MAX_CELLS));
        int dx, dy, dz;
        while (true) {
            dx = Math.max(1, (int) Math.ceil(ext.x / cell));
            dy = Math.max(1, (int) Math.ceil(ext.y / cell));
            dz = Math.max(1, (int) Math.ceil(ext.z / cell));
            if ((long) dx * dy * dz <= MAX_CELLS) break;
            cell *= 1.08f;
        }
        lb.putFloat(base, lo.x).putFloat(base + 4, lo.y).putFloat(base + 8, lo.z).putFloat(base + 12, cell)
                .putFloat(base + 16, dx).putFloat(base + 20, dy).putFloat(base + 24, dz).putFloat(base + 28, UNIFORM_SHARE);

        if (scores.length < n) {
            scores = new float[n];
            order = new int[n];
        }
        float[] power = new float[n];
        for (int i = 0; i < n; i++) {
            Scene.SphereLight l = lights.get(i);
            Vector3f c = l.radiance();
            float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
            power[i] = lum * Math.max(l.radius() * l.radius(), 1e-6f);
        }
        float half = cell * 0.5f;
        int cellsOffset = base + 32;
        for (int z = 0; z < dz; z++) {
            for (int y = 0; y < dy; y++) {
                for (int x = 0; x < dx; x++) {
                    float cx = lo.x + (x + 0.5f) * cell, cy = lo.y + (y + 0.5f) * cell, cz = lo.z + (z + 0.5f) * cell;
                    for (int i = 0; i < n; i++) {
                        Vector3f p = lights.get(i).position();
                        // distance from the light to the cell box
                        float ex = Math.max(Math.abs(p.x - cx) - half, 0f), ey = Math.max(Math.abs(p.y - cy) - half, 0f),
                                ez = Math.max(Math.abs(p.z - cz) - half, 0f);
                        float d2 = ex * ex + ey * ey + ez * ez;
                        scores[i] = power[i] / (d2 + 0.05f);
                        order[i] = i;
                    }
                    int keep = Math.min(K, n);
                    selectTop(n, keep);
                    int o = cellsOffset + ((z * dy + y) * dx + x) * ENTRIES_PER_CELL * 32;
                    for (int j = 0; j < K + 1; j += 2) {
                        int a = j < keep ? order[j] : 0xFFFF, b = j + 1 == K ? keep : j + 1 < keep ? order[j + 1] : 0xFFFF;
                        lb.putInt(o + j * 2, (a & 0xFFFF) | (b & 0xFFFF) << 16);
                    }
                }
            }
        }
    }

    /** Partial selection sort: moves the {@code keep} highest scores to the front of {@code order}. */
    private void selectTop(int n, int keep) {
        for (int j = 0; j < keep; j++) {
            int best = j;
            for (int i = j + 1; i < n; i++) if (scores[order[i]] > scores[order[best]]) best = i;
            int t = order[j];
            order[j] = order[best];
            order[best] = t;
        }
    }
}
