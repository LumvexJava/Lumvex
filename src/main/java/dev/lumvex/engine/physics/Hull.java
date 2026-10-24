package dev.lumvex.engine.physics;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * Builds collision hulls from vertex positions (packed xyz arrays).
 *
 * <p>{@link #enclosing} returns a k-DOP that is guaranteed to contain the whole mesh and is used for simple
 * colliders. {@link #extreme} picks actual mesh vertices and is used as the point cloud for Jolt convex shapes.
 */
public final class Hull {
    private Hull() { }

    /**
     * Returns the corners of a k-DOP around the mesh: the intersection of half spaces bounded by the farthest vertex
     * along each of a set of directions. More directions give a tighter fit and more corners; the densest direction
     * set whose hull stays within {@code maxVertices} corners is used, but never less than the bounding box.
     */
    public static float[] enclosing(float[] positions, int maxVertices) {
        float[] best = null;
        for (double[][] dirs : LEVELS) {
            float[] v = kdop(positions, dirs);
            if (best == null || v.length / 3 <= maxVertices) best = v;
            if (v.length / 3 > maxVertices) break;
        }
        return best;
    }

    /** The distinct vertices that are farthest along {@code n} directions spread evenly over the sphere. */
    public static float[] extreme(float[] positions, int n) {
        int count = positions.length / 3;
        Set<Integer> picked = new LinkedHashSet<>();
        for (double[] d : fibonacci(n)) {
            double best = -Double.MAX_VALUE;
            int bi = 0;
            for (int k = 0; k < count; k++) {
                double v = positions[k * 3] * d[0] + positions[k * 3 + 1] * d[1] + positions[k * 3 + 2] * d[2];
                if (v > best) { best = v; bi = k; }
            }
            picked.add(bi);
        }
        float[] out = new float[picked.size() * 3];
        int o = 0;
        for (int k : picked) {
            out[o++] = positions[k * 3];
            out[o++] = positions[k * 3 + 1];
            out[o++] = positions[k * 3 + 2];
        }
        return out;
    }

    // Direction sets for enclosing(), from coarse to fine.
    private static final List<double[][]> LEVELS = new ArrayList<>();

    static {
        double s = Math.sqrt(0.5), t = Math.sqrt(1 / 3.0);
        double[][] axes = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        double[][] xz = {{s, 0, s}, {-s, 0, s}, {s, 0, -s}, {-s, 0, -s}};
        double[][] xyyz = {{s, s, 0}, {-s, s, 0}, {s, -s, 0}, {-s, -s, 0}, {0, s, s}, {0, -s, s}, {0, s, -s}, {0, -s, -s}};
        double[][] corners = {{t, t, t}, {-t, t, t}, {t, -t, t}, {t, t, -t}, {-t, -t, t}, {-t, t, -t}, {t, -t, -t}, {-t, -t, -t}};
        LEVELS.add(axes);                                    // box, 8 corners
        LEVELS.add(concat(axes, xz));                        // box with beveled vertical edges, 16 corners
        LEVELS.add(concat(axes, xz, xyyz));                  // 18-DOP
        LEVELS.add(concat(axes, xz, xyyz, corners));         // 26-DOP
        for (int n : new int[]{40, 56, 80, 112}) LEVELS.add(concat(axes, fibonacci(n)));
    }

    private static double[][] concat(double[][]... parts) {
        List<double[]> l = new ArrayList<>();
        for (double[][] p : parts) l.addAll(List.of(p));
        return l.toArray(double[][]::new);
    }

    /** {@code n} unit vectors on a Fibonacci spiral, i.e. spread evenly over the sphere. */
    private static double[][] fibonacci(int n) {
        double golden = Math.PI * (3 - Math.sqrt(5));
        double[][] d = new double[n][];
        for (int i = 0; i < n; i++) {
            double y = 1 - 2 * (i + 0.5) / n, r = Math.sqrt(1 - y * y), phi = golden * i;
            d[i] = new double[]{Math.cos(phi) * r, y, Math.sin(phi) * r};
        }
        return d;
    }

    /**
     * Corners of the polytope {x : n_i . x <= max_v(n_i . v)}. Brute force over all plane triples, which is fine for
     * the at most ~120 directions used here.
     */
    static float[] kdop(float[] p, double[][] n) {
        int k = n.length, count = p.length / 3;
        double[] d = new double[k];
        for (int i = 0; i < k; i++) {
            double best = -Double.MAX_VALUE;
            for (int v = 0; v < count; v++) best = Math.max(best, p[v * 3] * n[i][0] + p[v * 3 + 1] * n[i][1] + p[v * 3 + 2] * n[i][2]);
            d[i] = best;
        }
        double scale = 1e-5 * Math.max(1, Math.abs(d[0]) + Math.abs(d[1]) + Math.abs(d[2]) + Math.abs(d[3]) + Math.abs(d[4]) + Math.abs(d[5]));
        List<double[]> verts = new ArrayList<>();
        for (int a = 0; a < k; a++) {
            for (int b = a + 1; b < k; b++) {
                for (int c = b + 1; c < k; c++) {
                    double[] x = solve(n[a], n[b], n[c], d[a], d[b], d[c]);
                    if (x == null) continue;
                    boolean inside = true;
                    for (int i = 0; i < k && inside; i++) inside = n[i][0] * x[0] + n[i][1] * x[1] + n[i][2] * x[2] <= d[i] + scale;
                    if (!inside) continue;
                    boolean dup = false;
                    for (double[] v : verts) {
                        if (Math.abs(v[0] - x[0]) + Math.abs(v[1] - x[1]) + Math.abs(v[2] - x[2]) < scale * 10) { dup = true; break; }
                    }
                    if (!dup) verts.add(x);
                }
            }
        }
        float[] out = new float[verts.size() * 3];
        for (int i = 0; i < verts.size(); i++) for (int c = 0; c < 3; c++) out[i * 3 + c] = (float) verts.get(i)[c];
        return out;
    }

    /** Intersection of three planes by Cramer's rule, or null if they are nearly parallel. */
    private static double[] solve(double[] a, double[] b, double[] c, double da, double db, double dc) {
        double[] bc = cross(b, c), ca = cross(c, a), ab = cross(a, b);
        double det = a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2];
        if (Math.abs(det) < 1e-9) return null;
        return new double[]{(da * bc[0] + db * ca[0] + dc * ab[0]) / det, (da * bc[1] + db * ca[1] + dc * ab[1]) / det,
                (da * bc[2] + db * ca[2] + dc * ab[2]) / det};
    }

    private static double[] cross(double[] u, double[] v) {
        return new double[]{u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]};
    }
}
