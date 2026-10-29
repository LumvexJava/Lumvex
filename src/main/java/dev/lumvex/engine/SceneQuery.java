package dev.lumvex.engine;

import dev.lumvex.render.scene.Mesh;
import dev.lumvex.render.scene.Scene;
import org.joml.Matrix3f;
import org.joml.Matrix4f;
import org.joml.Vector3f;

import java.util.Arrays;

import java.util.function.IntPredicate;

/**
 * Exact CPU-side ray casts against the render scene.
 *
 * <p>There is no acceleration structure on this side. Instances are culled by their world bounding boxes and the
 * remaining meshes are tested triangle by triangle, which is fast enough for the occasional query from scripts.
 */
public final class SceneQuery {
    /** Side of a grid cell in meters. */
    private static final float CELL = 4f;

    /** Most cells per axis; larger scenes get larger cells. */
    private static final int MAX_CELLS = 512;
    private final Scene scene;
    // Grid state, valid for the revisions below.
    private int builtStructure = -1, builtTransforms = -1, builtCount = -1;
    /** World bounds of every static instance, 6 floats each. */
    private float[] bounds = new float[0];
    private int[] dynamicInstances = new int[0];
    private float gridX, gridZ, cell;
    private int nx, nz;
    /** Instances per cell in CSR form: cellItems[cellStart[c] .. cellStart[c + 1]). */
    private int[] cellStart = new int[2], cellItems = new int[0];
    /** Per-query stamps, so an instance spanning several cells is tested once. */
    private int[] visited = new int[0];
    private int stamp;

    public SceneQuery(Scene scene) { this.scene = scene; }

    /** A ray hit. {@code t} is the distance along the ray; the normal is the geometric normal, facing the ray. */
    public record Hit(int instance, float t, Vector3f point, Vector3f normal) {}

    /** World-space AABB of a render instance as {minx, miny, minz, maxx, maxy, maxz}. */
    public float[] instanceBounds(int i) {
        Scene.Instance inst = scene.instances.get(i);
        float[] b = scene.meshes.get(inst.mesh).bounds();
        float[] r = {Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE};
        Vector3f p = new Vector3f();
        for (int c = 0; c < 8; c++) {
            inst.transform.transformPosition(p.set((c & 1) == 0 ? b[0] : b[3], (c & 2) == 0 ? b[1] : b[4], (c & 4) == 0 ? b[2] : b[5]));
            r[0] = Math.min(r[0], p.x); r[1] = Math.min(r[1], p.y); r[2] = Math.min(r[2], p.z);
            r[3] = Math.max(r[3], p.x); r[4] = Math.max(r[4], p.y); r[5] = Math.max(r[5], p.z);
        }
        return r;
    }

    /**
     * Rebuilds the grid when the scene's structure or transforms changed: the world bounds of every static instance,
     * filed in two passes (count per cell, then fill) under every cell of an x-z grid they touch; moving instances
     * are kept in a list of their own and tested every time.
     */
    private void ensureGrid() {
        int n = scene.instances.size();
        if (builtStructure == scene.structureRevision && builtTransforms == scene.transformRevision && builtCount == n) return;
        builtStructure = scene.structureRevision;
        builtTransforms = scene.transformRevision;
        builtCount = n;
        bounds = new float[6 * n];
        visited = new int[n];
        stamp = 0;
        float minX = Float.MAX_VALUE, minZ = Float.MAX_VALUE, maxX = -Float.MAX_VALUE, maxZ = -Float.MAX_VALUE;
        int dyn = 0;
        for (int i = 0; i < n; i++) {
            if (scene.instances.get(i).isDynamic()) { dyn++; continue; }
            float[] b = instanceBounds(i);
            System.arraycopy(b, 0, bounds, 6 * i, 6);
            minX = Math.min(minX, b[0]); minZ = Math.min(minZ, b[2]);
            maxX = Math.max(maxX, b[3]); maxZ = Math.max(maxZ, b[5]);
        }
        dynamicInstances = new int[dyn];
        for (int i = 0, k = 0; i < n; i++) if (scene.instances.get(i).isDynamic()) dynamicInstances[k++] = i;
        if (minX > maxX) minX = maxX = minZ = maxZ = 0;
        cell = Math.max(CELL, Math.max(maxX - minX, maxZ - minZ) / MAX_CELLS);
        gridX = minX;
        gridZ = minZ;
        nx = Math.max(1, (int) Math.ceil((maxX - minX) / cell) + 1);
        nz = Math.max(1, (int) Math.ceil((maxZ - minZ) / cell) + 1);
        int cells = nx * nz;
        int[] start = new int[cells + 1];
        for (int pass = 0; pass < 2; pass++) {
            int[] fill = pass == 1 ? Arrays.copyOf(start, cells) : null;
            for (int i = 0; i < n; i++) {
                if (scene.instances.get(i).isDynamic()) continue;
                int x0 = cellX(bounds[6 * i]), x1 = cellX(bounds[6 * i + 3]);
                int z0 = cellZ(bounds[6 * i + 2]), z1 = cellZ(bounds[6 * i + 5]);
                for (int z = z0; z <= z1; z++) {
                    for (int x = x0; x <= x1; x++) {
                        if (pass == 0) start[z * nx + x + 1]++;
                        else cellItems[fill[z * nx + x]++] = i;
                    }
                }
            }
            if (pass == 0) {
                for (int c = 1; c <= cells; c++) start[c] += start[c - 1];
                cellItems = new int[start[cells]];
            }
        }
        cellStart = start;
    }

    /** The grid column of x, clamped to the grid. */
    private int cellX(float x) { return Math.clamp((int) Math.floor((x - gridX) / cell), 0, nx - 1); }

    /** The grid row of z, clamped to the grid. */
    private int cellZ(float z) { return Math.clamp((int) Math.floor((z - gridZ) / cell), 0, nz - 1); }

    /**
     * Closest hit along a ray within {@code maxT}, or null. Tests the moving instances, then walks the part of the
     * ray above the grid cell by cell (2D DDA) and stops once the closest hit lies before the next cell.
     *
     * @param dir  normalized direction
     * @param skip instances to ignore; may be null
     */
    public Hit raycast(Vector3f origin, Vector3f dir, float maxT, IntPredicate skip) {
        ensureGrid();
        Query q = new Query(origin, dir, maxT, skip);
        for (int i : dynamicInstances) test(q, i, instanceBounds(i), 0);
        if (cellItems.length == 0) return q.best;
        if (++stamp == Integer.MAX_VALUE) {
            Arrays.fill(visited, 0);
            stamp = 1;
        }
        float tn = 0, tf = maxT;
        float[] lo = {gridX, gridZ}, hi = {gridX + nx * cell, gridZ + nz * cell}, oo = {origin.x, origin.z}, dd = {dir.x, dir.z};
        for (int c = 0; c < 2; c++) {
            if (Math.abs(dd[c]) < 1e-9f) {
                if (oo[c] < lo[c] || oo[c] > hi[c]) return q.best;
                continue;
            }
            float t0 = (lo[c] - oo[c]) / dd[c], t1 = (hi[c] - oo[c]) / dd[c];
            if (t0 > t1) { float x = t0; t0 = t1; t1 = x; }
            tn = Math.max(tn, t0);
            tf = Math.min(tf, t1);
        }
        if (tn > tf) return q.best;
        int cx = cellX(origin.x + dir.x * tn), cz = cellZ(origin.z + dir.z * tn);
        int sx = dir.x > 0 ? 1 : -1, sz = dir.z > 0 ? 1 : -1;
        boolean flatX = Math.abs(dir.x) < 1e-9f, flatZ = Math.abs(dir.z) < 1e-9f;
        float dtx = flatX ? Float.MAX_VALUE : cell / Math.abs(dir.x);
        float dtz = flatZ ? Float.MAX_VALUE : cell / Math.abs(dir.z);
        float nextX = flatX ? Float.MAX_VALUE : (gridX + (cx + (sx > 0 ? 1 : 0)) * cell - origin.x) / dir.x;
        float nextZ = flatZ ? Float.MAX_VALUE : (gridZ + (cz + (sz > 0 ? 1 : 0)) * cell - origin.z) / dir.z;
        while (true) {
            int c = cz * nx + cx;
            for (int k = cellStart[c]; k < cellStart[c + 1]; k++) {
                int i = cellItems[k];
                if (visited[i] == stamp) continue;
                visited[i] = stamp;
                test(q, i, bounds, 6 * i);
            }
            float cellEnd = Math.min(nextX, nextZ);
            if (q.bestT <= cellEnd || cellEnd > tf) break;
            if (nextX < nextZ) {
                cx += sx;
                nextX += dtx;
                if (cx < 0 || cx >= nx) break;
            } else {
                cz += sz;
                nextZ += dtz;
                if (cz < 0 || cz >= nz) break;
            }
        }
        return q.best;
    }

    /** State of one ray cast. */
    private static final class Query {
        final Vector3f origin, dir;
        final IntPredicate skip;
        final Matrix4f inv = new Matrix4f();
        final Vector3f o = new Vector3f(), d = new Vector3f();
        float bestT;
        Hit best;

        /** A query for the closest hit before {@code maxT}. */
        Query(Vector3f origin, Vector3f dir, float maxT, IntPredicate skip) {
            this.origin = origin;
            this.dir = dir;
            this.skip = skip;
            bestT = maxT;
        }
    }

    /**
     * Tests instance i, whose world bounds are bb[off .. off + 6), and records a closer hit in q. The ray is moved
     * into mesh space without normalizing its direction, so t stays a world distance; the normal goes back with the
     * inverse transpose.
     */
    private void test(Query q, int i, float[] bb, int off) {
        if (q.skip != null && q.skip.test(i)) return;
        if (!rayBox(q.origin, q.dir, bb, off, q.bestT)) return;
        Scene.Instance inst = scene.instances.get(i);
        Matrix4f inv = inst.transform.invert(q.inv);
        Vector3f o = inv.transformPosition(q.o.set(q.origin));
        Vector3f d = inv.transformDirection(q.d.set(q.dir));
        Mesh m = scene.meshes.get(inst.mesh);
        float[] v = m.vertices;
        int[] idx = m.indices;
        int st = Mesh.FLOATS_PER_VERTEX;
        int bestTri = -1;
        float bestT = q.bestT;
        for (int t = 0; t < idx.length; t += 3) {
            int a = idx[t] * st, b = idx[t + 1] * st, c = idx[t + 2] * st;
            float hit = triangle(o, d, v[a], v[a + 1], v[a + 2], v[b], v[b + 1], v[b + 2], v[c], v[c + 1], v[c + 2]);
            if (hit > 1e-5f && hit < bestT) {
                bestT = hit;
                bestTri = t;
            }
        }
        if (bestTri < 0) return;
        int a = idx[bestTri] * st, b = idx[bestTri + 1] * st, c = idx[bestTri + 2] * st;
        Vector3f e1 = new Vector3f(v[b] - v[a], v[b + 1] - v[a + 1], v[b + 2] - v[a + 2]);
        Vector3f e2 = new Vector3f(v[c] - v[a], v[c + 1] - v[a + 1], v[c + 2] - v[a + 2]);
        Vector3f n = e1.cross(e2);
        new Matrix3f(inv).transpose().transform(n).normalize();
        if (n.dot(q.dir) > 0) n.negate();
        q.bestT = bestT;
        q.best = new Hit(i, bestT, new Vector3f(q.dir).mul(bestT).add(q.origin), n);
    }

    // Moeller-Trumbore. Returns the ray parameter, or -1 on a miss.
    private static float triangle(Vector3f o, Vector3f d, float ax, float ay, float az, float bx, float by, float bz,
                                  float cx, float cy, float cz) {
        float e1x = bx - ax, e1y = by - ay, e1z = bz - az, e2x = cx - ax, e2y = cy - ay, e2z = cz - az;
        float px = d.y * e2z - d.z * e2y, py = d.z * e2x - d.x * e2z, pz = d.x * e2y - d.y * e2x;
        float det = e1x * px + e1y * py + e1z * pz;
        if (Math.abs(det) < 1e-12f) return -1;
        float inv = 1 / det;
        float tx = o.x - ax, ty = o.y - ay, tz = o.z - az;
        float u = (tx * px + ty * py + tz * pz) * inv;
        if (u < 0 || u > 1) return -1;
        float qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
        float w = (d.x * qx + d.y * qy + d.z * qz) * inv;
        if (w < 0 || u + w > 1) return -1;
        return (e2x * qx + e2y * qy + e2z * qz) * inv;
    }

    // Slab test against the AABB b[off .. off + 6), limited to [0, maxT].
    private static boolean rayBox(Vector3f o, Vector3f d, float[] b, int off, float maxT) {
        float tn = 0, tf = maxT;
        for (int c = 0; c < 3; c++) {
            float oc = o.get(c), dc = d.get(c);
            if (Math.abs(dc) < 1e-12f) {
                if (oc < b[off + c] || oc > b[off + 3 + c]) return false;
                continue;
            }
            float t0 = (b[off + c] - oc) / dc, t1 = (b[off + 3 + c] - oc) / dc;
            if (t0 > t1) { float x = t0; t0 = t1; t1 = x; }
            tn = Math.max(tn, t0);
            tf = Math.min(tf, t1);
            if (tn > tf) return false;
        }
        return true;
    }

}
