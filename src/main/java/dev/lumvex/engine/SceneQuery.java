package dev.lumvex.engine;

import dev.lumvex.render.scene.Mesh;
import dev.lumvex.render.scene.Scene;
import org.joml.Matrix3f;
import org.joml.Matrix4f;
import org.joml.Vector3f;

import java.util.function.IntPredicate;

/**
 * Exact CPU-side ray casts against the render scene.
 *
 * <p>There is no acceleration structure on this side. Instances are culled by their world bounding boxes and the
 * remaining meshes are tested triangle by triangle, which is fast enough for the occasional query from scripts.
 */
public final class SceneQuery {
    private final Scene scene;

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
     * Closest hit along a ray within {@code maxT}, or null.
     *
     * @param dir  normalized direction
     * @param skip instances to ignore; may be null
     */
    public Hit raycast(Vector3f origin, Vector3f dir, float maxT, IntPredicate skip) {
        Hit best = null;
        float bestT = maxT;
        Matrix4f inv = new Matrix4f();
        Vector3f o = new Vector3f(), d = new Vector3f();
        for (int i = 0; i < scene.instances.size(); i++) {
            if (skip != null && skip.test(i)) continue;
            float[] bb = instanceBounds(i);
            if (!rayBox(origin, dir, bb, bestT)) continue;
            Scene.Instance inst = scene.instances.get(i);
            inst.transform.invert(inv);
            inv.transformPosition(o.set(origin));
            // left unnormalized so t stays a world distance
            inv.transformDirection(d.set(dir));
            Mesh m = scene.meshes.get(inst.mesh);
            float[] v = m.vertices;
            int[] idx = m.indices;
            int st = Mesh.FLOATS_PER_VERTEX;
            int bestTri = -1;
            for (int t = 0; t < idx.length; t += 3) {
                int a = idx[t] * st, b = idx[t + 1] * st, c = idx[t + 2] * st;
                float hit = triangle(o, d, v[a], v[a + 1], v[a + 2], v[b], v[b + 1], v[b + 2], v[c], v[c + 1], v[c + 2]);
                if (hit > 1e-5f && hit < bestT) {
                    bestT = hit;
                    bestTri = t;
                }
            }
            if (bestTri < 0) continue;
            int a = idx[bestTri] * st, b = idx[bestTri + 1] * st, c = idx[bestTri + 2] * st;
            Vector3f e1 = new Vector3f(v[b] - v[a], v[b + 1] - v[a + 1], v[b + 2] - v[a + 2]);
            Vector3f e2 = new Vector3f(v[c] - v[a], v[c + 1] - v[a + 1], v[c + 2] - v[a + 2]);
            Vector3f n = e1.cross(e2);
            // normals transform with the inverse transpose
            new Matrix3f(inv).transpose().transform(n).normalize();
            if (n.dot(dir) > 0) n.negate();
            best = new Hit(i, bestT, new Vector3f(dir).mul(bestT).add(origin), n);
        }
        return best;
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

    // Slab test against an AABB, limited to [0, maxT].
    private static boolean rayBox(Vector3f o, Vector3f d, float[] b, float maxT) {
        float tn = 0, tf = maxT;
        for (int c = 0; c < 3; c++) {
            float oc = o.get(c), dc = d.get(c);
            if (Math.abs(dc) < 1e-12f) {
                if (oc < b[c] || oc > b[3 + c]) return false;
                continue;
            }
            float t0 = (b[c] - oc) / dc, t1 = (b[3 + c] - oc) / dc;
            if (t0 > t1) { float x = t0; t0 = t1; t1 = x; }
            tn = Math.max(tn, t0);
            tf = Math.min(tf, t1);
            if (tn > tf) return false;
        }
        return true;
    }

}
