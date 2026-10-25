package dev.lumvex.engine.physics;

import org.joml.Vector3f;

/**
 * GJK boolean intersection test for convex shapes defined by support functions.
 *
 * <p>The {@code margin} inflates the Minkowski difference: a positive margin also reports shapes that are closer
 * than that distance, a negative one only reports overlaps deeper than it.
 */
final class Gjk {
    private Gjk() { }

    /** A convex shape, known only by its support function. */
    interface Shape {
        /** Writes the point of the shape that is farthest in direction {@code d} to {@code out}. */
        void support(Vector3f d, Vector3f out);
    }

    /** Convex hull of the points {@code from} (inclusive) to {@code to} (exclusive) of a packed xyz array. */
    static Shape points(float[] p, int from, int to) {
        return (d, out) -> {
            float best = -Float.MAX_VALUE;
            int bi = from;
            for (int i = from; i < to; i++) {
                float v = p[i * 3] * d.x + p[i * 3 + 1] * d.y + p[i * 3 + 2] * d.z;
                if (v > best) { best = v; bi = i; }
            }
            out.set(p[bi * 3], p[bi * 3 + 1], p[bi * 3 + 2]);
        };
    }

    static Shape points(float[] p) { return points(p, 0, p.length / 3); }

    /** Whether the shapes intersect, taking the margin into account. Gives up after 64 iterations and says yes. */
    static boolean intersect(Shape a, Shape b, float margin) {
        Vector3f[] s = {new Vector3f(), new Vector3f(), new Vector3f(), new Vector3f()};
        int[] n = {0};
        Vector3f d = new Vector3f(1, 0.31f, 0.17f);
        Vector3f tmp = new Vector3f(), tb = new Vector3f();
        support(a, b, d, margin, s[0], tmp, tb);
        n[0] = 1;
        d.set(s[0]).negate();
        for (int iter = 0; iter < 64; iter++) {
            if (d.lengthSquared() < 1e-14f) return true;
            Vector3f p = new Vector3f();
            support(a, b, d, margin, p, tmp, tb);
            if (p.dot(d) < 0) return false;
            // The newest point always goes to s[0].
            for (int i = n[0]; i > 0; i--) s[i].set(s[i - 1]);
            s[0].set(p);
            n[0]++;
            if (simplex(s, n, d)) return true;
        }
        return true;
    }

    /** Support point of the Minkowski difference a - b, pushed out by {@code margin} along d. */
    private static void support(Shape a, Shape b, Vector3f d, float margin, Vector3f out, Vector3f ta, Vector3f tb) {
        a.support(d, ta);
        b.support(new Vector3f(d).negate(), tb);
        out.set(ta).sub(tb);
        if (margin != 0) {
            float l = d.length();
            if (l > 0) out.fma(margin / l, d);
        }
    }

    /**
     * Reduces the simplex to the feature closest to the origin and points d at the origin. Returns true once the
     * origin is enclosed.
     */
    private static boolean simplex(Vector3f[] s, int[] n, Vector3f d) {
        Vector3f a = s[0], ao = new Vector3f(a).negate();
        switch (n[0]) {
            case 2 -> {
                Vector3f b = s[1], ab = new Vector3f(b).sub(a);
                if (ab.dot(ao) > 0) d.set(ab).cross(ao).cross(ab);
                else { n[0] = 1; d.set(ao); }
                if (d.lengthSquared() < 1e-14f) return true;       // origin lies on the segment
                return false;
            }
            case 3 -> {
                return triangle(s, n, d, ao);
            }
            case 4 -> {
                Vector3f b = s[1], c = s[2], dd = s[3];
                Vector3f ab = new Vector3f(b).sub(a), ac = new Vector3f(c).sub(a), ad = new Vector3f(dd).sub(a);
                Vector3f abc = new Vector3f(ab).cross(ac), acd = new Vector3f(ac).cross(ad), adb = new Vector3f(ad).cross(ab);
                // Make all face normals point away from the opposite vertex.
                if (abc.dot(ad) > 0) abc.negate();
                if (acd.dot(ab) > 0) acd.negate();
                if (adb.dot(ac) > 0) adb.negate();
                if (abc.dot(ao) > 0) {
                    n[0] = 3;
                    return triangle(s, n, d, ao);
                }
                if (acd.dot(ao) > 0) {
                    s[1].set(c);
                    s[2].set(dd);
                    n[0] = 3;
                    return triangle(s, n, d, ao);
                }
                if (adb.dot(ao) > 0) {
                    s[2].set(b);
                    s[1].set(dd);
                    n[0] = 3;
                    return triangle(s, n, d, ao);
                }
                return true;
            }
            default -> {
                return false;
            }
        }
    }

    private static boolean triangle(Vector3f[] s, int[] n, Vector3f d, Vector3f ao) {
        Vector3f a = s[0], b = s[1], c = s[2];
        Vector3f ab = new Vector3f(b).sub(a), ac = new Vector3f(c).sub(a);
        Vector3f abc = new Vector3f(ab).cross(ac);
        if (abc.lengthSquared() < 1e-14f) {                 // degenerate triangle, continue with the edge a-b
            n[0] = 2;
            if (ab.dot(ao) > 0) d.set(ab).cross(ao).cross(ab);
            else { n[0] = 1; d.set(ao); }
            return d.lengthSquared() < 1e-14f;
        }
        if (new Vector3f(abc).cross(ac).dot(ao) > 0) {
            if (ac.dot(ao) > 0) {
                s[1].set(c);
                n[0] = 2;
                d.set(ac).cross(ao).cross(ac);
            } else {
                return segment(s, n, d, ao, ab);
            }
        } else if (new Vector3f(ab).cross(abc).dot(ao) > 0) {
            return segment(s, n, d, ao, ab);
        } else {
            float side = abc.dot(ao);
            if (Math.abs(side) < 1e-9f) return true;          // origin inside the triangle
            if (side > 0) d.set(abc);
            else {
                // Origin below the triangle: flip the winding so the normal faces it.
                Vector3f t = new Vector3f(b);
                s[1].set(c);
                s[2].set(t);
                d.set(abc).negate();
            }
        }
        return d.lengthSquared() < 1e-14f;
    }

    private static boolean segment(Vector3f[] s, int[] n, Vector3f d, Vector3f ao, Vector3f ab) {
        if (ab.dot(ao) > 0) {
            n[0] = 2;
            d.set(ab).cross(ao).cross(ab);
        } else {
            n[0] = 1;
            d.set(ao);
        }
        return d.lengthSquared() < 1e-14f;
    }
}
