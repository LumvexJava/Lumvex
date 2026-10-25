package dev.lumvex.game;

import java.util.Locale;

/**
 * A 3D vector used for positions, directions, sizes and Euler angles in degrees.
 *
 * <p>All operations return a new vector. {@link #ZERO} and {@link #UP} are shared instances and must not be
 * modified.
 */
public final class Vec3 {
    public final float x, y, z;

    public Vec3() { this(0, 0, 0); }

    public Vec3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }

    public static Vec3 of(float x, float y, float z) { return new Vec3(x, y, z); }

    /** (0, 0, 0) and (0, 1, 0). */
    public static final Vec3 ZERO = new Vec3(0, 0, 0), UP = new Vec3(0, 1, 0);

    public Vec3 add(Vec3 o) { return new Vec3(x + o.x, y + o.y, z + o.z); }

    public Vec3 add(float dx, float dy, float dz) { return new Vec3(x + dx, y + dy, z + dz); }

    public Vec3 sub(Vec3 o) { return new Vec3(x - o.x, y - o.y, z - o.z); }

    public Vec3 mul(float s) { return new Vec3(x * s, y * s, z * s); }

    public float dot(Vec3 o) { return x * o.x + y * o.y + z * o.z; }

    public Vec3 cross(Vec3 o) { return new Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x); }

    public float length() { return (float) Math.sqrt(dot(this)); }

    public float distance(Vec3 o) { return sub(o).length(); }

    /** Unit vector in the same direction. A zero vector stays zero. */
    public Vec3 normalized() {
        float l = length();
        return l < 1e-9f ? new Vec3() : mul(1 / l);
    }

    /** Linear interpolation: {@code t = 0} returns this, {@code t = 1} returns {@code o}. */
    public Vec3 lerp(Vec3 o, float t) { return new Vec3(x + (o.x - x) * t, y + (o.y - y) * t, z + (o.z - z) * t); }

    public float[] toArray() { return new float[]{x, y, z}; }

    static Vec3 of(float[] a) { return new Vec3(a[0], a[1], a[2]); }

    @Override public String toString() { return String.format(Locale.ROOT, "(%.3f, %.3f, %.3f)", x, y, z); }

    @Override public boolean equals(Object o) { return o instanceof Vec3 v && v.x == x && v.y == y && v.z == z; }

    @Override public int hashCode() { return Float.hashCode(x) * 961 + Float.hashCode(y) * 31 + Float.hashCode(z); }
}
