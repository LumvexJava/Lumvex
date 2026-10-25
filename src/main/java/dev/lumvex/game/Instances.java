package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc;
import org.joml.Matrix4f;
import org.joml.Vector3f;

/**
 * The copies of an object with an instances component, obtained with {@link GameObject#instances()}.
 *
 * <p>Instancing renders many copies of one model or prefab (trees, grass, crates) without a scene object per copy.
 * Transforms are relative to the owning object. Every change rebuilds the copies and their colliders, so things
 * that move every frame are better modeled as regular objects.
 *
 * <pre>{@code
 * Instances trees = find("Trees").instances();
 * for (int i = 0; i < trees.count(); i++) {
 *     if (trees.position(i).y < 0) trees.remove(i--);
 * }
 * trees.add(new Vec3(3, 0, 4), 45, 1.2f);   // position, yaw, scale
 * }</pre>
 */
public final class Instances {
    private final GameObject owner;
    private final SceneDoc.InstancesComponent ic;
    private final ScriptRuntime rt;

    Instances(GameObject owner, SceneDoc.InstancesComponent ic, ScriptRuntime rt) {
        this.owner = owner;
        this.ic = ic;
        this.rt = rt;
    }

    public int count() { return ic.count(); }

    /** The instanced model or prefab, e.g. {@code "models/tree_small_02_2k.glb"}. */
    public String source() { return ic.source; }

    /** Position of copy {@code i}, relative to the owning object. */
    public Vec3 position(int i) {
        float[] c = ic.get(check(i));
        return new Vec3(c[0], c[1], c[2]);
    }

    /** Position of copy {@code i} in world space. */
    public Vec3 worldPosition(int i) {
        float[] c = ic.get(check(i));
        Vector3f p = rt.doc.worldMatrix(owner.e, new Matrix4f()).transformPosition(new Vector3f(c[0], c[1], c[2]));
        return new Vec3(p.x, p.y, p.z);
    }

    /** Rotation of copy {@code i} as pitch, yaw, roll in degrees. */
    public Vec3 rotation(int i) {
        float[] c = ic.get(check(i));
        return new Vec3(c[3], c[4], c[5]);
    }

    public Vec3 scale(int i) {
        float[] c = ic.get(check(i));
        return new Vec3(c[6], c[7], c[8]);
    }

    public void setPosition(int i, Vec3 p) { set(i, 0, p.x, p.y, p.z); }

    public void setRotation(int i, float pitch, float yaw, float roll) { set(i, 3, pitch, yaw, roll); }

    public void setScale(int i, float s) { set(i, 6, s, s, s); }

    /** Adds a copy with the given yaw (degrees) and uniform scale and returns its index. */
    public int add(Vec3 position, float yaw, float scale) {
        int n = ic.add(position.x, position.y, position.z, 0, yaw, 0, scale, scale, scale);
        rt.structureChanged();
        return n;
    }

    /** Removes copy {@code i}. The indices of later copies shift down by one. */
    public void remove(int i) {
        ic.remove(check(i));
        rt.structureChanged();
    }

    /** Index of the copy closest to a world position, or -1 if there are no copies. */
    public int nearest(Vec3 world) {
        int best = -1;
        float bd = Float.MAX_VALUE;
        for (int i = 0; i < count(); i++) {
            float d = worldPosition(i).distance(world);
            if (d < bd) { bd = d; best = i; }
        }
        return best;
    }

    // offset: 0 = position, 3 = rotation, 6 = scale
    private void set(int i, int offset, float x, float y, float z) {
        float[] c = ic.get(check(i));
        c[offset] = x;
        c[offset + 1] = y;
        c[offset + 2] = z;
        ic.set(i, c);
        rt.structureChanged();
    }

    private int check(int i) {
        if (i < 0 || i >= ic.count()) throw new IndexOutOfBoundsException("copy " + i + " does not exist (" + ic.count() + " copies)");
        return i;
    }
}
