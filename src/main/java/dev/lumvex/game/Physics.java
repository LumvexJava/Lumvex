package dev.lumvex.game;

import java.util.ArrayList;
import java.util.List;

/**
 * Physics queries, available to scripts as {@code physics()}.
 *
 * <p>Only objects with a collider take part in collisions. Raycasts are different: they are traced against the
 * rendered geometry and hit every visible object.
 */
public final class Physics extends SystemScript {
    private final ScriptRuntime rt;

    Physics(ScriptRuntime rt) { this.rt = rt; }

    private PhysicsBackend b() { return rt.physicsBackend; }

    /** Gravity in m/s^2, by default (0, -9.81, 0). */
    public Vec3 gravity() {
        float[] g = b() == null ? new float[]{0, -9.81f, 0} : b().gravity();
        return new Vec3(g[0], g[1], g[2]);
    }

    /** Sets the gravity in m/s^2. */
    public void gravity(Vec3 g) { if (b() != null) b().setGravity(g.x, g.y, g.z); }

    /** Returns the first visible surface along the ray, or null if nothing is hit within {@code maxDistance}. */
    public RayHit raycast(Vec3 origin, Vec3 direction, float maxDistance) { return raycast(origin, direction, maxDistance, null); }

    /** Like {@link #raycast(Vec3, Vec3, float)}, but ignores one object, typically the one casting the ray. */
    public RayHit raycast(Vec3 origin, Vec3 direction, float maxDistance, GameObject ignore) {
        if (b() == null) return null;
        PhysicsBackend.Hit h = b().raycast(origin.toArray(), direction.normalized().toArray(), maxDistance, ignore == null ? null : ignore.id());
        if (h == null) return null;
        return new RayHit(h.entityId() == null ? null : rt.object(h.entityId()), Vec3.of(h.point()), Vec3.of(h.normal()), h.distance());
    }

    /** Objects whose colliders currently touch the collider of {@code o}. */
    public List<GameObject> touching(GameObject o) {
        List<GameObject> out = new ArrayList<>();
        if (b() == null) return out;
        for (String id : b().contacts(o.id())) {
            GameObject g = rt.object(id);
            if (g != null) out.add(g);
        }
        return out;
    }

    /** Whether the colliders of {@code a} and {@code b} touch right now. */
    public boolean touching(GameObject a, GameObject b) {
        return b() != null && b().contacts(a.id()).contains(b.id());
    }
}
