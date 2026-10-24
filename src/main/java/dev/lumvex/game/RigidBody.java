package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.ColliderComponent;

/**
 * A body simulated by the physics engine. An object has either a rigid body or a {@link Collider}, not both.
 *
 * <p>Dynamic bodies are moved by the simulation; push them with impulses, forces or velocities while the game
 * runs. Static and kinematic bodies are moved through the object's transform.
 */
public final class RigidBody extends Component {
    private final ColliderComponent data;

    /** A body shaped like the object's geometry. */
    public RigidBody(BodyType type) {
        this(new ColliderComponent());
        data.mode = "physics";
        data.body = type.name().toLowerCase();
    }

    /** A box-shaped body around the object's origin. */
    public RigidBody(BodyType type, Vec3 boxSize) {
        this(type);
        data.size = boxSize.toArray();
        data.offset = new float[3];
    }

    RigidBody(ColliderComponent data) { this.data = data; }

    public BodyType type() { return BodyType.valueOf(data.body.toUpperCase()); }

    /** Mass in kg. Only dynamic bodies use it. */
    public void setMass(float kg) {
        data.mass = kg;
        changed();
    }

    public float mass() { return data.mass; }

    /** Sliding friction, 0 (ice) to 1 (rubber). */
    public void setFriction(float friction) {
        data.friction = friction;
        changed();
    }

    /** Restitution, 0 (no bounce) to 1 (keeps all its speed). */
    public void setBounciness(float bounciness) {
        data.restitution = bounciness;
        changed();
    }

    // Running game

    /** Whether the simulation moves this body. False before the game runs. */
    public boolean isSimulated() {
        PhysicsBackend p = physics(false);
        return p != null && p.isDynamic(gameObject().id());
    }

    /** Linear velocity in m/s. */
    public Vec3 velocity() {
        float[] v = physics(true).velocity(gameObject().id());
        return new Vec3(v[0], v[1], v[2]);
    }

    public void setVelocity(Vec3 v) { physics(true).setVelocity(gameObject().id(), v.x, v.y, v.z); }

    /** Applies an impulse in kg*m/s, e.g. for a jump or a kick. */
    public void addImpulse(Vec3 impulse) { physics(true).addImpulse(gameObject().id(), impulse.x, impulse.y, impulse.z); }

    /** Applies a force in N for the current frame. Call it every frame for a continuous force. */
    public void addForce(Vec3 force) { physics(true).addForce(gameObject().id(), force.x, force.y, force.z); }

    @Override ColliderComponent data() { return data; }

    private PhysicsBackend physics(boolean required) {
        GameObject o = gameObject();
        PhysicsBackend p = o == null || o.rt == null ? null : o.rt.physicsBackend;
        if (p == null && required) throw new IllegalStateException("the physics of " + this + " only exists while the game runs");
        return p;
    }

    @Override public String toString() { return "RigidBody(" + (gameObject() == null ? "unattached" : gameObject().name()) + ")"; }
}
