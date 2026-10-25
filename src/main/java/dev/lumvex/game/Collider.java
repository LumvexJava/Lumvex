package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.ColliderComponent;

/**
 * A collision shape that only scripts move. It blocks objects moved with {@link GameObject#setPosition}, raises
 * collision events and, as a trigger, trigger events. For objects that fall, roll or get pushed, use a
 * {@link RigidBody} instead; an object has at most one of the two.
 *
 * <p>Without a box the shape is a convex hull around the object's geometry.
 */
public final class Collider extends Component {
    private final ColliderComponent data;

    /** A hull around the object's geometry. */
    public Collider() { this(new ColliderComponent()); }

    /** A box around the object's origin. */
    public Collider(Vec3 boxSize) { this(boxSize, Vec3.ZERO); }

    /** A box, shifted from the object's origin by {@code boxOffset}. */
    public Collider(Vec3 boxSize, Vec3 boxOffset) {
        this();
        data.size = boxSize.toArray();
        data.offset = boxOffset.toArray();
    }

    Collider(ColliderComponent data) { this.data = data; }

    /** A trigger never blocks; it only reports objects entering and leaving it, see {@link OnTrigger}. */
    public void setTrigger(boolean trigger) {
        data.trigger = trigger;
        changed();
    }

    public boolean isTrigger() { return data.trigger; }

    /** Corner limit of the hull around the geometry, 4 to 64. More corners fit round shapes more closely. */
    public void setHullVertices(int vertices) {
        data.hullVertices = Math.max(4, Math.min(64, vertices));
        changed();
    }

    @Override ColliderComponent data() { return data; }
}
