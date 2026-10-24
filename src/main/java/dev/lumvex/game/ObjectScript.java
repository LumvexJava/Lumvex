package dev.lumvex.game;

/**
 * Base class for scripts attached to a scene object.
 *
 * <p>One class can be attached to many objects, each with its own {@link Param} values, and an object can carry
 * several scripts. The transform shortcuts below act on {@link #object()}.
 */
public abstract class ObjectScript extends Script {
    /** The object this script is attached to. */
    protected final GameObject object() { return self; }

    /** Same as {@code object().position()}. */
    protected final Vec3 position() { return self.position(); }

    /** Same as {@code object().setPosition(x, y, z)}. */
    protected final void setPosition(float x, float y, float z) { self.setPosition(x, y, z); }

    /** Same as {@code object().move(dx, dy, dz)}. */
    protected final void move(float dx, float dy, float dz) { self.move(dx, dy, dz); }

    /** Same as {@code object().rotate(pitch, yaw, roll)}. */
    protected final void rotate(float pitch, float yaw, float roll) { self.rotate(pitch, yaw, roll); }
}
