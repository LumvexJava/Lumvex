package dev.lumvex.game;

import java.util.List;

/**
 * Bridge from the script API to the physics world. Implemented by the engine and only called on the render thread.
 * Bodies are addressed by entity id; vectors are {@code float[3]} in world space.
 */
public interface PhysicsBackend {
    boolean hasBody(String entityId);

    boolean isDynamic(String entityId);

    /** Linear velocity in m/s. */
    float[] velocity(String entityId);

    void setVelocity(String entityId, float x, float y, float z);

    /** Applies an impulse in kg*m/s. */
    void addImpulse(String entityId, float x, float y, float z);

    /** Applies a force in N for the current step. */
    void addForce(String entityId, float x, float y, float z);

    /** Gravity in m/s^2. */
    float[] gravity();

    void setGravity(float x, float y, float z);

    /** Closest hit within {@code maxDistance}, or null. {@code dir} must be normalized. */
    Hit raycast(float[] origin, float[] dir, float maxDistance, String ignoreEntity);

    /** A raycast hit. {@code entityId} is null for geometry that belongs to no entity. */
    record Hit(String entityId, float[] point, float[] normal, float distance) {}

    /** Entities whose colliders currently touch the given entity's collider. */
    List<String> contacts(String entityId);
}
