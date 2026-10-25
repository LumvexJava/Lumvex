package dev.lumvex.game;

/**
 * The first hit of a {@link Physics#raycast} query.
 *
 * <p>{@code object} is null when the ray hit geometry that does not belong to a scene object. {@code distance} is
 * measured along the ray from its origin.
 */
public record RayHit(GameObject object, Vec3 point, Vec3 normal, float distance) {}
