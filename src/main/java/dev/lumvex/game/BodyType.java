package dev.lumvex.game;

/** How the physics simulation treats a {@link RigidBody}. */
public enum BodyType {
    /** Never moves. Uses the exact triangles of the object, so it suits floors and level geometry. */
    STATIC,
    /** Simulated: falls, rolls and gets pushed. Uses a convex hull, or a true sphere for sphere meshes. */
    DYNAMIC,
    /** Moved by scripts; pushes dynamic bodies out of the way, e.g. a lift or a door. */
    KINEMATIC
}
