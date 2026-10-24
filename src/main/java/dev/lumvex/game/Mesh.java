package dev.lumvex.game;

import java.util.Arrays;

/**
 * A primitive shape for a {@link MeshRenderer}. All shapes are centered at the object's origin and given in meters.
 * Meshes are immutable values; equal shapes share their geometry on the GPU.
 */
public final class Mesh {
    enum Shape { BOX, SPHERE, PLANE, CYLINDER, TORUS }

    private final Shape shape;
    /** Size as stored in scene documents; its meaning depends on the shape. */
    private final float[] size;

    private Mesh(Shape shape, float... size) {
        this.shape = shape;
        this.size = size;
    }

    public static Mesh box(float width, float height, float depth) { return new Mesh(Shape.BOX, width, height, depth); }

    public static Mesh box(Vec3 size) { return box(size.x, size.y, size.z); }

    public static Mesh cube(float edge) { return box(edge, edge, edge); }

    public static Mesh sphere(float radius) { return new Mesh(Shape.SPHERE, radius, radius, radius); }

    /** A square in the xz plane, facing up. */
    public static Mesh plane(float edge) { return new Mesh(Shape.PLANE, edge / 2); }

    /** A closed cylinder along the y axis. */
    public static Mesh cylinder(float radius, float height) { return new Mesh(Shape.CYLINDER, 2 * radius, height, 2 * radius); }

    /** A ring in the xz plane. */
    public static Mesh torus(float majorRadius, float minorRadius) { return new Mesh(Shape.TORUS, majorRadius, minorRadius); }

    String shapeName() { return shape.name().toLowerCase(); }

    float[] size() { return size.clone(); }

    /**
     * Reads the shape of a scene document. The legacy shape {@code cube} is a unit box; unknown shapes fall back to
     * a box, as in the renderer.
     */
    static Mesh of(String shape, float[] size) {
        float[] s = size == null ? new float[]{1, 1, 1} : size;
        return switch (shape == null ? "box" : shape) {
            case "sphere" -> new Mesh(Shape.SPHERE, s);
            case "plane" -> new Mesh(Shape.PLANE, s[0]);
            case "cylinder" -> new Mesh(Shape.CYLINDER, s);
            case "torus" -> new Mesh(Shape.TORUS, s);
            case "cube" -> cube(1);
            default -> new Mesh(Shape.BOX, s);
        };
    }

    @Override public boolean equals(Object o) { return o instanceof Mesh m && m.shape == shape && Arrays.equals(m.size, size); }

    @Override public int hashCode() { return shape.hashCode() * 31 + Arrays.hashCode(size); }

    @Override public String toString() { return "Mesh(" + shapeName() + " " + Arrays.toString(size) + ")"; }
}
