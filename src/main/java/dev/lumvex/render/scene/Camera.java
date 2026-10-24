package dev.lumvex.render.scene;

import org.joml.Matrix4f;
import org.joml.Vector3f;

/**
 * A first-person camera with yaw and pitch in radians and a vertical field of view in degrees, plus thin-lens depth
 * of field parameters. Projection matrices use Vulkan's depth range.
 */
public final class Camera {
    public final Vector3f position = new Vector3f();
    public float yaw;
    public float pitch;
    public float fovYDegrees = 55f;
    public float aperture = 0.0f;
    public float focusDistance = 8f;

    private final Matrix4f view = new Matrix4f();
    private final Matrix4f proj = new Matrix4f();

    /** World to view: translate by -position, then rotate by yaw and pitch. */
    public Matrix4f view() {
        return view.identity().rotateX(pitch).rotateY(yaw).translate(-position.x, -position.y, -position.z);
    }

    public Matrix4f inverseView(Matrix4f dest) {
        return view().invert(dest);
    }

    /** Inverse projection with near and far planes at 0.01 m and 1000 m. */
    public Matrix4f inverseProjection(float aspect, Matrix4f dest) {
        return proj.setPerspective((float) Math.toRadians(fovYDegrees), aspect, 0.01f, 1000f, true).invert(dest);
    }

    /** Projection times view; the renderer uses last frame's matrix for reprojection. */
    public Matrix4f viewProjection(float aspect, Matrix4f dest) {
        return dest.setPerspective((float) Math.toRadians(fovYDegrees), aspect, 0.01f, 1000f, true).mul(view());
    }

    public Vector3f forward(Vector3f dest) {
        return inverseView(new Matrix4f()).transformDirection(dest.set(0, 0, -1)).normalize();
    }

    public Vector3f right(Vector3f dest) {
        return inverseView(new Matrix4f()).transformDirection(dest.set(1, 0, 0)).normalize();
    }
}
