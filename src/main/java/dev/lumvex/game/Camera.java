package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.CameraComponent;

/**
 * Makes an object the player camera. The view looks along the object's forward axis; pitch is positive when
 * looking down. A scene uses its first camera.
 */
public final class Camera extends Component {
    private final CameraComponent data;

    public Camera() { this(new CameraComponent()); }

    Camera(CameraComponent data) { this.data = data; }

    /** Vertical field of view in degrees. */
    public float fieldOfView() { return data.fov; }

    public void setFieldOfView(float degrees) {
        data.fov = degrees;
        changed();
    }

    @Override CameraComponent data() { return data; }
}
