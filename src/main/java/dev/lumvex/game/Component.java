package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc;

/**
 * Built-in data of a {@link GameObject}: what it looks like, how it collides, whether it is the camera or a light.
 * Behavior belongs in scripts, see {@link GameObject#addScript(ObjectScript)}.
 *
 * <p>A component belongs to one object. Changes made while the game runs take effect in the next frame.
 */
public abstract sealed class Component permits MeshRenderer, ModelRenderer, Collider, RigidBody, Camera, PointLight {
    private GameObject owner;

    Component() { }

    /** The object this component is attached to, or null before it is added to one. */
    public final GameObject gameObject() { return owner; }

    /** The scene document form that the renderer and physics read. Shared, not copied. */
    abstract SceneDoc.Component data();

    void attach(GameObject owner) {
        if (this.owner != null && this.owner != owner) {
            throw new IllegalStateException(getClass().getSimpleName() + " already belongs to " + this.owner.name());
        }
        this.owner = owner;
    }

    /** Tells a running game that the object has to be rebuilt. */
    final void changed() {
        if (owner != null) owner.structureChanged();
    }

    /** The typed view of a scene document component, or null for data that has no component class. */
    static Component wrap(SceneDoc.Component data) {
        return switch (data) {
            case SceneDoc.PrimitiveComponent p -> new MeshRenderer(p);
            case SceneDoc.ModelComponent m -> new ModelRenderer(m);
            case SceneDoc.ColliderComponent c -> "physics".equals(c.mode) ? new RigidBody(c) : new Collider(c);
            case SceneDoc.CameraComponent c -> new Camera(c);
            case SceneDoc.LightComponent l -> new PointLight(l);
            default -> null;
        };
    }
}
