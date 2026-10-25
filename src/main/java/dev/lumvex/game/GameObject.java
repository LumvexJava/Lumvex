package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc;
import dev.lumvex.engine.SceneDoc.Entity;
import org.joml.Matrix4f;
import org.joml.Quaternionf;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * An object in a scene: a transform with {@link Component components}, {@link ObjectScript scripts} and child
 * objects.
 *
 * <p>Reusable objects are subclasses that set themselves up in the constructor:
 *
 * <pre>{@code
 * public class Crate extends GameObject {
 *     public Crate(Vec3 position) {
 *         super("Crate");
 *         setPosition(position);
 *         addComponent(new MeshRenderer(Mesh.cube(1), Material.color(Color.hex("#A0703C"))));
 *         addComponent(new RigidBody(BodyType.DYNAMIC));
 *     }
 * }
 * }</pre>
 *
 * The same instance lives on while the game runs, so scripts can cast what they find to its class, see
 * {@link Script#findAll(Class)}.
 *
 * <p>Transforms are local to the parent. Rotations are Euler angles in degrees: x is pitch, y is yaw, z is roll.
 * Objects face along their local -z axis, see {@link #forward()}.
 */
public class GameObject {
    /** The running game, or null while the object is only being set up. */
    ScriptRuntime rt;
    final Entity e;
    final List<Script> scripts = new ArrayList<>();
    private final List<Component> components = new ArrayList<>();
    /** Children added before the object is part of a running game. */
    private final List<GameObject> children = new ArrayList<>();
    private GameObject parent;
    boolean destroyed;

    public GameObject(String name) { this(new Entity(name)); }

    /** Wraps an entity of a loaded scene document. */
    GameObject(Entity e) {
        this.e = e;
        for (SceneDoc.Component data : e.components) {
            Component c = Component.wrap(data);
            if (c != null) {
                c.attach(this);
                components.add(c);
            }
        }
    }

    public String name() { return e.name; }

    public void setName(String name) { e.name = name; }

    /** A unique id, stable across saves. */
    public String id() { return e.id; }

    public boolean isDestroyed() { return destroyed; }

    public List<String> tags() { return e.tags == null ? List.of() : Collections.unmodifiableList(e.tags); }

    public boolean hasTag(String tag) { return e.tags != null && e.tags.contains(tag); }

    public void addTag(String tag) {
        if (e.tags == null) e.tags = new ArrayList<>();
        if (!e.tags.contains(tag)) e.tags.add(tag);
    }

    public void removeTag(String tag) { if (e.tags != null) e.tags.remove(tag); }

    public Vec3 position() { return Vec3.of(e.position); }

    /**
     * Moves the object to a local position. While the game runs, an object with a blocking collider is swept
     * against other colliders: it stops at obstacles and slides along walls. Use {@link #teleport} to skip that.
     */
    public void setPosition(Vec3 p) {
        e.position = p.toArray();
        moved();
    }

    public void setPosition(float x, float y, float z) { setPosition(new Vec3(x, y, z)); }

    /**
     * Places the object at a local position without collision checks, e.g. for spawn points or elevators. Dynamic
     * physics bodies continue from the new position.
     */
    public void teleport(Vec3 p) {
        e.position = p.toArray();
        if (rt != null) rt.teleported(this);
    }

    public void teleport(float x, float y, float z) { teleport(new Vec3(x, y, z)); }

    /** Moves by an offset in meters, with the same collision handling as {@link #setPosition}. */
    public void move(Vec3 offset) { setPosition(position().add(offset)); }

    public void move(float dx, float dy, float dz) { setPosition(position().add(dx, dy, dz)); }

    /** Rotation in degrees (pitch, yaw, roll). */
    public Vec3 rotation() { return Vec3.of(e.rotation); }

    public void setRotation(Vec3 r) {
        e.rotation = r.toArray();
        moved();
    }

    public void setRotation(float pitch, float yaw, float roll) { setRotation(new Vec3(pitch, yaw, roll)); }

    /** Adds to the current rotation, e.g. {@code rotate(0, 45 * dt, 0)} every frame. */
    public void rotate(float pitch, float yaw, float roll) { setRotation(rotation().add(pitch, yaw, roll)); }

    public Vec3 scale() { return Vec3.of(e.scale); }

    public void setScale(Vec3 s) {
        e.scale = s.toArray();
        moved();
    }

    public void setScale(float uniform) { setScale(new Vec3(uniform, uniform, uniform)); }

    public Vec3 worldPosition() {
        Vector3f t = worldMatrix().getTranslation(new Vector3f());
        return new Vec3(t.x, t.y, t.z);
    }

    /** Unit vector the object faces, in world space. For the camera this is the view direction. */
    public Vec3 forward() { return direction(0, 0, -1); }

    public Vec3 right() { return direction(1, 0, 0); }

    public Vec3 up() { return direction(0, 1, 0); }

    /** Sets yaw and pitch so the object faces a world point. Roll is kept. */
    public void lookAt(Vec3 target) {
        Vec3 d = target.sub(worldPosition()).normalized();
        if (d.length() == 0) return;
        float pitch = (float) Math.toDegrees(Math.asin(Math.max(-1, Math.min(1, d.y))));
        // Camera pitch is positive when looking down, object pitch when tilting up.
        if (isCamera()) setRotation(-pitch, (float) Math.toDegrees(Math.atan2(d.x, -d.z)), 0);
        else setRotation(pitch, (float) Math.toDegrees(Math.atan2(-d.x, -d.z)), e.rotation[2]);
    }

    public void lookAt(GameObject other) { lookAt(other.worldPosition()); }

    public float distanceTo(GameObject other) { return worldPosition().distance(other.worldPosition()); }

    public float distanceTo(Vec3 point) { return worldPosition().distance(point); }

    /** The parent object, or null for a root object. */
    public GameObject parent() { return rt == null ? parent : e.parent == null ? null : rt.object(e.parent); }

    /**
     * Adds a child that follows this object. Its transform is relative to this object. While the game runs, the
     * child is added to the scene as well.
     */
    public <T extends GameObject> T addChild(T child) {
        GameObject c = child;
        if (c == this || c.parent() != null) throw new IllegalArgumentException(child + " already has a parent");
        c.e.parent = e.id;
        c.parent = this;
        if (rt == null) children.add(c);
        else rt.add(c);
        return child;
    }

    /**
     * Re-parents the object while the game runs, or detaches it with {@code null}. The world transform is
     * preserved; afterwards the object follows its new parent.
     */
    public void setParent(GameObject parent) {
        ScriptRuntime r = runtime();
        Matrix4f world = worldMatrix();
        Matrix4f local = parent == null ? world : parent.worldMatrix().invert().mul(world);
        e.parent = parent == null ? null : parent.e.id;
        this.parent = parent;
        Vector3f t = local.getTranslation(new Vector3f());
        Vector3f s = local.getScale(new Vector3f());
        Vector3f r3 = local.getNormalizedRotation(new Quaternionf()).getEulerAnglesYXZ(new Vector3f());
        e.position = new float[]{t.x, t.y, t.z};
        e.rotation = new float[]{(float) Math.toDegrees(r3.x), (float) Math.toDegrees(r3.y), (float) Math.toDegrees(r3.z)};
        e.scale = new float[]{s.x, s.y, s.z};
        r.moved(this);
        r.structureChanged();
    }

    public List<GameObject> children() {
        if (rt == null) return Collections.unmodifiableList(children);
        List<GameObject> out = new ArrayList<>();
        for (GameObject o : rt.objects()) if (e.id.equals(o.e.parent)) out.add(o);
        return out;
    }

    /**
     * Attaches a component and returns it. An object has at most one component of each class, and either a
     * {@link Collider} or a {@link RigidBody}.
     */
    public <T extends Component> T addComponent(T component) {
        if (component(component.getClass()) != null) {
            throw new IllegalStateException(this + " already has a " + component.getClass().getSimpleName());
        }
        if ((component instanceof Collider && component(RigidBody.class) != null)
                || (component instanceof RigidBody && component(Collider.class) != null)) {
            throw new IllegalStateException(this + " can have a Collider or a RigidBody, not both");
        }
        component.attach(this);
        components.add(component);
        e.components.add(component.data());
        structureChanged();
        return component;
    }

    /** The component of the given class, or null. */
    public <T extends Component> T component(Class<T> type) {
        for (Component c : components) if (type.isInstance(c)) return type.cast(c);
        return null;
    }

    public List<Component> components() { return Collections.unmodifiableList(components); }

    public void removeComponent(Component component) {
        if (!components.remove(component)) return;
        e.components.remove(component.data());
        structureChanged();
    }

    /**
     * Attaches a script and returns it. While the game runs, the script starts right away; otherwise it starts with
     * the scene.
     */
    public <T extends ObjectScript> T addScript(T script) {
        if (script.self != null) throw new IllegalStateException(script.describe() + " is already attached");
        script.self = this;
        scripts.add(script);
        if (rt != null) rt.begin(rt.bind(script));
        return script;
    }

    /** The attached script of the given type, or null. Allows typed calls: {@code script(Door.class).open(90)}. */
    public <T> T script(Class<T> type) {
        for (Script s : scripts) if (type.isInstance(s)) return type.cast(s);
        return null;
    }

    public List<Script> scripts() { return Collections.unmodifiableList(scripts); }

    /**
     * Calls an {@link Event} on this object's scripts, e.g. {@code find("Door").call("open", 45f)}, and returns the
     * first result. Arguments are checked against the method signature.
     */
    public Object call(String event, Object... args) { return runtime().call(scripts, "object \"" + name() + "\"", event, args); }

    public boolean hasEvent(String event) {
        for (Script s : scripts) if (s.hasEvent(event)) return true;
        return false;
    }

    /** The instanced copies, or null if the object has no instances component. */
    public Instances instances() {
        SceneDoc.InstancesComponent ic = e.get(SceneDoc.InstancesComponent.class);
        return ic == null ? null : new Instances(this, ic, runtime());
    }

    /**
     * Creates a copy of this object in the running scene, with copies of its components and of the scripts that the
     * scene document names. Objects defined as classes are better spawned again: {@code spawn(new Coin(at))}.
     */
    public GameObject copy() { return runtime().copy(this); }

    /** Removes the object and its children from the running scene. */
    public void destroy() { runtime().destroy(this); }

    @Override public String toString() { return getClass().getSimpleName() + "(" + name() + ")"; }

    /** All objects of this subtree, parents first. */
    List<GameObject> subtree() {
        List<GameObject> out = new ArrayList<>();
        out.add(this);
        for (GameObject c : children) out.addAll(c.subtree());
        return out;
    }

    /** Releases the children list once the runtime tracks the hierarchy. */
    void bind(ScriptRuntime rt) {
        this.rt = rt;
        children.clear();
    }

    boolean isCamera() { return component(Camera.class) != null; }

    void structureChanged() { if (rt != null) rt.structureChanged(); }

    private void moved() { if (rt != null) rt.moved(this); }

    private ScriptRuntime runtime() {
        if (rt == null) throw new IllegalStateException(this + " is not part of a running scene yet");
        return rt;
    }

    private Matrix4f worldMatrix() {
        if (rt != null) return rt.doc.worldMatrix(e, new Matrix4f());
        Matrix4f local = e.localMatrix(new Matrix4f());
        return parent == null ? local : parent.worldMatrix().mul(local);
    }

    private Vec3 direction(float x, float y, float z) {
        Matrix4f m;
        if (isCamera()) {
            // The renderer's camera builds its view as Rx(pitch) * Ry(yaw); match it here.
            m = new Matrix4f().rotateX((float) Math.toRadians(e.rotation[0])).rotateY((float) Math.toRadians(e.rotation[1])).invert();
        } else {
            m = worldMatrix();
        }
        Vector3f d = m.transformDirection(new Vector3f(x, y, z)).normalize();
        return new Vec3(d.x, d.y, d.z);
    }
}
