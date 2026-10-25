package dev.lumvex.game;

import java.util.ArrayList;
import java.util.List;

/**
 * Common base of all scripts. Extend {@link SceneScript}, {@link ObjectScript} or {@link UiScript} instead of this
 * class.
 *
 * <p>Lifecycle: {@link #onStart()} once after the scene is loaded, {@link #onUpdate(float)} every frame and
 * {@link #onStop()} when the game stops, the scene changes or the object is destroyed. Input, timers, collisions
 * and calls from other scripts are handled by annotated methods ({@link OnKey}, {@link OnMouse}, {@link OnClick},
 * {@link Every}, {@link OnCollision}, {@link OnTrigger}, {@link Event}).
 */
public abstract class Script {
    ScriptRuntime rt;
    GameObject self;
    boolean failed;

    /** Called once after the scene is loaded. The scene script starts first, then the objects. */
    public void onStart() { }

    /** Called every frame with the frame time in seconds. */
    public void onUpdate(float dt) { }

    /** Called when the game stops, another scene is loaded or the object is destroyed. */
    public void onStop() { }

    /** The script of the running scene. Scenes without a script get an empty one. */
    protected final SceneScript scene() { return rt.sceneScript(); }

    /** The scene script cast to its type, e.g. {@code scene(House.class).points}, or null if it has another type. */
    protected final <T extends SceneScript> T scene(Class<T> type) {
        SceneScript s = rt.sceneScript();
        return type.isInstance(s) ? type.cast(s) : null;
    }

    /** The first object with this name, or null (a warning is logged). */
    protected final GameObject find(String name) { return rt.find(name, true); }

    /** All objects with this name. */
    protected final List<GameObject> findAll(String name) { return rt.findAll(name); }

    /** All objects with this tag. */
    protected final List<GameObject> findByTag(String tag) { return rt.findByTag(tag); }

    /** The first object of the given class, e.g. {@code find(Player.class)}, or null. */
    protected final <T extends GameObject> T find(Class<T> type) {
        for (GameObject o : rt.objects()) if (type.isInstance(o)) return type.cast(o);
        return null;
    }

    /** All objects of the given class, e.g. {@code findAll(Coin.class)}. */
    protected final <T extends GameObject> List<T> findAll(Class<T> type) {
        List<T> out = new ArrayList<>();
        for (GameObject o : rt.objects()) if (type.isInstance(o)) out.add(type.cast(o));
        return out;
    }

    /** All objects in the running scene. */
    protected final List<GameObject> objects() { return rt.objects(); }

    /** The player camera, or null if the scene has none. */
    protected final GameObject camera() { return rt.camera(); }

    /**
     * Creates an object from a model ({@code "models/x.glb"}), a prefab ({@code "prefabs/x.prefab.json"}) or a
     * primitive ({@code "box"}, {@code "cube"}, {@code "sphere"}, {@code "plane"}, {@code "torus"}). Scripts on a
     * prefab's root object are started right away.
     */
    protected final GameObject spawn(String asset, Vec3 at) { return rt.spawn(asset, at); }

    protected final GameObject spawn(String asset, float x, float y, float z) { return rt.spawn(asset, new Vec3(x, y, z)); }

    /** Adds an object, e.g. {@code spawn(new Coin(at))}, with its children to the scene and starts its scripts. */
    protected final <T extends GameObject> T spawn(T object) { return rt.add(object); }

    /** Switches to another scene of the project at the end of the current frame. */
    protected final void loadScene(String name) { rt.requestScene(name); }

    /** Switches to the scene of the given class, e.g. {@code loadScene(Level2.class)}, at the end of the frame. */
    protected final void loadScene(Class<? extends Scene> scene) { rt.requestScene(scene); }

    /** Starts the running scene over with a fresh instance, at the end of the current frame. */
    protected final void reloadScene() { rt.requestScene(rt.sceneName()); }

    /** Ends the game. */
    protected final void stopGame() { rt.requestStop(); }

    /** Current keyboard and mouse input state. */
    protected final Input input() { return rt.input; }

    /** Render settings and post-processing controls. */
    protected final Render render() { return rt.render; }

    /** Frame timing and elapsed-time information. */
    protected final Time time() { return rt.time; }

    /**
     * A UI canvas by name or asset id ({@code "ui/Name.ui.json"}). Canvases that are not part of the scene are
     * loaded on first use.
     */
    protected final Canvas ui(String name) { return rt.ui(name); }

    /** The canvas of the given class, e.g. {@code ui(Hud.class).score.setText("3")}, or null. */
    protected final <T extends Canvas> T ui(Class<T> type) { return rt.ui(type); }

    /** Ray casts and physics operations for the running scene. */
    protected final Physics physics() { return rt.physics; }

    /** Prints the parts to the console, separated by spaces and prefixed with the script and object name. */
    protected final void log(Object... parts) {
        StringBuilder b = new StringBuilder("[").append(describe()).append("]");
        for (Object p : parts) b.append(' ').append(p);
        System.out.println(b);
    }

    /**
     * Calls an {@link Event} of this script by name and returns its result. The arguments are checked against the
     * method's parameters.
     */
    public final Object call(String event, Object... args) { return rt.call(List.of(this), describe(), event, args); }

    /** Whether this script declares an event with the given name. */
    public final boolean hasEvent(String event) { return ScriptClass.of(getClass()).events.containsKey(event); }

    /** "Class @ Object", used in log lines and error messages. */
    String describe() { return getClass().getSimpleName() + (self != null ? " @ " + self.name() : ""); }
}
