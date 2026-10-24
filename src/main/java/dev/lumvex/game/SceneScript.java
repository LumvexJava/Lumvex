package dev.lumvex.game;

/**
 * Base class for the script of a scene. A scene has at most one.
 *
 * <p>The scene script sets the scene up in {@link #onLoad()}, owns global state and render settings, switches
 * scenes and offers {@link Event events} to object scripts, e.g. {@code scene().call("score", 10)}.
 */
public abstract class SceneScript extends Script {
    /** Called after the scene is loaded and before any {@code onStart}. A good place to spawn objects. */
    public void onLoad() { }

    /** Name of the running scene. */
    protected final String sceneName() { return rt.sceneName(); }

    /** The scene script of the running game, or null if no game is running. */
    public static SceneScript current() {
        ScriptRuntime r = ScriptRuntime.current();
        return r == null ? null : r.sceneScript();
    }

    /** The scene script of the running game if it is of the given type, otherwise null. */
    public static <T extends SceneScript> T current(Class<T> type) {
        SceneScript s = current();
        return type.isInstance(s) ? type.cast(s) : null;
    }

    /** Stand-in for scenes without a script, so that calls on it report a missing event instead of failing. */
    static final class Empty extends SceneScript { }
}
