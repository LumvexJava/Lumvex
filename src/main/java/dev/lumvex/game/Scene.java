package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc;
import dev.lumvex.engine.SceneDoc.Entity;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A level of the game. Subclasses add their objects in {@link #build()}:
 *
 * <pre>{@code
 * public class Courtyard extends Scene {
 *     @Override
 *     protected void build() {
 *         lighting().setSun(new Vec3(0.6f, 0.35f, -0.5f), Color.WHITE, 3);
 *         add(new Ground(30));
 *         add(new Player(new Vec3(0, 1.65f, 7)));
 *         setScript(new Rules());
 *     }
 * }
 * }</pre>
 *
 * The game creates a fresh instance whenever the scene is loaded, see {@link Game#addScene}.
 */
public abstract class Scene {
    private final String name;
    private final SceneDoc document;
    private final List<GameObject> objects = new ArrayList<>();
    private final List<Canvas> canvases = new ArrayList<>();
    private final Lighting lighting;
    private final RenderSettings renderSettings;
    private SceneScript script;
    private boolean built;
    /** The running game, or null while the scene is being set up. */
    ScriptRuntime rt;

    /** A scene named after its class. */
    protected Scene() { this((String) null); }

    protected Scene(String name) {
        this.name = name != null ? name : getClass().getSimpleName();
        this.document = new SceneDoc();
        this.document.name = this.name;
        this.lighting = new Lighting(this);
        this.renderSettings = new RenderSettings(this);
    }

    /** Wraps a scene document that was loaded from a file. */
    private Scene(SceneDoc document) {
        this.name = document.name;
        this.document = document;
        this.lighting = new Lighting(this);
        this.renderSettings = new RenderSettings(this);
        for (Entity e : document.entities) objects.add(new GameObject(e));
        built = true;
    }

    /** Adds the objects, canvases and scene script. Called once, before the scene starts. */
    protected abstract void build();

    public final String name() { return name; }

    /**
     * Adds an object together with its children and returns it. While the game runs, its scripts start right
     * away.
     */
    public final <T extends GameObject> T add(T object) {
        if (object.parent() != null) throw new IllegalArgumentException(object + " is a child; add its root object instead");
        if (rt != null) return rt.add(object);
        for (GameObject o : object.subtree()) {
            objects.add(o);
            document.entities.add(o.e);
        }
        return object;
    }

    /** Adds a UI canvas, shown on top of the 3D view. */
    public final <T extends Canvas> T add(T canvas) {
        if (rt != null) throw new IllegalStateException("canvases are added while the scene is built");
        canvases.add(canvas);
        return canvas;
    }

    /** Sets the script that runs the rules of the scene, such as score and win conditions. */
    public final void setScript(SceneScript script) {
        if (rt != null) throw new IllegalStateException("the scene script is set while the scene is built");
        this.script = script;
    }

    public final Lighting lighting() { return lighting; }

    public final RenderSettings renderSettings() { return renderSettings; }

    @Override public String toString() { return "Scene(" + name + ")"; }

    /** Builds the scene on first use and returns its document, which the renderer reads. */
    final SceneDoc document() {
        if (!built) {
            built = true;
            build();
        }
        return document;
    }

    final List<GameObject> objects() { return Collections.unmodifiableList(objects); }

    final List<Canvas> canvases() { return Collections.unmodifiableList(canvases); }

    final SceneScript script() { return script; }

    final void settingsChanged() { if (rt != null) rt.settingsChanged(); }

    /** A scene backed by a document from a project's {@code scenes/} folder. */
    static Scene of(SceneDoc document) { return new Loaded(document); }

    private static final class Loaded extends Scene {
        Loaded(SceneDoc document) { super(document); }

        @Override protected void build() { }
    }
}
