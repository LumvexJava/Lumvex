package dev.lumvex.game;

import dev.lumvex.ui.UiDoc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * A 2D overlay on top of the 3D view, such as a HUD or a menu. Subclasses keep their elements in fields, so scripts
 * can update them directly:
 *
 * <pre>{@code
 * public class Hud extends Canvas {
 *     final Label score = add(new Label("Score", "Coins: 0"));
 *
 *     public Hud() {
 *         super("HUD");
 *         score.setPosition(Anchor.TOP_LEFT, 40, 30);
 *     }
 * }
 *
 * // in a script
 * ui(Hud.class).score.setText("Coins: 3");
 * }</pre>
 *
 * Hiding a canvas hides all of its elements.
 */
public class Canvas {
    final UiDoc doc;
    /** Key of the canvas in the running game: the asset id it was loaded from, or one derived from its name. */
    final String assetId;
    /** The running game, or null while the canvas is being set up. */
    ScriptRuntime rt;
    final List<UiScript> scripts = new ArrayList<>();
    private final Map<UiDoc.Element, UiElement> handles = new IdentityHashMap<>();

    public Canvas(String name) {
        this(new UiDoc(), UiDoc.assetId(name));
        doc.name = name;
    }

    /** Wraps a canvas document loaded from {@code assets/ui/}. */
    Canvas(UiDoc doc, String assetId) {
        this.doc = doc;
        this.assetId = assetId;
        for (UiDoc.Element e : doc.elements) handles.put(e, UiElement.wrap(e));
        for (UiElement el : handles.values()) el.bind(this);
    }

    public String name() { return doc.name; }

    /** Adds a top-level element with its children and returns it. */
    public final <T extends UiElement> T add(T element) {
        if (element.parent() != null || element.canvas != null) throw new IllegalArgumentException(element + " already has a parent");
        register(element);
        return element;
    }

    /** The element with the given name, or null. */
    public UiElement element(String name) {
        UiDoc.Element e = doc.byName(name);
        return e == null ? null : handle(e);
    }

    /** All elements in document order. */
    public List<UiElement> elements() {
        List<UiElement> out = new ArrayList<>();
        for (UiDoc.Element e : doc.elements) out.add(handle(e));
        return out;
    }

    public boolean isVisible() { return doc.visible; }

    public void setVisible(boolean visible) {
        if (doc.visible == visible) return;
        doc.visible = visible;
        changed();
    }

    public void show() { setVisible(true); }

    public void hide() { setVisible(false); }

    public void toggle() { setVisible(!doc.visible); }

    /** A modal canvas, such as a menu, frees the mouse while it is visible and blocks clicks into the world. */
    public void setModal(boolean modal) {
        doc.modal = modal;
        changed();
    }

    /** Draw order between canvases; higher is drawn on top. */
    public void setOrder(int order) {
        doc.order = order;
        changed();
    }

    /** The resolution that element positions and sizes refer to, 1920 x 1080 by default. */
    public void setReferenceSize(int width, int height) {
        doc.refWidth = width;
        doc.refHeight = height;
        changed();
    }

    /** Attaches a script that receives the clicks of all elements, see {@link OnClick}. */
    public <T extends UiScript> T addScript(T script) {
        if (script.canvas != null || script.element != null) throw new IllegalStateException(script.describe() + " is already attached");
        script.canvas = this;
        scripts.add(script);
        started(script);
        return script;
    }

    /** The canvas script of the given type, or null. */
    public <T> T script(Class<T> type) {
        for (UiScript s : scripts) if (type.isInstance(s)) return type.cast(s);
        return null;
    }

    /** Scripts attached to the canvas itself, not to single elements. */
    public List<UiScript> scripts() { return Collections.unmodifiableList(scripts); }

    /** Calls an {@link Event} on the canvas scripts and returns its result. */
    public Object call(String event, Object... args) {
        if (rt == null) throw new IllegalStateException(this + " is not shown in a running game");
        return rt.call(new ArrayList<>(scripts), "UI \"" + doc.name + "\"", event, args);
    }

    @Override public String toString() { return getClass().getSimpleName() + "(" + doc.name + ")"; }

    /** Adds an element subtree to the document. */
    void register(UiElement element) {
        for (UiElement el : element.subtree()) {
            doc.elements.add(el.e);
            handles.put(el.e, el);
            el.bind(this);
            for (UiScript s : el.scripts) started(s);
        }
        changed();
    }

    /** The one handle of a document element; elements added in code keep their own instance. */
    UiElement handle(UiDoc.Element e) {
        return handles.computeIfAbsent(e, k -> {
            UiElement el = UiElement.wrap(k);
            el.bind(this);
            return el;
        });
    }

    /** Starts a script attached after the canvas went live. */
    void started(UiScript script) {
        script.canvas = this;
        if (rt != null) rt.begin(rt.bind(script));
    }

    void changed() { if (rt != null) rt.uiChanged(); }
}
