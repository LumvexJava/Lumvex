package dev.lumvex.game;

import dev.lumvex.ui.UiDoc;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * An element of a {@link Canvas}: a {@link Label}, {@link Panel}, {@link Button} or {@link Image}. Every element can
 * show a background, an image and text.
 *
 * <p>Positions and sizes are in reference pixels of the canvas, 1920 x 1080 by default; the canvas is scaled to
 * the window. Children are laid out inside their parent and hidden with it.
 */
public abstract class UiElement {
    final UiDoc.Element e;
    Canvas canvas;
    private UiElement parent;
    /** Children added before the element is on a canvas. */
    private final List<UiElement> children = new ArrayList<>();
    final List<UiScript> scripts = new ArrayList<>();

    UiElement(String name, String type) { this(new UiDoc.Element(name, type)); }

    UiElement(UiDoc.Element e) { this.e = e; }

    /** The typed view of an element of a loaded canvas. */
    static UiElement wrap(UiDoc.Element e) {
        return switch (e.type) {
            case UiDoc.TEXT -> new Label(e);
            case UiDoc.BUTTON -> new Button(e);
            case UiDoc.IMAGE -> new Image(e);
            default -> new Panel(e);
        };
    }

    public String name() { return e.name; }

    /** The canvas, or null before the element is added to one. */
    public Canvas canvas() { return canvas; }

    // Layout

    /** Places the element at an anchor of its parent, offset by (x, y). */
    public void setPosition(Anchor anchor, float x, float y) {
        e.anchor = anchor.id;
        e.x = x;
        e.y = y;
        changed();
    }

    /** Moves the element relative to its anchor. */
    public void setPosition(float x, float y) { setPosition(anchor(), x, y); }

    public Anchor anchor() { return Anchor.of(e.anchor); }

    public void setSize(float width, float height) {
        e.width = width;
        e.height = height;
        changed();
    }

    public float width() { return e.width; }

    public float height() { return e.height; }

    // Appearance

    public String text() { return e.text == null ? "" : e.text; }

    public void setText(String text) {
        if (Objects.equals(e.text, text)) return;
        e.text = text;
        changed();
    }

    /** Font size in reference pixels. */
    public void setFont(float size, boolean bold) {
        e.fontSize = size;
        e.bold = bold;
        changed();
    }

    public void setTextAlign(TextAlign align) {
        e.align = align.id();
        changed();
    }

    public TextAlign textAlign() { return TextAlign.of(e.align); }

    public void setTextColor(Color color) {
        e.textColor = color.rgba();
        changed();
    }

    public void setBackground(Color color) {
        e.background = color.rgba();
        changed();
    }

    public void setCornerRadius(float radius) {
        e.cornerRadius = radius;
        changed();
    }

    public void setBorder(Color color, float width) {
        e.borderColor = color.rgba();
        e.borderWidth = width;
        changed();
    }

    // State

    /** Whether the element is actually shown: it, all of its parents and the canvas are visible. */
    public boolean isVisible() {
        if (canvas == null) return e.visible;
        return canvas.doc.visible && canvas.doc.shown(e);
    }

    public void setVisible(boolean visible) {
        if (e.visible == visible) return;
        e.visible = visible;
        changed();
    }

    public void show() { setVisible(true); }

    public void hide() { setVisible(false); }

    /** A disabled element is grayed out and ignores clicks. */
    public void setEnabled(boolean enabled) {
        if (e.enabled == enabled) return;
        e.enabled = enabled;
        changed();
    }

    public boolean isEnabled() { return e.enabled; }

    // Hierarchy

    /** Adds a child that is laid out inside this element, and returns it. */
    public <T extends UiElement> T addChild(T child) {
        UiElement c = child;
        if (c.parent != null || c.canvas != null) throw new IllegalArgumentException(child + " already has a parent");
        c.e.parent = e.id;
        c.parent = this;
        if (canvas == null) children.add(c);
        else canvas.register(c);
        return child;
    }

    /** The parent element, or null for a top-level element. */
    public UiElement parent() {
        if (parent != null || canvas == null) return parent;
        UiDoc.Element p = canvas.doc.find(e.parent);
        return p == null ? null : canvas.handle(p);
    }

    public List<UiElement> children() {
        if (canvas == null) return Collections.unmodifiableList(children);
        List<UiElement> out = new ArrayList<>();
        for (UiDoc.Element c : canvas.doc.children(e.id)) out.add(canvas.handle(c));
        return out;
    }

    // Scripts

    /** Attaches a script that receives the clicks on this element, see {@link OnClick}. */
    public <T extends UiScript> T addScript(T script) {
        if (script.canvas != null || script.element != null) throw new IllegalStateException(script.describe() + " is already attached");
        script.element = this;
        scripts.add(script);
        e.hasCodeScripts = true;
        if (canvas != null) canvas.started(script);
        return script;
    }

    /** The script of the given type attached to this element, or null. */
    public <T> T script(Class<T> type) {
        for (Script s : scripts) if (type.isInstance(s)) return type.cast(s);
        return null;
    }

    /** Calls an {@link Event} on the scripts attached to this element and returns its result. */
    public Object call(String event, Object... args) {
        if (canvas == null || canvas.rt == null) throw new IllegalStateException(this + " is not shown in a running game");
        return canvas.rt.call(new ArrayList<>(scripts), "UI element \"" + e.name + "\"", event, args);
    }

    /** Clicks the element as if the player had clicked it. */
    public void click() {
        if (canvas != null && canvas.rt != null) canvas.rt.uiClick(canvas, this);
    }

    /** Called when the element is clicked, after the scripts' {@link OnClick} handlers. */
    void clicked() { }

    @Override public String toString() { return getClass().getSimpleName() + "(" + name() + ")"; }

    // Internal

    /** This element and its children, parents first. */
    List<UiElement> subtree() {
        List<UiElement> out = new ArrayList<>();
        out.add(this);
        for (UiElement c : children) out.addAll(c.subtree());
        return out;
    }

    void bind(Canvas canvas) {
        this.canvas = canvas;
        children.clear();
    }

    private void changed() {
        if (canvas != null) canvas.changed();
    }
}
