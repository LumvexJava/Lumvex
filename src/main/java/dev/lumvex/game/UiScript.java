package dev.lumvex.game;

/**
 * Base class for scripts attached to a UI canvas or to a single UI element.
 *
 * <p>A script on the canvas receives the clicks of all its elements. Clicks bubble from the clicked element up
 * through its parents to the canvas, and every {@link OnClick} handler on the way is called.
 */
public abstract class UiScript extends Script {
    Canvas canvas;
    UiElement element;

    /** The canvas this script belongs to. */
    protected final Canvas canvas() { return canvas; }

    /** The element this script is attached to, or null for a canvas script. */
    protected final UiElement element() { return element; }

    @Override
    String describe() {
        String where = canvas != null ? canvas.name() : "?";
        return getClass().getSimpleName() + " @ UI " + where + (element != null ? "/" + element.name() : "");
    }
}
