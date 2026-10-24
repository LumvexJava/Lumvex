package dev.lumvex.game;

/**
 * The point of the parent that a {@link UiElement} is positioned from. The element's matching point sits there,
 * offset by its position: a {@code TOP_LEFT} element at (40, 30) has its top-left corner 40 px right of and 30 px
 * below the parent's top-left corner.
 */
public enum Anchor {
    TOP_LEFT("top-left"), TOP("top"), TOP_RIGHT("top-right"),
    LEFT("left"), CENTER("center"), RIGHT("right"),
    BOTTOM_LEFT("bottom-left"), BOTTOM("bottom"), BOTTOM_RIGHT("bottom-right"),
    /** Fills the parent; the position becomes the margin on all sides. */
    STRETCH("stretch");

    final String id;

    Anchor(String id) { this.id = id; }

    static Anchor of(String id) {
        for (Anchor a : values()) if (a.id.equals(id)) return a;
        return CENTER;
    }
}
