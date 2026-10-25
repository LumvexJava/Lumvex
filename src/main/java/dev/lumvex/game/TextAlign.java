package dev.lumvex.game;

/** Horizontal alignment of the text in a {@link UiElement}. */
public enum TextAlign {
    LEFT, CENTER, RIGHT;

    String id() { return name().toLowerCase(); }

    static TextAlign of(String id) { return id == null ? CENTER : valueOf(id.toUpperCase()); }
}
