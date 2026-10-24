package dev.lumvex.game;

import dev.lumvex.ui.UiDoc;

/** Text without a background, e.g. a score or a hint. */
public class Label extends UiElement {
    public Label(String name, String text) {
        super(name, UiDoc.TEXT);
        e.text = text;
    }

    Label(UiDoc.Element e) { super(e); }
}
