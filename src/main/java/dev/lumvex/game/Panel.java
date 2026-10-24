package dev.lumvex.game;

import dev.lumvex.ui.UiDoc;

/** A rectangle with a background that groups other elements, such as a dialog or a menu. */
public class Panel extends UiElement {
    public Panel(String name) { super(name, UiDoc.PANEL); }

    Panel(UiDoc.Element e) { super(e); }
}
