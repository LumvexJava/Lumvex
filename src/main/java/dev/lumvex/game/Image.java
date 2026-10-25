package dev.lumvex.game;

import dev.lumvex.ui.UiDoc;

/** A picture from the project's assets, scaled to the element. */
public class Image extends UiElement {
    public Image(String name, String imageAsset) {
        super(name, UiDoc.IMAGE);
        e.image = imageAsset;
    }

    Image(UiDoc.Element e) { super(e); }

    /** Shows another image asset, e.g. {@code "ui/icons/key.png"}. */
    public void setImage(String imageAsset) {
        e.image = imageAsset;
        if (canvas != null) canvas.changed();
    }
}
