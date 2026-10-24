package dev.lumvex.game;

import dev.lumvex.ui.UiDoc;

import java.util.ArrayList;
import java.util.List;

/** A clickable element. It lightens under the mouse and darkens while pressed. */
public class Button extends UiElement {
    private final List<Runnable> clickListeners = new ArrayList<>();

    public Button(String name, String text) {
        super(name, UiDoc.BUTTON);
        e.text = text;
        e.background = new float[]{0.15f, 0.15f, 0.18f, 0.9f};
        e.cornerRadius = 8;
    }

    Button(UiDoc.Element e) { super(e); }

    /** Runs {@code listener} on the game thread whenever the button is clicked. */
    public void addClickListener(Runnable listener) { clickListeners.add(listener); }

    public void removeClickListener(Runnable listener) { clickListeners.remove(listener); }

    /** Colors under the mouse and while pressed. Null keeps the automatic lighter and darker shades. */
    public void setStateColors(Color hover, Color pressed) {
        e.hoverColor = hover == null ? null : hover.rgba();
        e.pressedColor = pressed == null ? null : pressed.rgba();
        if (canvas != null) canvas.changed();
    }

    @Override void clicked() {
        for (Runnable r : new ArrayList<>(clickListeners)) r.run();
    }
}
