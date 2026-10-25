package dev.lumvex.game;

import java.lang.annotation.*;

/**
 * Marks a method that runs when the object is clicked.
 *
 * <p>In an object script the method fires when the player left-clicks the object or one of its children; while the
 * mouse is captured, the center of the screen is used. In a UI script it fires when the element is clicked. The
 * method takes no parameters, or a single {@link UiElement} (UI scripts only) receiving the clicked element.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface OnClick {
}
