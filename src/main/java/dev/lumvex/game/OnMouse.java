package dev.lumvex.game;

import java.lang.annotation.*;

/**
 * Marks a method that runs when a mouse button is pressed (or released) anywhere in the game view.
 * The method takes no parameters or {@code (float x, float y)}, the cursor position normalized to 0..1.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface OnMouse {
    /** {@code LEFT}, {@code RIGHT} or {@code MIDDLE}. */
    String value() default "LEFT";

    /** Fire on release instead of press. */
    boolean released() default false;
}
