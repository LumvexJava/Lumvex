package dev.lumvex.game;

import java.lang.annotation.*;

/**
 * Marks a method that runs when this object starts touching another object with the given tag.
 * Both objects need a collider. The method takes no parameters or a single {@link GameObject} (the other object).
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface OnCollision {
    /** Tag of the other object; empty matches any object. */
    String value() default "";
}
