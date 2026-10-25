package dev.lumvex.game;

import java.lang.annotation.*;

/**
 * Marks a method that runs when an object with the given tag enters a trigger, or leaves it if {@link #exit()} is
 * set. One of the two objects must have a collider marked as trigger; triggers report overlaps but do not block.
 * The method takes no parameters or a single {@link GameObject} (the other object).
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface OnTrigger {
    /** Tag of the other object; empty matches any object. */
    String value() default "";

    /** Fire when the object leaves the trigger instead of when it enters. */
    boolean exit() default false;
}
