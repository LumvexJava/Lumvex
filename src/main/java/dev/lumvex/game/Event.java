package dev.lumvex.game;

import java.lang.annotation.*;

/**
 * Marks a method that other scripts may call by name, e.g. {@code find("Door").call("open", 45f)}.
 *
 * <p>Arguments are checked against the method signature on every call and mismatches are reported with the
 * caller's file and line. For compile-time checking, call the method directly:
 * {@code find("Door").script(Door.class).open(45)}.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Event {
    /** Event name; defaults to the method name. */
    String value() default "";
}
