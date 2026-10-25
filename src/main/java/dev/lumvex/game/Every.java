package dev.lumvex.game;

import java.lang.annotation.*;

/**
 * Calls the annotated method (no parameters) at a fixed interval of game time.
 * Timers do not fire while the game is paused.
 *
 * <pre>{@code @Every(0.5f) void blink() { ... }}</pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface Every {
    /** Interval in seconds. */
    float value();
}
