package dev.lumvex.game;

import java.lang.annotation.*;

/**
 * Marks a method that runs when a key is pressed, or released if {@link #released()} is set.
 *
 * <p>Key names: {@code A}-{@code Z}, {@code 0}-{@code 9}, {@code F1}-{@code F12}, {@code SPACE}, {@code ENTER},
 * {@code ESCAPE}, {@code TAB}, {@code BACKSPACE}, {@code SHIFT}, {@code CTRL}, {@code ALT}, {@code UP},
 * {@code DOWN}, {@code LEFT}, {@code RIGHT}, and {@code ANY} for every key. The method takes no parameters or a
 * {@code String} receiving the key name.
 *
 * <pre>{@code @OnKey("E") void use() { ... }}</pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface OnKey {
    /** Key name, e.g. {@code "E"}. */
    String value();

    /** Fire on release instead of press. */
    boolean released() default false;
}
