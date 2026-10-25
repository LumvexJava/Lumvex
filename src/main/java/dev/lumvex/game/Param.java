package dev.lumvex.game;

import java.lang.annotation.*;

/**
 * Exposes a field as a per-object parameter. Values come from the scene file ({@code ScriptComponent.params});
 * fields without a value there keep their initializer.
 *
 * <p>Supported types: {@code float}, {@code double}, {@code int}, {@code long}, {@code boolean}, {@code String},
 * {@link Vec3}, {@link GameObject} (stored by name) and enums. The field initializer is the default value.
 *
 * <pre>{@code @Param("Opening angle in degrees") float angle = 90;}</pre>
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.FIELD)
public @interface Param {
    /** Description of the parameter. */
    String value() default "";
}
