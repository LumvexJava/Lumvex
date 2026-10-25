package dev.lumvex.game;

/**
 * Common base of the engine services available to scripts: {@link Input}, {@link Render}, {@link Time} and
 * {@link Physics}. Scripts get them through {@code input()}, {@code render()}, {@code time()} and {@code physics()}.
 */
public abstract class SystemScript {
    SystemScript() { }
}
