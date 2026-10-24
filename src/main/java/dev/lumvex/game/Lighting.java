package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.Settings;

/**
 * Sun, sky and environment of a {@link Scene}. While the game runs, scripts change these through
 * {@link Script#render()}.
 */
public final class Lighting {
    private final Scene scene;

    Lighting(Scene scene) { this.scene = scene; }

    /**
     * Sets the sun. {@code direction} points from the scene towards the sun; {@code intensity} is the irradiance,
     * around 3 for a clear day.
     */
    public void setSun(Vec3 direction, Color color, float intensity) {
        Settings s = settings();
        s.sunDirection = direction.normalized().toArray();
        s.sunColor = color.rgb();
        s.sunIntensity = intensity;
        scene.settingsChanged();
    }

    public Vec3 sunDirection() { return Vec3.of(settings().sunDirection); }

    public Color sunColor() { return Color.of(settings().sunColor); }

    public float sunIntensity() { return settings().sunIntensity; }

    /** Brightness of the procedural sky, 1 by default. */
    public void setSkyIntensity(float intensity) {
        settings().skyIntensity = intensity;
        scene.settingsChanged();
    }

    /**
     * Lights the scene with an HDR panorama of the project, e.g. {@code "hdri/meadow.hdr"}, instead of the
     * procedural sky. Pass null to return to the sky.
     */
    public void setEnvironment(String hdrAsset, float strength) {
        Settings s = settings();
        s.environment = hdrAsset;
        s.environmentStrength = strength;
        scene.settingsChanged();
    }

    /** Night mode: a dark sky with stars and a moon instead of the sun. */
    public void setNight(boolean night) {
        settings().night = night;
        scene.settingsChanged();
    }

    public boolean isNight() { return settings().night; }

    private Settings settings() { return scene.document().settings; }
}
