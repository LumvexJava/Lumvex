package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.Settings;

/**
 * Path tracer settings of a {@link Scene}. The defaults suit most scenes; fewer bounces trade indirect light for
 * speed.
 */
public final class RenderSettings {
    private final Scene scene;

    RenderSettings(Scene scene) { this.scene = scene; }

    /** Image brightness multiplier, 1 by default. */
    public void setExposure(float exposure) {
        settings().exposure = exposure;
        scene.settingsChanged();
    }

    public float exposure() { return settings().exposure; }

    /** Light bounces per path, 1 to 16; 8 by default. */
    public void setBounces(int bounces) {
        settings().bounces = Math.max(1, Math.min(16, bounces));
        scene.settingsChanged();
    }

    public int bounces() { return settings().bounces; }

    /** Frames that are accumulated while nothing moves; 100 by default. */
    public void setSampleLimit(int samples) {
        settings().sampleLimit = Math.max(1, samples);
        scene.settingsChanged();
    }

    /** Strength of the lens flares around bright lights, 0 to switch them off. */
    public void setLensFlares(float strength) {
        settings().lensFlares = strength;
        scene.settingsChanged();
    }

    public void setRenderMode(String mode) {
        settings().renderMode = mode;
        scene.settingsChanged();
    }

    public String renderMode() { return settings().renderMode; }

    private Settings settings() { return scene.document().settings; }
}
