package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc;

/**
 * Render settings of the running scene, available to scripts as {@code render()}.
 *
 * <p>Bounces, exposure, overscan, flares and the sample limit take effect immediately. Changes to the sky, sun or
 * HDRI rebuild the lighting, so the image has to converge again.
 */
public final class Render extends SystemScript {
    private final ScriptRuntime rt;

    Render(ScriptRuntime rt) { this.rt = rt; }

    private SceneDoc.Settings s() { return rt.doc.settings; }

    public int bounces() { return s().bounces; }

    /** Maximum light bounces per path, clamped to 1..32. 1 is direct light only; interiors look right at 4-8. */
    public void bounces(int n) { s().bounces = Math.max(1, Math.min(32, n)); rt.settingsChanged(); }

    public float exposure() { return s().exposure; }

    /** Exposure multiplier; 1 is neutral. */
    public void exposure(float e) { s().exposure = Math.max(0.001f, e); rt.settingsChanged(); }

    public float lensFlares() { return s().lensFlares; }

    /** Lens flare strength for the sun and bright lights; 0 disables them, 1 is the default. */
    public void lensFlares(float strength) { s().lensFlares = Math.max(0, strength); rt.settingsChanged(); }

    public int overscan() { return s().overscan; }

    /**
     * Extra border rendered beyond the visible image, in percent (0-50). Pixels that rotate into view then already
     * have a history, which keeps the screen edges from flickering. 10% costs about 21% more pixels.
     */
    public void overscan(int percent) { s().overscan = Math.max(0, Math.min(50, percent)); rt.settingsChanged(); }

    public int sampleLimit() { return s().sampleLimit; }

    /** Samples per pixel after which a still image is considered done and tracing stops; 0 means never. */
    public void sampleLimit(int n) { s().sampleLimit = Math.max(0, n); rt.settingsChanged(); }

    /** The HDRI asset used as sky, or null for the procedural sky. */
    public String environment() { return s().environment; }

    /** Sets the HDRI sky, e.g. {@code "hdri/pink_sunrise_4k.hdr"}; null switches to the procedural sky. */
    public void environment(String hdri) { s().environment = hdri; rt.structureChanged(); }

    public void environmentStrength(float f) { s().environmentStrength = f; rt.structureChanged(); }

    /**
     * Rotates the HDRI so its sun sits at {@code u} around the horizon: 0.5 faces -z, 0.75 faces +x, 0.25 faces -x.
     * A negative value keeps the panorama as captured.
     */
    public void environmentSun(float u) { s().environmentSunU = u; rt.structureChanged(); }

    /** Sun direction of the procedural sky. */
    public void sun(Vec3 direction) { s().sunDirection = direction.normalized().toArray(); rt.structureChanged(); }

    public void sunIntensity(float f) { s().sunIntensity = f; rt.structureChanged(); }

    /** Sun color, linear RGB in 0..1. */
    public void sunColor(Vec3 rgb) { s().sunColor = rgb.toArray(); rt.structureChanged(); }

    /** Brightness of the procedural sky. */
    public void sky(float intensity) { s().skyIntensity = intensity; rt.structureChanged(); }

    /**
     * Selects a debug visualization: 0 final image, 1 normals, 2 albedo, 3 denoiser noise estimate, 4 history length,
     * 5 influence zones of moving objects, 6 virtual texture mip levels, 7 path length, 8 raster check (hybrid
     * rendering: red where the rasterized and the traced primary surface differ).
     */
    public void debugView(int view) { rt.debugView = Math.max(0, Math.min(8, view)); }

    public int debugView() { return rt.debugView; }

    /** The engine's statistics line: frame rate, GPU time, object and triangle counts, physics time. */
    public String stats() { return rt.statsText; }

    /**
     * Enables the starry night sky. For a convincing night, also lower {@link #sky(float)} and
     * {@link #sunIntensity(float)}; the sun then acts as the moon.
     */
    public void night(boolean on) { s().night = on; rt.structureChanged(); }
}
