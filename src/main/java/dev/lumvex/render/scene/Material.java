package dev.lumvex.render.scene;

import org.joml.Vector3f;

import java.nio.ByteBuffer;

/**
 * A metallic-roughness material with dielectric transmission and emission. The GPU layout written by
 * {@link #write} must match {@code struct Material} in {@code common.glsl}.
 *
 * @param checkerScale   cell size of a world-space checker pattern applied to the albedo; 0 disables it
 * @param textureSet     virtual texture set, or -1. Texture values multiply albedo, roughness and metallic.
 * @param uvScale        texture repeats per UV unit (per meter on primitives)
 * @param normalStrength normal map strength; 0 disables the normal map
 * @param flags          combination of {@link #LIGHT_PROXY} and {@link #ALPHA_TEST}
 * @param parallax       relief depth of the texture set's height map in meters (parallax occlusion mapping); 0 off
 */
public record Material(Vector3f albedo, float roughness, float metallic,
                       Vector3f emission, float ior, float transmission, float checkerScale,
                       int textureSet, float uvScale, float normalStrength, int flags, float parallax) {

    public static final int GPU_SIZE = 64;
    /** {@code -Dlumvex.noparallax=true} turns parallax occlusion mapping off everywhere, for comparisons. */
    private static final boolean NO_PARALLAX = Boolean.getBoolean("lumvex.noparallax");

    /**
     * Keeps the factors in their physical range. An albedo above 1 would reflect more light than arrives: the surface
     * turns white and feeds bright noise into everything around it.
     */
    public Material {
        albedo = new Vector3f(albedo).max(new Vector3f(0)).min(new Vector3f(1));
        roughness = Math.clamp(roughness, 0f, 1f);
        metallic = Math.clamp(metallic, 0f, 1f);
        transmission = Math.clamp(transmission, 0f, 1f);
        parallax = Math.max(0f, parallax);
    }

    public Material(Vector3f albedo, float roughness, float metallic, Vector3f emission, float ior, float transmission,
                    float checkerScale, int textureSet, float uvScale, float normalStrength, int flags) {
        this(albedo, roughness, metallic, emission, ior, transmission, checkerScale, textureSet, uvScale, normalStrength, flags, 0f);
    }
    /**
     * Marks the visible surface of a {@link Scene.SphereLight}, such as a candle flame. It is seen by the camera and
     * in mirrors, but casts no shadow and is not counted again on diffuse bounces, since the sphere light already
     * delivers its light.
     */
    public static final int LIGHT_PROXY = 1;
    /** Cutout surface such as leaves; the texture's alpha mask decides per ray whether there is a hit. */
    public static final int ALPHA_TEST = 2;

    public Material(Vector3f albedo, float roughness, float metallic, Vector3f emission, float ior,
                    float transmission, float checkerScale) {
        this(albedo, roughness, metallic, emission, ior, transmission, checkerScale, -1, 1f, 1f, 0);
    }

    public Material(Vector3f albedo, float roughness, float metallic, Vector3f emission, float ior,
                    float transmission, float checkerScale, int textureSet, float uvScale, float normalStrength) {
        this(albedo, roughness, metallic, emission, ior, transmission, checkerScale, textureSet, uvScale, normalStrength, 0);
    }

    /** Fully rough dielectric. */
    public static Material diffuse(float r, float g, float b) {
        return new Material(new Vector3f(r, g, b), 1.0f, 0.0f, new Vector3f(), 1.5f, 0.0f, 0f);
    }

    public static Material plastic(float r, float g, float b, float roughness) {
        return new Material(new Vector3f(r, g, b), roughness, 0.0f, new Vector3f(), 1.5f, 0.0f, 0f);
    }

    public static Material metal(float r, float g, float b, float roughness) {
        return new Material(new Vector3f(r, g, b), roughness, 1.0f, new Vector3f(), 1.5f, 0.0f, 0f);
    }

    /** Smooth glass with a tint and an index of refraction. */
    public static Material glass(float r, float g, float b, float ior) {
        return new Material(new Vector3f(r, g, b), 0.0f, 0.0f, new Vector3f(), ior, 1.0f, 0f);
    }

    /** Black surface emitting {@code color * strength}. */
    public static Material emissive(float r, float g, float b, float strength) {
        return new Material(new Vector3f(0), 1.0f, 0.0f, new Vector3f(r, g, b).mul(strength), 1.5f, 0.0f, 0f);
    }

    /** Material driven entirely by its texture set (all factors 1). */
    public static Material textured(int textureSet, float uvScale) {
        return new Material(new Vector3f(1), 1f, 1f, new Vector3f(), 1.5f, 0f, 0f, textureSet, uvScale, 1f, 0);
    }

    public Material withChecker(float cellSize) {
        return new Material(albedo, roughness, metallic, emission, ior, transmission, cellSize, textureSet, uvScale, normalStrength, flags, parallax);
    }

    public Material withTexture(int set, float scale, float normalMapStrength) {
        return new Material(albedo, roughness, metallic, emission, ior, transmission, checkerScale, set, scale, normalMapStrength, flags, parallax);
    }

    public Material withEmission(Vector3f e) {
        return new Material(albedo, roughness, metallic, e, ior, transmission, checkerScale, textureSet, uvScale, normalStrength, flags, parallax);
    }

    public Material withFlags(int f) {
        return new Material(albedo, roughness, metallic, emission, ior, transmission, checkerScale, textureSet, uvScale, normalStrength, f, parallax);
    }

    /** The same material with a relief depth in meters for its height map, see {@link #parallax}. */
    public Material withParallax(float depth) {
        return new Material(albedo, roughness, metallic, emission, ior, transmission, checkerScale, textureSet, uvScale, normalStrength, flags, depth);
    }

    public boolean isLightProxy() { return (flags & LIGHT_PROXY) != 0; }

    public boolean isAlphaTested() { return (flags & ALPHA_TEST) != 0; }

    /** Writes the 64-byte GPU representation (scalar layout) at {@code offset}. */
    public void write(ByteBuffer buf, int offset) {
        buf.putFloat(offset, albedo.x).putFloat(offset + 4, albedo.y).putFloat(offset + 8, albedo.z)
                .putFloat(offset + 12, roughness)
                .putFloat(offset + 16, emission.x).putFloat(offset + 20, emission.y).putFloat(offset + 24, emission.z)
                .putFloat(offset + 28, metallic)
                .putFloat(offset + 32, ior)
                .putFloat(offset + 36, transmission)
                .putFloat(offset + 40, checkerScale)
                .putInt(offset + 44, textureSet)
                .putFloat(offset + 48, uvScale)
                .putFloat(offset + 52, normalStrength)
                .putInt(offset + 56, flags)
                .putFloat(offset + 60, NO_PARALLAX ? 0f : parallax);
    }
}
