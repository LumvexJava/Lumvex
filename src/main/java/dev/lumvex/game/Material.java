package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc.MaterialDef;

import java.util.Arrays;

/**
 * The surface of a {@link MeshRenderer}. Materials are immutable, so one instance can be shared by many objects:
 *
 * <pre>{@code
 * static final Material GOLD = Material.metal(Color.hex("#FFC857")).withRoughness(0.2f);
 * }</pre>
 *
 * <p>A material either describes its surface directly or refers to a material asset of the project, see
 * {@link #asset(String)}.
 */
public final class Material {
    private final MaterialDef def;
    private final String asset;

    private Material(MaterialDef def, String asset) {
        this.def = def;
        this.asset = asset;
    }

    /** A dielectric surface such as plaster, wood or plastic. Roughness defaults to 0.5. */
    public static Material color(Color albedo) {
        MaterialDef d = new MaterialDef();
        d.albedo = albedo.rgb();
        return new Material(d, null);
    }

    /** A metal. Roughness defaults to 0.3. */
    public static Material metal(Color albedo) {
        MaterialDef d = new MaterialDef();
        d.albedo = albedo.rgb();
        d.metallic = 1;
        d.roughness = 0.3f;
        return new Material(d, null);
    }

    /** Smooth clear glass with a tint and an index of refraction (1.5 for window glass). */
    public static Material glass(Color tint, float ior) {
        MaterialDef d = MaterialDef.glass(tint.r(), tint.g(), tint.b(), ior);
        return new Material(d, null);
    }

    /**
     * A material driven by a texture set, i.e. a folder below {@code assets/} with albedo, normal and roughness
     * maps. {@code uvScale} repeats the textures; 1 maps them once onto each unit of the mesh.
     */
    public static Material textured(String folder, float uvScale) {
        return new Material(MaterialDef.textured(folder, uvScale), null);
    }

    /** A material asset of the project, e.g. {@code "materials/brass.material.json"}. */
    public static Material asset(String id) { return new Material(null, id); }

    /** The same material in another color. */
    public Material withAlbedo(Color albedo) {
        MaterialDef d = editableCopy();
        d.albedo = albedo.rgb();
        return new Material(d, null);
    }

    /** The same material with another roughness: 0 is mirror-like, 1 is fully diffuse. */
    public Material withRoughness(float roughness) {
        MaterialDef d = editableCopy();
        d.roughness = roughness;
        return new Material(d, null);
    }

    /** The same material with another metallic factor: 0 for dielectrics, 1 for metals. */
    public Material withMetallic(float metallic) {
        MaterialDef d = editableCopy();
        d.metallic = metallic;
        return new Material(d, null);
    }

    /** The same material, glowing in the given color. {@code strength} is the radiance multiplier. */
    public Material withEmission(Color color, float strength) {
        MaterialDef d = editableCopy();
        d.emission = new float[]{color.r() * strength, color.g() * strength, color.b() * strength};
        return new Material(d, null);
    }

    /** The albedo, or null for a material asset. */
    public Color albedo() { return def == null ? null : Color.of(def.albedo); }

    public float roughness() { return def == null ? Float.NaN : def.roughness; }

    public float metallic() { return def == null ? Float.NaN : def.metallic; }

    /** The asset id, or null if the material is described inline. */
    public String assetId() { return asset; }

    /** A fresh document copy for storing in a scene. */
    MaterialDef toDef() { return def == null ? null : copy(def); }

    static Material of(MaterialDef def, String asset) {
        return asset != null ? new Material(null, asset) : new Material(def == null ? new MaterialDef() : copy(def), null);
    }

    private MaterialDef editableCopy() {
        if (def == null) throw new IllegalStateException("material asset " + asset + " cannot be changed in code");
        return copy(def);
    }

    private static MaterialDef copy(MaterialDef s) {
        MaterialDef d = new MaterialDef();
        d.albedo = s.albedo.clone();
        d.roughness = s.roughness;
        d.metallic = s.metallic;
        d.emission = s.emission.clone();
        d.ior = s.ior;
        d.transmission = s.transmission;
        d.textures = s.textures;
        d.uvScale = s.uvScale;
        d.normalStrength = s.normalStrength;
        d.lightProxy = s.lightProxy;
        return d;
    }

    @Override public String toString() {
        return asset != null ? "Material(" + asset + ")" : "Material(albedo=" + Arrays.toString(def.albedo) + ", roughness=" + def.roughness
                + ", metallic=" + def.metallic + ")";
    }
}
