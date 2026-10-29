package dev.lumvex.render.scene;

import org.joml.Matrix4f;
import org.joml.Vector3f;

import dev.lumvex.texture.TextureSet;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * The scene as the renderer sees it: meshes, materials, instances, sphere lights, sky settings and texture sets.
 * Built by {@code SceneBuilder}. Revision counters tell the renderer what changed, so it only re-uploads what it must.
 */
public final class Scene {

    /** Procedural animation: writes an instance's transform for a given time in seconds. */
    @FunctionalInterface
    public interface Animator {
        void update(float time, Matrix4f transform);
    }

    /** A mesh with a material at a transform; one TLAS entry. */
    public static final class Instance {
        public final int mesh;
        public int material;
        public final Matrix4f transform;
        public final Animator animator;

        /** Last frame's transform, for motion vectors. */
        public final Matrix4f prevTransform;
        public String name = "";

        public Instance named(String n) {
            name = n;
            return this;
        }

        Instance(int mesh, int material, Matrix4f transform, Animator animator) {
            this.mesh = mesh;
            this.material = material;
            this.transform = transform;
            this.animator = animator;
            this.prevTransform = new Matrix4f(transform);
        }

        /** Moved by a script this frame; treated like an animated instance. */
        public boolean scripted;

        public boolean isDynamic() { return animator != null || scripted; }
    }

    /** A small spherical light such as a bulb or a candle flame, sampled explicitly by the path tracer. */
    /**
     * A sphere light. {@code shape} says what geometry, if any, shows it to rays: none (a point light component), a
     * glowing sphere that coincides with it, or another glowing shape (boxes, panels, bulbs, flames). The path
     * tracer weighs light samples against reflection rays that hit that geometry, which needs to know this.
     */
    public record SphereLight(Vector3f position, float radius, Vector3f radiance, Shape shape) {
        public SphereLight(Vector3f position, float radius, Vector3f radiance) { this(position, radius, radiance, Shape.OTHER); }
    }

    /** Geometry of a sphere light; the ordinal is written to the light buffer (LIGHT_* in raygen.rgen). */
    public enum Shape { NONE, SPHERE, OTHER }

    // Change tracking. The renderer picks changes up at the start of the next frame; meshes, BLAS, texture tiles and
    // the environment that are already uploaded stay where they are.
    public int structureRevision;
    public int transformRevision;
    public int lightRevision;
    public int environmentRevision;
    public int textureRevision;

    /** Lights that only moved are updated without restarting accumulation; influence zones cover them. */
    public int lightMoveRevision;
    /**
     * Marks the last structural change as soft: the renderer keeps its history and only restarts inside the
     * influence zones of {@link #changedBoxes} and of the changed instances.
     */
    public boolean softChange;
    /** World AABBs of removed or changed objects plus an emissive flag: {minx, miny, minz, maxx, maxy, maxz, emissive}. */
    public final List<float[]> changedBoxes = new ArrayList<>();

    public void lightsMoved() { lightMoveRevision++; }

    public void structureChanged() { structureRevision++; }
    public void transformsChanged() { transformRevision++; }
    public void lightsChanged() { lightRevision++; }
    public void environmentChanged() { environmentRevision++; }
    public void texturesChanged() { textureRevision++; }

    public final List<SphereLight> lights = new ArrayList<>();

    /** Starry night sky; the sun acts as the moon. */
    public boolean night;
    /** Aerial perspective, see {@code SceneDoc.Settings.hazeDensity}. */
    public float hazeDensity, hazeFalloff = 0.08f, hazeSun = 0.15f;
    public final Vector3f hazeColor = new Vector3f(0.5f, 0.55f, 0.65f);
    /** HDR panorama used as sky; null selects the procedural sky. */
    public Environment environment;
    public float exposure = 1f;

    public final List<Mesh> meshes = new ArrayList<>();
    public final List<Material> materials = new ArrayList<>();
    public final List<Instance> instances = new ArrayList<>();
    /** Texture sets for virtual texturing. The renderer owns them once it has started. */
    public final List<TextureSet> textureSets = new ArrayList<>();
    /**
     * Each virtual texture pool holds slotsPerRow^2 tiles; 0 sizes the pools from the VRAM budget
     * {@code -Dlumvex.texturepool=<MB>} (default 3072 MB), see {@code VirtualTextures.slotsForBudget}.
     */
    public int texturePoolSlotsPerRow = 0;
    /** Block-compresses the tile pools: BC7/BC5 for near mips, BC1/BC5 for far mips. */
    public boolean textureCompression = true;
    /** Similarity tolerance for reusing tiles as stand-ins; 0 disables reuse. */
    public float textureSimilarity = 1f;

    public final Vector3f sunDirection = new Vector3f(0.5f, 0.8f, 0.3f).normalize();
    public final Vector3f sunColor = new Vector3f(1.0f, 0.95f, 0.85f);
    public float sunIntensity = 3.0f;
    public float skyIntensity = 1.0f;

    public final Vector3f cameraPosition = new Vector3f(0, 2, 8);
    public float cameraYaw = 0f;
    public float cameraPitch = 0.12f;
    public int addMesh(Mesh mesh) {
        meshes.add(mesh);
        return meshes.size() - 1;
    }

    /** Meshes replaced in place since the renderer last uploaded them. */
    public final Set<Integer> replacedMeshes = new HashSet<>();

    /** Replaces a mesh in place. The renderer rebuilds its BLAS with the next structural change. */
    public void replaceMesh(int index, Mesh mesh) {
        meshes.set(index, mesh);
        replacedMeshes.add(index);
    }

    public int addTextureSet(TextureSet set) {
        textureSets.add(set);
        return textureSets.size() - 1;
    }

    public int addMaterial(Material material) {
        materials.add(material);
        return materials.size() - 1;
    }

    public Instance addInstance(int mesh, int material, Matrix4f transform) {
        return addInstance(mesh, material, transform, null);
    }

    /** Adds an instance with an optional animator. The transform is copied. */
    public Instance addInstance(int mesh, int material, Matrix4f transform, Animator animator) {
        Instance inst = new Instance(mesh, material, new Matrix4f(transform), animator);
        instances.add(inst);
        return inst;
    }

    public boolean hasAnimation() {
        return instances.stream().anyMatch(i -> i.animator != null);
    }

    /** Advances all animated instances to the given time, keeping their previous transforms. */
    public void animate(float time) {
        for (Instance i : instances) {
            if (i.animator == null) continue;
            i.prevTransform.set(i.transform);
            i.animator.update(time, i.transform);
        }
    }

    /** Sets the previous transforms to the current ones, e.g. while animation is paused. */
    public void settle() {
        for (Instance i : instances) i.prevTransform.set(i.transform);
    }
}
