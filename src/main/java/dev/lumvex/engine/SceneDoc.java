package dev.lumvex.engine;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.joml.Matrix4f;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * The serialized form of a scene ({@code scenes/*.scene.json}): settings, an optional scene script and a flat list
 * of entities with components. {@link SceneBuilder} turns it into the render scene.
 *
 * <p>Asset references are paths relative to the project's {@code assets/} folder, for example
 * {@code models/chess_set_4k.glb}, {@code textures/wooden_floor_02_8k} or {@code hdri/pink_sunrise_4k.hdr}.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public final class SceneDoc {
    public String name = "New scene";
    public Settings settings = new Settings();
    /** The scene script, or null. */
    public ScriptComponent sceneScript;
    public List<Entity> entities = new ArrayList<>();

    /**
     * A scene object with a local transform and components. The fluent setters return the entity, so scenes can be
     * built in code: {@code new Entity("Lamp").at(0, 2, 0).with(light)}.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static final class Entity {
        public String id = UUID.randomUUID().toString();
        public String name = "Object";
        /** Id of the parent entity; null for root entities. */
        public String parent;
        public float[] position = {0, 0, 0};
        /** Euler angles in degrees (pitch, yaw, roll), applied in the order yaw, pitch, roll. */
        public float[] rotation = {0, 0, 0};
        public float[] scale = {1, 1, 1};
        public List<String> tags;
        public List<Component> components = new ArrayList<>();

        public Entity() {}

        public Entity(String name) { this.name = name; }

        public Entity at(float x, float y, float z) { position = new float[]{x, y, z}; return this; }

        public Entity rotated(float pitchDeg, float yawDeg, float rollDeg) { rotation = new float[]{pitchDeg, yawDeg, rollDeg}; return this; }

        public Entity scaled(float s) { scale = new float[]{s, s, s}; return this; }

        public Entity with(Component c) { components.add(c); return this; }

        public Entity tagged(String... t) {
            if (tags == null) tags = new ArrayList<>();
            tags.addAll(List.of(t));
            return this;
        }

        /** T * Ry * Rx * Rz * S. */
        public Matrix4f localMatrix(Matrix4f dest) {
            return dest.identity().translate(position[0], position[1], position[2])
                    .rotateY((float) Math.toRadians(rotation[1])).rotateX((float) Math.toRadians(rotation[0]))
                    .rotateZ((float) Math.toRadians(rotation[2])).scale(scale[0], scale[1], scale[2]);
        }

        /** The first component of the given type, or null. */
        public <T extends Component> T get(Class<T> type) {
            for (Component c : components) if (type.isInstance(c)) return type.cast(c);
            return null;
        }
    }

    /** Base class of all components. The JSON property {@code type} selects the subclass. */
    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
    @JsonSubTypes({
            @JsonSubTypes.Type(value = ModelComponent.class, name = "model"),
            @JsonSubTypes.Type(value = PrimitiveComponent.class, name = "primitive"),
            @JsonSubTypes.Type(value = LightComponent.class, name = "light"),
            @JsonSubTypes.Type(value = CameraComponent.class, name = "camera"),
            @JsonSubTypes.Type(value = ScriptComponent.class, name = "script"),
            @JsonSubTypes.Type(value = PrefabComponent.class, name = "prefab"),
            @JsonSubTypes.Type(value = ColliderComponent.class, name = "collider"),
            @JsonSubTypes.Type(value = UiComponent.class, name = "ui"),
            @JsonSubTypes.Type(value = InstancesComponent.class, name = "instances")})
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public abstract static class Component {}

    /** A glTF model. Meshes are loaded once per model file and shared by all entities that use it. */
    public static final class ModelComponent extends Component {
        public String model;
        /** If set, only nodes whose name starts with this prefix are used, e.g. one item of a set. */
        public String node;
        /** Centers the footprint of the selected nodes on the entity and puts their lowest point at y = 0. */
        public boolean recenter;
        /** Material asset that replaces all materials of the model; null keeps the model's own. */
        public String material;
        /** Radiance of the sphere lights created for candle flame cards in the model. */
        public float flameRadiance = 1400f;

        public ModelComponent() {}

        public ModelComponent(String model) { this.model = model; }
    }

    /**
     * A built-in shape: {@code box}/{@code cube} (size = edge lengths, UVs in meters), {@code sphere},
     * {@code cylinder}, {@code plane} (size[0] = half extent) or {@code torus}. Uses either a material asset or an
     * inline {@link MaterialDef}.
     */
    public static final class PrimitiveComponent extends Component {
        public String shape = "box";
        public float[] size = {1, 1, 1};
        public String material;
        public MaterialDef materialDef;

        public PrimitiveComponent() {}

        public PrimitiveComponent(String shape, float[] size, String material, MaterialDef def) {
            this.shape = shape; this.size = size; this.material = material; this.materialDef = def;
        }
    }

    /**
     * A light. {@code sphere} is a small emitter at the entity position (bulbs, candles); {@code sun} is a
     * directional light shining along the entity's -z axis.
     */
    public static final class LightComponent extends Component {
        public String kind = "sphere";
        public float[] color = {1.0f, 0.55f, 0.22f};
        /** Radiance for sphere lights, irradiance for suns. */
        public float intensity = 1400f;
        public float radius = 0.01f;
    }

    /** The camera the game starts with. */
    public static final class CameraComponent extends Component {
        public float fov = 55f;
        public boolean player = true;
    }

    /**
     * Many copies of a model or prefab without an entity per copy. Each copy takes {@link #STRIDE} floats in
     * {@link #data}: position, rotation (pitch, yaw, roll in degrees) and scale, relative to the entity.
     *
     * <p>All copies share the source's meshes, i.e. one BLAS and many TLAS entries. With a collider on the same
     * entity every copy gets its own static hull.
     */
    public static final class InstancesComponent extends Component {
        public static final int STRIDE = 9;
        public String source;
        /** Models only: restrict to nodes whose name starts with this prefix. */
        public String node;
        /** Models only: see {@link ModelComponent#recenter}. */
        public boolean recenter = true;
        public float[] data = new float[0];

        public InstancesComponent() {}

        public InstancesComponent(String source) { this.source = source; }

        @JsonIgnore
        public int count() { return data == null ? 0 : data.length / STRIDE; }

        /** Appends a copy and returns its index. */
        public int add(float x, float y, float z, float pitch, float yaw, float roll, float sx, float sy, float sz) {
            int n = count();
            float[] d = Arrays.copyOf(data == null ? new float[0] : data, (n + 1) * STRIDE);
            System.arraycopy(new float[]{x, y, z, pitch, yaw, roll, sx, sy, sz}, 0, d, n * STRIDE, STRIDE);
            data = d;
            return n;
        }

        /** Removes copy {@code i}; later copies shift down. Out-of-range indices are ignored. */
        public void remove(int i) {
            int n = count();
            if (i < 0 || i >= n) return;
            float[] d = new float[(n - 1) * STRIDE];
            System.arraycopy(data, 0, d, 0, i * STRIDE);
            System.arraycopy(data, (i + 1) * STRIDE, d, i * STRIDE, (n - 1 - i) * STRIDE);
            data = d;
        }

        public float[] get(int i) { return Arrays.copyOfRange(data, i * STRIDE, (i + 1) * STRIDE); }

        public void set(int i, float[] v) { System.arraycopy(v, 0, data, i * STRIDE, STRIDE); }

        /** Transform of copy {@code i} relative to the entity, composed like {@link Entity#localMatrix}. */
        public Matrix4f matrix(int i, Matrix4f dest) {
            int o = i * STRIDE;
            return dest.identity().translate(data[o], data[o + 1], data[o + 2])
                    .rotateY((float) Math.toRadians(data[o + 4])).rotateX((float) Math.toRadians(data[o + 3]))
                    .rotateZ((float) Math.toRadians(data[o + 5])).scale(data[o + 6], data[o + 7], data[o + 8]);
        }

        @JsonIgnore
        public boolean isPrefab() { return source != null && source.endsWith(".prefab.json"); }
    }

    /**
     * An instance of a prefab ({@code assets/prefabs/*.prefab.json}). The prefab's entities appear under this
     * entity's transform; editing the prefab file updates every instance.
     */
    public static final class PrefabComponent extends Component {
        public String prefab;

        public PrefabComponent() {}

        public PrefabComponent(String prefab) { this.prefab = prefab; }
    }

    /**
     * A script attachment: the fully qualified class name (e.g. {@code doors.Door} for
     * {@code scripts/doors/Door.java}) and values for its {@code @Param} fields. Missing parameters keep the default
     * from the code.
     */
    public static final class ScriptComponent extends Component {
        public String script;
        public Map<String, Object> params = new LinkedHashMap<>();

        public ScriptComponent() {}

        public ScriptComponent(String script) { this.script = script; }

        public ScriptComponent param(String name, Object value) { params.put(name, value); return this; }
    }

    /**
     * Collision settings.
     *
     * <p>Mode {@code simple} uses a k-DOP hull with at most {@link #hullVertices} corners; it raises collision and
     * trigger events and blocks objects moved by scripts. Mode {@code physics} creates a Jolt body: {@code static}
     * (exact triangles), {@code dynamic} (simulated convex hull) or {@code kinematic} (moved by scripts, pushes
     * dynamic bodies). A trigger never blocks and only raises trigger events. {@link #size} and {@link #offset}
     * replace the geometry with a box.
     */
    public static final class ColliderComponent extends Component {
        public String mode = "simple";
        public String body = "static";
        public int hullVertices = 12;
        public boolean trigger;
        public float mass = 10f;
        public float friction = 0.5f;
        public float restitution = 0.2f;
        public float[] size;
        public float[] offset;

        public ColliderComponent() {}

        public static ColliderComponent simple(int hullVertices) {
            ColliderComponent c = new ColliderComponent();
            c.hullVertices = hullVertices;
            return c;
        }

        /** A Jolt collider; {@code body} is {@code static}, {@code dynamic} or {@code kinematic}. */
        public static ColliderComponent physics(String body) {
            ColliderComponent c = new ColliderComponent();
            c.mode = "physics";
            c.body = body;
            return c;
        }

        /** Uses a box of the given size and offset instead of the geometry. */
        public ColliderComponent box(float sx, float sy, float sz, float ox, float oy, float oz) {
            size = new float[]{sx, sy, sz};
            offset = new float[]{ox, oy, oz};
            return this;
        }
    }

    /** Shows a UI canvas ({@code assets/ui/*.ui.json}) in this scene. */
    public static final class UiComponent extends Component {
        public String canvas;
        /** Whether the canvas is visible when the scene starts. */
        public boolean visible = true;

        public UiComponent() {}

        public UiComponent(String canvas) { this.canvas = canvas; }
    }

    /** A material, stored as {@code assets/materials/*.material.json} or inline in a primitive. */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static final class MaterialDef {
        public float[] albedo = {0.8f, 0.8f, 0.8f};
        public float roughness = 0.5f;
        public float metallic = 0f;
        public float[] emission = {0, 0, 0};
        public float ior = 1.5f;
        public float transmission = 0f;
        /** Texture set folder; null for an untextured material. The factors above multiply the maps. */
        public String textures;
        public float uvScale = 1f;
        public float normalStrength = 1f;
        /**
         * Relief depth in meters of the texture set's height map (parallax occlusion mapping with self-shadowing);
         * 0 keeps the surface flat. Texture sets without a height map stay flat anyway.
         */
        public float parallax = 0f;
        /** Marks the visible surface of a light: it glows for the camera but casts no shadow. */
        public boolean lightProxy;

        /** Relief depth of textured materials unless set otherwise: about the depth of a mortar joint. */
        public static final float DEFAULT_PARALLAX = 0.03f;

        /** A material driven entirely by its texture maps (all factors 1). */
        public static MaterialDef textured(String textures, float uvScale) {
            MaterialDef m = new MaterialDef();
            m.albedo = new float[]{1, 1, 1};
            m.roughness = 1f;
            m.metallic = 1f;
            m.textures = textures;
            m.uvScale = uvScale;
            m.parallax = DEFAULT_PARALLAX;
            return m;
        }

        /** Smooth glass with the given tint and index of refraction. */
        public static MaterialDef glass(float r, float g, float b, float ior) {
            MaterialDef m = new MaterialDef();
            m.albedo = new float[]{r, g, b};
            m.roughness = 0f;
            m.transmission = 1f;
            m.ior = ior;
            return m;
        }
    }

    /** Sky, sun and path tracer settings of a scene. */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public static final class Settings {
        /** HDR panorama used as sky and environment light; null selects the procedural sky. */
        public String environment;
        public float environmentStrength = 1f;
        /** Rotates the panorama so its sun sits at this horizontal position (0..1); negative keeps it as captured. */
        public float environmentSunU = -1f;
        /**
         * With a panorama: true takes the sun from its brightest spot, false keeps {@link #sunDirection},
         * {@link #sunColor} and {@link #sunIntensity} (for skies whose sun is hidden or too weak).
         */
        public boolean sunFromEnvironment = true;
        /** Starry night sky; the sun then acts as the moon. */
        public boolean night;
        /**
         * Aerial perspective: extinction per meter at height 0; 0 disables the haze. The density falls off with
         * height by {@code exp(-hazeFalloff * y)}.
         */
        public float hazeDensity = 0f;
        public float hazeFalloff = 0.08f;
        /** Radiance of the haze lit by the sky, in the units of the image before exposure. */
        public float[] hazeColor = {0.5f, 0.55f, 0.65f};
        /** Sunlight scattered by the haze towards the camera, relative to the sun's radiance (forward peak). */
        public float hazeSun = 0.15f;
        public float[] sunDirection = {0.5f, 0.8f, 0.3f};
        public float[] sunColor = {1.0f, 0.95f, 0.85f};
        public float sunIntensity = 3f;
        public float skyIntensity = 1f;
        public float exposure = 1f;
        /** Lens flare strength; 0 disables flares. */
        public float lensFlares = 1f;
        public int bounces = 8;
        public int sampleLimit = 100;
        /**
         * Extra border rendered beyond the visible image, in percent. Pixels that rotate into view already have a
         * history, which keeps the edges stable. The cost grows roughly with twice this percentage.
         */
        public int overscan = 10;
        /**
         * How the scene is rendered: {@code "hybrid"} (default; rasterized visibility, path traced light),
         * {@code "path"} (everything path traced, as before 1.2) or {@code "raster"} (rasterization only, no ray
         * tracing). GPUs without hardware ray tracing always render {@code "raster"}. See docs/HYBRID_RENDERING.md.
         */
        public String renderMode = "hybrid";
        /**
         * Strength of the denoiser's disocclusion fix, which fills newly visible pixels in from converged neighbors on
         * the same surface; 0 disables it, 1 is the default.
         */
        public float disocclusionFix = 1f;
    }

    /** Shared JSON mapper for scenes, prefabs, materials and project files. Ignores unknown properties. */
    public static final ObjectMapper JSON = new ObjectMapper()
            .enable(SerializationFeature.INDENT_OUTPUT)
            .disable(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES);

    /** Reads a scene or prefab file. */
    public static SceneDoc load(Path file) throws IOException {
        return JSON.readValue(file.toFile(), SceneDoc.class);
    }

    public void save(Path file) throws IOException {
        Files.createDirectories(file.toAbsolutePath().getParent());
        JSON.writeValue(file.toFile(), this);
    }

    public SceneDoc copy() {
        return JSON.convertValue(JSON.valueToTree(this), SceneDoc.class);
    }

    /**
     * The entity with this id, or null, through an index from id to list position (physics, transforms and scripts
     * look entities up every frame). Every hit is verified against the list, so edits of the public list never return
     * a wrong entity, they only cost one rebuild of the index. If the list looks unchanged but the id is missing, an
     * entity may have replaced another one at the same size, so the list is scanned.
     */
    public Entity find(String id) {
        if (id == null || id.isEmpty()) return null;
        Integer at = index.get(id);
        if (at != null && at < entities.size()) {
            Entity e = entities.get(at);
            if (e.id.equals(id)) return e;
        }
        if (entities != indexedList || entities.size() != indexedSize || at != null) {
            reindex();
            at = index.get(id);
            return at == null ? null : entities.get(at);
        }
        for (Entity e : entities) if (e.id.equals(id)) return e;
        return null;
    }

    @JsonIgnore private final Map<String, Integer> index = new HashMap<>();
    @JsonIgnore private List<Entity> indexedList;
    @JsonIgnore private int indexedSize = -1;

    /** Rebuilds the index from id to list position. */
    private void reindex() {
        index.clear();
        for (int i = 0; i < entities.size(); i++) index.put(entities.get(i).id, i);
        indexedList = entities;
        indexedSize = entities.size();
    }

    /**
     * The entity with its descendants, through an index from parent to children. The index is rebuilt when the list changes or after {@link #childrenChanged}, and every child
     * is checked against its parent field, so a stale index never moves an entity that left the hierarchy.
     */
    public Set<String> withDescendants(Set<String> ids) {
        if (entities != childrenList || entities.size() != childrenSize) {
            children.clear();
            for (Entity e : entities) if (e.parent != null) children.computeIfAbsent(e.parent, k -> new ArrayList<>()).add(e);
            childrenList = entities;
            childrenSize = entities.size();
        }
        Set<String> all = new HashSet<>(ids);
        java.util.ArrayDeque<String> open = new java.util.ArrayDeque<>(ids);
        while (!open.isEmpty()) {
            String id = open.poll();
            for (Entity c : children.getOrDefault(id, List.of())) {
                if (id.equals(c.parent) && all.add(c.id)) open.add(c.id);
            }
        }
        return all;
    }

    /** Call after entities were reparented without changing the entity list, see {@link #withDescendants}. */
    public void childrenChanged() { childrenSize = -1; }

    @JsonIgnore private final Map<String, List<Entity>> children = new HashMap<>();
    @JsonIgnore private List<Entity> childrenList;
    @JsonIgnore private int childrenSize = -1;

    /** World transform of an entity. Cycles in the parent chain are cut off. */
    public Matrix4f worldMatrix(Entity e, Matrix4f dest) {
        e.localMatrix(dest);
        Matrix4f tmp = new Matrix4f();
        Set<String> seen = new HashSet<>();
        for (Entity p = find(e.parent == null ? "" : e.parent); p != null && seen.add(p.id); p = p.parent == null ? null : find(p.parent)) {
            p.localMatrix(tmp).mul(dest, dest);
        }
        return dest;
    }
}
