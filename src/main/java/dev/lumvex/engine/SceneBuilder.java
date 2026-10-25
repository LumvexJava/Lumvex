package dev.lumvex.engine;

import dev.lumvex.engine.SceneDoc.*;
import dev.lumvex.render.scene.*;
import dev.lumvex.texture.TextureSet;
import org.joml.Matrix4f;
import org.joml.Vector3f;

import java.io.IOException;
import java.util.*;
import java.util.function.Supplier;

/**
 * Builds the renderer's {@link Scene} from a {@link SceneDoc} and keeps the two in sync. It also maps render
 * instances back to entities (picking) and provides entity geometry for collision and placement.
 *
 * <p>Expensive resources are registered once per session and reused across rebuilds: one mesh (and so one BLAS) per
 * model part or primitive shape, plus texture sets. A rebuild only regenerates the materials, instances and lights.
 */
public final class SceneBuilder {
    private static final Vector3f FLAME_COLOR = new Vector3f(1.0f, 0.55f, 0.22f);
    // warm incandescent
    private static final Vector3f BULB_COLOR = new Vector3f(1.0f, 0.72f, 0.42f);
    private static final int MAX_PREFAB_DEPTH = 8;

    // Switches for A/B measurements.
    static final boolean EMISSIVE_LIGHTS = !Boolean.getBoolean("lumvex.noEmissiveLights");
    static final boolean HARD_REBUILD = Boolean.getBoolean("lumvex.hardRebuild");

    private final AssetCache assets;
    private final Scene scene;

    // Registered once per session; these survive rebuilds.
    private final Map<String, Integer> primitiveMeshes = new HashMap<>();
    // "model#part" -> mesh
    private final Map<String, Integer> partMeshes = new HashMap<>();
    // texture asset -> set
    private final Map<String, Integer> textureSets = new HashMap<>();
    // model -> first texture set
    private final Map<String, Integer> modelTextureBase = new HashMap<>();
    private int flameMesh = -1;
    private String loadedEnvironment;
    private float loadedEnvironmentSunU = Float.NaN;

    /**
     * A render instance with its transform relative to the owning entity. Instances inside a copy of an instances
     * component also keep {@code pre} (the transform before the copy) and {@code post} (the part within the copy), so
     * a moved copy is updated as {@code pre * copy * post} without a rebuild.
     */
    private static final class InstanceRef {
        final int index;
        Matrix4f local;
        final Matrix4f pre, post;
        final int sub;

        InstanceRef(int index, Matrix4f local) { this(index, local, null, null, -1); }

        InstanceRef(int index, Matrix4f local, Matrix4f pre, Matrix4f post, int sub) {
            this.index = index;
            this.local = local;
            this.pre = pre;
            this.post = post;
            this.sub = sub;
        }

        int index() { return index; }

        Matrix4f local() { return local; }
    }

    /** A sphere light with its position and radius relative to the owning entity. */
    private record LightRef(int index, Vector3f localPos, float localRadius) {}

    // Rebuilt on every build.
    private final Map<String, List<InstanceRef>> entityInstances = new LinkedHashMap<>();
    private final Map<String, List<LightRef>> entityLights = new LinkedHashMap<>();
    private final List<String> instanceOwner = new ArrayList<>();
    /** Render instances by entity and copy index; physics needs fast access for thousands of copies. */
    private final Map<String, Map<Integer, List<InstanceRef>>> subRefs = new HashMap<>();
    private final Map<String, Integer> materialCache = new HashMap<>();
    private final Map<String, Matrix4f> recenterCache = new HashMap<>();
    private final Map<Mesh, List<float[]>> clusterCache = new IdentityHashMap<>();
    private boolean texturesAdded;
    private int lastChanged;

    // Emission state. Everything emitted belongs to the scene entity being processed; inside prefabs and instance
    // copies, `relative` is the transform from that entity to the current sub-object.
    private final Matrix4f ownerWorld = new Matrix4f(), relative = new Matrix4f();
    private int currentSub = -1;
    private final Matrix4f currentPre = new Matrix4f(), currentCopy = new Matrix4f();

    public SceneBuilder(AssetCache assets, Scene scene) {
        this.assets = assets;
        this.scene = scene;
    }

    public Scene scene() { return scene; }

    /** The entity that owns a render instance, or null. */
    public String entityOfInstance(int i) { return i >= 0 && i < instanceOwner.size() ? instanceOwner.get(i) : null; }

    /** All render instances of an entity. */
    public List<Integer> instancesOf(String entityId) {
        List<InstanceRef> l = entityInstances.get(entityId);
        List<Integer> out = new ArrayList<>();
        if (l != null) for (InstanceRef r : l) out.add(r.index());
        return out;
    }

    /** Triangles in an entity's local frame: packed xyz positions and three indices per triangle. */
    public record LocalMesh(float[] positions, int[] indices) {
        public boolean isEmpty() { return indices.length == 0; }
    }

    /**
     * The entity's geometry in its local frame, prefab parts included. Instanced copies are left out unless the
     * entity has no geometry of its own.
     */
    public LocalMesh localMesh(String entityId) {
        LocalMesh own = localMesh(entityId, -1);
        return own.isEmpty() ? localMesh(entityId, Integer.MIN_VALUE) : own;
    }

    /**
     * Merged local geometry of the render instances of one copy: {@code sub = -1} selects the entity's own instances,
     * {@code Integer.MIN_VALUE} selects all of them.
     */
    public LocalMesh localMesh(String entityId, int sub) {
        List<InstanceRef> all = entityInstances.get(entityId);
        if (all == null) return new LocalMesh(new float[0], new int[0]);
        List<InstanceRef> refs = all;
        if (sub != Integer.MIN_VALUE) {
            Map<Integer, List<InstanceRef>> bySub = subRefs.get(entityId);
            refs = bySub == null ? List.of() : bySub.getOrDefault(sub, List.of());
        }
        int nv = 0, ni = 0;
        for (InstanceRef r : refs) {
            Mesh m = scene.meshes.get(scene.instances.get(r.index()).mesh);
            nv += m.vertexCount();
            ni += m.indices.length;
        }
        float[] pos = new float[nv * 3];
        int[] idx = new int[ni];
        int vo = 0, io = 0;
        Vector3f p = new Vector3f();
        for (InstanceRef r : refs) {
            Mesh m = scene.meshes.get(scene.instances.get(r.index()).mesh);
            int st = Mesh.FLOATS_PER_VERTEX;
            for (int k = 0; k < m.vertexCount(); k++) {
                r.local().transformPosition(p.set(m.vertices[k * st], m.vertices[k * st + 1], m.vertices[k * st + 2]));
                pos[(vo + k) * 3] = p.x;
                pos[(vo + k) * 3 + 1] = p.y;
                pos[(vo + k) * 3 + 2] = p.z;
            }
            for (int k = 0; k < m.indices.length; k++) idx[io + k] = m.indices[k] + vo;
            vo += m.vertexCount();
            io += m.indices.length;
        }
        return new LocalMesh(pos, idx);
    }

    public void build(SceneDoc doc) throws IOException { build(doc, false); }

    /** Number of instances that were added, changed or removed by the last soft build. */
    public int lastChanged() { return lastChanged; }

    /**
     * Rebuilds the render scene. A soft build (used while playing) diffs against the previous build: the renderer
     * keeps the history of unchanged objects and only re-converges around the changed ones. If too much changed, for
     * example after a scene switch, it falls back to a full restart.
     */
    public void build(SceneDoc doc, boolean soft) throws IOException {
        Map<String, List<float[]>> before = soft && !HARD_REBUILD ? snapshot() : null;
        buildFull(doc, soft);
        lastChanged = 0;
        scene.softChange = false;
        scene.changedBoxes.clear();
        if (before == null) return;
        int changed = 0, removed = 0;
        List<Integer> changedInstances = new ArrayList<>();
        for (Map.Entry<String, List<InstanceRef>> en : entityInstances.entrySet()) {
            List<float[]> old = before.remove(en.getKey());
            List<InstanceRef> now = en.getValue();
            for (int k = 0; k < now.size(); k++) {
                Scene.Instance inst = scene.instances.get(now.get(k).index());
                float[] o = old != null && k < old.size() ? old.get(k) : null;
                if (o != null && o[0] == inst.mesh && o[1] == inst.material && same(o, inst.transform)) continue;
                changedInstances.add(now.get(k).index());
                if (o != null) scene.changedBoxes.add(box(o));
            }
        }
        for (List<float[]> gone : before.values()) for (float[] o : gone) { scene.changedBoxes.add(box(o)); removed++; }
        changed = changedInstances.size();
        if (changed + scene.changedBoxes.size() > 40) {
            scene.changedBoxes.clear();
            return;
        }
        for (int i : changedInstances) scene.instances.get(i).scripted = true;
        lastChanged = changed + removed;
        scene.softChange = true;
    }

    // Per entity and instance: {mesh, material, 16 matrix values, world AABB (6), emissive 0/1}.

    private Map<String, List<float[]>> snapshot() {
        Map<String, List<float[]>> m = new HashMap<>();
        for (Map.Entry<String, List<InstanceRef>> en : entityInstances.entrySet()) {
            List<float[]> l = new ArrayList<>();
            for (InstanceRef r : en.getValue()) {
                if (r.index() >= scene.instances.size()) continue;
                Scene.Instance inst = scene.instances.get(r.index());
                float[] v = new float[2 + 16 + 7];
                v[0] = inst.mesh;
                v[1] = inst.material;
                inst.transform.get(v, 2);
                float[] b = scene.meshes.get(inst.mesh).bounds();
                Vector3f p = new Vector3f();
                float[] bb = {Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE};
                for (int c = 0; c < 8; c++) {
                    inst.transform.transformPosition(p.set((c & 1) == 0 ? b[0] : b[3], (c & 2) == 0 ? b[1] : b[4], (c & 4) == 0 ? b[2] : b[5]));
                    bb[0] = Math.min(bb[0], p.x); bb[1] = Math.min(bb[1], p.y); bb[2] = Math.min(bb[2], p.z);
                    bb[3] = Math.max(bb[3], p.x); bb[4] = Math.max(bb[4], p.y); bb[5] = Math.max(bb[5], p.z);
                }
                System.arraycopy(bb, 0, v, 18, 6);
                Material mat = scene.materials.get(inst.material);
                v[24] = mat.emission().x + mat.emission().y + mat.emission().z > 0 ? 1 : 0;
                l.add(v);
            }
            m.put(en.getKey(), l);
        }
        return m;
    }

    private static boolean same(float[] o, Matrix4f m) {
        float[] a = new float[16];
        m.get(a);
        for (int i = 0; i < 16; i++) if (Math.abs(a[i] - o[2 + i]) > 1e-5f) return false;
        return true;
    }

    private static float[] box(float[] o) { return new float[]{o[18], o[19], o[20], o[21], o[22], o[23], o[24]}; }

    /** Regenerates materials, instances and lights. Meshes and texture sets stay registered. */
    private void buildFull(SceneDoc doc, boolean quiet) throws IOException {
        long t0 = System.nanoTime();
        prefetch(doc);
        scene.materials.clear();
        scene.instances.clear();
        scene.lights.clear();
        entityInstances.clear();
        entityLights.clear();
        instanceOwner.clear();
        currentSub = -1;
        subRefs.clear();
        // the model file may have changed since the last build
        recenterCache.clear();
        clusterCache.clear();
        texturesAdded = false;
        materialCache.clear();

        // Start from the settings' sun; a sun light entity may override it.
        Settings st = doc.settings;
        scene.sunDirection.set(st.sunDirection[0], st.sunDirection[1], st.sunDirection[2]).normalize();
        scene.sunColor.set(st.sunColor[0], st.sunColor[1], st.sunColor[2]);
        scene.sunIntensity = st.sunIntensity;

        Matrix4f world = new Matrix4f();
        for (Entity e : doc.entities) {
            doc.worldMatrix(e, world);
            ownerWorld.set(world);
            relative.identity();
            emit(e, e, world, 0);
        }
        applySettings(doc.settings);
        scene.structureChanged();
        scene.lightsChanged();
        if (texturesAdded) scene.texturesChanged();
        if (!quiet) System.out.printf("[engine] scene \"%s\": %d entities -> %d instances, %d meshes, %d materials, %d lights (%.1f s)%n",
                doc.name, doc.entities.size(), scene.instances.size(), scene.meshes.size(), scene.materials.size(),
                scene.lights.size(), (System.nanoTime() - t0) / 1e9);
    }

    /** Kicks off parallel loading of everything the scene references before the sequential build. */
    private void prefetch(SceneDoc doc) {
        for (Entity e : doc.entities) {
            for (Component c : e.components) {
                if (c instanceof ModelComponent m && m.model != null) assets.prefetchModel(m.model);
                if (c instanceof ModelComponent m && m.material != null) assets.prefetchMaterial(m.material);
                if (c instanceof PrimitiveComponent p && p.material != null) assets.prefetchMaterial(p.material);
                if (c instanceof PrimitiveComponent p && p.materialDef != null && p.materialDef.textures != null) {
                    assets.prefetchTextures(p.materialDef.textures);
                }
                if (c instanceof PrefabComponent pc && pc.prefab != null) {
                    try { prefetch(assets.prefab(pc.prefab)); } catch (RuntimeException ex) { System.err.println("[engine] " + ex.getMessage()); }
                }
                if (c instanceof InstancesComponent ic && ic.source != null) {
                    try {
                        if (ic.isPrefab()) prefetch(assets.prefab(ic.source));
                        else assets.prefetchModel(ic.source);
                    } catch (RuntimeException ex) { System.err.println("[engine] " + ex.getMessage()); }
                }
            }
        }
        // Material files are tiny; reading them now lets their texture sets start loading too.
        for (Entity e : doc.entities) {
            for (Component c : e.components) {
                String m = c instanceof ModelComponent mc ? mc.material : c instanceof PrimitiveComponent pc ? pc.material : null;
                if (m != null && assets.material(m).textures != null) assets.prefetchTextures(assets.material(m).textures);
            }
        }
    }

    /** Called at the start of a game frame: instances moved in the previous frame count as static again. */
    public void settleScripted() {
        for (Scene.Instance i : scene.instances) {
            if (!i.scripted) continue;
            i.scripted = false;
            i.prevTransform.set(i.transform);
        }
    }

    /**
     * Applies script-driven moves to the entities and their descendants. Their instances are flagged as dynamic for
     * this frame, so the renderer re-converges only around them. Returns whether anything moved.
     */
    public boolean moveScripted(SceneDoc doc, Set<String> movedIds) {
        if (movedIds.isEmpty()) return false;
        Set<String> all = new HashSet<>(movedIds);
        boolean grew = true;
        while (grew) {
            grew = false;
            for (Entity e : doc.entities) if (e.parent != null && all.contains(e.parent) && all.add(e.id)) grew = true;
        }
        Matrix4f world = new Matrix4f();
        boolean any = false, lights = false;
        for (Entity e : doc.entities) {
            if (!all.contains(e.id)) continue;
            doc.worldMatrix(e, world);
            List<InstanceRef> refs = entityInstances.get(e.id);
            if (refs != null) {
                for (InstanceRef r : refs) {
                    Scene.Instance inst = scene.instances.get(r.index());
                    inst.prevTransform.set(inst.transform);
                    inst.transform.set(world).mul(r.local());
                    inst.scripted = true;
                    any = true;
                }
            }
            List<LightRef> lrefs = entityLights.get(e.id);
            if (lrefs != null) {
                float s = world.getScale(new Vector3f()).y;
                for (LightRef l : lrefs) {
                    Scene.SphereLight old = scene.lights.get(l.index());
                    scene.lights.set(l.index(), new Scene.SphereLight(world.transformPosition(new Vector3f(l.localPos())),
                            l.localRadius() * s, old.radiance(), old.shape()));
                    lights = true;
                }
            }
        }
        if (lights) scene.lightsMoved();
        return any || lights;
    }

    /**
     * Emits the components of {@code src} at {@code world}. {@code src} is either the scene entity {@code owner}
     * itself or an object inside one of its prefabs or instanced copies; everything produced is attributed to
     * {@code owner}.
     */
    private void emit(Entity owner, Entity src, Matrix4f world, int depth) {
        for (Component c : src.components) {
            switch (c) {
                case PrimitiveComponent p -> addPrimitive(owner, p, world);
                case ModelComponent m -> addModel(owner, m, world);
                case LightComponent l -> addLight(owner, l, world);
                case CameraComponent cam -> {
                    if (cam.player && depth == 0) setCamera(owner, world);
                }
                case PrefabComponent pc -> {
                    if (pc.prefab == null) break;
                    if (depth >= MAX_PREFAB_DEPTH) { System.err.println("[engine] prefab nesting too deep: " + pc.prefab); break; }
                    SceneDoc prefab = assets.prefab(pc.prefab);
                    Matrix4f savedRel = new Matrix4f(relative);
                    Matrix4f childRel = new Matrix4f(), childWorld = new Matrix4f();
                    for (Entity child : prefab.entities) {
                        prefab.worldMatrix(child, childRel);
                        relative.set(savedRel).mul(childRel);
                        ownerWorld.mul(relative, childWorld);
                        emit(owner, child, childWorld, depth + 1);
                    }
                    relative.set(savedRel);
                }
                case InstancesComponent ic -> addInstances(owner, ic, depth);
                default -> { }
            }
        }
    }

    /**
     * Matches a node name against a model node filter. The filter is a {@code |}-separated list; each entry matches
     * the node itself and its sub-parts ({@code vintage_suitcase_01} matches {@code vintage_suitcase_01_bottom} but
     * not {@code vintage_suitcase_010}), and {@code name*} matches any prefix. An empty filter matches everything.
     */
    static boolean nodeMatches(String filter, String node) {
        if (filter == null || filter.isBlank()) return true;
        for (String t : filter.split("\\|")) {
            t = t.trim();
            if (t.isEmpty()) continue;
            if (t.endsWith("*") ? node.startsWith(t.substring(0, t.length() - 1)) : node.equals(t) || node.startsWith(t + "_")) return true;
        }
        return false;
    }

    /**
     * Emits the source once per copy, as if each copy were an object inside a prefab, so picking, lights and
     * transform updates work the same way. The copy state is saved and restored to support nesting.
     */
    private void addInstances(Entity owner, InstancesComponent ic, int depth) {
        if (ic.source == null || ic.count() == 0) return;
        if (depth >= MAX_PREFAB_DEPTH) { System.err.println("[engine] instances nested too deep: " + ic.source); return; }
        Entity src = new Entity(owner.name);
        if (ic.isPrefab()) src.with(new PrefabComponent(ic.source));
        else {
            ModelComponent mc = new ModelComponent(ic.source);
            mc.node = ic.node;
            mc.recenter = ic.recenter;
            src.with(mc);
        }
        Matrix4f savedRel = new Matrix4f(relative), t = new Matrix4f(), childWorld = new Matrix4f();
        int savedSub = currentSub;
        Matrix4f savedPre = new Matrix4f(currentPre), savedCopy = new Matrix4f(currentCopy);
        currentPre.set(savedRel);
        for (int i = 0; i < ic.count(); i++) {
            ic.matrix(i, t);
            relative.set(savedRel).mul(t);
            ownerWorld.mul(relative, childWorld);
            currentSub = i;
            currentCopy.set(t);
            emit(owner, src, childWorld, depth + 1);
        }
        relative.set(savedRel);
        currentSub = savedSub;
        currentPre.set(savedPre);
        currentCopy.set(savedCopy);
    }

    /** Adds a render instance at {@code world * local} and records it for transform updates. */
    private int instance(Entity e, int mesh, int material, Matrix4f world, Matrix4f local) {
        Scene.Instance inst = scene.addInstance(mesh, material, new Matrix4f(world).mul(local));
        inst.named(e.name);
        int index = scene.instances.size() - 1;
        // Kept relative to the owner, so a transform update only needs the owner's world matrix.
        Matrix4f rel = new Matrix4f(relative).mul(local);
        InstanceRef ref;
        // post = (pre * copy)^-1 * rel
        if (currentSub >= 0) {
            Matrix4f post = new Matrix4f(currentPre).mul(currentCopy).invert().mul(rel);
            ref = new InstanceRef(index, rel, new Matrix4f(currentPre), post, currentSub);
        } else {
            ref = new InstanceRef(index, rel);
        }
        entityInstances.computeIfAbsent(e.id, k -> new ArrayList<>()).add(ref);
        instanceOwner.add(e.id);
        subRefs.computeIfAbsent(e.id, k -> new HashMap<>()).computeIfAbsent(currentSub, k -> new ArrayList<>()).add(ref);
        return index;
    }

    /**
     * Adds a primitive; meshes are shared per shape and size. Emissive primitives are additionally represented by
     * sphere lights with the same output and their surface becomes a light proxy, because a path tracer only finds
     * small emissive surfaces by chance, which produces fireflies.
     */
    private void addPrimitive(Entity e, PrimitiveComponent p, Matrix4f world) {
        float[] sz = p.size == null ? new float[]{1, 1, 1} : p.size;
        String key = p.shape + ":" + Arrays.toString(sz);
        int mesh = primitiveMeshes.computeIfAbsent(key, k -> scene.addMesh(switch (p.shape) {
            case "sphere" -> MeshFactory.sphere(sz[0], 32, 24);
            case "plane" -> MeshFactory.plane(sz[0]);
            case "torus" -> MeshFactory.torus(sz[0], sz.length > 1 ? sz[1] : sz[0] * 0.3f, 48, 24);
            case "cube" -> MeshFactory.cube();
            case "cylinder" -> MeshFactory.cylinder(sz[0], sz.length > 1 ? sz[1] : sz[0], sz.length > 2 ? sz[2] : sz[0], 32);
            default -> MeshFactory.box(sz[0], sz[1], sz[2]);
        }));
        int mat = material(p.material, p.materialDef);
        Material m = scene.materials.get(mat);
        Vector3f em = m.emission();
        if (EMISSIVE_LIGHTS && em.x + em.y + em.z > 0 && !m.isLightProxy()) {
            Vector3f s = world.getScale(new Vector3f());
            float sy = Math.max(1e-6f, s.y);
            if (p.shape.equals("sphere")) {
                // the light coincides with the ball
                float radius = s.y * sz[0];
                scene.lights.add(new Scene.SphereLight(world.transformPosition(new Vector3f()), radius, new Vector3f(em),
                        Scene.Shape.SPHERE));
                entityLights.computeIfAbsent(e.id, k -> new ArrayList<>()).add(new LightRef(scene.lights.size() - 1,
                        relative.transformPosition(new Vector3f()), radius / sy));
            } else {
                // Boxes and planes: edge lengths in mesh space and in world space.
                float[] local = p.shape.equals("plane") ? new float[]{2 * sz[0], 0, 2 * sz[0]}
                        : p.shape.equals("cube") ? new float[]{1, 1, 1}
                        : new float[]{sz[0], sz.length > 1 ? sz[1] : sz[0], sz.length > 2 ? sz[2] : sz[0]};
                float[] d = {local[0] * s.x, local[1] * s.y, local[2] * s.z};
                int thin = d[0] <= d[1] && d[0] <= d[2] ? 0 : d[1] <= d[2] ? 1 : 2;
                float a1 = d[(thin + 1) % 3], a2 = d[(thin + 2) % 3];
                boolean panel = d[thin] < 0.25f * Math.min(a1, a2);
                // Match the emitted power: a full sphere in front of each side of a thin panel (4 pi r^2 = 2 A),
                // a half sphere protruding from a solid box.
                float radius = (float) Math.max(0.01, Math.sqrt(a1 * a2 / ((panel ? 4 : 2) * Math.PI)));
                float[] sides = panel ? new float[]{1, -1} : new float[]{0};
                for (float side : sides) {
                    // Place the sphere in front of the face rather than inside it. For panels mounted on a wall,
                    // the sphere on the wall side is simply shadowed by the wall.
                    Vector3f lp = new Vector3f();
                    float[] sc = {s.x, s.y, s.z};
                    lp.setComponent(thin, side * (radius + d[thin] / 2) / Math.max(1e-6f, sc[thin]));
                    scene.lights.add(new Scene.SphereLight(world.transformPosition(new Vector3f(lp)), radius, new Vector3f(em)));
                    entityLights.computeIfAbsent(e.id, k -> new ArrayList<>()).add(new LightRef(scene.lights.size() - 1,
                            relative.transformPosition(new Vector3f(lp)), radius / sy));
                }
            }
            Material proxy = m.withFlags(Material.LIGHT_PROXY);
            mat = cachedMaterial("proxy:" + mat, () -> proxy);
        }
        instance(e, mesh, mat, world, new Matrix4f());
    }

    /** A sun light replaces the scene's sun; a sphere light is placed at the entity and scales with it. */
    private void addLight(Entity e, LightComponent l, Matrix4f world) {
        Vector3f color = new Vector3f(l.color[0], l.color[1], l.color[2]);
        if ("sun".equals(l.kind)) {
            // direction towards the sun
            Vector3f dir = world.transformDirection(new Vector3f(0, 0, 1)).normalize();
            scene.sunDirection.set(dir);
            scene.sunColor.set(color);
            scene.sunIntensity = l.intensity;
            return;
        }
        scene.lights.add(new Scene.SphereLight(world.transformPosition(new Vector3f()), l.radius * world.getScale(new Vector3f()).y,
                color.mul(l.intensity), Scene.Shape.NONE));
        entityLights.computeIfAbsent(e.id, k -> new ArrayList<>()).add(new LightRef(scene.lights.size() - 1,
                relative.transformPosition(new Vector3f()), l.radius));
    }

    private void setCamera(Entity e, Matrix4f world) {
        world.getTranslation(scene.cameraPosition);
        scene.cameraYaw = (float) Math.toRadians(e.rotation[1]);
        scene.cameraPitch = (float) Math.toRadians(e.rotation[0]);
    }

    /**
     * Adds the parts of a model that pass the node filter. Some materials are treated specially:
     * <ul>
     *   <li>alpha-masked and emissive: candle flame cards, replaced by a flame shape and a sphere light per flame;</li>
     *   <li>bulb materials (see {@link #isBulb}): glowing proxies with a sphere light per bulb;</li>
     *   <li>cutouts such as leaves: alpha tested;</li>
     *   <li>other alpha-masked materials (window panes, lamp shades): rendered as glass.</li>
     * </ul>
     */
    private void addModel(Entity e, ModelComponent mc, Matrix4f world) {
        GltfLoader.Model model = assets.model(mc.model);
        int base = modelTextureBase.computeIfAbsent(mc.model, k -> {
            int b = scene.textureSets.size();
            for (TextureSet t : model.textureSets()) scene.addTextureSet(t);
            texturesAdded = true;
            return b;
        });
        int override = mc.material != null ? material(mc.material, null) : -1;
        int[] mat = new int[model.materials().size()];
        for (int i = 0; i < mat.length; i++) {
            GltfLoader.ModelMaterial mm = model.materials().get(i);
            Material m = mm.material();
            if (m.textureSet() >= 0) m = m.withTexture(m.textureSet() + base, m.uvScale(), m.normalStrength());
            if (mm.alphaMasked() && mm.emissive()) mat[i] = -1;
            else if (isBulb(mm.name())) mat[i] = -2;
            else if (override >= 0) mat[i] = override;
            else if (mm.cutout()) {
                Material fm = m.withFlags(m.flags() | Material.ALPHA_TEST);
                mat[i] = cachedMaterial(mc.model + "#" + i, () -> fm);
            }
            else if (mm.alphaMasked()) mat[i] = cachedMaterial("glass-pane", () -> Material.glass(0.97f, 0.97f, 0.97f, 1.5f));
            else {
                Material fm = m;
                mat[i] = cachedMaterial(mc.model + "#" + i, () -> fm);
            }
        }
        List<Integer> partIdx = new ArrayList<>();
        for (int i = 0; i < model.parts().size(); i++) {
            if (nodeMatches(mc.node, model.parts().get(i).nodeName())) partIdx.add(i);
        }
        Matrix4f local = new Matrix4f();
        // cached, since thousands of copies may share the model
        if (mc.recenter) {
            local.set(recenterCache.computeIfAbsent(mc.model + "|" + mc.node, k -> {
                Vector3f lo = new Vector3f(Float.MAX_VALUE), hi = new Vector3f(-Float.MAX_VALUE);
                for (int i : partIdx) bounds(model.parts().get(i).mesh(), lo, hi);
                return new Matrix4f().translate(-(lo.x + hi.x) / 2, -lo.y, -(lo.z + hi.z) / 2);
            }));
        }
        float sy = world.getScale(new Vector3f()).y;
        for (int i : partIdx) {
            GltfLoader.Part pt = model.parts().get(i);
            if (mat[pt.material()] == -2) {
                addBulbs(e, mc.model + "#" + i, pt.mesh(), world, local, sy, mc.flameRadiance);
                continue;
            }
            if (mat[pt.material()] < 0) {
                if (mc.flameRadiance > 0) addFlames(e, pt.mesh(), world, local, sy, mc.flameRadiance);
                continue;
            }
            int mesh = partMeshes.computeIfAbsent(mc.model + "#" + i, k -> scene.addMesh(pt.mesh()));
            instance(e, mesh, mat[pt.material()], world, local);
        }
    }

    /**
     * Replaces a flame card, which may hold several flames, by a glowing flame shape and a sphere light per flame.
     */
    private void addFlames(Entity e, Mesh card, Matrix4f world, Matrix4f local, float scaleY, float radiance) {
        if (flameMesh < 0) flameMesh = scene.addMesh(MeshFactory.sphere(16, 12));
        int flameMat = cachedMaterial("flame", () -> Material.emissive(FLAME_COLOR.x, FLAME_COLOR.y, FLAME_COLOR.z, 30f)
                .withFlags(Material.LIGHT_PROXY));
        for (float[] c : clusterCache.computeIfAbsent(card, SceneBuilder::clusters)) {
            // flame height in model space
            float h = Math.max(c[3], 0.02f / Math.max(scaleY, 1e-4f));
            Vector3f center = local.transformPosition(new Vector3f(c[0], c[1], c[2]));
            Matrix4f shape = new Matrix4f().translate(center).scale(h * 0.16f, h * 0.42f, h * 0.16f);
            instance(e, flameMesh, flameMat, world, shape);
            Vector3f lightLocal = new Vector3f(center).sub(0, h * 0.1f, 0);
            scene.lights.add(new Scene.SphereLight(world.transformPosition(new Vector3f(lightLocal)), h * 0.14f * scaleY,
                    new Vector3f(FLAME_COLOR).mul(radiance)));
            entityLights.computeIfAbsent(e.id, k -> new ArrayList<>()).add(new LightRef(scene.lights.size() - 1,
                    relative.transformPosition(new Vector3f(lightLocal)), h * 0.14f));
        }
    }

    /** Bulb materials are recognized by name: containing "bulb", starting with "lamps_" or ending with "_light". */
    static boolean isBulb(String material) {
        String n = material == null ? "" : material.toLowerCase(Locale.ROOT);
        return n.contains("bulb") || n.startsWith("lamps_") || n.endsWith("_light");
    }

    /**
     * Lamp bulbs: the bulb mesh glows as a light proxy and each connected bulb gets a sphere light. With zero
     * radiance the lamp is off and the bulbs are drawn as frosted glass.
     */
    private void addBulbs(Entity e, String key, Mesh bulbs, Matrix4f world, Matrix4f local, float scaleY, float radiance) {
        int mesh = partMeshes.computeIfAbsent(key, k -> scene.addMesh(bulbs));
        if (radiance <= 0) {
            instance(e, mesh, cachedMaterial("bulb-off", () -> Material.diffuse(0.85f, 0.83f, 0.78f)), world, local);
            return;
        }
        int glow = cachedMaterial("bulb-on", () -> Material.emissive(BULB_COLOR.x, BULB_COLOR.y, BULB_COLOR.z, 12f)
                .withFlags(Material.LIGHT_PROXY));
        instance(e, mesh, glow, world, local);
        for (float[] c : clusterCache.computeIfAbsent(bulbs, SceneBuilder::clusters)) {
            Vector3f center = local.transformPosition(new Vector3f(c[0], c[1], c[2]));
            float r = Math.max(c[3] * 0.3f, 0.01f) * scaleY;
            scene.lights.add(new Scene.SphereLight(world.transformPosition(new Vector3f(center)), r,
                    new Vector3f(BULB_COLOR).mul(radiance * 16f)));
            entityLights.computeIfAbsent(e.id, k -> new ArrayList<>()).add(new LightRef(scene.lights.size() - 1,
                    relative.transformPosition(new Vector3f(center)), r / Math.max(scaleY, 1e-4f)));
        }
    }

    /** Material index for a key; created on first use within a build. */
    private int cachedMaterial(String key, Supplier<Material> make) {
        return materialCache.computeIfAbsent(key, k -> scene.addMaterial(make.get()));
    }

    /** Material index for an asset or an inline definition; plain gray if neither is given. */
    private int material(String asset, MaterialDef inline) {
        if (asset != null) return cachedMaterial("asset:" + asset, () -> toMaterial(assets.material(asset)));
        if (inline != null) {
            try {
                String key = "inline:" + SceneDoc.JSON.writeValueAsString(inline);
                return cachedMaterial(key, () -> toMaterial(inline));
            } catch (IOException ex) {
                throw new IllegalStateException(ex);
            }
        }
        return cachedMaterial("default", () -> Material.diffuse(0.7f, 0.7f, 0.7f));
    }

    private Material toMaterial(MaterialDef d) {
        int set = -1;
        if (d.textures != null) {
            set = textureSets.computeIfAbsent(d.textures, k -> {
                texturesAdded = true;
                return scene.addTextureSet(assets.textures(k));
            });
        }
        return new Material(new Vector3f(d.albedo[0], d.albedo[1], d.albedo[2]), d.roughness, d.metallic,
                new Vector3f(d.emission[0], d.emission[1], d.emission[2]), d.ior, d.transmission, 0f,
                set, d.uvScale, d.normalStrength, d.lightProxy ? Material.LIGHT_PROXY : 0);
    }

    /**
     * Applies the scene settings. The HDR panorama is only reloaded when it changed; its sun is extracted and, if
     * requested, rotated, and then replaces the sun from the settings.
     */
    private void applySettings(Settings s) throws IOException {
        scene.night = s.night;
        scene.exposure = s.exposure;
        scene.skyIntensity = s.skyIntensity;
        if (!Objects.equals(s.environment, loadedEnvironment) || s.environmentSunU != loadedEnvironmentSunU) {
            scene.environment = null;
            if (s.environment != null) {
                Environment env = assets.environment(s.environment);
                env.extractSun(30f);
                if (s.environmentSunU >= 0) env.rotateSunTo(s.environmentSunU);
                scene.environment = env;
            }
            loadedEnvironment = s.environment;
            loadedEnvironmentSunU = s.environmentSunU;
            scene.environmentChanged();
        }
        if (scene.environment != null) {
            scene.night = false;
            scene.skyIntensity = s.environmentStrength;
            scene.sunDirection.set(scene.environment.sunDirection);
            scene.sunColor.set(scene.environment.sunColor);
            scene.sunIntensity = scene.environment.sunIrradiance * s.environmentStrength;
        }
    }

    /**
     * Splits a mesh into connected parts (union-find over triangles), e.g. the flames of a card or the bulbs of a
     * lamp, and returns {cx, cy, cz, height} per part. Parts whose centers are within 3 cm are merged.
     */
    private static List<float[]> clusters(Mesh card) {
        int n = card.vertexCount();
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) parent[i] = i;
        float[] v = card.vertices;
        int st = Mesh.FLOATS_PER_VERTEX;
        for (int t = 0; t < card.indices.length; t += 3) {
            union(parent, card.indices[t], card.indices[t + 1]);
            union(parent, card.indices[t], card.indices[t + 2]);
        }
        Map<Integer, float[]> agg = new HashMap<>();
        for (int i = 0; i < n; i++) {
            float[] a = agg.computeIfAbsent(find(parent, i), r -> new float[]{1e9f, 1e9f, 1e9f, -1e9f, -1e9f, -1e9f});
            for (int c = 0; c < 3; c++) { a[c] = Math.min(a[c], v[i * st + c]); a[3 + c] = Math.max(a[3 + c], v[i * st + c]); }
        }
        List<float[]> out = new ArrayList<>();
        for (float[] a : agg.values()) {
            float cx = (a[0] + a[3]) / 2, cy = (a[1] + a[4]) / 2, cz = (a[2] + a[5]) / 2, h = a[4] - a[1];
            boolean merged = false;
            for (float[] o : out) {
                if (Math.abs(o[0] - cx) < 0.03f && Math.abs(o[1] - cy) < 0.03f && Math.abs(o[2] - cz) < 0.03f) { merged = true; break; }
            }
            if (!merged) out.add(new float[]{cx, cy, cz, h});
        }
        return out;
    }

    // Union-find with path halving.
    private static int find(int[] p, int i) { while (p[i] != i) i = p[i] = p[p[i]]; return i; }

    private static void union(int[] p, int a, int b) { p[find(p, a)] = find(p, b); }

    /** Extends {@code lo}/{@code hi} by the mesh's vertex positions. */
    static void bounds(Mesh mesh, Vector3f lo, Vector3f hi) {
        float[] v = mesh.vertices;
        for (int i = 0; i < v.length; i += Mesh.FLOATS_PER_VERTEX) {
            lo.min(new Vector3f(v[i], v[i + 1], v[i + 2]));
            hi.max(new Vector3f(v[i], v[i + 1], v[i + 2]));
        }
    }
}
