package dev.lumvex.engine.physics;

import com.github.stephengold.joltjni.*;
import com.github.stephengold.joltjni.enumerate.EActivation;
import com.github.stephengold.joltjni.enumerate.EMotionQuality;
import com.github.stephengold.joltjni.enumerate.EMotionType;
import com.github.stephengold.joltjni.enumerate.EOverrideMassProperties;
import dev.lumvex.engine.SceneBuilder;
import dev.lumvex.engine.SceneDoc;
import dev.lumvex.engine.SceneDoc.ColliderComponent;
import dev.lumvex.engine.SceneDoc.Entity;
import dev.lumvex.engine.SceneQuery;
import dev.lumvex.game.PhysicsBackend;
import org.joml.Matrix4f;
import org.joml.Quaternionf;
import org.joml.Vector3f;

import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;

/**
 * Collision and rigid body physics for a running game. Lives on the render thread next to the {@link SceneBuilder}.
 *
 * <p>Every entity with a {@link ColliderComponent} gets a body:
 * <ul>
 *   <li>{@code simple}: an enclosing k-DOP hull. Contacts and the blocking of script-driven movement
 *       (move-and-slide) are computed here with GJK. The hull is also added to Jolt as a static or kinematic body,
 *       so dynamic bodies collide with it.</li>
 *   <li>{@code physics}: a Jolt body. Static bodies use the exact triangles; dynamic and kinematic bodies use a
 *       convex hull of up to 64 extreme points, or a real sphere for sphere primitives.</li>
 * </ul>
 * Each step writes the transforms of simulated bodies back into the scene document.
 *
 * <p>Without the Jolt native library only the simple collision features are available.
 */
public final class PhysicsWorld implements PhysicsBackend, AutoCloseable {
    private static final int LAYER_MOVING = 0, LAYER_STATIC = 1;
    private static final float CONTACT_MARGIN = 0.003f, BLOCK_MARGIN = -0.002f;
    /**
     * Penetration depth beyond which a body counts as stuck and may move freely to get out. It has to be clearly
     * larger than the GJK tolerance, otherwise a body resting against a wall could slip through it.
     */
    private static final float STUCK_MARGIN = -0.01f;
    /** Distance kept from a contact after a blocked move; larger than the uncertainty of the contact test. */
    private static final float CONTACT_GAP = 0.005f;
    private static boolean joltReady;

    private final SceneBuilder builder;
    private final SceneQuery query;
    private SceneDoc doc;
    private final Map<String, Body> bodies = new LinkedHashMap<>();
    /** Colliders without usable geometry; not retried until they disappear. */
    private final Set<String> failed = new HashSet<>();
    /** Touching pairs keyed "a|b" with a < b; the value tells whether it is a trigger pair. */
    private final Map<String, Boolean> pairs = new HashMap<>();
    private final List<Event> events = new ArrayList<>();
    private boolean firstContacts = true;
    /** Hash of each instanced entity's copies; a change rebuilds the copies' bodies. */
    private final Map<String, Integer> instanceSig = new HashMap<>();
    /** Bodies of instanced copies by owning entity, so moving the owner moves its copies. */
    private final Map<String, List<Body>> byOwner = new HashMap<>();

    private PhysicsSystem system;
    private BodyInterface bi;
    private TempAllocatorMalloc temp;
    private JobSystemThreadPool jobs;
    private final float[] gravity = {0, -9.81f, 0};
    /** Velocities set by scripts before the body was created, e.g. right after spawning. */
    private final Map<String, float[]> pendingVelocity = new HashMap<>();

    /** A contact between two entities began or ended. */
    public record Event(String a, String b, boolean trigger, boolean enter) {}

    /**
     * One collider. Copies of an instances component get one body each, with the id {@code entity#index}.
     * {@code local} holds hull points, or triangles (9 floats each) for mesh bodies.
     */
    private static final class Body {
        final String id;
        final ColliderComponent c;
        final boolean mesh;
        final float[] local;
        float[] world;
        final float[] aabb = new float[6];
        final Matrix4f matrix = new Matrix4f();
        float[] scale;
        /** Transform at the end of the previous step, the start of the next move-and-slide. */
        float[] lastPos, lastRot;
        int jolt = -1;
        EMotionType type;
        boolean moved;
        /** Ignores rotation; used for the player camera so that looking around never collides. */
        boolean upright;
        /** Sphere primitives get a true sphere shape in Jolt so they roll properly. */
        boolean sphere;
        float sphereRadius = 1;
        /** The entity this body belongs to; differs from {@code id} for instanced copies. */
        String owner;
        /** An instanced copy: always static, and copies never report contacts with each other. */
        boolean instance;

        Body(String id, ColliderComponent c, boolean mesh, float[] local) {
            this.id = id;
            this.owner = id;
            this.c = c;
            this.mesh = mesh;
            this.local = local;
        }

        boolean dynamic() { return !instance && "physics".equals(c.mode) && "dynamic".equals(c.body); }

        /** Whether the body stops script-driven movement of other bodies. */
        boolean blocking() { return !c.trigger && !dynamic(); }
    }

    public PhysicsWorld(SceneBuilder builder, SceneQuery query) {
        this.builder = builder;
        this.query = query;
        try {
            initJolt();
            temp = new TempAllocatorMalloc();
            jobs = new JobSystemThreadPool(Jolt.cMaxPhysicsJobs, Jolt.cMaxPhysicsBarriers,
                    Math.max(1, Runtime.getRuntime().availableProcessors() - 2));
            // Two layers; static bodies never need to be tested against each other.
            ObjectLayerPairFilterTable ovo = new ObjectLayerPairFilterTable(2);
            ovo.enableCollision(LAYER_MOVING, LAYER_MOVING);
            ovo.enableCollision(LAYER_MOVING, LAYER_STATIC);
            ovo.disableCollision(LAYER_STATIC, LAYER_STATIC);
            BroadPhaseLayerInterfaceTable bpl = new BroadPhaseLayerInterfaceTable(2, 2);
            bpl.mapObjectToBroadPhaseLayer(LAYER_MOVING, 0);
            bpl.mapObjectToBroadPhaseLayer(LAYER_STATIC, 1);
            ObjectVsBroadPhaseLayerFilterTable ovb = new ObjectVsBroadPhaseLayerFilterTable(bpl, 2, ovo, 2);
            system = new PhysicsSystem();
            system.init(8192, 0, 16384, 8192, bpl, ovb, ovo);
            system.setGravity(gravity[0], gravity[1], gravity[2]);
            bi = system.getBodyInterface();
        } catch (Throwable t) {
            System.out.println("[physics] Jolt not available (" + t + ") - simple collision only");
            system = null;
        }
    }

    /**
     * Loads the Jolt native library once per process. The DLL is extracted from the jar into the temp folder, using a
     * versioned file name so it can be reused by later runs.
     */
    private static synchronized void initJolt() throws Exception {
        if (joltReady) return;
        String res = "/windows/x86-64/com/github/stephengold/joltjni.dll";
        Path dir = Path.of(System.getProperty("java.io.tmpdir"), "lumvex-natives");
        Files.createDirectories(dir);
        Path dll = dir.resolve("joltjni-6.1.1.dll");
        if (!Files.exists(dll)) {
            try (InputStream in = PhysicsWorld.class.getResourceAsStream(res)) {
                if (in == null) throw new IllegalStateException("missing " + res);
                Path tmp = Files.createTempFile(dir, "jolt", ".dll");
                Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING);
                Files.move(tmp, dll, StandardCopyOption.REPLACE_EXISTING);
            }
        }
        System.load(dll.toAbsolutePath().toString());
        Jolt.registerDefaultAllocator();
        Jolt.installDefaultAssertCallback();
        Jolt.installDefaultTraceCallback();
        if (!Jolt.newFactory()) throw new IllegalStateException("Jolt factory");
        Jolt.registerTypes();
        joltReady = true;
        System.out.println("[physics] Jolt " + Jolt.versionString());
    }

    /** Brings the bodies in line with the scene after a rebuild: adds new colliders and removes stale ones. */
    public void sync(SceneDoc doc) {
        this.doc = doc;
        Set<String> present = new HashSet<>();
        boolean added = false;
        for (Entity e : doc.entities) {
            ColliderComponent c = e.get(ColliderComponent.class);
            if (c == null) continue;
            SceneDoc.InstancesComponent ic = e.get(SceneDoc.InstancesComponent.class);
            if (ic != null) {
                int sig = Arrays.hashCode(ic.data) * 31 + Objects.hashCode(ic.source);
                Integer old = instanceSig.put(e.id, sig);
                if (old != null && old != sig) dropCopies(e.id);
                for (int i = 0; i < ic.count(); i++) {
                    String id = e.id + "#" + i;
                    present.add(id);
                    if (bodies.containsKey(id) || failed.contains(id)) continue;
                    Body b = createCopy(e, c, i, id);
                    if (b == null) { failed.add(id); continue; }
                    bodies.put(id, b);
                    added = true;
                }
                continue;
            }
            present.add(e.id);
            if (bodies.containsKey(e.id) || failed.contains(e.id)) continue;
            Body b = create(e, c);
            if (b == null) { failed.add(e.id); continue; }
            bodies.put(e.id, b);
            added = true;
        }
        for (Iterator<Map.Entry<String, Body>> it = bodies.entrySet().iterator(); it.hasNext(); ) {
            Body b = it.next().getValue();
            if (present.contains(b.id)) continue;
            removeJolt(b);
            pairs.keySet().removeIf(k -> k.startsWith(b.id + "|") || k.endsWith("|" + b.id));
            it.remove();
        }
        failed.retainAll(present);
        instanceSig.keySet().removeIf(id -> doc.find(id) == null);
        byOwner.clear();
        for (Body b : bodies.values()) if (b.instance) byOwner.computeIfAbsent(b.owner, k -> new ArrayList<>()).add(b);
        if (added && system != null) system.optimizeBroadPhase();
    }

    private void dropCopies(String owner) {
        for (Iterator<Body> it = bodies.values().iterator(); it.hasNext(); ) {
            Body b = it.next();
            if (b.instance && b.owner.equals(owner)) { removeJolt(b); it.remove(); }
        }
        failed.removeIf(f -> f.startsWith(owner + "#"));
    }

    /** Static hull for copy {@code i} of an instances component, or null if the copy has no geometry. */
    private Body createCopy(Entity e, ColliderComponent c, int i, String id) {
        SceneBuilder.LocalMesh m = builder.localMesh(e.id, i);
        if (m.isEmpty()) return null;
        float[] pts = Hull.enclosing(m.positions(), c.hullVertices);
        Body b = new Body(id, c, false, pts);
        b.owner = e.id;
        b.instance = true;
        matrix(e, b, b.matrix);
        b.scale = scaleOf(b.matrix);
        b.world = transform(b.local, b.matrix, null);
        aabb(b.world, b.aabb);
        b.lastPos = e.position.clone();
        b.lastRot = e.rotation.clone();
        b.moved = true;
        if (system != null && !c.trigger) createJolt(e, b);
        return b;
    }

    /**
     * Builds the collision shape: a box if the collider has a size, otherwise the entity's geometry as exact
     * triangles (static physics bodies) or as a hull. Returns null if there is no geometry.
     */
    private Body create(Entity e, ColliderComponent c) {
        boolean physics = "physics".equals(c.mode);
        Body b;
        if (c.size != null) {
            float[] o = c.offset != null ? c.offset : new float[3];
            float[] pts = new float[24];
            for (int k = 0; k < 8; k++) {
                pts[k * 3] = o[0] + ((k & 1) == 0 ? -0.5f : 0.5f) * c.size[0];
                pts[k * 3 + 1] = o[1] + ((k & 2) == 0 ? -0.5f : 0.5f) * c.size[1];
                pts[k * 3 + 2] = o[2] + ((k & 4) == 0 ? -0.5f : 0.5f) * c.size[2];
            }
            b = new Body(e.id, c, false, pts);
        } else {
            SceneBuilder.LocalMesh m = builder.localMesh(e.id);
            if (m.isEmpty()) {
                System.out.println("[physics] \"" + e.name + "\": collider without geometry - give it a box size");
                return null;
            }
            if (physics && "static".equals(c.body) && !c.trigger) {
                float[] tri = new float[m.indices().length * 3];
                for (int i = 0; i < m.indices().length; i++) System.arraycopy(m.positions(), m.indices()[i] * 3, tri, i * 3, 3);
                b = new Body(e.id, c, true, tri);
            } else {
                b = new Body(e.id, c, false, physics ? Hull.extreme(m.positions(), 64) : Hull.enclosing(m.positions(), c.hullVertices));
            }
        }
        b.upright = e.get(SceneDoc.CameraComponent.class) != null;
        SceneDoc.PrimitiveComponent prim = e.get(SceneDoc.PrimitiveComponent.class);
        b.sphere = c.size == null && prim != null && "sphere".equals(prim.shape) && e.get(SceneDoc.ModelComponent.class) == null;
        if (b.sphere && prim.size != null) b.sphereRadius = prim.size[0];
        matrix(e, b, b.matrix);
        b.scale = scaleOf(b.matrix);
        b.world = transform(b.local, b.matrix, null);
        aabb(b.world, b.aabb);
        b.lastPos = e.position.clone();
        b.lastRot = e.rotation.clone();
        b.moved = true;
        if (system != null && !c.trigger) createJolt(e, b);
        return b;
    }

    /**
     * Creates the Jolt counterpart of a body. Simple colliders become kinematic if a script may move them, static
     * otherwise. Jolt shapes cannot carry a scale, so the points are scaled in advance.
     */
    private void createJolt(Entity e, Body b) {
        boolean physics = "physics".equals(b.c.mode);
        b.type = physics ? switch (b.c.body) {
            case "dynamic" -> EMotionType.Dynamic;
            case "kinematic" -> EMotionType.Kinematic;
            default -> EMotionType.Static;
        } : e.get(SceneDoc.ScriptComponent.class) != null ? EMotionType.Kinematic : EMotionType.Static;
        if (b.instance) b.type = EMotionType.Static;
        float[] s = b.scale;
        float[] pts = b.local.clone();
        for (int i = 0; i < pts.length; i++) pts[i] *= s[i % 3];
        ShapeSettings ss;
        if (b.mesh) {
            ss = new MeshShapeSettings(direct(pts));
        } else if (b.sphere) {
            ss = new SphereShapeSettings(b.sphereRadius * (Math.abs(s[0]) + Math.abs(s[1]) + Math.abs(s[2])) / 3);
        } else {
            pts = thicken(pts, 0.01f);
            ss = new ConvexHullShapeSettings(pts.length / 3, direct(pts), 0.0f);
        }
        ShapeResult res = ss.create();
        if (res.hasError()) {
            System.out.println("[physics] \"" + e.name + "\": invalid shape - " + res.getError());
            return;
        }
        ShapeRefC shape = res.get();
        Vector3f p = b.matrix.getTranslation(new Vector3f());
        Quaternionf q = b.matrix.getNormalizedRotation(new Quaternionf());
        BodyCreationSettings bcs = new BodyCreationSettings(shape.getPtr(), p.x, p.y, p.z, q.x, q.y, q.z, q.w, b.type,
                b.type == EMotionType.Static ? LAYER_STATIC : LAYER_MOVING);
        bcs.setFriction(b.c.friction);
        bcs.setRestitution(b.c.restitution);
        if (b.type == EMotionType.Dynamic) {
            // Continuous collision keeps small, fast bodies such as balls from tunneling through thin walls.
            bcs.setMotionQuality(EMotionQuality.LinearCast);
            MassProperties mp = new MassProperties();
            mp.setMass(Math.max(0.01f, b.c.mass));
            bcs.setOverrideMassProperties(EOverrideMassProperties.CalculateInertia);
            bcs.setMassPropertiesOverride(mp);
        }
        com.github.stephengold.joltjni.Body body = bi.createBody(bcs);
        if (body == null) return;
        b.jolt = body.getId();
        bi.addBody(b.jolt, b.type == EMotionType.Dynamic ? EActivation.Activate : EActivation.DontActivate);
        float[] v = pendingVelocity.remove(b.id);
        if (v != null && b.type == EMotionType.Dynamic) bi.setLinearVelocity(b.jolt, v[0], v[1], v[2]);
    }

    private void removeJolt(Body b) {
        if (system == null || b.jolt < 0) return;
        bi.removeBody(b.jolt);
        bi.destroyBody(b.jolt);
        b.jolt = -1;
    }

    /** Accepts the current transforms of teleported entities without sweeping from their old position. */
    public void teleported(Set<String> ids) {
        for (String id : ids) {
            Body b = bodies.get(id);
            Entity e = doc == null ? null : doc.find(id);
            if (b != null && e != null) {
                b.lastPos = e.position.clone();
                b.lastRot = e.rotation.clone();
            }
        }
    }

    /**
     * Advances one frame: constrains script-driven moves of blocking colliders, steps the simulation, writes the
     * transforms of dynamic bodies back into the scene and updates contacts (see {@link #takeEvents}). Returns all
     * entities that moved.
     */
    public Set<String> step(float dt, Set<String> scriptMoved) {
        Set<String> out = new LinkedHashSet<>(scriptMoved);
        for (Body b : bodies.values()) b.moved = false;
        Set<String> movedClosure = closure(scriptMoved);
        for (String id : scriptMoved) {
            Body b = bodies.get(id);
            Entity e = doc.find(id);
            if (b != null && e != null && b.blocking() && !b.mesh) resolve(b, e);
        }
        for (String id : movedClosure) {
            List<Body> copies = byOwner.get(id);
            Entity owner = copies == null ? null : doc.find(id);
            if (owner != null) {
                for (Body c : copies) {
                    matrix(owner, c, c.matrix);
                    c.world = transform(c.local, c.matrix, c.world);
                    aabb(c.world, c.aabb);
                    c.moved = true;
                    if (system != null && c.jolt >= 0) {
                        Vector3f p = c.matrix.getTranslation(new Vector3f());
                        Quaternionf q = c.matrix.getNormalizedRotation(new Quaternionf());
                        bi.setPositionAndRotation(c.jolt, p.x, p.y, p.z, q.x, q.y, q.z, q.w, EActivation.DontActivate);
                    }
                }
            }
            Body b = bodies.get(id);
            Entity e = doc.find(id);
            if (b == null || e == null) continue;
            matrix(e, b, b.matrix);
            b.world = transform(b.local, b.matrix, b.world);
            aabb(b.world, b.aabb);
            b.moved = true;
            if (system != null && b.jolt >= 0) {
                Vector3f p = b.matrix.getTranslation(new Vector3f());
                Quaternionf q = b.matrix.getNormalizedRotation(new Quaternionf());
                if (b.type == EMotionType.Kinematic && dt > 0) {
                    bi.moveKinematic(b.jolt, new RVec3(p.x, p.y, p.z), new Quat(q.x, q.y, q.z, q.w), dt);
                } else {
                    bi.setPositionAndRotation(b.jolt, p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                            b.type == EMotionType.Dynamic ? EActivation.Activate : EActivation.DontActivate);
                }
            }
        }
        if (system != null && dt > 0 && bodies.values().stream().anyMatch(Body::dynamic)) {
            float h = Math.min(dt, 1 / 20f);      // a long frame (loading, breakpoint) must not explode the simulation
            system.update(h, Math.max(1, (int) Math.ceil(h * 60)), temp, jobs);
            for (Body b : bodies.values()) {
                if (!b.dynamic() || b.jolt < 0 || movedClosure.contains(b.id) || !bi.isActive(b.jolt)) continue;
                Entity e = doc.find(b.owner);
                if (e == null) continue;
                RVec3 p = bi.getPosition(b.jolt);
                Quat q = bi.getRotation(b.jolt);
                Matrix4f world = new Matrix4f().translation((float) p.xx(), (float) p.yy(), (float) p.zz())
                        .rotate(new Quaternionf(q.getX(), q.getY(), q.getZ(), q.getW())).scale(b.scale[0], b.scale[1], b.scale[2]);
                Entity parent = e.parent == null ? null : doc.find(e.parent);
                Matrix4f local = parent == null ? world : doc.worldMatrix(parent, new Matrix4f()).invert().mul(world);
                Vector3f t = local.getTranslation(new Vector3f());
                Vector3f eu = local.getNormalizedRotation(new Quaternionf()).getEulerAnglesYXZ(new Vector3f());
                e.position = new float[]{t.x, t.y, t.z};
                e.rotation = new float[]{(float) Math.toDegrees(eu.x), (float) Math.toDegrees(eu.y), (float) Math.toDegrees(eu.z)};
                b.matrix.set(world);
                b.world = transform(b.local, b.matrix, b.world);
                aabb(b.world, b.aabb);
                b.moved = true;
                out.add(b.id);
            }
        }
        for (Body b : bodies.values()) {
            if (b.instance) continue;
            Entity e = doc.find(b.id);
            if (e != null) { b.lastPos = e.position.clone(); b.lastRot = e.rotation.clone(); }
        }
        updateContacts();
        return out;
    }

    /** Contact events since the last call. */
    public List<Event> takeEvents() {
        List<Event> r = new ArrayList<>(events);
        events.clear();
        return r;
    }

    /** World matrix of a body; upright bodies drop the rotation. */
    private Matrix4f matrix(Entity e, Body b, Matrix4f out) {
        doc.worldMatrix(e, out);
        if (b.upright) {
            Vector3f t = out.getTranslation(new Vector3f()), s = out.getScale(new Vector3f());
            out.identity().translate(t).scale(s);
        }
        return out;
    }

    /**
     * Move-and-slide for a script-driven move. If the straight path from the previous position is blocked, the
     * rotation change is dropped when it causes the collision, and the move is retried per axis (x, z, then y), each
     * as far as it is free. A body that was already stuck is let go so it can get out.
     */
    private void resolve(Body b, Entity e) {
        float[] np = e.position.clone(), nr = e.rotation.clone(), op = b.lastPos, or = b.lastRot;
        if (op == null) return;
        Set<String> family = family(e);
        float step = sweepStep(b);
        if (!sweepBlocked(b, e, op, np, nr, family, step)) return;
        if (blockedAt(b, e, op, or, family, STUCK_MARGIN)) return;
        float[] rot = nr;
        if (!Arrays.equals(nr, or) && blockedAt(b, e, op, nr, family, BLOCK_MARGIN)) rot = or;
        float[] pos = op.clone();
        for (int axis : new int[]{0, 2, 1}) {
            if (np[axis] == pos[axis]) continue;
            pos[axis] = advance(b, e, pos, axis, np[axis], rot, family, step);
        }
        e.position = pos;
        e.rotation = rot;
    }

    /** Sweep step length: 40% of the body's thinnest extent, so no collider can be skipped. */
    private static float sweepStep(Body b) {
        float[] box = aabb(b.world, new float[6]);
        float thin = Math.min(box[3] - box[0], Math.min(box[4] - box[1], box[5] - box[2]));
        return Math.max(0.01f, thin * 0.4f);
    }

    /** Whether the body collides anywhere on the straight path from {@code a} to {@code to}. */
    private boolean sweepBlocked(Body b, Entity e, float[] a, float[] to, float[] rot, Set<String> family, float step) {
        float dist = (float) Math.sqrt((to[0] - a[0]) * (to[0] - a[0]) + (to[1] - a[1]) * (to[1] - a[1]) + (to[2] - a[2]) * (to[2] - a[2]));
        int n = Math.max(1, (int) Math.ceil(dist / step));
        for (int i = 1; i <= n; i++) {
            float t = (float) i / n;
            float[] p = {a[0] + (to[0] - a[0]) * t, a[1] + (to[1] - a[1]) * t, a[2] + (to[2] - a[2]) * t};
            if (blockedAt(b, e, p, rot, family, BLOCK_MARGIN)) return true;
        }
        return false;
    }

    /**
     * Moves along one axis until blocked and bisects the contact point. Returns the reachable coordinate, backed off
     * by {@link #CONTACT_GAP}.
     */
    private float advance(Body b, Entity e, float[] pos, int axis, float target, float[] rot, Set<String> family, float step) {
        float start = pos[axis];
        int n = Math.max(1, (int) Math.ceil(Math.abs(target - start) / step));
        float[] t = pos.clone();
        float free = start;
        for (int i = 1; i <= n; i++) {
            float v = start + (target - start) * i / n;
            t[axis] = v;
            if (!blockedAt(b, e, t, rot, family, BLOCK_MARGIN)) { free = v; continue; }
            float lo = free, hi = v;
            for (int k = 0; k < 10; k++) {
                float mid = (lo + hi) / 2;
                t[axis] = mid;
                if (blockedAt(b, e, t, rot, family, BLOCK_MARGIN)) hi = mid;
                else lo = mid;
            }
            // Back off a little so the remaining axes can slide along the surface.
            float back = Math.signum(target - start) * CONTACT_GAP;
            return Math.abs(lo - start) > CONTACT_GAP ? lo - back : start;
        }
        return free;
    }

    /**
     * Whether the body would collide with a blocking body outside its own hierarchy at the given transform. Negative
     * margins require that much penetration.
     */
    private boolean blockedAt(Body b, Entity e, float[] pos, float[] rot, Set<String> family, float margin) {
        float[] sp = e.position, sr = e.rotation;
        e.position = pos;
        e.rotation = rot;
        Matrix4f m = matrix(e, b, new Matrix4f());
        e.position = sp;
        e.rotation = sr;
        float[] w = transform(b.local, m, null);
        float[] box = aabb(w, new float[6]);
        for (Body o : bodies.values()) {
            if (o == b || !o.blocking() || family.contains(o.owner)) continue;
            if (!boxesOverlap(box, o.aabb, 0)) continue;
            if (touches(w, box, o, margin)) return true;
        }
        return false;
    }

    /** Hull vs. body test. Mesh bodies are tested per triangle, after an AABB check. */
    private static boolean touches(float[] hull, float[] box, Body o, float margin) {
        Gjk.Shape a = Gjk.points(hull);
        if (!o.mesh) return Gjk.intersect(a, Gjk.points(o.world), margin);
        float m = Math.max(margin, 0);
        for (int t = 0; t < o.world.length / 9; t++) {
            int k = t * 9;
            float[] w = o.world;
            if (Math.max(w[k], Math.max(w[k + 3], w[k + 6])) < box[0] - m || Math.min(w[k], Math.min(w[k + 3], w[k + 6])) > box[3] + m) continue;
            if (Math.max(w[k + 1], Math.max(w[k + 4], w[k + 7])) < box[1] - m || Math.min(w[k + 1], Math.min(w[k + 4], w[k + 7])) > box[4] + m) continue;
            if (Math.max(w[k + 2], Math.max(w[k + 5], w[k + 8])) < box[2] - m || Math.min(w[k + 2], Math.min(w[k + 5], w[k + 8])) > box[5] + m) continue;
            if (Gjk.intersect(a, Gjk.points(w, t * 3, t * 3 + 3), margin)) return true;
        }
        return false;
    }

    /**
     * Detects contacts that began or ended. Only bodies that moved are tested (all non-instanced bodies in the first
     * frame), so the cost is moved x all rather than all pairs, which keeps scenes with thousands of static copies
     * cheap.
     */
    private void updateContacts() {
        List<Body> list = new ArrayList<>(bodies.values());
        Set<Body> done = new HashSet<>();
        for (Body a : list) {
            if (firstContacts ? a.instance : !a.moved) continue;
            done.add(a);
            for (Body b : list) {
                if (b == a || done.contains(b)) continue;
                if (a.mesh && b.mesh) continue;
                if (a.instance && b.instance) continue;
                if (a.owner.equals(b.owner)) continue;
                boolean touching = boxesOverlap(a.aabb, b.aabb, CONTACT_MARGIN)
                        && (a.mesh ? touches(b.world, b.aabb, a, CONTACT_MARGIN) : touches(a.world, a.aabb, b, CONTACT_MARGIN));
                String key = a.id.compareTo(b.id) < 0 ? a.id + "|" + b.id : b.id + "|" + a.id;
                Boolean was = pairs.get(key);
                if (touching == (was != null)) continue;
                boolean trigger = a.c.trigger || b.c.trigger;
                if (touching) {
                    pairs.put(key, trigger);
                    events.add(new Event(a.owner, b.owner, trigger, true));      // report the entity, not the copy
                } else {
                    pairs.remove(key);
                    events.add(new Event(a.owner, b.owner, was, false));
                }
            }
        }
        firstContacts = false;
    }

    @Override public boolean hasBody(String id) { return bodies.containsKey(id); }

    @Override public boolean isDynamic(String id) {
        Body b = bodies.get(id);
        return b != null && b.dynamic();
    }

    @Override public float[] velocity(String id) {
        Body b = bodies.get(id);
        if (system == null || b == null || b.jolt < 0) return new float[3];
        var v = bi.getLinearVelocity(b.jolt);
        return new float[]{v.getX(), v.getY(), v.getZ()};
    }

    /** For bodies that do not exist yet, the velocity is applied when they are created. */
    @Override public void setVelocity(String id, float x, float y, float z) {
        Body b = bodies.get(id);
        if (b == null) pendingVelocity.put(id, new float[]{x, y, z});
        if (system != null && b != null && b.jolt >= 0) {
            bi.setLinearVelocity(b.jolt, x, y, z);
            bi.activateBody(b.jolt);
        }
    }

    @Override public void addImpulse(String id, float x, float y, float z) {
        Body b = bodies.get(id);
        if (system != null && b != null && b.jolt >= 0 && b.dynamic()) bi.addImpulse(b.jolt, x, y, z);
    }

    @Override public void addForce(String id, float x, float y, float z) {
        Body b = bodies.get(id);
        if (system != null && b != null && b.jolt >= 0 && b.dynamic()) {
            bi.addForce(b.jolt, x, y, z);
            bi.activateBody(b.jolt);
        }
    }

    @Override public float[] gravity() { return gravity.clone(); }

    @Override public void setGravity(float x, float y, float z) {
        gravity[0] = x;
        gravity[1] = y;
        gravity[2] = z;
        if (system != null) system.setGravity(x, y, z);
        for (Body b : bodies.values()) if (system != null && b.dynamic() && b.jolt >= 0) bi.activateBody(b.jolt);
    }

    /** Traces against the rendered geometry rather than the colliders. */
    @Override public Hit raycast(float[] origin, float[] dir, float maxDistance, String ignoreEntity) {
        Set<Integer> skip = ignoreEntity == null ? Set.of() : new HashSet<>(builder.instancesOf(ignoreEntity));
        SceneQuery.Hit h = query.raycast(new Vector3f(origin[0], origin[1], origin[2]), new Vector3f(dir[0], dir[1], dir[2]).normalize(),
                maxDistance, skip::contains);
        if (h == null) return null;
        return new Hit(builder.entityOfInstance(h.instance()), new float[]{h.point().x, h.point().y, h.point().z},
                new float[]{h.normal().x, h.normal().y, h.normal().z}, h.t());
    }

    @Override public List<String> contacts(String id) {
        List<String> out = new ArrayList<>();
        for (String k : pairs.keySet()) {
            int bar = k.indexOf('|');
            String a = k.substring(0, bar), b = k.substring(bar + 1);
            if (a.equals(id)) out.add(b);
            else if (b.equals(id)) out.add(a);
        }
        return out;
    }

    /** World-space hull points of a collider, or null for mesh bodies and unknown ids. */
    public float[] worldHull(String id) {
        Body b = bodies.get(id);
        return b == null || b.mesh ? null : b.world.clone();
    }

    public int bodyCount() { return bodies.size(); }

    @Override
    public void close() {
        for (Body b : bodies.values()) removeJolt(b);
        bodies.clear();
        try {
            if (system != null) system.close();
            if (jobs != null) jobs.close();
            if (temp != null) temp.close();
        } catch (Throwable t) {
            System.out.println("[physics] close: " + t);
        }
        system = null;
    }

    /** The given entities plus all of their descendants. */
    private Set<String> closure(Set<String> ids) {
        Set<String> all = new HashSet<>(ids);
        boolean grew = true;
        while (grew) {
            grew = false;
            for (Entity e : doc.entities) if (e.parent != null && all.contains(e.parent) && all.add(e.id)) grew = true;
        }
        return all;
    }

    /** The entity with its descendants and ancestors. They move together, so they never block each other. */
    private Set<String> family(Entity e) {
        Set<String> f = closure(Set.of(e.id));
        Set<String> seen = new HashSet<>();
        for (Entity p = e.parent == null ? null : doc.find(e.parent); p != null && seen.add(p.id); p = p.parent == null ? null : doc.find(p.parent)) f.add(p.id);
        return f;
    }

    /** Transforms packed xyz points; {@code out} is reused if it has the right length. */
    private static float[] transform(float[] local, Matrix4f m, float[] out) {
        if (out == null || out.length != local.length) out = new float[local.length];
        for (int i = 0; i < local.length; i += 3) {
            float x = local[i], y = local[i + 1], z = local[i + 2];
            out[i] = m.m00() * x + m.m10() * y + m.m20() * z + m.m30();
            out[i + 1] = m.m01() * x + m.m11() * y + m.m21() * z + m.m31();
            out[i + 2] = m.m02() * x + m.m12() * y + m.m22() * z + m.m32();
        }
        return out;
    }

    /** AABB of packed xyz points as {minx, miny, minz, maxx, maxy, maxz}. */
    private static float[] aabb(float[] p, float[] out) {
        out[0] = out[1] = out[2] = Float.MAX_VALUE;
        out[3] = out[4] = out[5] = -Float.MAX_VALUE;
        for (int i = 0; i < p.length; i += 3) {
            for (int c = 0; c < 3; c++) {
                out[c] = Math.min(out[c], p[i + c]);
                out[3 + c] = Math.max(out[3 + c], p[i + c]);
            }
        }
        return out;
    }

    private static float[] scaleOf(Matrix4f m) {
        Vector3f s = m.getScale(new Vector3f());
        return new float[]{s.x, s.y, s.z};
    }

    private static FloatBuffer direct(float[] a) {
        FloatBuffer fb = ByteBuffer.allocateDirect(a.length * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        fb.put(a).flip();
        return fb;
    }

    /**
     * Jolt cannot build a hull from coplanar points, so flat shapes such as planes are extruded to a minimum
     * thickness along every axis that is thinner than {@code min}.
     */
    private static float[] thicken(float[] pts, float min) {
        float[] box = aabb(pts, new float[6]);
        for (int c = 0; c < 3; c++) {
            if (box[3 + c] - box[c] >= min) continue;
            int n = pts.length / 3;
            float[] out = new float[pts.length * 2];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < 3; k++) {
                    out[i * 3 + k] = pts[i * 3 + k];
                    out[(n + i) * 3 + k] = pts[i * 3 + k];
                }
                out[i * 3 + c] -= min / 2;
                out[(n + i) * 3 + c] += min / 2;
            }
            pts = out;
        }
        return pts;
    }

    private static boolean boxesOverlap(float[] a, float[] b, float m) {
        return a[0] <= b[3] + m && a[3] >= b[0] - m && a[1] <= b[4] + m && a[4] >= b[1] - m && a[2] <= b[5] + m && a[5] >= b[2] - m;
    }
}
