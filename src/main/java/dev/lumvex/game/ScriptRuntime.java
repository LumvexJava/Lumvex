package dev.lumvex.game;

import dev.lumvex.engine.SceneDoc;
import dev.lumvex.engine.SceneDoc.*;
import dev.lumvex.ui.UiDoc;
import dev.lumvex.ui.UiPainter;

import java.io.IOException;
import java.lang.reflect.*;
import java.nio.file.Path;
import java.util.*;
import java.util.function.Function;

/**
 * Runs the scripts of a playing scene.
 *
 * <p>The runtime works on its own copy of the scene document and is driven from the render thread: the host calls
 * {@link #update} once per frame and then applies the changes it reports through {@link #takeMoved},
 * {@link #takeStructure} and {@link #takeSettings}. Every call into user code goes through {@link #guard}, so a
 * failing script is reported with file and line instead of taking the game down. A script that throws from
 * {@code onUpdate} is disabled; the others keep running.
 */
public final class ScriptRuntime {
    /** Resources the runtime loads on demand. */
    public interface Host {
        /** A fresh instance of the scene with the given name. */
        Scene loadScene(String name) throws IOException;

        /** The name a scene class was added under, or null. */
        default String sceneName(Class<? extends Scene> type) { return null; }

        SceneDoc prefab(String id);

        /** A UI canvas by asset id ({@code ui/x.ui.json}), or null. */
        default UiDoc ui(String assetId) { return null; }
    }

    private static volatile ScriptRuntime current;

    /** The runtime of the game that is currently playing, or null. */
    public static ScriptRuntime current() { return current; }

    private final ScriptCompiler.Result compiled;
    private final Host host;
    private Scene scene;
    SceneDoc doc;
    final Input input;
    final Render render = new Render(this);
    final Time time = new Time();
    final Physics physics = new Physics(this);
    volatile PhysicsBackend physicsBackend;

    private final Map<String, GameObject> objects = new LinkedHashMap<>();
    private final List<Script> scripts = new ArrayList<>();
    private SceneScript sceneScript;

    private record Timer(Script script, Method method, float interval, float[] next) {}

    private final List<Timer> timers = new ArrayList<>();

    private final Set<String> moved = new LinkedHashSet<>();
    private boolean structure, settings, stopRequested;
    private String pendingScene;
    public volatile String lastError;
    public volatile int errors;

    public ScriptRuntime(ScriptCompiler.Result compiled, Scene scene, Input input, Host host) {
        this.compiled = compiled;
        this.scene = scene;
        this.doc = scene.document();
        this.input = input;
        this.host = host;
    }

    public void start() {
        current = this;
        enter(scene);
        System.out.println("[game] started " + doc.name + " (" + scripts.size() + " scripts)");
    }

    /**
     * Resets all per-scene state and binds the scene: its objects, scene script, object scripts and the canvases
     * placed in it. Then runs {@code onLoad} of the scene script and {@code onStart} of everything.
     */
    private void enter(Scene s) {
        SceneDoc d = s.document();
        scene = s;
        doc = d;
        s.rt = this;
        canvases.clear();
        uiPressed = null;
        uiHover = null;
        skipClick = false;
        uiDirty = true;
        objects.clear();
        scripts.clear();
        timers.clear();
        moved.clear();
        time.time = 0;
        for (GameObject go : s.objects()) {
            go.bind(this);
            objects.put(go.id(), go);
        }
        sceneScript = s.script() != null ? bind(s.script()) : null;
        if (sceneScript == null && d.sceneScript != null && d.sceneScript.script != null && !d.sceneScript.script.isBlank()) {
            Script script = instantiate(d.sceneScript.script, d.sceneScript.params, null, "scene \"" + d.name + "\"");
            if (script instanceof SceneScript ss) sceneScript = ss;
            else if (script != null) error("scene \"" + d.name + "\": " + d.sceneScript.script + " is not a scene script (extends SceneScript)", null);
        }
        if (sceneScript == null) {
            sceneScript = new SceneScript.Empty();
            sceneScript.rt = this;
        }
        List<Script> pending = new ArrayList<>();
        for (GameObject go : new ArrayList<>(objects.values())) pending.addAll(attach(go));
        for (Canvas c : s.canvases()) pending.addAll(show(c));
        for (Entity e : d.entities) {
            UiComponent uc = e.get(UiComponent.class);
            if (uc == null || uc.canvas == null || canvases.containsKey(uc.canvas)) continue;
            Canvas c = loadCanvas(uc.canvas, pending);
            if (c != null) c.doc.visible = uc.visible && c.doc.visible;
        }
        guard(sceneScript, "onLoad", sceneScript::onLoad);
        begin(sceneScript);
        for (Script script : pending) if (script.self == null || !script.self.destroyed) begin(script);
        structure = true;
        settings = true;
    }

    /** Advances one frame: input events, timers, {@code onUpdate}, then a pending scene switch. */
    public void update(float dt) {
        time.advance(dt);
        List<Input.Raw> events = input.beginFrame();
        uiHoverUpdate();
        for (Input.Raw r : events) dispatch(r);
        for (Timer t : new ArrayList<>(timers)) {
            if (t.script.failed || destroyed(t.script) || time.time < t.next[0]) continue;
            t.next[0] = time.time + t.interval;
            invoke(t.script, t.method, "@Every");
        }
        for (Script s : new ArrayList<>(scripts)) {
            if (s.failed || destroyed(s)) continue;
            guard(s, "onUpdate", () -> s.onUpdate(dt));
        }
        if (pendingScene != null) {
            String name = pendingScene;
            pendingScene = null;
            try {
                Scene next = host.loadScene(name);
                stopScripts();
                System.out.println("[game] scene -> " + name);
                enter(next);
            } catch (IOException ex) {
                error("loadScene(\"" + name + "\"): scene not found (" + ex.getMessage() + ")", null);
            }
        }
    }

    public void stop() {
        stopScripts();
        input.releaseAll();
        current = null;
        System.out.println("[game] stopped" + (errors > 0 ? " (" + errors + " script errors)" : ""));
    }

    private void stopScripts() {
        for (Script s : new ArrayList<>(scripts)) guard(s, "onStop", s::onStop);
        if (sceneScript != null && !scripts.contains(sceneScript)) guard(sceneScript, "onStop", sceneScript::onStop);
    }

    /** The scene document the scripts are working on. */
    public SceneDoc doc() { return doc; }

    public String sceneName() { return doc.name; }

    /** Ids of entities whose transform changed since the last call. */
    public Set<String> takeMoved() {
        Set<String> m = new LinkedHashSet<>(moved);
        moved.clear();
        return m;
    }

    /** Whether objects were added, removed or changed in a way that needs a scene rebuild since the last call. */
    public boolean takeStructure() {
        boolean s = structure;
        structure = false;
        return s;
    }

    /** Whether render settings changed since the last call. */
    public boolean takeSettings() {
        boolean s = settings;
        settings = false;
        return s;
    }

    /** Whether a script called {@code stopGame()}. */
    public boolean stopRequested() { return stopRequested; }

    /** Debug view selected by scripts; read by the host every frame. */
    public volatile int debugView;
    /** Statistics line provided by the host for {@link Render#stats()}. */
    public volatile String statsText = "";

    /** The player camera entity, or null. */
    public Entity cameraEntity() {
        for (GameObject o : objects.values()) {
            CameraComponent c = o.e.get(CameraComponent.class);
            if (c != null && c.player) return o.e;
        }
        return null;
    }

    public SceneScript sceneScript() { return sceneScript; }

    public void setPhysics(PhysicsBackend backend) { this.physicsBackend = backend; }

    /**
     * Reports a contact change between two entities after a physics step. Both sides are notified. Collisions only
     * fire on enter; triggers fire on enter and exit.
     */
    public void collision(String a, String b, boolean trigger, boolean enter) {
        GameObject ga = objects.get(a), gb = objects.get(b);
        if (ga == null || gb == null) return;
        collisionTo(ga, gb, trigger, enter);
        collisionTo(gb, ga, trigger, enter);
    }

    private void collisionTo(GameObject self, GameObject other, boolean trigger, boolean enter) {
        if (!trigger && !enter) return;
        for (Script s : new ArrayList<>(self.scripts)) {
            if (s.failed) continue;
            ScriptClass info = ScriptClass.of(s.getClass());
            if (trigger) {
                for (Method m : info.triggers) {
                    OnTrigger t = m.getAnnotation(OnTrigger.class);
                    if (t.exit() == enter || !(t.value().isEmpty() || other.hasTag(t.value()))) continue;
                    invoke(s, m, "@OnTrigger", other);
                }
            } else {
                for (Method m : info.collisions) {
                    OnCollision c = m.getAnnotation(OnCollision.class);
                    if (!(c.value().isEmpty() || other.hasTag(c.value()))) continue;
                    invoke(s, m, "@OnCollision", other);
                }
            }
        }
    }

    public List<Script> scripts() { return Collections.unmodifiableList(scripts); }

    /**
     * Binds the scripts attached to an object in code and instantiates those its entity names, including the ones
     * on the root entities of its prefab. The scripts are not started yet.
     */
    private List<Script> attach(GameObject go) {
        List<Script> out = new ArrayList<>();
        for (Script s : new ArrayList<>(go.scripts)) out.add(bind(s));
        for (SceneDoc.Component c : go.e.components) {
            if (c instanceof ScriptComponent sc && sc.script != null && !sc.script.isBlank()) {
                Script s = instantiate(sc.script, sc.params, go, "object \"" + go.name() + "\"");
                if (s != null) out.add(s);
            }
            if (c instanceof PrefabComponent pc && pc.prefab != null) {
                SceneDoc prefab;
                try { prefab = host.prefab(pc.prefab); } catch (RuntimeException ex) { continue; }
                for (Entity pe : prefab.entities) {
                    if (pe.parent != null) continue;
                    for (SceneDoc.Component pcc : pe.components) {
                        if (pcc instanceof ScriptComponent sc && sc.script != null && !sc.script.isBlank()) {
                            Script s = instantiate(sc.script, sc.params, go, "prefab " + pc.prefab);
                            if (s != null) out.add(s);
                        }
                    }
                }
            }
        }
        return out;
    }

    /**
     * Creates a script and injects its {@link Param} values. Returns null if the class is missing, of the wrong
     * kind or cannot be constructed; the reason is reported with {@code where} as context.
     */
    private Script instantiate(String className, Map<String, Object> params, GameObject go, String where) {
        Class<?> c = compiled.load(className);
        if (c == null) {
            error(where + ": script class \"" + className + "\" not found (does scripts/" + className.replace('.', '/')
                    + ".java exist and compile?)", null);
            return null;
        }
        if (!Script.class.isAssignableFrom(c)) { error(where + ": " + className + " is not a script", null); return null; }
        if (go != null && !ObjectScript.class.isAssignableFrom(c)) {
            error(where + ": " + className + " is not an object script (scene scripts belong in the scene settings)", null);
            return null;
        }
        ScriptClass info = ScriptClass.of(c);
        if (!info.instantiable) { error(where + ": " + className + " is abstract or has no empty constructor", null); return null; }
        Script s;
        try {
            Constructor<?> k = c.getDeclaredConstructor();
            k.setAccessible(true);
            s = (Script) k.newInstance();
        } catch (InvocationTargetException ex) {
            error(where + ": constructor of " + className + " - " + ex.getCause(), ex.getCause());
            return null;
        } catch (ReflectiveOperationException ex) {
            error(where + ": " + className + " - " + ex, null);
            return null;
        }
        s.rt = this;
        s.self = go;
        if (params != null) {
            for (Field f : info.params) {
                if (!params.containsKey(f.getName())) continue;
                try {
                    f.set(s, convert(params.get(f.getName()), f.getType(), true));
                } catch (IllegalArgumentException | IllegalAccessException ex) {
                    error(where + ": parameter " + c.getSimpleName() + "." + f.getName() + " (" + f.getType().getSimpleName() + "): "
                            + ex.getMessage(), null);
                }
            }
        }
        if (go != null) go.scripts.add(s);
        return s;
    }

    /** Connects a script that was created in code to this runtime. */
    <T extends Script> T bind(T script) {
        script.rt = this;
        return script;
    }

    /** Registers the script for updates, schedules its timers (first tick one interval from now) and starts it. */
    void begin(Script s) {
        if (!scripts.contains(s)) scripts.add(s);
        for (Method m : ScriptClass.of(s.getClass()).timers) {
            float iv = Math.max(0.001f, m.getAnnotation(Every.class).value());
            timers.add(new Timer(s, m, iv, new float[]{time.time + iv}));
        }
        guard(s, "onStart", s::onStart);
    }

    private static boolean destroyed(Script s) { return s.self != null && s.self.destroyed; }

    /**
     * Routes one input event. Mouse buttons go to the UI first; clicks the UI consumed, or any click while a modal
     * menu is open, never reach {@link OnMouse} handlers or the world. Object clicks bubble up the parent chain.
     */
    private void dispatch(Input.Raw r) {
        switch (r.type()) {
            case Input.KEY -> {
                for (Script s : new ArrayList<>(scripts)) {
                    if (s.failed || destroyed(s)) continue;
                    for (Method m : ScriptClass.of(s.getClass()).keys) {
                        OnKey k = m.getAnnotation(OnKey.class);
                        if (k.released() == r.down()) continue;
                        if (!k.value().equalsIgnoreCase(r.name()) && !k.value().equalsIgnoreCase("ANY")) continue;
                        invoke(s, m, "@OnKey(\"" + k.value() + "\")", r.name());
                    }
                }
            }
            case Input.BUTTON -> {
                if (uiButton(r)) return;
                for (Script s : new ArrayList<>(scripts)) {
                    if (s.failed || destroyed(s)) continue;
                    for (Method m : ScriptClass.of(s.getClass()).mice) {
                        OnMouse k = m.getAnnotation(OnMouse.class);
                        if (k.released() == r.down() || !k.value().equalsIgnoreCase(r.name())) continue;
                        invoke(s, m, "@OnMouse", r.x(), r.y());
                    }
                }
            }
            case Input.CLICK -> {
                // The pick result arrives after the button event; drop it if that button press went to the UI.
                if (skipClick || uiModalVisible()) { skipClick = false; return; }
                Set<String> seen = new HashSet<>();
                for (GameObject go = r.entity() == null ? null : objects.get(r.entity()); go != null && seen.add(go.id()); go = go.parent()) {
                    for (Script s : new ArrayList<>(go.scripts)) {
                        if (s.failed) continue;
                        for (Method m : ScriptClass.of(s.getClass()).clicks) invoke(s, m, "@OnClick");
                    }
                }
            }
            default -> { }
        }
    }

    /**
     * Invokes a handler, passing as many of {@code available} as it declares parameters. A handler whose parameter
     * types do not fit is reported and skipped.
     */
    private void invoke(Script s, Method m, String what, Object... available) {
        Object[] args = new Object[m.getParameterCount()];
        Class<?>[] types = m.getParameterTypes();
        for (int i = 0; i < args.length; i++) {
            Object v = i < available.length ? available[i] : null;
            try {
                args[i] = convert(v, types[i], false);
            } catch (IllegalArgumentException ex) {
                error(s.describe() + ": " + what + " " + ScriptClass.signature(m.getName(), m) + " - parameters do not fit ("
                        + ex.getMessage() + ")", null);
                return;
            }
        }
        guard(s, what + " " + m.getName(), () -> {
            try {
                m.invoke(s, args);
            } catch (InvocationTargetException ex) {
                throw new Wrapped(ex.getCause());
            } catch (IllegalAccessException ex) {
                throw new IllegalStateException(ex);
            }
        });
    }

    /**
     * Implements {@code call(...)} on scripts, objects and canvases. The first {@link Event} overload whose
     * parameters accept the arguments is invoked, once. Unknown events and argument mismatches are reported with
     * the caller's location. Returns the event's result, or null.
     */
    Object call(List<Script> targets, String targetName, String event, Object[] args) {
        if (args == null) args = new Object[0];
        String site = callSite();
        boolean found = false;
        Object result = null;
        boolean called = false;
        List<String> mismatches = new ArrayList<>();
        for (Script s : new ArrayList<>(targets)) {
            List<Method> ms = ScriptClass.of(s.getClass()).events.get(event);
            if (ms == null) continue;
            found = true;
            for (Method m : ms) {
                Object[] conv = tryConvert(m, args);
                if (conv == null) {
                    mismatches.add(s.getClass().getSimpleName() + "." + ScriptClass.signature(event, m));
                    continue;
                }
                Object[] r = new Object[1];
                boolean ok = guard(s, "event " + event, () -> {
                    try {
                        r[0] = m.invoke(s, conv);
                    } catch (InvocationTargetException ex) {
                        throw new Wrapped(ex.getCause());
                    } catch (IllegalAccessException ex) {
                        throw new IllegalStateException(ex);
                    }
                });
                if (ok && !called) result = r[0];
                called = true;
                break;
            }
        }
        if (!found) {
            List<String> have = new ArrayList<>();
            for (Script s : targets) have.addAll(ScriptClass.of(s.getClass()).eventSignatures());
            error(site + "call(\"" + event + "\"): " + targetName + " has no event \"" + event + "\""
                    + (have.isEmpty() ? " (no @Event methods at all)" : " - available: " + String.join(", ", have)), null);
        } else if (!called) {
            error(site + "call(\"" + event + "\"" + argList(args) + "): wrong arguments for " + String.join(" / ", mismatches)
                    + " - given: " + argTypes(args), null);
        }
        return result;
    }

    /** Converts the arguments to the method's parameter types, or returns null if they do not fit. */
    private Object[] tryConvert(Method m, Object[] args) {
        if (m.getParameterCount() != args.length) return null;
        Object[] out = new Object[args.length];
        Class<?>[] t = m.getParameterTypes();
        for (int i = 0; i < args.length; i++) {
            try {
                out[i] = convert(args[i], t[i], false);
            } catch (IllegalArgumentException ex) {
                return null;
            }
        }
        return out;
    }

    private static String argList(Object[] args) {
        StringBuilder b = new StringBuilder();
        for (Object a : args) b.append(", ").append(a instanceof String s ? "\"" + s + "\"" : String.valueOf(a));
        return b.toString();
    }

    private static String argTypes(Object[] args) {
        StringJoiner j = new StringJoiner(", ", "(", ")");
        for (Object a : args) j.add(a == null ? "null" : a.getClass().getSimpleName());
        return j.toString();
    }

    /**
     * Converts a value to type {@code t}. Numbers convert freely, but only whole numbers to integral types, and
     * strings convert to enum constants. Values read from a scene file ({@code fromScene}) additionally allow object
     * names for {@link GameObject}, 3-element lists for {@link Vec3} and strings for booleans.
     *
     * @throws IllegalArgumentException if the value does not fit
     */
    Object convert(Object v, Class<?> t, boolean fromScene) {
        Class<?> w = wrap(t);
        if (v == null) {
            if (t.isPrimitive()) throw new IllegalArgumentException("null for " + t.getSimpleName());
            return null;
        }
        if (w.isInstance(v)) return v;
        if (v instanceof Number n) {
            double d = n.doubleValue();
            if (w == Float.class) return (float) d;
            if (w == Double.class) return d;
            if (w == Integer.class || w == Long.class || w == Short.class || w == Byte.class) {
                if (d != Math.rint(d)) throw new IllegalArgumentException(v + " is not a whole number");
                if (w == Integer.class) return (int) d;
                if (w == Long.class) return (long) d;
                if (w == Short.class) return (short) d;
                return (byte) d;
            }
        }
        if (t.isEnum() && v instanceof String s) {
            for (Object c : t.getEnumConstants()) if (((Enum<?>) c).name().equalsIgnoreCase(s)) return c;
            throw new IllegalArgumentException("\"" + s + "\" does not exist in " + t.getSimpleName());
        }
        if (fromScene) {
            if (t == GameObject.class && v instanceof String s) return find(s, true);
            if (t == Vec3.class) {
                if (v instanceof float[] a && a.length == 3) return new Vec3(a[0], a[1], a[2]);
                if (v instanceof List<?> l && l.size() == 3 && l.stream().allMatch(x -> x instanceof Number)) {
                    return new Vec3(((Number) l.get(0)).floatValue(), ((Number) l.get(1)).floatValue(), ((Number) l.get(2)).floatValue());
                }
            }
            if (w == Boolean.class && v instanceof String s) return Boolean.parseBoolean(s);
            if (w == String.class) return String.valueOf(v);
        }
        throw new IllegalArgumentException(v.getClass().getSimpleName() + " instead of " + t.getSimpleName());
    }

    private static Class<?> wrap(Class<?> t) {
        if (!t.isPrimitive()) return t;
        if (t == float.class) return Float.class;
        if (t == double.class) return Double.class;
        if (t == int.class) return Integer.class;
        if (t == long.class) return Long.class;
        if (t == boolean.class) return Boolean.class;
        if (t == short.class) return Short.class;
        if (t == byte.class) return Byte.class;
        if (t == char.class) return Character.class;
        return t;
    }

    GameObject object(String id) { return objects.get(id); }

    GameObject find(String name, boolean warn) {
        for (GameObject o : objects.values()) if (o.name().equals(name)) return o;
        if (warn) warn(callSite() + "find(\"" + name + "\"): no object with this name in scene \"" + doc.name + "\"");
        return null;
    }

    List<GameObject> findAll(String name) {
        List<GameObject> out = new ArrayList<>();
        for (GameObject o : objects.values()) if (o.name().equals(name)) out.add(o);
        return out;
    }

    List<GameObject> findByTag(String tag) {
        List<GameObject> out = new ArrayList<>();
        for (GameObject o : objects.values()) if (o.hasTag(tag)) out.add(o);
        return out;
    }

    List<GameObject> objects() { return new ArrayList<>(objects.values()); }

    GameObject camera() {
        Entity e = cameraEntity();
        return e == null ? null : objects.get(e.id);
    }

    void moved(GameObject go) { if (!go.destroyed) moved.add(go.e.id); }

    private final Set<String> teleported = new LinkedHashSet<>();

    void teleported(GameObject go) {
        if (go.destroyed) return;
        moved.add(go.e.id);
        teleported.add(go.e.id);
    }

    /** Ids of entities teleported since the last call; physics places them without sweeping. */
    public Set<String> takeTeleported() {
        Set<String> t = new LinkedHashSet<>(teleported);
        teleported.clear();
        return t;
    }

    void structureChanged() { structure = true; }

    void settingsChanged() { settings = true; }

    void requestScene(String name) { pendingScene = name; }

    void requestScene(Class<? extends Scene> type) {
        String name = host.sceneName(type);
        if (name == null) {
            error(callSite() + "loadScene(" + type.getSimpleName() + ".class): the game has no such scene, see Game.addScene", null);
            return;
        }
        pendingScene = name;
    }

    void requestStop() { stopRequested = true; }

    GameObject spawn(String asset, Vec3 at) {
        String a = asset == null ? "" : asset.trim();
        String base = a.contains("/") ? a.substring(a.lastIndexOf('/') + 1) : a;
        Entity e = new Entity(base.replace(".glb", "").replace(".prefab.json", ""));
        if (a.startsWith("models/")) e.with(new ModelComponent(a));
        else if (a.startsWith("prefabs/")) e.with(new PrefabComponent(a));
        else if (Set.of("box", "cube", "sphere", "plane", "torus", "cylinder").contains(a)) {
            MaterialDef m = new MaterialDef();
            m.roughness = 0.6f;
            e.with(new PrimitiveComponent(a, a.equals("plane") ? new float[]{1} : a.equals("torus") ? new float[]{0.5f, 0.15f}
                    : new float[]{1, 1, 1}, null, m));
        } else {
            error(callSite() + "spawn(\"" + asset + "\"): unknown - allowed: models/....glb, prefabs/....prefab.json, box, cube, sphere, plane, torus", null);
            return null;
        }
        if (at != null) e.position = at.toArray();
        return add(e);
    }

    GameObject copy(GameObject src) {
        Entity c;
        try {
            // A JSON round trip is the simplest deep copy of an entity with all of its components.
            c = SceneDoc.JSON.readValue(SceneDoc.JSON.writeValueAsString(src.e), Entity.class);
        } catch (IOException ex) {
            throw new IllegalStateException(ex);
        }
        c.id = UUID.randomUUID().toString();
        return add(c);
    }

    private GameObject add(Entity e) { return add(new GameObject(e)); }

    /** Adds an object and its children to the running scene and starts their scripts. */
    <T extends GameObject> T add(T root) {
        List<GameObject> all = root.subtree();
        for (GameObject go : all) {
            doc.entities.add(go.e);
            go.bind(this);
            objects.put(go.id(), go);
        }
        for (GameObject go : all) for (Script s : attach(go)) begin(s);
        structure = true;
        return root;
    }

    void destroy(GameObject go) {
        if (go.destroyed) return;
        List<GameObject> all = new ArrayList<>();
        collect(go, all);
        for (GameObject o : all) {
            o.destroyed = true;
            for (Script s : o.scripts) guard(s, "onStop", s::onStop);
            scripts.removeAll(o.scripts);
            timers.removeIf(t -> o.scripts.contains(t.script));
            objects.remove(o.id());
            doc.entities.remove(o.e);
            moved.remove(o.id());
        }
        structure = true;
    }

    private void collect(GameObject go, List<GameObject> out) {
        out.add(go);
        for (GameObject c : go.children()) if (!out.contains(c)) collect(c, out);
    }

    private final Map<String, Canvas> canvases = new LinkedHashMap<>();
    private volatile boolean uiDirty = true;
    private Canvas uiPressedCanvas, uiHoverCanvas;
    private UiDoc.Element uiPressed, uiHover;
    private boolean skipClick;
    private volatile int screenW = 1920, screenH = 1080;

    void uiChanged() { uiDirty = true; }

    /** Looks up a canvas by name or asset id, loading it (and starting its scripts) on first use. */
    Canvas ui(String name) {
        for (Canvas c : canvases.values()) if (c.doc.name.equals(name) || c.assetId.equals(name)) return c;
        String id = name.contains("/") ? name : UiDoc.assetId(name);
        List<Script> pending = new ArrayList<>();
        Canvas c = loadCanvas(id, pending);
        if (c == null) {
            warn(callSite() + "ui(\"" + name + "\"): no UI canvas with this name (" + id + ")");
            return null;
        }
        for (Script s : pending) begin(s);
        return c;
    }

    /** The canvas of the given class in the running scene, or null. */
    <T extends Canvas> T ui(Class<T> type) {
        for (Canvas c : canvases.values()) if (type.isInstance(c)) return type.cast(c);
        warn(callSite() + "ui(" + type.getSimpleName() + ".class): the scene has no such canvas");
        return null;
    }

    /**
     * Loads a private copy of a canvas asset and shows it. Its scripts are added to {@code pending} for the caller
     * to start.
     */
    private Canvas loadCanvas(String assetId, List<Script> pending) {
        UiDoc d;
        try {
            d = host.ui(assetId);
        } catch (RuntimeException ex) {
            d = null;
        }
        if (d == null) {
            error("UI canvas " + assetId + " not found", null);
            return null;
        }
        Canvas c = new Canvas(d.copy(), assetId);
        pending.addAll(show(c));
        return c;
    }

    /**
     * Shows a canvas in the running scene: binds the scripts attached in code and instantiates those its document
     * names. Returns the scripts, which are not started yet.
     */
    private List<Script> show(Canvas c) {
        c.rt = this;
        canvases.put(c.assetId, c);
        List<Script> pending = new ArrayList<>();
        for (UiScript s : new ArrayList<>(c.scripts)) pending.add(bind(s));
        for (ScriptComponent sc : c.doc.scripts) {
            UiScript s = instantiateUi(sc, c, null);
            if (s != null) {
                c.scripts.add(s);
                pending.add(s);
            }
        }
        for (UiElement el : c.elements()) {
            for (UiScript s : new ArrayList<>(el.scripts)) {
                s.canvas = c;
                pending.add(bind(s));
            }
            for (ScriptComponent sc : el.e.scripts) {
                UiScript s = instantiateUi(sc, c, el);
                if (s != null) {
                    el.scripts.add(s);
                    pending.add(s);
                }
            }
        }
        uiDirty = true;
        return pending;
    }

    private UiScript instantiateUi(ScriptComponent sc, Canvas c, UiElement el) {
        if (sc.script == null || sc.script.isBlank()) return null;
        String where = "UI \"" + c.doc.name + "\"" + (el != null ? " / " + el.name() : "");
        Script s = instantiate(sc.script, sc.params, null, where);
        if (s == null) return null;
        if (!(s instanceof UiScript u)) {
            error(where + ": " + sc.script + " is not a UI script (extends UiScript)", null);
            return null;
        }
        u.canvas = c;
        u.element = el;
        return u;
    }

    /**
     * Delivers a click to the scripts of the element, then of its ancestors, then of the canvas, and finally to the
     * element's own click listeners.
     */
    void uiClick(Canvas c, UiElement clicked) {
        Set<UiElement> seen = new HashSet<>();
        for (UiElement el = clicked; el != null && seen.add(el); el = el.parent()) {
            for (Script s : new ArrayList<>(el.scripts)) {
                if (s.failed) continue;
                for (Method m : ScriptClass.of(s.getClass()).clicks) invoke(s, m, "@OnClick", clicked);
            }
        }
        for (UiScript s : new ArrayList<>(c.scripts)) {
            if (s.failed) continue;
            for (Method m : ScriptClass.of(s.getClass()).clicks) invoke(s, m, "@OnClick", clicked);
        }
        try {
            clicked.clicked();
        } catch (RuntimeException ex) {
            error("click listener of " + clicked + ": " + ex, ex);
        }
    }

    private boolean uiModalVisible() {
        for (Canvas c : canvases.values()) if (c.doc.visible && c.doc.modal) return true;
        return false;
    }

    /** Topmost element under a normalized view position, as {canvas, element}, or null. */
    private Object[] uiHit(float x, float y) {
        List<Canvas> l = new ArrayList<>(canvases.values());
        l.sort(Comparator.comparingInt((Canvas c) -> c.doc.order).reversed());
        for (Canvas c : l) {
            if (!c.doc.visible) continue;
            UiDoc.Element e = UiPainter.hit(c.doc, x * screenW, y * screenH, screenW, screenH);
            if (e != null) return new Object[]{c, e};
        }
        return null;
    }

    /**
     * Handles a mouse button for the UI and returns whether the UI consumed it. A click completes when the left
     * button is released over the element it was pressed on. While a modal menu is open, all buttons are consumed.
     */
    private boolean uiButton(Input.Raw r) {
        if (!"LEFT".equalsIgnoreCase(r.name())) return uiModalVisible();
        if (r.down()) {
            Object[] h = uiHit(r.x(), r.y());
            if (h != null) {
                uiPressedCanvas = (Canvas) h[0];
                uiPressed = (UiDoc.Element) h[1];
                uiDirty = true;
                skipClick = true;
                return true;
            }
            if (uiModalVisible()) { skipClick = true; return true; }
            return false;
        }
        if (uiPressed != null) {
            Object[] h = uiHit(r.x(), r.y());
            Canvas c = uiPressedCanvas;
            UiDoc.Element e = uiPressed;
            uiPressed = null;
            uiPressedCanvas = null;
            uiDirty = true;
            if (h != null && h[1] == e) uiClick(c, c.handle(e));
            return true;
        }
        return uiModalVisible();
    }

    private void uiHoverUpdate() {
        input.uiModal = uiModalVisible();
        Object[] h = canvases.isEmpty() ? null : uiHit(input.mouseX(), input.mouseY());
        UiDoc.Element e = h == null ? null : (UiDoc.Element) h[1];
        if (e != uiHover) {
            uiHover = e;
            uiHoverCanvas = h == null ? null : (Canvas) h[0];
            uiDirty = true;
        }
    }

    /** Size of the output image in pixels; the UI is laid out for it. */
    public void setScreen(int w, int h) {
        if (w == screenW && h == screenH) return;
        screenW = w;
        screenH = h;
        uiDirty = true;
    }

    /** Whether the UI needs to be redrawn; clears the flag. */
    public boolean takeUiDirty() {
        boolean d = uiDirty;
        uiDirty = false;
        return d;
    }

    /**
     * Draws all visible canvases into a premultiplied ARGB buffer of {@code w} x {@code h}, or returns null if there
     * is nothing to draw. {@code assets} resolves image asset ids to files.
     */
    public int[] renderUi(int w, int h, Function<String, Path> assets) {
        List<UiDoc> docs = new ArrayList<>();
        Map<UiDoc, UiPainter.State> states = new HashMap<>();
        for (Canvas c : canvases.values()) {
            docs.add(c.doc);
            String hover = c == uiHoverCanvas && uiHover != null ? uiHover.id : null;
            String pressed = c == uiPressedCanvas && uiPressed != null ? uiPressed.id : null;
            states.put(c.doc, new UiPainter.State(hover, pressed));
        }
        return UiPainter.render(docs, states, w, h, assets);
    }

    /** Carries an exception thrown by a script through a reflective call. */
    private static final class Wrapped extends RuntimeException {
        Wrapped(Throwable cause) { super(cause); }
    }

    /**
     * Runs script code and reports anything it throws, with the script's file and line. A failure in
     * {@code onUpdate} disables the script so the error is not repeated every frame. Returns false on failure.
     */
    private boolean guard(Script s, String what, Runnable r) {
        try {
            r.run();
            return true;
        } catch (Throwable t) {
            Throwable cause = t instanceof Wrapped ? t.getCause() : t;
            boolean update = what.equals("onUpdate");
            if (update) s.failed = true;
            error(s.describe() + "." + what + ": " + cause + (update ? "  -> script paused" : ""), cause);
            return false;
        }
    }

    private void error(String msg, Throwable t) {
        String where = t == null ? "" : location(t.getStackTrace());
        String text = "[script error] " + (where.isEmpty() ? "" : where + ": ") + msg;
        lastError = text;
        errors++;
        System.out.println(text);
    }

    private void warn(String msg) { System.out.println("[script] warning: " + msg); }

    /** Location of the script line currently calling into the engine, e.g. {@code "scripts/Door.java:23: "}. */
    private String callSite() {
        String l = location(new Throwable().getStackTrace());
        return l.isEmpty() ? "" : l + ": ";
    }

    // First stack frame that belongs to a project script, formatted as scripts/<package path>/<File>.java:<line>.
    private String location(StackTraceElement[] st) {
        for (StackTraceElement el : st) {
            if (!compiled.isScriptClass(el.getClassName()) || el.getFileName() == null) continue;
            String pkg = el.getClassName().contains(".") ? el.getClassName().substring(0, el.getClassName().lastIndexOf('.')).replace('.', '/') + "/" : "";
            return "scripts/" + pkg + el.getFileName() + ":" + el.getLineNumber();
        }
        return "";
    }
}
