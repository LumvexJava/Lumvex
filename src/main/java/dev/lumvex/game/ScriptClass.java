package dev.lumvex.game;

import java.lang.reflect.*;
import java.util.*;

/**
 * Reflection metadata of a script class: its kind, {@link Param} fields, {@link Event} methods and annotated
 * handlers. Inherited members are included, with overrides taking precedence. Instances are cached per class.
 */
public final class ScriptClass {
    public enum Kind { SCENE, OBJECT, UI, OTHER }

    public final Class<?> type;
    public final Kind kind;
    public final boolean instantiable;
    final List<Field> params = new ArrayList<>();
    final Map<String, List<Method>> events = new LinkedHashMap<>();
    final List<Method> keys = new ArrayList<>(), mice = new ArrayList<>(), clicks = new ArrayList<>(), timers = new ArrayList<>(),
            collisions = new ArrayList<>(), triggers = new ArrayList<>();

    private static final ClassValue<ScriptClass> CACHE = new ClassValue<>() {
        @Override protected ScriptClass computeValue(Class<?> type) { return new ScriptClass(type); }
    };

    public static ScriptClass of(Class<?> type) { return CACHE.get(type); }

    private ScriptClass(Class<?> type) {
        this.type = type;
        this.kind = SceneScript.class.isAssignableFrom(type) ? Kind.SCENE : ObjectScript.class.isAssignableFrom(type) ? Kind.OBJECT
                : UiScript.class.isAssignableFrom(type) ? Kind.UI : Kind.OTHER;
        boolean ctor;
        try {
            type.getDeclaredConstructor();
            ctor = true;
        } catch (NoSuchMethodException e) {
            ctor = false;
        }
        this.instantiable = ctor && !Modifier.isAbstract(type.getModifiers()) && !type.isInterface() && !type.isAnonymousClass()
                && !type.isLocalClass() && (type.getEnclosingClass() == null || Modifier.isStatic(type.getModifiers()));
        Set<String> seenMethods = new HashSet<>(), seenFields = new HashSet<>();
        List<Class<?>> chain = new ArrayList<>();
        for (Class<?> c = type; c != null && c != Script.class && c != Object.class; c = c.getSuperclass()) chain.add(c);
        // Superclass parameters first, in declaration order.
        for (int i = chain.size() - 1; i >= 0; i--) {
            for (Field f : chain.get(i).getDeclaredFields()) {
                if (!f.isAnnotationPresent(Param.class) || Modifier.isStatic(f.getModifiers()) || !seenFields.add(f.getName())) continue;
                f.setAccessible(true);
                params.add(f);
            }
        }
        // Subclass methods first, so an override hides the inherited method with the same signature.
        for (Class<?> c : chain) {
            for (Method m : c.getDeclaredMethods()) {
                if (m.isSynthetic() || m.isBridge()) continue;
                if (!seenMethods.add(m.getName() + Arrays.toString(m.getParameterTypes()))) continue;
                boolean any = false;
                if (m.isAnnotationPresent(Event.class)) {
                    String n = m.getAnnotation(Event.class).value();
                    events.computeIfAbsent(n.isEmpty() ? m.getName() : n, k -> new ArrayList<>()).add(m);
                    any = true;
                }
                if (m.isAnnotationPresent(OnKey.class)) { keys.add(m); any = true; }
                if (m.isAnnotationPresent(OnMouse.class)) { mice.add(m); any = true; }
                if (m.isAnnotationPresent(OnClick.class)) { clicks.add(m); any = true; }
                if (m.isAnnotationPresent(Every.class)) { timers.add(m); any = true; }
                if (m.isAnnotationPresent(OnCollision.class)) { collisions.add(m); any = true; }
                if (m.isAnnotationPresent(OnTrigger.class)) { triggers.add(m); any = true; }
                if (any) m.setAccessible(true);
            }
        }
    }

    /** Fully qualified class name, e.g. {@code doors.Door}. */
    public String name() { return type.getName(); }

    public String simpleName() { return type.getSimpleName(); }

    /** Event signatures such as {@code open(float angle)}. */
    public List<String> eventSignatures() {
        List<String> out = new ArrayList<>();
        for (Map.Entry<String, List<Method>> e : events.entrySet()) for (Method m : e.getValue()) out.add(signature(e.getKey(), m));
        return out;
    }

    // Parameter names are only available if the class was compiled with -parameters.
    static String signature(String name, Method m) {
        StringJoiner j = new StringJoiner(", ", name + "(", ")");
        for (Parameter p : m.getParameters()) j.add(p.getType().getSimpleName() + (p.isNamePresent() ? " " + p.getName() : ""));
        return j.toString();
    }

    @Override public String toString() { return simpleName(); }
}
