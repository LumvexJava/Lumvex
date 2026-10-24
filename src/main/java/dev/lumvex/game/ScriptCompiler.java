package dev.lumvex.game;

import javax.tools.*;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.StringWriter;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Stream;

/**
 * Provides the script classes of a game: either compiled in memory from a project's {@code scripts/} folder with the
 * JDK compiler, or taken from classes that are already on the application's class path
 * ({@link #fromClassLoader}).
 */
public final class ScriptCompiler {
    private ScriptCompiler() { }

    /** A compiler diagnostic. {@code file} is relative to the project folder. */
    public record Problem(String file, long line, long column, String message, boolean error) {
        @Override public String toString() {
            return file + ":" + line + ": " + (error ? "error" : "warning") + ": " + message;
        }
    }

    /** Outcome of a compile: diagnostics, the script classes found and the loader that holds them. */
    public static final class Result {
        public final boolean ok;
        public final List<Problem> problems;
        /** Instantiable scene, object and UI scripts, sorted by name. */
        public final List<ScriptClass> scripts;
        public final double seconds;
        private final ClassLoader loader;
        private final Set<String> names;

        Result(boolean ok, List<Problem> problems, List<ScriptClass> scripts, double seconds, ClassLoader loader, Set<String> names) {
            this.ok = ok;
            this.problems = problems;
            this.scripts = scripts;
            this.seconds = seconds;
            this.loader = loader;
            this.names = names;
        }

        /** Loads a class through this result's loader; null if it does not exist or nothing was loaded. */
        public Class<?> load(String className) {
            if (loader == null || className == null) return null;
            try {
                return Class.forName(className, true, loader);
            } catch (Throwable t) {
                return null;
            }
        }

        /** Whether the class belongs to the scripts; used to map stack frames to script files. */
        public boolean isScriptClass(String binaryName) {
            int d = binaryName.indexOf('$');
            return names.contains(d < 0 ? binaryName : binaryName.substring(0, d)) || names.contains(binaryName);
        }
    }

    static List<Path> sources(Path dir) {
        List<Path> out = new ArrayList<>();
        if (!Files.isDirectory(dir)) return out;
        try (Stream<Path> s = Files.walk(dir)) {
            s.filter(p -> p.toString().endsWith(".java") && Files.isRegularFile(p)).sorted().forEach(out::add);
        } catch (IOException ignored) {
        }
        return out;
    }

    /** Compiles all scripts in the folder and loads them. Requires a JDK. */
    public static Result compile(Path dir) {
        long t0 = System.nanoTime();
        List<Path> files = sources(dir);
        List<Problem> problems = new ArrayList<>();
        if (files.isEmpty()) return new Result(true, problems, List.of(), 0, null, Set.of());
        JavaCompiler javac = ToolProvider.getSystemJavaCompiler();
        if (javac == null) {
            problems.add(new Problem("-", 0, 0, "no Java compiler found: the engine must run on a JDK (not a JRE)", true));
            return new Result(false, problems, List.of(), 0, null, Set.of());
        }
        DiagnosticCollector<JavaFileObject> diags = new DiagnosticCollector<>();
        Map<String, byte[]> classes = new LinkedHashMap<>();
        try (StandardJavaFileManager std = javac.getStandardFileManager(diags, Locale.ENGLISH, StandardCharsets.UTF_8)) {
            // Class files go to memory instead of disk.
            JavaFileManager fm = new ForwardingJavaFileManager<>(std) {
                @Override
                public JavaFileObject getJavaFileForOutput(Location location, String className, JavaFileObject.Kind kind, FileObject sibling) {
                    return new SimpleJavaFileObject(URI.create("mem:///" + className.replace('.', '/') + kind.extension), kind) {
                        @Override public OutputStream openOutputStream() {
                            return new ByteArrayOutputStream() {
                                @Override public void close() { classes.put(className, toByteArray()); }
                            };
                        }
                    };
                }
            };
            List<String> options = List.of("-classpath", System.getProperty("java.class.path"), "-parameters", "-g",
                    "-encoding", "UTF-8", "-proc:none", "-Xlint:-options", "-implicit:none");
            List<JavaFileObject> units = new ArrayList<>();
            for (JavaFileObject o : std.getJavaFileObjectsFromPaths(files)) units.add(o);
            boolean ok = javac.getTask(new StringWriter(), fm, diags, options, null, units).call();
            Path root = dir.getParent() != null ? dir.getParent() : dir;
            for (Diagnostic<? extends JavaFileObject> d : diags.getDiagnostics()) {
                if (d.getKind() == Diagnostic.Kind.NOTE) continue;
                String file = d.getSource() == null ? "-" : root.relativize(Path.of(d.getSource().toUri())).toString().replace('\\', '/');
                problems.add(new Problem(file, d.getLineNumber(), d.getColumnNumber(), d.getMessage(Locale.ENGLISH),
                        d.getKind() == Diagnostic.Kind.ERROR));
            }
            if (!ok) return new Result(false, problems, List.of(), (System.nanoTime() - t0) / 1e9, null, Set.of());
        } catch (IOException | RuntimeException ex) {
            problems.add(new Problem("-", 0, 0, "compiler: " + ex, true));
            return new Result(false, problems, List.of(), 0, null, Set.of());
        }
        return define(classes, problems, (System.nanoTime() - t0) / 1e9);
    }

    /**
     * Uses classes that are already compiled, typically those of the application embedding the engine. Scenes
     * resolve their script classes by name through {@code loader}; {@code classes} only lists the known scripts and
     * may be empty.
     */
    public static Result fromClassLoader(ClassLoader loader, Collection<Class<?>> classes) {
        List<ScriptClass> scripts = new ArrayList<>();
        Set<String> names = new LinkedHashSet<>();
        for (Class<?> c : classes) {
            if (!Script.class.isAssignableFrom(c)) throw new IllegalArgumentException(c.getName() + " is not a script");
            ScriptClass sc = ScriptClass.of(c);
            if (sc.instantiable && sc.kind != ScriptClass.Kind.OTHER) scripts.add(sc);
            names.add(c.getName());
        }
        scripts.sort(Comparator.comparing(ScriptClass::name));
        return new Result(true, new ArrayList<>(), scripts, 0, loader, names);
    }

    // Defines the classes in a child loader of the engine's loader. Classes that fail to link become problems.
    private static Result define(Map<String, byte[]> classes, List<Problem> problems, double seconds) {
        ClassLoader loader = new ClassLoader("scripts", ScriptCompiler.class.getClassLoader()) {
            @Override protected Class<?> findClass(String name) throws ClassNotFoundException {
                byte[] b = classes.get(name);
                if (b == null) throw new ClassNotFoundException(name);
                return defineClass(name, b, 0, b.length);
            }
        };
        List<ScriptClass> scripts = new ArrayList<>();
        for (String name : classes.keySet()) {
            try {
                Class<?> c = Class.forName(name, false, loader);
                if (!Script.class.isAssignableFrom(c)) continue;
                ScriptClass sc = ScriptClass.of(c);
                if (sc.instantiable && sc.kind != ScriptClass.Kind.OTHER) scripts.add(sc);
            } catch (Throwable t) {
                problems.add(new Problem(name, 0, 0, "cannot be loaded: " + t, true));
            }
        }
        scripts.sort(Comparator.comparing(ScriptClass::name));
        return new Result(true, problems, scripts, seconds, loader, classes.keySet());
    }
}
