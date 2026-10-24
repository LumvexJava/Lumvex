package dev.lumvex.engine;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/**
 * A project folder: {@code project.json} with the settings below, plus {@code assets/}, {@code scenes/} and
 * {@code scripts/}. Asset ids used in scenes are paths relative to {@code assets/}.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
public final class Project {
    public String name = "Project";
    public String engineVersion = "1.0";
    /** Scene that is opened and played first, without the {@code .scene.json} suffix. */
    public String startScene;
    /** Upper bound for the resolution of texture sets; larger images are downscaled on load. */
    public int materialTextureSize = 8192;
    /** Upper bound for textures embedded in models. */
    public int modelTextureSize = 2048;
    /** Each virtual texture pool holds {@code textureSlotsPerRow^2} tiles. */
    public int textureSlotsPerRow = 40;
    /** Window title of an exported game; null uses the project name. */
    public String gameTitle;
    public boolean gameFullscreen = true;

    @JsonIgnore public Path root;

    public static final String FILE = "project.json";

    /** Opens an existing project. Fails if the folder contains no {@code project.json}. */
    public static Project open(Path root) throws IOException {
        Path f = root.resolve(FILE);
        if (!Files.isRegularFile(f)) throw new IOException("no " + FILE + " in " + root.toAbsolutePath());
        Project p = SceneDoc.JSON.readValue(f.toFile(), Project.class);
        p.root = root.toAbsolutePath().normalize();
        return p;
    }

    /** Creates the folder layout and {@code project.json} of a new project. */
    public static Project create(Path root, String name) throws IOException {
        Project p = new Project();
        p.name = name;
        p.root = root.toAbsolutePath().normalize();
        for (String d : new String[]{"assets", "scenes", "scripts"}) Files.createDirectories(p.root.resolve(d));
        p.save();
        return p;
    }

    /**
     * A project that exists only in memory, for games that build their scenes in code. Assets are read from
     * {@code root/assets}; nothing is written.
     */
    public static Project inMemory(Path root, String name) {
        Project p = new Project();
        p.name = name != null ? name : "Game";
        p.root = root.toAbsolutePath().normalize();
        return p;
    }

    public void save() throws IOException {
        SceneDoc.JSON.writeValue(root.resolve(FILE).toFile(), this);
    }

    @JsonIgnore public Path assets() { return root.resolve("assets"); }

    @JsonIgnore public Path scenesDir() { return root.resolve("scenes"); }

    /** Resolves an asset id to its file. */
    public Path asset(String id) { return assets().resolve(id); }

    @JsonIgnore public Path scriptsDir() { return root.resolve("scripts"); }

    /** Resolves an id that may also point into {@code scripts/}. */
    public Path file(String id) { return id.startsWith("scripts/") ? root.resolve(id) : asset(id); }

    public Path sceneFile(String sceneName) { return scenesDir().resolve(sceneName + ".scene.json"); }

    /** Names of all scenes in the project, sorted. */
    @JsonIgnore public List<String> sceneNames() throws IOException {
        List<String> out = new ArrayList<>();
        if (!Files.isDirectory(scenesDir())) return out;
        try (Stream<Path> s = Files.list(scenesDir())) {
            s.map(p -> p.getFileName().toString()).filter(n -> n.endsWith(".scene.json"))
                    .map(n -> n.substring(0, n.length() - ".scene.json".length())).sorted().forEach(out::add);
        }
        return out;
    }
}
