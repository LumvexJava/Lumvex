package dev.lumvex.engine;

import dev.lumvex.render.scene.Environment;
import dev.lumvex.render.scene.GltfLoader;
import dev.lumvex.texture.PbrImages;
import dev.lumvex.texture.TextureSet;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.*;

/**
 * CPU-side cache of decoded assets. Each model, texture set and material is loaded once and shared between scene
 * builds, so switching scenes or restarting play mode never decodes the same file twice.
 *
 * <p>Loads run on a small thread pool; the {@code prefetch} methods start them early so independent images decode
 * in parallel, and the getters block only on the asset they return.
 */
public final class AssetCache implements AutoCloseable {
    private final Project project;
    private final ExecutorService pool = Executors.newFixedThreadPool(
            Math.clamp(Runtime.getRuntime().availableProcessors() / 2, 2, 8), r -> {
                Thread t = new Thread(r, "asset-loader");
                t.setDaemon(true);
                return t;
            });
    private final Map<String, Future<GltfLoader.Model>> models = new ConcurrentHashMap<>();
    private final Map<String, Future<TextureSet>> textures = new ConcurrentHashMap<>();
    private final Map<String, Future<SceneDoc.MaterialDef>> materials = new ConcurrentHashMap<>();
    private final Map<String, SceneDoc> prefabs = new ConcurrentHashMap<>();

    public AssetCache(Project project) { this.project = project; }

    public Project project() { return project; }

    public void prefetchModel(String id) {
        models.computeIfAbsent(id, k -> pool.submit(() -> GltfLoader.load(project.asset(k), project.modelTextureSize, 0)));
    }

    public void prefetchTextures(String id) {
        textures.computeIfAbsent(id, k -> pool.submit(() -> PbrImages.loadFolder(project.asset(k), project.materialTextureSize)));
    }

    public void prefetchMaterial(String id) {
        materials.computeIfAbsent(id, k -> pool.submit(() -> SceneDoc.JSON.readValue(project.asset(k).toFile(), SceneDoc.MaterialDef.class)));
    }

    public GltfLoader.Model model(String id) { prefetchModel(id); return get(models.get(id), id); }

    public TextureSet textures(String id) { prefetchTextures(id); return get(textures.get(id), id); }

    public SceneDoc.MaterialDef material(String id) { prefetchMaterial(id); return get(materials.get(id), id); }

    /** Returns a prefab document, cached until {@link #invalidate}. */
    public SceneDoc prefab(String id) {
        return prefabs.computeIfAbsent(id, k -> {
            try {
                return SceneDoc.load(project.asset(k));
            } catch (IOException e) {
                throw new IllegalStateException("prefab " + k + ": " + e.getMessage(), e);
            }
        });
    }

    /**
     * Loads an HDR environment. Not cached: sun extraction modifies the panorama, and reloading a 4K .hdr only takes
     * about 0.1 s.
     */
    public Environment environment(String id) throws IOException { return Environment.load(project.asset(id)); }

    private static <T> T get(Future<T> f, String id) {
        try {
            return f.get();
        } catch (ExecutionException e) {
            throw new IllegalStateException("asset " + id + ": " + e.getCause().getMessage(), e.getCause());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    @Override
    public void close() { pool.shutdownNow(); }
}
