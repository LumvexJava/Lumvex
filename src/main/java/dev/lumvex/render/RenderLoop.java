package dev.lumvex.render;

import dev.lumvex.engine.AssetCache;
import dev.lumvex.engine.SceneBuilder;
import dev.lumvex.engine.SceneDoc;
import dev.lumvex.engine.SceneQuery;
import dev.lumvex.engine.physics.PhysicsWorld;
import dev.lumvex.render.scene.Camera;
import dev.lumvex.render.scene.Scene;
import dev.lumvex.game.ScriptCompiler;

import dev.lumvex.game.ScriptRuntime;
import dev.lumvex.vk.VulkanCanvas;
import org.joml.Matrix4f;
import org.joml.Vector3f;

import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.function.BiConsumer;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * The render thread. It owns the {@link Renderer}, the render scene and its {@link SceneBuilder}, and runs the
 * scripts and the physics of the game, so none of them ever race.
 *
 * <p>Other threads hand over requests ({@link #play}, {@link #pick}, {@link #query}, ...), which are applied
 * between frames.
 */
public final class RenderLoop implements Runnable {
    private final VulkanCanvas canvas;
    private final AssetCache assets;
    private final ConcurrentLinkedQueue<Runnable> tasks = new ConcurrentLinkedQueue<>();
    private final Thread thread = new Thread(this, "render");
    private volatile boolean running = true;

    private final Scene scene = new Scene();
    private SceneBuilder builder;
    private SceneDoc renderDoc;
    private Renderer renderer;
    private final Camera camera = new Camera();

    /** Fly camera controls for scenes without a player camera, written by the UI thread. */
    public final Input input = new Input();
    /** Script input, written by the UI thread. */
    public final dev.lumvex.game.Input gameInput = new dev.lumvex.game.Input();
    public volatile String status = "Loading scene ...";
    public volatile boolean ready;
    public volatile boolean playing, paused;
    /** Number of scene rebuilds while playing. */
    public volatile int rebuilds;
    /** Smoothed physics time per frame in ms. */
    public volatile double physicsMs;

    private Consumer<String> onError = e -> { };
    private Runnable onReady = () -> { };
    private Runnable onPlayStopped = () -> { };
    private ScriptRuntime runtime;
    private PhysicsWorld physics;
    private String stats = "";

    /** Fly camera controls. */
    public static final class Input {
        public volatile boolean forward, back, left, right, up, down, fast, look;
        private float dx, dy, wheel;

        public synchronized void addLook(float x, float y) { dx += x; dy += y; }

        public synchronized void addWheel(float w) { wheel += w; }

        /** Returns and resets the accumulated {dx, dy, wheel}. */
        synchronized float[] consume() {
            float[] r = {dx, dy, wheel};
            dx = dy = wheel = 0;
            return r;
        }
    }

    public RenderLoop(VulkanCanvas canvas, AssetCache assets) {
        this.canvas = canvas;
        this.assets = assets;
        thread.setDaemon(true);
    }

    /** Starts the render thread with the given scene. Exactly one of {@code onReady} and {@code onError} is called. */
    public void start(SceneDoc initial, Runnable onReady, Consumer<String> onError) {
        this.renderDoc = initial;
        this.onReady = onReady;
        this.onError = onError;
        thread.start();
    }

    /** Stops the render thread, waiting up to 5 s. */
    public void stop() {
        running = false;
        try { thread.join(5000); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    /** Reports the entity (or null) and view depth at an image pixel. The callback runs on the render thread. */
    public void pick(int x, int y, BiConsumer<String, Float> callback) {
        tasks.add(() -> renderer.requestPick(x, y, r -> callback.accept(builder.entityOfInstance((int) r[0] - 1), r[1])));
    }

    /**
     * Starts the scripts and the physics on {@code gameScene}. {@code onStopped} runs on the render thread when the game
     * ends, either through {@link #stopPlay} or a script's {@code stopGame()}.
     */
    public void play(dev.lumvex.game.Scene gameScene, ScriptCompiler.Result compiled, ScriptRuntime.Host host, Runnable onStopped) {
        tasks.add(() -> {
            if (runtime != null) return;
            onPlayStopped = onStopped;
            paused = false;
            runtime = new ScriptRuntime(compiled, gameScene, gameInput, host);
            physics = new PhysicsWorld(builder, new SceneQuery(scene));
            runtime.setPhysics(physics);
            playing = true;
            try {
                runtime.start();
            } catch (Throwable t) {
                t.printStackTrace();
            }
            applyRuntime(0);
        });
    }

    public void stopPlay() { tasks.add(this::endPlay); }

    /** See {@link Renderer#setRenderScale}. */
    public void setRenderScale(float scale) { tasks.add(() -> renderer.setRenderScale(scale)); }

    /** Pauses or resumes scripts and physics. */
    public void setPaused(boolean p) { paused = p; }

    /** Runs {@code f} on the render thread with the script runtime, which is null when not playing. */
    public <T> CompletableFuture<T> query(Function<ScriptRuntime, T> f) {
        CompletableFuture<T> cf = new CompletableFuture<>();
        tasks.add(() -> {
            try { cf.complete(f.apply(runtime)); } catch (Throwable t) { cf.completeExceptionally(t); }
        });
        return cf;
    }

    private void endPlay() {
        if (runtime == null) return;
        try { runtime.stop(); } catch (Throwable t) { t.printStackTrace(); }
        runtime = null;
        renderer.debugMode = 0;
        renderer.setUiImage(null, 0, 0);
        if (physics != null) physics.close();
        physics = null;
        playing = false;
        paused = false;
        builder.settleScripted();
        onPlayStopped.run();
    }

    /**
     * Pushes this frame's script changes through physics into the render scene. {@code dt = 0} skips the simulation.
     * Returns whether anything moved.
     */
    private boolean applyRuntime(float dt) {
        SceneDoc d = runtime.doc();
        renderDoc = d;
        boolean animated = false;
        try {
            boolean structure = runtime.takeStructure();
            Set<String> moved = runtime.takeMoved();
            if (structure) {
                rebuilds++;
                // soft build: only changed regions restart
                builder.build(d, true);
                animated = builder.lastChanged() > 0;
                updateStats();
            }
            if (physics != null) {
                long t0 = System.nanoTime();
                physics.sync(d);
                physics.teleported(runtime.takeTeleported());
                moved = physics.step(dt, moved);
                for (PhysicsWorld.Event ev : physics.takeEvents()) runtime.collision(ev.a(), ev.b(), ev.trigger(), ev.enter());
                physicsMs = physicsMs * 0.9 + (System.nanoTime() - t0) / 1e6 * 0.1;
            }
            if (!moved.isEmpty()) animated |= builder.moveScripted(d, moved);
        } catch (java.io.IOException e) {
            e.printStackTrace();
        }
        if (runtime.takeSettings()) applySettings(d);
        return animated;
    }

    private int uiW, uiH;

    /** Re-renders the game UI when the scripts changed it or the output size changed. */
    private void updateUi() {
        if (runtime == null) return;
        int w = renderer.width(), h = renderer.height();
        Function<String, Path> files = id -> assets.project().asset(id);
        runtime.setScreen(w, h);
        if (runtime.takeUiDirty() || w != uiW || h != uiH) {
            renderer.setUiImage(runtime.renderUi(w, h, files), w, h);
            uiW = w;
            uiH = h;
        }
    }

    /** Follows the player camera; returns false if the scene has none. */
    private boolean playerView() {
        SceneDoc.Entity e = runtime == null ? null : runtime.cameraEntity();
        if (e == null) return false;
        Vector3f p = renderDoc.worldMatrix(e, new Matrix4f()).getTranslation(new Vector3f());
        camera.position.set(p);
        camera.yaw = (float) Math.toRadians(e.rotation[1]);
        camera.pitch = (float) Math.toRadians(e.rotation[0]);
        SceneDoc.CameraComponent c = e.get(SceneDoc.CameraComponent.class);
        camera.fovYDegrees = c.fov;
        return true;
    }

    @Override
    public void run() {
        try {
            builder = new SceneBuilder(assets, scene);
            builder.build(renderDoc);
            while (!canvas.isDisplayable() || canvas.getWidth() == 0) Thread.sleep(20);
            Renderer.compileSer = false;
            renderer = new Renderer(canvas, scene, false, false);
            applySettings(renderDoc);
            camera.position.set(scene.cameraPosition);
            camera.yaw = scene.cameraYaw;
            camera.pitch = scene.cameraPitch;
            updateStats();
            ready = true;
            onReady.run();
        } catch (Throwable t) {
            t.printStackTrace();
            onError.accept(String.valueOf(t.getMessage()));
            return;
        }
        long last = System.nanoTime();
        double fpsTime = 0;
        int fpsFrames = 0;
        double fps = 0;
        while (running) {
            Runnable task;
            while ((task = tasks.poll()) != null) {
                try { task.run(); } catch (Throwable t) { t.printStackTrace(); }
            }
            long now = System.nanoTime();
            float dt = Math.min(0.1f, (now - last) / 1e9f);
            last = now;
            boolean animated = false;
            if (runtime != null) {
                builder.settleScripted();
                if (!paused) {
                    try {
                        runtime.update(dt);
                    } catch (Throwable t) {
                        t.printStackTrace();
                    }
                }
                if (runtime.stopRequested()) endPlay();
                else {
                    animated = applyRuntime(paused ? 0 : dt);
                    if (renderer.debugMode != runtime.debugView) {
                        renderer.debugMode = runtime.debugView;
                        renderer.resetAccumulation();
                    }
                    runtime.statsText = status;
                }
            }
            boolean moving = updateCamera(dt) || animated;
            try { updateUi(); } catch (Throwable t) { t.printStackTrace(); }
            try {
                renderer.render(camera, animated);
            } catch (Throwable t) {
                t.printStackTrace();
                onError.accept(String.valueOf(t.getMessage()));
                break;
            }
            fpsFrames++;
            fpsTime += dt;
            if (fpsTime > 0.5) {
                fps = fpsFrames / fpsTime;
                fpsFrames = 0;
                fpsTime = 0;
                String physicsText = physics != null && physics.bodyCount() > 0
                        ? String.format("  |  physics %d bodies %.2f ms", physics.bodyCount(), physicsMs) : "";
                status = String.format("%s  %.0f fps  |  GPU %.1f ms  |  %s%s", renderer.idle() ? "converged" : "rendering",
                        fps, renderer.gpuTotalMs(), stats, physicsText);
            }
            // A converged, static image needs no work; idle a little.
            sleep(renderer.idle() && !moving && tasks.isEmpty() ? 16 : 0);
        }
        renderer.close();
    }

    private static void sleep(long ms) {
        if (ms <= 0) return;
        try { Thread.sleep(ms); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }

    /** Follows the player camera, or applies fly camera input. Returns whether the camera moved. */
    private boolean updateCamera(float dt) {
        float[] d = input.consume();
        if (playerView()) return false;
        boolean moved = false;
        if (input.look && (d[0] != 0 || d[1] != 0)) {
            camera.yaw += d[0] * 0.0035f;
            camera.pitch = Math.max(-1.55f, Math.min(1.55f, camera.pitch + d[1] * 0.0035f));
            moved = true;
        }
        float speed = (input.fast ? 6f : 1.8f) * dt;
        Vector3f f = camera.forward(new Vector3f()), r = camera.right(new Vector3f());
        if (input.look) {
            if (input.forward) { camera.position.fma(speed, f); moved = true; }
            if (input.back) { camera.position.fma(-speed, f); moved = true; }
            if (input.right) { camera.position.fma(speed, r); moved = true; }
            if (input.left) { camera.position.fma(-speed, r); moved = true; }
            if (input.up) { camera.position.y += speed; moved = true; }
            if (input.down) { camera.position.y -= speed; moved = true; }
        }
        if (d[2] != 0) { camera.position.fma(-d[2] * 0.25f, f); moved = true; }
        return moved;
    }

    private void applySettings(SceneDoc doc) {
        renderer.maxBounces = doc.settings.bounces;
        renderer.sampleLimit = doc.settings.sampleLimit;
        renderer.exposure = doc.settings.exposure;
        renderer.flareStrength = doc.settings.lensFlares;
        renderer.setOverscan(Integer.getInteger("lumvex.overscan", doc.settings.overscan) / 100f);
    }

    // Triangles are reported twice: stored (each mesh once) and rendered (summed over instances).
    private void updateStats() {
        Set<Integer> used = new HashSet<>();
        long rendered = 0;
        for (Scene.Instance i : scene.instances) {
            used.add(i.mesh);
            rendered += scene.meshes.get(i.mesh).triangleCount();
        }
        long unique = 0;
        for (int m : used) unique += scene.meshes.get(m).triangleCount();
        stats = String.format("%d objects, %d instances  |  triangles: %,d stored, %,d rendered", renderDoc.entities.size(),
                scene.instances.size(), unique, rendered);
    }
}
