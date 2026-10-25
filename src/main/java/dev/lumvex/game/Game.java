package dev.lumvex.game;

import dev.lumvex.engine.AssetCache;
import dev.lumvex.engine.Project;
import dev.lumvex.engine.SceneDoc;
import dev.lumvex.render.RenderLoop;
import dev.lumvex.ui.UiDoc;
import dev.lumvex.vk.VulkanCanvas;

import javax.swing.JFrame;
import javax.swing.JOptionPane;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.WindowConstants;
import java.awt.AWTException;
import java.awt.BorderLayout;
import java.awt.Cursor;
import java.awt.Point;
import java.awt.Robot;
import java.awt.Toolkit;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Runs a game in its own window. This is the entry point for applications that use the engine:
 *
 * <pre>{@code
 * public static void main(String[] args) throws IOException {
 *     Game game = new Game("Coin Courtyard");
 *     game.setWindowSize(1600, 900);
 *     game.addScene(new Courtyard());
 *     game.start();
 * }
 * }</pre>
 *
 * Scenes are classes of the application; models, textures and other assets are read from the {@code assets/}
 * folder of the working directory. A project with scene files is run with {@link #fromProject(Path)} instead.
 *
 * <p>Keyboard and mouse input is routed to the scripts.
 */
public final class Game {
    private final Path projectDir;
    private final Map<String, Supplier<? extends Scene>> scenes = new LinkedHashMap<>();
    private final Map<Class<? extends Scene>, String> sceneNames = new LinkedHashMap<>();
    private String startScene, title;
    private boolean fullscreen;
    // NaN: leave the renderer's default
    private float renderScale = Float.NaN;
    private int width = 1600, height = 900;
    private ClassLoader scriptLoader;
    private final List<Class<?>> scriptClasses = new ArrayList<>();
    private boolean exitOnQuit = true;
    private Runnable onStarted = () -> { };
    private Consumer<String> onError;

    private Project project;
    private AssetCache assets;
    private final VulkanCanvas canvas = new VulkanCanvas();
    private RenderLoop render;
    private JFrame frame;
    private Robot robot;
    private boolean mouseLocked, quit;
    private int lastX, lastY;

    /** A game whose scenes are added with {@link #addScene}. */
    public Game(String title) {
        this.projectDir = null;
        this.title = title;
    }

    private Game(Path projectDir) { this.projectDir = projectDir; }

    /**
     * A game that runs a project folder, i.e. the folder containing {@code project.json}, {@code scenes/} and
     * {@code assets/}.
     */
    public static Game fromProject(Path projectDir) { return new Game(projectDir); }

    /**
     * Adds a scene through a factory that creates a fresh instance whenever the scene is loaded, e.g.
     * {@code () -> new Level(3)}. Use it for scenes whose constructor takes parameters. The first scene added is the
     * start scene.
     */
    public void addScene(Supplier<? extends Scene> factory) {
        Scene sample = factory.get();
        register(sample.getClass(), sample.name(), factory);
    }

    /**
     * Adds a scene. The game plays this instance first; a restart creates a new one through the scene's
     * constructor without parameters, which the class must therefore have. The first scene added is the start
     * scene.
     */
    public void addScene(Scene scene) {
        Constructor<? extends Scene> constructor = emptyConstructor(scene.getClass());
        Scene[] unused = {scene};
        register(scene.getClass(), scene.name(), () -> {
            Scene next = unused[0];
            if (next != null) {
                unused[0] = null;
                return next;
            }
            try {
                return constructor.newInstance();
            } catch (ReflectiveOperationException ex) {
                throw new IllegalStateException("cannot create a new " + scene.getClass().getSimpleName(), ex);
            }
        });
    }

    private void register(Class<? extends Scene> type, String name, Supplier<? extends Scene> factory) {
        if (projectDir != null) throw new IllegalStateException("a project game loads its scenes from " + projectDir);
        if (scenes.putIfAbsent(name, factory) != null) throw new IllegalArgumentException("there already is a scene named " + name);
        sceneNames.putIfAbsent(type, name);
    }

    private static <T extends Scene> Constructor<T> emptyConstructor(Class<T> type) {
        try {
            Constructor<T> c = type.getDeclaredConstructor();
            c.setAccessible(true);
            return c;
        } catch (NoSuchMethodException ex) {
            throw new IllegalArgumentException(type.getSimpleName() + " needs a constructor without parameters to be restarted;"
                    + " otherwise pass a factory: addScene(() -> new " + type.getSimpleName() + "(...))");
        }
    }

    /** The scene to start with. Defaults to the first scene added, or the project's start scene. */
    public void setStartScene(String name) { this.startScene = name; }

    /** Starts with the scene of the given class, which must have been added before. */
    public void setStartScene(Class<? extends Scene> scene) {
        String name = sceneNames.get(scene);
        if (name == null) throw new IllegalArgumentException(scene.getSimpleName() + " was not added, see addScene");
        this.startScene = name;
    }

    /** Window title. A project game defaults to the project's game title. */
    public void setTitle(String title) { this.title = title; }

    /** A decorated window of the given size. This is the default, at 1600 x 900. */
    public void setWindowSize(int width, int height) {
        this.width = width;
        this.height = height;
        this.fullscreen = false;
    }

    /** A borderless window that fills the screen. */
    public void setFullscreen(boolean fullscreen) { this.fullscreen = fullscreen; }

    /**
     * The fraction of the window's pixels that is path traced, per axis, from 0.25 to 1; the image is scaled up to
     * the window. Lower values trade sharpness for frame rate. The default, 0, picks it automatically: windows with
     * more than 1920 x 1080 pixels, as on high-DPI displays, are traced at about that many pixels.
     */
    public void setRenderScale(float scale) {
        this.renderScale = scale;
        if (render != null) render.setRenderScale(scale);
    }

    /**
     * Project games only: uses compiled classes of the application as scripts instead of compiling the project's
     * {@code scripts/} folder. Scene files name their script classes; they are looked up through the class loader
     * of the given classes, so any script on that class path works, listed or not.
     */
    public void addScriptClasses(Class<?>... classes) {
        scriptClasses.addAll(Arrays.asList(classes));
        if (scriptLoader == null && classes.length > 0) scriptLoader = classes[0].getClassLoader();
    }

    /** Whether ending the game exits the JVM (the default). If false, only the window is closed. */
    public void setExitOnQuit(boolean exit) { this.exitOnQuit = exit; }

    /** Called once the first scene is loaded and its scripts are running. */
    public void setOnStarted(Runnable listener) { this.onStarted = listener; }

    /**
     * Called with a message when the renderer fails, on the event dispatch thread. By default an error dialog is
     * shown and the game ends.
     */
    public void setOnError(Consumer<String> listener) { this.onError = listener; }

    /** The project, or null before {@link #start()}. */
    public Project project() { return project; }

    /** The render loop, or null before {@link #start()}. Use {@link RenderLoop#query} to inspect the running scripts. */
    public RenderLoop renderLoop() { return render; }

    /** The game window, or null before {@link #start()}. */
    public JFrame window() { return frame; }

    /**
     * Opens the window and starts the game as soon as the renderer is ready. Returns immediately; the game runs on
     * its own threads.
     */
    public void start() throws IOException {
        project = projectDir != null ? Project.open(projectDir) : Project.inMemory(Path.of(""), title);
        assets = new AssetCache(project);
        render = new RenderLoop(canvas, assets);
        if (!Float.isNaN(renderScale)) render.setRenderScale(renderScale);
        String first = firstScene();
        ScriptRuntime.Host host = new ScriptRuntime.Host() {
            @Override public Scene loadScene(String name) throws IOException { return Game.this.loadScene(name); }

            @Override public String sceneName(Class<? extends Scene> type) { return sceneNames.get(type); }

            @Override public SceneDoc prefab(String id) { return assets.prefab(id); }

            @Override public UiDoc ui(String id) {
                try { return UiDoc.load(project.asset(id)); } catch (IOException ex) { return null; }
            }
        };
        // The scripts run on the first instance; the renderer builds the static scene from a second one.
        Scene scene = loadScene(first);
        SceneDoc preview = loadScene(first).document();
        ScriptCompiler.Result scripts = loadScripts(scene);
        if (projectDir != null) {
            System.out.println("[game] scripts: " + scripts.scripts.size() + " classes" + (scripts.ok ? "" : " - ERRORS " + scripts.problems));
        }
        // Vulkan paints the whole canvas; avoid AWT flicker
        System.setProperty("sun.awt.noerasebackground", "true");
        try {
            if (SwingUtilities.isEventDispatchThread()) openWindow();
            else SwingUtilities.invokeAndWait(this::openWindow);
        } catch (Exception ex) {
            throw new IOException("the window could not be opened", ex);
        }
        render.start(preview, () -> {
                    System.out.println("[game] scene \"" + first + "\" loaded");
                    render.play(scene, scripts, host, () -> SwingUtilities.invokeLater(() -> quit(0)));
                    onStarted.run();
                },
                err -> SwingUtilities.invokeLater(() -> {
                    canvas.showStartupError();
                    if (onError != null) {
                        onError.accept(err);
                    } else {
                        JOptionPane.showMessageDialog(frame, "Graphics error:\n" + err, frame.getTitle(), JOptionPane.ERROR_MESSAGE);
                        quit(1);
                    }
                }));
    }

    /** Ends the game, like {@code stopGame()} in a script. */
    public void quit() { quit(0); }

    private String firstScene() throws IOException {
        if (startScene != null) return startScene;
        if (projectDir == null) {
            if (scenes.isEmpty()) throw new IllegalStateException("no scene added, see Game.addScene");
            return scenes.keySet().iterator().next();
        }
        return project.startScene != null ? project.startScene : project.sceneNames().get(0);
    }

    private Scene loadScene(String name) throws IOException {
        if (projectDir != null) return Scene.of(SceneDoc.load(project.sceneFile(name)));
        Supplier<? extends Scene> factory = scenes.get(name);
        if (factory == null) throw new IOException("no scene named " + name + "; scenes: " + scenes.keySet());
        return factory.get();
    }

    private ScriptCompiler.Result loadScripts(Scene scene) {
        if (projectDir == null) return ScriptCompiler.fromClassLoader(scene.getClass().getClassLoader(), scriptClasses);
        if (scriptLoader != null) return ScriptCompiler.fromClassLoader(scriptLoader, scriptClasses);
        return ScriptCompiler.compile(project.scriptsDir());
    }

    // Runs on the event dispatch thread.
    private void openWindow() {
        frame = new JFrame(title != null ? title : project.gameTitle != null ? project.gameTitle : project.name);
        frame.setBackground(java.awt.Color.BLACK);
        frame.add(canvas, BorderLayout.CENTER);
        frame.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        frame.addWindowListener(new WindowAdapter() {
            @Override public void windowClosing(WindowEvent e) { quit(0); }
        });
        if (fullscreen) {
            frame.setUndecorated(true);
            frame.setExtendedState(JFrame.MAXIMIZED_BOTH);
        } else {
            frame.setSize(width, height);
            frame.setLocationRelativeTo(null);
        }
        installInput();
        try { robot = new Robot(); } catch (AWTException ignored) { }
        new Timer(40, e -> syncMouseLock()).start();
        canvas.showShaderLoading();
        frame.setVisible(true);
        canvas.requestFocusInWindow();
    }

    /** Shuts everything down once. With {@code exitOnQuit} the JVM exits with {@code code}. */
    private void quit(int code) {
        if (quit) return;
        quit = true;
        render.stopPlay();
        render.stop();
        assets.close();
        if (frame != null) frame.dispose();
        System.out.println("[game] ended");
        if (exitOnQuit) System.exit(code);
    }

    /**
     * Forwards keyboard and mouse events to the scripts. A left click also picks the object under the cursor (the
     * screen center while the mouse is captured) for {@code @OnClick}. In scenes without a player camera, holding the
     * right mouse button flies the camera with WASD.
     */
    private void installInput() {
        canvas.addKeyListener(new KeyAdapter() {
            @Override public void keyPressed(KeyEvent e) { key(e, true); }
            @Override public void keyReleased(KeyEvent e) { key(e, false); }
        });
        canvas.addMouseListener(new MouseAdapter() {
            @Override public void mousePressed(MouseEvent e) {
                canvas.requestFocusInWindow();
                lastX = e.getX();
                lastY = e.getY();
                float w = Math.max(1, canvas.getWidth()), h = Math.max(1, canvas.getHeight());
                render.gameInput.button(button(e), true, mouseLocked ? 0.5f : e.getX() / w, mouseLocked ? 0.5f : e.getY() / h);
                if (SwingUtilities.isLeftMouseButton(e)) {
                    double sc = canvas.pixelScale();
                    int x = (int) ((mouseLocked ? canvas.getWidth() / 2 : e.getX()) * sc), y = (int) ((mouseLocked ? canvas.getHeight() / 2 : e.getY()) * sc);
                    render.pick(x, y, (id, depth) -> render.gameInput.clicked(id));
                }
                // scenes without a player camera
                if (SwingUtilities.isRightMouseButton(e)) render.input.look = true;
            }

            @Override public void mouseReleased(MouseEvent e) {
                float w = Math.max(1, canvas.getWidth()), h = Math.max(1, canvas.getHeight());
                render.gameInput.button(button(e), false, e.getX() / w, e.getY() / h);
                if (SwingUtilities.isRightMouseButton(e)) {
                    RenderLoop.Input in = render.input;
                    in.look = false;
                    in.forward = in.back = in.left = in.right = in.up = in.down = false;
                }
            }
        });
        canvas.addMouseMotionListener(new MouseMotionAdapter() {
            @Override public void mouseMoved(MouseEvent e) { mouse(e); }

            @Override public void mouseDragged(MouseEvent e) {
                if (render.input.look) render.input.addLook(e.getX() - lastX, e.getY() - lastY);
                mouse(e);
            }
        });
    }

    private void key(KeyEvent e, boolean down) {
        if (down && e.getKeyCode() == KeyEvent.VK_ESCAPE && mouseLocked) render.gameInput.unlock();
        render.gameInput.key(e.getKeyCode(), down);
        RenderLoop.Input in = render.input;
        in.fast = e.isShiftDown();
        if (in.look) {
            switch (e.getKeyCode()) {
                case KeyEvent.VK_W -> in.forward = down;
                case KeyEvent.VK_S -> in.back = down;
                case KeyEvent.VK_A -> in.left = down;
                case KeyEvent.VK_D -> in.right = down;
                case KeyEvent.VK_E, KeyEvent.VK_SPACE -> in.up = down;
                case KeyEvent.VK_Q -> in.down = down;
                default -> { }
            }
        }
    }

    /**
     * While the mouse is captured, movement is measured from the canvas center and the cursor is warped back there
     * after each event; otherwise it is the delta to the previous position.
     */
    private void mouse(MouseEvent e) {
        float w = Math.max(1, canvas.getWidth()), h = Math.max(1, canvas.getHeight());
        if (mouseLocked) {
            int dx = e.getX() - canvas.getWidth() / 2, dy = e.getY() - canvas.getHeight() / 2;
            if (dx == 0 && dy == 0) return;
            render.gameInput.mouseMoved(dx, dy, 0.5f, 0.5f);
            centerMouse();
        } else {
            render.gameInput.mouseMoved(e.getX() - lastX, e.getY() - lastY, e.getX() / w, e.getY() / h);
        }
        lastX = e.getX();
        lastY = e.getY();
    }

    private static String button(MouseEvent e) {
        return SwingUtilities.isLeftMouseButton(e) ? "LEFT" : SwingUtilities.isRightMouseButton(e) ? "RIGHT" : "MIDDLE";
    }

    /**
     * Polled by a timer: captures the mouse while a script requests it, no menu is open and the window has focus.
     * A captured cursor is invisible and held at the canvas center.
     */
    private void syncMouseLock() {
        boolean want = render.playing && render.gameInput.mouseLocked() && !render.gameInput.uiWantsMouse() && frame.isActive();
        if (want == mouseLocked) return;
        mouseLocked = want;
        if (want) {
            canvas.setCursor(Toolkit.getDefaultToolkit().createCustomCursor(new BufferedImage(16, 16, BufferedImage.TYPE_INT_ARGB),
                    new Point(0, 0), "none"));
            centerMouse();
        } else {
            canvas.setCursor(Cursor.getDefaultCursor());
        }
    }

    private void centerMouse() {
        if (robot == null || !canvas.isShowing()) return;
        Point p = canvas.getLocationOnScreen();
        robot.mouseMove(p.x + canvas.getWidth() / 2, p.y + canvas.getHeight() / 2);
    }
}
