# Lumvex Core

Lumvex Core is a real-time path tracing game engine written in Java. Instead of providing a standalone editor, it works as a code-first library: you write your scenes, gameplay logic, and UI directly in Java. 

Under the hood, it runs on Vulkan 1.3 using hardware ray tracing (RT cores), Jolt for physics, and shaderc to compile GLSL shaders at startup.

## Requirements

* **OS:** Windows x64
* **GPU:** Vulkan 1.3 with hardware ray tracing support (NVIDIA RTX or AMD RX 6000+)
* **Java:** JDK 21 (A full JDK is required if you compile external project scripts at runtime; otherwise a JRE is sufficient)
* **Maven:** 3.9+ (optional)

## Building from Source

```bash
mvn package   # Builds target/lumvex-core-1.0.0.jar (+ sources) and copies dependencies to lib/
mvn install   # Installs the engine into your local Maven repository (~/.m2)
```

### Adding Lumvex to your Project

Once installed to your local repository, add it via **Maven**:

```xml
<dependency>
    <groupId>dev.lumvex</groupId>
    <artifactId>lumvex-core</artifactId>
    <version>1.0.0</version>
</dependency>
```

Or with **Gradle**:
```kotlin
repositories { mavenLocal() }
dependencies { implementation("dev.lumvex:lumvex-core:1.0.0") }
```

**Without a build tool:** Add `target/lumvex-core-1.0.0.jar` and all JARs inside `lib/` to your classpath (in IntelliJ: *Project Structure → Libraries*). You can also attach `lumvex-core-1.0.0-sources.jar` for IDE documentation.
To run from the command line:
```bash
java -cp "lumvex-core-1.0.0.jar;lib/*;your-classes" your.Main
```

## Quick Start

Everything needed to build a game lives in `dev.lumvex.game`. A typical setup consists of a `Scene` for your level, `GameObject` subclasses for entities, scripts for behavior, and a `Canvas` for the HUD.

```java
import dev.lumvex.game.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Game game = new Game("Hello Lumvex");
        game.setWindowSize(1280, 720);
        game.addScene(new HelloScene()); // restarting the scene creates a fresh instance
        game.start();                    // non-blocking; runs on its own threads
    }
}

public class HelloScene extends Scene {
    @Override
    protected void build() {
        lighting().setSun(new Vec3(0.6f, 0.5f, 0.4f), Color.WHITE, 3);
        add(new Floor(40));
        add(new SpinningCube(new Vec3(0, 0.6f, 0)));
        add(new Viewer(new Vec3(0, 1.4f, 3.6f)));
    }
}

public class SpinningCube extends GameObject {
    public SpinningCube(Vec3 position) {
        super("Cube");
        setPosition(position);
        
        Material gold = Material.metal(Color.hex("#FFC857")).withRoughness(0.2f);
        addComponent(new MeshRenderer(Mesh.cube(0.8f), gold));
        addScript(new Spin(45));
    }
}

public class Spin extends ObjectScript {
    private float degreesPerSecond;

    public Spin(float degreesPerSecond) { 
        this.degreesPerSecond = degreesPerSecond; 
    }

    @Override
    public void onUpdate(float dt) { 
        rotate(0, degreesPerSecond * dt, 0); 
    }

    @OnClick
    void reverse() { 
        degreesPerSecond = -degreesPerSecond; 
    }
}
```

**[Read the Wiki & Step-by-Step Tutorials](https://github.com/LumvexJava/Lumvex/wiki)**

### Core Concepts (`dev.lumvex.game`)

* **`Game`** — Manages the window and main loop (`addScene`, `setWindowSize`, `setFullscreen`, `setRenderScale`, `start`, `quit`). Render scale adjusts automatically on HiDPI displays.
* **`Scene`** — Represents a level. Override `build()` to populate it using `add(object)` or `add(canvas)`, and configure `lighting()` or `renderSettings()`.
* **`GameObject`** — Base entity with a transform. Can hold components (`addComponent`), scripts (`addScript`), and child objects (`addChild`).
* **Components** — `MeshRenderer`, `ModelRenderer`, `Collider`, `RigidBody`, `Camera`, `PointLight`.
* **Primitives / Math** — Immutable value types like `Mesh`, `Material`, `Color`, and `Vec3`.
* **Scripting** — `ObjectScript`, `SceneScript`, and `UiScript`
* **UI** — Built-in HUD elements (`Canvas`, `Label`, `Panel`, `Button`, `Image`).

Objects instantiated during `Scene.build()` can be queried and manipulated directly by scripts at runtime:
```java
findAll(Coin.class);
ui(Hud.class).showScore(3);
scene(GameRules.class).coinCollected();
spawn(new Coin(at));
```

## Working with Assets

By default, assets are loaded relative to the `assets/` directory in your working folder:

```java
new ModelRenderer("models/chair.glb");
Material.textured("textures/oak", 2);
lighting().setEnvironment("hdri/meadow.hdr", 1);
```

* **Models (`models/<name>.glb`):** Binary glTF with embedded textures. Supports metallic-roughness PBR, alpha masking, and blending.
* **Texture Sets (`textures/<name>/`):** Folders containing PBR maps (albedo, OpenGL normal, roughness, metallic, AO, or packed ARM). Files are matched automatically by suffix (`diff`, `nor_gl`, `rough`, `metal`, `ao`, `arm`).
* **Materials (`materials/<name>.material.json`):** JSON definitions specifying `albedo`, `roughness`, `metallic`, `emission`, `transmission`, `ior`, `textures`, `uvScale`, and `normalStrength`.
* **Skies (`hdri/<name>.hdr`):** Equirectangular Radiance RGBE images.

*Tip: Assets downloaded from [Poly Haven](https://polyhaven.com) (`.glb` models, texture folders, `.hdr` skies) work out of the box without conversion.*

## JSON Scene Projects (Optional)

Instead of hardcoding scenes in Java, you can load a structured project directory (`project.json`, `scenes/*.scene.json`, `assets/`, and `scripts/`):

```java
Game game = Game.fromProject(Path.of("my-game"));
game.setStartScene("Main");
game.addScriptClasses(Door.class); // optional: register pre-compiled script classes
game.start();
```

Scene files reference scripts by their fully qualified class name. If you don't register them via `addScriptClasses`, the engine will compile the `.java` files inside the project's `scripts/` folder in-memory at startup (requires a JDK).

## Architecture & API Stability

```
src/main/java/dev/lumvex/
 ├── game/        Public game API (Scenes, GameObjects, Components)
 ├── script/      Scripting API
 ├── player/      Entry point and window management
 ├── engine/      Project loader, SceneDoc, AssetCache, Jolt physics integration
 └── render/, vk/, texture/, scene/, ui/ 
                  Low-level Vulkan backend, virtual texturing, ray tracing & UI rendering
```

**Public API Notice:** `dev.lumvex.game` as well as `SceneDoc` and `Project` (in `dev.lumvex.engine`) are considered stable. Internal packages (`render`, `vk`, `texture`, `ui`, etc.) are left public so you can tinker with the engine, but their APIs may change between releases.

## AI Disclosure

AI tools were used during development in a supporting role—specifically for refactoring and restructuring code to improve readability, generating automated debugging tests, assisting with bug diagnostics, and translating/drafting documentation. All core engine design, Vulkan/rendering pipelines, and runtime systems were engineered and reviewed by human developers.

## License

Licensed under the [MIT License](LICENSE).
