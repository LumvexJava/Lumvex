# Lumvex Core

Lumvex Core is a real-time rendering „engine“ written in Java. It has no UI interface per se but is completely bound by code (you have to write your scenes / gameplay logic / objects etc. in Java yourself (like it should be). Lumvex only handles the rendering so you can fully concentrate on your project.

Under the hood, it runs on Vulkan 1.3 and Jolt for physics. The renderer is a Java port of the [kajiya](https://github.com/EmbarkStudios/kajiya) pipeline (rasterized G-buffer, ray traced sun shadows, an irradiance cache, ReSTIR diffuse GI, ReSTIR reflections, SSGI, TAA, motion blur and bloom/tonemapping). Its HLSL shaders are compiled to SPIR-V with `dxc` (see `tools/compile_kajiya_shaders.sh`); the compiled SPIR-V ships in the jar, so `dxc` is only needed when you change a shader.

<img width="320" height="180" alt="01-autumn-atrium" src="https://github.com/user-attachments/assets/08faf122-6c93-4fab-8e70-179b9a175ba3" />
<img width="320" height="180" alt="02-neon-reflections" src="https://github.com/user-attachments/assets/9421af78-d211-432f-8757-2179b23d85be" />
<img width="320" height="180" alt="03-material-studies" src="https://github.com/user-attachments/assets/5d0193f6-245a-49d2-94bf-2df4887dddc7" />


* **GPU:** Vulkan 1.3 with hardware ray tracing support (NVIDIA RTX or AMD RX 6000+ (not sure about amd gpu support)) for the full renderer. Any other Vulkan 1.3 GPU (e.g. Intel or AMD integrated graphics) runs in the raster mode.
* **Java:** JDK 21 (A full JDK is required if you compile external project scripts at runtime; otherwise a JRE is sufficient)
* **Maven:** 3.9+ (optional)
* **OS:** Windows x86-64 or Linux x86-64 with an X11 display (including XWayland);  (Wayland surfaces are not supported!). Maven selects the matching LWJGL and Jolt native libraries for the build OS.

## Building from Source (how to get started with setting up):
You have to install maven by yourself (https://maven.apache.org/download.cgi)
```bash
mvn package   # Builds target/lumvex-core-1.3.0.jar (+ sources) and copies dependencies to lib/
mvn install   # Installs the engine into your local Maven repository (~/.m2)
```
After following the commands, you should be able to "add" the engine into IntelliJ using the pom.xml file (Maven Project):

```xml
<dependency>
    <groupId>dev.lumvex</groupId>
    <artifactId>lumvex-core</artifactId>
    <version>1.3.0</version>
</dependency>
```

Or with Gradle:
```kotlin
repositories { mavenLocal() }
dependencies { implementation("dev.lumvex:lumvex-core:1.3.0") }
```


## Quick Start

Everything needed to build a game lives in `dev.lumvex.game`. A typical setup consists of a `Scene` for your level, `GameObject` subclasses for entities, scripts for behavior, and a `Canvas` for the HUD.

Here are some scripts to help you get started:

```java
import dev.lumvex.game.*;

public class Main {
    public static void main(String[] args) throws Exception {
        Game game = new Game("Hello Lumvex");
        game.setWindowSize(1280, 720);
        game.addScene(new HelloScene());
        game.start();                    
        
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

**[Tutorials & Doc (both unfinished)](https://github.com/LumvexJava/Lumvex/wiki)**


## Rendering

The kajiya pipeline needs hardware ray tracing (RTX 20+/RX 6000+) for the full image. Without it only the rasterized G-buffer, the sun and the sky are lit. The former render modes (`hybrid`, `path`, `raster`) and their settings are accepted but ignored; temporal accumulation, `sampleLimit`, lens flares, overscan and the old debug views no longer exist.

| JVM option | Effect |
| --- | --- |
| `-Dlumvex.gpu=<part of the name>` | picks a GPU, e.g. `-Dlumvex.gpu=intel` for the integrated one |
| `-Dlumvex.noraytracing=true` | ignores ray tracing hardware |
| `-Dlumvex.gputimes=true` | prints the GPU frame time every 120 frames |
| `-Dlumvex.validation=true` | enables the Vulkan validation layer |
| `-Dlumvex.dxc=true` | recompiles the HLSL shaders with `dxc` from the Vulkan SDK instead of loading the shipped SPIR-V |
| `-Dlumvex.autoshot=<file>` (`.autoshotFrame=N`, `.autoshotExit=true`) | saves a screenshot after N frames |





By default, assets are loaded relative to the `assets/` directory in your working folder:

```java
new ModelRenderer("models/chair.glb");
Material.textured("textures/oak", 2);
lighting().setEnvironment("hdri/meadow.hdr", 1);
```
*Tip: Assets downloaded from [Poly Haven](https://polyhaven.com) (`.glb` models, texture folders, `.hdr` skies) work out of the box without conversion, so keep in mind.*

Scene files reference scripts by their fully qualified class name. If you don't register them via `addScriptClasses`, the engine will compile the `.java` files inside the project's `scripts/` folder in-memory at startup (requires a JDK).

## Architecture:

```
src/main/java/dev/lumvex/
 ├── game/        Public game API (Scenes, GameObjects, Components)
 ├─- script/      Scripting API
 ├── player/      Entry point and window management
 ├── engine/      Project loader, SceneDoc, AssetCache, Jolt physics integration
 └── render/, vk/, texture/, scene/, ui/ 
                  Vulkan backend, virtual texturing, ray tracing & UI rendering
```

*Note: Code marked with `//DEV` has been tested and works, but has not been commented yet.*

## Credits

The renderer is a Java port of **[kajiya](https://github.com/EmbarkStudios/kajiya)** by Embark Studios (Copyright (c) 2019 Embark Studios, dual-licensed under MIT OR Apache-2.0). The Java code in `dev.lumvex.kajiya` was rewritten from kajiya's Rust code, and the HLSL shaders in `src/main/resources/kajiya/shaders` were taken from kajiya (some of them adapted).

kajiya itself ships third-party code that is also included here:
- AMD FidelityFX Shadow Denoiser (MIT)
- [MinimalAtmosphere](https://github.com/Fewes/MinimalAtmosphere) by Felix Westin (MIT)
- Blue noise textures by Christoph Peters (CC0)
- Sampler tables from [blue-noise-sampler](https://github.com/Jasper-Bekkers/blue-noise-sampler)

Full details and license texts: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [`licenses/`](licenses).

Thank you to Embark Studios and all kajiya contributors!

## AI Disclosure

In the early stages of the project I used AI tools for refactoring and for writing code comments. I no longer use AI tools for this, except for help with the license/credits documentation.

The core design of the engine and the runtime systems were developed and reviewed by human developers (it's just me, so blame me if something doesn't work). Starting with version 1.3.0, the render pipelines contain code from [kajiya](https://github.com/EmbarkStudios/kajiya) (ported from Rust to Java, shaders taken over), see [Credits](#credits).

## License

Licensed under the [MIT License](LICENSE). Third-party code (kajiya and others) keeps its own license, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
