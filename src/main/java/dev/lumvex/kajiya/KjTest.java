package dev.lumvex.kajiya;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.kajiya.render.Fmt;
import dev.lumvex.kajiya.world.SceneAdapter;
import dev.lumvex.kajiya.world.View;
import dev.lumvex.render.scene.*;
import dev.lumvex.vk.RenderSurface;
import dev.lumvex.vk.VkContext;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.vulkan.VkInstance;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.List;

public final class KjTest {
    public static void main(String[] args) throws Exception {
        int w = Integer.getInteger("kj.w", 1280), h = Integer.getInteger("kj.h", 720);
        int frames = Integer.getInteger("kj.frames", 40);
        String out = System.getProperty("kj.out", "kj_test.png");
        RenderSurface headless = new RenderSurface() {
            public List<String> instanceExtensions() { return List.of(); }
            public long createSurface(VkInstance i) { return 0; }
            public int[] framebufferSize() { return new int[]{w, h}; }
            public boolean consumeResized() { return false; }
            public void waitWhileEmpty() { }
            public boolean headless() { return true; }
        };
        VkContext ctx = new VkContext(headless, Boolean.getBoolean("kj.validation"));
        KjRenderer r = new KjRenderer(ctx);

        Scene scene = new Scene();
        int floor = scene.addMesh(MeshFactory.plane(20));
        int cube = scene.addMesh(MeshFactory.cube());
        int sphere = scene.addMesh(MeshFactory.sphere(0.8f, 48, 24));
        int mFloor = scene.addMaterial(Material.diffuse(0.6f, 0.6f, 0.6f));
        int mRed = scene.addMaterial(Material.diffuse(0.8f, 0.1f, 0.1f));
        int mGold = scene.addMaterial(Material.metal(1.0f, 0.78f, 0.34f, 0.2f));
        int mGlow = scene.addMaterial(Material.emissive(1f, 0.6f, 0.2f, 8f));
        scene.addInstance(floor, mFloor, new Matrix4f());
        scene.addInstance(cube, mRed, new Matrix4f().translate(-2f, 0.5f, 0f));
        scene.addInstance(sphere, mGold, new Matrix4f().translate(0f, 0.8f, 0f));
        scene.addInstance(sphere, mGlow, new Matrix4f().translate(2.2f, 0.8f, 0.5f).scale(0.5f));
        if ("room".equals(System.getProperty("kj.scene"))) {
            java.nio.ByteBuffer alb = java.nio.ByteBuffer.allocateDirect(128 * 128 * 4), nrm = java.nio.ByteBuffer.allocateDirect(128 * 128 * 4), orm = java.nio.ByteBuffer.allocateDirect(128 * 128 * 4);
            for (int y = 0; y < 128; y++) for (int x = 0; x < 128; x++) {
                int i = (y * 128 + x) * 4;
                boolean c = ((x / 16) + (y / 16)) % 2 == 0;
                alb.put(i, (byte) (c ? 230 : 60)).put(i + 1, (byte) (c ? 200 : 40)).put(i + 2, (byte) (c ? 160 : 30)).put(i + 3, (byte) 255);
                nrm.put(i, (byte) 128).put(i + 1, (byte) 128).put(i + 2, (byte) 255).put(i + 3, (byte) 255);
                orm.put(i, (byte) 255).put(i + 1, (byte) 200).put(i + 2, (byte) 0).put(i + 3, (byte) 255);
            }
            int ts = scene.addTextureSet(dev.lumvex.texture.TextureSet.of("checker", 128, alb, nrm, orm));
            int mTex = scene.addMaterial(Material.textured(ts, 1.5f));
            int wall = scene.addMesh(MeshFactory.box(8f, 4f, 0.2f));
            int mWhite = scene.addMaterial(Material.diffuse(0.75f, 0.75f, 0.75f));
            int mGreen = scene.addMaterial(Material.diffuse(0.1f, 0.6f, 0.15f));
            int light = scene.addMesh(MeshFactory.box(2f, 0.1f, 2f));
            int mLight = scene.addMaterial(Material.emissive(1f, 0.9f, 0.8f, 15f));
            scene.instances.clear();
            scene.addInstance(floor, mTex, new Matrix4f());
            scene.addInstance(floor, mWhite, new Matrix4f().translate(0, 4f, 0).rotateX((float) Math.PI));
            scene.addInstance(wall, mWhite, new Matrix4f().translate(0, 2f, -4f));
            scene.addInstance(wall, mGreen, new Matrix4f().translate(-4f, 2f, 0).rotateY((float) Math.PI / 2));
            scene.addInstance(wall, mRed, new Matrix4f().translate(4f, 2f, 0).rotateY((float) Math.PI / 2));
            scene.addInstance(light, mLight, new Matrix4f().translate(0, 3.9f, 0));
            scene.addInstance(cube, mWhite, new Matrix4f().translate(-1.2f, 0.5f, -1f).rotateY(0.5f));
            scene.addInstance(sphere, mGold, new Matrix4f().translate(1.3f, 0.8f, 0.2f));
        }
        SceneAdapter adapter = new SceneAdapter(r.g, r.world, scene);
        scene.structureChanged();
        adapter.sync();

        if (System.getProperty("kj.env") != null) {
            dev.lumvex.render.scene.Environment env = dev.lumvex.render.scene.Environment.load(java.nio.file.Path.of(System.getProperty("kj.env")));
            env.extractSun(30f);
            r.setEnvironment(env);
            r.sunDirection.set(env.sunDirection);
        }
        Camera cam = new Camera();
        cam.position.set(0, 2.2f, 7f);
        if ("room".equals(System.getProperty("kj.scene"))) { cam.position.set(0, 2f, 3.6f); cam.pitch = 0.05f; r.sunSizeMultiplier = 0f; r.sunColorMultiplier.set(0, 0, 0); r.evShift = 2f; }
        cam.pitch = 0.15f;
        r.sunDirection.set(0.5f, 0.7f, 0.4f).normalize();

        Img target = new Img(ctx, new Img.Desc(w, h, Fmt.RGBA8));
        Img ui = new Img(ctx, new Img.Desc(w, h, Fmt.RGBA8));
        for (int f = 0; f < frames; f++) {
            View v = View.of(cam.view(), cam.fovYDegrees, (float) w / h, 0.01f);
            Img[] result = new Img[1];
            r.g.submit(() -> {
                Img main = r.recordFrame(v, w, h, 1f / 60f);
                r.blit(main, ui, target);
                result[0] = main;
            });
        }
        if (Boolean.getBoolean("kj.resize")) {
            int w2 = w * 3 / 4, h2 = h * 3 / 4;
            Img target2 = new Img(ctx, new Img.Desc(w2, h2, Fmt.RGBA8));
            Img ui2 = new Img(ctx, new Img.Desc(w2, h2, Fmt.RGBA8));
            r.g.flushPools();
            for (int f = 0; f < 5; f++) {
                View v = View.of(cam.view(), cam.fovYDegrees, (float) w2 / h2, 0.01f);
                r.g.submit(() -> r.blit(r.recordFrame(v, w2 * 2 / 3, h2 * 2 / 3, w2, h2, 1f / 60f), ui2, target2));
            }
            System.out.println("[kj] resize/render-scale frames OK");
        }
        float[] px = r.g.readback(target);
        BufferedImage bi = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        double sum = 0;
        for (int i = 0; i < w * h; i++) {
            int rr = Math.round(px[i * 4] * 255), gg = Math.round(px[i * 4 + 1] * 255), bb = Math.round(px[i * 4 + 2] * 255);
            sum += (rr + gg + bb) / 3.0;
            bi.setRGB(i % w, i / w, (rr << 16) | (gg << 8) | bb);
        }
        ImageIO.write(bi, "png", new File(out));
        System.out.printf("[kj] wrote %s, mean value %.2f/255%n", out, sum / (w * h));
        r.close();
        ctx.close();
    }
}
