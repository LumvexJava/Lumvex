// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.kajiya.render.*;
import dev.lumvex.kajiya.world.View;
import dev.lumvex.kajiya.world.World;
import dev.lumvex.vk.Buffer;
import dev.lumvex.vk.VkContext;
import org.joml.Vector3f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.lwjgl.vulkan.VK13.*;

public final class KjRenderer implements AutoCloseable {
    public final VkContext ctx;
    public final Gfx g;
    public final World world;
    public final Ssgi ssgi = new Ssgi();
    public final Ircache ircache = new Ircache();
    public final Rtdgi rtdgi = new Rtdgi();
    public final Rtr rtr = new Rtr();
    public final ShadowDenoise shadowDenoise = new ShadowDenoise();
    public final Taa taa = new Taa();
    public final Post post = new Post();

    public final Vector3f sunDirection = new Vector3f(0.5f, 0.8f, 0.3f).normalize();
    public final Vector3f sunColorMultiplier = new Vector3f(1, 1, 1);
    public final Vector3f skyAmbient = new Vector3f();
    public float sunSizeMultiplier = 1f;
    public float evShift = 0f;
    public float contrast = 1f;
    public int debugShadingMode = 0;
    public boolean useMotionBlur = true;
    public int renderFlags;
    public float roughnessScale = 1f;

    private int frameIdx;
    private View prevView;
    private float preMult = 1f, preMultPrev = 1f, postMult = 1f;
    private final boolean rt;
    private static final int JITTER_COUNT = 128;
    private final float[][] jitter = new float[JITTER_COUNT][2];

    public KjRenderer(VkContext ctx) {
        this.ctx = ctx;
        this.g = new Gfx(ctx);
        this.world = new World(g);
        this.rt = ctx.rayTracingSupported;
        for (int i = 0; i < JITTER_COUNT; i++) {
            jitter[i][0] = radicalInverse(i + 1, 2) - 0.5f;
            jitter[i][1] = radicalInverse(i + 1, 3) - 0.5f;
        }
    }

    private static float radicalInverse(int n, int base) {
        float val = 0, invBase = 1f / base, invBi = invBase;
        while (n > 0) {
            int d = n % base;
            val += d * invBi;
            n = (int) (n * invBase);
            invBi *= invBase;
        }
        return val;
    }

    public boolean dynamicExposure;
    public float adaptSpeedLog2 = 1.5f, exposureBias = -2.1f;
    private float evFast, evSlow;

    private void updatePreExposure() {
        float smoothed = 0f;
        if (dynamicExposure) {
            float ev = Math.max(-16f, Math.min(16f, -post.imageLog2Lum));
            float dt = (1f / 60f) * (float) Math.pow(2.0, adaptSpeedLog2);
            evFast += (ev - evFast) * (1f - (float) Math.exp(-1.0 * dt));
            evSlow += (ev - evSlow) * (1f - (float) Math.exp(-0.25 * dt));
            smoothed = (evSlow + evFast) * 0.5f + exposureBias;
        }
        float evMult = (float) Math.pow(2.0, evShift + smoothed);
        preMultPrev = preMult;
        preMult = preMult * 0.9f + evMult * 0.1f;
        postMult = evMult / preMult;
    }

    private void writeFrameConstants(View view, int w, int h, float dt) {
        View prev = prevView == null ? view : prevView;
        float[] j = jitter[frameIdx % JITTER_COUNT];
        taa.jitterX = j[0];
        taa.jitterY = j[1];
        int off = g.dynAlloc(View.SIZE);
        ByteBuffer b = g.dynMap().slice(off, View.SIZE).order(ByteOrder.LITTLE_ENDIAN);
        View.write(b, view, prev, j[0], j[1], w, h);
        ircache.updateEyePosition(view.eye());
        float realSunAngularRadius = (float) Math.toRadians(0.53) * 0.5f;
        b.putFloat(720, sunDirection.x).putFloat(724, sunDirection.y).putFloat(728, sunDirection.z).putFloat(732, 0);
        b.putInt(736, frameIdx).putFloat(740, dt).putFloat(744, (float) Math.cos(sunSizeMultiplier * realSunAngularRadius)).putInt(748, 0);
        b.putFloat(752, sunColorMultiplier.x).putFloat(756, sunColorMultiplier.y).putFloat(760, sunColorMultiplier.z).putFloat(764, 0);
        b.putFloat(768, skyAmbient.x).putFloat(772, skyAmbient.y).putFloat(776, skyAmbient.z).putFloat(780, 0);
        b.putFloat(784, preMult).putFloat(788, preMultPrev).putFloat(792, preMult / preMultPrev).putFloat(796, 0);
        b.putInt(800, renderFlags).putFloat(804, roughnessScale).putInt(808, 0).putInt(812, 0);
        Vector3f c = ircache.gridCenter();
        b.putFloat(816, c.x).putFloat(820, c.y).putFloat(824, c.z).putFloat(828, 1f);
        for (int i = 0; i < Ircache.CASCADE_COUNT; i++) {
            int[] cc = ircache.cascadeConstants(i);
            for (int k = 0; k < 8; k++) b.putInt(832 + i * 32 + k * 4, cc[k]);
        }
        int n = world.instances.size();
        int instOff = g.dynAlloc(Math.max(1, n) * 4);
        if (n > 0) {
            //DEV: one bulk copy instead of a put per instance
            java.nio.FloatBuffer fb = g.dynMap().slice(instOff, n * 4).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
            fb.put(world.emissiveArray(), 0, n);
        }
        int triOff = g.dynAlloc(64);
        Pass.setFrameOffsets(off, instOff, triOff);
        prevView = view;
    }

    private Img environment;

    public void setEnvironment(dev.lumvex.render.scene.Environment env) {
        if (environment != null) {
            g.flushPools();
            environment.close();
            environment = null;
        }
        if (env == null) return;
        int w = env.width, h = env.height;
        ByteBuffer b = ByteBuffer.allocateDirect(w * h * 8).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < w * h; i++) {
            b.putShort(Float.floatToFloat16(Math.min(env.rgb[i * 3], 65504f))).putShort(Float.floatToFloat16(Math.min(env.rgb[i * 3 + 1], 65504f)))
                    .putShort(Float.floatToFloat16(Math.min(env.rgb[i * 3 + 2], 65504f))).putShort(Float.floatToFloat16(1f));
        }
        b.flip();
        environment = Tex.upload(g, w, h, Fmt.RGBA16F, b);
    }

    public void debugMode(int mode) {
        debugShadingMode = rt ? 0 : 4;
    }

    public Img record(org.lwjgl.vulkan.VkCommandBuffer cmd, View view, int w, int h, int outW, int outH, float dt) { return record(cmd, 0, view, w, h, outW, outH, dt); }

    public Img record(org.lwjgl.vulkan.VkCommandBuffer cmd, int slot, View view, int w, int h, int outW, int outH, float dt) {
        g.cmd = cmd;
        g.beginFrame(slot);
        g.markReset();
        return recordFrame(view, w, h, outW, outH, dt);
    }

    public Img recordFrame(View view, int w, int h, float dt) {
        return recordFrame(view, w, h, w, h, dt);
    }

    public Img recordFrame(View view, int w, int h, int outW, int outH, float dt) {
        updatePreExposure();
        writeFrameConstants(view, w, h, dt);
        long tlas = 0;
        if (rt) {
            world.recordTlasBuild();
            tlas = world.tlas.handle;
        }

        Img accum = g.temporal("root.accum", new Img.Desc(w, h, Fmt.RGBA16F));
        Img skyCube = environment != null ? Basics.iblCube(g, environment) : Basics.skyCube(g);
        Img convolvedSky = Basics.convolveCube(g, skyCube);

        Img[] vel = new Img[1];
        Basics.cullViewProj = new org.joml.Matrix4f(view.viewToClip).mul(view.worldToView);
        Basics.cullEye = view.eye(); //DEV
        Gbuf gb = Basics.rasterMeshes(g, world, w, h, vel);
        g.mark("raster");
        Img reprojection = Basics.reprojectionMap(g, gb, vel[0], null);
        Img ssgiTex = ssgi.render(g, gb, reprojection, accum);

        g.mark("reproj+ssgi");
        ircache.prepare(g);
        Img wrc = g.image(new Img.Desc(1, 1, Fmt.R8));

        if (rt) ircache.traceIrradiance(g, convolvedSky, tlas, wrc);

        g.mark("ircache");
        Img sunShadow = rt ? Basics.sunShadowMask(g, gb, tlas) : g.image(gb.depth.desc.withFormat(Fmt.R8));
        Rtdgi.Reprojected rep = rt ? rtdgi.reproject(g, reprojection) : null;
        Img denoisedShadow = sunSizeMultiplier > 0f ? shadowDenoise.render(g, gb, sunShadow, reprojection) : sunShadow;
        if (rt) ircache.sumUpIrradianceForSampling(g);

        g.mark("shadows");
        Img rtdgiTex;
        Rtr.Traced traced;
        if (rt) {
            Rtdgi.Output out = rtdgi.render(g, rep, gb, reprojection, convolvedSky, ircache, wrc, tlas, ssgiTex);
            rtdgiTex = out.irradiance();
            traced = rtr.trace(g, gb, reprojection, skyCube, tlas, rtdgiTex, out.candidates(), ircache, wrc);
        } else {
            rtdgiTex = g.image(new Img.Desc(1, 1, Fmt.RGBA8));
            traced = rtr.dummy(g, gb);
        }
        g.mark("rtdgi+rtr");
        Img rtrTex = Rtr.filterTemporal(g, traced, gb, reprojection);

        Img lit = g.image(new Img.Desc(w, h, Fmt.RGBA16F));
        Basics.lightGbuffer(g, gb, denoisedShadow, rtrTex, rtdgiTex, ircache, wrc, accum, lit, skyCube, convolvedSky, debugShadingMode, 0);

        g.mark("light");
        Img aa = taa.render(g, lit, reprojection, gb.depth, outW, outH);
        g.mark("taa");
        Img finalInput = useMotionBlur ? Post.motionBlur(g, aa, gb.depth, reprojection) : aa;
        g.mark("motionblur");
        Img out = post.render(g, finalInput, postMult, contrast);
        g.mark("post");

        world.storePrevTransforms();
        frameIdx++;
        return out;
    }

    /** Converts the tonemapped image to sRGB and composites a premultiplied UI image over it. */
    public void blit(Img main, Img ui, Img target) {
        Pass.compute(g, "final_blit.hlsl").bind(main).bind(ui).bind(target)
                .constants(new Cb().f4(main.desc.extentInvExtent()).f4(target.desc.extentInvExtent()))
                .dispatch(target.width(), target.height(), 1);
    }

    @Override
    public void close() {
        world.close();
        if (environment != null) environment.close();
        g.close();
    }
}
