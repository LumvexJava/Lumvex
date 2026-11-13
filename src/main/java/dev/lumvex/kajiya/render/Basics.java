// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.kajiya.world.World;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK13.*;

public final class Basics {

    public static Img skyCube(Gfx g) {
        Img sky = g.image(Img.Desc.cube(64, Fmt.RGBA16F));
        Pass.compute(g, "sky/comp_cube.hlsl").bind(sky, Img.View.ARRAY).dispatch(64, 64, 6);
        return sky;
    }

    public static Img convolveCube(Gfx g, Img input) {
        Img out = g.image(Img.Desc.cube(16, Fmt.RGBA16F));
        Pass.compute(g, "convolve_cube.hlsl").bind(input).bind(out, Img.View.ARRAY).constants(new Cb().u(16)).dispatch(16, 16, 6);
        return out;
    }

    public static Img iblCube(Gfx g, Img equirect) {
        final int width = 1024;
        Img out = g.image(Img.Desc.cube(width, Fmt.RGBA16F));
        Pass.compute(g, "ibl/ibl_cube.hlsl").bind(equirect).bind(out, Img.View.ARRAY).constants(new Cb().u(width)).dispatch(width, width, 6);
        return out;
    }

    public static Img reprojectionMap(Gfx g, Gbuf gb, Img velocity, PingPong unused) {
        Img out = g.image(gb.depth.desc.withFormat(Fmt.R16G16B16A16_SNORM));
        Img prevDepth = g.temporal("reprojection.prev_depth", gb.depth.desc.withFormat(Fmt.R32F));
        Pass.compute(g, "calculate_reprojection_map.hlsl").bind(gb.depth).bind(gb.normal).bind(prevDepth).bind(velocity).bind(out)
                .constants(new Cb().f4(out.desc.extentInvExtent())).dispatch(out.width(), out.height(), 1);
        Pass.compute(g, "copy_depth_to_r.hlsl").bind(gb.depth).bind(prevDepth).dispatch(prevDepth.width(), prevDepth.height(), 1);
        return out;
    }

    public static Img sunShadowMask(Gfx g, Gbuf gb, long tlas) {
        Img out = g.image(gb.depth.desc.withFormat(Fmt.R8));
        Pass.rt(g, "rt/trace_sun_shadow_mask.rgen.hlsl", List.of("rt/shadow.rmiss.hlsl", "rt/shadow.rmiss.hlsl"), List.of())
                .bind(gb.depth).bind(gb.normal).bind(out).tlas(tlas).traceRays(out.width(), out.height(), 1);
        return out;
    }

    public static void lightGbuffer(Gfx g, Gbuf gb, Img shadowMask, Img rtr, Img rtdgi, Ircache ircache, Img wrc, Img temporalOutput, Img output,
                                    Img skyCube, Img convolvedSkyCube, int debugShadingMode, int debugShowWrc) {
        ircache.bind(Pass.compute(g, "light_gbuffer.hlsl").bind(gb.gbuffer).bind(gb.depth).bind(shadowMask).bind(rtr).bind(rtdgi))
                .bind(wrc).bind(temporalOutput).bind(output).bind(skyCube).bind(convolvedSkyCube)
                .constants(new Cb().f4(gb.gbuffer.desc.extentInvExtent()).u(debugShadingMode).u(debugShowWrc))
                .dispatch(gb.gbuffer.width(), gb.gbuffer.height(), 1);
    }

    private static final int[] GBUFFER_FORMATS = {Fmt.A2R10G10B10, Fmt.RGBA32F, Fmt.RGBA16F};

    public static org.joml.Matrix4f cullViewProj;
    public static org.joml.Vector3f cullEye; //DEV
    private static final float CULL_MIN_ANGLE = Float.parseFloat(System.getProperty("lumvex.cullangle", "0.0012"));

    public static Gbuf rasterMeshes(Gfx g, World world, int w, int h, Img[] velocityOut) {
        Img normal = g.image(new Img.Desc(w, h, Fmt.A2R10G10B10));
        Img gbuffer = g.image(new Img.Desc(w, h, Fmt.RGBA32F));
        Img depth = g.image(new Img.Desc(w, h, Fmt.D32));
        Img velocity = g.image(new Img.Desc(w, h, Fmt.RGBA16F));
        velocityOut[0] = velocity;
        Gbuf gb = new Gbuf(normal, gbuffer, depth);

        Pipe pipe = g.raster("raster_simple_vs.hlsl", "raster_simple_ps.hlsl", GBUFFER_FORMATS, Fmt.D32, 8, false);
        int n = world.instances.size();
        int off = g.dynAlloc(Math.max(1, n) * 96);
        ByteBuffer m = g.dynMap().order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < n; i++) {
            World.Instance inst = world.instances.get(i);
            int p = off + i * 96;
            put34(m, p, inst.transform);
            put34(m, p + 48, inst.prevTransform);
        }
        g.barrier();
        Pass pass = new Pass(g, pipe).constantsAt(off);

        try (MemoryStack stack = stackPush()) {
            VkRenderingAttachmentInfo.Buffer colors = VkRenderingAttachmentInfo.calloc(3, stack);
            Img[] targets = {normal, gbuffer, velocity};
            for (int i = 0; i < 3; i++) {
                colors.get(i).sType$Default().imageView(targets[i].view(Img.View.DEFAULT)).imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                        .loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR).storeOp(VK_ATTACHMENT_STORE_OP_STORE);
            }
            VkRenderingAttachmentInfo depthAtt = VkRenderingAttachmentInfo.calloc(stack).sType$Default().imageView(depth.view(Img.View.DEFAULT))
                    .imageLayout(VK_IMAGE_LAYOUT_GENERAL).loadOp(VK_ATTACHMENT_LOAD_OP_CLEAR).storeOp(VK_ATTACHMENT_STORE_OP_STORE);
            depthAtt.clearValue().depthStencil().depth(0f);
            VkRenderingInfo ri = VkRenderingInfo.calloc(stack).sType$Default().layerCount(1).pColorAttachments(colors).pDepthAttachment(depthAtt);
            ri.renderArea().extent().set(w, h);
            pass.begin();
            vkCmdBeginRendering(g.cmd, ri);
            vkCmdSetViewport(g.cmd, 0, VkViewport.calloc(1, stack).x(0).y(h).width(w).height(-h).minDepth(0).maxDepth(1));
            vkCmdSetScissor(g.cmd, 0, VkRect2D.calloc(1, stack).extent(e -> e.set(w, h)));
            java.nio.IntBuffer pc = stack.mallocInt(2);
            org.joml.FrustumIntersection fi = cullViewProj != null && !Boolean.getBoolean("lumvex.nocull") ? new org.joml.FrustumIntersection(cullViewProj, false) : null;
            int drawn = 0;
            long tris = 0;
            World.Instance[] arr = world.instArray();
            for (int i = 0; i < n; i++) {
                World.Instance inst = arr[i];
                World.Geometry geo = inst.geo;
                if (fi != null && !visibleCached(fi, inst)) continue;
                drawn++;
                tris += geo.indexCount / 3;
                vkCmdBindIndexBuffer(g.cmd, world.bindless.vertices.handle, geo.index, VK_INDEX_TYPE_UINT32);
                vkCmdPushConstants(g.cmd, pipe.layout, VK_SHADER_STAGE_ALL_GRAPHICS, 0, pc.put(0, i).put(1, inst.mesh));
                vkCmdDrawIndexed(g.cmd, geo.indexCount, 1, 0, 0, 0);
            }
            dev.lumvex.render.RenderTuning.drawnInstances = drawn;
            dev.lumvex.render.RenderTuning.culledInstances = n - drawn;
            dev.lumvex.render.RenderTuning.drawnTriangles = tris;
            vkCmdEndRendering(g.cmd);
        }
        return gb;
    }

    private static final org.joml.Vector3f TMN = new org.joml.Vector3f(), TMX = new org.joml.Vector3f(), TV = new org.joml.Vector3f();
    private static final float DETAIL_DIST = Float.parseFloat(System.getProperty("lumvex.detaildist", "130"));

    /** Culling with the instance's cached world bounds (recomputed only after it moved). */
    private static boolean visibleCached(org.joml.FrustumIntersection fi, World.Instance inst) {
        float[] w = inst.wb;
        if (!inst.wbValid) {
            org.joml.Matrix4f t = inst.transform;
            float[] b = inst.geo.bounds;
            if (t.m00() == 1f && t.m11() == 1f && t.m22() == 1f && t.m01() == 0f && t.m02() == 0f && t.m10() == 0f && t.m12() == 0f && t.m20() == 0f && t.m21() == 0f) {
                w[0] = b[0] + t.m30(); w[1] = b[1] + t.m31(); w[2] = b[2] + t.m32();
                w[3] = b[3] + t.m30(); w[4] = b[4] + t.m31(); w[5] = b[5] + t.m32();
            } else {
                org.joml.Vector3f mn = TMN.set(Float.MAX_VALUE), mx = TMX.set(-Float.MAX_VALUE), v = TV;
                for (int i = 0; i < 8; i++) {
                    t.transformPosition((i & 1) == 0 ? b[0] : b[3], (i & 2) == 0 ? b[1] : b[4], (i & 4) == 0 ? b[2] : b[5], v);
                    mn.min(v);
                    mx.max(v);
                }
                w[0] = mn.x; w[1] = mn.y; w[2] = mn.z; w[3] = mx.x; w[4] = mx.y; w[5] = mx.z;
            }
            inst.wbValid = true;
        }
        if (!fi.testAab(w[0], w[1], w[2], w[3], w[4], w[5])) return false;
        org.joml.Vector3f eye = cullEye;
        if (eye == null) return true;
        World.MeshEntry me = inst.me;
        if (me.detail) {
            float ex = Math.max(Math.max(w[0] - eye.x, 0f), eye.x - w[3]), ey = Math.max(Math.max(w[1] - eye.y, 0f), eye.y - w[4]), ez = Math.max(Math.max(w[2] - eye.z, 0f), eye.z - w[5]);
            float d2 = ex * ex + ey * ey + ez * ez;
            if (d2 >= me.farDist * me.farDist || d2 < me.nearDist * me.nearDist) return false;
        }
        float dx = (w[0] + w[3]) * 0.5f - eye.x, dy = (w[1] + w[4]) * 0.5f - eye.y, dz = (w[2] + w[5]) * 0.5f - eye.z;
        float r = (float) Math.sqrt((w[3] - w[0]) * (w[3] - w[0]) + (w[4] - w[1]) * (w[4] - w[1]) + (w[5] - w[2]) * (w[5] - w[2])) * 0.5f;
        float d = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
        return r > 12f || r > CULL_MIN_ANGLE * d || d < r;
    }

    private static boolean visible(org.joml.FrustumIntersection fi, org.joml.Matrix4f t, float[] b, org.joml.Vector3f eye, World.MeshEntry me) {
        org.joml.Vector3f mn = TMN.set(Float.MAX_VALUE), mx = TMX.set(-Float.MAX_VALUE), v = TV; //DEV: reused, no garbage per instance and frame
        if (t.m00() == 1f && t.m11() == 1f && t.m22() == 1f && t.m01() == 0f && t.m02() == 0f && t.m10() == 0f && t.m12() == 0f && t.m20() == 0f && t.m21() == 0f) {
            //DEV: pure translation (baked tiles and the ghost copies of the endless city): no corner transforms needed
            mn.set(b[0] + t.m30(), b[1] + t.m31(), b[2] + t.m32());
            mx.set(b[3] + t.m30(), b[4] + t.m31(), b[5] + t.m32());
        } else {
            for (int i = 0; i < 8; i++) {
                t.transformPosition((i & 1) == 0 ? b[0] : b[3], (i & 2) == 0 ? b[1] : b[4], (i & 4) == 0 ? b[2] : b[5], v);
                mn.min(v);
                mx.max(v);
            }
        }
        if (!fi.testAab(mn, mx)) return false;
        if (eye == null) return true;
        if (me.detail) {
            float ex = Math.max(Math.max(mn.x - eye.x, 0f), eye.x - mx.x), ey = Math.max(Math.max(mn.y - eye.y, 0f), eye.y - mx.y), ez = Math.max(Math.max(mn.z - eye.z, 0f), eye.z - mx.z);
            float d2 = ex * ex + ey * ey + ez * ez;
            if (d2 >= me.farDist * me.farDist || d2 < me.nearDist * me.nearDist) return false;
        }
        // tiny or very far instances: drop below ~1 px
        float r = mn.distance(mx) * 0.5f, dx = (mn.x + mx.x) * 0.5f - eye.x, dy = (mn.y + mx.y) * 0.5f - eye.y, dz = (mn.z + mx.z) * 0.5f - eye.z;
        float d = (float) Math.sqrt(dx * dx + dy * dy + dz * dz);
        return r > 12f || r > CULL_MIN_ANGLE * d || d < r;
    }

    private static void put34(ByteBuffer m, int p, org.joml.Matrix4f t) {
        m.putFloat(p, t.m00()).putFloat(p + 4, t.m10()).putFloat(p + 8, t.m20()).putFloat(p + 12, t.m30());
        m.putFloat(p + 16, t.m01()).putFloat(p + 20, t.m11()).putFloat(p + 24, t.m21()).putFloat(p + 28, t.m31());
        m.putFloat(p + 32, t.m02()).putFloat(p + 36, t.m12()).putFloat(p + 40, t.m22()).putFloat(p + 44, t.m32());
    }
}
