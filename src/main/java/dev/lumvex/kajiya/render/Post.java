// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.vk.Buffer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.lwjgl.vulkan.VK13.*;

public final class Post {
    private static final int BIN_COUNT = 256;
    private static final double MIN_LOG2 = -16.0, MAX_LOG2 = 16.0;

    private Buffer histogramHost;
    public float imageLog2Lum;
    public float lowClip, highClip;

    public static Img blurPyramid(Gfx g, Img input) {
        Img.Desc half = Fmt.half(input.desc);
        int mips = Math.max(1, Fmt.mipCount(half.width(), half.height()) - 1);
        Img out = g.image(half.withFormat(Fmt.B10G11R11).withMips(mips));
        Pass.compute(g, "blur.hlsl").bind(input).bindMip(out, 0).dispatch(out.width(), out.height(), 1);
        for (int m = 1; m < mips; m++) {
            Pass.compute(g, "blur.hlsl").bindMip(out, m - 1).bindMip(out, m)
                    .dispatch(Math.max(1, out.width() >> m), Math.max(1, out.height() >> m), 1);
        }
        return out;
    }

    public static Img revBlurPyramid(Gfx g, Img in) {
        Img out = g.image(in.desc);
        int mips = in.desc.mips();
        for (int m = mips - 2; m >= 0; m--) {
            int w = Math.max(1, out.width() >> m), h = Math.max(1, out.height() >> m);
            Pass.compute(g, "rev_blur.hlsl").bindMip(in, m).bindMip(out, m + 1).bindMip(out, m)
                    .constants(new Cb().u(w).u(h).f(0.5f)).dispatch(w, h, 1);
        }
        return out;
    }

    private Buffer calculateHistogram(Gfx g, Img pyramid) {
        if (histogramHost == null) histogramHost = new Buffer(g.ctx, 4L * BIN_COUNT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.READBACK);
        Buffer tmp = g.buffer(4L * BIN_COUNT, 0);
        int level = Math.max(0, pyramid.desc.mips() - 7);
        int mw = Math.max(1, (pyramid.width() + (1 << level) - 1) >> level), mh = Math.max(1, (pyramid.height() + (1 << level) - 1) >> level);
        Pass.compute(g, "post/luminance_histogram_clear.hlsl").bind(tmp).dispatch(BIN_COUNT, 1, 1);
        Pass.compute(g, "post/luminance_histogram_calculate.hlsl").bindMip(pyramid, level).bind(tmp)
                .constants(new Cb().u2(mw, mh)).dispatch(mw, mh, 1);
        Pass.compute(g, "post/luminance_histogram_copy.hlsl").bind(tmp).bind(histogramHost).dispatch(BIN_COUNT, 1, 1);
        return tmp;
    }

    private void readBackHistogram() {
        if (histogramHost == null) return;
        ByteBuffer m = histogramHost.map().order(ByteOrder.LITTLE_ENDIAN);
        int[] hist = new int[BIN_COUNT];
        long total = 0;
        for (int i = 0; i < BIN_COUNT; i++) {
            hist[i] = m.getInt(i * 4);
            total += hist[i] & 0xFFFFFFFFL;
        }
        double lo = Math.min(lowClip, 1.0), hi = Math.min(highClip, 1.0 - lo);
        long reject = (long) (total * lo), use = (long) (total * (1.0 - lo - hi));
        double sum = 0;
        long used = 0, leftReject = reject, leftUse = use;
        for (int i = 0; i < BIN_COUNT; i++) {
            double t = (i + 0.5) / BIN_COUNT;
            long c = hist[i] & 0xFFFFFFFFL;
            long countToUse = Math.min(Math.max(0, c - leftReject), leftUse);
            leftReject = Math.max(0, leftReject - c);
            leftUse = Math.max(0, leftUse - countToUse);
            sum += t * countToUse;
            used += countToUse;
        }
        double mean = sum / Math.max(1, used);
        imageLog2Lum = (float) (MIN_LOG2 + mean * (MAX_LOG2 - MIN_LOG2));
    }

    public Img render(Gfx g, Img input, float postExposureMult, float contrast) {
        readBackHistogram();
        Img blur = blurPyramid(g, input);
        Buffer histogram = calculateHistogram(g, blur);
        Img rev = revBlurPyramid(g, blur);
        Img out = g.image(input.desc.withFormat(Fmt.B10G11R11));
        Pass.compute(g, "post_combine.hlsl").bind(input).bind(blur).bind(rev).bind(histogram).bind(out)
                .constants(new Cb().f4(out.desc.extentInvExtent()).f(postExposureMult).f(contrast)).dispatch(out.width(), out.height(), 1);
        return out;
    }

    public static Img motionBlur(Gfx g, Img input, Img depth, Img reprojectionMap) {
        final int tile = 16;
        Img rx = g.image(Fmt.divUp(reprojectionMap.desc, tile, 1).withFormat(Fmt.RG16F));
        Pass.compute(g, "motion_blur/velocity_reduce_x.hlsl").bind(reprojectionMap).bind(rx).dispatch(rx.width(), rx.height(), 1);
        Img ry = g.image(Fmt.divUp(rx.desc, 1, tile));
        Pass.compute(g, "motion_blur/velocity_reduce_y.hlsl").bind(rx).bind(ry).dispatch(rx.width(), rx.height(), 1);
        Img dilated = g.image(ry.desc);
        Pass.compute(g, "motion_blur/velocity_dilate.hlsl").bind(ry).bind(dilated).dispatch(dilated.width(), dilated.height(), 1);
        Img out = g.image(input.desc);
        Pass.compute(g, "motion_blur/motion_blur.hlsl").bind(input).bind(reprojectionMap).bind(dilated).bind(depth).bind(out)
                .constants(new Cb().f4(depth.desc.extentInvExtent()).f4(out.desc.extentInvExtent()).f(1.0f))
                .dispatch(out.width(), out.height(), 1);
        return out;
    }
}
