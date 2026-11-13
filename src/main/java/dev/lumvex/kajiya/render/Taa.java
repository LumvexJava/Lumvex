// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;

public final class Taa {
    private final PingPong temporal = new PingPong("taa"), velocity = new PingPong("taa.velocity"), smoothVar = new PingPong("taa.smooth_var");
    public float jitterX, jitterY;

    private static Img.Desc fmt(int w, int h, int f) { return new Img.Desc(w, h, f); }

    public Img render(Gfx g, Img input, Img reprojectionMap, Img depth, int outW, int outH) {
        Img[] t = temporal.get(g, fmt(outW, outH, Fmt.RGBA16F));
        Img[] vel = velocity.get(g, fmt(outW, outH, Fmt.RG16F));
        Img reprojectedHistory = g.image(fmt(outW, outH, Fmt.RGBA16F));
        Img closestVelocity = g.image(fmt(outW, outH, Fmt.RG16F));

        Pass.compute(g, "taa/reproject_history.hlsl").bind(t[1]).bind(reprojectionMap).bind(depth).bind(reprojectedHistory).bind(closestVelocity)
                .constants(new Cb().f4(input.desc.extentInvExtent()).f4(reprojectedHistory.desc.extentInvExtent()))
                .dispatch(reprojectedHistory.width(), reprojectedHistory.height(), 1);

        Img[] sv = smoothVar.get(g, fmt(outW, outH, Fmt.RGBA16F));
        Img filteredInput = g.image(fmt(input.width(), input.height(), Fmt.RGBA16F));
        Img filteredDeviation = g.image(fmt(input.width(), input.height(), Fmt.RGBA16F));
        Pass.compute(g, "taa/filter_input.hlsl").bind(input).bind(depth).bind(filteredInput).bind(filteredDeviation)
                .dispatch(filteredInput.width(), filteredInput.height(), 1);

        Img filteredHistory = g.image(fmt(filteredInput.width(), filteredInput.height(), Fmt.RGBA16F));
        Pass.compute(g, "taa/filter_history.hlsl").bind(reprojectedHistory).bind(filteredHistory)
                .constants(new Cb().f4(reprojectedHistory.desc.extentInvExtent()).f4(input.desc.extentInvExtent()))
                .dispatch(filteredHistory.width(), filteredHistory.height(), 1);

        Img inputProb = g.image(fmt(input.width(), input.height(), Fmt.R16F));
        Pass.compute(g, "taa/input_prob.hlsl").bind(input).bind(filteredInput).bind(filteredDeviation).bind(reprojectedHistory).bind(filteredHistory)
                .bind(reprojectionMap).bind(depth).bind(sv[1]).bind(vel[1]).bind(inputProb)
                .constants(new Cb().f4(input.desc.extentInvExtent())).dispatch(inputProb.width(), inputProb.height(), 1);

        Img prob1 = g.image(inputProb.desc);
        Pass.compute(g, "taa/filter_prob.hlsl").bind(inputProb).bind(prob1).dispatch(prob1.width(), prob1.height(), 1);
        Img prob2 = g.image(inputProb.desc);
        Pass.compute(g, "taa/filter_prob2.hlsl").bind(prob1).bind(prob2).dispatch(prob1.width(), prob1.height(), 1);

        Img thisFrame = g.image(fmt(outW, outH, Fmt.RGBA16F));
        Pass.compute(g, "taa/taa.hlsl").bind(input).bind(reprojectedHistory).bind(reprojectionMap).bind(closestVelocity).bind(vel[1]).bind(depth)
                .bind(sv[1]).bind(prob2).bind(t[0]).bind(thisFrame).bind(sv[0]).bind(vel[0])
                .constants(new Cb().f4(input.desc.extentInvExtent()).f4(t[0].desc.extentInvExtent()))
                .dispatch(t[0].width(), t[0].height(), 1);
        return thisFrame;
    }
}
