// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;

public final class Gbuf {
    public final Img normal, gbuffer, depth;
    private Img halfNormal, halfDepth;

    public Gbuf(Img normal, Img gbuffer, Img depth) {
        this.normal = normal;
        this.gbuffer = gbuffer;
        this.depth = depth;
    }

    public Img halfViewNormal(Gfx g) {
        if (halfNormal == null) {
            halfNormal = g.image(Fmt.half(gbuffer.desc).withFormat(Fmt.RGBA8_SNORM));
            Pass.compute(g, "extract_half_res_gbuffer_view_normal_rgba8.hlsl").bind(gbuffer).bind(halfNormal)
                    .constants(new Cb().f4(gbuffer.desc.extentInvExtent()).f4(halfNormal.desc.extentInvExtent()))
                    .dispatch(halfNormal.width(), halfNormal.height(), 1);
        }
        return halfNormal;
    }

    public Img halfDepth(Gfx g) {
        if (halfDepth == null) {
            halfDepth = g.image(Fmt.half(gbuffer.desc).withFormat(Fmt.R32F));
            Pass.compute(g, "extract_half_res_depth.hlsl").bind(depth).bind(halfDepth)
                    .constants(new Cb().f4(depth.desc.extentInvExtent()).f4(halfDepth.desc.extentInvExtent()))
                    .dispatch(halfDepth.width(), halfDepth.height(), 1);
        }
        return halfDepth;
    }
}
