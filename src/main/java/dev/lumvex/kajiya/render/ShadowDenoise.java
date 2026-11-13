// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;

public final class ShadowDenoise {
    private final PingPong accum = new PingPong("shadow_denoise_accum"), moments = new PingPong("shadow_denoise_moments");

    public Img render(Gfx g, Gbuf gb, Img shadowMask, Img reprojectionMap) {
        Img.Desc gd = gb.gbuffer.desc;
        int bw = (gd.width() + 7) / 8, bh = (gd.height() + 3) / 4;
        Img bitpacked = g.image(new Img.Desc(bw, bh, Fmt.R32UI));
        Pass.compute(g, "shadow_denoise/bitpack_shadow_mask.hlsl").bind(shadowMask).bind(bitpacked)
                .constants(new Cb().f4(gd.extentInvExtent()).u2(bw, bh)).dispatch(bw * 2, bh, 1);

        Img[] mom = moments.get(g, gd.withFormat(Fmt.RGBA16F));
        Img.Desc spatialDesc = new Img.Desc(gd.width(), gd.height(), Fmt.RG16F);
        Img[] acc = accum.get(g, spatialDesc);
        Img spatialInput = g.image(spatialDesc);
        Img metadata = g.image(new Img.Desc(bw, bh, Fmt.R32UI));

        Pass.compute(g, "shadow_denoise/megakernel.hlsl").bind(shadowMask).bind(bitpacked).bind(mom[1]).bind(acc[1]).bind(reprojectionMap)
                .bind(mom[0]).bind(spatialInput).bind(metadata)
                .constants(new Cb().f4(gd.extentInvExtent()).u2(bw, bh)).dispatch(gd.width(), gd.height(), 1);

        Img temp = g.image(spatialDesc);
        spatial(g, 1, spatialInput, acc[0], metadata, gb, bw, bh);
        spatial(g, 2, acc[0], temp, metadata, gb, bw, bh);
        spatial(g, 4, temp, spatialInput, metadata, gb, bw, bh);
        return spatialInput;
    }

    private static void spatial(Gfx g, int step, Img in, Img out, Img metadata, Gbuf gb, int bw, int bh) {
        Pass.compute(g, "shadow_denoise/spatial_filter.hlsl").bind(in).bind(metadata).bind(gb.normal).bind(gb.depth).bind(out)
                .constants(new Cb().f4(out.desc.extentInvExtent()).u2(bw, bh).u(step)).dispatch(out.width(), out.height(), 1);
    }
}
