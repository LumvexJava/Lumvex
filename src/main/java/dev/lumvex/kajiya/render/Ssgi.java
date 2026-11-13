// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;

public final class Ssgi {
    private final PingPong ssgiTex = new PingPong("ssgi");

    public Img render(Gfx g, Gbuf gb, Img reprojectionMap, Img prevRadiance) {
        Img.Desc gd = gb.gbuffer.desc;
        Img halfNormal = gb.halfViewNormal(g), halfDepth = gb.halfDepth(g);
        Img ssgi = g.image(Fmt.half(gd).withFormat(Fmt.R16F));
        Pass.compute(g, "ssgi/ssgi.hlsl").bind(gb.gbuffer).bind(halfDepth).bind(halfNormal).bind(prevRadiance).bind(reprojectionMap).bind(ssgi)
                .constants(new Cb().f4(gd.extentInvExtent()).f4(ssgi.desc.extentInvExtent())).dispatch(ssgi.width(), ssgi.height(), 1);
        return filter(g, ssgi, gb, reprojectionMap);
    }

    private Img filter(Gfx g, Img input, Gbuf gb, Img reprojectionMap) {
        Img.Desc gd = gb.gbuffer.desc;
        Img halfNormal = gb.halfViewNormal(g), halfDepth = gb.halfDepth(g);
        Img spatial = g.image(Fmt.half(gd).withFormat(Fmt.R16F));
        Pass.compute(g, "ssgi/spatial_filter.hlsl").bind(input).bind(halfDepth).bind(halfNormal).bind(spatial).dispatch(spatial.width(), spatial.height(), 1);

        Img upsampled = g.image(gd.withFormat(Fmt.R16F));
        Pass.compute(g, "ssgi/upsample.hlsl").bind(spatial).bind(gb.depth).bind(gb.gbuffer).bind(upsampled).dispatch(upsampled.width(), upsampled.height(), 1);

        Img[] pair = ssgiTex.get(g, new Img.Desc(gd.width(), gd.height(), Fmt.R16F));
        Img filtered = g.image(gd.withFormat(Fmt.R8));
        Pass.compute(g, "ssgi/temporal_filter.hlsl").bind(upsampled).bind(pair[1]).bind(reprojectionMap).bind(filtered).bind(pair[0])
                .constants(new Cb().f4(pair[0].desc.extentInvExtent())).dispatch(pair[0].width(), pair[0].height(), 1);
        return filtered;
    }
}
