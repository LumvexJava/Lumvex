// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;

import java.util.List;

public final class Rtdgi {
    private final PingPong radiance = new PingPong("rtdgi.radiance"), rayOrig = new PingPong("rtdgi.ray_orig"), ray = new PingPong("rtdgi.ray"),
            reservoir = new PingPong("rtdgi.reservoir"), candidate = new PingPong("rtdgi.candidate"), invalidity = new PingPong("rtdgi.invalidity"),
            temporal2 = new PingPong("rtdgi.temporal2"), temporal2Var = new PingPong("rtdgi.temporal2_var"), hitNormal = new PingPong("rtdgi.hit_normal");
    public int spatialReusePassCount = 2;
    public boolean useRaytracedReservoirVisibility = false;

    public record Reprojected(Img history, Img temporalOutput) {}

    public record Candidates(Img radiance, Img normal, Img hit) {}

    public record Output(Img irradiance, Candidates candidates) {}

    private static final List<String> MISS = List.of("rt/gbuffer.rmiss.hlsl", "rt/shadow.rmiss.hlsl");
    private static final List<String> HIT = List.of("rt/gbuffer.rchit.hlsl");

    private static Img.Desc temporalDesc(int w, int h) {
        return new Img.Desc(w, h, Fmt.RGBA16F);
    }

    public Reprojected reproject(Gfx g, Img reprojectionMap) {
        Img[] t2 = temporal2.get(g, temporalDesc(reprojectionMap.width(), reprojectionMap.height()));
        Img reprojected = g.image(temporalDesc(reprojectionMap.width(), reprojectionMap.height()));
        Pass.compute(g, "rtdgi/fullres_reproject.hlsl").bind(t2[1]).bind(reprojectionMap).bind(reprojected)
                .constants(new Cb().f4(reprojected.desc.extentInvExtent())).dispatch(reprojected.width(), reprojected.height(), 1);
        return new Reprojected(reprojected, t2[0]);
    }

    private Img temporalFilter(Gfx g, Img input, Gbuf gb, Img reprojectionMap, Img reprojectedHistory, Img invalidityTex, Img temporalOutput) {
        Img[] var = temporal2Var.get(g, new Img.Desc(input.width(), input.height(), Fmt.RG16F));
        Img filtered = g.image(gb.gbuffer.desc.withFormat(Fmt.RGBA16F));
        Pass.compute(g, "rtdgi/temporal_filter.hlsl").bind(input).bind(reprojectedHistory).bind(var[1]).bind(reprojectionMap).bind(invalidityTex)
                .bind(filtered).bind(temporalOutput).bind(var[0])
                .constants(new Cb().f4(temporalOutput.desc.extentInvExtent()).f4(gb.gbuffer.desc.extentInvExtent()))
                .dispatch(temporalOutput.width(), temporalOutput.height(), 1);
        return filtered;
    }

    private static Img spatial(Gfx g, Img input, Gbuf gb, Img ssao) {
        Img out = g.image(temporalDesc(input.width(), input.height()));
        Pass.compute(g, "rtdgi/spatial_filter.hlsl").bind(input).bind(gb.depth).bind(ssao).bind(gb.normal).bind(out)
                .constants(new Cb().f4(out.desc.extentInvExtent())).dispatch(out.width(), out.height(), 1);
        return out;
    }

    public Output render(Gfx g, Reprojected rep, Gbuf gb, Img reprojectionMap, Img skyCube, Ircache ircache, Img wrcAtlas, long tlas, Img ssao) {
        Img halfSsao = g.image(Fmt.half(ssao.desc).withFormat(Fmt.R8_SNORM));
        Pass.compute(g, "extract_half_res_ssao.hlsl").bind(ssao).bind(halfSsao)
                .constants(new Cb().f4(ssao.desc.extentInvExtent()).f4(halfSsao.desc.extentInvExtent())).dispatch(halfSsao.width(), halfSsao.height(), 1);

        Img.Desc gd = gb.gbuffer.desc;
        Img.Desc half = Fmt.half(gd);
        int hw = half.width(), hh = half.height();
        Img[] hitNormalPair = hitNormal.get(g, temporalDesc(hw, hh).withFormat(Fmt.RGBA8));
        Img[] candPair = candidate.get(g, new Img.Desc(hw, hh, Fmt.RGBA16F));
        Img candRadiance = g.image(half.withFormat(Fmt.RGBA16F));
        Img candNormal = g.image(half.withFormat(Fmt.RGBA8_SNORM));
        Img candHit = g.image(half.withFormat(Fmt.RGBA16F));
        Img temporalReservoirPacked = g.image(half.withFormat(Fmt.RGBA32UI));
        Img halfDepth = gb.halfDepth(g);
        Img[] invPair = invalidity.get(g, new Img.Desc(hw, hh, Fmt.RG16F));
        Img[] radPair = radiance.get(g, temporalDesc(hw, hh));
        Img[] rayOrigPair = rayOrig.get(g, new Img.Desc(hw, hh, Fmt.RGBA32F));
        Img[] rayPair = ray.get(g, new Img.Desc(hw, hh, Fmt.RGBA16F));
        Img halfNormal = gb.halfViewNormal(g);
        Img validityPre = g.image(half.withFormat(Fmt.R8));
        Img[] resPair = reservoir.get(g, new Img.Desc(hw, hh, Fmt.RG32UI));

        ircache.bind(Pass.rt(g, "rtdgi/diffuse_validate.rgen.hlsl", MISS, HIT)
                .bind(halfNormal).bind(gb.depth).bind(rep.history()).bind(resPair[1]).bind(rayPair[1]).bind(reprojectionMap))
                .bind(wrcAtlas).bind(skyCube).bind(radPair[1]).bind(rayOrigPair[1]).bind(validityPre)
                .constants(new Cb().f4(gd.extentInvExtent())).tlas(tlas).traceRays(candRadiance.width(), candRadiance.height(), 1);

        Img validityInput = g.image(half.withFormat(Fmt.R8));
        ircache.bind(Pass.rt(g, "rtdgi/trace_diffuse.rgen.hlsl", MISS, HIT)
                .bind(halfNormal).bind(gb.depth).bind(rep.history()).bind(reprojectionMap))
                .bind(wrcAtlas).bind(skyCube).bind(rayOrigPair[1]).bind(candRadiance).bind(candNormal).bind(candHit).bind(validityPre).bind(validityInput)
                .constants(new Cb().f4(gd.extentInvExtent())).tlas(tlas).traceRays(candRadiance.width(), candRadiance.height(), 1);

        Pass.compute(g, "rtdgi/temporal_validity_integrate.hlsl").bind(validityInput).bind(invPair[1]).bind(reprojectionMap).bind(halfNormal).bind(halfDepth)
                .bind(invPair[0])
                .constants(new Cb().f4(gd.extentInvExtent()).f4(invPair[0].desc.extentInvExtent()))
                .dispatch(invPair[0].width(), invPair[0].height(), 1);

        Pass.compute(g, "rtdgi/restir_temporal.hlsl").bind(halfNormal).bind(gb.depth).bind(candRadiance).bind(candNormal).bind(candHit).bind(radPair[1])
                .bind(rayOrigPair[1]).bind(rayPair[1]).bind(resPair[1]).bind(reprojectionMap).bind(hitNormalPair[1]).bind(candPair[1]).bind(invPair[0])
                .bind(radPair[0]).bind(rayOrigPair[0]).bind(rayPair[0]).bind(hitNormalPair[0]).bind(resPair[0]).bind(candPair[0]).bind(temporalReservoirPacked)
                .constants(new Cb().f4(gd.extentInvExtent()))
                .dispatch(radPair[0].width(), radPair[0].height(), 1);

        Img radianceTex = radPair[0];
        Img temporalReservoir = resPair[0];

        Img res0 = g.image(half.withFormat(Fmt.RG32UI)), res1 = g.image(half.withFormat(Fmt.RG32UI));
        Img br0 = g.image(half.withFormat(Fmt.B10G11R11)), br1 = g.image(half.withFormat(Fmt.B10G11R11));
        Img reservoirInput = temporalReservoir, bouncedInput = radianceTex;
        for (int i = 0; i < spatialReusePassCount; i++) {
            int raymarch = i + 1 == spatialReusePassCount ? 1 : 0;
            Pass.compute(g, "rtdgi/restir_spatial.hlsl").bind(reservoirInput).bind(bouncedInput).bind(halfNormal).bind(halfDepth).bind(gb.depth).bind(halfSsao)
                    .bind(temporalReservoirPacked).bind(rep.history()).bind(res0).bind(br0)
                    .constants(new Cb().f4(gd.extentInvExtent()).f4(res0.desc.extentInvExtent()).u(i).u(raymarch).u(useRaytracedReservoirVisibility ? 1 : 0))
                    .dispatch(res0.width(), res0.height(), 1);
            Img t = res0;
            res0 = res1;
            res1 = t;
            t = br0;
            br0 = br1;
            br1 = t;
            reservoirInput = res1;
            bouncedInput = br1;
        }

        if (useRaytracedReservoirVisibility) {
            Pass.rt(g, "rtdgi/restir_check.rgen.hlsl", MISS, HIT).bind(halfDepth).bind(temporalReservoirPacked).bind(reservoirInput)
                    .constants(new Cb().f4(gd.extentInvExtent())).tlas(tlas).traceRays(candRadiance.width(), candRadiance.height(), 1);
        }

        Img irradiance = g.image(gd.withFormat(Fmt.RGBA16F));
        Pass.compute(g, "rtdgi/restir_resolve.hlsl").bind(radianceTex).bind(reservoirInput).bind(gb.gbuffer).bind(gb.depth).bind(halfNormal).bind(halfDepth)
                .bind(ssao).bind(candRadiance).bind(candHit).bind(temporalReservoirPacked).bind(bouncedInput).bind(irradiance)
                .constants(new Cb().f4(gd.extentInvExtent()).f4(irradiance.desc.extentInvExtent()))
                .dispatch(irradiance.width(), irradiance.height(), 1);

        Img filtered = temporalFilter(g, irradiance, gb, reprojectionMap, rep.history(), invPair[0], rep.temporalOutput());
        Img sp = spatial(g, filtered, gb, ssao);
        return new Output(sp, new Candidates(candRadiance, candNormal, candHit));
    }
}
