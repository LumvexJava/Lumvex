// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.vk.Buffer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.List;

import static org.lwjgl.vulkan.VK13.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

public final class Rtr {
    private final PingPong temporal = new PingPong("rtr.temporal"), rayLen = new PingPong("rtr.ray_len"), irradiance = new PingPong("rtr.irradiance"),
            rayOrig = new PingPong("rtr.ray_orig"), ray = new PingPong("rtr.ray"), reservoir = new PingPong("rtr.reservoir"),
            rng = new PingPong("rtr.rng"), hitNormal = new PingPong("rtr.hit_normal");
    private Buffer rankingTile, scramblingTile, sobol;
    public boolean reuseRtdgiRays = true;
    public static ByteBuffer spatialResolveOffsets;

    private static final List<String> MISS = List.of("rt/gbuffer.rmiss.hlsl", "rt/shadow.rmiss.hlsl");
    private static final List<String> HIT = List.of("rt/gbuffer.rchit.hlsl");

    public static final class Traced {
        public Img resolved, temporalOutput, history, rayLen, restirInvalidity;
    }

    public static ByteBuffer resource(String name) {
        try (InputStream in = Rtr.class.getResourceAsStream("/kajiya/data/" + name)) {
            byte[] b = in.readAllBytes();
            ByteBuffer bb = ByteBuffer.allocateDirect(b.length).order(java.nio.ByteOrder.LITTLE_ENDIAN);
            bb.put(b).flip();
            return bb;
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public static ByteBuffer offsets() {
        if (spatialResolveOffsets == null) spatialResolveOffsets = resource("rtr_spatial_resolve_offsets.bin");
        return spatialResolveOffsets;
    }

    public static Cb withOffsets(Cb cb) {
        return cb.rawBytes(offsets());
    }

    private void ensureLuts(Gfx g) {
        if (rankingTile != null) return;
        rankingTile = Buffer.deviceLocal(g.ctx, resource("rtr_ranking_tile.bin"), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        scramblingTile = Buffer.deviceLocal(g.ctx, resource("rtr_scrambling_tile.bin"), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        sobol = Buffer.deviceLocal(g.ctx, resource("rtr_sobol.bin"), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    }

    private static Img.Desc temporalDesc(int w, int h) {
        return new Img.Desc(w, h, Fmt.RGBA16F);
    }

    public Traced trace(Gfx g, Gbuf gb, Img reprojectionMap, Img skyCube, long tlas, Img rtdgiIrradiance, Rtdgi.Candidates cands, Ircache ircache, Img wrcAtlas) {
        ensureLuts(g);
        Img.Desc gd = gb.gbuffer.desc;
        Img.Desc half = Fmt.half(gd);
        Img refl0 = cands.radiance(), refl1 = cands.hit(), refl2 = cands.normal();

        Img[] rngPair = rng.get(g, new Img.Desc(half.width(), half.height(), Fmt.R32UI));

        ircache.bind(Pass.rt(g, "rtr/reflection.rgen.hlsl", MISS, HIT).bind(gb.gbuffer).bind(gb.depth).bind(rankingTile).bind(scramblingTile).bind(sobol)
                .bind(rtdgiIrradiance).bind(skyCube))
                .bind(wrcAtlas).bind(refl0).bind(refl1).bind(refl2).bind(rngPair[0])
                .constants(new Cb().f4(gd.extentInvExtent()).u(reuseRtdgiRays ? 1 : 0)).tlas(tlas).traceRays(refl0.width(), refl0.height(), 1);

        Img halfNormal = gb.halfViewNormal(g);
        Img halfDepth = gb.halfDepth(g);

        Img[] rayOrigPair = rayOrig.get(g, new Img.Desc(half.width(), half.height(), Fmt.RGBA32F));
        Img restirInvalidity = g.image(refl0.desc.withFormat(Fmt.R8));

        Img[] hitNormalPair = hitNormal.get(g, temporalDesc(half.width(), half.height()).withFormat(Fmt.RGBA8));
        Img[] irrPair = irradiance.get(g, temporalDesc(half.width(), half.height()));
        Img[] resPair = reservoir.get(g, new Img.Desc(half.width(), half.height(), Fmt.RG32UI));
        Img[] rayPair = ray.get(g, temporalDesc(half.width(), half.height()));

        ircache.bind(Pass.rt(g, "rtr/reflection_validate.rgen.hlsl", MISS, HIT).bind(gb.gbuffer).bind(gb.depth).bind(rtdgiIrradiance).bind(skyCube)
                .bind(restirInvalidity))
                .bind(wrcAtlas).bind(rayOrigPair[1]).bind(rayPair[1]).bind(rngPair[1]).bind(irrPair[1]).bind(resPair[1])
                .constants(new Cb().f4(gd.extentInvExtent())).tlas(tlas).traceRays(Fmt.half(refl0.width()), Fmt.half(refl0.height()), 1);

        Pass.compute(g, "rtr/rtr_restir_temporal.hlsl").bind(gb.gbuffer).bind(halfNormal).bind(gb.depth).bind(refl0).bind(refl1).bind(refl2).bind(irrPair[1])
                .bind(rayOrigPair[1]).bind(rayPair[1]).bind(rngPair[1]).bind(resPair[1]).bind(reprojectionMap).bind(hitNormalPair[1])
                .bind(irrPair[0]).bind(rayOrigPair[0]).bind(rayPair[0]).bind(rngPair[0]).bind(hitNormalPair[0]).bind(resPair[0])
                .constants(new Cb().f4(gd.extentInvExtent())).dispatch(irrPair[0].width(), irrPair[0].height(), 1);

        Img resolved = g.image(gd.withFormat(Fmt.B10G11R11));
        Img[] tPair = temporal.get(g, temporalDesc(gd.width(), gd.height()));
        Img[] lenPair = rayLen.get(g, new Img.Desc(gd.width(), gd.height(), Fmt.RG16F));

        Pass.compute(g, "rtr/resolve.hlsl").bind(gb.gbuffer).bind(gb.depth).bind(refl0).bind(refl1).bind(refl2).bind(tPair[1]).bind(reprojectionMap)
                .bind(halfNormal).bind(halfDepth).bind(lenPair[1]).bind(irrPair[0]).bind(rayPair[0]).bind(resPair[0]).bind(rayOrigPair[0]).bind(hitNormalPair[0])
                .bind(resolved).bind(lenPair[0])
                .constants(withOffsets(new Cb().f4(resolved.desc.extentInvExtent()))).dispatch(resolved.width(), resolved.height(), 1);

        Traced t = new Traced();
        t.resolved = resolved;
        t.temporalOutput = tPair[0];
        t.history = tPair[1];
        t.rayLen = lenPair[0];
        t.restirInvalidity = restirInvalidity;
        return t;
    }

    public Traced dummy(Gfx g, Gbuf gb) {
        Img.Desc gd = gb.gbuffer.desc;
        Traced t = new Traced();
        t.resolved = g.image(gd.withFormat(Fmt.RGBA8));
        Img[] tPair = temporal.get(g, temporalDesc(gd.width(), gd.height()));
        t.temporalOutput = tPair[0];
        t.history = tPair[1];
        t.rayLen = rayLen.get(g, new Img.Desc(gd.width(), gd.height(), Fmt.RGBA8))[0];
        t.restirInvalidity = g.image(new Img.Desc(1, 1, Fmt.R8));
        return t;
    }

    public static Img filterTemporal(Gfx g, Traced t, Gbuf gb, Img reprojectionMap) {
        Pass.compute(g, "rtr/temporal_filter.hlsl").bind(t.resolved).bind(t.history).bind(gb.depth).bind(t.rayLen).bind(reprojectionMap)
                .bind(t.restirInvalidity).bind(gb.gbuffer).bind(t.temporalOutput)
                .constants(new Cb().f4(t.temporalOutput.desc.extentInvExtent())).dispatch(t.resolved.width(), t.resolved.height(), 1);
        Pass.compute(g, "rtr/spatial_cleanup.hlsl").bind(t.temporalOutput).bind(gb.depth).bind(gb.normal).bind(t.resolved)
                .constants(withOffsets(new Cb())).dispatch(t.resolved.width(), t.resolved.height(), 1);
        return t.resolved;
    }
}
