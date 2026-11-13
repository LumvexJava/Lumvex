// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.vk.Buffer;
import org.joml.Vector3f;

import java.util.List;

public final class Ircache {
    public static final int CASCADE_COUNT = 12;
    public static final int CASCADE_SIZE = 32;
    public static final int MAX_ENTRIES = 1024 * 64;
    public static final int SAMPLES_PER_FRAME = 4;
    public static final int VALIDATION_SAMPLES_PER_FRAME = 4;
    private static final float CELL_DIAMETER = 0.16f * 0.125f;
    private static final int MAX_GRID_CELLS = CASCADE_SIZE * CASCADE_SIZE * CASCADE_SIZE * CASCADE_COUNT;
    private static final int INDIRECTION_COUNT = 1024 * 1024;

    private boolean initialized;
    private final Vector3f gridCenter = new Vector3f();
    private final int[][] curScroll = new int[CASCADE_COUNT][3], prevScroll = new int[CASCADE_COUNT][3];
    private int parity;
    public boolean enableScroll = true;

    public Buffer meta, gridMeta, gridMeta2, entryCell, spatial, irradiance, aux, life, pool, entryIndirection, repoProposal, repoProposalCount;
    private boolean pendingSum;
    private Buffer pendingArgs;

    public void updateEyePosition(Vector3f eye) {
        if (!enableScroll) return;
        gridCenter.set(eye);
        for (int c = 0; c < CASCADE_COUNT; c++) {
            float d = CELL_DIAMETER * (1 << c);
            for (int k = 0; k < 3; k++) {
                prevScroll[c][k] = curScroll[c][k];
                curScroll[c][k] = (int) Math.floor((k == 0 ? eye.x : k == 1 ? eye.y : eye.z) / d) - CASCADE_SIZE / 2;
            }
        }
    }

    public Vector3f gridCenter() { return gridCenter; }

    public int[] cascadeConstants(int c) {
        return new int[]{curScroll[c][0], curScroll[c][1], curScroll[c][2], 0,
                curScroll[c][0] - prevScroll[c][0], curScroll[c][1] - prevScroll[c][1], curScroll[c][2] - prevScroll[c][2], 0};
    }

    public Pass bind(Pass p) {
        return p.bind(meta).bind(pool).bind(repoProposal).bind(repoProposalCount).bind(gridMeta).bind(entryCell).bind(spatial).bind(irradiance).bind(life);
    }

    public void prepare(Gfx g) {
        meta = g.tbuf("ircache.meta_buf", 4 * 8);
        gridMeta = g.tbuf("ircache.grid_meta_buf", 8L * MAX_GRID_CELLS);
        gridMeta2 = g.tbuf("ircache.grid_meta_buf2", 8L * MAX_GRID_CELLS);
        entryCell = g.tbuf("ircache.entry_cell_buf", 4L * MAX_ENTRIES);
        spatial = g.tbuf("ircache.spatial_buf", 16L * MAX_ENTRIES);
        irradiance = g.tbuf("ircache.irradiance_buf", 3 * 16L * MAX_ENTRIES);
        aux = g.tbuf("ircache.aux_buf", 4L * 16 * 16 * MAX_ENTRIES);
        life = g.tbuf("ircache.life_buf", 4L * MAX_ENTRIES);
        pool = g.tbuf("ircache.pool_buf", 4L * MAX_ENTRIES);
        entryIndirection = g.tbuf("ircache.entry_indirection_buf", 4L * INDIRECTION_COUNT);
        repoProposal = g.tbuf("ircache.reposition_proposal_buf", 16L * MAX_ENTRIES);
        repoProposalCount = g.tbuf("ircache.reposition_proposal_count_buf", 4L * MAX_ENTRIES);

        if (parity == 1) {
            Buffer t = gridMeta;
            gridMeta = gridMeta2;
            gridMeta2 = t;
        }

        if (!initialized) {
            Pass.compute(g, "ircache/clear_ircache_pool.hlsl").bind(pool).bind(life).dispatch(MAX_ENTRIES, 1, 1);
            initialized = true;
        } else {
            Pass.compute(g, "ircache/scroll_cascades.hlsl").bind(gridMeta).bind(gridMeta2).bind(entryCell).bind(irradiance).bind(life).bind(pool).bind(meta)
                    .dispatch(CASCADE_SIZE, CASCADE_SIZE, CASCADE_SIZE * CASCADE_COUNT);
            Buffer t = gridMeta;
            gridMeta = gridMeta2;
            gridMeta2 = t;
            parity = (parity + 1) % 2;
        }

        Buffer args = g.buffer(32, 0);
        Pass.compute(g, "ircache/prepare_age_dispatch_args.hlsl").bind(meta).bind(args).dispatch(1, 1, 1);

        Buffer occupancy = g.buffer(4L * MAX_ENTRIES, 0);
        Pass.compute(g, "ircache/age_ircache_entries.hlsl").bind(meta).bind(gridMeta).bind(entryCell).bind(life).bind(pool).bind(spatial)
                .bind(repoProposal).bind(repoProposalCount).bind(irradiance).bind(occupancy).dispatchIndirect(args, 0);

        prefixScan(g, occupancy);

        Pass.compute(g, "ircache/ircache_compact_entries.hlsl").bind(meta).bind(life).bind(occupancy).bind(entryIndirection).dispatchIndirect(args, 0);
    }

    private static void prefixScan(Gfx g, Buffer input) {
        final int segment = 1024;
        Pass.compute(g, "prefix_scan/inclusive_prefix_scan.hlsl").bind(input).dispatch(segment * segment / 2, 1, 1);
        Buffer sums = g.buffer(4L * segment, 0);
        Pass.compute(g, "prefix_scan/inclusive_prefix_scan_segments.hlsl").bind(input).bind(sums).dispatch(segment / 2, 1, 1);
        Pass.compute(g, "prefix_scan/inclusive_prefix_scan_merge.hlsl").bind(input).bind(sums).dispatch(segment * segment / 2, 1, 1);
    }

    private static final List<String> MISS2 = List.of("rt/gbuffer.rmiss.hlsl", "rt/shadow.rmiss.hlsl");
    private static final List<String> HIT = List.of("rt/gbuffer.rchit.hlsl");

    public void traceIrradiance(Gfx g, Img skyCube, long tlas, Img wrcAtlas) {
        Buffer args = g.buffer(64, 0);
        Pass.compute(g, "ircache/prepare_trace_dispatch_args.hlsl").bind(meta).bind(args).dispatch(1, 1, 1);

        Pass.compute(g, "ircache/reset_entry.hlsl").bind(life).bind(meta).bind(irradiance).bind(aux).bind(entryIndirection).dispatchIndirect(args, 32);

        Pass.rt(g, "ircache/trace_accessibility.rgen.hlsl", List.of("rt/shadow.rmiss.hlsl", "rt/shadow.rmiss.hlsl"), List.of())
                .bind(spatial).bind(life).bind(repoProposal).bind(meta).bind(aux).bind(entryIndirection).tlas(tlas).traceRaysIndirect(args, 16);

        Pass.rt(g, "ircache/ircache_validate.rgen.hlsl", MISS2, HIT)
                .bind(spatial).bind(skyCube).bind(gridMeta).bind(life).bind(repoProposal).bind(repoProposalCount).bind(wrcAtlas)
                .bind(meta).bind(aux).bind(pool).bind(entryIndirection).bind(entryCell).tlas(tlas)
                .traceRays(MAX_ENTRIES * VALIDATION_SAMPLES_PER_FRAME, 1, 1);

        Pass.rt(g, "ircache/trace_irradiance.rgen.hlsl", MISS2, HIT)
                .bind(spatial).bind(skyCube).bind(gridMeta).bind(life).bind(repoProposal).bind(repoProposalCount).bind(wrcAtlas)
                .bind(meta).bind(aux).bind(pool).bind(entryIndirection).bind(entryCell).tlas(tlas)
                .traceRays(MAX_ENTRIES * SAMPLES_PER_FRAME, 1, 1);

        pendingSum = true;
        pendingArgs = args;
    }

    public void sumUpIrradianceForSampling(Gfx g) {
        if (!pendingSum) return;
        Pass.compute(g, "ircache/sum_up_irradiance.hlsl").bind(life).bind(meta).bind(irradiance).bind(aux).bind(entryIndirection).dispatchIndirect(pendingArgs, 32);
        pendingSum = false;
    }
}
