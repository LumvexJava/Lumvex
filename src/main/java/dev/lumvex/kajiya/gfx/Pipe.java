// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import java.util.TreeMap;

public final class Pipe {
    public final String name;
    public final long pipeline;
    public final long layout;
    public final int bindPoint;
    public final long set0Layout;
    public final TreeMap<Integer, Spirv.Binding> set0;
    public final int maxSet;
    public final int[] local;
    /** Ray tracing shader binding table regions: address, stride, size for raygen, miss, hit. */
    public long[] sbt;
    public dev.lumvex.vk.Buffer[] sbtBuffers;

    public final long set3Layout;
    public final TreeMap<Integer, Spirv.Binding> set3;

    public Pipe(String name, long pipeline, long layout, int bindPoint, long set0Layout, TreeMap<Integer, Spirv.Binding> set0,
                int maxSet, int[] local, long set3Layout, TreeMap<Integer, Spirv.Binding> set3) {
        this.set3Layout = set3Layout;
        this.set3 = set3;
        this.name = name;
        this.pipeline = pipeline;
        this.layout = layout;
        this.bindPoint = bindPoint;
        this.set0Layout = set0Layout;
        this.set0 = set0;
        this.maxSet = maxSet;
        this.local = local;
    }
}
