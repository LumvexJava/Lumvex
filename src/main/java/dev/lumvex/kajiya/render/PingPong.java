// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.Gfx;
import dev.lumvex.kajiya.gfx.Img;

public final class PingPong {
    private String out, hist;

    public PingPong(String name) {
        out = name + ":0";
        hist = name + ":1";
    }

    /** Returns {output, history} and swaps their roles for the next frame. */
    public Img[] get(Gfx g, Img.Desc d) {
        Img o = g.temporal(out, d), h = g.temporal(hist, d);
        String t = out;
        out = hist;
        hist = t;
        return new Img[]{o, h};
    }
}
