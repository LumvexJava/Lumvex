// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import org.joml.Matrix4f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class Cb {
    private ByteBuffer b = ByteBuffer.allocate(4096).order(ByteOrder.LITTLE_ENDIAN);
    private int pos;

    private void align(int size, int align) {
        pos = (pos + align - 1) & ~(align - 1);
        int row = pos & ~15;
        if (size <= 16 && pos + size > row + 16) pos = row + 16;
        if (pos + size > b.capacity()) {
            ByteBuffer n = ByteBuffer.allocate(b.capacity() * 2).order(ByteOrder.LITTLE_ENDIAN);
            n.put(b.array(), 0, b.capacity());
            b = n;
        }
    }

    public Cb f(float v) { align(4, 4); b.putFloat(pos, v); pos += 4; return this; }

    public Cb u(int v) { align(4, 4); b.putInt(pos, v); pos += 4; return this; }

    public Cb f2(float x, float y) { align(8, 8); b.putFloat(pos, x).putFloat(pos + 4, y); pos += 8; return this; }

    public Cb u2(int x, int y) { align(8, 8); b.putInt(pos, x).putInt(pos + 4, y); pos += 8; return this; }

    public Cb f3(float x, float y, float z) { align(12, 4); b.putFloat(pos, x).putFloat(pos + 4, y).putFloat(pos + 8, z); pos += 12; return this; }

    public Cb u3(int x, int y, int z) { align(12, 4); b.putInt(pos, x).putInt(pos + 4, y).putInt(pos + 8, z); pos += 12; return this; }

    public Cb f4(float x, float y, float z, float w) {
        align(16, 16);
        b.putFloat(pos, x).putFloat(pos + 4, y).putFloat(pos + 8, z).putFloat(pos + 12, w);
        pos += 16;
        return this;
    }

    public Cb u4(int x, int y, int z, int w) {
        align(16, 16);
        b.putInt(pos, x).putInt(pos + 4, y).putInt(pos + 8, z).putInt(pos + 12, w);
        pos += 16;
        return this;
    }

    public Cb f4(float[] v) { return f4(v[0], v[1], v[2], v[3]); }

    public Cb m4(Matrix4f m) {
        align(64, 16);
        m.get(pos, b);
        pos += 64;
        return this;
    }

    public Cb rawBytes(ByteBuffer src) {
        align(16, 16);
        int n = src.remaining();
        while (pos + n > b.capacity()) {
            ByteBuffer nb = ByteBuffer.allocate(b.capacity() * 2).order(ByteOrder.LITTLE_ENDIAN);
            nb.put(b.array(), 0, b.capacity());
            b = nb;
        }
        for (int i = 0; i < n; i++) b.put(pos + i, src.get(src.position() + i));
        pos += n;
        return this;
    }

    public Cb pad16() { pos = (pos + 15) & ~15; return this; }

    public int size() { return (pos + 15) & ~15; }

    public ByteBuffer data() { return b.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(0).limit(Math.max(16, size())); }
}
