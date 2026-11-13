// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.world;

//DEV

import org.joml.Matrix4f;
import org.joml.Vector3f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class View {
    public final Matrix4f worldToView = new Matrix4f(), viewToWorld = new Matrix4f(), viewToClip = new Matrix4f(), clipToView = new Matrix4f();

    public static View of(Matrix4f worldToView, float fovYDegrees, float aspect, float near) {
        View v = new View();
        v.worldToView.set(worldToView);
        worldToView.invert(v.viewToWorld);
        float h = (float) (1.0 / Math.tan(Math.toRadians(fovYDegrees) * 0.5));
        float w = h / aspect;
        v.viewToClip.set(w, 0, 0, 0, 0, h, 0, 0, 0, 0, 0, -1, 0, 0, near, 0);
        v.clipToView.set(1f / w, 0, 0, 0, 0, 1f / h, 0, 0, 0, 0, 0, 1f / near, 0, 0, -1, 0);
        return v;
    }

    public Vector3f eye() { return viewToWorld.getTranslation(new Vector3f()); }

    public static final int SIZE = 1216;

    /** Writes the FrameConstants struct of inc/frame_constants.hlsl; ircache cascades are written by the caller. */
    public static void write(ByteBuffer b, View cur, View prev, float jitterX, float jitterY, int renderW, int renderH) {
        b.order(ByteOrder.LITTLE_ENDIAN);
        float clipX = 2f * jitterX / renderW, clipY = 2f * jitterY / renderH;
        Matrix4f jitter = new Matrix4f().translation(-clipX, -clipY, 0), jitterInv = new Matrix4f().translation(clipX, clipY, 0);
        Matrix4f viewToSample = new Matrix4f(jitter).mul(cur.viewToClip);
        Matrix4f sampleToView = new Matrix4f(cur.clipToView).mul(jitterInv);
        Matrix4f clipToPrevClip = new Matrix4f(prev.viewToClip).mul(prev.worldToView).mul(cur.viewToWorld).mul(cur.clipToView);
        int p = 0;
        for (Matrix4f m : new Matrix4f[]{cur.viewToClip, cur.clipToView, viewToSample, sampleToView, cur.worldToView, cur.viewToWorld, clipToPrevClip,
                prev.viewToClip, prev.clipToView, prev.worldToView, prev.viewToWorld}) {
            m.get(p, b);
            p += 64;
        }
        b.putFloat(p, jitterX).putFloat(p + 4, jitterY).putFloat(p + 8, clipX).putFloat(p + 12, clipY);
    }
}
