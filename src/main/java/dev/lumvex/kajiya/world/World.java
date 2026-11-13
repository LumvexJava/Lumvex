// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.world;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.vk.Buffer;
import org.joml.Matrix4f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.system.MemoryUtil.memAlloc;
import static org.lwjgl.system.MemoryUtil.memFree;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
import static org.lwjgl.vulkan.VK13.*;

public final class World implements AutoCloseable {

    public static final int MATERIAL_SIZE = 152;
    public static final int TLAS_MAX_INSTANCES = 65536;
    public static final int TEX_WHITE = 3, TEX_FLAT_NORMAL = 4, TEX_SPEC_DEFAULT = 5;

    public static final class Geometry {
        public int core, uv, mat, aux, tangent, index;
        public int vertexCount, indexCount;
        public Accel blas;
        public final float[] bounds = new float[6];
    }

    public static final class Material {
        public float[] baseColor = {1, 1, 1, 1};
        public int normalMap = TEX_FLAT_NORMAL, specMap = TEX_SPEC_DEFAULT, albedoMap = TEX_WHITE, emissiveMap = TEX_WHITE;
        public float roughness = 1f, metalness = 0f;
        public float[] emissive = {0, 0, 0};
        public int flags;
        public float[] uvTransform = {1, 0, 0, 1, 0, 0};
    }

    //DEV: distances (m) where prop LOD 0/1/2 end and where decals disappear
    public static final float LOD1_DIST = Float.parseFloat(System.getProperty("lumvex.lod1", "40")), LOD2_DIST = Float.parseFloat(System.getProperty("lumvex.lod2", "100")),
            LOD_MAX = Float.parseFloat(System.getProperty("lumvex.lodmax", "230")), DECAL_DIST = Float.parseFloat(System.getProperty("lumvex.detaildist", "130"));

    //DEV: interior geometry (rooms, furniture) is only drawn inside this distance; impostors take over props beyond the geometry LODs
    public static final float INTERIOR_DIST = Float.parseFloat(System.getProperty("lumvex.interiordist", "70")), IMPOSTOR_DIST = Float.parseFloat(System.getProperty("lumvex.impostordist", "520"));

    public static final class MeshEntry {
        public int geometry;
        public boolean emissive, noRt, detail; //DEV: detail = decals and props, distance culled
        public float nearDist = 0f, farDist = Float.MAX_VALUE; //DEV: camera distance range in which the mesh is drawn (prop LODs, decals)
        public float[] emissiveRgb = {0, 0, 0};
    }

    public static final class Instance {
        public int mesh;
        public final Matrix4f transform = new Matrix4f();
        public final Matrix4f prevTransform = new Matrix4f();
        public float emissiveMultiplier = 1f;
        //DEV: cached world bounds and lookups so the per frame loops do not chase lists and transform corners
        public final float[] wb = new float[6];
        public boolean wbValid;
        public MeshEntry me;
        public Geometry geo;
    }

    //DEV: indices of the instances that moved this frame or the one before (prev transforms are only stored for these)
    public int[] dynIdx = new int[0];
    public int dynCount;
    private Instance[] instArray = new Instance[0];
    private float[] emissiveArray = new float[0];

    private boolean arraysDirty = true; //DEV: a rebuild with the same instance count must not reuse the stale arrays

    public Instance[] instArray() {
        if (arraysDirty || instArray.length != instances.size()) { instArray = instances.toArray(new Instance[0]); emissiveArray = new float[0]; arraysDirty = false; }
        return instArray;
    }

    /** Per instance emission multipliers as one array (written into the frame buffer in a single copy). */
    public float[] emissiveArray() {
        if (emissiveArray.length != instances.size()) {
            emissiveArray = new float[instances.size()];
            for (int i = 0; i < emissiveArray.length; i++) emissiveArray[i] = instances.get(i).emissiveMultiplier;
        }
        return emissiveArray;
    }

    private final Gfx g;
    public final Bindless bindless;
    public final List<Geometry> geometries = new ArrayList<>();
    public final List<MeshEntry> meshes = new ArrayList<>();
    public final List<Instance> instances = new ArrayList<>();
    public Accel tlas;
    public boolean tlasDirty = true;
    private boolean tlasBuilt;
    private Buffer tlasScratch;
    private long[] tlasScratchSize = new long[1];
    private Img brdfLut, bezoldLut, blueNoise, white, flatNormal, specDefault;

    public World(Gfx g) {
        this.g = g;
        bindless = new Bindless(g);
        bindless.appendVertexData(memAllocZero(256));
        if (g.ctx.rayTracingSupported) {
            tlas = Accel.top(g.ctx, TLAS_MAX_INSTANCES, tlasScratchSize);
            tlasScratch = new Buffer(g.ctx, tlasScratchSize[0] + g.ctx.minScratchOffsetAlignment,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, Buffer.DEVICE);
        }
        initLuts();
    }

    private static ByteBuffer memAllocZero(int n) {
        ByteBuffer b = ByteBuffer.allocateDirect(n);
        return b;
    }

    private void initLuts() {
        brdfLut = new Img(g.ctx, new Img.Desc(64, 64, VK_FORMAT_R16G16B16A16_SFLOAT));
        bezoldLut = new Img(g.ctx, new Img.Desc(64, 1, VK_FORMAT_R16G16_SFLOAT));
        g.submit(() -> {
            Pass.compute(g, "lut/brdf_fg.hlsl").bind(brdfLut).dispatch(64, 64, 1);
            Pass.compute(g, "lut/bezold_brucke.hlsl").bind(bezoldLut).dispatch(64, 1, 1);
        });
        blueNoise = Tex.fromResource(g, "/kajiya/images/LDR_RGBA_0.png", false, false);
        int a = bindless.addImage(brdfLut), b = bindless.addImage(blueNoise), c = bindless.addImage(bezoldLut);
        if (a != Bindless.LUT_BRDF_FG || b != Bindless.LUT_BLUE_NOISE || c != Bindless.LUT_BEZOLD_BRUCKE) throw new IllegalStateException("bindless LUT ids");
        white = Tex.solid(g, 255, 255, 255, 255, true);
        flatNormal = Tex.solid(g, 128, 128, 255, 255, false);
        specDefault = Tex.solid(g, 255, 255, 0, 255, false);
        if (bindless.addImage(white) != TEX_WHITE || bindless.addImage(flatNormal) != TEX_FLAT_NORMAL || bindless.addImage(specDefault) != TEX_SPEC_DEFAULT) {
            throw new IllegalStateException("bindless default texture ids");
        }
    }

    private static int packNormal(float x, float y, float z) {
        int px = (int) ((Math.clamp(x, -1f, 1f) * 0.5f + 0.5f) * 2047f);
        int py = (int) ((Math.clamp(y, -1f, 1f) * 0.5f + 0.5f) * 1023f);
        int pz = (int) ((Math.clamp(z, -1f, 1f) * 0.5f + 0.5f) * 2047f);
        return (pz << 21) | (py << 11) | px;
    }

    public int addGeometry(float[] v, int[] idx) {
        int n = v.length / 8;
        Geometry geo = new Geometry();
        geo.vertexCount = n;
        geo.indexCount = idx.length;
        float[] tan = tangents(v, idx, n);
        int total = idx.length * 4 + n * 16 + n * 8 + n * 4 + n * 16;
        ByteBuffer b = memAlloc(total + 64).order(ByteOrder.LITTLE_ENDIAN);
        int base = (int) bindless.vertexWritten;
        geo.index = base + b.position();
        for (int i : idx) b.putInt(i);
        geo.core = base + b.position();
        float[] bb = {Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE};
        for (int i = 0; i < n; i++) {
            b.putFloat(v[i * 8]).putFloat(v[i * 8 + 1]).putFloat(v[i * 8 + 2]).putInt(packNormal(v[i * 8 + 3], v[i * 8 + 4], v[i * 8 + 5]));
            for (int k = 0; k < 3; k++) {
                bb[k] = Math.min(bb[k], v[i * 8 + k]);
                bb[3 + k] = Math.max(bb[3 + k], v[i * 8 + k]);
            }
        }
        System.arraycopy(bb, 0, geo.bounds, 0, 6);
        geo.uv = base + b.position();
        for (int i = 0; i < n; i++) b.putFloat(v[i * 8 + 6]).putFloat(v[i * 8 + 7]);
        geo.mat = base + b.position();
        for (int i = 0; i < n; i++) b.putInt(0);
        geo.tangent = base + b.position();
        for (int i = 0; i < n * 4; i++) b.putFloat(tan[i]);
        geo.aux = 0;
        b.flip();
        int off = bindless.appendVertexData(b);
        memFree(b);
        if (off != base) throw new IllegalStateException("vertex offset mismatch");
        if (g.ctx.rayTracingSupported) {
            long addr = bindless.vertices.address;
            geo.blas = Accel.bottom(g.ctx, addr + geo.core, 16, n - 1, addr + geo.index, idx.length / 3);
        }
        geometries.add(geo);
        return geometries.size() - 1;
    }

    private static float[] tangents(float[] v, int[] idx, int n) {
        float[] t = new float[n * 3], bt = new float[n * 3], out = new float[n * 4];
        for (int i = 0; i + 2 < idx.length; i += 3) {
            int a = idx[i], b = idx[i + 1], c = idx[i + 2];
            float e1x = v[b * 8] - v[a * 8], e1y = v[b * 8 + 1] - v[a * 8 + 1], e1z = v[b * 8 + 2] - v[a * 8 + 2];
            float e2x = v[c * 8] - v[a * 8], e2y = v[c * 8 + 1] - v[a * 8 + 1], e2z = v[c * 8 + 2] - v[a * 8 + 2];
            float du1 = v[b * 8 + 6] - v[a * 8 + 6], dv1 = v[b * 8 + 7] - v[a * 8 + 7];
            float du2 = v[c * 8 + 6] - v[a * 8 + 6], dv2 = v[c * 8 + 7] - v[a * 8 + 7];
            float det = du1 * dv2 - du2 * dv1;
            if (Math.abs(det) < 1e-12f) continue;
            float r = 1f / det;
            float tx = (e1x * dv2 - e2x * dv1) * r, ty = (e1y * dv2 - e2y * dv1) * r, tz = (e1z * dv2 - e2z * dv1) * r;
            float bx = (e2x * du1 - e1x * du2) * r, by = (e2y * du1 - e1y * du2) * r, bz = (e2z * du1 - e1z * du2) * r;
            for (int k : new int[]{a, b, c}) {
                t[k * 3] += tx; t[k * 3 + 1] += ty; t[k * 3 + 2] += tz;
                bt[k * 3] += bx; bt[k * 3 + 1] += by; bt[k * 3 + 2] += bz;
            }
        }
        for (int i = 0; i < n; i++) {
            float nx = v[i * 8 + 3], ny = v[i * 8 + 4], nz = v[i * 8 + 5];
            float tx = t[i * 3], ty = t[i * 3 + 1], tz = t[i * 3 + 2];
            float d = nx * tx + ny * ty + nz * tz;
            tx -= nx * d; ty -= ny * d; tz -= nz * d;
            float len = (float) Math.sqrt(tx * tx + ty * ty + tz * tz);
            if (len < 1e-8f) {
                // any vector orthogonal to the normal
                if (Math.abs(nx) < 0.9f) { tx = 0; ty = -nz; tz = ny; } else { tx = nz; ty = 0; tz = -nx; }
                len = (float) Math.sqrt(tx * tx + ty * ty + tz * tz);
            }
            tx /= len; ty /= len; tz /= len;
            float cx = ny * tz - nz * ty, cy = nz * tx - nx * tz, cz = nx * ty - ny * tx;
            float w = (cx * bt[i * 3] + cy * bt[i * 3 + 1] + cz * bt[i * 3 + 2]) < 0 ? -1f : 1f;
            out[i * 4] = tx; out[i * 4 + 1] = ty; out[i * 4 + 2] = tz; out[i * 4 + 3] = w;
        }
        return out;
    }

    public int addMesh(int geometry, Material m) {
        Geometry geo = geometries.get(geometry);
        ByteBuffer b = memAlloc(MATERIAL_SIZE).order(ByteOrder.LITTLE_ENDIAN);
        for (float f : m.baseColor) b.putFloat(f);
        b.putInt(m.normalMap).putInt(m.specMap).putInt(m.albedoMap).putInt(m.emissiveMap);
        b.putFloat(m.roughness).putFloat(m.metalness);
        for (float f : m.emissive) b.putFloat(f);
        b.putInt(m.flags);
        for (int slot = 0; slot < 4; slot++) for (float f : m.uvTransform) b.putFloat(f);
        b.flip();
        int matOffset = bindless.appendVertexData(b);
        memFree(b);
        int idx = meshes.size();
        if (idx >= Bindless.MAX_MESHES) throw new IllegalStateException("too many mesh/material combinations (limit " + Bindless.MAX_MESHES + ")");
        bindless.setMesh(idx, new int[]{geo.core, geo.uv, geo.mat, geo.aux, geo.tangent, matOffset, geo.index});
        MeshEntry e = new MeshEntry();
        e.geometry = geometry;
        e.emissiveRgb = m.emissive.clone();
        e.noRt = (m.flags & 366) != 0;
        e.detail = (m.flags & 500) != 0;
        if ((m.flags & 16) != 0) { e.nearDist = 0f; e.farDist = LOD1_DIST; }
        else if ((m.flags & 32) != 0) { e.nearDist = LOD1_DIST; e.farDist = LOD2_DIST; }
        else if ((m.flags & 64) != 0) { e.nearDist = LOD2_DIST; e.farDist = LOD_MAX; }
        else if ((m.flags & 128) != 0) { e.nearDist = 0f; e.farDist = INTERIOR_DIST; }
        else if ((m.flags & 256) != 0) { e.nearDist = LOD_MAX; e.farDist = IMPOSTOR_DIST; }
        else if ((m.flags & 4) != 0) { e.farDist = DECAL_DIST; }
        e.emissive = m.emissive[0] > 0 || m.emissive[1] > 0 || m.emissive[2] > 0;
        meshes.add(e);
        return idx;
    }

    public Instance addInstance(int mesh, Matrix4f transform) {
        Instance i = new Instance();
        i.mesh = mesh;
        i.transform.set(transform);
        i.prevTransform.set(transform);
        i.me = meshes.get(mesh);
        i.geo = geometries.get(i.me.geometry);
        instances.add(i);
        arraysDirty = true;
        return i;
    }

    public void storePrevTransforms() {
        for (int k = 0; k < dynCount; k++) {
            Instance i = instances.get(dynIdx[k]);
            i.prevTransform.set(i.transform);
        }
    }

    public void recordTlasBuild() {
        if (tlas == null) return;
        if (!tlasDirty && tlasBuilt) return;
        tlasDirty = false;
        tlasBuilt = true;
        int off = g.dynAlloc(Math.max(1, instances.size()) * 64);
        ByteBuffer m = g.dynMap().order(ByteOrder.LITTLE_ENDIAN);
        int n = 0;
        for (int i = 0; i < instances.size() && n < TLAS_MAX_INSTANCES; i++) {
            Instance inst = instances.get(i);
            if (meshes.get(inst.mesh).noRt) continue;
            int p = off + n * 64;
            n++;
            Matrix4f t = inst.transform;
            m.putFloat(p, t.m00()).putFloat(p + 4, t.m10()).putFloat(p + 8, t.m20()).putFloat(p + 12, t.m30());
            m.putFloat(p + 16, t.m01()).putFloat(p + 20, t.m11()).putFloat(p + 24, t.m21()).putFloat(p + 28, t.m31());
            m.putFloat(p + 32, t.m02()).putFloat(p + 36, t.m12()).putFloat(p + 40, t.m22()).putFloat(p + 44, t.m32());
            m.putInt(p + 48, (inst.mesh & 0xFFFFFF) | (0xFF << 24));
            m.putInt(p + 52, org.lwjgl.vulkan.KHRAccelerationStructure.VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR << 24);
            m.putLong(p + 56, geometries.get(meshes.get(inst.mesh).geometry).blas.address);
        }
        long scratch = dev.lumvex.vk.VkUtil.align(tlasScratch.address, g.ctx.minScratchOffsetAlignment);
        tlas.recordTopBuild(g.cmd, g.dynAddress(off), n, scratch);
        g.barrier();
    }

    public boolean hasEmissive() {
        for (Instance i : instances) if (meshes.get(i.mesh).emissive) return true;
        return false;
    }

    @Override
    public void close() {
        for (Geometry geo : geometries) if (geo.blas != null) geo.blas.close();
        if (tlas != null) tlas.close();
        if (tlasScratch != null) tlasScratch.close();
        bindless.close();
        for (Img i : new Img[]{brdfLut, bezoldLut, blueNoise, white, flatNormal, specDefault}) if (i != null) i.close();
    }
}
