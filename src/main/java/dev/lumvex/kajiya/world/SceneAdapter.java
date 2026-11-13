// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.world;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.render.scene.Scene;
import dev.lumvex.texture.TextureSet;
import org.joml.Matrix4f;
import org.joml.Vector3f;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class SceneAdapter {
    private final Gfx g;
    private final World world;
    private final Scene scene;
    private final Map<Integer, Integer> geometry = new HashMap<>();
    private final Map<Integer, dev.lumvex.render.scene.Mesh> geometrySource = new HashMap<>();
    private final Map<String, Integer> meshEntries = new HashMap<>();
    private final Map<Integer, int[]> textureIds = new HashMap<>();
    private final List<Img> textures = new ArrayList<>();
    private int seenStructure = -1;
    private int lightGeometry = -1;

    public SceneAdapter(Gfx g, World world, Scene scene) {
        this.g = g;
        this.world = world;
        this.scene = scene;
    }

    public boolean sync() {
        boolean rebuilt = false;
        if (seenStructure != scene.structureRevision) {
            rebuild();
            seenStructure = scene.structureRevision;
            rebuilt = true;
            world.tlasDirty = true;
        }
        int n = Math.min(scene.instances.size(), world.instances.size());
        boolean moved = false;
        //DEV: static instances (almost all of the city and its ghost copies) are never compared or copied again
        if (rebuilt || wasDyn.length != n) wasDyn = new boolean[n];
        if (world.dynIdx.length < n) world.dynIdx = new int[Math.max(n, 64)];
        int dyn = 0;
        for (int i = 0; i < n; i++) {
            Scene.Instance si = scene.instances.get(i);
            boolean nowDyn = si.isDynamic();
            if (!rebuilt && !nowDyn && !wasDyn[i]) continue;
            World.Instance wi = world.instances.get(i);
            if (!wi.transform.equals(si.transform) && !wi.me.noRt) moved = true; //DEV: raster-only instances (arms) never force a TLAS rebuild
            wi.transform.set(si.transform);
            wi.wbValid = false;
            if (rebuilt) wi.prevTransform.set(si.transform);
            world.dynIdx[dyn++] = i;
            wasDyn[i] = nowDyn;
        }
        world.dynCount = dyn;
        //DEV: moving instances (rotors, beacons) refit the TLAS only every few frames, a rebuild over all instances per frame is far too costly
        if (moved) tlasPending = true;
        if (tlasPending && ++tlasWait >= TLAS_PERIOD) {
            world.tlasDirty = true;
            tlasPending = false;
            tlasWait = 0;
        }
        return rebuilt;
    }

    private static final int TLAS_PERIOD = Integer.getInteger("lumvex.tlasperiod", 4);
    private boolean tlasPending;
    private boolean[] wasDyn = new boolean[0];
    private int tlasWait;

    private void rebuild() {
        world.instances.clear();
        scene.replacedMeshes.clear();
        for (int i = 0; i < scene.meshes.size(); i++) {
            var m = scene.meshes.get(i);
            if (geometrySource.get(i) != m) {
                geometry.put(i, world.addGeometry(m.vertices, m.indices));
                geometrySource.put(i, m);
            }
        }
        for (Scene.Instance inst : scene.instances) {
            var mat = scene.materials.get(inst.material);
            String key = inst.mesh + ":" + mat.hashCode() + ":" + mat;
            int entry = meshEntries.computeIfAbsent(key, k -> world.addMesh(geometry.get(inst.mesh), convert(mat)));
            world.addInstance(entry, inst.transform);
        }
        for (Scene.SphereLight l : scene.lights) {
            if (l.shape() != Scene.Shape.NONE) continue;
            if (lightGeometry < 0) {
                var s = dev.lumvex.render.scene.MeshFactory.sphere(1f, 12, 6);
                lightGeometry = world.addGeometry(s.vertices, s.indices);
            }
            World.Material w = new World.Material();
            w.baseColor = new float[]{0, 0, 0, 1};
            w.emissive = new float[]{l.radiance().x, l.radiance().y, l.radiance().z};
            int entry = world.addMesh(lightGeometry, w);
            world.addInstance(entry, new Matrix4f().translate(l.position()).scale(Math.max(0.01f, l.radius())));
        }
    }

    private World.Material convert(dev.lumvex.render.scene.Material m) {
        World.Material w = new World.Material();
        w.baseColor = new float[]{m.albedo().x, m.albedo().y, m.albedo().z, 1f};
        w.roughness = m.roughness();
        w.metalness = m.metallic();
        w.emissive = new float[]{m.emission().x, m.emission().y, m.emission().z};
        w.flags = (m.flags() & 4) != 0 ? 2 : (m.flags() & 8) != 0 ? 4 : (m.flags() & 16) != 0 ? 8 : (m.flags() & 32) != 0 ? 16 : (m.flags() & 64) != 0 ? 32 : (m.flags() & 128) != 0 ? 64 : (m.flags() & 256) != 0 ? 128 : (m.flags() & 512) != 0 ? 256 : 0;
        int set = m.textureSet();
        if (set >= 0 && set < scene.textureSets.size()) {
            int[] ids = textureIds.computeIfAbsent(set, this::uploadTextureSet);
            w.albedoMap = ids[0];
            if (m.normalStrength() > 0) w.normalMap = ids[1];
            w.specMap = ids[2];
            float s = m.uvScale();
            w.uvTransform = new float[]{s, 0, 0, s, 0, 0};
        }
        return w;
    }

    private int[] uploadTextureSet(int set) {
        TextureSet ts = scene.textureSets.get(set);
        int mips = ts.mipCount;
        ByteBuffer[] albedo = new ByteBuffer[mips], normal = new ByteBuffer[mips], spec = new ByteBuffer[mips];
        for (int m = 0; m < mips; m++) {
            albedo[m] = ts.level(TextureSet.ALBEDO, m).duplicate().order(ByteOrder.LITTLE_ENDIAN);
            normal[m] = ts.level(TextureSet.NORMAL, m).duplicate().order(ByteOrder.LITTLE_ENDIAN);
            ByteBuffer orm = ts.level(TextureSet.ORM, m);
            int n = orm.capacity() / 4;
            ByteBuffer s = ByteBuffer.allocateDirect(n * 4);
            for (int i = 0; i < n; i++) {
                s.put(i * 4, orm.get(i * 4 + 1)).put(i * 4 + 1, orm.get(i * 4 + 2)).put(i * 4 + 2, (byte) 0).put(i * 4 + 3, (byte) 255);
            }
            spec[m] = s;
        }
        int size = ts.size;
        Img a = Tex.rgba8Chain(g, size, size, albedo, true), nrm = Tex.rgba8Chain(g, size, size, normal, false), sp = Tex.rgba8Chain(g, size, size, spec, false);
        textures.add(a);
        textures.add(nrm);
        textures.add(sp);
        return new int[]{world.bindless.addImage(a), world.bindless.addImage(nrm), world.bindless.addImage(sp)};
    }

    public void close() {
        for (Img i : textures) i.close();
    }
}
