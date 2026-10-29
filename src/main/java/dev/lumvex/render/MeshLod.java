package dev.lumvex.render;

import dev.lumvex.render.scene.Mesh;

import java.util.IdentityHashMap;
import java.util.Map;

public final class MeshLod {
    public static final int MAX_LEVELS = 4;

    public final int[][] indices;
    public final float[] error;

    private MeshLod(int[][] indices, float[] error) {
        this.indices = indices;
        this.error = error;
    }

    public int levels() { return indices.length; }

    private static final Map<Mesh, MeshLod> CACHE = new IdentityHashMap<>();

    public static synchronized MeshLod of(Mesh mesh) {
        return CACHE.computeIfAbsent(mesh, m -> new MeshLod(new int[][]{m.indices}, new float[]{0f}));
    }

    public static synchronized void retain(java.util.Collection<Mesh> meshes) {
        java.util.Set<Mesh> keep = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        keep.addAll(meshes);
        CACHE.keySet().retainAll(keep);
    }
}
