package dev.lumvex.render.scene;

import org.joml.Vector3f;

import java.util.Arrays;

/**
 * An immutable indexed triangle mesh. Each mesh is uploaded once and gets its own BLAS; a different shape is a new
 * mesh. Vertices are interleaved as px py pz nx ny nz u v (32 bytes).
 */
public final class Mesh {
    public static final int FLOATS_PER_VERTEX = 8;
    public static final int STRIDE = FLOATS_PER_VERTEX * Float.BYTES;

    public final String name;
    public final float[] vertices;
    public final int[] indices;

    public Mesh(String name, float[] vertices, int[] indices) {
        this.name = name;
        this.vertices = vertices;
        this.indices = indices;
    }

    public int vertexCount() { return vertices.length / FLOATS_PER_VERTEX; }

    private float[] cachedBounds;

    /** Object-space bounds {minX, minY, minZ, maxX, maxY, maxZ}. Returns a copy. */
    public float[] bounds() {
        if (cachedBounds != null) return cachedBounds.clone();
        cachedBounds = computeBounds();
        return cachedBounds.clone();
    }

    private float[] computeBounds() {
        float[] b = {Float.MAX_VALUE, Float.MAX_VALUE, Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE, -Float.MAX_VALUE};
        for (int i = 0; i < vertices.length; i += FLOATS_PER_VERTEX) {
            for (int k = 0; k < 3; k++) {
                b[k] = Math.min(b[k], vertices[i + k]);
                b[3 + k] = Math.max(b[3 + k], vertices[i + k]);
            }
        }
        return b;
    }

    public int triangleCount() { return indices.length / 3; }

    /** Incremental mesh construction. {@code vertex(...)} returns the index of the new vertex. */
    public static final class Builder {
        private final String name;
        private float[] v = new float[1024];
        private int[] i = new int[1024];
        private int vn, in;

        public Builder(String name) { this.name = name; }

        public int vertex(float px, float py, float pz, float nx, float ny, float nz, float u, float w) {
            if (vn + FLOATS_PER_VERTEX > v.length) v = Arrays.copyOf(v, v.length * 2);
            v[vn++] = px; v[vn++] = py; v[vn++] = pz;
            v[vn++] = nx; v[vn++] = ny; v[vn++] = nz;
            v[vn++] = u; v[vn++] = w;
            return vn / FLOATS_PER_VERTEX - 1;
        }

        public int vertex(float px, float py, float pz, float nx, float ny, float nz) { return vertex(px, py, pz, nx, ny, nz, 0, 0); }

        public int vertex(Vector3f p, Vector3f n) { return vertex(p.x, p.y, p.z, n.x, n.y, n.z, 0, 0); }

        public int vertex(Vector3f p, Vector3f n, float u, float w) { return vertex(p.x, p.y, p.z, n.x, n.y, n.z, u, w); }

        public void triangle(int a, int b, int c) {
            if (in + 3 > i.length) i = Arrays.copyOf(i, i.length * 2);
            i[in++] = a; i[in++] = b; i[in++] = c;
        }

        /** Adds the quad a-b-c-d as two triangles. */
        public void quad(int a, int b, int c, int d) {
            triangle(a, b, c);
            triangle(a, c, d);
        }

        public Mesh build() {
            return new Mesh(name, Arrays.copyOf(v, vn), Arrays.copyOf(i, in));
        }
    }
}
