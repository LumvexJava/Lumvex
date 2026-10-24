package dev.lumvex.render.scene;

import org.joml.Vector3f;

/**
 * Procedural primitive meshes with counter-clockwise winding and outward normals.
 *
 * <p>Planes, boxes and cylinders get UVs in meters, so {@code uvScale} means repeats per meter and the texel
 * density is the same on every face. Spheres and tori use UVs in texture repeats.
 */
public final class MeshFactory {
    private MeshFactory() {}

    /** Square in the xz plane facing +y, with half the edge length given. */
    public static Mesh plane(float halfSize) {
        Mesh.Builder b = new Mesh.Builder("plane");
        int a = b.vertex(-halfSize, 0, -halfSize, 0, 1, 0, -halfSize, -halfSize);
        int c = b.vertex(-halfSize, 0, halfSize, 0, 1, 0, -halfSize, halfSize);
        int d = b.vertex(halfSize, 0, halfSize, 0, 1, 0, halfSize, halfSize);
        int e = b.vertex(halfSize, 0, -halfSize, 0, 1, 0, halfSize, -halfSize);
        b.quad(a, c, d, e);
        return b.build();
    }

    /** Unit cube centered at the origin. */
    public static Mesh cube() {
        return box(1, 1, 1);
    }

    /** Box centered at the origin with flat normals; each face has its own vertices. */
    public static Mesh box(float sx, float sy, float sz) {
        Mesh.Builder b = new Mesh.Builder("box");
        Vector3f size = new Vector3f(sx, sy, sz);
        float[][] normals = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
        for (float[] n : normals) {
            Vector3f normal = new Vector3f(n[0], n[1], n[2]);
            Vector3f up = Math.abs(normal.y) > 0.5f ? new Vector3f(0, 0, 1) : new Vector3f(0, 1, 0);
            Vector3f right = new Vector3f(up).cross(normal);   // (right, up, normal) is right-handed
            Vector3f u = new Vector3f(normal).cross(right);
            Vector3f center = new Vector3f(normal).mul(size).mul(0.5f);
            Vector3f halfRight = new Vector3f(right).mul(size).mul(0.5f);
            Vector3f halfUp = new Vector3f(u).mul(size).mul(0.5f);
            float width = halfRight.length() * 2, height = halfUp.length() * 2;
            int[] idx = new int[4];
            float[][] corners = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
            for (int k = 0; k < 4; k++) {
                Vector3f p = new Vector3f(center)
                        .fma(corners[k][0], halfRight)
                        .fma(corners[k][1], halfUp);
                idx[k] = b.vertex(p, normal, (corners[k][0] * 0.5f + 0.5f) * width, (0.5f - corners[k][1] * 0.5f) * height);
            }
            b.quad(idx[0], idx[1], idx[2], idx[3]);
        }
        return b.build();
    }

    /** Unit sphere, see {@link #sphere(float, int, int)}. */
    public static Mesh sphere(int segments, int rings) { return sphere(1, segments, rings); }

    /**
     * Sphere centered at the origin. U spans two texture repeats around the equator so texels stay square (the
     * circumference is twice the pole-to-pole length).
     */
    public static Mesh sphere(float radius, int segments, int rings) {
        Mesh.Builder b = new Mesh.Builder("sphere");
        for (int r = 0; r <= rings; r++) {
            double theta = Math.PI * r / rings;
            for (int s = 0; s <= segments; s++) {
                double phi = 2 * Math.PI * s / segments;
                float x = (float) (Math.sin(theta) * Math.cos(phi));
                float y = (float) Math.cos(theta);
                float z = (float) (Math.sin(theta) * Math.sin(phi));
                b.vertex(x * radius, y * radius, z * radius, x, y, z, 2f * s / segments, (float) r / rings);
            }
        }
        int row = segments + 1;
        for (int r = 0; r < rings; r++) {
            for (int s = 0; s < segments; s++) {
                int i0 = r * row + s, i1 = i0 + 1, i2 = i0 + row, i3 = i2 + 1;
                if (r != 0) b.triangle(i0, i1, i2);
                if (r != rings - 1) b.triangle(i1, i3, i2);
            }
        }
        return b.build();
    }

    /** Closed cylinder along y, centered at the origin. {@code sx} and {@code sz} are diameters, {@code sy} the height. */
    public static Mesh cylinder(float sx, float sy, float sz, int segments) {
        Mesh.Builder b = new Mesh.Builder("cylinder");
        float rx = sx / 2, rz = sz / 2, h = sy / 2;
        float circ = (float) (Math.PI * (rx + rz));
        int[] bottom = new int[segments + 1], top = new int[segments + 1];
        for (int s = 0; s <= segments; s++) {
            double a = 2 * Math.PI * s / segments;
            float c = (float) Math.cos(a), sn = (float) Math.sin(a);
            Vector3f n = new Vector3f(c / Math.max(rx, 1e-6f), 0, sn / Math.max(rz, 1e-6f)).normalize();
            float u = circ * s / segments;
            bottom[s] = b.vertex(c * rx, -h, sn * rz, n.x, n.y, n.z, u, sy);
            top[s] = b.vertex(c * rx, h, sn * rz, n.x, n.y, n.z, u, 0);
        }
        for (int s = 0; s < segments; s++) b.quad(bottom[s], bottom[s + 1], top[s + 1], top[s]);
        for (int side = -1; side <= 1; side += 2) {
            int center = b.vertex(0, side * h, 0, 0, side, 0, 0, 0);
            int[] ring = new int[segments + 1];
            for (int s = 0; s <= segments; s++) {
                double a = 2 * Math.PI * s / segments;
                float x = (float) Math.cos(a) * rx, z = (float) Math.sin(a) * rz;
                ring[s] = b.vertex(x, side * h, z, 0, side, 0, x, z);
            }
            for (int s = 0; s < segments; s++) {
                if (side > 0) b.triangle(center, ring[s + 1], ring[s]);
                else b.triangle(center, ring[s], ring[s + 1]);
            }
        }
        return b.build();
    }

    /**
     * Torus around the y axis. The texture repeats about {@code majorRadius / minorRadius} times along the ring to
     * avoid stretching.
     */
    public static Mesh torus(float majorRadius, float minorRadius, int majorSegments, int minorSegments) {
        Mesh.Builder b = new Mesh.Builder("torus");
        float uRepeat = Math.max(1, Math.round(majorRadius / minorRadius));
        for (int i = 0; i <= majorSegments; i++) {
            double u = 2 * Math.PI * i / majorSegments;
            float cu = (float) Math.cos(u), su = (float) Math.sin(u);
            for (int j = 0; j <= minorSegments; j++) {
                double v = 2 * Math.PI * j / minorSegments;
                float cv = (float) Math.cos(v), sv = (float) Math.sin(v);
                float nx = cu * cv, ny = sv, nz = su * cv;
                b.vertex((majorRadius + minorRadius * cv) * cu, minorRadius * sv, (majorRadius + minorRadius * cv) * su,
                        nx, ny, nz, uRepeat * i / majorSegments, (float) j / minorSegments);
            }
        }
        int row = minorSegments + 1;
        for (int i = 0; i < majorSegments; i++) {
            for (int j = 0; j < minorSegments; j++) {
                int a = i * row + j, c = a + row;
                b.triangle(a, a + 1, c);
                b.triangle(a + 1, c + 1, c);
            }
        }
        return b.build();
    }
}
