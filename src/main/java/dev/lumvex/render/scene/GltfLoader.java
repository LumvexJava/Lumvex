package dev.lumvex.render.scene;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import dev.lumvex.texture.PbrImages;
import dev.lumvex.texture.TextureSet;
import org.joml.Matrix3f;
import org.joml.Matrix4f;
import org.joml.Vector3f;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A minimal loader for binary glTF 2.0 ({@code .glb}): triangle primitives, the node hierarchy (baked into the
 * vertices) and metallic-roughness materials with embedded images. glTF uses the engine's conventions, y up and
 * meters, so no conversion is needed.
 */
public final class GltfLoader {
    private GltfLoader() {}

    /** One mesh per node and primitive; {@code material} indexes {@link Model#materials}. */
    public record Part(Mesh mesh, int material, String nodeName) {}

    /**
     * A material of the model. {@code alphaMasked} means the glTF alpha mode is not OPAQUE; {@code cutout} means its
     * texture actually has holes, as with leaves.
     */
    public record ModelMaterial(String name, Material material, boolean emissive, boolean alphaMasked, boolean cutout) {
        public ModelMaterial(String name, Material material, boolean emissive, boolean alphaMasked) {
            this(name, material, emissive, alphaMasked, false);
        }
    }

    public record Model(String name, List<Part> parts, List<ModelMaterial> materials, List<TextureSet> textureSets,
                        Vector3f min, Vector3f max) {
        public Vector3f size() { return new Vector3f(max).sub(min); }
    }

    private static final ObjectMapper JSON = new ObjectMapper();

    /**
     * Loads a {@code .glb} file.
     *
     * @param maxTextureSize textures larger than this are downscaled; a 4096 set takes about 256 MB of RAM
     * @param textureSetBase scene index the first texture set of this model will get
     */
    public static Model load(Path file, int maxTextureSize, int textureSetBase) throws IOException {
        long t0 = System.nanoTime();
        ByteBuffer glb = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        if (glb.getInt(0) != 0x46546C67) throw new IOException("not a .glb file: " + file);
        int jsonLen = glb.getInt(12);
        JsonNode root = JSON.readTree(new String(glb.array(), 20, jsonLen, StandardCharsets.UTF_8));
        int binStart = 20 + jsonLen + 8;
        ByteBuffer bin = glb.slice(binStart, glb.capacity() - binStart).order(ByteOrder.LITTLE_ENDIAN);

        // Materials that use the same images share one texture set.
        List<ModelMaterial> materials = new ArrayList<>();
        List<TextureSet> sets = new ArrayList<>();
        Map<String, Integer> setByImages = new HashMap<>();
        Map<Integer, BufferedImage> imageCache = new HashMap<>();
        for (JsonNode m : root.path("materials")) {
            JsonNode pbr = m.path("pbrMetallicRoughness");
            float[] base = floats(pbr.path("baseColorFactor"), new float[]{1, 1, 1, 1});
            float metal = (float) pbr.path("metallicFactor").asDouble(1.0);
            float rough = (float) pbr.path("roughnessFactor").asDouble(1.0);
            float[] emissive = floats(m.path("emissiveFactor"), new float[]{0, 0, 0});
            float emissiveStrength = (float) m.path("extensions").path("KHR_materials_emissive_strength").path("emissiveStrength").asDouble(1.0);
            int baseTex = texImage(root, pbr.path("baseColorTexture"));
            int mrTex = texImage(root, pbr.path("metallicRoughnessTexture"));
            int norTex = texImage(root, m.path("normalTexture"));
            int aoTex = texImage(root, m.path("occlusionTexture"));
            int emTex = texImage(root, m.path("emissiveTexture"));
            boolean masked = !"OPAQUE".equals(m.path("alphaMode").asText("OPAQUE"));
            String name = m.path("name").asText("material");

            Material mat;
            int colorTex = baseTex >= 0 ? baseTex : emTex;
            if (colorTex >= 0) {
                String key = colorTex + "/" + mrTex + "/" + norTex + "/" + aoTex;
                Integer setIndex = setByImages.get(key);
                if (setIndex == null) {
                    BufferedImage albedo = image(root, bin, colorTex, imageCache);
                    int size = PbrImages.sizeFor(albedo, maxTextureSize);
                    sets.add(PbrImages.build(name, size, albedo,
                            norTex >= 0 ? image(root, bin, norTex, imageCache) : null,
                            // roughness in G
                            mrTex >= 0 ? image(root, bin, mrTex, imageCache) : null, 1,
                            // metallic in B
                            mrTex >= 0 ? image(root, bin, mrTex, imageCache) : null, 2,
                            aoTex >= 0 ? image(root, bin, aoTex, imageCache) : null));
                    setIndex = textureSetBase + sets.size() - 1;
                    setByImages.put(key, setIndex);
                }
                float normalScale = (float) m.path("normalTexture").path("scale").asDouble(1.0);
                mat = new Material(new Vector3f(base[0], base[1], base[2]), mrTex >= 0 ? rough : rough, mrTex >= 0 ? metal : 0f,
                        new Vector3f(emissive[0], emissive[1], emissive[2]).mul(emissiveStrength), 1.5f, 0f, 0f,
                        setIndex, 1f, norTex >= 0 ? normalScale : 0f);
                if (mrTex < 0) mat = new Material(mat.albedo(), rough, metal, mat.emission(), 1.5f, 0f, 0f, setIndex, 1f, mat.normalStrength());
            } else {
                mat = new Material(new Vector3f(base[0], base[1], base[2]), rough, metal,
                        new Vector3f(emissive[0], emissive[1], emissive[2]).mul(emissiveStrength), 1.5f, 0f, 0f);
            }
            boolean isEmissive = emissive[0] + emissive[1] + emissive[2] > 0;
            int ts = mat.textureSet() - textureSetBase;
            boolean cutout = masked && ts >= 0 && ts < sets.size() && sets.get(ts).alphaMask != null;
            materials.add(new ModelMaterial(name, mat, isEmissive, masked, cutout));
        }
        if (materials.isEmpty()) materials.add(new ModelMaterial("default", Material.plastic(0.7f, 0.7f, 0.7f, 0.5f), false, false));

        // Nodes become parts with their world transforms baked in.
        List<Part> parts = new ArrayList<>();
        Vector3f min = new Vector3f(Float.MAX_VALUE), max = new Vector3f(-Float.MAX_VALUE);
        JsonNode scene = root.path("scenes").path(root.path("scene").asInt(0));
        for (JsonNode n : scene.path("nodes")) {
            visit(root, bin, n.asInt(), new Matrix4f(), parts, min, max);
        }
        System.out.printf("[gltf] %-28s %d parts, %d materials, %d texture sets, size %.2f x %.2f x %.2f m, %.1f s%n",
                file.getFileName(), parts.size(), materials.size(), sets.size(),
                max.x - min.x, max.y - min.y, max.z - min.z, (System.nanoTime() - t0) / 1e9);
        return new Model(file.getFileName().toString(), parts, materials, sets, min, max);
    }

    private static void visit(JsonNode root, ByteBuffer bin, int nodeIndex, Matrix4f parent, List<Part> parts,
                              Vector3f min, Vector3f max) {
        JsonNode node = root.path("nodes").path(nodeIndex);
        Matrix4f local = new Matrix4f();
        if (node.has("matrix")) {
            float[] m = floats(node.path("matrix"), null);
            local.set(m);
        } else {
            float[] t = floats(node.path("translation"), new float[]{0, 0, 0});
            float[] r = floats(node.path("rotation"), new float[]{0, 0, 0, 1});
            float[] s = floats(node.path("scale"), new float[]{1, 1, 1});
            local.translationRotateScale(t[0], t[1], t[2], r[0], r[1], r[2], r[3], s[0], s[1], s[2]);
        }
        Matrix4f world = new Matrix4f(parent).mul(local);
        if (node.has("mesh")) {
            JsonNode mesh = root.path("meshes").path(node.path("mesh").asInt());
            for (JsonNode prim : mesh.path("primitives")) {
                // only triangle lists
                if (prim.path("mode").asInt(4) != 4) continue;
                Mesh m = primitive(root, bin, prim, world, node.path("name").asText(""));
                if (m == null) continue;
                for (int i = 0; i < m.vertices.length; i += Mesh.FLOATS_PER_VERTEX) {
                    min.min(new Vector3f(m.vertices[i], m.vertices[i + 1], m.vertices[i + 2]));
                    max.max(new Vector3f(m.vertices[i], m.vertices[i + 1], m.vertices[i + 2]));
                }
                parts.add(new Part(m, prim.path("material").asInt(0), node.path("name").asText("")));
            }
        }
        for (JsonNode c : node.path("children")) visit(root, bin, c.asInt(), world, parts, min, max);
    }

    /** Converts a primitive to a world-space mesh, or returns null if it has no positions. */
    private static Mesh primitive(JsonNode root, ByteBuffer bin, JsonNode prim, Matrix4f world, String name) {
        JsonNode attr = prim.path("attributes");
        if (!attr.has("POSITION")) return null;
        float[] pos = readFloats(root, bin, attr.path("POSITION").asInt(), 3);
        int count = pos.length / 3;
        float[] nrm = attr.has("NORMAL") ? readFloats(root, bin, attr.path("NORMAL").asInt(), 3) : null;
        float[] uv = attr.has("TEXCOORD_0") ? readFloats(root, bin, attr.path("TEXCOORD_0").asInt(), 2) : null;
        int[] idx = prim.has("indices") ? readIndices(root, bin, prim.path("indices").asInt()) : sequence(count);

        Matrix3f normalMatrix = new Matrix3f(world).invert().transpose();
        float[] v = new float[count * Mesh.FLOATS_PER_VERTEX];
        Vector3f p = new Vector3f(), n = new Vector3f();
        for (int i = 0; i < count; i++) {
            world.transformPosition(p.set(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2]));
            if (nrm != null) normalMatrix.transform(n.set(nrm[i * 3], nrm[i * 3 + 1], nrm[i * 3 + 2])).normalize();
            int o = i * Mesh.FLOATS_PER_VERTEX;
            v[o] = p.x; v[o + 1] = p.y; v[o + 2] = p.z;
            v[o + 3] = n.x; v[o + 4] = n.y; v[o + 5] = n.z;
            v[o + 6] = uv != null ? uv[i * 2] : 0; v[o + 7] = uv != null ? uv[i * 2 + 1] : 0;
        }
        if (nrm == null) smoothNormals(v, idx);
        // mirrored: restore counter-clockwise winding
        if (world.determinant() < 0) {
            for (int i = 0; i < idx.length; i += 3) { int t = idx[i + 1]; idx[i + 1] = idx[i + 2]; idx[i + 2] = t; }
        }
        return new Mesh(name, v, idx);
    }

    /** Area-weighted smooth normals for primitives that come without normals. */
    private static void smoothNormals(float[] v, int[] idx) {
        int stride = Mesh.FLOATS_PER_VERTEX;
        for (int t = 0; t < idx.length; t += 3) {
            int a = idx[t] * stride, b = idx[t + 1] * stride, c = idx[t + 2] * stride;
            Vector3f e1 = new Vector3f(v[b] - v[a], v[b + 1] - v[a + 1], v[b + 2] - v[a + 2]);
            Vector3f e2 = new Vector3f(v[c] - v[a], v[c + 1] - v[a + 1], v[c + 2] - v[a + 2]);
            Vector3f fn = e1.cross(e2);
            for (int k : new int[]{a, b, c}) { v[k + 3] += fn.x; v[k + 4] += fn.y; v[k + 5] += fn.z; }
        }
        for (int i = 0; i < v.length; i += stride) {
            Vector3f n = new Vector3f(v[i + 3], v[i + 4], v[i + 5]);
            if (n.lengthSquared() > 0) n.normalize(); else n.set(0, 1, 0);
            v[i + 3] = n.x; v[i + 4] = n.y; v[i + 5] = n.z;
        }
    }

    private static int componentSize(int type) {
        return switch (type) { case 5120, 5121 -> 1; case 5122, 5123 -> 2; default -> 4; };
    }

    /** Reads an accessor as floats, honoring byte stride and normalized integer components. */
    private static float[] readFloats(JsonNode root, ByteBuffer bin, int accessorIndex, int comps) {
        JsonNode acc = root.path("accessors").path(accessorIndex);
        JsonNode view = root.path("bufferViews").path(acc.path("bufferView").asInt());
        int count = acc.path("count").asInt();
        int type = acc.path("componentType").asInt();
        boolean normalized = acc.path("normalized").asBoolean(false);
        int cs = componentSize(type);
        int stride = view.path("byteStride").asInt(cs * comps);
        int base = view.path("byteOffset").asInt(0) + acc.path("byteOffset").asInt(0);
        float[] out = new float[count * comps];
        for (int i = 0; i < count; i++) {
            for (int c = 0; c < comps; c++) {
                int o = base + i * stride + c * cs;
                out[i * comps + c] = switch (type) {
                    case 5126 -> bin.getFloat(o);
                    case 5123 -> normalized ? (bin.getShort(o) & 0xFFFF) / 65535f : bin.getShort(o) & 0xFFFF;
                    case 5121 -> normalized ? (bin.get(o) & 0xFF) / 255f : bin.get(o) & 0xFF;
                    case 5122 -> normalized ? Math.max(bin.getShort(o) / 32767f, -1f) : bin.getShort(o);
                    case 5120 -> normalized ? Math.max(bin.get(o) / 127f, -1f) : bin.get(o);
                    default -> bin.getInt(o);
                };
            }
        }
        return out;
    }

    private static int[] readIndices(JsonNode root, ByteBuffer bin, int accessorIndex) {
        JsonNode acc = root.path("accessors").path(accessorIndex);
        JsonNode view = root.path("bufferViews").path(acc.path("bufferView").asInt());
        int count = acc.path("count").asInt();
        int type = acc.path("componentType").asInt();
        int cs = componentSize(type);
        int base = view.path("byteOffset").asInt(0) + acc.path("byteOffset").asInt(0);
        int[] out = new int[count];
        for (int i = 0; i < count; i++) {
            int o = base + i * cs;
            out[i] = switch (type) { case 5121 -> bin.get(o) & 0xFF; case 5123 -> bin.getShort(o) & 0xFFFF; default -> bin.getInt(o); };
        }
        return out;
    }

    // Index buffer for non-indexed primitives.
    private static int[] sequence(int n) {
        int[] s = new int[n];
        for (int i = 0; i < n; i++) s[i] = i;
        return s;
    }

    private static int texImage(JsonNode root, JsonNode texRef) {
        if (!texRef.has("index")) return -1;
        return root.path("textures").path(texRef.path("index").asInt()).path("source").asInt(-1);
    }

    /** Decodes an embedded PNG or JPEG image, cached by image index. */
    private static BufferedImage image(JsonNode root, ByteBuffer bin, int imageIndex, Map<Integer, BufferedImage> cache) {
        return cache.computeIfAbsent(imageIndex, i -> {
            JsonNode img = root.path("images").path(i);
            JsonNode view = root.path("bufferViews").path(img.path("bufferView").asInt());
            byte[] data = new byte[view.path("byteLength").asInt()];
            bin.get(view.path("byteOffset").asInt(0), data);
            try {
                return PbrImages.read(data);
            } catch (IOException e) {
                throw new IllegalStateException("cannot decode image " + img.path("name").asText(), e);
            }
        });
    }

    private static float[] floats(JsonNode arr, float[] def) {
        if (arr == null || !arr.isArray()) return def;
        float[] f = new float[arr.size()];
        for (int i = 0; i < f.length; i++) f[i] = (float) arr.get(i).asDouble();
        return f;
    }
}
