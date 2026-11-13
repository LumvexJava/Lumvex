// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import static org.lwjgl.vulkan.VK13.*;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

public final class Spirv {

    public record Binding(int type, int count, String name) {}

    public record EntryPoint(int model, String name) {}

    public final int[] words;
    public final TreeMap<Integer, TreeMap<Integer, Binding>> sets = new TreeMap<>();
    public final List<EntryPoint> entryPoints = new ArrayList<>();
    public int localX = 1, localY = 1, localZ = 1;

    private static final int OP_NAME = 5, OP_ENTRY_POINT = 15, OP_EXECUTION_MODE = 16, OP_TYPE_IMAGE = 25, OP_TYPE_SAMPLER = 26,
            OP_TYPE_SAMPLED_IMAGE = 27, OP_TYPE_ARRAY = 28, OP_TYPE_RUNTIME_ARRAY = 29, OP_TYPE_STRUCT = 30,
            OP_TYPE_POINTER = 32, OP_CONSTANT = 43, OP_VARIABLE = 59, OP_DECORATE = 71, OP_TYPE_ACCEL = 5341;
    private static final int DEC_BLOCK = 2, DEC_BUFFER_BLOCK = 3, DEC_BINDING = 33, DEC_SET = 34;

    public static Spirv parse(byte[] code) {
        ByteBuffer bb = ByteBuffer.wrap(code).order(ByteOrder.LITTLE_ENDIAN);
        int[] w = new int[code.length / 4];
        bb.asIntBuffer().get(w);
        return new Spirv(w);
    }

    private Spirv(int[] words) {
        this.words = words;
        Map<Integer, String> names = new HashMap<>();
        Map<Integer, int[]> types = new HashMap<>();
        Map<Integer, Integer> consts = new HashMap<>();
        Map<Integer, Integer> setOf = new HashMap<>(), bindingOf = new HashMap<>();
        Map<Integer, Integer> blockKind = new HashMap<>();
        List<int[]> vars = new ArrayList<>();
        int first = -1;
        for (int i = 5; i < words.length; ) {
            int wc = words[i] >>> 16, op = words[i] & 0xFFFF;
            if (wc == 0) break;
            switch (op) {
                case OP_NAME -> names.putIfAbsent(words[i + 1], str(i + 2, i + wc));
                case OP_ENTRY_POINT -> {
                    String n = str(i + 3, i + wc);
                    entryPoints.add(new EntryPoint(words[i + 1], n));
                    if (first < 0) first = words[i + 2];
                }
                case OP_EXECUTION_MODE -> {
                    if (words[i + 2] == 17) {
                        localX = words[i + 3];
                        localY = words[i + 4];
                        localZ = words[i + 5];
                    }
                }
                case OP_DECORATE -> {
                    int target = words[i + 1], dec = words[i + 2];
                    if (dec == DEC_SET) setOf.put(target, words[i + 3]);
                    else if (dec == DEC_BINDING) bindingOf.put(target, words[i + 3]);
                    else if (dec == DEC_BLOCK) blockKind.put(target, DEC_BLOCK);
                    else if (dec == DEC_BUFFER_BLOCK) blockKind.put(target, DEC_BUFFER_BLOCK);
                }
                case OP_TYPE_IMAGE -> types.put(words[i + 1], new int[]{op, words[i + 3], words[i + 7]});
                case OP_TYPE_SAMPLER, OP_TYPE_SAMPLED_IMAGE, OP_TYPE_STRUCT, OP_TYPE_ACCEL -> types.put(words[i + 1], new int[]{op});
                case OP_TYPE_ARRAY -> types.put(words[i + 1], new int[]{op, words[i + 2], words[i + 3]});
                case OP_TYPE_RUNTIME_ARRAY -> types.put(words[i + 1], new int[]{op, words[i + 2]});
                case OP_TYPE_POINTER -> types.put(words[i + 1], new int[]{op, words[i + 2], words[i + 3]});
                case OP_CONSTANT -> consts.put(words[i + 2], words[i + 3]);
                case OP_VARIABLE -> vars.add(new int[]{words[i + 1], words[i + 2], words[i + 3]});
                default -> { }
            }
            i += wc;
        }
        for (int[] v : vars) {
            int id = v[1];
            if (!setOf.containsKey(id) || !bindingOf.containsKey(id)) continue;
            int[] ptr = types.get(v[0]);
            int pointee = ptr[2], count = 1;
            int[] t = types.get(pointee);
            while (t != null && (t[0] == OP_TYPE_ARRAY || t[0] == OP_TYPE_RUNTIME_ARRAY)) {
                count = t[0] == OP_TYPE_RUNTIME_ARRAY ? 0 : consts.getOrDefault(t[2], 1);
                pointee = t[1];
                t = types.get(pointee);
            }
            int ty;
            switch (t[0]) {
                case OP_TYPE_IMAGE -> {
                    boolean buffer = t[1] == 5;
                    ty = t[2] == 2 ? (buffer ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                            : (buffer ? VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE);
                }
                case OP_TYPE_SAMPLER -> ty = VK_DESCRIPTOR_TYPE_SAMPLER;
                case OP_TYPE_SAMPLED_IMAGE -> ty = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                case OP_TYPE_ACCEL -> ty = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
                case OP_TYPE_STRUCT -> {
                    boolean ssbo = v[2] == 12 || blockKind.get(pointee) != null && blockKind.get(pointee) == DEC_BUFFER_BLOCK;
                    ty = ssbo ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                }
                default -> throw new IllegalStateException("unsupported descriptor type for %" + id);
            }
            sets.computeIfAbsent(setOf.get(id), k -> new TreeMap<>())
                    .put(bindingOf.get(id), new Binding(ty, count, names.getOrDefault(id, "")));
        }
    }

    private String str(int from, int to) {
        StringBuilder sb = new StringBuilder();
        for (int i = from; i < to; i++) {
            int w = words[i];
            for (int b = 0; b < 4; b++) {
                int c = (w >>> (8 * b)) & 0xFF;
                if (c == 0) return sb.toString();
                sb.append((char) c);
            }
        }
        return sb.toString();
    }

    public int maxSet() { return sets.isEmpty() ? -1 : sets.lastKey(); }
}
