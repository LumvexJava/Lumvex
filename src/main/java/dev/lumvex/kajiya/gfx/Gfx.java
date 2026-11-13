// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import dev.lumvex.vk.Buffer;
import dev.lumvex.vk.VkContext;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.*;

import static dev.lumvex.vk.VkUtil.check;
import static dev.lumvex.vk.VkUtil.align;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.memAddress;
import static org.lwjgl.vulkan.KHRAccelerationStructure.*;
import static org.lwjgl.vulkan.KHRBufferDeviceAddress.VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.*;
import static org.lwjgl.vulkan.VK13.*;

public final class Gfx implements AutoCloseable {

    public static final int DYN_ALIGN = 256;
    public static final int DYN_SIZE = 64 * 1024 * 1024;
    public static final int DYN_UNIFORM_RANGE = 16384;
    public static final int DYN_STORAGE_RANGE = 12 * 1024 * 1024; //DEV: was 1 MiB = only 10.9k instances (96 bytes each) were visible; now about 131k
    public static final int BINDLESS_COUNT = 4096;
    public static final int ALL_STAGES = VK_SHADER_STAGE_ALL;

    public final VkContext ctx;
    public final VkDevice dev;
    public VkCommandBuffer cmd;

    private final Map<String, byte[]> spvCache = new HashMap<>();
    private final Map<String, Pipe> pipes = new HashMap<>();
    private final Map<String, Long> samplers = new HashMap<>();
    private final Map<Long, Long> modules = new HashMap<>();
    public final long bindlessLayout;
    public final long frameLayout;
    public long bindlessSet;
    private long bindlessPool;
    public long frameSet;
    private long frameSetPool;
    public final Buffer dyn;
    private final ByteBuffer dynMap;
    private int dynOffset;
    private long descriptorPool;
    private final long[] slotPools = new long[2];
    private int slot, dynBase;
    private final List<Img>[] imgUsedS = new List[]{new ArrayList<Img>(), new ArrayList<Img>()};
    private final List<Buffer>[] bufUsedS = new List[]{new ArrayList<Buffer>(), new ArrayList<Buffer>()};
    private final List<Long>[] bufKeyS = new List[]{new ArrayList<Long>(), new ArrayList<Long>()};

    private final Map<Img.Desc, ArrayDeque<Img>> imgFree = new HashMap<>();
    private List<Img> imgUsed = imgUsedS[0];
    private final Map<String, Img> temporals = new HashMap<>();
    private final Map<Long, ArrayDeque<Buffer>> bufFree = new HashMap<>();
    private List<Buffer> bufUsed = bufUsedS[0];
    private List<Long> bufUsedKey = bufKeyS[0];

    public Gfx(VkContext ctx) {
        this.ctx = ctx;
        this.dev = ctx.device;
        try (MemoryStack stack = stackPush()) {
            LongBuffer lp = stack.mallocLong(1);

            // set 1: bindless
            VkDescriptorSetLayoutBinding.Buffer b1 = VkDescriptorSetLayoutBinding.calloc(4, stack);
            for (int i = 0; i < 3; i++) b1.get(i).binding(i).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).stageFlags(ALL_STAGES);
            b1.get(3).binding(3).descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).descriptorCount(BINDLESS_COUNT).stageFlags(ALL_STAGES);
            IntBuffer flags1 = stack.ints(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
                    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
                    VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
                            | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT);
            VkDescriptorSetLayoutBindingFlagsCreateInfo bf1 = VkDescriptorSetLayoutBindingFlagsCreateInfo.calloc(stack).sType$Default().pBindingFlags(flags1);
            check(vkCreateDescriptorSetLayout(dev, VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pNext(bf1.address())
                    .flags(VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT).pBindings(b1), null, lp), "bindless layout");
            bindlessLayout = lp.get(0);

            VkDescriptorPoolSize.Buffer ps1 = VkDescriptorPoolSize.calloc(2, stack);
            ps1.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(3);
            ps1.get(1).type(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).descriptorCount(BINDLESS_COUNT);
            check(vkCreateDescriptorPool(dev, VkDescriptorPoolCreateInfo.calloc(stack).sType$Default()
                    .flags(VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT).maxSets(1).pPoolSizes(ps1), null, lp), "bindless pool");
            bindlessPool = lp.get(0);
            VkDescriptorSetVariableDescriptorCountAllocateInfo vc = VkDescriptorSetVariableDescriptorCountAllocateInfo.calloc(stack)
                    .sType$Default().pDescriptorCounts(stack.ints(BINDLESS_COUNT));
            check(vkAllocateDescriptorSets(dev, VkDescriptorSetAllocateInfo.calloc(stack).sType$Default().pNext(vc.address())
                    .descriptorPool(bindlessPool).pSetLayouts(stack.longs(bindlessLayout)), lp), "bindless set");
            bindlessSet = lp.get(0);

            // set 2: frame constants
            VkDescriptorSetLayoutBinding.Buffer b2 = VkDescriptorSetLayoutBinding.calloc(3, stack);
            b2.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC).descriptorCount(1).stageFlags(ALL_STAGES);
            b2.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC).descriptorCount(1).stageFlags(ALL_STAGES);
            b2.get(2).binding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC).descriptorCount(1).stageFlags(ALL_STAGES);
            IntBuffer flags2 = stack.ints(VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
                    VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
            VkDescriptorSetLayoutBindingFlagsCreateInfo bf2 = VkDescriptorSetLayoutBindingFlagsCreateInfo.calloc(stack).sType$Default().pBindingFlags(flags2);
            check(vkCreateDescriptorSetLayout(dev, VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pNext(bf2.address())
                    .pBindings(b2), null, lp), "frame layout");
            frameLayout = lp.get(0);
            VkDescriptorPoolSize.Buffer ps2 = VkDescriptorPoolSize.calloc(2, stack);
            ps2.get(0).type(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC).descriptorCount(1);
            ps2.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC).descriptorCount(2);
            check(vkCreateDescriptorPool(dev, VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(1).pPoolSizes(ps2), null, lp), "frame pool");
            frameSetPool = lp.get(0);
            check(vkAllocateDescriptorSets(dev, VkDescriptorSetAllocateInfo.calloc(stack).sType$Default().descriptorPool(frameSetPool)
                    .pSetLayouts(stack.longs(frameLayout)), lp), "frame set");
            frameSet = lp.get(0);

            dyn = new Buffer(ctx, DYN_SIZE + DYN_STORAGE_RANGE, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                    Buffer.UPLOAD);
            dynMap = dyn.map();
            VkDescriptorBufferInfo.Buffer bi = VkDescriptorBufferInfo.calloc(3, stack);
            bi.get(0).buffer(dyn.handle).range(DYN_UNIFORM_RANGE);
            bi.get(1).buffer(dyn.handle).range(DYN_STORAGE_RANGE);
            bi.get(2).buffer(dyn.handle).range(DYN_STORAGE_RANGE);
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(3, stack);
            w.get(0).sType$Default().dstSet(frameSet).dstBinding(0).descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC).descriptorCount(1).pBufferInfo(VkDescriptorBufferInfo.create(bi.address(), 1));
            w.get(1).sType$Default().dstSet(frameSet).dstBinding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC).descriptorCount(1).pBufferInfo(VkDescriptorBufferInfo.create(bi.address() + VkDescriptorBufferInfo.SIZEOF, 1));
            w.get(2).sType$Default().dstSet(frameSet).dstBinding(2).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC).descriptorCount(1).pBufferInfo(VkDescriptorBufferInfo.create(bi.address() + 2L * VkDescriptorBufferInfo.SIZEOF, 1));
            vkUpdateDescriptorSets(dev, w, null);
        }
        createDescriptorPool();
        slotPools[0] = descriptorPool;
        createDescriptorPool();
        slotPools[1] = descriptorPool;
        descriptorPool = slotPools[0];
    }

    private void createDescriptorPool() {
        try (MemoryStack stack = stackPush()) {
            VkDescriptorPoolSize.Buffer ps = VkDescriptorPoolSize.calloc(7, stack);
            int[][] sizes = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32768},
                    {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 32768},
                    {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32768}, {VK_DESCRIPTOR_TYPE_SAMPLER, 4096},
                    {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1024}};
            for (int i = 0; i < sizes.length; i++) ps.get(i).type(sizes[i][0]).descriptorCount(sizes[i][1]);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateDescriptorPool(dev, VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(8192).pPoolSizes(ps), null, lp), "descriptor pool");
            descriptorPool = lp.get(0);
        }
    }

    public void beginFrame() { beginFrame(0); }

    public void beginFrame(int s) {
        slot = s;
        imgUsed = imgUsedS[s];
        bufUsed = bufUsedS[s];
        bufUsedKey = bufKeyS[s];
        descriptorPool = slotPools[s];
        vkResetDescriptorPool(dev, descriptorPool, 0);
        dynBase = s * (DYN_SIZE / 2);
        dynOffset = dynBase;
        endFrame();
    }

    public void flushPools() {
        vkDeviceWaitIdle(dev);
        for (int k = 0; k < 2; k++) {
            for (Img i : imgUsedS[k]) imgFree.computeIfAbsent(i.desc, kk -> new ArrayDeque<>()).add(i);
            imgUsedS[k].clear();
            for (int i = 0; i < bufUsedS[k].size(); i++) bufFree.computeIfAbsent(bufKeyS[k].get(i), kk -> new ArrayDeque<>()).add(bufUsedS[k].get(i));
            bufUsedS[k].clear();
            bufKeyS[k].clear();
        }
        for (ArrayDeque<Img> q : imgFree.values()) for (Img i : q) i.close();
        imgFree.clear();
        for (ArrayDeque<Buffer> q : bufFree.values()) for (Buffer b : q) b.close();
        bufFree.clear();
    }

    public void endFrame() {
        for (Img i : imgUsed) imgFree.computeIfAbsent(i.desc, k -> new ArrayDeque<>()).add(i);
        imgUsed.clear();
        for (int i = 0; i < bufUsed.size(); i++) bufFree.computeIfAbsent(bufUsedKey.get(i), k -> new ArrayDeque<>()).add(bufUsed.get(i));
        bufUsed.clear();
        bufUsedKey.clear();
    }

    public int dynPush(ByteBuffer data) {
        int size = data.remaining();
        int off = dynAlloc(size);
        dynMap.slice(off, size).put(data.duplicate());
        return off;
    }

    public int dynAlloc(int size) {
        int off = dynOffset;
        dynOffset = (int) align(off + size, DYN_ALIGN);
        if (dynOffset > dynBase + DYN_SIZE / 2) throw new IllegalStateException("dynamic constants exhausted");
        return off;
    }

    public ByteBuffer dynMap() { return dynMap; }

    public long dynAddress(int offset) { return dyn.address + offset; }

    public Img image(Img.Desc d) {
        ArrayDeque<Img> q = imgFree.get(d);
        Img i = q != null && !q.isEmpty() ? q.poll() : new Img(ctx, d);
        imgUsed.add(i);
        return i;
    }

    public Img image(int w, int h, int format) { return image(new Img.Desc(w, h, format)); }

    public Img temporal(String name, Img.Desc d) {
        Img i = temporals.get(name);
        if (i == null || !i.desc.equals(d)) {
            if (i != null) {
                vkDeviceWaitIdle(dev);
                i.close();
            }
            i = new Img(ctx, d);
            temporals.put(name, i);
        }
        return i;
    }

    public Buffer buffer(long size, int extraUsage) {
        long key = (size << 20) ^ extraUsage;
        ArrayDeque<Buffer> q = bufFree.get(key);
        Buffer b = q != null && !q.isEmpty() ? q.poll() : newBuffer(size, extraUsage);
        bufUsed.add(b);
        bufUsedKey.add(key);
        return b;
    }

    private final Map<String, Buffer> tbufs = new HashMap<>();

    public Buffer tbuf(String name, long size) {
        Buffer b = tbufs.get(name);
        if (b == null || b.size < size) {
            if (b != null) {
                vkDeviceWaitIdle(dev);
                b.close();
            }
            b = newBuffer(size, 0);
            Buffer fb = b;
            ctx.submitImmediate(c -> vkCmdFillBuffer(c, fb.handle, 0, VK_WHOLE_SIZE, 0));
            tbufs.put(name, b);
        }
        return b;
    }

    public Buffer newBuffer(long size, int extraUsage) {
        return new Buffer(ctx, Math.max(16, size), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | extraUsage, Buffer.DEVICE);
    }

    public static final boolean PROFILE = Boolean.getBoolean("lumvex.passtimes");
    private long markPool;
    private final List<String> markNames = new ArrayList<>();
    private int marksPending;

    public void markReset() {
        if (!PROFILE) return;
        if (markPool == 0) {
            try (MemoryStack stack = stackPush()) {
                LongBuffer lp = stack.mallocLong(1);
                check(vkCreateQueryPool(dev, VkQueryPoolCreateInfo.calloc(stack).sType$Default().queryType(VK_QUERY_TYPE_TIMESTAMP).queryCount(64), null, lp), "mark pool");
                markPool = lp.get(0);
            }
        }
        if (marksPending > 0) {
            try (MemoryStack stack = stackPush()) {
                LongBuffer r = stack.mallocLong(marksPending);
                if (vkGetQueryPoolResults(dev, markPool, 0, marksPending, r, 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
                    StringBuilder sb = new StringBuilder("[passes]");
                    for (int i = 1; i < marksPending; i++) sb.append(String.format(" %s %.2f", markNames.get(i), (r.get(i) - r.get(i - 1)) * ctx.timestampPeriodNs / 1e6));
                    if (++profileFrames % 120 == 0) System.out.println(sb);
                }
            }
        }
        markNames.clear();
        marksPending = 0;
        vkCmdResetQueryPool(cmd, markPool, 0, 64);
        mark("start");
    }

    private int profileFrames;

    public void mark(String name) {
        if (!PROFILE || markPool == 0 || marksPending >= 64) return;
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, markPool, marksPending++);
        markNames.add(name);
    }

    public void barrier() {
        try (MemoryStack stack = stackPush()) {
            VkMemoryBarrier2.Buffer mb = VkMemoryBarrier2.calloc(1, stack).sType$Default()
                    .srcStageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT).srcAccessMask(VK_ACCESS_2_MEMORY_WRITE_BIT)
                    .dstStageMask(VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT).dstAccessMask(VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            vkCmdPipelineBarrier2(cmd, VkDependencyInfo.calloc(stack).sType$Default().pMemoryBarriers(mb));
        }
    }

    public byte[] spirv(String shader) {
        return spvCache.computeIfAbsent(shader, s -> {
            String res = "/kajiya/spv/" + s.replace('/', '_') + ".spv";
            if (!Boolean.getBoolean("lumvex.dxc")) {
                try (InputStream in = Gfx.class.getResourceAsStream(res)) {
                    if (in != null) return in.readAllBytes();
                } catch (IOException e) {
                    throw new IllegalStateException(e);
                }
            }
            return ShaderLib.compile(s);
        });
    }

    private long module(byte[] code) {
        try (MemoryStack stack = stackPush()) {
            ByteBuffer bb = stack.malloc(code.length);
            bb.put(code).flip();
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateShaderModule(dev, VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(bb), null, lp), "shader module");
            return lp.get(0);
        }
    }

    public long sampler(String name) {
        return samplers.computeIfAbsent(name, n -> {
            String spec = n.substring("sampler_".length());
            int filter = spec.charAt(0) == 'n' ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            int mip = spec.charAt(1) == 'n' ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
            String am = spec.substring(2);
            int addr = switch (am) {
                case "r" -> VK_SAMPLER_ADDRESS_MODE_REPEAT;
                case "mr" -> VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
                case "c" -> VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
                case "cb" -> VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
                default -> throw new IllegalArgumentException(n);
            };
            try (MemoryStack stack = stackPush()) {
                LongBuffer lp = stack.mallocLong(1);
                check(vkCreateSampler(dev, VkSamplerCreateInfo.calloc(stack).sType$Default().magFilter(filter).minFilter(filter)
                        .mipmapMode(mip).addressModeU(addr).addressModeV(addr).addressModeW(addr).maxLod(VK_LOD_CLAMP_NONE)
                        .borderColor(VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK), null, lp), "sampler");
                return lp.get(0);
            }
        });
    }

    private static int stageOf(Spirv sp) {
        int model = sp.entryPoints.get(0).model();
        return switch (model) {
            case 0 -> VK_SHADER_STAGE_VERTEX_BIT;
            case 4 -> VK_SHADER_STAGE_FRAGMENT_BIT;
            case 5 -> VK_SHADER_STAGE_COMPUTE_BIT;
            case 5313 -> VK_SHADER_STAGE_RAYGEN_BIT_KHR;
            case 5314 -> VK_SHADER_STAGE_INTERSECTION_BIT_KHR;
            case 5315 -> VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
            case 5316 -> VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            case 5317 -> VK_SHADER_STAGE_MISS_BIT_KHR;
            default -> throw new IllegalStateException("execution model " + model);
        };
    }

    private long set0Layout(TreeMap<Integer, Spirv.Binding> set, int stages) {
        try (MemoryStack stack = stackPush()) {
            int n = set.size();
            VkDescriptorSetLayoutBinding.Buffer bs = VkDescriptorSetLayoutBinding.calloc(n, stack);
            IntBuffer flags = stack.mallocInt(n);
            int i = 0;
            for (var e : set.entrySet()) {
                Spirv.Binding b = e.getValue();
                int type = switch (b.type()) {
                    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER -> VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
                    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER -> b.name().endsWith("_dyn") ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    default -> b.type();
                };
                VkDescriptorSetLayoutBinding lb = bs.get(i).binding(e.getKey()).descriptorType(type)
                        .descriptorCount(Math.max(1, b.count())).stageFlags(stages);
                if (type == VK_DESCRIPTOR_TYPE_SAMPLER) lb.pImmutableSamplers(stack.longs(sampler(b.name())));
                flags.put(i, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
                i++;
            }
            VkDescriptorSetLayoutBindingFlagsCreateInfo bf = VkDescriptorSetLayoutBindingFlagsCreateInfo.calloc(stack).sType$Default().pBindingFlags(flags);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateDescriptorSetLayout(dev, VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pNext(bf.address()).pBindings(bs), null, lp), "set0 layout");
            return lp.get(0);
        }
    }

    private long emptyLayout;

    private long emptyLayout() {
        if (emptyLayout == 0) {
            try (MemoryStack stack = stackPush()) {
                LongBuffer lp = stack.mallocLong(1);
                check(vkCreateDescriptorSetLayout(dev, VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default(), null, lp), "empty layout");
                emptyLayout = lp.get(0);
            }
        }
        return emptyLayout;
    }

    private record Layout(long set0, long pipeline, TreeMap<Integer, Spirv.Binding> bindings, int maxSet, long set3, TreeMap<Integer, Spirv.Binding> bindings3) {}

    private Layout layout(List<Spirv> shaders, int stages, int pushSize, int pushStages) {
        TreeMap<Integer, Spirv.Binding> merged = new TreeMap<>();
        int maxSet = -1;
        for (Spirv sp : shaders) {
            maxSet = Math.max(maxSet, sp.maxSet());
            var s0 = sp.sets.get(0);
            if (s0 != null) for (var e : s0.entrySet()) merged.putIfAbsent(e.getKey(), e.getValue());
        }
        long l0 = merged.isEmpty() ? emptyLayout() : set0Layout(merged, stages);
        TreeMap<Integer, Spirv.Binding> merged3 = new TreeMap<>();
        for (Spirv sp : shaders) {
            var s3 = sp.sets.get(3);
            if (s3 != null) for (var e : s3.entrySet()) merged3.putIfAbsent(e.getKey(), e.getValue());
        }
        long l3 = merged3.isEmpty() ? emptyLayout() : set0Layout(merged3, stages);
        try (MemoryStack stack = stackPush()) {
            int sets = Math.max(1, maxSet + 1);
            LongBuffer ls = stack.mallocLong(sets);
            ls.put(0, l0);
            if (sets > 1) ls.put(1, bindlessLayout);
            if (sets > 2) ls.put(2, frameLayout);
            if (sets > 3) ls.put(3, l3);
            VkPipelineLayoutCreateInfo li = VkPipelineLayoutCreateInfo.calloc(stack).sType$Default().pSetLayouts(ls);
            if (pushSize > 0) {
                VkPushConstantRange.Buffer pr = VkPushConstantRange.calloc(1, stack).stageFlags(pushStages).offset(0).size(pushSize);
                li.pPushConstantRanges(pr);
            }
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreatePipelineLayout(dev, li, null, lp), "pipeline layout");
            return new Layout(l0, lp.get(0), merged, maxSet, l3, merged3);
        }
    }

    public Pipe compute(String shader) {
        return pipes.computeIfAbsent("cs:" + shader, k -> {
            Spirv sp = Spirv.parse(spirv(shader));
            Layout l = layout(List.of(sp), VK_SHADER_STAGE_COMPUTE_BIT, 0, 0);
            try (MemoryStack stack = stackPush()) {
                long mod = module(spirv(shader));
                VkComputePipelineCreateInfo.Buffer ci = VkComputePipelineCreateInfo.calloc(1, stack).sType$Default().layout(l.pipeline);
                ci.stage().sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(mod).pName(stack.UTF8("main"));
                LongBuffer lp = stack.mallocLong(1);
                check(vkCreateComputePipelines(dev, VK_NULL_HANDLE, ci, null, lp), "compute pipeline " + shader);
                vkDestroyShaderModule(dev, mod, null);
                return new Pipe(shader, lp.get(0), l.pipeline, VK_PIPELINE_BIND_POINT_COMPUTE, l.set0, l.bindings, l.maxSet,
                        new int[]{sp.localX, sp.localY, sp.localZ}, l.set3, l.bindings3);
            }
        });
    }

    public Pipe raster(String vs, String ps, int[] colorFormats, int depthFormat, int pushSize, boolean cull) {
        return pipes.computeIfAbsent("gfx:" + vs + ps, k -> {
            Spirv svs = Spirv.parse(spirv(vs)), sps = Spirv.parse(spirv(ps));
            int stages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
            Layout l = layout(List.of(svs, sps), stages, pushSize, VK_SHADER_STAGE_ALL_GRAPHICS);
            try (MemoryStack stack = stackPush()) {
                long mv = module(spirv(vs)), mp = module(spirv(ps));
                VkPipelineShaderStageCreateInfo.Buffer st = VkPipelineShaderStageCreateInfo.calloc(2, stack);
                st.get(0).sType$Default().stage(VK_SHADER_STAGE_VERTEX_BIT).module(mv).pName(stack.UTF8("main"));
                st.get(1).sType$Default().stage(VK_SHADER_STAGE_FRAGMENT_BIT).module(mp).pName(stack.UTF8("main"));
                VkPipelineColorBlendAttachmentState.Buffer cba = VkPipelineColorBlendAttachmentState.calloc(colorFormats.length, stack);
                for (int i = 0; i < colorFormats.length; i++) cba.get(i).colorWriteMask(0xF);
                IntBuffer cf = stack.mallocInt(colorFormats.length);
                for (int i = 0; i < colorFormats.length; i++) cf.put(i, colorFormats[i]);
                VkPipelineRenderingCreateInfo ri = VkPipelineRenderingCreateInfo.calloc(stack).sType$Default().pColorAttachmentFormats(cf).depthAttachmentFormat(depthFormat);
                VkGraphicsPipelineCreateInfo.Buffer ci = VkGraphicsPipelineCreateInfo.calloc(1, stack).sType$Default().pNext(ri.address())
                        .pStages(st)
                        .pVertexInputState(VkPipelineVertexInputStateCreateInfo.calloc(stack).sType$Default())
                        .pInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo.calloc(stack).sType$Default().topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST))
                        .pViewportState(VkPipelineViewportStateCreateInfo.calloc(stack).sType$Default().viewportCount(1).scissorCount(1))
                        .pRasterizationState(VkPipelineRasterizationStateCreateInfo.calloc(stack).sType$Default().polygonMode(VK_POLYGON_MODE_FILL)
                                .cullMode(cull ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE).frontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE).lineWidth(1f))
                        .pMultisampleState(VkPipelineMultisampleStateCreateInfo.calloc(stack).sType$Default().rasterizationSamples(VK_SAMPLE_COUNT_1_BIT))
                        .pDepthStencilState(VkPipelineDepthStencilStateCreateInfo.calloc(stack).sType$Default().depthTestEnable(true).depthWriteEnable(true)
                                .depthCompareOp(VK_COMPARE_OP_GREATER_OR_EQUAL).maxDepthBounds(1f))
                        .pColorBlendState(VkPipelineColorBlendStateCreateInfo.calloc(stack).sType$Default().pAttachments(cba))
                        .pDynamicState(VkPipelineDynamicStateCreateInfo.calloc(stack).sType$Default().pDynamicStates(stack.ints(VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR)))
                        .layout(l.pipeline);
                LongBuffer lp = stack.mallocLong(1);
                check(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, ci, null, lp), "graphics pipeline " + vs);
                vkDestroyShaderModule(dev, mv, null);
                vkDestroyShaderModule(dev, mp, null);
                return new Pipe(vs, lp.get(0), l.pipeline, VK_PIPELINE_BIND_POINT_GRAPHICS, l.set0, l.bindings, l.maxSet, new int[]{1, 1, 1}, l.set3, l.bindings3);
            }
        });
    }

    public Pipe rayTracing(String rgen, List<String> miss, List<String> hit) {
        return pipes.computeIfAbsent("rt:" + rgen + miss + hit, k -> {
            List<String> all = new ArrayList<>();
            all.add(rgen);
            all.addAll(miss);
            all.addAll(hit);
            List<Spirv> sps = new ArrayList<>();
            for (String s : all) sps.add(Spirv.parse(spirv(s)));
            int stages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
            Layout l = layout(sps, stages, 0, 0);
            try (MemoryStack stack = stackPush()) {
                int n = all.size();
                VkPipelineShaderStageCreateInfo.Buffer st = VkPipelineShaderStageCreateInfo.calloc(n, stack);
                VkRayTracingShaderGroupCreateInfoKHR.Buffer gr = VkRayTracingShaderGroupCreateInfoKHR.calloc(n, stack);
                long[] mods = new long[n];
                for (int i = 0; i < n; i++) {
                    mods[i] = module(spirv(all.get(i)));
                    int stage = i == 0 ? VK_SHADER_STAGE_RAYGEN_BIT_KHR : i <= miss.size() ? VK_SHADER_STAGE_MISS_BIT_KHR : VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
                    st.get(i).sType$Default().stage(stage).module(mods[i]).pName(stack.UTF8("main"));
                    VkRayTracingShaderGroupCreateInfoKHR g = gr.get(i).sType$Default().generalShader(VK_SHADER_UNUSED_KHR)
                            .closestHitShader(VK_SHADER_UNUSED_KHR).anyHitShader(VK_SHADER_UNUSED_KHR).intersectionShader(VK_SHADER_UNUSED_KHR);
                    if (stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR) g.type(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR).closestHitShader(i);
                    else g.type(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR).generalShader(i);
                }
                VkRayTracingPipelineCreateInfoKHR.Buffer ci = VkRayTracingPipelineCreateInfoKHR.calloc(1, stack).sType$Default()
                        .pStages(st).pGroups(gr).maxPipelineRayRecursionDepth(1).layout(l.pipeline);
                LongBuffer lp = stack.mallocLong(1);
                check(vkCreateRayTracingPipelinesKHR(dev, VK_NULL_HANDLE, VK_NULL_HANDLE, ci, null, lp), "rt pipeline " + rgen);
                for (long m : mods) vkDestroyShaderModule(dev, m, null);
                Pipe p = new Pipe(rgen, lp.get(0), l.pipeline, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, l.set0, l.bindings, l.maxSet, new int[]{1, 1, 1}, l.set3, l.bindings3);
                buildSbt(p, n, miss.size(), hit.size());
                return p;
            }
        });
    }

    private void buildSbt(Pipe p, int groups, int misses, int hits) {
        int hs = ctx.shaderGroupHandleSize;
        int stride = (int) align(hs, ctx.shaderGroupHandleAlignment);
        ByteBuffer handles = org.lwjgl.system.MemoryUtil.memAlloc(hs * groups);
        check(vkGetRayTracingShaderGroupHandlesKHR(dev, p.pipeline, 0, groups, handles), "sbt handles");
        int[] first = {0, 1, 1 + misses};
        int[] count = {1, misses, hits};
        p.sbt = new long[9];
        p.sbtBuffers = new Buffer[3];
        for (int r = 0; r < 3; r++) {
            if (count[r] == 0) continue;
            ByteBuffer data = org.lwjgl.system.MemoryUtil.memAlloc(stride * count[r]);
            org.lwjgl.system.MemoryUtil.memSet(data, 0);
            for (int i = 0; i < count[r]; i++) {
                for (int b = 0; b < hs; b++) data.put(i * stride + b, handles.get((first[r] + i) * hs + b));
            }
            Buffer buf = Buffer.deviceLocal(ctx, data, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
            p.sbtBuffers[r] = buf;
            p.sbt[r * 3] = buf.address;
            p.sbt[r * 3 + 1] = stride;
            p.sbt[r * 3 + 2] = (long) stride * count[r];
            org.lwjgl.system.MemoryUtil.memFree(data);
        }
        org.lwjgl.system.MemoryUtil.memFree(handles);
    }

    public long descriptorPool() { return descriptorPool; }

    public void submit(Runnable record) {
        beginFrame();
        try (MemoryStack stack = stackPush()) {
            PointerBuffer pp = stack.mallocPointer(1);
            check(vkAllocateCommandBuffers(dev, VkCommandBufferAllocateInfo.calloc(stack).sType$Default().commandPool(ctx.commandPool)
                    .level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1), pp), "alloc cmd");
            cmd = new VkCommandBuffer(pp.get(0), dev);
            check(vkBeginCommandBuffer(cmd, VkCommandBufferBeginInfo.calloc(stack).sType$Default()
                    .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)), "begin cmd");
            record.run();
            check(vkEndCommandBuffer(cmd), "end cmd");
            check(vkQueueSubmit(ctx.queue, VkSubmitInfo.calloc(stack).sType$Default().pCommandBuffers(pp), VK_NULL_HANDLE), "submit");
            check(vkQueueWaitIdle(ctx.queue), "wait");
            vkFreeCommandBuffers(dev, ctx.commandPool, pp);
        }
        cmd = null;
        endFrame();
    }

    public float[] readback(Img img) {
        int bpp = switch (img.desc.format()) {
            case VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32_SFLOAT -> 8;
            case VK_FORMAT_R32G32B32A32_SFLOAT -> 16;
            case VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_A2R10G10B10_UNORM_PACK32 -> 4;
            case VK_FORMAT_R16_SFLOAT -> 2;
            case VK_FORMAT_R8_UNORM -> 1;
            default -> throw new IllegalArgumentException("readback format " + img.desc.format());
        };
        long size = (long) img.width() * img.height() * bpp;
        try (Buffer rb = new Buffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, Buffer.READBACK)) {
            ctx.submitImmediate(c -> {
                try (MemoryStack stack = stackPush()) {
                    VkBufferImageCopy.Buffer r = VkBufferImageCopy.calloc(1, stack);
                    r.get(0).imageSubresource().aspectMask(img.aspect).layerCount(1);
                    r.get(0).imageExtent().set(img.width(), img.height(), 1);
                    vkCmdCopyImageToBuffer(c, img.image, VK_IMAGE_LAYOUT_GENERAL, rb.handle, r);
                }
            });
            ByteBuffer m = rb.map().order(java.nio.ByteOrder.LITTLE_ENDIAN);
            float[] out;
            switch (img.desc.format()) {
                case VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16_SFLOAT -> {
                    out = new float[(int) (size / 2)];
                    for (int i = 0; i < out.length; i++) out[i] = Float.float16ToFloat(m.getShort(i * 2));
                }
                case VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8_UNORM -> {
                    out = new float[(int) size];
                    for (int i = 0; i < out.length; i++) out[i] = (m.get(i) & 0xFF) / 255f;
                }
                case VK_FORMAT_A2R10G10B10_UNORM_PACK32 -> {
                    out = new float[(int) size];
                    for (int i = 0; i < size / 4; i++) {
                        int v = m.getInt(i * 4);
                        out[i * 4] = ((v >> 20) & 1023) / 1023f;
                        out[i * 4 + 1] = ((v >> 10) & 1023) / 1023f;
                        out[i * 4 + 2] = (v & 1023) / 1023f;
                        out[i * 4 + 3] = ((v >> 30) & 3) / 3f;
                    }
                }
                default -> {
                    out = new float[(int) (size / 4)];
                    for (int i = 0; i < out.length; i++) out[i] = m.getFloat(i * 4);
                }
            }
            return out;
        }
    }

    @Override
    public void close() {
        vkDeviceWaitIdle(dev);
        for (Img i : temporals.values()) i.close();
        for (ArrayDeque<Img> q : imgFree.values()) for (Img i : q) i.close();
        for (int k = 0; k < 2; k++) for (Img i : imgUsedS[k]) i.close();
        dyn.close();
    }
}
