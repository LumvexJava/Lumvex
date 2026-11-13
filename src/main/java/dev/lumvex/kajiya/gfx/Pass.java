// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import dev.lumvex.vk.Buffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.KHRAccelerationStructure.*;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.*;
import static org.lwjgl.vulkan.VK13.*;

public final class Pass {

    private record ImgBind(Img img, Img.View view, int mip, int mips) {}

    private record Const(int offset) {}

    private record Accel(long handle) {}

    private static final java.util.Set<String> WARNED = new java.util.HashSet<>();
    private final Gfx g;
    private final Pipe pipe;
    private final List<Object> binds = new ArrayList<>();
    private byte[] push;

    public Pass(Gfx g, Pipe pipe) {
        this.g = g;
        this.pipe = pipe;
    }

    public static Pass compute(Gfx g, String shader) { return new Pass(g, g.compute(shader)); }

    public static Pass rt(Gfx g, String rgen, List<String> miss, List<String> hit) { return new Pass(g, g.rayTracing(rgen, miss, hit)); }

    public Pass bind(Img i) { binds.add(new ImgBind(i, Img.View.DEFAULT, 0, i.desc.mips())); return this; }

    public Pass bind(Img i, Img.View view) { binds.add(new ImgBind(i, view, 0, i.desc.mips())); return this; }

    public Pass bindMip(Img i, int mip) { binds.add(new ImgBind(i, Img.View.DEFAULT, mip, 1)); return this; }

    public Pass bind(Buffer b) { binds.add(b); return this; }

    public Pass bindAccel(long handle) { binds.add(new Accel(handle)); return this; }

    public Pass tlas(long handle) { tlas = handle; return this; }

    private long tlas;

    public Pass skip() { binds.add(null); return this; }

    public Pass constants(Cb cb) { binds.add(new Const(g.dynPush(cb.data()))); return this; }

    public Pass constantsAt(int offset) { binds.add(new Const(offset)); return this; }

    public Pass push(byte[] data) { push = data; return this; }

    public Pass dispatch(int x, int y, int z) {
        if (x <= 0 || y <= 0 || z <= 0) return this;
        begin();
        int[] l = pipe.local;
        vkCmdDispatch(g.cmd, (x + l[0] - 1) / l[0], (y + l[1] - 1) / l[1], (z + l[2] - 1) / l[2]);
        return this;
    }

    public Pass dispatchIndirect(Buffer args, long offset) {
        begin();
        vkCmdDispatchIndirect(g.cmd, args.handle, offset);
        return this;
    }

    public Pass traceRays(int x, int y, int z) {
        if (x <= 0 || y <= 0 || z <= 0) return this;
        begin();
        try (MemoryStack stack = stackPush()) {
            vkCmdTraceRaysKHR(g.cmd, region(stack, 0), region(stack, 1), region(stack, 2),
                    VkStridedDeviceAddressRegionKHR.calloc(stack), x, y, z);
        }
        return this;
    }

    public Pass traceRaysIndirect(Buffer args, long offset) {
        begin();
        try (MemoryStack stack = stackPush()) {
            vkCmdTraceRaysIndirectKHR(g.cmd, region(stack, 0), region(stack, 1), region(stack, 2),
                    VkStridedDeviceAddressRegionKHR.calloc(stack), args.address + offset);
        }
        return this;
    }

    private VkStridedDeviceAddressRegionKHR region(MemoryStack stack, int r) {
        return VkStridedDeviceAddressRegionKHR.calloc(stack).deviceAddress(pipe.sbt[r * 3]).stride(pipe.sbt[r * 3 + 1]).size(pipe.sbt[r * 3 + 2]);
    }

    public void begin() {
        g.barrier();
        vkCmdBindPipeline(g.cmd, pipe.bindPoint, pipe.pipeline);
        try (MemoryStack stack = stackPush()) {
            if (pipe.maxSet >= 2) {
                vkCmdBindDescriptorSets(g.cmd, pipe.bindPoint, pipe.layout, 2, stack.longs(g.frameSet), frameOffsets);
            }
            if (pipe.maxSet >= 1) {
                vkCmdBindDescriptorSets(g.cmd, pipe.bindPoint, pipe.layout, 1, stack.longs(g.bindlessSet), null);
            }
            if (pipe.set0Layout != 0 && !pipe.set0.isEmpty()) bindSet0(stack);
            if (pipe.maxSet >= 3 && !pipe.set3.isEmpty()) {
                LongBuffer lp = stack.mallocLong(1);
                check(vkAllocateDescriptorSets(g.dev, VkDescriptorSetAllocateInfo.calloc(stack).sType$Default().descriptorPool(g.descriptorPool())
                        .pSetLayouts(stack.longs(pipe.set3Layout)), lp), "alloc set3");
                VkWriteDescriptorSetAccelerationStructureKHR as = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack).sType$Default()
                        .pAccelerationStructures(stack.longs(tlas));
                VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(1, stack).sType$Default().pNext(as.address()).dstSet(lp.get(0)).dstBinding(0)
                        .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
                vkUpdateDescriptorSets(g.dev, w, null);
                vkCmdBindDescriptorSets(g.cmd, pipe.bindPoint, pipe.layout, 3, lp, null);
            }
            if (push != null) {
                java.nio.ByteBuffer pb = stack.malloc(push.length);
                pb.put(push).flip();
                vkCmdPushConstants(g.cmd, pipe.layout, VK_SHADER_STAGE_ALL, 0, pb);
            }
        }
    }

    public static int[] frameOffsetsArray = new int[3];
    private static final java.nio.IntBuffer frameOffsets = org.lwjgl.system.MemoryUtil.memAllocInt(3);

    public static void setFrameOffsets(int globals, int instanceDyn, int triLights) {
        frameOffsets.put(0, globals).put(1, instanceDyn).put(2, triLights);
    }

    private void bindSet0(MemoryStack stack) {
        LongBuffer lp = stack.mallocLong(1);
        check(vkAllocateDescriptorSets(g.dev, VkDescriptorSetAllocateInfo.calloc(stack).sType$Default().descriptorPool(g.descriptorPool())
                .pSetLayouts(stack.longs(pipe.set0Layout)), lp), "alloc set0 for " + pipe.name);
        long set = lp.get(0);
        int n = pipe.set0.size();
        VkWriteDescriptorSet.Buffer writes = VkWriteDescriptorSet.calloc(n, stack);
        int count = 0;
        List<Integer> dynOffsets = new ArrayList<>();
        for (var e : pipe.set0.entrySet()) {
            int idx = e.getKey();
            Spirv.Binding sb = e.getValue();
            if (sb.type() == VK_DESCRIPTOR_TYPE_SAMPLER) continue;
            Object o = idx < binds.size() ? binds.get(idx) : null;
            if (o == null) {
                if (idx < 32 && WARNED.add(pipe.name + "#" + idx)) System.err.println("[kajiya] pass " + pipe.name + ": binding " + idx + " (" + sb.name() + ") not bound");
                continue;
            }
            VkWriteDescriptorSet w = writes.get(count++).sType$Default().dstSet(set).dstBinding(idx).dstArrayElement(0).descriptorCount(1);
            switch (sb.type()) {
                case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE -> {
                    ImgBind ib = (ImgBind) o;
                    VkDescriptorImageInfo.Buffer ii = VkDescriptorImageInfo.calloc(1, stack)
                            .imageView(ib.img.view(ib.view, ib.mip, ib.mips)).imageLayout(VK_IMAGE_LAYOUT_GENERAL);
                    w.descriptorType(sb.type()).pImageInfo(ii);
                }
                case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER -> {
                    boolean dyn = sb.name().endsWith("_dyn");
                    if (dyn) {
                        w.descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
                        VkDescriptorBufferInfo.Buffer bi = VkDescriptorBufferInfo.calloc(1, stack).buffer(g.dyn.handle).range(Gfx.DYN_STORAGE_RANGE);
                        w.pBufferInfo(bi);
                        dynOffsets.add(((Const) o).offset());
                    } else {
                        w.descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
                        w.pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(((Buffer) o).handle).range(VK_WHOLE_SIZE));
                    }
                }
                case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER -> {
                    w.descriptorType(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC);
                    w.pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(g.dyn.handle).range(Gfx.DYN_UNIFORM_RANGE));
                    dynOffsets.add(((Const) o).offset());
                }
                case VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR -> {
                    VkWriteDescriptorSetAccelerationStructureKHR as = VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack).sType$Default()
                            .pAccelerationStructures(stack.longs(((Accel) o).handle()));
                    w.pNext(as.address()).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
                }
                default -> throw new IllegalStateException("pass " + pipe.name + ": unsupported descriptor type " + sb.type());
            }
        }
        writes.limit(count);
        vkUpdateDescriptorSets(g.dev, writes, null);
        java.nio.IntBuffer offs = stack.mallocInt(dynOffsets.size());
        for (int i = 0; i < dynOffsets.size(); i++) offs.put(i, dynOffsets.get(i));
        vkCmdBindDescriptorSets(g.cmd, pipe.bindPoint, pipe.layout, 0, stack.longs(set), offs);
    }
}
