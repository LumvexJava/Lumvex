// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import dev.lumvex.vk.Buffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
import static org.lwjgl.vulkan.VK13.*;

public final class Bindless implements AutoCloseable {
    public static final int MAX_MESHES = 8192;
    public static final int MESH_STRIDE = 28;
    public static final long VERTEX_CAPACITY = 1000L * 1024 * 1024;
    public static final int LUT_BRDF_FG = 0, LUT_BLUE_NOISE = 1, LUT_BEZOLD_BRUCKE = 2;

    private final Gfx g;
    public final Buffer meshes;
    public final Buffer vertices;
    public final Buffer textureSizes;
    public long vertexWritten;
    private int nextImage;
    private final ByteBuffer sizes;
    private final ByteBuffer meshMap;

    public Bindless(Gfx g) {
        this.g = g;
        meshes = new Buffer(g.ctx, (long) MAX_MESHES * MESH_STRIDE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        vertices = new Buffer(g.ctx, VERTEX_CAPACITY, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, Buffer.DEVICE);
        textureSizes = new Buffer(g.ctx, (long) Gfx.BINDLESS_COUNT * 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        sizes = textureSizes.map().order(ByteOrder.LITTLE_ENDIAN);
        meshMap = meshes.map().order(ByteOrder.LITTLE_ENDIAN);
        try (MemoryStack stack = stackPush()) {
            VkDescriptorBufferInfo.Buffer bi = VkDescriptorBufferInfo.calloc(3, stack);
            bi.get(0).buffer(meshes.handle).range(VK_WHOLE_SIZE);
            bi.get(1).buffer(vertices.handle).range(VK_WHOLE_SIZE);
            bi.get(2).buffer(textureSizes.handle).range(VK_WHOLE_SIZE);
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(3, stack);
            for (int i = 0; i < 3; i++) {
                w.get(i).sType$Default().dstSet(g.bindlessSet).dstBinding(i).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                        .pBufferInfo(VkDescriptorBufferInfo.create(bi.address() + (long) i * VkDescriptorBufferInfo.SIZEOF, 1));
            }
            vkUpdateDescriptorSets(g.dev, w, null);
        }
    }

    public int addImage(Img img) {
        int id = nextImage++;
        try (MemoryStack stack = stackPush()) {
            VkDescriptorImageInfo.Buffer ii = VkDescriptorImageInfo.calloc(1, stack).imageView(img.view(Img.View.DEFAULT)).imageLayout(VK_IMAGE_LAYOUT_GENERAL);
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(1, stack).sType$Default().dstSet(g.bindlessSet).dstBinding(3)
                    .dstArrayElement(id).descriptorType(VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE).descriptorCount(1).pImageInfo(ii);
            vkUpdateDescriptorSets(g.dev, w, null);
        }
        float[] e = img.desc.extentInvExtent();
        for (int i = 0; i < 4; i++) sizes.putFloat(id * 16 + i * 4, e[i]);
        return id;
    }

    public int imageCount() { return nextImage; }

    public int appendVertexData(ByteBuffer data) {
        int off = (int) vertexWritten;
        int size = data.remaining();
        if (vertexWritten + size > VERTEX_CAPACITY) throw new IllegalStateException("vertex buffer full");
        try (Buffer staging = new Buffer(g.ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.HOST)) {
            staging.map().put(data.duplicate());
            g.ctx.submitImmediate(cmd -> {
                try (MemoryStack stack = stackPush()) {
                    vkCmdCopyBuffer(cmd, staging.handle, vertices.handle, VkBufferCopy.calloc(1, stack).dstOffset(off).size(size));
                }
            });
        }
        vertexWritten += (size + 15) & ~15;
        return off;
    }

    public void setMesh(int idx, int[] offsets) {
        for (int i = 0; i < 7; i++) meshMap.putInt(idx * MESH_STRIDE + i * 4, offsets[i]);
    }

    @Override
    public void close() {
        meshes.close();
        vertices.close();
        textureSizes.close();
    }
}
