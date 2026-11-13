// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import dev.lumvex.vk.Buffer;
import dev.lumvex.vk.VkContext;
import dev.lumvex.vk.VkUtil;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.KHRAccelerationStructure.*;
import static org.lwjgl.vulkan.VK13.*;

public final class Accel implements AutoCloseable {
    private final VkContext ctx;
    public final long handle;
    public final long address;
    private final Buffer storage;

    private Accel(VkContext ctx, int type, long size) {
        this.ctx = ctx;
        storage = new Buffer(ctx, size, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, Buffer.DEVICE);
        try (MemoryStack stack = stackPush()) {
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateAccelerationStructureKHR(ctx.device, VkAccelerationStructureCreateInfoKHR.calloc(stack).sType$Default()
                    .buffer(storage.handle).size(size).type(type), null, lp), "create AS");
            handle = lp.get(0);
            address = vkGetAccelerationStructureDeviceAddressKHR(ctx.device,
                    VkAccelerationStructureDeviceAddressInfoKHR.calloc(stack).sType$Default().accelerationStructure(handle));
        }
    }

    public static Accel bottom(VkContext ctx, long vertexAddress, int stride, int maxVertex, long indexAddress, int triangles) {
        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureGeometryKHR.Buffer geom = VkAccelerationStructureGeometryKHR.calloc(1, stack);
            geom.get(0).sType$Default().geometryType(VK_GEOMETRY_TYPE_TRIANGLES_KHR).flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
            VkAccelerationStructureGeometryTrianglesDataKHR tri = geom.get(0).geometry().triangles();
            tri.sType$Default().vertexFormat(VK_FORMAT_R32G32B32_SFLOAT).vertexStride(stride).maxVertex(maxVertex).indexType(VK_INDEX_TYPE_UINT32);
            tri.vertexData().deviceAddress(vertexAddress);
            tri.indexData().deviceAddress(indexAddress);
            VkAccelerationStructureBuildGeometryInfoKHR.Buffer bi = VkAccelerationStructureBuildGeometryInfoKHR.calloc(1, stack);
            bi.get(0).sType$Default().type(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR).flags(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                    .mode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR).geometryCount(1).pGeometries(geom);
            VkAccelerationStructureBuildSizesInfoKHR sizes = VkAccelerationStructureBuildSizesInfoKHR.calloc(stack).sType$Default();
            vkGetAccelerationStructureBuildSizesKHR(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, bi.get(0), stack.ints(triangles), sizes);
            Accel as = new Accel(ctx, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes.accelerationStructureSize());
            long align = ctx.minScratchOffsetAlignment;
            try (Buffer scratch = new Buffer(ctx, sizes.buildScratchSize() + align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, Buffer.DEVICE)) {
                bi.get(0).dstAccelerationStructure(as.handle).scratchData().deviceAddress(VkUtil.align(scratch.address, align));
                VkAccelerationStructureBuildRangeInfoKHR.Buffer range = VkAccelerationStructureBuildRangeInfoKHR.calloc(1, stack).primitiveCount(triangles);
                ctx.submitImmediate(cmd -> vkCmdBuildAccelerationStructuresKHR(cmd, bi, stack.pointers(range.address())));
            }
            return as;
        }
    }

    public static Accel top(VkContext ctx, int maxInstances, long[] scratchSize) {
        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureBuildGeometryInfoKHR.Buffer bi = topInfo(stack, 0);
            VkAccelerationStructureBuildSizesInfoKHR sizes = VkAccelerationStructureBuildSizesInfoKHR.calloc(stack).sType$Default();
            vkGetAccelerationStructureBuildSizesKHR(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, bi.get(0), stack.ints(maxInstances), sizes);
            scratchSize[0] = sizes.buildScratchSize();
            return new Accel(ctx, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizes.accelerationStructureSize());
        }
    }

    private static VkAccelerationStructureBuildGeometryInfoKHR.Buffer topInfo(MemoryStack stack, long instanceAddress) {
        VkAccelerationStructureGeometryKHR.Buffer geom = VkAccelerationStructureGeometryKHR.calloc(1, stack);
        geom.get(0).sType$Default().geometryType(VK_GEOMETRY_TYPE_INSTANCES_KHR).flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
        geom.get(0).geometry().instances().sType$Default().arrayOfPointers(false).data().deviceAddress(instanceAddress);
        VkAccelerationStructureBuildGeometryInfoKHR.Buffer bi = VkAccelerationStructureBuildGeometryInfoKHR.calloc(1, stack);
        bi.get(0).sType$Default().type(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR).flags(VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR)
                .mode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR).geometryCount(1).pGeometries(geom);
        return bi;
    }

    public void recordTopBuild(VkCommandBuffer cmd, long instanceAddress, int count, long scratchAddress) {
        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureBuildGeometryInfoKHR.Buffer bi = topInfo(stack, instanceAddress);
            bi.get(0).dstAccelerationStructure(handle).scratchData().deviceAddress(scratchAddress);
            VkAccelerationStructureBuildRangeInfoKHR.Buffer range = VkAccelerationStructureBuildRangeInfoKHR.calloc(1, stack).primitiveCount(count);
            vkCmdBuildAccelerationStructuresKHR(cmd, bi, stack.pointers(range.address()));
        }
    }

    @Override
    public void close() {
        vkDestroyAccelerationStructureKHR(ctx.device, handle, null);
        storage.close();
    }
}
