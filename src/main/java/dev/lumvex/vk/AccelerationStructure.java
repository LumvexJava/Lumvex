package dev.lumvex.vk;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.KHRAccelerationStructure.*;
import static org.lwjgl.vulkan.VK13.*;

/** A bottom- or top-level acceleration structure, the BVH the RT cores traverse for {@code traceRayEXT}. */
public final class AccelerationStructure implements AutoCloseable {
    private final VkContext ctx;
    public final long handle;
    public final Buffer storage;
    public final long address;

    private AccelerationStructure(VkContext ctx, int type, long size) {
        this.ctx = ctx;
        storage = new Buffer(ctx, size,
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                Buffer.DEVICE);
        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureCreateInfoKHR info = VkAccelerationStructureCreateInfoKHR.calloc(stack)
                    .sType$Default()
                    .buffer(storage.handle)
                    .size(size)
                    .type(type);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateAccelerationStructureKHR(ctx.device, info, null, lp), "vkCreateAccelerationStructureKHR");
            handle = lp.get(0);
            address = vkGetAccelerationStructureDeviceAddressKHR(ctx.device,
                    VkAccelerationStructureDeviceAddressInfoKHR.calloc(stack).sType$Default().accelerationStructure(handle));
        }
    }

    /** Builds a BLAS for an indexed triangle mesh and waits for the build to finish. */
    public static AccelerationStructure buildBottomLevel(VkContext ctx, Buffer vertices, int vertexCount,
                                                         int vertexStride, Buffer indices, int triangleCount) {
        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureGeometryKHR.Buffer geometry = VkAccelerationStructureGeometryKHR.calloc(1, stack);
            geometry.get(0)
                    .sType$Default()
                    .geometryType(VK_GEOMETRY_TYPE_TRIANGLES_KHR)
                    .flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
            VkAccelerationStructureGeometryTrianglesDataKHR tri = geometry.get(0).geometry().triangles();
            tri.sType$Default()
                    .vertexFormat(VK_FORMAT_R32G32B32_SFLOAT)
                    .vertexStride(vertexStride)
                    .maxVertex(vertexCount - 1)
                    .indexType(VK_INDEX_TYPE_UINT32);
            tri.vertexData().deviceAddress(vertices.address);
            tri.indexData().deviceAddress(indices.address);

            return build(ctx, stack, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, geometry, triangleCount,
                    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
        }
    }

    /**
     * Creates an unbuilt TLAS with room for {@code maxInstances}; {@code scratchSizeOut} receives the scratch size,
     * large enough for both full builds and refits.
     */
    public static AccelerationStructure createTopLevel(VkContext ctx, Buffer instanceBuffer, int maxInstances, long[] scratchSizeOut) {
        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureGeometryKHR.Buffer geometry = topLevelGeometry(stack, instanceBuffer);
            VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfo = buildInfo(stack,
                    VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, geometry, TLAS_FLAGS);
            VkAccelerationStructureBuildSizesInfoKHR sizes = VkAccelerationStructureBuildSizesInfoKHR.calloc(stack).sType$Default();
            vkGetAccelerationStructureBuildSizesKHR(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                    buildInfo.get(0), stack.ints(maxInstances), sizes);
            scratchSizeOut[0] = Math.max(sizes.buildScratchSize(), sizes.updateScratchSize());
            return new AccelerationStructure(ctx, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizes.accelerationStructureSize());
        }
    }

    /** Refits are allowed: moving a few instances only updates the bounding volumes above them. */
    private static final int TLAS_FLAGS = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
            | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

    /**
     * Records a TLAS build: a full one, or with {@code refit} an update of the last build with the same instances in
     * new places, which costs a fraction of a build but lets the tree degrade as instances move far. The caller is
     * responsible for the barrier before tracing.
     */
    public void recordTopLevelBuild(VkCommandBuffer cmd, Buffer instanceBuffer, int instanceCount, long scratchAddress, boolean refit) {
        try (MemoryStack stack = stackPush()) {
            VkAccelerationStructureGeometryKHR.Buffer geometry = topLevelGeometry(stack, instanceBuffer);
            VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfo = buildInfo(stack,
                    VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, geometry, TLAS_FLAGS);
            buildInfo.get(0).dstAccelerationStructure(handle).scratchData().deviceAddress(scratchAddress);
            if (refit) buildInfo.get(0).mode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR).srcAccelerationStructure(handle);
            VkAccelerationStructureBuildRangeInfoKHR.Buffer range = VkAccelerationStructureBuildRangeInfoKHR.calloc(1, stack)
                    .primitiveCount(instanceCount);
            vkCmdBuildAccelerationStructuresKHR(cmd, buildInfo, stack.pointers(range.address()));
        }
    }

    private static VkAccelerationStructureGeometryKHR.Buffer topLevelGeometry(MemoryStack stack, Buffer instanceBuffer) {
        VkAccelerationStructureGeometryKHR.Buffer geometry = VkAccelerationStructureGeometryKHR.calloc(1, stack);
        geometry.get(0)
                .sType$Default()
                .geometryType(VK_GEOMETRY_TYPE_INSTANCES_KHR)
                .flags(VK_GEOMETRY_OPAQUE_BIT_KHR);
        VkAccelerationStructureGeometryInstancesDataKHR inst = geometry.get(0).geometry().instances();
        inst.sType$Default().arrayOfPointers(false);
        inst.data().deviceAddress(instanceBuffer.address);
        return geometry;
    }

    private static VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfo(MemoryStack stack, int type,
                                                                                VkAccelerationStructureGeometryKHR.Buffer geometry, int flags) {
        VkAccelerationStructureBuildGeometryInfoKHR.Buffer info = VkAccelerationStructureBuildGeometryInfoKHR.calloc(1, stack);
        info.get(0)
                .sType$Default()
                .type(type)
                .flags(flags)
                .mode(VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR)
                .geometryCount(1)
                .pGeometries(geometry);
        return info;
    }

    /** Creates and builds a structure synchronously with a temporary scratch buffer. */
    private static AccelerationStructure build(VkContext ctx, MemoryStack stack, int type,
                                               VkAccelerationStructureGeometryKHR.Buffer geometry, int primitiveCount, int flags) {
        VkAccelerationStructureBuildGeometryInfoKHR.Buffer buildInfo = buildInfo(stack, type, geometry, flags);
        VkAccelerationStructureBuildSizesInfoKHR sizes = VkAccelerationStructureBuildSizesInfoKHR.calloc(stack).sType$Default();
        vkGetAccelerationStructureBuildSizesKHR(ctx.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                buildInfo.get(0), stack.ints(primitiveCount), sizes);

        AccelerationStructure as = new AccelerationStructure(ctx, type, sizes.accelerationStructureSize());
        long align = ctx.minScratchOffsetAlignment;
        try (Buffer scratch = new Buffer(ctx, sizes.buildScratchSize() + align,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, Buffer.DEVICE)) {
            buildInfo.get(0).dstAccelerationStructure(as.handle)
                    .scratchData().deviceAddress(VkUtil.align(scratch.address, align));
            VkAccelerationStructureBuildRangeInfoKHR.Buffer range = VkAccelerationStructureBuildRangeInfoKHR.calloc(1, stack)
                    .primitiveCount(primitiveCount);
            ctx.submitImmediate(cmd -> vkCmdBuildAccelerationStructuresKHR(cmd, buildInfo, stack.pointers(range.address())));
        }
        return as;
    }

    @Override
    public void close() {
        vkDestroyAccelerationStructureKHR(ctx.device, handle, null);
        storage.close();
    }
}
