package dev.lumvex.vk;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.memByteBuffer;
import static org.lwjgl.system.MemoryUtil.memCopy;
import static org.lwjgl.vulkan.VK13.*;

/** A buffer with a dedicated memory allocation and, if requested, a device address. */
public final class Buffer implements AutoCloseable {

    public static final int HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    public static final int DEVICE = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    /**
     * For data the CPU writes and shaders read frequently: device-local memory that is mapped through the PCIe BAR.
     * Falls back to {@link #HOST} where no such memory type exists.
     */
    public static final int UPLOAD = HOST | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    /** For data the CPU reads back: cached system memory, since uncached reads cost about 0.1 us each. */
    public static final int READBACK = HOST | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;

    private final VkContext ctx;
    public final long handle;
    public final long memory;
    public final long size;
    public final long address;
    private ByteBuffer mapped;

    public Buffer(VkContext ctx, long size, int usage, int memoryFlags) {
        this.ctx = ctx;
        this.size = size;
        VkDevice device = ctx.device;
        boolean wantsAddress = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
        try (MemoryStack stack = stackPush()) {
            VkBufferCreateInfo info = VkBufferCreateInfo.calloc(stack)
                    .sType$Default()
                    .size(size)
                    .usage(usage)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateBuffer(device, info, null, lp), "vkCreateBuffer");
            handle = lp.get(0);

            VkMemoryRequirements req = VkMemoryRequirements.malloc(stack);
            vkGetBufferMemoryRequirements(device, handle, req);

            VkMemoryAllocateInfo alloc = VkMemoryAllocateInfo.calloc(stack)
                    .sType$Default()
                    .allocationSize(req.size())
                    .memoryTypeIndex(memoryType(ctx, req.memoryTypeBits(), memoryFlags));
            if (wantsAddress) {
                VkMemoryAllocateFlagsInfo flags = VkMemoryAllocateFlagsInfo.calloc(stack)
                        .sType$Default()
                        .flags(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
                alloc.pNext(flags.address());
            }
            check(vkAllocateMemory(device, alloc, null, lp), "vkAllocateMemory");
            memory = lp.get(0);
            check(vkBindBufferMemory(device, handle, memory, 0), "vkBindBufferMemory");

            if (wantsAddress) {
                address = vkGetBufferDeviceAddress(device, VkBufferDeviceAddressInfo.calloc(stack)
                        .sType$Default()
                        .buffer(handle));
            } else {
                address = 0;
            }
        }
    }

    private static int memoryType(VkContext ctx, int typeBits, int flags) {
        try {
            return ctx.findMemoryType(typeBits, flags);
        } catch (IllegalStateException e) {
            if (flags == UPLOAD || flags == READBACK) return ctx.findMemoryType(typeBits, HOST);
            throw e;
        }
    }

    /** Maps the buffer persistently; it must be host-visible. */
    public ByteBuffer map() {
        if (mapped == null) {
            try (MemoryStack stack = stackPush()) {
                PointerBuffer pp = stack.mallocPointer(1);
                check(vkMapMemory(ctx.device, memory, 0, size, 0, pp), "vkMapMemory");
                mapped = memByteBuffer(pp.get(0), (int) size);
            }
        }
        return mapped;
    }

    /** Creates a device-local buffer and uploads {@code data} through a staging buffer. */
    public static Buffer deviceLocal(VkContext ctx, ByteBuffer data, int usage) {
        long size = data.remaining();
        try (Buffer staging = new Buffer(ctx, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HOST)) {
            memCopy(data, staging.map());
            Buffer dst = new Buffer(ctx, size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, DEVICE);
            ctx.submitImmediate(cmd -> {
                try (MemoryStack stack = stackPush()) {
                    VkBufferCopy.Buffer region = VkBufferCopy.calloc(1, stack).size(size);
                    vkCmdCopyBuffer(cmd, staging.handle, dst.handle, region);
                }
            });
            return dst;
        }
    }

    @Override
    public void close() {
        if (mapped != null) vkUnmapMemory(ctx.device, memory);
        vkDestroyBuffer(ctx.device, handle, null);
        vkFreeMemory(ctx.device, memory, null);
    }
}
