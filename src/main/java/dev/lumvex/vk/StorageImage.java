package dev.lumvex.vk;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK13.*;

/** A 2D storage image that stays in {@code VK_IMAGE_LAYOUT_GENERAL} for its whole lifetime. */
public final class StorageImage implements AutoCloseable {
    private final VkContext ctx;
    public final long image;
    public final long memory;
    public final long view;
    public final int format;
    public final int width;
    public final int height;

    public StorageImage(VkContext ctx, int width, int height, int format) {
        this.ctx = ctx;
        this.width = width;
        this.height = height;
        this.format = format;
        VkDevice device = ctx.device;
        try (MemoryStack stack = stackPush()) {
            VkImageCreateInfo info = VkImageCreateInfo.calloc(stack)
                    .sType$Default()
                    .imageType(VK_IMAGE_TYPE_2D)
                    .format(format)
                    .mipLevels(1)
                    .arrayLayers(1)
                    .samples(VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
            info.extent().set(width, height, 1);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateImage(device, info, null, lp), "vkCreateImage");
            image = lp.get(0);

            VkMemoryRequirements req = VkMemoryRequirements.malloc(stack);
            vkGetImageMemoryRequirements(device, image, req);
            VkMemoryAllocateInfo alloc = VkMemoryAllocateInfo.calloc(stack)
                    .sType$Default()
                    .allocationSize(req.size())
                    .memoryTypeIndex(ctx.findMemoryType(req.memoryTypeBits(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
            check(vkAllocateMemory(device, alloc, null, lp), "vkAllocateMemory(image)");
            memory = lp.get(0);
            check(vkBindImageMemory(device, image, memory, 0), "vkBindImageMemory");

            VkImageViewCreateInfo viewInfo = VkImageViewCreateInfo.calloc(stack)
                    .sType$Default()
                    .image(image)
                    .viewType(VK_IMAGE_VIEW_TYPE_2D)
                    .format(format);
            viewInfo.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
            check(vkCreateImageView(device, viewInfo, null, lp), "vkCreateImageView");
            view = lp.get(0);
        }

        ctx.submitImmediate(cmd -> {
            try (MemoryStack stack = stackPush()) {
                VkImageMemoryBarrier.Buffer barrier = VkImageMemoryBarrier.calloc(1, stack);
                barrier.get(0)
                        .sType$Default()
                        .srcAccessMask(0)
                        .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT)
                        .oldLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                        .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                        .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                        .image(image)
                        .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1));
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        0, null, null, barrier);
            }
        });
    }

    @Override
    public void close() {
        vkDestroyImageView(ctx.device, view, null);
        vkDestroyImage(ctx.device, image, null);
        vkFreeMemory(ctx.device, memory, null);
    }
}
