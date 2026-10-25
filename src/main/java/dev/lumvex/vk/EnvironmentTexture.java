package dev.lumvex.vk;

import dev.lumvex.render.scene.Environment;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK13.*;

/** The HDR environment as an RGBA16F equirectangular texture; wraps horizontally and clamps vertically. */
public final class EnvironmentTexture implements AutoCloseable {
    private final VkContext ctx;
    public final long image, memory, view, sampler;

    /** With a null environment a 1x1 black placeholder is created, so the descriptor stays valid. */
    public EnvironmentTexture(VkContext ctx, Environment env) {
        this.ctx = ctx;
        int w = env != null ? env.width : 1, h = env != null ? env.height : 1;
        try (MemoryStack stack = stackPush()) {
            VkImageCreateInfo info = VkImageCreateInfo.calloc(stack)
                    .sType$Default()
                    .imageType(VK_IMAGE_TYPE_2D)
                    .format(VK_FORMAT_R16G16B16A16_SFLOAT)
                    .mipLevels(1)
                    .arrayLayers(1)
                    .samples(VK_SAMPLE_COUNT_1_BIT)
                    .tiling(VK_IMAGE_TILING_OPTIMAL)
                    .usage(VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                    .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
            info.extent().set(w, h, 1);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateImage(ctx.device, info, null, lp), "vkCreateImage(env)");
            image = lp.get(0);
            VkMemoryRequirements req = VkMemoryRequirements.malloc(stack);
            vkGetImageMemoryRequirements(ctx.device, image, req);
            check(vkAllocateMemory(ctx.device, VkMemoryAllocateInfo.calloc(stack)
                    .sType$Default()
                    .allocationSize(req.size())
                    .memoryTypeIndex(ctx.findMemoryType(req.memoryTypeBits(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), null, lp),
                    "vkAllocateMemory(env)");
            memory = lp.get(0);
            check(vkBindImageMemory(ctx.device, image, memory, 0), "vkBindImageMemory(env)");
            VkImageViewCreateInfo vi = VkImageViewCreateInfo.calloc(stack)
                    .sType$Default().image(image).viewType(VK_IMAGE_VIEW_TYPE_2D).format(VK_FORMAT_R16G16B16A16_SFLOAT);
            vi.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
            check(vkCreateImageView(ctx.device, vi, null, lp), "vkCreateImageView(env)");
            view = lp.get(0);
            check(vkCreateSampler(ctx.device, VkSamplerCreateInfo.calloc(stack)
                    .sType$Default()
                    .magFilter(VK_FILTER_LINEAR).minFilter(VK_FILTER_LINEAR)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .maxLod(0f), null, lp), "vkCreateSampler(env)");
            sampler = lp.get(0);
        }

        try (Buffer staging = new Buffer(ctx, (long) w * h * 8, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.HOST)) {
            ByteBuffer px = staging.map();
            for (int i = 0; i < w * h; i++) {
                float r = env != null ? env.rgb[i * 3] : 0, g = env != null ? env.rgb[i * 3 + 1] : 0, b = env != null ? env.rgb[i * 3 + 2] : 0;
                px.putShort(i * 8, Float.floatToFloat16(Math.min(r, 65000f)))
                        .putShort(i * 8 + 2, Float.floatToFloat16(Math.min(g, 65000f)))
                        .putShort(i * 8 + 4, Float.floatToFloat16(Math.min(b, 65000f)))
                        .putShort(i * 8 + 6, Float.floatToFloat16(1f));
            }
            ctx.submitImmediate(cmd -> {
                try (MemoryStack stack = stackPush()) {
                    barrier(cmd, stack, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
                    VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(1, stack);
                    region.get(0).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
                    region.get(0).imageExtent().set(w, h, 1);
                    vkCmdCopyBufferToImage(cmd, staging.handle, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
                    barrier(cmd, stack, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
                }
            });
        }
    }

    private void barrier(VkCommandBuffer cmd, MemoryStack stack, int oldL, int newL, int src, int dst) {
        VkImageMemoryBarrier.Buffer b = VkImageMemoryBarrier.calloc(1, stack);
        b.get(0).sType$Default().srcAccessMask(src).dstAccessMask(dst).oldLayout(oldL).newLayout(newL)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).image(image)
                .subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, null, null, b);
    }

    @Override
    public void close() {
        vkDestroySampler(ctx.device, sampler, null);
        vkDestroyImageView(ctx.device, view, null);
        vkDestroyImage(ctx.device, image, null);
        vkFreeMemory(ctx.device, memory, null);
    }
}
