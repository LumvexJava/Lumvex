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

    /**
     * Uploads the environment with its mip chain, all levels back to back in one staging buffer. With a null
     * environment a 1x1 black placeholder is created, so the descriptor stays valid.
     */
    public EnvironmentTexture(VkContext ctx, Environment env) {
        this.ctx = ctx;
        int w = env != null ? env.width : 1, h = env != null ? env.height : 1;
        levels = 1 + 31 - Integer.numberOfLeadingZeros(Math.max(w, h));
        try (MemoryStack stack = stackPush()) {
            VkImageCreateInfo info = VkImageCreateInfo.calloc(stack)
                    .sType$Default()
                    .imageType(VK_IMAGE_TYPE_2D)
                    .format(VK_FORMAT_R16G16B16A16_SFLOAT)
                    .mipLevels(levels)
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
            vi.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(levels).layerCount(1);
            check(vkCreateImageView(ctx.device, vi, null, lp), "vkCreateImageView(env)");
            view = lp.get(0);
            check(vkCreateSampler(ctx.device, VkSamplerCreateInfo.calloc(stack)
                    .sType$Default()
                    .magFilter(VK_FILTER_LINEAR).minFilter(VK_FILTER_LINEAR)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_LINEAR)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_REPEAT)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .maxLod(VK_LOD_CLAMP_NONE), null, lp), "vkCreateSampler(env)");
            sampler = lp.get(0);
        }

        float[][] mips = new float[levels][];
        int[] mw = new int[levels], mh = new int[levels];
        mips[0] = env != null ? env.rgb : new float[3];
        mw[0] = w;
        mh[0] = h;
        long bytes = 0;
        for (int l = 0; l < levels; l++) {
            if (l > 0) {
                mw[l] = Math.max(1, mw[l - 1] / 2);
                mh[l] = Math.max(1, mh[l - 1] / 2);
                mips[l] = downsample(mips[l - 1], mw[l - 1], mh[l - 1], mw[l], mh[l]);
            }
            bytes += (long) mw[l] * mh[l] * 8;
        }
        try (Buffer staging = new Buffer(ctx, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.HOST)) {
            ByteBuffer px = staging.map();
            long[] offset = new long[levels];
            int at = 0;
            for (int l = 0; l < levels; l++) {
                offset[l] = at;
                float[] rgb = mips[l];
                for (int i = 0; i < mw[l] * mh[l]; i++) {
                    px.putShort(at, Float.floatToFloat16(Math.min(rgb[i * 3], 65000f)))
                            .putShort(at + 2, Float.floatToFloat16(Math.min(rgb[i * 3 + 1], 65000f)))
                            .putShort(at + 4, Float.floatToFloat16(Math.min(rgb[i * 3 + 2], 65000f)))
                            .putShort(at + 6, Float.floatToFloat16(1f));
                    at += 8;
                }
            }
            ctx.submitImmediate(cmd -> {
                try (MemoryStack stack = stackPush()) {
                    barrier(cmd, stack, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
                    VkBufferImageCopy.Buffer region = VkBufferImageCopy.calloc(levels, stack);
                    for (int l = 0; l < levels; l++) {
                        region.get(l).bufferOffset(offset[l]);
                        region.get(l).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(l).layerCount(1);
                        region.get(l).imageExtent().set(mw[l], mh[l], 1);
                    }
                    vkCmdCopyBufferToImage(cmd, staging.handle, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, region);
                    barrier(cmd, stack, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
                }
            });
        }
    }

    /** Number of mip levels, down to 1 x 1. */
    public final int levels;

    /** Halves an RGB image with a 2 x 2 box filter; odd sizes drop the last row or column. */
    private static float[] downsample(float[] src, int sw, int sh, int dw, int dh) {
        float[] dst = new float[dw * dh * 3];
        for (int y = 0; y < dh; y++) {
            int y0 = Math.min(2 * y, sh - 1), y1 = Math.min(2 * y + 1, sh - 1);
            for (int x = 0; x < dw; x++) {
                int x0 = Math.min(2 * x, sw - 1), x1 = Math.min(2 * x + 1, sw - 1);
                for (int c = 0; c < 3; c++) {
                    dst[(y * dw + x) * 3 + c] = 0.25f * (src[(y0 * sw + x0) * 3 + c] + src[(y0 * sw + x1) * 3 + c]
                            + src[(y1 * sw + x0) * 3 + c] + src[(y1 * sw + x1) * 3 + c]);
                }
            }
        }
        return dst;
    }

    /** Records a layout transition of all mip levels. */
    private void barrier(VkCommandBuffer cmd, MemoryStack stack, int oldL, int newL, int src, int dst) {
        VkImageMemoryBarrier.Buffer b = VkImageMemoryBarrier.calloc(1, stack);
        b.get(0).sType$Default().srcAccessMask(src).dstAccessMask(dst).oldLayout(oldL).newLayout(newL)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).image(image)
                .subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(levels).layerCount(1);
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
