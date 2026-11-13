// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import dev.lumvex.vk.VkContext;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.LongBuffer;
import java.util.HashMap;
import java.util.Map;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK13.*;

public final class Img implements AutoCloseable {

    public enum View { DEFAULT, ARRAY, MIP0_STORAGE }

    public record Desc(int width, int height, int layers, int mips, int format, boolean cube, int extraUsage) {
        public Desc(int width, int height, int format) { this(width, height, 1, 1, format, false, 0); }
        public Desc withFormat(int f) { return new Desc(width, height, layers, mips, f, cube, extraUsage); }
        public Desc withSize(int w, int h) { return new Desc(w, h, layers, mips, format, cube, extraUsage); }
        public Desc withUsage(int u) { return new Desc(width, height, layers, mips, format, cube, extraUsage | u); }
        public Desc withMips(int m) { return new Desc(width, height, layers, m, format, cube, extraUsage); }
        public static Desc cube(int size, int format) { return new Desc(size, size, 6, 1, format, true, 0); }
        public float[] extentInvExtent() { return new float[]{width, height, 1f / width, 1f / height}; }
    }

    private final VkContext ctx;
    public final Desc desc;
    public final long image;
    public final long memory;
    public final boolean depth;
    public final int aspect;
    private final Map<Long, Long> views = new HashMap<>();
    public String name = "";

    public Img(VkContext ctx, Desc d) {
        this.ctx = ctx;
        this.desc = d;
        depth = d.format == VK_FORMAT_D32_SFLOAT || d.format == VK_FORMAT_D16_UNORM;
        aspect = depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
        try (MemoryStack stack = stackPush()) {
            VkFormatProperties fp = VkFormatProperties.malloc(stack);
            vkGetPhysicalDeviceFormatProperties(ctx.physicalDevice, d.format, fp);
            int feat = fp.optimalTilingFeatures();
            int usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | d.extraUsage;
            if (depth) usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
            else {
                if ((feat & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
                if ((feat & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
            }
            VkImageCreateInfo info = VkImageCreateInfo.calloc(stack).sType$Default()
                    .flags(d.cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0)
                    .imageType(VK_IMAGE_TYPE_2D).format(d.format).mipLevels(d.mips).arrayLayers(d.layers)
                    .samples(VK_SAMPLE_COUNT_1_BIT).tiling(VK_IMAGE_TILING_OPTIMAL).usage(usage)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE).initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
            info.extent().set(d.width, d.height, 1);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateImage(ctx.device, info, null, lp), "vkCreateImage");
            image = lp.get(0);
            VkMemoryRequirements req = VkMemoryRequirements.malloc(stack);
            vkGetImageMemoryRequirements(ctx.device, image, req);
            VkMemoryAllocateInfo alloc = VkMemoryAllocateInfo.calloc(stack).sType$Default().allocationSize(req.size())
                    .memoryTypeIndex(ctx.findMemoryType(req.memoryTypeBits(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT));
            check(vkAllocateMemory(ctx.device, alloc, null, lp), "vkAllocateMemory(image)");
            memory = lp.get(0);
            check(vkBindImageMemory(ctx.device, image, memory, 0), "vkBindImageMemory");
        }
        ctx.submitImmediate(cmd -> {
            try (MemoryStack stack = stackPush()) {
                VkImageMemoryBarrier.Buffer b = VkImageMemoryBarrier.calloc(1, stack);
                b.get(0).sType$Default().srcAccessMask(0).dstAccessMask(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT)
                        .oldLayout(VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK_IMAGE_LAYOUT_GENERAL)
                        .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).image(image)
                        .subresourceRange(r -> r.aspectMask(aspect).levelCount(d.mips).layerCount(d.layers));
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, null, null, b);
                if (!depth && !d.cube) {
                    VkClearColorValue zero = VkClearColorValue.calloc(stack);
                    VkImageSubresourceRange.Buffer range = VkImageSubresourceRange.calloc(1, stack)
                            .aspectMask(aspect).levelCount(d.mips).layerCount(d.layers);
                    vkCmdClearColorImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, zero, range);
                }
            }
        });
    }

    public int width() { return desc.width; }

    public int height() { return desc.height; }

    public long view(View kind) { return view(kind, 0, desc.mips); }

    public long view(View kind, int baseMip, int mipCount) {
        long key = ((long) kind.ordinal() << 40) | ((long) baseMip << 20) | mipCount;
        Long v = views.get(key);
        if (v != null) return v;
        try (MemoryStack stack = stackPush()) {
            int type;
            if (kind == View.ARRAY) type = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            else if (desc.cube) type = VK_IMAGE_VIEW_TYPE_CUBE;
            else type = desc.layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
            VkImageViewCreateInfo vi = VkImageViewCreateInfo.calloc(stack).sType$Default().image(image).viewType(type)
                    .format(desc.format);
            vi.subresourceRange().aspectMask(aspect).baseMipLevel(baseMip).levelCount(mipCount).baseArrayLayer(0).layerCount(desc.layers);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateImageView(ctx.device, vi, null, lp), "vkCreateImageView");
            views.put(key, lp.get(0));
            return lp.get(0);
        }
    }

    @Override
    public void close() {
        for (long v : views.values()) vkDestroyImageView(ctx.device, v, null);
        views.clear();
        vkDestroyImage(ctx.device, image, null);
        vkFreeMemory(ctx.device, memory, null);
    }
}
