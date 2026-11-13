// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.gfx;

//DEV

import dev.lumvex.vk.Buffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK13.*;

public final class Tex {

    public static Img rgba8(Gfx g, int w, int h, byte[] rgba, boolean srgb, boolean mips) {
        int levels = mips ? 32 - Integer.numberOfLeadingZeros(Math.max(w, h)) : 1;
        int format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        Img img = new Img(g.ctx, new Img.Desc(w, h, 1, levels, format, false, 0));
        try (Buffer staging = new Buffer(g.ctx, rgba.length, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.HOST)) {
            staging.map().put(rgba);
            g.ctx.submitImmediate(cmd -> {
                try (MemoryStack stack = stackPush()) {
                    VkBufferImageCopy.Buffer r = VkBufferImageCopy.calloc(1, stack);
                    r.get(0).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
                    r.get(0).imageExtent().set(w, h, 1);
                    vkCmdCopyBufferToImage(cmd, staging.handle, img.image, VK_IMAGE_LAYOUT_GENERAL, r);
                    for (int l = 1; l < levels; l++) {
                        memBarrier(cmd, stack);
                        VkImageBlit.Buffer blit = VkImageBlit.calloc(1, stack);
                        blit.get(0).srcSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(l - 1).layerCount(1);
                        blit.get(0).srcOffsets(1).set(Math.max(1, w >> (l - 1)), Math.max(1, h >> (l - 1)), 1);
                        blit.get(0).dstSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(l).layerCount(1);
                        blit.get(0).dstOffsets(1).set(Math.max(1, w >> l), Math.max(1, h >> l), 1);
                        vkCmdBlitImage(cmd, img.image, VK_IMAGE_LAYOUT_GENERAL, img.image, VK_IMAGE_LAYOUT_GENERAL, blit, VK_FILTER_LINEAR);
                    }
                    memBarrier(cmd, stack);
                }
            });
        }
        return img;
    }

    public static Img upload(Gfx g, int w, int h, int format, java.nio.ByteBuffer data) {
        Img img = new Img(g.ctx, new Img.Desc(w, h, format));
        try (Buffer staging = new Buffer(g.ctx, data.remaining(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.HOST)) {
            staging.map().put(data.duplicate());
            g.ctx.submitImmediate(cmd -> {
                try (MemoryStack stack = stackPush()) {
                    VkBufferImageCopy.Buffer r = VkBufferImageCopy.calloc(1, stack);
                    r.get(0).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).layerCount(1);
                    r.get(0).imageExtent().set(w, h, 1);
                    vkCmdCopyBufferToImage(cmd, staging.handle, img.image, VK_IMAGE_LAYOUT_GENERAL, r);
                    memBarrier(cmd, stack);
                }
            });
        }
        return img;
    }

    public static Img rgba8Chain(Gfx g, int w, int h, java.nio.ByteBuffer[] levels, boolean srgb) {
        int format = srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
        Img img = new Img(g.ctx, new Img.Desc(w, h, 1, levels.length, format, false, 0));
        long total = 0;
        for (java.nio.ByteBuffer l : levels) total += l.remaining();
        try (Buffer staging = new Buffer(g.ctx, total, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.HOST)) {
            java.nio.ByteBuffer m = staging.map();
            long[] offsets = new long[levels.length];
            long off = 0;
            for (int i = 0; i < levels.length; i++) {
                offsets[i] = off;
                m.put((int) off, levels[i].duplicate(), 0, levels[i].remaining());
                off += levels[i].remaining();
            }
            g.ctx.submitImmediate(cmd -> {
                try (MemoryStack stack = stackPush()) {
                    VkBufferImageCopy.Buffer r = VkBufferImageCopy.calloc(levels.length, stack);
                    for (int i = 0; i < levels.length; i++) {
                        r.get(i).bufferOffset(offsets[i]).imageSubresource().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).mipLevel(i).layerCount(1);
                        r.get(i).imageExtent().set(Math.max(1, w >> i), Math.max(1, h >> i), 1);
                    }
                    vkCmdCopyBufferToImage(cmd, staging.handle, img.image, VK_IMAGE_LAYOUT_GENERAL, r);
                    memBarrier(cmd, stack);
                }
            });
        }
        return img;
    }

    private static void memBarrier(VkCommandBuffer cmd, MemoryStack stack) {
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, mb, null, null);
    }

    public static Img solid(Gfx g, int r, int gr, int b, int a, boolean srgb) {
        return rgba8(g, 1, 1, new byte[]{(byte) r, (byte) gr, (byte) b, (byte) a}, srgb, false);
    }

    public static Img fromResource(Gfx g, String resource, boolean srgb, boolean mips) {
        try (InputStream in = Tex.class.getResourceAsStream(resource)) {
            if (in == null) throw new IllegalStateException("missing resource " + resource);
            return fromImage(g, ImageIO.read(in), srgb, mips);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    public static Img fromImage(Gfx g, BufferedImage bi, boolean srgb, boolean mips) {
        int w = bi.getWidth(), h = bi.getHeight();
        byte[] px = new byte[w * h * 4];
        int[] argb = bi.getRGB(0, 0, w, h, null, 0, w);
        for (int i = 0; i < argb.length; i++) {
            px[i * 4] = (byte) (argb[i] >> 16);
            px[i * 4 + 1] = (byte) (argb[i] >> 8);
            px[i * 4 + 2] = (byte) argb[i];
            px[i * 4 + 3] = (byte) (argb[i] >>> 24);
        }
        return rgba8(g, w, h, px, srgb, mips);
    }
}
