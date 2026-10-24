package dev.lumvex.vk;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.IntBuffer;
import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.KHRSurface.*;
import static org.lwjgl.vulkan.KHRSwapchain.*;
import static org.lwjgl.vulkan.VK13.*;

/** Swapchain whose images are filled by blitting the renderer's output image. */
public final class Swapchain implements AutoCloseable {
    private final VkContext ctx;
    public final long handle;
    public final int format;
    public final int width;
    public final int height;
    public final long[] images;
    /** One semaphore per image, so a semaphore is never reused while its presentation is pending. */
    public final long[] renderFinished;

    /** Uses FIFO with vsync, otherwise mailbox or immediate, and one image more than the minimum. */
    public Swapchain(VkContext ctx, int fbWidth, int fbHeight, boolean vsync, long oldSwapchain) {
        this.ctx = ctx;
        VkPhysicalDevice pd = ctx.physicalDevice;
        try (MemoryStack stack = stackPush()) {
            VkSurfaceCapabilitiesKHR caps = VkSurfaceCapabilitiesKHR.malloc(stack);
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(pd, ctx.surface, caps);
            if ((caps.supportedUsageFlags() & VK_IMAGE_USAGE_TRANSFER_DST_BIT) == 0) {
                throw new IllegalStateException("Swapchain images do not support TRANSFER_DST");
            }

            IntBuffer ip = stack.mallocInt(1);
            vkGetPhysicalDeviceSurfaceFormatsKHR(pd, ctx.surface, ip, null);
            VkSurfaceFormatKHR.Buffer formats = VkSurfaceFormatKHR.malloc(ip.get(0), stack);
            vkGetPhysicalDeviceSurfaceFormatsKHR(pd, ctx.surface, ip, formats);
            VkSurfaceFormatKHR chosen = formats.get(0);
            for (VkSurfaceFormatKHR f : formats) {
                // UNORM, because the output image already holds gamma-encoded values.
                if ((f.format() == VK_FORMAT_B8G8R8A8_UNORM || f.format() == VK_FORMAT_R8G8B8A8_UNORM)
                        && f.colorSpace() == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    chosen = f;
                    break;
                }
            }
            format = chosen.format();

            vkGetPhysicalDeviceSurfacePresentModesKHR(pd, ctx.surface, ip, null);
            IntBuffer modes = stack.mallocInt(ip.get(0));
            vkGetPhysicalDeviceSurfacePresentModesKHR(pd, ctx.surface, ip, modes);
            int presentMode = VK_PRESENT_MODE_FIFO_KHR;
            if (!vsync) {
                for (int i = 0; i < modes.capacity(); i++) {
                    if (modes.get(i) == VK_PRESENT_MODE_MAILBOX_KHR) { presentMode = VK_PRESENT_MODE_MAILBOX_KHR; break; }
                    if (modes.get(i) == VK_PRESENT_MODE_IMMEDIATE_KHR) presentMode = VK_PRESENT_MODE_IMMEDIATE_KHR;
                }
            }

            int w, h;
            if (caps.currentExtent().width() != 0xFFFFFFFF) {
                w = caps.currentExtent().width();
                h = caps.currentExtent().height();
            } else {
                w = Math.clamp(fbWidth, caps.minImageExtent().width(), caps.maxImageExtent().width());
                h = Math.clamp(fbHeight, caps.minImageExtent().height(), caps.maxImageExtent().height());
            }
            width = w;
            height = h;

            int imageCount = caps.minImageCount() + 1;
            if (caps.maxImageCount() > 0) imageCount = Math.min(imageCount, caps.maxImageCount());

            VkSwapchainCreateInfoKHR info = VkSwapchainCreateInfoKHR.calloc(stack)
                    .sType$Default()
                    .surface(ctx.surface)
                    .minImageCount(imageCount)
                    .imageFormat(format)
                    .imageColorSpace(chosen.colorSpace())
                    .imageArrayLayers(1)
                    .imageUsage(VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
                    .imageSharingMode(VK_SHARING_MODE_EXCLUSIVE)
                    .preTransform(caps.currentTransform())
                    .compositeAlpha(VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                    .presentMode(presentMode)
                    .clipped(true)
                    .oldSwapchain(oldSwapchain);
            info.imageExtent().set(width, height);

            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateSwapchainKHR(ctx.device, info, null, lp), "vkCreateSwapchainKHR");
            handle = lp.get(0);

            vkGetSwapchainImagesKHR(ctx.device, handle, ip, null);
            LongBuffer imgs = stack.mallocLong(ip.get(0));
            vkGetSwapchainImagesKHR(ctx.device, handle, ip, imgs);
            images = new long[imgs.capacity()];
            renderFinished = new long[imgs.capacity()];
            VkSemaphoreCreateInfo semInfo = VkSemaphoreCreateInfo.calloc(stack).sType$Default();
            for (int i = 0; i < images.length; i++) {
                images[i] = imgs.get(i);
                check(vkCreateSemaphore(ctx.device, semInfo, null, lp), "vkCreateSemaphore");
                renderFinished[i] = lp.get(0);
            }
        }
    }

    @Override
    public void close() {
        for (long s : renderFinished) vkDestroySemaphore(ctx.device, s, null);
        vkDestroySwapchainKHR(ctx.device, handle, null);
    }
}
