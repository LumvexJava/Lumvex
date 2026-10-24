package dev.lumvex.vk;

import org.lwjgl.vulkan.VkInstance;

import java.util.List;

/** Presentation target of the renderer: creates the Vulkan surface and reports its size in physical pixels. */
public interface RenderSurface {
    /** Instance extensions required to create the surface. */
    List<String> instanceExtensions();

    long createSurface(VkInstance instance);

    /** Drawable size in pixels; 0 x 0 while minimized or hidden. */
    int[] framebufferSize();

    /** Returns true once after each resize. */
    boolean consumeResized();

    /** Blocks while the surface has no area, e.g. while minimized. */
    void waitWhileEmpty();

    /** Called after a frame was handed to the swapchain; the surface no longer needs to paint itself. */
    default void framePresented() { }

    /** Offscreen rendering without a swapchain, e.g. for thumbnails. */
    default boolean headless() { return false; }
}
