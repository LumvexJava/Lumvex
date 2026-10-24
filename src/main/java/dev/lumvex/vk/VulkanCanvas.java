package dev.lumvex.vk;

import dev.lumvex.vk.RenderSurface;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.jawt.*;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkWin32SurfaceCreateInfoKHR;

import java.awt.*;
import java.nio.LongBuffer;
import java.util.List;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.jawt.JAWTFunctions.*;
import static org.lwjgl.vulkan.KHRSurface.VK_KHR_SURFACE_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRWin32Surface.VK_KHR_WIN32_SURFACE_EXTENSION_NAME;
import static org.lwjgl.vulkan.KHRWin32Surface.vkCreateWin32SurfaceKHR;

/**
 * A heavyweight AWT canvas with a Vulkan surface on its native window (HWND, obtained through JAWT). The renderer
 * presents into it from its own thread; AWT never paints it.
 */
public final class VulkanCanvas extends Canvas implements RenderSurface {
    private volatile int lastW, lastH;
    private volatile boolean resized;
    /** Whether Vulkan has shown a frame; until then the canvas paints itself black instead of the system white. */
    private volatile boolean presented;

    public VulkanCanvas() {
        setBackground(Color.BLACK);
        setFocusable(true);
        addComponentListener(new java.awt.event.ComponentAdapter() {
            @Override public void componentResized(java.awt.event.ComponentEvent e) { resized = true; }
        });
    }

    // Vulkan owns the pixels once it presents; AWT must not clear or paint them then.
    @Override public void paint(Graphics g) {
        if (presented) return;
        g.setColor(Color.BLACK);
        g.fillRect(0, 0, getWidth(), getHeight());
    }

    @Override public void update(Graphics g) { paint(g); }

    @Override public void framePresented() { presented = true; }

    @Override
    public List<String> instanceExtensions() {
        return List.of(VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
    }

    /** The canvas's HWND, obtained through JAWT. The canvas must be displayable. */
    public long hwnd() {
        try (MemoryStack stack = stackPush()) {
            JAWT awt = JAWT.calloc(stack).version(JAWT_VERSION_1_4);
            if (!JAWT_GetAWT(awt)) throw new IllegalStateException("JAWT_GetAWT failed");
            JAWTDrawingSurface ds = JAWT_GetDrawingSurface(this, awt.GetDrawingSurface());
            if (ds == null) throw new IllegalStateException("JAWT: no drawing surface (canvas not displayable?)");
            try {
                int lock = JAWT_DrawingSurface_Lock(ds, ds.Lock());
                if ((lock & JAWT_LOCK_ERROR) != 0) throw new IllegalStateException("JAWT lock failed");
                try {
                    JAWTDrawingSurfaceInfo dsi = JAWT_DrawingSurface_GetDrawingSurfaceInfo(ds, ds.GetDrawingSurfaceInfo());
                    try {
                        return JAWTWin32DrawingSurfaceInfo.create(dsi.platformInfo()).hwnd();
                    } finally {
                        JAWT_DrawingSurface_FreeDrawingSurfaceInfo(dsi, ds.FreeDrawingSurfaceInfo());
                    }
                } finally {
                    JAWT_DrawingSurface_Unlock(ds, ds.Unlock());
                }
            } finally {
                JAWT_FreeDrawingSurface(ds, awt.FreeDrawingSurface());
            }
        }
    }

    @Override
    public long createSurface(VkInstance instance) {
        long hwnd = hwnd();
        try (MemoryStack stack = stackPush()) {
            VkWin32SurfaceCreateInfoKHR info = VkWin32SurfaceCreateInfoKHR.calloc(stack).sType$Default()
                    .hinstance(org.lwjgl.system.windows.WindowsLibrary.HINSTANCE).hwnd(hwnd);
            LongBuffer lp = stack.mallocLong(1);
            int r = vkCreateWin32SurfaceKHR(instance, info, null, lp);
            if (r != 0) throw new IllegalStateException("vkCreateWin32SurfaceKHR: " + r);
            return lp.get(0);
        }
    }

    /** Size in physical pixels, i.e. with the display scaling applied. */
    @Override
    public int[] framebufferSize() {
        double scale = 1.0;
        GraphicsConfiguration gc = getGraphicsConfiguration();
        if (gc != null) scale = gc.getDefaultTransform().getScaleX();
        int w = (int) Math.round(getWidth() * scale), h = (int) Math.round(getHeight() * scale);
        if (w != lastW || h != lastH) { lastW = w; lastH = h; resized = true; }
        return new int[]{w, h};
    }

    /** Physical pixels per logical pixel; converts mouse coordinates to image pixels. */
    public double pixelScale() {
        GraphicsConfiguration gc = getGraphicsConfiguration();
        return gc == null ? 1.0 : gc.getDefaultTransform().getScaleX();
    }

    @Override
    public boolean consumeResized() {
        framebufferSize();
        boolean r = resized;
        resized = false;
        return r;
    }

    @Override
    public void waitWhileEmpty() {
        try { Thread.sleep(50); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
    }
}
