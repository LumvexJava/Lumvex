package dev.lumvex.kajiya;

//DEV

import dev.lumvex.kajiya.gfx.*;
import dev.lumvex.vk.RenderSurface;
import dev.lumvex.vk.VkContext;
import org.lwjgl.vulkan.VkInstance;

import java.util.List;

import static org.lwjgl.vulkan.VK13.*;

public final class KjSmoke {
    public static void main(String[] args) {
        RenderSurface headless = new RenderSurface() {
            public List<String> instanceExtensions() { return List.of(); }
            public long createSurface(VkInstance i) { return 0; }
            public int[] framebufferSize() { return new int[]{64, 64}; }
            public boolean consumeResized() { return false; }
            public void waitWhileEmpty() { }
            public boolean headless() { return true; }
        };
        VkContext ctx = new VkContext(headless, Boolean.getBoolean("kj.validation"));
        Gfx g = new Gfx(ctx);
        Bindless bl = new Bindless(g);
        Img lut = new Img(ctx, new Img.Desc(64, 64, VK_FORMAT_R16G16B16A16_SFLOAT));
        g.submit(() -> Pass.compute(g, "lut/brdf_fg.hlsl").bind(lut).dispatch(64, 64, 1));
        float[] px = g.readback(lut);
        for (int y : new int[]{0, 32, 63}) {
            int o = (y * 64 + 32) * 4;
            System.out.printf("brdf_fg[32,%d] = %.4f %.4f %.4f%n", y, px[o], px[o + 1], px[o + 2]);
        }
        bl.close();
        g.close();
        ctx.close();
    }
}
