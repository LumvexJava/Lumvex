// Java port of parts of kajiya (Rust): https://github.com/EmbarkStudios/kajiya
// Original work Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md.
// Port and modifications: Copyright (c) 2026 LumvexJava (MIT).

package dev.lumvex.kajiya.render;

//DEV

import dev.lumvex.kajiya.gfx.Img;

import static org.lwjgl.vulkan.VK13.*;

public final class Fmt {
    public static final int RGBA16F = VK_FORMAT_R16G16B16A16_SFLOAT, RGBA32F = VK_FORMAT_R32G32B32A32_SFLOAT, D32 = VK_FORMAT_D32_SFLOAT,
            A2R10G10B10 = VK_FORMAT_A2R10G10B10_UNORM_PACK32, R16G16B16A16_SNORM = VK_FORMAT_R16G16B16A16_SNORM, R32F = VK_FORMAT_R32_SFLOAT,
            R16F = VK_FORMAT_R16_SFLOAT, R8 = VK_FORMAT_R8_UNORM, RGBA8 = VK_FORMAT_R8G8B8A8_UNORM, RGBA8_SNORM = VK_FORMAT_R8G8B8A8_SNORM,
            R8_SNORM = VK_FORMAT_R8_SNORM, B10G11R11 = VK_FORMAT_B10G11R11_UFLOAT_PACK32, RG32UI = VK_FORMAT_R32G32_UINT,
            RGBA32UI = VK_FORMAT_R32G32B32A32_UINT, RG16F = VK_FORMAT_R16G16_SFLOAT, R32UI = VK_FORMAT_R32_UINT;

    public static int half(int v) { return (v + 1) / 2; }

    public static Img.Desc half(Img.Desc d) { return d.withSize(half(d.width()), half(d.height())); }

    public static Img.Desc divUp(Img.Desc d, int x, int y) { return d.withSize(Math.max(1, (d.width() + x - 1) / x), Math.max(1, (d.height() + y - 1) / y)); }

    public static int mipCount(int w, int h) { return 32 - Integer.numberOfLeadingZeros(Math.max(w, h)); }

    private Fmt() { }
}
