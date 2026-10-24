package dev.lumvex.vk;

import static org.lwjgl.vulkan.VK10.VK_SUCCESS;

/** Vulkan result checks and alignment helpers. */
public final class VkUtil {
    private VkUtil() {}

    /** Throws if {@code result} is not {@code VK_SUCCESS}. */
    public static void check(int result, String what) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(what + " failed with VkResult " + result);
        }
    }

    /** Rounds up to a multiple of {@code alignment}, which must be a power of two. */
    public static long align(long value, long alignment) {
        return (value + alignment - 1) & -alignment;
    }

    public static int align(int value, int alignment) {
        return (value + alignment - 1) & -alignment;
    }
}
