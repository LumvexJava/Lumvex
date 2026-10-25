package dev.lumvex.vk;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.function.BiConsumer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.memAlloc;
import static org.lwjgl.system.MemoryUtil.memFree;
import static org.lwjgl.util.shaderc.Shaderc.*;
import static org.lwjgl.vulkan.VK13.*;

/** Compiles the GLSL shaders under {@code /shaders} on the class path to SPIR-V with shaderc. */
public final class ShaderCompiler implements AutoCloseable {
    private static final Pattern INCLUDE = Pattern.compile("^\\s*#include\\s+\"([^\"]+)\"\\s*$", Pattern.MULTILINE);

    private final long compiler;
    private final long options;
    private BiConsumer<String, Integer> progressListener = (shader, percent) -> { };
    private int expectedShaderCount;
    private int compiledShaderCount;

    /** Targets Vulkan 1.3 / SPIR-V 1.6 with performance optimizations. */
    public ShaderCompiler() {
        compiler = shaderc_compiler_initialize();
        options = shaderc_compile_options_initialize();
        shaderc_compile_options_set_target_env(options, shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_3);
        shaderc_compile_options_set_target_spirv(options, shaderc_spirv_version_1_6);
        shaderc_compile_options_set_optimization_level(options, shaderc_optimization_level_performance);
    }

    /** Reports successful module creation as a percentage of the expected startup shaders. */
    public void setProgressListener(int expectedShaderCount, BiConsumer<String, Integer> listener) {
        if (expectedShaderCount <= 0) throw new IllegalArgumentException("expectedShaderCount must be positive");
        if (compiledShaderCount != 0) throw new IllegalStateException("shader compilation has already started");
        this.expectedShaderCount = expectedShaderCount;
        this.progressListener = Objects.requireNonNull(listener, "listener");
    }

    /** Adds a macro definition to all subsequent compilations. */
    public void define(String name, String value) {
        shaderc_compile_options_add_macro_definition(options, name, value);
    }

    /** Loads a shader source and expands its {@code #include "file"} directives recursively. */
    public static String loadSource(String name) {
        try (InputStream in = ShaderCompiler.class.getResourceAsStream("/shaders/" + name)) {
            if (in == null) throw new IllegalArgumentException("Shader not found: " + name);
            String src = new String(in.readAllBytes(), StandardCharsets.UTF_8);
            Matcher m = INCLUDE.matcher(src);
            StringBuilder sb = new StringBuilder();
            while (m.find()) m.appendReplacement(sb, Matcher.quoteReplacement(loadSource(m.group(1))));
            m.appendTail(sb);
            return sb.toString();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Compiles a shader and creates a shader module from it.
     *
     * @param kind         shaderc shader kind, e.g. {@code shaderc_raygen_shader}
     * @param localDefines macros for this module only, inserted after {@code #version}
     * @throws IllegalStateException with the compiler log if compilation fails
     */
    public long createModule(VkContext ctx, String name, int kind, String... localDefines) {
        String text = loadSource(name);
        if (localDefines.length > 0) {
            int eol = text.indexOf('\n') + 1;
            StringBuilder defs = new StringBuilder();
            for (String d : localDefines) defs.append("#define ").append(d).append(" 1\n");
            text = text.substring(0, eol) + defs + text.substring(eol);
        }
        // The expanded source can exceed LWJGL's 64 KB MemoryStack, so it is passed from the heap.
        byte[] bytes = text.getBytes(java.nio.charset.StandardCharsets.UTF_8);
        ByteBuffer source = memAlloc(bytes.length);
        source.put(bytes).flip();
        long result;
        try (MemoryStack stack = stackPush()) {
            result = shaderc_compile_into_spv(compiler, source, kind, stack.UTF8(name), stack.UTF8("main"), options);
        } finally {
            memFree(source);
        }
        try {
            if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) {
                throw new IllegalStateException("Shader compilation failed (" + name + "):\n" + shaderc_result_get_error_message(result));
            }
            ByteBuffer spirv = shaderc_result_get_bytes(result);
            ByteBuffer copy = memAlloc(spirv.remaining());
            copy.put(spirv).flip();
            try (MemoryStack stack = stackPush()) {
                VkShaderModuleCreateInfo info = VkShaderModuleCreateInfo.calloc(stack).sType$Default().pCode(copy);
                LongBuffer lp = stack.mallocLong(1);
                check(vkCreateShaderModule(ctx.device, info, null, lp), "vkCreateShaderModule(" + name + ")");
                long module = lp.get(0);
                if (expectedShaderCount > 0) {
                    int percent = Math.min(100, ++compiledShaderCount * 100 / expectedShaderCount);
                    progressListener.accept(name, percent);
                }
                return module;
            } finally {
                memFree(copy);
            }
        } finally {
            shaderc_result_release(result);
        }
    }

    @Override
    public void close() {
        shaderc_compile_options_release(options);
        shaderc_compiler_release(compiler);
    }
}
