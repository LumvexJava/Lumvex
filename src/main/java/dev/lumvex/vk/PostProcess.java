package dev.lumvex.vk;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_compute_shader;
import static org.lwjgl.vulkan.VK13.*;

/**
 * The compute passes that follow ray tracing:
 * <ol>
 *   <li>variance estimation from the history into pingA</li>
 *   <li>a-trous iterations ping-ponging between pingA and pingB with step widths 1, 2, 4, ...</li>
 *   <li>temporal filtering: the reprojected result of the previous frame, clamped to the local neighborhood</li>
 *   <li>the specular share apart: a-trous iterations between specPingA and specPingB whose reach is each pixel's
 *   reflection footprint (roughness, reflected-ray length), see {@code specular.comp}, then the temporal stage
 *   again, moved with the reflections</li>
 *   <li>composite: remodulation with albedo, tone mapping, lens flares and the game UI into the RGBA8 output</li>
 * </ol>
 * Filtering happens on demodulated illumination (radiance divided by albedo), so texture detail is never blurred.
 * The history holds diffuse light and the specular share in separate images.
 * The renderer double-buffers the history, so each descriptor set exists once per history parity.
 */
public final class PostProcess implements AutoCloseable {
    private static final int ATROUS_PC_SIZE = 32;
    private static final int COMPOSITE_PC_SIZE = 44;
    private static final int TEMPORAL_PC_SIZE = 16;
    private static final int SPECULAR_PC_SIZE = 20;

    /** One parity of the history; {@code color} holds the diffuse light, {@code specular} the specular share. */
    public record HistoryImages(StorageImage color, StorageImage guideNormalDepth, StorageImage guideAlbedo,
                                StorageImage meta, StorageImage motion, StorageImage specular, StorageImage specGuide) {}

    private final VkContext ctx;
    private final long atrousSetLayout, compositeSetLayout, temporalSetLayout, specularSetLayout;
    private final long atrousLayout, compositeLayout, temporalLayout, specularLayout;
    private final long atrousPipeline, compositePipeline, temporalPipeline, specularPipeline, specularTemporalPipeline;
    private final long[] specularTemporalSet = new long[2];
    private final long[] specularSetA = new long[2];  // reads specPingA, writes specPingB
    private final long[] specularSetB = new long[2];  // reads specPingB, writes specPingA
    private final long[] temporalSet = new long[2];
    private final long descriptorPool;
    private final long[] setA = new long[2];          // reads pingA, writes pingB
    private final long[] setB = new long[2];          // reads pingB, writes pingA
    private final long[] compositeSet = new long[2];

    public float sigmaLuminance = 4.0f;
    public float sigmaNormal = 128.0f;
    public float sigmaDepth = 0.02f;
    public float sigmaAlbedo = 0.1f;
    /** Filter illumination instead of radiance; the albedo is multiplied back in during compositing. */
    public boolean demodulate = true;
    public boolean temporal = true;
    public float temporalAlphaMin = 0.2f, temporalGamma = 1.0f, temporalTrustSamples = 16f;
    /** Filter the specular share by its reflection footprint; off composites the raw specular history. */
    public boolean specularFilter = true;
    /** Largest reflection footprint the specular filter blurs, in traced pixels. */
    public float specularMaxRadius = 48f;

    /** Images are bound later with {@link #updateDescriptors}. */
    public PostProcess(VkContext ctx, ShaderCompiler compiler) {
        this.ctx = ctx;
        VkDevice device = ctx.device;
        int img = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, buf = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        try (MemoryStack stack = stackPush()) {
            atrousSetLayout = createSetLayout(stack, new int[]{img, img, img, img, img, img});
            compositeSetLayout = createSetLayout(stack, new int[]{img, img, img, img, img, buf, img, img, buf, img, img, img, img, img});
            specularSetLayout = createSetLayout(stack, new int[]{img, img, img, img, img, img});
            specularLayout = createPipelineLayout(stack, specularSetLayout, SPECULAR_PC_SIZE);
            specularPipeline = createComputePipeline(stack, compiler, "specular.comp", specularLayout);
            temporalSetLayout = createSetLayout(stack, new int[]{img, img, img, img, img, img, img});
            temporalLayout = createPipelineLayout(stack, temporalSetLayout, TEMPORAL_PC_SIZE);
            temporalPipeline = createComputePipeline(stack, compiler, "temporal.comp", temporalLayout);
            specularTemporalPipeline = createComputePipeline(stack, compiler, "temporal.comp", temporalLayout, "LUMVEX_SPECULAR");
            atrousLayout = createPipelineLayout(stack, atrousSetLayout, ATROUS_PC_SIZE);
            compositeLayout = createPipelineLayout(stack, compositeSetLayout, COMPOSITE_PC_SIZE);
            atrousPipeline = createComputePipeline(stack, compiler, "atrous.comp", atrousLayout);
            compositePipeline = createComputePipeline(stack, compiler, "composite.comp", compositeLayout);

            VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(2, stack);
            sizes.get(0).type(img).descriptorCount(2 * (6 + 6 + 14 + 7 + 6 + 6 + 7));
            sizes.get(1).type(buf).descriptorCount(2 * 2);
            LongBuffer lp = stack.mallocLong(14);
            check(vkCreateDescriptorPool(device, VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(14).pPoolSizes(sizes), null, lp), "vkCreateDescriptorPool(post)");
            descriptorPool = lp.get(0);
            check(vkAllocateDescriptorSets(device, VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default()
                    .descriptorPool(descriptorPool)
                    .pSetLayouts(stack.longs(atrousSetLayout, atrousSetLayout, compositeSetLayout,
                            atrousSetLayout, atrousSetLayout, compositeSetLayout, temporalSetLayout, temporalSetLayout,
                            specularSetLayout, specularSetLayout, specularSetLayout, specularSetLayout,
                            temporalSetLayout, temporalSetLayout)), lp),
                    "vkAllocateDescriptorSets(post)");
            for (int p = 0; p < 2; p++) {
                setA[p] = lp.get(p * 3);
                setB[p] = lp.get(p * 3 + 1);
                compositeSet[p] = lp.get(p * 3 + 2);
                temporalSet[p] = lp.get(6 + p);
                specularSetA[p] = lp.get(8 + p * 2);
                specularSetB[p] = lp.get(9 + p * 2);
                specularTemporalSet[p] = lp.get(12 + p);
            }
        }
    }

    private long createSetLayout(MemoryStack stack, int[] types) {
        VkDescriptorSetLayoutBinding.Buffer b = VkDescriptorSetLayoutBinding.calloc(types.length, stack);
        for (int i = 0; i < types.length; i++) {
            b.get(i).binding(i).descriptorType(types[i]).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
        }
        LongBuffer lp = stack.mallocLong(1);
        check(vkCreateDescriptorSetLayout(ctx.device, VkDescriptorSetLayoutCreateInfo.calloc(stack)
                .sType$Default().pBindings(b), null, lp), "vkCreateDescriptorSetLayout(post)");
        return lp.get(0);
    }

    private long createPipelineLayout(MemoryStack stack, long setLayout, int pcSize) {
        LongBuffer lp = stack.mallocLong(1);
        check(vkCreatePipelineLayout(ctx.device, VkPipelineLayoutCreateInfo.calloc(stack)
                .sType$Default()
                .pSetLayouts(stack.longs(setLayout))
                .pPushConstantRanges(VkPushConstantRange.calloc(1, stack)
                        .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(pcSize)), null, lp),
                "vkCreatePipelineLayout(post)");
        return lp.get(0);
    }

    private long createComputePipeline(MemoryStack stack, ShaderCompiler compiler, String shader, long layout,
                                       String... defines) {
        long module = compiler.createModule(ctx, shader, shaderc_glsl_compute_shader, defines);
        VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
        info.get(0).sType$Default().layout(layout);
        info.get(0).stage().sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
        LongBuffer lp = stack.mallocLong(1);
        check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, info, null, lp), "vkCreateComputePipelines(" + shader + ")");
        vkDestroyShaderModule(ctx.device, module, null);
        return lp.get(0);
    }

    /** Binds the images; call after every resize. {@code history[p]} is the set written at parity {@code p}. */
    public void updateDescriptors(HistoryImages[] history, StorageImage pingA, StorageImage pingB,
                                  StorageImage output, Buffer flares, StorageImage[] filtered, Buffer ui,
                                  StorageImage specPingA, StorageImage specPingB, StorageImage[] specFiltered) {
        try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(2 * (6 + 6 + 14 + 7 + 6 + 6 + 7), stack);
            int i = 0;
            for (int p = 0; p < 2; p++) {
                HistoryImages h = history[p];
                StorageImage[] a = {h.color, h.guideNormalDepth, h.guideAlbedo, pingA, pingB, h.meta};
                StorageImage[] b = {h.color, h.guideNormalDepth, h.guideAlbedo, pingB, pingA, h.meta};
                for (int k = 0; k < 6; k++) image(stack, w.get(i++), setA[p], k, a[k]);
                for (int k = 0; k < 6; k++) image(stack, w.get(i++), setB[p], k, b[k]);
                image(stack, w.get(i++), compositeSet[p], 0, h.color);
                image(stack, w.get(i++), compositeSet[p], 1, pingA);
                image(stack, w.get(i++), compositeSet[p], 2, pingB);
                image(stack, w.get(i++), compositeSet[p], 3, output);
                image(stack, w.get(i++), compositeSet[p], 4, h.meta);
                buffer(stack, w.get(i++), compositeSet[p], 5, flares);
                image(stack, w.get(i++), compositeSet[p], 6, h.guideAlbedo);
                image(stack, w.get(i++), compositeSet[p], 7, filtered[p]);
                buffer(stack, w.get(i++), compositeSet[p], 8, ui);
                image(stack, w.get(i++), compositeSet[p], 9, h.specular);
                image(stack, w.get(i++), compositeSet[p], 10, specPingA);
                image(stack, w.get(i++), compositeSet[p], 11, specPingB);
                image(stack, w.get(i++), compositeSet[p], 12, h.specGuide);
                image(stack, w.get(i++), compositeSet[p], 13, specFiltered[p]);
                StorageImage[] st = {specPingA, specPingB, h.specGuide, h.meta, specFiltered[1 - p], specFiltered[p],
                        history[1 - p].meta};
                for (int k = 0; k < 7; k++) image(stack, w.get(i++), specularTemporalSet[p], k, st[k]);
                StorageImage[] sa = {h.specular, h.specGuide, h.guideNormalDepth, h.meta, specPingA, specPingB};
                StorageImage[] sb = {h.specular, h.specGuide, h.guideNormalDepth, h.meta, specPingB, specPingA};
                for (int k = 0; k < 6; k++) image(stack, w.get(i++), specularSetA[p], k, sa[k]);
                for (int k = 0; k < 6; k++) image(stack, w.get(i++), specularSetB[p], k, sb[k]);
                StorageImage[] t = {pingA, pingB, h.motion, h.meta, filtered[1 - p], filtered[p], history[1 - p].meta};
                for (int k = 0; k < 7; k++) image(stack, w.get(i++), temporalSet[p], k, t[k]);
            }
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    private static void image(MemoryStack stack, VkWriteDescriptorSet w, long set, int binding, StorageImage img) {
        w.sType$Default().dstSet(set).dstBinding(binding).descriptorCount(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                .pImageInfo(VkDescriptorImageInfo.calloc(1, stack).imageView(img.view).imageLayout(VK_IMAGE_LAYOUT_GENERAL));
    }

    private static void buffer(MemoryStack stack, VkWriteDescriptorSet w, long set, int binding, Buffer buf) {
        w.sType$Default().dstSet(set).dstBinding(binding).descriptorCount(1)
                .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(buf.handle).range(VK_WHOLE_SIZE));
    }

    /** Records variance estimation and the a-trous passes. Returns the image to composite: 1 = pingA, 2 = pingB. */
    public int recordDenoise(VkCommandBuffer cmd, MemoryStack stack, int parity, int width, int height,
                             int historyLimit, int iterations) {
        ByteBuffer pc = stack.malloc(ATROUS_PC_SIZE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atrousPipeline);

        // Variance estimation writes pingA, which is the destination of setB.
        dispatchAtrous(cmd, stack, pc, setB[parity], 0, 1, historyLimit, width, height);
        for (int i = 0; i < iterations; i++) {
            computeBarrier(cmd, stack);
            dispatchAtrous(cmd, stack, pc, (i & 1) == 0 ? setA[parity] : setB[parity], 1, 1 << i, historyLimit, width, height);
        }
        return (iterations & 1) == 0 ? 1 : 2;
    }

    // mode 0 estimates variance, mode 1 filters.
    private void dispatchAtrous(VkCommandBuffer cmd, MemoryStack stack, ByteBuffer pc, long set, int mode, int step,
                                int historyLimit, int width, int height) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, atrousLayout, 0, stack.longs(set), null);
        pc.putInt(0, mode).putInt(4, step).putInt(8, historyLimit)
                .putFloat(12, sigmaLuminance).putFloat(16, sigmaNormal).putFloat(20, sigmaDepth).putFloat(24, sigmaAlbedo)
                .putInt(28, demodulate ? 1 : 0);
        vkCmdPushConstants(cmd, atrousLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    }

    /** {@code source}: 0 = raw history, 1 = pingA, 2 = pingB, 3 = temporal result. */
    public void recordComposite(VkCommandBuffer cmd, MemoryStack stack, int parity, int width, int height,
                                float scaleX, float scaleY, int marginX, int marginY, float exposure,
                                int debugMode, int source, int specularSource, int historyLimit, float flareStrength) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compositePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compositeLayout, 0, stack.longs(compositeSet[parity]), null);
        ByteBuffer pc = stack.malloc(COMPOSITE_PC_SIZE);
        pc.putFloat(0, exposure).putInt(4, debugMode).putInt(8, source).putInt(12, historyLimit).putFloat(16, flareStrength)
                .putInt(20, demodulate && source != 0 ? 1 : 0).putInt(24, marginX).putInt(28, marginY)
                .putFloat(32, scaleX).putFloat(36, scaleY).putInt(40, specularSource);
        vkCmdPushConstants(cmd, compositeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    }

    /** Rebinds a reallocated UI buffer. */
    public void updateUiBuffer(Buffer ui) {
        try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(2, stack);
            for (int p = 0; p < 2; p++) buffer(stack, w.get(p), compositeSet[p], 8, ui);
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    /** Rebinds a reallocated lens flare buffer. */
    public void updateFlareBuffer(Buffer flares) {
        try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(2, stack);
            for (int p = 0; p < 2; p++) buffer(stack, w.get(p), compositeSet[p], 5, flares);
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    /** Records the temporal pass on the spatial filter's output and returns the composite source (3). */
    public int recordTemporal(VkCommandBuffer cmd, MemoryStack stack, int parity, int width, int height, int source) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporalLayout, 0, stack.longs(temporalSet[parity]), null);
        ByteBuffer pc = stack.malloc(TEMPORAL_PC_SIZE);
        pc.putInt(0, source).putFloat(4, temporalAlphaMin).putFloat(8, temporalGamma).putFloat(12, temporalTrustSamples);
        vkCmdPushConstants(cmd, temporalLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        return 3;
    }

    /**
     * Records the specular filter: preparation into specPingB, then {@code iterations} a-trous steps of widths 1, 2,
     * 4, ... Returns the composite's specular source: 1 = specPingA, 2 = specPingB.
     *
     * @param pixelAngle view angle of one traced pixel in radians
     * @param moved whether the camera moved this frame, which keeps the specular history short
     */
    public int recordSpecular(VkCommandBuffer cmd, MemoryStack stack, int parity, int width, int height,
                              int iterations, float pixelAngle, boolean moved) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, specularPipeline);
        ByteBuffer pc = stack.malloc(SPECULAR_PC_SIZE);
        pc.putInt(16, moved ? 1 : 0);
        dispatchSpecular(cmd, stack, pc, specularSetA[parity], 0, 1, pixelAngle, width, height);   // writes specPingB
        for (int i = 0; i < iterations; i++) {
            computeBarrier(cmd, stack);
            dispatchSpecular(cmd, stack, pc, (i & 1) == 0 ? specularSetB[parity] : specularSetA[parity], 1, 1 << i,
                    pixelAngle, width, height);
        }
        return (iterations & 1) == 0 ? 2 : 1;
    }

    private void dispatchSpecular(VkCommandBuffer cmd, MemoryStack stack, ByteBuffer pc, long set, int mode, int step,
                                  float pixelAngle, int width, int height) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, specularLayout, 0, stack.longs(set), null);
        pc.putInt(0, mode).putInt(4, step).putFloat(8, pixelAngle).putFloat(12, specularMaxRadius);
        vkCmdPushConstants(cmd, specularLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
    }

    /**
     * Records the temporal stage of the specular share on the specular filter's output ({@code source}: 1 =
     * specPingA, 2 = specPingB), moved with the reflections. Returns the composite's specular source (3).
     */
    public int recordSpecularTemporal(VkCommandBuffer cmd, MemoryStack stack, int parity, int width, int height, int source) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, specularTemporalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporalLayout, 0, stack.longs(specularTemporalSet[parity]), null);
        ByteBuffer pc = stack.malloc(TEMPORAL_PC_SIZE);
        pc.putInt(0, source).putFloat(4, temporalAlphaMin).putFloat(8, temporalGamma).putFloat(12, temporalTrustSamples);
        vkCmdPushConstants(cmd, temporalLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
        vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
        return 3;
    }

    /** Compute-to-compute memory barrier. */
    public static void computeBarrier(VkCommandBuffer cmd, MemoryStack stack) {
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, mb, null, null);
    }

    @Override
    public void close() {
        vkDestroyDescriptorPool(ctx.device, descriptorPool, null);
        vkDestroyPipeline(ctx.device, atrousPipeline, null);
        vkDestroyPipeline(ctx.device, compositePipeline, null);
        vkDestroyPipeline(ctx.device, temporalPipeline, null);
        vkDestroyPipeline(ctx.device, specularPipeline, null);
        vkDestroyPipeline(ctx.device, specularTemporalPipeline, null);
        vkDestroyPipelineLayout(ctx.device, specularLayout, null);
        vkDestroyDescriptorSetLayout(ctx.device, specularSetLayout, null);
        vkDestroyPipelineLayout(ctx.device, temporalLayout, null);
        vkDestroyDescriptorSetLayout(ctx.device, temporalSetLayout, null);
        vkDestroyPipelineLayout(ctx.device, atrousLayout, null);
        vkDestroyPipelineLayout(ctx.device, compositeLayout, null);
        vkDestroyDescriptorSetLayout(ctx.device, atrousSetLayout, null);
        vkDestroyDescriptorSetLayout(ctx.device, compositeSetLayout, null);
    }
}
