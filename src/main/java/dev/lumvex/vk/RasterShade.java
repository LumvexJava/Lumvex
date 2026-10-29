package dev.lumvex.vk;

import dev.lumvex.texture.VirtualTextures;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_compute_shader;
import static org.lwjgl.vulkan.VK13.*;

/**
 * The raster render mode: shades the raster pre-pass's visibility buffer with a compute shader
 * ({@code raster_shade.comp}), without ray tracing. Used when a scene asks for {@code "raster"} and on GPUs without
 * hardware ray tracing, where it is the only mode. Lighting: direct sunlight without shadows, sky light from the
 * environment's mip chain, a blurred sky reflection, emission; no sphere lights and no indirect light.
 *
 * <p>It writes the images ray generation writes (color, guides, meta, specular share) of history parity {@code p},
 * so the composite pass, the haze and picking need no changes. Buffer addresses are read as two 32-bit halves, so
 * GPUs without 64-bit integers in shaders work.
 */
public final class RasterShade implements AutoCloseable {
    private final VkContext ctx;
    private final long setLayout, layout, pipeline, pool;
    private final long[] sets = new long[2];
    private final int pushSize;

    /**
     * Creates the compute pipeline of {@code raster_shade.comp} and one descriptor set per parity, with the bindings
     * given as {binding, type, count}.
     */
    public RasterShade(VkContext ctx, ShaderCompiler compiler, int pushConstantSize) {
        this.ctx = ctx;
        this.pushSize = pushConstantSize;
        int img = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, buf = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, smp = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        int[][] bindings = {{0, img, 1}, {1, img, 1}, {2, img, 1}, {3, buf, 1}, {4, buf, 1}, {5, img, 1}, {6, img, 1},
                {7, buf, 1}, {8, img, 1}, {16, smp, 6}, {17, buf, 1}, {18, buf, 1}, {19, buf, 1}, {21, smp, 1}};
        try (MemoryStack stack = stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer b = VkDescriptorSetLayoutBinding.calloc(bindings.length, stack);
            for (int i = 0; i < bindings.length; i++) {
                b.get(i).binding(bindings[i][0]).descriptorType(bindings[i][1]).descriptorCount(bindings[i][2])
                        .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            }
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateDescriptorSetLayout(ctx.device, VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default()
                    .pBindings(b), null, lp), "vkCreateDescriptorSetLayout(raster shade)");
            setLayout = lp.get(0);
            check(vkCreatePipelineLayout(ctx.device, VkPipelineLayoutCreateInfo.calloc(stack).sType$Default()
                    .pSetLayouts(stack.longs(setLayout))
                    .pPushConstantRanges(VkPushConstantRange.calloc(1, stack).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT)
                            .offset(0).size(pushConstantSize)), null, lp), "vkCreatePipelineLayout(raster shade)");
            layout = lp.get(0);
            long module = compiler.createModule(ctx, "raster_shade.comp", shaderc_glsl_compute_shader);
            VkComputePipelineCreateInfo.Buffer ci = VkComputePipelineCreateInfo.calloc(1, stack);
            ci.get(0).sType$Default().layout(layout);
            ci.get(0).stage().sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
            check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, ci, null, lp), "vkCreateComputePipelines(raster shade)");
            pipeline = lp.get(0);
            vkDestroyShaderModule(ctx.device, module, null);

            VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(3, stack);
            sizes.get(0).type(img).descriptorCount(2 * 6);
            sizes.get(1).type(buf).descriptorCount(2 * 6);
            sizes.get(2).type(smp).descriptorCount(2 * 7);
            check(vkCreateDescriptorPool(ctx.device, VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(2)
                    .pPoolSizes(sizes), null, lp), "vkCreateDescriptorPool(raster shade)");
            pool = lp.get(0);
            LongBuffer s = stack.mallocLong(2);
            check(vkAllocateDescriptorSets(ctx.device, VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(pool).pSetLayouts(stack.longs(setLayout, setLayout)), s), "vkAllocateDescriptorSets(raster shade)");
            sets[0] = s.get(0);
            sets[1] = s.get(1);
        }
    }

    /** Binds the images: set {@code p} writes history {@code p}; call after every resize. */
    public void writeImages(PostProcess.HistoryImages[] history, StorageImage visibility) {
        try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(12, stack);
            int i = 0;
            for (int p = 0; p < 2; p++) {
                PostProcess.HistoryImages h = history[p];
                StorageImage[] images = {visibility, h.color(), h.guideNormalDepth(), h.meta(), h.specular(), h.guideAlbedo()};
                int[] binding = {0, 1, 2, 5, 6, 8};
                for (int k = 0; k < images.length; k++) {
                    w.get(i++).sType$Default().dstSet(sets[p]).dstBinding(binding[k]).descriptorCount(1)
                            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                            .pImageInfo(VkDescriptorImageInfo.calloc(1, stack).imageView(images[k].view).imageLayout(VK_IMAGE_LAYOUT_GENERAL));
                }
            }
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    /** Binds the scene: instances, materials, transforms, virtual textures and the environment. */
    public void writeScene(Buffer instances, Buffer materials, Buffer transforms, VirtualTextures vt, EnvironmentTexture env) {
        try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(16, stack);
            VkDescriptorImageInfo.Buffer atlas = VkDescriptorImageInfo.calloc(6, stack);
            for (int c = 0; c < 6; c++) atlas.get(c).sampler(vt.sampler).imageView(vt.atlasView[c]).imageLayout(VK_IMAGE_LAYOUT_GENERAL);
            Buffer[] buffers = {instances, materials, transforms, vt.pageTable, vt.feedback, vt.setInfo};
            int[] binding = {3, 4, 7, 17, 18, 19};
            int i = 0;
            for (int p = 0; p < 2; p++) {
                for (int k = 0; k < buffers.length; k++) {
                    w.get(i++).sType$Default().dstSet(sets[p]).dstBinding(binding[k]).descriptorCount(1)
                            .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                            .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(buffers[k].handle).range(VK_WHOLE_SIZE));
                }
                w.get(i++).sType$Default().dstSet(sets[p]).dstBinding(16).descriptorCount(6)
                        .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).pImageInfo(atlas);
                w.get(i++).sType$Default().dstSet(sets[p]).dstBinding(21).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(VkDescriptorImageInfo.calloc(1, stack).sampler(env.sampler).imageView(env.view)
                                .imageLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL));
            }
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    /**
     * Shades history {@code parity} from the visibility buffer. {@code pushConstants} are the ones ray generation
     * gets (camera, sun, sky, flags); the raster pre-pass must have run, with its barrier to compute reads.
     */
    public void record(VkCommandBuffer cmd, MemoryStack stack, int parity, int width, int height, ByteBuffer pushConstants) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, stack.longs(sets[parity]), null);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushConstants.limit(pushSize));
        vkCmdDispatch(cmd, (width + 7) / 8, (height + 7) / 8, 1);
        pushConstants.clear();
    }

    @Override
    public void close() {
        vkDestroyPipeline(ctx.device, pipeline, null);
        vkDestroyPipelineLayout(ctx.device, layout, null);
        vkDestroyDescriptorPool(ctx.device, pool, null);
        vkDestroyDescriptorSetLayout(ctx.device, setLayout, null);
    }
}
