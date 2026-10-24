package dev.lumvex.vk;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_compute_shader;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR;
import static org.lwjgl.vulkan.VK13.*;

/**
 * The world-space radiance cache that paths end in, and its resolve pass ({@code cache_resolve.comp}).
 *
 * <p>Training paths add samples to a cell's accumulator. Once per frame, between training and the image pass, the
 * resolve pass blends them into the cell's mean with a history of at most {@link #maxFrames} frames - fewer while
 * objects move, and only {@link #maxFramesDynamic} inside their influence zones - and frees cells that nobody
 * trained for {@link #staleFrames} frames. That keeps the cache valid while lights move, so paths may end in it at
 * any time (after SHaRC, NVIDIA's spatially hashed radiance cache).
 *
 * <p>Buffer layout (std430, see {@code cache.glsl}): {@code uint keys[SLOTS] | uint meta[SLOTS] |
 * vec4 resolved[SLOTS] | vec4 accumulator[SLOTS]}.
 */
public final class RadianceCache implements AutoCloseable {
    /** Slots of the hash table; must match CACHE_SIZE in cache.glsl. */
    public static final int SLOTS = 1 << 20;
    private static final int BYTES_PER_SLOT = 4 + 4 + 16 + 16;
    private static final int GROUP_SIZE = 256;

    /** History of a cell in frames of training, while nothing moves. */
    public float maxFrames = 256;
    /** History of a cell while objects move: light bounced off them reaches far beyond their influence zones. */
    public float maxFramesAnimated = 32;
    /** History of a cell inside the influence zone of a moving object or light. */
    public float maxFramesDynamic = 3;
    /** Upper bound of a cell's weight in samples; a converged cell keeps learning at this rate. */
    public float maxWeight = 1024;
    /** Frames without training after which a cell is freed for others. */
    public int staleFrames = 1024;

    private final VkContext ctx;
    public final Buffer buffer;
    private final long setLayout, pipelineLayout, pipeline, pool, set;
    private boolean clearPending;

    public RadianceCache(VkContext ctx, ShaderCompiler compiler) {
        this.ctx = ctx;
        buffer = new Buffer(ctx, (long) SLOTS * BYTES_PER_SLOT,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, Buffer.DEVICE);
        ctx.submitImmediate(c -> vkCmdFillBuffer(c, buffer.handle, 0, VK_WHOLE_SIZE, 0));
        try (MemoryStack stack = stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer b = VkDescriptorSetLayoutBinding.calloc(1, stack);
            b.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateDescriptorSetLayout(ctx.device, VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(b), null, lp), "vkCreateDescriptorSetLayout(cache)");
            setLayout = lp.get(0);
            VkPushConstantRange.Buffer range = VkPushConstantRange.calloc(1, stack);
            range.get(0).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(16);
            check(vkCreatePipelineLayout(ctx.device, VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(setLayout)).pPushConstantRanges(range), null, lp),
                    "vkCreatePipelineLayout(cache)");
            pipelineLayout = lp.get(0);

            long module = compiler.createModule(ctx, "cache_resolve.comp", shaderc_glsl_compute_shader);
            VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
            info.get(0).sType$Default().layout(pipelineLayout);
            info.get(0).stage().sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
            check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, info, null, lp), "vkCreateComputePipelines(cache)");
            pipeline = lp.get(0);
            vkDestroyShaderModule(ctx.device, module, null);

            VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(1, stack);
            sizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1);
            check(vkCreateDescriptorPool(ctx.device, VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(1).pPoolSizes(sizes), null, lp), "vkCreateDescriptorPool(cache)");
            pool = lp.get(0);
            check(vkAllocateDescriptorSets(ctx.device, VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(pool).pSetLayouts(stack.longs(setLayout)), lp),
                    "vkAllocateDescriptorSets(cache)");
            set = lp.get(0);

            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(1, stack);
            w.get(0).sType$Default().dstSet(set).dstBinding(0).descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(buffer.handle).range(VK_WHOLE_SIZE));
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    /** Empties the cache at the start of the next frame, e.g. after the scene or the light settings changed. */
    public void clear() { clearPending = true; }

    /** Records the pending clear, if any, before the frame's first shader access. */
    public void recordClear(VkCommandBuffer cmd, MemoryStack stack) {
        if (!clearPending) return;
        vkCmdFillBuffer(cmd, buffer.handle, 0, VK_WHOLE_SIZE, 0);
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, mb, null, null);
        clearPending = false;
    }

    /**
     * Records the resolve pass. Training writes must be visible to it; the image pass must wait for it.
     *
     * @param animated whether objects moved this frame
     */
    public void recordResolve(VkCommandBuffer cmd, MemoryStack stack, boolean animated) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, stack.longs(set), null);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, stack.malloc(16)
                .putFloat(0, animated ? Math.min(maxFramesAnimated, maxFrames) : maxFrames)
                .putFloat(4, maxFramesDynamic).putFloat(8, maxWeight).putInt(12, staleFrames));
        vkCmdDispatch(cmd, SLOTS / GROUP_SIZE, 1, 1);
    }

    @Override
    public void close() {
        vkDestroyPipeline(ctx.device, pipeline, null);
        vkDestroyPipelineLayout(ctx.device, pipelineLayout, null);
        vkDestroyDescriptorPool(ctx.device, pool, null);
        vkDestroyDescriptorSetLayout(ctx.device, setLayout, null);
        buffer.close();
    }
}
