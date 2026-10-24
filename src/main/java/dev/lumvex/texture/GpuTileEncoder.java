package dev.lumvex.texture;

import dev.lumvex.vk.Buffer;
import dev.lumvex.vk.ShaderCompiler;
import dev.lumvex.vk.VkContext;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_compute_shader;
import static org.lwjgl.vulkan.VK13.*;

/**
 * Block-compresses texture tiles on the GPU ({@code bc_encode.comp}, producing the same bitstream as
 * {@link BlockCompression}). The CPU only copies raw RGBA8 tiles into the upload buffer; a single dispatch per frame
 * encodes all of them in microseconds per tile, versus about 28 ms per tile on a CPU thread.
 *
 * <p>Formats: 0 = BC1, 1 = BC5, 2 = BC7 (mode 6).
 */
public final class GpuTileEncoder implements AutoCloseable {
    private static final int JOB_BYTES = 16;

    private final VkContext ctx;
    /** Encoded blocks; the source of the copies into the tile atlases. */
    public final Buffer encoded;
    private final Buffer jobs;
    private final ByteBuffer jobData;
    private final int maxJobs;
    private final long setLayout, pipelineLayout, pipeline, pool, set;
    private int jobCount, maxBlocks;

    /** {@code maxJobs} bounds the number of tile channels encoded per frame. */
    public GpuTileEncoder(VkContext ctx, ShaderCompiler compiler, Buffer raw, long encodedBytes, int maxJobs) {
        this.ctx = ctx;
        this.maxJobs = maxJobs;
        encoded = new Buffer(ctx, encodedBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.DEVICE);
        jobs = new Buffer(ctx, (long) JOB_BYTES * maxJobs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        jobData = jobs.map();
        try (MemoryStack stack = stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer b = VkDescriptorSetLayoutBinding.calloc(3, stack);
            for (int i = 0; i < 3; i++) {
                b.get(i).binding(i).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            }
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateDescriptorSetLayout(ctx.device, VkDescriptorSetLayoutCreateInfo.calloc(stack).sType$Default().pBindings(b), null, lp),
                    "vkCreateDescriptorSetLayout(bc)");
            setLayout = lp.get(0);
            check(vkCreatePipelineLayout(ctx.device, VkPipelineLayoutCreateInfo.calloc(stack).sType$Default().pSetLayouts(stack.longs(setLayout)), null, lp),
                    "vkCreatePipelineLayout(bc)");
            pipelineLayout = lp.get(0);
            long module = compiler.createModule(ctx, "bc_encode.comp", shaderc_glsl_compute_shader);
            VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
            info.get(0).sType$Default().layout(pipelineLayout);
            info.get(0).stage().sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
            check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, info, null, lp), "vkCreateComputePipelines(bc)");
            pipeline = lp.get(0);
            vkDestroyShaderModule(ctx.device, module, null);

            check(vkCreateDescriptorPool(ctx.device, VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(1)
                    .pPoolSizes(VkDescriptorPoolSize.calloc(1, stack).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(3)), null, lp),
                    "vkCreateDescriptorPool(bc)");
            pool = lp.get(0);
            check(vkAllocateDescriptorSets(ctx.device, VkDescriptorSetAllocateInfo.calloc(stack).sType$Default().descriptorPool(pool)
                    .pSetLayouts(stack.longs(setLayout)), lp), "vkAllocateDescriptorSets(bc)");
            set = lp.get(0);
            Buffer[] bufs = {raw, encoded, jobs};
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(3, stack);
            for (int i = 0; i < 3; i++) {
                w.get(i).sType$Default().dstSet(set).dstBinding(i).descriptorCount(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(bufs[i].handle).range(VK_WHOLE_SIZE));
            }
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    public void begin() { jobCount = 0; maxBlocks = 0; }

    /** Queues a {@code width} x {@code width} RGBA8 tile at {@code rawOffset} for encoding to {@code encodedOffset}. */
    public void add(long rawOffset, long encodedOffset, int format, int width) {
        if (jobCount >= maxJobs) throw new IllegalStateException("too many encode jobs");
        int o = jobCount++ * JOB_BYTES;
        jobData.putInt(o, (int) (rawOffset / 4)).putInt(o + 4, (int) (encodedOffset / 4)).putInt(o + 8, format).putInt(o + 12, width);
        maxBlocks = Math.max(maxBlocks, (width / 4) * (width / 4));
    }

    public int jobCount() { return jobCount; }

    /** Records the dispatch plus a barrier that makes {@link #encoded} readable by transfers. */
    public void record(VkCommandBuffer cmd, MemoryStack stack) {
        if (jobCount == 0) return;
        VkMemoryBarrier.Buffer hb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT).dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, hb, null, null);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, stack.longs(set), null);
        vkCmdDispatch(cmd, (maxBlocks + 63) / 64, jobCount, 1);
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT).dstAccessMask(VK_ACCESS_TRANSFER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, mb, null, null);
    }

    @Override
    public void close() {
        vkDestroyPipeline(ctx.device, pipeline, null);
        vkDestroyPipelineLayout(ctx.device, pipelineLayout, null);
        vkDestroyDescriptorPool(ctx.device, pool, null);
        vkDestroyDescriptorSetLayout(ctx.device, setLayout, null);
        encoded.close();
        jobs.close();
    }
}
