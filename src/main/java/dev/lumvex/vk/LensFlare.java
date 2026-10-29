package dev.lumvex.vk;

import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_compute_shader;
import static org.lwjgl.vulkan.KHRAccelerationStructure.*;
import static org.lwjgl.vulkan.VK13.*;

/**
 * Lens flares for the sun and bright lights. Each frame the host lists the sources; {@code flare.comp} measures
 * their visibility with 64 ray queries each and projects them to the screen, and {@code composite.comp} draws glow,
 * starburst, ghosts and halo from the result.
 *
 * <p>Buffer layout (std430): {@code mat4 viewProj | vec4 camPos (w = count) | vec4 screen | Flare[max]}, where a
 * flare is three vec4.
 */
public final class LensFlare implements AutoCloseable {
    private static final int HEADER = 96, ENTRY = 48;
    private final VkContext ctx;
    public final int maxFlares;
    public final Buffer buffer;
    private final ByteBuffer mapped;
    private final long setLayout, pipelineLayout, pipeline, pool, set;
    private int count;
    private boolean reset = true;

    public LensFlare(VkContext ctx, ShaderCompiler compiler, long tlas, int maxFlares) {
        this.ctx = ctx;
        this.maxFlares = maxFlares;
        buffer = new Buffer(ctx, HEADER + (long) ENTRY * maxFlares, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        mapped = buffer.map();
        for (int i = 0; i < HEADER + ENTRY * maxFlares; i += 4) mapped.putInt(i, 0);
        try (MemoryStack stack = stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer b = VkDescriptorSetLayoutBinding.calloc(2, stack);
            b.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            b.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateDescriptorSetLayout(ctx.device, VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(b), null, lp), "vkCreateDescriptorSetLayout(flare)");
            setLayout = lp.get(0);
            check(vkCreatePipelineLayout(ctx.device, VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(setLayout)), null, lp), "vkCreatePipelineLayout(flare)");
            pipelineLayout = lp.get(0);

            long module = compiler.createModule(ctx, "flare.comp", shaderc_glsl_compute_shader);
            VkComputePipelineCreateInfo.Buffer info = VkComputePipelineCreateInfo.calloc(1, stack);
            info.get(0).sType$Default().layout(pipelineLayout);
            info.get(0).stage().sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
            check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, info, null, lp), "vkCreateComputePipelines(flare)");
            pipeline = lp.get(0);
            vkDestroyShaderModule(ctx.device, module, null);

            VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(2, stack);
            sizes.get(0).type(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1);
            sizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1);
            check(vkCreateDescriptorPool(ctx.device, VkDescriptorPoolCreateInfo.calloc(stack)
                    .sType$Default().maxSets(1).pPoolSizes(sizes), null, lp), "vkCreateDescriptorPool(flare)");
            pool = lp.get(0);
            check(vkAllocateDescriptorSets(ctx.device, VkDescriptorSetAllocateInfo.calloc(stack)
                    .sType$Default().descriptorPool(pool).pSetLayouts(stack.longs(setLayout)), lp), "vkAllocateDescriptorSets(flare)");
            set = lp.get(0);

            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(2, stack);
            w.get(0).sType$Default().dstSet(set).dstBinding(0).descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR)
                    .pNext(VkWriteDescriptorSetAccelerationStructureKHR.calloc(stack).sType$Default()
                            .pAccelerationStructures(stack.longs(tlas)).address());
            w.get(1).sType$Default().dstSet(set).dstBinding(1).descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                    .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(buffer.handle).range(VK_WHOLE_SIZE));
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    /** Clears the source list. The GPU must be done with the previous frame. */
    public void begin() { count = 0; }

    /** Adds a directional source such as the sun. */
    public void addDirectional(Vector3f direction, float angularRadius, Vector3f intensity) {
        if (count >= maxFlares) return;
        put(direction.x, direction.y, direction.z, -angularRadius, intensity.x, intensity.y, intensity.z);
    }

    // A negative radius marks a directional source with an angular radius.
    private void put(float x, float y, float z, float w, float r, float g, float b) {
        int o = HEADER + count * ENTRY;
        mapped.putFloat(o, x).putFloat(o + 4, y).putFloat(o + 8, z).putFloat(o + 12, w)
                .putFloat(o + 16, r).putFloat(o + 20, g).putFloat(o + 24, b).putFloat(o + 28, 0);
        count++;
    }

    /** Discards the temporally smoothed visibilities, e.g. after a scene or resolution change. */
    public void resetHistory() { reset = true; }

    public int count() { return count; }

    /** Prints the previous frame's GPU results, for debugging. */
    public void dump() {
        StringBuilder sb = new StringBuilder("[flare]");
        for (int i = 0; i < count; i++) {
            int o = HEADER + i * ENTRY;
            sb.append(String.format(" (%.0f,%.0f v%.2f i%.2f)", mapped.getFloat(o + 32), mapped.getFloat(o + 36),
                    mapped.getFloat(o + 40), mapped.getFloat(o + 16)));
        }
        System.out.println(sb);
    }

    /** Records the visibility pass. The TLAS build must be visible to compute shaders. */
    public void record(VkCommandBuffer cmd, Matrix4f viewProj, Vector3f camPos, int width, int height, int frame) {
        viewProj.get(0, mapped);
        mapped.putFloat(64, camPos.x).putFloat(68, camPos.y).putFloat(72, camPos.z).putFloat(76, count)
                .putFloat(80, width).putFloat(84, height).putFloat(88, frame).putFloat(92, reset ? 1f : 0.45f);
        reset = false;
        if (count == 0) return;
        try (MemoryStack stack = stackPush()) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, stack.longs(set), null);
            vkCmdDispatch(cmd, count, 1, 1);
        }
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
