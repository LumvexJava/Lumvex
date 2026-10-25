package dev.lumvex.vk;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import static dev.lumvex.vk.VkUtil.align;
import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.memCopy;
import static org.lwjgl.util.shaderc.Shaderc.*;
import static org.lwjgl.vulkan.KHRAccelerationStructure.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
import static org.lwjgl.vulkan.KHRRayTracingPipeline.*;
import static org.lwjgl.vulkan.VK13.*;

/**
 * The ray tracing pipeline (raygen, two miss shaders, closest and any hit) and its shader binding table.
 *
 * <p>Descriptor set 0, one set per history parity (see {@code Renderer}):
 * <pre>
 *   0      TLAS
 *   3, 4   per-instance info (buffer addresses, material), materials
 *   1, 2   current history: color (rgb mean, a = E[L^2]), guide normal + path distance
 *   5, 6   current history: guide albedo, meta (sample count, view depth, instance, normal)
 *   7-10   previous history: color, guide normal/distance, guide albedo, meta
 *   11     radiance cache
 *   12     screen motion
 *   14, 15 animated objects, per-instance motion
 *   16-19  virtual texturing: atlases, page table, feedback, set info
 *   20, 21 sphere lights, HDR environment
 *   22     ray statistics
 *   23, 24 ReSTIR reservoirs (current, previous)
 *   25     alpha masks
 * </pre>
 */
public final class RtPipeline implements AutoCloseable {
    public static final int PUSH_CONSTANT_SIZE = 256;

    private static final int RT_STAGES = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR
            | VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

    private final VkContext ctx;
    public final long descriptorSetLayout;
    public final long pipelineLayout;
    public final long pipeline;
    public final Buffer sbt;

    public final VkStridedDeviceAddressRegionKHR raygenRegion = VkStridedDeviceAddressRegionKHR.calloc();
    /** Ray generation shader of the radiance cache training pass. */
    public final VkStridedDeviceAddressRegionKHR trainRegion = VkStridedDeviceAddressRegionKHR.calloc();
    public final VkStridedDeviceAddressRegionKHR missRegion = VkStridedDeviceAddressRegionKHR.calloc();
    public final VkStridedDeviceAddressRegionKHR hitRegion = VkStridedDeviceAddressRegionKHR.calloc();
    public final VkStridedDeviceAddressRegionKHR callableRegion = VkStridedDeviceAddressRegionKHR.calloc();

    public RtPipeline(VkContext ctx, ShaderCompiler compiler) {
        this.ctx = ctx;
        VkDevice device = ctx.device;
        try (MemoryStack stack = stackPush()) {
            int[] images = {1, 2, 5, 6, 7, 8, 9, 10, 12, 23, 24, 26, 27, 28, 29};
            VkDescriptorSetLayoutBinding.Buffer bindings = VkDescriptorSetLayoutBinding.calloc(14 + images.length, stack);
            bindings.get(13 + images.length).binding(25).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_ANY_HIT_BIT_KHR);
            bindings.get(12 + images.length).binding(22).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            bindings.get(11 + images.length).binding(21).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            bindings.get(10 + images.length).binding(20).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            // 16: atlases for the near and far pool, three channels each
            bindings.get(6 + images.length).binding(16).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .descriptorCount(6).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            for (int k = 0; k < 3; k++) {
                bindings.get(7 + images.length + k).binding(17 + k).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            }
            bindings.get(4 + images.length).binding(14).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            bindings.get(5 + images.length).binding(15).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            bindings.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR).descriptorCount(1).stageFlags(RT_STAGES);
            bindings.get(1).binding(3).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).stageFlags(RT_STAGES);
            bindings.get(2).binding(4).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1).stageFlags(RT_STAGES);
            bindings.get(3).binding(11).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            for (int i = 0; i < images.length; i++) {
                bindings.get(4 + i).binding(images[i]).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .descriptorCount(1).stageFlags(VK_SHADER_STAGE_RAYGEN_BIT_KHR);
            }
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateDescriptorSetLayout(device, VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(bindings), null, lp), "vkCreateDescriptorSetLayout");
            descriptorSetLayout = lp.get(0);

            VkPushConstantRange.Buffer pcRange = VkPushConstantRange.calloc(1, stack)
                    .stageFlags(RT_STAGES).offset(0).size(PUSH_CONSTANT_SIZE);
            check(vkCreatePipelineLayout(device, VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default()
                    .pSetLayouts(stack.longs(descriptorSetLayout))
                    .pPushConstantRanges(pcRange), null, lp), "vkCreatePipelineLayout");
            pipelineLayout = lp.get(0);

            long raygen = compiler.createModule(ctx, "raygen.rgen", shaderc_glsl_raygen_shader);
            // The training pass is the same source compiled with its own main().
            long trainRaygen = compiler.createModule(ctx, "raygen.rgen", shaderc_glsl_raygen_shader, "LUMVEX_TRAIN_PASS");
            long miss = compiler.createModule(ctx, "miss.rmiss", shaderc_glsl_miss_shader);
            long shadowMiss = compiler.createModule(ctx, "shadow.rmiss", shaderc_glsl_miss_shader);
            long closestHit = compiler.createModule(ctx, "closesthit.rchit", shaderc_glsl_closesthit_shader);
            long anyHit = compiler.createModule(ctx, "alpha.rahit", shaderc_glsl_anyhit_shader);

            VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(6, stack);
            ByteBuffer main = stack.UTF8("main");
            stages.get(0).sType$Default().stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR).module(raygen).pName(main);
            stages.get(1).sType$Default().stage(VK_SHADER_STAGE_RAYGEN_BIT_KHR).module(trainRaygen).pName(main);
            stages.get(2).sType$Default().stage(VK_SHADER_STAGE_MISS_BIT_KHR).module(miss).pName(main);
            stages.get(3).sType$Default().stage(VK_SHADER_STAGE_MISS_BIT_KHR).module(shadowMiss).pName(main);
            stages.get(4).sType$Default().stage(VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR).module(closestHit).pName(main);
            stages.get(5).sType$Default().stage(VK_SHADER_STAGE_ANY_HIT_BIT_KHR).module(anyHit).pName(main);

            // Groups: raygen, training raygen, miss, shadow miss, triangle hit group.
            VkRayTracingShaderGroupCreateInfoKHR.Buffer groups = VkRayTracingShaderGroupCreateInfoKHR.calloc(5, stack);
            for (int i = 0; i < 4; i++) {
                groups.get(i).sType$Default()
                        .type(VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR)
                        .generalShader(i)
                        .closestHitShader(VK_SHADER_UNUSED_KHR)
                        .anyHitShader(VK_SHADER_UNUSED_KHR)
                        .intersectionShader(VK_SHADER_UNUSED_KHR);
            }
            groups.get(4).sType$Default()
                    .type(VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR)
                    .generalShader(VK_SHADER_UNUSED_KHR)
                    .closestHitShader(4)
                    .anyHitShader(5)
                    .intersectionShader(VK_SHADER_UNUSED_KHR);

            VkRayTracingPipelineCreateInfoKHR.Buffer info = VkRayTracingPipelineCreateInfoKHR.calloc(1, stack);
            info.get(0).sType$Default()
                    .pStages(stages)
                    .pGroups(groups)
                    .maxPipelineRayRecursionDepth(Math.min(2, ctx.maxRayRecursionDepth))
                    .layout(pipelineLayout);
            check(vkCreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, info, null, lp),
                    "vkCreateRayTracingPipelinesKHR");
            pipeline = lp.get(0);

            vkDestroyShaderModule(device, raygen, null);
            vkDestroyShaderModule(device, trainRaygen, null);
            vkDestroyShaderModule(device, miss, null);
            vkDestroyShaderModule(device, shadowMiss, null);
            vkDestroyShaderModule(device, closestHit, null);
            vkDestroyShaderModule(device, anyHit, null);

            int handleSize = ctx.shaderGroupHandleSize;
            int handleStride = align(handleSize, ctx.shaderGroupHandleAlignment);
            int base = ctx.shaderGroupBaseAlignment;
            int groupCount = 5;
            ByteBuffer handles = stack.malloc(groupCount * handleSize);
            check(vkGetRayTracingShaderGroupHandlesKHR(device, pipeline, 0, groupCount, handles),
                    "vkGetRayTracingShaderGroupHandlesKHR");

            long raygenSize = align(handleStride, base);
            long missSize = align(2L * handleStride, base);
            long hitSize = align(handleStride, base);
            long total = 2 * raygenSize + missSize + hitSize + base; // extra room to align the start address
            sbt = new Buffer(ctx, total,
                    VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    Buffer.UPLOAD);
            long start = align(sbt.address, base);
            int startOffset = (int) (start - sbt.address);
            ByteBuffer dst = sbt.map();

            long missStart = 2 * raygenSize;
            copyHandle(handles, 0, handleSize, dst, startOffset);
            copyHandle(handles, 1, handleSize, dst, (int) (startOffset + raygenSize));
            copyHandle(handles, 2, handleSize, dst, (int) (startOffset + missStart));
            copyHandle(handles, 3, handleSize, dst, (int) (startOffset + missStart + handleStride));
            copyHandle(handles, 4, handleSize, dst, (int) (startOffset + missStart + missSize));

            raygenRegion.deviceAddress(start).stride(raygenSize).size(raygenSize);
            trainRegion.deviceAddress(start + raygenSize).stride(raygenSize).size(raygenSize);
            missRegion.deviceAddress(start + missStart).stride(handleStride).size(missSize);
            hitRegion.deviceAddress(start + missStart + missSize).stride(handleStride).size(hitSize);
        }
    }

    private static void copyHandle(ByteBuffer handles, int group, int handleSize, ByteBuffer dst, int offset) {
        memCopy(handles.slice(group * handleSize, handleSize), dst.slice(offset, handleSize));
    }

    @Override
    public void close() {
        sbt.close();
        raygenRegion.free();
        trainRegion.free();
        missRegion.free();
        hitRegion.free();
        callableRegion.free();
        vkDestroyPipeline(ctx.device, pipeline, null);
        vkDestroyPipelineLayout(ctx.device, pipelineLayout, null);
        vkDestroyDescriptorSetLayout(ctx.device, descriptorSetLayout, null);
    }
}
