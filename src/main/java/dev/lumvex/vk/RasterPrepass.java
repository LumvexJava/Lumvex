package dev.lumvex.vk;

import dev.lumvex.render.MeshLod;
import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.*;

import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.List;

import static dev.lumvex.vk.VkUtil.check;
import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.system.MemoryUtil.memAlloc;
import static org.lwjgl.system.MemoryUtil.memFree;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_compute_shader;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_fragment_shader;
import static org.lwjgl.util.shaderc.Shaderc.shaderc_glsl_vertex_shader;
import static org.lwjgl.vulkan.VK13.*;

/**
 * The raster pre-pass of the hybrid and raster render modes: rasterizes the scene into a visibility buffer before
 * the light is computed.
 *
 * <p>Each texel of {@link #visibility} holds {@code (instance + 1, i0, i1, i2)} of the nearest surface at the pixel
 * center, the instance and the three vertex indices of its triangle, 0 for sky. Ray generation (and the raster
 * mode's shading pass) rebuild the hit from it with one ray-triangle intersection.
 *
 * <ul>
 *   <li>Vulkan 1.3 dynamic rendering, no render pass or framebuffer objects.</li>
 *   <li><b>Frustum and occlusion culling on the GPU in two phases</b> ({@code raster_cull.comp}). Phase 0 draws
 *   the instances in the frustum that were visible in the last frame; their depth is reduced to a Hi-Z pyramid
 *   ({@code hiz.comp}, the farthest depth of each block), and phase 1 draws the instances in the frustum that are not
 *   hidden behind it and were not drawn yet. Anything that comes into view is drawn in the same frame.
 *   {@code -Dlumvex.noocclusion=true} turns phase 1 off.</li>
 *   <li><b>GPU instancing:</b> instances grouped by mesh, one indexed indirect draw per mesh, level of detail and
 *   phase, from one shared index buffer. The draw list is only rebuilt when the scene's structure changes
 *   ({@link #setDraws}); moving objects just update the transform buffer.</li>
 *   <li>Vertices are pulled through buffer device addresses from the buffers the acceleration structures are built
 *   from; the transforms come from the buffer ray generation reads too.</li>
 *   <li>Reversed, infinite depth (1 at the near plane, 0 at infinity) in a 32-bit float buffer.</li>
 *   <li>Double-sided, like the TLAS instances.</li>
 *   <li>Alpha-tested instances are only drawn in the raster mode, with a second pipeline that runs the alpha test;
 *   in the hybrid mode raygen tests them by ray.</li>
 * </ul>
 * The images have the size of the traced image including its overscan border.
 */
public final class RasterPrepass implements AutoCloseable {
    public static final int VISIBILITY_FORMAT = VK_FORMAT_R32G32B32A32_UINT;
    private static final int DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;
    /** mat4 view-projection and the device address of the shared index buffer. */
    private static final int PUSH_SIZE = 72;
    private static final int LEVELS = MeshLod.MAX_LEVELS;
    /** Projected geometric error in pixels below which a coarser level of detail is used. */
    public float lodPixels = Float.parseFloat(System.getProperty("lumvex.lodpixels", "1"));
    /** VkDrawIndexedIndirectCommand: indexCount, instanceCount, firstIndex, vertexOffset, firstInstance. */
    private static final int DRAW_SIZE = 20;
    /** Culling parameters: mat4, five planes, uvec4 counts, vec4 screen, vec4 camera. */
    private static final int PARAMS_SIZE = 64 + 5 * 16 + 16 + 16 + 16;
    /** Occlusion culling (phase 1) on; off draws everything in the frustum. */
    public boolean occlusionCulling = !Boolean.getBoolean("lumvex.noocclusion");

    private final VkContext ctx;
    private final long setLayout, pipelineLayout, pipeline, pipelineCutout, pool, set;
    /** Mesh groups drawn with the opaque pipeline; the groups behind them are alpha-tested (raster mode only). */
    private int opaqueGroups;
    private final long cullSetLayout, cullLayout, cullPipeline, cullSet;
    private final long hizSetLayout, hizLayout, hizPipeline, sampler;
    private long hizPool;
    private long[] hizSets = new long[0];
    /** Visibility buffer, kept in {@code VK_IMAGE_LAYOUT_GENERAL} so ray generation reads it as a storage image. */
    public StorageImage visibility;
    private long depthImage, depthMemory, depthView;
    /** Hi-Z pyramid: one view of all levels for culling, one per level for writing. */
    private long hizImage, hizMemory, hizView;
    private long[] hizLevelViews = new long[0];
    private int hizLevels, hizWidth, hizHeight;
    private int width, height;
    /**
     * Render instance indices grouped by mesh, the group of each entry, every group's mesh bounds, the indirect draws
     * of both phases as uploaded (template, instanceCount 0) and as filled by the culling pass, the instances each
     * phase draws (phase 1 behind phase 0's), and whether each entry was visible in the last frame.
     */
    private Buffer drawList, drawGroup, groupBounds, commandTemplate, drawCommands, culled, visibleLast, params;
    /** Per group: number of levels and their errors, the first triangle of each level; per culled entry its level's first triangle. */
    private Buffer groupLods, culledFirstTri;
    /** Every drawn mesh's indices, back to back; each draw starts at its mesh's offset. */
    private Buffer indexBuffer;
    private int drawCount, listCount;
    private boolean visibleLastStale = true;
    private final Vector4f[] planes = {new Vector4f(), new Vector4f(), new Vector4f(), new Vector4f(), new Vector4f()};
    private final Vector3f camera = new Vector3f();
    private float pixelsPerRadian;

    /**
     * Creates the pipelines and their descriptor sets:
     * <ul>
     * <li>the draw: bindings 0-3 for the vertex stage (instances, transforms, culled list, first triangles), 4
     * materials and 5 alpha masks for the fragment stage; an opaque pipeline and one with the alpha test, which
     * discards fragments and so loses early depth testing;</li>
     * <li>the culling pass: storage buffers, and the Hi-Z pyramid at binding 8;</li>
     * <li>the Hi-Z pass: the level below (or the depth buffer) and the level to write.</li>
     * </ul>
     */
    public RasterPrepass(VkContext ctx, ShaderCompiler compiler) {
        this.ctx = ctx;
        VkDevice device = ctx.device;
        try (MemoryStack stack = stackPush()) {
            VkDescriptorSetLayoutBinding.Buffer b = VkDescriptorSetLayoutBinding.calloc(6, stack);
            for (int i = 0; i < 6; i++) {
                b.get(i).binding(i).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(1)
                        .stageFlags(i < 4 ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT);
            }
            LongBuffer lp = stack.mallocLong(1);
            check(vkCreateDescriptorSetLayout(device, VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(b), null, lp), "vkCreateDescriptorSetLayout(raster)");
            setLayout = lp.get(0);
            check(vkCreatePipelineLayout(device, VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default()
                    .pSetLayouts(stack.longs(setLayout))
                    .pPushConstantRanges(VkPushConstantRange.calloc(1, stack)
                            .stageFlags(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT).offset(0).size(PUSH_SIZE)),
                    null, lp), "vkCreatePipelineLayout(raster)");
            pipelineLayout = lp.get(0);

            VkDescriptorSetLayoutBinding.Buffer cb = VkDescriptorSetLayoutBinding.calloc(11, stack);
            for (int i = 0; i < 11; i++) {
                cb.get(i).binding(i).descriptorType(i == 8 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .descriptorCount(1).stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            }
            check(vkCreateDescriptorSetLayout(device, VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(cb), null, lp), "vkCreateDescriptorSetLayout(raster cull)");
            cullSetLayout = lp.get(0);
            check(vkCreatePipelineLayout(device, VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default()
                    .pSetLayouts(stack.longs(cullSetLayout))
                    .pPushConstantRanges(VkPushConstantRange.calloc(1, stack)
                            .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(4)),
                    null, lp), "vkCreatePipelineLayout(raster cull)");
            cullLayout = lp.get(0);
            cullPipeline = computePipeline(stack, compiler, "raster_cull.comp", cullLayout);

            VkDescriptorSetLayoutBinding.Buffer hb = VkDescriptorSetLayoutBinding.calloc(2, stack);
            hb.get(0).binding(0).descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            hb.get(1).binding(1).descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(1)
                    .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT);
            check(vkCreateDescriptorSetLayout(device, VkDescriptorSetLayoutCreateInfo.calloc(stack)
                    .sType$Default().pBindings(hb), null, lp), "vkCreateDescriptorSetLayout(hiz)");
            hizSetLayout = lp.get(0);
            check(vkCreatePipelineLayout(device, VkPipelineLayoutCreateInfo.calloc(stack)
                    .sType$Default().pSetLayouts(stack.longs(hizSetLayout)), null, lp), "vkCreatePipelineLayout(hiz)");
            hizLayout = lp.get(0);
            hizPipeline = computePipeline(stack, compiler, "hiz.comp", hizLayout);
            check(vkCreateSampler(device, VkSamplerCreateInfo.calloc(stack).sType$Default()
                    .magFilter(VK_FILTER_NEAREST).minFilter(VK_FILTER_NEAREST)
                    .mipmapMode(VK_SAMPLER_MIPMAP_MODE_NEAREST)
                    .addressModeU(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeV(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .addressModeW(VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE)
                    .maxLod(VK_LOD_CLAMP_NONE), null, lp), "vkCreateSampler(hiz)");
            sampler = lp.get(0);

            VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(2, stack);
            sizes.get(0).type(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER).descriptorCount(16);
            sizes.get(1).type(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(1);
            check(vkCreateDescriptorPool(device, VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(2)
                    .pPoolSizes(sizes), null, lp), "vkCreateDescriptorPool(raster)");
            pool = lp.get(0);
            LongBuffer sets = stack.mallocLong(2);
            check(vkAllocateDescriptorSets(device, VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(pool).pSetLayouts(stack.longs(setLayout, cullSetLayout)), sets), "vkAllocateDescriptorSets(raster)");
            set = sets.get(0);
            cullSet = sets.get(1);

            long vert = compiler.createModule(ctx, "raster.vert", shaderc_glsl_vertex_shader);
            long frag = compiler.createModule(ctx, "raster.frag", shaderc_glsl_fragment_shader);
            long fragCutout = compiler.createModule(ctx, "raster.frag", shaderc_glsl_fragment_shader, "LUMVEX_ALPHA_TEST");
            ByteBuffer main = stack.UTF8("main");
            VkPipelineShaderStageCreateInfo.Buffer stages = VkPipelineShaderStageCreateInfo.calloc(2, stack);
            stages.get(0).sType$Default().stage(VK_SHADER_STAGE_VERTEX_BIT).module(vert).pName(main);
            stages.get(1).sType$Default().stage(VK_SHADER_STAGE_FRAGMENT_BIT).module(frag).pName(main);

            VkPipelineRenderingCreateInfo rendering = VkPipelineRenderingCreateInfo.calloc(stack)
                    .sType$Default()
                    .colorAttachmentCount(1)
                    .pColorAttachmentFormats(stack.ints(VISIBILITY_FORMAT))
                    .depthAttachmentFormat(DEPTH_FORMAT);
            VkPipelineColorBlendAttachmentState.Buffer blend = VkPipelineColorBlendAttachmentState.calloc(1, stack)
                    .colorWriteMask(VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT
                            | VK_COLOR_COMPONENT_A_BIT);
            VkGraphicsPipelineCreateInfo.Buffer info = VkGraphicsPipelineCreateInfo.calloc(1, stack);
            info.get(0).sType$Default()
                    .pNext(rendering.address())
                    .pStages(stages)
                    .pVertexInputState(VkPipelineVertexInputStateCreateInfo.calloc(stack).sType$Default())
                    .pInputAssemblyState(VkPipelineInputAssemblyStateCreateInfo.calloc(stack).sType$Default()
                            .topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST))
                    .pViewportState(VkPipelineViewportStateCreateInfo.calloc(stack).sType$Default()
                            .viewportCount(1).scissorCount(1))
                    .pRasterizationState(VkPipelineRasterizationStateCreateInfo.calloc(stack).sType$Default()
                            .polygonMode(VK_POLYGON_MODE_FILL)
                            .cullMode(VK_CULL_MODE_NONE)
                            .frontFace(VK_FRONT_FACE_COUNTER_CLOCKWISE)
                            .lineWidth(1f))
                    .pMultisampleState(VkPipelineMultisampleStateCreateInfo.calloc(stack).sType$Default()
                            .rasterizationSamples(VK_SAMPLE_COUNT_1_BIT))
                    .pDepthStencilState(VkPipelineDepthStencilStateCreateInfo.calloc(stack).sType$Default()
                            .depthTestEnable(true)
                            .depthWriteEnable(true)
                            .depthCompareOp(VK_COMPARE_OP_GREATER))
                    .pColorBlendState(VkPipelineColorBlendStateCreateInfo.calloc(stack).sType$Default().pAttachments(blend))
                    .pDynamicState(VkPipelineDynamicStateCreateInfo.calloc(stack).sType$Default()
                            .pDynamicStates(stack.ints(VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR)))
                    .layout(pipelineLayout);
            check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, info, null, lp), "vkCreateGraphicsPipelines(raster)");
            pipeline = lp.get(0);
            stages.get(1).module(fragCutout);
            check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, info, null, lp), "vkCreateGraphicsPipelines(raster cutout)");
            pipelineCutout = lp.get(0);
            vkDestroyShaderModule(device, vert, null);
            vkDestroyShaderModule(device, frag, null);
            vkDestroyShaderModule(device, fragCutout, null);
        }
        params = new Buffer(ctx, PARAMS_SIZE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, Buffer.UPLOAD);
        allocateLists(256, 64);
        ByteBuffer one = memAlloc(12);
        one.putInt(0, 0).putInt(4, 1).putInt(8, 2);
        indexBuffer = Buffer.deviceLocal(ctx, one, INDEX_USAGE);
        memFree(one);
    }

    /** The shared index buffer is bound for the draws and read by the fragment stage through its address. */
    private static final int INDEX_USAGE = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    /** Compiles a compute shader and creates its pipeline. */
    private long computePipeline(MemoryStack stack, ShaderCompiler compiler, String shader, long layout) {
        long module = compiler.createModule(ctx, shader, shaderc_glsl_compute_shader);
        VkComputePipelineCreateInfo.Buffer ci = VkComputePipelineCreateInfo.calloc(1, stack);
        ci.get(0).sType$Default().layout(layout);
        ci.get(0).stage().sType$Default().stage(VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(stack.UTF8("main"));
        LongBuffer lp = stack.mallocLong(1);
        check(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, ci, null, lp), "vkCreateComputePipelines(" + shader + ")");
        vkDestroyShaderModule(ctx.device, module, null);
        return lp.get(0);
    }

    /**
     * Allocates the draw list buffers for {@code entries} instances in {@code groups} mesh groups; the culled list
     * has a range as long as the draw list per phase and level.
     */
    private void allocateLists(int entries, int groups) {
        int storage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        drawList = new Buffer(ctx, 4L * entries, storage, Buffer.UPLOAD);
        drawGroup = new Buffer(ctx, 4L * entries, storage, Buffer.UPLOAD);
        culled = new Buffer(ctx, 2L * LEVELS * 4 * entries, storage, Buffer.DEVICE);
        culledFirstTri = new Buffer(ctx, 2L * LEVELS * 4 * entries, storage, Buffer.DEVICE);
        groupLods = new Buffer(ctx, 32L * groups, storage, Buffer.UPLOAD);
        visibleLast = new Buffer(ctx, 4L * entries, storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, Buffer.DEVICE);
        groupBounds = new Buffer(ctx, 32L * groups, storage, Buffer.UPLOAD);
        commandTemplate = new Buffer(ctx, 2L * LEVELS * DRAW_SIZE * groups, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, Buffer.UPLOAD);
        drawCommands = new Buffer(ctx, 2L * LEVELS * DRAW_SIZE * groups,
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | storage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, Buffer.DEVICE);
        visibleLastStale = true;
    }

    /** Frees the buffers made by {@link #allocateLists}. */
    private void closeLists() {
        drawList.close();
        drawGroup.close();
        culled.close();
        culledFirstTri.close();
        groupLods.close();
        visibleLast.close();
        groupBounds.close();
        commandTemplate.close();
        drawCommands.close();
    }

    /**
     * Binds the per-instance buffers shared with ray tracing: buffer addresses and material per instance, and the
     * object-to-world matrices (mat4x3). Call again whenever either is reallocated; the GPU must be idle.
     */
    public void setInstanceBuffers(Buffer instanceInfo, Buffer transforms) {
        writeSet(instanceInfo, transforms);
    }

    private Buffer instanceInfo, transforms;

    /** Binds the materials and the alpha masks for the alpha-tested groups; call when either is replaced. */
    public void setAlphaBuffers(Buffer materials, Buffer alphaMasks) {
        try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(2, stack);
            Buffer[] b = {materials, alphaMasks};
            for (int i = 0; i < 2; i++) {
                w.get(i).sType$Default().dstSet(set).dstBinding(4 + i).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(b[i].handle).range(VK_WHOLE_SIZE));
            }
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    /**
     * Writes the buffers into the draw set (bindings 0-3) and the culling set (the rest, binding 8 being the Hi-Z):
     * the draw reads the culled list, the culling pass reads the full list and writes the culled one. A null
     * argument keeps the buffer bound before.
     */
    private void writeSet(Buffer info, Buffer xf) {
        if (info != null) instanceInfo = info;
        if (xf != null) transforms = xf;
        if (instanceInfo == null || transforms == null) return;
        try (MemoryStack stack = stackPush()) {
            Buffer[] buffers = {instanceInfo, transforms, culled, culledFirstTri,
                    transforms, drawList, drawGroup, groupBounds, drawCommands, culled, visibleLast, params, null, groupLods, culledFirstTri};
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(buffers.length - 1, stack);
            int k = 0;
            for (int i = 0; i < buffers.length; i++) {
                if (buffers[i] == null) continue;
                w.get(k++).sType$Default().dstSet(i < 4 ? set : cullSet).dstBinding(i < 4 ? i : i - 4).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
                        .pBufferInfo(VkDescriptorBufferInfo.calloc(1, stack).buffer(buffers[i].handle).range(VK_WHOLE_SIZE));
            }
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
        writeHizSets();
    }

    /**
     * Sets what to draw: {@code instances} lists render instance indices grouped by mesh, and group {@code g} covers
     * {@code instances[groupStart[g] .. groupStart[g + 1])}; its mesh has the levels of detail {@code groupLods.get(g)}
     * and the local bounds {@code bounds[6g .. 6g + 6)} (min xyz, max xyz). The GPU must not be using the previous
     * lists (call between frames, after the fence).
     */
    public void setDraws(int[] instances, int instanceCount, int[] groupStart, List<MeshLod> lods, float[] bounds, int groups) {
        setDraws(instances, instanceCount, groupStart, lods, bounds, groups, groups);
    }

    /**
     * As above; groups from {@code opaqueGroups} on are drawn with the alpha test (the raster mode's cutouts). Builds
     * one index buffer for all levels of all meshes and the indirect draw templates: each phase and level has its own
     * range of the culled list, and its instance count starts at 0 every frame, counted up by the culling passes.
     */
    public void setDraws(int[] instances, int instanceCount, int[] groupStart, List<MeshLod> lods, float[] bounds, int groups,
                         int opaqueGroups) {
        this.opaqueGroups = opaqueGroups;
        if (4L * Math.max(instanceCount, 1) > drawList.size || 2L * LEVELS * DRAW_SIZE * Math.max(groups, 1) > drawCommands.size) {
            vkDeviceWaitIdle(ctx.device);
            int entries = (int) Math.max(drawList.size / 4, 2L * instanceCount);
            int g = (int) Math.max(drawCommands.size / (2L * LEVELS * DRAW_SIZE), 2L * groups);
            closeLists();
            allocateLists(entries, g);
            writeSet(null, null);
        }
        long totalIndices = 0;
        for (int g = 0; g < groups; g++) for (int[] idx : lods.get(g).indices) totalIndices += idx.length;
        ByteBuffer ib = memAlloc((int) Math.max(4, 4 * totalIndices));
        int[][] firstIndex = new int[Math.max(groups, 1)][LEVELS];
        int at = 0;
        for (int g = 0; g < groups; g++) {
            int[][] levels = lods.get(g).indices;
            for (int l = 0; l < levels.length; l++) {
                firstIndex[g][l] = at;
                ib.asIntBuffer().position(at).put(levels[l]);
                at += levels[l].length;
            }
        }
        vkDeviceWaitIdle(ctx.device);
        indexBuffer.close();
        indexBuffer = Buffer.deviceLocal(ctx, ib, INDEX_USAGE);
        memFree(ib);

        ByteBuffer list = drawList.map(), group = drawGroup.map(), bb = groupBounds.map(), cmds = commandTemplate.map();
        ByteBuffer gl = groupLods.map();
        for (int g = 0; g < groups; g++) {
            for (int i = groupStart[g]; i < groupStart[g + 1]; i++) {
                list.putInt(4 * i, instances[i]);
                group.putInt(4 * i, g);
            }
            for (int c = 0; c < 3; c++) {
                bb.putFloat(32 * g + 4 * c, bounds[6 * g + c]);
                bb.putFloat(32 * g + 16 + 4 * c, bounds[6 * g + 3 + c]);
            }
            MeshLod lod = lods.get(g);
            gl.putFloat(32 * g, lod.levels());
            for (int l = 1; l < LEVELS; l++) gl.putFloat(32 * g + 4 * l, l < lod.levels() ? lod.error[l] : 0f);
            for (int l = 0; l < LEVELS; l++) gl.putInt(32 * g + 16 + 4 * l, l < lod.levels() ? firstIndex[g][l] / 3 : 0);
            for (int phase = 0; phase < 2; phase++) {
                for (int l = 0; l < LEVELS; l++) {
                    int o = DRAW_SIZE * ((phase * groups + g) * LEVELS + l);
                    cmds.putInt(o, l < lod.levels() ? lod.indices[l].length : 0)
                            .putInt(o + 4, 0)
                            .putInt(o + 8, l < lod.levels() ? firstIndex[g][l] : 0)
                            .putInt(o + 12, 0)
                            .putInt(o + 16, groupStart[g] + (phase * LEVELS + l) * instanceCount);
                }
            }
        }
        drawCount = groups;
        listCount = instanceCount;
        visibleLastStale = true;
    }

    /**
     * (Re)creates the visibility and depth buffers and the Hi-Z pyramid (half the depth buffer's size, halved per
     * level down to 1 x 1) with its descriptor sets; call whenever the traced image size changes. All images stay in
     * {@code VK_IMAGE_LAYOUT_GENERAL}: written by attachments and compute, read by compute.
     */
    public void resize(int width, int height) {
        destroyImages();
        this.width = width;
        this.height = height;
        visibility = new StorageImage(ctx, width, height, VISIBILITY_FORMAT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        VkDevice device = ctx.device;
        try (MemoryStack stack = stackPush()) {
            LongBuffer lp = stack.mallocLong(1);
            depthImage = createImage(stack, DEPTH_FORMAT, width, height, 1,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            depthMemory = bind(stack, depthImage);
            VkImageViewCreateInfo view = VkImageViewCreateInfo.calloc(stack)
                    .sType$Default()
                    .image(depthImage)
                    .viewType(VK_IMAGE_VIEW_TYPE_2D)
                    .format(DEPTH_FORMAT);
            view.subresourceRange().aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).levelCount(1).layerCount(1);
            check(vkCreateImageView(device, view, null, lp), "vkCreateImageView(depth)");
            depthView = lp.get(0);

            hizWidth = Math.max(1, (width + 1) / 2);
            hizHeight = Math.max(1, (height + 1) / 2);
            hizLevels = 1;
            for (int w = hizWidth, h = hizHeight; w > 1 || h > 1; w = Math.max(1, (w + 1) / 2), h = Math.max(1, (h + 1) / 2)) hizLevels++;
            hizImage = createImage(stack, VK_FORMAT_R32_SFLOAT, hizWidth, hizHeight, hizLevels,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
            hizMemory = bind(stack, hizImage);
            view.image(hizImage).format(VK_FORMAT_R32_SFLOAT);
            view.subresourceRange().aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).baseMipLevel(0).levelCount(hizLevels);
            check(vkCreateImageView(device, view, null, lp), "vkCreateImageView(hiz)");
            hizView = lp.get(0);
            hizLevelViews = new long[hizLevels];
            for (int l = 0; l < hizLevels; l++) {
                view.subresourceRange().baseMipLevel(l).levelCount(1);
                check(vkCreateImageView(device, view, null, lp), "vkCreateImageView(hiz level)");
                hizLevelViews[l] = lp.get(0);
            }
            ctx.submitImmediate(c -> {
                try (MemoryStack s = stackPush()) {
                    VkImageMemoryBarrier.Buffer ib = VkImageMemoryBarrier.calloc(1, s);
                    ib.get(0).sType$Default().oldLayout(VK_IMAGE_LAYOUT_UNDEFINED).newLayout(VK_IMAGE_LAYOUT_GENERAL)
                            .dstAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                            .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED).dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                            .image(hizImage)
                            .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(hizLevels).layerCount(1));
                    vkCmdPipelineBarrier(c, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, null, null, ib);
                }
            });
            if (hizPool != 0) vkDestroyDescriptorPool(device, hizPool, null);
            VkDescriptorPoolSize.Buffer sizes = VkDescriptorPoolSize.calloc(2, stack);
            sizes.get(0).type(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER).descriptorCount(hizLevels);
            sizes.get(1).type(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE).descriptorCount(hizLevels);
            check(vkCreateDescriptorPool(device, VkDescriptorPoolCreateInfo.calloc(stack).sType$Default().maxSets(hizLevels)
                    .pPoolSizes(sizes), null, lp), "vkCreateDescriptorPool(hiz)");
            hizPool = lp.get(0);
            LongBuffer layouts = stack.mallocLong(hizLevels);
            for (int l = 0; l < hizLevels; l++) layouts.put(l, hizSetLayout);
            LongBuffer sets = stack.mallocLong(hizLevels);
            check(vkAllocateDescriptorSets(device, VkDescriptorSetAllocateInfo.calloc(stack).sType$Default()
                    .descriptorPool(hizPool).pSetLayouts(layouts), sets), "vkAllocateDescriptorSets(hiz)");
            hizSets = new long[hizLevels];
            for (int l = 0; l < hizLevels; l++) hizSets[l] = sets.get(l);
        }
        writeHizSets();
        visibleLastStale = true;
    }

    /**
     * Writes the Hi-Z descriptor sets, level 0 reading the depth buffer and every further level the one below, and
     * binds the whole pyramid to the culling set.
     */
    private void writeHizSets() {
        if (hizSets.length == 0) return;
        try (MemoryStack stack = stackPush()) {
            VkWriteDescriptorSet.Buffer w = VkWriteDescriptorSet.calloc(2 * hizLevels + 1, stack);
            for (int l = 0; l < hizLevels; l++) {
                w.get(2 * l).sType$Default().dstSet(hizSets[l]).dstBinding(0).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                        .pImageInfo(VkDescriptorImageInfo.calloc(1, stack).sampler(sampler)
                                .imageView(l == 0 ? depthView : hizLevelViews[l - 1]).imageLayout(VK_IMAGE_LAYOUT_GENERAL));
                w.get(2 * l + 1).sType$Default().dstSet(hizSets[l]).dstBinding(1).descriptorCount(1)
                        .descriptorType(VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
                        .pImageInfo(VkDescriptorImageInfo.calloc(1, stack).imageView(hizLevelViews[l]).imageLayout(VK_IMAGE_LAYOUT_GENERAL));
            }
            w.get(2 * hizLevels).sType$Default().dstSet(cullSet).dstBinding(8).descriptorCount(1)
                    .descriptorType(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)
                    .pImageInfo(VkDescriptorImageInfo.calloc(1, stack).sampler(sampler).imageView(hizView).imageLayout(VK_IMAGE_LAYOUT_GENERAL));
            vkUpdateDescriptorSets(ctx.device, w, null);
        }
    }

    /** Creates a 2D image with optimal tiling, without memory. */
    private long createImage(MemoryStack stack, int format, int w, int h, int levels, int usage) {
        VkImageCreateInfo info = VkImageCreateInfo.calloc(stack)
                .sType$Default()
                .imageType(VK_IMAGE_TYPE_2D)
                .format(format)
                .mipLevels(levels)
                .arrayLayers(1)
                .samples(VK_SAMPLE_COUNT_1_BIT)
                .tiling(VK_IMAGE_TILING_OPTIMAL)
                .usage(usage)
                .sharingMode(VK_SHARING_MODE_EXCLUSIVE)
                .initialLayout(VK_IMAGE_LAYOUT_UNDEFINED);
        info.extent().set(w, h, 1);
        LongBuffer lp = stack.mallocLong(1);
        check(vkCreateImage(ctx.device, info, null, lp), "vkCreateImage(raster)");
        return lp.get(0);
    }

    /** Allocates device-local memory for an image and binds it; returns the memory. */
    private long bind(MemoryStack stack, long image) {
        VkMemoryRequirements req = VkMemoryRequirements.malloc(stack);
        vkGetImageMemoryRequirements(ctx.device, image, req);
        LongBuffer lp = stack.mallocLong(1);
        check(vkAllocateMemory(ctx.device, VkMemoryAllocateInfo.calloc(stack)
                .sType$Default()
                .allocationSize(req.size())
                .memoryTypeIndex(ctx.findMemoryType(req.memoryTypeBits(), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)), null, lp),
                "vkAllocateMemory(raster)");
        check(vkBindImageMemory(ctx.device, image, lp.get(0), 0), "vkBindImageMemory(raster)");
        return lp.get(0);
    }

    /**
     * Records the pass: orders it after last frame's reads of the visibility buffer (last frame's depth is not
     * needed), culls and draws in two phases (see the class comment) and makes the result visible to ray generation.
     * {@code viewProj} is the reversed-Z view-projection of the traced image, {@code pixelsPerRadian} the screen
     * pixels per radian at the image center, for the choice of the level of detail.
     */
    public void record(VkCommandBuffer cmd, MemoryStack stack, Matrix4f viewProj, Vector3f camera, float pixelsPerRadian) {
        this.camera.set(camera);
        this.pixelsPerRadian = pixelsPerRadian;
        VkImageMemoryBarrier.Buffer barriers = VkImageMemoryBarrier.calloc(2, stack);
        barriers.get(0).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_READ_BIT)
                .dstAccessMask(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
                .oldLayout(VK_IMAGE_LAYOUT_GENERAL)
                .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .image(visibility.image)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_COLOR_BIT).levelCount(1).layerCount(1));
        barriers.get(1).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_READ_BIT)
                .dstAccessMask(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                .oldLayout(VK_IMAGE_LAYOUT_UNDEFINED)
                .newLayout(VK_IMAGE_LAYOUT_GENERAL)
                .srcQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .dstQueueFamilyIndex(VK_QUEUE_FAMILY_IGNORED)
                .image(depthImage)
                .subresourceRange(r -> r.aspectMask(VK_IMAGE_ASPECT_DEPTH_BIT).levelCount(1).layerCount(1));
        vkCmdPipelineBarrier(cmd, ctx.rtStage | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                        | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT
                        | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, null, null, barriers);
        if (drawCount == 0) {
            draw(cmd, stack, viewProj, -1);
        } else {
            prepareCulling(cmd, stack, viewProj);
            cull(cmd, stack, 0);
            draw(cmd, stack, viewProj, 0);
            if (occlusionCulling) {
                buildHiz(cmd, stack);
                cull(cmd, stack, 1);
                draw(cmd, stack, viewProj, 1);
            }
        }
        VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                ctx.rtStage | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, mb, null, null);
    }

    /**
     * Draws phase {@code phase}'s commands, the opaque groups and then the alpha-tested ones; phase 0 (or -1, nothing
     * to draw) clears the buffers first, the depth to 0, which is infinitely far.
     */
    private void draw(VkCommandBuffer cmd, MemoryStack stack, Matrix4f viewProj, int phase) {
        boolean clear = phase <= 0;
        VkRenderingAttachmentInfo.Buffer color = VkRenderingAttachmentInfo.calloc(1, stack);
        color.get(0).sType$Default()
                .imageView(visibility.view)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .loadOp(clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE)
                .clearValue().color().uint32(0, 0).uint32(1, 0).uint32(2, 0).uint32(3, 0);
        VkRenderingAttachmentInfo depth = VkRenderingAttachmentInfo.calloc(stack).sType$Default()
                .imageView(depthView)
                .imageLayout(VK_IMAGE_LAYOUT_GENERAL)
                .loadOp(clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD)
                .storeOp(VK_ATTACHMENT_STORE_OP_STORE);
        depth.clearValue().depthStencil().depth(0f);
        VkRenderingInfo info = VkRenderingInfo.calloc(stack).sType$Default()
                .layerCount(1)
                .pColorAttachments(color)
                .pDepthAttachment(depth);
        info.renderArea().extent().set(width, height);
        vkCmdBeginRendering(cmd, info);
        if (phase >= 0) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, stack.longs(set), null);
            vkCmdBindIndexBuffer(cmd, indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);
            vkCmdSetViewport(cmd, 0, VkViewport.calloc(1, stack).width(width).height(height).minDepth(0f).maxDepth(1f));
            VkRect2D.Buffer scissor = VkRect2D.calloc(1, stack);
            scissor.get(0).extent().set(width, height);
            vkCmdSetScissor(cmd, 0, scissor);
            ByteBuffer pc = stack.malloc(PUSH_SIZE);
            viewProj.get(0, pc);
            pc.putLong(64, indexBuffer.address);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, pc);
            long base = (long) phase * drawCount * LEVELS * DRAW_SIZE;
            if (opaqueGroups > 0) vkCmdDrawIndexedIndirect(cmd, drawCommands.handle, base, opaqueGroups * LEVELS, DRAW_SIZE);
            if (drawCount > opaqueGroups) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineCutout);
                vkCmdDrawIndexedIndirect(cmd, drawCommands.handle, base + (long) opaqueGroups * LEVELS * DRAW_SIZE,
                        (drawCount - opaqueGroups) * LEVELS, DRAW_SIZE);
            }
        }
        vkCmdEndRendering(cmd);
    }

    /**
     * Writes the culling parameters (the frustum planes after Gribb and Hartmann: x >= -w, x <= w, y >= -w, y <= w and
     * w >= 0 in front of the eye), resets the draw commands' instance counts and, after a new draw list or size, the
     * visibility of the last frame.
     */
    private void prepareCulling(VkCommandBuffer cmd, MemoryStack stack, Matrix4f viewProj) {
        Vector4f r0 = viewProj.getRow(0, new Vector4f()), r1 = viewProj.getRow(1, new Vector4f()), r3 = viewProj.getRow(3, new Vector4f());
        planes[0].set(r3).add(r0);
        planes[1].set(r3).sub(r0);
        planes[2].set(r3).add(r1);
        planes[3].set(r3).sub(r1);
        planes[4].set(r3);
        ByteBuffer p = params.map();
        viewProj.get(0, p);
        for (int i = 0; i < 5; i++) planes[i].get(64 + 16 * i, p);
        p.putInt(144, listCount).putInt(148, drawCount).putInt(152, hizLevels).putInt(156, occlusionCulling ? 1 : 0);
        p.putFloat(160, width).putFloat(164, height).putFloat(168, hizWidth).putFloat(172, hizHeight);
        p.putFloat(176, camera.x).putFloat(180, camera.y).putFloat(184, camera.z).putFloat(188, pixelsPerRadian / Math.max(lodPixels, 1e-3f));

        VkBufferCopy.Buffer copy = VkBufferCopy.calloc(1, stack).size(2L * LEVELS * DRAW_SIZE * drawCount);
        vkCmdCopyBuffer(cmd, commandTemplate.handle, drawCommands.handle, copy);
        if (visibleLastStale) {
            vkCmdFillBuffer(cmd, visibleLast.handle, 0, VK_WHOLE_SIZE, 0);
            visibleLastStale = false;
        }
        VkMemoryBarrier.Buffer toCompute = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, toCompute, null, null);
    }

    /** One culling phase; afterwards its draw commands are ready for the indirect draw. */
    private void cull(VkCommandBuffer cmd, MemoryStack stack, int phase) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cullLayout, 0, stack.longs(cullSet), null);
        vkCmdPushConstants(cmd, cullLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, stack.ints(phase));
        vkCmdDispatch(cmd, (listCount + 63) / 64, 1, 1);
        VkMemoryBarrier.Buffer toDraw = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, toDraw, null, null);
    }

    /** Reduces the depth of phase 0 to the Hi-Z pyramid, one dispatch per level, before phase 1 draws on it. */
    private void buildHiz(VkCommandBuffer cmd, MemoryStack stack) {
        VkMemoryBarrier.Buffer depthDone = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, depthDone, null, null);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizPipeline);
        VkMemoryBarrier.Buffer level = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT)
                .dstAccessMask(VK_ACCESS_SHADER_READ_BIT);
        int w = hizWidth, h = hizHeight;
        for (int l = 0; l < hizLevels; l++) {
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, hizLayout, 0, stack.longs(hizSets[l]), null);
            vkCmdDispatch(cmd, (w + 7) / 8, (h + 7) / 8, 1);
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, level, null, null);
            w = Math.max(1, (w + 1) / 2);
            h = Math.max(1, (h + 1) / 2);
        }
        VkMemoryBarrier.Buffer toDepth = VkMemoryBarrier.calloc(1, stack).sType$Default()
                .srcAccessMask(VK_ACCESS_SHADER_READ_BIT)
                .dstAccessMask(VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT, 0, toDepth, null, null);
    }

    /** Number of indirect draws per phase, one per mesh. */
    public int drawCount() { return drawCount; }

    /** Frees the images made by {@link #resize}. */
    private void destroyImages() {
        if (visibility == null) return;
        visibility.close();
        vkDestroyImageView(ctx.device, depthView, null);
        vkDestroyImage(ctx.device, depthImage, null);
        vkFreeMemory(ctx.device, depthMemory, null);
        for (long v : hizLevelViews) vkDestroyImageView(ctx.device, v, null);
        vkDestroyImageView(ctx.device, hizView, null);
        vkDestroyImage(ctx.device, hizImage, null);
        vkFreeMemory(ctx.device, hizMemory, null);
        visibility = null;
    }

    @Override
    public void close() {
        destroyImages();
        closeLists();
        params.close();
        indexBuffer.close();
        if (hizPool != 0) vkDestroyDescriptorPool(ctx.device, hizPool, null);
        vkDestroySampler(ctx.device, sampler, null);
        vkDestroyPipeline(ctx.device, hizPipeline, null);
        vkDestroyPipelineLayout(ctx.device, hizLayout, null);
        vkDestroyDescriptorSetLayout(ctx.device, hizSetLayout, null);
        vkDestroyPipeline(ctx.device, cullPipeline, null);
        vkDestroyPipelineLayout(ctx.device, cullLayout, null);
        vkDestroyDescriptorSetLayout(ctx.device, cullSetLayout, null);
        vkDestroyPipeline(ctx.device, pipeline, null);
        vkDestroyPipeline(ctx.device, pipelineCutout, null);
        vkDestroyPipelineLayout(ctx.device, pipelineLayout, null);
        vkDestroyDescriptorPool(ctx.device, pool, null);
        vkDestroyDescriptorSetLayout(ctx.device, setLayout, null);
    }
}
