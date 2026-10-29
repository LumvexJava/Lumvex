#version 460
// Raster pre-pass of the hybrid renderer, vertex stage. The whole scene is one indexed multi-draw-indirect call per
// culling phase: one draw per mesh, one instance per render instance of that mesh (see RasterPrepass.java). The
// indices come from one buffer holding every mesh's indices, so the GPU's vertex cache shares the vertices of
// neighboring triangles; the vertices themselves are pulled from the same buffers the acceleration structures are
// built from.
// Buffer addresses are two 32-bit halves, so the pass also runs on GPUs without 64-bit integers in shaders.
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require

layout(buffer_reference, scalar) readonly buffer Vertices { float v[]; }; // pos, normal, uv

struct InstanceInfo {
    uvec2    vertexAddress;
    uvec2    indexAddress;
    uint     materialIndex;
    uint     flags;
};
layout(set = 0, binding = 0, scalar) readonly buffer InstanceBuffer { InstanceInfo instances[]; };
layout(set = 0, binding = 1, scalar) readonly buffer InstanceTransforms { mat4x3 objectToWorld[]; };
// Visible render instances grouped by mesh (written by raster_cull.comp); draw d covers
// drawInstance[firstInstance .. firstInstance + instanceCount).
layout(set = 0, binding = 2, std430) readonly buffer DrawList { uint drawInstance[]; };
// The first triangle of the drawn level of detail in the shared index buffer, per entry of the culled list.
layout(set = 0, binding = 3, std430) readonly buffer FirstTri { uint drawFirstTri[]; };

layout(push_constant, scalar) uniform Frame {
    mat4 viewProj;      // reversed-Z projection (1 near, 0 at infinity) times view
    uvec2 indices;      // the shared index buffer, read by the fragment stage
} pc;

const uint VERTEX_FLOATS = 8u;

// Instance and first triangle of the drawn level for the visibility buffer; the fragment stage adds the triangle.
layout(location = 0) flat out uvec2 visible;
// Texture coordinates and material, for the alpha test of the raster mode's cutouts (raster.frag).
layout(location = 1) out vec2 uv;
layout(location = 2) flat out uint material;

// Transforms one vertex of an instance: gl_InstanceIndex (which includes the draw's firstInstance) picks the
// instance from the culled list, gl_VertexIndex is the mesh's own vertex index. The camera matrices follow the
// OpenGL convention (y up), Vulkan's framebuffer rows go down like the image rows of ray generation, so y is flipped.
void main() {
    uint inst = drawInstance[gl_InstanceIndex];
    InstanceInfo info = instances[inst];
    uint idx = uint(gl_VertexIndex);
    Vertices vb = Vertices(info.vertexAddress);
    uint o = VERTEX_FLOATS * idx;
    vec3 world = objectToWorld[inst] * vec4(vb.v[o], vb.v[o + 1u], vb.v[o + 2u], 1.0);
    vec4 clip = pc.viewProj * vec4(world, 1.0);
    clip.y = -clip.y;
    gl_Position = clip;
    visible = uvec2(inst + 1u, drawFirstTri[gl_InstanceIndex]);
    uv = vec2(vb.v[o + 6u], vb.v[o + 7u]);
    material = info.materialIndex;
}
