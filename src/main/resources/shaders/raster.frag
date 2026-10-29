#version 460
// Raster pre-pass of the hybrid renderer, fragment stage: writes the visibility buffer, (instance + 1, the three
// vertex indices of the triangle) of the nearest surface at each pixel center; 0 is sky. Ray generation rebuilds the
// hit from it instead of tracing the primary ray, see rasterPrimary() in raygen.rgen. The vertex indices rather than
// a triangle number, because the pass is built for levels of detail, whose triangles the mesh itself does not have;
// they still use the mesh's vertices.
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_buffer_reference_uvec2 : require
#extension GL_EXT_scalar_block_layout : require

layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };

layout(push_constant, scalar) uniform Frame {
    mat4 viewProj;
    uvec2 indices;      // the shared index buffer of all levels of all meshes, as two 32-bit halves
} pc;

layout(location = 0) flat in uvec2 visible;   // instance + 1, first triangle of the drawn level
layout(location = 1) in vec2 uv;
layout(location = 2) flat in uint material;
layout(location = 0) out uvec4 visibility;

#ifdef LUMVEX_ALPHA_TEST
// Alpha test of the raster mode's cutouts (leaves): the texture set's 1-bit alpha mask, as in alpha.rahit.
struct Material {
    vec3 albedo; float roughness; vec3 emission; float metallic; float ior; float transmission; float checkerScale;
    int textureSet; float uvScale; float normalStrength; uint flags; float parallax;
};
layout(set = 0, binding = 4, scalar) readonly buffer MaterialBuffer { Material materials[]; };
layout(set = 0, binding = 5, std430) readonly buffer AlphaMasks { uint alphaWords[]; };
const uint MATERIAL_ALPHA_TEST = 2u;

// Whether the alpha mask is clear at the fragment's texture coordinates.
bool cutAway() {
    Material m = materials[material];
    if ((m.flags & MATERIAL_ALPHA_TEST) == 0u || m.textureSet < 0) return false;
    uint offset = alphaWords[2u * uint(m.textureSet)];
    uint side = alphaWords[2u * uint(m.textureSet) + 1u];
    if (offset == 0xFFFFFFFFu || side == 0u) return false;
    uvec2 p = min(uvec2(fract(uv * m.uvScale) * float(side)), uvec2(side - 1u));
    uint bit = p.y * side + p.x;
    return (alphaWords[offset + (bit >> 5u)] & (1u << (bit & 31u))) == 0u;
}
#endif

// Writes the instance and the vertex indices of the triangle, found from the drawn level's first triangle plus
// gl_PrimitiveID, which counts the triangles of this instance's draw.
void main() {
#ifdef LUMVEX_ALPHA_TEST
    if (cutAway()) discard;
#endif
    uint base = 3u * (visible.y + uint(gl_PrimitiveID));
    Indices ib = Indices(pc.indices);
    visibility = uvec4(visible.x, ib.i[base], ib.i[base + 1u], ib.i[base + 2u]);
}
