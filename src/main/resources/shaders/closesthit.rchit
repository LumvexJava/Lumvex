#version 460
#include "common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec2 attribs;

 // pos, normal, uv
layout(set = 0, binding = 3, scalar) readonly buffer InstanceBuffer { InstanceInfo instances[]; };

#include "surface.glsl"

// Interpolates the hit attributes in world space; normals transform with the inverse transpose. Shading happens in
// raygen.
void main() {
    InstanceInfo inst = instances[gl_InstanceCustomIndexEXT];
    Triangle tri = fetchTriangle(inst, uint(gl_PrimitiveID));
    mat3 normalMatrix = transpose(mat3(gl_WorldToObjectEXT));
    payload = surfaceHit(tri, attribs, mat3(gl_ObjectToWorldEXT), normalMatrix, inst.materialIndex,
                         gl_InstanceCustomIndexEXT, gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT, gl_HitTEXT);
}
