#version 460
#include "common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;
hitAttributeEXT vec2 attribs;

layout(buffer_reference, scalar) readonly buffer Vertices { float v[]; }; // pos, normal, uv
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };

layout(set = 0, binding = 3, scalar) readonly buffer InstanceBuffer { InstanceInfo instances[]; };

const uint VERTEX_FLOATS = 8u;

vec3 fetchPosition(Vertices vb, uint idx) { uint o = VERTEX_FLOATS * idx; return vec3(vb.v[o], vb.v[o + 1], vb.v[o + 2]); }
vec3 fetchNormal(Vertices vb, uint idx)   { uint o = VERTEX_FLOATS * idx; return vec3(vb.v[o + 3], vb.v[o + 4], vb.v[o + 5]); }
vec2 fetchUV(Vertices vb, uint idx)       { uint o = VERTEX_FLOATS * idx; return vec2(vb.v[o + 6], vb.v[o + 7]); }

// Interpolates the hit attributes in world space. Shading happens in raygen.
void main() {
    InstanceInfo inst = instances[gl_InstanceCustomIndexEXT];
    Vertices vb = Vertices(inst.vertexAddress);
    Indices ib = Indices(inst.indexAddress);

    uint base = 3u * uint(gl_PrimitiveID);
    uint i0 = ib.i[base], i1 = ib.i[base + 1u], i2 = ib.i[base + 2u];
    vec3 p0 = fetchPosition(vb, i0), p1 = fetchPosition(vb, i1), p2 = fetchPosition(vb, i2);
    vec3 n0 = fetchNormal(vb, i0), n1 = fetchNormal(vb, i1), n2 = fetchNormal(vb, i2);
    vec2 t0 = fetchUV(vb, i0), t1 = fetchUV(vb, i1), t2 = fetchUV(vb, i2);

    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 nObj = n0 * bary.x + n1 * bary.y + n2 * bary.z;

    // Normals transform with the inverse transpose.
    vec3 N = normalize((nObj * gl_WorldToObjectEXT).xyz);
    vec3 Ng = normalize((cross(p1 - p0, p2 - p0) * gl_WorldToObjectEXT).xyz);
    if (dot(Ng, N) < 0.0) Ng = -Ng;

    // World-space edges for the texel density and the tangent frame.
    mat3 toWorld = mat3(gl_ObjectToWorldEXT);
    vec3 e1 = toWorld * (p1 - p0), e2 = toWorld * (p2 - p0);
    vec2 d1 = t1 - t0, d2 = t2 - t0;
    float uvArea = abs(d1.x * d2.y - d2.x * d1.y);
    float worldArea = length(cross(e1, e2));
    // Base LOD for ray cones (Akenine-Moeller et al. 2019); texture size and cone width are added in raygen.
    float density = 0.5 * log2(max(uvArea, 1e-12) / max(worldArea, 1e-12));

    vec3 T, B;
    float r = d1.x * d2.y - d2.x * d1.y;
    if (abs(r) > 1e-12) {
        T = (e1 * d2.y - e2 * d1.y) / r;
        B = (e2 * d1.x - e1 * d2.x) / r;
    } else {
        onb(N, T, B);
    }
    T = T - N * dot(N, T);
    T = dot(T, T) > 1e-12 ? normalize(T) : normalize(cross(abs(N.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0), N));
    float sign = dot(cross(N, T), B) < 0.0 ? -1.0 : 1.0;

    payload.position = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    payload.hitT = gl_HitTEXT;
    payload.normal = N;
    payload.geoNormal = Ng;
    payload.materialIndex = inst.materialIndex;
    payload.instanceIndex = gl_InstanceCustomIndexEXT;
    payload.uv = t0 * bary.x + t1 * bary.y + t2 * bary.z;
    payload.texelDensity = density;
    payload.tangentSign = sign;
    payload.tangent = T;
}
