// LUMVEX_RASTER_SHADE: included by the raster-only shading pass (raster_shade.comp), which also runs on GPUs
// without ray tracing or 64-bit integers; buffer addresses are then two 32-bit halves.
#ifndef LUMVEX_RASTER_SHADE
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#else
#extension GL_EXT_buffer_reference_uvec2 : require
#endif
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require

const float PI = 3.14159265358979323846;

// Data layouts shared with the Java side.

struct HitPayload {
    vec3  position;
    float hitT;          // negative on a miss
    vec3  normal;        // world-space shading normal
    uint  materialIndex;
    vec3  geoNormal;     // world-space geometric normal, same hemisphere as `normal`
    uint  instanceIndex;
    vec2  uv;
    float texelDensity;  // 0.5 * log2(uv area / world area) of the triangle, the base LOD
    float tangentSign;   // bitangent = sign * cross(normal, tangent)
    vec3  tangent;       // world space, orthogonal to `normal`
    float pad;
};

struct Material {
    vec3  albedo;
    float roughness;
    vec3  emission;
    float metallic;
    float ior;
    float transmission;
    float checkerScale;  // > 0: world-space checker pattern on the albedo
    int   textureSet;    // -1 if untextured
    float uvScale;
    float normalStrength;
    uint  flags;         // MATERIAL_* bits
    float parallax;      // relief depth of the height map in meters, 0 = flat
};

struct InstanceInfo {
#ifndef LUMVEX_RASTER_SHADE
    uint64_t vertexAddress;
    uint64_t indexAddress;
#else
    uvec2    vertexAddress;
    uvec2    indexAddress;
#endif
    uint     materialIndex;
    uint     flags;          // bit 0: animated object
};

layout(push_constant, scalar) uniform PushConstants {
    mat4  invView;
    mat4  invProj;
    uint  frame;           // monotonic, for RNG seeds
    uint  maxBounces;
    uint  samplesPerFrame;
    uint  historyLimit;    // max samples per pixel, 0 = unlimited
    vec3  sunDirection;
    float sunIntensity;
    vec3  sunColor;
    float skyIntensity;
    uint  debugMode;
    float aperture;
    float focusDistance;
    uint  flags;           // FLAG_* below
    mat4  prevViewProj;    // previous frame's view-projection, for reprojection
} pc;

const uint FLAG_HISTORY_VALID = 1u;  // the previous history may be reused
const uint FLAG_CAMERA_MOVED  = 2u;
const uint FLAG_CACHE         = 4u;  // reserved
const uint FLAG_ANIMATED      = 8u;  // objects moved: motion vectors and influence zones
const uint FLAG_NIGHT         = 16u; // starry night sky
const uint FLAG_ENV           = 32u; // HDR environment instead of the procedural sky

// Random numbers

// PCG hash
uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Uniform random number in [0, 1)
float rnd(inout uint state) {
    state = pcgHash(state);
    return float(state >> 8) * (1.0 / 16777216.0);
}

// Sampling

// Orthonormal basis around a unit normal (Duff et al. 2017)
void onb(vec3 n, out vec3 t, out vec3 b) {
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = vec3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = vec3(c, s + n.y * n.y * a, -n.y);
}

// Cosine-weighted direction in the hemisphere around n
vec3 cosineSampleHemisphere(vec3 n, inout uint seed) {
    float r1 = rnd(seed), r2 = rnd(seed);
    float phi = 2.0 * PI * r1;
    float r = sqrt(r2);
    vec3 t, b;
    onb(n, t, b);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(0.0, 1.0 - r2)));
}

// Half vector distributed by the GGX normal distribution
vec3 sampleGGX(vec3 n, float alpha, inout uint seed) {
    float r1 = rnd(seed), r2 = rnd(seed);
    float phi = 2.0 * PI * r1;
    float cosTheta = sqrt((1.0 - r2) / (1.0 + (alpha * alpha - 1.0) * r2));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    vec3 t, b;
    onb(n, t, b);
    return normalize(t * (sinTheta * cos(phi)) + b * (sinTheta * sin(phi)) + n * cosTheta);
}

// Uniform direction in the cone around dir with cos(half angle) = cosMax
vec3 sampleCone(vec3 dir, float cosMax, inout uint seed) {
    float r1 = rnd(seed), r2 = rnd(seed);
    float cosTheta = mix(1.0, cosMax, r1);
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    float phi = 2.0 * PI * r2;
    vec3 t, b;
    onb(dir, t, b);
    return normalize(t * (sinTheta * cos(phi)) + b * (sinTheta * sin(phi)) + dir * cosTheta);
}

// Uniform point on the unit disk
vec2 sampleDisk(inout uint seed) {
    float r = sqrt(rnd(seed));
    float phi = 2.0 * PI * rnd(seed);
    return r * vec2(cos(phi), sin(phi));
}

// BRDF terms


vec3 fresnelSchlick(vec3 f0, float cosTheta) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Unpolarized Fresnel reflectance of a dielectric; 1 on total internal reflection. eta = n_i / n_t.
float fresnelDielectric(float cosI, float eta) {
    float sinT2 = eta * eta * (1.0 - cosI * cosI);
    if (sinT2 >= 1.0) return 1.0;
    float cosT = sqrt(1.0 - sinT2);
    float rs = (eta * cosI - cosT) / (eta * cosI + cosT);
    float rp = (cosI - eta * cosT) / (cosI + eta * cosT);
    return 0.5 * (rs * rs + rp * rp);
}

float ggxD(float NdotH, float alpha) {
    float a2 = alpha * alpha;
    float d = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

// Smith G1 for GGX
float smithG1(float NdotX, float alpha) {
    float a2 = alpha * alpha;
    return 2.0 * NdotX / (NdotX + sqrt(a2 + (1.0 - a2) * NdotX * NdotX));
}

// Environment

const float SUN_COS_ANGLE = 0.99955; // about 1.7 degrees angular radius

// Procedural sky: horizon-to-zenith gradient and a glow around the sun. At night a deep blue zenith over a paler
// horizon (airglow, distant towns), the moon's halo from the same glow terms, and stars from a sparse hash over
// direction cells, fading out towards the horizon.
vec3 skyColor(vec3 dir) {
    bool night = (pc.flags & FLAG_NIGHT) != 0u;
    vec3 zenith = night ? vec3(0.010, 0.016, 0.040) : vec3(0.18, 0.36, 0.78);
    vec3 horizon = night ? vec3(0.040, 0.046, 0.065) : vec3(0.80, 0.86, 0.95);
    vec3 groundCol = night ? vec3(0.012, 0.012, 0.014) : vec3(0.25, 0.23, 0.21);
    vec3 sky = dir.y >= 0.0
        ? mix(horizon, zenith, pow(dir.y, 0.45))
        : mix(horizon, groundCol, clamp(-dir.y * 5.0, 0.0, 1.0));
    float s = max(dot(dir, normalize(pc.sunDirection)), 0.0);
    sky += pc.sunColor * (pow(s, 64.0) * 0.6 + pow(s, 8.0) * 0.1) * (night ? 0.06 : 1.0);
    sky *= pc.skyIntensity;
    if (night && dir.y > 0.0) {
        vec3 d = dir * 700.0;
        vec3 cell = floor(d);
        float h = fract(sin(dot(cell, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
        if (h > 0.9982) {
            float star = smoothstep(0.45, 0.0, length(fract(d) - 0.5)) * (h - 0.9982) / 0.0018;
            sky += vec3(0.75, 0.82, 1.0) * star * 0.5 * smoothstep(0.02, 0.2, dir.y);
        }
    }
    return sky;
}

// Radiance of the visible sun disk, normalized to integrate to sunIntensity.
vec3 sunDisk(vec3 dir) {
    if (dot(dir, normalize(pc.sunDirection)) < SUN_COS_ANGLE) return vec3(0.0);
    float solidAngle = 2.0 * PI * (1.0 - SUN_COS_ANGLE);
    return pc.sunColor * pc.sunIntensity / solidAngle;
}

// ACES filmic tone curve (Narkowicz fit)
vec3 acesFilm(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
