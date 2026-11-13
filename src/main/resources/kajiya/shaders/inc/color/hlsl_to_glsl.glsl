// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/inc/color/hlsl_to_glsl.glsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#define float2 vec2
#define float3 vec3
#define float4 vec4
#define static
#define mul(A, B) ((A) * (B))
#define atan2(y, x) atan(y, x)
#define lerp(a, b, t) mix(a, b, t)
#define saturate(a) clamp(a, 0.0, 1.0)
#define frac(x) fract(x)

#define float3x3(a, b, c, d, e, f, g, h, i) transpose(mat3(a, b, c, d, e, f, g, h, i))
#define float2x2(a, b, c, d) transpose(mat2(a, b, c, d))
