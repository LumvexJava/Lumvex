// Adapted from kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/ibl/ibl_cube.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.
// Modifications for Lumvex: Copyright (c) 2026 LumvexJava (MIT).

#include "../inc/atmosphere.hlsl"
#include "../inc/sun.hlsl"
#include "../inc/cube_map.hlsl"
#include "../inc/samplers.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tex;
[[vk::binding(1)]] RWTexture2DArray<float4> output_tex;
[[vk::binding(2)]] cbuffer _ {
    uint cube_width;
};

// https://learnopengl.com/PBR/IBL/Diffuse-irradiance
//DEV: Lumvex panorama mapping, u = 0.5 + atan2(x, -z) / 2pi, v = acos(y) / pi (see render.scene.Environment)
float2 direction_to_spherical_map_uv(float3 v) {
    return float2(0.5 + atan2(v.x, -v.z) * 0.15915494, acos(clamp(v.y, -1.0, 1.0)) * 0.31830989);
}

[numthreads(8, 8, 1)]
void main(in uint3 px : SV_DispatchThreadID) {
    uint face = px.z;
    float2 uv = (px.xy + 0.5) / cube_width;
    float3 dir = normalize(mul(CUBE_MAP_FACE_ROTATIONS[face], float3(uv * 2 - 1, -1.0)));

    uv = direction_to_spherical_map_uv(dir);

    float3 output = input_tex.SampleLevel(sampler_llr, uv, 0).rgb;

    output_tex[px] = float4(frame_constants.pre_exposure * output, 1);
}
