// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/lut/bezold_brucke.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#include "../inc/color/math.hlsl"

#include "../inc/color/standard_observer.hlsl"
#include "../inc/color/ipt.hlsl"
#include "../inc/color/bezold_brucke.hlsl"

[[vk::binding(0)]] RWTexture1D<float2> output_tex;

[numthreads(64, 1, 1)]
void main(in uint px : SV_DispatchThreadID) {
    const float2 xy = bb_lut_coord_to_xy_white_offset((px + 0.5) / 64) + white_D65_xy;
    float3 XYZ = CIE_xyY_to_XYZ(float3(xy, 1.0));

    const float3 shifted_XYZ = bezold_brucke_shift_XYZ_brute_force(XYZ, 1.0);
    const float2 shifted_xy = normalize(CIE_XYZ_to_xyY(shifted_XYZ).xy - white_D65_xy) + white_D65_xy;

    output_tex[px] = shifted_xy - xy;
}
