// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/copy_depth_to_r.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

[[vk::binding(0)]] Texture2D<float> input_tex;
[[vk::binding(1)]] RWTexture2D<float> output_tex;


[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    output_tex[px] = input_tex[px];
}
