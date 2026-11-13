//DEV: HLSL port of rust-shaders/src/rev_blur.rs
#include "inc/samplers.hlsl"

[[vk::binding(0)]] Texture2D<float4> input_tail_tex;
[[vk::binding(1)]] Texture2D<float4> input_tex;
[[vk::binding(2)]] RWTexture2D<float4> output_tex;
[[vk::binding(3)]] cbuffer _ {
    uint output_extent_x;
    uint output_extent_y;
    float self_weight;
};

[numthreads(8, 8, 1)]
void main(uint2 px: SV_DispatchThreadID) {
    float4 pyramid_col = input_tail_tex[px];
    float2 inv_size = 1.0 / float2(output_extent_x, output_extent_y);

    static const int K = 1;
    float4 self_col = 0;
    for (int y = -K; y <= K; ++y) {
        for (int x = -K; x <= K; ++x) {
            float2 uv = (float2(px) + 0.5 + float2(x, y)) * inv_size;
            self_col += input_tex.SampleLevel(sampler_lnc, uv, 0);
        }
    }
    self_col /= (2 * K + 1) * (2 * K + 1);

    static const float exponential_falloff = 0.6;
    output_tex[px] = lerp(self_col, pyramid_col, self_weight * exponential_falloff);
}
