// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/post/luminance_histogram_clear.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

[[vk::binding(0)]] RWStructuredBuffer<uint> output_buffer;

[numthreads(256, 1, 1)]
void main(uint bin: SV_DispatchThreadID) {
    output_buffer[bin] = 0;
}
