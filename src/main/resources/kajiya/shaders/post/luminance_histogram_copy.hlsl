// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/post/luminance_histogram_copy.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

[[vk::binding(0)]] StructuredBuffer<uint> src_buffer;
[[vk::binding(1)]] RWStructuredBuffer<uint> dst_buffer;

[numthreads(64, 1, 1)]
void main(uint bin: SV_DispatchThreadID) {
    dst_buffer[bin] = src_buffer[bin];
}
