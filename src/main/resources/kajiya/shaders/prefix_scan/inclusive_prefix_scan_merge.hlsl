// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/prefix_scan/inclusive_prefix_scan_merge.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#define THREAD_GROUP_SIZE 512
#define SEGMENT_SIZE (THREAD_GROUP_SIZE * 2)

[[vk::binding(0)]] RWByteAddressBuffer inout_buf;
[[vk::binding(1)]] ByteAddressBuffer segment_sum_buf;

uint2 load_input2(uint idx, uint segment) {
    const uint2 internal_sum = inout_buf.Load2(sizeof(uint) * (idx + segment * SEGMENT_SIZE));
    const uint prev_segment_sum = select(segment == 0, 0, segment_sum_buf.Load(sizeof(uint) * (segment - 1)));

    return internal_sum + prev_segment_sum;
}

void store_output2(uint idx, uint segment, uint2 val) {
    inout_buf.Store2(sizeof(uint) * (idx + segment * SEGMENT_SIZE), val);
}

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint idx: SV_GroupThreadID, uint segment: SV_GroupID) {
    store_output2(idx * 2, segment, load_input2(idx * 2, segment));
}
