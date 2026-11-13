// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/ircache/clear_ircache_pool.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#include "ircache_constants.hlsl"

[[vk::binding(0)]] RWStructuredBuffer<uint> ircache_pool_buf;
[[vk::binding(1)]] RWStructuredBuffer<uint> ircache_life_buf;

[numthreads(64, 1, 1)]
void main(uint idx: SV_DispatchThreadID) {
    ircache_pool_buf[idx] = idx;
    ircache_life_buf[idx] = IRCACHE_ENTRY_LIFE_RECYCLED;
}
