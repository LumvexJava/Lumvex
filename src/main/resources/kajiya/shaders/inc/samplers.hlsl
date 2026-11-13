// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/inc/samplers.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#ifndef SAMPLERS_HLSL
#define SAMPLERS_HLSL

[[vk::binding(32)]] SamplerState sampler_lnc;
[[vk::binding(33)]] SamplerState sampler_llr;
[[vk::binding(34)]] SamplerState sampler_nnc;
[[vk::binding(35)]] SamplerState sampler_llc;

#endif