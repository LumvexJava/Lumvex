// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/inc/bindless_textures.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#ifndef BINDLESS_TEXTURES_HLSL
#define BINDLESS_TEXTURES_HLSL

[[vk::binding(2, 1)]] StructuredBuffer<float4> bindless_texture_sizes;
[[vk::binding(3, 1)]] Texture2D bindless_textures[];

// Pre-integrated FG texture for the GGX BRDF
static const uint BINDLESS_LUT_BRDF_FG = 0;

static const uint BINDLESS_LUT_BLUE_NOISE_256_LDR_RGBA_0 = 1;

static const uint BINDLESS_LUT_BEZOLD_BRUCKE = 2;

#endif
