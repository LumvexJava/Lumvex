// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/wrc/bindings.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#ifndef WRC_BINDINGS_HLSL
#define WRC_BINDINGS_HLSL

#define DEFINE_WRC_BINDINGS(b0) \
    [[vk::binding(b0)]] Texture2D<float4> wrc_radiance_atlas_tex;

#endif  // WRC_BINDINGS_HLSL