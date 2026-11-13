// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/inc/bindless.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#include "mesh.hlsl"

[[vk::binding(0, 1)]] StructuredBuffer<Mesh> meshes;
[[vk::binding(1, 1)]] ByteAddressBuffer vertices;
#include "bindless_textures.hlsl"
