// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/inc/lights/packed.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#ifndef LIGHTS_PACKED_HLSL
#define LIGHTS_PACKED_HLSL

struct TriangleLightPacked {
    float packed[12];
};

#endif