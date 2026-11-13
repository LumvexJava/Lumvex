// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/rt/gbuffer.rmiss.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#include "../inc/pack_unpack.hlsl"
#include "../inc/rt.hlsl"

[shader("miss")]
void main(inout GbufferRayPayload payload: SV_RayPayload) {}
