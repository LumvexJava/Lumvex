// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/rt/shadow.rmiss.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

struct ShadowPayload {
    bool is_shadowed;
};

[shader("miss")]
void main(inout ShadowPayload payload : SV_RayPayload) {
    payload.is_shadowed = false;
}
