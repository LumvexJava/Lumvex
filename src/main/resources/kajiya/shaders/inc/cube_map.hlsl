// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/inc/cube_map.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

static const float3x3 CUBE_MAP_FACE_ROTATIONS[6] = {
    float3x3(0,0,-1, 0,-1,0, -1,0,0),   // right
    float3x3(0,0,1, 0,-1,0, 1,0,0),     // left

    float3x3(1,0,0, 0,0,-1, 0,1,0),     // top
    float3x3(1,0,0, 0,0,1, 0,-1,0),     // bottom

    float3x3(1,0,0, 0,-1,0, 0,0,-1),    // back
    float3x3(-1,0,0, 0,-1,0, 0,0,1),    // front
};