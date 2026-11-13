// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/rtr/rtr_restir_pack_unpack.inc.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

struct RtrRestirRayOrigin {
    float3 ray_origin_eye_offset_ws;
    float roughness;
    uint frame_index_mod4;

    static RtrRestirRayOrigin from_raw(float4 raw) {
        RtrRestirRayOrigin res;
        res.ray_origin_eye_offset_ws = raw.xyz;

        float2 misc = unpack_2x16f_uint(asuint(raw.w));
        res.roughness = misc.x;
        res.frame_index_mod4 = uint(misc.y) & 3;
        return res;
    }

    float4 to_raw() {
        return float4(
            ray_origin_eye_offset_ws,
            asfloat(pack_2x16f_uint(float2(roughness, frame_index_mod4)))
        );
    }
};
