// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/rtdgi/candidate_ray_dir.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

float3 rtdgi_candidate_ray_dir(uint2 px, float3x3 tangent_to_world) {
    float2 urand = blue_noise_for_pixel(px, frame_constants.frame_index).xy;

    #if 0
        const uint salt = 35110969;
        uint rng = hash3(uint3(px + salt, frame_constants.frame_index));
        urand += float2(
            uint_to_u01_float(hash1_mut(rng)),
            uint_to_u01_float(hash1_mut(rng))
        ) / 256.0;

        urand = frac(urand);
    #endif

#if 1
    float3 wi = uniform_sample_hemisphere(urand);
#else
    DiffuseBrdf brdf;
    brdf.albedo = 1;
    float3 wi = brdf.sample(float3(0, 0, 1), urand).wi;
#endif

    return mul(tangent_to_world, wi);
}
