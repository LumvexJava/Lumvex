// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/inc/uv.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#ifndef UV_HLSL
#define UV_HLSL

float2 get_uv(int2 pix, float4 texSize) {
	return (float2(pix) + 0.5) * texSize.zw;
}

float2 get_uv(float2 pix, float4 texSize) {
	return (pix + 0.5) * texSize.zw;
}

float2 cs_to_uv(float2 cs) {
	return cs * float2(0.5, -0.5) + float2(0.5, 0.5);
}

float2 uv_to_cs(float2 uv) {
	return (uv - 0.5.xx) * float2(2, -2);
}

#endif
