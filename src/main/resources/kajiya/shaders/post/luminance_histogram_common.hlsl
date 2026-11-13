// From kajiya: https://github.com/EmbarkStudios/kajiya (assets/shaders/post/luminance_histogram_common.hlsl)
// Copyright (c) 2019 Embark Studios - licensed under MIT OR Apache-2.0,
// see licenses/kajiya/ and THIRD_PARTY_NOTICES.md in the Lumvex repository.

#ifndef LUMINANCE_HISTOGRAM_COMMON_HLSL
#define LUMINANCE_HISTOGRAM_COMMON_HLSL

static const uint LUMINANCE_HISTOGRAM_BIN_COUNT = 256;
static const float LUMINANCE_HISTOGRAM_MIN_LOG2 = -16.0;
static const float LUMINANCE_HISTOGRAM_MAX_LOG2 = 16.0;

#endif  // LUMINANCE_HISTOGRAM_COMMON_HLSL
