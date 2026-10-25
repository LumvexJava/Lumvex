#version 460
#include "common.glsl"

layout(location = 0) rayPayloadInEXT HitPayload payload;

void main() {
    payload.hitT = -1.0; // the sky is evaluated in raygen
}
