#version 460
#include "common.glsl"

layout(location = 1) rayPayloadInEXT bool shadowed;

void main() {
    shadowed = false;
}
