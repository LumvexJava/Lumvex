// Sparse virtual texturing, see VirtualTextures.java.
//
// Texture sets are split into VT_TILE x VT_TILE tiles per mip. vtPage maps a tile to its own slot and optionally
// to a similar stand-in tile of the same mip, usable beyond a certain distance. vtRequest records the smallest
// distance each tile was needed at. Entries are pool << 30 | (slot + 1); pool 0 holds mips 0-1, pool 1 the rest.

const uint VT_TILE = 128u;
const uint VT_BORDER = 4u;
const uint VT_SLOT = VT_TILE + 2u * VT_BORDER;

layout(set = 0, binding = 16) uniform sampler2D vtAtlas[6];                        // [pool * 3 + channel]: albedo, normal xy, ORM
layout(set = 0, binding = 17, std430) readonly buffer VtPageTable { uvec4 vtPage[]; };  // own, stand-in, stand-in distance, upload frame
layout(set = 0, binding = 18, std430) buffer VtFeedback { uint vtRequest[]; };          // min distance in cm
layout(set = 0, binding = 19, std430) readonly buffer VtSets { uvec4 vtHeader; uvec4 vtSet[]; }; // header: perRow, atlas, frame; per set: size, mips, firstTile
const float VT_FADE_FRAMES = 8.0;   // frames over which a new tile fades in

uint vtTilesPerSide(uint size, uint mip) { return max(1u, (size >> mip) / VT_TILE); }

// Global tile index: the set's first tile, plus the tiles of all finer mips, plus the row-major index.
uint vtTileId(uint set, uint mip, uvec2 tile) {
    uvec4 s = vtSet[set];
    uint id = s.z;
    for (uint m = 0u; m < mip; ++m) { uint t = vtTilesPerSide(s.x, m); id += t * t; }
    return id + tile.y * vtTilesPerSide(s.x, mip) + tile.x;
}

// Mip level from the triangle's texel density, the texture size and the ray cone footprint.
float vtLod(uint set, float texelDensity, float uvScale, float coneWidth, float cosTheta) {
    float size = float(vtSet[set].x);
    return texelDensity + log2(size * uvScale) + log2(max(coneWidth, 1e-6) / max(cosTheta, 0.05));
}

uvec2 vtTileOf(uint size, uint mip, vec2 uvw) {
    uint dim = size >> mip;
    return min(uvec2(uvw * float(dim)) / VT_TILE, uvec2(vtTilesPerSide(size, mip) - 1u));
}

// Records that the tile is needed at this distance; the smallest distance in cm wins.
void vtRequestTile(uint set, vec2 uv, uint mip, float distance) {
    uvec4 s = vtSet[set];
    mip = min(mip, s.y - 1u);
    uint id = vtTileId(set, mip, vtTileOf(s.x, mip, fract(uv)));
    uint cm = uint(clamp(distance * 100.0, 0.0, 4.0e9));
    // Most pixels report no smaller distance than the stored one; checking first avoids heavy atomic contention.
    if (vtRequest[id] > cm) atomicMin(vtRequest[id], cm);
}

struct VtSample {
    vec4  albedo;      // linear (the atlases are sRGB)
    vec3  normal;      // tangent space, unit length
    vec3  orm;         // ao, roughness, metallic
    float mipWanted;
    float mipUsed;
    float standIn;     // 0: own tile, 1: stand-in, in between: cross-fade
};

// Reads all channels of a resident tile at a texel position inside the tile.
void vtFetch(uint entry, vec2 local, out vec4 albedo, out vec2 nxy, out vec3 orm) {
    uint pool = entry >> 30u;
    uint slot = (entry & 0x3FFFFFFFu) - 1u;
    uint perRow = vtHeader.x;
    vec2 atlasPx = vec2(float(slot % perRow), float(slot / perRow)) * float(VT_SLOT) + float(VT_BORDER) + local;
    vec2 auv = atlasPx / float(vtHeader.y);
    albedo = textureLod(vtAtlas[pool * 3u], auv, 0.0);
    nxy = textureLod(vtAtlas[pool * 3u + 1u], auv, 0.0).xy * 2.0 - 1.0;
    orm = textureLod(vtAtlas[pool * 3u + 2u], auv, 0.0).xyz;
}

// Samples a texture set. The LOD is rounded stochastically, which averages to trilinear filtering over frames.
// Falls back to the next coarser resident mip; the distance selects stand-in tiles.
VtSample vtSample(uint set, vec2 uv, float lod, float distance, inout uint seed, bool request) {
    uvec4 s = vtSet[set];
    float maxMip = float(s.y - 1u);
    lod = clamp(lod, 0.0, maxMip);
    uint wanted = uint(min(floor(lod + rnd(seed)), maxMip));
    vec2 uvw = fract(uv);
    if (request) vtRequestTile(set, uvw, wanted, distance);

    uint mip = wanted;
    uvec4 e = uvec4(0u);
    for (; mip < s.y; ++mip) {
        e = vtPage[vtTileId(set, mip, vtTileOf(s.x, mip, uvw))];
        if (e.x != 0u || e.y != 0u) break;
    }
    mip = min(mip, s.y - 1u);   // the coarsest mip is pinned, so the loop always finds a slot

    uint dim = s.x >> mip;
    vec2 local = uvw * float(dim) - vec2(vtTileOf(s.x, mip, uvw) * VT_TILE);

    // Beyond standInDist the stand-in is good enough. The last 25% before it cross-fade, so the real tile blends
    // in smoothly as the camera approaches.
    float standInDist = uintBitsToFloat(e.z);
    float w = 0.0;
    if (e.y != 0u) w = e.x == 0u ? 1.0 : smoothstep(0.75 * standInDist, standInDist, distance);

    VtSample r;
    vec2 nxy;
    if (w < 1.0) vtFetch(e.x, local, r.albedo, nxy, r.orm);
    if (w > 0.0) {
        vec4 a2; vec2 n2; vec3 o2;
        vtFetch(e.y, local, a2, n2, o2);
        if (w >= 1.0) { r.albedo = a2; nxy = n2; r.orm = o2; }
        else { r.albedo = mix(r.albedo, a2, w); nxy = mix(nxy, n2, w); r.orm = mix(r.orm, o2, w); }
    }
    // Blend a tile that just arrived with the next coarser resident level to avoid popping.
    float age = float(vtHeader.z - e.w);
    if (e.x != 0u && w < 1.0 && age < VT_FADE_FRAMES && mip + 1u < s.y) {
        uint cmip = mip + 1u;
        uvec4 ce = uvec4(0u);
        for (; cmip < s.y; ++cmip) {
            ce = vtPage[vtTileId(set, cmip, vtTileOf(s.x, cmip, uvw))];
            if (ce.x != 0u) break;
        }
        if (cmip < s.y && ce.x != 0u) {
            vec2 clocal = uvw * float(s.x >> cmip) - vec2(vtTileOf(s.x, cmip, uvw) * VT_TILE);
            vec4 a2; vec2 n2; vec3 o2;
            vtFetch(ce.x, clocal, a2, n2, o2);
            float k = 1.0 - (age + 1.0) / (VT_FADE_FRAMES + 1.0);   // weight of the coarse level
            r.albedo = mix(r.albedo, a2, k); nxy = mix(nxy, n2, k); r.orm = mix(r.orm, o2, k);
        }
    }
    r.normal = vec3(nxy, sqrt(max(1.0 - dot(nxy, nxy), 0.0)));   // BC5 only stores x and y
    r.mipWanted = float(wanted);
    r.mipUsed = float(mip);
    r.standIn = w;
    return r;
}
