// World-space radiance cache, shared by the path tracer (raygen.rgen) and its resolve pass (cache_resolve.comp).
// The layout must match dev.lumvex.vk.RadianceCache.
//
// Training paths add samples to a cell's accumulator; once per frame the resolve pass blends them into the cell's
// resolved value with a bounded history and frees cells nobody trained for a while. Paths only ever read the
// resolved value, so the cache follows changing light within a few frames instead of freezing (after SHaRC, NVIDIA's
// spatially hashed radiance cache).
#ifndef LUMVEX_CACHE_GLSL
#define LUMVEX_CACHE_GLSL

const uint CACHE_SIZE = 1u << 23;       // slots; must match RadianceCache.SLOTS
const uint CACHE_AGE_MASK = 0xffffu;    // cacheMeta: frames since the cell was last trained
const uint CACHE_DYNAMIC = 0x80000000u; // cacheMeta: trained inside the influence zone of a moving object this frame

#define RADIANCE_CACHE_BUFFER(SET, BINDING)                                                            \
    layout(set = SET, binding = BINDING, std430) buffer RadianceCache {                               \
        uint  cacheKeys[CACHE_SIZE];     /* hash check value, 0 for a free slot */                    \
        uint  cacheMeta[CACHE_SIZE];     /* age and CACHE_DYNAMIC */                                   \
        vec4  cacheResolved[CACHE_SIZE]; /* rgb mean outgoing radiance / albedo, a weight in samples */ \
        float cacheAccum[];              /* per slot: rgb sum and sample count of this frame */        \
    };

#endif
