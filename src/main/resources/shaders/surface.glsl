// Hit attributes of a triangle in world space. Shared by the closest-hit shader and by raygen's raster primary
// surface (see rasterPrimary), so both produce exactly the same payload. Include after common.glsl.

layout(buffer_reference, scalar) readonly buffer Vertices { float v[]; }; // pos, normal, uv
layout(buffer_reference, scalar) readonly buffer Indices { uint i[]; };

const uint VERTEX_FLOATS = 8u;

// Position, normal and texture coordinates of a vertex.
vec3 fetchPosition(Vertices vb, uint idx) { uint o = VERTEX_FLOATS * idx; return vec3(vb.v[o], vb.v[o + 1], vb.v[o + 2]); }
vec3 fetchNormal(Vertices vb, uint idx)   { uint o = VERTEX_FLOATS * idx; return vec3(vb.v[o + 3], vb.v[o + 4], vb.v[o + 5]); }
vec2 fetchUV(Vertices vb, uint idx)       { uint o = VERTEX_FLOATS * idx; return vec2(vb.v[o + 6], vb.v[o + 7]); }

struct Triangle {
    vec3 p0, p1, p2;   // object space
    vec3 n0, n1, n2;
    vec2 t0, t1, t2;
};

Triangle fetchTriangleVertices(InstanceInfo inst, uvec3 v);

// The triangle of the instance's mesh with this index.
Triangle fetchTriangle(InstanceInfo inst, uint primitive) {
    Indices ib = Indices(inst.indexAddress);
    uint base = 3u * primitive;
    return fetchTriangleVertices(inst, uvec3(ib.i[base], ib.i[base + 1u], ib.i[base + 2u]));
}

// The triangle of three vertices of the instance's mesh, e.g. from the raster pass's visibility buffer.
Triangle fetchTriangleVertices(InstanceInfo inst, uvec3 v) {
    Vertices vb = Vertices(inst.vertexAddress);
    uint i0 = v.x, i1 = v.y, i2 = v.z;
    Triangle t;
    t.p0 = fetchPosition(vb, i0); t.p1 = fetchPosition(vb, i1); t.p2 = fetchPosition(vb, i2);
    t.n0 = fetchNormal(vb, i0);   t.n1 = fetchNormal(vb, i1);   t.n2 = fetchNormal(vb, i2);
    t.t0 = fetchUV(vb, i0);       t.t1 = fetchUV(vb, i1);       t.t2 = fetchUV(vb, i2);
    return t;
}

// Interpolates the hit attributes. `attribs` are the barycentrics of p1 and p2, `toWorld` the object-to-world
// rotation and scale, `normalMatrix` its inverse transpose. The world-space edges give the texel density, the base
// LOD for ray cones (Akenine-Moeller et al. 2019; texture size and cone width are added in raygen), and the tangent
// frame.
HitPayload surfaceHit(Triangle tri, vec2 attribs, mat3 toWorld, mat3 normalMatrix, uint materialIndex,
                      uint instanceIndex, vec3 position, float hitT) {
    vec3 bary = vec3(1.0 - attribs.x - attribs.y, attribs.x, attribs.y);
    vec3 nObj = tri.n0 * bary.x + tri.n1 * bary.y + tri.n2 * bary.z;

    vec3 N = normalize(normalMatrix * nObj);
    vec3 Ng = normalize(normalMatrix * cross(tri.p1 - tri.p0, tri.p2 - tri.p0));
    if (dot(Ng, N) < 0.0) Ng = -Ng;

    vec3 e1 = toWorld * (tri.p1 - tri.p0), e2 = toWorld * (tri.p2 - tri.p0);
    vec2 d1 = tri.t1 - tri.t0, d2 = tri.t2 - tri.t0;
    float uvArea = abs(d1.x * d2.y - d2.x * d1.y);
    float worldArea = length(cross(e1, e2));
    float density = 0.5 * log2(max(uvArea, 1e-12) / max(worldArea, 1e-12));

    vec3 T, B;
    float r = d1.x * d2.y - d2.x * d1.y;
    if (abs(r) > 1e-12) {
        T = (e1 * d2.y - e2 * d1.y) / r;
        B = (e2 * d1.x - e1 * d2.x) / r;
    } else {
        onb(N, T, B);
    }
    T = T - N * dot(N, T);
    T = dot(T, T) > 1e-12 ? normalize(T) : normalize(cross(abs(N.y) < 0.99 ? vec3(0, 1, 0) : vec3(1, 0, 0), N));
    float sign = dot(cross(N, T), B) < 0.0 ? -1.0 : 1.0;

    HitPayload h;
    h.position = position;
    h.hitT = hitT;
    h.normal = N;
    h.geoNormal = Ng;
    h.materialIndex = materialIndex;
    h.instanceIndex = instanceIndex;
    h.uv = tri.t0 * bary.x + tri.t1 * bary.y + tri.t2 * bary.z;
    h.texelDensity = density;
    h.tangentSign = sign;
    h.tangent = T;
    h.pad = 0.0;
    return h;
}
