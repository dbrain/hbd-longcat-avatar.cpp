// Exact, geometry-preserving triangle cleanup for generated meshes.
//
// Marching-cubes grids can produce a few distinct indices at exactly the same
// interpolated position.  Those zero-area triangles do not improve the
// surface; they poison strict topology/physics importers.  This routine welds
// only bit-identical positions, removes the resulting degenerate/duplicate
// faces, and compacts unused vertices.  It deliberately never snaps nearby
// geometry, smooths, fills, or drops a real surface face.
#pragma once
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mesh_exact_clean {

struct Report {
    int64_t input_vertices = 0;
    int64_t input_faces = 0;
    int64_t welded_vertices = 0;
    int64_t dropped_degenerate_faces = 0;
    int64_t dropped_duplicate_faces = 0;
    int64_t output_vertices = 0;
    int64_t output_faces = 0;
};

struct PositionKey {
    uint32_t x, y, z;
    bool operator==(const PositionKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct PositionKeyHash {
    size_t operator()(const PositionKey& k) const noexcept {
        size_t h = k.x;
        h ^= (size_t)k.y * 0x9e3779b1u + (h << 6) + (h >> 2);
        h ^= (size_t)k.z * 0x85ebca6bu + (h << 6) + (h >> 2);
        return h;
    }
};
struct FaceKey {
    uint32_t a, b, c;
    bool operator==(const FaceKey& o) const { return a == o.a && b == o.b && c == o.c; }
};
struct FaceKeyHash {
    size_t operator()(const FaceKey& k) const noexcept {
        size_t h = k.a;
        h ^= (size_t)k.b * 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= (size_t)k.c * 0xc2b2ae3d27d4eb4full + (h << 6) + (h >> 2);
        return h;
    }
};
inline uint32_t bits(float v) {
    // Treat signed zero as the same geometric point.
    if (v == 0.f) return 0;
    uint32_t out = 0;
    std::memcpy(&out, &v, sizeof(out));
    return out;
}
inline FaceKey sorted_face(uint32_t a, uint32_t b, uint32_t c) {
    if (a > b) { const uint32_t t = a; a = b; b = t; }
    if (b > c) { const uint32_t t = b; b = c; c = t; }
    if (a > b) { const uint32_t t = a; a = b; b = t; }
    return {a, b, c};
}

inline Report clean(std::vector<float>& verts, std::vector<int64_t>& faces) {
    Report r;
    r.input_vertices = (int64_t)(verts.size() / 3);
    r.input_faces = (int64_t)(faces.size() / 3);
    if (verts.size() % 3 || faces.size() % 3 || r.input_vertices > INT32_MAX) return r;

    std::unordered_map<PositionKey, uint32_t, PositionKeyHash> positions;
    positions.reserve((size_t)r.input_vertices * 2);
    std::vector<uint32_t> remap((size_t)r.input_vertices);
    std::vector<float> welded;
    welded.reserve(verts.size());
    for (uint32_t i = 0; i < (uint32_t)r.input_vertices; ++i) {
        const PositionKey key{bits(verts[(size_t)i * 3]), bits(verts[(size_t)i * 3 + 1]),
                              bits(verts[(size_t)i * 3 + 2])};
        const auto it = positions.find(key);
        if (it != positions.end()) {
            remap[i] = it->second;
            ++r.welded_vertices;
        } else {
            const uint32_t id = (uint32_t)(welded.size() / 3);
            positions.emplace(key, id);
            remap[i] = id;
            welded.push_back(verts[(size_t)i * 3]);
            welded.push_back(verts[(size_t)i * 3 + 1]);
            welded.push_back(verts[(size_t)i * 3 + 2]);
        }
    }

    std::unordered_set<FaceKey, FaceKeyHash> seen;
    seen.reserve((size_t)r.input_faces * 2);
    std::vector<int64_t> cleaned_faces;
    cleaned_faces.reserve(faces.size());
    for (size_t f = 0; f < faces.size(); f += 3) {
        const int64_t i0 = faces[f], i1 = faces[f + 1], i2 = faces[f + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0 || i0 >= r.input_vertices || i1 >= r.input_vertices || i2 >= r.input_vertices) {
            ++r.dropped_degenerate_faces;
            continue;
        }
        const uint32_t a = remap[(size_t)i0], b = remap[(size_t)i1], c = remap[(size_t)i2];
        if (a == b || b == c || a == c) {
            ++r.dropped_degenerate_faces;
            continue;
        }
        if (!seen.emplace(sorted_face(a, b, c)).second) {
            ++r.dropped_duplicate_faces;
            continue;
        }
        cleaned_faces.push_back(a);
        cleaned_faces.push_back(b);
        cleaned_faces.push_back(c);
    }

    std::vector<uint32_t> compact(welded.size() / 3, UINT32_MAX);
    std::vector<float> out_verts;
    out_verts.reserve(welded.size());
    for (int64_t& idx : cleaned_faces) {
        uint32_t& out = compact[(size_t)idx];
        if (out == UINT32_MAX) {
            out = (uint32_t)(out_verts.size() / 3);
            out_verts.push_back(welded[(size_t)idx * 3]);
            out_verts.push_back(welded[(size_t)idx * 3 + 1]);
            out_verts.push_back(welded[(size_t)idx * 3 + 2]);
        }
        idx = out;
    }
    verts.swap(out_verts);
    faces.swap(cleaned_faces);
    r.output_vertices = (int64_t)(verts.size() / 3);
    r.output_faces = (int64_t)(faces.size() / 3);
    return r;
}

} // namespace mesh_exact_clean
