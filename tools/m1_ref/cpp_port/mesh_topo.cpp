// mesh_topo — topology audit for GLB meshes.
//
// `open` / `nonmanifold` are computed after exact position welding, which is
// the delivery invariant for a GLB with chart-split UV vertices.  A previous
// version used a lossy integer hash as the *identity* of a quantised position;
// unrelated vertices could collide and be reported as fabricated non-manifold
// fans.  Near-coincidence is useful geometric evidence, but it is not topology
// and is therefore reported separately.
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>
#include "glb_reader.hpp"

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
struct CellKey {
    int64_t x, y, z;
    bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
};
struct CellKeyHash {
    size_t operator()(const CellKey& k) const noexcept {
        size_t h = (size_t)k.x;
        h ^= (size_t)k.y * 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
        h ^= (size_t)k.z * 0xc2b2ae3d27d4eb4full + (h << 6) + (h >> 2);
        return h;
    }
};

static uint32_t fbits(float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    return bits;
}
static uint64_t edge_key(uint32_t a, uint32_t b) {
    if (a > b) { uint32_t t = a; a = b; b = t; }
    return (uint64_t(a) << 32) | uint64_t(b);
}

static void edge_stats(const std::vector<int64_t>& faces, const std::vector<uint32_t>& canon,
                       int64_t& open, int64_t& nonmanifold, int64_t& degenerate) {
    std::unordered_map<uint64_t, int> edge_use;
    edge_use.reserve(faces.size());
    degenerate = 0;
    for (size_t f = 0; f + 2 < faces.size(); f += 3) {
        const int64_t i0 = faces[f], i1 = faces[f + 1], i2 = faces[f + 2];
        if (i0 < 0 || i1 < 0 || i2 < 0 || (size_t)i0 >= canon.size() ||
            (size_t)i1 >= canon.size() || (size_t)i2 >= canon.size()) {
            ++degenerate;
            continue;
        }
        const uint32_t tri[3] = {canon[(size_t)i0], canon[(size_t)i1], canon[(size_t)i2]};
        if (tri[0] == tri[1] || tri[1] == tri[2] || tri[2] == tri[0]) {
            ++degenerate;
            continue;
        }
        ++edge_use[edge_key(tri[0], tri[1])];
        ++edge_use[edge_key(tri[1], tri[2])];
        ++edge_use[edge_key(tri[2], tri[0])];
    }
    open = 0;
    nonmanifold = 0;
    for (const auto& e : edge_use) {
        if (e.second == 1) ++open;
        else if (e.second > 2) ++nonmanifold;
    }
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        std::printf("usage: mesh_topo <mesh.glb> [near_eps_frac=1e-5]\n");
        return 1;
    }
    const double eps_frac = argc == 3 ? std::strtod(argv[2], nullptr) : 1e-5;
    if (!(eps_frac > 0.0)) {
        std::fprintf(stderr, "near_eps_frac must be positive\n");
        return 1;
    }
    glb::Mesh mesh;
    if (!glb::read_glb(argv[1], mesh)) {
        std::fprintf(stderr, "read fail: %s\n", argv[1]);
        return 1;
    }
    const size_t V = mesh.verts.size() / 3;
    const size_t F = mesh.faces.size() / 3;
    if (V > UINT32_MAX) {
        std::fprintf(stderr, "mesh has too many vertices for topology audit\n");
        return 1;
    }

    std::unordered_map<PositionKey, uint32_t, PositionKeyHash> exact_cells;
    exact_cells.reserve(V * 2);
    std::vector<uint32_t> exact(V);
    uint32_t exact_vertices = 0;
    for (uint32_t i = 0; i < V; ++i) {
        const PositionKey key{fbits(mesh.verts[(size_t)i * 3]), fbits(mesh.verts[(size_t)i * 3 + 1]),
                              fbits(mesh.verts[(size_t)i * 3 + 2])};
        const auto it = exact_cells.find(key);
        if (it == exact_cells.end()) {
            exact_cells.emplace(key, exact_vertices);
            exact[i] = exact_vertices++;
        } else {
            exact[i] = it->second;
        }
    }
    int64_t open = 0, nonmanifold = 0, degenerate = 0;
    edge_stats(mesh.faces, exact, open, nonmanifold, degenerate);

    // A non-gating proximity signal: it detects a very-close double surface or
    // near duplicate without reclassifying it as a shared topological vertex.
    double mn[3] = {1e30, 1e30, 1e30}, mx[3] = {-1e30, -1e30, -1e30};
    for (size_t i = 0; i < V; ++i)
        for (int d = 0; d < 3; ++d) {
            const double v = mesh.verts[i * 3 + (size_t)d];
            mn[d] = std::fmin(mn[d], v);
            mx[d] = std::fmax(mx[d], v);
        }
    const double dx = mx[0] - mn[0], dy = mx[1] - mn[1], dz = mx[2] - mn[2];
    const double diag = std::sqrt(dx * dx + dy * dy + dz * dz);
    const double inv = diag > 0.0 ? 1.0 / (diag * eps_frac) : 1.0;
    std::unordered_map<CellKey, uint32_t, CellKeyHash> near_cells;
    near_cells.reserve(V * 2);
    uint64_t near_coincident = 0;
    for (uint32_t i = 0; i < V; ++i) {
        const CellKey key{(int64_t)std::llround(mesh.verts[(size_t)i * 3] * inv),
                          (int64_t)std::llround(mesh.verts[(size_t)i * 3 + 1] * inv),
                          (int64_t)std::llround(mesh.verts[(size_t)i * 3 + 2] * inv)};
        const auto it = near_cells.find(key);
        if (it == near_cells.end()) near_cells.emplace(key, i);
        else ++near_coincident;
    }
    std::printf("%s: V=%zu exactV=%u F=%zu open=%lld nonmanifold=%lld degenerate=%lld "
                "near_coincident=%llu (near_eps=%.1e*diag)\n",
                argv[1], V, exact_vertices, F, (long long)open, (long long)nonmanifold,
                (long long)degenerate, (unsigned long long)near_coincident, eps_frac);
    return 0;
}
