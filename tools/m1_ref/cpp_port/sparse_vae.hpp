// Sparse-VAE upsample primitives for the Pixal3D M3a port (FlexiDualGridVaeDecoder.upsample).
//
// Correctness-first, fp32, IMPERATIVE host orchestration — the natural shape for a
// data-dependent sparse decoder (coords grow per-stage from subdiv decisions, so a
// static graph can't express it). The one net-new op (submanifold sparse conv) is the
// spike kernel (sparse_subm_conv_cpu.hpp CPU mirror + sparse_subm_conv.cu CUDA, both
// validated ~1e-7 on these exact layers). The dense per-token ops (LayerNorm32 / SiLU /
// Linear) are already ggml-proven in M1/M2 — here they're trivial host fp32 loops that
// match torch exactly. Coord-growth (SparseChannel2Spatial) is host index arithmetic.
//
// Weight conventions (from stage3a_capture.py export):
//   conv weight  : spike [V=27, Cin, Cout]  (transpose(1,2,3,4,0).reshape; validated vs flex_gemm)
//   Linear weight: torch [out, in]           (y = x @ W^T + b)
//   norm weight  : [C] gamma, [C] beta (affine) or none (non-affine)
//   coords       : int32 [N,4] = (batch, c1, c2, c3)
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cmath>
#include <vector>
#include <unordered_map>
#include "../../sparse_spike/sparse_subm_conv_cpu.hpp"   // launch_subm_conv_cpu

#ifdef M3A_USE_CUDA
#include <cuda_runtime.h>
// declared in sparse_subm_conv.cu (compiled + linked by build.sh cuda)
extern "C" void launch_subm_conv_cuda(
    const float* d_feats, const uint32_t* d_nmap, const float* d_weight,
    const float* d_bias, float* d_out, int N, int Cin, int Cout, int V, cudaStream_t stream);
#endif

namespace svae {

static const uint32_t SENTINEL = 0xFFFFFFFFu;

// ---- rulebook (neighbor map), z-major k offsets — matches sparse_conv.cpp / spike ----
static inline int64_t coord_key(int32_t b, int32_t c1, int32_t c2, int32_t c3) {
    auto m = [](int32_t v) { return (int64_t)(v & 0xFFFFF); };
    return (((((int64_t)b << 20) | m(c1)) << 20 | m(c2)) << 20) | m(c3);
}

// nmap[N*27]: for each voxel, the input index at coord+offset[v] (z-major), or SENTINEL.
inline std::vector<uint32_t> build_nmap(const int32_t* coords, int N) {
    // kernel offsets z-major: v = kz*9+ky*3+kx, offset (kz-1, ky-1, kx-1) on (c1,c2,c3)
    int offs[27][3];
    int idx = 0;
    for (int kz = 0; kz < 3; kz++)
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++) { offs[idx][0]=kz-1; offs[idx][1]=ky-1; offs[idx][2]=kx-1; idx++; }
    std::unordered_map<int64_t,int> index;
    index.reserve((size_t)N * 2);
    for (int i = 0; i < N; i++)
        index[coord_key(coords[i*4+0], coords[i*4+1], coords[i*4+2], coords[i*4+3])] = i;
    std::vector<uint32_t> nmap((size_t)N * 27, SENTINEL);
    for (int i = 0; i < N; i++) {
        int32_t b=coords[i*4+0], c1=coords[i*4+1], c2=coords[i*4+2], c3=coords[i*4+3];
        for (int v = 0; v < 27; v++) {
            auto it = index.find(coord_key(b, c1+offs[v][0], c2+offs[v][1], c3+offs[v][2]));
            if (it != index.end()) nmap[(size_t)i*27 + v] = (uint32_t)it->second;
        }
    }
    return nmap;
}

// ---- submanifold sparse conv: dispatch to spike CPU mirror or CUDA kernel ----
inline std::vector<float> subm_conv(const std::vector<float>& feats, const std::vector<uint32_t>& nmap,
                                    const std::vector<float>& weight, const std::vector<float>& bias,
                                    int N, int Cin, int Cout, bool use_cuda) {
    const int V = 27;
#ifdef M3A_USE_CUDA
    if (use_cuda) {
        float *d_feats, *d_weight, *d_bias, *d_out; uint32_t* d_nmap;
        cudaMalloc(&d_feats, (size_t)N*Cin*4);
        cudaMalloc(&d_weight, (size_t)V*Cin*Cout*4);
        cudaMalloc(&d_bias, (size_t)Cout*4);
        cudaMalloc(&d_out, (size_t)N*Cout*4);
        cudaMalloc(&d_nmap, (size_t)N*V*4);
        cudaMemcpy(d_feats, feats.data(), (size_t)N*Cin*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weight, weight.data(), (size_t)V*Cin*Cout*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, bias.data(), (size_t)Cout*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_nmap, nmap.data(), (size_t)N*V*4, cudaMemcpyHostToDevice);
        launch_subm_conv_cuda(d_feats, d_nmap, d_weight, d_bias, d_out, N, Cin, Cout, V, 0);
        cudaDeviceSynchronize();
        std::vector<float> out((size_t)N*Cout);
        cudaMemcpy(out.data(), d_out, (size_t)N*Cout*4, cudaMemcpyDeviceToHost);
        cudaFree(d_feats); cudaFree(d_weight); cudaFree(d_bias); cudaFree(d_out); cudaFree(d_nmap);
        return out;
    }
#endif
    (void)use_cuda;
    // OpenMP-parallel CPU conv — bit-identical math to launch_subm_conv_cpu (spike mirror):
    // out[n,co] = bias[co] + sum_v sum_ci feats[nmap[n,v],ci]*weight[v,ci,co]. Rows independent.
    std::vector<float> out((size_t)N * Cout);
    const float* f = feats.data(); const uint32_t* nm = nmap.data();
    const float* w = weight.data(); const float* b = bias.empty() ? nullptr : bias.data();
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float* orow = out.data() + (size_t)n * Cout;
        for (int co = 0; co < Cout; co++) orow[co] = b ? b[co] : 0.f;
        for (int v = 0; v < V; v++) {
            uint32_t j = nm[(size_t)n * V + v];
            if (j == SENTINEL) continue;
            const float* fj = f + (size_t)j * Cin;
            const float* wv = w + (size_t)v * Cin * Cout;   // [Cin,Cout]
            for (int ci = 0; ci < Cin; ci++) {
                float fc = fj[ci];
                const float* wr = wv + (size_t)ci * Cout;
                for (int co = 0; co < Cout; co++) orow[co] += fc * wr[co];
            }
        }
    }
    return out;
}

// ---- dense per-token ops (fp32, match torch) ----
// torch nn.Linear: weight [out,in], y[n,o] = sum_i x[n,i]*W[o,i] + b[o].
inline std::vector<float> linear(const std::vector<float>& x, const std::vector<float>& W,
                                 const std::vector<float>& b, int N, int Cin, int Cout) {
    std::vector<float> y((size_t)N * Cout);
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const float* xr = x.data() + (size_t)n * Cin;
        float* yr = y.data() + (size_t)n * Cout;
        for (int o = 0; o < Cout; o++) {
            const float* wr = W.data() + (size_t)o * Cin;
            float acc = b.empty() ? 0.f : b[o];
            for (int i = 0; i < Cin; i++) acc += xr[i] * wr[i];
            yr[o] = acc;
        }
    }
    return y;
}

// LayerNorm over C (population mean/var), then *gamma+beta if affine. eps=1e-6.
inline void layernorm(std::vector<float>& x, const std::vector<float>& gamma,
                      const std::vector<float>& beta, int N, int C, float eps) {
    bool affine = !gamma.empty();
#pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float* r = x.data() + (size_t)n * C;
        double mean = 0; for (int c = 0; c < C; c++) mean += r[c]; mean /= C;
        double var = 0; for (int c = 0; c < C; c++) { double d = r[c]-mean; var += d*d; } var /= C;
        double inv = 1.0 / std::sqrt(var + eps);
        for (int c = 0; c < C; c++) {
            double v = (r[c] - mean) * inv;
            if (affine) v = v * gamma[c] + beta[c];
            r[c] = (float)v;
        }
    }
}

inline void silu_inplace(std::vector<float>& x) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)x.size(); i++) { float v = x[i]; x[i] = v / (1.f + std::exp(-v)); }
}

inline void add_inplace(std::vector<float>& a, const std::vector<float>& b) {
    for (size_t i = 0; i < a.size(); i++) a[i] += b[i];
}

// ---- SparseChannel2Spatial coord growth (factor 2) ----
// Given coords[N,4] and sub[N*8] bool, produce new_coords[M,4] + parent[M] + child[M],
// iterating n ascending then child ascending (== torch nonzero() order).
struct C2SIdx { std::vector<int32_t> coords; std::vector<int> parent; std::vector<int> child; int M; };
inline C2SIdx c2s_grow(const int32_t* coords, int N, const uint8_t* sub) {
    C2SIdx r; r.M = 0;
    for (int n = 0; n < N; n++) for (int k = 0; k < 8; k++) if (sub[(size_t)n*8 + k]) r.M++;
    r.coords.resize((size_t)r.M * 4); r.parent.resize(r.M); r.child.resize(r.M);
    int m = 0;
    for (int n = 0; n < N; n++) {
        int32_t b=coords[n*4+0], c1=coords[n*4+1], c2=coords[n*4+2], c3=coords[n*4+3];
        for (int k = 0; k < 8; k++) {
            if (!sub[(size_t)n*8 + k]) continue;
            r.coords[(size_t)m*4+0] = b;
            r.coords[(size_t)m*4+1] = 2*c1 + ( k       & 1);   // i=0 -> +subidx%2     on c1
            r.coords[(size_t)m*4+2] = 2*c2 + ((k >> 1) & 1);   // i=1 -> +(subidx//2)%2 on c2
            r.coords[(size_t)m*4+3] = 2*c3 + ((k >> 2) & 1);   // i=2 -> +(subidx//4)%2 on c3
            r.parent[m] = n; r.child[m] = k; m++;
        }
    }
    return r;
}

// gather children (child-major): src[N, 8*per_child] reshaped [N*8, per_child];
// out[m] = src[parent[m]*8 + child[m]] = src[parent[m]][child[m]*per_child : +per_child].
inline std::vector<float> gather_children(const std::vector<float>& src, const C2SIdx& ix,
                                          int per_child) {
    std::vector<float> out((size_t)ix.M * per_child);
#pragma omp parallel for schedule(static)
    for (int m = 0; m < ix.M; m++) {
        const float* sp = src.data() + ((size_t)ix.parent[m]*8 + ix.child[m]) * per_child;
        float* op = out.data() + (size_t)m * per_child;
        for (int c = 0; c < per_child; c++) op[c] = sp[c];
    }
    return out;
}

// repeat_interleave along channel dim: x[M, c] -> [M, c*factor], out[:, i*factor+r] = x[:, i].
inline std::vector<float> repeat_interleave(const std::vector<float>& x, int M, int c, int factor) {
    std::vector<float> out((size_t)M * c * factor);
#pragma omp parallel for schedule(static)
    for (int m = 0; m < M; m++) {
        const float* xr = x.data() + (size_t)m * c;
        float* op = out.data() + (size_t)m * c * factor;
        for (int i = 0; i < c; i++) for (int r = 0; r < factor; r++) op[i*factor + r] = xr[i];
    }
    return out;
}

// ---- O-Voxel flexible_dual_grid_to_mesh (eval, split_weight provided) ----
// Per voxel: vertex = (coord + dual_vertex)*voxel_size + aabb0. For each of 3 edge dirs,
// if intersected[n][e]: the 4 neighbor voxels (coord+offset) form a quad iff all 4 exist
// (hashmap lookup); split into 2 triangles by the diagonal with larger quad_lerp product.
// Iterates n asc, dir asc (== torch edge_neighbor_voxel[intersected_flag] row-major order)
// so face vertex-indices + order match the real o_voxel output exactly.
struct Mesh { std::vector<float> verts; std::vector<int64_t> faces; int N, F; };

inline Mesh flexible_dual_grid_to_mesh(const int32_t* coords, int N,
                                       const float* dual_vertices, const int8_t* intersected,
                                       const float* quad_lerp, int grid_size) {
    // 3 edge dirs x 4 neighbor offsets (x,y,z) — verbatim from o_voxel
    static const int OFF[3][4][3] = {
        {{0,0,0},{0,0,1},{0,1,1},{0,1,0}},   // x-axis
        {{0,0,0},{1,0,0},{1,0,1},{0,0,1}},   // y-axis
        {{0,0,0},{0,1,0},{1,1,0},{1,0,0}},   // z-axis
    };
    static const int SPLIT1[6] = {0,1,2,0,2,3};
    static const int SPLIT2[6] = {0,1,3,3,1,2};
    const float voxel_size = 1.0f / (float)grid_size, aabb0 = -0.5f;

    Mesh m; m.N = N;
    m.verts.resize((size_t)N * 3);
    for (int i = 0; i < N; i++)
        for (int d = 0; d < 3; d++)
            m.verts[(size_t)i*3 + d] = ((float)coords[i*4+1+d] + dual_vertices[(size_t)i*3+d]) * voxel_size + aabb0;

    // hashmap coord(x,y,z) -> voxel index
    std::unordered_map<int64_t,int> index;
    index.reserve((size_t)N * 2);
    for (int i = 0; i < N; i++)
        index[coord_key(0, coords[i*4+1], coords[i*4+2], coords[i*4+3])] = i;

    // assemble quads in torch order, split into triangles
    m.faces.reserve((size_t)N * 6);
    int q[4];
    for (int n = 0; n < N; n++) {
        int32_t x = coords[n*4+1], y = coords[n*4+2], z = coords[n*4+3];
        for (int e = 0; e < 3; e++) {
            if (!intersected[(size_t)n*3 + e]) continue;
            bool ok = true;
            for (int k = 0; k < 4; k++) {
                auto it = index.find(coord_key(0, x + OFF[e][k][0], y + OFF[e][k][1], z + OFF[e][k][2]));
                if (it == index.end()) { ok = false; break; }
                q[k] = it->second;
            }
            if (!ok) continue;
            float w02 = quad_lerp[q[0]] * quad_lerp[q[2]];
            float w13 = quad_lerp[q[1]] * quad_lerp[q[3]];
            const int* sp = (w02 > w13) ? SPLIT1 : SPLIT2;
            for (int t = 0; t < 6; t++) m.faces.push_back((int64_t)q[sp[t]]);
        }
    }
    m.F = (int)(m.faces.size() / 3);
    return m;
}

// write a binary little-endian PLY (verts [N*3] f32, faces [F*3] int) — loadable in
// MeshLab/Blender. A tangible geometry artifact for the untextured mesh.
inline bool write_ply(const char* path, const std::vector<float>& verts,
                      const std::vector<int64_t>& faces) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    int64_t V = (int64_t)verts.size() / 3, F = (int64_t)faces.size() / 3;
    std::fprintf(f, "ply\nformat binary_little_endian 1.0\n");
    std::fprintf(f, "element vertex %lld\nproperty float x\nproperty float y\nproperty float z\n", (long long)V);
    std::fprintf(f, "element face %lld\nproperty list uchar int vertex_indices\n", (long long)F);
    std::fprintf(f, "end_header\n");
    std::fwrite(verts.data(), sizeof(float), verts.size(), f);
    for (int64_t i = 0; i < F; i++) {
        unsigned char n = 3; std::fwrite(&n, 1, 1, f);
        int idx[3] = {(int)faces[i*3], (int)faces[i*3+1], (int)faces[i*3+2]};
        std::fwrite(idx, sizeof(int), 3, f);
    }
    std::fclose(f);
    return true;
}

} // namespace svae
