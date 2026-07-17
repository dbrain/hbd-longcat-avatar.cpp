// p3sam_sonata.hpp — native C++ (CPU fp32) port of the Sonata PTv3 ENCODER used by P3-SAM
// (XPart/.../sonata/model.py, enc_mode=True). Mirrors the fp32 non-flash oracle:
//   embedding (Linear 9->48 + LN + GELU)
//   serialize (z/z-trans/hilbert/hilbert-trans, shuffle_orders=False) + sparsify
//   enc stage s in 0..4: [GridPooling if s>0] + enc_depths[s] Blocks
//     Block: feat += cpe(feat)               cpe = SubMConv3d(C,3) -> Linear(C,C) -> LN(C)
//            feat += attn(norm1(feat))        SerializedAttention (patch K=min(N,1024))
//            feat += mlp(norm2(feat))         Linear(C,4C)->GELU->Linear(4C,C)
//   unpool concat all 5 stage feats -> 1232 ; mlp(1232->512->512->512) ; feat[inverse]->[Nin,512]
// LayerNorm eps=1e-5, GELU exact-erf. SubMConv weight = spconv [Cout,3,3,3,Cin].
//
// All feature math depends only on grid_coord (coord floats are never read by the encoder),
// so we carry grid_coord (int) per stage and recompute serialization from it.
#pragma once
#include "p3sam_heads.hpp"        // Weights, linear, gelu
#include "p3sam_serialize.hpp"    // serialize_order
#include <cstdint>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <string>

namespace p3sam {

inline void layernorm(std::vector<float>& x, int M, int C, const NpyArray& w, const NpyArray& b,
                      float eps = 1e-5f) {
    const float* g = w.f32(); const float* bb = b.f32();
    for (int m = 0; m < M; m++) {
        float* r = x.data() + (size_t)m * C;
        double mean = 0; for (int c = 0; c < C; c++) mean += r[c]; mean /= C;
        double var = 0; for (int c = 0; c < C; c++) { double d = r[c] - mean; var += d * d; } var /= C;
        float inv = 1.0f / std::sqrt((float)var + eps);
        for (int c = 0; c < C; c++) r[c] = ((r[c] - (float)mean) * inv) * g[c] + bb[c];
    }
}

// ---- SubMConv3d (submanifold, K=3) on grid_coord (N,3). weight = spconv [Cout,3,3,3,Cin].
inline std::vector<float> submconv3d(const int32_t* grid, int N, const float* feat, int Cin,
                                     const float* w, const float* bias, int Cout, int sign = +1) {
    std::unordered_map<int64_t, int> idx; idx.reserve(N * 2);
    auto key = [](int32_t x, int32_t y, int32_t z) {
        return ((int64_t)(x & 0x1FFFFF) << 42) | ((int64_t)(y & 0x1FFFFF) << 21) | (int64_t)(z & 0x1FFFFF);
    };
    for (int i = 0; i < N; i++) idx[key(grid[i*3], grid[i*3+1], grid[i*3+2])] = i;
    std::vector<float> out((size_t)N * Cout);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 256)
    #endif
    for (int n = 0; n < N; n++) {
        std::vector<double> acc(Cout, 0.0);
        int32_t px = grid[n*3], py = grid[n*3+1], pz = grid[n*3+2];
        for (int v = 0; v < 27; v++) {
            int k0 = v / 9, k1 = (v / 3) % 3, k2 = v % 3;
            int dx = (k0 - 1) * sign, dy = (k1 - 1) * sign, dz = (k2 - 1) * sign;
            auto it = idx.find(key(px + dx, py + dy, pz + dz));
            if (it == idx.end()) continue;
            const float* fj = feat + (size_t)it->second * Cin;
            for (int ci = 0; ci < Cin; ci++) {
                double fv = fj[ci];
                // w[co,k0,k1,k2,ci] = co*27*Cin + v*Cin + ci
                size_t base = (size_t)v * Cin + ci;
                for (int co = 0; co < Cout; co++) acc[co] += fv * w[(size_t)co * 27 * Cin + base];
            }
        }
        for (int co = 0; co < Cout; co++) out[(size_t)n*Cout+co] = (float)acc[co] + bias[co];
    }
    return out;
}

// ---- serialized patch attention (non-flash fp32 oracle). order_id in 0..3, H heads.
inline std::vector<float> serial_attention(Weights& W, const std::string& bn, int order_id,
                                           const int32_t* grid, int N, const float* feat, int C, int H) {
    const int Kmax = 1024;
    int K = std::min(N, Kmax);
    std::vector<int64_t> ser_inv;
    auto ser_ord = serialize_order(grid, N, order_id, &ser_inv);   // length N
    // padded length + pad index array (structure get_padding_and_inverse, single batch)
    int Npatch = (N + K - 1) / K, Npad = Npatch * K;
    std::vector<int64_t> pad(Npad);
    for (int i = 0; i < Npad; i++) pad[i] = (i < N) ? i : 0;
    int rem = N % K;
    if (N > K && rem != 0) {
        // wrap last partial patch: pad[Npad-K+rem : Npad] = pad[Npad-2K+rem : Npad-K]
        for (int j = 0; j < K - rem; j++)
            pad[Npad - K + rem + j] = (Npad - 2 * K + rem + j);
    }
    // qkv = Linear(feat) then gather by order_full = ser_ord[pad]
    std::vector<float> qkv; linear(feat, N, C, W.get(bn + ".qkv.weight"), W.get(bn + ".qkv.bias"), qkv, 3 * C);
    const int Ch = C / H;
    const float scale = 1.0f / std::sqrt((float)Ch);
    std::vector<float> fpad((size_t)Npad * C, 0.0f);
    // process each patch independently
    #ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1)
    #endif
    for (int p = 0; p < Npatch; p++) {
        std::vector<float> attn(K);
        // gather qkv rows for this patch: row r -> point ser_ord[pad[p*K+r]]
        // q,k,v layout in qkv row: [0:C]=q [C:2C]=k [2C:3C]=v ; heads split contiguous Ch
        for (int h = 0; h < H; h++) {
            for (int r = 0; r < K; r++) {
                int pt_q = (int)ser_ord[pad[(size_t)p*K + r]];
                const float* qrow = qkv.data() + (size_t)pt_q * 3 * C + 0 * C + h * Ch;
                // attn[c] = softmax_c( scale * q . k_c )
                double mx = -1e30;
                for (int c = 0; c < K; c++) {
                    int pt_k = (int)ser_ord[pad[(size_t)p*K + c]];
                    const float* krow = qkv.data() + (size_t)pt_k * 3 * C + 1 * C + h * Ch;
                    double dot = 0; for (int d = 0; d < Ch; d++) dot += (double)qrow[d] * krow[d];
                    dot *= scale; attn[c] = (float)dot; if (dot > mx) mx = dot;
                }
                double sum = 0;
                for (int c = 0; c < K; c++) { double e = std::exp((double)attn[c] - mx); attn[c] = (float)e; sum += e; }
                float invs = (float)(1.0 / sum);
                // out = Σ_c attn_c * v_c
                float* o = fpad.data() + (size_t)(p*K + r) * C + h * Ch;
                for (int c = 0; c < K; c++) {
                    int pt_v = (int)ser_ord[pad[(size_t)p*K + c]];
                    const float* vrow = qkv.data() + (size_t)pt_v * 3 * C + 2 * C + h * Ch;
                    float a = attn[c] * invs;
                    for (int d = 0; d < Ch; d++) o[d] += a * vrow[d];
                }
            }
        }
    }
    // feat_out[n] = fpad[ ser_inv[n] ] ; then proj
    std::vector<float> gathered((size_t)N * C);
    for (int n = 0; n < N; n++)
        std::copy_n(fpad.data() + (size_t)ser_inv[n] * C, C, gathered.data() + (size_t)n * C);
    std::vector<float> out; linear(gathered.data(), N, C, W.get(bn + ".proj.weight"), W.get(bn + ".proj.bias"), out, C);
    return out;
}

// ---- one Block ----
inline void block(Weights& W, const std::string& bn, int order_id, const int32_t* grid,
                  int N, std::vector<float>& feat, int C, int H) {
    // cpe: SubMConv3d -> Linear -> LN ; feat += cpe
    {
        const NpyArray& cw = W.get(bn + ".cpe.0.weight");   // [Cout,3,3,3,Cin]
        auto c0 = submconv3d(grid, N, feat.data(), C, cw.f32(), W.get(bn + ".cpe.0.bias").f32(), C);
        std::vector<float> c1; linear(c0.data(), N, C, W.get(bn + ".cpe.1.weight"), W.get(bn + ".cpe.1.bias"), c1, C);
        layernorm(c1, N, C, W.get(bn + ".cpe.2.weight"), W.get(bn + ".cpe.2.bias"));
        for (size_t i = 0; i < feat.size(); i++) feat[i] += c1[i];
    }
    // attn: feat += attn(norm1(feat))
    {
        std::vector<float> x = feat;
        layernorm(x, N, C, W.get(bn + ".norm1.0.weight"), W.get(bn + ".norm1.0.bias"));
        auto a = serial_attention(W, bn + ".attn", order_id, grid, N, x.data(), C, H);
        for (size_t i = 0; i < feat.size(); i++) feat[i] += a[i];
    }
    // mlp: feat += mlp(norm2(feat))
    {
        std::vector<float> x = feat;
        layernorm(x, N, C, W.get(bn + ".norm2.0.weight"), W.get(bn + ".norm2.0.bias"));
        std::vector<float> h1; linear(x.data(), N, C, W.get(bn + ".mlp.0.fc1.weight"), W.get(bn + ".mlp.0.fc1.bias"), h1, 4 * C);
        for (auto& v : h1) v = gelu(v);
        std::vector<float> h2; linear(h1.data(), N, 4 * C, W.get(bn + ".mlp.0.fc2.weight"), W.get(bn + ".mlp.0.fc2.bias"), h2, C);
        for (size_t i = 0; i < feat.size(); i++) feat[i] += h2[i];
    }
}

// ---- GridPooling: returns new grid (Mnew,3), cluster (N,), pooled+norm+gelu feat (Mnew,Cout)
struct PoolOut { std::vector<int32_t> grid; std::vector<int64_t> cluster; std::vector<float> feat; int M; };
inline PoolOut grid_pool(Weights& W, const std::string& dn, const int32_t* grid, int N,
                         const float* feat, int Cin, int Cout, int stride) {
    // new grid = unique sorted rows of (grid//stride); cluster = row id
    std::vector<std::array<int32_t,3>> g(N);
    for (int n = 0; n < N; n++) {
        auto fl = [&](int32_t v){ return (int32_t)std::floor((double)v / stride); }; // trunc==floor for >=0
        g[n] = { grid[n*3]/stride, grid[n*3+1]/stride, grid[n*3+2]/stride };
        (void)fl;
    }
    std::vector<int> ord(N); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b){
        if (g[a][0]!=g[b][0]) return g[a][0]<g[b][0];
        if (g[a][1]!=g[b][1]) return g[a][1]<g[b][1];
        return g[a][2]<g[b][2];
    });
    PoolOut P; P.cluster.assign(N, 0);
    std::vector<int> rep;   // first member of each cluster (in sorted order)
    int cid = -1;
    std::array<int32_t,3> prev{INT32_MIN,INT32_MIN,INT32_MIN};
    for (int s = 0; s < N; s++) {
        int n = ord[s];
        if (s == 0 || g[n] != prev) { cid++; P.grid.push_back(g[n][0]); P.grid.push_back(g[n][1]); P.grid.push_back(g[n][2]); prev = g[n]; }
        P.cluster[n] = cid;
    }
    P.M = cid + 1;
    // proj(feat) then segment max per cluster
    std::vector<float> proj; linear(feat, N, Cin, W.get(dn + ".proj.weight"), W.get(dn + ".proj.bias"), proj, Cout);
    P.feat.assign((size_t)P.M * Cout, -1e30f);
    for (int n = 0; n < N; n++) {
        int64_t c = P.cluster[n];
        const float* s = proj.data() + (size_t)n * Cout;
        float* d = P.feat.data() + (size_t)c * Cout;
        for (int j = 0; j < Cout; j++) if (s[j] > d[j]) d[j] = s[j];
    }
    // norm + gelu
    layernorm(P.feat, P.M, Cout, W.get(dn + ".norm.0.weight"), W.get(dn + ".norm.0.bias"));
    for (auto& v : P.feat) v = gelu(v);
    return P;
}

struct StageData { std::vector<int32_t> grid; std::vector<int64_t> cluster; std::vector<float> feat; int N, C; };

// Full encoder. Inputs: tf_grid (M0,3 int32) and tf_feat (M0,9 float). tf_inverse (Nin,) maps
// final point.inverse. Returns feats_N (Nin,512). Also fills per-stage feats for validation.
inline std::vector<float> run_sonata(Weights& W, const int32_t* tf_grid, const float* tf_feat,
                                     int M0, const int64_t* inverse, int Nin,
                                     std::vector<StageData>* stages_out = nullptr) {
    const int enc_ch[5]   = {48, 96, 192, 384, 512};
    const int enc_head[5] = {3, 6, 12, 24, 32};
    const int enc_dep[5]  = {3, 3, 3, 12, 3};
    const int stride = 2;
    // embedding: Linear(9->48)+LN+GELU
    std::vector<float> feat;
    linear(tf_feat, M0, 9, W.get("sonata.embedding.stem.linear.weight"),
           W.get("sonata.embedding.stem.linear.bias"), feat, 48);
    layernorm(feat, M0, 48, W.get("sonata.embedding.stem.norm.weight"), W.get("sonata.embedding.stem.norm.bias"));
    for (auto& v : feat) v = gelu(v);

    std::vector<StageData> S(5);
    std::vector<int32_t> grid(tf_grid, tf_grid + (size_t)M0 * 3);
    int N = M0;
    for (int s = 0; s < 5; s++) {
        int C = enc_ch[s];
        if (s > 0) {
            std::string dn = "sonata.enc.enc" + std::to_string(s) + ".down";
            PoolOut P = grid_pool(W, dn, grid.data(), N, feat.data(), enc_ch[s-1], C, stride);
            grid = P.grid; feat = P.feat; N = P.M;
            S[s].cluster = P.cluster;   // maps stage(s-1) points -> stage s clusters
        }
        for (int b = 0; b < enc_dep[s]; b++) {
            std::string bn = "sonata.enc.enc" + std::to_string(s) + ".block" + std::to_string(b);
            int order_id = b % 4;
            block(W, bn, order_id, grid.data(), N, feat, C, enc_head[s]);
        }
        S[s].grid = grid; S[s].feat = feat; S[s].N = N; S[s].C = C;
    }
    if (stages_out) *stages_out = S;    // snapshot raw per-stage feats BEFORE unpool overwrites them
    // unpool concat: start at stage4, walk down concatenating child[cluster] to parent
    std::vector<float> cur = S[4].feat; int curN = S[4].N, curC = S[4].C;
    for (int s = 4; s >= 1; s--) {
        // parent = S[s-1]; child = cur (stage s); cluster S[s].cluster maps parent point -> child idx
        int pN = S[s-1].N, pC = S[s-1].C;
        const std::vector<int64_t>& cl = S[s].cluster;
        int newC = pC + curC;
        std::vector<float> nf((size_t)pN * newC);
        for (int n = 0; n < pN; n++) {
            float* d = nf.data() + (size_t)n * newC;
            const float* pf = S[s-1].feat.data() + (size_t)n * pC;
            for (int c = 0; c < pC; c++) d[c] = pf[c];
            const float* cf = cur.data() + (size_t)cl[n] * curC;
            for (int c = 0; c < curC; c++) d[pC + c] = cf[c];
        }
        S[s-1].feat = nf; curC = newC; curN = pN; cur = nf;
        // note S[s-1].feat now holds the growing concat (used by next iter as parent feat)
    }
    // cur = feat_1232 at stage0 resolution (M0, 1232)
    // head mlp 1232->512->512->512 (keys mlp.{0,2,4})
    std::vector<float> m0; linear(cur.data(), M0, curC, W.get("mlp.0.weight"), W.get("mlp.0.bias"), m0, 512);
    for (auto& v : m0) v = gelu(v);
    std::vector<float> m2; linear(m0.data(), M0, 512, W.get("mlp.2.weight"), W.get("mlp.2.bias"), m2, 512);
    for (auto& v : m2) v = gelu(v);
    std::vector<float> mlp512; linear(m2.data(), M0, 512, W.get("mlp.4.weight"), W.get("mlp.4.bias"), mlp512, 512);
    // feats_N = mlp512[inverse]
    std::vector<float> feats_N((size_t)Nin * 512);
    for (int n = 0; n < Nin; n++)
        std::copy_n(mlp512.data() + (size_t)inverse[n] * 512, 512, feats_N.data() + (size_t)n * 512);
    return feats_N;
}

} // namespace p3sam
