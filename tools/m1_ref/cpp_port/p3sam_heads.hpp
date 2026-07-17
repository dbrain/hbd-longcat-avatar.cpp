// p3sam_heads.hpp — native C++ port of the P3-SAM seg/iou heads (model.forward in
// auto_mask_no_postprocess.py L55-111). Per prompt point K and per surface point N:
//   feats_seg = [feat(512), point(3), prompt(3)]                       (518)
//   seg_mlp_{1,2,3}: 518->512->512->1  GELU                 -> stage1 logits [N,K,3]
//   stage2: feats_seg_2 = [feats_seg, mask1](521)
//           g = seg_s2_mlp_g(521->256->256->256); g = max_N(g) bcast    (256)
//           feats_seg_3 = [g, feats_seg_2](777)
//           seg_s2_mlp_{1,2,3}: 777->256->256->1            -> stage2 logits [N,K,3]
//   iou: feats_iou = [g, feats_seg, mask2](777); iou_mlp(777->256->256->256);
//        g2 = max_N(.); iou_mlp_out(256->256->256->3); sigmoid          [K,3]
// nn.GELU() default = exact erf. Linear y = x@W^T + b (W is [out,in]).
#pragma once
#include "../../sparse_spike/npy.hpp"
#include <cmath>
#include <string>
#include <vector>
#include <unordered_map>
#include <stdexcept>

namespace p3sam {

struct Weights {
    std::string dir;
    std::unordered_map<std::string, NpyArray> cache;
    explicit Weights(std::string d) : dir(std::move(d)) {}
    const NpyArray& get(const std::string& key) {
        auto it = cache.find(key);
        if (it != cache.end()) return it->second;
        std::string san; for (char c : key) { if (c == '.') san += "__"; else san += c; }  // '.' -> '__'
        std::string path = dir + "/" + san + ".npy";
        FILE* fp = fopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("p3sam weight not found: " + path);
        fclose(fp);
        cache[key] = npy_load(path);
        return cache[key];
    }
};

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752440f)); }

// out[M,O] = in[M,I] @ W[O,I]^T + b[O]
inline void linear(const float* in, int M, int I, const NpyArray& W, const NpyArray& b,
                   std::vector<float>& out, int O) {
    const float* w = W.f32(); const float* bb = b.f32();
    out.resize((size_t)M * O);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int m = 0; m < M; m++) {
        const float* x = in + (size_t)m * I;
        float* y = out.data() + (size_t)m * O;
        for (int o = 0; o < O; o++) {
            const float* wr = w + (size_t)o * I;
            float acc = bb[o];
            for (int i = 0; i < I; i++) acc += x[i] * wr[i];
            y[o] = acc;
        }
    }
}

// 3-layer MLP (Linear,GELU,Linear,GELU,Linear) with keys <name>.{0,2,4}.{weight,bias}
inline std::vector<float> mlp3(Weights& W, const std::string& name, const float* in,
                               int M, int I, int& O_out) {
    const NpyArray& w0 = W.get(name + ".0.weight"); int H0 = (int)w0.shape[0];
    const NpyArray& w2 = W.get(name + ".2.weight"); int H2 = (int)w2.shape[0];
    const NpyArray& w4 = W.get(name + ".4.weight"); int H4 = (int)w4.shape[0];
    std::vector<float> a, c, e;
    linear(in, M, I, w0, W.get(name + ".0.bias"), a, H0);
    for (auto& v : a) v = gelu(v);
    linear(a.data(), M, H0, w2, W.get(name + ".2.bias"), c, H2);
    for (auto& v : c) v = gelu(v);
    linear(c.data(), M, H2, w4, W.get(name + ".4.bias"), e, H4);
    O_out = H4;
    return e;
}

// Run heads on N points x K prompts. Returns stage1/stage2 logits [K,N,3] (row=k) and
// iou [K,3]. feat: [N,512], pts:[N,3], prompt:[K,3].
struct HeadOut { std::vector<float> seg1, seg2, iou; int N, K; };
inline HeadOut run_heads(Weights& W, const float* feat, const float* pts, int N,
                         const float* prompt, int K) {
    HeadOut R; R.N = N; R.K = K;
    const int FS = 518;       // 512 + 3 + 3
    // feats_seg flat per (n,k): we iterate k outer to mirror [K,N,*] golden layout
    std::vector<float> fs((size_t)N * FS);
    R.seg1.resize((size_t)K * N * 3);
    R.seg2.resize((size_t)K * N * 3);
    R.iou.resize((size_t)K * 3);
    // We process one prompt at a time (max-pool is within a prompt's N points).
    std::vector<float> mask1((size_t)N * 3), g((size_t)N * 256), fs2((size_t)N * 521),
                       fs3((size_t)N * 777), mask2((size_t)N * 3), fiou((size_t)N * 777);
    for (int k = 0; k < K; k++) {
        // build feats_seg [N,518]
        for (int n = 0; n < N; n++) {
            float* d = fs.data() + (size_t)n * FS;
            const float* f = feat + (size_t)n * 512;
            for (int i = 0; i < 512; i++) d[i] = f[i];
            d[512] = pts[n*3+0]; d[513] = pts[n*3+1]; d[514] = pts[n*3+2];
            d[515] = prompt[k*3+0]; d[516] = prompt[k*3+1]; d[517] = prompt[k*3+2];
        }
        // stage1: seg_mlp_{1,2,3} -> mask1 [N,3]
        int o; auto m1 = mlp3(W, "seg_mlp_1", fs.data(), N, FS, o);
        auto m2 = mlp3(W, "seg_mlp_2", fs.data(), N, FS, o);
        auto m3 = mlp3(W, "seg_mlp_3", fs.data(), N, FS, o);
        for (int n = 0; n < N; n++) {
            mask1[n*3+0] = m1[n]; mask1[n*3+1] = m2[n]; mask1[n*3+2] = m3[n];
            R.seg1[((size_t)k*N + n)*3+0] = m1[n];
            R.seg1[((size_t)k*N + n)*3+1] = m2[n];
            R.seg1[((size_t)k*N + n)*3+2] = m3[n];
        }
        // feats_seg_2 [N,521] = [fs, mask1]
        for (int n = 0; n < N; n++) {
            float* d = fs2.data() + (size_t)n * 521;
            const float* s = fs.data() + (size_t)n * FS;
            for (int i = 0; i < FS; i++) d[i] = s[i];
            d[518] = mask1[n*3+0]; d[519] = mask1[n*3+1]; d[520] = mask1[n*3+2];
        }
        // g = seg_s2_mlp_g [N,256]; max over N -> [256]; broadcast
        int og; auto gv = mlp3(W, "seg_s2_mlp_g", fs2.data(), N, 521, og);   // [N,256]
        std::vector<float> gmax(256, -1e30f);
        for (int n = 0; n < N; n++) for (int c = 0; c < 256; c++)
            gmax[c] = std::max(gmax[c], gv[(size_t)n*256+c]);
        // feats_seg_3 [N,777] = [gmax, feats_seg_2]
        for (int n = 0; n < N; n++) {
            float* d = fs3.data() + (size_t)n * 777;
            for (int c = 0; c < 256; c++) d[c] = gmax[c];
            const float* s = fs2.data() + (size_t)n * 521;
            for (int i = 0; i < 521; i++) d[256 + i] = s[i];
        }
        int o2; auto s1 = mlp3(W, "seg_s2_mlp_1", fs3.data(), N, 777, o2);
        auto s2 = mlp3(W, "seg_s2_mlp_2", fs3.data(), N, 777, o2);
        auto s3 = mlp3(W, "seg_s2_mlp_3", fs3.data(), N, 777, o2);
        for (int n = 0; n < N; n++) {
            mask2[n*3+0] = s1[n]; mask2[n*3+1] = s2[n]; mask2[n*3+2] = s3[n];
            R.seg2[((size_t)k*N + n)*3+0] = s1[n];
            R.seg2[((size_t)k*N + n)*3+1] = s2[n];
            R.seg2[((size_t)k*N + n)*3+2] = s3[n];
        }
        // iou: feats_iou [N,777] = [gmax, feats_seg(518), mask2(3)]
        for (int n = 0; n < N; n++) {
            float* d = fiou.data() + (size_t)n * 777;
            for (int c = 0; c < 256; c++) d[c] = gmax[c];
            const float* s = fs.data() + (size_t)n * FS;
            for (int i = 0; i < FS; i++) d[256 + i] = s[i];
            d[256+518+0] = mask2[n*3+0]; d[256+518+1] = mask2[n*3+1]; d[256+518+2] = mask2[n*3+2];
        }
        int oi; auto iv = mlp3(W, "iou_mlp", fiou.data(), N, 777, oi);   // [N,256]
        std::vector<float> imax(256, -1e30f);
        for (int n = 0; n < N; n++) for (int c = 0; c < 256; c++)
            imax[c] = std::max(imax[c], iv[(size_t)n*256+c]);
        int oo; auto io = mlp3(W, "iou_mlp_out", imax.data(), 1, 256, oo);  // [1,3]
        for (int c = 0; c < 3; c++) R.iou[(size_t)k*3+c] = 1.0f / (1.0f + std::exp(-io[c]));
    }
    return R;
}

} // namespace p3sam
