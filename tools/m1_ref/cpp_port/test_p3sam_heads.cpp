// Validate p3sam_heads.hpp vs hd_seg1/seg2_logits + hd_iou goldens.
// Build: g++ -O2 -std=c++17 test_p3sam_heads.cpp -o test_p3sam_heads
#include "p3sam_heads.hpp"
#include <cstdio>
#include <cmath>

static void cmp(const char* tag, const float* a, const float* b, size_t n) {
    double se = 0, sa = 0, sb = 0, maxe = 0, mean_abs = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i]; se += d * d; sa += (double)a[i]*a[i]; sb += (double)b[i]*b[i];
        maxe = std::max(maxe, std::fabs(d)); mean_abs += std::fabs(d);
    }
    double cos = se >= 0 ? (1.0 - se / (std::sqrt(sa)*std::sqrt(sb) + 1e-30)) : 0;
    // proper cosine
    double dot = 0; for (size_t i = 0; i < n; i++) dot += (double)a[i]*b[i];
    cos = dot / (std::sqrt(sa)*std::sqrt(sb) + 1e-30);
    printf("  %-16s cos=%.6f  mean_abs=%.3e  max_abs=%.3e\n", tag, cos, mean_abs/n, maxe);
}

int main(int argc, char** argv) {
    std::string G = argc > 1 ? argv[1] : "/mnt/hdd/3d/avatar-shootout/p3sam_goldens";
    std::string Wdir = G + "/weights_npy";
    auto feat = npy_load(G + "/hd_feats_sub.npy");    // (N,512)
    auto pts  = npy_load(G + "/hd_points_sub.npy");   // (N,3)
    auto prm  = npy_load(G + "/hd_prompt_sub.npy");   // (K,3)
    auto g1   = npy_load(G + "/hd_seg1_logits.npy");  // (K,N,3)
    auto g2   = npy_load(G + "/hd_seg2_logits.npy");
    auto gi   = npy_load(G + "/hd_iou.npy");          // (K,3)
    int N = (int)feat.shape[0], K = (int)prm.shape[0];
    printf("N=%d K=%d\n", N, K);
    p3sam::Weights W(Wdir);
    auto R = p3sam::run_heads(W, feat.f32(), pts.f32(), N, prm.f32(), K);
    cmp("seg1_logits", R.seg1.data(), g1.f32(), R.seg1.size());
    cmp("seg2_logits", R.seg2.data(), g2.f32(), R.seg2.size());
    cmp("iou",         R.iou.data(),  gi.f32(), R.iou.size());
    return 0;
}
