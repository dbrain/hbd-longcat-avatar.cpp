// Validate p3sam_sonata.hpp vs banked backbone goldens (feat_s0..s4, grid_s*, pool_inv_s*,
// feat_1232, feats_N). Build: g++ -O3 -fopenmp -std=c++17 test_p3sam_sonata.cpp -o test_p3sam_sonata
#include "p3sam_sonata.hpp"
#include <cstdio>
#include <cmath>

static double cosine(const float* a, const float* b, size_t n) {
    double dot=0,sa=0,sb=0; for (size_t i=0;i<n;i++){dot+=(double)a[i]*b[i];sa+=(double)a[i]*a[i];sb+=(double)b[i]*b[i];}
    return dot/(std::sqrt(sa)*std::sqrt(sb)+1e-30);
}
static double meanabs(const float* a, const float* b, size_t n) {
    double s=0; for (size_t i=0;i<n;i++) s+=std::fabs((double)a[i]-b[i]); return s/n;
}

int main(int argc, char** argv) {
    std::string G = argc>1?argv[1]:"/mnt/hdd/3d/avatar-shootout/p3sam_goldens";
    p3sam::Weights W(G + "/weights_npy");
    auto tfg = npy_load(G + "/tf_grid_coord.npy");   // (M0,3) int32
    auto tff = npy_load(G + "/tf_feat.npy");          // (M0,9)
    auto inv = npy_load(G + "/tf_inverse_final.npy"); // (Nin,) int64
    int M0 = (int)tfg.shape[0], Nin = (int)inv.shape[0];
    printf("M0=%d Nin=%d\n", M0, Nin);
    std::vector<p3sam::StageData> S;
    auto feats_N = p3sam::run_sonata(W, tfg.i32(), tff.f32(), M0, inv.i64(), Nin, &S);

    // per-stage grid + cluster + feat checks
    for (int s = 0; s < 5; s++) {
        auto fg = npy_load(G + "/feat_s" + std::to_string(s) + ".npy");
        auto gg = npy_load(G + "/grid_s" + std::to_string(s) + ".npy");
        int Ng = (int)fg.shape[0], Cg = (int)fg.shape[1];
        long grid_mis = 0;
        if ((int)S[s].N == Ng) {
            for (size_t i = 0; i < (size_t)Ng*3; i++) if (S[s].grid[i] != gg.i32()[i]) grid_mis++;
        } else grid_mis = -1;
        long cl_mis = 0;
        if (s > 0) {
            auto pc = npy_load(G + "/pool_inv_s" + std::to_string(s) + ".npy");
            if ((int)S[s].cluster.size() == (int)pc.numel())
                for (size_t i = 0; i < S[s].cluster.size(); i++) if (S[s].cluster[i] != pc.i64()[i]) cl_mis++;
            else cl_mis = -1;
        }
        double cos = (S[s].N==Ng) ? cosine(S[s].feat.data(), fg.f32(), (size_t)Ng*Cg) : 0;
        double ma  = (S[s].N==Ng) ? meanabs(S[s].feat.data(), fg.f32(), (size_t)Ng*Cg) : 0;
        printf("  stage%d N=%d/%d C=%d  grid_mis=%ld cluster_mis=%ld  feat_cos=%.6f mean_abs=%.3e\n",
               s, S[s].N, Ng, Cg, grid_mis, cl_mis, cos, ma);
    }
    auto g1232 = npy_load(G + "/feat_1232.npy");
    auto gN = npy_load(G + "/feats_N.npy");
    printf("  feats_N  cos=%.6f mean_abs=%.3e\n", cosine(feats_N.data(), gN.f32(), gN.numel()),
           meanabs(feats_N.data(), gN.f32(), gN.numel()));
    return 0;
}
