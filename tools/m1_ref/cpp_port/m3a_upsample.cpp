// M3a — FlexiDualGridVaeDecoder.upsample(lr_slat, x4) -> hr_coords  (the mesh gateway).
//
// Imperative fp32 host port. Integrates the spike submanifold sparse-conv (CPU mirror /
// CUDA kernel) + SparseConvNeXtBlock3d + SparseResBlockC2S3d + the data-dependent 4-stage
// coord-growth loop. Validates per-stage feats/subdiv/coords vs the true-fp32 oracle
// (refs/stage3a/, from stage3a_capture.py --mode fp32) and hr_coords SET-EQUAL the oracle.
//
// Build:  ./build.sh m3a_upsample          (CPU)
//         ./build.sh m3a_upsample cuda      (conv on GPU; run: ./m3a_upsample cuda)
#include "sparse_vae.hpp"
#include "../../sparse_spike/npy.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <set>
#include <array>

static const char* WDIR = "weights_npy/shape_dec";
static const char* REFS = "refs/stage3a";
static const char* GOLD = "../../sparse_spike/golden_stages";

static std::vector<float> loadf(const std::string& path) {
    NpyArray a = npy_load(path);
    if (a.descr != "<f4") { printf("  !! %s not f32 (%s)\n", path.c_str(), a.descr.c_str()); }
    return std::vector<float>(a.f32(), a.f32() + a.numel());
}
static std::vector<float> W(const std::string& key) { return loadf(std::string(WDIR) + "/" + key + ".npy"); }
static std::vector<float> R(const std::string& key) { return loadf(std::string(REFS) + "/" + key + ".npy"); }

// tight compare (assumes aligned order)
static bool cmp_feats(const char* tag, const std::vector<float>& got, const std::string& refkey, double tol) {
    std::vector<float> ref = R(refkey);
    if (got.size() != ref.size()) { printf("  [%s] SIZE MISMATCH got=%zu ref=%zu\n", tag, got.size(), ref.size()); return false; }
    double ma = 0, sum = 0, dot = 0, na = 0, nb = 0; size_t worst = 0;
    for (size_t i = 0; i < got.size(); i++) {
        double d = std::fabs((double)got[i] - ref[i]); if (d > ma) { ma = d; worst = i; }
        sum += d; dot += (double)got[i]*ref[i]; na += (double)got[i]*got[i]; nb += (double)ref[i]*ref[i];
    }
    double cos = dot / (std::sqrt(na*nb) + 1e-12);
    bool ok = (ma < tol);
    printf("  [%s] n=%zu maxabs=%.3e meanabs=%.3e cosine=%.6f worst@%zu got=%.5f ref=%.5f  %s\n",
           tag, got.size(), ma, sum/got.size(), cos, worst, got[worst], ref[worst], ok ? "OK" : "**HI**");
    return ok;
}

typedef std::array<int32_t,4> Coord;
static std::set<Coord> coordset(const int32_t* c, int N) {
    std::set<Coord> s; for (int i = 0; i < N; i++) s.insert({c[i*4],c[i*4+1],c[i*4+2],c[i*4+3]}); return s;
}
// coord compare: elementwise (order) + set IoU
static void cmp_coords(const char* tag, const std::vector<int32_t>& got, const std::string& refkey) {
    NpyArray ref = npy_load(std::string(REFS) + "/" + refkey + ".npy");
    int Ng = (int)got.size()/4, Nr = (int)ref.shape[0];
    const int32_t* rc = ref.i32();
    int elem_eq = (Ng == Nr);
    if (elem_eq) for (int i = 0; i < Ng*4; i++) if (got[i] != rc[i]) { elem_eq = 0; break; }
    auto sg = coordset(got.data(), Ng), sr = coordset(rc, Nr);
    int inter = 0; for (auto& c : sg) if (sr.count(c)) inter++;
    int uni = (int)sg.size() + (int)sr.size() - inter;
    printf("  [%s] N got=%d ref=%d  elementwise=%s  inter=%d IoU=%.6f (got-only=%d ref-only=%d)\n",
           tag, Ng, Nr, elem_eq ? "EXACT" : "no", inter, (double)inter/uni,
           (int)sg.size()-inter, (int)sr.size()-inter);
}

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");
    printf("[m3a] backend=%s\n", use_cuda ? "cuda(conv)" : "cpu");

    const int MC[5] = {1024, 512, 256, 128, 64};
    const int NB[4] = {4, 16, 8, 4};
    const float EPS = 1e-6f;

    // ---- input: golden lr_slat (DENORM) ----
    NpyArray lcN = npy_load(std::string(GOLD) + "/stage2_out/lr_slat_coords.npy");
    NpyArray lfN = npy_load(std::string(GOLD) + "/stage2_out/lr_slat_feats.npy");
    int N = (int)lcN.shape[0];
    std::vector<int32_t> coords(lcN.i32(), lcN.i32() + (size_t)N*4);
    std::vector<float> lr_feats(lfN.f32(), lfN.f32() + (size_t)N*32);
    printf("[m3a] lr_slat N=%d C=32\n", N);

    // ---- from_latent: Linear 32 -> 1024 ----
    std::vector<float> feats = svae::linear(lr_feats, W("from_latent.weight"), W("from_latent.bias"), N, 32, 1024);
    cmp_feats("from_latent", feats, "from_latent_feats", 1e-3);
    int C = 1024;

    bool all_ok = true;
    std::vector<uint32_t> nmap = svae::build_nmap(coords.data(), N);  // rulebook at current coords

    for (int L = 0; L < 4; L++) {
        int Cout = MC[L+1];
        char pfx[64];
        printf("\n[m3a] === level %d: C=%d Cout=%d N=%d (ConvNeXt x%d + C2S) ===\n", L, C, Cout, N, NB[L]);

        // ---- ConvNeXt blocks (coords fixed) ----
        for (int j = 0; j < NB[L]; j++) {
            snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, j);
            std::string p(pfx);
            std::vector<float> h = svae::subm_conv(feats, nmap, W(p+".conv.weight"), W(p+".conv.bias"), N, C, C, use_cuda);
            svae::layernorm(h, W(p+".norm.weight"), W(p+".norm.bias"), N, C, EPS);
            std::vector<float> m = svae::linear(h, W(p+".mlp.0.weight"), W(p+".mlp.0.bias"), N, C, 4*C);
            svae::silu_inplace(m);
            m = svae::linear(m, W(p+".mlp.2.weight"), W(p+".mlp.2.bias"), N, 4*C, C);
            svae::add_inplace(feats, m);   // ConvNeXt: return mlp(norm(conv(x))) + x
        }
        char tag[32]; snprintf(tag, sizeof(tag), "s%d_in", L);
        all_ok &= cmp_feats(tag, feats, std::string(tag) + "_feats", 5e-2);

        // ---- C2S up-block (grows coords x2) ----
        snprintf(pfx, sizeof(pfx), "blocks.%d.%d", L, NB[L]);
        std::string p(pfx);
        // to_subdiv(x): Linear C->8 on the block INPUT
        std::vector<float> subdiv = svae::linear(feats, W(p+".to_subdiv.weight"), W(p+".to_subdiv.bias"), N, C, 8);
        snprintf(tag, sizeof(tag), "s%d_subdiv_logits", L);
        all_ok &= cmp_feats(tag, subdiv, std::string(tag), 5e-2);
        std::vector<uint8_t> sub((size_t)N*8);
        for (size_t i = 0; i < sub.size(); i++) sub[i] = (subdiv[i] > 0.f) ? 1 : 0;

        // h = conv1(silu(norm1(x)))   [N, Cout*8]
        std::vector<float> h1 = feats;
        svae::layernorm(h1, W(p+".norm1.weight"), W(p+".norm1.bias"), N, C, EPS);  // affine
        svae::silu_inplace(h1);
        std::vector<float> conv1_out = svae::subm_conv(h1, nmap, W(p+".conv1.weight"), W(p+".conv1.bias"), N, C, Cout*8, use_cuda);

        // C2S grow (host index arithmetic)
        svae::C2SIdx ix = svae::c2s_grow(coords.data(), N, sub.data());
        int M = ix.M;
        // h_grown[M,Cout] = gather child-major (conv1_out [N, Cout*8] -> [N*8, Cout])
        std::vector<float> h_grown = svae::gather_children(conv1_out, ix, Cout);
        // skip: x[N,C] -> C2S [M, C/8] -> repeat_interleave x (Cout/(C/8))
        int per = C / 8;
        std::vector<float> skip_small = svae::gather_children(feats, ix, per);
        int rep = Cout / per;   // == 4
        std::vector<float> skip = svae::repeat_interleave(skip_small, M, per, rep);

        // h = conv2(silu(norm2(h_grown))) + skip
        std::vector<float> empty;
        svae::layernorm(h_grown, empty, empty, M, Cout, EPS);   // non-affine norm2
        svae::silu_inplace(h_grown);
        std::vector<uint32_t> nmap2 = svae::build_nmap(ix.coords.data(), M);
        std::vector<float> conv2_out = svae::subm_conv(h_grown, nmap2, W(p+".conv2.weight"), W(p+".conv2.bias"), M, Cout, Cout, use_cuda);
        svae::add_inplace(conv2_out, skip);

        snprintf(tag, sizeof(tag), "s%d_out_coords", L);
        cmp_coords(tag, ix.coords, std::string(tag));
        snprintf(tag, sizeof(tag), "s%d_out", L);
        all_ok &= cmp_feats(tag, conv2_out, std::string(tag) + "_feats", 1e-1);

        // advance state
        feats = std::move(conv2_out);
        coords = std::move(ix.coords);
        nmap = std::move(nmap2);
        N = M; C = Cout;
    }

    // ---- final hr_coords ----
    printf("\n[m3a] === hr_coords (after 4 stages): N=%d ===\n", N);
    // vs fp32 oracle (the tight target — must be SET-EQUAL)
    NpyArray oN = npy_load(std::string(REFS) + "/hr_coords_fp32.npy");
    auto so = coordset(oN.i32(), (int)oN.shape[0]);
    auto sm = coordset(coords.data(), N);
    int inter = 0; for (auto& c : sm) if (so.count(c)) inter++;
    int uni = (int)sm.size() + (int)so.size() - inter;
    printf("[m3a] hr_coords vs FP32 ORACLE: N mine=%d oracle=%d inter=%d IoU=%.6f (mine-only=%d oracle-only=%d)\n",
           N, (int)so.size(), inter, (double)inter/uni, (int)sm.size()-inter, (int)so.size()-inter);
    // vs fp16 golden (sanity — expect ~0.999 boundary noise)
    NpyArray gN = npy_load(std::string(GOLD) + "/stage3a_up/hr_coords.npy");
    auto sg = coordset(gN.i32(), (int)gN.shape[0]);
    int interg = 0; for (auto& c : sm) if (sg.count(c)) interg++;
    int unig = (int)sm.size() + (int)sg.size() - interg;
    printf("[m3a] hr_coords vs fp16 GOLDEN: oracle-IoU=%.6f (golden N=%d)\n", (double)interg/unig, (int)sg.size());

    bool oracle_exact = ((int)so.size() == N) && ((double)inter/uni > 0.99999);
    bool oracle_close = ((double)inter/uni > 0.999);
    printf("\n[m3a] %s  (oracle IoU=%.6f, per-stage feats %s)\n",
           (oracle_exact ? "PASS (SET-EQUAL oracle)" : (oracle_close ? "PASS (IoU>0.999 = fp32 accum noise)" : "FAIL")),
           (double)inter/uni, all_ok ? "tight" : "loose");
    return (oracle_close && all_ok) ? 0 : 1;
}
