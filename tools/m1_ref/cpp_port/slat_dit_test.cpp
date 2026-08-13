// M2 Shape-SLat LR sparse DiT single forward in ggml.
// Mirrors tools/m1_ref/slat_dit.py (numpy ref, cross-checked vs real torch
// ElasticSLatFlowModel @ maxabs 6e-6). Validates v[N,32] vs refs/slat_dit_v.npy.
#include "slat_dit_graph.hpp"
#include <cmath>

static const char* WDIR = "weights_npy/slat_flow_512";
static const char* REFS = "refs";
static const char* GOLD = "../../sparse_spike/golden_stages";

int main(int argc, char** argv) {
    bool use_cuda = (argc > 1 && std::string(argv[1]) == "cuda");

    // coords [N,4] -> coords_xyz [N,3]
    NpyArray cN = npy_load(std::string(REFS) + "/slat_coords.npy");
    int N = (int)cN.shape[0];
    const int32_t* c4 = cN.i32();
    std::vector<int32_t> coords_xyz((size_t)N * 3);
    for (int n = 0; n < N; n++) for (int j = 0; j < 3; j++) coords_xyz[n * 3 + j] = c4[n * 4 + 1 + j];

    M1Harness H(WDIR, 512, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t x_ne[4] = {slatdit::INCH, N, 1, 1};   // xin [32,N] (ne0=ch)
    ggml_tensor* xin = H.input("x", 2, x_ne);
    int64_t t_ne[4] = {1, 1, 1, 1}; ggml_tensor* tin = H.input("t", 1, t_ne);
    int64_t g_ne[4] = {1024, 5, 1, 1}; ggml_tensor* gin = H.input("global", 2, g_ne);
    int64_t p_ne[4] = {slatdit::PROJ_IN, N, 1, 1}; ggml_tensor* pin = H.input("proj", 2, p_ne);

    ggml_tensor* block0 = nullptr;
    ggml_tensor* vout = slatdit::build_slat_dit_forward(ctx, H, N, xin, tin, gin, pin, coords_xyz.data(), &block0);
    ggml_set_output(vout);
    ggml_cgraph* gf = new_graph(ctx, 32768);
    ggml_build_forward_expand(gf, vout);
    H.alloc_and_upload(gf);

    H.upload_input_npy(xin, std::string(REFS) + "/slat_dit_x.npy");
    H.upload_input_npy(tin, std::string(REFS) + "/slat_dit_t.npy");
    H.upload_input_npy(gin, std::string(GOLD) + "/stage2_cond/global.npy");
    H.upload_input_npy(pin, std::string(GOLD) + "/stage2_cond/proj_feats.npy");
    H.compute(gf);

    printf("[slat_dit] backend=%s  N=%d\n", use_cuda ? "cuda" : "cpu", N);
    CmpStats v = compare_to_npy(H, vout, std::string(REFS) + "/slat_dit_v.npy", true, "v");
    bool ok = (use_cuda ? v.maxabs < 5e-3 : v.maxabs < 5e-4);
    printf("[slat_dit] %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
