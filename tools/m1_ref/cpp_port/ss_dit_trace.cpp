// Compare one frozen, first-step SS DiT forward against Python block checkpoints.
// Diagnostic only: PIXAL3D_SS_TRACE_DIR must point at a golden_stage_hook trace
// containing stage1_noise/, stage1_cond/, and stage1_block_trace/.
#include "ss_dit_graph.hpp"
#include <cstdio>

static const char* FLOW_W = "weights_npy/ss_flow";

struct Stats {
    double maxabs = 0, meanabs = 0, cosine = 0;
};

static Stats compare_token_major(ggml_tensor* native, const NpyArray& reference) {
    const int64_t n = ssdit::SEQ * ssdit::C;
    if (reference.numel() != n) throw std::runtime_error("checkpoint element-count mismatch");
    std::vector<float> got(n);
    ggml_backend_tensor_get(native, got.data(), 0, n * sizeof(float));
    const float* ref = reference.f32();  // Python [1, token, channel], C-order.
    double sum = 0, dot = 0, ng = 0, nr = 0;
    for (int64_t tok = 0; tok < ssdit::SEQ; ++tok) {
        for (int64_t ch = 0; ch < ssdit::C; ++ch) {
            const double a = got[ch + ssdit::C * tok];  // ggml [channel, token]
            const double b = ref[tok * ssdit::C + ch];
            const double d = std::fabs(a - b);
            sum += d;
            if (d > 0) { /* keep max update branch hot and explicit */ }
            dot += a * b; ng += a * a; nr += b * b;
        }
    }
    Stats out;
    for (int64_t tok = 0; tok < ssdit::SEQ; ++tok)
        for (int64_t ch = 0; ch < ssdit::C; ++ch)
            out.maxabs = std::max(out.maxabs, std::fabs((double)got[ch + ssdit::C * tok] - ref[tok * ssdit::C + ch]));
    out.meanabs = sum / n;
    out.cosine = dot / (std::sqrt(ng) * std::sqrt(nr) + 1e-30);
    return out;
}

int main(int argc, char** argv) {
    const char* trace_env = std::getenv("PIXAL3D_SS_TRACE_DIR");
    if (!trace_env) {
        std::fprintf(stderr, "PIXAL3D_SS_TRACE_DIR is required\n");
        return 2;
    }
    const std::string trace = trace_env;
    const bool use_cuda = argc > 1 && std::string(argv[1]) == "cuda";
    const std::vector<int> indices{0, 4, 9, 14, 19, 24, 29};

    M1Harness H(FLOW_W, 512, use_cuda);
    ggml_context* ctx = H.ctx;
    int64_t x_ne[4] = {ssdit::SEQ, ssdit::INCH, 1, 1};
    int64_t t_ne[4] = {1, 1, 1, 1};
    int64_t g_ne[4] = {1024, 5, 1, 1};
    int64_t p_ne[4] = {1024, ssdit::SEQ, 1, 1};
    ggml_tensor* xin = H.input("x", 2, x_ne);
    ggml_tensor* tin = H.input("t", 1, t_ne);
    ggml_tensor* gin = H.input("global", 2, g_ne);
    ggml_tensor* pin = H.input("proj", 2, p_ne);
    std::vector<ggml_tensor*> taps;
    ggml_tensor* vout = ssdit::build_ss_dit_forward(ctx, H, xin, tin, gin, pin,
                                                     nullptr, &indices, &taps);
    if (taps.size() != indices.size()) throw std::runtime_error("native trace taps missing");
    ggml_set_output(vout);
    ggml_cgraph* graph = new_graph(ctx, 32768);
    ggml_build_forward_expand(graph, vout);
    for (ggml_tensor* tap : taps) ggml_build_forward_expand(graph, tap);
    H.alloc_and_upload(graph);

    NpyArray noise = npy_load(trace + "/stage1_noise/noise.npy");
    NpyArray global = npy_load(trace + "/stage1_cond/global.npy");
    NpyArray proj = npy_load(trace + "/stage1_cond/proj.npy");
    H.upload_input_raw(xin, std::vector<float>(noise.f32(), noise.f32() + noise.numel()));
    H.upload_input_raw(gin, std::vector<float>(global.f32(), global.f32() + global.numel()));
    H.upload_input_raw(pin, std::vector<float>(proj.f32(), proj.f32() + proj.numel()));
    H.upload_input_raw(tin, std::vector<float>{1000.0f});
    H.compute(graph);

    std::printf("[ss_dit_trace] backend=%s precision=%s trace=%s\n", use_cuda ? "cuda" : "cpu",
                ssdit::use_native_bf16() ? "mixed-bf16" : "f32", trace.c_str());
    for (size_t i = 0; i < indices.size(); ++i) {
        char name[64]; std::snprintf(name, sizeof(name), "/stage1_block_trace/block_%02d.npy", indices[i]);
        Stats s = compare_token_major(taps[i], npy_load(trace + name));
        std::printf("  block %02d: maxabs=%.6e meanabs=%.6e cosine=%.9f\n",
                    indices[i], s.maxabs, s.meanabs, s.cosine);
    }
    return 0;
}
