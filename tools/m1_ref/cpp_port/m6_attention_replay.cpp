// Replay the captured Python block-0 self-attention Q/K/V through the native
// attention implementation.  This isolates SDPA parity from the surrounding
// DiT graph. Build: ./build.sh m6_attention_replay cuda
#include "m1_ggml.hpp"
#include <cmath>
#include <cstdio>

int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <M6 trace directory>\n", argv[0]); return 2; }
    const std::string root = argv[1];
    const char * kind_env = std::getenv("M6_ATTENTION_KIND");
    const std::string kind = kind_env ? kind_env : "self";
    if (kind != "self" && kind != "cross") throw std::runtime_error("M6_ATTENTION_KIND must be self or cross");
    const std::string stem = root + "/python_m6_" + kind + "_attn";
    NpyArray qn = npy_load(stem + "_sdpa_q_step_00.npy");
    NpyArray kn = npy_load(stem + "_sdpa_k_step_00.npy");
    NpyArray vn = npy_load(stem + "_sdpa_v_step_00.npy");
    NpyArray rn = npy_load(stem + "_preout_step_00.npy");
    if (qn.shape.size() != 3 || kn.shape.size() != 3 || vn.shape != kn.shape ||
        qn.shape[1] != kn.shape[1] || qn.shape[2] != kn.shape[2]) throw std::runtime_error("invalid M6 Q/K/V trace");
    const int64_t n = qn.shape[0], nk = kn.shape[0], h = qn.shape[1], d = qn.shape[2];
    if (d != 128 || rn.numel() != n*h*d) throw std::runtime_error("invalid M6 attention dimensions");

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) throw std::runtime_error("CUDA unavailable");
    ggml_init_params ip{(size_t) 4096 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, h, n);
    ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, h, nk);
    ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, h, nk);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
    ggml_tensor * out = attention(ctx, q, k, v, 1.0f / std::sqrt((float) d), nullptr, true);
    out = ggml_cast(ctx, out, GGML_TYPE_F32);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) throw std::runtime_error("allocation failed");
    // NumPy [token,head,dim] has the same contiguous byte ordering as ggml [dim,head,token].
    ggml_backend_tensor_set(q, qn.f32(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(k, kn.f32(), 0, ggml_nbytes(k));
    ggml_backend_tensor_set(v, vn.f32(), 0, ggml_nbytes(v));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    std::vector<float> got((size_t) n*h*d);
    ggml_backend_tensor_get(out, got.data(), 0, got.size() * sizeof(float));
    double mae = 0, mse = 0, maxabs = 0, dot = 0, aa = 0, bb = 0;
    for (size_t i = 0; i < got.size(); ++i) { double a = got[i], b = rn.f32()[i], e = a-b;
        mae += std::fabs(e); mse += e*e; maxabs = std::max(maxabs, std::fabs(e)); dot += a*b; aa += a*a; bb += b*b; }
    const double count = got.size();
    std::printf("M6_ATTENTION_REPLAY kind=%s Nq=%lld Nk=%lld H=%lld D=%lld mae=%.9g rmse=%.9g maxabs=%.9g cosine=%.9g\n",
        kind.c_str(), (long long) n, (long long) nk, (long long) h, (long long) d,
        mae/count, std::sqrt(mse/count), maxabs, dot/(std::sqrt(aa*bb)+1e-30));
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend);
    return 0;
}
