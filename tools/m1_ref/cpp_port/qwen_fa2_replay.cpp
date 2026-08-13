// Replays a captured upstream Qwen FA2 call through the native causal-GQA bridge.
#include "m1_ggml.hpp"
#include <cstdio>
#include <cmath>

int main(int argc, char ** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <trace-dir>\n", argv[0]); return 2; }
    const std::string d = argv[1];
    NpyArray qn = npy_load(d + "/q_bthd.npy"), kn = npy_load(d + "/k_bthd.npy");
    NpyArray vn = npy_load(d + "/v_bthd.npy"), rn = npy_load(d + "/o_bthd.npy");
    if (qn.shape.size() != 4 || kn.shape.size() != 4 || vn.shape != kn.shape || rn.shape.size() != 4 ||
        qn.shape[3] != 128 || kn.shape[3] != 128 || qn.shape[0] != kn.shape[0] ||
        rn.shape[0] != qn.shape[0] || rn.shape[1] != qn.shape[2] || rn.shape[2] != qn.shape[1] || rn.shape[3] != 128)
        throw std::runtime_error("expected q [B,16,Q,128], k/v [B,8,K,128], o [B,Q,16,128]");
    const int B = (int) qn.shape[0], Q = (int) qn.shape[2], K = (int) kn.shape[2], H = (int) qn.shape[1], KH = (int) kn.shape[1];
    if (H != 16 || KH != 8) throw std::runtime_error("expected Qwen 16q/8kv heads");
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) throw std::runtime_error("CUDA unavailable");
    ggml_init_params ip{(size_t)4096 * ggml_tensor_overhead(), nullptr, true};
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 128, Q, H, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 128, K, KH, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_BF16, 128, K, KH, B);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
    ggml_tensor * o = ggml_flash_attn_ext_qwen_causal_gqa(ctx, q, k, v, 1.0f / std::sqrt(128.0f));
    ggml_set_output(o);
    ggml_cgraph * gf = ggml_new_graph(ctx); ggml_build_forward_expand(gf, o);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) throw std::runtime_error("allocation failed");
    auto bf16 = [](const NpyArray & a) { std::vector<ggml_bf16_t> r(a.numel()); for (size_t i=0;i<r.size();++i) r[i]=ggml_fp32_to_bf16(a.f32()[i]); return r; };
    auto qb = bf16(qn), kb = bf16(kn), vb = bf16(vn);
    ggml_backend_tensor_set(q, qb.data(), 0, qb.size()*sizeof(qb[0]));
    ggml_backend_tensor_set(k, kb.data(), 0, kb.size()*sizeof(kb[0]));
    ggml_backend_tensor_set(v, vb.data(), 0, vb.size()*sizeof(vb[0]));
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("compute failed");
    std::vector<ggml_bf16_t> ob((size_t)B*Q*H*128); ggml_backend_tensor_get(o, ob.data(), 0, ob.size()*sizeof(ob[0]));
    double mae=0,mse=0,mx=0; for(size_t i=0;i<ob.size();++i){double e=(double)ggml_bf16_to_fp32(ob[i])-rn.f32()[i];mae+=std::fabs(e);mse+=e*e;mx=std::max(mx,std::fabs(e));}
    std::printf("QWEN_FA2_REPLAY B=%d Q=%d K=%d mae=%.9g rmse=%.9g maxabs=%.9g\n", B,Q,K,mae/ob.size(),std::sqrt(mse/ob.size()),mx);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend);
}
