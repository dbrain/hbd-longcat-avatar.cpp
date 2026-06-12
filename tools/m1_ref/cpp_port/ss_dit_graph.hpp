// Reusable SS DiT forward graph builder (shared by ss_dit_test + sampler).
// build_ss_dit_forward(...) constructs the 30-block ModulatedTransformerCrossBlock
// forward to a v-prediction [token,8] tensor. See ss_dit_test.cpp for the validated
// reference; identical math (mirrors tools/m1_ref/ss_dit.py).
#pragma once
#include "m1_ggml.hpp"
#include <cmath>

namespace ssdit {
static const int C = 1536, NB = 30, NH = 12, HD = 128, SEQ = 4096, INCH = 8, TE_HALF = 128;

// complex interleaved RoPE: out=x*COS+rot(x)*SIN, rot[2p]=-x[2p+1],rot[2p+1]=x[2p].
static inline ggml_tensor* rope_inter(ggml_context* ctx, ggml_tensor* t, ggml_tensor* COS, ggml_tensor* SIN) {
    int64_t d = t->ne[0], nh = t->ne[1], tok = t->ne[2];
    ggml_tensor* t5 = ggml_reshape_4d(ctx, t, 2, d / 2, nh, tok);
    ggml_tensor* x0 = ggml_cont(ctx, ggml_view_4d(ctx, t5, 1, d / 2, nh, tok, t5->nb[1], t5->nb[2], t5->nb[3], 0));
    ggml_tensor* x1 = ggml_cont(ctx, ggml_view_4d(ctx, t5, 1, d / 2, nh, tok, t5->nb[1], t5->nb[2], t5->nb[3], t5->nb[0]));
    ggml_tensor* rot5 = ggml_concat(ctx, ggml_neg(ctx, x1), x0, 0);
    ggml_tensor* rot = ggml_reshape_3d(ctx, rot5, d, nh, tok);
    return ggml_add(ctx, ggml_mul(ctx, t, COS), ggml_mul(ctx, rot, SIN));
}

// host helpers for COS/SIN/freqs (defined below)
static inline void fill_freqs(std::vector<float>& fr);
static inline void fill_cos_sin(const std::string& rope_npy, std::vector<float>& cosb, std::vector<float>& sinb);

// Inputs: xin [token,8], tin [1] (t_scaled), gin [1024,5], pin [1024,token].
// rope COS/SIN/freqs are created internally as PERSISTENT constants (survive recompute).
// Returns vout [token,8]; block0 optional.
static inline ggml_tensor* build_ss_dit_forward(ggml_context* ctx, M1Harness& H,
        ggml_tensor* xin, ggml_tensor* tin, ggml_tensor* gin, ggml_tensor* pin,
        ggml_tensor** block0_out = nullptr) {
    // rope constants (persistent — NOT gallocr-managed inputs)
    std::vector<float> frb; fill_freqs(frb);
    std::vector<float> cosb, sinb; fill_cos_sin(H.wdir + "/rope_phases.npy", cosb, sinb);
    int64_t fr_ne[4] = {TE_HALF, 1, 1, 1};
    ggml_tensor* freqs = H.const_tensor("rope_freqs", 1, fr_ne, std::move(frb));
    int64_t cs_ne[4] = {HD, 1, SEQ, 1};
    ggml_tensor* COS = H.const_tensor("rope_cos", 3, cs_ne, std::move(cosb));
    ggml_tensor* SIN = H.const_tensor("rope_sin", 3, cs_ne, std::move(sinb));
    // t-embedding + share_mod modulation
    ggml_tensor* args = ggml_mul(ctx, freqs, tin);
    ggml_tensor* temb = ggml_concat(ctx, ggml_cos(ctx, args), ggml_sin(ctx, args), 0);  // [256]
    temb = lin(ctx, H.weight("t_embedder.mlp.0.weight"), H.weight("t_embedder.mlp.0.bias"), temb);
    temb = silu_(ctx, temb);
    temb = lin(ctx, H.weight("t_embedder.mlp.2.weight"), H.weight("t_embedder.mlp.2.bias"), temb);
    temb = silu_(ctx, temb);
    ggml_tensor* tmod = lin(ctx, H.weight("adaLN_modulation.1.weight"), H.weight("adaLN_modulation.1.bias"), temb);

    ggml_tensor* h = ggml_cont(ctx, ggml_transpose(ctx, xin));  // [8, token]
    h = lin(ctx, H.weight("input_layer.weight"), H.weight("input_layer.bias"), h);  // [1536, token]

    for (int bi = 0; bi < NB; bi++) {
        std::string bp = "blocks." + std::to_string(bi) + ".";
        ggml_tensor* mod6 = ggml_add(ctx, H.weight(bp + "modulation"), tmod);
        auto chunk = [&](int i) { return ggml_view_1d(ctx, mod6, C, (size_t)i * C * ggml_element_size(mod6)); };
        ggml_tensor* s_msa = chunk(0), *sc_msa = chunk(1), *g_msa = chunk(2);
        ggml_tensor* s_mlp = chunk(3), *sc_mlp = chunk(4), *g_mlp = chunk(5);

        ggml_tensor* a = layernorm(ctx, h, nullptr, nullptr, 1e-6f);
        a = modulate(ctx, a, sc_msa, s_msa);
        ggml_tensor* qkv = lin(ctx, H.weight(bp + "self_attn.to_qkv.weight"), H.weight(bp + "self_attn.to_qkv.bias"), a);
        qkv = ggml_reshape_4d(ctx, qkv, HD, NH, 3, SEQ);
        auto take = [&](int three) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, qkv, HD, NH, 1, SEQ, qkv->nb[1], qkv->nb[2], qkv->nb[3], (size_t)three * qkv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, SEQ);
        };
        ggml_tensor* q = take(0), *k = take(1), *v = take(2);
        q = mh_rms_norm(ctx, q, H.weight(bp + "self_attn.q_rms_norm.gamma"));
        k = mh_rms_norm(ctx, k, H.weight(bp + "self_attn.k_rms_norm.gamma"));
        q = rope_inter(ctx, q, COS, SIN);
        k = rope_inter(ctx, k, COS, SIN);
        ggml_tensor* sa = attention(ctx, q, k, v, 1.0f / std::sqrt((float)HD));
        sa = lin(ctx, H.weight(bp + "self_attn.to_out.weight"), H.weight(bp + "self_attn.to_out.bias"), sa);
        sa = ggml_mul(ctx, sa, g_msa);
        h = ggml_add(ctx, h, sa);

        ggml_tensor* hn = layernorm(ctx, h, H.weight(bp + "norm2.weight"), H.weight(bp + "norm2.bias"), 1e-6f);
        std::string cab = bp + "cross_attn.cross_attn_block.";
        ggml_tensor* cq = lin(ctx, H.weight(cab + "to_q.weight"), H.weight(cab + "to_q.bias"), hn);
        cq = ggml_reshape_3d(ctx, cq, HD, NH, SEQ);
        ggml_tensor* ckv = lin(ctx, H.weight(cab + "to_kv.weight"), H.weight(cab + "to_kv.bias"), gin);
        ckv = ggml_reshape_4d(ctx, ckv, HD, NH, 2, 5);
        auto takekv = [&](int two) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, ckv, HD, NH, 1, 5, ckv->nb[1], ckv->nb[2], ckv->nb[3], (size_t)two * ckv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, 5);
        };
        ggml_tensor* ck = takekv(0), *cv = takekv(1);
        cq = mh_rms_norm(ctx, cq, H.weight(cab + "q_rms_norm.gamma"));
        ck = mh_rms_norm(ctx, ck, H.weight(cab + "k_rms_norm.gamma"));
        ggml_tensor* ca = attention(ctx, cq, ck, cv, 1.0f / std::sqrt((float)HD));
        ca = lin(ctx, H.weight(cab + "to_out.weight"), H.weight(cab + "to_out.bias"), ca);
        ggml_tensor* pj = lin(ctx, H.weight(bp + "cross_attn.proj_linear.weight"), H.weight(bp + "cross_attn.proj_linear.bias"), pin);
        h = ggml_add(ctx, h, ggml_add(ctx, ca, pj));

        ggml_tensor* m = layernorm(ctx, h, nullptr, nullptr, 1e-6f);
        m = modulate(ctx, m, sc_mlp, s_mlp);
        m = lin(ctx, H.weight(bp + "mlp.mlp.0.weight"), H.weight(bp + "mlp.mlp.0.bias"), m);
        m = gelu_tanh_(ctx, m);
        m = lin(ctx, H.weight(bp + "mlp.mlp.2.weight"), H.weight(bp + "mlp.mlp.2.bias"), m);
        m = ggml_mul(ctx, m, g_mlp);
        h = ggml_add(ctx, h, m);

        if (bi == 0 && block0_out) { h = ggml_cont(ctx, h); *block0_out = h; ggml_set_output(h); }
    }
    h = layernorm(ctx, h, nullptr, nullptr, 1e-5f);
    h = lin(ctx, H.weight("out_layer.weight"), H.weight("out_layer.bias"), h);  // [8, token]
    return ggml_cont(ctx, ggml_transpose(ctx, h));  // [token, 8]
}

// host helpers for COS/SIN/freqs
static inline void fill_freqs(std::vector<float>& fr) {
    fr.resize(TE_HALF);
    for (int i = 0; i < TE_HALF; i++) fr[i] = std::exp(-std::log(10000.0f) * i / TE_HALF);
}
static inline void fill_cos_sin(const std::string& rope_npy, std::vector<float>& cosb, std::vector<float>& sinb) {
    NpyArray rp = npy_load(rope_npy);  // [4096,64,2]
    const float* rpf = rp.f32();
    cosb.assign((size_t)HD * SEQ, 0.f); sinb.assign((size_t)HD * SEQ, 0.f);
    for (int t = 0; t < SEQ; t++)
        for (int p = 0; p < HD / 2; p++) {
            float cz = rpf[((size_t)t * (HD / 2) + p) * 2 + 0];
            float sz = rpf[((size_t)t * (HD / 2) + p) * 2 + 1];
            size_t base = (size_t)t * HD;
            cosb[base + 2 * p] = cz; cosb[base + 2 * p + 1] = cz;
            sinb[base + 2 * p] = sz; sinb[base + 2 * p + 1] = sz;
        }
}
}  // namespace ssdit
