// Reusable SS DiT forward graph builder (shared by ss_dit_test + sampler).
// build_ss_dit_forward(...) constructs the 30-block ModulatedTransformerCrossBlock
// forward to a v-prediction [token,8] tensor. See ss_dit_test.cpp for the validated
// reference; identical math (mirrors tools/m1_ref/ss_dit.py).
#pragma once
#include "m1_ggml.hpp"
#include <algorithm>
#include <cmath>

namespace ssdit {
static const int C = 1536, NB = 30, NH = 12, HD = 128, SEQ = 4096, INCH = 8, TE_HALF = 128;

// Pixal's flow torso is trained and deployed in mixed BF16/F32.  Its conversion
// helper changes only Linear modules inside `blocks`; LayerNorm parameters,
// RMSNorm gains, and the shared block modulation parameter remain F32.  The
// input projection, time embedding/share-modulation, final norm, and output
// projection also remain F32.
// Keeping the old F32 port as the default preserves the existing component
// parity tests.  PIXAL3D_SS_BF16=1 is the native-model-contract diagnostic
// mode: it emulates BF16 block values as a BF16 round-trip around each block
// operation.  ggml's CUDA elementwise broadcast/norm kernels do not accept
// BF16 operands, so the actual operation remains F32 but sees and returns the
// same BF16-representable values that the PyTorch torso carries between ops.
static inline bool use_native_bf16() {
    const char* e = std::getenv("PIXAL3D_SS_BF16");
    return e && std::atoi(e) != 0;
}

static inline ggml_tensor* torso_cast(ggml_context* ctx, ggml_tensor* x, bool bf16) {
    if (!bf16) return x;
    // CUDA's add/norm paths reject a BF16 input, but F32<-BF16 preserves the
    // exact BF16 rounding point and makes the result usable by those paths.
    return ggml_cast(ctx, ggml_cast(ctx, x, GGML_TYPE_BF16), GGML_TYPE_F32);
}

// Tensor-core accumulation is F32, as in PyTorch.  PyTorch then stores the
// linear result in BF16, represented here by torso_cast before the next op.
static inline ggml_tensor* torso_lin(ggml_context* ctx, ggml_tensor* W, ggml_tensor* b,
                                     ggml_tensor* x, bool bf16) {
    if (!bf16) return lin(ctx, W, b, x);
    // Diagnostic A/B: preserve the exact BF16 operands but execute the GEMM
    // through F32 storage/accumulation.  This isolates cuBLAS BF16 output
    // rounding/order from the surrounding model contract on the fixed M6
    // oracle.  Production remains the direct BF16 tensor-core path.
    const bool f32_gemm = std::getenv("PIXAL3D_BF16_F32_GEMM") != nullptr;
    ggml_tensor* w = ggml_cast(ctx, W, GGML_TYPE_BF16);
    ggml_tensor* a = ggml_cast(ctx, x, GGML_TYPE_BF16);
    if (f32_gemm) { w = ggml_cast(ctx, w, GGML_TYPE_F32); a = ggml_cast(ctx, a, GGML_TYPE_F32); }
    ggml_tensor* y = ggml_mul_mat(ctx, w, a);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    if (b) y = ggml_add(ctx, y, torso_cast(ctx, b, true));
    return torso_cast(ctx, y, true);
}

static inline ggml_tensor* torso_layernorm(ggml_context* ctx, ggml_tensor* x,
                                            ggml_tensor* w, ggml_tensor* b, float eps,
                                            bool bf16) {
    if (!bf16) return layernorm(ctx, x, w, b, eps);
    // LayerNorm32 upcasts its BF16 activation to F32; its affine parameters
    // were never converted by Pixal's MIX_PRECISION_MODULES helper.
    return torso_cast(ctx, layernorm(ctx, torso_cast(ctx, x, true),
                                    w, b, eps), true);
}

static inline ggml_tensor* torso_rmsnorm(ggml_context* ctx, ggml_tensor* x,
                                          ggml_tensor* gamma, bool bf16) {
    if (!bf16) return mh_rms_norm(ctx, x, gamma);
    // MultiHeadRMSNorm.gamma is a Parameter, not a Linear parameter, so it
    // stays F32 in the reference.  The result is cast back to the activation
    // dtype at the module boundary.
    return torso_cast(ctx, mh_rms_norm(ctx, torso_cast(ctx, x, true),
                                       gamma), true);
}

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
static inline void fill_cos_sin(M1Harness& H, std::vector<float>& cosb, std::vector<float>& sinb);

// Inputs: xin [token,8], tin [1] (t_scaled), gin [1024,5], pin [1024,token].
// rope COS/SIN/freqs are created internally as PERSISTENT constants (survive recompute).
// Returns vout [token,8]; block0 optional.
static inline ggml_tensor* build_ss_dit_forward(ggml_context* ctx, M1Harness& H,
        ggml_tensor* xin, ggml_tensor* tin, ggml_tensor* gin, ggml_tensor* pin,
        ggml_tensor** block0_out = nullptr,
        const std::vector<int>* block_tap_indices = nullptr,
        std::vector<ggml_tensor*>* block_taps = nullptr) {
    const bool bf16 = use_native_bf16();
    // rope constants (persistent — NOT gallocr-managed inputs)
    std::vector<float> frb; fill_freqs(frb);
    std::vector<float> cosb, sinb; fill_cos_sin(H, cosb, sinb);
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
    tmod = torso_cast(ctx, tmod, bf16);

    ggml_tensor* h = ggml_cont(ctx, ggml_transpose(ctx, xin));  // [8, token]
    h = lin(ctx, H.weight("input_layer.weight"), H.weight("input_layer.bias"), h);  // [1536, token]
    h = torso_cast(ctx, h, bf16);
    // The reference casts both projection-conditioning tensors before entering
    // the BF16 torso.  They are not merely weights of a BF16 linear: the
    // original values are otherwise retained through cross-attention/proj.
    ggml_tensor* torso_gin = torso_cast(ctx, gin, bf16);
    ggml_tensor* torso_pin = torso_cast(ctx, pin, bf16);

    for (int bi = 0; bi < NB; bi++) {
        std::string bp = "blocks." + std::to_string(bi) + ".";
        // `modulation` is a raw Parameter and stays F32.  Python adds it to
        // BF16 tmod in F32, then explicitly casts the sum to mod.dtype.
        ggml_tensor* mod6 = torso_cast(ctx, ggml_add(ctx, H.weight(bp + "modulation"), tmod), bf16);
        auto chunk = [&](int i) { return ggml_view_1d(ctx, mod6, C, (size_t)i * C * ggml_element_size(mod6)); };
        ggml_tensor* s_msa = chunk(0), *sc_msa = chunk(1), *g_msa = chunk(2);
        ggml_tensor* s_mlp = chunk(3), *sc_mlp = chunk(4), *g_mlp = chunk(5);

        ggml_tensor* a = torso_layernorm(ctx, h, nullptr, nullptr, 1e-6f, bf16);
        a = torso_cast(ctx, modulate(ctx, a, sc_msa, s_msa), bf16);
        ggml_tensor* qkv = torso_lin(ctx, H.weight(bp + "self_attn.to_qkv.weight"), H.weight(bp + "self_attn.to_qkv.bias"), a, bf16);
        qkv = ggml_reshape_4d(ctx, qkv, HD, NH, 3, SEQ);
        auto take = [&](int three) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, qkv, HD, NH, 1, SEQ, qkv->nb[1], qkv->nb[2], qkv->nb[3], (size_t)three * qkv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, SEQ);
        };
        ggml_tensor* q = take(0), *k = take(1), *v = take(2);
        q = torso_rmsnorm(ctx, q, H.weight(bp + "self_attn.q_rms_norm.gamma"), bf16);
        k = torso_rmsnorm(ctx, k, H.weight(bp + "self_attn.k_rms_norm.gamma"), bf16);
        q = torso_cast(ctx, rope_inter(ctx, q, COS, SIN), bf16);
        k = torso_cast(ctx, rope_inter(ctx, k, COS, SIN), bf16);
        ggml_tensor* sa = torso_cast(ctx, attention(ctx, q, k, torso_cast(ctx, v, bf16), 1.0f / std::sqrt((float)HD)), bf16);
        sa = torso_lin(ctx, H.weight(bp + "self_attn.to_out.weight"), H.weight(bp + "self_attn.to_out.bias"), sa, bf16);
        sa = torso_cast(ctx, ggml_mul(ctx, sa, g_msa), bf16);
        h = torso_cast(ctx, ggml_add(ctx, h, sa), bf16);

        ggml_tensor* hn = torso_layernorm(ctx, h, H.weight(bp + "norm2.weight"), H.weight(bp + "norm2.bias"), 1e-6f, bf16);
        std::string cab = bp + "cross_attn.cross_attn_block.";
        ggml_tensor* cq = torso_lin(ctx, H.weight(cab + "to_q.weight"), H.weight(cab + "to_q.bias"), hn, bf16);
        cq = ggml_reshape_3d(ctx, cq, HD, NH, SEQ);
        ggml_tensor* ckv = torso_lin(ctx, H.weight(cab + "to_kv.weight"), H.weight(cab + "to_kv.bias"), torso_gin, bf16);
        ckv = ggml_reshape_4d(ctx, ckv, HD, NH, 2, 5);
        auto takekv = [&](int two) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, ckv, HD, NH, 1, 5, ckv->nb[1], ckv->nb[2], ckv->nb[3], (size_t)two * ckv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, 5);
        };
        ggml_tensor* ck = takekv(0), *cv = takekv(1);
        cq = torso_rmsnorm(ctx, cq, H.weight(cab + "q_rms_norm.gamma"), bf16);
        ck = torso_rmsnorm(ctx, ck, H.weight(cab + "k_rms_norm.gamma"), bf16);
        ggml_tensor* ca = torso_cast(ctx, attention(ctx, cq, ck, torso_cast(ctx, cv, bf16), 1.0f / std::sqrt((float)HD)), bf16);
        ca = torso_lin(ctx, H.weight(cab + "to_out.weight"), H.weight(cab + "to_out.bias"), ca, bf16);
        ggml_tensor* pj = torso_lin(ctx, H.weight(bp + "cross_attn.proj_linear.weight"), H.weight(bp + "cross_attn.proj_linear.bias"), torso_pin, bf16);
        h = torso_cast(ctx, ggml_add(ctx, h, torso_cast(ctx, ggml_add(ctx, ca, pj), bf16)), bf16);

        ggml_tensor* m = torso_layernorm(ctx, h, nullptr, nullptr, 1e-6f, bf16);
        m = torso_cast(ctx, modulate(ctx, m, sc_mlp, s_mlp), bf16);
        m = torso_lin(ctx, H.weight(bp + "mlp.mlp.0.weight"), H.weight(bp + "mlp.mlp.0.bias"), m, bf16);
        m = torso_cast(ctx, gelu_tanh_(ctx, m), bf16);
        m = torso_lin(ctx, H.weight(bp + "mlp.mlp.2.weight"), H.weight(bp + "mlp.mlp.2.bias"), m, bf16);
        m = torso_cast(ctx, ggml_mul(ctx, m, g_mlp), bf16);
        h = torso_cast(ctx, ggml_add(ctx, h, m), bf16);

        if (bi == 0 && block0_out) { h = ggml_cont(ctx, h); *block0_out = h; ggml_set_output(h); }
        if (block_taps && block_tap_indices &&
            std::find(block_tap_indices->begin(), block_tap_indices->end(), bi) != block_tap_indices->end()) {
            h = ggml_cont(ctx, h);
            block_taps->push_back(h);
            ggml_set_output(h);
        }
    }
    // Explicit Python manual_cast(h, x.dtype) before the F32 final norm/head.
    if (bf16) h = ggml_cast(ctx, h, GGML_TYPE_F32);
    h = layernorm(ctx, h, nullptr, nullptr, 1e-5f);
    h = lin(ctx, H.weight("out_layer.weight"), H.weight("out_layer.bias"), h);  // [8, token]
    return ggml_cont(ctx, ggml_transpose(ctx, h));  // [token, 8]
}

// host helpers for COS/SIN/freqs
static inline void fill_freqs(std::vector<float>& fr) {
    fr.resize(TE_HALF);
    for (int i = 0; i < TE_HALF; i++) fr[i] = std::exp(-std::log(10000.0f) * i / TE_HALF);
}
static inline void fill_cos_sin(M1Harness& H, std::vector<float>& cosb, std::vector<float>& sinb) {
    std::vector<float> rp = H.host_f32("rope_phases");  // [4096,64,2]
    const float* rpf = rp.data();
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
