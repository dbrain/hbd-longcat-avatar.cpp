// TRELLIS-2 CROSS-MODE tex DiT forward graph (SLatFlowModel, image_attn_mode="cross").
// This is the model the tex_goldens were captured from (Trellis2TexturingPipeline tex_slat_flow_model).
// It is a DIFFERENT model from pixal3d's own proj-mode `slat_flow_imgshape2tex_1024` (which uses
// cross_attn.cross_attn_block.* + proj_linear over 5 global tokens). Here the cross-attn is STANDARD
// full-token cross-attention: to_kv runs over the FULL image cond [Ntok,1024], no proj_linear.
//
// Same skeleton as slat_dit_graph.hpp (self-attn 3D-RoPE + RMSNorm q/k, share_mod modulation, tanh-GELU
// MLP, t_embedder). Deltas vs the geometry graph:
//   - input_layer takes 64ch (x is fed pre-concatenated [x32 || shape_norm32]); in=64, out=32.
//   - cross-attn weight prefix is `blocks.{i}.cross_attn.` (NOT `.cross_attn_block.`).
//   - cross-attn KV context = cin [1024, Ntok] (all image tokens), NO proj add, NO gate.
// Config (slat_flow_imgshape2tex_dit_1_3B_1024_bf16.json): C=1536, NB=30, NH=12, HD=128, in=64,
// out=32, cond_channels=1024, mlp 1536->8192->1536, pe_mode rope, share_mod, qk_rms_norm(_cross).
#pragma once
#include "m1_ggml.hpp"
#include "ss_dit_graph.hpp"     // fill_freqs, rope_inter
#include "slat_dit_graph.hpp"   // fill_sparse_cos_sin, constants
#include <cmath>

namespace texdit {
using slatdit::C; using slatdit::NB; using slatdit::NH; using slatdit::HD;
using slatdit::TE_HALF;
static const int IN_CH = 64, OUT_CH = 32;

// xin [64,N] (ne0=ch, pre-concat [x||shape_norm]); tin [1] (t_scaled = 1000*t); cin [1024,Ntok]
// (image cond tokens); coords_xyz[N,3] i32. Returns vout [32,N] (ne0=ch) == tex pred_v [N,32] raw.
static inline ggml_tensor* build_tex_dit_cross_forward(ggml_context* ctx, M1Harness& H, int N, int Ntok,
        ggml_tensor* xin, ggml_tensor* tin, ggml_tensor* cin, const int32_t* coords_xyz) {
    std::vector<float> frb; ssdit::fill_freqs(frb);
    std::vector<float> cosb, sinb; slatdit::fill_sparse_cos_sin(coords_xyz, N, cosb, sinb);
    int64_t fr_ne[4] = {TE_HALF, 1, 1, 1};
    ggml_tensor* freqs = H.const_tensor("rope_freqs", 1, fr_ne, std::move(frb));
    int64_t cs_ne[4] = {HD, 1, N, 1};
    ggml_tensor* COS = H.const_tensor("srope_cos", 3, cs_ne, std::move(cosb));
    ggml_tensor* SIN = H.const_tensor("srope_sin", 3, cs_ne, std::move(sinb));

    ggml_tensor* args = ggml_mul(ctx, freqs, tin);
    ggml_tensor* temb = ggml_concat(ctx, ggml_cos(ctx, args), ggml_sin(ctx, args), 0);
    temb = lin(ctx, H.weight("t_embedder.mlp.0.weight"), H.weight("t_embedder.mlp.0.bias"), temb);
    temb = silu_(ctx, temb);
    temb = lin(ctx, H.weight("t_embedder.mlp.2.weight"), H.weight("t_embedder.mlp.2.bias"), temb);
    temb = silu_(ctx, temb);
    ggml_tensor* tmod = lin(ctx, H.weight("adaLN_modulation.1.weight"), H.weight("adaLN_modulation.1.bias"), temb);

    ggml_tensor* h = lin(ctx, H.weight("input_layer.weight"), H.weight("input_layer.bias"), xin);  // [1536,N]

    // Flash-attn padding mask for the self-attn (shared across blocks), as in slat_dit_graph.
    ggml_tensor* fa_mask = nullptr;
    if ((pix_fast_prec() && std::getenv("PIXAL3D_FLASH")) || geo_flash()) {
        int64_t nkvpad = GGML_PAD((int64_t)N, 256), nqpad = GGML_PAD((int64_t)N, 256);
        std::vector<uint16_t> md((size_t)nkvpad * nqpad, 0);
        uint16_t maskval = 0xFBFF;   // -65504
        for (int64_t j = 0; j < nqpad; j++)
            for (int64_t kk = N; kk < nkvpad; kk++) md[(size_t)j * nkvpad + kk] = maskval;
        int64_t mne[4] = {nkvpad, nqpad, 1, 1};
        fa_mask = H.const_tensor_f16("fa_mask", 2, mne, std::move(md));
    }

    for (int bi = 0; bi < NB; bi++) {
        std::string bp = "blocks." + std::to_string(bi) + ".";
        ggml_tensor* mod6 = ggml_add(ctx, H.weight(bp + "modulation"), tmod);
        auto chunk = [&](int i) { return ggml_view_1d(ctx, mod6, C, (size_t)i * C * ggml_element_size(mod6)); };
        ggml_tensor* s_msa = chunk(0), *sc_msa = chunk(1), *g_msa = chunk(2);
        ggml_tensor* s_mlp = chunk(3), *sc_mlp = chunk(4), *g_mlp = chunk(5);

        // --- self-attn (norm1 affine-free, modulate, 3D RoPE, RMSNorm q/k) ---
        ggml_tensor* a = layernorm(ctx, h, nullptr, nullptr, 1e-6f);
        a = modulate(ctx, a, sc_msa, s_msa);
        ggml_tensor* qkv = lin(ctx, H.weight(bp + "self_attn.to_qkv.weight"), H.weight(bp + "self_attn.to_qkv.bias"), a);
        qkv = ggml_reshape_4d(ctx, qkv, HD, NH, 3, N);
        auto take = [&](int three) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, qkv, HD, NH, 1, N, qkv->nb[1], qkv->nb[2], qkv->nb[3], (size_t)three * qkv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, N);
        };
        ggml_tensor* q = take(0), *k = take(1), *v = take(2);
        q = mh_rms_norm(ctx, q, H.weight(bp + "self_attn.q_rms_norm.gamma"));
        k = mh_rms_norm(ctx, k, H.weight(bp + "self_attn.k_rms_norm.gamma"));
        q = ssdit::rope_inter(ctx, q, COS, SIN);
        k = ssdit::rope_inter(ctx, k, COS, SIN);
        ggml_tensor* sa = attention(ctx, q, k, v, 1.0f / std::sqrt((float)HD), fa_mask);
        sa = lin(ctx, H.weight(bp + "self_attn.to_out.weight"), H.weight(bp + "self_attn.to_out.bias"), sa);
        sa = ggml_mul(ctx, sa, g_msa);
        h = ggml_add(ctx, h, sa);

        // --- cross-attn (norm2 AFFINE, full-token cross over cin, RMSNorm q/k, no gate/proj) ---
        ggml_tensor* hn = layernorm(ctx, h, H.weight(bp + "norm2.weight"), H.weight(bp + "norm2.bias"), 1e-6f);
        std::string ca_p = bp + "cross_attn.";
        ggml_tensor* cq = lin(ctx, H.weight(ca_p + "to_q.weight"), H.weight(ca_p + "to_q.bias"), hn);
        cq = ggml_reshape_3d(ctx, cq, HD, NH, N);
        ggml_tensor* ckv = lin(ctx, H.weight(ca_p + "to_kv.weight"), H.weight(ca_p + "to_kv.bias"), cin);  // [3072,Ntok]
        ckv = ggml_reshape_4d(ctx, ckv, HD, NH, 2, Ntok);
        auto takekv = [&](int two) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, ckv, HD, NH, 1, Ntok, ckv->nb[1], ckv->nb[2], ckv->nb[3], (size_t)two * ckv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, Ntok);
        };
        ggml_tensor* ck = takekv(0), *cv = takekv(1);
        cq = mh_rms_norm(ctx, cq, H.weight(ca_p + "q_rms_norm.gamma"));
        ck = mh_rms_norm(ctx, ck, H.weight(ca_p + "k_rms_norm.gamma"));
        ggml_tensor* ca = attention(ctx, cq, ck, cv, 1.0f / std::sqrt((float)HD));
        ca = lin(ctx, H.weight(ca_p + "to_out.weight"), H.weight(ca_p + "to_out.bias"), ca);
        h = ggml_add(ctx, h, ca);

        // --- MLP (norm3 affine-free, modulate, tanh-GELU, gate) ---
        ggml_tensor* m = layernorm(ctx, h, nullptr, nullptr, 1e-6f);
        m = modulate(ctx, m, sc_mlp, s_mlp);
        m = lin(ctx, H.weight(bp + "mlp.mlp.0.weight"), H.weight(bp + "mlp.mlp.0.bias"), m);
        m = gelu_tanh_(ctx, m);
        m = lin(ctx, H.weight(bp + "mlp.mlp.2.weight"), H.weight(bp + "mlp.mlp.2.bias"), m);
        m = ggml_mul(ctx, m, g_mlp);
        h = ggml_add(ctx, h, m);
    }
    h = layernorm(ctx, h, nullptr, nullptr, 1e-5f);
    h = lin(ctx, H.weight("out_layer.weight"), H.weight("out_layer.bias"), h);  // [32, N]
    return ggml_cont(ctx, h);
}
}  // namespace texdit
