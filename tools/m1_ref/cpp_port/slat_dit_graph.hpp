// M2 Shape-SLat LR sparse DiT forward graph (ElasticSLatFlowModel / SLatFlowModel.forward).
// For batch=1 the "full varlen sparse attention" is plain attention over N voxel tokens, so
// the block math is IDENTICAL to the SS dense DiT (ss_dit_graph.hpp). Differences:
//   in/out channels = 32, proj_in = 2048, N voxel tokens, RoPE phases from coords (3D sparse
//   rope) instead of the saved grid rope_phases. Mirrors tools/m1_ref/slat_dit.py.
#pragma once
#include "m1_ggml.hpp"
#include "ss_dit_graph.hpp"   // reuse rope_inter, fill_freqs
#include <cmath>

namespace slatdit {
static const int C = 1536, NB = 30, NH = 12, HD = 128, INCH = 32, PROJ_IN = 2048, TE_HALF = 128;
static const int ROPE_DIM = 3, FREQ_DIM = HD / 2 / ROPE_DIM;  // 21
static const float ROPE_THETA = 10000.0f;

// sparse 3D rope COS/SIN [d=128,1,N] from voxel coords_xyz[N,3] (per-pair replicated).
// phase[n, axis*21+f] = coord[n,axis]/10000^(f/21); 3*21=63 angles + 1 zero-pad pair = 64.
static inline void fill_sparse_cos_sin(const int32_t* coords_xyz, int N,
                                       std::vector<float>& cosb, std::vector<float>& sinb) {
    cosb.assign((size_t)HD * N, 0.f); sinb.assign((size_t)HD * N, 1.f * 0.f);
    std::vector<float> freqs(FREQ_DIM);
    for (int f = 0; f < FREQ_DIM; f++) freqs[f] = 1.0f / std::pow(ROPE_THETA, (float)f / FREQ_DIM);
    for (int n = 0; n < N; n++) {
        size_t base = (size_t)n * HD;
        for (int p = 0; p < HD / 2; p++) {
            float ang = 0.f;
            if (p < ROPE_DIM * FREQ_DIM) {  // p<63
                int axis = p / FREQ_DIM, f = p % FREQ_DIM;
                ang = (float)coords_xyz[(size_t)n * 3 + axis] * freqs[f];
            }  // p==63 -> ang 0 (pad)
            float c = std::cos(ang), s = std::sin(ang);
            cosb[base + 2 * p] = c; cosb[base + 2 * p + 1] = c;
            sinb[base + 2 * p] = s; sinb[base + 2 * p + 1] = s;
        }
    }
}

// xin [32,N] (ne0=ch), tin [1] (t_scaled), gin [1024,5], pin [2048,N]; coords_xyz[N,3] i32.
// Returns vout [32,N] (ne0=ch) == slat_dit_v.npy [N,32] raw. rope consts persistent.
static inline ggml_tensor* build_slat_dit_forward(ggml_context* ctx, M1Harness& H, int N,
        ggml_tensor* xin, ggml_tensor* tin, ggml_tensor* gin, ggml_tensor* pin,
        const int32_t* coords_xyz, ggml_tensor** block0_out = nullptr) {
    std::vector<float> frb; ssdit::fill_freqs(frb);
    std::vector<float> cosb, sinb; fill_sparse_cos_sin(coords_xyz, N, cosb, sinb);
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

    ggml_tensor* h = lin(ctx, H.weight("input_layer.weight"), H.weight("input_layer.bias"), xin);  // [1536,N] (xin [32,N])

    // Flash-attn padding mask (PIXAL3D_FLASH + FAST): F16 [n_kv_pad, n_q_pad], -inf on the padded
    // keys, built ONCE and shared by all NB self-attn blocks (one ~47MB tensor, not NB copies).
    ggml_tensor* fa_mask = nullptr;
    if (pix_fast_prec() && std::getenv("PIXAL3D_FLASH")) {
        int64_t nkvpad = GGML_PAD((int64_t)N, 256), nqpad = GGML_PAD((int64_t)N, 256);
        std::vector<uint16_t> md((size_t)nkvpad * nqpad, 0);   // 0x0000 = keep
        uint16_t maskval = std::getenv("FLASH_ZM") ? 0x0000 : 0xFBFF;   // diag: all-zero mask
        for (int64_t j = 0; j < nqpad; j++)
            for (int64_t kk = N; kk < nkvpad; kk++) md[(size_t)j * nkvpad + kk] = maskval;  // -65504 pad keys
        int64_t mne[4] = {nkvpad, nqpad, 1, 1};
        fa_mask = H.const_tensor_f16("fa_mask", 2, mne, std::move(md));
    }

    for (int bi = 0; bi < NB; bi++) {
        std::string bp = "blocks." + std::to_string(bi) + ".";
        ggml_tensor* mod6 = ggml_add(ctx, H.weight(bp + "modulation"), tmod);
        auto chunk = [&](int i) { return ggml_view_1d(ctx, mod6, C, (size_t)i * C * ggml_element_size(mod6)); };
        ggml_tensor* s_msa = chunk(0), *sc_msa = chunk(1), *g_msa = chunk(2);
        ggml_tensor* s_mlp = chunk(3), *sc_mlp = chunk(4), *g_mlp = chunk(5);

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

        ggml_tensor* hn = layernorm(ctx, h, H.weight(bp + "norm2.weight"), H.weight(bp + "norm2.bias"), 1e-6f);
        std::string cab = bp + "cross_attn.cross_attn_block.";
        ggml_tensor* cq = lin(ctx, H.weight(cab + "to_q.weight"), H.weight(cab + "to_q.bias"), hn);
        cq = ggml_reshape_3d(ctx, cq, HD, NH, N);
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
    h = lin(ctx, H.weight("out_layer.weight"), H.weight("out_layer.bias"), h);  // [32, N]
    return ggml_cont(ctx, h);  // [32, N] (ne0=ch) == slat_dit_v.npy [N,32] raw
}
}  // namespace slatdit
