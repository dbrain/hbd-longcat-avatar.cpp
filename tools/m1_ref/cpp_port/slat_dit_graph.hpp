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

// Structured-latent flow models use the same selective mixed-BF16 contract as
// the sparse-structure DiT: only Linear modules below `blocks` are converted.
// LayerNorm affine parameters, RMSNorm gamma, and the shared modulation
// Parameter remain F32.  Keep the historical F32 graph as the default so
// existing component goldens remain meaningful; this opt-in mode is validated
// separately against the Pixal stage-4 texture oracle before it can be used by
// a production material run.
static inline bool use_native_bf16() {
    const char* e = std::getenv("PIXAL3D_SLAT_BF16");
    return e && std::atoi(e) != 0;
}

// M6 can be evaluated in its model-contract BF16 torso without changing M2
// or M3b geometry.  Keep that experiment on a separate opt-in so a texture
// diagnostic cannot silently alter the geometry cache that conditions it.
static inline bool use_m6_bf16() {
    const char* e = std::getenv("PIXAL3D_M6_BF16");
    return e && std::atoi(e) != 0;
}

// Pixal evaluates the adaLN expression literally as h * (1 + scale) + shift
// in BF16.  Rewriting it as h * scale + h + shift is equivalent in F32 but
// not after BF16 rounding at each CUDA operation; the latter was the first
// material-path divergence on the fixed M6 oracle.  ggml elementwise kernels
// require F32 here, so reify every Python BF16 operation with torso_cast.
static inline ggml_tensor* torso_modulate(ggml_context* ctx, ggml_tensor* h,
                                          ggml_tensor* scale, ggml_tensor* shift,
                                          ggml_tensor* one, bool bf16) {
    if (!bf16) return modulate(ctx, h, scale, shift);
    ggml_tensor* factor = ssdit::torso_cast(ctx, ggml_add(ctx, scale, one), true);
    ggml_tensor* out = ssdit::torso_cast(ctx, ggml_mul(ctx, h, factor), true);
    return ssdit::torso_cast(ctx, ggml_add(ctx, out, shift), true);
}

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
        const int32_t* coords_xyz, ggml_tensor** block0_out = nullptr,
        ggml_tensor** input_layer_out = nullptr, ggml_tensor** self_attn0_out = nullptr,
        ggml_tensor** self_attn0_preout = nullptr, ggml_tensor** self_attn0_qkv = nullptr,
        ggml_tensor** self_attn0_qkv_input = nullptr, ggml_tensor** tmod_out = nullptr,
        ggml_tensor** self_attn0_norm1 = nullptr, ggml_tensor** cross_attn0_out = nullptr,
        ggml_tensor** proj_linear0_out = nullptr, ggml_tensor** mlp0_out = nullptr,
        ggml_tensor** mlp0_input = nullptr, ggml_tensor** mlp0_linear0 = nullptr,
        ggml_tensor** mlp0_hidden = nullptr, ggml_tensor** cross_attn0_preout = nullptr) {
    // M2/M3b and M6 share this graph.  Native M6 has the 64-channel
    // noise||shape input.  Production keeps the geometry flows on the locked
    // F32 route, but the checkpoint itself declares those flows BF16 too.  An
    // explicit geometry-only diagnostic switch lets the sampler goldens decide
    // whether that model-contract precision improves the cascade; it is never
    // inherited by the clean production wrapper.
    const bool m6_precision = xin->ne[0] == 64;
    const char* geo_bf16_env = std::getenv("PIXAL3D_SLAT_BF16_GEOMETRY");
    const bool geometry_bf16 = geo_bf16_env && std::atoi(geo_bf16_env) != 0;
    const bool bf16 = (use_native_bf16() && (m6_precision || geometry_bf16)) ||
                      (m6_precision && use_m6_bf16());
    std::vector<float> frb; ssdit::fill_freqs(frb);
    std::vector<float> cosb, sinb; fill_sparse_cos_sin(coords_xyz, N, cosb, sinb);
    int64_t fr_ne[4] = {TE_HALF, 1, 1, 1};
    ggml_tensor* freqs = H.const_tensor("rope_freqs", 1, fr_ne, std::move(frb));
    std::vector<float> oneb(C, 1.0f);
    int64_t one_ne[4] = {C, 1, 1, 1};
    ggml_tensor* mod_one = H.const_tensor("slat_mod_one", 2, one_ne, std::move(oneb));
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
    tmod = ssdit::torso_cast(ctx, tmod, bf16);
    if (tmod_out) { tmod = ggml_cont(ctx, tmod); *tmod_out = tmod; ggml_set_output(tmod); }

    ggml_tensor* h = lin(ctx, H.weight("input_layer.weight"), H.weight("input_layer.bias"), xin);  // [1536,N] (xin [32,N])
    h = ssdit::torso_cast(ctx, h, bf16);
    if (input_layer_out) { h = ggml_cont(ctx, h); *input_layer_out = h; ggml_set_output(h); }
    ggml_tensor* torso_gin = ssdit::torso_cast(ctx, gin, bf16);
    ggml_tensor* torso_pin = ssdit::torso_cast(ctx, pin, bf16);

    // Flash-attn padding mask (PIXAL3D_FLASH + FAST): F16 [n_kv_pad, n_q_pad], -inf on the padded
    // keys, built ONCE and shared by all NB self-attn blocks (one ~47MB tensor, not NB copies).
    ggml_tensor* fa_mask = nullptr;
    if ((pix_fast_prec() && std::getenv("PIXAL3D_FLASH")) || geo_flash()) {
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
        ggml_tensor* mod6 = ssdit::torso_cast(ctx, ggml_add(ctx, H.weight(bp + "modulation"), tmod), bf16);
        auto chunk = [&](int i) { return ggml_view_1d(ctx, mod6, C, (size_t)i * C * ggml_element_size(mod6)); };
        ggml_tensor* s_msa = chunk(0), *sc_msa = chunk(1), *g_msa = chunk(2);
        ggml_tensor* s_mlp = chunk(3), *sc_mlp = chunk(4), *g_mlp = chunk(5);

        ggml_tensor* a = ssdit::torso_layernorm(ctx, h, nullptr, nullptr, 1e-6f, bf16);
        if (bi == 0 && self_attn0_norm1) { a = ggml_cont(ctx, a); *self_attn0_norm1 = a; ggml_set_output(a); }
        a = torso_modulate(ctx, a, sc_msa, s_msa, mod_one, bf16);
        if (bi == 0 && self_attn0_qkv_input) { a = ggml_cont(ctx, a); *self_attn0_qkv_input = a; ggml_set_output(a); }
        ggml_tensor* qkv = ssdit::torso_lin(ctx, H.weight(bp + "self_attn.to_qkv.weight"), H.weight(bp + "self_attn.to_qkv.bias"), a, bf16);
        if (bi == 0 && self_attn0_qkv) { qkv = ggml_cont(ctx, qkv); *self_attn0_qkv = qkv; ggml_set_output(qkv); }
        qkv = ggml_reshape_4d(ctx, qkv, HD, NH, 3, N);
        auto take = [&](int three) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, qkv, HD, NH, 1, N, qkv->nb[1], qkv->nb[2], qkv->nb[3], (size_t)three * qkv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, N);
        };
        ggml_tensor* q = take(0), *k = take(1), *v = take(2);
        q = ssdit::torso_rmsnorm(ctx, q, H.weight(bp + "self_attn.q_rms_norm.gamma"), bf16);
        k = ssdit::torso_rmsnorm(ctx, k, H.weight(bp + "self_attn.k_rms_norm.gamma"), bf16);
        q = ssdit::torso_cast(ctx, ssdit::rope_inter(ctx, q, COS, SIN), bf16);
        k = ssdit::torso_cast(ctx, ssdit::rope_inter(ctx, k, COS, SIN), bf16);
        ggml_tensor* sa = ssdit::torso_cast(ctx, attention(ctx, q, k, ssdit::torso_cast(ctx, v, bf16), 1.0f / std::sqrt((float)HD), fa_mask, m6_precision), bf16);
        if (bi == 0 && self_attn0_preout) { sa = ggml_cont(ctx, sa); *self_attn0_preout = sa; ggml_set_output(sa); }
        sa = ssdit::torso_lin(ctx, H.weight(bp + "self_attn.to_out.weight"), H.weight(bp + "self_attn.to_out.bias"), sa, bf16);
        if (bi == 0 && self_attn0_out) { sa = ggml_cont(ctx, sa); *self_attn0_out = sa; ggml_set_output(sa); }
        sa = ssdit::torso_cast(ctx, ggml_mul(ctx, sa, g_msa), bf16);
        h = ssdit::torso_cast(ctx, ggml_add(ctx, h, sa), bf16);

        ggml_tensor* hn = ssdit::torso_layernorm(ctx, h, H.weight(bp + "norm2.weight"), H.weight(bp + "norm2.bias"), 1e-6f, bf16);
        std::string cab = bp + "cross_attn.cross_attn_block.";
        ggml_tensor* cq = ssdit::torso_lin(ctx, H.weight(cab + "to_q.weight"), H.weight(cab + "to_q.bias"), hn, bf16);
        cq = ggml_reshape_3d(ctx, cq, HD, NH, N);
        ggml_tensor* ckv = ssdit::torso_lin(ctx, H.weight(cab + "to_kv.weight"), H.weight(cab + "to_kv.bias"), torso_gin, bf16);
        ckv = ggml_reshape_4d(ctx, ckv, HD, NH, 2, 5);
        auto takekv = [&](int two) {
            ggml_tensor* t = ggml_cont(ctx, ggml_view_4d(ctx, ckv, HD, NH, 1, 5, ckv->nb[1], ckv->nb[2], ckv->nb[3], (size_t)two * ckv->nb[2]));
            return ggml_reshape_3d(ctx, t, HD, NH, 5);
        };
        ggml_tensor* ck = takekv(0), *cv = takekv(1);
        cq = ssdit::torso_rmsnorm(ctx, cq, H.weight(cab + "q_rms_norm.gamma"), bf16);
        ck = ssdit::torso_rmsnorm(ctx, ck, H.weight(cab + "k_rms_norm.gamma"), bf16);
        ggml_tensor* ca = ssdit::torso_cast(ctx, attention(ctx, cq, ck, ssdit::torso_cast(ctx, cv, bf16), 1.0f / std::sqrt((float)HD), nullptr, m6_precision), bf16);
        if (bi == 0 && cross_attn0_preout) { ca = ggml_cont(ctx, ca); *cross_attn0_preout = ca; ggml_set_output(ca); }
        ca = ssdit::torso_lin(ctx, H.weight(cab + "to_out.weight"), H.weight(cab + "to_out.bias"), ca, bf16);
        if (bi == 0 && cross_attn0_out) { ca = ggml_cont(ctx, ca); *cross_attn0_out = ca; ggml_set_output(ca); }
        ggml_tensor* pj = ssdit::torso_lin(ctx, H.weight(bp + "cross_attn.proj_linear.weight"), H.weight(bp + "cross_attn.proj_linear.bias"), torso_pin, bf16);
        if (bi == 0 && proj_linear0_out) { pj = ggml_cont(ctx, pj); *proj_linear0_out = pj; ggml_set_output(pj); }
        h = ssdit::torso_cast(ctx, ggml_add(ctx, h, ssdit::torso_cast(ctx, ggml_add(ctx, ca, pj), bf16)), bf16);

        ggml_tensor* m = ssdit::torso_layernorm(ctx, h, nullptr, nullptr, 1e-6f, bf16);
        m = torso_modulate(ctx, m, sc_mlp, s_mlp, mod_one, bf16);
        if (bi == 0 && mlp0_input) { m = ggml_cont(ctx, m); *mlp0_input = m; ggml_set_output(m); }
        m = ssdit::torso_lin(ctx, H.weight(bp + "mlp.mlp.0.weight"), H.weight(bp + "mlp.mlp.0.bias"), m, bf16);
        if (bi == 0 && mlp0_linear0) { m = ggml_cont(ctx, m); *mlp0_linear0 = m; ggml_set_output(m); }
        m = ssdit::torso_cast(ctx, gelu_tanh_(ctx, m), bf16);
        if (bi == 0 && mlp0_hidden) { m = ggml_cont(ctx, m); *mlp0_hidden = m; ggml_set_output(m); }
        m = ssdit::torso_lin(ctx, H.weight(bp + "mlp.mlp.2.weight"), H.weight(bp + "mlp.mlp.2.bias"), m, bf16);
        if (bi == 0 && mlp0_out) { m = ggml_cont(ctx, m); *mlp0_out = m; ggml_set_output(m); }
        m = ssdit::torso_cast(ctx, ggml_mul(ctx, m, g_mlp), bf16);
        h = ssdit::torso_cast(ctx, ggml_add(ctx, h, m), bf16);

        if (bi == 0 && block0_out) { h = ggml_cont(ctx, h); *block0_out = h; ggml_set_output(h); }
    }
    if (bf16) h = ggml_cast(ctx, h, GGML_TYPE_F32);
    h = layernorm(ctx, h, nullptr, nullptr, 1e-5f);
    h = lin(ctx, H.weight("out_layer.weight"), H.weight("out_layer.bias"), h);  // [32, N]
    return ggml_cont(ctx, h);  // [32, N] (ne0=ch) == slat_dit_v.npy [N,32] raw
}
}  // namespace slatdit
