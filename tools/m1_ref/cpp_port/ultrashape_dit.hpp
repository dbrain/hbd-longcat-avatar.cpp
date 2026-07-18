// ultrashape_dit.hpp — ggml graph for the UltraShape RefineDiT (the flow-matching denoiser).
//
// ULTRASHAPE-ARCH.md §2. 21 post-norm DiT blocks over seq = [time_token, K voxel tokens]:
//   x = x_embedder(noisy_latent[64])          -> [2048, K]
//   c = t_embedder(timesteps(t))              -> [2048, 1]  (time token, prepended)
//   x = cat([c, x])                           -> [2048, 1+K]
//   per block: (U-Net skip on blk 11..20) -> +attn1(norm1,RoPE) -> +attn2(norm2, cond) -> +FFN(norm3)
//   final_layer: norm_final -> drop time token -> linear(2048->64)
//
// Wrinkles vs the VAE blocks:
//  - SEPARATE to_q/to_k/to_v (no fused c_qkv), but cat((q,k,v),-1).view(.,16,384).split(128) re-scrambles
//    them into the per-head-interleaved layout -> replicated EXACTLY by concat(q;k;v) + groups=3 head-slice.
//  - qk_norm is **RMSNorm (weight only, eps 1e-6)** on head_dim=128, applied per head (us_dit_rms).
//  - 3D RoPE (precompute_freqs_cis_3d_interpolated, head_dim 128, theta 1e4, res 128 -> scale 1.0) on
//    self-attn q,k; the time token gets identity RoPE (cos=1,sin=0). Half-split rotate_half.
//  - cross-attn (attn2): q from x, k/v from cond[1024]; cat((k,v),-1) groups=2 head-slice; qk RMSNorm.
//  - U-Net skips: blocks 11..20 pop a pushed earlier-block output: skip_norm(skip_linear(cat[skip,x])).
//  - MoE (blocks 15..20): top-2-of-8 softmax gate + shared expert. ggml_mul_mat_id over stacked experts.
//    THE PARITY TRAP — gate run in fp32 (lin forces PREC_F32, soft_max fp32). Non-MoE blocks use plain MLP.
#pragma once
#include "m1_ggml.hpp"
#include "vecset_encoder.hpp"   // vs_take_head_slice / vs_to_heads
#include <string>
#include <cmath>
#include <vector>

struct UsDitCfg {
    int hidden = 2048;
    int heads = 16;
    int depth = 21;
    int in_channels = 64;
    int context_dim = 1024;
    int num_moe_layers = 6;     // last 6 blocks
    int num_experts = 8;
    int moe_top_k = 2;
    int freq_embed = 8192;      // t_embedder frequency_embedding_size = hidden*4
    int voxel_query_res = 128;
    float ln_eps = 1e-6f;
    float qk_eps = 1e-6f;       // RMSNorm eps
    float rope_theta = 10000.0f;

    int head_dim() const { return hidden / heads; }   // 128
    float attn_scale() const { return 1.0f / std::sqrt((float) head_dim()); }
    bool is_moe(int layer) const { return depth - layer <= num_moe_layers; }   // 15..20
    bool has_skip(int layer) const { return layer > depth / 2; }               // 11..20
    bool pushes(int layer) const { return layer < depth / 2; }                 // 0..9
};

// --- host precomputes (no weights) ---------------------------------------------------------------

// Timesteps(num_channels) sinusoidal embedding of a scalar t (downscale_freq_shift=0, scale=1,
// max_period=10000): half=nc/2; exponent=-ln(1e4)*arange(half)/half; emb=t*exp(exponent);
// out = cat([sin(emb), cos(emb)]).  Returns length-nc vector.
static inline std::vector<float> us_timesteps_embed(float t, int nc) {
    int half = nc / 2;
    std::vector<float> out(nc);
    for (int i = 0; i < half; ++i) {
        float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half);
        float a = t * freq;
        out[i] = std::sin(a);
        out[half + i] = std::cos(a);
    }
    return out;
}

// 3D RoPE cos/sin for the full sequence [identity_time_token, K voxel tokens].
// precompute_freqs_cis_3d_interpolated(dim=head_dim, theta, trained=current=voxel_query_res -> scale 1).
// voxel_idx: row-major [K,3] (int grid coords as float). Returns cos and sin each [head_dim*(1+K)]
// row-major over [token, head_dim] (token 0 = identity: cos=1, sin=0).
static inline void us_rope_3d(const float* voxel_idx, int64_t K, const UsDitCfg& cfg,
                              std::vector<float>& cos_out, std::vector<float>& sin_out) {
    const int dim = cfg.head_dim();                  // 128
    const int dim_x = dim / 3, dim_y = dim / 3, dim_z = dim - dim_x - dim_y;  // 42,42,44
    const int target_len = dim / 2;                  // 64
    auto mk_freqs = [&](int d) {
        std::vector<float> f;
        for (int i = 0; i < d; i += 2) f.push_back(1.0f / std::pow(cfg.rope_theta, (float)i / (float)d));
        return f;
    };
    std::vector<float> fx = mk_freqs(dim_x), fy = mk_freqs(dim_y), fz = mk_freqs(dim_z);
    int nfx = dim_x / 2 + (dim_x % 2);               // 21
    int nfy = dim_y / 2 + (dim_y % 2);               // 21
    fx.resize(nfx);
    fy.resize(nfy);
    fz.resize(target_len - nfx - nfy);               // 22
    const int64_t S = 1 + K;
    cos_out.assign((size_t)S * dim, 0.0f);
    sin_out.assign((size_t)S * dim, 0.0f);
    // token 0 = identity
    for (int d = 0; d < dim; ++d) { cos_out[d] = 1.0f; sin_out[d] = 0.0f; }
    for (int64_t p = 0; p < K; ++p) {
        float ix = voxel_idx[p * 3 + 0], iy = voxel_idx[p * 3 + 1], iz = voxel_idx[p * 3 + 2];
        // scale_factor = current/trained = 1.0 -> pos == idx
        std::vector<float> args; args.reserve(target_len);
        for (float f : fx) args.push_back(ix * f);
        for (float f : fy) args.push_back(iy * f);
        for (float f : fz) args.push_back(iz * f);   // 64 values
        float* co = cos_out.data() + (size_t)(p + 1) * dim;
        float* si = sin_out.data() + (size_t)(p + 1) * dim;
        for (int j = 0; j < target_len; ++j) {       // duplicated to fill 128
            co[j] = std::cos(args[j]); co[target_len + j] = std::cos(args[j]);
            si[j] = std::sin(args[j]); si[target_len + j] = std::sin(args[j]);
        }
    }
}

// --- graph helpers -------------------------------------------------------------------------------

// per-head RMSNorm (weight only, eps): x [head_dim, heads, S]; gamma [head_dim] broadcasts over heads.
static inline ggml_tensor* us_dit_rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gamma, float eps) {
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), gamma);
}

// apply_rotary_emb: x*cos + rotate_half(x)*sin.  x [hd,heads,S]; cos/sin [hd,1,S] (broadcast heads).
static inline ggml_tensor* us_rope_apply(ggml_context* ctx, ggml_tensor* x, ggml_tensor* cos, ggml_tensor* sin) {
    ggml_tensor* rot = rotate_half(ctx, x);
    return ggml_add(ctx, ggml_mul(ctx, x, cos), ggml_mul(ctx, rot, sin));
}

// generic FFN: proj_out( gelu_erf( proj_in(x) ) ), both +bias.  names supplied (mlp.fc1/fc2 or net.0.proj/net.2).
static inline ggml_tensor* us_ffn(M1Harness& H, ggml_context* ctx, const std::string& in_w, const std::string& in_b,
                                  const std::string& out_w, const std::string& out_b, ggml_tensor* x) {
    ggml_tensor* h = lin(ctx, H.weight(in_w), H.weight(in_b), x);
    h = gelu_erf_(ctx, h);
    return lin(ctx, H.weight(out_w), H.weight(out_b), h);
}

// Self-attention (attn1): separate q/k/v -> concat -> groups=3 head-slice -> qk RMSNorm -> RoPE -> SDPA -> out_proj.
static inline ggml_tensor* us_dit_self_attn(M1Harness& H, ggml_context* ctx, const UsDitCfg& cfg,
                                            const std::string& p, ggml_tensor* x,
                                            ggml_tensor* rcos, ggml_tensor* rsin) {
    const int hd = cfg.head_dim(), nh = cfg.heads;
    ggml_tensor* q = lin(ctx, H.weight(p + "to_q.weight"), nullptr, x);   // [2048, S]
    ggml_tensor* k = lin(ctx, H.weight(p + "to_k.weight"), nullptr, x);
    ggml_tensor* v = lin(ctx, H.weight(p + "to_v.weight"), nullptr, x);
    ggml_tensor* packed = ggml_concat(ctx, ggml_concat(ctx, q, k, 0), v, 0);  // [6144, S] = cat(q,k,v)
    ggml_tensor* qh = vs_take_head_slice(ctx, packed, hd, nh, 3, 0);  // [hd, nh, S]
    ggml_tensor* kh = vs_take_head_slice(ctx, packed, hd, nh, 3, 1);
    ggml_tensor* vh = vs_take_head_slice(ctx, packed, hd, nh, 3, 2);
    qh = us_dit_rms(ctx, qh, H.weight(p + "q_norm.weight"), cfg.qk_eps);
    kh = us_dit_rms(ctx, kh, H.weight(p + "k_norm.weight"), cfg.qk_eps);
    qh = us_rope_apply(ctx, qh, rcos, rsin);
    kh = us_rope_apply(ctx, kh, rcos, rsin);

    // USR_DIT_FLASH (opt-in, default OFF -> byte-identical to prod): fuse the refine DiT self-attention
    // with ggml_flash_attn_ext. This is the S=8192 self-attn whose dense [S, S, heads] fp32 scores
    // tensor is ~4.3 GB = the pipeline's VRAM PEAK (held ~813s), and it never flashed because this call
    // site passes no mask. Flash removes the scores tensor entirely (VRAM) and fuses the softmax (time).
    // fp32 accumulation kept; only K/V storage is f16. Null mask safe (S=8192 is x256). V is power-of-2
    // pre-scaled (exact in f16) against the documented low-t V-overflow NaN (m1_ggml.hpp attention()).
    static const bool dit_flash = []{ const char* e=std::getenv("USR_DIT_FLASH"); return e ? std::atoi(e)!=0 : false; }();  // image_to_rig sets =1 (prod default); =0 disables. Tests (no env) get fp32.
    if (dit_flash) {
        const int64_t d = qh->ne[0], nhd = qh->ne[1], tq = qh->ne[2];
        ggml_tensor* qf = ggml_cont(ctx, ggml_permute(ctx, qh, 0, 2, 1, 3));   // [d, S, head] F32
        ggml_tensor* kf = ggml_cont(ctx, ggml_permute(ctx, kh, 0, 2, 1, 3));
        ggml_tensor* vf = ggml_cont(ctx, ggml_permute(ctx, vh, 0, 2, 1, 3));
        const float vsc = 1.0f / 64.0f;
        vf = ggml_scale(ctx, vf, vsc);
        kf = ggml_cast(ctx, kf, GGML_TYPE_F16);
        vf = ggml_cast(ctx, vf, GGML_TYPE_F16);
        ggml_tensor* r = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, cfg.attn_scale(), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
        ggml_tensor* o = ggml_cont_2d(ctx, r, d * nhd, tq);
        o = ggml_scale(ctx, o, 1.0f / vsc);
        return lin(ctx, H.weight(p + "out_proj.weight"), H.weight(p + "out_proj.bias"), o);
    }

    ggml_tensor* o = attention(ctx, qh, kh, vh, cfg.attn_scale());    // [2048, S]
    return lin(ctx, H.weight(p + "out_proj.weight"), H.weight(p + "out_proj.bias"), o);
}

// Cross-attention (attn2): q from x, k/v from cond; cat((k,v)) groups=2 head-slice; qk RMSNorm; SDPA; out_proj.
static inline ggml_tensor* us_dit_cross_attn(M1Harness& H, ggml_context* ctx, const UsDitCfg& cfg,
                                             const std::string& p, ggml_tensor* x, ggml_tensor* cond) {
    const int hd = cfg.head_dim(), nh = cfg.heads;
    ggml_tensor* q = lin(ctx, H.weight(p + "to_q.weight"), nullptr, x);     // [2048, S]
    ggml_tensor* k = lin(ctx, H.weight(p + "to_k.weight"), nullptr, cond);  // [2048, Tc]
    ggml_tensor* v = lin(ctx, H.weight(p + "to_v.weight"), nullptr, cond);
    ggml_tensor* kv = ggml_concat(ctx, k, v, 0);                  // [4096, Tc]
    ggml_tensor* qh = vs_to_heads(ctx, q, hd, nh);                // [hd, nh, S]
    ggml_tensor* kh = vs_take_head_slice(ctx, kv, hd, nh, 2, 0);  // [hd, nh, Tc]
    ggml_tensor* vh = vs_take_head_slice(ctx, kv, hd, nh, 2, 1);
    qh = us_dit_rms(ctx, qh, H.weight(p + "q_norm.weight"), cfg.qk_eps);
    kh = us_dit_rms(ctx, kh, H.weight(p + "k_norm.weight"), cfg.qk_eps);

    // USR_DIT_FLASH: flash the cross-attention too. Its dense [Tc, S, heads] f32 scores (Tc=cond tokens)
    // are the DiT's VRAM peak at high N (query-tiled by the GLOBAL PIXAL3D_ATTN_CAP_MB, which other
    // stages -- geo DiT/VAE/decode -- want HIGH for speed). Flashing here removes them WITHOUT touching
    // that global cap. KV = Tc cond tokens (small); q = S DiT tokens. fp32 accum; f16 K/V storage.
    static const bool dit_flash = []{ const char* e=std::getenv("USR_DIT_FLASH"); return e ? std::atoi(e)!=0 : false; }();  // image_to_rig sets =1 (prod default); =0 disables. Tests (no env) get fp32.
    if (dit_flash) {
        const int64_t d = qh->ne[0], nhd = qh->ne[1], tq = qh->ne[2];
        ggml_tensor* qf = ggml_cont(ctx, ggml_permute(ctx, qh, 0, 2, 1, 3));   // [d, S, head]
        ggml_tensor* kf = ggml_cont(ctx, ggml_permute(ctx, kh, 0, 2, 1, 3));   // [d, Tc, head]
        ggml_tensor* vf = ggml_cont(ctx, ggml_permute(ctx, vh, 0, 2, 1, 3));
        const float vsc = 1.0f / 64.0f;
        vf = ggml_scale(ctx, vf, vsc);
        kf = ggml_cast(ctx, kf, GGML_TYPE_F16);
        vf = ggml_cast(ctx, vf, GGML_TYPE_F16);
        ggml_tensor* r = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, cfg.attn_scale(), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
        ggml_tensor* o = ggml_cont_2d(ctx, r, d * nhd, tq);
        o = ggml_scale(ctx, o, 1.0f / vsc);
        return lin(ctx, H.weight(p + "out_proj.weight"), H.weight(p + "out_proj.bias"), o);
    }

    ggml_tensor* o = attention(ctx, qh, kh, vh, cfg.attn_scale());// [2048, S]
    return lin(ctx, H.weight(p + "out_proj.weight"), H.weight(p + "out_proj.bias"), o);
}

// MoE: top-2-of-8 softmax-routed experts + always-on shared expert.  h = norm3(x) [2048, S].
static inline ggml_tensor* us_moe(M1Harness& H, ggml_context* ctx, const UsDitCfg& cfg,
                                  const std::string& p, ggml_tensor* h) {
    const int E = cfg.num_experts, K = cfg.moe_top_k, D = cfg.hidden, F = cfg.freq_embed;
    int64_t N = h->ne[1];
    // --- gate (fp32) ---
    ggml_tensor* logits = lin(ctx, H.weight(p + "gate.weight"), nullptr, h);  // [E, N]
    ggml_tensor* probs = ggml_soft_max(ctx, logits);                          // [E, N]
    ggml_tensor* sel = ggml_top_k(ctx, probs, K);                             // [K, N] I32
    ggml_tensor* w = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, E, N), sel);  // [1, K, N]
    // --- stacked expert weights ---
    std::vector<std::string> w0k, b0k, w2k, b2k;
    for (int e = 0; e < E; ++e) {
        std::string ep = p + "experts." + std::to_string(e) + ".";
        w0k.push_back(ep + "net.0.proj.weight"); b0k.push_back(ep + "net.0.proj.bias");
        w2k.push_back(ep + "net.2.weight");      b2k.push_back(ep + "net.2.bias");
    }
    int64_t w0_ne[4] = {D, F, E, 1}, b0_ne[4] = {F, E, 1, 1};
    int64_t w2_ne[4] = {F, D, E, 1}, b2_ne[4] = {D, E, 1, 1};
    ggml_tensor* W0 = H.weight_stack(w0k, 3, w0_ne);   // [D, F, E]
    ggml_tensor* B0 = H.weight_stack(b0k, 2, b0_ne);   // [F, E]
    ggml_tensor* W2 = H.weight_stack(w2k, 3, w2_ne);   // [F, D, E]
    ggml_tensor* B2 = H.weight_stack(b2k, 2, b2_ne);   // [D, E]
    ggml_tensor* hb = ggml_reshape_3d(ctx, h, D, 1, N);                       // [D, 1, N]
    ggml_tensor* sel_flat = ggml_reshape_1d(ctx, sel, K * N);                 // gather bias per (k,token)
    // (ggml_mul_mat_set_prec asserts OP==MUL_MAT, so it can't force fp32 on mul_mat_id; CPU is fp32
    //  regardless, and the parity-critical gate matmul above already forces PREC_F32.)

    // USR_MOE_CHUNK: process the expert path in token tiles. The expert intermediate a0 [F=8192,K,N]
    // and its bias gather gb0 [F,K,N] are the DiT's dominant N-scaling tensors (~2x F*K*N*4 bytes =
    // ~4.3 GB at N=16384) AND the mul_mat_id runtime pool scales with them -> the N>=16384 OOM.
    // Tiling caps them at F*K*chunk; the result is bit-identical (per-token routing, no cross-token
    // op). Default 0 = untiled (exact legacy path). The gate (sel/w) stays full — it's tiny [E/K,N].
    static const int64_t moe_chunk = []{ const char* e=std::getenv("USR_MOE_CHUNK"); return e?atoll(e):(int64_t)0; }();  // image_to_rig sets =8192 (prod default: untiles at N<=8192, tiles above to fit); =0 disables
    auto expert_path = [&](ggml_tensor* hb_c, ggml_tensor* sel_c, ggml_tensor* w_c, ggml_tensor* h_c,
                           int64_t n) -> ggml_tensor* {
        ggml_tensor* sf = ggml_reshape_1d(ctx, sel_c, K * n);
        ggml_tensor* a0 = ggml_mul_mat_id(ctx, W0, hb_c, sel_c);                              // [F,K,n]
        ggml_tensor* gb0 = ggml_reshape_3d(ctx, ggml_get_rows(ctx, B0, sf), F, K, n);         // [F,K,n]
        a0 = gelu_erf_(ctx, ggml_add(ctx, a0, gb0));
        ggml_tensor* a2 = ggml_mul_mat_id(ctx, W2, a0, sel_c);                                // [D,K,n]
        ggml_tensor* gb2 = ggml_reshape_3d(ctx, ggml_get_rows(ctx, B2, sf), D, K, n);         // [D,K,n]
        a2 = ggml_mul(ctx, ggml_add(ctx, a2, gb2), w_c);
        ggml_tensor* acc = ggml_cont(ctx, ggml_view_2d(ctx, a2, D, n, a2->nb[2], 0));
        for (int j = 1; j < K; ++j)
            acc = ggml_add(ctx, acc, ggml_cont(ctx, ggml_view_2d(ctx, a2, D, n, a2->nb[2], (size_t)j*a2->nb[1])));
        ggml_tensor* shared = us_ffn(H, ctx, p+"shared_experts.net.0.proj.weight", p+"shared_experts.net.0.proj.bias",
                                     p+"shared_experts.net.2.weight", p+"shared_experts.net.2.bias", h_c);
        return ggml_add(ctx, acc, shared);                                                   // [D,n]
    };
    if (moe_chunk <= 0 || moe_chunk >= N) {
        return expert_path(hb, sel, w, h, N);
    }
    // tiled: slice tokens [t0, t0+tc) off hb/sel/w/h, run experts, concat back along ne1.
    ggml_tensor* out = nullptr;
    for (int64_t t0 = 0; t0 < N; t0 += moe_chunk) {
        int64_t tc = std::min<int64_t>(moe_chunk, N - t0);
        ggml_tensor* hb_c = ggml_cont(ctx, ggml_view_3d(ctx, hb, D, 1, tc, hb->nb[1], hb->nb[2], (size_t)t0*hb->nb[2]));
        ggml_tensor* sel_c = ggml_cont(ctx, ggml_view_2d(ctx, sel, K, tc, sel->nb[1], (size_t)t0*sel->nb[1]));
        ggml_tensor* w_c  = ggml_cont(ctx, ggml_view_3d(ctx, w,  1, K, tc, w->nb[1], w->nb[2], (size_t)t0*w->nb[2]));
        ggml_tensor* h_c  = ggml_cont(ctx, ggml_view_2d(ctx, h,  D, tc, h->nb[1], (size_t)t0*h->nb[1]));
        ggml_tensor* part = expert_path(hb_c, sel_c, w_c, h_c, tc);                           // [D,tc]
        out = out ? ggml_concat(ctx, out, part, 1) : part;
    }
    return out;
}

// One DiT block.  x [2048, S].  skip = popped earlier-block output (or null).  cond [1024, Tc].
static inline ggml_tensor* us_dit_block(M1Harness& H, ggml_context* ctx, const UsDitCfg& cfg, int layer,
                                        ggml_tensor* x, ggml_tensor* cond, ggml_tensor* skip,
                                        ggml_tensor* rcos, ggml_tensor* rsin) {
    const std::string p = "blocks." + std::to_string(layer) + ".";
    if (cfg.has_skip(layer)) {
        // USR_F16_SKIP stores the U-Net skip stack in F16 (~half the held VRAM; the skips are the
        // biggest N-scaling live tensors after flash removes the attn scores). Cast back to F32 here
        // so the concat/skip_linear run at full precision. Near-lossless (residual features); A/B meshes.
        if (skip->type != GGML_TYPE_F32) skip = ggml_cast(ctx, skip, GGML_TYPE_F32);
        ggml_tensor* cat = ggml_concat(ctx, skip, x, 0);   // [4096, S] = cat([skip, x], -1)
        x = lin(ctx, H.weight(p + "skip_linear.weight"), H.weight(p + "skip_linear.bias"), cat);  // [2048,S]
        x = layernorm(ctx, x, H.weight(p + "skip_norm.weight"), H.weight(p + "skip_norm.bias"), cfg.ln_eps);
    }
    // self-attn
    ggml_tensor* n1 = layernorm(ctx, x, H.weight(p + "norm1.weight"), H.weight(p + "norm1.bias"), cfg.ln_eps);
    x = ggml_add(ctx, x, us_dit_self_attn(H, ctx, cfg, p + "attn1.", n1, rcos, rsin));
    // cross-attn
    ggml_tensor* n2 = layernorm(ctx, x, H.weight(p + "norm2.weight"), H.weight(p + "norm2.bias"), cfg.ln_eps);
    x = ggml_add(ctx, x, us_dit_cross_attn(H, ctx, cfg, p + "attn2.", n2, cond));
    // FFN / MoE
    ggml_tensor* n3 = layernorm(ctx, x, H.weight(p + "norm3.weight"), H.weight(p + "norm3.bias"), cfg.ln_eps);
    ggml_tensor* ff = cfg.is_moe(layer) ? us_moe(H, ctx, cfg, p + "moe.", n3)
                                        : us_ffn(H, ctx, p + "mlp.fc1.weight", p + "mlp.fc1.bias",
                                                 p + "mlp.fc2.weight", p + "mlp.fc2.bias", n3);
    return ggml_add(ctx, x, ff);
}

// t_embedder: timesteps(t) -> mlp.0 (2048->8192) -> GELU -> mlp.2 (8192->2048).  ts_embed [2048,1] (host).
static inline ggml_tensor* us_t_embedder(M1Harness& H, ggml_context* ctx, const UsDitCfg&, ggml_tensor* ts_embed) {
    ggml_tensor* h = lin(ctx, H.weight("t_embedder.mlp.0.weight"), H.weight("t_embedder.mlp.0.bias"), ts_embed);
    h = gelu_erf_(ctx, h);
    return lin(ctx, H.weight("t_embedder.mlp.2.weight"), H.weight("t_embedder.mlp.2.bias"), h);  // [2048, 1]
}

// Full RefineDiT.  x_lat [in_channels=64, K] (noisy latent), ts_embed [2048,1] (host timesteps),
// cond [1024, Tc], rcos/rsin [head_dim, 1, 1+K] (host RoPE).  Returns noise_pred [64, K].
static inline ggml_tensor* us_refine_dit(M1Harness& H, ggml_context* ctx, const UsDitCfg& cfg,
                                         ggml_tensor* x_lat, ggml_tensor* ts_embed, ggml_tensor* cond,
                                         ggml_tensor* rcos, ggml_tensor* rsin) {
    const bool dump = (std::getenv("PIXAL3D_DIT_DUMP") != nullptr);
    ggml_tensor* c = us_t_embedder(H, ctx, cfg, ts_embed);                                  // [2048, 1]
    ggml_tensor* xe = lin(ctx, H.weight("x_embedder.weight"), H.weight("x_embedder.bias"), x_lat);  // [2048, K]
    if (dump) { ggml_set_output(c); ggml_set_name(c, "dbg_c"); ggml_set_output(xe); ggml_set_name(xe, "dbg_xe"); }
    ggml_tensor* x = ggml_concat(ctx, c, xe, 1);                                            // [2048, 1+K]
    static const bool f16skip = std::getenv("USR_F16_SKIP") != nullptr;
    std::vector<ggml_tensor*> skip_stack;
    for (int layer = 0; layer < cfg.depth; ++layer) {
        ggml_tensor* skip = nullptr;
        if (cfg.has_skip(layer)) { skip = skip_stack.back(); skip_stack.pop_back(); }
        x = us_dit_block(H, ctx, cfg, layer, x, cond, skip, rcos, rsin);
        if (dump) { char nm[24]; snprintf(nm, sizeof(nm), "dbg_blk%d", layer); ggml_set_output(x); ggml_set_name(x, nm); }
        // Push the skip in F16 (USR_F16_SKIP) — halves the held U-Net skip VRAM at high N.
        if (cfg.pushes(layer)) skip_stack.push_back(f16skip ? ggml_cast(ctx, x, GGML_TYPE_F16) : x);
    }
    // final_layer: norm_final -> drop time token -> linear(2048->64)
    x = layernorm(ctx, x, H.weight("final_layer.norm_final.weight"), H.weight("final_layer.norm_final.bias"), cfg.ln_eps);
    int64_t K = x->ne[1] - 1;
    x = ggml_cont(ctx, ggml_view_2d(ctx, x, cfg.hidden, K, x->nb[1], x->nb[1]));            // drop token 0
    return lin(ctx, H.weight("final_layer.linear.weight"), H.weight("final_layer.linear.bias"), x);  // [64, K]
}
