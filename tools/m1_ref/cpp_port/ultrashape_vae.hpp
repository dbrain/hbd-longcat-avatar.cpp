// ultrashape_vae.hpp — ggml graph for the UltraShape ShapeVAE DECODE path.
//
// Two pieces (ULTRASHAPE-ARCH.md §3), both vecset/Michelangelo-family blocks we already know
// (`vecset_encoder.hpp`), with ONE new wrinkle vs the rig encoder: qk_norm here is a per-head
// **LayerNorm** (weight+bias, eps 1e-6) on head_dim=64, NOT RMSNorm.
//
//   vae.forward(latents):  post_kl(Linear 64->1024) -> Transformer(16x ResidualAttentionBlock self-attn)
//   geo_decoder(queries, latents):  query_proj(Linear fourier_out=51 -> 1024) on fourier(queries)
//                                   -> ResidualCrossAttentionBlock(query, latents) -> ln_post -> output_proj(->1)
//
// Config (configs/infer_dit_refine.yaml vae_config): width=1024 heads=16 (head_dim 64) num_freqs=8
// include_pi=false include_input=true point_feats=4 num_decoder_layers=16 qkv_bias=false qk_norm=true
// ln eps=1e-6 GELU=erf attention scale=1/sqrt(64). FourierEmbedder.out_dim = 3*(8*2+1)=51.
//
// The FourierEmbedder is a deterministic per-point transform (no weights) -> done on the HOST
// (us_fourier_embed below); this graph consumes the already-embedded [51, Nq] query tensor.
#pragma once
#include "m1_ggml.hpp"
#include "vecset_encoder.hpp"   // vs_take_head_slice / vs_to_heads (shared head-split helpers)
#include <string>
#include <cmath>
#include <vector>

struct UsVaeCfg {
    int width = 1024;
    int heads = 16;
    int n_decoder_layers = 16;
    int num_freqs = 8;
    bool include_pi = false;
    bool include_input = true;
    int embed_dim = 64;        // VAE latent channels (post_kl input)
    int mlp_scale = 4;
    float ln_eps = 1e-6f;
    std::string prefix = "";   // weights live at top level: post_kl.*, transformer.*, geo_decoder.*

    int head_dim() const { return width / heads; }
    int fourier_out() const { return 3 * (num_freqs * 2 + (include_input ? 1 : 0)); }  // 51
    float attn_scale() const { return 1.0f / std::sqrt((float) head_dim()); }
};

// Host FourierEmbedder (logspace freqs 2^0..2^(num_freqs-1), include_pi=false, include_input=true).
// queries: row-major [Nq,3] (xyz). Returns row-major [Nq, out_dim] with out_dim = 3*(2*nf+1):
//   [x, y, z,  sin(x*f0..),sin(y*..),sin(z*..),  cos(x*f0..),cos(y*..),cos(z*..)]
// matching torch: embed = (x[...,None]*freq).view(...,-1)  (per-coord freq-contiguous),
//                 cat(x, embed.sin(), embed.cos()).
static inline std::vector<float> us_fourier_embed(const float* queries, int64_t Nq, const UsVaeCfg& cfg) {
    const int nf = cfg.num_freqs;
    std::vector<float> freqs(nf);
    for (int i = 0; i < nf; ++i) freqs[i] = std::pow(2.0f, (float)i) * (cfg.include_pi ? (float)M_PI : 1.0f);
    const int out_dim = cfg.fourier_out();
    std::vector<float> out((size_t)Nq * out_dim);
    for (int64_t p = 0; p < Nq; ++p) {
        const float* q = queries + p * 3;
        float* o = out.data() + (size_t)p * out_dim;
        // include_input: raw xyz first
        o[0] = q[0]; o[1] = q[1]; o[2] = q[2];
        // embed[c*nf + f] = q[c]*freq[f]; sin block then cos block
        float* sblk = o + 3;
        float* cblk = o + 3 + 3 * nf;
        for (int c = 0; c < 3; ++c)
            for (int f = 0; f < nf; ++f) {
                float v = q[c] * freqs[f];
                sblk[c * nf + f] = std::sin(v);
                cblk[c * nf + f] = std::cos(v);
            }
    }
    return out;
}

// generate_dense_grid_points(bbox_min,bbox_max,octree_res,indexing="ij"): linspace per axis with
// octree_res+1 points; meshgrid ij -> flattened [i,j,k] = i*G*G + j*G + k -> xyz = (x[i],y[j],z[k]).
// Returns row-major [G^3, 3] with G = octree_res+1.  bounds=±b (symmetric).
static inline std::vector<float> us_dense_grid_queries(int octree_res, float b, int& G) {
    G = octree_res + 1;
    std::vector<float> ax(G);
    for (int i = 0; i < G; ++i) ax[i] = -b + (2.0f * b) * (float)i / (float)(G - 1);  // linspace(-b,b,G)
    std::vector<float> out((size_t)G * G * G * 3);
    size_t idx = 0;
    for (int i = 0; i < G; ++i)
        for (int j = 0; j < G; ++j)
            for (int k = 0; k < G; ++k) {
                out[idx++] = ax[i]; out[idx++] = ax[j]; out[idx++] = ax[k];
            }
    return out;
}

// per-head LayerNorm (qk_norm). x [head_dim, heads, T]; norm over ne0=head_dim, affine w/b [head_dim].
static inline ggml_tensor* us_qk_ln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w, ggml_tensor* b, float eps) {
    return layernorm(ctx, x, w, b, eps);
}

// MLP: c_proj( gelu_erf( c_fc(x) ) ).
static inline ggml_tensor* us_mlp(M1Harness& H, ggml_context* ctx, const UsVaeCfg&,
                                  const std::string& p, ggml_tensor* x) {
    ggml_tensor* h = lin(ctx, H.weight(p + "c_fc.weight"), H.weight(p + "c_fc.bias"), x);
    h = gelu_erf_(ctx, h);
    return lin(ctx, H.weight(p + "c_proj.weight"), H.weight(p + "c_proj.bias"), h);
}

// Self-attention (MultiheadAttention): c_qkv (no bias) -> split q,k,v -> qk LayerNorm -> SDPA -> c_proj (+bias).
// `ap` = attention prefix (e.g. "...attn."); qk_norm weights at ap+"attention.{q,k}_norm.{weight,bias}".
static inline ggml_tensor* us_self_attn(M1Harness& H, ggml_context* ctx, const UsVaeCfg& cfg,
                                        const std::string& ap, ggml_tensor* x) {
    const int hd = cfg.head_dim(), nh = cfg.heads;
    ggml_tensor* qkv = lin(ctx, H.weight(ap + "c_qkv.weight"), nullptr, x);   // [3*width, T] (qkv_bias=false)
    ggml_tensor* q = vs_take_head_slice(ctx, qkv, hd, nh, 3, 0);   // [hd, nh, T]
    ggml_tensor* k = vs_take_head_slice(ctx, qkv, hd, nh, 3, 1);
    ggml_tensor* v = vs_take_head_slice(ctx, qkv, hd, nh, 3, 2);
    q = us_qk_ln(ctx, q, H.weight(ap + "attention.q_norm.weight"), H.weight(ap + "attention.q_norm.bias"), cfg.ln_eps);
    k = us_qk_ln(ctx, k, H.weight(ap + "attention.k_norm.weight"), H.weight(ap + "attention.k_norm.bias"), cfg.ln_eps);
    ggml_tensor* o = attention(ctx, q, k, v, cfg.attn_scale());   // [width, T]
    return lin(ctx, H.weight(ap + "c_proj.weight"), H.weight(ap + "c_proj.bias"), o);
}

// Cross-attention (MultiheadCrossAttention): c_q(query, no bias), c_kv(data, no bias) -> qk LN -> SDPA -> c_proj.
static inline ggml_tensor* us_cross_attn(M1Harness& H, ggml_context* ctx, const UsVaeCfg& cfg,
                                         const std::string& ap, ggml_tensor* query, ggml_tensor* data) {
    const int hd = cfg.head_dim(), nh = cfg.heads;
    ggml_tensor* q = lin(ctx, H.weight(ap + "c_q.weight"), nullptr, query);   // [width, Tq]
    ggml_tensor* kv = lin(ctx, H.weight(ap + "c_kv.weight"), nullptr, data);  // [2*width, Tk]
    ggml_tensor* qh = vs_to_heads(ctx, q, hd, nh);                // [hd, nh, Tq]
    ggml_tensor* k = vs_take_head_slice(ctx, kv, hd, nh, 2, 0);   // [hd, nh, Tk]
    ggml_tensor* v = vs_take_head_slice(ctx, kv, hd, nh, 2, 1);
    qh = us_qk_ln(ctx, qh, H.weight(ap + "attention.q_norm.weight"), H.weight(ap + "attention.q_norm.bias"), cfg.ln_eps);
    k  = us_qk_ln(ctx, k,  H.weight(ap + "attention.k_norm.weight"), H.weight(ap + "attention.k_norm.bias"), cfg.ln_eps);

    // USR_DECODE_FLASH (opt-in, default OFF -> byte-identical to prod): fuse this cross-attention with
    // ggml_flash_attn_ext instead of the dense QK^T -> soft_max_f32 -> AV path. nsys (TF32 off, prod)
    // put soft_max_f32 at 27% of the decode -- the dense path materialises a [Tk, Tq, heads] fp32
    // scores tensor (Tk=8192 latents, Tq=chunk) and streams it through a standalone softmax; flash
    // fuses all three into one kernel with no scores tensor (also the decode's VRAM peak). Independent
    // of PIXAL3D_FAST: fp32 ACCUMULATION is kept (set_prec F32), so the only precision change is F16
    // STORAGE of K/V -- occupancy is a threshold, so this is A/B'd as MESHES, not timed. The stale
    // m1_ggml.hpp comment "cross-attn's 5 kv tokens don't benefit" predates this 8192-kv call site.
    static const bool decode_flash = getenv("USR_DECODE_FLASH") != nullptr;
    if (decode_flash) {
        const int64_t d = qh->ne[0], nhd = qh->ne[1], tq = qh->ne[2], tk = k->ne[2];
        ggml_tensor* qf = ggml_cont(ctx, ggml_permute(ctx, qh, 0, 2, 1, 3));   // [d, tq, head] F32
        ggml_tensor* kf = ggml_cont(ctx, ggml_permute(ctx, k,  0, 2, 1, 3));   // [d, tk, head]
        ggml_tensor* vf = ggml_cont(ctx, ggml_permute(ctx, v,  0, 2, 1, 3));   // [d, tk, head]
        // V pre-scale by a power of 2 (EXACT in F16: shifts the exponent, zero precision loss) so the
        // flash PV accumulator can't overflow F16 -> +inf -> NaN, then undo on the output. Same fix as
        // m1_ggml.hpp's attention(). Tk=8192 and Tq=chunk are both multiples of 256, so a NULL mask is
        // safe (the documented null-mask MMA over-read only bites non-256 n_kv).
        const float vsc = 1.0f / 64.0f;
        vf = ggml_scale(ctx, vf, vsc);
        kf = ggml_cast(ctx, kf, GGML_TYPE_F16);
        vf = ggml_cast(ctx, vf, GGML_TYPE_F16);
        ggml_tensor* r = ggml_flash_attn_ext(ctx, qf, kf, vf, nullptr, cfg.attn_scale(), 0.0f, 0.0f);
        ggml_flash_attn_ext_set_prec(r, GGML_PREC_F32);
        ggml_tensor* o = ggml_cont_2d(ctx, r, d * nhd, tq);                    // [d,head,tq] -> [width, Tq]
        o = ggml_scale(ctx, o, 1.0f / vsc);
        (void)tk;
        return lin(ctx, H.weight(ap + "c_proj.weight"), H.weight(ap + "c_proj.bias"), o);
    }

    ggml_tensor* o = attention(ctx, qh, k, v, cfg.attn_scale());  // [width, Tq]
    return lin(ctx, H.weight(ap + "c_proj.weight"), H.weight(ap + "c_proj.bias"), o);
}

// ResidualAttentionBlock (self): x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)).
static inline ggml_tensor* us_self_block(M1Harness& H, ggml_context* ctx, const UsVaeCfg& cfg,
                                         const std::string& p, ggml_tensor* x) {
    ggml_tensor* x1 = layernorm(ctx, x, H.weight(p + "ln_1.weight"), H.weight(p + "ln_1.bias"), cfg.ln_eps);
    x = ggml_add(ctx, x, us_self_attn(H, ctx, cfg, p + "attn.", x1));
    ggml_tensor* x2 = layernorm(ctx, x, H.weight(p + "ln_2.weight"), H.weight(p + "ln_2.bias"), cfg.ln_eps);
    return ggml_add(ctx, x, us_mlp(H, ctx, cfg, p + "mlp.", x2));
}

// ResidualCrossAttentionBlock: x = x + attn(ln_1(x), ln_2(data)); x = x + mlp(ln_3(x)).
static inline ggml_tensor* us_cross_block(M1Harness& H, ggml_context* ctx, const UsVaeCfg& cfg,
                                          const std::string& p, ggml_tensor* x, ggml_tensor* data) {
    ggml_tensor* xn = layernorm(ctx, x, H.weight(p + "ln_1.weight"), H.weight(p + "ln_1.bias"), cfg.ln_eps);
    ggml_tensor* dn = layernorm(ctx, data, H.weight(p + "ln_2.weight"), H.weight(p + "ln_2.bias"), cfg.ln_eps);
    x = ggml_add(ctx, x, us_cross_attn(H, ctx, cfg, p + "attn.", xn, dn));
    ggml_tensor* x3 = layernorm(ctx, x, H.weight(p + "ln_3.weight"), H.weight(p + "ln_3.bias"), cfg.ln_eps);
    return ggml_add(ctx, x, us_mlp(H, ctx, cfg, p + "mlp.", x3));
}

// vae.forward(latents): post_kl -> 16x self-attn Transformer.  latents_in [embed_dim=64, K] -> [width=1024, K].
static inline ggml_tensor* us_vae_transformer(M1Harness& H, ggml_context* ctx, const UsVaeCfg& cfg,
                                              ggml_tensor* latents) {
    ggml_tensor* x = lin(ctx, H.weight(cfg.prefix + "post_kl.weight"), H.weight(cfg.prefix + "post_kl.bias"), latents);
    for (int i = 0; i < cfg.n_decoder_layers; ++i)
        x = us_self_block(H, ctx, cfg, cfg.prefix + "transformer.resblocks." + std::to_string(i) + ".", x);
    return x;   // [width, K]
}

// geo_decoder: query_embed [fourier_out=51, Nq] (host-fourier'd), latents [width, K] -> occupancy logits [1, Nq].
static inline ggml_tensor* us_geo_decoder(M1Harness& H, ggml_context* ctx, const UsVaeCfg& cfg,
                                          ggml_tensor* query_embed, ggml_tensor* latents) {
    const std::string g = cfg.prefix + "geo_decoder.";
    ggml_tensor* qe = lin(ctx, H.weight(g + "query_proj.weight"), H.weight(g + "query_proj.bias"), query_embed); // [width, Nq]
    ggml_tensor* x = us_cross_block(H, ctx, cfg, g + "cross_attn_decoder.", qe, latents);  // [width, Nq]
    x = layernorm(ctx, x, H.weight(g + "ln_post.weight"), H.weight(g + "ln_post.bias"), cfg.ln_eps);
    return lin(ctx, H.weight(g + "output_proj.weight"), H.weight(g + "output_proj.bias"), x); // [1, Nq]
}
