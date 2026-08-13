// vecset_encoder.hpp — ggml graph for the SkinTokens VecSet mesh-condition encoder
// (the R1 spike of HANDOFF-RIGGING-skintokens.md: the one genuinely-new GPU primitive;
// doubles as Hunyuan3D shape-VAE groundwork — both are the 3DShape2VecSet / Michelangelo
// perceiver family).
//
// Architecture (CONFIRMED from the proven checkpoint
// experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt, model_config.mesh_encoder,
// __target__=michelangelo_encoder => ShapeAsLatentPerceiverEncoder, no_query=True):
//
//   pc[N,3], feats(normals)[N,3]  --(host)-->  FPS-sampled queries sampled_pc[Q,3] (Q=token_num)
//   data        = input_proj( cat[ fourier(pc),        feats        ] )   # [N, width]  (kv)
//   sampled_data= input_proj( cat[ fourier(sampled_pc),sampled_feats] )   # [Q, width]  (query)
//   latents = ResidualCrossAttentionBlock(sampled_data, data)             # [Q, width]
//   latents = Transformer(num_encoder_layers x ResidualAttentionBlock)(latents)
//   latents = ln_post(latents)                                            # [Q, width]
//
// Exact hyperparameters (from the checkpoint, NOT the paper):
//   width=512  heads=8 (head_dim=64)  num_encoder_layers=8  num_freqs=8  include_pi=FALSE
//   include_input=TRUE  point_feats=3  token_num(Q)=512  mlp_width_scale=4  use_ln_post=TRUE
//   qkv_bias=FALSE (c_q/c_kv/c_qkv have NO bias; c_proj/input_proj/mlp DO)  ln eps=1e-5
//   GELU = nn.GELU() = exact erf gelu   attention scale = 1/sqrt(head_dim) (fp32 softmax)
//
// The Fourier embed is a deterministic per-point transform (no weights) -> done on the HOST in
// vecset_encode.cpp; this graph consumes the already-embedded [in=fourier_out+point_feats, *] data.
//
// Weight keys = the PyTorch state_dict suffixes under `prefix` (default the checkpoint's
// "mesh_encoder.encoder.") so the GGUF packer (R0/pack step) can write them verbatim.
#pragma once
#include "m1_ggml.hpp"
#include <string>
#include <cmath>

struct VecsetCfg {
    int width = 512;
    int heads = 8;
    int n_layers = 8;
    int num_freqs = 8;
    bool include_pi = false;
    bool include_input = true;
    int point_feats = 3;
    int mlp_scale = 4;
    bool use_ln_post = true;
    bool qkv_bias = false;
    float ln_eps = 1e-5f;
    std::string prefix = "mesh_encoder.encoder.";

    int head_dim() const { return width / heads; }
    // FourierEmbedder.out_dim = input_dim * (num_freqs*2 + (include_input?1:0))   (input_dim=3)
    int fourier_out() const { return 3 * (num_freqs * 2 + (include_input ? 1 : 0)); }
    int in_dim() const { return fourier_out() + point_feats; }   // input_proj in features
    float attn_scale() const { return 1.0f / std::sqrt((float) head_dim()); }
};

// Split a packed projection [groups*head_dim, heads, T] view into one [head_dim, heads, T] slice
// (group g of `groups`). Used for qkv (groups=3) and kv (groups=2). Returns a contiguous tensor.
static inline ggml_tensor* vs_take_head_slice(ggml_context* ctx, ggml_tensor* packed,
                                              int head_dim, int heads, int groups, int g) {
    int64_t T = packed->ne[1];                       // packed: [groups*head_dim*heads, T]
    // reshape to [groups*head_dim, heads, T]
    ggml_tensor* r = ggml_reshape_3d(ctx, packed, (int64_t)groups * head_dim, heads, T);
    ggml_tensor* s = ggml_view_3d(ctx, r, head_dim, heads, T, r->nb[1], r->nb[2],
                                  (size_t) g * head_dim * r->nb[0]);
    return ggml_cont(ctx, s);                         // [head_dim, heads, T]
}

// Reshape a [width, T] projection into [head_dim, heads, T] for attention().
static inline ggml_tensor* vs_to_heads(ggml_context* ctx, ggml_tensor* x, int head_dim, int heads) {
    return ggml_reshape_3d(ctx, x, head_dim, heads, x->ne[1]);
}

// MLP: c_proj( gelu_erf( c_fc(x) ) ).  x [width, T].
static inline ggml_tensor* vs_mlp(M1Harness& H, ggml_context* ctx, const VecsetCfg& cfg,
                                  const std::string& p, ggml_tensor* x) {
    ggml_tensor* h = lin(ctx, H.weight(p + "c_fc.weight"), H.weight(p + "c_fc.bias"), x);
    h = gelu_erf_(ctx, h);
    return lin(ctx, H.weight(p + "c_proj.weight"), H.weight(p + "c_proj.bias"), h);
}

// Self-attention (MultiheadAttention): c_qkv (no bias) -> split q,k,v -> attention -> c_proj (+bias).
static inline ggml_tensor* vs_self_attn(M1Harness& H, ggml_context* ctx, const VecsetCfg& cfg,
                                        const std::string& p, ggml_tensor* x) {
    const int hd = cfg.head_dim(), nh = cfg.heads;
    ggml_tensor* qkv = lin(ctx, H.weight(p + "c_qkv.weight"),
                           cfg.qkv_bias ? H.weight(p + "c_qkv.bias") : nullptr, x);  // [3*width, T]
    ggml_tensor* q = vs_take_head_slice(ctx, qkv, hd, nh, 3, 0);   // [hd, nh, T]
    ggml_tensor* k = vs_take_head_slice(ctx, qkv, hd, nh, 3, 1);
    ggml_tensor* v = vs_take_head_slice(ctx, qkv, hd, nh, 3, 2);
    ggml_tensor* o = attention(ctx, q, k, v, cfg.attn_scale());    // [width, T]
    return lin(ctx, H.weight(p + "c_proj.weight"), H.weight(p + "c_proj.bias"), o);
}

// Cross-attention (MultiheadCrossAttention): c_q(query) (no bias), c_kv(data) (no bias) -> attention -> c_proj.
static inline ggml_tensor* vs_cross_attn(M1Harness& H, ggml_context* ctx, const VecsetCfg& cfg,
                                         const std::string& p, ggml_tensor* query, ggml_tensor* data) {
    const int hd = cfg.head_dim(), nh = cfg.heads;
    ggml_tensor* q = lin(ctx, H.weight(p + "c_q.weight"),
                         cfg.qkv_bias ? H.weight(p + "c_q.bias") : nullptr, query);   // [width, Tq]
    ggml_tensor* kv = lin(ctx, H.weight(p + "c_kv.weight"),
                          cfg.qkv_bias ? H.weight(p + "c_kv.bias") : nullptr, data);  // [2*width, Tk]
    ggml_tensor* qh = vs_to_heads(ctx, q, hd, nh);                 // [hd, nh, Tq]
    ggml_tensor* k = vs_take_head_slice(ctx, kv, hd, nh, 2, 0);    // [hd, nh, Tk]
    ggml_tensor* v = vs_take_head_slice(ctx, kv, hd, nh, 2, 1);
    ggml_tensor* o = attention(ctx, qh, k, v, cfg.attn_scale());   // [width, Tq]
    return lin(ctx, H.weight(p + "c_proj.weight"), H.weight(p + "c_proj.bias"), o);
}

// ResidualCrossAttentionBlock: x = x + attn(ln_1(x), ln_2(data)); x = x + mlp(ln_3(x)).
static inline ggml_tensor* vs_cross_block(M1Harness& H, ggml_context* ctx, const VecsetCfg& cfg,
                                          const std::string& p, ggml_tensor* x, ggml_tensor* data) {
    ggml_tensor* xn = layernorm(ctx, x, H.weight(p + "ln_1.weight"), H.weight(p + "ln_1.bias"), cfg.ln_eps);
    ggml_tensor* dn = layernorm(ctx, data, H.weight(p + "ln_2.weight"), H.weight(p + "ln_2.bias"), cfg.ln_eps);
    x = ggml_add(ctx, x, vs_cross_attn(H, ctx, cfg, p + "attn.", xn, dn));
    ggml_tensor* x3 = layernorm(ctx, x, H.weight(p + "ln_3.weight"), H.weight(p + "ln_3.bias"), cfg.ln_eps);
    return ggml_add(ctx, x, vs_mlp(H, ctx, cfg, p + "mlp.", x3));
}

// ResidualAttentionBlock (self): x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)).
static inline ggml_tensor* vs_self_block(M1Harness& H, ggml_context* ctx, const VecsetCfg& cfg,
                                         const std::string& p, ggml_tensor* x) {
    ggml_tensor* x1 = layernorm(ctx, x, H.weight(p + "ln_1.weight"), H.weight(p + "ln_1.bias"), cfg.ln_eps);
    x = ggml_add(ctx, x, vs_self_attn(H, ctx, cfg, p + "attn.", x1));
    ggml_tensor* x2 = layernorm(ctx, x, H.weight(p + "ln_2.weight"), H.weight(p + "ln_2.bias"), cfg.ln_eps);
    return ggml_add(ctx, x, vs_mlp(H, ctx, cfg, p + "mlp.", x2));
}

// Full encoder. data_embed [in_dim, N] (kv, full point cloud), sampled_embed [in_dim, Q] (query).
// Both already Fourier-embedded + normals-concatenated on the host. Returns latents [width, Q].
static inline ggml_tensor* build_vecset_encoder(M1Harness& H, ggml_context* ctx, const VecsetCfg& cfg,
                                                ggml_tensor* data_embed, ggml_tensor* sampled_embed) {
    const std::string e = cfg.prefix;
    // shared input_proj (Linear in_dim -> width, with bias) applied to both kv data and queries.
    ggml_tensor* data = lin(ctx, H.weight(e + "input_proj.weight"), H.weight(e + "input_proj.bias"), data_embed);     // [width, N]
    ggml_tensor* x    = lin(ctx, H.weight(e + "input_proj.weight"), H.weight(e + "input_proj.bias"), sampled_embed);  // [width, Q]
    x = vs_cross_block(H, ctx, cfg, e + "cross_attn.", x, data);
    for (int i = 0; i < cfg.n_layers; ++i)
        x = vs_self_block(H, ctx, cfg, e + "self_attn.resblocks." + std::to_string(i) + ".", x);
    if (cfg.use_ln_post)
        x = layernorm(ctx, x, H.weight(e + "ln_post.weight"), H.weight(e + "ln_post.bias"), cfg.ln_eps);
    return x;   // [width, Q]
}
