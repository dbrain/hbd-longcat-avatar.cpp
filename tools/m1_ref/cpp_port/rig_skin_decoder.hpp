// rig_skin_decoder.hpp — header-only ggml port of the SkinTokens skin-VAE (FSQ-CVAE)
// "R4" cond-encoder + skin-weight decoder (SkinFSQCVAEModel, Tripo2 cond_encoder/decoder).
// PURE-FP32, CPU (or CUDA via the M1Harness backend select). Reuses rig::Fsq (rig_fsq.hpp)
// and the m1_ggml.hpp attention/linear/layernorm helpers.
//
// Source dataflow (Python, all CONFIRMED from the checkpoint + code read):
//   src/model/skin_vae/autoencoders/skin_fsq_cvae_model.py  (_encode / _decode)
//   src/model/skin_vae/autoencoders/autoencoder_kl_tripo2.py (Tripo2Encoder / Tripo2Decoder)
//   src/model/skin_vae/transformers/tripo2_transformer.py    (DiTBlock)
//   src/model/skin_vae/attention_processor.py                (Tripo2AttnProcessor2_0)
//   src/model/skin_vae/embeddings.py                         (FrequencyPositionalEmbedding, use_pmpe)
//   src/model/tokenrig.py::decode (~L536)                    (token->index->code->decoder loop)
//
// Config (REAL, from grpo_1400.ckpt model_config.model):
//   width=768 heads=12 head_dim=64 latent=512  cond_channels=3
//   cond_encoder: 2 layers -> blocks = [cross(attn2)+ff, self(attn1)+ff, self(attn1)+ff] (3 blocks)
//   decoder:     10 layers -> blocks[0..9] = self(attn1)+ff ; blocks[10] = cross(attn2)+ff
//   embedder: FrequencyPositionalEmbedding(num_freqs=8, logspace, include_pi=True, include_input=True,
//             use_pmpe=True) -> out_dim 51 (=3*(8*2+1))
//   FSQ levels=[8,8,8,8,8] dim=512 (rig::Fsq).  attn scale=1/sqrt(64)=0.125.  all LayerNorm eps=1e-5.
//   to_q/to_k/to_v: NO bias ; to_out.0: bias.  cross-attn KV: norm_cross = LayerNorm(768) affine.
//   ff = net.0.proj (Linear 768->3072) -> GELU(exact erf) -> net.2 (Linear 3072->768).
//   qk_norm=False (no RMS on q/k) ; image_rotary_emb=None (no RoPE).
//
// DECODER DATAFLOW (the O(N) point question, RESOLVED — Tripo2Decoder.forward):
//   memory = post_quant( cat([z(4 code tokens), cond_latents(384)], dim=1) )   -> [388, 768]
//   for block in blocks[0..9]:  memory = self_attn_block(memory)               # O(388^2), CHEAP
//   kv_cache = memory
//   q = proj_query( embed_cat(query_points) )                                  # [N=8192, 768]
//   l = blocks[10] cross-attn: q attends kv_cache (388 keys)                   # O(N * 388), O(N) in pts
//   logits = proj_out( norm_out(l) )                                           # [N, 1]
//   skin_pred = sigmoid(logits)                                                # YES — sigmoid applied
//   stack J joints -> [N, J].
// => The 8192 points DO NOT self-attend; they enter only at the final cross-attn. O(N), CPU-cheap.
//
// ggml layout: a torch tensor [d0..dk] is ggml ne={dk..d0}. nn.Linear[out,in] -> ggml ne={in,out};
// ggml_mul_mat(W,x) with x ne0=in yields the Linear. Tokens are the ne1 (column) axis: [width, T].
#pragma once
#include "m1_ggml.hpp"
#include "rig_fsq.hpp"
#include "vecset_encoder.hpp"   // vs_take_head_slice (Tripo2 interleaved group head-split)
#include <string>
#include <vector>
#include <cmath>

namespace rig {

struct SkinVaeCfg {
    int width = 768;
    int heads = 12;
    int latent = 512;
    int cond_channels = 3;
    int num_freqs = 8;
    bool include_pi = true;
    bool include_input = true;
    bool use_pmpe = true;
    int n_cond_self = 2;     // cond_encoder self-attn blocks (after the cross block) -> blocks 1,2
    int n_dec_self = 10;     // decoder self-attn blocks (blocks 0..9)
    int tokens_per_skin = 4;
    float ln_eps = 1e-5f;
    std::string prefix = "model.";

    int head_dim() const { return width / heads; }
    float attn_scale() const { return 1.0f / std::sqrt((float) head_dim()); }
    // FrequencyPositionalEmbedding.out_dim = input_dim(3) * (num_freqs*2 + include_input)
    int embed_out() const { return 3 * (num_freqs * 2 + (include_input ? 1 : 0)); }
    // proj_in/proj_query input feature dim = cond_channels(3) + embed_out(51) = 54
    int in_feat() const { return cond_channels + embed_out(); }
};

// ---------------------------------------------------------------------------
// host Fourier/PMPE embed (FrequencyPositionalEmbedding, use_pmpe=True), exact.
//   freqs[k]   = 2^k * (pi if include_pi)                          (logspace)
//   phase[i]   = (num_freqs^(1-(i+1)/num_freqs) + (i+1)/num_freqs) * 2*pi
//   embed[c,k] = coord_c * freqs[k]
//   pphase[c,k]= coord_c * pi*0.5 + phase[k]
//   res = cat( sin(embed)+sin(pphase) , cos(embed)+cos(pphase) )   over (c,k) flattened c-major
//   out = cat( xyz , res )                                          (include_input)
// then we concat the point features (normals, cond_channels) -> [in_feat, *].
// Layout matches torch x[...,None]*freqs then .view(...,-1): index = c*num_freqs + k.
// Returns a flat [in_feat * N] column-major buffer (ggml ne={in_feat, N}).
// ---------------------------------------------------------------------------
static inline std::vector<float> skin_embed_with_feats(const float* pts, const float* feats,
                                                       int64_t N, const SkinVaeCfg& cfg) {
    const int F = cfg.num_freqs;
    std::vector<double> freq(F), phase(F);
    for (int k = 0; k < F; ++k) {
        double f = std::pow(2.0, (double) k);
        if (cfg.include_pi) f *= M_PI;
        freq[k] = f;
    }
    if (cfg.use_pmpe) {
        for (int i = 0; i < F; ++i) {
            double ph = std::pow((double) F, 1.0 - (double)(i + 1) / (double) F) + (double)(i + 1) / (double) F;
            phase[i] = ph * M_PI * 2.0;
        }
    }
    const int in_feat = cfg.in_feat();
    std::vector<float> out((size_t) N * in_feat);
    for (int64_t i = 0; i < N; ++i) {
        const float* p = pts + i * 3;
        float* o = out.data() + (size_t) i * in_feat;
        int w = 0;
        if (cfg.include_input) for (int c = 0; c < 3; ++c) o[w++] = p[c];        // xyz
        // sin block (c-major, then k)
        for (int c = 0; c < 3; ++c) for (int k = 0; k < F; ++k) {
            double e = (double) p[c] * freq[k];
            double s = std::sin(e);
            if (cfg.use_pmpe) s += std::sin((double) p[c] * M_PI * 0.5 + phase[k]);
            o[w++] = (float) s;
        }
        // cos block
        for (int c = 0; c < 3; ++c) for (int k = 0; k < F; ++k) {
            double e = (double) p[c] * freq[k];
            double cc = std::cos(e);
            if (cfg.use_pmpe) cc += std::cos((double) p[c] * M_PI * 0.5 + phase[k]);
            o[w++] = (float) cc;
        }
        if (feats) for (int c = 0; c < cfg.cond_channels; ++c) o[w++] = feats[i * 3 + c]; // normals
    }
    return out;
}

// ---------------------------------------------------------------------------
// ggml block helpers (mirror DiTBlock + Tripo2AttnProcessor2_0).
// All x tensors are [width, T] (ne0=width = the channel dim, ne1=T = tokens).
// ---------------------------------------------------------------------------

// FeedForward: net.2( gelu_erf( net.0.proj(x) ) ).
static inline ggml_tensor* sv_ff(M1Harness& H, ggml_context* ctx, const std::string& p, ggml_tensor* x) {
    ggml_tensor* h = lin(ctx, H.weight(p + "ff.net.0.proj.weight"), H.weight(p + "ff.net.0.proj.bias"), x);
    h = gelu_erf_(ctx, h);
    return lin(ctx, H.weight(p + "ff.net.2.weight"), H.weight(p + "ff.net.2.bias"), h);
}

// IMPORTANT — Tripo2AttnProcessor2_0 "split heads FIRST, then split qkv": q,k,v are concatenated
// on the channel dim BEFORE the head reshape, so head h does NOT get the standard contiguous
// q-slice [h*hd:(h+1)*hd]. For self-attn: qkv=cat(q,k,v) [3*width] -> view [heads, 3*hd] -> split;
// i.e. head h's q = flat[h*3hd : h*3hd+hd]. This is the INTERLEAVED group split implemented by
// vs_take_head_slice(packed, hd, heads, groups, g) (reshape [groups*hd, heads, T] then slice g).
// (Naive per-linear reshape is WRONG here — it cost ~0.5 maxabs in attn1.)
// For cross-attn: q IS standard (query reshaped alone), but kv=cat(k,v) uses the 2-group split.

// self-attn (attn1): to_q/to_k/to_v (no bias) -> Tripo2 interleaved 3-group head split -> attn.
static inline ggml_tensor* sv_self_attn(M1Harness& H, ggml_context* ctx, const SkinVaeCfg& cfg,
                                        const std::string& p, ggml_tensor* x) {
    const int hd = cfg.head_dim(), nh = cfg.heads;
    ggml_tensor* q = lin(ctx, H.weight(p + "attn1.to_q.weight"), nullptr, x);   // [width, T]
    ggml_tensor* k = lin(ctx, H.weight(p + "attn1.to_k.weight"), nullptr, x);
    ggml_tensor* v = lin(ctx, H.weight(p + "attn1.to_v.weight"), nullptr, x);
    ggml_tensor* qkv = ggml_concat(ctx, ggml_concat(ctx, q, k, 0), v, 0);       // [3*width, T]
    ggml_tensor* qh = vs_take_head_slice(ctx, qkv, hd, nh, 3, 0);               // [hd, nh, T]
    ggml_tensor* kh = vs_take_head_slice(ctx, qkv, hd, nh, 3, 1);
    ggml_tensor* vh = vs_take_head_slice(ctx, qkv, hd, nh, 3, 2);
    ggml_tensor* o  = attention(ctx, qh, kh, vh, cfg.attn_scale());             // [width, T]
    return lin(ctx, H.weight(p + "attn1.to_out.0.weight"), H.weight(p + "attn1.to_out.0.bias"), o);
}

// cross-attn (attn2): to_q(query) standard head split; KV = to_k/to_v(norm_cross(mem)) with the
// Tripo2 interleaved 2-group split.  query [width, Tq], mem [width, Tk].
static inline ggml_tensor* sv_cross_attn(M1Harness& H, ggml_context* ctx, const SkinVaeCfg& cfg,
                                         const std::string& p, ggml_tensor* query, ggml_tensor* mem) {
    const int hd = cfg.head_dim(), nh = cfg.heads;
    ggml_tensor* kv_in = layernorm(ctx, mem, H.weight(p + "attn2.norm_cross.weight"),
                                   H.weight(p + "attn2.norm_cross.bias"), cfg.ln_eps);
    ggml_tensor* q = lin(ctx, H.weight(p + "attn2.to_q.weight"), nullptr, query);   // [width, Tq]
    ggml_tensor* k = lin(ctx, H.weight(p + "attn2.to_k.weight"), nullptr, kv_in);   // [width, Tk]
    ggml_tensor* v = lin(ctx, H.weight(p + "attn2.to_v.weight"), nullptr, kv_in);
    ggml_tensor* qh = ggml_reshape_3d(ctx, q, hd, nh, q->ne[1]);                    // [hd, nh, Tq] (standard)
    ggml_tensor* kv = ggml_concat(ctx, k, v, 0);                                    // [2*width, Tk]
    ggml_tensor* kh = vs_take_head_slice(ctx, kv, hd, nh, 2, 0);                    // [hd, nh, Tk]
    ggml_tensor* vh = vs_take_head_slice(ctx, kv, hd, nh, 2, 1);
    ggml_tensor* o  = attention(ctx, qh, kh, vh, cfg.attn_scale());                 // [width, Tq]
    return lin(ctx, H.weight(p + "attn2.to_out.0.weight"), H.weight(p + "attn2.to_out.0.bias"), o);
}

// self-attn DiTBlock: x = x + attn1(norm1(x)); x = x + ff(norm3(x)).
static inline ggml_tensor* sv_self_block(M1Harness& H, ggml_context* ctx, const SkinVaeCfg& cfg,
                                         const std::string& p, ggml_tensor* x) {
    ggml_tensor* n1 = layernorm(ctx, x, H.weight(p + "norm1.weight"), H.weight(p + "norm1.bias"), cfg.ln_eps);
    x = ggml_add(ctx, x, sv_self_attn(H, ctx, cfg, p, n1));
    ggml_tensor* n3 = layernorm(ctx, x, H.weight(p + "norm3.weight"), H.weight(p + "norm3.bias"), cfg.ln_eps);
    return ggml_add(ctx, x, sv_ff(H, ctx, p, n3));
}

// cross-attn DiTBlock (encoder block0 / decoder block10):
//   x = x + attn2(norm2(x), mem); x = x + ff(norm3(x)).
static inline ggml_tensor* sv_cross_block(M1Harness& H, ggml_context* ctx, const SkinVaeCfg& cfg,
                                          const std::string& p, ggml_tensor* x, ggml_tensor* mem) {
    ggml_tensor* n2 = layernorm(ctx, x, H.weight(p + "norm2.weight"), H.weight(p + "norm2.bias"), cfg.ln_eps);
    x = ggml_add(ctx, x, sv_cross_attn(H, ctx, cfg, p, n2, mem));
    ggml_tensor* n3 = layernorm(ctx, x, H.weight(p + "norm3.weight"), H.weight(p + "norm3.bias"), cfg.ln_eps);
    return ggml_add(ctx, x, sv_ff(H, ctx, p, n3));
}

// ---------------------------------------------------------------------------
// ENTRY (a) cond-encoder: (cond_q_embed[in_feat, 384], cond_kv_embed[in_feat, N]) -> cond_latents[512, 384].
//   cond_q   = proj_in(cond_q_embed)         (the FPS-sampled 384 cond query points)
//   cond_kv  = proj_in(cond_kv_embed)         (the full point cloud, embedded)
//   blocks[0] cross-attn(cond_q <- cond_kv); blocks[1..2] self-attn; norm_out
//   cond_latents = cond_quant(...)            (Linear 768->512)
// NB: the 384 cond query points come from a nondeterministic FPS sample (numpy PCG64 + fps);
// to validate bit-exact you must feed the SAME sampled points the golden used. (Decoder check
// below uses the BANKED golden cond_latents and is the primary validation.)
// ---------------------------------------------------------------------------
static inline ggml_tensor* build_cond_encoder(M1Harness& H, ggml_context* ctx, const SkinVaeCfg& cfg,
                                              ggml_tensor* cond_q_embed, ggml_tensor* cond_kv_embed) {
    const std::string e = cfg.prefix + "cond_encoder.";
    ggml_tensor* kv = lin(ctx, H.weight(e + "proj_in.weight"), H.weight(e + "proj_in.bias"), cond_kv_embed); // [768, N]
    ggml_tensor* x  = lin(ctx, H.weight(e + "proj_in.weight"), H.weight(e + "proj_in.bias"), cond_q_embed);  // [768, 384]
    x = sv_cross_block(H, ctx, cfg, e + "blocks.0.", x, kv);
    for (int i = 0; i < cfg.n_cond_self; ++i)
        x = sv_self_block(H, ctx, cfg, e + "blocks." + std::to_string(i + 1) + ".", x);
    x = layernorm(ctx, x, H.weight(e + "norm_out.weight"), H.weight(e + "norm_out.bias"), cfg.ln_eps);
    return lin(ctx, H.weight(cfg.prefix + "cond_quant.weight"),
               H.weight(cfg.prefix + "cond_quant.bias"), x);   // [512, 384]
}

// ---------------------------------------------------------------------------
// ENTRY (b) decoder for ONE joint group:
//   z_codes [512, Tz]   (FSQ.indices_to_codes of the tokens_per_skin codes for this joint, as columns)
//   cond_lat [512, 384] (the cond latent memory)
//   q_embed  [in_feat, N] (the query points, host-embedded)
//   -> skin_pred [N] (after sigmoid).
// memory = post_quant( cat([z, cond], over tokens) ) -> [768, Tz+384]
// (cat over tokens => concat on ggml axis 1; z first then cond, matching torch cat dim=1).
// ---------------------------------------------------------------------------
static inline ggml_tensor* build_decoder_joint(M1Harness& H, ggml_context* ctx, const SkinVaeCfg& cfg,
                                               ggml_tensor* z_codes, ggml_tensor* cond_lat,
                                               ggml_tensor* q_embed) {
    const std::string d = cfg.prefix + "decoder.";
    ggml_tensor* mem_in = ggml_concat(ctx, z_codes, cond_lat, 1);    // [512, Tz+384]
    ggml_tensor* mem = lin(ctx, H.weight(cfg.prefix + "post_quant.weight"),
                           H.weight(cfg.prefix + "post_quant.bias"), mem_in);  // [768, Tz+384]
    for (int i = 0; i < cfg.n_dec_self; ++i)
        mem = sv_self_block(H, ctx, cfg, d + "blocks." + std::to_string(i) + ".", mem);
    // query stream
    ggml_tensor* q = lin(ctx, H.weight(d + "proj_query.weight"), H.weight(d + "proj_query.bias"), q_embed); // [768, N]
    ggml_tensor* l = sv_cross_block(H, ctx, cfg, d + "blocks.10.", q, mem);
    l = layernorm(ctx, l, H.weight(d + "norm_out.weight"), H.weight(d + "norm_out.bias"), cfg.ln_eps);
    ggml_tensor* logit = lin(ctx, H.weight(d + "proj_out.weight"), H.weight(d + "proj_out.bias"), l); // [1, N]
    return ggml_sigmoid(ctx, logit);   // [1, N]  (final activation = sigmoid, per Tripo2Decoder.forward)
}

} // namespace rig
