#ifndef __NAVA_HPP__
#define __NAVA_HPP__

#include <cmath>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "core/ggml_extend.hpp"
#include "model/common/rope.hpp"

// NAVA — 6.3B joint audio-video MMDiT (ernie-research/NAVA) ported into the
// sd.cpp tree. This is the DiT backbone only (Phase 1 of HANDOFF-nava.md): a
// single forward that COMPILES + RUNS. Numeric correctness is validated
// separately by the human against a PyTorch bf16 reference — see the
// "VERIFIED FORWARD SPEC" section of HANDOFF-nava.md, which this file mirrors
// line-for-line.
//
// House style is copied from src/wan.hpp (WanSelfAttention / WanAttentionBlock /
// modulate_* / Wan / WanRunner). Tensor names match the GGUF
// (models/nava-dit-f16.gguf, prefix "backbone.") verbatim.
//
// ggml ne-order convention: a torch [N, L, dim] tensor is ggml ne [dim, L, N].

namespace NAVA {

    constexpr int NAVA_GRAPH_SIZE = 32768;

    struct NavaParams {
        int64_t dim         = 3072;
        int64_t ffn_dim     = 14336;
        int64_t num_heads   = 24;
        int64_t head_dim    = 128;  // dim / num_heads
        int num_double      = 10;
        int num_single      = 20;
        int64_t video_dim   = 48;
        int64_t audio_dim   = 128;
        int64_t text_dim    = 4096;
        int64_t text_len    = 512;
        int freq_dim        = 256;
        float eps           = 1e-6f;
        int theta           = 10000;
        // video patch[1,2,2]; temporal kernel was squeezed to 1.
        std::tuple<int, int, int> patch_size = {1, 2, 2};
        // RoPE split for the 64 complex dims (d_head/2): [T,H,W] = [22,21,21]
        // complex == [44,42,42] real. Audio rotates only the first 22 complex
        // dims, scaled by 0.24; remaining 42 complex pass through identity.
        int rope_t_pairs    = 22;
        int rope_h_pairs    = 21;
        int rope_w_pairs    = 21;
        float audio_freq_scale = 0.24f;
    };

    // ---------------------------------------------------------------------
    // Joint positional encoding (pe), built host-side.
    //
    // Produces the SAME packed buffer that Rope::embed_nd(Matrix) emits and that
    // Rope::apply_rope / ggml_rope_pe consume: a flat float array laid out as
    // ggml ne [2, 2, d_head/2, L_total], i.e. per (token, complex-pair) the 4
    // values [cos, -sin, sin, cos] (see Rope::rope() in rope.hpp lines 88-96).
    //
    // Video tokens (flatten order f-major, then h, then w over the patch grid):
    //   pairs   0..21 (T, 22 pairs): angle = f * freq_T[pair]
    //   pairs  22..42 (H, 21 pairs): angle = h * freq_H[pair-22]
    //   pairs  43..63 (W, 21 pairs): angle = w * freq_W[pair-43]
    //   (all UNSCALED, theta=10000)
    // Audio tokens (position p = 0..L_audio-1):
    //   pairs   0..21 (22 pairs): angle = p * (0.24 * freq_T[pair])
    //   pairs  22..63 (42 pairs): identity (cos=1, sin=0)
    //
    // freq for a sub-rope of real-dim `dim` (== 2*pairs): the house Rope::rope()
    // uses scale = linspace(0, (dim-2)/dim, dim/2) and omega = theta^(-scale),
    // i.e. omega[j] = theta^(-2j/dim). We reproduce that exactly so the joint pe
    // is bit-compatible with the in-tree RoPE path. (The HANDOFF wrote the freq
    // as 10000^(-2j/dim) which is the same thing.)
    __STATIC_INLINE__ std::vector<float> gen_nava_joint_pe(int f_len,
                                                           int h_len,
                                                           int w_len,
                                                           int audio_len,
                                                           const NavaParams& p) {
        const int d_half = static_cast<int>(p.head_dim / 2);  // 64 complex pairs
        const int t_pairs = p.rope_t_pairs;                   // 22
        const int h_pairs = p.rope_h_pairs;                   // 21
        const int w_pairs = p.rope_w_pairs;                   // 21
        GGML_ASSERT(t_pairs + h_pairs + w_pairs == d_half);

        auto make_omega = [&](int pairs) {
            // real dim = 2*pairs; omega[j] = theta^(-2j/dim)
            int dim = 2 * pairs;
            std::vector<float> omega(pairs);
            for (int j = 0; j < pairs; ++j) {
                float scale = (dim > 1) ? (2.f * j) / static_cast<float>(dim) : 0.f;
                omega[j]    = 1.0f / std::pow(static_cast<float>(p.theta), scale);
            }
            return omega;
        };
        std::vector<float> omega_t = make_omega(t_pairs);
        std::vector<float> omega_h = make_omega(h_pairs);
        std::vector<float> omega_w = make_omega(w_pairs);

        const int64_t L_vid   = (int64_t)f_len * h_len * w_len;
        const int64_t L_total = L_vid + audio_len;

        // Layout: for each token, d_half pairs, each pair 4 floats [cos,-sin,sin,cos].
        std::vector<float> pe((size_t)L_total * d_half * 4, 0.0f);

        auto write_pair = [&](size_t tok, int pair, float angle) {
            size_t base       = (tok * (size_t)d_half + (size_t)pair) * 4;
            float c           = std::cos(angle);
            float s           = std::sin(angle);
            pe[base + 0]      = c;
            pe[base + 1]      = -s;
            pe[base + 2]      = s;
            pe[base + 3]      = c;
        };

        // VIDEO tokens
        size_t tok = 0;
        for (int fi = 0; fi < f_len; ++fi) {
            for (int hi = 0; hi < h_len; ++hi) {
                for (int wi = 0; wi < w_len; ++wi, ++tok) {
                    for (int j = 0; j < t_pairs; ++j)
                        write_pair(tok, j, fi * omega_t[j]);
                    for (int j = 0; j < h_pairs; ++j)
                        write_pair(tok, t_pairs + j, hi * omega_h[j]);
                    for (int j = 0; j < w_pairs; ++j)
                        write_pair(tok, t_pairs + h_pairs + j, wi * omega_w[j]);
                }
            }
        }
        // AUDIO tokens
        for (int pi = 0; pi < audio_len; ++pi, ++tok) {
            for (int j = 0; j < t_pairs; ++j)
                write_pair(tok, j, pi * (p.audio_freq_scale * omega_t[j]));
            // remaining pairs: identity (cos=1, sin=0) — initialize explicitly so
            // the packed [cos,-sin,sin,cos] = [1,0,0,1].
            for (int j = t_pairs; j < d_half; ++j)
                write_pair(tok, j, 0.0f);
        }
        return pe;
    }

    // ---------------------------------------------------------------------
    // modulate helpers (mirror WAN::modulate_add / modulate_mul). e is the
    // per-stream modulation chunk, ggml ne [dim, 1, N] (broadcast over tokens).
    static ggml_tensor* mod_add(ggml_context* ctx, ggml_tensor* x, ggml_tensor* e) {
        return ggml_add(ctx, x, e);
    }
    static ggml_tensor* mod_mul(ggml_context* ctx, ggml_tensor* x, ggml_tensor* e) {
        return ggml_mul(ctx, x, e);
    }

    // ---------------------------------------------------------------------
    // Modulation 2-timestep collapse (bit-exact VRAM/speed win).
    //
    // The per-token I2V timestep has only TWO unique values: 0 for the clean
    // anchor prefix and `tval` for the rest. So AdaLN modulation is computed on a
    // COMPACT e [dim,6,2] (col0 = anchor t=0, col1 = tval) instead of the full
    // [dim,6,L_total]. After chunk6's add+chunk+permute each modulation chunk is a
    // compact [dim,2,1]; these helpers expand it per-token to [dim,L,1] with pure
    // copies (repeat/concat — no arithmetic), so the downstream op graph and bits
    // are byte-identical to the old per-token path.

    // Single-anchor 2-segment expand: first n_anchor tokens take col0 (anchor),
    // the remaining L-n_anchor take col1 (tval). c: compact [dim,2,1].
    static ggml_tensor* expand_mod_chunk(ggml_context* ctx, ggml_tensor* c,
                                         int64_t n_anchor, int64_t L) {
        int64_t dim = c->ne[0];
        GGML_ASSERT(c->ne[1] == 2);
        auto col0 = ggml_cont(ctx, ggml_view_3d(ctx, c, dim, 1, 1, c->nb[1], c->nb[2], 0));
        auto col1 = ggml_cont(ctx, ggml_view_3d(ctx, c, dim, 1, 1, c->nb[1], c->nb[2], c->nb[1]));
        if (n_anchor <= 0) {
            return ggml_repeat_4d(ctx, col1, dim, L, 1, 1);
        }
        if (n_anchor >= L) {
            return ggml_repeat_4d(ctx, col0, dim, L, 1, 1);
        }
        auto a = ggml_repeat_4d(ctx, col0, dim, n_anchor, 1, 1);
        auto r = ggml_repeat_4d(ctx, col1, dim, L - n_anchor, 1, 1);
        return ggml_concat(ctx, a, r, 1);  // [dim, L, 1]
    }

    // SingleBlock 4-segment expand over concat[video++audio]: video has its own
    // clean anchor (n_clean_i of L_vid), audio its own (n_anchor_a of L_aud), and
    // they are INDEPENDENT. c: compact [dim,2,1].
    static ggml_tensor* expand_mod_chunk_single(ggml_context* ctx, ggml_tensor* c,
                                                int64_t n_clean_i, int64_t L_vid,
                                                int64_t n_anchor_a, int64_t L_aud) {
        int64_t dim = c->ne[0];
        GGML_ASSERT(c->ne[1] == 2);
        auto col0 = ggml_cont(ctx, ggml_view_3d(ctx, c, dim, 1, 1, c->nb[1], c->nb[2], 0));
        auto col1 = ggml_cont(ctx, ggml_view_3d(ctx, c, dim, 1, 1, c->nb[1], c->nb[2], c->nb[1]));
        ggml_tensor* out = nullptr;
        auto append = [&](ggml_tensor* col, int64_t n) {
            if (n <= 0) return;
            auto seg = ggml_repeat_4d(ctx, col, dim, n, 1, 1);
            out = out ? ggml_concat(ctx, out, seg, 1) : seg;
        };
        append(col0, n_clean_i);             // video anchor prefix
        append(col1, L_vid - n_clean_i);     // video body
        append(col0, n_anchor_a);            // audio anchor prefix
        append(col1, L_aud - n_anchor_a);    // audio body
        GGML_ASSERT(out != nullptr && out->ne[1] == L_vid + L_aud);
        return out;  // [dim, L_total, 1]
    }

    // ---------------------------------------------------------------------
    // One self/cross attention "side" (q/k/v/o + qk RMSNorm). Mirrors
    // WanSelfAttention but exposed so the joint double-block can drive video and
    // audio sides separately and concat in the middle.
    class AttnProj : public GGMLBlock {
    public:
        int64_t dim;
        int64_t num_heads;
        int64_t head_dim;
        std::string suffix;  // "" for video side, "_audio" for audio side

        AttnProj(int64_t dim, int64_t num_heads, float eps, const std::string& suffix = "")
            : dim(dim), num_heads(num_heads), suffix(suffix) {
            head_dim                       = dim / num_heads;
            blocks["q" + suffix]           = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["k" + suffix]           = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["v" + suffix]           = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["o" + suffix]           = std::shared_ptr<GGMLBlock>(new Linear(dim, dim));
            blocks["norm_q" + suffix]      = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
            blocks["norm_k" + suffix]      = std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps));
        }

        // Project x -> q/k/v with qk-RMSNorm (over full dim, PRE head-reshape),
        // then reshape to heads: [head_dim, n_head, L, N].
        std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*>
        qkv(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q" + suffix]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k" + suffix]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v" + suffix]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q" + suffix]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k" + suffix]);

            int64_t N = x->ne[2];
            int64_t L = x->ne[1];

            auto q = norm_q->forward(ctx, q_proj->forward(ctx, x));
            auto k = norm_k->forward(ctx, k_proj->forward(ctx, x));
            auto v = v_proj->forward(ctx, x);

            q = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, L, N);
            k = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, L, N);
            v = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, L, N);
            return {q, k, v};
        }

        ggml_tensor* out(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o" + suffix]);
            return o_proj->forward(ctx, x);
        }

        // Public sub-block accessor (the double-block drives the "_audio" twins).
        std::shared_ptr<GGMLBlock> sub(const std::string& name) {
            auto it = blocks.find(name);
            return it == blocks.end() ? nullptr : it->second;
        }
        // Add a sub-block (used to graft the "_audio" twins onto self/cross attn).
        void add_sub(const std::string& name, std::shared_ptr<GGMLBlock> b) {
            blocks[name] = std::move(b);
        }

        // Project an arbitrary x -> q/k/v with a given suffix's linears+norms, then
        // reshape to heads. Reused for the audio twins in the double block.
        std::tuple<ggml_tensor*, ggml_tensor*, ggml_tensor*>
        qkv_suffix(GGMLRunnerContext* ctx, ggml_tensor* x, const std::string& sfx) {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q" + sfx]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k" + sfx]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v" + sfx]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q" + sfx]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k" + sfx]);
            int64_t N   = x->ne[2];
            int64_t L   = x->ne[1];
            auto q      = norm_q->forward(ctx, q_proj->forward(ctx, x));
            auto k      = norm_k->forward(ctx, k_proj->forward(ctx, x));
            auto v      = v_proj->forward(ctx, x);
            q           = ggml_reshape_4d(ctx->ggml_ctx, q, head_dim, num_heads, L, N);
            k           = ggml_reshape_4d(ctx->ggml_ctx, k, head_dim, num_heads, L, N);
            v           = ggml_reshape_4d(ctx->ggml_ctx, v, head_dim, num_heads, L, N);
            return {q, k, v};
        }

        ggml_tensor* out_suffix(GGMLRunnerContext* ctx, ggml_tensor* x, const std::string& sfx) {
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o" + sfx]);
            return o_proj->forward(ctx, x);
        }

        // Cross-attention with a given suffix.
        ggml_tensor* cross_suffix(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context, const std::string& sfx) {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q" + sfx]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k" + sfx]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v" + sfx]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o" + sfx]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q" + sfx]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k" + sfx]);
            auto q      = norm_q->forward(ctx, q_proj->forward(ctx, x));
            auto k      = norm_k->forward(ctx, k_proj->forward(ctx, context));
            auto v      = v_proj->forward(ctx, context);
            auto y      = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v,
                                                 num_heads, nullptr, false, ctx->flash_attn_enabled);
            return o_proj->forward(ctx, y);
        }

        // Cross-attention to a shared text context (no rope). x: [dim, Lq, N],
        // context: [dim, Lk, N].
        ggml_tensor* cross(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* context) {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["q" + suffix]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["k" + suffix]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["v" + suffix]);
            auto o_proj = std::dynamic_pointer_cast<Linear>(blocks["o" + suffix]);
            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q" + suffix]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k" + suffix]);

            auto q = norm_q->forward(ctx, q_proj->forward(ctx, x));
            auto k = norm_k->forward(ctx, k_proj->forward(ctx, context));
            auto v = v_proj->forward(ctx, context);

            auto y = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v,
                                            num_heads, nullptr, false, ctx->flash_attn_enabled);
            return o_proj->forward(ctx, y);
        }
    };

    // ---------------------------------------------------------------------
    // DOUBLE block (×10, "Hierarchical Alignment"). Separate video/audio self &
    // cross attention; SHARED ffn + norm3. Joint self-attn over cat[vid, aud].
    class DoubleBlock : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t ffn_dim;
        int64_t num_heads;
        float eps;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            // Modulation tables are used in elementwise add/mul against F32
            // activations; force F32 so the CPU backend's binary ops accept them
            // (the loader up-converts the F16 gguf tensor). On CUDA F16 would also
            // work, but F32 here keeps the harness backend-agnostic and is tiny.
            (void)tensor_storage_map;
            params["modulation.modulation"]       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 6, 1);
            params["modulation_audio.modulation"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 6, 1);
        }

    public:
        DoubleBlock(int64_t dim, int64_t ffn_dim, int64_t num_heads, float eps)
            : dim(dim), ffn_dim(ffn_dim), num_heads(num_heads), eps(eps) {
            // self_attn holds video proj + _audio twins in one GGMLBlock so names
            // land as self_attn.q / self_attn.q_audio etc.
            auto graft_audio = [&](std::shared_ptr<AttnProj>& ap) {
                ap->add_sub("q_audio", std::shared_ptr<GGMLBlock>(new Linear(dim, dim)));
                ap->add_sub("k_audio", std::shared_ptr<GGMLBlock>(new Linear(dim, dim)));
                ap->add_sub("v_audio", std::shared_ptr<GGMLBlock>(new Linear(dim, dim)));
                ap->add_sub("o_audio", std::shared_ptr<GGMLBlock>(new Linear(dim, dim)));
                ap->add_sub("norm_q_audio", std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps)));
                ap->add_sub("norm_k_audio", std::shared_ptr<GGMLBlock>(new RMSNorm(dim, eps)));
            };
            auto self_attn = std::make_shared<AttnProj>(dim, num_heads, eps, "");
            graft_audio(self_attn);
            blocks["self_attn"] = self_attn;

            auto cross_attn = std::make_shared<AttnProj>(dim, num_heads, eps, "");
            graft_audio(cross_attn);
            blocks["cross_attn"] = cross_attn;

            // SHARED ffn + norm3 (affine LayerNorm = cross_attn_norm). norm1/norm2
            // are param-free (AdaLN only) -> handled inline with ggml_ext_layer_norm.
            blocks["ffn.0"] = std::shared_ptr<GGMLBlock>(new Linear(dim, ffn_dim));
            blocks["ffn.2"] = std::shared_ptr<GGMLBlock>(new Linear(ffn_dim, dim));
            blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, true));
        }

        // returns 6 modulation chunks for the given e (ev or ea).
        // e: [dim, 6, Ne] — Ne==1 text-mode (uniform-t) / Ne==2 compact I2V
        // (col0=anchor t=0, col1=tval). modparam [dim,6,1] broadcasts over Ne.
        // n_anchor/L_stream drive the per-token expansion in the I2V (Ne==2) case.
        std::vector<ggml_tensor*> chunk6(GGMLRunnerContext* ctx, ggml_tensor* e, const std::string& which,
                                         int64_t n_anchor, int64_t L_stream) {
            auto modparam      = params[which];                          // [dim, 6, 1]
            e                  = ggml_add(ctx->ggml_ctx, e, modparam);   // [dim, 6, Ne]
            const bool compact = e->ne[2] == 2;                          // compact I2V (2-col) path; >2 = full per-token (debug)
            auto chunks        = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1); // 6 x [dim, 1, Ne]
            // Lay each chunk as [dim, Ne, 1]; then in I2V expand to per-token [dim, L_stream, 1].
            // When Ne==1 (text-mode / uniform-t) this is a no-op shape that broadcasts over
            // the token axis exactly like before — text-mode is unchanged.
            for (auto& c : chunks) {
                c = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, c, 0, 2, 1, 3));
                if (compact) {
                    c = expand_mod_chunk(ctx->ggml_ctx, c, n_anchor, L_stream);  // [dim, L_stream, 1]
                }
            }
            return chunks;
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,        // [dim, L_total, N]
                             int64_t L_vid,
                             ggml_tensor* e_vid,    // [dim, 6, Ne] (Ne=1 text / 2 compact I2V)
                             ggml_tensor* e_audio,  // [dim, 6, Ne]
                             ggml_tensor* pe,
                             ggml_tensor* context,
                             int64_t n_clean_i  = 0,    // video clean-anchor count (I2V)
                             int64_t n_anchor_a = 0,    // audio clean-anchor count (I2V)
                             bool joint_attn = true) {  // false = masking_modality (intra-modal only)
            int64_t L_total = x->ne[1];
            int64_t L_aud   = L_total - L_vid;

            auto self_attn  = std::dynamic_pointer_cast<AttnProj>(blocks["self_attn"]);
            auto cross_attn = std::dynamic_pointer_cast<AttnProj>(blocks["cross_attn"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);
            auto norm3      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);

            auto ev = chunk6(ctx, e_vid, "modulation.modulation", n_clean_i, L_vid);
            auto ea = chunk6(ctx, e_audio, "modulation_audio.modulation", n_anchor_a, L_aud);

            // split streams
            auto xv = ggml_ext_slice(ctx->ggml_ctx, x, 1, 0, L_vid);        // [dim, L_vid, N]
            auto xa = ggml_ext_slice(ctx->ggml_ctx, x, 1, L_vid, L_total);  // [dim, L_aud, N]

            // ---- joint self-attn ----
            // modulated norm1 (param-free LN) per stream
            auto xv_n = ggml_ext_layer_norm(ctx->ggml_ctx, xv, nullptr, nullptr, eps);
            xv_n      = mod_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, xv_n, mod_mul(ctx->ggml_ctx, xv_n, ev[1])), ev[0]);
            auto xa_n = ggml_ext_layer_norm(ctx->ggml_ctx, xa, nullptr, nullptr, eps);
            xa_n      = mod_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, xa_n, mod_mul(ctx->ggml_ctx, xa_n, ea[1])), ea[0]);

            (void)L_aud;
            // per-stream q/k/v (video uses "" suffix, audio uses "_audio").
            ggml_tensor *qv, *kv, *vv, *qa, *ka, *va;
            std::tie(qv, kv, vv) = self_attn->qkv_suffix(ctx, xv_n, "");
            std::tie(qa, ka, va) = self_attn->qkv_suffix(ctx, xa_n, "_audio");
            ggml_tensor *attn_v, *attn_a;
            if (joint_attn) {
                // concat over the token dim (ne dim 2): [head_dim, n_head, L, N]
                auto q = ggml_concat(ctx->ggml_ctx, qv, qa, 2);
                auto k = ggml_concat(ctx->ggml_ctx, kv, ka, 2);
                auto v = ggml_concat(ctx->ggml_ctx, vv, va, 2);
                // joint rope + attention over the whole sequence (B=1 unpadded -> no mask)
                auto attn = Rope::attention(ctx, q, k, v, pe, nullptr, 1.0f, true, ctx->flash_attn_enabled);  // [dim, L_total, N]
                attn_v = ggml_ext_slice(ctx->ggml_ctx, attn, 1, 0, L_vid);
                attn_a = ggml_ext_slice(ctx->ggml_ctx, attn, 1, L_vid, L_total);
            } else {
                // masking_modality (align_3d_cfg): SEPARATE intra-modal self-attention —
                // video tokens attend only to video, audio only to audio (no cross-modal).
                // The joint pe (ne[3]==L_total) is the per-token concat of video++audio
                // freqs, so slice it per stream to match each sub-sequence.
                auto pe_v   = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, L_vid);
                auto pe_a   = ggml_ext_slice(ctx->ggml_ctx, pe, 3, L_vid, L_total);
                attn_v      = Rope::attention(ctx, qv, kv, vv, pe_v, nullptr, 1.0f, true, ctx->flash_attn_enabled);
                attn_a      = Rope::attention(ctx, qa, ka, va, pe_a, nullptr, 1.0f, true, ctx->flash_attn_enabled);
            }
            auto yv     = self_attn->out_suffix(ctx, attn_v, "");
            auto ya_sa  = self_attn->out_suffix(ctx, attn_a, "_audio");
            // gated residual
            xv = ggml_add(ctx->ggml_ctx, xv, mod_mul(ctx->ggml_ctx, yv, ev[2]));
            xa = ggml_add(ctx->ggml_ctx, xa, mod_mul(ctx->ggml_ctx, ya_sa, ea[2]));

            // ---- per-stream cross-attn (UNGATED), to shared text context ----
            auto cv = cross_attn->cross_suffix(ctx, norm3->forward(ctx, xv), context, "");
            xv      = ggml_add(ctx->ggml_ctx, xv, cv);
            auto ca = cross_attn->cross_suffix(ctx, norm3->forward(ctx, xa), context, "_audio");
            xa      = ggml_add(ctx->ggml_ctx, xa, ca);

            // ---- shared FFN (modulated + gated) ----
            auto ffn = [&](ggml_tensor* xin, std::vector<ggml_tensor*>& e) {
                auto y = ggml_ext_layer_norm(ctx->ggml_ctx, xin, nullptr, nullptr, eps);
                y      = mod_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, y, mod_mul(ctx->ggml_ctx, y, e[4])), e[3]);
                y      = ffn_0->forward(ctx, y);
                y      = ggml_ext_gelu(ctx->ggml_ctx, y, true);
                y      = ffn_2->forward(ctx, y);
                return ggml_add(ctx->ggml_ctx, xin, mod_mul(ctx->ggml_ctx, y, e[5]));
            };
            xv = ffn(xv, ev);
            xa = ffn(xa, ea);

            return ggml_concat(ctx->ggml_ctx, xv, xa, 1);  // [dim, L_total, N]
        }
    };

    // ---------------------------------------------------------------------
    // SINGLE block (×20, "Unified Fusion"). ONE shared param set over the whole
    // concat[video ++ audio]. Self-attn still uses the joint pe.
    class SingleBlock : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t ffn_dim;
        int64_t num_heads;
        float eps;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            (void)tensor_storage_map;  // force F32 (see DoubleBlock::init_params note)
            params["modulation.modulation"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 6, 1);
        }

    public:
        SingleBlock(int64_t dim, int64_t ffn_dim, int64_t num_heads, float eps)
            : dim(dim), ffn_dim(ffn_dim), num_heads(num_heads), eps(eps) {
            blocks["self_attn"]  = std::make_shared<AttnProj>(dim, num_heads, eps, "");
            blocks["cross_attn"] = std::make_shared<AttnProj>(dim, num_heads, eps, "");
            blocks["ffn.0"]      = std::shared_ptr<GGMLBlock>(new Linear(dim, ffn_dim));
            blocks["ffn.2"]      = std::shared_ptr<GGMLBlock>(new Linear(ffn_dim, dim));
            blocks["norm3"]      = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,    // [dim, L_total, N]
                             ggml_tensor* e,    // [dim, 6, Ne] (Ne=1 text / 2 compact I2V)
                             ggml_tensor* pe,
                             ggml_tensor* context,
                             int64_t L_vid    = -1,      // video token count (masking split + I2V expand)
                             int64_t n_clean_i  = 0,     // video clean-anchor count (I2V)
                             int64_t n_anchor_a = 0,     // audio clean-anchor count (I2V)
                             bool joint_attn  = true) {  // false = masking_modality (intra-modal only)
            int64_t L = x->ne[1];
            int64_t N = x->ne[2];

            auto self_attn  = std::dynamic_pointer_cast<AttnProj>(blocks["self_attn"]);
            auto cross_attn = std::dynamic_pointer_cast<AttnProj>(blocks["cross_attn"]);
            auto ffn_0      = std::dynamic_pointer_cast<Linear>(blocks["ffn.0"]);
            auto ffn_2      = std::dynamic_pointer_cast<Linear>(blocks["ffn.2"]);
            auto norm3      = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);

            auto modparam      = params["modulation.modulation"];
            e                  = ggml_add(ctx->ggml_ctx, e, modparam);  // [dim,6,Ne] (Ne=1 text / 2 compact I2V)
            const bool compact = e->ne[2] == 2;                         // compact I2V (2-col) path; >2 = full per-token (debug)
            auto es            = ggml_ext_chunk(ctx->ggml_ctx, e, 6, 1);// 6 x [dim,1,Ne]
            // -> [dim, Ne, 1]; then in I2V expand to per-token [dim, L_total, 1] over the
            // concat[video++audio] sequence (video + audio carry INDEPENDENT clean anchors).
            // (no-op broadcast when Ne==1; text-mode unchanged.)
            for (auto& c : es) {
                c = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, c, 0, 2, 1, 3));
                if (compact) {
                    c = expand_mod_chunk_single(ctx->ggml_ctx, c, n_clean_i, L_vid, n_anchor_a, L - L_vid);
                }
            }

            // self-attn (joint pe over the whole sequence)
            auto y = ggml_ext_layer_norm(ctx->ggml_ctx, x, nullptr, nullptr, eps);
            y      = mod_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, y, mod_mul(ctx->ggml_ctx, y, es[1])), es[0]);
            ggml_tensor *q, *k, *v;
            std::tie(q, k, v) = self_attn->qkv(ctx, y);
            ggml_tensor* attn;
            if (joint_attn) {
                attn = Rope::attention(ctx, q, k, v, pe, nullptr, 1.0f, true, ctx->flash_attn_enabled);
            } else {
                // masking_modality: split q/k/v at L_vid (token axis ne[2]) + slice pe (ne[3])
                // -> separate intra-modal attention, then re-concat over the token dim.
                auto qv   = ggml_ext_slice(ctx->ggml_ctx, q, 2, 0, L_vid);
                auto kv   = ggml_ext_slice(ctx->ggml_ctx, k, 2, 0, L_vid);
                auto vv   = ggml_ext_slice(ctx->ggml_ctx, v, 2, 0, L_vid);
                auto qa   = ggml_ext_slice(ctx->ggml_ctx, q, 2, L_vid, L);
                auto ka   = ggml_ext_slice(ctx->ggml_ctx, k, 2, L_vid, L);
                auto va   = ggml_ext_slice(ctx->ggml_ctx, v, 2, L_vid, L);
                auto pe_v = ggml_ext_slice(ctx->ggml_ctx, pe, 3, 0, L_vid);
                auto pe_a = ggml_ext_slice(ctx->ggml_ctx, pe, 3, L_vid, L);
                auto av   = Rope::attention(ctx, qv, kv, vv, pe_v, nullptr, 1.0f, true, ctx->flash_attn_enabled);
                auto aa   = Rope::attention(ctx, qa, ka, va, pe_a, nullptr, 1.0f, true, ctx->flash_attn_enabled);
                attn      = ggml_concat(ctx->ggml_ctx, av, aa, 1);  // [dim, L_total, N]
            }
            attn              = self_attn->out(ctx, attn);
            x                 = ggml_add(ctx->ggml_ctx, x, mod_mul(ctx->ggml_ctx, attn, es[2]));

            // cross-attn (ungated)
            auto cx = cross_attn->cross(ctx, norm3->forward(ctx, x), context);
            x       = ggml_add(ctx->ggml_ctx, x, cx);

            // ffn (modulated + gated)
            auto yf = ggml_ext_layer_norm(ctx->ggml_ctx, x, nullptr, nullptr, eps);
            yf      = mod_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, yf, mod_mul(ctx->ggml_ctx, yf, es[4])), es[3]);
            yf      = ffn_0->forward(ctx, yf);
            yf      = ggml_ext_gelu(ctx->ggml_ctx, yf, true);
            yf      = ffn_2->forward(ctx, yf);
            x       = ggml_add(ctx->ggml_ctx, x, mod_mul(ctx->ggml_ctx, yf, es[5]));

            (void)L;
            (void)N;
            return x;
        }
    };

    // ---------------------------------------------------------------------
    // Head / head_audio. Param-free norm + Linear + modulation param [dim,2,1];
    // uses the per-token TIME emb `e` (not e0). e: [dim, N] or [dim, 1, N].
    class NavaHead : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t out_dim;
        float eps;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            (void)tensor_storage_map;  // force F32 (see DoubleBlock::init_params note)
            params["modulation"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 2, 1);
        }

    public:
        NavaHead(int64_t dim, int64_t out_dim, float eps)
            : dim(dim), out_dim(out_dim), eps(eps) {
            blocks["head"] = std::shared_ptr<GGMLBlock>(new Linear(dim, out_dim));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* e_time,
                             int64_t n_anchor = 0) {
            // x: [dim, L, N]; e_time: [dim, Ne]. The modulation is broadcast over the L tokens.
            auto head     = std::dynamic_pointer_cast<Linear>(blocks["head"]);
            auto modparam = params["modulation"];  // [dim,2,1]

            // e_time: [dim, Ne] — Ne==1 (text/uniform-t, broadcast over tokens) or
            // Ne==2 (compact I2V clean anchor: col0=t0, col1=tval). Build shift/scale.
            int64_t Ne = e_time->ne[1];
            auto e     = ggml_reshape_3d(ctx->ggml_ctx, e_time, dim, 1, Ne);  // [dim,1,Ne]
            e          = ggml_repeat_4d(ctx->ggml_ctx, e, dim, 2, Ne, 1);     // [dim,2,Ne]
            e          = ggml_add(ctx->ggml_ctx, e, modparam);                // [dim,2,Ne] (modparam [dim,2,1] broadcasts over Ne)
            auto es    = ggml_ext_chunk(ctx->ggml_ctx, e, 2, 1);              // 2 x [dim,1,Ne]
            // -> [dim, Ne, 1] so modulation applies per-token (broadcasts when Ne==1).
            auto shift = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, es[0], 0, 2, 1, 3));
            auto scale = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, es[1], 0, 2, 1, 3));
            if (Ne == 2) {  // compact I2V: expand the 2 columns to per-token [dim, L, 1]
                int64_t L = x->ne[1];
                shift = expand_mod_chunk(ctx->ggml_ctx, shift, n_anchor, L);
                scale = expand_mod_chunk(ctx->ggml_ctx, scale, n_anchor, L);
            }

            x = ggml_ext_layer_norm(ctx->ggml_ctx, x, nullptr, nullptr, eps);
            x = mod_add(ctx->ggml_ctx, ggml_add(ctx->ggml_ctx, x, mod_mul(ctx->ggml_ctx, x, scale)), shift);
            x = head->forward(ctx, x);
            return x;  // [out_dim, L, N]
        }
    };

    // ---------------------------------------------------------------------
    // The full NAVA backbone.
    class Nava : public GGMLBlock {
    protected:
        NavaParams params_;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            // patch_embedding video: gguf weight ne [2,2,48,3072] (kw,kh,IC,OC). We
            // load it raw and reshape to ggml_ext_conv_3d's [kw,kh,kd=1, IC*OC] at
            // forward time.
            enum ggml_type wt = get_type(prefix + "patch_embedding.weight", tensor_storage_map, GGML_TYPE_F16);
            params["patch_embedding.weight"] = ggml_new_tensor_4d(ctx, wt, std::get<2>(params_.patch_size), std::get<1>(params_.patch_size), params_.video_dim, params_.dim);
            params["patch_embedding.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params_.dim);

            // patch_embedding_audio.0: Conv1d(128->3072, k7) weight ne [7,128,3072]
            enum ggml_type wa0 = get_type(prefix + "patch_embedding_audio.0.weight", tensor_storage_map, GGML_TYPE_F16);
            params["patch_embedding_audio.0.weight"] = ggml_new_tensor_3d(ctx, wa0, 7, params_.audio_dim, params_.dim);
            params["patch_embedding_audio.0.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, params_.dim);
            // SwiGLU ConvMLP w1/w2/w3 (Conv1d k7, no bias)
            enum ggml_type wsw = get_type(prefix + "patch_embedding_audio.2.w1.weight", tensor_storage_map, GGML_TYPE_F16);
            params["patch_embedding_audio.2.w1.weight"] = ggml_new_tensor_3d(ctx, wsw, 7, params_.dim, 8192);
            params["patch_embedding_audio.2.w2.weight"] = ggml_new_tensor_3d(ctx, wsw, 7, 8192, params_.dim);
            params["patch_embedding_audio.2.w3.weight"] = ggml_new_tensor_3d(ctx, wsw, 7, params_.dim, 8192);
        }

    public:
        Nava() {}
        Nava(NavaParams p)
            : params_(p) {
            // text_embedding (Linear 4096->3072, GELU-tanh, Linear 3072->3072)
            blocks["text_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(p.text_dim, p.dim));
            blocks["text_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(p.dim, p.dim));

            // time_embedding (Linear 256->3072, SiLU, Linear 3072->3072)
            blocks["time_embedding.0"] = std::shared_ptr<GGMLBlock>(new Linear(p.freq_dim, p.dim));
            blocks["time_embedding.2"] = std::shared_ptr<GGMLBlock>(new Linear(p.dim, p.dim));

            // time_projection (SiLU, Linear 3072->18432)
            blocks["time_projection.1"] = std::shared_ptr<GGMLBlock>(new Linear(p.dim, p.dim * 6));

            for (int i = 0; i < p.num_double; ++i) {
                blocks["double_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new DoubleBlock(p.dim, p.ffn_dim, p.num_heads, p.eps));
            }
            for (int i = 0; i < p.num_single; ++i) {
                blocks["single_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new SingleBlock(p.dim, p.ffn_dim, p.num_heads, p.eps));
            }

            blocks["head"]       = std::shared_ptr<GGMLBlock>(new NavaHead(p.dim, p.video_dim * std::get<1>(p.patch_size) * std::get<2>(p.patch_size), p.eps));
            blocks["head_audio"] = std::shared_ptr<GGMLBlock>(new NavaHead(p.dim, p.audio_dim, p.eps));

            // speaker_embedding (SpkToken) — voice clone. model_mm.py SpkToken:
            //   net = Sequential(LayerNorm(192), Linear(192,dim), SiLU, Linear(dim,dim))
            //   out_norm = LayerNorm(dim); null_token [1,dim] (only used for the all-zero
            //   "fake" embedding, which we never splice — we splice REAL embeddings only).
            // The projected [dim] token replaces the <extra_id_2> position in the
            // text-embedded context (post text_embedding), feeding the audio stream's
            // shared context (model_mm.py:1515-1527,1644).
            blocks["speaker_embedding.net.0"]    = std::shared_ptr<GGMLBlock>(new LayerNorm(192));
            blocks["speaker_embedding.net.1"]    = std::shared_ptr<GGMLBlock>(new Linear(192, p.dim));
            blocks["speaker_embedding.net.3"]    = std::shared_ptr<GGMLBlock>(new Linear(p.dim, p.dim));
            blocks["speaker_embedding.out_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(p.dim));
        }

        const NavaParams& cfg() const { return params_; }

        // Public accessor for sub-blocks (the runner drives the block loop).
        std::shared_ptr<GGMLBlock> get_block(const std::string& name) {
            auto it = blocks.find(name);
            return it == blocks.end() ? nullptr : it->second;
        }

        // VIDEO patchify: Conv3d(48->3072, k[1,2,2], s[1,2,2]) on [W,H,F,48] (ggml ne)
        // -> [w', h', f', 3072] -> flatten to [3072, L_vid, N] (token order f-major,
        // then h, then w — matching the joint pe).
        ggml_tensor* patchify_video(GGMLRunnerContext* ctx, ggml_tensor* vid) {
            // vid ggml ne: [W, H, F, 48]
            ggml_tensor* w = params["patch_embedding.weight"];  // [2,2,48,3072]
            ggml_tensor* b = params["patch_embedding.bias"];
            // reshape weight to ggml_ext_conv_3d layout [kw,kh,kd=1, IC*OC]
            ggml_tensor* w3d = ggml_reshape_4d(ctx->ggml_ctx, w, w->ne[0], w->ne[1], 1, params_.video_dim * params_.dim);

            auto x = ggml_ext_conv_3d(ctx->ggml_ctx, vid, w3d, b, params_.video_dim,
                                      std::get<2>(params_.patch_size), std::get<1>(params_.patch_size), std::get<0>(params_.patch_size),
                                      0, 0, 0, 1, 1, 1);
            // x ggml ne: [w', h', f', dim] (N folded into last dim; here N=1)
            int64_t W2 = x->ne[0], H2 = x->ne[1], F2 = x->ne[2], D = x->ne[3];
            // -> [dim, w'*h'*f'] then transpose to [dim, L]
            x = ggml_reshape_3d(ctx->ggml_ctx, x, W2 * H2 * F2, D, 1);              // [w'h'f', dim, 1]
            x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [dim, w'h'f', 1]
            return x;
        }

        // AUDIO patchify: Conv1d(128->3072,k7,p3)->SiLU->SwiGLU ConvMLP. Input audio
        // ggml ne [128, Lraw] (channels-first per the conv) -> output [dim, L_audio, 1].
        ggml_tensor* patchify_audio(GGMLRunnerContext* ctx, ggml_tensor* audio) {
            // audio ggml ne [128, Lraw]: ggml_conv_1d wants x as [Lraw, IC=128, 1].
            auto x = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, audio, 1, 0, 2, 3));  // [Lraw,128]
            x      = ggml_reshape_3d(ctx->ggml_ctx, x, x->ne[0], params_.audio_dim, 1);                       // [Lraw,128,1]

            ggml_tensor* w0 = params["patch_embedding_audio.0.weight"];
            ggml_tensor* b0 = params["patch_embedding_audio.0.bias"];
            x               = ggml_conv_1d(ctx->ggml_ctx, w0, x, 1, 3, 1);  // [L, 3072, 1]
            x               = ggml_add(ctx->ggml_ctx, x, ggml_reshape_3d(ctx->ggml_ctx, b0, 1, params_.dim, 1));
            x               = ggml_silu(ctx->ggml_ctx, x);

            // SwiGLU ConvMLP: out = w2( silu(w1(x)) * w3(x) ), all Conv1d k7 p3, no bias.
            ggml_tensor* w1 = params["patch_embedding_audio.2.w1.weight"];
            ggml_tensor* w2 = params["patch_embedding_audio.2.w2.weight"];
            ggml_tensor* w3 = params["patch_embedding_audio.2.w3.weight"];
            auto g          = ggml_conv_1d(ctx->ggml_ctx, w1, x, 1, 3, 1);  // [L, 8192, 1]
            g               = ggml_silu(ctx->ggml_ctx, g);
            auto h          = ggml_conv_1d(ctx->ggml_ctx, w3, x, 1, 3, 1);  // [L, 8192, 1]
            auto gh         = ggml_mul(ctx->ggml_ctx, g, h);                // [L, 8192, 1]
            auto out        = ggml_conv_1d(ctx->ggml_ctx, w2, gh, 1, 3, 1); // [L, 3072, 1]

            // -> [dim, L_audio, 1]
            out = ggml_ext_cont(ctx->ggml_ctx, ggml_ext_torch_permute(ctx->ggml_ctx, out, 1, 0, 2, 3));  // [3072, L, 1]
            return out;
        }

        // Build the per-stream AdaLN e0 [dim,6,N] and the per-token time emb e [dim,N]
        // from a scalar timestep. For this first-draft harness we use a SINGLE shared
        // timestep per stream (no I2V clean-anchor per-token override yet — see TODO).
        std::pair<ggml_tensor*, ggml_tensor*> time_embed(GGMLRunnerContext* ctx, ggml_tensor* timestep) {
            auto te0  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.0"]);
            auto te2  = std::dynamic_pointer_cast<Linear>(blocks["time_embedding.2"]);
            auto tp1  = std::dynamic_pointer_cast<Linear>(blocks["time_projection.1"]);

            auto e = ggml_ext_timestep_embedding(ctx->ggml_ctx, timestep, params_.freq_dim);  // [256, N]
            e      = te0->forward(ctx, e);
            e      = ggml_silu_inplace(ctx->ggml_ctx, e);
            e      = te2->forward(ctx, e);  // [dim, N]

            auto e0 = ggml_silu(ctx->ggml_ctx, e);
            e0      = tp1->forward(ctx, e0);                                                  // [6*dim, N]
            e0      = ggml_reshape_3d(ctx->ggml_ctx, e0, params_.dim, 6, e0->ne[1]);          // [dim, 6, N]
            return {e0, e};
        }

        ggml_tensor* embed_text(GGMLRunnerContext* ctx, ggml_tensor* context) {
            auto t0 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.0"]);
            auto t2 = std::dynamic_pointer_cast<Linear>(blocks["text_embedding.2"]);
            // If context is already post-embed (dim==3072), pass through.
            if (context->ne[0] == params_.dim) {
                return context;
            }
            context = t0->forward(ctx, context);
            context = ggml_ext_gelu(ctx->ggml_ctx, context, true);
            context = t2->forward(ctx, context);
            return context;  // [dim, 512, N]
        }

        // Voice clone: project a speaker embedding [192,1] through SpkToken and splice
        // the resulting [dim,1] token into the (already text_embedded) context [dim,512,1]
        // at each spk position. We only call this with a REAL (nonzero) embedding, so the
        // null_token gating in model_mm.py SpkToken.forward is unneeded here.
        ggml_tensor* splice_spk(GGMLRunnerContext* ctx, ggml_tensor* context_e,
                                ggml_tensor* spk_emb, const std::vector<int>& spk_pos) {
            auto ln0   = std::dynamic_pointer_cast<LayerNorm>(blocks["speaker_embedding.net.0"]);
            auto l1    = std::dynamic_pointer_cast<Linear>(blocks["speaker_embedding.net.1"]);
            auto l3    = std::dynamic_pointer_cast<Linear>(blocks["speaker_embedding.net.3"]);
            auto onorm = std::dynamic_pointer_cast<LayerNorm>(blocks["speaker_embedding.out_norm"]);
            auto h = ln0->forward(ctx, spk_emb);   // [192,1]
            h      = l1->forward(ctx, h);          // [dim,1]
            h      = ggml_silu(ctx->ggml_ctx, h);
            h      = l3->forward(ctx, h);          // [dim,1]
            h      = onorm->forward(ctx, h);       // [dim,1] -- the spk token
            for (int p : spk_pos) {
                if (p < 0 || p >= (int)context_e->ne[1]) continue;
                context_e = ggml_set_2d(ctx->ggml_ctx, context_e, h, context_e->nb[1],
                                        (size_t)p * context_e->nb[1]);
            }
            return context_e;
        }
    };

    // ---------------------------------------------------------------------
    // Runner. CPU or CUDA backend; one forward. Captures per-block hidden states
    // + final velocities as debug tensors for the human's numeric diff.
    struct NavaRunner : public GGMLRunner {
        NavaParams nava_params;
        Nava nava;
        std::vector<float> pe_vec;
        sd::Tensor<float> t2_host;  // compact {0, tval} timestep (modulation 2-timestep collapse); must outlive compute
        std::string prefix;

        // grid sizes for the current forward (set by build_graph)
        int f_len = 0, h_len = 0, w_len = 0, audio_len = 0;
        int64_t L_vid = 0;
        // when set, build_graph emits ONE output = concat(vel_video, pad(vel_audio,192))
        // so compute_va() can recover BOTH stream velocities for joint denoising.
        bool return_joint = false;
        // when set, the self-attention runs SEPARATE intra-modal (video-only / audio-only)
        // instead of joint cross-modal — the align_3d_cfg "masking_modality" forward pass.
        bool mask_modality = false;
        // voice clone: when apply_spk, splice the projected speaker embedding (spk_emb_host
        // [192,1]) into the text-embedded context at spk_pos_v before the blocks.
        bool apply_spk = false;
        sd::Tensor<float> spk_emb_host;
        std::vector<int> spk_pos_v;

        NavaRunner(ggml_backend_t backend,
                   ggml_backend_t params_backend,
                   const String2TensorStorage& tensor_storage_map = {},
                   const std::string prefix                       = "backbone")
            : GGMLRunner(backend, params_backend), prefix(prefix) {
            nava = Nava(nava_params);
            nava.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return "nava"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            nava.get_param_tensors(tensors, prefix);
        }

        // x_video ggml ne [W,H,F,48]; audio ggml ne [128,Lraw]; context [text_dim or dim, 512, 1];
        // timestep scalar.
        ggml_cgraph* build_graph(const sd::Tensor<float>& video,
                                 const sd::Tensor<float>& audio,
                                 const sd::Tensor<float>& context,
                                 const sd::Tensor<float>& timestep) {
            ggml_cgraph* gf = new_graph_custom(NAVA_GRAPH_SIZE);

            ggml_tensor* vid = make_input(video);
            ggml_tensor* aud = make_input(audio);
            ggml_tensor* ctx_t = make_input(context);
            // timestep input is built compact below (after L_vid/audio_len are known).

            auto rc = get_context();
            // RENDER (joint) path: skip the validation-only debug-tensor taps. Their
            // ggml_cont+dup+cpy snapshot ops share the gallocr compute buffer and, when
            // compute_va() is the FIRST/only call type (as in the render loop), the
            // allocator reuses the joint output's video region for a snapshot scratch
            // → audio values leak into the video channels → the velocity is wrong by
            // ~25 dB every step → the trajectory under-denoises and explodes. The taps
            // are only consumed by the single-forward validation harness (compute(),
            // !return_joint), so disabling them here is free and makes compute_va()
            // == compute() regardless of gallocr reservation order.
            if (return_joint) {
                rc.debug_tensors = nullptr;
            }

            // grid from the conv output sizes
            int W = (int)vid->ne[0], H = (int)vid->ne[1], F = (int)vid->ne[2];
            int ph = std::get<1>(nava_params.patch_size), pw = std::get<2>(nava_params.patch_size);
            f_len = F;  // temporal patch=1
            h_len = H / ph;
            w_len = W / pw;
            // audio token count: Conv1d k7 p3 s1 preserves length. audio ggml ne is
            // [128, Lraw] -> Lraw == ne[1] == shape()[1].
            audio_len = (int)audio.shape()[1];

            L_vid = (int64_t)f_len * h_len * w_len;

            // joint pe -> ggml ne [2,2,d_head/2,L_total]
            pe_vec = gen_nava_joint_pe(f_len, h_len, w_len, audio_len, nava_params);
            int64_t L_total = L_vid + audio_len;
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, nava_params.head_dim / 2, L_total);
            set_backend_tensor_data(pe, pe_vec.data());

            // --- modulation 2-timestep collapse ---
            // The per-token I2V timestep has only 2 unique values: 0 (clean anchor) and
            // tval (rest). Feed time_embed a COMPACT {0, tval} so the AdaLN modulation
            // tensors stay [dim,6,2] instead of [dim,6,L_total]; the blocks expand the 2
            // columns back per-token with pure copies (bit-exact). Recover the per-stream
            // anchor counts host-side from the timestep we were handed.
            int64_t n_clean_i = 0, n_anchor_a = 0;
            int64_t dbg_anchor_idx = 0, dbg_tval_idx = 0;
            const bool per_token = (timestep.numel() > 1);
            ggml_tensor* t = nullptr;
            if (per_token) {
                GGML_ASSERT(timestep.numel() == L_total);
                const float* td = timestep.data();
                float tval = 0.0f;
                for (int64_t i = 0; i < L_total; ++i) {
                    if (td[i] != 0.0f) { tval = td[i]; break; }
                }
                while (n_clean_i < L_vid && td[n_clean_i] == 0.0f) ++n_clean_i;
                while (n_anchor_a < audio_len && td[L_vid + n_anchor_a] == 0.0f) ++n_anchor_a;
                // host-side indices of an anchor (t==0) and a tval token, for the
                // NAVA_MOD_FROM_FULL debug mode (slice the compact 2-col table out of the
                // full-width matmul output to test expansion logic vs matmul-width rounding).
                for (int64_t i = 0; i < L_total; ++i) { if (td[i] == 0.0f) { dbg_anchor_idx = i; break; } }
                for (int64_t i = 0; i < L_total; ++i) { if (td[i] != 0.0f) { dbg_tval_idx = i; break; } }
                t2_host = sd::Tensor<float>({2});
                t2_host.data()[0] = 0.0f;     // col0 = anchor (t=0)
                t2_host.data()[1] = tval;     // col1 = tval
                t = make_input(t2_host);
            } else {
                t = make_input(timestep);
            }

            // --- prepare ---
            auto x_vid   = nava.patchify_video(&rc, vid);    // [dim, L_vid, 1]
            auto x_aud   = nava.patchify_audio(&rc, aud);    // [dim, L_audio, 1]
            auto x       = ggml_concat(compute_ctx, x_vid, x_aud, 1);  // [dim, L_total, 1]

            auto [e0, e_time] = nava.time_embed(&rc, t);     // e0 [dim,6,Ne]; e_time [dim,Ne]
            // Ne==1: uniform-t (text-mode) — all streams share e0/e_time (broadcast).
            // Ne==2: compact I2V clean anchor (col0=t0, col1=tval). The blocks/heads
            // expand the 2 columns to per-token [dim,*,1] using the anchor counts above.
            // No per-stream slicing here anymore — that's the whole VRAM win.
            ggml_tensor* e_vid    = e0;
            ggml_tensor* e_audio  = e0;
            ggml_tensor* e_single = e0;
            ggml_tensor* e_head_v = e_time;
            ggml_tensor* e_head_a = e_time;

            // BIT-EXACT mode (NAVA_MOD_FROM_FULL=1): reproduce the OLD prod path's modulation
            // EXACTLY. The compact default runs the quantized time_projection matmul at width
            // 2 → ggml-cuda picks the MMVQ kernel (F32 activations), which is MORE accurate
            // than the old full-width MMQ path (int8-quantized activations) — i.e. the compact
            // latent moves TOWARD the PyTorch reference, not a bug. When exact reproduction of
            // the old C++ baseline is required, this mode runs the matmul at full width and
            // slices out the 2 unique columns (bit-identical, proven), at the cost of
            // materializing the full table transiently. Same downstream expansion either way.
            if (per_token && std::getenv("NAVA_MOD_FROM_FULL") != nullptr) {
                ggml_tensor* tf = make_input(timestep);
                auto [e0f, e_timef] = nava.time_embed(&rc, tf);  // full-width matmul (old prod)
                auto a6 = ggml_ext_slice(compute_ctx, e0f, 2, dbg_anchor_idx, dbg_anchor_idx + 1);
                auto v6 = ggml_ext_slice(compute_ctx, e0f, 2, dbg_tval_idx, dbg_tval_idx + 1);
                e0 = ggml_concat(compute_ctx, a6, v6, 2);        // [dim,6,2] compact, from full
                auto a1 = ggml_ext_slice(compute_ctx, e_timef, 1, dbg_anchor_idx, dbg_anchor_idx + 1);
                auto v1 = ggml_ext_slice(compute_ctx, e_timef, 1, dbg_tval_idx, dbg_tval_idx + 1);
                e_time = ggml_concat(compute_ctx, a1, v1, 1);    // [dim,2] compact, from full
                e_vid = e_audio = e_single = e0;
                e_head_v = e_head_a = e_time;
            }

            auto context_e = nava.embed_text(&rc, ctx_t);    // [dim, 512, 1]
            if (apply_spk && !spk_emb_host.empty() && !spk_pos_v.empty()) {
                ggml_tensor* spk_in = make_input(spk_emb_host);  // [192,1]
                context_e = nava.splice_spk(&rc, context_e, spk_in, spk_pos_v);
            }
            rc.capture_tensor("context_embedded", context_e);

            // --- double blocks ---
            for (int i = 0; i < nava_params.num_double; ++i) {
                auto blk = std::dynamic_pointer_cast<DoubleBlock>(
                    nava.get_block("double_blocks." + std::to_string(i)));
                x = blk->forward(&rc, x, L_vid, e_vid, e_audio, pe, context_e, n_clean_i, n_anchor_a, !mask_modality);
                rc.capture_tensor("double_blocks." + std::to_string(i), x);
            }

            // --- single blocks: ONE modulation set over the full concat[video++audio].
            // e_single is e0 (uniform-t) or the full per-token e0 [dim,6,L_total] (I2V). ---
            for (int i = 0; i < nava_params.num_single; ++i) {
                auto blk = std::dynamic_pointer_cast<SingleBlock>(
                    nava.get_block("single_blocks." + std::to_string(i)));
                x = blk->forward(&rc, x, e_single, pe, context_e, L_vid, n_clean_i, n_anchor_a, !mask_modality);
                rc.capture_tensor("single_blocks." + std::to_string(i), x);
            }

            // --- heads ---
            auto head       = std::dynamic_pointer_cast<NavaHead>(nava.get_block("head"));
            auto head_audio = std::dynamic_pointer_cast<NavaHead>(nava.get_block("head_audio"));

            auto x_v = ggml_ext_slice(compute_ctx, x, 1, 0, L_vid);
            auto x_a = ggml_ext_slice(compute_ctx, x, 1, L_vid, L_total);

            auto vel_v = head->forward(&rc, x_v, e_head_v, n_clean_i);       // [48*ph*pw, L_vid, 1]
            auto vel_a = head_audio->forward(&rc, x_a, e_head_a, n_anchor_a);// [128, L_audio, 1]

            rc.capture_tensor("velocity_video_patched", vel_v);
            rc.capture_tensor("velocity_audio", vel_a);

            if (return_joint) {
                // ONE output node carrying both stream velocities: pad audio ch
                // 128->192 and concat after the L_vid video tokens. compute_va()
                // splits it back host-side. (vel_v ne0=192=48*ph*pw, vel_a ne0=128.)
                int64_t vd     = vel_v->ne[0];
                // CRITICAL: protect vel_v / vel_a as graph outputs BEFORE building the
                // concat. Otherwise the gallocr treats them as dead once concat has
                // consumed them and may place the `joint` output buffer OVERLAPPING
                // vel_v's storage — then ggml_concat reads vel_v while writing an
                // aliased joint, corrupting the video velocity (~25 dB off EVERY step,
                // even step 0). compute() (non-joint) already set_output's these, which
                // is exactly why compute() was faithful while compute_va() was not.
                ggml_set_output(vel_v);
                ggml_set_output(vel_a);
                auto vel_a_pad = ggml_pad(compute_ctx, vel_a, (int)(vd - vel_a->ne[0]), 0, 0, 0);
                auto joint     = ggml_concat(compute_ctx, vel_v, vel_a_pad, 1);  // [192, L_total, 1]
                ggml_set_output(joint);
                ggml_build_forward_expand(gf, joint);
                if (std::getenv("NAVA_DIT_VRAM_HIST") != nullptr) {
                    int nn = ggml_graph_n_nodes(gf);
                    // top tensors by size + op histogram (probe what fills the compute buffer)
                    std::vector<std::pair<int64_t, ggml_tensor*>> sz;
                    std::map<int, int64_t> opbytes;
                    int64_t total = 0;
                    for (int i = 0; i < nn; i++) {
                        ggml_tensor* n = ggml_graph_node(gf, i);
                        int64_t b = (int64_t) ggml_nbytes(n);
                        sz.push_back({b, n});
                        opbytes[(int) n->op] += b;
                        total += b;
                    }
                    std::sort(sz.begin(), sz.end(), [](auto& a, auto& b) { return a.first > b.first; });
                    printf("=== NAVA DiT graph: %d nodes, sum-of-node-bytes %.1f MB (gallocr packs this into the compute buffer) ===\n",
                           nn, total / 1e6);
                    printf("  TOP 35 tensors by size:\n");
                    for (int i = 0; i < 35 && i < (int) sz.size(); i++) {
                        ggml_tensor* n = sz[i].second;
                        printf("   %7.1f MB  %-14s [%lld,%lld,%lld,%lld]\n", sz[i].first / 1e6,
                               ggml_op_name(n->op), (long long) n->ne[0], (long long) n->ne[1],
                               (long long) n->ne[2], (long long) n->ne[3]);
                    }
                    printf("  bytes by op:\n");
                    std::vector<std::pair<int64_t, int>> ob;
                    for (auto& kv : opbytes) ob.push_back({kv.second, kv.first});
                    std::sort(ob.begin(), ob.end(), [](auto& a, auto& b) { return a.first > b.first; });
                    for (int i = 0; i < 10 && i < (int) ob.size(); i++)
                        printf("   %7.1f MB  %s\n", ob[i].first / 1e6, ggml_op_name((ggml_op) ob[i].second));
                }
                return gf;
            }

            // NOTE: video unpatchify (token->[48,F,H,W]) is left to the human's diff
            // harness / sampler (Phase 2). We output the patched velocity so the
            // per-token numbers can be matched directly against the PyTorch
            // pre-unpatchify tensor.
            //
            // Expand audio FIRST then video, so the graph's FINAL output node — the
            // one GGMLRunner::compute returns — is the VIDEO velocity. (Audio is
            // still recoverable from the velocity_audio debug tap.)
            // ggml_set_output protects the returned buffer from gallocr reuse by the
            // appended debug-snapshot ops (else the return value is clobbered).
            ggml_set_output(vel_a);
            ggml_set_output(vel_v);
            ggml_build_forward_expand(gf, vel_a);
            ggml_build_forward_expand(gf, vel_v);
            return gf;
        }

        // Returns the final patched video velocity. Per-block hidden states and the
        // audio velocity are captured via the GGMLRunner debug-tensor path (their
        // mean/std/min/max are auto-logged during execute_graph), which is enough to
        // confirm the full forward executed. The human's numeric diff harness can
        // extend this to dump each captured tensor.
        sd::Tensor<float> compute(int n_threads,
                                  const sd::Tensor<float>& video,
                                  const sd::Tensor<float>& audio,
                                  const sd::Tensor<float>& context,
                                  const sd::Tensor<float>& timestep) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(video, audio, context, timestep);
            };
            auto r = GGMLRunner::compute<float>(get_graph, n_threads, false);
            return r.value_or(sd::Tensor<float>());
        }

        // Joint forward: returns {patched video velocity [192,L_vid], audio velocity
        // [128,L_audio]} from ONE forward so the sampler can denoise both streams in
        // lockstep (NAVA is a joint AV model — the video coheres only when the audio
        // stream is co-denoised at the matching noise level).
        std::pair<sd::Tensor<float>, sd::Tensor<float>>
        compute_va(int n_threads,
                   const sd::Tensor<float>& video,
                   const sd::Tensor<float>& audio,
                   const sd::Tensor<float>& context,
                   const sd::Tensor<float>& timestep,
                   bool mask_modality_arg = false,
                   bool apply_spk_arg = false,
                   const sd::Tensor<float>* spk_emb_arg = nullptr,
                   const std::vector<int>* spk_pos_arg = nullptr) {
            return_joint  = true;
            mask_modality = mask_modality_arg;
            apply_spk     = apply_spk_arg && spk_emb_arg && !spk_emb_arg->empty()
                            && spk_pos_arg && !spk_pos_arg->empty();
            if (apply_spk) { spk_emb_host = *spk_emb_arg; spk_pos_v = *spk_pos_arg; }
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(video, audio, context, timestep);
            };
            auto r       = GGMLRunner::compute<float>(get_graph, n_threads, false);
            return_joint  = false;
            mask_modality = false;
            apply_spk     = false;
            sd::Tensor<float> joint = r.value_or(sd::Tensor<float>());  // [192, L_total]
            if (joint.empty()) {
                return {sd::Tensor<float>(), sd::Tensor<float>()};
            }
            const int64_t vd = joint.shape()[0];  // 192
            const int64_t Lv = L_vid;
            const int64_t La = audio_len;
            sd::Tensor<float> vv({vd, Lv});
            sd::Tensor<float> va({(int64_t)128, La});
            const float* js = joint.data();  // ne0 (vd) fastest: idx(c,tok)=tok*vd+c
            for (int64_t tk = 0; tk < Lv; ++tk) {
                for (int64_t c = 0; c < vd; ++c) {
                    vv.data()[tk * vd + c] = js[tk * vd + c];
                }
            }
            for (int64_t tk = 0; tk < La; ++tk) {
                for (int64_t c = 0; c < 128; ++c) {
                    va.data()[tk * 128 + c] = js[(Lv + tk) * vd + c];
                }
            }
            return {std::move(vv), std::move(va)};
        }
    };

}  // namespace NAVA

#endif  // __NAVA_HPP__
