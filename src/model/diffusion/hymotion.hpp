#ifndef __SD_MODEL_DIFFUSION_HYMOTION_HPP__
#define __SD_MODEL_DIFFUSION_HYMOTION_HPP__

// ---------------------------------------------------------------------------
// HY-Motion 1.0 (Tencent Hunyuan) -- text -> humanoid motion, native ggml port.
//
//   prompt --[Qwen3-8B hidden states + CLIP-L pooled]--> MMDiT --[Euler x50]-->
//   latent [T,201] --> denorm --> rot6d local joint rotations + root translation
//
// WHAT THIS IS: a Flux/HunyuanVideo-family flow-matching MMDiT -- 9 double-stream
// + 18 single-stream blocks, feat_dim 1280, 20 heads. NO VQ, NO VAE, no motion
// tokenizer: continuous flow matching straight on the raw 201-dim frame feature.
// 360 motion tokens + 128 text tokens = 488. Tiny.
//
// THE 201-DIM FRAME (config.yml: input_dim: 201; paper eq. f in R^201):
//     [  0:  3]  root translation                       3
//     [  3:  9]  root_rot6d   (global body orientation) 6
//     [  9:135]  21 local joint rotations as rot6d    126   <- the whole point
//     [135:201]  22 joint positions   (DECODER IGNORES) 66
//   Skeleton is SMPL-H's first 22 joints, NO HANDS (body_model.py pads 30
//   identity rotations for fingers).
//
// RELATIONSHIP TO flux.hpp -- READ THIS BEFORE "SIMPLIFYING" ANYTHING:
// Structurally MMSingleStreamBlock is Flux::SingleStreamBlock and
// MMDoubleStreamBlock is Flux::DoubleStreamBlock, with identical shapes and
// identical modulation semantics. They are re-implemented here rather than
// reused because of four concrete divergences, each of which would be a silent
// wrong-output bug if you just called the Flux block:
//
//   1. WEIGHT NAMES differ throughout (HY `q_norm` vs Flux `norm.query_norm`,
//      HY `norm` vs Flux `pre_norm`, HY `modulation.linear` vs Flux
//      `modulation.lin`, HY `motion_mlp.fc1` vs Flux `img_mlp.0`). The GGUF
//      keeps HY's names verbatim (see tools/convert_hymotion.py) precisely so
//      this mapping is checkable by eye instead of buried in a converter.
//   2. CONCAT ORDER IS REVERSED. Flux attends over [txt, img]; HY attends over
//      [motion, text]. Same math, different token order -- so the RoPE table and
//      the mask must be built for the [motion, text] order too.
//   3. Flux::RMSNorm names its parameter `scale`; HY's checkpoint says `weight`.
//      We use the RMSNorm from ggml_extend.hpp, which uses `weight`.
//   4. The additive narrowband/padding mask (below) has no Flux analogue.
//
// Everything genuinely heavy IS reused: Rope::rope / Rope::attention (including
// the fused ggml_rope_pe CUDA kernel), ggml_ext_attention_ext, Linear/LayerNorm/
// RMSNorm, and the GGMLBlock/GGMLRunner loader plumbing.
//
// ---------------------------------------------------------------------------
// FOUR TRAPS THAT WILL SILENTLY PRODUCE PLAUSIBLE-BUT-WRONG MOTION.
// Every one of these is a thing the reference does that you would not guess.
//
// TRAP 1 -- `apply_rope_to_single_branch: false` MEANS THE OPPOSITE OF HOW IT READS.
//   The flag name suggests "don't rope the single-stream branch". It actually
//   selects WHICH tokens get RoPE:
//       true  -> RoPE applied to the motion branch ALONE (before concat)
//       false -> RoPE applied to the FULL CONCATENATED [motion, text] stream
//   config.yml sets **false**, so text tokens carry temporal RoPE positions
//   360..487, continuing the motion index. In MMSingleStreamBlock the reference's
//   q1/q2 split-then-concat is a literal no-op under `false`. We implement the
//   `false` path only (rope_on_concat), which is exactly Flux's behaviour.
//
// TRAP 2 -- THE rot6d LAYOUT IS INTERLEAVED, NOT [0:3]+[3:6].
//   utils/geometry.py contains TWO mutually-transposed conventions. The decoder
//   uses `rot6d_to_rotation_matrix`, which does `rot6d.view(...,3,2)` and takes
//   a1 = x[...,0], a2 = x[...,1]:
//       a1 = (rot6d[0], rot6d[2], rot6d[4])     <- NOT rot6d[0:3]
//       a2 = (rot6d[1], rot6d[3], rot6d[5])     <- NOT rot6d[3:6]
//   and stacks the Gram-Schmidt basis as COLUMNS (dim=-1), so
//       R = [b1 | b2 | b3],  i.e. rot6d is the first two COLUMNS of R,
//       flattened row-major: rot6d = [R00,R01,R10,R11,R20,R21].
//   The other function in the same file (`rotation_6d_to_matrix`, pytorch3d
//   style) uses d6[:3]/d6[3:] and stacks as ROWS. It is NOT used by the decoder.
//   Reaching for the standard pytorch3d convention yields a perfectly valid
//   rotation matrix that is simply the wrong rotation -- no crash, no NaN, just
//   subtly broken motion. See hymotion_decode.hpp.
//
// TRAP 3 -- TWO DIFFERENT time_factors.
//   The main DiT's TimestepEmbeddingEncoder is built with time_factor=1000.0
//   (config.yml). The token refiner's is built WITHOUT passing time_factor, so
//   it gets the constructor default **1.0**. Same class, same t, different
//   sinusoidal input. Getting this wrong perturbs only the text conditioning,
//   which is exactly the kind of error that looks like "the model is a bit bad".
//
// TRAP 4 -- narrowband is a +/-60 FRAME BAND, and the number is not in config.yml.
//   mask_mode: narrowband, but `narrowband_length` is absent from config.yml, so
//   it takes the code default 2.0, which __init__ multiplies by 30.0 -> a window
//   of 60 frames. Motion self-attention is banded to |i-j| <= 60 (+/-2 seconds).
//   Additionally T->M attention is hard-disabled: text queries never see motion
//   keys, while motion queries DO see text. The band applies ONLY to the [M->M]
//   quadrant.
//
// Sources, all verified against the shipped code + the 1.0B checkpoint:
//   hymotion/network/hymotion_mmdit.py, network/{encoders,token_refiner,
//   modulate_layers,positional_encoding,attention,bricks}.py,
//   pipeline/motion_diffusion.py, utils/geometry.py, HY-Motion-1.0/config.yml.
// ---------------------------------------------------------------------------

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "ggml-cpu.h"

#include "core/ggml_extend.hpp"
#include "model/common/rope.hpp"

#define HYMOTION_GRAPH_SIZE 8192

namespace HYMotion {

    struct Params {
        int64_t input_dim         = 201;
        int64_t feat_dim          = 1280;
        int64_t ctxt_input_dim    = 4096;   // Qwen3-8B hidden
        int64_t vtxt_input_dim    = 768;    // CLIP-L pooled
        int64_t num_heads         = 20;
        int64_t head_dim          = 64;
        int64_t double_blocks     = 9;      // num_layers // 3
        int64_t single_blocks     = 18;     // num_layers - num_layers // 3
        int64_t refiner_blocks    = 2;
        float mlp_ratio           = 4.0f;
        float time_factor         = 1000.0f;  // main DiT only -- see TRAP 3
        float rope_theta          = 10000.0f;
        int64_t train_frames      = 360;    // always denoise 360 and crop
        int64_t max_length_llm    = 128;
        int64_t narrowband_window = 60;     // +/-60 frames -- see TRAP 4
        int64_t fps               = 30;
        int64_t num_joints        = 22;     // SMPL-H body, no hands

        int64_t mlp_hidden() const { return (int64_t)(feat_dim * mlp_ratio); }
    };

    // modulate(x, shift, scale) = x * (1 + scale) + shift   [modulate_layers.py]
    __STATIC_INLINE__ ggml_tensor* modulate(ggml_context* ctx,
                                            ggml_tensor* x,
                                            ggml_tensor* shift,
                                            ggml_tensor* scale) {
        // x: [C, L, N], shift/scale: [C, 1, N] -> broadcast over L
        x = ggml_add(ctx, x, ggml_mul(ctx, x, scale));
        x = ggml_add(ctx, x, shift);
        return x;
    }

    // apply_gate(x, gate) = x * gate   [modulate_layers.py, tanh=False]
    __STATIC_INLINE__ ggml_tensor* apply_gate(ggml_context* ctx, ggml_tensor* x, ggml_tensor* gate) {
        return ggml_mul(ctx, x, gate);
    }

    // ModulateDiT: linear(silu(x)) -> [factor*C], then .chunk(factor, dim=-1).
    // chunk i is the contiguous slice [i*C, (i+1)*C) of the last dim, which in
    // ggml (ne[0] fastest) is a view at byte offset i*C.
    struct ModulateDiT : public GGMLBlock {
        int64_t dim;
        int factor;

        ModulateDiT(int64_t dim, int factor)
            : dim(dim), factor(factor) {
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(dim, dim * factor, true));
        }

        // vec: [C, 1, N] -> factor tensors of [C, 1, N]
        std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* vec) {
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);
            auto out    = linear->forward(ctx, ggml_silu(ctx->ggml_ctx, vec));  // [factor*C, 1, N]

            std::vector<ggml_tensor*> chunks;
            chunks.reserve(factor);
            for (int i = 0; i < factor; ++i) {
                chunks.push_back(ggml_view_3d(ctx->ggml_ctx, out,
                                              dim, out->ne[1], out->ne[2],
                                              out->nb[1], out->nb[2],
                                              (size_t)i * dim * out->nb[0]));
            }
            return chunks;
        }
    };

    // encoders.py MLP: fc1 -> act -> fc2
    struct MLP : public GGMLBlock {
        bool gelu_tanh;  // main blocks: gelu_tanh. refiner: silu.

        MLP(int64_t in_dim, int64_t hidden, int64_t out_dim, bool gelu_tanh)
            : gelu_tanh(gelu_tanh) {
            blocks["fc1"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, hidden, true));
            // FFN down-proj: 1/128 pre-scale, the same anti-overflow tool ZImage's w2 and
            // block.hpp's net.2 use, for the same reason. With F16 weights the CUDA backend
            // casts the F32 activation to F16 and accumulates in F16 (PREC_DEFAULT ->
            // cublasGemmEx COMPUTE_16F), so a large-K GEMM can cross the 65504 ceiling ->
            // inf -> NaN, while CPU (F32 accumulate) stays clean. MEASURED on this model:
            // the DiT went 100% NaN on CUDA for every t >= ~0.335 (clean below, CPU clean
            // everywhere) because activations grow with t; the scale is an exact identity
            // (x*s -> GEMM -> x/s, bias added after) and removes the overflow at its source.
            blocks["fc2"] = std::shared_ptr<GGMLBlock>(new Linear(hidden, out_dim, true, false, false, 1.f / 128.f));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto fc1 = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
            auto fc2 = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);
            x        = fc1->forward(ctx, x);
            x        = gelu_tanh ? ggml_ext_gelu(ctx->ggml_ctx, x, true) : ggml_silu(ctx->ggml_ctx, x);
            x        = fc2->forward(ctx, x);
            return x;
        }
    };

    // encoders.py MLPEncoder(num_layers=2): Linear -> act -> Linear.
    // nn.Sequential indices give the weight names linears.0 / linears.2.
    struct MLPEncoder : public GGMLBlock {
        MLPEncoder(int64_t in_dim, int64_t feat_dim) {
            blocks["linears.0"] = std::shared_ptr<GGMLBlock>(new Linear(in_dim, feat_dim, true));
            blocks["linears.2"] = std::shared_ptr<GGMLBlock>(new Linear(feat_dim, feat_dim, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            auto l0 = std::dynamic_pointer_cast<Linear>(blocks["linears.0"]);
            auto l2 = std::dynamic_pointer_cast<Linear>(blocks["linears.2"]);
            x       = l0->forward(ctx, x);
            x       = ggml_silu(ctx->ggml_ctx, x);  // act_type="silu" at both call sites
            x       = l2->forward(ctx, x);
            return x;
        }
    };

    // encoders.py TimestepEmbeddingEncoder: Linear -> SiLU -> Linear over a
    // sinusoidal embedding. The sinusoid itself is computed host-side and passed
    // in (see sinusoidal_embedding) -- it is 1280 floats per step and doing it on
    // the host keeps the exact cos-then-sin ordering obvious.
    struct TimestepEmbeddingEncoder : public GGMLBlock {
        TimestepEmbeddingEncoder(int64_t embedding_dim, int64_t feat_dim) {
            blocks["blocks.0"] = std::shared_ptr<GGMLBlock>(new Linear(embedding_dim, feat_dim, true));
            blocks["blocks.2"] = std::shared_ptr<GGMLBlock>(new Linear(feat_dim, feat_dim, true));
        }

        // sinu: [embedding_dim, 1, N] -> [feat_dim, 1, N]
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* sinu) {
            auto b0 = std::dynamic_pointer_cast<Linear>(blocks["blocks.0"]);
            auto b2 = std::dynamic_pointer_cast<Linear>(blocks["blocks.2"]);
            auto x  = b0->forward(ctx, sinu);
            x       = ggml_silu(ctx->ggml_ctx, x);  // act_type="silu" at both call sites
            x       = b2->forward(ctx, x);
            return x;
        }
    };

    // encoders.py TimestepEmbeddingEncoder.sinusodial_embedding [sic].
    //   timesteps *= time_factor
    //   freqs[i]   = exp(-log(10000) * i / half)          i in [0, half)
    //   emb        = cat([cos(t*freqs), sin(t*freqs)])    <-- COS FIRST
    // Identical to Flux's timestep_embedding (including time_factor=1000).
    __STATIC_INLINE__ std::vector<float> sinusoidal_embedding(float t,
                                                              int64_t dim,
                                                              float time_factor,
                                                              float temperature = 10000.0f) {
        const float tt      = t * time_factor;
        const int64_t half  = dim / 2;
        std::vector<float> emb((size_t)dim, 0.0f);
        for (int64_t i = 0; i < half; ++i) {
            const float freq = std::exp(-std::log(temperature) * (float)i / (float)half);
            const float arg  = tt * freq;
            emb[(size_t)i]          = std::cos(arg);
            emb[(size_t)(half + i)] = std::sin(arg);
        }
        // dim is even for feat_dim 1280; the reference zero-pads odd dims.
        return emb;
    }

    // -----------------------------------------------------------------------
    // token_refiner.py -- HunyuanVideo-style text refiner over the Qwen3 stream.
    //
    // NOTE the two deviations from the main blocks, both from constructor
    // defaults that the caller never overrides:
    //   qk_norm_type = "layer"  -> LayerNorm (weight AND bias, confirmed in the
    //                              ckpt) on head_dim, NOT the RMSNorm the main
    //                              blocks use.
    //   mlp_act_type = "silu"   -> NOT the gelu_tanh the main blocks use.
    // Also: adaLN here has factor=2 and yields TWO GATES (gate_msa, gate_mlp) --
    // there is no shift/scale and no modulate() call, unlike FinalLayer's
    // factor=2 which yields (shift, scale).
    // -----------------------------------------------------------------------
    struct IndividualTokenRefinerBlock : public GGMLBlock {
        int64_t num_heads, head_dim, feat_dim;

        IndividualTokenRefinerBlock(int64_t feat_dim, int64_t num_heads, int64_t mlp_hidden)
            : num_heads(num_heads), head_dim(feat_dim / num_heads), feat_dim(feat_dim) {
            blocks["norm1"]             = std::shared_ptr<GGMLBlock>(new LayerNorm(feat_dim, 1e-6f, true));
            blocks["self_attn_qkv"]     = std::shared_ptr<GGMLBlock>(new Linear(feat_dim, feat_dim * 3, true));
            blocks["self_attn_q_norm"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(head_dim, 1e-6f, true));
            blocks["self_attn_k_norm"]  = std::shared_ptr<GGMLBlock>(new LayerNorm(head_dim, 1e-6f, true));
            blocks["self_attn_proj"]    = std::shared_ptr<GGMLBlock>(new Linear(feat_dim, feat_dim, true));
            blocks["norm2"]             = std::shared_ptr<GGMLBlock>(new LayerNorm(feat_dim, 1e-6f, true));
            blocks["mlp"]               = std::shared_ptr<GGMLBlock>(new MLP(feat_dim, mlp_hidden, feat_dim, /*gelu_tanh*/ false));
            blocks["adaLN_modulation"]  = std::shared_ptr<GGMLBlock>(new ModulateDiT(feat_dim, 2));
        }

        // x: [C, L, N]; c: [C, 1, N]; mask: [L, L] additive F32 or null
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* c, ggml_tensor* mask) {
            auto ctx0        = ctx->ggml_ctx;
            auto norm1       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto qkv_l       = std::dynamic_pointer_cast<Linear>(blocks["self_attn_qkv"]);
            auto proj        = std::dynamic_pointer_cast<Linear>(blocks["self_attn_proj"]);
            auto norm2       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto mlp         = std::dynamic_pointer_cast<MLP>(blocks["mlp"]);
            auto adaLN       = std::dynamic_pointer_cast<ModulateDiT>(blocks["adaLN_modulation"]);

            auto gates    = adaLN->forward(ctx, c);  // {gate_msa, gate_mlp}
            auto gate_msa = gates[0];
            auto gate_mlp = gates[1];

            const int64_t L = x->ne[1], N = x->ne[2];

            auto norm_x = norm1->forward(ctx, x);
            auto qkv    = qkv_l->forward(ctx, norm_x);  // [3C, L, N]

            // "B L (K H D) -> K B L H D": K is the SLOWEST of the three, so q/k/v
            // are the three contiguous C-sized slabs of the last dim, and within a
            // slab the layout is head-major (h outer, d inner) == ggml [d_head,
            // n_head, ...].
            //
            // LAYOUT CONTRACT -- this block is the ONLY place we call
            // ggml_ext_attention_ext directly instead of through Rope::attention,
            // because the refiner has no RoPE. Rope::attention passes
            // skip_reshape=TRUE, which is only valid because apply_rope() has
            // already restructured q/k into [d_head, L, n_head*N] (and leaves v as
            // [d_head, n_head, L, N] -- an asymmetric layout that is easy to walk
            // into). With no rope we must instead use skip_reshape=FALSE and hand
            // the wrapper FLAT [C, L, N] tensors so it does the head split itself.
            // Passing already-split [d_head, n_head, L, N] with skip_reshape=false
            // makes the wrapper read C=64 and compute d_head = 64/20 = 3 (integer
            // division), then reshape 163840 elements into 153600 -> the
            // ggml.c:3725 nelements assert.
            //
            // q/k still need the per-head LayerNorm over head_dim, so: split ->
            // norm over ne[0]=head_dim -> flatten back to [C, L, N]. The flatten is
            // exact (the tensor is contiguous and head-major), and the wrapper's
            // reshape_4d(d_head, n_head, L, N) re-splits it identically.
            auto split_norm = [&](int i, std::shared_ptr<GGMLBlock> nrm) {
                auto t = ggml_ext_cont(ctx0, ggml_view_4d(ctx0, qkv, head_dim, num_heads, L, N,
                                                          qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2],
                                                          (size_t)i * feat_dim * qkv->nb[0]));
                t      = std::dynamic_pointer_cast<LayerNorm>(nrm)->forward(ctx, t);  // over head_dim
                return ggml_reshape_3d(ctx0, t, feat_dim, L, N);
            };
            auto q = split_norm(0, blocks["self_attn_q_norm"]);
            auto k = split_norm(1, blocks["self_attn_k_norm"]);
            auto v = ggml_ext_cont(ctx0, ggml_view_3d(ctx0, qkv, feat_dim, L, N,
                                                      qkv->nb[1], qkv->nb[2],
                                                      (size_t)2 * feat_dim * qkv->nb[0]));

            // No RoPE in the refiner -> skip_reshape=false, flat [C, L, N] in.
            auto attn = ggml_ext_attention_ext(ctx0, ctx->backend, q, k, v, num_heads, mask,
                                               /*skip_reshape*/ false, /*flash_attn*/ false);
            x         = ggml_add(ctx0, x, apply_gate(ctx0, proj->forward(ctx, attn), gate_msa));
            x         = ggml_add(ctx0, x, apply_gate(ctx0, mlp->forward(ctx, norm2->forward(ctx, x)), gate_mlp));
            return x;
        }
    };

    struct SingleTokenRefiner : public GGMLBlock {
        Params p;

        SingleTokenRefiner(const Params& p)
            : p(p) {
            blocks["input_embedder"]  = std::shared_ptr<GGMLBlock>(new Linear(p.feat_dim, p.feat_dim, true));
            blocks["context_encoder"] = std::shared_ptr<GGMLBlock>(new MLPEncoder(p.feat_dim, p.feat_dim));
            blocks["timestep_encoder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbeddingEncoder(p.feat_dim, p.feat_dim));
            for (int64_t i = 0; i < p.refiner_blocks; ++i) {
                blocks["individual_token_refiner.blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new IndividualTokenRefinerBlock(p.feat_dim, p.num_heads, p.mlp_hidden()));
            }
        }

        // x:        [C, L_txt, N]   (already through ctxt_encoder)
        // t_sinu:   [C, 1, N]       sinusoidal(t, feat_dim, time_factor=1.0)  <- TRAP 3
        // mean_w:   [L_txt, 1, N]   normalised mask weights for the masked mean
        // attn_mask:[L_txt, L_txt]  additive, or null
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* t_sinu,
                             ggml_tensor* mean_w,
                             ggml_tensor* attn_mask) {
            auto ctx0 = ctx->ggml_ctx;
            auto emb  = std::dynamic_pointer_cast<Linear>(blocks["input_embedder"]);
            auto cenc = std::dynamic_pointer_cast<MLPEncoder>(blocks["context_encoder"]);
            auto tenc = std::dynamic_pointer_cast<TimestepEmbeddingEncoder>(blocks["timestep_encoder"]);

            auto t_rep = tenc->forward(ctx, t_sinu);  // [C, 1, N]

            // context_aware = (x * mask).sum(1) / mask.sum(1)  -- we pass the
            // already-normalised weights so this is a plain weighted sum, i.e. a
            // [C,L]x[L,1] matmul per batch.
            auto ctx_rep = ggml_mul_mat(ctx0, ggml_ext_cont(ctx0, ggml_transpose(ctx0, x)), mean_w);
            ctx_rep      = ggml_reshape_3d(ctx0, ctx_rep, x->ne[0], 1, x->ne[2]);  // [C, 1, N]
            ctx_rep      = cenc->forward(ctx, ctx_rep);

            auto c = ggml_add(ctx0, t_rep, ctx_rep);  // [C, 1, N]

            x = emb->forward(ctx, x);
            for (int64_t i = 0; i < p.refiner_blocks; ++i) {
                auto blk = std::dynamic_pointer_cast<IndividualTokenRefinerBlock>(
                    blocks["individual_token_refiner.blocks." + std::to_string(i)]);
                x = blk->forward(ctx, x, c, attn_mask);
            }
            return x;
        }
    };

    // -----------------------------------------------------------------------
    // MMDoubleStreamBlock == Flux::DoubleStreamBlock (reversed concat order).
    // -----------------------------------------------------------------------
    struct MMDoubleStreamBlock : public GGMLBlock {
        int64_t num_heads, head_dim, feat_dim;

        MMDoubleStreamBlock(int64_t feat_dim, int64_t num_heads, int64_t mlp_hidden)
            : num_heads(num_heads), head_dim(feat_dim / num_heads), feat_dim(feat_dim) {
            for (const std::string s : {"motion", "text"}) {
                blocks[s + "_mod"]      = std::shared_ptr<GGMLBlock>(new ModulateDiT(feat_dim, 6));
                blocks[s + "_norm1"]    = std::shared_ptr<GGMLBlock>(new LayerNorm(feat_dim, 1e-6f, false));
                blocks[s + "_qkv"]      = std::shared_ptr<GGMLBlock>(new Linear(feat_dim, feat_dim * 3, true));
                blocks[s + "_q_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-6f));
                blocks[s + "_k_norm"]   = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-6f));
                blocks[s + "_out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(feat_dim, feat_dim, true));
                blocks[s + "_norm2"]    = std::shared_ptr<GGMLBlock>(new LayerNorm(feat_dim, 1e-6f, false));
                blocks[s + "_mlp"]      = std::shared_ptr<GGMLBlock>(new MLP(feat_dim, mlp_hidden, feat_dim, /*gelu_tanh*/ true));
            }
        }

        // Build q/k/v [head_dim, num_heads, L, N] from a [3C, L, N] qkv.
        std::vector<ggml_tensor*> qkv_split(ggml_context* ctx0, ggml_tensor* qkv, int64_t L, int64_t N) {
            std::vector<ggml_tensor*> out;
            for (int i = 0; i < 3; ++i) {
                out.push_back(ggml_ext_cont(ctx0, ggml_view_4d(ctx0, qkv, head_dim, num_heads, L, N,
                                                               qkv->nb[0] * head_dim, qkv->nb[1], qkv->nb[2],
                                                               (size_t)i * feat_dim * qkv->nb[0])));
            }
            return out;
        }

        // motion: [C, L_m, N]; text: [C, L_t, N]; adapter: [C, 1, N]
        // pe: RoPE table over the CONCATENATED [motion, text] stream (TRAP 1)
        std::vector<ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                          ggml_tensor* motion,
                                          ggml_tensor* text,
                                          ggml_tensor* adapter,
                                          ggml_tensor* pe,
                                          ggml_tensor* mask) {
            auto ctx0 = ctx->ggml_ctx;
            auto B    = [&](const std::string& n) { return blocks[n]; };

            auto m_mod = std::dynamic_pointer_cast<ModulateDiT>(B("motion_mod"))->forward(ctx, adapter);
            auto t_mod = std::dynamic_pointer_cast<ModulateDiT>(B("text_mod"))->forward(ctx, adapter);
            // chunk order: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
            const int SH = 0, SC = 1, GA = 2, SH2 = 3, SC2 = 4, GA2 = 5;

            const int64_t L_m = motion->ne[1], L_t = text->ne[1], N = motion->ne[2];

            auto m_norm1 = std::dynamic_pointer_cast<LayerNorm>(B("motion_norm1"));
            auto m_mod_x = modulate(ctx0, m_norm1->forward(ctx, motion), m_mod[SH], m_mod[SC]);
            auto m_qkv   = qkv_split(ctx0, std::dynamic_pointer_cast<Linear>(B("motion_qkv"))->forward(ctx, m_mod_x), L_m, N);
            m_qkv[0]     = std::dynamic_pointer_cast<RMSNorm>(B("motion_q_norm"))->forward(ctx, m_qkv[0]);
            m_qkv[1]     = std::dynamic_pointer_cast<RMSNorm>(B("motion_k_norm"))->forward(ctx, m_qkv[1]);

            auto t_norm1 = std::dynamic_pointer_cast<LayerNorm>(B("text_norm1"));
            auto t_mod_x = modulate(ctx0, t_norm1->forward(ctx, text), t_mod[SH], t_mod[SC]);
            auto t_qkv   = qkv_split(ctx0, std::dynamic_pointer_cast<Linear>(B("text_qkv"))->forward(ctx, t_mod_x), L_t, N);
            t_qkv[0]     = std::dynamic_pointer_cast<RMSNorm>(B("text_q_norm"))->forward(ctx, t_qkv[0]);
            t_qkv[1]     = std::dynamic_pointer_cast<RMSNorm>(B("text_k_norm"))->forward(ctx, t_qkv[1]);

            // TRAP 1 / divergence 2: HY concatenates (motion, text) -- the REVERSE
            // of Flux's (txt, img) -- and RoPE is then applied to the whole thing.
            auto q = ggml_concat(ctx0, m_qkv[0], t_qkv[0], 2);
            auto k = ggml_concat(ctx0, m_qkv[1], t_qkv[1], 2);
            auto v = ggml_concat(ctx0, m_qkv[2], t_qkv[2], 2);

            // The spike passed a per-call /*flash*/ false here. Master's Rope::attention
            // dropped that parameter and reads ctx->flash_attn_enabled instead, which is
            // off unless a runner turns it on -- and HYMotionRunner never does. Same
            // behaviour, one fewer knob; if flash is ever enabled on this runner, the
            // masked branch needs re-checking before trusting it.
            auto attn = Rope::attention(ctx, q, k, v, pe, mask, 1.0f, /*rope_interleaved*/ true);
            // attn: [C, L_m + L_t, N]
            auto m_attn = ggml_view_3d(ctx0, attn, attn->ne[0], L_m, N, attn->nb[1], attn->nb[2], 0);
            auto t_attn = ggml_view_3d(ctx0, attn, attn->ne[0], L_t, N, attn->nb[1], attn->nb[2], (size_t)L_m * attn->nb[1]);

            motion = ggml_add(ctx0, motion, apply_gate(ctx0, std::dynamic_pointer_cast<Linear>(B("motion_out_proj"))->forward(ctx, m_attn), m_mod[GA]));
            {
                auto n2  = std::dynamic_pointer_cast<LayerNorm>(B("motion_norm2"))->forward(ctx, motion);
                auto mm  = std::dynamic_pointer_cast<MLP>(B("motion_mlp"))->forward(ctx, modulate(ctx0, n2, m_mod[SH2], m_mod[SC2]));
                motion   = ggml_add(ctx0, motion, apply_gate(ctx0, mm, m_mod[GA2]));
            }

            text = ggml_add(ctx0, text, apply_gate(ctx0, std::dynamic_pointer_cast<Linear>(B("text_out_proj"))->forward(ctx, t_attn), t_mod[GA]));
            {
                auto n2 = std::dynamic_pointer_cast<LayerNorm>(B("text_norm2"))->forward(ctx, text);
                auto tm = std::dynamic_pointer_cast<MLP>(B("text_mlp"))->forward(ctx, modulate(ctx0, n2, t_mod[SH2], t_mod[SC2]));
                text    = ggml_add(ctx0, text, apply_gate(ctx0, tm, t_mod[GA2]));
            }
            return {motion, text};
        }
    };

    // -----------------------------------------------------------------------
    // MMSingleStreamBlock == Flux::SingleStreamBlock, verbatim in structure.
    // Under rope_on_concat (TRAP 1) the reference's q1/q2 split-then-concat is a
    // no-op, so this is exactly Flux: rope the whole stream, attend, gate.
    // -----------------------------------------------------------------------
    struct MMSingleStreamBlock : public GGMLBlock {
        int64_t num_heads, head_dim, feat_dim, mlp_hidden;

        MMSingleStreamBlock(int64_t feat_dim, int64_t num_heads, int64_t mlp_hidden)
            : num_heads(num_heads), head_dim(feat_dim / num_heads), feat_dim(feat_dim), mlp_hidden(mlp_hidden) {
            blocks["modulation"] = std::shared_ptr<GGMLBlock>(new ModulateDiT(feat_dim, 3));
            blocks["norm"]       = std::shared_ptr<GGMLBlock>(new LayerNorm(feat_dim, 1e-6f, false));
            blocks["linear1"]    = std::shared_ptr<GGMLBlock>(new Linear(feat_dim, feat_dim * 3 + mlp_hidden, true));
            // linear2 is this block's FFN/attn down-proj (K = feat_dim + mlp_hidden, the
            // largest K in the model) -- same 1/128 anti-overflow pre-scale as MLP::fc2.
            blocks["linear2"]    = std::shared_ptr<GGMLBlock>(new Linear(feat_dim + mlp_hidden, feat_dim, true, false, false, 1.f / 128.f));
            blocks["q_norm"]     = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-6f));
            blocks["k_norm"]     = std::shared_ptr<GGMLBlock>(new RMSNorm(head_dim, 1e-6f));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* adapter,
                             ggml_tensor* pe,
                             ggml_tensor* mask) {
            auto ctx0 = ctx->ggml_ctx;
            auto mod  = std::dynamic_pointer_cast<ModulateDiT>(blocks["modulation"])->forward(ctx, adapter);
            // chunk order: shift_msa, scale_msa, gate_msa
            auto shift = mod[0], scale = mod[1], gate = mod[2];

            const int64_t L = x->ne[1], N = x->ne[2];

            auto norm    = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto x_mod   = modulate(ctx0, norm->forward(ctx, x), shift, scale);
            auto qkv_mlp = std::dynamic_pointer_cast<Linear>(blocks["linear1"])->forward(ctx, x_mod);  // [3C+H, L, N]

            auto slice = [&](int i) {
                return ggml_ext_cont(ctx0, ggml_view_4d(ctx0, qkv_mlp, head_dim, num_heads, L, N,
                                                        qkv_mlp->nb[0] * head_dim, qkv_mlp->nb[1], qkv_mlp->nb[2],
                                                        (size_t)i * feat_dim * qkv_mlp->nb[0]));
            };
            auto q = std::dynamic_pointer_cast<RMSNorm>(blocks["q_norm"])->forward(ctx, slice(0));
            auto k = std::dynamic_pointer_cast<RMSNorm>(blocks["k_norm"])->forward(ctx, slice(1));
            auto v = slice(2);

            auto attn = Rope::attention(ctx, q, k, v, pe, mask, 1.0f, true);  // [C, L, N]

            auto mlp = ggml_view_3d(ctx0, qkv_mlp, mlp_hidden, L, N, qkv_mlp->nb[1], qkv_mlp->nb[2],
                                    (size_t)3 * feat_dim * qkv_mlp->nb[0]);
            mlp      = ggml_ext_gelu(ctx0, ggml_ext_cont(ctx0, mlp), true);  // mlp_act_type="gelu_tanh"

            auto cat = ggml_concat(ctx0, attn, mlp, 0);  // [C + H, L, N]
            auto out = std::dynamic_pointer_cast<Linear>(blocks["linear2"])->forward(ctx, cat);
            return ggml_add(ctx0, x, apply_gate(ctx0, out, gate));
        }
    };

    // encoders.py FinalLayer. adaLN factor=2 -> (shift, scale). NOTE this is the
    // same factor as the refiner's adaLN but a DIFFERENT meaning (there: two
    // gates). Do not unify them.
    struct FinalLayer : public GGMLBlock {
        FinalLayer(int64_t feat_dim, int64_t out_dim) {
            blocks["norm_final"]       = std::shared_ptr<GGMLBlock>(new LayerNorm(feat_dim, 1e-6f, false));
            blocks["adaLN_modulation"] = std::shared_ptr<GGMLBlock>(new ModulateDiT(feat_dim, 2));
            blocks["linear"]           = std::shared_ptr<GGMLBlock>(new Linear(feat_dim, out_dim, true));
        }

        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, ggml_tensor* adapter) {
            auto ctx0 = ctx->ggml_ctx;
            auto mod  = std::dynamic_pointer_cast<ModulateDiT>(blocks["adaLN_modulation"])->forward(ctx, adapter);
            auto nf   = std::dynamic_pointer_cast<LayerNorm>(blocks["norm_final"]);
            x         = modulate(ctx0, nf->forward(ctx, x), mod[0], mod[1]);
            return std::dynamic_pointer_cast<Linear>(blocks["linear"])->forward(ctx, x);
        }
    };

    // -----------------------------------------------------------------------
    // The DiT.
    // -----------------------------------------------------------------------
    struct HunyuanMotionMMDiT : public GGMLBlock {
        Params p;

        HunyuanMotionMMDiT(const Params& p)
            : p(p) {
            blocks["input_encoder"]    = std::shared_ptr<GGMLBlock>(new Linear(p.input_dim, p.feat_dim, true));
            blocks["ctxt_encoder"]     = std::shared_ptr<GGMLBlock>(new Linear(p.ctxt_input_dim, p.feat_dim, true));
            blocks["vtxt_encoder"]     = std::shared_ptr<GGMLBlock>(new MLPEncoder(p.vtxt_input_dim, p.feat_dim));
            blocks["timestep_encoder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbeddingEncoder(p.feat_dim, p.feat_dim));
            blocks["text_refiner"]     = std::shared_ptr<GGMLBlock>(new SingleTokenRefiner(p));
            for (int64_t i = 0; i < p.double_blocks; ++i) {
                blocks["double_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new MMDoubleStreamBlock(p.feat_dim, p.num_heads, p.mlp_hidden()));
            }
            for (int64_t i = 0; i < p.single_blocks; ++i) {
                blocks["single_blocks." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new MMSingleStreamBlock(p.feat_dim, p.num_heads, p.mlp_hidden()));
            }
            blocks["final_layer"] = std::shared_ptr<GGMLBlock>(new FinalLayer(p.feat_dim, p.input_dim));
        }

        // x:            [input_dim, L_m, N]      noisy latent (L_m == train_frames)
        // ctxt:         [ctxt_dim, L_t, N]       Qwen3-8B hidden states
        // vtxt:         [vtxt_dim, 1, N]         CLIP-L pooled
        // t_sinu_main:  [feat_dim, 1, N]         sinusoidal(t, time_factor=1000)
        // t_sinu_refine:[feat_dim, 1, N]         sinusoidal(t, time_factor=1)  <- TRAP 3
        // ctxt_mean_w:  [L_t, 1, N]              normalised text mask weights
        // pe:           [2,2,head_dim/2,L_m+L_t] RoPE over the concatenated stream
        // mask_double / mask_single: [L_q, L_k] additive F32
        // mask_refiner: [L_t, L_t] additive F32
        ggml_tensor* forward(GGMLRunnerContext* ctx,
                             ggml_tensor* x,
                             ggml_tensor* ctxt,
                             ggml_tensor* vtxt,
                             ggml_tensor* t_sinu_main,
                             ggml_tensor* t_sinu_refine,
                             ggml_tensor* ctxt_mean_w,
                             ggml_tensor* pe,
                             ggml_tensor* mask_double,
                             ggml_tensor* mask_single,
                             ggml_tensor* mask_refiner) {
            auto ctx0 = ctx->ggml_ctx;

            auto motion = std::dynamic_pointer_cast<Linear>(blocks["input_encoder"])->forward(ctx, x);

            auto t_feat = std::dynamic_pointer_cast<TimestepEmbeddingEncoder>(blocks["timestep_encoder"])->forward(ctx, t_sinu_main);
            auto v_feat = std::dynamic_pointer_cast<MLPEncoder>(blocks["vtxt_encoder"])->forward(ctx, vtxt);
            auto adapter = ggml_add(ctx0, t_feat, v_feat);  // [C, 1, N]

            auto ctxt_feat = std::dynamic_pointer_cast<Linear>(blocks["ctxt_encoder"])->forward(ctx, ctxt);
            ctxt_feat      = std::dynamic_pointer_cast<SingleTokenRefiner>(blocks["text_refiner"])
                            ->forward(ctx, ctxt_feat, t_sinu_refine, ctxt_mean_w, mask_refiner);

            for (int64_t i = 0; i < p.double_blocks; ++i) {
                auto blk = std::dynamic_pointer_cast<MMDoubleStreamBlock>(blocks["double_blocks." + std::to_string(i)]);
                auto o   = blk->forward(ctx, motion, ctxt_feat, adapter, pe, mask_double);
                motion   = o[0];
                ctxt_feat = o[1];
            }

            const int64_t split_len = motion->ne[1];
            auto h = ggml_concat(ctx0, motion, ctxt_feat, 1);  // [C, L_m + L_t, N]
            for (int64_t i = 0; i < p.single_blocks; ++i) {
                auto blk = std::dynamic_pointer_cast<MMSingleStreamBlock>(blocks["single_blocks." + std::to_string(i)]);
                h        = blk->forward(ctx, h, adapter, pe, mask_single);
            }

            h = ggml_ext_cont(ctx0, ggml_view_3d(ctx0, h, h->ne[0], split_len, h->ne[2], h->nb[1], h->nb[2], 0));
            // insert_start_token / with_long_skip_connection are both FALSE for this
            // checkpoint (confirmed: no `start_token` or `long_skip_net` tensors).
            return std::dynamic_pointer_cast<FinalLayer>(blocks["final_layer"])->forward(ctx, h, adapter);
        }
    };

    // -----------------------------------------------------------------------
    // Host-side construction of the RoPE table and the attention masks.
    // -----------------------------------------------------------------------

    // 1D temporal RoPE over positions 0..(L_m+L_t-1) of the CONCATENATED stream.
    // Rope::rope produces omega[j] = theta^(-2j/D) and pe = [[cos,-sin],[sin,cos]],
    // which is bit-for-bit the reference's `use_real=True` path:
    //     cos/sin are repeat_interleave(2) over head_dim and rotate_half pairs
    //     ADJACENT elements -> out[2j]   = x[2j]cos - x[2j+1]sin
    //                          out[2j+1] = x[2j]sin + x[2j+1]cos
    // i.e. the interleaved (GPT-J style) convention, matching rope_interleaved=true.
    __STATIC_INLINE__ std::vector<float> build_pe(int64_t total_len, int64_t head_dim, float theta) {
        std::vector<float> pos((size_t)total_len);
        for (int64_t i = 0; i < total_len; ++i) {
            pos[(size_t)i] = (float)i;
        }
        auto rows = Rope::rope(pos, (int)head_dim, theta);
        return Rope::flatten(rows);  // [total_len][head_dim/2 * 4] -> ne [2,2,D/2,L]
    }

    // The additive attention mask, built exactly as _build_dmm_attn_mask_shared /
    // _build_smm_attn_mask_shared do:
    //
    //             motion_k    text_k
    //   motion_q  [M->M]      [M->T]
    //   text_q    [T->M]      [T->T]
    //
    //   * the narrowband band and the motion key-padding apply to [M->M] only
    //   * every column j is additionally masked by that key's padding
    //   * [T->M] is then hard-set to -inf (text never sees motion)
    //
    // LAYOUT -- ggml_ext_attention_ext HAS TWO DIFFERENT MASK CONVENTIONS and picks
    // between them on `flash_attn`. Getting this wrong is silent when L_q == L_k.
    //   flash_attn=true  -> build_kqv() ggml_transpose()s an F32 mask and casts it to
    //                       F16, so the caller supplies ne = [L_q, L_k].
    //   flash_attn=false -> the fallback does `kq = ggml_mul_mat(k, q)` giving
    //                       ne = [L_k, L_q, n_head*N], then `ggml_add_inplace(kq,
    //                       mask)` RAW -- no transpose, no cast. The caller must
    //                       supply ne = [L_k, L_q].
    // We pass flash_attn=false (488 tokens; flash is pointless here), so: ne =
    // [L_k, L_q], ggml element (i0=key, i1=query), buffer index q*L + k. This
    // matches CLIP's causal mask (clip.hpp:543-550 writes -inf at vec[i1*n + i0]
    // when i0 > i1, which is only causal if i0 is the KEY index).
    //
    // Feeding the transposed mask is NOT a benign mixup: the motion key-padding
    // makes columns [motion_valid, L_m) entirely -inf, and transposed those become
    // entirely -inf ROWS -> softmax over an all-masked row -> NaN -> the whole
    // forward is NaN. Which is exactly what it did.
    //
    // motion_valid: number of leading valid motion frames (the rest are padding)
    // text_valid:   number of leading valid text tokens
    __STATIC_INLINE__ std::vector<float> build_attn_mask(int64_t L_m,
                                                         int64_t L_t,
                                                         int64_t motion_valid,
                                                         int64_t text_valid,
                                                         int64_t narrowband_window) {
        const int64_t L   = L_m + L_t;
        const float NEG   = -INFINITY;
        std::vector<float> m((size_t)(L * L), 0.0f);
        // ne = [L_k, L_q]: i0 = key, i1 = query.
        auto at = [&](int64_t q, int64_t k) -> float& { return m[(size_t)(q * L + k)]; };

        for (int64_t q = 0; q < L; ++q) {
            for (int64_t k = 0; k < L; ++k) {
                float v = 0.0f;
                // narrowband: [M->M] quadrant only, |i-j| <= window
                if (q < L_m && k < L_m) {
                    if (std::llabs((long long)(q - k)) > narrowband_window) {
                        v = NEG;
                    }
                }
                // key padding, applied to every row
                const bool key_valid = (k < L_m) ? (k < motion_valid) : ((k - L_m) < text_valid);
                if (!key_valid) {
                    v = NEG;
                }
                // disable T->M
                if (q >= L_m && k < L_m) {
                    v = NEG;
                }
                at(q, k) = v;
            }
        }
        return m;
    }

    // The refiner's own mask: bidirectional (mask1 & mask2) over the text tokens,
    // plus the reference's NaN guard -- a query row with no visible key falls back
    // to seeing itself (token_refiner.py:113-120).
    __STATIC_INLINE__ std::vector<float> build_refiner_mask(int64_t L_t, int64_t text_valid) {
        const float NEG = -INFINITY;
        std::vector<float> m((size_t)(L_t * L_t), 0.0f);
        // ne = [L_k, L_q]: i0 = key, i1 = query. See build_attn_mask.
        auto at = [&](int64_t q, int64_t k) -> float& { return m[(size_t)(q * L_t + k)]; };
        for (int64_t q = 0; q < L_t; ++q) {
            const bool q_valid = q < text_valid;
            for (int64_t k = 0; k < L_t; ++k) {
                const bool k_valid = k < text_valid;
                at(q, k)           = (q_valid && k_valid) ? 0.0f : NEG;
            }
            if (!q_valid) {
                at(q, q) = 0.0f;  // NaN guard: all-masked row sees the diagonal
            }
        }
        return m;
    }

    // -----------------------------------------------------------------------
    // Runner.
    // -----------------------------------------------------------------------
    struct HYMotionRunner : public GGMLRunner {
        Params p;
        HunyuanMotionMMDiT dit;

        // Host buffers referenced by set_backend_tensor_data; must outlive compute().
        std::vector<float> pe_vec, mask_d_vec, mask_r_vec, tsin_main_vec, tsin_ref_vec, mean_w_vec;
        std::vector<float> x_vec, ctxt_vec, vtxt_vec;
        int64_t L_m = 0, L_t = 0, N = 1;

        // Master's GGMLRunner takes (compute backend, weight manager) -- the second
        // backend the spike passed here (params_backend, i.e. "where the weights live")
        // is now the weight manager's business, not the runner's. The DiT is only
        // ~2.1 GB at f16, so the example allocates the params straight onto the compute
        // backend and passes a pass-through manager; see examples/hymotion/main.cpp.
        HYMotionRunner(ggml_backend_t backend,
                       std::shared_ptr<RunnerWeightManager> weight_manager,
                       const Params& params,
                       const String2TensorStorage& tensor_storage_map = {})
            : GGMLRunner(backend, weight_manager), p(params), dit(params) {
            dit.init(params_ctx, tensor_storage_map, "");
        }

        std::string get_desc() override { return "hymotion"; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string& prefix) {
            dit.get_param_tensors(tensors, prefix);
        }

        // x:    [input_dim * L_m * N]   noisy latent, host
        // ctxt: [ctxt_dim  * L_t * N]   Qwen3-8B hidden states, host
        // vtxt: [vtxt_dim  * 1   * N]   CLIP-L pooled, host
        // t:    scalar flow time in [0,1]
        // motion_valid / text_valid: leading valid counts for the masks
        ggml_cgraph* build_graph(float t, int64_t motion_valid, int64_t text_valid) {
            ggml_cgraph* gf = new_graph_custom(HYMOTION_GRAPH_SIZE);
            auto ctx0       = compute_ctx;

            auto x    = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, p.input_dim, L_m, N);
            auto ctxt = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, p.ctxt_input_dim, L_t, N);
            auto vtxt = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, p.vtxt_input_dim, 1, N);
            set_backend_tensor_data(x, x_vec.data());
            set_backend_tensor_data(ctxt, ctxt_vec.data());
            set_backend_tensor_data(vtxt, vtxt_vec.data());

            // TRAP 3: the main encoder uses time_factor=1000, the refiner's uses 1.
            tsin_main_vec.clear();
            tsin_ref_vec.clear();
            for (int64_t n = 0; n < N; ++n) {
                auto a = sinusoidal_embedding(t, p.feat_dim, p.time_factor);
                auto b = sinusoidal_embedding(t, p.feat_dim, 1.0f);
                tsin_main_vec.insert(tsin_main_vec.end(), a.begin(), a.end());
                tsin_ref_vec.insert(tsin_ref_vec.end(), b.begin(), b.end());
            }
            auto tsin_main = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, p.feat_dim, 1, N);
            auto tsin_ref  = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, p.feat_dim, 1, N);
            set_backend_tensor_data(tsin_main, tsin_main_vec.data());
            set_backend_tensor_data(tsin_ref, tsin_ref_vec.data());

            // Normalised text-mask weights for the refiner's masked mean:
            //   (x * mask).sum(1) / mask.sum(1).clamp_min(1e-6)
            mean_w_vec.assign((size_t)(L_t * N), 0.0f);
            for (int64_t n = 0; n < N; ++n) {
                const float denom = std::max((float)text_valid, 1e-6f);
                for (int64_t i = 0; i < text_valid && i < L_t; ++i) {
                    mean_w_vec[(size_t)(n * L_t + i)] = 1.0f / denom;
                }
            }
            auto mean_w = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, L_t, 1, N);
            set_backend_tensor_data(mean_w, mean_w_vec.data());

            // RoPE over the concatenated [motion, text] stream (TRAP 1).
            pe_vec      = build_pe(L_m + L_t, p.head_dim, p.rope_theta);
            auto pe     = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 2, 2, p.head_dim / 2, L_m + L_t);
            set_backend_tensor_data(pe, pe_vec.data());

            // The double- and single-stream masks are built by the same rules over
            // the same [motion, text] token layout, so they are identical here; the
            // reference builds them via two functions only because the single blocks
            // take split_len rather than (motion_len, text_len).
            mask_d_vec  = build_attn_mask(L_m, L_t, motion_valid, text_valid, p.narrowband_window);
            auto mask_d = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, L_m + L_t, L_m + L_t);
            set_backend_tensor_data(mask_d, mask_d_vec.data());

            mask_r_vec  = build_refiner_mask(L_t, text_valid);
            auto mask_r = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, L_t, L_t);
            set_backend_tensor_data(mask_r, mask_r_vec.data());

            auto rc = get_context();
            // The spike forced rc.allow_fused_rope = false on CPU here: its
            // Rope::apply_rope preferred the CUDA-only fused ggml_rope_pe kernel and
            // the CPU backend aborted with "op not implemented: ROPE_PE". Master's
            // rope.hpp has no fused path and no allow_fused_rope flag -- apply_rope is
            // always the cont+mul+add chain -- so there is nothing to switch off.
            if (getenv("HYMOTION_NO_MASK")) { mask_d = nullptr; mask_r = nullptr; }
            auto out = dit.forward(&rc, x, ctxt, vtxt, tsin_main, tsin_ref, mean_w, pe, mask_d, mask_d, mask_r);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        // Returns the predicted flow [input_dim, L_m, N].
        sd::Tensor<float> compute(int n_threads, float t, int64_t motion_valid, int64_t text_valid) {
            auto get_graph = [&]() -> ggml_cgraph* { return build_graph(t, motion_valid, text_valid); };
            auto r         = GGMLRunner::compute<float>(get_graph, n_threads, false);
            return r.value_or(sd::Tensor<float>());
        }
    };

}  // namespace HYMotion

#endif  // __SD_MODEL_DIFFUSION_HYMOTION_HPP__
