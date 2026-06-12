#ifndef __WAV2VEC2_HPP__
#define __WAV2VEC2_HPP__

#include <cmath>
#include <cstdint>
#include <vector>

#include "ggml_extend.hpp"
#include "longcat_audio.hpp"  // reuse load_wav_16k_mono
#include "model.h"

// ---------------------------------------------------------------------------
// Wan2.2-S2V-14B audio pathway: wav2vec2-large-xlsr-53 encoder + CausalAudioEncoder.
//
// Reference ground truth (M1, non-causal stock):
//   /mnt/hdd/live-avatar/ref/Wan2.2/wan/modules/s2v/audio_encoder.py
//     - AudioEncoder.extract_audio_feat: wav2vec2 (HF Wav2Vec2ForCTC) -> ALL 25
//       hidden states (input embeds + 24 layers); linear_interpolation 50fps->30fps
//       (video_rate=30, align_corners=True); get_audio_embed_bucket_fps(fps=16, m=0)
//       -> per-latent-frame bucketed embed [25, T_a].
//   /mnt/hdd/live-avatar/ref/Wan2.2/wan/modules/s2v/audio_utils.py
//     - CausalAudioEncoder: learnable [1,25,1,1] SiLU-weighted sum over the 25
//       layers -> MotionEncoder_tc -> audio tokens [B, F, n_tok, dim].
//   /mnt/hdd/live-avatar/ref/Wan2.2/wan/modules/s2v/auxi_blocks.py
//     - MotionEncoder_tc: CausalConv1d(replicate-pad) local/global branches,
//       strides 1->2->2, LayerNorm(no-affine)+SiLU, padding_tokens; need_global=True
//       (enable_adain) so it returns (x_global, x_local).
//
// wav2vec2-large-xlsr-53 config (verified from config.json):
//   hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
//   intermediate_size=4096, conv_dim=[512]*7, conv_kernel=[10,3,3,3,3,2,2],
//   conv_stride=[5,2,2,2,2,2,2], conv_bias=true, feat_extract_norm="layer",
//   do_stable_layer_norm=true, num_conv_pos_embeddings=128,
//   num_conv_pos_embedding_groups=16, layer_norm_eps=1e-5, hidden_act=gelu.
//   Net feature-extractor stride = 5*2*2*2*2*2*2 = 320 -> 16kHz/320 = 50 fps.
//
// This file implements:
//   1. CPU log-mel? NO — wav2vec2 consumes the RAW normalized waveform (Wav2Vec2
//      FeatureExtractor does zero-mean/unit-var normalization, then a Conv1d stack).
//      So the CPU front-end here is just: load wav, mean/var normalize.
//   2. The encoder graph (GGMLRunner): 7 LayerNorm-conv feature-extractor layers,
//      feature_projection (LN + Linear), pos_conv_embed (grouped Conv1d), 24
//      stable-layer-norm (pre-LN) transformer layers + final encoder.layer_norm,
//      capturing all 25 hidden states stacked as [d_model, T, 25].
//   3. CausalAudioEncoder graph: SiLU-weighted layer sum + MotionEncoder_tc.
//   4. CPU host helpers for the 50->30fps interp and the fps=16/m=0 bucketing.
//
// GGUF NAMING (matched to PyTorch state_dict of model_s2v.py + the HF wav2vec2;
//   a parallel agent writes /mnt/hdd/live-avatar/gguf/NAMING.md — match exactly
//   when ready). Until then we mirror the HF names under prefix "audio_encoder.":
//     audio_encoder.feature_extractor.conv_layers.{i}.conv.{weight,bias}
//     audio_encoder.feature_extractor.conv_layers.{i}.layer_norm.{weight,bias}
//     audio_encoder.feature_projection.layer_norm.{weight,bias}
//     audio_encoder.feature_projection.projection.{weight,bias}
//     audio_encoder.encoder.pos_conv_embed.conv.{weight_g,weight_v,bias}  (weight-norm)
//     audio_encoder.encoder.layer_norm.{weight,bias}
//     audio_encoder.encoder.layers.{i}.attention.{q,k,v,out}_proj.{weight,bias}
//     audio_encoder.encoder.layers.{i}.layer_norm.{weight,bias}              (pre-attn LN)
//     audio_encoder.encoder.layers.{i}.feed_forward.intermediate_dense.{weight,bias}
//     audio_encoder.encoder.layers.{i}.feed_forward.output_dense.{weight,bias}
//     audio_encoder.encoder.layers.{i}.final_layer_norm.{weight,bias}        (pre-ffn LN)
//   CausalAudioEncoder under prefix "casual_audio_encoder." (note the upstream
//   typo "casual"): see CausalAudioEncoderBlock below for the exact subnames.
//
// TODO(NAMING.md): the HF Wav2Vec2ForCTC checkpoint wraps the encoder under
//   "wav2vec2." — the gguf converter must strip that and re-root under
//   "audio_encoder.". Also weight-norm pos_conv may be pre-fused to a single
//   "conv.weight" at convert time; if so, drop the weight_g/weight_v split here
//   (see Wav2Vec2PosConvEmbed::init_params note).
namespace WAV2VEC2 {

    struct Wav2Vec2Params {
        int64_t d_model    = 1024;
        int n_layers       = 24;
        int64_t n_heads    = 16;
        int64_t head_dim   = 64;  // 1024/16
        int64_t ffn_dim    = 4096;
        int n_conv_layers  = 7;
        std::vector<int64_t> conv_dim    = {512, 512, 512, 512, 512, 512, 512};
        std::vector<int>     conv_kernel = {10, 3, 3, 3, 3, 2, 2};
        std::vector<int>     conv_stride = {5, 2, 2, 2, 2, 2, 2};
        int pos_conv_kernel = 128;
        int pos_conv_groups = 16;
        float eps           = 1e-5f;
        int n_hidden_states = 25;  // input embeds + 24 layers
    };

    // --------------------------------------------------------------------
    // Feature-extractor conv layer (feat_extract_norm == "layer"):
    //   Conv1d(no bias? -> conv_bias=true here) -> LayerNorm over channels -> GELU.
    // HF stores conv weight as [out, in, kernel]; ggml_conv_1d wants [kernel,in,out].
    // The layer_norm is applied over the channel dim AFTER transposing to [.., C].
    // --------------------------------------------------------------------
    class Wav2Vec2ConvLayer : public GGMLBlock {
    protected:
        int64_t in_dim, out_dim;
        int kernel, stride;
        float eps;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            (void)tensor_storage_map;
            params["conv.weight"]       = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kernel, in_dim, out_dim);
            params["conv.bias"]         = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
            params["layer_norm.weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
            params["layer_norm.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
        }

    public:
        Wav2Vec2ConvLayer(int64_t in_dim, int64_t out_dim, int kernel, int stride, float eps)
            : in_dim(in_dim), out_dim(out_dim), kernel(kernel), stride(stride), eps(eps) {}

        // x: [T_in, in_dim, 1] (ggml ne: [length, channels, batch]); returns [T_out, out_dim, 1].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w  = params["conv.weight"];
            ggml_tensor* b  = params["conv.bias"];
            ggml_tensor* ln_w = params["layer_norm.weight"];
            ggml_tensor* ln_b = params["layer_norm.bias"];

            x = ggml_conv_1d(ctx->ggml_ctx, w, x, stride, 0, 1);  // [T_out, out_dim, 1], padding 0
            x = ggml_add(ctx->ggml_ctx, x, ggml_reshape_3d(ctx->ggml_ctx, b, 1, out_dim, 1));

            // LayerNorm over channels: ggml_norm normalizes over dim0, so transpose
            // [T,C,1] -> [C,T,1], norm, scale/shift, transpose back.
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [C, T, 1]
            x = ggml_norm(ctx->ggml_ctx, x, eps);
            x = ggml_add(ctx->ggml_ctx, ggml_mul(ctx->ggml_ctx, x, ln_w), ln_b);
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [T, C, 1]
            x = ggml_gelu(ctx->ggml_ctx, x);
            return x;
        }
    };

    // --------------------------------------------------------------------
    // pos_conv_embed: grouped Conv1d (kernel 128, groups 16) + SamePad (drop last
    // sample when kernel even) + GELU, added as a residual to hidden states.
    // HF uses weight-norm (weight_g / weight_v). For M1 we expect the converter to
    // pre-fuse into a single conv.weight; we load that. (If the gguf keeps weight_g/
    // weight_v, extend init_params + reconstruct w = g * v / ||v||.)  TODO(NAMING.md).
    //
    // ggml has no native grouped conv-1d; we emulate by splitting the input channels
    // into `groups` chunks and running a conv per group, then concatenating. This is
    // structurally correct (built once on the 1024-d hidden states).
    // --------------------------------------------------------------------
    class Wav2Vec2PosConvEmbed : public GGMLBlock {
    protected:
        int64_t d_model;
        int kernel, groups;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            (void)tensor_storage_map;
            // fused conv weight [kernel, d_model/groups, d_model], bias [d_model]
            params["conv.weight"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kernel, d_model / groups, d_model);
            params["conv.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, d_model);
        }

    public:
        Wav2Vec2PosConvEmbed(int64_t d_model, int kernel, int groups)
            : d_model(d_model), kernel(kernel), groups(groups) {}

        // x: [d_model, T, 1]; returns positional residual [d_model, T, 1].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["conv.weight"];  // [kernel, d_model/groups, d_model]
            ggml_tensor* b = params["conv.bias"];
            int64_t T = x->ne[1];

            // transpose to [T, d_model, 1] for ggml_conv_1d (length-major).
            auto h = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [T, d_model, 1]

            int64_t in_per_g  = d_model / groups;
            int64_t out_per_g = d_model / groups;
            // pad left by (kernel-1) so output length keeps T (HF pads (kernel)//2 then
            // SamePad trims 1 for even kernel -> net output length == T). We replicate
            // that exact length by padding kernel-1 on the left and 0 on the right, then
            // trimming the trailing (kernel even) sample below.
            std::vector<ggml_tensor*> outs;
            for (int g = 0; g < groups; g++) {
                auto h_g = ggml_view_3d(ctx->ggml_ctx, h, T, in_per_g, 1,
                                        h->nb[1], h->nb[2], g * in_per_g * h->nb[1]);
                h_g      = ggml_cont(ctx->ggml_ctx, h_g);  // [T, in_per_g, 1]
                auto w_g = ggml_view_3d(ctx->ggml_ctx, w, kernel, in_per_g, out_per_g,
                                        w->nb[1], w->nb[2], g * out_per_g * w->nb[2]);
                w_g      = ggml_cont(ctx->ggml_ctx, w_g);  // [kernel, in_per_g, out_per_g]
                // pad left kernel-1 on the length axis (dim0).
                auto h_pad = ggml_pad_ext(ctx->ggml_ctx, h_g, kernel - 1, 0, 0, 0, 0, 0, 0, 0);
                auto o_g   = ggml_conv_1d(ctx->ggml_ctx, w_g, h_pad, 1, 0, 1);  // [T(+1 if even), out_per_g, 1]
                outs.push_back(o_g);
            }
            ggml_tensor* o = outs[0];
            for (size_t i = 1; i < outs.size(); i++) {
                o = ggml_concat(ctx->ggml_ctx, o, outs[i], 1);  // concat along channel dim
            }
            // SamePad: kernel 128 is even -> drop the last time sample to restore length T.
            if (kernel % 2 == 0) {
                o = ggml_ext_slice(ctx->ggml_ctx, o, 0, 0, T);
            }
            o = ggml_add(ctx->ggml_ctx, o, ggml_reshape_3d(ctx->ggml_ctx, b, 1, d_model, 1));
            o = ggml_gelu(ctx->ggml_ctx, o);
            o = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, o, 1, 0, 2, 3));  // [d_model, T, 1]
            return o;
        }
    };

    // Stable-layer-norm (pre-LN) transformer layer.
    class Wav2Vec2EncoderLayer : public GGMLBlock {
    protected:
        int64_t d_model, n_heads, ffn_dim;
        float eps;

    public:
        Wav2Vec2EncoderLayer(int64_t d_model, int64_t n_heads, int64_t ffn_dim, float eps)
            : d_model(d_model), n_heads(n_heads), ffn_dim(ffn_dim), eps(eps) {
            blocks["attention.q_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(d_model, d_model, true));
            blocks["attention.k_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(d_model, d_model, true));
            blocks["attention.v_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(d_model, d_model, true));
            blocks["attention.out_proj"] = std::shared_ptr<GGMLBlock>(new Linear(d_model, d_model, true));
            blocks["layer_norm"]         = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model, eps, true, true));
            blocks["feed_forward.intermediate_dense"] = std::shared_ptr<GGMLBlock>(new Linear(d_model, ffn_dim, true));
            blocks["feed_forward.output_dense"]       = std::shared_ptr<GGMLBlock>(new Linear(ffn_dim, d_model, true));
            blocks["final_layer_norm"]   = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model, eps, true, true));
        }

        // x: [d_model, T, 1]; pre-LN (stable layer norm) variant.
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, bool flash_attn) {
            auto q_proj = std::dynamic_pointer_cast<Linear>(blocks["attention.q_proj"]);
            auto k_proj = std::dynamic_pointer_cast<Linear>(blocks["attention.k_proj"]);
            auto v_proj = std::dynamic_pointer_cast<Linear>(blocks["attention.v_proj"]);
            auto outp   = std::dynamic_pointer_cast<Linear>(blocks["attention.out_proj"]);
            auto ln     = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);
            auto fc1    = std::dynamic_pointer_cast<Linear>(blocks["feed_forward.intermediate_dense"]);
            auto fc2    = std::dynamic_pointer_cast<Linear>(blocks["feed_forward.output_dense"]);
            auto lnf    = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer_norm"]);

            // attention (pre-LN)
            auto residual = x;
            auto h = ln->forward(ctx, x);
            auto q = q_proj->forward(ctx, h);
            auto k = k_proj->forward(ctx, h);
            auto v = v_proj->forward(ctx, h);
            auto attn = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v,
                                                n_heads, nullptr, false, flash_attn);  // [d_model, T, 1]
            attn = outp->forward(ctx, attn);
            x    = ggml_add(ctx->ggml_ctx, residual, attn);

            // ffn (pre-LN), gelu
            residual = x;
            h = lnf->forward(ctx, x);
            h = fc1->forward(ctx, h);
            h = ggml_gelu(ctx->ggml_ctx, h);
            h = fc2->forward(ctx, h);
            x = ggml_add(ctx->ggml_ctx, residual, h);
            return x;
        }
    };

    class Wav2Vec2Encoder : public GGMLBlock {
    protected:
        Wav2Vec2Params p;

    public:
        Wav2Vec2Encoder() {}
        Wav2Vec2Encoder(Wav2Vec2Params p)
            : p(p) {
            // feature extractor conv layers
            int64_t in_dim = 1;
            for (int i = 0; i < p.n_conv_layers; i++) {
                int64_t out_dim = p.conv_dim[i];
                blocks["feature_extractor.conv_layers." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new Wav2Vec2ConvLayer(in_dim, out_dim, p.conv_kernel[i], p.conv_stride[i], p.eps));
                in_dim = out_dim;
            }
            // feature projection
            blocks["feature_projection.layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(p.conv_dim.back(), p.eps, true, true));
            blocks["feature_projection.projection"] = std::shared_ptr<GGMLBlock>(new Linear(p.conv_dim.back(), p.d_model, true));
            // pos conv
            blocks["encoder.pos_conv_embed"] = std::shared_ptr<GGMLBlock>(new Wav2Vec2PosConvEmbed(p.d_model, p.pos_conv_kernel, p.pos_conv_groups));
            // stable-layer-norm transformer
            for (int i = 0; i < p.n_layers; i++) {
                blocks["encoder.layers." + std::to_string(i)] =
                    std::shared_ptr<GGMLBlock>(new Wav2Vec2EncoderLayer(p.d_model, p.n_heads, p.ffn_dim, p.eps));
            }
            blocks["encoder.layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(p.d_model, p.eps, true, true));
        }

        // wav: [T_samples, 1, 1] raw normalized waveform.
        // Returns ALL 25 hidden states stacked as [d_model, T, 25].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* wav, bool flash_attn) {
            // feature extractor
            ggml_tensor* x = wav;  // [T, 1, 1]
            for (int i = 0; i < p.n_conv_layers; i++) {
                auto conv = std::dynamic_pointer_cast<Wav2Vec2ConvLayer>(blocks["feature_extractor.conv_layers." + std::to_string(i)]);
                x = conv->forward(ctx, x);  // [T_i, conv_dim, 1]
            }
            // x: [T, 512, 1] -> transpose to [512, T, 1] feature vectors
            x = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [512, T, 1]

            // feature projection (LN + Linear). HF returns norm_hidden_states; the
            // projected output is the transformer input.
            auto fp_ln   = std::dynamic_pointer_cast<LayerNorm>(blocks["feature_projection.layer_norm"]);
            auto fp_proj = std::dynamic_pointer_cast<Linear>(blocks["feature_projection.projection"]);
            x = fp_ln->forward(ctx, x);
            x = fp_proj->forward(ctx, x);  // [d_model, T, 1]

            // positional conv embedding (residual add)
            auto pos_conv = std::dynamic_pointer_cast<Wav2Vec2PosConvEmbed>(blocks["encoder.pos_conv_embed"]);
            auto pos = pos_conv->forward(ctx, x);  // [d_model, T, 1]
            x = ggml_add(ctx->ggml_ctx, x, pos);

            auto lnf = std::dynamic_pointer_cast<LayerNorm>(blocks["encoder.layer_norm"]);

            int64_t T = x->ne[1];
            std::vector<ggml_tensor*> hs;
            // hidden_states[0] = pre-layer hidden state (post pos_conv). In stable-LN
            // Wav2Vec2 the final encoder.layer_norm is applied AFTER all layers; HF's
            // output_hidden_states captures the pre-final-LN states for [0..n-1] and the
            // post-final-LN for the last entry.
            hs.push_back(x);
            for (int i = 0; i < p.n_layers; i++) {
                auto layer = std::dynamic_pointer_cast<Wav2Vec2EncoderLayer>(blocks["encoder.layers." + std::to_string(i)]);
                x = layer->forward(ctx, x, flash_attn);
                if (i == p.n_layers - 1) {
                    hs.push_back(lnf->forward(ctx, x));
                } else {
                    hs.push_back(x);
                }
            }

            // stack -> [d_model, T, 25]
            ggml_tensor* stacked = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, hs[0]), p.d_model, T, 1);
            for (size_t i = 1; i < hs.size(); i++) {
                auto h = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, hs[i]), p.d_model, T, 1);
                stacked = ggml_concat(ctx->ggml_ctx, stacked, h, 2);
            }
            return stacked;  // [d_model, T, 25]
        }
    };

    struct Wav2Vec2EncoderRunner : public GGMLRunner {
        Wav2Vec2Params p;
        Wav2Vec2Encoder encoder;
        std::string desc = "wav2vec2-encoder";

        Wav2Vec2EncoderRunner(ggml_backend_t backend,
                              ggml_backend_t params_backend,
                              const String2TensorStorage& tensor_storage_map = {},
                              const std::string prefix                       = "audio_encoder")
            : GGMLRunner(backend, params_backend) {
            encoder = Wav2Vec2Encoder(p);
            encoder.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return desc; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            encoder.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& wav_tensor) {
            ggml_cgraph* gf  = new_graph_custom(GGML_DEFAULT_GRAPH_SIZE * 8);
            ggml_tensor* wav = make_input(wav_tensor);  // [T_samples, 1, 1]
            auto runner_ctx  = get_context();
            ggml_tensor* out = encoder.forward(&runner_ctx, wav, runner_ctx.flash_attn_enabled);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        // wav_tensor: [T_samples, 1, 1] (normalized). Returns [d_model, T, 25].
        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& wav_tensor) {
            auto get_graph = [&]() -> ggml_cgraph* { return build_graph(wav_tensor); };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }
    };

    // --------------------------------------------------------------------
    // CausalAudioEncoder (casual_audio_encoder.*): learnable [1,25,1,1] SiLU
    // weights -> weighted sum over the 25 layers -> MotionEncoder_tc.
    //
    // MotionEncoder_tc (need_global=True / enable_adain):
    //   local branch:  conv1_local (in_dim -> hidden//4 * num_heads, k3 s1) ->
    //                  reshape (b n) -> norm1(hidden//4) -> SiLU -> conv2 (->hidden//2, k3 s2)
    //                  -> norm2 -> SiLU -> conv3 (->hidden, k3 s2) -> norm3 -> SiLU
    //                  -> reshape b t n c -> append padding_token -> x_local [b,t,n+1,c]
    //   global branch: conv1_global (in_dim -> hidden//4, k3 s1) -> norm1 -> SiLU
    //                  -> conv2 -> norm2 -> SiLU -> conv3 -> norm3 -> SiLU
    //                  -> final_linear -> x_global [b,t,1,c]
    //   returns (x_global, x_local).
    //
    //   in_dim = audio_dim = 1024 (DiT-side audio), hidden_dim = dim = 5120,
    //   num_heads = num_audio_token = 4 -> x_local has 4 local + 1 padding = 5 tokens.
    //   So merged audio tokens per frame = [b, f, 5, 5120].  CausalConv1d uses
    //   replicate left-pad of (kernel-1) (causal). The two stride-2 convs over the
    //   time axis collapse the prepended motion_frames context.
    // --------------------------------------------------------------------
    struct CausalAudioEncoderParams {
        int64_t in_dim    = 1024;   // audio_dim
        int64_t out_dim   = 5120;   // DiT dim
        int num_token     = 4;      // num_audio_token -> num_heads of MotionEncoder
        int num_layers    = 25;     // wav2vec hidden states
        bool need_global  = true;   // enable_adain
        float eps         = 1e-6f;
    };

    // CausalConv1d (replicate left-pad kernel-1) as a GGMLBlock.
    class CausalConv1d : public GGMLBlock {
    protected:
        int64_t in_dim, out_dim;
        int kernel, stride;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            (void)tensor_storage_map;
            params["conv.weight"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, kernel, in_dim, out_dim);
            params["conv.bias"]   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_dim);
        }

    public:
        CausalConv1d(int64_t in_dim, int64_t out_dim, int kernel, int stride)
            : in_dim(in_dim), out_dim(out_dim), kernel(kernel), stride(stride) {}

        // x: [T, in_dim, B]; returns [T_out, out_dim, B]. Replicate left-pad (kernel-1).
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x) {
            ggml_tensor* w = params["conv.weight"];
            ggml_tensor* b = params["conv.bias"];
            int pad = kernel - 1;
            // replicate-pad the leftmost time sample `pad` times. ggml lacks a direct
            // replicate-pad on dim0; we emulate by slicing the first frame and
            // concatenating it pad times on the left.
            if (pad > 0) {
                auto first = ggml_ext_slice(ctx->ggml_ctx, x, 0, 0, 1);  // [1, in_dim, B]
                ggml_tensor* padcat = first;
                for (int i = 1; i < pad; i++) {
                    padcat = ggml_concat(ctx->ggml_ctx, padcat, first, 0);
                }
                x = ggml_concat(ctx->ggml_ctx, padcat, x, 0);
            }
            x = ggml_conv_1d(ctx->ggml_ctx, w, x, stride, 0, 1);  // [T_out, out_dim, B]
            x = ggml_add(ctx->ggml_ctx, x, ggml_reshape_3d(ctx->ggml_ctx, b, 1, out_dim, 1));
            return x;
        }
    };

    class MotionEncoderTC : public GGMLBlock {
    protected:
        CausalAudioEncoderParams p;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            (void)tensor_storage_map;
            // padding_tokens [1,1,1,hidden] -> ggml 1d of hidden
            params["padding_tokens"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.out_dim);
        }

    public:
        MotionEncoderTC() {}
        MotionEncoderTC(CausalAudioEncoderParams p)
            : p(p) {
            int64_t h4 = p.out_dim / 4;
            int64_t h2 = p.out_dim / 2;
            blocks["conv1_local"] = std::shared_ptr<GGMLBlock>(new CausalConv1d(p.in_dim, h4 * p.num_token, 3, 1));
            if (p.need_global) {
                blocks["conv1_global"] = std::shared_ptr<GGMLBlock>(new CausalConv1d(p.in_dim, h4, 3, 1));
                blocks["final_linear"] = std::shared_ptr<GGMLBlock>(new Linear(p.out_dim, p.out_dim, true));
            }
            blocks["conv2"] = std::shared_ptr<GGMLBlock>(new CausalConv1d(h4, h2, 3, 2));
            blocks["conv3"] = std::shared_ptr<GGMLBlock>(new CausalConv1d(h2, p.out_dim, 3, 2));
            // norm1/2/3 are LayerNorm elementwise_affine=False (no params).
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(h4, p.eps, false, true));
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(h2, p.eps, false, true));
            blocks["norm3"] = std::shared_ptr<GGMLBlock>(new LayerNorm(p.out_dim, p.eps, false, true));
        }

        // weighted_feat: [in_dim, F, 1] (per-frame, b f c -> here ggml [c, F, 1]).
        // Returns {x_global [out_dim, 1, F], x_local [out_dim, num_token+1, F]} —
        // ggml ne layout [c, n_tok, F] so the DiT side can rearrange per frame.
        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* x_cf) {
            // x_cf: [in_dim, F, 1]; conv path wants [T=F, in_dim, B].
            auto to_time = [&](ggml_tensor* t) {
                return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, t, 1, 0, 2, 3));  // [c,F,1]->[F,c,1]
            };
            auto to_chan = [&](ggml_tensor* t) {
                return ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, t, 1, 0, 2, 3));  // [F,c,1]->[c,F,1]
            };

            auto conv1_local = std::dynamic_pointer_cast<CausalConv1d>(blocks["conv1_local"]);
            auto conv2       = std::dynamic_pointer_cast<CausalConv1d>(blocks["conv2"]);
            auto conv3       = std::dynamic_pointer_cast<CausalConv1d>(blocks["conv3"]);
            auto norm1       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm1"]);
            auto norm2       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm2"]);
            auto norm3       = std::dynamic_pointer_cast<LayerNorm>(blocks["norm3"]);

            int64_t h4 = p.out_dim / 4;

            bool dbg = getenv("S2V_CAE_DEBUG") != nullptr;
            auto D = [&](const char* tag, ggml_tensor* t) {
                if (dbg) fprintf(stderr, "[CAE] %-14s ne=[%lld,%lld,%lld,%lld] nb0=%zu cont=%d\n", tag,
                                 (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
                                 t->nb[0], ggml_is_contiguous(t));
            };
            // ---- local branch ----
            // conv1_local: [F, in_dim, 1] -> [F, h4*num_token, 1]
            auto xt = to_time(x_cf);
            D("x_cf->time", xt);
            xt = conv1_local->forward(ctx, xt);  // [F, h4*num_token, 1]
            D("conv1_local", xt);
            int64_t Fl = xt->ne[0];
            // rearrange 'b (n c) t -> (b n) t c': split channel into [num_token, h4],
            // treat num_token as batch. ggml layout: xt is [F, num_token*h4, 1]; we want
            // [h4, F, num_token] for the per-head LayerNorm over the h4 channels.
            // current xt ne=[F, num_token*h4, 1]; permute to [num_token*h4, F, 1]:
            auto xc = to_chan(xt);  // [num_token*h4, F, 1]
            xc = ggml_reshape_3d(ctx->ggml_ctx, xc, h4, p.num_token, Fl);          // [h4, num_token, F]
            xc = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, xc, 0, 2, 1, 3));  // [h4, F, num_token]
            xc = norm1->forward(ctx, xc);                                          // LN over h4
            xc = ggml_silu(ctx->ggml_ctx, xc);
            // conv2 over time: needs [F, h4, num_token]
            auto xtc = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, xc, 1, 0, 2, 3));  // [F, h4, num_token]
            xtc = conv2->forward(ctx, xtc);  // [F/2, h2, num_token]
            // norm2 over channels: [h2, F2, num_token]
            auto xc2 = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, xtc, 1, 0, 2, 3));  // [h2, F2, num_token]
            xc2 = norm2->forward(ctx, xc2);
            xc2 = ggml_silu(ctx->ggml_ctx, xc2);
            auto xtc2 = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, xc2, 1, 0, 2, 3));  // [F2, h2, num_token]
            xtc2 = conv3->forward(ctx, xtc2);  // [F3, out_dim, num_token]
            auto xc3 = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, xtc2, 1, 0, 2, 3));  // [out_dim, F3, num_token]
            xc3 = norm3->forward(ctx, xc3);
            xc3 = ggml_silu(ctx->ggml_ctx, xc3);
            // xc3: [out_dim, F3, num_token]; append padding token -> [out_dim, F3, num_token+1].
            ggml_tensor* pad = params["padding_tokens"];  // [out_dim]
            int64_t F3 = xc3->ne[1];
            auto pad_b = ggml_reshape_3d(ctx->ggml_ctx, pad, p.out_dim, 1, 1);
            pad_b      = ggml_repeat_4d(ctx->ggml_ctx, pad_b, p.out_dim, F3, 1, 1);  // [out_dim, F3, 1]
            D("xc3(local)", xc3);
            auto x_local = ggml_concat(ctx->ggml_ctx, xc3, pad_b, 2);                // [out_dim, F3, num_token+1]
            D("x_local.cat", x_local);
            // reorder to [out_dim, num_token+1, F3] (per-frame token grouping)
            x_local = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x_local, 0, 2, 1, 3));
            D("x_local.perm", x_local);

            if (!p.need_global) {
                return {nullptr, x_local};
            }

            // ---- global branch ---- (single head)
            auto conv1_global = std::dynamic_pointer_cast<CausalConv1d>(blocks["conv1_global"]);
            auto final_linear = std::dynamic_pointer_cast<Linear>(blocks["final_linear"]);
            auto g = conv1_global->forward(ctx, to_time(x_cf));  // [F, h4, 1]
            auto gc = to_chan(g);                                // [h4, F, 1]
            gc = norm1->forward(ctx, gc);
            gc = ggml_silu(ctx->ggml_ctx, gc);
            auto gt = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, gc, 1, 0, 2, 3));  // [F, h4, 1]
            gt = conv2->forward(ctx, gt);                                                     // [F2, h2, 1]
            auto gc2 = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, gt, 1, 0, 2, 3));  // [h2, F2, 1]
            gc2 = norm2->forward(ctx, gc2);
            gc2 = ggml_silu(ctx->ggml_ctx, gc2);
            auto gt2 = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, gc2, 1, 0, 2, 3));  // [F2, h2, 1]
            gt2 = conv3->forward(ctx, gt2);                                                     // [F3, out_dim, 1]
            auto gc3 = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, gt2, 1, 0, 2, 3));  // [out_dim, F3, 1]
            gc3 = norm3->forward(ctx, gc3);
            gc3 = ggml_silu(ctx->ggml_ctx, gc3);
            gc3 = final_linear->forward(ctx, gc3);  // [out_dim, F3, 1]
            // x_global -> [out_dim, 1, F3]
            auto x_global = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, gc3, 0, 2, 1, 3));  // [out_dim,1,F3]
            return {x_global, x_local};
        }
    };

    class CausalAudioEncoderBlock : public GGMLBlock {
    protected:
        CausalAudioEncoderParams p;

        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map = {}, const std::string prefix = "") override {
            // PyTorch `weights` is [1, num_layers, 1, 1] -> ggml ne [1, 1, num_layers, 1].
            // Declare it with that exact ne so the loader's per-dim shape check passes;
            // we flatten to [num_layers] in forward(). Force F32 (the loader converts
            // the gguf's F16 storage on load) — it feeds a broadcast ggml_mul against
            // F32 features, and ggml binary ops require matching src1 dtype.
            (void)tensor_storage_map;
            params["weights"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, p.num_layers);
        }

    public:
        CausalAudioEncoderBlock() {}
        CausalAudioEncoderBlock(CausalAudioEncoderParams p)
            : p(p) {
            blocks["encoder"] = std::shared_ptr<GGMLBlock>(new MotionEncoderTC(p));
        }

        // features: [in_dim, F, num_layers] (ggml ne) = the stacked wav2vec hidden
        // states already cropped to in_dim & F latent frames. (PyTorch features are
        // [B, num_layers, dim, video_length]; here per-frame channels-first.)
        // Returns {x_global, x_local} from the MotionEncoder.
        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx, ggml_tensor* features) {
            ggml_tensor* w = params["weights"];  // ggml ne [1, 1, num_layers]
            w              = ggml_reshape_1d(ctx->ggml_ctx, w, p.num_layers);  // [num_layers]
            // weights = SiLU(weights); weights_sum = sum; weighted = sum_l(feat_l * w_l)/sum.
            auto ws = ggml_silu(ctx->ggml_ctx, w);                    // [num_layers]
            // weighted sum over layer dim (dim2). Broadcast ws over [in_dim, F, num_layers].
            auto ws_b = ggml_reshape_3d(ctx->ggml_ctx, ws, 1, 1, p.num_layers);
            if (getenv("S2V_CAE_DEBUG"))
                fprintf(stderr, "[CAE] feats ne=[%lld,%lld,%lld] cont=%d ; ws_b ne=[%lld,%lld,%lld] cont=%d\n",
                        (long long)features->ne[0], (long long)features->ne[1], (long long)features->ne[2], ggml_is_contiguous(features),
                        (long long)ws_b->ne[0], (long long)ws_b->ne[1], (long long)ws_b->ne[2], ggml_is_contiguous(ws_b));
            auto wf   = ggml_mul(ctx->ggml_ctx, features, ws_b);      // [in_dim, F, num_layers]
            // sum over the layer dim -> [in_dim, F].  ggml_sum_rows sums ne[0], so
            // permute the layer axis (src ne2) into ne[0]: in_dim(ne0)->pos1,
            // F(ne1)->pos2, num_layers(ne2)->pos0  ==> ggml_permute(.,1,2,0,3).
            auto wf_p = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, wf, 1, 2, 0, 3));  // [num_layers, in_dim, F]
            auto summed = ggml_sum_rows(ctx->ggml_ctx, wf_p);        // [1, in_dim, F]
            // divide by sum(ws)
            auto ws_sum = ggml_sum_rows(ctx->ggml_ctx, ggml_reshape_2d(ctx->ggml_ctx, ws, p.num_layers, 1));  // [1,1]
            summed = ggml_div(ctx->ggml_ctx, summed, ws_sum);
            auto weighted = ggml_reshape_3d(ctx->ggml_ctx, summed, p.in_dim, summed->ne[2], 1);  // [in_dim, F, 1]

            auto enc = std::dynamic_pointer_cast<MotionEncoderTC>(blocks["encoder"]);
            return enc->forward(ctx, weighted);
        }
    };

    struct CausalAudioEncoderRunner : public GGMLRunner {
        CausalAudioEncoderParams p;
        CausalAudioEncoderBlock enc;
        std::string desc = "casual-audio-encoder";

        CausalAudioEncoderRunner(ggml_backend_t backend,
                                 ggml_backend_t params_backend,
                                 const String2TensorStorage& tensor_storage_map = {},
                                 const std::string prefix                       = "casual_audio_encoder")
            : GGMLRunner(backend, params_backend) {
            enc = CausalAudioEncoderBlock(p);
            enc.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override { return desc; }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            enc.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& feats, bool want_global) {
            ggml_cgraph* gf  = new_graph_custom(GGML_DEFAULT_GRAPH_SIZE * 4);
            ggml_tensor* f   = make_input(feats);
            auto runner_ctx  = get_context();
            auto out = enc.forward(&runner_ctx, f);
            // The runner extracts the graph's FINAL node. Expand exactly the one we
            // want last so it is the designated output:
            //   x_local  [dim, n_tok=5, F]  -> per-frame injector context
            //   x_global [dim, 1, F]        -> AdaLN temb (enable_adain)
            ggml_tensor* outp = (want_global && out.first) ? out.first : out.second;
            ggml_build_forward_expand(gf, outp);
            return gf;
        }

        // x_local: per-frame audio tokens [dim, 5, F].
        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& feats) {
            auto get_graph = [&]() -> ggml_cgraph* { return build_graph(feats, /*want_global=*/false); };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }

        // x_global: per-frame global embedding [dim, 1, F] (AdaLN temb). need_global.
        sd::Tensor<float> compute_global(int n_threads, const sd::Tensor<float>& feats) {
            auto get_graph = [&]() -> ggml_cgraph* { return build_graph(feats, /*want_global=*/true); };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }
    };

    // --------------------------------------------------------------------
    // CPU host helpers.
    // --------------------------------------------------------------------

    // Wav2Vec2FeatureExtractor: zero-mean / unit-variance normalization of the raw
    // 16kHz mono waveform (do_normalize=True). Returns the normalized waveform.
    inline std::vector<float> normalize_waveform(const std::vector<float>& wav) {
        if (wav.empty()) return wav;
        double mean = 0.0;
        for (float v : wav) mean += v;
        mean /= (double)wav.size();
        double var = 0.0;
        for (float v : wav) var += (v - mean) * (v - mean);
        var /= (double)wav.size();
        double inv = 1.0 / std::sqrt(var + 1e-7);
        std::vector<float> out(wav.size());
        for (size_t i = 0; i < wav.size(); i++) out[i] = (float)((wav[i] - mean) * inv);
        return out;
    }

    // linear_interpolation over time (align_corners=True) from in_fps to out_fps.
    // hs: [d_model, T_enc, 25] (ggml ne) -> resampled [d_model, out_len, 25].
    // Returns flat layout [layer][out_t][channel] matching the PyTorch
    // [num_layers, T, dim] used by get_audio_embed_bucket_fps.
    inline std::vector<float> interp_50_to_30(const sd::Tensor<float>& hs,
                                              int video_length,
                                              int& out_len) {
        int64_t d_model = hs.shape()[0];
        int64_t T_enc   = hs.shape()[1];
        int64_t n_hs    = hs.shape()[2];
        const float* p  = hs.data();
        auto at = [&](int64_t c, int64_t t, int64_t l) -> float {
            return p[l * T_enc * d_model + t * d_model + c];
        };
        // output_len = int(T_enc/50 * 30) unless video_length pins it.
        out_len = video_length > 0 ? video_length : (int)((double)T_enc / 50.0 * 30.0);
        std::vector<float> out((size_t)n_hs * out_len * d_model, 0.0f);
        for (int o = 0; o < out_len; o++) {
            float src = out_len == 1 ? 0.0f : (float)o * (float)(T_enc - 1) / (float)(out_len - 1);
            int t0   = (int)std::floor(src);
            int t1   = std::min<int>(t0 + 1, (int)T_enc - 1);
            float w1 = src - (float)t0;
            float w0 = 1.0f - w1;
            for (int64_t l = 0; l < n_hs; l++) {
                for (int64_t c = 0; c < d_model; c++) {
                    float v0 = at(c, t0, l);
                    float v1 = at(c, t1, l);
                    out[((size_t)l * out_len + o) * d_model + c] = w0 * v0 + w1 * v1;
                }
            }
        }
        return out;  // [num_layers][out_len][d_model]
    }

    // get_audio_embed_bucket_fps(fps=16, m=0, video_rate=30): with m=0 and
    // audio_sample_stride = int(30/16) = 1, chosen_idx = [bi] (single center frame).
    // Produces the per-bucket embedding [num_layers, audio_dim, bucket_num] (here we
    // return ggml-ne [audio_dim, bucket_num, num_layers] so it slots directly into
    // CausalAudioEncoderRunner's [in_dim, F, num_layers] input).
    //
    // interp: [num_layers][T][d_model] from interp_50_to_30. Returns the bucketed
    // tensor and sets num_repeat (= min_batch_num).
    inline sd::Tensor<float> audio_embed_bucket_fps(const std::vector<float>& interp,
                                                    int64_t num_layers,
                                                    int64_t d_model,
                                                    int audio_frame_num,
                                                    int infer_frames,
                                                    int fps,
                                                    int& num_repeat) {
        const int video_rate = 30;
        double scale = (double)video_rate / (double)fps;
        int min_batch_num = (int)((double)audio_frame_num / ((double)infer_frames * scale)) + 1;
        int bucket_num = min_batch_num * infer_frames;
        num_repeat = min_batch_num;
        int audio_sample_stride = (int)((double)video_rate / (double)fps);  // 1 for fps=16

        // get_sample_indices(original_fps=30, total=audio_frame_num+padd, target_fps=16,
        // num_sample=bucket_num, fixed_start=0). required_duration = bucket_num/16.
        int padd = (int)std::ceil((double)bucket_num / (double)fps * (double)video_rate) - audio_frame_num;
        int total = audio_frame_num + padd;
        double start_time = 0.0;
        double required_duration = (double)bucket_num / (double)fps;
        double end_time = start_time + required_duration;
        std::vector<int> batch_idx(bucket_num);
        for (int i = 0; i < bucket_num; i++) {
            double tp = start_time + (end_time - start_time) * (double)i / (double)bucket_num;  // endpoint=False
            int fi = (int)std::round(tp * (double)video_rate);
            fi = std::max(0, std::min(fi, total - 1));
            batch_idx[i] = fi;
        }

        sd::Tensor<float> out({d_model, (int64_t)bucket_num, num_layers});
        float* d = out.data();
        auto at = [&](int64_t l, int64_t t, int64_t c) -> float {
            return interp[((size_t)l * (size_t)/*T*/0 + 0) * d_model + 0];  // placeholder, replaced below
        };
        (void)at;
        int T = audio_frame_num;
        for (int bi_i = 0; bi_i < bucket_num; bi_i++) {
            int bi = batch_idx[bi_i];
            for (int64_t l = 0; l < num_layers; l++) {
                for (int64_t c = 0; c < d_model; c++) {
                    float v;
                    if (bi < T) {
                        // m=0 -> chosen_idx = [bi]; single center frame
                        v = interp[((size_t)l * T + bi) * d_model + c];
                    } else {
                        v = 0.0f;
                    }
                    d[((size_t)l * bucket_num + bi_i) * d_model + c] = v;
                }
            }
        }
        (void)audio_sample_stride;
        return out;  // [d_model, bucket_num, num_layers]
    }

}  // namespace WAV2VEC2

#endif  // __WAV2VEC2_HPP__
