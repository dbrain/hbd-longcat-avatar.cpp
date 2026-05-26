#ifndef __LONGCAT_AUDIO_HPP__
#define __LONGCAT_AUDIO_HPP__

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

#include "ggml_extend.hpp"
#include "model.h"

// LongCat-Video-Avatar 1.5 audio pathway.
//
// Reference (~/dev/longcat-video-ref):
//   - longcat_video/pipeline_longcat_video_avatar.py :: get_audio_embedding_whisper
//     (wav -> WhisperFeatureExtractor log-mel -> whisper-large-v3 ENCODER ->
//      ALL 33 hidden states grouped into 5 features -> [T_video, 5, 1280])
//   - run_demo_avatar_single_audio_to_video.py L215-221 (the +/-2 window over
//     video frames -> audio_emb [B, T_video, W=5, S=5, C=1280])
//   - longcat_video/modules/avatar/longcat_video_dit_avatar.py L420-441
//     (host-side windowing: first-frame window + latter-frame windows w/ vae_scale=4)
//   - longcat_video/modules/avatar/blocks.py :: AudioProjModel (dual proj1/proj1_vf
//     -> relu -> proj2 -> relu -> proj3 -> [.,32,768] -> LayerNorm)
//
// This file implements (CPU mel + GPU encoder graph + GPU/host audio_proj):
//   1. WhisperFeatureExtractor-equivalent 128-bin log-mel (CPU).
//   2. The whisper-large-v3 encoder (conv1 s1 + conv2 s2 + pos-embed + 32 pre-LN
//      transformer layers + final layer_norm) as a GGMLRunner, capturing all 33
//      hidden states (input embeds + 32 layer outputs).
//   3. The host-side audio windowing + AudioProjModel, producing audio_hidden_states
//      [768, 32, N_t] (per latent frame) consumed by the per-block audio cross-attn.
//
// Constants for avatar-v1.5 (verified from base_model_int8/config.json):
//   audio_window=5, audio_block=5, audio_channel=1280, intermediate_dim=512,
//   output_dim=768, context_tokens=32, vae_scale=4, audio_prenorm=False.
//   save_fps=25, audio_stride=1, ENC_FPS=50 (whisper encoder output rate).
namespace LONGCAT_AUDIO {

    // ---------------------------------------------------------------------
    // Minimal WAV reader: PCM16 / PCM32 / IEEE float, mono or multi-channel
    // (averaged to mono). Linear-resamples to 16 kHz if needed. The test input
    // is already 16 kHz mono PCM16, so the common path is a straight read.
    // Returns the mono float waveform in [-1, 1].
    // ---------------------------------------------------------------------
    inline bool load_wav_16k_mono(const std::string& path, std::vector<float>& out) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            LOG_ERROR("audio: cannot open wav '%s'", path.c_str());
            return false;
        }
        std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (buf.size() < 44 || std::memcmp(buf.data(), "RIFF", 4) != 0 || std::memcmp(buf.data() + 8, "WAVE", 4) != 0) {
            LOG_ERROR("audio: '%s' is not a RIFF/WAVE file", path.c_str());
            return false;
        }
        auto rd_u32 = [&](size_t o) { uint32_t v; std::memcpy(&v, buf.data() + o, 4); return v; };
        auto rd_u16 = [&](size_t o) { uint16_t v; std::memcpy(&v, buf.data() + o, 2); return v; };

        uint16_t audio_format = 1, channels = 1, bits = 16;
        uint32_t sample_rate = 16000;
        size_t data_off = 0, data_len = 0;
        size_t p = 12;
        while (p + 8 <= buf.size()) {
            uint32_t cksz = rd_u32(p + 4);
            if (std::memcmp(buf.data() + p, "fmt ", 4) == 0) {
                audio_format = rd_u16(p + 8);
                channels     = rd_u16(p + 10);
                sample_rate  = rd_u32(p + 12);
                bits         = rd_u16(p + 22);
            } else if (std::memcmp(buf.data() + p, "data", 4) == 0) {
                data_off = p + 8;
                data_len = cksz;
                break;
            }
            p += 8 + cksz + (cksz & 1);
        }
        if (data_off == 0 || channels == 0) {
            LOG_ERROR("audio: no data chunk in '%s'", path.c_str());
            return false;
        }
        data_len = std::min(data_len, buf.size() - data_off);

        std::vector<float> mono;
        const char* d = buf.data() + data_off;
        if (audio_format == 1 && bits == 16) {
            size_t n = data_len / 2;
            size_t frames = n / channels;
            mono.resize(frames);
            for (size_t i = 0; i < frames; i++) {
                float acc = 0.f;
                for (int c = 0; c < channels; c++) {
                    int16_t s;
                    std::memcpy(&s, d + (i * channels + c) * 2, 2);
                    acc += s / 32768.0f;
                }
                mono[i] = acc / channels;
            }
        } else if (audio_format == 3 && bits == 32) {
            size_t n = data_len / 4;
            size_t frames = n / channels;
            mono.resize(frames);
            for (size_t i = 0; i < frames; i++) {
                float acc = 0.f;
                for (int c = 0; c < channels; c++) {
                    float s;
                    std::memcpy(&s, d + (i * channels + c) * 4, 4);
                    acc += s;
                }
                mono[i] = acc / channels;
            }
        } else if (audio_format == 1 && bits == 32) {
            size_t n = data_len / 4;
            size_t frames = n / channels;
            mono.resize(frames);
            for (size_t i = 0; i < frames; i++) {
                float acc = 0.f;
                for (int c = 0; c < channels; c++) {
                    int32_t s;
                    std::memcpy(&s, d + (i * channels + c) * 4, 4);
                    acc += s / 2147483648.0f;
                }
                mono[i] = acc / channels;
            }
        } else {
            LOG_ERROR("audio: unsupported wav format=%u bits=%u in '%s'", audio_format, bits, path.c_str());
            return false;
        }

        if (sample_rate == 16000) {
            out = std::move(mono);
        } else {
            // linear resample to 16 kHz
            double ratio = 16000.0 / (double)sample_rate;
            size_t out_n = (size_t)(mono.size() * ratio);
            out.resize(out_n);
            for (size_t i = 0; i < out_n; i++) {
                double src = i / ratio;
                size_t i0  = (size_t)src;
                size_t i1  = std::min(i0 + 1, mono.size() - 1);
                float w    = (float)(src - i0);
                out[i]     = mono[i0] * (1 - w) + mono[i1] * w;
            }
            LOG_INFO("audio: resampled %u Hz -> 16000 Hz (%zu samples)", sample_rate, out_n);
        }

        // RUNTIME mouth-softening knob: optional 2nd-order Butterworth low-pass on the
        // 16 kHz mono signal (re-introduces the v1.0 wav2vec2 path's _smooth_transients,
        // which the v1.5 Whisper path dropped — it softens the plosive/sibilant
        // transients that spike visemes). Cutoff Hz from LONGCAT_AUDIO_LOWPASS; 0/unset =
        // off (default, signal unchanged). Forward biquad (causal; the slight group delay
        // is negligible vs the 50fps audio window).
        if (const char* lpe = getenv("LONGCAT_AUDIO_LOWPASS")) {
            float fc = (float)atof(lpe);
            if (fc > 0.0f && fc < 8000.0f && !out.empty()) {
                const double fs    = 16000.0;
                const double wc    = std::tan(M_PI * fc / fs);  // pre-warped
                const double wc2   = wc * wc;
                const double k     = std::sqrt(2.0) * wc;       // Butterworth Q=1/sqrt(2)
                const double norm  = 1.0 / (1.0 + k + wc2);
                const double b0    = wc2 * norm;
                const double b1    = 2.0 * b0;
                const double b2    = b0;
                const double a1    = 2.0 * (wc2 - 1.0) * norm;
                const double a2    = (1.0 - k + wc2) * norm;
                double x1 = 0, x2 = 0, y1 = 0, y2 = 0;
                for (size_t i = 0; i < out.size(); i++) {
                    double x0 = out[i];
                    double y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
                    x2 = x1; x1 = x0;
                    y2 = y1; y1 = y0;
                    out[i] = (float)y0;
                }
                LOG_INFO("audio: applied %.0f Hz Butterworth low-pass (mouth-softening)", fc);
            }
        }
        return true;
    }

    // ---------------------------------------------------------------------
    // WhisperFeatureExtractor (whisper-large-v3): 128-bin log-mel spectrogram.
    //   sampling_rate 16000, n_fft 400, hop_length 160, feature_size 128.
    //   Hann window (periodic), power spectrum, slaney mel filterbank
    //   (htk=False, norm="slaney"), log10, dynamic-range clamp to (max-8),
    //   then (x+4)/4 normalization. Pads the wav with reflect to center frames
    //   (HF uses center=True via n_fft//2 padding) — matched here.
    // ---------------------------------------------------------------------
    struct WhisperMel {
        static constexpr int kSampleRate = 16000;
        static constexpr int kNFFT       = 400;
        static constexpr int kHop        = 160;
        static constexpr int kNMels      = 128;

        std::vector<float> mel_filters;  // [kNMels * (kNFFT/2+1)] row-major (mel outer)
        std::vector<float> hann;         // [kNFFT]

        WhisperMel() {
            build_hann();
            build_mel_filters();
        }

        void build_hann() {
            hann.resize(kNFFT);
            // torch.hann_window(periodic=True): w[n] = 0.5 - 0.5*cos(2*pi*n/N)
            for (int n = 0; n < kNFFT; n++) {
                hann[n] = 0.5f - 0.5f * std::cos(2.0 * M_PI * n / (double)kNFFT);
            }
        }

        static double hz_to_mel_slaney(double f) {
            const double f_min  = 0.0;
            const double f_sp   = 200.0 / 3.0;
            double mels         = (f - f_min) / f_sp;
            const double min_lhz = 1000.0;
            const double min_lmel = (min_lhz - f_min) / f_sp;
            const double logstep  = std::log(6.4) / 27.0;
            if (f >= min_lhz) {
                mels = min_lmel + std::log(f / min_lhz) / logstep;
            }
            return mels;
        }

        static double mel_to_hz_slaney(double mel) {
            const double f_min   = 0.0;
            const double f_sp    = 200.0 / 3.0;
            double freq          = f_min + f_sp * mel;
            const double min_lhz = 1000.0;
            const double min_lmel = (min_lhz - f_min) / f_sp;
            const double logstep  = std::log(6.4) / 27.0;
            if (mel >= min_lmel) {
                freq = min_lhz * std::exp(logstep * (mel - min_lmel));
            }
            return freq;
        }

        // Slaney-normalised mel filterbank, identical to
        // transformers.audio_utils.mel_filter_bank(htk=False, norm="slaney").
        void build_mel_filters() {
            const int n_freqs = kNFFT / 2 + 1;  // 201
            mel_filters.assign((size_t)kNMels * n_freqs, 0.0f);

            // fft bin frequencies
            std::vector<double> fftfreqs(n_freqs);
            for (int i = 0; i < n_freqs; i++) {
                fftfreqs[i] = (double)i * kSampleRate / (double)kNFFT;
            }

            const double f_min = 0.0;
            const double f_max = kSampleRate / 2.0;  // 8000
            const double mel_min = hz_to_mel_slaney(f_min);
            const double mel_max = hz_to_mel_slaney(f_max);

            // kNMels+2 mel points
            std::vector<double> mel_pts(kNMels + 2);
            for (int i = 0; i < kNMels + 2; i++) {
                double m  = mel_min + (mel_max - mel_min) * i / (double)(kNMels + 1);
                mel_pts[i] = mel_to_hz_slaney(m);
            }

            std::vector<double> fdiff(kNMels + 1);
            for (int i = 0; i < kNMels + 1; i++) {
                fdiff[i] = mel_pts[i + 1] - mel_pts[i];
            }

            for (int m = 0; m < kNMels; m++) {
                for (int k = 0; k < n_freqs; k++) {
                    double lower = (fftfreqs[k] - mel_pts[m]) / fdiff[m];
                    double upper = (mel_pts[m + 2] - fftfreqs[k]) / fdiff[m + 1];
                    double val   = std::max(0.0, std::min(lower, upper));
                    // slaney normalisation
                    double enorm = 2.0 / (mel_pts[m + 2] - mel_pts[m]);
                    mel_filters[(size_t)m * n_freqs + k] = (float)(val * enorm);
                }
            }
        }

        // naive DFT magnitude^2 over a windowed frame; n_fft is small (400) and
        // the audio is short, so an O(n_fft^2) DFT per frame is fine.
        void power_spectrum(const float* frame, std::vector<double>& power) const {
            const int n_freqs = kNFFT / 2 + 1;
            power.assign(n_freqs, 0.0);
            for (int k = 0; k < n_freqs; k++) {
                double re = 0.0, im = 0.0;
                for (int n = 0; n < kNFFT; n++) {
                    double ang = -2.0 * M_PI * k * n / (double)kNFFT;
                    re += frame[n] * std::cos(ang);
                    im += frame[n] * std::sin(ang);
                }
                power[k] = re * re + im * im;
            }
        }

        // wav -> log-mel [kNMels, n_frames] (ggml-ne order dim0=n_mels? NO: we
        // return row-major mel-outer [kNMels][n_frames], i.e. data[m*n_frames+f]).
        // n_frames follows HF center-padded framing: 1 + len/hop (drops the final
        // n_fft//2-only frame, like np.abs(stft)[..., :-1]).
        std::vector<float> log_mel(const std::vector<float>& wav, int& n_frames_out) const {
            const int n_freqs = kNFFT / 2 + 1;
            const int pad     = kNFFT / 2;  // 200, reflect pad for center=True

            // reflect pad
            std::vector<float> padded((size_t)wav.size() + 2 * pad);
            for (int i = 0; i < pad; i++) {
                int src        = std::min((int)wav.size() - 1, pad - i);
                padded[i]      = src >= 0 && src < (int)wav.size() ? wav[src] : 0.0f;
            }
            for (size_t i = 0; i < wav.size(); i++) {
                padded[pad + i] = wav[i];
            }
            for (int i = 0; i < pad; i++) {
                int src = (int)wav.size() - 2 - i;
                padded[pad + wav.size() + i] = (src >= 0 && src < (int)wav.size()) ? wav[src] : 0.0f;
            }

            int n_frames = 1 + (int)((padded.size() - kNFFT) / kHop);
            if (n_frames < 1) {
                n_frames = 0;
            }
            // HF drops the last STFT frame (stft[..., :-1])
            if (n_frames > 0) {
                n_frames -= 1;
            }
            n_frames_out = n_frames;

            std::vector<float> out((size_t)kNMels * n_frames, 0.0f);
            std::vector<float> frame(kNFFT);
            std::vector<double> power;
            float global_max = -1e30f;

            for (int f = 0; f < n_frames; f++) {
                int start = f * kHop;
                for (int n = 0; n < kNFFT; n++) {
                    frame[n] = padded[start + n] * hann[n];
                }
                power_spectrum(frame.data(), power);
                for (int m = 0; m < kNMels; m++) {
                    double acc = 0.0;
                    const float* filt = &mel_filters[(size_t)m * n_freqs];
                    for (int k = 0; k < n_freqs; k++) {
                        acc += filt[k] * power[k];
                    }
                    double logv = std::log10(std::max(acc, 1e-10));
                    float v     = (float)logv;
                    out[(size_t)m * n_frames + f] = v;
                    if (v > global_max) {
                        global_max = v;
                    }
                }
            }

            // dynamic range clamp + normalize: max(x, max-8); (x+4)/4
            float floor = global_max - 8.0f;
            for (auto& v : out) {
                v = std::max(v, floor);
                v = (v + 4.0f) / 4.0f;
            }
            return out;
        }
    };

    // ---------------------------------------------------------------------
    // Whisper-large-v3 ENCODER (encoder-only gguf, prefix audio_encoder.*).
    //   conv1 (k3 s1 p1, 128->1280) + gelu, conv2 (k3 s2 p1, 1280->1280) + gelu,
    //   + embed_positions, then 32 pre-LN transformer layers (eager MHA,
    //   k_proj no bias), final layer_norm. We capture ALL 33 hidden states
    //   (post conv+posembed input embeds, then each of 32 layer outputs).
    // ---------------------------------------------------------------------
    struct WhisperEncoderParams {
        int n_layers      = 32;
        int64_t d_model   = 1280;
        int64_t n_heads   = 20;
        int64_t head_dim  = 64;
        int64_t ffn_dim   = 5120;
        int64_t n_mels    = 128;
        int max_source_positions = 1500;
        float eps         = 1e-5f;
    };

    class WhisperEncoderLayer : public GGMLBlock {
    protected:
        int64_t d_model;
        int64_t n_heads;
        int64_t head_dim;
        int64_t ffn_dim;
        float eps;

    public:
        WhisperEncoderLayer(int64_t d_model, int64_t n_heads, int64_t head_dim, int64_t ffn_dim, float eps)
            : d_model(d_model), n_heads(n_heads), head_dim(head_dim), ffn_dim(ffn_dim), eps(eps) {
            blocks["self_attn_layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model, eps, true, true));
            blocks["self_attn.q_proj"]     = std::shared_ptr<GGMLBlock>(new Linear(d_model, d_model, true));
            blocks["self_attn.k_proj"]     = std::shared_ptr<GGMLBlock>(new Linear(d_model, d_model, false));  // NO bias
            blocks["self_attn.v_proj"]     = std::shared_ptr<GGMLBlock>(new Linear(d_model, d_model, true));
            blocks["self_attn.out_proj"]   = std::shared_ptr<GGMLBlock>(new Linear(d_model, d_model, true));
            blocks["final_layer_norm"]     = std::shared_ptr<GGMLBlock>(new LayerNorm(d_model, eps, true, true));
            blocks["fc1"]                  = std::shared_ptr<GGMLBlock>(new Linear(d_model, ffn_dim, true));
            blocks["fc2"]                  = std::shared_ptr<GGMLBlock>(new Linear(ffn_dim, d_model, true));
        }

        // x: [d_model, T, 1]
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* x, bool flash_attn) {
            auto ln1     = std::dynamic_pointer_cast<LayerNorm>(blocks["self_attn_layer_norm"]);
            auto q_proj  = std::dynamic_pointer_cast<Linear>(blocks["self_attn.q_proj"]);
            auto k_proj  = std::dynamic_pointer_cast<Linear>(blocks["self_attn.k_proj"]);
            auto v_proj  = std::dynamic_pointer_cast<Linear>(blocks["self_attn.v_proj"]);
            auto outp    = std::dynamic_pointer_cast<Linear>(blocks["self_attn.out_proj"]);
            auto ln2     = std::dynamic_pointer_cast<LayerNorm>(blocks["final_layer_norm"]);
            auto fc1     = std::dynamic_pointer_cast<Linear>(blocks["fc1"]);
            auto fc2     = std::dynamic_pointer_cast<Linear>(blocks["fc2"]);

            int64_t T = x->ne[1];
            int64_t N = x->ne[2];

            // self-attention (pre-LN). Whisper scales q by head_dim^-0.5 INSIDE
            // q_proj output (HF applies self.scaling to q); ggml_ext_attention_ext
            // applies 1/sqrt(d_head) itself, so we let it handle the scale.
            auto residual = x;
            auto h        = ln1->forward(ctx, x);
            auto q        = q_proj->forward(ctx, h);
            auto k        = k_proj->forward(ctx, h);
            auto v        = v_proj->forward(ctx, h);
            auto attn     = ggml_ext_attention_ext(ctx->ggml_ctx, ctx->backend, q, k, v,
                                                   n_heads, nullptr, false, flash_attn);  // [d_model, T, N]
            attn          = outp->forward(ctx, attn);
            x             = ggml_add(ctx->ggml_ctx, residual, attn);

            // ffn (pre-LN), gelu
            residual = x;
            h        = ln2->forward(ctx, x);
            h        = fc1->forward(ctx, h);
            h        = ggml_ext_gelu(ctx->ggml_ctx, h, false);
            h        = fc2->forward(ctx, h);
            x        = ggml_add(ctx->ggml_ctx, residual, h);

            (void)T;
            (void)N;
            return x;
        }
    };

    class WhisperEncoder : public GGMLBlock {
    protected:
        WhisperEncoderParams p;

        // conv1/conv2 front-end + positional embedding are direct params of the
        // encoder (gguf names audio_encoder.conv1.* / conv2.* / embed_positions.*).
        // ggml stores conv1 weight as [kernel=3, in, out] == ggml_conv_1d's [K,IC,OC].
        void init_params(ggml_context* ctx, const String2TensorStorage& tensor_storage_map, const std::string prefix = "") override {
            (void)tensor_storage_map;
            params["conv1.weight"]           = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, p.n_mels, p.d_model);
            params["conv1.bias"]             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.d_model);
            params["conv2.weight"]           = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 3, p.d_model, p.d_model);
            params["conv2.bias"]             = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, p.d_model);
            params["embed_positions.weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, p.d_model, p.max_source_positions);
        }

    public:
        WhisperEncoder() {}
        WhisperEncoder(WhisperEncoderParams p)
            : p(p) {
            for (int i = 0; i < p.n_layers; i++) {
                blocks["layers." + std::to_string(i)] = std::shared_ptr<GGMLBlock>(
                    new WhisperEncoderLayer(p.d_model, p.n_heads, p.head_dim, p.ffn_dim, p.eps));
            }
            blocks["layer_norm"] = std::shared_ptr<GGMLBlock>(new LayerNorm(p.d_model, p.eps, true, true));
        }

        // mel: [T_mel, n_mels, 1]. Captures all 33 hidden states (input embeds +
        // 32 layer outputs) and returns them STACKED as [d_model, T, 33].
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* mel, bool flash_attn) {
            auto lnf   = std::dynamic_pointer_cast<LayerNorm>(blocks["layer_norm"]);
            ggml_tensor* pos = params["embed_positions.weight"];  // [d_model, max_pos]
            int64_t d_model  = p.d_model;

            // conv front-end. conv1: k3 s1 p1; conv2: k3 s2 p1; gelu after each.
            ggml_tensor* w1 = params["conv1.weight"];
            ggml_tensor* b1 = params["conv1.bias"];
            ggml_tensor* w2 = params["conv2.weight"];
            ggml_tensor* b2 = params["conv2.bias"];
            auto x = ggml_conv_1d(ctx->ggml_ctx, w1, mel, 1, 1, 1);  // [T_mel, d_model, 1]
            x      = ggml_add(ctx->ggml_ctx, x, ggml_reshape_3d(ctx->ggml_ctx, b1, 1, d_model, 1));
            x      = ggml_gelu(ctx->ggml_ctx, x);
            x      = ggml_conv_1d(ctx->ggml_ctx, w2, x, 2, 1, 1);  // [T_out, d_model, 1]
            x      = ggml_add(ctx->ggml_ctx, x, ggml_reshape_3d(ctx->ggml_ctx, b2, 1, d_model, 1));
            x      = ggml_gelu(ctx->ggml_ctx, x);
            x      = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, x, 1, 0, 2, 3));  // [d_model, T, 1]
            int64_t T = x->ne[1];

            // add positional embeddings (first T rows)
            auto pos_slice = ggml_view_2d(ctx->ggml_ctx, pos, pos->ne[0], T, pos->nb[1], 0);
            x              = ggml_add(ctx->ggml_ctx, x, ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, pos_slice), p.d_model, T, 1));

            // hidden_states[0] = input embeds (post conv + posembed). HF returns
            // the pre-layer hidden state as hidden_states[0].
            std::vector<ggml_tensor*> hs;
            hs.push_back(x);

            for (int i = 0; i < p.n_layers; i++) {
                auto layer = std::dynamic_pointer_cast<WhisperEncoderLayer>(blocks["layers." + std::to_string(i)]);
                x          = layer->forward(ctx, x, flash_attn);
                // HF appends the hidden state AFTER each layer; the final entry
                // (index 32) is post-final-layer-norm. We apply layer_norm only to
                // the captured copy for the last layer to match get_audio_embedding's
                // hidden_states tuple (encoder applies final LN to last_hidden_state,
                // and hidden_states[-1] == post-LN output in HF Whisper).
                if (i == p.n_layers - 1) {
                    hs.push_back(lnf->forward(ctx, x));
                } else {
                    hs.push_back(x);
                }
            }

            // stack along a new dim -> [d_model, T, 33]
            ggml_tensor* stacked = hs[0];
            stacked              = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, stacked), p.d_model, T, 1);
            for (size_t i = 1; i < hs.size(); i++) {
                auto h = ggml_reshape_3d(ctx->ggml_ctx, ggml_cont(ctx->ggml_ctx, hs[i]), p.d_model, T, 1);
                stacked = ggml_concat(ctx->ggml_ctx, stacked, h, 2);
            }
            return stacked;  // [d_model, T, 33]
        }
    };

    struct WhisperEncoderRunner : public GGMLRunner {
        WhisperEncoderParams p;
        WhisperEncoder encoder;
        std::string desc = "whisper-encoder";

        WhisperEncoderRunner(ggml_backend_t backend,
                             ggml_backend_t params_backend,
                             const String2TensorStorage& tensor_storage_map = {},
                             const std::string prefix                       = "audio_encoder")
            : GGMLRunner(backend, params_backend) {
            encoder = WhisperEncoder(p);
            encoder.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return desc;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            encoder.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& mel_tensor) {
            ggml_cgraph* gf  = new_graph_custom(GGML_DEFAULT_GRAPH_SIZE * 8);
            ggml_tensor* mel = make_input(mel_tensor);  // [T_mel, n_mels, 1]
            auto runner_ctx  = get_context();
            ggml_tensor* out = encoder.forward(&runner_ctx, mel, runner_ctx.flash_attn_enabled);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        // mel_tensor: [T_mel, n_mels, 1]; returns [d_model, T, 33].
        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& mel_tensor) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(mel_tensor);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }
    };

#if 0
    // NOTE: The AudioProjModel runs INSIDE the DiT graph (LONGCAT_AVATAR::LongCatAvatar::
    // audio_proj, loaded under model.diffusion_model.audio_proj.*). The standalone
    // AudioProjModel/AudioProjRunner below are kept (disabled) as a reference of the
    // dual-window MLP shape only — do NOT instantiate (would double-load audio_proj).
    struct AudioProjParams {
        int64_t input_dim    = 32000;  // seq_len(5)*blocks(5)*channels(1280)
        int64_t input_dim_vf = 51200;  // seq_len_vf(8)*blocks(5)*channels(1280)
        int64_t intermediate = 512;
        int64_t context_tokens = 32;
        int64_t output_dim   = 768;
    };

    class AudioProjModel : public GGMLBlock {
    protected:
        AudioProjParams p;

    public:
        AudioProjModel() {}
        AudioProjModel(AudioProjParams p)
            : p(p) {
            blocks["proj1"]    = std::shared_ptr<GGMLBlock>(new Linear(p.input_dim, p.intermediate, true));
            blocks["proj1_vf"] = std::shared_ptr<GGMLBlock>(new Linear(p.input_dim_vf, p.intermediate, true));
            blocks["proj2"]    = std::shared_ptr<GGMLBlock>(new Linear(p.intermediate, p.intermediate, true));
            blocks["proj3"]    = std::shared_ptr<GGMLBlock>(new Linear(p.intermediate, p.context_tokens * p.output_dim, true));
            blocks["norm"]     = std::shared_ptr<GGMLBlock>(new LayerNorm(p.output_dim, 1e-5f, true, true));
        }

        // first:  [input_dim, n_first]    (n_first = 1)
        // latter: [input_dim_vf, n_latter]
        // returns [output_dim, context_tokens, n_first + n_latter]
        ggml_tensor* forward(GGMLRunnerContext* ctx, ggml_tensor* first, ggml_tensor* latter) {
            auto proj1    = std::dynamic_pointer_cast<Linear>(blocks["proj1"]);
            auto proj1_vf = std::dynamic_pointer_cast<Linear>(blocks["proj1_vf"]);
            auto proj2    = std::dynamic_pointer_cast<Linear>(blocks["proj2"]);
            auto proj3    = std::dynamic_pointer_cast<Linear>(blocks["proj3"]);
            auto norm     = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);

            auto a = ggml_relu(ctx->ggml_ctx, proj1->forward(ctx, first));      // [512, n_first]
            auto b = ggml_relu(ctx->ggml_ctx, proj1_vf->forward(ctx, latter));  // [512, n_latter]
            auto c = ggml_concat(ctx->ggml_ctx, a, b, 1);                       // [512, N_t]
            c      = ggml_relu(ctx->ggml_ctx, proj2->forward(ctx, c));          // [512, N_t]
            c      = proj3->forward(ctx, c);                                    // [32*768, N_t]

            int64_t N_t = c->ne[1];
            c = ggml_reshape_3d(ctx->ggml_ctx, c, p.output_dim, p.context_tokens, N_t);  // [768, 32, N_t]
            c = norm->forward(ctx, c);
            return c;  // [768, 32, N_t]
        }
    };

    struct AudioProjRunner : public GGMLRunner {
        AudioProjParams p;
        AudioProjModel proj;
        std::string desc = "audio-proj";

        AudioProjRunner(ggml_backend_t backend,
                        ggml_backend_t params_backend,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string prefix                       = "model.diffusion_model.audio_proj")
            : GGMLRunner(backend, params_backend) {
            proj = AudioProjModel(p);
            proj.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return desc;
        }

        void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
            proj.get_param_tensors(tensors, prefix);
        }

        ggml_cgraph* build_graph(const sd::Tensor<float>& first, const sd::Tensor<float>& latter) {
            ggml_cgraph* gf = new_graph_custom(GGML_DEFAULT_GRAPH_SIZE);
            ggml_tensor* f  = make_input(first);
            ggml_tensor* l  = make_input(latter);
            auto runner_ctx = get_context();
            ggml_tensor* out = proj.forward(&runner_ctx, f, l);
            ggml_build_forward_expand(gf, out);
            return gf;
        }

        sd::Tensor<float> compute(int n_threads, const sd::Tensor<float>& first, const sd::Tensor<float>& latter) {
            auto get_graph = [&]() -> ggml_cgraph* {
                return build_graph(first, latter);
            };
            return take_or_empty(GGMLRunner::compute<float>(get_graph, n_threads, false));
        }
    };
#endif  // standalone AudioProjModel/AudioProjRunner reference

    // ---------------------------------------------------------------------
    // Host-side audio token preparation.
    //
    // Reproduces (CPU, fp32):
    //   pipeline get_audio_embedding_whisper: group the 33 whisper hidden states
    //     into 5 features (mean of [0:8],[8:16],[16:24],[24:32]; last=hs[32]),
    //     each at ENC_FPS=50, then linearly interpolate (align_corners=True) to
    //     fps=25 with output_len = video_length -> full_audio_emb [video_length, 5, 1280].
    //   demo center_indices: per video frame t, window full_audio_emb[t-2..t+2]
    //     (clamped) -> audio_emb [T_video, W=5, S=5, C=1280].
    //   DiT forward windowing (L420-441, vae_scale=4, audio_window=5): build the
    //     first-frame window (proj1, dim 32000) and latter-frame windows (proj1_vf,
    //     dim 51200), N_t latent frames total.
    //
    // whisper_hs: [d_model, T_enc, 33] (ggml-ne) from WhisperEncoderRunner.
    // Produces `first` [32000, 1] and `latter` [51200, n_latter] ready for AudioProjRunner.
    struct AudioWindowConfig {
        int audio_window = 5;
        int vae_scale    = 4;
        int n_feats      = 5;     // grouped whisper features (blocks)
        int64_t channels = 1280;
        float enc_fps    = 50.f;
        float fps        = 25.f;
    };

    // group whisper hidden states [d_model, T_enc, 33] into [T_enc, 5, d_model]
    // (5 grouped features) then linearly interpolate T_enc(@enc_fps) -> video_length(@fps).
    // Returns full_audio_emb flat [video_length * 5 * d_model], layout
    // [t][feat][channel] (t outer).
    inline std::vector<float> build_full_audio_emb(const sd::Tensor<float>& whisper_hs,
                                                   int video_length,
                                                   const AudioWindowConfig& cfg) {
        int64_t d_model = whisper_hs.shape()[0];
        int64_t T_enc   = whisper_hs.shape()[1];
        int64_t n_hs    = whisper_hs.shape()[2];  // 33
        const float* hs = whisper_hs.data();
        auto at = [&](int64_t c, int64_t t, int64_t l) -> float {
            return hs[l * T_enc * d_model + t * d_model + c];
        };

        // grouped[t][feat][c], feat in [0,5)
        const int F = cfg.n_feats;
        std::vector<float> grouped((size_t)T_enc * F * d_model, 0.0f);
        auto group_ranges = std::vector<std::pair<int, int>>{{0, 8}, {8, 16}, {16, 24}, {24, 32}, {32, 33}};
        for (int64_t t = 0; t < T_enc; t++) {
            for (int f = 0; f < F; f++) {
                int lo = group_ranges[f].first;
                int hi = std::min<int>(group_ranges[f].second, (int)n_hs);
                float inv = 1.0f / (float)(hi - lo);
                for (int64_t c = 0; c < d_model; c++) {
                    float acc = 0.0f;
                    for (int l = lo; l < hi; l++) {
                        acc += at(c, t, l);
                    }
                    grouped[((size_t)t * F + f) * d_model + c] = acc * inv;
                }
            }
        }

        // F.interpolate(mode='linear', align_corners=True) from T_enc -> video_length.
        // The interpolation is over the time axis (per feat,channel independent).
        int out_len = video_length;
        std::vector<float> full((size_t)out_len * F * d_model, 0.0f);
        for (int o = 0; o < out_len; o++) {
            float src;
            if (out_len == 1) {
                src = 0.0f;
            } else {
                src = (float)o * (float)(T_enc - 1) / (float)(out_len - 1);  // align_corners=True
            }
            int t0   = (int)std::floor(src);
            int t1   = std::min<int>(t0 + 1, (int)T_enc - 1);
            float w1 = src - (float)t0;
            float w0 = 1.0f - w1;
            for (int f = 0; f < F; f++) {
                for (int64_t c = 0; c < d_model; c++) {
                    float v0 = grouped[((size_t)t0 * F + f) * d_model + c];
                    float v1 = grouped[((size_t)t1 * F + f) * d_model + c];
                    full[((size_t)o * F + f) * d_model + c] = w0 * v0 + w1 * v1;
                }
            }
        }
        return full;  // [video_length][5][d_model]
    }

    // Build proj1/proj1_vf inputs from full_audio_emb. Returns N_t (audio latent
    // frames = 1 + (T_video-1)/vae_scale).
    inline int build_proj_inputs(const std::vector<float>& full_audio_emb,
                                 int T_video,
                                 const AudioWindowConfig& cfg,
                                 sd::Tensor<float>& first_out,
                                 sd::Tensor<float>& latter_out,
                                 int frame_offset = 0) {
        const int W   = cfg.audio_window;       // 5
        const int S   = cfg.n_feats;            // 5
        const int64_t C = cfg.channels;         // 1280
        const int n   = cfg.vae_scale;          // 4
        const int mid = W / 2;                  // 2

        int video_length = (int)(full_audio_emb.size() / ((size_t)S * C));
        // audio_emb[t][w][s][c]: full_audio_emb[clamp(frame_offset+t-mid+w)][s][c].
        // frame_offset shifts the window into the GLOBAL audio timeline so a
        // continuation segment starting at global video-frame `frame_offset`
        // windows the correct slice of the audio (lip-sync continues across segments).
        auto audio_emb = [&](int t, int w, int s, int64_t c) -> float {
            int idx = frame_offset + t - mid + w;
            idx     = std::max(0, std::min(idx, video_length - 1));
            return full_audio_emb[((size_t)idx * S + s) * C + c];
        };

        int n_latter = (T_video - 1) / n;  // n_t
        int N_t      = 1 + n_latter;

        // first frame window: [W, S, C] -> flat input_dim (W*S*C = 32000), layout w-outer.
        int64_t input_dim = (int64_t)W * S * C;
        first_out         = sd::Tensor<float>({input_dim, 1});
        {
            float* d = first_out.data();
            for (int w = 0; w < W; w++) {
                for (int s = 0; s < S; s++) {
                    for (int64_t c = 0; c < C; c++) {
                        d[((size_t)w * S + s) * C + c] = audio_emb(0, w, s, c);
                    }
                }
            }
        }

        // latter frames: for each n_t, concat 8 window-entries (= W-1+vae_scale).
        // Per latter latent frame nt (covering video frames [1 + nt*n .. 1 + nt*n + n-1]),
        // with the 4 sub-frames' windows reduced to 8 entries:
        //   first sub-frame (n-index 0): w in [0..mid]   -> mid+1=3 entries
        //   middle sub-frames (n-index 1..n-2): w == mid -> 1 entry each -> (n-2)=2
        //   last sub-frame (n-index n-1): w in [mid..W-1]-> W-mid=3 entries
        // concat order: first, middle, last  (matches reference torch.concat).
        int64_t input_dim_vf = (int64_t)(W - 1 + n) * S * C;  // 8*5*1280 = 51200
        latter_out           = sd::Tensor<float>({input_dim_vf, (int64_t)std::max(1, n_latter)});
        if (n_latter > 0) {
            float* d = latter_out.data();
            for (int nt = 0; nt < n_latter; nt++) {
                // video frame indices for this latent frame's sub-frames:
                // latter_frame_audio_emb = audio_cond[:, 1:], reshaped (n_t n).
                // So sub-frame k maps to video frame (1 + nt*n + k).
                float* dst = d + (size_t)nt * input_dim_vf;
                int e      = 0;  // window-entry index [0..8)
                auto emit  = [&](int vframe, int w) {
                    for (int s = 0; s < S; s++) {
                        for (int64_t c = 0; c < C; c++) {
                            dst[((size_t)e * S + s) * C + c] = audio_emb(vframe, w, s, c);
                        }
                    }
                    e++;
                };
                // first sub-frame: w 0..mid
                int vf0 = 1 + nt * n + 0;
                for (int w = 0; w <= mid; w++) {
                    emit(vf0, w);
                }
                // middle sub-frames: n-index 1..n-2, w == mid
                for (int k = 1; k <= n - 2; k++) {
                    emit(1 + nt * n + k, mid);
                }
                // last sub-frame: n-index n-1, w mid..W-1
                int vfl = 1 + nt * n + (n - 1);
                for (int w = mid; w < W; w++) {
                    emit(vfl, w);
                }
                GGML_ASSERT(e == W - 1 + n);
            }
        }
        return N_t;
    }

}  // namespace LONGCAT_AUDIO

#endif  // __LONGCAT_AUDIO_HPP__
