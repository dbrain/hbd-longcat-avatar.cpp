// Dumps every MiniMax-H3 tensor permutation as JSON so it can be diffed element-for-element
// against the PR's reference implementation running under real torch.
//
// Three implementations of the SAME permutation exist in this port and all three are checked:
//   1. the ggml op            DiT::patchify_3d / unpatchify_3d, MiniMaxH3Runner::pack_audio /
//                             unpack_audio / merge_av, ViT3DDecoder::unpatchify
//   2. the host-side twin     minimax_h3_patchify_cond_rows / minimax_h3_pack_audio_cond_rows /
//                             minimax_h3_swap_audio_axes / minimax_h3_unpack_audio_latent
//   3. comfy's reference      patchify_video / unpatchify_video / pack_audio / unpack_audio
//
// (1) and (2) come out of this binary, (3) out of ref_permute_torch.py, and diff_permute.py
// requires all three to agree exactly.  Every tensor is `arange`-filled, so a permutation error
// cannot hide behind a plausible-looking value.
//
// A CPU ggml build is enough -- no CUDA, no weights.  See the header of diff_permute.py for the
// exact build and run commands.
//
// The case matrix must stay in lockstep with ref_permute_torch.py.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-cpu.h"
#include "model/diffusion/dit.hpp"
#include "model/diffusion/minimax_h3.hpp"
#include "model/diffusion/minimax_h3_host_layout.hpp"
#include "model/vae/minimax_h3_vae.hpp"

// The engine's headers reference the CUDA backend unconditionally; a CPU-only ggml does not
// export these four.  They are never reached on this path -- no graph here runs through
// GGMLRunner::compute -- so a `false` stub is the whole requirement.
extern "C" {
bool ggml_backend_is_cuda(ggml_backend_t) {
    return false;
}
void ggml_backend_cuda_trim_memory(ggml_backend_t) {}
bool ggml_cuda_nvfp4_weight_global_folded(ggml_backend_t, const char*) {
    return false;
}
bool ggml_cuda_nvfp4_f16_dst_available(ggml_backend_t) {
    return false;
}
}

namespace {

    int g_failures    = 0;
    int64_t g_checked = 0;

    // ------------------------------------------------------------------------------------
    // tiny ggml scaffolding: one context, graphs computed straight on the CPU
    // ------------------------------------------------------------------------------------

    struct Scratch {
        ggml_context* ctx = nullptr;

        explicit Scratch(size_t bytes) {
            ggml_init_params p = {bytes, nullptr, false};
            ctx                = ggml_init(p);
            GGML_ASSERT(ctx != nullptr);
        }
        ~Scratch() {
            ggml_free(ctx);
        }
        Scratch(const Scratch&)            = delete;
        Scratch& operator=(const Scratch&) = delete;
    };

    ggml_tensor* arange_tensor(ggml_context* ctx, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3) {
        ggml_tensor* t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne0, ne1, ne2, ne3);
        float* d       = static_cast<float*>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); ++i) {
            d[i] = static_cast<float>(i);
        }
        return t;
    }

    std::vector<float> arange_vec(int64_t n) {
        std::vector<float> v(static_cast<size_t>(n));
        for (int64_t i = 0; i < n; ++i) {
            v[static_cast<size_t>(i)] = static_cast<float>(i);
        }
        return v;
    }

    std::vector<float> compute_flat(ggml_context* ctx, ggml_tensor* out) {
        ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, false);
        ggml_build_forward_expand(gf, out);
        ggml_graph_compute_with_ctx(ctx, gf, 1);
        GGML_ASSERT(ggml_is_contiguous(out));
        const int64_t n = ggml_nelements(out);
        std::vector<float> v(static_cast<size_t>(n));
        std::memcpy(v.data(), out->data, static_cast<size_t>(n) * sizeof(float));
        return v;
    }

    // ------------------------------------------------------------------------------------
    // JSON emission (integers throughout -- every value is an arange index)
    // ------------------------------------------------------------------------------------

    void emit_array(const char* key, const std::vector<float>& v, bool last = false) {
        std::printf("\"%s\":[", key);
        for (size_t i = 0; i < v.size(); ++i) {
            std::printf(i + 1 == v.size() ? "%d" : "%d,", static_cast<int>(v[i]));
        }
        std::printf(last ? "]" : "],");
    }

    void check_equal(const char* what, const std::vector<float>& got, const std::vector<float>& want) {
        if (got.size() != want.size()) {
            std::fprintf(stderr, "FAIL %s: %zu elements, expected %zu\n", what, got.size(), want.size());
            ++g_failures;
            return;
        }
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != want[i]) {
                std::fprintf(stderr,
                             "FAIL %s: element %zu is %g, expected %g\n",
                             what,
                             i,
                             static_cast<double>(got[i]),
                             static_cast<double>(want[i]));
                ++g_failures;
                return;
            }
        }
        g_checked += static_cast<int64_t>(got.size());
    }

    // ------------------------------------------------------------------------------------
    // case matrix -- keep in lockstep with ref_permute_torch.py
    // ------------------------------------------------------------------------------------

    struct PatchCase {
        const char* name;
        int64_t c, t, h, w;
    };

    const PatchCase PATCH_CASES[] = {
        {"min", 24, 1, 2, 2},
        {"t2va_small", 24, 5, 8, 12},
        {"chan3", 3, 2, 4, 6},
        {"wide", 24, 3, 4, 20},
        {"tall", 24, 2, 20, 4},
        {"odd_c", 7, 4, 6, 6},
    };

    struct AudioCase {
        const char* name;
        int64_t c, ch, t;
    };

    const AudioCase AUDIO_CASES[] = {
        {"min", 32, 2, 1},
        {"t2va", 32, 2, 7},
        {"tiny", 4, 2, 5},
        {"mono", 32, 1, 3},
        {"full_207", 32, 2, 207},
    };

    struct VaeCase {
        const char* name;
        int64_t c, pt, ph, t, h, w;  // ph == pw: the decoder reads one `patch_size` for both
    };

    const VaeCase VAE_CASES[] = {
        {"vae_real_1", 3, 4, 16, 1, 1, 1},
        {"vae_real_2x2", 3, 4, 16, 1, 2, 2},
        {"vae_real_t2", 3, 4, 16, 2, 1, 3},
        {"vae_synth", 2, 2, 3, 2, 3, 5},
    };

    // ------------------------------------------------------------------------------------
    // 1. conditioning-row / target-row patchify
    // ------------------------------------------------------------------------------------

    void run_patchify() {
        std::printf("\"patchify\":[");
        bool first = true;
        for (const PatchCase& c : PATCH_CASES) {
            if (!first) {
                std::printf(",");
            }
            first = false;
            Scratch s(512ull * 1024 * 1024);

            // ggml [W, H, T, C] == torch [1, C, T, H, W] laid out contiguously
            ggml_tensor* x = arange_tensor(s.ctx, c.w, c.h, c.t, c.c);
            ggml_tensor* rows =
                DiT::patchify_3d(s.ctx, x, /*pt*/ 1, /*ph*/ 2, /*pw*/ 2, /*N*/ 1, /*patch_last*/ true);
            std::vector<float> ggml_rows = compute_flat(s.ctx, rows);

            // the runner feeds patchify_3d's output straight back to unpatchify_3d
            ggml_tensor* rows_in = arange_tensor(s.ctx, c.c * 4, c.t * (c.h / 2) * (c.w / 2), 1, 1);
            std::memcpy(rows_in->data, ggml_rows.data(), ggml_rows.size() * sizeof(float));
            ggml_tensor* back               = DiT::unpatchify_3d(s.ctx,
                                                                 rows_in,
                                                                 c.t,
                                                                 c.h / 2,
                                                                 c.w / 2,
                                                                 /*pt*/ 1,
                                                                 /*ph*/ 2,
                                                                 /*pw*/ 2,
                                                                 /*patch_last*/ true);
            std::vector<float> ggml_unpatch = compute_flat(s.ctx, back);

            // host-side twin, same input buffer
            sd::Tensor<float> latent({c.w, c.h, c.t, c.c}, arange_vec(c.c * c.t * c.h * c.w));
            sd::Tensor<float> host = minimax_h3_patchify_cond_rows(latent);

            // The two must agree before either is compared to torch: this is the pair the
            // W23 warning is about (a keyframe scanned differently from the target renders fine).
            check_equal((std::string("patchify/") + c.name + " host-vs-ggml").c_str(), host.values(), ggml_rows);
            check_equal((std::string("patchify/") + c.name + " roundtrip").c_str(),
                        ggml_unpatch,
                        arange_vec(c.c * c.t * c.h * c.w));

            std::printf("{\"name\":\"%s\",", c.name);
            emit_array("rows", ggml_rows);
            emit_array("rows_host", host.values());
            emit_array("unpatch", ggml_unpatch, true);
            std::printf("}");
        }
        std::printf("],");
    }

    // ------------------------------------------------------------------------------------
    // 2. audio pack / unpack, the {T,C,S} <-> {T,S,C} swap, and the packed-latent transport
    // ------------------------------------------------------------------------------------

    void run_audio(MiniMaxH3::MiniMaxH3Runner& runner) {
        std::printf("\"audio\":[");
        bool first = true;
        for (const AudioCase& c : AUDIO_CASES) {
            if (!first) {
                std::printf(",");
            }
            first = false;
            Scratch s(512ull * 1024 * 1024);

            // request layout: ggml [T, ch, C, 1] == torch [1, C, ch, T]
            ggml_tensor* ax              = arange_tensor(s.ctx, c.t, c.ch, c.c, 1);
            ggml_tensor* rows            = runner.pack_audio(s.ctx, ax);
            std::vector<float> ggml_pack = compute_flat(s.ctx, rows);

            ggml_tensor* rows_in = arange_tensor(s.ctx, c.c, c.ch * c.t, 1, 1);
            std::memcpy(rows_in->data, ggml_pack.data(), ggml_pack.size() * sizeof(float));
            ggml_tensor* back              = runner.unpack_audio(s.ctx, rows_in, c.ch);
            std::vector<float> ggml_unpack = compute_flat(s.ctx, back);

            // host-side twin reads the same request-layout buffer
            sd::Tensor<float> req({c.t, c.ch, c.c}, arange_vec(c.c * c.ch * c.t));
            sd::Tensor<float> host_pack = minimax_h3_pack_audio_cond_rows(req);

            // VAE layout {T, C, S} -> request layout {T, S, C}
            sd::Tensor<float> vae({c.t, c.c, c.ch}, arange_vec(c.c * c.ch * c.t));
            sd::Tensor<float> swapped = minimax_h3_swap_audio_axes(vae);

            check_equal((std::string("audio/") + c.name + " host-vs-ggml pack").c_str(),
                        host_pack.values(),
                        ggml_pack);
            check_equal((std::string("audio/") + c.name + " roundtrip").c_str(),
                        ggml_unpack,
                        arange_vec(c.c * c.ch * c.t));
            // the swap is its own inverse
            check_equal((std::string("audio/") + c.name + " swap involution").c_str(),
                        minimax_h3_swap_audio_axes(swapped).values(),
                        vae.values());

            // Full transport: merge_av flat-appends the request-layout audio into the trailing
            // channels of the video latent, and the host helper peels it back out.  Both are the
            // shipping implementations, so this closes the loop the sampler actually walks.
            // (minimax_h3_unpack_audio_latent hardcodes stereo, so mono is out of scope for it.)
            if (c.ch == 2) {
                const int64_t vw = 4, vh = 4, vt = 2, vc = 24;
                ggml_tensor* vx                = arange_tensor(s.ctx, vw, vh, vt, vc);
                ggml_tensor* merged            = runner.merge_av(s.ctx, vx, ax);
                std::vector<float> packed_flat = compute_flat(s.ctx, merged);
                sd::Tensor<float> packed({merged->ne[0], merged->ne[1], merged->ne[2], merged->ne[3]},
                                         packed_flat);
                sd::Tensor<float> peeled =
                    minimax_h3_unpack_audio_latent(packed, static_cast<int>(c.t), static_cast<int>(vc), static_cast<int>(c.c));
                check_equal((std::string("audio/") + c.name + " merge_av transport").c_str(),
                            peeled.values(),
                            arange_vec(c.c * c.ch * c.t));
                if (peeled.shape() != std::vector<int64_t>({c.t, 2, c.c, 1})) {
                    std::fprintf(stderr, "FAIL audio/%s merge_av transport: wrong shape out\n", c.name);
                    ++g_failures;
                }
            }

            std::printf("{\"name\":\"%s\",", c.name);
            emit_array("pack", ggml_pack);
            emit_array("pack_host", host_pack.values());
            emit_array("unpack", ggml_unpack);
            emit_array("vae_to_request", swapped.values(), true);
            std::printf("}");
        }
        std::printf("],");
    }

    // ------------------------------------------------------------------------------------
    // 3. the video VAE decoder's 3-pass unpatchify permute chain
    // ------------------------------------------------------------------------------------

    void run_vae() {
        std::printf("\"vae_unpatchify\":[");
        bool first = true;
        for (const VaeCase& c : VAE_CASES) {
            if (!first) {
                std::printf(",");
            }
            first = false;
            Scratch s(512ull * 1024 * 1024);

            MiniMaxH3Video::MiniMaxH3VAEConfig config;
            config.out_channels                = c.c;
            config.spatial_downsample_factors  = {static_cast<int>(c.ph)};
            config.temporal_downsample_factors = {static_cast<int>(c.pt)};
            config.decoder_num_layers          = 1;  // unpatchify() reads none of the blocks
            MiniMaxH3Video::ViT3DDecoder decoder(config);
            GGML_ASSERT(decoder.patch_size == c.ph && decoder.patch_size_t == c.pt);

            const int64_t cols      = c.c * c.pt * c.ph * c.ph;
            const int64_t rows      = c.t * c.h * c.w;
            ggml_tensor* h          = arange_tensor(s.ctx, cols, rows, 1, 1);
            ggml_tensor* out        = decoder.unpatchify(s.ctx, h, c.w, c.h, c.t);
            std::vector<float> flat = compute_flat(s.ctx, out);

            const int64_t want[4] = {c.w * c.ph, c.h * c.ph, c.t * c.pt, c.c};
            for (int i = 0; i < 4; ++i) {
                if (out->ne[i] != want[i]) {
                    std::fprintf(stderr, "FAIL vae/%s: ne[%d] = %lld, expected %lld\n",
                                 c.name, i, static_cast<long long>(out->ne[i]), static_cast<long long>(want[i]));
                    ++g_failures;
                }
            }

            std::printf("{\"name\":\"%s\",", c.name);
            emit_array("out", flat, true);
            std::printf("}");
        }
        std::printf("]");
    }

}  // namespace

int main() {
    // The runner owns pack_audio / unpack_audio / merge_av.  An empty tensor map gives the
    // default config, which is all these three read (they read nothing at all, in fact) -- no
    // weights, no CUDA, no compute buffer.
    ggml_backend_t backend = sd_backend_cpu_init();
    GGML_ASSERT(backend != nullptr);
    MiniMaxH3::MiniMaxH3Runner runner(backend);

    std::printf("{");
    run_patchify();
    run_audio(runner);
    run_vae();
    std::printf("}\n");

    if (g_failures != 0) {
        std::fprintf(stderr, "\nminimax_h3_permutation_test: %d INTERNAL CHECK(S) FAILED\n", g_failures);
        return 1;
    }
    std::fprintf(stderr,
                 "minimax_h3_permutation_test: internal checks PASS -- %lld elements compared "
                 "(host twin vs ggml op, round-trips, merge_av transport)\n",
                 static_cast<long long>(g_checked));
    return 0;
}
