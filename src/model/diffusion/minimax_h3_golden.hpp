#ifndef __SD_MODEL_DIFFUSION_MINIMAX_H3_GOLDEN_HPP__
#define __SD_MODEL_DIFFUSION_MINIMAX_H3_GOLDEN_HPP__

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "core/tensor.hpp"
#include "core/util.h"
#include "ggml.h"
#include "model/diffusion/minimax_h3_host_layout.hpp"

// ─────────────────────────────────────────────────────────────────────────────────────────────
// MiniMax-H3 GOLDEN PARITY HARNESS -- engine side.
//
// Every open H3 defect so far has been chased by comparing OUTPUTS: our clip against ComfyUI's.
// That can only ever say THAT we differ.  This header exists so the two implementations can be
// driven from a BYTE-IDENTICAL state and their IMMEDIATE per-step output compared, which says
// WHERE.
//
// It is also, deliberately, a regression fixture rather than a one-off: capture ComfyUI's
// intermediates once (tools/h3_golden_capture.sh), then re-run tools/h3_golden_compare.py after
// any quant / kernel / model change and find out in seconds whether we drifted.
//
// ★ THE FOUR THINGS THAT MUST BE FROZEN, and why each one matters
//   1. the initial noise    -- comfy draws it with torch's Philox from `seed`; we draw it with our
//                              own RNG.  Same seed, DIFFERENT numbers.  Nothing downstream is
//                              comparable until this is pinned.  MINIMAX_H3_GOLDEN_NOISE.
//   2. the text conditioning-- comfy runs Qwen3-VL-32B at nvfp4_awq, we run our own nvfp4 GGUF.
//                              The hidden states differ by percent, so the DiT sees a different
//                              `context` and EVERY block differs for a reason that has nothing to
//                              do with the DiT.  MINIMAX_H3_GOLDEN_CONTEXT.
//   3. the sigma schedule   -- measured "within 1%" of comfy's, which is not the same as equal.
//                              A 1% sigma difference is a 1% velocity difference for free.
//                              MINIMAX_H3_GOLDEN_SIGMAS.
//   4. x at every step      -- even with 1-3 frozen, step k>1 starts from OUR step k-1 output, so
//                              a divergence at step 1 contaminates every later step and the
//                              comparison stops being a forward-pass measurement.  Teacher-forcing
//                              x from the golden makes every step an INDEPENDENT forward-pass
//                              comparison.  MINIMAX_H3_GOLDEN_FORCE_X.
//
// Run WITHOUT forcing first: if step 1's velocity matches and x1 does not, the defect is in the
// SAMPLER UPDATE, not in the forward pass.  Then run WITH forcing to localise the forward pass.
//
// LAYOUT NOTE (the thing that makes all of this cheap).  ComfyUI samples on
// `pack_latents([video, audio])` = [video C-order flat | audio C-order flat], and our packed
// latent is [W, H, T, 24+extra] whose flat buffer is video C-order [1,24,T,H,W] followed, at
// offset 24*W*H*T, by audio C-order [1,32,2,Ta].  The two are THE SAME BYTES in the same order.
// So a golden .npy pair drops straight in, and a dump drops straight out, with no permutation --
// which also means a layout bug cannot hide inside the harness itself.
//
// Everything here is a no-op unless its env var is set.  Refusals are HARD (return false /
// GGML_ABORT) on purpose: a silently-skipped injection would be reported as a passing parity run.
// ─────────────────────────────────────────────────────────────────────────────────────────────
namespace minimax_h3_golden {

    namespace detail {

        inline std::string env_or_empty(const char* name) {
            const char* v = getenv(name);
            return (v == nullptr || v[0] == '\0' || strcmp(v, "0") == 0) ? std::string() : std::string(v);
        }

        // Minimal numpy .npy v1/v2 reader, float32 / C-order only.  Anything else is refused
        // rather than reinterpreted -- a fortran-order or float64 golden read as C-order f32 is
        // exactly the class of silent corruption this harness exists to catch.
        inline bool read_npy_f32(const std::string& path,
                                 std::vector<float>& out,
                                 std::vector<int64_t>& shape_c_order) {
            std::ifstream f(path, std::ios::binary);
            if (!f.is_open()) {
                LOG_ERROR("[H3-GOLDEN] cannot open '%s'", path.c_str());
                return false;
            }
            unsigned char magic[10] = {0};
            f.read(reinterpret_cast<char*>(magic), 10);
            if (!f.good() || magic[0] != 0x93 || memcmp(magic + 1, "NUMPY", 5) != 0) {
                LOG_ERROR("[H3-GOLDEN] '%s' is not a .npy file", path.c_str());
                return false;
            }
            size_t header_len = 0;
            if (magic[6] == 1) {
                header_len = static_cast<size_t>(magic[8]) | (static_cast<size_t>(magic[9]) << 8);
            } else {
                unsigned char extra[2] = {0};
                f.read(reinterpret_cast<char*>(extra), 2);
                header_len = static_cast<size_t>(magic[8]) | (static_cast<size_t>(magic[9]) << 8) |
                             (static_cast<size_t>(extra[0]) << 16) | (static_cast<size_t>(extra[1]) << 24);
            }
            std::string header(header_len, '\0');
            f.read(header.data(), static_cast<std::streamsize>(header_len));
            if (!f.good()) {
                LOG_ERROR("[H3-GOLDEN] '%s' has a truncated header", path.c_str());
                return false;
            }
            if (header.find("'<f4'") == std::string::npos && header.find("\"<f4\"") == std::string::npos) {
                LOG_ERROR("[H3-GOLDEN] REFUSING '%s': not float32 little-endian. Header: %s",
                          path.c_str(), header.c_str());
                return false;
            }
            if (header.find("'fortran_order': True") != std::string::npos) {
                LOG_ERROR("[H3-GOLDEN] REFUSING '%s': fortran_order=True", path.c_str());
                return false;
            }
            const size_t sp = header.find("'shape'");
            const size_t lp = sp == std::string::npos ? std::string::npos : header.find('(', sp);
            const size_t rp = lp == std::string::npos ? std::string::npos : header.find(')', lp);
            if (rp == std::string::npos) {
                LOG_ERROR("[H3-GOLDEN] '%s' has an unreadable shape field", path.c_str());
                return false;
            }
            shape_c_order.clear();
            int64_t count = 1;
            {
                const std::string dims = header.substr(lp + 1, rp - lp - 1);
                std::string tok;
                for (size_t i = 0; i <= dims.size(); i++) {
                    const char c = i < dims.size() ? dims[i] : ',';
                    if (c == ',') {
                        if (!tok.empty()) {
                            shape_c_order.push_back(std::strtoll(tok.c_str(), nullptr, 10));
                            count *= shape_c_order.back();
                        }
                        tok.clear();
                    } else if (std::isdigit(static_cast<unsigned char>(c))) {
                        tok += c;
                    }
                }
            }
            if (count <= 0) {
                LOG_ERROR("[H3-GOLDEN] '%s' declares an empty array", path.c_str());
                return false;
            }
            out.resize(static_cast<size_t>(count));
            f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count) * 4);
            if (!f.good() && f.gcount() != static_cast<std::streamsize>(count) * 4) {
                LOG_ERROR("[H3-GOLDEN] '%s' payload is short (wanted %lld floats)", path.c_str(),
                          static_cast<long long>(count));
                return false;
            }
            return true;
        }

        // `shape` is in GGML order (ne0 fastest); the numpy header is written reversed, so a
        // {W,H,T,C,1} tensor lands as (1,C,T,H,W) -- exactly comfy's latent shape.
        inline bool write_npy_f32(const std::string& path,
                                  const float* data,
                                  const std::vector<int64_t>& shape) {
            std::ofstream out(path, std::ios::binary);
            if (!out.is_open()) {
                LOG_ERROR("[H3-GOLDEN] cannot write '%s'", path.c_str());
                return false;
            }
            std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (";
            int64_t count    = 1;
            for (size_t i = shape.size(); i-- > 0;) {
                dict += std::to_string(shape[i]);
                dict += ", ";
                count *= shape[i];
            }
            dict += "), }";
            size_t header_len = dict.size() + 1;
            while ((10 + header_len) % 64 != 0) {
                dict += ' ';
                header_len = dict.size() + 1;
            }
            dict += '\n';
            const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
            const uint16_t len16         = static_cast<uint16_t>(dict.size());
            out.write(reinterpret_cast<const char*>(magic), 8);
            out.write(reinterpret_cast<const char*>(&len16), 2);
            out.write(dict.data(), static_cast<std::streamsize>(dict.size()));
            out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count) * 4);
            return out.good();
        }

        // Geometry of the packed latent, derived once and asserted, so every function below reads
        // the same numbers rather than each recomputing them slightly differently.
        struct Packed {
            int64_t W = 0, H = 0, T = 0, C = 0;
            int64_t spatial      = 0;  // W*H*T
            int64_t video_values = 0;  // 24 * spatial
            int64_t audio_values = 0;  // 32 * 2 * audio_t
            int64_t video_ch     = 0;
            int64_t audio_t      = 0;
            bool ok              = false;
        };

        inline Packed geometry(const sd::Tensor<float>& packed, int audio_t, int video_channels) {
            Packed p;
            if (packed.empty() || packed.dim() < 4) {
                return p;
            }
            p.W            = packed.shape()[0];
            p.H            = packed.shape()[1];
            p.T            = packed.shape()[2];
            p.C            = packed.shape()[3];
            p.video_ch     = video_channels;
            p.audio_t      = audio_t;
            p.spatial      = p.W * p.H * p.T;
            p.video_values = static_cast<int64_t>(video_channels) * p.spatial;
            p.audio_values = static_cast<int64_t>(audio_t) * 2 * 32;
            p.ok           = p.C >= video_channels && packed.numel() >= p.video_values + p.audio_values;
            return p;
        }

    }  // namespace detail

    // ── which steps get FULL arrays dumped ────────────────────────────────────────────────────
    //
    // MINIMAX_H3_GOLDEN_STEPS="1,10,20", or "all"/unset for every step.  Full arrays are ~4.6 MB
    // of video per tensor at 1344x768/39f and there are three per step, so 20 steps is ~280 MB per
    // run -- fine for one investigation, not fine for a fixture you re-run weekly.  Scalars are
    // still written for every step by the existing MINIMAX_H3_DIAG bundle.
    //
    // NON-NUMERIC TOKENS ARE IGNORED, not parsed as 0.  "all" through atoi() is 0, which would
    // select a step that does not exist and dump NOTHING -- a silent empty run that looks exactly
    // like a working one until the comparison says "no step has both arrays".
    inline bool step_selected(int step) {
        static bool parsed = false;
        static std::set<int> wanted;
        if (!parsed) {
            parsed              = true;
            const std::string s = detail::env_or_empty("MINIMAX_H3_GOLDEN_STEPS");
            std::string tok;
            auto flush = [&]() {
                if (!tok.empty() &&
                    tok.find_first_not_of("0123456789") == std::string::npos) {
                    wanted.insert(std::atoi(tok.c_str()));
                }
                tok.clear();
            };
            for (size_t i = 0; i <= s.size(); i++) {
                const char c = i < s.size() ? s[i] : ',';
                if (c == ',') {
                    flush();
                } else if (!std::isspace(static_cast<unsigned char>(c))) {
                    tok += c;
                }
            }
        }
        return wanted.empty() || wanted.count(std::abs(step)) > 0;
    }

    // ── 1. initial noise ──────────────────────────────────────────────────────────────────────
    //
    // MINIMAX_H3_GOLDEN_NOISE=<dir-or-prefix>
    //   reads <prefix>.noise.video.npy  (1, 24, T, H, W)
    //     and <prefix>.noise.audio.npy  (1, 32,  2, Ta)
    // and overwrites the corresponding regions of the freshly drawn noise tensor.  The padding
    // tail of the last packed channel (the slack past 32*2*Ta) is left as drawn: the DiT never
    // reads it, and zeroing it would make a self-vs-self diff of the packed tensor lie.
    //
    // Element counts must match EXACTLY.  A golden captured at a different resolution or frame
    // count is refused, never resized -- silently resampling it would produce a plausible,
    // meaningless parity number.
    inline bool inject_initial_noise(sd::Tensor<float>& noise, int audio_t, int video_channels) {
        const std::string prefix = detail::env_or_empty("MINIMAX_H3_GOLDEN_NOISE");
        if (prefix.empty()) {
            return true;
        }
        const detail::Packed p = detail::geometry(noise, audio_t, video_channels);
        if (!p.ok) {
            LOG_ERROR("[H3-GOLDEN] MINIMAX_H3_GOLDEN_NOISE set but this render's latent is not a "
                      "packed MiniMax-H3 AV tensor");
            return false;
        }

        std::vector<float> v, a;
        std::vector<int64_t> vs, as;
        if (!detail::read_npy_f32(prefix + ".noise.video.npy", v, vs)) {
            return false;
        }
        if (p.audio_values > 0 && !detail::read_npy_f32(prefix + ".noise.audio.npy", a, as)) {
            return false;
        }
        if (static_cast<int64_t>(v.size()) != p.video_values) {
            LOG_ERROR("[H3-GOLDEN] REFUSING %s.noise.video.npy: %zu floats, this render wants %lld "
                      "(%lldx%lldx%lld x %lld ch). Resolution/frames must match the capture.",
                      prefix.c_str(), v.size(), (long long)p.video_values,
                      (long long)p.W, (long long)p.H, (long long)p.T, (long long)p.video_ch);
            return false;
        }
        if (static_cast<int64_t>(a.size()) != p.audio_values) {
            LOG_ERROR("[H3-GOLDEN] REFUSING %s.noise.audio.npy: %zu floats, this render wants %lld "
                      "(audio_t=%d). Frame count must match the capture.",
                      prefix.c_str(), a.size(), (long long)p.audio_values, audio_t);
            return false;
        }
        std::copy_n(v.data(), v.size(), noise.data());
        if (!a.empty()) {
            std::copy_n(a.data(), a.size(), noise.data() + p.video_values);
        }
        LOG_INFO("[H3-GOLDEN] initial noise REPLACED from %s.noise.{video,audio}.npy "
                 "(%lld video + %lld audio floats). Our RNG is out of the picture; both engines "
                 "now start from the same numbers.",
                 prefix.c_str(), (long long)v.size(), (long long)a.size());
        return true;
    }

    // ── 2. text conditioning ──────────────────────────────────────────────────────────────────
    //
    // MINIMAX_H3_GOLDEN_CONTEXT=<file.npy>, shape (1, L, D).
    //   D == 5376 (hidden_size)  -> comfy's PRE-REFINED embeds.  MiniMaxH3Model::forward_text
    //                               detects the width and skips condition_proj + token_refiner,
    //                               exactly as upstream hoists them out of the step loop.  This
    //                               removes the text encoder AND the refiner from the comparison.
    //   D == 5120 (text_dim)     -> comfy's RAW Qwen3-VL layer-50 states.  Keeps our refiner in
    //                               the comparison, which is what you want once the DiT is clean.
    //
    // ⚠️ The token COUNT must match what our tokenizer produced, because c_token_types /
    // text_token_tags -- which decide the AdaLN modality row of every text row -- are still ours.
    // A mismatch is refused; it is itself a finding, and one that invalidates everything after it.
    inline bool inject_context(sd::Tensor<float>& c_crossattn) {
        const std::string path = detail::env_or_empty("MINIMAX_H3_GOLDEN_CONTEXT");
        if (path.empty()) {
            return true;
        }
        std::vector<float> data;
        std::vector<int64_t> shape;
        if (!detail::read_npy_f32(path, data, shape)) {
            return false;
        }
        if (shape.size() != 3 || shape[0] != 1) {
            LOG_ERROR("[H3-GOLDEN] REFUSING %s: expected (1, L, D), got %zu dims", path.c_str(), shape.size());
            return false;
        }
        const int64_t L = shape[1];
        const int64_t D = shape[2];
        if (!c_crossattn.empty() && c_crossattn.shape()[1] != L) {
            LOG_ERROR("[H3-GOLDEN] REFUSING %s: it holds %lld tokens, our tokenizer produced %lld. "
                      "The modality tags and c_token_types are still OURS and are indexed by row, "
                      "so a different token count would silently mis-tag the text span. Fix the "
                      "tokenizer parity first -- this mismatch IS the finding.",
                      path.c_str(), (long long)L, (long long)c_crossattn.shape()[1]);
            return false;
        }
        // {D, L, 1} in GGML order == (1, L, D) in numpy order, which is the layout the DiT reads.
        sd::Tensor<float> replacement({D, L, 1});
        std::copy_n(data.data(), data.size(), replacement.data());
        c_crossattn = std::move(replacement);
        LOG_INFO("[H3-GOLDEN] c_crossattn REPLACED from %s (%lld tokens x %lld dims) -- %s",
                 path.c_str(), (long long)L, (long long)D,
                 D == 5376 ? "pre-refined embeds: the text encoder AND the token refiner are now "
                             "out of the comparison"
                           : "raw text-encoder states: our refiner still runs");
        return true;
    }

    // MINIMAX_H3_GOLDEN_DUMP_CONTEXT=<file.npy> writes the conditioner's raw Qwen hidden states
    // before an optional golden substitution.  This is the missing control for whole-pipeline
    // quality comparisons: a DiT parity run can pass perfectly with injected Comfy conditioning
    // while the production text encoder still hands that same DiT a materially different prompt.
    inline bool dump_context(const sd::Tensor<float>& c_crossattn) {
        const std::string path = detail::env_or_empty("MINIMAX_H3_GOLDEN_DUMP_CONTEXT");
        if (path.empty()) {
            return true;
        }
        if (c_crossattn.empty() || c_crossattn.dim() < 2) {
            LOG_ERROR("[H3-GOLDEN] cannot dump raw context to %s: expected a non-empty context tensor",
                      path.c_str());
            return false;
        }
        // Runtime tensors elide the singleton batch dimension and are {D,L}; spell it back into
        // the file so this is directly comparable to Comfy's (1,L,D) capture.
        const std::vector<int64_t> shape = {c_crossattn.shape()[0], c_crossattn.shape()[1], 1};
        if (!detail::write_npy_f32(path, c_crossattn.data(), shape)) {
            return false;
        }
        LOG_INFO("[H3-GOLDEN] raw conditioner context -> %s (%lld tokens x %lld dims)",
                 path.c_str(),
                 static_cast<long long>(c_crossattn.shape()[1]),
                 static_cast<long long>(c_crossattn.shape()[0]));
        return true;
    }

    // ── 3. sigma schedule ─────────────────────────────────────────────────────────────────────
    //
    // MINIMAX_H3_GOLDEN_SIGMAS=<file.txt> -- one float per line, INCLUDING the trailing 0.
    // Replaces the whole schedule.  "Within 1% of comfy at every step" is a 1% velocity
    // difference handed to the comparison for free; this makes it zero.
    inline bool override_sigmas(std::vector<float>& sigmas) {
        const std::string path = detail::env_or_empty("MINIMAX_H3_GOLDEN_SIGMAS");
        if (path.empty()) {
            return true;
        }
        std::ifstream f(path);
        if (!f.is_open()) {
            LOG_ERROR("[H3-GOLDEN] cannot open %s", path.c_str());
            return false;
        }
        std::vector<float> loaded;
        std::string line;
        while (std::getline(f, line)) {
            if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
                continue;
            }
            loaded.push_back(static_cast<float>(std::atof(line.c_str())));
        }
        if (loaded.size() < 2) {
            LOG_ERROR("[H3-GOLDEN] REFUSING %s: %zu sigmas is not a schedule", path.c_str(), loaded.size());
            return false;
        }
        LOG_INFO("[H3-GOLDEN] sigma schedule REPLACED from %s: %zu entries (was %zu), "
                 "sigma[0]=%.6f sigma[-1]=%.6f",
                 path.c_str(), loaded.size(), sigmas.size(), loaded.front(), loaded.back());
        sigmas = std::move(loaded);
        return true;
    }

    // ── 4. per-step teacher forcing ───────────────────────────────────────────────────────────
    //
    // MINIMAX_H3_GOLDEN_FORCE_X=<prefix>, reading <prefix>.s<k>.x.{video,audio}.npy.
    //
    // ★ This is what turns a trajectory comparison into a FORWARD-PASS comparison.  Without it,
    // step k starts from our own step k-1 output, so any step-1 divergence is amplified through
    // the run and "we differ at step 7" carries no information about step 7.  With it, every step
    // is an independent evaluation of the same function at the same point.
    //
    // A missing file for a given step is NOT an error -- the capture is allowed to be sparse
    // (step 1 / middle / last).  It logs, and that step simply runs unforced; the comparison tool
    // reads the same list and marks unforced steps as non-independent.
    inline void force_x(sd::Tensor<float>& x, int step, int audio_t, int video_channels) {
        const std::string prefix = detail::env_or_empty("MINIMAX_H3_GOLDEN_FORCE_X");
        if (prefix.empty()) {
            return;
        }
        const detail::Packed p = detail::geometry(x, audio_t, video_channels);
        if (!p.ok) {
            return;
        }
        const std::string base = prefix + ".s" + std::to_string(std::abs(step)) + ".x";
        std::vector<float> v, a;
        std::vector<int64_t> vs, as;
        {
            std::ifstream probe(base + ".video.npy", std::ios::binary);
            if (!probe.good()) {
                LOG_INFO("[H3-GOLDEN] step %d not forced (no %s.video.npy) -- this step's comparison is "
                         "NOT independent of the previous one",
                         step, base.c_str());
                return;
            }
        }
        if (!detail::read_npy_f32(base + ".video.npy", v, vs)) {
            GGML_ABORT("[H3-GOLDEN] %s.video.npy exists but could not be read", base.c_str());
        }
        if (static_cast<int64_t>(v.size()) != p.video_values) {
            GGML_ABORT("[H3-GOLDEN] %s.video.npy holds %zu floats, this render wants %lld. A forced "
                       "x of the wrong shape would silently compare two different problems.",
                       base.c_str(), v.size(), (long long)p.video_values);
        }
        std::copy_n(v.data(), v.size(), x.data());
        if (p.audio_values > 0 && detail::read_npy_f32(base + ".audio.npy", a, as)) {
            if (static_cast<int64_t>(a.size()) != p.audio_values) {
                GGML_ABORT("[H3-GOLDEN] %s.audio.npy holds %zu floats, this render wants %lld",
                           base.c_str(), a.size(), (long long)p.audio_values);
            }
            std::copy_n(a.data(), a.size(), x.data() + p.video_values);
        }
        LOG_INFO("[H3-GOLDEN] step %d: x FORCED from %s.{video,audio}.npy -- this step is now an "
                 "independent forward-pass comparison",
                 step, base.c_str());
    }

    // Replace the finished packed latent immediately before the VAEs. The capture node writes
    // ComfyUI's final pair as
    //   <prefix>0.npy  video (1, 24, T, H, W)
    //   <prefix>1.npy  audio (1, 32, 2, Ta)
    // and those payloads are byte-for-byte the two contiguous halves of our packed tensor.
    // This isolates each decoder from every DiT, sampler and quantisation difference.
    inline bool force_final(sd::Tensor<float>& packed, int audio_t, int video_channels) {
        const std::string prefix = detail::env_or_empty("MINIMAX_H3_GOLDEN_FINAL");
        if (prefix.empty()) {
            return true;
        }
        const detail::Packed p = detail::geometry(packed, audio_t, video_channels);
        if (!p.ok) {
            return false;
        }
        std::vector<float> video;
        std::vector<float> audio;
        std::vector<int64_t> video_shape;
        std::vector<int64_t> audio_shape;
        if (!detail::read_npy_f32(prefix + "0.npy", video, video_shape) ||
            !detail::read_npy_f32(prefix + "1.npy", audio, audio_shape)) {
            return false;
        }
        if (static_cast<int64_t>(video.size()) != p.video_values ||
            static_cast<int64_t>(audio.size()) != p.audio_values) {
            LOG_ERROR("[H3-GOLDEN] REFUSING final latent %s{0,1}.npy: got %zu video + %zu audio "
                      "floats, this render wants %lld + %lld",
                      prefix.c_str(), video.size(), audio.size(),
                      (long long)p.video_values, (long long)p.audio_values);
            return false;
        }
        std::copy_n(video.data(), video.size(), packed.data());
        std::copy_n(audio.data(), audio.size(), packed.data() + p.video_values);
        LOG_INFO("[H3-GOLDEN] final latent FORCED from %s{0,1}.npy -- both VAEs now decode "
                 "ComfyUI's exact output",
                 prefix.c_str());
        return true;
    }

    // ── 5. dumping ────────────────────────────────────────────────────────────────────────────
    //
    // MINIMAX_H3_GOLDEN_DUMP=<prefix> writes, per selected step,
    //   <prefix>.s<k>.x.{video,audio}.npy
    //   <prefix>.s<k>.velocity.{video,audio}.npy
    //   <prefix>.s<k>.x0.{video,audio}.npy
    // in comfy's own array shapes, so the comparison tool never has to know which side produced a
    // file.  The existing MINIMAX_H3_DIAG_NPY writes the AUDIO half only; this is the video
    // control it was missing, which is the half that decides "audio-specific or global".
    inline void dump_packed(const char* what,
                            const sd::Tensor<float>& packed,
                            int step,
                            int audio_t,
                            int video_channels) {
        const std::string prefix = detail::env_or_empty("MINIMAX_H3_GOLDEN_DUMP");
        if (prefix.empty() || !step_selected(step)) {
            return;
        }
        const detail::Packed p = detail::geometry(packed, audio_t, video_channels);
        if (!p.ok) {
            return;
        }
        const std::string base = prefix + ".s" + std::to_string(std::abs(step)) + "." + what;
        detail::write_npy_f32(base + ".video.npy", packed.data(), {p.W, p.H, p.T, p.video_ch, 1});
        if (p.audio_values > 0) {
            detail::write_npy_f32(base + ".audio.npy", packed.data() + p.video_values,
                                  {p.audio_t, 2, 32, 1});
        }
    }

    // The final sampled latent, under the same naming (step index "final"), so a golden run and a
    // production run can be diffed by the same tool with no special case.
    inline void dump_final(const sd::Tensor<float>& packed, int audio_t, int video_channels) {
        const std::string prefix = detail::env_or_empty("MINIMAX_H3_GOLDEN_DUMP");
        if (prefix.empty()) {
            return;
        }
        const detail::Packed p = detail::geometry(packed, audio_t, video_channels);
        if (!p.ok) {
            return;
        }
        const std::string base = prefix + ".final";
        detail::write_npy_f32(base + ".video.npy", packed.data(), {p.W, p.H, p.T, p.video_ch, 1});
        if (p.audio_values > 0) {
            detail::write_npy_f32(base + ".audio.npy", packed.data() + p.video_values,
                                  {p.audio_t, 2, 32, 1});
        }
        LOG_INFO("[H3-GOLDEN] final latent written to %s.{video,audio}.npy", base.c_str());
    }

    // Emitted once per render so a dump directory is self-describing: the compare tool refuses to
    // score a pair whose geometry lines disagree.
    inline void log_geometry(const sd::Tensor<float>& packed, int audio_t, int video_channels) {
        if (detail::env_or_empty("MINIMAX_H3_GOLDEN_DUMP").empty()) {
            return;
        }
        const detail::Packed p = detail::geometry(packed, audio_t, video_channels);
        LOG_INFO("[H3-GOLDEN] geometry W=%lld H=%lld T=%lld packed_C=%lld video_C=%lld audio_t=%lld "
                 "video_values=%lld audio_values=%lld",
                 (long long)p.W, (long long)p.H, (long long)p.T, (long long)p.C,
                 (long long)p.video_ch, (long long)p.audio_t,
                 (long long)p.video_values, (long long)p.audio_values);
    }

}  // namespace minimax_h3_golden

#endif  // __SD_MODEL_DIFFUSION_MINIMAX_H3_GOLDEN_HPP__
