// NAVA DiT harness — two modes.
//
//   nava <gguf> [<input_dir>] [<out_dir>]
//       Phase-1 single-forward harness (numeric-diff reference). Loads
//       models/nava-dit-f16.gguf, runs ONE forward of the dual-stream
//       audio-video MMDiT, writes the patched video velocity (+ auto-logged
//       per-block stats). Numeric correctness is validated separately against
//       a PyTorch bf16 reference. (The DiT is VERIFIED — see HANDOFF-nava.md
//       2026-06-03e: all 30 blocks 62-78 dB.)
//
//   nava render --prompt "..." [opts]
//       Phase-2 SILENT text-to-video render. prompt-context -> seeded noise ->
//       Euler flow-match sampling w/ cond+uncond CFG (calling the verified DiT)
//       -> Wan2.2 48ch VAE decode -> frames -> webm. Writes a run dir
//       (clip.webm + meta.json) for the eye-test viewer tools/nava_eyetest_server.py.
//       See docs/nava-phase2-sampler.md.
//       Render opts:
//         --prompt "..."        (currently informational; context comes from --context or dummy)
//         --context <file.bin>  raw umT5 [4096,512] OR post-embed [3072,512] cond context.
//                               (REQUIRED for a real render — dump from NAVA's text_model;
//                                omitted => deterministic DUMMY context, garbage-but-valid output.)
//         --neg-context <file>  uncond context (default: zeros_like(context) — NAVA's MVP path).
//         --steps N             Euler steps (default 10; use 2 for the smoke).
//         --frames N            latent temporal frames (default 13). pixel frames = (N-1)*4+1.
//         --width W --height H  PIXEL resolution (default 832x480). latent = /16; DiT grid = /32.
//         --seed S              noise seed (default 42).
//         --cfg X               video guidance scale (default 3.0).
//         --shift X             flow-match shift (default 5.0).
//         --fps N               webm fps (default 24).
//         --out-name NAME       run dir name under RUNS_DIR (default a timestamp).
//         --runs-dir DIR        base output dir (default /mnt/hdd/nava/cpp-runs).
//         --vae <gguf>          Wan2.2 48ch VAE gguf (default models/longcat-wan-vae-f16.gguf).
//         --cuda                use the CUDA backend (default CPU; CPU is for the smoke only).
//         --label "..."         meta.json label.
//
// CPU concessions (NOT model truth; a CUDA service build runs natively):
//   * LONGCAT_NO_FUSED_ROPE=1 — the fused interleaved-RoPE op (ggml_rope_pe) is
//     CUDA-only; force the portable chain-RoPE path.
//   * modulation/head-mod params allocated F32 (CPU rejects f32(+)f16; the gguf
//     has them F16). Handled inside nava.hpp.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "ggml.h"
#include "model.h"
#include "stable-diffusion.h"
#include "tensor.hpp"
#include "tensor_ggml.hpp"
#include "util.h"

#include "common/media_io.h"  // load_image_from_file (stb_image decode)

#include "nava.hpp"
#include "rng_philox.hpp"
#include "unipc.hpp"
#include "vae.hpp"
#include "wan.hpp"
#include "ltx_audio_vae.h"
#include "t5.hpp"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#else
// CPU-only build shim: ggml_extend.hpp references ggml_backend_cuda_host_buffer_type
// unconditionally (lap-32 pinned offload params). On a pure-CPU build that pinned
// path is never taken (it requires a non-CPU runtime backend), but the symbol must
// still resolve at link time. This stub returns null so the alloc falls back to the
// pageable path. (Only present when GGML_USE_CUDA is undefined; never built into the
// supported CUDA service binary.)
extern "C" ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void) {
    return nullptr;
}
#endif

static void log_cb(enum sd_log_level_t level, const char* text, void* /*data*/) {
    fputs(text, level == SD_LOG_ERROR ? stderr : stdout);
}

static bool path_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

static bool ends_with_ci(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    for (size_t i = 0; i < suf.size(); ++i)
        if (std::tolower((unsigned char)s[s.size() - suf.size() + i]) != suf[i]) return false;
    return true;
}

// True for real encoded-image inputs (vs a preprocessed raw RGB .bin / latent .bin).
static bool looks_like_image_file(const std::string& p) {
    return ends_with_ci(p, ".png") || ends_with_ci(p, ".jpg") || ends_with_ci(p, ".jpeg")
        || ends_with_ci(p, ".bmp") || ends_with_ci(p, ".webp") || ends_with_ci(p, ".tga");
}

// PIL-compatible separable Lanczos-3 resize for interleaved RGB8. Python NAVA resizes
// with PIL Image.LANCZOS; stb_image_resize has no Lanczos and the closest cubic
// (CATMULLROM) measurably diverges through the chaotic DiT (darker audio). This mirrors
// PIL's ImagingResampleInner: precompute normalized coeffs per output pixel
// (support = 3*max(scale,1)), horizontal pass -> uint8 temp, then vertical pass -> uint8.
static double nava_sinc(double x) {
    if (x == 0.0) return 1.0;
    x *= M_PI;
    return std::sin(x) / x;
}
static double nava_lanczos(double x) {
    return (-3.0 < x && x < 3.0) ? nava_sinc(x) * nava_sinc(x / 3.0) : 0.0;
}
// One-dimension coefficient table (PIL precompute_coeffs).
static void nava_lanczos_coeffs(int in_size, int out_size,
                                std::vector<int>& bounds,       // 2*out_size: [xmin,k]
                                std::vector<double>& kk,        // out_size*ksize
                                int& ksize) {
    double scale       = (double)in_size / (double)out_size;
    double filterscale = scale < 1.0 ? 1.0 : scale;
    double support     = 3.0 * filterscale;  // Lanczos a=3
    ksize = (int)std::ceil(support) * 2 + 1;
    bounds.assign((size_t)out_size * 2, 0);
    kk.assign((size_t)out_size * ksize, 0.0);
    for (int xx = 0; xx < out_size; ++xx) {
        double center = (xx + 0.5) * scale;
        double ww = 0.0, ss = 1.0 / filterscale;
        int xmin = (int)(center - support + 0.5); if (xmin < 0) xmin = 0;
        int xmax = (int)(center + support + 0.5); if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double* k = &kk[(size_t)xx * ksize];
        int x = 0;
        for (; x < xmax; ++x) { double w = nava_lanczos((x + xmin - center + 0.5) * ss); k[x] = w; ww += w; }
        for (x = 0; x < xmax; ++x) if (ww != 0.0) k[x] /= ww;
        bounds[xx * 2 + 0] = xmin;
        bounds[xx * 2 + 1] = xmax;
    }
}
static inline uint8_t nava_clip8(double v) {
    if (v <= 0.0) return 0;
    if (v >= 255.0) return 255;
    return (uint8_t)(v + 0.5);
}
static void nava_lanczos_resize_rgb8(const uint8_t* src, int sw, int sh,
                                     std::vector<uint8_t>& dst, int dw, int dh) {
    std::vector<int> hb, vb; std::vector<double> hk, vk; int hks = 0, vks = 0;
    nava_lanczos_coeffs(sw, dw, hb, hk, hks);
    nava_lanczos_coeffs(sh, dh, vb, vk, vks);
    // Horizontal pass: src [sh,sw,3] -> temp [sh,dw,3].
    std::vector<uint8_t> temp((size_t)sh * dw * 3);
    for (int y = 0; y < sh; ++y)
        for (int xx = 0; xx < dw; ++xx) {
            int xmin = hb[xx * 2], xmax = hb[xx * 2 + 1];
            const double* k = &hk[(size_t)xx * hks];
            for (int c = 0; c < 3; ++c) {
                double acc = 0.0;
                for (int i = 0; i < xmax; ++i) acc += src[((size_t)y * sw + (xmin + i)) * 3 + c] * k[i];
                temp[((size_t)y * dw + xx) * 3 + c] = nava_clip8(acc);
            }
        }
    // Vertical pass: temp [sh,dw,3] -> dst [dh,dw,3].
    dst.assign((size_t)dh * dw * 3, 0);
    for (int yy = 0; yy < dh; ++yy) {
        int ymin = vb[yy * 2], ymax = vb[yy * 2 + 1];
        const double* k = &vk[(size_t)yy * vks];
        for (int xx = 0; xx < dw; ++xx)
            for (int c = 0; c < 3; ++c) {
                double acc = 0.0;
                for (int i = 0; i < ymax; ++i) acc += temp[((size_t)(ymin + i) * dw + xx) * 3 + c] * k[i];
                dst[((size_t)yy * dw + xx) * 3 + c] = nava_clip8(acc);
            }
    }
}

// Load an encoded image and reproduce Python NAVA's I2V first-frame preprocessing
// (nava_src/vae/local_video_vae.py:_resize_center_crop): aspect-PRESERVING resize so
// the image fills the (W,H) target, then a center crop, then x/255 -> [0,1]. The
// result is a ggml ne [W,H,1,3] tensor ready for the WAN VAE encoder — identical in
// layout to tools/nava_prep_image.py, but handled in-port so callers pass a real
// image instead of a pre-baked bin. Resampling is PIL-LANCZOS-parity
// (nava_lanczos_resize_rgb8) to match Python bit-for-bit-ish; framing/FOV is exact.
static bool load_image_resize_center_crop(const std::string& path, int W, int H,
                                          sd::Tensor<float>& out) {
    int sw = 0, sh = 0;
    uint8_t* src = load_image_from_file(path.c_str(), sw, sh, /*expected_w=*/0, /*expected_h=*/0, 3);
    if (!src) { printf("ERROR: failed to decode image %s\n", path.c_str()); return false; }
    // Python: scale = max(W/sw, H/sh); resize to (round(sw*scale), round(sh*scale)).
    double scale = std::max((double)W / (double)sw, (double)H / (double)sh);
    int rw = (int)std::lround(sw * scale);
    int rh = (int)std::lround(sh * scale);
    if (rw < W) rw = W;
    if (rh < H) rh = H;
    std::vector<uint8_t> resized;
    nava_lanczos_resize_rgb8(src, sw, sh, resized, rw, rh);  // PIL Image.LANCZOS parity
    free(src);
    // Center crop (rw,rh) -> (W,H).
    int left = (rw - W) / 2, top = (rh - H) / 2;
    // ggml ne [W,H,1,3]: c slowest, then y, x fastest (== prep script's [C,1,H,W] C-order).
    std::vector<float> data((size_t)3 * H * W);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                data[((size_t)c * H + y) * W + x] =
                    resized[(((size_t)(top + y) * rw) + (left + x)) * 3 + c] / 255.0f;
    out = sd::Tensor<float>({(int64_t)W, (int64_t)H, 1, 3}, std::move(data));
    printf("I2V: decoded %s %dx%d -> resize %dx%d -> center-crop %dx%d (Python-parity)\n",
           path.c_str(), sw, sh, rw, rh, W, H);
    return true;
}

// Match python's HuggingfaceTokenizer text preprocessing (nava_src .../tokenizers.py):
// the umT5 sentencepiece normalizer applies NFKC, mapping full-width forms to ASCII
// (U+FF01..U+FF5E -> cp-0xFEE0, U+3000 -> space). The cpp t5 unigram tokenizer does NOT
// apply that charsmap, so full-width CJK punctuation ('，' U+FF0C, '：' U+FF1A, ...)
// mis-tokenizes (id 356/619 instead of 275/283), corrupting the whole umT5 context via
// bidirectional attention. Also apply whitespace_clean (collapse \s+ -> ' ', strip).
static std::string nava_normalize_text(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    auto emit = [&](uint32_t cp) {
        if (cp < 0x80) out.push_back((char)cp);
        else if (cp < 0x800) { out.push_back((char)(0xC0 | (cp >> 6))); out.push_back((char)(0x80 | (cp & 0x3F))); }
        else if (cp < 0x10000) { out.push_back((char)(0xE0 | (cp >> 12))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
        else { out.push_back((char)(0xF0 | (cp >> 18))); out.push_back((char)(0x80 | ((cp >> 12) & 0x3F))); out.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); out.push_back((char)(0x80 | (cp & 0x3F))); }
    };
    size_t i = 0, n = in.size();
    while (i < n) {
        unsigned char c = (unsigned char)in[i];
        uint32_t cp; int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6) { cp = c & 0x1F; len = 2; }
        else if ((c >> 4) == 0xE) { cp = c & 0x0F; len = 3; }
        else if ((c >> 3) == 0x1E) { cp = c & 0x07; len = 4; }
        else { cp = c; len = 1; }
        for (int k = 1; k < len && i + k < n; ++k) cp = (cp << 6) | ((unsigned char)in[i + k] & 0x3F);
        i += len;
        if (cp >= 0xFF01 && cp <= 0xFF5E) cp -= 0xFEE0;  // full-width forms -> ASCII (NFKC)
        else if (cp == 0x3000) cp = 0x20;                // ideographic space -> space
        emit(cp);
    }
    // whitespace_clean: collapse whitespace runs to a single space, then strip.
    std::string ws;
    ws.reserve(out.size());
    bool prev_sp = false;
    for (char ch : out) {
        bool sp = (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v');
        if (sp) { if (!prev_sp) ws.push_back(' '); prev_sp = true; }
        else { ws.push_back(ch); prev_sp = false; }
    }
    size_t a = ws.find_first_not_of(' '), b = ws.find_last_not_of(' ');
    return (a == std::string::npos) ? std::string() : ws.substr(a, b - a + 1);
}

static void mkdirs(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); ++i) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) {
            if (!cur.empty() && cur != "/") {
                mkdir(cur.c_str(), 0777);
            }
        }
    }
}

static void dump_stats(const char* tag, const sd::Tensor<float>& t) {
    if (t.empty()) {
        printf("%-26s EMPTY\n", tag);
        return;
    }
    double sum = 0.0, sq = 0.0, mn = 1e30, mx = -1e30;
    for (int64_t i = 0; i < t.numel(); ++i) {
        double v = t.data()[i];
        sum += v;
        sq += v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    double mean = sum / (double)t.numel();
    double var  = sq / (double)t.numel() - mean * mean;
    printf("%-26s shape=%s mean=%.5f std=%.5f min=%.4f max=%.4f\n",
           tag, sd::tensor_shape_to_string(t.shape()).c_str(), mean,
           var > 0 ? sqrt(var) : 0.0, mn, mx);
}

static void write_bin(const std::string& path, const sd::Tensor<float>& t, const std::string& name) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) { printf("cannot open %s\n", path.c_str()); return; }
    int32_t n_dims = (int32_t)t.dim();
    int32_t len    = (int32_t)name.size();
    int32_t ttype  = (int32_t)GGML_TYPE_F32;
    fwrite(&n_dims, sizeof(int32_t), 1, f);
    fwrite(&len, sizeof(int32_t), 1, f);
    fwrite(&ttype, sizeof(int32_t), 1, f);
    for (int i = 0; i < n_dims; ++i) {
        int32_t d = (int32_t)t.shape()[(size_t)i];
        fwrite(&d, sizeof(int32_t), 1, f);
    }
    fwrite(name.data(), 1, (size_t)len, f);
    fwrite(t.data(), sizeof(float), (size_t)t.numel(), f);
    fclose(f);
    printf("wrote %s\n", path.c_str());
}

// Minimal RIFF/WAVE reader for the `audio-encode` subcommand. Supports PCM16,
// PCM32, IEEE-float32, and WAVE_FORMAT_EXTENSIBLE wrapping those. Returns an
// sd::Tensor<float> ne [samples, channels] (ne0=samples fastest, ne1=channels)
// = the planar layout the LTX audio encoder expects. The model needs 16 kHz; we
// require it (ffmpeg -ar 16000 produces it) rather than resample in-binary.
static bool read_wav_16k(const std::string& path, sd::Tensor<float>& out, uint32_t& sample_rate) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { printf("audio-encode: cannot open wav %s\n", path.c_str()); return false; }
    auto rd_u32 = [&](uint32_t& v) { return fread(&v, 4, 1, f) == 1; };
    auto rd_u16 = [&](uint16_t& v) { return fread(&v, 2, 1, f) == 1; };
    char tag[4];
    uint32_t riff_size = 0;
    if (fread(tag, 1, 4, f) != 4 || memcmp(tag, "RIFF", 4) != 0) { fclose(f); printf("audio-encode: not RIFF\n"); return false; }
    if (!rd_u32(riff_size) || fread(tag, 1, 4, f) != 4 || memcmp(tag, "WAVE", 4) != 0) { fclose(f); printf("audio-encode: not WAVE\n"); return false; }

    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t srate = 0;
    bool have_fmt = false, have_data = false;
    std::vector<uint8_t> data;
    while (fread(tag, 1, 4, f) == 4) {
        uint32_t chunk_size = 0;
        if (!rd_u32(chunk_size)) break;
        if (memcmp(tag, "fmt ", 4) == 0) {
            uint16_t block_align = 0;
            uint32_t byte_rate = 0;
            rd_u16(fmt); rd_u16(channels); rd_u32(srate); rd_u32(byte_rate); rd_u16(block_align); rd_u16(bits);
            long consumed = 16;
            if (fmt == 0xFFFE && chunk_size >= 40) {  // EXTENSIBLE: read the real subformat tag
                uint16_t ext_size = 0, valid_bits = 0; uint32_t chmask = 0; uint16_t subfmt = 0;
                rd_u16(ext_size); rd_u16(valid_bits); rd_u32(chmask); rd_u16(subfmt);
                fmt = subfmt; consumed += 10;
            }
            if ((long)chunk_size > consumed) fseek(f, chunk_size - consumed, SEEK_CUR);
            have_fmt = true;
        } else if (memcmp(tag, "data", 4) == 0) {
            data.resize(chunk_size);
            if (fread(data.data(), 1, chunk_size, f) != chunk_size) { fclose(f); printf("audio-encode: short data\n"); return false; }
            have_data = true;
        } else {
            fseek(f, chunk_size + (chunk_size & 1), SEEK_CUR);  // skip (pad to even)
        }
    }
    fclose(f);
    if (!have_fmt || !have_data) { printf("audio-encode: missing fmt/data chunk\n"); return false; }
    sample_rate = srate;

    const int64_t bytes_per_sample = bits / 8;
    if (bytes_per_sample == 0 || channels == 0) { printf("audio-encode: bad fmt\n"); return false; }
    const int64_t total = (int64_t)data.size() / bytes_per_sample;
    const int64_t frames = total / channels;

    // de-interleave -> planar ne [frames, channels]
    out = sd::Tensor<float>({frames, (int64_t)channels});
    float* dst = out.data();
    const uint8_t* p = data.data();
    auto sample_at = [&](int64_t idx) -> float {
        const uint8_t* s = p + idx * bytes_per_sample;
        if (fmt == 3 && bits == 32) { float v; memcpy(&v, s, 4); return v; }                 // IEEE float
        if (fmt == 1 && bits == 16) { int16_t v; memcpy(&v, s, 2); return v / 32768.0f; }     // PCM16
        if (fmt == 1 && bits == 32) { int32_t v; memcpy(&v, s, 4); return v / 2147483648.0f; }// PCM32
        if (fmt == 1 && bits == 24) {                                                          // PCM24
            int32_t v = (s[0] | (s[1] << 8) | (s[2] << 16)); if (v & 0x800000) v |= ~0xFFFFFF;
            return v / 8388608.0f;
        }
        return 0.0f;
    };
    for (int64_t i = 0; i < frames; ++i)
        for (int64_t c = 0; c < channels; ++c)
            dst[c * frames + i] = sample_at(i * channels + c);  // ne1(c)-strided, ne0(i)-fast
    return true;
}

// deterministic dummy fill (small magnitudes, sin-based so it's reproducible)
static sd::Tensor<float> dummy(std::vector<int64_t> shape, float scale = 0.02f) {
    sd::Tensor<float> t(shape);
    for (int64_t i = 0; i < t.numel(); ++i) {
        t.data()[i] = scale * std::sin(0.1f * (float)(i % 997));
    }
    return t;
}

static sd::Tensor<float> load_or_dummy(const std::string& dir,
                                       const std::string& fname,
                                       std::vector<int64_t> dummy_shape) {
    if (!dir.empty()) {
        std::string p = dir + "/" + fname;
        if (path_exists(p)) {
            auto t = sd::load_tensor_from_file_as_tensor<float>(p);
            printf("loaded %s\n", p.c_str());
            return t;
        }
    }
    return dummy(std::move(dummy_shape));
}

// ---------------------------------------------------------------------------
// Phase-1 single-forward harness (unchanged behaviour).
// ---------------------------------------------------------------------------
static int run_single_forward(int argc, char** argv) {
    std::string gguf_path = argv[1];
    std::string in_dir, out_dir = ".";
    int positional = 0;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (positional == 0) {
            in_dir = a;
            positional++;
        } else if (positional == 1) {
            out_dir = a;
            positional++;
        }
    }

    setenv("LONGCAT_NO_FUSED_ROPE", "1", /*overwrite=*/0);

    const int W = 16, H = 16, F = 2;  // -> grid h'=8, w'=8, f'=2 -> L_vid=128
    const int Lraw = 32;              // -> L_audio = 32

    auto video    = load_or_dummy(in_dir, "video.bin",    {W, H, F, 48});
    auto audio    = load_or_dummy(in_dir, "audio.bin",    {128, Lraw});
    auto context  = load_or_dummy(in_dir, "context.bin",  {4096, 512});
    auto timestep = load_or_dummy(in_dir, "timestep.bin", {1});
    if (in_dir.empty() || !path_exists(in_dir + "/timestep.bin")) {
        timestep.data()[0] = 500.0f;
    }

    // NAVA_I2V=1: replace the scalar timestep with the PER-TOKEN clean-anchor t
    // (first frame's h'*w' VIDEO tokens = 0, all other video + all audio = t) so the
    // per-token-timestep path can be validated against the PyTorch first_frame_is_clean
    // reference (~/dev/NAVA/nava_dump_i2v_ref.py).
    if (getenv("NAVA_I2V")) {
        // derive the grid from the LOADED video tensor [W,H,F,48], not the smoke
        // fallback consts, so the per-token t is correct at any resolution.
        const int Wv = (int)video.shape()[0], Hv = (int)video.shape()[1], Fv = (int)video.shape()[2];
        const int h_grid = Hv / 2, w_grid = Wv / 2;
        const int64_t L_vid = (int64_t)Fv * h_grid * w_grid;
        const int64_t L_aud = audio.shape()[1];
        const int64_t L_total = L_vid + L_aud;
        const int64_t n_clean = (int64_t)h_grid * w_grid;         // first-frame spatial tokens
        const float tval = timestep.data()[0];
        sd::Tensor<float> t_tok({L_total});
        for (int64_t i = 0; i < L_total; ++i) t_tok.data()[i] = (i < n_clean) ? 0.0f : tval;
        timestep = std::move(t_tok);
        printf("NAVA_I2V: per-token t [L_total=%lld], first %lld tokens = 0, rest = %.1f\n",
               (long long)L_total, (long long)n_clean, tval);
    }

    dump_stats("input video",    video);
    dump_stats("input audio",    audio);
    dump_stats("input context",  context);
    dump_stats("input timestep", timestep);

    ggml_backend_t backend = ggml_backend_cpu_init();
    printf("backend: CPU\n");

    ModelLoader loader;
    if (!loader.init_from_file(gguf_path, "")) {
        printf("failed to init loader from %s\n", gguf_path.c_str());
        return 1;
    }
    const bool head_f32 = getenv("NAVA_HEAD_F32") != nullptr;
    auto& tsm = loader.get_tensor_storage_map();
    for (auto& [name, ts] : tsm) {
        if (ends_with(name, "weight") && ts.type == GGML_TYPE_F16) {
            ts.expected_type = GGML_TYPE_F16;
        }
        // Experiment: upcast the two head Linear weights to F32 to isolate whether
        // the ~44 dB head error is F16-vs-bf16 precision or a structural op bug.
        if (head_f32 && (name == "backbone.head.head.weight"
                         || name == "backbone.head_audio.head.weight")) {
            ts.expected_type = GGML_TYPE_F32;
            printf("NAVA_HEAD_F32: upcasting %s to F32\n", name.c_str());
        }
    }

    auto runner = std::make_shared<NAVA::NavaRunner>(backend, backend, tsm, "backbone");
    runner->alloc_params_buffer();

    std::map<std::string, ggml_tensor*> tensors;
    runner->get_param_tensors(tensors, "backbone");
    printf("expecting %zu param tensors\n", tensors.size());
    if (!loader.load_tensors(tensors)) {
        printf("failed to load NAVA tensors\n");
        return 1;
    }
    printf("NAVA backbone loaded\n");

    int n_threads = 8;
    // NAVA_MASK_MODALITY=1: run the separate intra-modal attention path (align_3d_cfg
    // masking_modality forward) so it can be diffed vs the PyTorch mmask reference.
    if (getenv("NAVA_MASK_MODALITY")) {
        runner->mask_modality = true;
        printf("NAVA_MASK_MODALITY: separate intra-modal self-attention\n");
    }
    int64_t t0 = ggml_time_ms();
    auto vel_video = runner->compute(n_threads, video, audio, context, timestep);
    int64_t t1 = ggml_time_ms();
    printf("forward done in %lld ms\n", (long long)(t1 - t0));

    // VALIDATE compute_va() == compute(): the render uses compute_va (joint
    // video+audio via pad/concat/split); the per-block validation used compute().
    // The video velocity MUST be bit-identical between them — any diff is a
    // render-only bug (slicing/stride in the joint path).
    {
        // pass the same mask_modality the compute() above used, else this compares
        // masked-compute() vs unmasked-compute_va() (compute_va resets the flag).
        auto vv = runner->compute_va(n_threads, video, audio, context, timestep, runner->mask_modality).first;
        dump_stats("compute()    vel_video", vel_video);
        dump_stats("compute_va() vv        ", vv);
        if (!vv.empty() && vv.numel() == vel_video.numel()) {
            double mx = 0, sa = 0;
            for (int64_t i = 0; i < vv.numel(); ++i) {
                double d = std::fabs((double)vv.data()[i] - (double)vel_video.data()[i]);
                if (d > mx) mx = d;
                sa += d;
            }
            printf(">>> compute_va vs compute: maxAbsDiff=%.6g  meanAbsDiff=%.6g  (MUST be ~0)\n",
                   mx, sa / (double)vv.numel());
        } else {
            printf(">>> compute_va vs compute: SHAPE MISMATCH vv.numel=%lld vel.numel=%lld\n",
                   (long long)vv.numel(), (long long)vel_video.numel());
        }
    }

    dump_stats("velocity_video_patched", vel_video);
    if (!vel_video.empty()) {
        write_bin(out_dir + "/nava_velocity_video_patched.bin", vel_video, "velocity_video_patched");
    } else {
        printf("WARNING: empty output\n");
        return 1;
    }
    printf("(audio velocity + per-block hidden states are auto-logged above as debug-tensor stats)\n");

    // --- VALIDATE compute_va() vs compute() -------------------------------
    // The render path uses compute_va() (joint pad+concat+host-split). It must
    // return a video velocity bit-identical to compute()'s. Compare here.
    if (getenv("NAVA_VALIDATE_VA")) {
        printf("\n=== NAVA_VALIDATE_VA: compute() vs compute_va() ===\n");
        auto [vv, va] = runner->compute_va(n_threads, video, audio, context, timestep);
        dump_stats("compute_va vv (video vel)", vv);
        dump_stats("compute_va va (audio vel)", va);
        if (vv.empty()) {
            printf("WARNING: compute_va returned empty vv\n");
            return 1;
        }
        if (vv.shape().size() != 2 || vv.shape()[0] != vel_video.shape()[0]
            || vv.shape()[1] != vel_video.shape()[1]) {
            printf("SHAPE MISMATCH: compute() vv ne[%lld,%lld] vs compute_va() vv ne[%lld,%lld]\n",
                   (long long)vel_video.shape()[0], (long long)vel_video.shape()[1],
                   (long long)vv.shape()[0], (long long)vv.shape()[1]);
            return 1;
        }
        const int64_t n = (int64_t)vel_video.shape()[0] * vel_video.shape()[1];
        double maxabs = 0.0, sumsq_diff = 0.0, sumsq_ref = 0.0;
        int64_t argmax = -1;
        const float* a = vel_video.data();
        const float* b = vv.data();
        for (int64_t i = 0; i < n; ++i) {
            double d = std::fabs((double)a[i] - (double)b[i]);
            if (d > maxabs) { maxabs = d; argmax = i; }
            sumsq_diff += (double)(a[i] - b[i]) * (a[i] - b[i]);
            sumsq_ref  += (double)a[i] * a[i];
        }
        double mse  = sumsq_diff / (double)n;
        double peak = 0.0;
        for (int64_t i = 0; i < n; ++i) peak = std::max(peak, (double)std::fabs(a[i]));
        double psnr = (mse > 0.0) ? 20.0 * std::log10(peak) - 10.0 * std::log10(mse) : 999.0;
        printf("compute_va vv vs compute() vv: N=%lld maxabsdiff=%.6g (at idx %lld: %.6g vs %.6g) "
               "rel_l2=%.6g PSNR=%.2f dB\n",
               (long long)n, maxabs, (long long)argmax,
               argmax >= 0 ? a[argmax] : 0.0f, argmax >= 0 ? b[argmax] : 0.0f,
               std::sqrt(sumsq_diff / (sumsq_ref + 1e-30)), psnr);
        if (maxabs < 1e-4) {
            printf("RESULT: compute_va is CLEAN (bit-identical within tol).\n");
        } else {
            printf("RESULT: compute_va DIFFERS — BUG in joint pad/concat/split path.\n");
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Phase-2 render mode.
// ---------------------------------------------------------------------------

struct RenderOpts {
    std::string gguf      = "models/nava-dit-f16.gguf";
    std::string vae       = "models/longcat-wan-vae-f16.gguf";
    std::string audio_vae = "models/nava-ltx-audio-vae-f16.gguf";  // LTX audio VAE; "" => silent webm
    std::string context;       // cond context .bin (raw umT5 [4096,512] or [3072,512]); empty => dummy
    std::string neg_context;   // VIDEO uncond context; empty => zeros_like(context)
    std::string audio_neg_context;  // AUDIO uncond context; empty => falls back to neg_context
    std::string prompt    = "(informational only — context comes from --context)";
    // Inline text-encode (Task: umT5 in-process). --umt5 <gguf> turns on the
    // inline path: the prompt (from --prompt or --prompt-file) is encoded through
    // umT5 here, umT5 is freed, THEN the DiT loads — no precomputed --context bin
    // and no shell-out to `encode-prompt`. spk-pos is resolved from the inline
    // encode (no .spkpos sidecar needed).
    std::string umt5;
    std::string prompt_file;
    std::string label;
    std::string out_name;
    std::string runs_dir  = "/mnt/hdd/nava/cpp-runs";
    int steps             = 10;
    int frames            = 13;   // latent temporal frames
    int width             = 832;  // pixel
    int height            = 480;  // pixel
    int fps               = 24;
    uint64_t seed         = 42;
    float cfg             = 3.0f;
    float cfg_align       = 3.0f;   // video_align_guidance_scale (align_3d_cfg)
    float cfg_align_audio = 2.0f;   // audio_align_guidance_scale (align_3d_cfg)
    float shift           = 5.0f;
    bool cuda             = false;
    std::string image;         // I2V: input image; frame-0 clean-latent anchor (empty => T2V)
    // Voice clone (Task 2). --spk-emb: a [192] f32 .bin (ReDimNet speaker embedding,
    // precomputed offline). --spk-pos: comma-sep token indices for the <extra_id_2>
    // splice; if empty, auto-loaded from "<context>.spkpos" (written by encode-prompt).
    // --timbre-cfg: timbre_align_guidance_scale (>0 enables the 4th timbre-uncond audio
    // forward, model_mm.py/pipeline_nava.py; 0 = off). Clone active iff spk-emb + spk-pos.
    std::string spk_emb;
    std::string spk_pos_str;
    float timbre_cfg      = 0.0f;
    // Clip continuation (Task 3). --video-anchor: a prior segment's tail video latent
    // [W_lat,H_lat,N,48] (diffusion convention) — its N frames are pinned as clean
    // anchors (frames 0..N-1, timestep 0). --audio-anchor: prior tail audio latent
    // [128,K] — first K audio tokens pinned clean. Both chain DIRECTLY from a prior
    // render's dumped latents (NAVA_DUMP_LATENT / NAVA_DUMP_AUDIO_LATENT); no VAE
    // re-encode needed. --image still gives the single-frame i2v anchor (N=1).
    std::string video_anchor;
    std::string audio_anchor;
    // Continuation seam softener (Track 1, SDEdit warm-start). --warm-latent: a prior
    // segment's dumped diffusion video latent [W_lat,H_lat,Kf,48] (from NAVA_DUMP_LATENT).
    // Its LAST frame (a velocity-carrying P-frame, std~0.88 — NOT a reset I-frame) is
    // broadcast across all f_len frames as the clean x0; the video latent is init'd
    // (1-sigma)*x0 + sigma*noise and the VIDEO stream denoises from the schedule step
    // whose sigma <= --warm-strength. AUDIO keeps the full schedule (Track 1 leaves the
    // per-segment speech path untouched). 0 = disabled.
    std::string warm_latent;
    float warm_strength   = 0.0f;
    int   warm_overlap    = 4;     // # leading frames seeded from the prior tail SEQUENCE
    // SYNC mode (Track 1+2): when set, the AUDIO stream is also warm-started from the prior
    // segment's tail audio latent [128,audio_len] at the SAME sigma_held and both streams
    // denoise the truncated schedule together — no freeze, no video/audio noise-level
    // desync (which corrupts the joint audio), and speech flows across the seam.
    std::string warm_audio;
};

// FlowMatchScheduler (Euler), distilled from NAVA scheduler/flow_match.py.
// sigmas = linspace(sigma_max, sigma_min, N); shift transform; timesteps = sigma*1000.
struct FlowMatchSched {
    std::vector<float> sigmas;
    std::vector<float> timesteps;
    void set_timesteps(int n, float shift) {
        const float sigma_max = 1.0f;
        const float sigma_min = 0.003f / 1.002f;
        sigmas.resize(n);
        for (int i = 0; i < n; ++i) {
            // NAVA's FlowMatchScheduler uses extra_one_step=True:
            // sigmas = linspace(sigma_max, sigma_min, n+1)[:-1]  -> divide by n (NOT n-1),
            // so the steps stay at higher sigma and the FINAL Euler step is one big jump
            // sigma_last -> 0. (Dividing by n-1 reaches sigma_min and crawls the tail,
            // a different denoising trajectory that destabilizes strong CFG guidance.)
            float s = (n == 1) ? sigma_max
                               : sigma_max + (sigma_min - sigma_max) * ((float)i / (float)n);
            // shift transform: s <- shift*s / (1 + (shift-1)*s)
            sigmas[i] = shift * s / (1.0f + (shift - 1.0f) * s);
        }
        timesteps.resize(n);
        for (int i = 0; i < n; ++i) timesteps[i] = sigmas[i] * 1000.0f;
    }
    // sigma_next for step i (0 at the final step).
    float sigma_next(int i) const {
        return (i + 1 < (int)sigmas.size()) ? sigmas[i + 1] : 0.0f;
    }
};

// Unpatchify the DiT head output [192, L_vid] (ggml ne; 192 = (p=1,q=2,r=2,c=48),
// c fastest; L_vid token order f-major then h then w) into a VAE-latent video
// tensor with ggml ne [W_lat, H_lat, F, 48] (W fastest) — exactly what WanVAE
// decode + diffusion_to_vae_latents expect. Mirrors NAVA model_mm.py:
//   u.view(F,H',W',1,2,2,48); einsum('fhwpqrc->cfphqwr') -> [48,F,H'*2,W'*2].
// (here H'=h_grid, W'=w_grid are the DiT conv grid = VAE-latent /2; output H_lat=H'*2.)
static sd::Tensor<float> unpatchify_video(const sd::Tensor<float>& patched,
                                          int f_len, int h_grid, int w_grid) {
    const int pq = 2, pr = 2, c = 48;
    const int64_t L_vid = (int64_t)f_len * h_grid * w_grid;
    // patched ggml ne [192, L_vid]; data ne0 (the 192) fastest.
    // within a token the 192 are ordered (p,q,r,c) with c fastest (matches the
    // .view(...,1,2,2,48) layout: last dim c is fastest).
    const float* src = patched.data();
    const int64_t H_lat = (int64_t)h_grid * pq;
    const int64_t W_lat = (int64_t)w_grid * pr;
    // output ggml ne [W_lat, H_lat, F, 48]; data idx = ((cc*F + ff)*H_lat + yy)*W_lat + xx
    sd::Tensor<float> out({W_lat, H_lat, (int64_t)f_len, (int64_t)c});
    float* dst = out.data();
    for (int ff = 0; ff < f_len; ++ff) {
        for (int hh = 0; hh < h_grid; ++hh) {
            for (int ww = 0; ww < w_grid; ++ww) {
                int64_t tok = ((int64_t)ff * h_grid + hh) * w_grid + ww;
                const float* tbase = src + tok * (int64_t)(pq * pr * c);
                for (int q = 0; q < pq; ++q) {        // sub-row within patch (H)
                    for (int r = 0; r < pr; ++r) {    // sub-col within patch (W)
                        for (int cc = 0; cc < c; ++cc) {
                            // src layout (p=1,q,r,c): index = ((q*pr + r)*c) + cc
                            float v = tbase[((int64_t)q * pr + r) * c + cc];
                            int64_t yy = (int64_t)hh * pq + q;
                            int64_t xx = (int64_t)ww * pr + r;
                            int64_t didx = (((int64_t)cc * f_len + ff) * H_lat + yy) * W_lat + xx;
                            dst[didx] = v;
                        }
                    }
                }
            }
        }
    }
    (void)L_vid;
    return out;
}

static std::string now_stamp() {
    time_t t = time(nullptr);
    struct tm tmv;
    localtime_r(&t, &tmv);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tmv);
    return buf;
}

// ---------------------------------------------------------------------------
// umT5 text-context encode — shared by `encode-prompt`, the inline `render
// --umt5`/`--prompt` path, and `encode-prompts` (batch). Factoring this keeps
// the three paths byte-identical: the load-bearing tokenizer transform
// (<S> -> <S><extra_id_2> , NFKC normalize, 512-cap with EOS-at-511) lives in
// ONE place. See the encode-prompt subcommand below for the full rationale of
// each step.
// ---------------------------------------------------------------------------

// mmap capability for the umT5 load. Default OFF, opt-in via NAVA_UMT5_MMAP=1.
// MEASURED A/B (peter_prompt, warm page cache, CUDA backend): mmap is a WASH on
// load time (2.14s on vs 2.27s off) and RAISES host RSS 1.4GB -> 7.3GB, because
// the whole 6GB gguf maps resident while the weights are simultaneously copied
// to the GPU params buffer (the encode is transient — umT5 is freed before the
// DiT loads, so nothing is reused). The prod sd-server/longcat mmap RAM win is
// for LONG-LIVED CPU-resident weights; it does not transfer to this transient
// GPU encode. Kept as opt-in so a future CPU-backend / pinned-cache use can flip
// it on, and values are bit-identical either way (proven by cmp).
static bool nava_umt5_use_mmap() { return getenv("NAVA_UMT5_MMAP") != nullptr; }

static const char* kNavaT5Prefix = "text_encoders.t5xxl.transformer";

// Load umT5-xxl into a RESIDENT T5Embedder on `backend`. Caller owns the
// returned embedder and MUST reset() it to free the ~6GB params buffer before
// the DiT loads (umT5 + DiT do not co-reside on a 12GB card). The ModelLoader
// is local: the plain load_tensors path COPIES file data into the alloc'd
// params buffer, so the mmap can be released as soon as load returns.
static std::shared_ptr<T5Embedder> nava_load_umt5(const std::string& gguf,
                                                  ggml_backend_t backend,
                                                  bool use_mmap,
                                                  double* load_secs = nullptr) {
    int64_t t0 = ggml_time_ms();
    ModelLoader ml;
    if (!ml.init_from_file_and_convert_name(gguf)) {
        printf("umt5 load fail: %s\n", gguf.c_str());
        return nullptr;
    }
    auto& tsm = ml.get_tensor_storage_map();
    auto t5   = std::make_shared<T5Embedder>(backend, backend, tsm, kNavaT5Prefix, /*is_umt5=*/true);
    t5->alloc_params_buffer();
    std::map<std::string, ggml_tensor*> tensors;
    t5->get_param_tensors(tensors, kNavaT5Prefix);
    if (!ml.load_tensors(tensors, {}, /*n_threads=*/0, /*use_mmap=*/use_mmap)) {
        printf("umt5 tensors load fail\n");
        return nullptr;
    }
    if (load_secs) *load_secs = (ggml_time_ms() - t0) / 1000.0;
    return t5;
}

// Encode ONE raw prompt through a resident umT5 into a NAVA cond context
// [4096,512] (zero-padded). If spk_pos_out != null, fills it with the
// <extra_id_2>(=256297) token positions for voice-clone splicing. Returns an
// empty tensor on encode failure. This is the EXACT logic of the original
// encode-prompt subcommand — keep them identical.
static sd::Tensor<float> nava_encode_prompt_ctx(const std::shared_ptr<T5Embedder>& t5,
                                                const std::string& raw_prompt,
                                                std::vector<int>* spk_pos_out) {
    // NAVA data-loader caption transform (t2v.py, use_speech_special_token=false):
    // insert <extra_id_2> after each <S>; the TRAILING SPACE is load-bearing so
    // the unigram tokenizer emits the metaspace on the following word (matches HF).
    std::string text = raw_prompt;
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
    {
        const std::string from = "<S>", to = "<S><extra_id_2> ";
        for (size_t p = 0; (p = text.find(from, p)) != std::string::npos; p += to.size())
            text.replace(p, from.size(), to);
    }
    text = nava_normalize_text(text);
    auto tw      = t5->tokenize(text, 0, false);
    auto& tokens = std::get<0>(tw);
    auto& masks  = std::get<2>(tw);
    if (const char* tp = getenv("NAVA_DUMP_TOKENS")) {
        FILE* tf = fopen(tp, "w");
        if (tf) { for (auto id : tokens) fprintf(tf, "%d\n", (int)id); fclose(tf);
                  printf("dumped %zu token ids -> %s\n", tokens.size(), tp); }
    }
    const size_t kMaxTok = 512;
    if (tokens.size() > kMaxTok) {
        int eos = tokens.back();  // keep 511 content + EOS at 511 (bidirectional encoder)
        tokens.resize(kMaxTok - 1);
        tokens.push_back(eos);
        masks.assign(kMaxTok, 0.0f);
    }
    if (spk_pos_out) {
        spk_pos_out->clear();
        for (size_t i = 0; i < tokens.size(); ++i)
            if (tokens[i] == 256297) spk_pos_out->push_back((int)i);
    }
    auto input_ids = sd::Tensor<int32_t>::from_vector(tokens);
    auto attn_mask = sd::Tensor<float>::from_vector(masks);
    auto emb       = t5->model.compute(8, input_ids, attn_mask);  // ne [4096, L]
    if (emb.empty()) return sd::Tensor<float>();
    int64_t C  = emb.shape()[0];
    int64_t L  = emb.shape().size() > 1 ? emb.shape()[1] : 1;
    int64_t Lc = std::min<int64_t>(L, 512);
    sd::Tensor<float> ctx({C, 512});
    std::fill(ctx.data(), ctx.data() + ctx.numel(), 0.0f);
    for (int64_t tok = 0; tok < Lc; ++tok)
        for (int64_t c = 0; c < C; ++c)
            ctx.data()[c + C * tok] = emb.data()[c + C * tok];
    return ctx;
}

// Read a whole file into a string (prompt text). Returns false if unreadable.
static bool nava_read_file(const std::string& path, std::string& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    out.clear();
    if (n > 0) { out.resize((size_t)n); size_t r = fread(&out[0], 1, (size_t)n, f); out.resize(r); }
    fclose(f);
    return true;
}

static int run_render(int argc, char** argv) {
    RenderOpts o;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char* def) -> std::string {
            if (i + 1 < argc) return argv[++i];
            return def;
        };
        if (a == "--prompt") o.prompt = next("");
        else if (a == "--prompt-file") o.prompt_file = next("");
        else if (a == "--umt5") o.umt5 = next("");
        else if (a == "--context") o.context = next("");
        else if (a == "--neg-context") o.neg_context = next("");
        else if (a == "--audio-neg-context") o.audio_neg_context = next("");
        else if (a == "--steps") o.steps = atoi(next("10").c_str());
        else if (a == "--frames") o.frames = atoi(next("13").c_str());
        else if (a == "--width") o.width = atoi(next("832").c_str());
        else if (a == "--height") o.height = atoi(next("480").c_str());
        else if (a == "--fps") o.fps = atoi(next("24").c_str());
        else if (a == "--seed") o.seed = (uint64_t)strtoull(next("42").c_str(), nullptr, 10);
        else if (a == "--image") o.image = next("");
        else if (a == "--cfg") o.cfg = atof(next("3.0").c_str());
        else if (a == "--cfg-align") o.cfg_align = atof(next("3.0").c_str());
        else if (a == "--cfg-align-audio") o.cfg_align_audio = atof(next("2.0").c_str());
        else if (a == "--shift") o.shift = atof(next("5.0").c_str());
        else if (a == "--out-name") o.out_name = next("");
        else if (a == "--runs-dir") o.runs_dir = next("");
        else if (a == "--vae") o.vae = next("");
        else if (a == "--audio-vae") o.audio_vae = next("");
        else if (a == "--no-audio") o.audio_vae = "";
        else if (a == "--gguf") o.gguf = next("");
        else if (a == "--label") o.label = next("");
        else if (a == "--spk-emb") o.spk_emb = next("");
        else if (a == "--spk-pos") o.spk_pos_str = next("");
        else if (a == "--timbre-cfg") o.timbre_cfg = atof(next("0.0").c_str());
        else if (a == "--video-anchor") o.video_anchor = next("");
        else if (a == "--audio-anchor") o.audio_anchor = next("");
        else if (a == "--warm-latent") o.warm_latent = next("");
        else if (a == "--warm-strength") o.warm_strength = atof(next("0.0").c_str());
        else if (a == "--warm-overlap") o.warm_overlap = atoi(next("4").c_str());
        else if (a == "--warm-audio") o.warm_audio = next("");
        else if (a == "--cuda") o.cuda = true;
        else { printf("unknown render arg: %s\n", a.c_str()); return 1; }
    }
    if (o.out_name.empty()) o.out_name = "nava-" + now_stamp();
    if (o.label.empty()) o.label = o.out_name;

    setenv("LONGCAT_NO_FUSED_ROPE", "1", /*overwrite=*/0);

    // Latent grid. VAE spatial downsample = 16; DiT patch [1,2,2] downsamples by
    // another 2 -> conv grid = latent/2 = pixel/32. (See NAVA t2v.py:277-278.)
    const int W_lat = o.width / 16;
    const int H_lat = o.height / 16;
    const int w_grid = W_lat / 2;
    const int h_grid = H_lat / 2;
    const int f_len  = o.frames;
    // Audio stream length. NAVA is a JOINT audio-video MMDiT — every block does
    // joint self-attention over [video ++ audio]. A degenerate audio stream
    // (1 token, near-zero) is OUT of the trained distribution and can poison the
    // joint attention. Size it to the video duration (audio_tokens_per_sec=25,
    // video_fps=24 per nava_run.yaml), min 8, so the joint forward sees a
    // plausible audio stream even though we render silent.
    // PyTorch uses ceil (t2v.py:417: math.ceil(video_duration * audio_tokens_per_sec)),
    // NOT round — match it so the joint audio sequence length is identical.
    const int audio_len = std::max(8, (int)std::ceil(((f_len - 1) * 4 + 1) / 24.0 * 25.0));

    printf("=== NAVA render (SILENT) ===\n");
    printf("prompt: %s\n", o.prompt.c_str());
    printf("steps=%d frames=%d (pixel=%d) res=%dx%d latent=[48,%d,%d,%d] grid=[%d,%d,%d] seed=%llu cfg=%.2f shift=%.2f\n",
           o.steps, f_len, (f_len - 1) * 4 + 1, o.width, o.height, f_len, H_lat, W_lat,
           f_len, h_grid, w_grid, (unsigned long long)o.seed, o.cfg, o.shift);

    // backend
    ggml_backend_t backend = nullptr;
    if (o.cuda) {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
        printf("backend: CUDA\n");
#else
        printf("--cuda requested but binary built without CUDA; falling back to CPU\n");
#endif
    }
    if (backend == nullptr) {
        backend = ggml_backend_cpu_init();
        printf("backend: CPU\n");
    }

    int n_threads = 8;
    int64_t wall_t0 = ggml_time_ms();

    // ----- INLINE umT5 text-encode (--umt5) — encode THEN free, before DiT -----
    // Makes a multi-segment render self-sufficient (raw text -> clip) with no
    // precomputed --context bin and no per-segment shell-out. umT5 (~6.4GB) is
    // resident only for the encode and freed here, so it never co-resides with
    // the ~7GB DiT. The image VAE (below) and DiT load after, strictly
    // sequential. Bit-identical to `nava encode-prompt` (shared helper).
    sd::Tensor<float> inline_ctx;
    std::vector<int>  inline_spk_pos;
    bool have_inline_ctx = false;
    if (!o.umt5.empty()) {
        std::string ptext;
        if (!o.prompt_file.empty()) {
            if (!nava_read_file(o.prompt_file, ptext)) {
                printf("ERROR: --prompt-file %s unreadable\n", o.prompt_file.c_str());
                return 1;
            }
        } else {
            ptext = o.prompt;
        }
        const bool mmap = nava_umt5_use_mmap();
        double load_s = 0;
        auto t5 = nava_load_umt5(o.umt5, backend, mmap, &load_s);
        if (!t5) return 1;
        int64_t e0 = ggml_time_ms();
        inline_ctx = nava_encode_prompt_ctx(t5, ptext, &inline_spk_pos);
        double enc_s = (ggml_time_ms() - e0) / 1000.0;
        t5.reset();  // FREE umT5 (~6.4GB) BEFORE the DiT loads
        if (inline_ctx.numel() == 0) { printf("ERROR: inline umT5 encode returned empty\n"); return 1; }
        have_inline_ctx = true;
        printf("INLINE umT5: load %.2fs + encode %.2fs (mmap=%s) -> cond ctx [4096,512], %zu spk pos\n",
               load_s, enc_s, mmap ? "on" : "off", inline_spk_pos.size());
        if (const char* dp = getenv("NAVA_DUMP_INLINE_CTX")) {
            write_bin(dp, inline_ctx, "context");
            printf("NAVA_DUMP_INLINE_CTX: wrote inline cond ctx -> %s\n", dp);
        }
    }

    // ----- I2V: encode the input image -> frame-0 clean-latent anchor -----
    // --image accepts a real encoded image (.png/.jpg/...): decoded + Python-parity
    // resize/center-crop/normalize in-port via load_image_resize_center_crop (matches
    // nava_src/vae/local_video_vae.py:_resize_center_crop). A pre-baked RGB bin (ggml ne
    // [W,H,1,3], [0,1] from tools/nava_prep_image.py) is still accepted for back-compat.
    // VAE-encode it (decode_only=false) BEFORE the 12.6GB DiT loads (VRAM: VAE alone fits,
    // then freed). The anchor
    // is spliced at frame 0 and pinned each step via the per-token clean-anchor timestep.
    // Clean-anchor video latent [W_lat,H_lat,N,48] (diffusion convention): N tail frames
    // of a prior segment (--video-anchor) OR a single i2v image (--image, N=1).
    sd::Tensor<float> vid_anchor;
    int n_anchor_v = 0;
    if (!o.video_anchor.empty()) {
        if (!path_exists(o.video_anchor)) { printf("ERROR: --video-anchor %s not found\n", o.video_anchor.c_str()); return 1; }
        vid_anchor = sd::load_tensor_from_file_as_tensor<float>(o.video_anchor);
        if (vid_anchor.dim() < 4 || vid_anchor.shape()[0] != (int64_t)W_lat
            || vid_anchor.shape()[1] != (int64_t)H_lat || vid_anchor.shape()[3] != 48) {
            printf("ERROR: --video-anchor ne %s != [%d,%d,N,48]\n",
                   sd::tensor_shape_to_string(vid_anchor.shape()).c_str(), W_lat, H_lat);
            return 1;
        }
        n_anchor_v = (int)vid_anchor.shape()[2];
        printf("CONTINUATION: video anchor %s -> %d clean frame(s)\n", o.video_anchor.c_str(), n_anchor_v);
    } else if (!o.image.empty()) {
        if (!path_exists(o.image)) { printf("ERROR: --image %s not found\n", o.image.c_str()); return 1; }
        sd::Tensor<float> img_rgb;
        if (looks_like_image_file(o.image)) {
            // Real image file: decode + Python-parity resize/center-crop/normalize in-port.
            if (!load_image_resize_center_crop(o.image, W_lat * 16, H_lat * 16, img_rgb)) return 1;
        } else {
            // Legacy: pre-baked RGB bin (ggml ne [W,H,1,3], [0,1]) from tools/nava_prep_image.py.
            img_rgb = sd::load_tensor_from_file_as_tensor<float>(o.image);
            printf("I2V: loaded preprocessed RGB bin %s shape=%s\n", o.image.c_str(),
                   sd::tensor_shape_to_string(img_rgb.shape()).c_str());
        }
        auto encvae = std::make_shared<WAN::WanVAERunner>(
            backend, backend, String2TensorStorage{}, "", /*decode_only=*/false, VERSION_WAN2_2_TI2V);
        {
            ModelLoader vl;
            if (!vl.init_from_file_and_convert_name(o.vae, "vae.")) { printf("I2V VAE loader fail\n"); return 1; }
            encvae->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> vt;
            encvae->get_param_tensors(vt, "first_stage_model");
            if (!vl.load_tensors(vt)) { printf("I2V VAE tensors fail\n"); return 1; }
        }
        sd_tiling_params_t et = {};
        // Python encodes the I2V first frame in ONE full-frame VAE forward. C++ used to
        // tile (24x24 latent tiles, 0.25 overlap), and the SEAM BLENDING perturbed the
        // frame-0 anchor latent by ~14% (reldiff 0.137 -> 0.005 vs Python when off),
        // which propagated into the audio via cross-modal attention. Full-frame is the
        // default now: it's a SINGLE frame encoded in isolation before the DiT loads
        // (VAE freed right after), so peak ~2.6GB << the DiT sampling peak (~8.9GB) — no
        // real VRAM cost. NAVA_VAE_ENCODE_TILE=1 forces tiling back for huge inputs.
        et.enabled = (W_lat > 24 || H_lat > 24) && (getenv("NAVA_VAE_ENCODE_TILE") != nullptr);
        et.temporal_tiling = false; et.tile_size_x = 24; et.tile_size_y = 24; et.target_overlap = 0.25f;
        printf("I2V: VAE encode tiling=%s\n", et.enabled ? "ON (24x24/0.25)" : "OFF (full-frame, Python-parity)");
        auto mu = encvae->encode(n_threads, img_rgb, et, /*encode_video=*/true, false);
        if (mu.empty()) { printf("ERROR: I2V VAE encode failed\n"); return 1; }
        printf("I2V: encoded mu shape=%s\n", sd::tensor_shape_to_string(mu.shape()).c_str());
        sd::Tensor<float> mu5 = mu;
        mu5.reshape_({(int64_t)W_lat, (int64_t)H_lat, 1, 48, 1});
        vid_anchor = encvae->vae_to_diffusion_latents(mu5);
        vid_anchor.reshape_({(int64_t)W_lat, (int64_t)H_lat, 1, 48});
        n_anchor_v = 1;
        encvae.reset();  // free VAE before the DiT loads
        dump_stats("I2V anchor latent (diffusion)", vid_anchor);
        if (const char* ap = getenv("NAVA_DUMP_ANCHOR")) {
            write_bin(ap, vid_anchor, "vid_anchor");  // [W_lat,H_lat,1,48] diffusion latent
            printf("NAVA_DUMP_ANCHOR: wrote frame-0 anchor -> %s\n", ap);
        }
    }
    // Clean-anchor audio latent [128,K]: first K audio tokens pinned to a prior segment's
    // tail audio latent (--audio-anchor). Chains directly from a dumped audio latent.
    sd::Tensor<float> aud_anchor;
    int n_anchor_a = 0;
    if (!o.audio_anchor.empty()) {
        if (!path_exists(o.audio_anchor)) { printf("ERROR: --audio-anchor %s not found\n", o.audio_anchor.c_str()); return 1; }
        aud_anchor = sd::load_tensor_from_file_as_tensor<float>(o.audio_anchor);
        if (aud_anchor.shape()[0] != 128) {
            printf("ERROR: --audio-anchor ne %s != [128,K]\n", sd::tensor_shape_to_string(aud_anchor.shape()).c_str());
            return 1;
        }
        n_anchor_a = (int)aud_anchor.shape()[1];
        if (n_anchor_a >= audio_len) n_anchor_a = audio_len - 1;  // keep >=1 token to denoise
        printf("CONTINUATION: audio anchor %s -> %d clean token(s) (of %d)\n",
               o.audio_anchor.c_str(), n_anchor_a, audio_len);
    }
    const bool i2v = (n_anchor_v > 0) || (n_anchor_a > 0);  // any clean-anchor => per-token timestep

    // ----- load DiT -----
    ModelLoader loader;
    if (!loader.init_from_file(o.gguf, "")) {
        printf("failed to init loader from %s\n", o.gguf.c_str());
        return 1;
    }
    auto& tsm = loader.get_tensor_storage_map();
    for (auto& [name, ts] : tsm) {
        if (ends_with(name, "weight") && ts.type == GGML_TYPE_F16) {
            ts.expected_type = GGML_TYPE_F16;
        }
    }
    auto runner = std::make_shared<NAVA::NavaRunner>(backend, backend, tsm, "backbone");
    // Flash attention: fuses the joint self-attn (L_total ~5k tokens, head_dim 128) so the
    // O(L^2) scores tensor is never materialized -> big activation-VRAM + speed win on the
    // DiT phase. head_dim 128 is FA-friendly. Default ON; NAVA_NO_FLASH=1 reverts to the
    // unfused path (the numerically-validated reference). Prec is F32 inside FA.
    if (getenv("NAVA_NO_FLASH") == nullptr) {
        runner->set_flash_attention_enabled(true);
        printf("flash attention: ON\n");
    }
    runner->alloc_params_buffer();
    {
        std::map<std::string, ggml_tensor*> tensors;
        runner->get_param_tensors(tensors, "backbone");
        if (!loader.load_tensors(tensors)) {
            printf("failed to load NAVA tensors\n");
            return 1;
        }
    }
    int64_t load_t1 = ggml_time_ms();
    printf("NAVA backbone loaded (%.2fs)\n", (load_t1 - wall_t0) / 1000.0f);

    // ----- contexts -----
    // cond context: raw umT5 [4096,512] (DiT text_embedding runs) or post-embed
    // [3072,512]. Dummy if not provided (smoke). uncond = zeros_like (NAVA MVP).
    // BUGFIX (the Phase-2 incoherence): load the cond context DIRECTLY by its
    // (typically absolute) path. The old code used load_or_dummy(".", o.context)
    // which builds "./" + "/abs/path" = ".//abs/path" — a RELATIVE path that never
    // exists, so it SILENTLY fell back to the sin-based dummy context. The render
    // then denoised with a dummy prompt embedding (context_embedded std 0.036 vs
    // the real 0.21) → every clip was unconditioned garbage AND the dummy-cond vs
    // real-neg CFG guidance was biased, driving the std runaway. (ctx_neg below was
    // already loaded correctly via load_tensor_from_file_as_tensor.)
    sd::Tensor<float> ctx_pos;
    if (have_inline_ctx) {
        ctx_pos = std::move(inline_ctx);
        printf("cond context from INLINE umT5 encode (ne0=%lld)\n",
               (long long)ctx_pos.shape()[0]);
    } else if (!o.context.empty()) {
        if (!path_exists(o.context)) {
            printf("ERROR: --context %s not found\n", o.context.c_str());
            return 1;
        }
        ctx_pos = sd::load_tensor_from_file_as_tensor<float>(o.context);
        printf("loaded cond context %s (ne0=%lld)\n", o.context.c_str(),
               (long long)ctx_pos.shape()[0]);
    } else {
        printf("NOTE: no --context => DUMMY cond context (garbage-but-valid render).\n");
        ctx_pos = dummy({4096, 512});
    }
    sd::Tensor<float> ctx_neg;
    if (!o.neg_context.empty()) {
        ctx_neg = sd::load_tensor_from_file_as_tensor<float>(o.neg_context);
    } else {
        ctx_neg = sd::Tensor<float>(ctx_pos.shape());  // zeros_like
        for (int64_t i = 0; i < ctx_neg.numel(); ++i) ctx_neg.data()[i] = 0.0f;
    }
    // AUDIO uncond context. PyTorch (pipeline_nava.py negative_prompt_mode=True) uses
    // a SEPARATE audio negative prompt ("机械音…") for the audio stream's CFG uncond,
    // NOT the video neg-prompt. Using the wrong audio uncond corrupts the co-denoised
    // audio velocity, which feeds back through the joint self-attention every step and
    // makes the video diverge. Fall back to the video neg only if not supplied.
    sd::Tensor<float> ctx_audio_neg;
    const bool have_audio_neg = !o.audio_neg_context.empty();
    if (have_audio_neg) {
        ctx_audio_neg = sd::load_tensor_from_file_as_tensor<float>(o.audio_neg_context);
        printf("audio uncond uses SEPARATE audio-neg context: %s\n", o.audio_neg_context.c_str());
    } else {
        printf("NOTE: no --audio-neg-context => audio uncond reuses the VIDEO neg context.\n");
    }

    // ----- Voice clone: load the speaker embedding + splice positions -----
    // spk_emb_host is the ReDimNet [192] embedding (precomputed offline). spk_pos_v are
    // the <extra_id_2> token indices in the 512-len context (from --spk-pos or the
    // "<context>.spkpos" sidecar written by encode-prompt). Clone is active iff BOTH set.
    sd::Tensor<float> spk_emb_host;
    std::vector<int> spk_pos_v;
    if (!o.spk_emb.empty()) {
        if (!path_exists(o.spk_emb)) { printf("ERROR: --spk-emb %s not found\n", o.spk_emb.c_str()); return 1; }
        spk_emb_host = sd::load_tensor_from_file_as_tensor<float>(o.spk_emb);
        spk_emb_host.reshape_({192, 1});
        std::string poss = o.spk_pos_str;
        if (poss.empty() && have_inline_ctx) {  // inline encode resolved spk pos in-process
            spk_pos_v = inline_spk_pos;
            printf("spk_pos: %zu position(s) from inline umT5 encode\n", spk_pos_v.size());
        } else if (poss.empty()) {  // fall back to the encode-prompt sidecar
            std::string sp = o.context + ".spkpos";
            FILE* sf = fopen(sp.c_str(), "r");
            if (sf) { char ln[32]; while (fgets(ln, sizeof(ln), sf)) { int v = atoi(ln); if (ln[0]>='0'&&ln[0]<='9') spk_pos_v.push_back(v); } fclose(sf);
                      printf("spk_pos: loaded %zu position(s) from %s\n", spk_pos_v.size(), sp.c_str()); }
        } else {
            size_t p = 0; while (p < poss.size()) { size_t c = poss.find(',', p); spk_pos_v.push_back(atoi(poss.substr(p, c==std::string::npos?c:c-p).c_str())); if (c==std::string::npos) break; p = c+1; }
        }
        if (spk_pos_v.empty()) { printf("WARNING: --spk-emb given but no spk_pos resolved => clone DISABLED.\n"); }
    }
    const bool clone = !spk_emb_host.empty() && !spk_pos_v.empty();
    const bool effective_timbre = clone && o.timbre_cfg > 0.0f;
    if (clone) {
        printf("VOICE CLONE: spk_emb=%s, spk_pos=[", o.spk_emb.c_str());
        for (size_t i = 0; i < spk_pos_v.size(); ++i) printf("%s%d", i?",":"", spk_pos_v[i]);
        printf("], timbre_cfg=%.2f%s\n", o.timbre_cfg, effective_timbre ? " (active)" : "");
    }

    // ----- seeded noise in VAE-latent space [W_lat, H_lat, F, 48] -----
    PhiloxRNG rng(o.seed);
    sd::Tensor<float> latent({(int64_t)W_lat, (int64_t)H_lat, (int64_t)f_len, 48});
    {
        auto r = rng.randn((uint32_t)latent.numel());
        for (int64_t i = 0; i < latent.numel(); ++i) latent.data()[i] = r[(size_t)i];
    }
    // Audio latent: proper std-1 Gaussian noise (NOT a near-zero dummy), so the
    // joint self-attention sees an in-distribution audio stream. We still render
    // silent (the audio velocity is discarded here; joint denoise is the next step
    // if this proves the joint forward was being poisoned by the degenerate dummy).
    sd::Tensor<float> audio_latent({128, (int64_t)audio_len});
    {
        auto r = rng.randn((uint32_t)audio_latent.numel());
        for (int64_t i = 0; i < audio_latent.numel(); ++i) audio_latent.data()[i] = r[(size_t)i];
    }
    bool injected_video_init = false;
    if (const char* p = getenv("NAVA_INJECT_VIDEO")) {
        auto inj = sd::load_tensor_from_file_as_tensor<float>(p);
        if (inj.shape() != latent.shape()) {
            printf("ERROR: NAVA_INJECT_VIDEO ne %s != latent %s\n",
                   sd::tensor_shape_to_string(inj.shape()).c_str(),
                   sd::tensor_shape_to_string(latent.shape()).c_str());
            return 1;
        }
        latent = std::move(inj);
        injected_video_init = true;
        printf("NAVA_INJECT_VIDEO: sampler video init <- %s\n", p);
    }
    if (const char* p = getenv("NAVA_INJECT_AUDIO")) {
        auto inj = sd::load_tensor_from_file_as_tensor<float>(p);
        if (inj.shape() != audio_latent.shape()) {
            printf("ERROR: NAVA_INJECT_AUDIO ne %s != audio_latent %s\n",
                   sd::tensor_shape_to_string(inj.shape()).c_str(),
                   sd::tensor_shape_to_string(audio_latent.shape()).c_str());
            return 1;
        }
        audio_latent = std::move(inj);
        printf("NAVA_INJECT_AUDIO: sampler audio init <- %s\n", p);
    }

    // I2V: token bookkeeping + splice the clean anchor into frame 0. The first
    // h_grid*w_grid VIDEO tokens (frame 0) carry timestep 0 every step (clean anchor)
    // and frame 0's latent is re-pinned to the image after every Euler step.
    const int h_grid_i      = H_lat / 2, w_grid_i = W_lat / 2;
    const int64_t L_vid_i   = (int64_t)f_len * h_grid_i * w_grid_i;
    const int64_t n_clean_i = (int64_t)n_anchor_v * h_grid_i * w_grid_i;  // clean VIDEO tokens (N frames)
    // Pin clean anchors into a tensor. Python's predict_eps() does this as a
    // per-forward input substitution for I2V: the model sees frame 0 clean, but
    // latents_vision itself is not overwritten after scheduler.step().
    auto splice_anchor_into = [&](sd::Tensor<float>* vlat, sd::Tensor<float>* alat) {
        if (vlat && n_anchor_v > 0) {
            const int64_t W = W_lat, H = H_lat, F = f_len;  // latent ne [W,H,F,48]; anchor [W,H,N,48]
            for (int64_t f = 0; f < n_anchor_v; ++f)
                for (int64_t c = 0; c < 48; ++c)
                    for (int64_t y = 0; y < H; ++y)
                        for (int64_t x = 0; x < W; ++x)
                            vlat->data()[x + W * (y + H * (f + F * c))] =
                                vid_anchor.data()[x + W * (y + H * (f + (int64_t)n_anchor_v * c))];
        }
        if (alat && n_anchor_a > 0) {  // audio_latent ne [128,audio_len]; aud_anchor [128,K]
            for (int64_t k = 0; k < n_anchor_a; ++k)
                for (int64_t c = 0; c < 128; ++c)
                    alat->data()[k * 128 + c] = aud_anchor.data()[k * 128 + c];
        }
    };
    const bool model_input_anchor = getenv("NAVA_MODEL_INPUT_ANCHOR") != nullptr;
    // Default generation keeps the historical C++ behavior: the clean anchor is
    // part of sampler state and is re-pinned after each update. The Python-style
    // per-forward substitution diagnostic is available with NAVA_MODEL_INPUT_ANCHOR=1.
    // Python i2v keeps the sampler state as free noise and substitutes the clean
    // first frame only in predict_eps(). Keep the historical C++ repinned-state
    // path by default, but make NAVA_MODEL_INPUT_ANCHOR=1 match Python exactly.
    splice_anchor_into(model_input_anchor ? nullptr : &latent, &audio_latent);
    const bool repin_after_step = !model_input_anchor || getenv("NAVA_REPIN_AFTER_STEP") != nullptr;

    // ----- Euler flow-match sampling (JOINT video+audio denoise) -----
    // NAVA is a joint AV MMDiT: every block self-attends over [video ++ audio].
    // We co-denoise the audio stream in lockstep so the video tokens always attend
    // to an audio stream at the matching noise level (a frozen / degenerate audio
    // stream poisons the joint attention and the video never coheres).
    const float cfg_audio = 2.0f;  // nava_run.yaml audio_guidance_scale
    // align_3d_cfg (nava_640.yaml): a 3rd "masking_modality" forward (separate
    // intra-modal attention) adds an alignment guidance term to BOTH streams. This
    // is what locks audio content to the video on hard prompts — its omission was
    // the audio->noise divergence. Default ON to match python; NAVA_NO_ALIGN_CFG=1
    // reverts to the old 2-way CFG for A/B.
    const float cfg_video_align = o.cfg_align;        // video_align_guidance_scale
    const float cfg_audio_align = o.cfg_align_audio;  // audio_align_guidance_scale
    const bool  align_cfg       = getenv("NAVA_NO_ALIGN_CFG") == nullptr;
    FlowMatchSched sched;
    sched.set_timesteps(o.steps, o.shift);
    // UniPC multistep solver (validated vs PyTorch FlowUniPCMultistepScheduler) is the
    // DEFAULT: it matches python prod (nava_run.yaml scheduler_unipc=true). Its higher-order
    // corrector tames the big low-sigma Euler steps that otherwise leave residual noise/"fuzz"
    // at low step counts (and can run the std away). Opt out with NAVA_EULER=1 (plain Euler).
    const bool use_unipc = getenv("NAVA_EULER") == nullptr;
    // Per-stream sampler: the AUDIO stream can use a DIFFERENT solver than video.
    // UniPC's higher-order corrector cleans up the VIDEO fuzz, but on the audio stream
    // it can run aud_std away (-> the voice diverges to noise on hard prompts). Set
    // NAVA_AUDIO_EULER=1 to keep video on UniPC but denoise audio with plain Euler.
    const bool use_unipc_audio = use_unipc && getenv("NAVA_AUDIO_EULER") == nullptr;
    UniPCSched usched_v, usched_a;
    if (use_unipc) {
        usched_v.set_timesteps(o.steps, o.shift);
        usched_a.set_timesteps(o.steps, o.shift);
    }
    printf("sampler: video=%s  audio=%s\n",
           use_unipc ? "UniPC" : "Euler",
           use_unipc_audio ? "UniPC" : "Euler");

    // ----- WARM-START (SDEdit) video init — continuation seam softener (Track 1) -----
    // Pinning a clean I-frame anchor (--image) resets the leading frame to std~0.5,
    // wiping the motion velocity the full-sequence encode held (std~0.88) → a hard cut.
    // Instead we seed the WHOLE video latent from the prior segment's velocity-carrying
    // tail latent and denoise the video stream from a partial sigma (img2img/strength).
    // The continuity-vs-LIVENESS tension (measured by eye): uniformly warm-starting EVERY
    // frame from the prior latent kills the model's free micro-motion (blinks, cheeks) →
    // "dead-eyed" + jitter. M=1 stays alive because its post-anchor frames denoise from PURE
    // NOISE (full freedom). GRADED fix: warm ONLY the OVERLAP region (first O video frames /
    // Na audio tokens) to carry the prior MOTION across the seam, and leave the BODY frames as
    // pure noise on the full schedule (free → lively). The overlap is SOFT-ANCHORED: held at
    // sigma_held (a partial-noise SDEdit level, repinned each step) until the schedule's sigma
    // descends to it (video_start_step), then released to denoise with the body. The model
    // propagates the overlap motion into the free body via its own temporal attention. Both
    // streams share sigma_held so the joint attention never desyncs (no broken audio). The
    // overlap frames reproduce the prior tail, so they're dropped in the concat → seam flows.
    int   video_start_step = 0;     // step at which the soft-anchored overlap releases
    bool  warm_active      = false;
    float sigma_held       = 0.0f;
    int   warm_ov_v        = 0;     // # soft-anchored video overlap frames
    int   warm_ov_a        = 0;     // # soft-anchored audio overlap tokens
    std::vector<float> warm_pin_v;  // saved warm-init for the overlap video frames (repin)
    std::vector<float> warm_pin_a;  // saved warm-init for the overlap audio tokens (repin)
    if (!o.warm_latent.empty() && o.warm_strength > 0.0f) {
        if (!path_exists(o.warm_latent)) { printf("ERROR: --warm-latent %s not found\n", o.warm_latent.c_str()); return 1; }
        auto wl = sd::load_tensor_from_file_as_tensor<float>(o.warm_latent);  // [W,H,Kf,48] diffusion latent
        if (wl.dim() < 4 || wl.shape()[0] != (int64_t)W_lat || wl.shape()[1] != (int64_t)H_lat
            || wl.shape()[3] != 48) {
            printf("ERROR: --warm-latent ne %s != [%d,%d,Kf,48]\n",
                   sd::tensor_shape_to_string(wl.shape()).c_str(), W_lat, H_lat);
            return 1;
        }
        const int64_t Kf = wl.shape()[2];
        // start step = first schedule step whose sigma <= the requested strength.
        const std::vector<float>& sig = use_unipc ? usched_v.sigmas : sched.sigmas;
        int ss = 0;
        while (ss < o.steps && sig[(size_t)ss] > o.warm_strength) ss++;
        if (ss >= o.steps) ss = o.steps - 1;  // leave >=1 release step
        video_start_step = ss;
        sigma_held       = sig[(size_t)ss];
        int O = o.warm_overlap;
        if (O < 1) O = 1;
        if (O > (int)Kf) O = (int)Kf;
        if (O > f_len) O = f_len;
        warm_ov_v = O;
        const int64_t WH = (int64_t)W_lat * H_lat;
        warm_pin_v.resize((size_t)O * 48 * WH);
        // Seed overlap frames [0,O) from the prior tail SEQUENCE [Kf-O, Kf) (carries the real
        // frame-to-frame velocity); body frames [O, f_len) are LEFT as pure noise (free).
        for (int64_t c = 0; c < 48; ++c)
            for (int64_t f = 0; f < O; ++f)
                for (int64_t p = 0; p < WH; ++p) {
                    const float   x0  = wl.data()[p + WH * ((Kf - (int64_t)O + f) + Kf * c)];
                    const int64_t di  = p + WH * (f + (int64_t)f_len * c);
                    const float   val = (1.0f - sigma_held) * x0 + sigma_held * latent.data()[di];
                    latent.data()[di] = val;
                    warm_pin_v[((size_t)f * 48 + c) * WH + p] = val;
                }
        warm_active = true;
        printf("WARM-START (graded): strength=%.3f start_step=%d/%d sigma_held=%.4f overlap=%d/%d Kf=%lld (body=free noise)\n",
               o.warm_strength, video_start_step, o.steps, sigma_held, O, f_len, (long long)Kf);
        dump_stats("warm-start video init", latent);
    }
    // Audio overlap: warm the first Na audio tokens from the prior tail audio latent at the
    // SAME sigma_held (continuity), soft-anchored like the video overlap; the rest is free
    // noise → new speech stays lively. Na ~ the video overlap in TIME (25 audio tok/s).
    if (warm_active && !o.warm_audio.empty()) {
        if (!path_exists(o.warm_audio)) { printf("ERROR: --warm-audio %s not found\n", o.warm_audio.c_str()); return 1; }
        auto wa = sd::load_tensor_from_file_as_tensor<float>(o.warm_audio);  // [128, Ka]
        if (wa.shape()[0] != 128) {
            printf("ERROR: --warm-audio ne %s != [128,Ka]\n", sd::tensor_shape_to_string(wa.shape()).c_str());
            return 1;
        }
        const int64_t Ka = wa.shape()[1];
        int Na = (int)llround((double)warm_ov_v * (double)audio_len / (double)f_len);
        if (Na < 1) Na = 1;
        if (Na > audio_len) Na = audio_len;
        warm_ov_a = Na;
        warm_pin_a.resize((size_t)Na * 128);
        for (int64_t t = 0; t < Na; ++t) {
            const int64_t st = (t < Ka) ? t : (Ka - 1);  // align to prior tail; clamp at its end
            for (int64_t c = 0; c < 128; ++c) {
                const float a0  = wa.data()[st * 128 + c];
                const float val = (1.0f - sigma_held) * a0 + sigma_held * audio_latent.data()[t * 128 + c];
                audio_latent.data()[t * 128 + c] = val;
                warm_pin_a[(size_t)t * 128 + c]  = val;
            }
        }
        printf("WARM-START AUDIO (graded): overlap=%d/%d tokens, Ka=%lld (rest=free noise)\n",
               Na, audio_len, (long long)Ka);
        dump_stats("warm-start audio init", audio_latent);
    }
    // Per-token timestep needed whenever tokens sit at different noise levels: i2v clean
    // anchors (t=0) OR the soft-anchored warm overlap (held at sigma_held until release).
    const bool per_token_ts = i2v || warm_active;

    // Optional per-step trajectory dump for cpp-vs-PyTorch sampler faithfulness probe.
    const char* dump_traj = getenv("NAVA_DUMP_TRAJ");
    const char* dump_audio_traj = getenv("NAVA_DUMP_AUDIO_TRAJ");
    if (dump_traj) {
        std::string d = dump_traj;
        mkdirs(d);
        write_bin(d + "/vid_noise.bin", latent, "vid_noise");
        write_bin(d + "/aud_noise.bin", audio_latent, "aud_noise");
        printf("NAVA_DUMP_TRAJ active -> %s\n", d.c_str());
    }
    if (dump_audio_traj) {
        std::string d = dump_audio_traj;
        mkdirs(d);
        write_bin(d + "/aud_noise.bin", audio_latent, "aud_noise");
        printf("NAVA_DUMP_AUDIO_TRAJ active -> %s\n", d.c_str());
    }

    // ---- NAVA step cache (TeaCache/EasyCache-style velocity reuse) ----
    // The cond velocity is extremely stable step-to-step (measured: cos>0.999 in the
    // mid-trajectory). On a "stable" step we SKIP all 3 DiT forwards and reuse the last
    // computed velocities. Skip is predicted from the relative latent change times the
    // learned input->output transformation rate (EasyCache), accumulated until it crosses
    // NAVA_CACHE_THRESH. Active only in [start,end] fraction of the schedule. Disabled by
    // default (thresh 0); the timbre 4th-forward path disables it. Tunables:
    //   NAVA_CACHE_THRESH (e.g. 0.15) NAVA_CACHE_START (0.2) NAVA_CACHE_END (0.95)
    const float cache_thresh = getenv("NAVA_CACHE_THRESH") ? atof(getenv("NAVA_CACHE_THRESH")) : 0.0f;
    const float cache_start  = getenv("NAVA_CACHE_START") ? atof(getenv("NAVA_CACHE_START")) : 0.20f;
    const float cache_end    = getenv("NAVA_CACHE_END") ? atof(getenv("NAVA_CACHE_END")) : 0.95f;
    const bool  cache_on     = cache_thresh > 0.0f && !effective_timbre &&
                               getenv("NAVA_SEPARATE_AUDIO_NEG") == nullptr;
    // AUDIO-AWARE veto (NAVA_CACHE_AUDIO_AWARE=1): the base EasyCache skip predictor is
    // VIDEO-only (latent + vv_cond), so it skips steps that are stable for video but NOT
    // for the faster-moving audio stream → reuses a stale audio velocity → audio refinement
    // loss (the 25-step audio regresses toward 10-step "off tone"). With this on, a SECOND
    // accumulator tracks the audio stream (audio_latent + va_cond) and a step is skipped
    // only when BOTH streams are predicted stable — audio instability vetoes the skip.
    const bool  cache_audio_aware = getenv("NAVA_CACHE_AUDIO_AWARE") != nullptr;
    sd::Tensor<float> c_vv_cond, c_va_cond, c_vv_uncond, c_va_uncond, c_vv_mmask, c_va_mmask;
    std::vector<float> cache_prev_latent, cache_prev_vvcond;
    std::vector<float> cache_prev_alatent, cache_prev_vacond;  // audio-aware
    float cache_rate = 0.0f, cache_accum = 0.0f;
    float cache_rate_a = 0.0f, cache_accum_a = 0.0f;           // audio-aware
    bool cache_have = false, cache_have_rate = false;
    bool cache_have_rate_a = false;                            // audio-aware
    int  cache_skipped = 0;

    int64_t sample_t0 = ggml_time_ms();
    for (int step = 0; step < o.steps; ++step) {
        // Python's FlowUniPCMultistepScheduler exposes int64 timesteps, but the
        // existing C++ trajectory is measurably closer with the float sigma*1000
        // value. Keep float timesteps by default; NAVA_UNIPC_INT_TIMESTEP=1
        // enables the strict Python-source probe path.
        const float sampler_t = use_unipc ? usched_v.timesteps[step] : sched.timesteps[step];
        const float sampler_sigma = use_unipc ? usched_v.sigmas[step] : sched.sigmas[step];
        const bool audio_uses_unipc_schedule = use_unipc_audio || use_unipc;
        const float audio_sigma = audio_uses_unipc_schedule ? usched_a.sigmas[step] : sched.sigmas[step];
        const float audio_sigma_next = audio_uses_unipc_schedule ? usched_a.sigmas[step + 1]
                                                                 : sched.sigma_next(step);
        const bool use_int_timestep = use_unipc && getenv("NAVA_UNIPC_INT_TIMESTEP") != nullptr;
        const float tval = use_int_timestep ? static_cast<float>(static_cast<int64_t>(sampler_t))
                                            : sampler_t;
        // Graded warm-start: the soft-anchored overlap (first warm_ov_v video frames /
        // warm_ov_a audio tokens) is HELD at sigma_held until the schedule sigma descends to
        // it (step < video_start_step) → those tokens carry the held timestep; everything else
        // (clean anchors = 0, free body = live tval) carries its own level.
        const float held_t      = sigma_held * 1000.0f;
        const bool  ov_held     = warm_active && (step < video_start_step);
        const int64_t tpf       = (int64_t)h_grid_i * w_grid_i;  // video tokens per frame
        // I2V / warm-start: per-token timestep. T2V: scalar timestep (uniform-t).
        // Validated vs PyTorch (blocks 100-120 dB).
        sd::Tensor<float> ts;
        if (per_token_ts) {
            const int64_t L_total = L_vid_i + audio_len;
            ts = sd::Tensor<float>({L_total});
            for (int64_t i = 0; i < L_vid_i; ++i) {
                const int64_t f = (tpf > 0) ? (i / tpf) : 0;
                ts.data()[i] = (i < n_clean_i)                         ? 0.0f      // clean anchor
                             : (ov_held && f < warm_ov_v)             ? held_t    // soft-anchored overlap
                                                                       : tval;     // free body
            }
            for (int64_t j = 0; j < audio_len; ++j)
                ts.data()[L_vid_i + j] = (j < (int64_t)n_anchor_a)        ? 0.0f
                                       : (ov_held && j < warm_ov_a)       ? held_t
                                                                          : tval;
        } else {
            ts = sd::Tensor<float>({1});
            ts.data()[0] = tval;
        }

        // Optional diagnostic: Python predict_eps() substitutes the I2V first
        // frame into the model input each forward, while scheduler.step() still
        // receives the evolving latents_vision sample.
        sd::Tensor<float> latent_model;
        sd::Tensor<float> audio_model;
        const sd::Tensor<float>* latent_in = &latent;
        const sd::Tensor<float>* audio_in  = &audio_latent;
        if (i2v && model_input_anchor) {
            latent_model = latent;
            audio_model  = audio_latent;
            splice_anchor_into(&latent_model, &audio_model);
            latent_in = &latent_model;
            audio_in  = &audio_model;
        }

        // ---- step-cache skip decision (EasyCache: predict from latent change x rate) ----
        bool did_skip = false;
        if (cache_on && cache_have && cache_have_rate &&
            (int64_t) cache_prev_latent.size() == latent.numel()) {
            const float progress = (float) step / (float) o.steps;
            if (progress >= cache_start && progress <= cache_end) {
                double dch = 0.0, nrm = 0.0;
                const float* ld = latent.data();
                for (int64_t i = 0; i < latent.numel(); ++i) { dch += std::fabs(ld[i] - cache_prev_latent[i]); nrm += std::fabs(ld[i]); }
                const float rel = nrm > 0.0 ? (float) (dch / nrm) : 0.0f;
                cache_accum += cache_rate * rel;
                bool vid_stable = cache_accum < cache_thresh;
                // audio-aware: a second accumulator on the audio stream must ALSO be stable
                bool aud_stable = true;
                if (cache_audio_aware && cache_have_rate_a &&
                    (int64_t) cache_prev_alatent.size() == audio_latent.numel()) {
                    double ach = 0.0, anrm = 0.0;
                    const float* ad = audio_latent.data();
                    for (int64_t i = 0; i < audio_latent.numel(); ++i) { ach += std::fabs(ad[i] - cache_prev_alatent[i]); anrm += std::fabs(ad[i]); }
                    const float arel = anrm > 0.0 ? (float) (ach / anrm) : 0.0f;
                    cache_accum_a += cache_rate_a * arel;
                    aud_stable = cache_accum_a < cache_thresh;
                }
                if (vid_stable && aud_stable) did_skip = true;
                else { cache_accum = 0.0f; cache_accum_a = 0.0f; }
            }
        }

        // cond + uncond JOINT forwards -> both stream velocities each.
        // COND: when cloning, splice the projected speaker embedding into the shared
        // (spk-selected, model_mm.py:1644) context. UNCOND is always spk=None (the python
        // uncond pass passes spk_embs=None -> context_vid(neg) for both streams).
        sd::Tensor<float> vv_cond, va_cond, vv_uncond, va_uncond;
        if (did_skip) {
            vv_cond = c_vv_cond; va_cond = c_va_cond; vv_uncond = c_vv_uncond; va_uncond = c_va_uncond;
            cache_skipped++;
        } else {
            std::tie(vv_cond, va_cond) = runner->compute_va(n_threads, *latent_in, *audio_in, ctx_pos, ts,
                                                            /*mask_modality=*/false, /*apply_spk=*/clone, &spk_emb_host, &spk_pos_v);
            if (vv_cond.empty()) {
                printf("ERROR: DiT forward returned empty at step %d\n", step);
                return 1;
            }
            if (step == 0 && getenv("NAVA_STOP_AFTER_STEP0_COND") != nullptr) {
                if (dump_audio_traj) {
                    const std::string d = dump_audio_traj;
                    write_bin(d + "/va_cond_00.bin", va_cond, "va_cond");
                }
                printf("NAVA_STOP_AFTER_STEP0_COND active -> stopped after first conditional DiT forward\n");
                return 0;
            }
            std::tie(vv_uncond, va_uncond) = runner->compute_va(n_threads, *latent_in, *audio_in, ctx_neg, ts);
            if (vv_uncond.empty()) {
                printf("ERROR: DiT forward returned empty at step %d\n", step);
                return 1;
            }
        }
        // AUDIO uncond context. PyTorch's MMDiT merges to a SINGLE shared context for
        // both streams: when spk_embed is None (our case — spk is stubbed), the merge
        // picks context_VID, so the audio stream's uncond uses the VIDEO negative and
        // the audio negative is DISCARDED (model_mm.py:1644, uncond pass spk=None at
        // pipeline_nava.py:491). So by default we reuse va_uncond (video-neg joint pass)
        // for audio — matching python — and SKIP the separate audio-neg forward.
        // NAVA_SEPARATE_AUDIO_NEG=1 restores the old behaviour (correct only once a real
        // speaker embed is wired, where the merge would pick context_audio).
        static const bool separate_audio_neg =
            have_audio_neg && getenv("NAVA_SEPARATE_AUDIO_NEG") != nullptr;
        sd::Tensor<float> va_audio_uncond;
        if (separate_audio_neg) {
            auto [vv_audio_unc, va_audio_unc] =
                runner->compute_va(n_threads, *latent_in, *audio_in, ctx_audio_neg, ts);
            (void)vv_audio_unc;
            va_audio_uncond = std::move(va_audio_unc);
        }
        const sd::Tensor<float>& va_uncond_for_audio =
            separate_audio_neg ? va_audio_uncond : va_uncond;
        const float dsig_video = use_unipc ? usched_v.sigmas[step + 1] - usched_v.sigmas[step]
                                           : sched.sigma_next(step) - sched.sigmas[step];
        const float dsig_audio = audio_sigma_next - audio_sigma;

        // align_3d_cfg: 3rd forward with masking_modality=true (separate intra-modal
        // attention, same POSITIVE context) -> alignment guidance anchor for both streams.
        sd::Tensor<float> vv_mmask, va_mmask;
        if (align_cfg) {
            if (did_skip) {
                vv_mmask = c_vv_mmask; va_mmask = c_va_mmask;
            } else {
                std::tie(vv_mmask, va_mmask) =
                    runner->compute_va(n_threads, *latent_in, *audio_in, ctx_pos, ts, /*mask_modality=*/true,
                                       /*apply_spk=*/clone, &spk_emb_host, &spk_pos_v);
            }
        }

        // ---- update step cache (computed steps only) + learn the transformation rate ----
        if (cache_on && !did_skip) {
            if (cache_have && !cache_prev_vvcond.empty() &&
                (int64_t) cache_prev_vvcond.size() == vv_cond.numel() &&
                (int64_t) cache_prev_latent.size() == latent.numel()) {
                double vch = 0, vnrm = 0; const float* vd = vv_cond.data();
                for (int64_t i = 0; i < vv_cond.numel(); ++i) { vch += std::fabs(vd[i] - cache_prev_vvcond[i]); vnrm += std::fabs(vd[i]); }
                double lch = 0, lnrm = 0; const float* ld = latent.data();
                for (int64_t i = 0; i < latent.numel(); ++i) { lch += std::fabs(ld[i] - cache_prev_latent[i]); lnrm += std::fabs(ld[i]); }
                const float vrel = vnrm > 0 ? (float) (vch / vnrm) : 0.0f;
                const float lrel = lnrm > 0 ? (float) (lch / lnrm) : 0.0f;
                if (lrel > 1e-9f) { cache_rate = vrel / lrel; cache_have_rate = true; }
            }
            // audio-aware: learn the audio transformation rate (va_cond change / audio_latent change)
            if (cache_audio_aware && cache_have && !cache_prev_vacond.empty() &&
                (int64_t) cache_prev_vacond.size() == va_cond.numel() &&
                (int64_t) cache_prev_alatent.size() == audio_latent.numel()) {
                double ach = 0, anrm = 0; const float* ad = va_cond.data();
                for (int64_t i = 0; i < va_cond.numel(); ++i) { ach += std::fabs(ad[i] - cache_prev_vacond[i]); anrm += std::fabs(ad[i]); }
                double alch = 0, alnrm = 0; const float* al = audio_latent.data();
                for (int64_t i = 0; i < audio_latent.numel(); ++i) { alch += std::fabs(al[i] - cache_prev_alatent[i]); alnrm += std::fabs(al[i]); }
                const float arel  = anrm > 0 ? (float) (ach / anrm) : 0.0f;
                const float alrel = alnrm > 0 ? (float) (alch / alnrm) : 0.0f;
                if (alrel > 1e-9f) { cache_rate_a = arel / alrel; cache_have_rate_a = true; }
            }
            c_vv_cond = vv_cond; c_va_cond = va_cond;
            c_vv_uncond = vv_uncond; c_va_uncond = va_uncond;
            c_vv_mmask = vv_mmask; c_va_mmask = va_mmask;
            cache_have = true;
            cache_prev_vvcond.assign(vv_cond.data(), vv_cond.data() + vv_cond.numel());
            cache_prev_latent.assign(latent.data(), latent.data() + latent.numel());
            cache_prev_vacond.assign(va_cond.data(), va_cond.data() + va_cond.numel());
            cache_prev_alatent.assign(audio_latent.data(), audio_latent.data() + audio_latent.numel());
            cache_accum = 0.0f;
            cache_accum_a = 0.0f;
        }
        // timbre CFG: a 4th AUDIO forward with spk=None on the POSITIVE context
        // (pipeline_nava.py eps_timbre_uncond). The term g_timbre*(cond - timbre_uncond)
        // amplifies the speaker's contribution. Audio-only (video timbre term is commented
        // out in python). Only its audio velocity is used.
        sd::Tensor<float> va_timbre;
        if (effective_timbre) {
            sd::Tensor<float> vv_tmp;
            std::tie(vv_tmp, va_timbre) =
                runner->compute_va(n_threads, *latent_in, *audio_in, ctx_pos, ts, /*mask_modality=*/false);
            (void)vv_tmp;
        }
        if (dump_audio_traj) {
            char fn[96];
            const std::string d = dump_audio_traj;
            snprintf(fn, sizeof(fn), "/va_cond_%02d.bin", step);
            write_bin(d + fn, va_cond, "va_cond");
            snprintf(fn, sizeof(fn), "/va_uncond_%02d.bin", step);
            write_bin(d + fn, va_uncond_for_audio, "va_uncond");
            if (!va_mmask.empty()) {
                snprintf(fn, sizeof(fn), "/va_mmask_%02d.bin", step);
                write_bin(d + fn, va_mmask, "va_mmask");
            }
        }

        // --- video: CFG combine (patched) -> unpatchify -> Euler step ---
        // align on : eps = ec + g*(ec-eu) + ga*(ec-emmask)   (cond base, 3-term)
        // align off: eps = eu + g*(ec-eu)                     (legacy 2-term)
        sd::Tensor<float> v_p(vv_cond.shape());
        for (int64_t i = 0; i < v_p.numel(); ++i) {
            float vc = vv_cond.data()[i], vu = vv_uncond.data()[i];
            if (align_cfg && !vv_mmask.empty()) {
                float vm      = vv_mmask.data()[i];
                v_p.data()[i] = vc + o.cfg * (vc - vu) + cfg_video_align * (vc - vm);
            } else {
                v_p.data()[i] = vu + o.cfg * (vc - vu);
            }
        }
        auto v = unpatchify_video(v_p, f_len, h_grid, w_grid);
        if (dump_traj) {
            char fn[80];
            snprintf(fn, sizeof(fn), "/vel_vid_cfg_%02d.bin", step);
            write_bin(std::string(dump_traj) + fn, v, "vel_vid_cfg");
            snprintf(fn, sizeof(fn), "/vel_vid_cond_%02d.bin", step);
            write_bin(std::string(dump_traj) + fn, vv_cond, "vel_vid_cond");
            snprintf(fn, sizeof(fn), "/vel_vid_uncond_%02d.bin", step);
            write_bin(std::string(dump_traj) + fn, vv_uncond, "vel_vid_uncond");
        }
        // Body + overlap denoise every step (full schedule). The soft-anchored overlap is
        // repinned below while still held, so its UniPC update here is discarded each early
        // step — same mechanism as the i2v clean-anchor repin.
        if (use_unipc) {
            latent = usched_v.step(v, latent);  // x0-predict + multistep corrector
        } else {
            for (int64_t i = 0; i < latent.numel(); ++i) {
                latent.data()[i] += v.data()[i] * dsig_video;
            }
        }
        if (repin_after_step) {
            splice_anchor_into(&latent, &audio_latent);
        }
        // Graded warm-start: re-pin the soft-anchored video overlap to its held warm-init
        // until release (keeps frames [0,warm_ov_v) at sigma_held while the body denoises).
        if (warm_active && ov_held && warm_ov_v > 0) {
            const int64_t WH = (int64_t)W_lat * H_lat;
            for (int64_t f = 0; f < warm_ov_v; ++f)
                for (int64_t c = 0; c < 48; ++c)
                    for (int64_t p = 0; p < WH; ++p)
                        latent.data()[p + WH * (f + (int64_t)f_len * c)] =
                            warm_pin_v[((size_t)f * 48 + c) * WH + p];
        }

        // --- audio: CFG combine -> Euler step (same schedule/shift -> same dsig) ---
        static const bool freeze_audio = getenv("NAVA_FREEZE_AUDIO") != nullptr;
        if (!freeze_audio && !va_cond.empty() && va_cond.numel() == audio_latent.numel()
            && va_uncond_for_audio.numel() == audio_latent.numel()) {
            sd::Tensor<float> va_cfg(audio_latent.shape());
            for (int64_t i = 0; i < audio_latent.numel(); ++i) {
                float ac = va_cond.data()[i], au = va_uncond_for_audio.data()[i];
                float eps;
                if (align_cfg && !va_mmask.empty() && va_mmask.numel() == audio_latent.numel()) {
                    float am = va_mmask.data()[i];
                    eps = ac + cfg_audio * (ac - au) + cfg_audio_align * (ac - am);  // cond-base 3-term
                } else if (effective_timbre) {
                    eps = ac + cfg_audio * (ac - au);                                // cond-base 2-term (timbre, no align)
                } else {
                    eps = au + cfg_audio * (ac - au);                                // legacy uncond-base 2-term
                }
                // timbre term (audio only): + g_timbre*(cond - timbre_uncond)
                if (effective_timbre && !va_timbre.empty() && va_timbre.numel() == audio_latent.numel()) {
                    eps += o.timbre_cfg * (ac - va_timbre.data()[i]);
                }
                va_cfg.data()[i] = eps;
            }
            if (dump_traj) {
                char fn[80];
                snprintf(fn, sizeof(fn), "/va_cfg_%02d.bin", step);
                write_bin(std::string(dump_traj) + fn, va_cfg, "va_cfg");
                snprintf(fn, sizeof(fn), "/va_cond_%02d.bin", step);
                write_bin(std::string(dump_traj) + fn, va_cond, "va_cond");
            }
            if (dump_audio_traj) {
                char fn[96];
                const std::string d = dump_audio_traj;
                snprintf(fn, sizeof(fn), "/va_cfg_%02d.bin", step);
                write_bin(d + fn, va_cfg, "va_cfg");
            }
            if (use_unipc_audio) {
                audio_latent = usched_a.step(va_cfg, audio_latent);
            } else {
                for (int64_t i = 0; i < audio_latent.numel(); ++i) {
                    audio_latent.data()[i] += va_cfg.data()[i] * dsig_audio;
                }
            }
            if (repin_after_step) {
                splice_anchor_into(&latent, &audio_latent);
            }
            // Graded warm-start: re-pin the soft-anchored audio overlap to its held warm-init
            // until release (continuity handoff) while the rest of the speech denoises free.
            if (warm_active && ov_held && warm_ov_a > 0) {
                for (int64_t t = 0; t < warm_ov_a; ++t)
                    for (int64_t c = 0; c < 128; ++c)
                        audio_latent.data()[t * 128 + c] = warm_pin_a[(size_t)t * 128 + c];
            }
            if (dump_traj) {
                char fn[80];
                snprintf(fn, sizeof(fn), "/aud_step_%02d.bin", step);
                write_bin(std::string(dump_traj) + fn, audio_latent, "aud_step");
            }
            if (dump_audio_traj) {
                char fn[96];
                snprintf(fn, sizeof(fn), "/aud_step_%02d.bin", step);
                write_bin(std::string(dump_audio_traj) + fn, audio_latent, "aud_step");
            }
        }
        double asm_ = 0, asq = 0; int64_t an = audio_latent.numel();
        for (int64_t i = 0; i < an; ++i) { double a = audio_latent.data()[i]; asm_ += a; asq += a*a; }
        double astd = sqrt(asq/(double)an - (asm_/(double)an)*(asm_/(double)an));

        // track latent std per step to see whether the trajectory converges or diverges.
        double sm = 0, sq = 0; int64_t nn = latent.numel();
        for (int64_t i = 0; i < nn; ++i) { double vv = latent.data()[i]; sm += vv; sq += vv * vv; }
        double lmean = sm / (double)nn, lstd = sqrt(sq / (double)nn - lmean * lmean);
        printf("  step %2d/%d  t=%.2f model_t=%.2f sigma=%.4f dsig_v=%+.4f dsig_a=%+.4f  latent mean=%+.3f std=%.3f  aud_std=%.3f\n",
               step + 1, o.steps, sampler_t, tval, sampler_sigma, dsig_video, dsig_audio, lmean, lstd, astd);

        if (dump_traj) {
            char fn[64];
            snprintf(fn, sizeof(fn), "/vid_step_%02d.bin", step);
            write_bin(std::string(dump_traj) + fn, latent, "vid_step");
        }
    }
    int64_t sample_t1 = ggml_time_ms();
    float sample_s = (sample_t1 - sample_t0) / 1000.0f;
    if (cache_on) {
        printf("step-cache: skipped %d/%d steps (reused all 3 forwards) thresh=%.3f rate=%.3f%s rate_a=%.3f\n",
               cache_skipped, o.steps, cache_thresh, cache_rate,
               cache_audio_aware ? " [AUDIO-AWARE]" : "", cache_rate_a);
    }
    dump_stats("final latent", latent);

    // Optional: dump the final video latent [W_lat,H_lat,F,48] so it can be decoded
    // by PyTorch's own wan VAE (cross-decode → isolates a cpp VAE bug from a cpp DiT bug).
    if (const char* lp = getenv("NAVA_DUMP_LATENT")) {
        write_bin(lp, latent, "final_latent");
    }

    // free the DiT before VAE decode to lower peak memory.
    runner.reset();

    // ----- VAE decode -----
    // NAVA latent normalization (RESOLVED — see report): NAVA's LocalVideoVAEAdapter
    // has scaling_factor=1.0/shift_factor=0.0 (no renorm in the pipeline), and the
    // Wan2.2 video VAE applies z = z*std + mean INSIDE decode. The in-tree
    // WanVAERunner mirrors this exactly: diffusion_to_vae_latents = latents*std/1.0 + mean
    // with scale_factor=1.0, and the 48ch mean/std constants are BIT-IDENTICAL to
    // NAVA vae2_2.py (verified). So: latent (diffusion convention) ->
    // diffusion_to_vae_latents -> decode. decode() then maps [-1,1] -> [0,1].
    auto vae = std::make_shared<WAN::WanVAERunner>(
        backend, backend, String2TensorStorage{}, "", /*decode_only=*/true, VERSION_WAN2_2_TI2V);
    {
        ModelLoader vloader;
        if (!vloader.init_from_file_and_convert_name(o.vae, "vae.")) {
            printf("failed to init VAE loader from %s\n", o.vae.c_str());
            return 1;
        }
        vae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> vtensors;
        vae->get_param_tensors(vtensors, "first_stage_model");
        if (!vloader.load_tensors(vtensors)) {
            printf("failed to load VAE tensors\n");
            return 1;
        }
    }
    printf("VAE loaded; decoding...\n");

    sd_tiling_params_t tiling = {};
    // The Wan2.2 16x VAE decode buffer is spatial-bound (~7GB at 24x24 latent on the
    // 3060). Tile spatially at the proven-fit 24x24 latent size so full-bucket
    // (832x480 -> 52x30 latent) fits 12GB. Single-tile no-op when <= 24x24.
    // The 7.7GB decode compute buffer scales ~quadratically with the tile size. 24x24
    // peaks ~8.8GB (over the 7.5GB budget); smaller tiles cut it (16x16 ~ (16/24)^2 =
    // 0.44x). NAVA_VAE_TILE overrides the latent tile edge (default 24).
    int vae_tile = 24;
    if (const char* vt = getenv("NAVA_VAE_TILE")) { int t = atoi(vt); if (t >= 8 && t <= 64) vae_tile = t; }
    tiling.enabled        = (W_lat > vae_tile || H_lat > vae_tile);
    tiling.temporal_tiling = false;  // wan VAE is causal-conv3d => temporal already streamed
    tiling.tile_size_x    = vae_tile;
    tiling.tile_size_y    = vae_tile;
    tiling.target_overlap = 0.25f;
    printf("VAE tiling: %dx%d latent (enabled=%d)\n", vae_tile, vae_tile, (int)tiling.enabled);
    int64_t dec_t0 = ggml_time_ms();
    // WanVAERunner expects a 5D video latent [W,H,T,C,1] (get_latents_mean_std
    // reads channel at ne dim 3 for 5D; a 4D tensor is treated as an IMAGE
    // [W,H,C,1] with channel at dim 2). Our sampling latent is [W,H,T,48] -> add
    // the trailing batch singleton.
    sd::Tensor<float> latent5 = latent;
    latent5.reshape_({(int64_t)W_lat, (int64_t)H_lat, (int64_t)f_len, 48, 1});
    auto latent_vae = vae->diffusion_to_vae_latents(latent5);
    auto rgb = vae->decode(n_threads, latent_vae, tiling, /*decode_video=*/true, false, false);
    int64_t dec_t1 = ggml_time_ms();
    if (rgb.empty()) {
        printf("ERROR: VAE decode failed\n");
        return 1;
    }
    printf("VAE decode %.2fs; rgb shape=%s\n", (dec_t1 - dec_t0) / 1000.0f,
           sd::tensor_shape_to_string(rgb.shape()).c_str());
    dump_stats("decoded rgb [0,1]", rgb);

    // ----- continuation: dump the tail decoded frames LOSSLESSLY -----
    // NAVA_DUMP_TAIL=<dir> writes the last K decoded pixel frames (K=NAVA_TAIL_K, default 9
    // -> M=3 anchor) as lossless PNGs straight from the `rgb` tensor (NOT the VP9 webm). The
    // webm round-trip injects compression artifacts that COMPOUND through every chained
    // re-encode (sharpness/noise explodes seg-over-seg); chaining from these lossless frames
    // keeps the drift-sink clean. Feed them to `nava encode-video` -> `render --video-anchor`.
    if (const char* td = getenv("NAVA_DUMP_TAIL")) {
        int tail_k = 9;
        if (const char* tk = getenv("NAVA_TAIL_K")) { int v = atoi(tk); if (v >= 1) tail_k = v; }
        int64_t Tp = rgb.shape()[2];
        if (tail_k > (int)Tp) tail_k = (int)Tp;
        std::string dir = td;
        mkdirs(dir);
        for (int j = 0; j < tail_k; ++j) {
            int64_t fi = Tp - tail_k + j;  // oldest-first temporal order
            sd_image_t im = tensor_to_sd_image(rgb, (int)fi);
            char fn[64];
            snprintf(fn, sizeof(fn), "/f%03d.png", j + 1);
            write_image_to_file(dir + fn, im.data, (int)im.width, (int)im.height, (int)im.channel, "", 100);
            free(im.data);
        }
        printf("NAVA_DUMP_TAIL: wrote last %d decoded frames (lossless PNG) -> %s/f001..f%03d.png\n",
               tail_k, dir.c_str(), tail_k);
    }

    // ----- audio: decode the co-denoised audio latent -> waveform (LTX audio VAE) -----
    // audio_latent (ggml ne [128, audio_len]) carries the joint-denoised audio stream.
    // The LTX audio VAE applies its own per_channel_statistics renorm INSIDE decode and
    // the pipeline unscale (scaling_factor=1.0/shift_factor=0.0) is a no-op for NAVA, so
    // we feed the RAW latent. Output: planar waveform ggml ne [n_samples, n_ch] @ 48 kHz
    // (bwe). We interleave it into an sd_audio_t and let the webm muxer Opus-encode it.
    sd_audio_t* audio_track = nullptr;
    sd_audio_t audio_obj{};
    std::vector<float> audio_interleaved;  // must outlive the webm write below
    // Optional: dump the co-denoised audio latent (ggml ne [128, audio_len]) so the
    // cpp decode can be validated against nava_audio_vae_decode_ref.py on the SAME
    // latent. NAVA_DUMP_AUDIO_LATENT=<path> (or =1 -> /tmp/nava_audio_latent.bin).
    if (const char* dl = getenv("NAVA_DUMP_AUDIO_LATENT")) {
        std::string p = (std::string(dl) == "1") ? "/tmp/nava_audio_latent.bin" : dl;
        write_bin(p, audio_latent, "audio_latent");
        printf("dumped co-denoised audio latent -> %s (ne [128, %d])\n", p.c_str(), audio_len);
    }
    if (!o.audio_vae.empty()) {
        vae.reset();  // free the video VAE; the 340 MB audio VAE loads into the freed VRAM
        int64_t a_t0 = ggml_time_ms();
        ModelLoader aloader;
        if (!aloader.init_from_file(o.audio_vae)) {
            printf("WARN: failed to load audio VAE '%s' -> SILENT webm\n", o.audio_vae.c_str());
        } else {
            auto& atsm = aloader.get_tensor_storage_map();
            auto avae  = std::make_shared<LTXV::LTXAudioVAERunner>(backend, backend, atsm, "");
            avae->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> at;
            avae->get_param_tensors(at, "");
            if (!aloader.load_tensors(at)) {
                printf("WARN: audio VAE tensors load failed -> SILENT webm\n");
            } else {
                auto wav     = avae->decode(n_threads, audio_latent);  // ne [n_samples, n_ch]
                int64_t a_t1 = ggml_time_ms();
                if (wav.empty()) {
                    printf("WARN: audio VAE decode returned empty -> SILENT webm\n");
                } else {
                    const auto& ws  = wav.shape();
                    int64_t n_samp  = ws[0];
                    int64_t n_ch    = ws.size() > 1 ? ws[1] : 1;
                    if (n_ch < 1) n_ch = 1;
                    printf("audio VAE decode %.2fs; waveform ne=%s -> %lld samp x %lld ch @ %d Hz\n",
                           (a_t1 - a_t0) / 1000.0f, sd::tensor_shape_to_string(ws).c_str(),
                           (long long)n_samp, (long long)n_ch, avae->config.output_sample_rate());
                    dump_stats("audio waveform", wav);
                    if (const char* wp = getenv("NAVA_DUMP_WAV")) {
                        write_bin(wp, wav, "audio_waveform");  // planar [n_samp, n_ch] f32
                    }
                    // ggml is planar (ne0=n_samples fastest): data[c*n_samp + i].
                    // sd_audio_t wants interleaved: data[i*n_ch + c].
                    audio_interleaved.resize((size_t)(n_samp * n_ch));
                    const float* wd = wav.data();
                    for (int64_t i = 0; i < n_samp; ++i)
                        for (int64_t c = 0; c < n_ch; ++c)
                            audio_interleaved[(size_t)(i * n_ch + c)] = wd[(size_t)(c * n_samp + i)];
                    audio_obj.sample_rate  = (uint32_t)avae->config.output_sample_rate();
                    audio_obj.channels     = (uint32_t)n_ch;
                    audio_obj.sample_count = (uint64_t)n_samp;
                    audio_obj.data         = audio_interleaved.data();
                    audio_track            = &audio_obj;
                }
            }
        }
    }

    // ----- frames -> webm -----
    // rgb ggml ne [W,H,T,C], already [0,1] (decode applied (x+1)/2). Build sd_image_t per frame.
    int64_t Tpix = rgb.shape()[2];
    std::vector<sd_image_t> images;
    images.reserve((size_t)Tpix);
    for (int64_t fi = 0; fi < Tpix; ++fi) {
        images.push_back(tensor_to_sd_image(rgb, (int)fi));  // converts [0,1] f32 -> u8
    }

    std::string run_dir = o.runs_dir + "/" + o.out_name;
    mkdirs(run_dir);
    std::string webm_path = run_dir + "/clip.webm";

#ifdef SD_USE_WEBM
    int rc = create_webm_from_sd_images(webm_path.c_str(), images.data(), (int)images.size(), o.fps, 90, audio_track);
#else
    // fall back to whatever the build provides (mjpg avi etc.) but keep the name.
    int rc = create_video_from_sd_images(webm_path.c_str(), images.data(), (int)images.size(), o.fps, 90, audio_track);
#endif
    int64_t wall_t1 = ggml_time_ms();
    float wall_s = (wall_t1 - wall_t0) / 1000.0f;

    for (auto& im : images) free(im.data);

    bool ok = (rc == 0);
    printf("webm write rc=%d -> %s\n", rc, webm_path.c_str());

    // ----- meta.json (schema per tools/nava_eyetest_server.py) -----
    {
        std::string mp = run_dir + "/meta.json";
        FILE* mf = fopen(mp.c_str(), "wb");
        if (mf) {
            float s_per_step = o.steps > 0 ? sample_s / o.steps : 0.0f;
            fprintf(mf, "{\n");
            fprintf(mf, "  \"label\": \"%s\",\n", o.label.c_str());
            fprintf(mf, "  \"ok\": %s,\n", ok ? "true" : "false");
            fprintf(mf, "  \"rc\": %d,\n", rc);
            fprintf(mf, "  \"phase\": \"phase2-silent\",\n");
            fprintf(mf, "  \"w\": %d, \"h\": %d,\n", o.width, o.height);
            fprintf(mf, "  \"resolution\": \"%dx%d\",\n", o.width, o.height);
            fprintf(mf, "  \"frames\": %lld,\n", (long long)Tpix);
            fprintf(mf, "  \"latent_frames\": %d,\n", f_len);
            fprintf(mf, "  \"steps\": %d,\n", o.steps);
            fprintf(mf, "  \"seed\": %llu,\n", (unsigned long long)o.seed);
            fprintf(mf, "  \"cfg\": %.3f,\n", o.cfg);
            fprintf(mf, "  \"shift\": %.3f,\n", o.shift);
            fprintf(mf, "  \"wall_s\": %.2f,\n", wall_s);
            fprintf(mf, "  \"load_s\": %.2f,\n", (load_t1 - wall_t0) / 1000.0f);
            fprintf(mf, "  \"s_per_step\": %.3f,\n", s_per_step);
            fprintf(mf, "  \"prompt\": \"%s\",\n", o.prompt.c_str());
            fprintf(mf, "  \"backend\": \"%s\",\n", o.cuda ? "cuda" : "cpu");
            fprintf(mf, "  \"notes\": \"%s\"\n",
                    o.context.empty() ? "DUMMY context (smoke; no real prompt)" : "umT5 context.bin");
            fprintf(mf, "}\n");
            fclose(mf);
            printf("wrote %s\n", mp.c_str());
        }
    }

    printf("=== render done: %s (wall %.2fs, sample %.2fs, %d frames) ===\n",
           run_dir.c_str(), wall_s, sample_s, (int)Tpix);
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// UniPC scheduler numerical validation vs PyTorch reference.
//   nava unipc-test <dir>
// Reads raw float32 .bin files written by unipc_ref.py:
//   init.bin [1,64], model_outputs.bin [N,64], traj.bin [N,64] (N=10).
// Runs UniPCSched and prints per-step + overall max-abs-error.
// ---------------------------------------------------------------------------
static std::vector<float> read_f32_bin(const std::string& path, size_t count) {
    std::vector<float> v(count);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        printf("cannot open %s\n", path.c_str());
        return {};
    }
    size_t got = fread(v.data(), sizeof(float), count, f);
    fclose(f);
    if (got != count) {
        printf("short read %s: got %zu want %zu\n", path.c_str(), got, count);
        return {};
    }
    return v;
}

static int run_unipc_test(int argc, char** argv) {
    std::string dir = (argc >= 3) ? argv[2] : "/tmp/unipc_ref";
    const int N = 10;
    const int D = 64;

    auto init = read_f32_bin(dir + "/init.bin", (size_t)D);
    auto mos  = read_f32_bin(dir + "/model_outputs.bin", (size_t)N * D);
    auto traj = read_f32_bin(dir + "/traj.bin", (size_t)N * D);
    if (init.empty() || mos.empty() || traj.empty()) {
        printf("FAIL: could not load reference .bin files from %s\n", dir.c_str());
        return 1;
    }

    UniPCSched sched;
    sched.set_timesteps(N, 5.0f);

    printf("sigmas (cpp): ");
    for (float s : sched.sigmas) printf("%.6f ", s);
    printf("\n");

    sd::Tensor<float> sample({1, (int64_t)D}, init);

    double overall_max = 0.0;
    bool any_bad       = false;
    for (int k = 0; k < N; ++k) {
        std::vector<float> mo_k(mos.begin() + (size_t)k * D, mos.begin() + (size_t)(k + 1) * D);
        sd::Tensor<float> mo({1, (int64_t)D}, mo_k);
        sample = sched.step(mo, sample);

        double step_max = 0.0;
        for (int i = 0; i < D; ++i) {
            float v = sample.data()[i];
            if (std::isnan(v) || std::isinf(v)) any_bad = true;
            double e = std::fabs((double)v - (double)traj[(size_t)k * D + i]);
            if (e > step_max) step_max = e;
        }
        if (step_max > overall_max) overall_max = step_max;
        printf("step %2d  sigma_t=%.6f -> %.6f   max_abs_err=%.3e\n",
               k, sched.sigmas[(size_t)k], sched.sigmas[(size_t)k + 1], step_max);
    }

    printf("\nfinal head (cpp): ");
    for (int i = 0; i < 8; ++i) printf("%.6f ", sample.data()[i]);
    printf("\nfinal head (py):  ");
    for (int i = 0; i < 8; ++i) printf("%.6f ", traj[(size_t)(N - 1) * D + i]);
    printf("\n\nOVERALL MAX ABS ERROR = %.6e\n", overall_max);
    printf("NaN/Inf present: %s\n", any_bad ? "YES" : "no");
    bool pass = (overall_max < 1e-4) && !any_bad;
    printf("GATE (<1e-4): %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    sd_set_log_callback(log_cb, nullptr);

    if (argc >= 2 && std::string(argv[1]) == "render") {
        return run_render(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "unipc-test") {
        return run_unipc_test(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "encode-prompt") {
        // nava encode-prompt <umt5_gguf> <prompt.txt> <out.bin>  — encode a text prompt
        // through umT5-xxl (GPU) into a NAVA context bin [4096,512] (zero-padded), matching
        // dump_three_ctx.py. Makes the cpp nava pipeline self-sufficient (text -> context).
        if (argc < 5) {
            printf("usage: %s encode-prompt <umt5_gguf> <prompt.txt> <out.bin>\n", argv[0]);
            return 1;
        }
        std::string gguf = argv[2], pf = argv[3], out = argv[4];
        std::string text;
        if (!nava_read_file(pf, text)) { printf("cannot open prompt file %s\n", pf.c_str()); return 1; }
        ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        const bool mmap = nava_umt5_use_mmap();
        double load_s = 0;
        auto t5 = nava_load_umt5(gguf, backend, mmap, &load_s);
        if (!t5) return 1;
        printf("umt5 loaded (%.2fs, mmap=%s); prompt is %zu bytes\n", load_s, mmap ? "on" : "off", text.size());
        // Shared helper does the load-bearing caption transform (<S> -> <S><extra_id_2> ),
        // NFKC normalize, 512-cap (511 content + EOS), and spk-pos collection — IDENTICAL
        // to the inline render path (bit-exactness by construction).
        std::vector<int> spk_pos;
        sd::Tensor<float> ctx = nava_encode_prompt_ctx(t5, text, &spk_pos);
        if (ctx.numel() == 0) { printf("umt5 encode returned empty\n"); return 1; }
        // Voice clone: record the <extra_id_2>(=256297) position(s) to a sidecar
        // <out>.spkpos, so `render --spk-emb` can splice the projected speaker embedding.
        {
            std::string sp = out + ".spkpos";
            FILE* sf = fopen(sp.c_str(), "w");
            if (sf) {
                for (int p : spk_pos) fprintf(sf, "%d\n", p);
                fclose(sf);
                printf("spk_pos: %zu <extra_id_2> token(s) -> %s\n", spk_pos.size(), sp.c_str());
            }
        }
        write_bin(out, ctx, "context");
        dump_stats("umt5 context", ctx);
        printf("encoded -> %s  ne=[%lld,512]\n", out.c_str(), (long long)ctx.shape()[0]);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "encode-prompts") {
        // nava encode-prompts <umt5_gguf> <out_prefix> <p0.txt> <p1.txt> ...
        //   Batch text-encode for a multi-segment CHAIN: load umT5 ONCE, encode every
        //   segment's prompt while it stays resident, then free — so an N-segment chain
        //   pays 1x umT5 load instead of N. Writes <out_prefix>{i}.bin (+ .spkpos sidecar)
        //   for each prompt; a chain driver then loops `nava render --context <prefix>{i}.bin`.
        //   (For a SINGLE self-contained render, prefer `nava render --umt5 ... --prompt`.)
        if (argc < 5) {
            printf("usage: %s encode-prompts <umt5_gguf> <out_prefix> <p0.txt> [p1.txt ...]\n", argv[0]);
            return 1;
        }
        std::string gguf = argv[2], prefix = argv[3];
        std::vector<std::string> pfiles;
        for (int i = 4; i < argc; ++i) pfiles.push_back(argv[i]);
        ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        const bool mmap = nava_umt5_use_mmap();
        double load_s = 0;
        int64_t t_all0 = ggml_time_ms();
        auto t5 = nava_load_umt5(gguf, backend, mmap, &load_s);  // 1x load for the whole batch
        if (!t5) return 1;
        printf("umt5 loaded ONCE (%.2fs, mmap=%s) for %zu segment prompt(s)\n",
               load_s, mmap ? "on" : "off", pfiles.size());
        for (size_t i = 0; i < pfiles.size(); ++i) {
            std::string text;
            if (!nava_read_file(pfiles[i], text)) { printf("cannot open prompt file %s\n", pfiles[i].c_str()); return 1; }
            std::vector<int> spk_pos;
            int64_t e0 = ggml_time_ms();
            sd::Tensor<float> ctx = nava_encode_prompt_ctx(t5, text, &spk_pos);
            if (ctx.numel() == 0) { printf("umt5 encode returned empty for %s\n", pfiles[i].c_str()); return 1; }
            double enc_s = (ggml_time_ms() - e0) / 1000.0;
            std::string out = prefix + std::to_string(i) + ".bin";
            {
                std::string sp = out + ".spkpos";
                FILE* sf = fopen(sp.c_str(), "w");
                if (sf) { for (int p : spk_pos) fprintf(sf, "%d\n", p); fclose(sf); }
            }
            write_bin(out, ctx, "context");
            printf("  [%zu] %s -> %s  (encode %.2fs, %zu spk pos)\n",
                   i, pfiles[i].c_str(), out.c_str(), enc_s, spk_pos.size());
        }
        t5.reset();  // free umT5 once the whole batch is encoded
        printf("encode-prompts: %zu prompt(s) in one residency, total %.2fs (load %.2fs)\n",
               pfiles.size(), (ggml_time_ms() - t_all0) / 1000.0, load_s);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "encode-video") {
        // nava encode-video --vae <gguf> --width W --height H --out <out.bin> [--cuda] f0 f1 ...
        //
        // Re-encode K decoded PIXEL frames (in temporal order, oldest->newest) through the
        // Wan2.2 16x VAE into a diffusion-latent ANCHOR block ggml ne [W_lat,H_lat,M,48]
        // (M = 1 + (K-1)/4; the VAE encodes frame 0 alone as a causal I-frame std~0.5 and
        // each following 4-frame chunk as a P-frame std~0.9 — a valid temporal sequence).
        // This is the CORRECT clip-continuation primitive: feed the output to
        //   nava render --video-anchor <out.bin>
        // to pin those M frames as clean motion anchors for the next segment. The legacy
        // raw-latent --video-anchor (slicing a prior render's P-frame into the I-frame slot)
        // is WRONG and smears the decode (see HANDOFF-nava-CONTINUATION-NEXT.md).
        //
        // K=1 (single frame) reproduces the `render --image` i2v anchor BIT-FOR-BIT (same VAE,
        // same full-frame encode, same vae_to_diffusion_latents) — a numeric parity gate.
        std::string vae = "models/wan2.2-vae-48ch-f16.gguf", out;
        int W = 0, H = 0;
        bool want_cuda = false;
        std::vector<std::string> frame_files;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--vae") { if (i + 1 < argc) vae = argv[++i]; }
            else if (a == "--width") { if (i + 1 < argc) W = atoi(argv[++i]); }
            else if (a == "--height") { if (i + 1 < argc) H = atoi(argv[++i]); }
            else if (a == "--out") { if (i + 1 < argc) out = argv[++i]; }
            else if (a == "--cuda") { want_cuda = true; }
            else frame_files.push_back(a);
        }
        if (W <= 0 || H <= 0 || out.empty() || frame_files.empty()) {
            printf("usage: %s encode-video --vae <gguf> --width W --height H --out <out.bin> [--cuda] f0.png f1.png ...\n",
                   argv[0]);
            printf("  W,H are PIXEL dims (multiples of 16). K frames -> M = 1+(K-1)/4 latent anchor frames.\n");
            return 1;
        }
        const int K = (int)frame_files.size();
        const int W_lat = W / 16, H_lat = H / 16;
        const int M = 1 + (K - 1) / 4;
        if (W % 16 != 0 || H % 16 != 0) { printf("ERROR: W,H must be multiples of 16 (got %dx%d)\n", W, H); return 1; }
        if (((K - 1) % 4) != 0)
            printf("WARN: K=%d is not 1+4*m; the final P-chunk is partial (<4 frames). M=%d. "
                   "For clean chunking use K in {1,5,9,13,...}.\n", K, M);
        printf("=== NAVA encode-video: %d pixel frame(s) -> [%d,%d,%d,48] diffusion anchor (vae=%s) ===\n",
               K, W_lat, H_lat, M, vae.c_str());

        ggml_backend_t backend = nullptr;
        if (want_cuda) {
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
            printf("backend: CUDA\n");
#endif
        }
        if (!backend) { backend = ggml_backend_cpu_init(); printf("backend: CPU\n"); }
        const int n_threads = 8;

        // Load + Python-parity resize/center-crop each frame, stack into ggml ne [W,H,K,3].
        // Per-frame data idx = x + W*(y + H*(0 + 1*c)); stacked idx = x + W*(y + H*(f + K*c)).
        sd::Tensor<float> frames({(int64_t)W, (int64_t)H, (int64_t)K, 3});
        for (int f = 0; f < K; ++f) {
            sd::Tensor<float> img;
            if (looks_like_image_file(frame_files[(size_t)f])) {
                if (!load_image_resize_center_crop(frame_files[(size_t)f], W, H, img)) return 1;
            } else {
                img = sd::load_tensor_from_file_as_tensor<float>(frame_files[(size_t)f]);
                printf("loaded preprocessed RGB bin %s shape=%s\n", frame_files[(size_t)f].c_str(),
                       sd::tensor_shape_to_string(img.shape()).c_str());
            }
            if (img.shape()[0] != (int64_t)W || img.shape()[1] != (int64_t)H) {
                printf("ERROR: frame %d ne %s != [%d,%d,1,3]\n", f,
                       sd::tensor_shape_to_string(img.shape()).c_str(), W, H);
                return 1;
            }
            for (int64_t c = 0; c < 3; ++c)
                for (int64_t y = 0; y < H; ++y)
                    for (int64_t x = 0; x < W; ++x)
                        frames.data()[x + W * (y + H * (f + (int64_t)K * c))] =
                            img.data()[x + W * (y + H * (0 + 1 * c))];
        }
        // Explicit 5D video tensor [W,H,K,3,1] (temporal at ne2) — symmetric to the decode
        // path's latent5 [W,H,T,48,1]; dim==5 means _compute does NOT unsqueeze (the 4D
        // single-image path relies on the unsqueeze batch->temporal convention).
        frames.reshape_({(int64_t)W, (int64_t)H, (int64_t)K, 3, 1});

        auto encvae = std::make_shared<WAN::WanVAERunner>(
            backend, backend, String2TensorStorage{}, "", /*decode_only=*/false, VERSION_WAN2_2_TI2V);
        {
            ModelLoader vl;
            if (!vl.init_from_file_and_convert_name(vae, "vae.")) { printf("VAE loader fail: %s\n", vae.c_str()); return 1; }
            encvae->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> vt;
            encvae->get_param_tensors(vt, "first_stage_model");
            if (!vl.load_tensors(vt)) { printf("VAE tensors load fail\n"); return 1; }
        }
        sd_tiling_params_t et = {};
        // Full-frame encode (Python-parity) — same as render --image. Tiling seams perturb
        // the anchor latent (~14%); a single block is encoded in isolation so VRAM is fine.
        et.enabled = (W_lat > 24 || H_lat > 24) && (getenv("NAVA_VAE_ENCODE_TILE") != nullptr);
        et.temporal_tiling = false; et.tile_size_x = 24; et.tile_size_y = 24; et.target_overlap = 0.25f;
        printf("VAE encode tiling=%s\n", et.enabled ? "ON (24x24/0.25)" : "OFF (full-frame)");
        auto mu = encvae->encode(n_threads, frames, et, /*circular_x=*/true, /*circular_y=*/false);
        if (mu.empty()) { printf("ERROR: VAE encode failed\n"); return 1; }
        printf("encoded mu shape=%s (expect [%d,%d,%d,48] or trailing-singleton variant)\n",
               sd::tensor_shape_to_string(mu.shape()).c_str(), W_lat, H_lat, M);
        // Normalize to a clean 4D [W_lat,H_lat,M,48] regardless of trailing singletons.
        if (mu.numel() != (int64_t)W_lat * H_lat * M * 48) {
            printf("ERROR: mu numel %lld != %lld — layout mismatch (K=%d -> M=%d?)\n",
                   (long long)mu.numel(), (long long)((int64_t)W_lat * H_lat * M * 48), K, M);
            return 1;
        }
        // vae_to_diffusion_latents reads channels at ne dim 3 for a 5D tensor -> [W,H,M,48,1].
        sd::Tensor<float> mu5 = mu;
        mu5.reshape_({(int64_t)W_lat, (int64_t)H_lat, (int64_t)M, 48, 1});
        auto anchor = encvae->vae_to_diffusion_latents(mu5);
        anchor.reshape_({(int64_t)W_lat, (int64_t)H_lat, (int64_t)M, 48});
        encvae.reset();
        dump_stats("video anchor (diffusion)", anchor);
        // Per-frame std diagnostic: frame 0 should be I-frame (~0.5), frames 1+ P-frames (~0.9).
        for (int f = 0; f < M; ++f) {
            double s = 0, sq = 0; int64_t cnt = 0;
            for (int64_t c = 0; c < 48; ++c)
                for (int64_t y = 0; y < H_lat; ++y)
                    for (int64_t x = 0; x < W_lat; ++x) {
                        double v = anchor.data()[x + W_lat * (y + H_lat * (f + (int64_t)M * c))];
                        s += v; sq += v * v; ++cnt;
                    }
            double m = s / (double)cnt, sd_ = sqrt(sq / (double)cnt - m * m);
            printf("  anchor frame %d: mean=%+.4f std=%.4f%s\n", f, m, sd_,
                   f == 0 ? "  (I-frame, expect ~0.5)" : "  (P-frame, expect ~0.9)");
        }
        write_bin(out, anchor, "vid_anchor");
        printf("=== encode-video done -> %s  ne=[%d,%d,%d,48] ===\n", out.c_str(), W_lat, H_lat, M);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "encode-stream") {
        // nava encode-stream --vae <g> --width W --height H --out <out.bin> [--cuda]
        //                    [--decode-preview <png>] prime0.png ... primeN.png target.png
        //
        // STREAMING / CACHED VAE ENCODE (clip-continuity gate). All positional args are
        // PIXEL frames in temporal order; the LAST is the target to encode, the preceding
        // ones are the priming tail. The Wan2.2 VAE encoder's causal cache (_enc_feat_map)
        // is primed with the tail, then the target is encoded as a standalone 1-frame chunk
        // WITHOUT clearing the cache, so it may carry P-frame temporal state instead of the
        // history-less I-frame reset that a fresh encode produces. With ONE positional arg
        // (no priming) this reproduces the M=1 baseline I-encode (== render --image anchor),
        // a numeric parity gate.
        std::string vae = "models/wan2.2-vae-48ch-f16.gguf", out, preview;
        int W = 0, H = 0;
        bool want_cuda = false;
        std::vector<std::string> frame_files;
        for (int i = 2; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "--vae") { if (i + 1 < argc) vae = argv[++i]; }
            else if (a == "--width") { if (i + 1 < argc) W = atoi(argv[++i]); }
            else if (a == "--height") { if (i + 1 < argc) H = atoi(argv[++i]); }
            else if (a == "--out") { if (i + 1 < argc) out = argv[++i]; }
            else if (a == "--decode-preview") { if (i + 1 < argc) preview = argv[++i]; }
            else if (a == "--cuda") { want_cuda = true; }
            else frame_files.push_back(a);
        }
        if (W <= 0 || H <= 0 || out.empty() || frame_files.empty()) {
            printf("usage: %s encode-stream --vae <g> --width W --height H --out <out.bin> [--cuda]\n"
                   "                        [--decode-preview <png>] prime0.png ... target.png\n", argv[0]);
            return 1;
        }
        if (W % 16 != 0 || H % 16 != 0) { printf("ERROR: W,H must be multiples of 16 (got %dx%d)\n", W, H); return 1; }
        const int K = (int)frame_files.size();
        const int W_lat = W / 16, H_lat = H / 16;
        const int n_prime = K - 1;
        printf("=== NAVA encode-stream: prime=%d frame(s) + 1 target -> cached [%d,%d,1,48] (vae=%s) ===\n",
               n_prime, W_lat, H_lat, vae.c_str());

        ggml_backend_t backend = nullptr;
        if (want_cuda) {
#ifdef GGML_USE_CUDA
            backend = ggml_backend_cuda_init(0);
            printf("backend: CUDA\n");
#endif
        }
        if (!backend) { backend = ggml_backend_cpu_init(); printf("backend: CPU\n"); }
        const int n_threads = 8;

        sd::Tensor<float> frames({(int64_t)W, (int64_t)H, (int64_t)K, 3});
        for (int f = 0; f < K; ++f) {
            sd::Tensor<float> img;
            if (looks_like_image_file(frame_files[(size_t)f])) {
                if (!load_image_resize_center_crop(frame_files[(size_t)f], W, H, img)) return 1;
            } else {
                img = sd::load_tensor_from_file_as_tensor<float>(frame_files[(size_t)f]);
            }
            if (img.shape()[0] != (int64_t)W || img.shape()[1] != (int64_t)H) {
                printf("ERROR: frame %d ne %s != [%d,%d,1,3]\n", f,
                       sd::tensor_shape_to_string(img.shape()).c_str(), W, H);
                return 1;
            }
            for (int64_t c = 0; c < 3; ++c)
                for (int64_t y = 0; y < H; ++y)
                    for (int64_t x = 0; x < W; ++x)
                        frames.data()[x + W * (y + H * (f + (int64_t)K * c))] =
                            img.data()[x + W * (y + H * (0 + 1 * c))];
        }
        frames.reshape_({(int64_t)W, (int64_t)H, (int64_t)K, 3, 1});

        auto encvae = std::make_shared<WAN::WanVAERunner>(
            backend, backend, String2TensorStorage{}, "", /*decode_only=*/false, VERSION_WAN2_2_TI2V);
        {
            ModelLoader vl;
            if (!vl.init_from_file_and_convert_name(vae, "vae.")) { printf("VAE loader fail: %s\n", vae.c_str()); return 1; }
            encvae->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> vt;
            encvae->get_param_tensors(vt, "first_stage_model");
            if (!vl.load_tensors(vt)) { printf("VAE tensors load fail\n"); return 1; }
        }

        // NB: encode-video's full-frame path runs with circular OFF (its circular_x arg
        // only takes effect in the tiled path), so leave the default to stay bit-parity
        // with the M=1 baseline anchor.
        auto mu = encvae->encode_streaming(n_threads, frames);
        if (mu.empty()) { printf("ERROR: streaming encode failed\n"); return 1; }
        if (mu.numel() != (int64_t)W_lat * H_lat * 48) {
            printf("ERROR: mu numel %lld != %lld (target should be ONE latent frame)\n",
                   (long long)mu.numel(), (long long)((int64_t)W_lat * H_lat * 48));
            return 1;
        }
        // diffusion-space anchor (same normalization as encode-video / render --image),
        // so std is directly comparable to the I~0.5 / P~0.9 bands.
        sd::Tensor<float> mu5 = mu;
        mu5.reshape_({(int64_t)W_lat, (int64_t)H_lat, 1, 48, 1});
        auto anchor = encvae->vae_to_diffusion_latents(mu5);
        anchor.reshape_({(int64_t)W_lat, (int64_t)H_lat, 1, 48});
        dump_stats("cached target anchor (diffusion)", anchor);
        {
            double s = 0, sq = 0; int64_t cnt = 0;
            for (int64_t i = 0; i < anchor.numel(); ++i) { double v = anchor.data()[i]; s += v; sq += v * v; ++cnt; }
            double m = s / cnt, st = sqrt(sq / cnt - m * m);
            printf("  target latent: mean=%+.4f std=%.4f  (I-frame ~0.5, P-frame ~0.9)\n", m, st);
        }
        write_bin(out, anchor, "vid_anchor");

        if (!preview.empty()) {
            // Decode the single mu latent back to pixels (history-less 1-frame decode) and
            // save a PNG to eyeball what the cached latent represents.
            sd_tiling_params_t dt = {};
            dt.enabled = false;
            sd::Tensor<float> mu1 = mu;
            mu1.reshape_({(int64_t)W_lat, (int64_t)H_lat, 1, 48, 1});
            auto rgb = encvae->decode(n_threads, mu1, dt, /*decode_video=*/true, false, false, true);
            if (!rgb.empty()) {
                sd_image_t im = tensor_to_sd_image(rgb, 0);
                write_image_to_file(preview, im.data, (int)im.width, (int)im.height, (int)im.channel, "", 100);
                free(im.data);
                printf("decode preview -> %s\n", preview.c_str());
            } else {
                printf("WARN: decode preview failed\n");
            }
        }
        encvae.reset();
        printf("=== encode-stream done -> %s  ne=[%d,%d,1,48] ===\n", out.c_str(), W_lat, H_lat);
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "audio-encode") {
        // nava audio-encode <audio_vae_gguf> <in.wav> <out_latent.bin> [--cuda]
        //   wav (16 kHz, mono/stereo) -> log-mel -> LTX audio encoder -> latent ne [128, ceil(T_mel/4)].
        //   CPU by default (encoder is tiny ~350MB); --cuda available for the service build.
        if (argc < 5) { printf("usage: %s audio-encode <audio_vae_gguf> <in.wav> <out_latent.bin> [--cuda]\n", argv[0]); return 1; }
        bool use_cuda = false;
        for (int i = 5; i < argc; ++i) if (std::string(argv[i]) == "--cuda") use_cuda = true;

        sd::Tensor<float> wave;
        uint32_t sr = 0;
        if (!read_wav_16k(argv[3], wave, sr)) return 1;
        if (sr != 16000) {
            printf("audio-encode: wav is %u Hz; require 16000 Hz (run: ffmpeg -i in -ar 16000 -ac 1 out.wav)\n", sr);
            return 1;
        }
        printf("audio-encode: wav '%s' sr=%u ne=[%lld,%lld] dur=%.4fs\n",
               argv[3], sr, (long long)wave.shape()[0], (long long)wave.shape()[1],
               (double)wave.shape()[0] / sr);

        ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
        if (use_cuda) backend = ggml_backend_cuda_init(0);
#else
        (void)use_cuda;
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        ModelLoader ml;
        if (!ml.init_from_file(argv[2])) { printf("audio-encode: vae load fail\n"); return 1; }
        auto& tsm = ml.get_tensor_storage_map();
        auto avae = std::make_shared<LTXV::LTXAudioVAERunner>(backend, backend, tsm, "");
        if (!avae->config.has_encoder) {
            printf("audio-encode: gguf has NO encoder tensors. Repack with: "
                   "tools/convert_ltx_audio_vae.py --with-encoder\n");
            return 1;
        }
        avae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> at;
        avae->get_param_tensors(at, "");
        if (!ml.load_tensors(at)) { printf("audio-encode: vae tensors fail\n"); return 1; }
        printf("audio-encode: vae loaded (%zu params); encoder ready\n", at.size());

        auto latent = avae->encode(8, wave);
        if (latent.empty()) { printf("audio-encode: ERROR empty latent\n"); return 1; }
        dump_stats("audio_latent", latent);
        printf("audio-encode: latent ne=%s\n", sd::tensor_shape_to_string(latent.shape()).c_str());
        write_bin(argv[4], latent, "audio_latent");
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "ltx-audio-test") {
        // nava ltx-audio-test <gguf> <latent.bin> [out_wave.bin]  — load the audio VAE gguf
        // into the in-tree LTXV decoder + decode on CUDA (CPU hits an F16 assert), dump waveform.
        if (argc < 4) { printf("usage: %s ltx-audio-test <gguf> <latent.bin> [out.bin]\n", argv[0]); return 1; }
        ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(0);
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        ModelLoader ml;
        if (!ml.init_from_file(argv[2])) { printf("audio vae load fail\n"); return 1; }
        auto& tsm = ml.get_tensor_storage_map();
        auto avae = std::make_shared<LTXV::LTXAudioVAERunner>(backend, backend, tsm, "");
        avae->alloc_params_buffer();
        std::map<std::string, ggml_tensor*> at;
        avae->get_param_tensors(at, "");
        if (!ml.load_tensors(at)) { printf("audio vae tensors fail\n"); return 1; }
        printf("audio vae loaded (%zu params); config sample_rate=%d\n", at.size(), avae->config.sample_rate);
        auto z = sd::load_tensor_from_file_as_tensor<float>(argv[3]);
        printf("latent shape=%s\n", sd::tensor_shape_to_string(z.shape()).c_str());
        auto wav = avae->decode(8, z);
        if (wav.empty()) { printf("ERROR: audio decode returned empty\n"); return 1; }
        dump_stats("waveform", wav);
        if (argc >= 5) write_bin(argv[4], wav, "waveform");
        return 0;
    }
    if (argc < 2) {
        printf("usage:\n");
        printf("  %s <gguf> [<input_dir>] [<out_dir>]      # phase-1 single forward\n", argv[0]);
        printf("  %s render --umt5 <gguf> --prompt \"...\"     # phase-2 render, inline text-encode\n", argv[0]);
        printf("  %s render --context ctx.bin [opts]        # phase-2 render from precomputed ctx\n", argv[0]);
        printf("  %s encode-prompt <umt5> <p.txt> <out.bin> # text -> umT5 cond context bin\n", argv[0]);
        printf("  %s encode-prompts <umt5> <prefix> p0 p1.. # batch encode (1x umT5 load for a chain)\n", argv[0]);
        printf("  %s audio-encode <avae> <in.wav> <out.bin> # wav(16k) -> LTX audio latent [128,T/4]\n", argv[0]);
        printf("  %s unipc-test [<ref_dir>]                 # UniPC scheduler numeric validation\n", argv[0]);
        return 1;
    }
    return run_single_forward(argc, argv);
}
