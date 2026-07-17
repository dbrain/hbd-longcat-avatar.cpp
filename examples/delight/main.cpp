// ---------------------------------------------------------------------------------------------
// sd-delight — Hunyuan3D-Delight-v2-0 as a de-lighting pre-process.
//
// The model is an SD-2.1-class InstructPix2Pix UNet (in_channels 8, cross_attention_dim 1024,
// v_prediction). sd.cpp already implements the whole ip2p contract; this tool only drives it and
// adds the two things Tencent's python does around the pipeline (alpha erosion + white fill, and
// the optional recorrect_rgb colour-drift guard), plus the RATIO composition described below.
//
// THE 512-vs-NATIVE PROBLEM
// ------------------------
// Delight runs at 512x512 (that is what it was trained for). Our texture projection loads the
// source at NATIVE resolution (tex_project.hpp load_rgba01 -> stbi_load(..., 4), no resize); the
// soldier input is 1060x1060. Round-tripping 512^2 delit pixels back to 1060^2 would destroy the
// eye/button/buckle detail that is the entire reason we project the real photo instead of letting
// a generative painter re-imagine it.
//
// So we do not use the delit pixels as the output. We use them only to ESTIMATE THE SHADING:
//
//     S   = lowpass( orig512 / delight(orig512) )     // shading is low-frequency
//     out = orig_native / upsample(S)                 // 100% of the native detail preserved
//
// Upsampling *shading* 512->1060 is near-lossless precisely because shading is low-frequency.
// This works in IMAGE space (a regular grid), so the lowpass is trivially safe -- unlike atlas
// space, where a neighbourhood op crosses unrelated charts.
//
// --mode direct uses the delit pixels upsampled, for A/B. NOTE: it does NOT currently
// reproduce Tencent -- upstream ALWAYS applies recorrect_rgb (dehighlight_utils.py:104) and
// our --recorrect defaults OFF. Pass --recorrect with --mode direct for reference behaviour.
//
// BACKGROUND HANDLING (matters more than it looks)
// ------------------------------------------------
// Tencent always feeds the net an image whose background has been filled WHITE (their reference is
// rembg'd RGBA). Our pipeline's --image is a BLACK-background RGB matte (make_matte), i.e. no alpha
// at all. Feeding a black background to a net trained on white ones is off-distribution, and worse,
// the ratio would divide by a near-zero delit background. So we derive a mask (explicit --alpha, the
// file's own alpha, or a background key), white-fill for the delight pass, and restore the original
// background in the output. The lowpass is mask-normalised (blur(S*m)/blur(m)) so the white
// background cannot bleed shading across the silhouette.
//
// The output PNG keeps the INPUT's channel count, because tex_project.hpp decides has_alpha from the
// file's native channel count -- emitting RGBA for an RGB input would silently change front-view
// behaviour downstream.
// ---------------------------------------------------------------------------------------------

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

struct DelightParams {
    std::string model_path;
    std::string image_path;
    std::string output_path;
    std::string alpha_path;
    std::string save_delit_path;

    int steps      = 50;    // Tencent: num_inference_steps=50
    int64_t seed   = 42;    // Tencent: torch.manual_seed(42)
    // BOTH 1.0 = Tencent's REFERENCE SEMANTICS, and ~2x faster (1 UNet eval/step, not 2).
    // The old defaults were 1.5/1.5 with a comment pointing at a "note in main()" and a "README
    // note" -- NEITHER EXISTED. The 1.5 was unjustified anywhere in the tree. Measured against the
    // real reference pipeline (torch/diffusers, same weights, byte-identical 512 input): the change
    // is worth 5.18 MAE, UNDER the reference's own seed-to-seed noise floor of 12.04 -- i.e. real in
    // the code, invisible in the output. Take it for correctness + speed, not for looks.
    // Upstream's own image_guidance_scale=1.5 is DEAD CODE: diffusers gates CFG on
    // (guidance_scale > 1.0 && image_guidance_scale >= 1.0), and Tencent passes guidance_scale=1.0,
    // so the reference runs ONE conditional pass with no CFG at all (verified at runtime:
    // do_classifier_free_guidance = False). In sd.cpp, txt_cfg == img_cfg makes use_uncond=false,
    // so --cfg is NOT the text CFG here -- it acts as the image-guidance scale. Both at 1.0 makes
    // both gates false => pred = pred_cond => exactly the reference.
    float txt_cfg  = 1.0f;  // Tencent: guidance_scale=1.0 (dehighlight_utils.py:26)
    float img_cfg  = 1.0f;  // see above -- 1.5 upstream is dead code; 1.0 reproduces the reference
    int size       = 512;   // Tencent: image.resize((512, 512))
    int n_threads  = 0;     // 0 = leave sd_ctx_params_init's physical-core default alone
    // Tencent loads the pipeline with torch_dtype=torch.float16. The delight UNet ships FP32 on disk
    // (3.46GB) while its VAE/text-encoder ship FP16, so leaving wtype at the file's native type would
    // load ~4.3GB of params instead of ~2.6GB, for no fidelity the reference implementation keeps.
    sd_type_t wtype = SD_TYPE_F16;

    bool recorrect       = false;  // OFF by default; see note in main()
    float recorrect_scale = 0.95f;

    std::string mode      = "ratio";  // ratio | direct
    // DEFAULT = "auto": conditioning-weighted blend of the per-channel and achromatic estimators.
    // Owner's standing requirement: this is an automated "best output" workflow, so anything that
    // would otherwise be a manual per-image judgement gets automated, with an override kept.
    std::string ratio_est = "auto";   // auto | mean | lum | pixel -- see the ratio block
    float lum_k           = 0.10f;    // auto: k = lum_k * subject mean delit luminance
    float sigma           = 12.0f;    // lowpass sigma, in --size pixels
    float smin       = 0.25f;
    float smax       = 4.0f;
    bool ratio_srgb  = false;  // default: do the division in LINEAR light

    std::string bg = "auto";  // auto | black | white | none
    bool verbose   = false;
};

// ------------------------------- colour helpers -------------------------------

static inline float srgb_to_linear(float c) {
    return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}
static inline float linear_to_srgb(float c) {
    return c <= 0.0031308f ? c * 12.92f : 1.055f * std::pow(c, 1.0f / 2.4f) - 0.055f;
}
static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ------------------------------- gaussian blur -------------------------------

static std::vector<float> gaussian_kernel(float sigma, int& radius) {
    radius = (int)std::ceil(3.0f * sigma);
    if (radius < 1) {
        radius = 1;
    }
    std::vector<float> k((size_t)(2 * radius + 1));
    float sum = 0.0f;
    for (int i = -radius; i <= radius; i++) {
        float v            = std::exp(-(float)(i * i) / (2.0f * sigma * sigma));
        k[(size_t)(i + radius)] = v;
        sum += v;
    }
    for (auto& v : k) {
        v /= sum;
    }
    return k;
}

// Separable blur of a single-channel plane, clamped at the borders.
static void blur_plane(std::vector<float>& p, int w, int h, float sigma) {
    int radius        = 0;
    std::vector<float> k = gaussian_kernel(sigma, radius);
    std::vector<float> tmp((size_t)w * h, 0.0f);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; i++) {
                int sx = x + i;
                sx     = sx < 0 ? 0 : (sx >= w ? w - 1 : sx);
                acc += p[(size_t)y * w + sx] * k[(size_t)(i + radius)];
            }
            tmp[(size_t)y * w + x] = acc;
        }
    }
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            float acc = 0.0f;
            for (int i = -radius; i <= radius; i++) {
                int sy = y + i;
                sy     = sy < 0 ? 0 : (sy >= h ? h - 1 : sy);
                acc += tmp[(size_t)sy * w + x] * k[(size_t)(i + radius)];
            }
            p[(size_t)y * w + x] = acc;
        }
    }
}

// ------------------------------- args -------------------------------

static void print_usage() {
    printf(
        "usage: sd-delight --model <hunyuan3d-delight-v2-0 dir> --image <in.png> --output <out.png>\n"
        "\n"
        "  De-lights an image with Hunyuan3D-Delight-v2-0 (SD2.1-class InstructPix2Pix).\n"
        "\n"
        "  --model  <dir>     diffusers directory (unet/ vae/ text_encoder/)\n"
        "  --image  <path>    input image (native resolution is preserved)\n"
        "  --output <path>    output PNG (same channel count as the input)\n"
        "  --alpha  <path>    optional RGBA/greyscale image supplying the subject mask\n"
        "\n"
        "  --mode   <m>       ratio (default) | direct\n"
        "                       ratio  = out = orig_native / upsample(lowpass(orig512/delit512))\n"
        "                                keeps ALL native detail; only shading is transferred\n"
        "                       direct = Tencent's behaviour: the 512 delit pixels, upsampled\n"
        "  --ratio-est <e>    auto (default) | mean | lum | pixel  -- the shading estimator\n"
        "                       auto  = blend mean/lum by conditioning (w = dl/(dl+k)); no\n"
        "                               manual per-image call. --ratio-lum-k tunes k.\n"
        "                       mean  = lowpass(orig)/lowpass(delit) PER CHANNEL (robust in very\n"
        "                               dark regions, but a 3-channel S can shift HUE there)\n"
        "                       lum   = as mean, but ONE achromatic scalar applied to all 3\n"
        "                               channels -- cannot shift hue; use on dark subjects\n"
        "                       pixel = lowpass(orig/delit)\n"
        "  --sigma  <f>       ratio lowpass sigma in 512-space pixels (default 12)\n"
        "  --smin/--smax <f>  clamp on the shading ratio (default 0.25 / 4.0)\n"
        "  --ratio-srgb       do the ratio in sRGB space (default: linear light)\n"
        "\n"
        "  --steps  <n>       default 50 (Tencent)\n"
        "  --seed   <n>       default 42 (Tencent)\n"
        "  --cfg    <f>       default 1.0 = Tencent (dehighlight_utils.py:26). NOTE: sd.cpp treats\n"
        "                       this as IMAGE guidance when it equals --img-cfg, not text guidance\n"
        "  --img-cfg <f>      default 1.0. Upstream says 1.5 but it is DEAD CODE there (diffusers\n"
        "                       gates CFG on guidance_scale>1.0, which Tencent sets to 1.0)\n"
        "  --size   <n>       delight working resolution (default 512)\n"
        "  --recorrect        apply Tencent's recorrect_rgb (default OFF for --mode ratio)\n"
        "  --recorrect-scale <f>  default 0.95\n"
        "  --bg <auto|black|white|none>  background key for mask derivation (default auto)\n"
        "  --type <f16|f32|default>  weight type (default f16, matching Tencent's torch.float16;\n"
        "                       'default' keeps each tensor's on-disk type -> ~4.3GB, the UNet is fp32)\n"
        "  --save-delit <p>   also write the raw %dx%d delit image (debug/A-B)\n"
        "  --threads <n>      default: physical core count\n"
        "  -v, --verbose\n",
        512, 512);
}

static bool parse_args(int argc, const char** argv, DelightParams& p) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next     = [&](const char* what) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", what);
                exit(1);
            }
            return argv[++i];
        };
        if (a == "--model") {
            p.model_path = next("--model");
        } else if (a == "--image") {
            p.image_path = next("--image");
        } else if (a == "--output" || a == "-o") {
            p.output_path = next("--output");
        } else if (a == "--alpha") {
            p.alpha_path = next("--alpha");
        } else if (a == "--save-delit") {
            p.save_delit_path = next("--save-delit");
        } else if (a == "--mode") {
            p.mode = next("--mode");
        } else if (a == "--ratio-est") {
            p.ratio_est = next("--ratio-est");
        } else if (a == "--ratio-lum-k") {
            p.lum_k = (float)atof(next("--ratio-lum-k"));
        } else if (a == "--steps") {
            p.steps = atoi(next("--steps"));
        } else if (a == "--seed") {
            p.seed = atoll(next("--seed"));
        } else if (a == "--cfg" || a == "--cfg-scale") {
            p.txt_cfg = (float)atof(next("--cfg"));
        } else if (a == "--img-cfg" || a == "--img-cfg-scale") {
            p.img_cfg = (float)atof(next("--img-cfg"));
        } else if (a == "--size") {
            p.size = atoi(next("--size"));
        } else if (a == "--sigma") {
            p.sigma = (float)atof(next("--sigma"));
        } else if (a == "--smin") {
            p.smin = (float)atof(next("--smin"));
        } else if (a == "--smax") {
            p.smax = (float)atof(next("--smax"));
        } else if (a == "--ratio-srgb") {
            p.ratio_srgb = true;
        } else if (a == "--recorrect") {
            p.recorrect = true;
        } else if (a == "--recorrect-scale") {
            p.recorrect_scale = (float)atof(next("--recorrect-scale"));
        } else if (a == "--bg") {
            p.bg = next("--bg");
        } else if (a == "--threads") {
            p.n_threads = atoi(next("--threads"));
        } else if (a == "--type") {
            std::string t = next("--type");
            if (t == "f16") {
                p.wtype = SD_TYPE_F16;
            } else if (t == "f32") {
                p.wtype = SD_TYPE_F32;
            } else if (t == "default") {
                p.wtype = SD_TYPE_COUNT;  // whatever each tensor is on disk
            } else {
                fprintf(stderr, "error: --type must be f16, f32 or default\n");
                return false;
            }
        } else if (a == "-v" || a == "--verbose") {
            p.verbose = true;
        } else if (a == "-h" || a == "--help") {
            print_usage();
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument '%s'\n", a.c_str());
            print_usage();
            return false;
        }
    }
    if (p.model_path.empty() || p.image_path.empty() || p.output_path.empty()) {
        fprintf(stderr, "error: --model, --image and --output are required\n");
        print_usage();
        return false;
    }
    if (p.mode != "ratio" && p.mode != "direct") {
        fprintf(stderr, "error: --mode must be 'ratio' or 'direct'\n");
        return false;
    }
    if (p.ratio_est != "mean" && p.ratio_est != "pixel" && p.ratio_est != "lum" && p.ratio_est != "auto") {
        fprintf(stderr, "error: --ratio-est must be 'mean' or 'pixel'\n");
        return false;
    }
    return true;
}

static void log_cb(enum sd_log_level_t level, const char* text, void* data) {
    DelightParams* p = (DelightParams*)data;
    if (!p->verbose && level == SD_LOG_DEBUG) {
        return;
    }
    fputs(text, level == SD_LOG_ERROR ? stderr : stdout);
    fflush(level == SD_LOG_ERROR ? stderr : stdout);
}

int main(int argc, const char** argv) {
    DelightParams p;
    if (!parse_args(argc, argv, p)) {
        return 1;
    }
    sd_set_log_callback(log_cb, (void*)&p);

    // ---------------- load the native image ----------------
    int nw = 0, nh = 0, nc_file = 0;
    unsigned char* native = stbi_load(p.image_path.c_str(), &nw, &nh, &nc_file, 4);
    if (native == nullptr) {
        fprintf(stderr, "error: failed to load '%s': %s\n", p.image_path.c_str(), stbi_failure_reason());
        return 1;
    }
    const bool file_has_alpha = (nc_file == 4 || nc_file == 2);
    printf("[delight] input %dx%d (%d ch in file, alpha=%s)\n", nw, nh, nc_file, file_has_alpha ? "yes" : "no");

    const size_t npx = (size_t)nw * nh;

    // ---------------- derive the subject mask ----------------
    // Priority: --alpha > the file's own alpha > background key.
    std::vector<float> mask_native(npx, 1.0f);
    std::string mask_src = "none (whole image)";

    if (!p.alpha_path.empty()) {
        int aw = 0, ah = 0, ac = 0;
        unsigned char* ad = stbi_load(p.alpha_path.c_str(), &aw, &ah, &ac, 4);
        if (ad == nullptr) {
            fprintf(stderr, "error: failed to load --alpha '%s': %s\n", p.alpha_path.c_str(), stbi_failure_reason());
            stbi_image_free(native);
            return 1;
        }
        if (aw != nw || ah != nh) {
            fprintf(stderr, "error: --alpha is %dx%d but --image is %dx%d\n", aw, ah, nw, nh);
            stbi_image_free(ad);
            stbi_image_free(native);
            return 1;
        }
        // An RGBA alpha source uses its alpha; a greyscale one uses luma.
        const bool a_has_alpha = (ac == 4 || ac == 2);
        for (size_t i = 0; i < npx; i++) {
            mask_native[i] = a_has_alpha ? ad[i * 4 + 3] / 255.0f : ad[i * 4 + 0] / 255.0f;
        }
        stbi_image_free(ad);
        mask_src = "--alpha " + p.alpha_path;
    } else if (file_has_alpha) {
        for (size_t i = 0; i < npx; i++) {
            mask_native[i] = native[i * 4 + 3] / 255.0f;
        }
        mask_src = "input alpha channel";
    } else if (p.bg != "none") {
        // Background key. "auto" samples the four corners and only keys if they agree and are
        // near-black or near-white; anything else means we cannot tell, so we do not guess.
        std::string key = p.bg;
        if (key == "auto") {
            const size_t corners[4] = {0, (size_t)(nw - 1), (size_t)(nh - 1) * nw, npx - 1};
            float lo = 1.0f, hi = 0.0f;
            for (size_t c : corners) {
                float l = (0.2126f * native[c * 4 + 0] + 0.7152f * native[c * 4 + 1] + 0.0722f * native[c * 4 + 2]) / 255.0f;
                lo      = std::fmin(lo, l);
                hi      = std::fmax(hi, l);
            }
            if (hi < 0.06f) {
                key = "black";
            } else if (lo > 0.94f) {
                key = "white";
            } else {
                key = "none";
                printf("[delight] --bg auto: corners are not uniformly black or white "
                       "(luma %.3f..%.3f); no background key applied\n",
                       lo, hi);
            }
        }
        if (key == "black" || key == "white") {
            const float ref = (key == "white") ? 1.0f : 0.0f;
            const float tol = 0.06f;
            for (size_t i = 0; i < npx; i++) {
                float l = (0.2126f * native[i * 4 + 0] + 0.7152f * native[i * 4 + 1] + 0.0722f * native[i * 4 + 2]) / 255.0f;
                mask_native[i] = (std::fabs(l - ref) <= tol) ? 0.0f : 1.0f;
            }
            mask_src = "--bg " + key + " key";
        } else if (key != "none") {
            fprintf(stderr, "error: --bg must be auto, black, white or none\n");
            stbi_image_free(native);
            return 1;
        }
    }
    printf("[delight] subject mask: %s\n", mask_src.c_str());

    // ---------------- build the 512 delight input ----------------
    const int S = p.size;
    std::vector<unsigned char> small_rgba((size_t)S * S * 4);
    if (!stbir_resize_uint8(native, nw, nh, 0, small_rgba.data(), S, S, 0, 4)) {
        fprintf(stderr, "error: resize to %dx%d failed\n", S, S);
        stbi_image_free(native);
        return 1;
    }
    std::vector<float> mask_small((size_t)S * S, 1.0f);
    {
        std::vector<unsigned char> m8(npx), ms8((size_t)S * S);
        for (size_t i = 0; i < npx; i++) {
            m8[i] = (unsigned char)clampf(mask_native[i] * 255.0f + 0.5f, 0.0f, 255.0f);
        }
        stbir_resize_uint8(m8.data(), nw, nh, 0, ms8.data(), S, S, 0, 1);
        for (size_t i = 0; i < (size_t)S * S; i++) {
            mask_small[i] = ms8[i] / 255.0f;
        }
    }

    // Tencent: 3x3 alpha erosion, then fill the eroded-out background WHITE.
    // The erosion pulls the mask in so that anti-aliased silhouette pixels (which carry background
    // colour) are treated as background rather than subject.
    std::vector<float> mask_eroded = mask_small;
    {
        std::vector<float> src = mask_small;
        for (int y = 0; y < S; y++) {
            for (int x = 0; x < S; x++) {
                float mn = 1.0f;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int sx = x + dx, sy = y + dy;
                        if (sx < 0 || sy < 0 || sx >= S || sy >= S) {
                            continue;
                        }
                        mn = std::fmin(mn, src[(size_t)sy * S + sx]);
                    }
                }
                mask_eroded[(size_t)y * S + x] = mn;
            }
        }
    }

    // rgb_in = what the network actually sees (white-filled background).
    std::vector<unsigned char> rgb_in((size_t)S * S * 3);
    for (size_t i = 0; i < (size_t)S * S; i++) {
        const bool is_bg = mask_eroded[i] <= 0.5f;
        for (int k = 0; k < 3; k++) {
            rgb_in[i * 3 + k] = is_bg ? 255 : small_rgba[i * 4 + k];
        }
    }

    // ---------------- run the model ----------------
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);
    ctx_params.model_path      = p.model_path.c_str();
    ctx_params.vae_decode_only = false;  // ip2p encodes the reference image
    ctx_params.wtype           = p.wtype;
    if (p.n_threads > 0) {
        ctx_params.n_threads = p.n_threads;
    }
    // The delight scheduler config is v_prediction. Set it explicitly: it is what the model IS, and
    // it also skips is_using_v_parameterization_for_sd2()'s probe (the diffusers loader never reads
    // scheduler_config.json, so without this the probe would run).
    ctx_params.prediction = V_PRED;

    sd_ctx_t* ctx = new_sd_ctx(&ctx_params);
    if (ctx == nullptr) {
        fprintf(stderr, "error: new_sd_ctx failed for '%s'\n", p.model_path.c_str());
        stbi_image_free(native);
        return 1;
    }

    sd_image_t ref_image = {(uint32_t)S, (uint32_t)S, 3, rgb_in.data()};

    sd_img_gen_params_t gen;
    sd_img_gen_params_init(&gen);
    gen.prompt          = "";  // Tencent: prompt=""
    gen.negative_prompt = "";
    gen.ref_images      = &ref_image;
    gen.ref_images_count = 1;
    gen.width           = S;
    gen.height          = S;
    gen.seed            = p.seed;
    gen.batch_count     = 1;
    // Tencent overrides DDIM -> EulerAncestralDiscreteScheduler.
    gen.sample_params.sample_method     = EULER_A_SAMPLE_METHOD;
    gen.sample_params.scheduler         = DISCRETE_SCHEDULER;
    gen.sample_params.sample_steps      = p.steps;
    gen.sample_params.guidance.txt_cfg  = p.txt_cfg;
    gen.sample_params.guidance.img_cfg  = p.img_cfg;

    printf("[delight] running: %d steps, seed %" PRId64 ", txt_cfg %.2f, img_cfg %.2f, euler_a, v-pred\n",
           p.steps, (int64_t)p.seed, p.txt_cfg, p.img_cfg);

    sd_image_t* results = generate_image(ctx, &gen);
    if (results == nullptr || results[0].data == nullptr) {
        fprintf(stderr, "error: generate_image failed\n");
        free_sd_ctx(ctx);
        stbi_image_free(native);
        return 1;
    }
    if (results[0].width != (uint32_t)S || results[0].height != (uint32_t)S || results[0].channel != 3) {
        fprintf(stderr, "error: unexpected result %ux%ux%u\n", results[0].width, results[0].height, results[0].channel);
        free(results[0].data);
        free(results);
        free_sd_ctx(ctx);
        stbi_image_free(native);
        return 1;
    }

    // delit512 in [0,1]
    std::vector<float> delit((size_t)S * S * 3);
    for (size_t i = 0; i < (size_t)S * S * 3; i++) {
        delit[i] = results[0].data[i] / 255.0f;
    }
    free(results[0].data);
    free(results);

    // The model is done. Free it now: at ~2.6GB (F16) it can never be co-resident with the
    // geometry stage, and this is a pre-process -- the card must be empty before geometry starts.
    free_sd_ctx(ctx);
    ctx = nullptr;

    if (!p.save_delit_path.empty()) {
        std::vector<unsigned char> d8((size_t)S * S * 3);
        for (size_t i = 0; i < (size_t)S * S * 3; i++) {
            d8[i] = (unsigned char)clampf(delit[i] * 255.0f + 0.5f, 0.0f, 255.0f);
        }
        stbi_write_png(p.save_delit_path.c_str(), S, S, 3, d8.data(), S * 3);
        printf("[delight] wrote raw delit -> %s\n", p.save_delit_path.c_str());
    }

    // orig512 as the net saw it (white-filled), in [0,1].
    std::vector<float> orig_s((size_t)S * S * 3);
    for (size_t i = 0; i < (size_t)S * S * 3; i++) {
        orig_s[i] = rgb_in[i] / 255.0f;
    }

    // ---------------- optional recorrect_rgb ----------------
    // Tencent maps the DELIT output's per-channel mean/std onto the ORIGINAL's, then keeps whichever
    // of {delit, corrected} is closer (MSE) to the original. It is a colour-drift guard for their
    // direct-pixel use. For --mode ratio it is counter-productive: it deliberately pulls the result's
    // statistics back toward the LIT original, which is exactly the shading the ratio is trying to
    // measure. Hence: default OFF, and OFF is the right default for ratio.
    if (p.recorrect) {
        std::vector<float> corrected = delit;
        for (int c = 0; c < 3; c++) {
            double s_sum = 0, t_sum = 0;
            size_t n = 0;
            for (size_t i = 0; i < (size_t)S * S; i++) {
                if (mask_eroded[i] > 0.5f) {
                    s_sum += delit[i * 3 + c];
                    t_sum += orig_s[i * 3 + c];
                    n++;
                }
            }
            if (n < 2) {
                continue;
            }
            double s_mean = s_sum / (double)n, t_mean = t_sum / (double)n;
            double s_var = 0, t_var = 0;
            for (size_t i = 0; i < (size_t)S * S; i++) {
                if (mask_eroded[i] > 0.5f) {
                    double ds = delit[i * 3 + c] - s_mean, dt = orig_s[i * 3 + c] - t_mean;
                    s_var += ds * ds;
                    t_var += dt * dt;
                }
            }
            // torch.std is the unbiased (n-1) estimator; match it.
            double s_std = std::sqrt(s_var / (double)(n - 1));
            double t_std = std::sqrt(t_var / (double)(n - 1));
            if (s_std < 1e-6) {
                continue;
            }
            for (size_t i = 0; i < (size_t)S * S; i++) {
                double v = (delit[i * 3 + c] - p.recorrect_scale * s_mean) * (t_std / s_std) +
                           p.recorrect_scale * t_mean;
                corrected[i * 3 + c] = clampf((float)v, 0.0f, 1.0f);
            }
        }
        double src_mse = 0, mod_mse = 0;
        for (size_t i = 0; i < (size_t)S * S * 3; i++) {
            double a = delit[i] - orig_s[i];
            double b = corrected[i] - orig_s[i];
            src_mse += a * a;
            mod_mse += b * b;
        }
        if (src_mse < mod_mse) {
            printf("[delight] recorrect_rgb: raw delit is closer to the input; keeping raw\n");
        } else {
            printf("[delight] recorrect_rgb: applied\n");
            delit = corrected;
        }
    }

    // ---------------- compose ----------------
    const int out_ch = file_has_alpha ? 4 : 3;
    std::vector<unsigned char> out((size_t)npx * out_ch);

    if (p.mode == "direct") {
        // Tencent's behaviour: the delit pixels themselves, upsampled to native.
        std::vector<unsigned char> d8((size_t)S * S * 3), up((size_t)npx * 3);
        for (size_t i = 0; i < (size_t)S * S * 3; i++) {
            d8[i] = (unsigned char)clampf(delit[i] * 255.0f + 0.5f, 0.0f, 255.0f);
        }
        stbir_resize_uint8(d8.data(), S, S, 0, up.data(), nw, nh, 0, 3);
        for (size_t i = 0; i < npx; i++) {
            const bool is_bg = mask_native[i] <= 0.5f;
            for (int k = 0; k < 3; k++) {
                // Outside the subject, keep the ORIGINAL background rather than the net's idea of it.
                out[i * out_ch + k] = is_bg ? native[i * 4 + k] : up[i * 3 + k];
            }
            if (out_ch == 4) {
                out[i * 4 + 3] = native[i * 4 + 3];
            }
        }
    } else {
        // ratio: S = lowpass(orig512 / delit512), then out = orig_native / upsample(S).
        const bool lin = !p.ratio_srgb;
        std::vector<std::vector<float>> Sp(3, std::vector<float>((size_t)S * S, 1.0f));
        std::vector<float> wplane((size_t)S * S, 0.0f);
        for (size_t i = 0; i < (size_t)S * S; i++) {
            wplane[i] = (mask_eroded[i] > 0.5f) ? 1.0f : 0.0f;
        }

        // Mask-normalised lowpass: blur(x*m)/blur(m). Without the normalisation the white background
        // would bleed a spurious shading estimate across the silhouette.
        std::vector<float> wblur = wplane;
        blur_plane(wblur, S, S, p.sigma);
        auto normalise = [&](std::vector<float>& plane) {
            blur_plane(plane, S, S, p.sigma);
            for (size_t i = 0; i < (size_t)S * S; i++) {
                plane[i] = (wblur[i] > 1e-4f) ? plane[i] / wblur[i] : 0.0f;
            }
        };

        const float eps = 1e-3f;
        if (p.ratio_est == "pixel") {
            // S = lowpass(orig/delit): the per-pixel ratio, then smoothed.
            for (size_t i = 0; i < (size_t)S * S; i++) {
                for (int c = 0; c < 3; c++) {
                    float o = orig_s[i * 3 + c], d = delit[i * 3 + c];
                    if (lin) {
                        o = srgb_to_linear(o);
                        d = srgb_to_linear(d);
                    }
                    float r = (d > eps) ? (o / d) : 1.0f;
                    if (!std::isfinite(r)) {
                        r = 1.0f;
                    }
                    Sp[c][i] = clampf(r, p.smin, p.smax) * wplane[i];  // pre-multiplied for the blur
                }
            }
            for (int c = 0; c < 3; c++) {
                normalise(Sp[c]);
                for (size_t i = 0; i < (size_t)S * S; i++) {
                    Sp[c][i] = (wblur[i] > 1e-4f) ? clampf(Sp[c][i], p.smin, p.smax) : 1.0f;
                }
            }
        } else {
            // S = lowpass(orig)/lowpass(delit): a ratio of two SMOOTH quantities, so it never divides
            // by a single near-black pixel. Better conditioned than the per-pixel form wherever the
            // subject is very dark (a black bearskin / dark uniform is exactly that case), and it
            // recovers the same S under the multiplicative model orig = S*albedo with S slowly varying.
            std::vector<std::vector<float>> op(3, std::vector<float>((size_t)S * S)), dp(3, std::vector<float>((size_t)S * S));
            for (size_t i = 0; i < (size_t)S * S; i++) {
                for (int c = 0; c < 3; c++) {
                    float o = orig_s[i * 3 + c], d = delit[i * 3 + c];
                    if (lin) {
                        o = srgb_to_linear(o);
                        d = srgb_to_linear(d);
                    }
                    op[c][i] = o * wplane[i];
                    dp[c][i] = d * wplane[i];
                }
            }
            for (int c = 0; c < 3; c++) {
                normalise(op[c]);
                normalise(dp[c]);
            }
            if (p.ratio_est == "lum" || p.ratio_est == "auto") {
                // S = ONE achromatic scalar per pixel, applied to all three channels.
                // Under near-white light shading is achromatic: orig = S*albedo with S a SCALAR. A
                // PER-CHANNEL S instead lets the delight model's own colour drift ride out as a HUE
                // shift -- catastrophic exactly where the albedo is near-black (boots, bearskin),
                // because all three lowpassed channels are tiny there, so their three independent
                // ratios are noise-dominated and diverge. MEASURED on the soldier: saturation
                // 0.183 -> 0.357 (+94.7%) at input luma 0.02-0.05, per-channel spread p90 = 4.0x,
                // and 19.2% of dark pixels pinned to the smin clamp (= the visible BANDS; only 5.2%
                // over the whole subject). Owner spotted it by eye as "rainbowy bands on boots + hat
                // -- both black items". One scalar CANNOT shift hue, by construction.
                // "auto" (the DEFAULT) blends the two ESTIMATORS by how well-conditioned the
                // per-channel one is, so nothing here needs a human decision:
                //   w = dl / (dl + k)   ->  w->1 where the subject is bright  (per-channel S: also
                //                           removes a CHROMATIC light cast, which lum cannot)
                //                       ->  w->0 as dl->0                     (achromatic S: hue-safe)
                // The noise in a ratio o/d grows like 1/d, so dl IS the conditioning of the estimate --
                // this weights by the actual thing that fails, not by an invented brightness threshold.
                // k is self-calibrating: a fraction of THIS subject's own mean delit luminance, so it
                // carries across subjects/exposures without retuning (--ratio-lum-k overrides).
                // ratio_est=="lum" pins w=0 (pure achromatic) for A/B.
                double acc = 0.0;
                size_t nacc = 0;
                std::vector<float> dlum((size_t)S * S, 0.0f);
                for (size_t i = 0; i < (size_t)S * S; i++) {
                    dlum[i] = 0.2126f * dp[0][i] + 0.7152f * dp[1][i] + 0.0722f * dp[2][i];
                    // wplane (the eroded SUBJECT mask), NOT wblur (the blurred mask): wblur > 1e-4
                    // reaches far outside the silhouette into the blur skirt, so it averaged over
                    // 48.4% of the frame instead of the subject's 21.3% and read dl_mean 12% high.
                    // k is supposed to calibrate to THIS SUBJECT's luminance -- so average the subject.
                    if (wplane[i] > 0.5f) {
                        acc += dlum[i];
                        nacc++;
                    }
                }
                const float dl_mean = nacc ? (float)(acc / (double)nacc) : 1.0f;
                const float k       = std::max(p.lum_k * dl_mean, 1e-6f);
                const bool pin_lum  = (p.ratio_est == "lum");
                for (size_t i = 0; i < (size_t)S * S; i++) {
                    const float ol = 0.2126f * op[0][i] + 0.7152f * op[1][i] + 0.0722f * op[2][i];
                    const float dl = dlum[i];
                    float r_lum    = (dl > eps) ? (ol / dl) : 1.0f;
                    if (!std::isfinite(r_lum)) {
                        r_lum = 1.0f;
                    }
                    const float w = pin_lum ? 0.0f : (dl / (dl + k));
                    for (int c = 0; c < 3; c++) {
                        float r_c = (dp[c][i] > eps) ? (op[c][i] / dp[c][i]) : 1.0f;
                        if (!std::isfinite(r_c)) {
                            r_c = 1.0f;
                        }
                        float r = r_lum + w * (r_c - r_lum);
                        if (wblur[i] <= 1e-4f) {
                            r = 1.0f;
                        }
                        Sp[c][i] = clampf(r, p.smin, p.smax);
                    }
                }
            } else {
                for (int c = 0; c < 3; c++) {
                    for (size_t i = 0; i < (size_t)S * S; i++) {
                        float r = (dp[c][i] > eps) ? (op[c][i] / dp[c][i]) : 1.0f;
                        if (!std::isfinite(r) || wblur[i] <= 1e-4f) {
                            r = 1.0f;
                        }
                        Sp[c][i] = clampf(r, p.smin, p.smax);
                    }
                }
            }
        }

        // Upsample the shading 512 -> native. Near-lossless: it is low-frequency by construction.
        std::vector<float> Sup((size_t)npx * 3);
        {
            std::vector<float> packed((size_t)S * S * 3);
            for (size_t i = 0; i < (size_t)S * S; i++) {
                for (int c = 0; c < 3; c++) {
                    packed[i * 3 + c] = Sp[c][i];
                }
            }
            stbir_resize_float(packed.data(), S, S, 0, Sup.data(), nw, nh, 0, 3);
        }

        for (size_t i = 0; i < npx; i++) {
            const bool is_bg = mask_native[i] <= 0.5f;
            for (int k = 0; k < 3; k++) {
                if (is_bg) {
                    out[i * out_ch + k] = native[i * 4 + k];
                    continue;
                }
                float o = native[i * 4 + k] / 255.0f;
                if (lin) {
                    o = srgb_to_linear(o);
                }
                float s = clampf(Sup[i * 3 + k], p.smin, p.smax);
                float v = o / s;
                if (lin) {
                    v = linear_to_srgb(clampf(v, 0.0f, 1.0f));
                }
                out[i * out_ch + k] = (unsigned char)clampf(v * 255.0f + 0.5f, 0.0f, 255.0f);
            }
            if (out_ch == 4) {
                out[i * 4 + 3] = native[i * 4 + 3];
            }
        }
    }

    if (!stbi_write_png(p.output_path.c_str(), nw, nh, out_ch, out.data(), nw * out_ch)) {
        fprintf(stderr, "error: failed to write '%s'\n", p.output_path.c_str());
        stbi_image_free(native);
        return 1;
    }
    if (p.mode == "ratio") {
        printf("[delight] ratio: est=%s space=%s sigma=%.1f clamp=[%.2f,%.2f] recorrect=%s\n",
               p.ratio_est.c_str(), p.ratio_srgb ? "srgb" : "linear", p.sigma, p.smin, p.smax,
               p.recorrect ? "on" : "off");
    }
    printf("[delight] wrote %dx%d (%d ch, mode=%s) -> %s\n", nw, nh, out_ch, p.mode.c_str(), p.output_path.c_str());

    stbi_image_free(native);
    return 0;
}
