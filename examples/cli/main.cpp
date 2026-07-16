#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// #include "preprocessing.hpp"
#include "stable-diffusion.h"

#include "common/common.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "image_metadata.h"

namespace fs = std::filesystem;

const char* previews_str[] = {
    "none",
    "proj",
    "tae",
    "vae",
};

std::regex format_specifier_regex("(?:[^%]|^)(?:%%)*(%\\d{0,3}d)");

struct SDCliParams {
    SDMode mode             = IMG_GEN;
    std::string output_path = "output.png";
    int output_begin_idx    = -1;
    std::string image_path;
    std::string metadata_format = "text";

    bool verbose          = false;
    bool canny_preprocess = false;
    bool convert_name     = false;

    // nvfp4-twolevel diffusion imatrix:
    //   --collect-imatrix <out.gguf>  (img_gen): enable DiT activation collection,
    //                                   write imatrix gguf after generation.
    //   --imatrix <in.gguf>           (convert): feed AWQ importance into nvfp4 quant.
    std::string collect_imatrix_path;
    std::string imatrix_path;

    preview_t preview_method = PREVIEW_NONE;
    int preview_interval     = 1;
    std::string preview_path = "preview.png";
    int preview_fps          = 16;
    bool taesd_preview       = false;
    bool preview_noisy       = false;
    bool color               = false;
    bool metadata_raw        = false;
    bool metadata_brief      = false;
    bool metadata_all        = false;

    bool normal_exit = false;

    ArgOptions get_options() {
        ArgOptions options;

        options.string_options = {
            {"-o",
             "--output",
             "path to write result image to. you can use printf-style %d format specifiers for image sequences (default: ./output.png) (eg. output_%03d.png). Single-file video outputs support .avi, .webm, and animated .webp",
             &output_path},
            {"",
             "--image",
             "path to the image to inspect (for metadata mode)",
             &image_path},
            {"",
             "--metadata-format",
             "metadata output format, one of [text, json] (default: text)",
             &metadata_format},
            {"",
             "--preview-path",
             "path to write preview image to (default: ./preview.png). Multi-frame previews support .avi, .webm, and animated .webp",
             &preview_path},
            {"",
             "--collect-imatrix",
             "(img_gen) collect per-DiT-Linear activation importance during generation and write an imatrix gguf to this path",
             &collect_imatrix_path},
            {"",
             "--imatrix",
             "(convert) path to an imatrix gguf to drive AWQ-style importance for nvfp4 quantization",
             &imatrix_path},
        };

        options.int_options = {
            {"",
             "--preview-interval",
             "interval in denoising steps between consecutive updates of the image preview file (default is 1, meaning updating at every step)",
             &preview_interval},
            {"",
             "--output-begin-idx",
             "starting index for output image sequence, must be non-negative (default 0 if specified %d in output path, 1 otherwise)",
             &output_begin_idx},
        };

        options.bool_options = {
            {"",
             "--canny",
             "apply canny preprocessor (edge detection)",
             true, &canny_preprocess},
            {"",
             "--convert-name",
             "convert tensor name (for convert mode)",
             true, &convert_name},
            {"-v",
             "--verbose",
             "print extra info",
             true, &verbose},
            {"",
             "--color",
             "colors the logging tags according to level",
             true, &color},
            {"",
             "--taesd-preview-only",
             std::string("prevents usage of taesd for decoding the final image. (for use with --preview ") + previews_str[PREVIEW_TAE] + ")",
             true, &taesd_preview},
            {"",
             "--preview-noisy",
             "enables previewing noisy inputs of the models rather than the denoised outputs",
             true, &preview_noisy},
            {"",
             "--metadata-raw",
             "include raw hex previews for unparsed metadata payloads",
             true, &metadata_raw},
            {"",
             "--metadata-brief",
             "truncate long metadata text values in text output",
             true, &metadata_brief},
            {"",
             "--metadata-all",
             "include structural/container entries such as IHDR, IDAT, and non-metadata JPEG segments",
             true, &metadata_all},

        };

        auto on_mode_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* mode_c_str = argv[index];
            if (mode_c_str != nullptr) {
                int mode_found = -1;
                for (int i = 0; i < MODE_COUNT; i++) {
                    if (!strcmp(mode_c_str, modes_str[i])) {
                        mode_found = i;
                    }
                }
                if (mode_found == -1) {
                    LOG_ERROR("error: invalid mode %s, must be one of [%s]\n",
                              mode_c_str, SD_ALL_MODES_STR);
                    exit(1);
                }
                mode = (SDMode)mode_found;
            }
            return 1;
        };

        auto on_preview_arg = [&](int argc, const char** argv, int index) {
            if (++index >= argc) {
                return -1;
            }
            const char* preview = argv[index];
            int preview_found   = -1;
            for (int m = 0; m < PREVIEW_COUNT; m++) {
                if (!strcmp(preview, previews_str[m])) {
                    preview_found = m;
                }
            }
            if (preview_found == -1) {
                LOG_ERROR("error: preview method %s", preview);
                return -1;
            }
            preview_method = (preview_t)preview_found;
            return 1;
        };

        auto on_help_arg = [&](int argc, const char** argv, int index, bool& valid) {
            normal_exit = true;
            valid       = true;
            return -1;
        };

        options.manual_options = {
            {"-M",
             "--mode",
             "run mode, one of [img_gen, vid_gen, upscale, convert, metadata], default: img_gen",
             on_mode_arg},
            {"",
             "--preview",
             std::string("preview method. must be one of the following [") + previews_str[0] + ", " + previews_str[1] + ", " + previews_str[2] + ", " + previews_str[3] + "] (default is " + previews_str[PREVIEW_NONE] + ")",
             on_preview_arg},
            {"-h",
             "--help",
             "show this help message and exit",
             on_help_arg},
        };

        return options;
    };

    bool resolve() {
        if (mode == CONVERT) {
            if (output_path == "output.png") {
                output_path = "output.gguf";
            }
        }
        return true;
    }

    bool validate() {
        if (mode != METADATA) {
            if (output_path.length() == 0) {
                LOG_ERROR("error: the following arguments are required: output_path");
                return false;
            }
        } else {
            if (image_path.empty()) {
                LOG_ERROR("error: metadata mode needs an image path (--image)");
                return false;
            }
            if (metadata_format != "text" && metadata_format != "json") {
                LOG_ERROR("error: invalid metadata format %s, must be one of [text, json]",
                          metadata_format.c_str());
                return false;
            }
        }
        return true;
    }

    bool resolve_and_validate() {
        if (!resolve()) {
            return false;
        }
        if (!validate()) {
            return false;
        }
        return true;
    }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "SDCliParams {\n"
            << "  mode: " << modes_str[mode] << ",\n"
            << "  output_path: \"" << output_path << "\",\n"
            << "  image_path: \"" << image_path << "\",\n"
            << "  metadata_format: \"" << metadata_format << "\",\n"
            << "  verbose: " << (verbose ? "true" : "false") << ",\n"
            << "  color: " << (color ? "true" : "false") << ",\n"
            << "  canny_preprocess: " << (canny_preprocess ? "true" : "false") << ",\n"
            << "  convert_name: " << (convert_name ? "true" : "false") << ",\n"
            << "  preview_method: " << previews_str[preview_method] << ",\n"
            << "  preview_interval: " << preview_interval << ",\n"
            << "  preview_path: \"" << preview_path << "\",\n"
            << "  preview_fps: " << preview_fps << ",\n"
            << "  taesd_preview: " << (taesd_preview ? "true" : "false") << ",\n"
            << "  preview_noisy: " << (preview_noisy ? "true" : "false") << ",\n"
            << "  metadata_raw: " << (metadata_raw ? "true" : "false") << ",\n"
            << "  metadata_brief: " << (metadata_brief ? "true" : "false") << ",\n"
            << "  metadata_all: " << (metadata_all ? "true" : "false") << "\n"
            << "}";
        return oss.str();
    }
};

void print_usage(int argc, const char* argv[], const std::vector<ArgOptions>& options_list) {
    std::cout << version_string() << "\n";
    std::cout << "Usage: " << argv[0] << " [options]\n\n";
    std::cout << "CLI Options:\n";
    options_list[0].print();
    std::cout << "\nContext Options:\n";
    options_list[1].print();
    std::cout << "\nGeneration Options:\n";
    options_list[2].print();
}

void parse_args(int argc, const char** argv, SDCliParams& cli_params, SDContextParams& ctx_params, SDGenerationParams& gen_params) {
    std::vector<ArgOptions> options_vec = {cli_params.get_options(), ctx_params.get_options(), gen_params.get_options()};

    if (!parse_options(argc, argv, options_vec)) {
        print_usage(argc, argv, options_vec);
        exit(cli_params.normal_exit ? 0 : 1);
    }

    bool valid = cli_params.resolve_and_validate();
    if (valid && cli_params.mode != METADATA) {
        valid = ctx_params.resolve_and_validate(cli_params.mode) &&
                gen_params.resolve_and_validate(cli_params.mode,
                                                ctx_params.lora_model_dir,
                                                ctx_params.hires_upscalers_dir);
    }

    if (!valid) {
        print_usage(argc, argv, options_vec);
        exit(1);
    }
}

void sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    SDCliParams* cli_params = (SDCliParams*)data;
    log_print(level, log, cli_params->verbose, cli_params->color);
}

bool load_images_from_dir(const std::string dir,
                          std::vector<SDImageOwner>& images,
                          int expected_width  = 0,
                          int expected_height = 0,
                          int max_image_num   = 0,
                          bool verbose        = false) {
    if (!fs::exists(dir) || !fs::is_directory(dir)) {
        LOG_ERROR("'%s' is not a valid directory\n", dir.c_str());
        return false;
    }

    std::vector<fs::directory_entry> entries;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            entries.push_back(entry);
        }
    }

    std::sort(entries.begin(), entries.end(),
              [](const fs::directory_entry& a, const fs::directory_entry& b) {
                  return a.path().filename().string() < b.path().filename().string();
              });

    for (const auto& entry : entries) {
        std::string path = entry.path().string();
        std::string ext  = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".webp") {
            LOG_DEBUG("load image %zu from '%s'", images.size(), path.c_str());
            int width             = 0;
            int height            = 0;
            uint8_t* image_buffer = load_image_from_file(path.c_str(), width, height, expected_width, expected_height);
            if (image_buffer == nullptr) {
                LOG_ERROR("load image from '%s' failed", path.c_str());
                return false;
            }

            images.emplace_back(sd_image_t{(uint32_t)width,
                                           (uint32_t)height,
                                           3,
                                           image_buffer});

            if (max_image_num > 0 && static_cast<int>(images.size()) >= max_image_num) {
                break;
            }
        }
    }
    return true;
}

void step_callback(int step, int frame_count, sd_image_t* image, bool is_noisy, void* data) {
    (void)step;
    (void)is_noisy;
    SDCliParams* cli_params = (SDCliParams*)data;
    // is_noisy is set to true if the preview corresponds to noisy latents, false if it's denoised latents
    // unused in this app, it will either be always noisy or always denoised here
    if (frame_count == 1) {
        if (!write_image_to_file(cli_params->preview_path,
                                 image->data,
                                 image->width,
                                 image->height,
                                 image->channel)) {
            LOG_ERROR("save preview image to '%s' failed", cli_params->preview_path.c_str());
        }
    } else {
        if (create_video_from_sd_images(cli_params->preview_path.c_str(), image, frame_count, cli_params->preview_fps) != 0) {
            LOG_ERROR("save preview video to '%s' failed", cli_params->preview_path.c_str());
        }
    }
}

std::string format_frame_idx(std::string pattern, int frame_idx) {
    std::smatch match;
    std::string result = pattern;
    while (std::regex_search(result, match, format_specifier_regex)) {
        std::string specifier = match.str(1);
        char buffer[32];
        snprintf(buffer, sizeof(buffer), specifier.c_str(), frame_idx);
        result.replace(match.position(1), match.length(1), buffer);
    }

    // Then replace all '%%' with '%'
    size_t pos = 0;
    while ((pos = result.find("%%", pos)) != std::string::npos) {
        result.replace(pos, 2, "%");
        pos += 1;
    }
    return result;
}

static fs::path get_video_audio_sidecar_path(const SDCliParams& cli_params) {
    fs::path out_path     = cli_params.output_path;
    fs::path base_path    = out_path;
    fs::path ext          = out_path.has_extension() ? out_path.extension() : fs::path{};
    std::string ext_lower = ext.string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    const EncodedImageFormat output_format = encoded_image_format_from_path(out_path.string());
    if (!ext.empty()) {
        if (output_format == EncodedImageFormat::JPEG ||
            output_format == EncodedImageFormat::PNG ||
            output_format == EncodedImageFormat::WEBP ||
            ext_lower == ".avi" ||
            ext_lower == ".webm") {
            base_path.replace_extension();
        }
    }
    base_path += ".wav";
    return base_path;
}

bool save_results(const SDCliParams& cli_params,
                  const SDContextParams& ctx_params,
                  const SDGenerationParams& gen_params,
                  sd_image_t* results,
                  int num_results,
                  const sd_audio_t* generated_audio = nullptr) {
    if (results == nullptr || num_results <= 0) {
        return false;
    }

    namespace fs      = std::filesystem;
    fs::path out_path = cli_params.output_path;

    if (!out_path.parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(out_path.parent_path(), ec);
        if (ec) {
            LOG_ERROR("failed to create directory '%s': %s",
                      out_path.parent_path().string().c_str(), ec.message().c_str());
            return false;
        }
    }

    fs::path base_path = out_path;
    fs::path ext       = out_path.has_extension() ? out_path.extension() : fs::path{};

    std::string ext_lower = ext.string();
    std::transform(ext_lower.begin(), ext_lower.end(), ext_lower.begin(), ::tolower);
    const EncodedImageFormat output_format = encoded_image_format_from_path(out_path.string());
    if (!ext.empty()) {
        if (output_format == EncodedImageFormat::JPEG ||
            output_format == EncodedImageFormat::PNG ||
            output_format == EncodedImageFormat::WEBP ||
            ext_lower == ".avi" ||
            ext_lower == ".webm") {
            base_path.replace_extension();
        }
    }

    int output_begin_idx = cli_params.output_begin_idx;
    if (output_begin_idx < 0) {
        output_begin_idx = 0;
    }

    auto write_image = [&](const fs::path& path, int idx) {
        const sd_image_t& img = results[idx];
        if (!img.data)
            return false;

        const int64_t metadata_seed = cli_params.mode == VID_GEN ? gen_params.seed : gen_params.seed + idx;
        std::string params          = gen_params.embed_image_metadata
                                          ? get_image_params(ctx_params, gen_params, metadata_seed, cli_params.mode)
                                          : "";
        const bool ok               = write_image_to_file(path.string(), img.data, img.width, img.height, img.channel, params, 90);
        LOG_INFO("save result image %d to '%s' (%s)", idx, path.string().c_str(), ok ? "success" : "failure");
        return ok;
    };

    auto write_audio_sidecar = [&](const fs::path& wav_path) {
        if (generated_audio == nullptr) {
            return;
        }
        if (write_wav_to_file(wav_path.string(),
                              generated_audio->data,
                              generated_audio->sample_count,
                              generated_audio->channels,
                              generated_audio->sample_rate)) {
            LOG_INFO("save result audio to '%s'", wav_path.string().c_str());
        } else {
            LOG_WARN("failed to save result audio to '%s'", wav_path.string().c_str());
        }
    };

    int sucessful_reults = 0;

    if (std::regex_search(cli_params.output_path, format_specifier_regex)) {
        if (output_format == EncodedImageFormat::UNKNOWN)
            ext = ".png";
        fs::path pattern = base_path;
        pattern += ext;

        for (int i = 0; i < num_results; ++i) {
            fs::path img_path = format_frame_idx(pattern.string(), output_begin_idx + i);
            if (write_image(img_path, i)) {
                sucessful_reults++;
            }
        }
        LOG_INFO("%d/%d images saved", sucessful_reults, num_results);
        return sucessful_reults != 0;
    }

    if (cli_params.mode == VID_GEN && num_results > 1) {
        if (ext_lower != ".avi" && ext_lower != ".webp" && ext_lower != ".webm")
            ext = ".avi";
        fs::path video_path = base_path;
        video_path += ext;
        std::string final_ext_lower = ext.string();
        std::transform(final_ext_lower.begin(), final_ext_lower.end(), final_ext_lower.begin(), ::tolower);
        const bool mux_audio = generated_audio != nullptr && (final_ext_lower == ".avi" || final_ext_lower == ".webm");
        if (create_video_from_sd_images(video_path.string().c_str(), results, num_results, gen_params.fps, 90, mux_audio ? generated_audio : nullptr) == 0) {
            LOG_INFO("save result video to '%s'", video_path.string().c_str());
            if (generated_audio != nullptr && !mux_audio) {
                fs::path wav_path = video_path;
                wav_path.replace_extension(".wav");
                write_audio_sidecar(wav_path);
            }
            return true;
        } else {
            LOG_ERROR("Failed to save result video to '%s'", video_path.string().c_str());
            return false;
        }
    }

    if (output_format == EncodedImageFormat::UNKNOWN)
        ext = ".png";

    for (int i = 0; i < num_results; ++i) {
        fs::path img_path = base_path;
        if (num_results > 1) {
            img_path += "_" + std::to_string(output_begin_idx + i);
        }
        img_path += ext;
        if (write_image(img_path, i)) {
            sucessful_reults++;
        }
    }
    LOG_INFO("%d/%d images saved", sucessful_reults, num_results);
    if (generated_audio != nullptr) {
        write_audio_sidecar(get_video_audio_sidecar_path(cli_params));
    }
    return sucessful_reults != 0;
}

// Per-segment continuity match for continuation chaining (LongCat-Avatar).
// In a chained segment (seg>0) the leading `cond_n` decoded frames are the
// re-rendered prior-segment tail, HELD FIXED via the denoise mask, so they
// carry the prior segment's true exposure. The trailing generated frames come
// fresh from the 8-step-DMD denoise and land DARKER (lap-15: cond ~95-100,
// generated ~75-84) — a cond->gen brightness STEP that the seam re-encode
// propagates and compounds. This matches each generated frame's per-channel
// mean+std to the cond region (a smarter fix than lap-15's uniform offset,
// which referenced cond-vs-prior-tail — already matched — and was a wash).
// Operates on the decoded RGB BEFORE the cond frames are dropped + before the
// tail is captured for re-encode, so it closes the compounding loop. DEFAULT-ON
// for chained renders (opt out LONGCAT_CONT_NO_EXPOSURE_MATCH=1); strength env
// LONGCAT_CONT_EXPOSURE_STRENGTH (default 1.0 = full match; 0.5 = half, gentler).
// MEASURED (3x33f Q4_K): kills the compounding per-segment brightness ramp —
// luma 109.6->115.1->133.3 (drift +23.7) collapses to 109.6->110.3->111.0
// (drift +1.5). Seam frame-to-frame Δ residual (5.9x->4.7x) is structural
// (pose discontinuity between independently-sampled segments), not exposure.
static void continuity_match_segment(sd_image_t* frames, int n, int cond_n, float strength) {
    if (frames == nullptr || n <= cond_n || cond_n <= 0)
        return;
    const int C = (int)frames[0].channel;
    if (C < 1)
        return;
    const size_t px_per_frame = (size_t)frames[0].width * frames[0].height;
    if (px_per_frame == 0)
        return;
    // per-channel mean/std over the cond region [0..cond_n) and gen region [cond_n..n)
    std::vector<double> cm(C, 0), cv(C, 0), gm(C, 0), gv(C, 0);
    auto accum = [&](int f0, int f1, std::vector<double>& m, std::vector<double>& v) {
        std::vector<double> s(C, 0), s2(C, 0);
        size_t cnt = 0;
        for (int f = f0; f < f1; ++f) {
            const uint8_t* d = frames[f].data;
            for (size_t p = 0; p < px_per_frame; ++p)
                for (int c = 0; c < C; ++c) {
                    double x = d[p * C + c];
                    s[c] += x;
                    s2[c] += x * x;
                }
            cnt += px_per_frame;
        }
        for (int c = 0; c < C; ++c) {
            m[c] = s[c] / cnt;
            double var = s2[c] / cnt - m[c] * m[c];
            v[c] = var > 1e-6 ? std::sqrt(var) : 1e-3;
        }
    };
    accum(0, cond_n, cm, cv);
    accum(cond_n, n, gm, gv);
    // affine map each generated frame's channels: out = cm + (x-gm)*(cv/gv),
    // blended by `strength` toward identity.
    for (int f = cond_n; f < n; ++f) {
        uint8_t* d = frames[f].data;
        for (int c = 0; c < C; ++c) {
            double gain  = cv[c] / gv[c];
            double bias  = cm[c] - gm[c] * gain;
            // blend toward identity (gain=1,bias=0)
            gain = 1.0 + strength * (gain - 1.0);
            bias = strength * bias;
            for (size_t p = 0; p < px_per_frame; ++p) {
                double x   = (double)d[p * C + c] * gain + bias;
                d[p * C + c] = (uint8_t)(x < 0 ? 0 : (x > 255 ? 255 : x + 0.5));
            }
        }
    }
}

int main(int argc, const char* argv[]) {
    if (argc > 1 && std::string(argv[1]) == "--version") {
        std::cout << version_string() << "\n";
        return EXIT_SUCCESS;
    }

    SDCliParams cli_params;
    SDContextParams ctx_params;
    SDGenerationParams gen_params;

    parse_args(argc, argv, cli_params, ctx_params, gen_params);
    sd_set_log_callback(sd_log_cb, (void*)&cli_params);
    log_verbose = cli_params.verbose;
    log_color   = cli_params.color;

    if (cli_params.mode == METADATA) {
        MetadataReadOptions options;
        options.output_format      = cli_params.metadata_format == "json"
                                         ? MetadataOutputFormat::JSON
                                         : MetadataOutputFormat::TEXT;
        options.include_raw        = cli_params.metadata_raw;
        options.brief              = cli_params.metadata_brief;
        options.include_structural = cli_params.metadata_all;

        std::string error;
        if (!print_image_metadata(cli_params.image_path, options, std::cout, error)) {
            LOG_ERROR("%s", error.c_str());
            return 1;
        }
        return 0;
    }

    if (gen_params.video_frames > 4) {
        size_t last_dot_pos   = cli_params.preview_path.find_last_of(".");
        std::string base_path = cli_params.preview_path;
        std::string file_ext  = "";
        if (last_dot_pos != std::string::npos) {  // filename has extension
            base_path = cli_params.preview_path.substr(0, last_dot_pos);
            file_ext  = cli_params.preview_path.substr(last_dot_pos);
            std::transform(file_ext.begin(), file_ext.end(), file_ext.begin(), ::tolower);
        }
        if (file_ext == ".png") {
            cli_params.preview_path = base_path + ".avi";
        }
    }
    cli_params.preview_fps = gen_params.fps;
    if (cli_params.preview_method == PREVIEW_PROJ)
        cli_params.preview_fps /= 4;

    sd_set_preview_callback(step_callback,
                            cli_params.preview_method,
                            cli_params.preview_interval,
                            !cli_params.preview_noisy,
                            cli_params.preview_noisy,
                            (void*)&cli_params);

    LOG_DEBUG("version: %s", version_string().c_str());
    LOG_DEBUG("%s", sd_get_system_info());
    LOG_DEBUG("%s", cli_params.to_string().c_str());
    LOG_DEBUG("%s", ctx_params.to_string().c_str());
    LOG_DEBUG("%s", gen_params.to_string().c_str());

    if (cli_params.mode == CONVERT) {
        bool success = convert_with_imatrix(ctx_params.model_path.c_str(),
                                            ctx_params.vae_path.c_str(),
                                            cli_params.output_path.c_str(),
                                            ctx_params.wtype,
                                            ctx_params.tensor_type_rules.c_str(),
                                            cli_params.convert_name,
                                            cli_params.imatrix_path.c_str());
        if (!success) {
            LOG_ERROR("convert '%s'/'%s' to '%s' failed",
                      ctx_params.model_path.c_str(),
                      ctx_params.vae_path.c_str(),
                      cli_params.output_path.c_str());
            return 1;
        } else {
            LOG_INFO("convert '%s'/'%s' to '%s' success",
                     ctx_params.model_path.c_str(),
                     ctx_params.vae_path.c_str(),
                     cli_params.output_path.c_str());
            return 0;
        }
    }

    bool vae_decode_only = true;

    auto load_image_and_update_size = [&](const std::string& path,
                                          SDImageOwner& image,
                                          bool resize_image    = true,
                                          int expected_channel = 3) -> bool {
        int expected_width  = 0;
        int expected_height = 0;
        if (resize_image && gen_params.width_and_height_are_set()) {
            expected_width  = gen_params.width;
            expected_height = gen_params.height;
        }

        if (!load_sd_image_from_file(image.put(), path.c_str(), expected_width, expected_height, expected_channel)) {
            LOG_ERROR("load image from '%s' failed", path.c_str());
            return false;
        }

        gen_params.set_width_and_height_if_unset(image.get().width, image.get().height);
        return true;
    };

    if (gen_params.init_image_path.size() > 0) {
        vae_decode_only = false;
        if (!load_image_and_update_size(gen_params.init_image_path, gen_params.init_image)) {
            return 1;
        }
    }

    if (gen_params.end_image_path.size() > 0) {
        vae_decode_only = false;
        if (!load_image_and_update_size(gen_params.end_image_path, gen_params.end_image)) {
            return 1;
        }
    }

    if (gen_params.keyframe_paths.size() > 0) {
        vae_decode_only = false;
        gen_params.keyframe_images.clear();
        for (auto& path : gen_params.keyframe_paths) {
            SDImageOwner kf_image({0, 0, 3, nullptr});
            if (!load_image_and_update_size(path, kf_image)) {
                return 1;
            }
            gen_params.keyframe_images.push_back(std::move(kf_image));
        }
    }

    if (gen_params.ref_image_paths.size() > 0) {
        vae_decode_only = false;
        gen_params.ref_images.clear();
        for (auto& path : gen_params.ref_image_paths) {
            SDImageOwner ref_image({0, 0, 3, nullptr});
            if (!load_image_and_update_size(path, ref_image, false)) {
                return 1;
            }
            gen_params.ref_images.push_back(std::move(ref_image));
        }
    }

    if (gen_params.mask_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.mask_image.put(),
                                     gen_params.mask_image_path.c_str(),
                                     gen_params.get_resolved_width(),
                                     gen_params.get_resolved_height(),
                                     1)) {
            LOG_ERROR("load image from '%s' failed", gen_params.mask_image_path.c_str());
            return 1;
        }
    } else {
        sd_image_t generated_mask = {0, 0, 1, nullptr};
        generated_mask.data       = (uint8_t*)malloc(gen_params.get_resolved_width() * gen_params.get_resolved_height());
        if (generated_mask.data == nullptr) {
            LOG_ERROR("malloc mask image failed");
            return 1;
        }
        generated_mask.width  = gen_params.get_resolved_width();
        generated_mask.height = gen_params.get_resolved_height();
        memset(generated_mask.data, 255, gen_params.get_resolved_width() * gen_params.get_resolved_height());
        gen_params.mask_image.reset(generated_mask);
    }

    if (gen_params.control_image_path.size() > 0) {
        if (!load_sd_image_from_file(gen_params.control_image.put(),
                                     gen_params.control_image_path.c_str(),
                                     gen_params.get_resolved_width(),
                                     gen_params.get_resolved_height())) {
            LOG_ERROR("load image from '%s' failed", gen_params.control_image_path.c_str());
            return 1;
        }
        if (cli_params.canny_preprocess) {  // apply preprocessor
            preprocess_canny(gen_params.control_image.get(),
                             0.08f,
                             0.08f,
                             0.8f,
                             1.0f,
                             false);
        }
    }

    if (!gen_params.control_video_path.empty()) {
        gen_params.control_frames.clear();
        if (!load_images_from_dir(gen_params.control_video_path,
                                  gen_params.control_frames,
                                  gen_params.get_resolved_width(),
                                  gen_params.get_resolved_height(),
                                  gen_params.video_frames,
                                  cli_params.verbose)) {
            return 1;
        }
    }

    if (!gen_params.pm_id_images_dir.empty()) {
        gen_params.pm_id_images.clear();
        if (!load_images_from_dir(gen_params.pm_id_images_dir,
                                  gen_params.pm_id_images,
                                  0,
                                  0,
                                  0,
                                  cli_params.verbose)) {
            return 1;
        }
    }

    if (cli_params.mode == VID_GEN) {
        vae_decode_only = false;
    }

    if (gen_params.hires_enabled &&
        (gen_params.resolved_hires_upscaler == SD_HIRES_UPSCALER_MODEL ||
         gen_params.resolved_hires_upscaler == SD_HIRES_UPSCALER_LANCZOS ||
         gen_params.resolved_hires_upscaler == SD_HIRES_UPSCALER_NEAREST)) {
        vae_decode_only = false;
    }

    sd_ctx_params_t sd_ctx_params = ctx_params.to_sd_ctx_params_t(vae_decode_only, true, cli_params.taesd_preview);

    SDImageVec results;
    int num_results             = 0;
    sd_audio_t* generated_audio = nullptr;

    if (cli_params.mode == UPSCALE) {
        num_results = 1;
        results.push_back(gen_params.init_image.release());
    } else {
        SDCtxPtr sd_ctx(new_sd_ctx(&sd_ctx_params));

        if (sd_ctx == nullptr) {
            LOG_INFO("new_sd_ctx_t failed");
            return 1;
        }

        if (gen_params.sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen_params.sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
        }

        if (gen_params.high_noise_sample_params.sample_method == SAMPLE_METHOD_COUNT) {
            gen_params.high_noise_sample_params.sample_method = sd_get_default_sample_method(sd_ctx.get());
        }

        if (gen_params.sample_params.scheduler == SCHEDULER_COUNT) {
            gen_params.sample_params.scheduler = sd_get_default_scheduler(sd_ctx.get(), gen_params.sample_params.sample_method);
        }

        if (cli_params.mode == IMG_GEN) {
            sd_img_gen_params_t img_gen_params = gen_params.to_sd_img_gen_params_t();

            // nvfp4-twolevel: enable diffusion imatrix collection before render.
            if (!cli_params.collect_imatrix_path.empty()) {
                sd_imatrix_collect_begin("diffusion_model");
            }

            num_results = gen_params.batch_count;
            results.adopt(generate_image(sd_ctx.get(), &img_gen_params), num_results);

            if (!cli_params.collect_imatrix_path.empty()) {
                int n_written = 0;
                if (!sd_imatrix_collect_write_gguf(cli_params.collect_imatrix_path.c_str(), &n_written)) {
                    LOG_ERROR("failed to write imatrix to '%s'", cli_params.collect_imatrix_path.c_str());
                } else {
                    LOG_INFO("imatrix: wrote %d DiT Linear tensors to '%s'",
                             n_written, cli_params.collect_imatrix_path.c_str());
                }
            }
        } else if (cli_params.mode == VID_GEN) {
            sd_vid_gen_params_t vid_gen_params = gen_params.to_sd_vid_gen_params_t();

            // nvfp4: enable diffusion imatrix collection before the video render. The graph
            // hook is mode-agnostic (accumulates per-column activation 2nd-moment on every
            // DiT Linear mul_mat), so this mirrors the IMG_GEN path and lets an avatar
            // calibration imatrix be collected with `-M vid_gen --collect-imatrix <out>`.
            if (!cli_params.collect_imatrix_path.empty()) {
                sd_imatrix_collect_begin("diffusion_model");
            }

            // LongCat-Avatar mouth-exaggeration knobs -> runtime env (the library reads
            // these per-render; the API will set the equivalent fields directly).
            if (gen_params.audio_mouth_scale != 1.0f) {
                setenv("LONGCAT_AUDIO_MOUTH_SCALE", std::to_string(gen_params.audio_mouth_scale).c_str(), 1);
            }
            if (gen_params.audio_lowpass > 0.0f) {
                setenv("LONGCAT_AUDIO_LOWPASS", std::to_string(gen_params.audio_lowpass).c_str(), 1);
            }
            // LTXAV relip knobs -> runtime env (a2v guidance / ramp / ref-tstride). CLI is a COLD
            // process (no warm-worker sticky-value problem), so bridge only when the field was set
            // to a NON-default via --a2v-guidance / JSON — otherwise leave the env untouched so a
            // harness that sets LTXAV_A2V_GUIDANCE / LTXAV_RELIP_REF_TSTRIDE directly (e.g.
            // still wins. The warm server, by contrast, MUST always-overwrite
            // (apply_ltx_relip_env) to stop a prior render's value bleeding across requests.
            if (gen_params.a2v_guidance != 1.0f) {
                setenv("LTXAV_A2V_GUIDANCE", std::to_string(gen_params.a2v_guidance).c_str(), 1);
            }
            if (gen_params.a2v_ramp_end != 1.0f) {
                setenv("LTXAV_A2V_RAMP_END", std::to_string(gen_params.a2v_ramp_end).c_str(), 1);
            }
            if (gen_params.relip_ref_tstride != 1) {
                setenv("LTXAV_RELIP_REF_TSTRIDE", std::to_string(gen_params.relip_ref_tstride).c_str(), 1);
            }
            // FEATURE 2 (NAG) -> runtime env. Bridge only when NAG is actually enabled (--nag-scale
            // != 0) so a harness that sets LTXAV_NAG_* directly still wins on default runs. When on,
            // bridge all four so the engine (sample() + resolve use_uncond force) sees a coherent set.
            if (gen_params.nag_scale != 0.0f) {
                setenv("LTXAV_NAG_SCALE", std::to_string(gen_params.nag_scale).c_str(), 1);
                setenv("LTXAV_NAG_ALPHA", std::to_string(gen_params.nag_alpha).c_str(), 1);
                setenv("LTXAV_NAG_TAU", std::to_string(gen_params.nag_tau).c_str(), 1);
                setenv("LTXAV_NAG_UNTIL_SIGMA", std::to_string(gen_params.nag_until_sigma).c_str(), 1);
            }

            int n_segments = std::max(1, gen_params.segments);
            if (gen_params.ltx_chain_segments > 0) {
                // ============================================================================
                // LTXAV IN-PROCESS N-SEGMENT VIDEO CHAINING (separate from the avatar loop
                // below; do NOT entangle). One sd-cli process renders N video segments with
                // the ~11GB DiT kept RESIDENT across all of them (no per-segment reload).
                //  - seg 0: plain i2v from --init-img (cont_latent = nullptr).
                //  - seg N>0: continues from the PRIOR segment's video-latent tail (last K
                //    latent frames, VIDEO channels only) handed to the in-memory LTXAV
                //    cont-latent branch in prepare_video_generation_latents.
                // Stitch mirrors the file-based chain: drop OVERLAP_PX = 1+(K-1)*8 pixel
                // frames off the head of every seg>0 (the re-rendered overlap), append rest.
                // ============================================================================
                int n_chain = gen_params.ltx_chain_segments;
                // Per-segment prompts (the "director" layer): one prompt per line in
                // --ltx-chain-prompts. Fewer lines than segments reuses the last; empty
                // reuses the single -p for every segment.
                std::vector<std::string> seg_prompts;
                if (!gen_params.ltx_chain_prompts_path.empty()) {
                    std::ifstream pf(gen_params.ltx_chain_prompts_path);
                    if (!pf) {
                        LOG_ERROR("failed to open --ltx-chain-prompts %s", gen_params.ltx_chain_prompts_path.c_str());
                        return 1;
                    }
                    std::string line;
                    while (std::getline(pf, line)) {
                        if (!line.empty() && line.back() == '\r') {
                            line.pop_back();
                        }
                        if (!line.empty()) {
                            seg_prompts.push_back(line);
                        }
                    }
                    LOG_INFO("LTXAV chain: loaded %zu per-segment prompt(s) from %s",
                             seg_prompts.size(), gen_params.ltx_chain_prompts_path.c_str());
                }

                // Resolve per-segment prompt strings (fewer lines reuse the last; none =>
                // NULL => generate_video_chain reuses the base -p). Storage must outlive the call.
                std::vector<std::string> resolved_prompts;
                resolved_prompts.reserve(n_chain);
                for (int seg = 0; seg < n_chain; ++seg) {
                    if (!seg_prompts.empty()) {
                        resolved_prompts.push_back(seg_prompts[std::min((size_t)seg, seg_prompts.size() - 1)]);
                    } else {
                        resolved_prompts.emplace_back();  // empty -> NULL ptr below -> base prompt
                    }
                }
                std::vector<const char*> prompt_ptrs;
                prompt_ptrs.reserve(n_chain);
                for (const auto& s : resolved_prompts) {
                    prompt_ptrs.push_back(s.empty() ? nullptr : s.c_str());
                }

                std::string save_dir;
                if (const char* sdir = getenv("LTX_CHAIN_SAVE_DIR")) {
                    save_dir = sdir;
                }

                // The whole chain (resident DiT, in-memory latent carry, prompts pre-encoded
                // in one TE window, per-segment lip-sync audio, stitch) lives in the library so
                // the CLI and the worker-isolated server share one implementation.
                sd_vid_chain_params_t chain_params = {};
                chain_params.retake_segment     = -1;  // off (aggregate {} would give 0 = a valid index)
                chain_params.n_segments         = n_chain;
                chain_params.cont_latent_frames = std::max(1, gen_params.cont_latent_take);
                chain_params.segment_prompts    = prompt_ptrs.data();
                chain_params.chain_audio_dir    = gen_params.ltx_chain_audio_dir.empty()
                                                      ? nullptr
                                                      : gen_params.ltx_chain_audio_dir.c_str();
                chain_params.save_dir           = save_dir.empty() ? nullptr : save_dir.c_str();

                sd_image_t* chain_frames = nullptr;
                int         chain_count  = 0;
                sd_audio_t* chain_audio  = nullptr;
                if (!generate_video_chain(sd_ctx.get(), &vid_gen_params, &chain_params,
                                          &chain_frames, &chain_count, &chain_audio)) {
                    LOG_ERROR("LTXAV chain failed");
                    return 1;
                }
                results.adopt(chain_frames, chain_count);
                num_results     = chain_count;
                generated_audio = chain_audio;
            } else if (n_segments > 1) {
                // LongCat-Avatar continuation chaining: render N segments, each
                // conditioned on the prior segment's LATENT tail (no decode/re-encode
                // roundtrip — generate_video_ex hands back the diffusion latent), and
                // stitch them into one continuous clip. Segment 0 is the full render
                // from the init image; each later segment drops its leading cond frames
                // (a re-render of the prior tail) and appends the rest.
                int cond_vframes = std::max(1, gen_params.cont_cond_frames);  // overlap in VIDEO frames
                int seg_frames   = gen_params.video_frames;
                // Wan VAE temporal: num_cond_latents latents <-> 1+(N-1)*4 video frames.
                int num_cond_latents = 1 + (cond_vframes - 1) / 4;
                int cond_decoded_v   = 1 + (num_cond_latents - 1) * 4;  // video frames the cond latents decode to
                if (cond_decoded_v >= seg_frames) {
                    LOG_ERROR("--cont-cond-frames (%d -> %d latents -> %d decoded frames) must leave new frames in --video-frames (%d)",
                              cond_vframes, num_cond_latents, cond_decoded_v, seg_frames);
                    return 1;
                }
                int new_per_seg = seg_frames - cond_decoded_v;  // genuinely new video frames per chained segment
                LOG_INFO("continuation: %d segments, overlap %d video frames = %d cond latents (%d decoded), %d new frames/seg",
                         n_segments, cond_vframes, num_cond_latents, cond_decoded_v, new_per_seg);

                // Keep the DiT + whisper resident across segments (the TE is freed once
                // by the GPU-TE deferred-load flow; the DiT must persist or later
                // segments render against freed GPU memory).
                sd_ctx_keep_diffusion_model_resident(sd_ctx.get(), true);

                std::vector<float> stitched_audio;  // 16k mono, full timeline
                uint32_t audio_sr = 0;
                // DRIFT SINK (decode->re-encode): a chained segment conditions on the
                // prior segment's tail. v1 fed the prior segment's RAW diffusion latent
                // tail straight back — but that skips the reference pipeline's per-chain
                // re-regularization, so 8-step DMD sampling error (color/exposure cast +
                // identity morph) COMPOUNDED at each seam. The fix mirrors generate_vc:
                // DECODE the prior segment's tail to pixels (we already have them — they
                // are the segment's decoded output frames) and RE-ENCODE them through the
                // Wan VAE back to a fresh diffusion latent. The encode->decode round-trip
                // snaps the cond back onto the VAE data manifold every step, the drift
                // sink the reference relies on. We feed the LAST cond_decoded_v decoded
                // frames; sd_ctx_encode_video_frames returns num_cond_latents latents.
                std::vector<sd_image_t> cond_tail_frames;  // last cond_decoded_v decoded frames of prior seg
                std::vector<float> cond_latent_buf;        // re-encoded cond latent fed to current seg
                // A/B escape hatch: LONGCAT_CONT_RAW_LATENT=1 restores the v1 raw-latent
                // passthrough (the drifting path) for measuring the seam fix. Default is
                // the decode->re-encode drift sink.
                bool raw_latent = getenv("LONGCAT_CONT_RAW_LATENT") != nullptr;
                std::vector<float> prev_latent;  // v1 raw passthrough only
                int prev_lw = 0, prev_lh = 0, prev_lt = 0, prev_lc = 0;

                // REFERENCE ANCHOR (generate_avc): the reference keeps a persistent,
                // un-drifted anchor latent and PREPENDS it on EVERY continuation segment
                // -> [ref(1), cond_tail(N), noise...]; without it continuation drifts off
                // the already-drifted prior tail (watercolour melt). The reference's
                // anchor is `ref_latent = latent[:, :, :1]` — segment 0's (ai2v) output
                // latent FRAME 0, which the ai2v path holds fixed = the VAE-encoded
                // portrait latent. We capture exactly that from segment 0's returned
                // latent (no separate up-front VAE encode — that would fight the resident
                // DiT for VRAM). Held constant across all later segments. Opt out
                // (legacy ref-free continuation) with LONGCAT_CONT_NO_REF_ANCHOR=1.
                bool use_ref_anchor = (getenv("LONGCAT_CONT_NO_REF_ANCHOR") == nullptr);
                std::vector<float> ref_anchor_latent;  // [Wl*Hl*1*Cl] contiguous f32 (frame 0)
                bool ref_anchor_ready = false;

                for (int seg = 0; seg < n_segments; ++seg) {
                    // Reference anchor: armed for every continuation segment (>0); seg0
                    // is the plain ai2v render (no cont latent, no ref split) — its
                    // latent frame 0 BECOMES the anchor for all later segments.
                    if (seg > 0 && use_ref_anchor && ref_anchor_ready) {
                        vid_gen_params.cont_ref_latent      = ref_anchor_latent.data();
                        vid_gen_params.cont_ref_img_index   = gen_params.cont_ref_img_index;
                        vid_gen_params.cont_mask_frame_range = gen_params.cont_mask_frame_range;
                    } else {
                        vid_gen_params.cont_ref_latent = nullptr;
                    }
                    if (seg == 0) {
                        vid_gen_params.cont_latent        = nullptr;
                        vid_gen_params.cont_latent_frames = 0;
                        vid_gen_params.audio_frame_offset = 0;
                    } else if (raw_latent) {
                        // v1: feed the prior segment's RAW diffusion-latent tail (drifts).
                        size_t plane = (size_t)prev_lw * prev_lh;
                        int    keep  = std::min(num_cond_latents, prev_lt);
                        cond_latent_buf.assign((size_t)plane * keep * prev_lc, 0.f);
                        for (int c = 0; c < prev_lc; ++c) {
                            for (int nf = 0; nf < keep; ++nf) {
                                int src_t        = prev_lt - keep + nf;
                                const float* src = prev_latent.data() + ((size_t)c * prev_lt + src_t) * plane;
                                float* dst       = cond_latent_buf.data() + ((size_t)c * keep + nf) * plane;
                                std::memcpy(dst, src, plane * sizeof(float));
                            }
                        }
                        vid_gen_params.cont_latent        = cond_latent_buf.data();
                        vid_gen_params.cont_latent_frames = keep;
                        vid_gen_params.audio_frame_offset = seg * new_per_seg;
                    } else {
                        // re-encode the prior segment's decoded tail frames -> fresh
                        // diffusion latent (the drift sink). Take the trailing
                        // num_cond_latents latent frames as the cond conditioning.
                        int rlw = 0, rlh = 0, rlt = 0, rlc = 0;
                        float* reenc = sd_ctx_encode_video_frames(
                            sd_ctx.get(), cond_tail_frames.data(), (int)cond_tail_frames.size(),
                            gen_params.width, gen_params.height, &rlw, &rlh, &rlt, &rlc);
                        if (reenc == nullptr) {
                            LOG_ERROR("continuation: re-encode of segment %d tail failed", seg);
                            return 1;
                        }
                        // keep the LAST num_cond_latents latent frames (layout
                        // [Wl,Hl,Tl,Cl,1], temporal dim 2).
                        size_t plane = (size_t)rlw * rlh;
                        int    keep  = std::min(num_cond_latents, rlt);
                        cond_latent_buf.assign((size_t)plane * keep * rlc, 0.f);
                        for (int c = 0; c < rlc; ++c) {
                            for (int nf = 0; nf < keep; ++nf) {
                                int src_t        = rlt - keep + nf;
                                const float* src = reenc + ((size_t)c * rlt + src_t) * plane;
                                float* dst       = cond_latent_buf.data() + ((size_t)c * keep + nf) * plane;
                                std::memcpy(dst, src, plane * sizeof(float));
                            }
                        }
                        free(reenc);
                        LOG_INFO("continuation: re-encoded %zu tail frames -> %d cond latents (drift sink)",
                                 cond_tail_frames.size(), keep);
                        vid_gen_params.cont_latent        = cond_latent_buf.data();
                        vid_gen_params.cont_latent_frames = keep;
                        vid_gen_params.audio_frame_offset = seg * new_per_seg;
                    }
                    // distinct seed per segment so the noise frames differ
                    vid_gen_params.seed = (gen_params.seed < 0) ? gen_params.seed : gen_params.seed + seg;

                    LOG_INFO("=== continuation segment %d/%d (audio_offset=%d frames) ===",
                             seg + 1, n_segments, vid_gen_params.audio_frame_offset);

                    sd_image_t* seg_video = nullptr;
                    int seg_count         = 0;
                    sd_audio_t* seg_audio = nullptr;
                    // Default (drift sink) re-encodes pixels, so no raw latent needed.
                    // raw_latent mode requests the diffusion latent to pass through.
                    // The ref-anchor also needs the latent (segment 0's frame 0).
                    bool want_latent = raw_latent || (use_ref_anchor && seg == 0);
                    float* lat_out = nullptr;
                    int    lw = 0, lh = 0, lt = 0, lc = 0;
                    if (!generate_video_ex(sd_ctx.get(), &vid_gen_params, &seg_video, &seg_count, &seg_audio,
                                           want_latent ? &lat_out : nullptr,
                                           want_latent ? &lw : nullptr, want_latent ? &lh : nullptr,
                                           want_latent ? &lt : nullptr, want_latent ? &lc : nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr,
                                           nullptr, nullptr, nullptr, nullptr, nullptr)) {
                        LOG_ERROR("continuation segment %d failed", seg + 1);
                        free_sd_audio(seg_audio);
                        free(seg_video);
                        free(lat_out);
                        return 1;
                    }
                    if (raw_latent && lat_out != nullptr) {
                        prev_latent.assign(lat_out, lat_out + (size_t)lw * lh * lt * lc);
                        prev_lw = lw; prev_lh = lh; prev_lt = lt; prev_lc = lc;
                    }
                    // Capture segment 0's latent FRAME 0 as the persistent ref anchor
                    // (= the reference's `ref_latent = latent[:, :, :1]`). Layout is
                    // [Wl, Hl, Tl, Cl] contiguous; frame 0 is the temporal stride 0 slice
                    // per channel: src[(c*Tl + 0)*plane + p].
                    if (use_ref_anchor && seg == 0 && lat_out != nullptr && lt > 0) {
                        size_t plane = (size_t)lw * lh;
                        ref_anchor_latent.assign(plane * (size_t)lc, 0.f);
                        for (int c = 0; c < lc; ++c) {
                            const float* src = lat_out + ((size_t)c * lt + 0) * plane;
                            std::memcpy(ref_anchor_latent.data() + (size_t)c * plane, src, plane * sizeof(float));
                        }
                        ref_anchor_ready = true;
                        LOG_INFO("continuation: captured ref anchor from seg0 latent frame 0 (%dx%d, %d ch)", lw, lh, lc);
                    }
                    if (lat_out != nullptr) {
                        free(lat_out);
                    }

                    // frames: segment 0 keeps all; later segments drop the cond tail re-render.
                    int drop = (seg == 0) ? 0 : cond_decoded_v;
                    // SMARTER per-segment continuity fix (opt-in): match the generated
                    // frames' per-channel mean+std to this segment's cond region (the
                    // held-fixed prior tail) so the cond->gen brightness STEP doesn't
                    // propagate + compound through the re-encode. Applied BEFORE drop +
                    // before the tail capture, so the matched pixels feed both the
                    // stitched output and the next segment's cond. seg0 has no cond region.
                    if (seg > 0 && getenv("LONGCAT_CONT_NO_EXPOSURE_MATCH") == nullptr) {
                        float strength = 1.0f;
                        if (const char* se = getenv("LONGCAT_CONT_EXPOSURE_STRENGTH"))
                            strength = (float)atof(se);
                        continuity_match_segment(seg_video, seg_count, drop, strength);
                    }
                    // capture this segment's LAST cond_decoded_v decoded frames (deep copy)
                    // to re-encode as the NEXT segment's cond conditioning (drift sink).
                    for (auto& f : cond_tail_frames) free(f.data);
                    cond_tail_frames.clear();
                    if (seg + 1 < n_segments && !raw_latent) {
                        int tail_start = seg_count - cond_decoded_v;
                        if (tail_start < 0) tail_start = 0;
                        for (int i = tail_start; i < seg_count; ++i) {
                            sd_image_t copy = seg_video[i];
                            size_t nbytes   = (size_t)copy.width * copy.height * copy.channel;
                            copy.data       = (uint8_t*)malloc(nbytes);
                            std::memcpy(copy.data, seg_video[i].data, nbytes);
                            cond_tail_frames.push_back(copy);
                        }
                    }
                    for (int i = 0; i < seg_count; ++i) {
                        if (i < drop) {
                            free(seg_video[i].data);  // discard re-rendered cond frame
                        } else {
                            results.push_back(seg_video[i]);  // adopt ownership
                        }
                    }
                    free(seg_video);

                    // audio: seg_audio already starts at this segment's offset
                    // (audio_frame_offset). Drop the cond-frame portion for seg>0,
                    // then append; this reconstructs the full continuous timeline.
                    if (seg_audio != nullptr && seg_audio->data != nullptr) {
                        audio_sr            = seg_audio->sample_rate;
                        size_t drop_samples = (size_t)((double)drop / 25.0 * (double)audio_sr);
                        if (drop_samples > seg_audio->sample_count) {
                            drop_samples = seg_audio->sample_count;
                        }
                        stitched_audio.insert(stitched_audio.end(),
                                              seg_audio->data + drop_samples,
                                              seg_audio->data + seg_audio->sample_count);
                    }
                    free_sd_audio(seg_audio);
                }
                for (auto& f : cond_tail_frames) free(f.data);
                cond_tail_frames.clear();

                num_results = (int)results.size();
                LOG_INFO("continuation: stitched %d segments -> %d frames", n_segments, num_results);

                if (!stitched_audio.empty() && audio_sr > 0) {
                    sd_audio_t* a = (sd_audio_t*)malloc(sizeof(sd_audio_t));
                    if (a != nullptr) {
                        a->sample_rate  = audio_sr;
                        a->channels     = 1;
                        a->sample_count = (uint64_t)stitched_audio.size();
                        a->data         = (float*)malloc(stitched_audio.size() * sizeof(float));
                        if (a->data != nullptr) {
                            std::memcpy(a->data, stitched_audio.data(), stitched_audio.size() * sizeof(float));
                            generated_audio = a;
                        } else {
                            free(a);
                        }
                    }
                }
            } else {
                sd_image_t* generated_video = nullptr;
                if (!generate_video(sd_ctx.get(), &vid_gen_params, &generated_video, &num_results, &generated_audio)) {
                    generated_video = nullptr;
                }
                results.adopt(generated_video, num_results);
            }

            // nvfp4: write the accumulated diffusion imatrix (every DiT Linear driven by
            // the render(s) above — one 8-step avatar denoise covers all 48 blocks).
            if (!cli_params.collect_imatrix_path.empty()) {
                int n_written = 0;
                if (!sd_imatrix_collect_write_gguf(cli_params.collect_imatrix_path.c_str(), &n_written)) {
                    LOG_ERROR("failed to write imatrix to '%s'", cli_params.collect_imatrix_path.c_str());
                } else {
                    LOG_INFO("imatrix: wrote %d DiT Linear tensors to '%s'",
                             n_written, cli_params.collect_imatrix_path.c_str());
                }
            }
        }

        if (!results) {
            LOG_ERROR("generate failed");
            return 1;
        }
    }

    int upscale_factor = 4;  // unused for RealESRGAN_x4plus_anime_6B.pth
    if (ctx_params.esrgan_path.size() > 0 && gen_params.upscale_repeats > 0) {
        UpscalerCtxPtr upscaler_ctx(new_upscaler_ctx(ctx_params.esrgan_path.c_str(),
                                                     ctx_params.offload_params_to_cpu,
                                                     ctx_params.diffusion_conv_direct,
                                                     ctx_params.n_threads,
                                                     gen_params.upscale_tile_size,
                                                     ctx_params.backend.c_str(),
                                                     ctx_params.params_backend.c_str()));

        if (upscaler_ctx == nullptr) {
            LOG_ERROR("new_upscaler_ctx failed");
        } else {
            for (int i = 0; i < num_results; i++) {
                if (results[i].data == nullptr) {
                    continue;
                }
                SDImageOwner current_image(results[i]);
                results[i] = {0, 0, 0, nullptr};
                for (int u = 0; u < gen_params.upscale_repeats; ++u) {
                    SDImageOwner upscaled_image(upscale(upscaler_ctx.get(), current_image.get(), upscale_factor));
                    if (upscaled_image.get().data == nullptr) {
                        LOG_ERROR("upscale failed");
                        break;
                    }
                    current_image = std::move(upscaled_image);
                }
                results[i] = current_image.release();  // Set the final upscaled image as the result
            }
        }
    }

    if (!save_results(cli_params, ctx_params, gen_params, results.data(), num_results, generated_audio)) {
        free_sd_audio(generated_audio);
        return 1;
    }

    free_sd_audio(generated_audio);

    return 0;
}
