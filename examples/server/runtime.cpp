#include "runtime.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>

#include "common/common.h"
#include "common/log.h"

namespace fs = std::filesystem;

static std::string lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

static bool is_supported_model_ext(const fs::path& p) {
    auto ext = lower_ascii(p.extension().string());
    return ext == ".gguf" || ext == ".pt" || ext == ".pth" || ext == ".safetensors";
}

static const std::string k_base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

std::string base64_encode(const std::vector<uint8_t>& bytes) {
    std::string ret;
    int val  = 0;
    int valb = -6;
    for (uint8_t c : bytes) {
        val = (val << 8) + c;
        valb += 8;
        while (valb >= 0) {
            ret.push_back(k_base64_chars[(val >> valb) & 0x3F]);
            valb -= 6;
        }
    }
    if (valb > -6) {
        ret.push_back(k_base64_chars[((val << 8) >> (valb + 8)) & 0x3F]);
    }
    while (ret.size() % 4) {
        ret.push_back('=');
    }
    return ret;
}

bool base64_decode(const std::string& text, std::vector<uint8_t>& bytes) {
    bytes.clear();
    int accumulator = 0;
    int bits = -8;
    for (unsigned char c : text) {
        if (c == '=') {
            break;
        }
        int value = -1;
        if (c >= 'A' && c <= 'Z') {
            value = c - 'A';
        } else if (c >= 'a' && c <= 'z') {
            value = c - 'a' + 26;
        } else if (c >= '0' && c <= '9') {
            value = c - '0' + 52;
        } else if (c == '+') {
            value = 62;
        } else if (c == '/') {
            value = 63;
        }
        if (value < 0) {
            return false;
        }
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if (bits >= 0) {
            bytes.push_back(static_cast<uint8_t>((accumulator >> bits) & 0xff));
            bits -= 8;
        }
    }
    return true;
}

std::string normalize_output_format(std::string output_format) {
    std::transform(output_format.begin(), output_format.end(), output_format.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return output_format;
}

std::vector<std::string> supported_img_output_formats(bool allow_webp) {
    std::vector<std::string> formats = {"png", "jpeg"};
#ifdef SD_USE_WEBP
    if (allow_webp) {
        formats.push_back("webp");
    }
#else
    (void)allow_webp;
#endif
    return formats;
}

std::vector<std::string> supported_vid_output_formats() {
    std::vector<std::string> formats;
#ifdef SD_USE_WEBM
    formats.push_back("webm");
#endif
#ifdef SD_USE_WEBP
    formats.push_back("webp");
#endif
    formats.push_back("avi");
    return formats;
}

static std::string valid_vid_output_formats_message() {
    const std::vector<std::string> formats = supported_vid_output_formats();

    std::string message = "invalid output_format, must be one of [";
    for (size_t i = 0; i < formats.size(); ++i) {
        if (i > 0) {
            message += ", ";
        }
        message += formats[i];
    }
    message += "]";
    return message;
}

bool assign_output_options(ImgGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           bool allow_webp,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    const std::vector<std::string> valid_formats = supported_img_output_formats(allow_webp);
    const bool valid_format                      = std::find(valid_formats.begin(),
                                                             valid_formats.end(),
                                                             request.output_format) != valid_formats.end();
    if (!valid_format) {
        error_message = "invalid output_format, must be one of [";
        for (size_t i = 0; i < valid_formats.size(); ++i) {
            if (i > 0) {
                error_message += ", ";
            }
            error_message += valid_formats[i];
        }
        error_message += "]";
        return false;
    }

    return true;
}

bool assign_output_options(VidGenJobRequest& request,
                           std::string output_format,
                           int output_compression,
                           std::string& error_message) {
    request.output_format      = normalize_output_format(std::move(output_format));
    request.output_compression = std::clamp(output_compression, 0, 100);

    if (request.output_format == "avi") {
        return true;
    }

    if (request.output_format == "webm") {
#ifdef SD_USE_WEBM
        return true;
#else
        error_message = valid_vid_output_formats_message();
        return false;
#endif
    }

    if (request.output_format == "webp") {
#ifdef SD_USE_WEBP
        return true;
#else
        error_message = valid_vid_output_formats_message();
        return false;
#endif
    }

    error_message = valid_vid_output_formats_message();
    return false;
}

std::string video_mime_type(const std::string& output_format) {
    if (output_format == "webm") {
        return "video/webm";
    }
    if (output_format == "webp") {
        return "image/webp";
    }
    return "video/x-msvideo";
}

bool runtime_supports_generation_mode(const ServerRuntime& runtime, SDMode mode) {
    if (mode == VID_GEN) {
        return sd_ctx_supports_video_generation(runtime.sd_ctx);
    }
    if (mode == IMG_GEN) {
        return sd_ctx_supports_image_generation(runtime.sd_ctx);
    }
    return true;
}

bool runtime_is_draining(const ServerRuntime& runtime) {
    return runtime.gpu_sharing != nullptr && runtime.gpu_sharing->draining.load();
}

std::map<std::string, std::string> runtime_diffusion_model_variants(const ServerRuntime& runtime) {
    std::map<std::string, std::string> variants;
    // A supervised child has --diffusion-model replaced with the selected
    // variant. Its environment preserves the original logical map so `base`
    // still means base rather than merely "the model selected at spawn".
    if (const char* worker_map = std::getenv("SD_SERVER_WORKER_VARIANT_MAP");
        worker_map != nullptr && worker_map[0] != '\0') {
        std::stringstream entries(worker_map);
        std::string entry;
        while (std::getline(entries, entry, ';')) {
            const size_t equals = entry.find('=');
            if (equals == std::string::npos || equals == 0 || equals + 1 >= entry.size()) continue;
            variants[entry.substr(0, equals)] = entry.substr(equals + 1);
        }
        return variants;
    }
    if (runtime.ctx_params != nullptr && !runtime.ctx_params->diffusion_model_path.empty()) {
        variants.emplace("base", runtime.ctx_params->diffusion_model_path);
    }
    if (runtime.svr_params != nullptr && !runtime.svr_params->diffusion_model_edit_path.empty()) {
        variants["edit"] = runtime.svr_params->diffusion_model_edit_path;
    }
    if (runtime.svr_params == nullptr) return variants;
    std::stringstream entries(runtime.svr_params->diffusion_model_variants_spec);
    std::string entry;
    while (std::getline(entries, entry, ';')) {
        const size_t equals = entry.find('=');
        if (equals == std::string::npos || equals == 0 || equals + 1 >= entry.size()) continue;
        const std::string name = entry.substr(0, equals);
        const std::string path = entry.substr(equals + 1);
        if (!name.empty() && !path.empty()) variants[name] = path;
    }
    return variants;
}

std::string unsupported_generation_mode_error(SDMode mode) {
    if (mode == VID_GEN) {
        return "loaded model does not support vid_gen";
    }
    if (mode == IMG_GEN) {
        return "loaded model does not support img_gen";
    }
    return "loaded model does not support requested mode";
}

ArgOptions SDSvrParams::get_options() {
    ArgOptions options;

    options.string_options = {
        {"-l", "--listen-ip", "server listen ip (default: 127.0.0.1)", 0, &listen_ip},
        {"", "--serve-html-path", "path to HTML file to serve at root (optional)", 0, &serve_html_path},
        {"", "--diffusion-model-edit", "alternate DiT GGUF for model=edit worker requests", 0, &diffusion_model_edit_path},
        {"", "--diffusion-model-variants", "named DiT variants: name=path;name=path", 0, &diffusion_model_variants_spec},
    };

    options.int_options = {
        {"", "--listen-port", "server listen port (default: 1234)", &listen_port},
    };

    options.bool_options = {
        {"-v", "--verbose", "print extra info", true, &verbose},
        {"", "--color", "colors the logging tags according to level", true, &color},
    };

    auto on_help_arg = [&](int, const char**, int, bool& valid) {
        normal_exit = true;
        valid       = true;
        return -1;
    };

    options.manual_options = {
        {"-h", "--help", "show this help message and exit", on_help_arg},
    };
    return options;
}

bool SDSvrParams::validate() {
    if (listen_ip.empty()) {
        LOG_ERROR("error: the following arguments are required: listen_ip");
        return false;
    }

    if (listen_port < 0 || listen_port > 65535) {
        LOG_ERROR("error: listen_port should be in the range [0, 65535]");
        return false;
    }

    if (!serve_html_path.empty() && !fs::exists(serve_html_path)) {
        LOG_ERROR("error: serve_html_path file does not exist: %s", serve_html_path.c_str());
        return false;
    }
    return true;
}

bool SDSvrParams::resolve_and_validate() {
    if (!validate()) {
        return false;
    }
    return true;
}

std::string SDSvrParams::to_string() const {
    std::ostringstream oss;
    oss << "SDSvrParams {\n"
        << "  listen_ip: " << listen_ip << ",\n"
        << "  listen_port: \"" << listen_port << "\",\n"
        << "  serve_html_path: \"" << serve_html_path << "\",\n"
        << "  diffusion_model_edit_path: \"" << diffusion_model_edit_path << "\",\n"
        << "  diffusion_model_variants_spec: \"" << diffusion_model_variants_spec << "\",\n"
        << "}";
    return oss.str();
}

// Read `<lora_dir>/loras.json` — the optional, operator-editable description of the adapters in
// this directory. See LoraEntry in runtime.h for why it lives here and not in a client.
//
// Shape (every key optional; unlisted files are simply undescribed):
//
//   {
//     "ltx2.3-transition.gguf": {
//       "label": "Transition / morph",
//       "blurb": "Smooth first-to-last-frame morphs and scene transitions.",
//       "default_multiplier": 1.0,
//       "trigger": "zhuanchang",
//       "trigger_required": true
//     }
//   }
//
// A malformed manifest is a WARNING, never fatal: the adapters themselves still load, and taking
// the whole LoRA directory offline because someone left a trailing comma would be a far worse
// failure than serving the files undescribed. Keys are matched against the entry's `path` (the
// name relative to the LoRA dir, extension included — the same string a request must send).
// `LTX_LORA_MANIFEST` overrides the location. It exists because the natural home —
// `<lora_dir>/loras.json` — is awkward for a containerised deployment: the LoRA directory is
// already a bind mount, and mounting a single tracked file INSIDE another bind mount means nesting
// two mounts, which resolves through any symlink already at that path and silently lands the file
// somewhere else. A separate path plus this variable is unambiguous. Unset keeps the natural
// default, which is what a bare-metal run wants.
static json load_lora_manifest(const fs::path& lora_dir) {
    fs::path manifest = lora_dir / "loras.json";
    bool explicitly_configured = false;
    if (const char* override_path = std::getenv("LTX_LORA_MANIFEST")) {
        if (override_path[0] != '\0') {
            manifest             = override_path;
            explicitly_configured = true;
        }
    }
    std::error_code ec;
    if (!fs::exists(manifest, ec) || !fs::is_regular_file(manifest, ec)) {
        // An absent DEFAULT is the ordinary "this directory is undescribed" case and says nothing.
        // An absent CONFIGURED path is an operator error, and a silent one: every adapter quietly
        // reverts to the client's neutral 1.0, which for several of them is the wrong strength and
        // shows up only as a render that under- or over-fires. So it gets said out loud.
        if (explicitly_configured) {
            LOG_WARN("LTX_LORA_MANIFEST=%s does not exist; every adapter will be UNDESCRIBED and "
                     "clients will fall back to a neutral 1.0 strength\n",
                     manifest.u8string().c_str());
        }
        return json::object();
    }
    std::ifstream f(manifest, std::ios::binary);
    if (!f) {
        LOG_WARN("lora manifest %s exists but could not be opened; adapters will be undescribed\n",
                 manifest.u8string().c_str());
        return json::object();
    }
    json parsed = json::parse(f, nullptr, false);
    if (parsed.is_discarded() || !parsed.is_object()) {
        LOG_WARN("lora manifest %s is not a JSON object; adapters will be undescribed\n",
                 manifest.u8string().c_str());
        return json::object();
    }
    return parsed;
}

json lora_entry_json(const LoraEntry& e) {
    json item;
    item["name"] = e.name;
    item["path"] = e.path;
    // Manifest fields are OMITTED rather than defaulted when unstated, so a client can tell
    // "the operator states 1.0" from "nobody has said" — see LoraEntry in runtime.h.
    if (!e.label.empty()) {
        item["label"] = e.label;
    }
    if (!e.blurb.empty()) {
        item["blurb"] = e.blurb;
    }
    if (e.default_multiplier >= 0.0f) {
        item["default_multiplier"] = e.default_multiplier;
    }
    if (!e.trigger.empty()) {
        item["trigger"]          = e.trigger;
        item["trigger_required"] = e.trigger_required;
    }
    return item;
}

std::vector<LoraEntry> scan_lora_dir(const std::string& lora_model_dir) {
    std::vector<LoraEntry> new_cache;

    fs::path lora_dir = lora_model_dir;
    if (fs::exists(lora_dir) && fs::is_directory(lora_dir)) {
        // Re-read per refresh rather than caching: the directory is bind-mounted, so editing the
        // manifest is meant to take effect without restarting the engine, exactly as dropping a new
        // .gguf into the directory already does.
        json manifest = load_lora_manifest(lora_dir);

        for (auto& entry : fs::recursive_directory_iterator(lora_dir, fs::directory_options::skip_permission_denied)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path& p = entry.path();
            if (!is_supported_model_ext(p)) {
                continue;
            }

            LoraEntry lora_entry;
            lora_entry.name     = p.stem().u8string();
            lora_entry.fullpath = p.u8string();
            std::string rel     = p.lexically_relative(lora_dir).u8string();
            std::replace(rel.begin(), rel.end(), '\\', '/');
            lora_entry.path = rel;

            // Manifest lookup by the wire name first, then by stem, so an entry keyed
            // "foo.safetensors" still describes the "foo.gguf" it was converted to (the engine
            // prefers the gguf; losing the trained multiplier on that swap would be silent).
            auto it = manifest.find(rel);
            if (it == manifest.end()) {
                it = manifest.find(lora_entry.name);
            }
            if (it != manifest.end() && it->is_object()) {
                const json& m               = *it;
                lora_entry.label            = m.value("label", std::string());
                lora_entry.blurb            = m.value("blurb", std::string());
                lora_entry.default_multiplier = m.value("default_multiplier", -1.0f);
                lora_entry.trigger          = m.value("trigger", std::string());
                lora_entry.trigger_required = m.value("trigger_required", false);
            }

            new_cache.push_back(std::move(lora_entry));
        }
    }

    std::sort(new_cache.begin(), new_cache.end(), [](const LoraEntry& a, const LoraEntry& b) {
        return a.path < b.path;
    });
    return new_cache;
}

void refresh_lora_cache(ServerRuntime& rt) {
    std::vector<LoraEntry> new_cache = scan_lora_dir(rt.ctx_params->lora_model_dir);
    {
        std::lock_guard<std::mutex> lock(*rt.lora_mutex);
        *rt.lora_cache = std::move(new_cache);
    }
}

std::string get_lora_full_path(ServerRuntime& rt, const std::string& path) {
    std::lock_guard<std::mutex> lock(*rt.lora_mutex);
    auto it = std::find_if(rt.lora_cache->begin(), rt.lora_cache->end(),
                           [&](const LoraEntry& entry) { return entry.path == path; });
    return it != rt.lora_cache->end() ? it->fullpath : "";
}

void refresh_upscaler_cache(ServerRuntime& rt) {
    std::vector<UpscalerEntry> new_cache;

    fs::path upscaler_dir = rt.ctx_params->hires_upscalers_dir;
    if (fs::exists(upscaler_dir) && fs::is_directory(upscaler_dir)) {
        for (auto& entry : fs::directory_iterator(upscaler_dir)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            const fs::path& p = entry.path();
            if (!is_supported_model_ext(p)) {
                continue;
            }

            UpscalerEntry upscaler_entry;
            upscaler_entry.name       = p.stem().u8string();
            upscaler_entry.fullpath   = fs::absolute(p).lexically_normal().u8string();
            upscaler_entry.model_name = "ESRGAN_4x";
            upscaler_entry.path       = p.filename().u8string();

            new_cache.push_back(std::move(upscaler_entry));
        }
    }

    std::sort(new_cache.begin(), new_cache.end(), [](const UpscalerEntry& a, const UpscalerEntry& b) {
        return a.name < b.name;
    });

    {
        std::lock_guard<std::mutex> lock(*rt.upscaler_mutex);
        *rt.upscaler_cache = std::move(new_cache);
    }
}

int64_t unix_timestamp_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}
