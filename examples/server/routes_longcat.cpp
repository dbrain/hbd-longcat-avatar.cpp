#include "routes.h"

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "async_jobs.h"
#include "common/media_io.h"

namespace {

namespace fs = std::filesystem;

std::atomic<bool> avatar_rendering{false};
std::atomic<uint64_t> avatar_temp_sequence{0};

class ScopedAvatarAudioFile {
public:
    ~ScopedAvatarAudioFile() {
        if (!path.empty()) {
            std::error_code error;
            fs::remove(path, error);
        }
    }

    bool write(const std::vector<uint8_t>& bytes) {
        if (bytes.empty()) {
            return false;
        }
        std::error_code error;
        const fs::path directory = fs::temp_directory_path(error);
        if (error) {
            return false;
        }
        path = directory / ("longcat-avatar-" + std::to_string(unix_timestamp_now()) + "-" +
                            std::to_string(avatar_temp_sequence.fetch_add(1)) + ".wav");
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        return output.good();
    }

    std::string string() const {
        return path.string();
    }

private:
    fs::path path;
};

bool decode_avatar_blob(const std::string& value, std::vector<uint8_t>& bytes) {
    std::string encoded = value;
    if (encoded.rfind("data:", 0) == 0) {
        const size_t comma = encoded.find(',');
        if (comma == std::string::npos) {
            return false;
        }
        encoded = encoded.substr(comma + 1);
    }
    return base64_decode(encoded, bytes) && !bytes.empty();
}

bool parse_avatar_video_request(const json& body,
                                ServerRuntime& runtime,
                                VidGenJobRequest& request,
                                std::string& error_message) {
    request.gen_params = *runtime.default_gen_params;
    refresh_lora_cache(runtime);
    if (!request.gen_params.from_json_str(body.dump(), [&](const std::string& path) {
            return get_lora_full_path(runtime, path);
        })) {
        error_message = "invalid generation parameters";
        return false;
    }
    if (!request.gen_params.resolve_and_validate(VID_GEN, "", runtime.ctx_params->hires_upscalers_dir, true)) {
        error_message = "invalid generation parameters";
        return false;
    }
    request.output_format = "webm";
    request.output_compression = body.value("output_compression", 100);
    return true;
}

void apply_avatar_bsa(const json& body, sd_vid_gen_params_t& params) {
    if (!body.contains("bsa") || !body["bsa"].is_object()) {
        return;
    }
    const json& bsa = body["bsa"];
    params.bsa_enabled = bsa.value("enable", false) ? 1 : 0;
    params.bsa_radius = bsa.value("radius", params.bsa_radius);
    params.bsa_self_frame = bsa.value("self_frame", params.bsa_self_frame != 0) ? 1 : 0;
    params.bsa_bookend = bsa.value("bookend", params.bsa_bookend != 0) ? 1 : 0;
    params.bsa_cube_h = bsa.value("cube_h", params.bsa_cube_h);
    params.bsa_cube_w = bsa.value("cube_w", params.bsa_cube_w);
}

}  // namespace

void register_longcat_avatar_endpoints(httplib::Server& svr, ServerRuntime& rt) {
    ServerRuntime* runtime = &rt;

    svr.Get("/health", [runtime](const httplib::Request&, httplib::Response& res) {
        res.set_content(json({{"status", "ok"},
                              {"busy", avatar_rendering.load()},
                              {"draining", runtime_is_draining(*runtime)},
                              {"loaded", runtime->gpu_sharing == nullptr || runtime->gpu_sharing->diffusion_loaded.load()}})
                            .dump(),
                        "application/json");
    });

    svr.Post("/generate", [runtime](const httplib::Request& req, httplib::Response& res) {
        if (runtime_is_draining(*runtime)) {
            res.status = 503;
            res.set_content(R"({"error":"service draining — not accepting new renders"})", "application/json");
            return;
        }
        bool expected = false;
        if (!avatar_rendering.compare_exchange_strong(expected, true)) {
            res.status = 429;
            res.set_content(R"({"error":"busy: a render is already in flight"})", "application/json");
            return;
        }
        struct ResetBusy {
            ~ResetBusy() { avatar_rendering.store(false); }
        } reset_busy;

        try {
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            json body = json::parse(req.body);
            const std::string image = body.value("image", std::string());
            const std::string audio = body.value("audio", std::string());
            if (image.empty() || audio.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"image and audio are required"})", "application/json");
                return;
            }
            if (!body.contains("init_image")) {
                body["init_image"] = image.rfind("data:", 0) == 0
                                         ? image
                                         : std::string("data:image/png;base64,") + image;
            }
            if (body.contains("segment_frames") && !body.contains("video_frames")) {
                body["video_frames"] = body["segment_frames"];
            }

            std::vector<uint8_t> audio_bytes;
            if (!decode_avatar_blob(audio, audio_bytes)) {
                res.status = 400;
                res.set_content(R"({"error":"audio must be base64 WAV data"})", "application/json");
                return;
            }
            ScopedAvatarAudioFile audio_file;
            if (!audio_file.write(audio_bytes)) {
                res.status = 500;
                res.set_content(R"({"error":"could not stage avatar audio"})", "application/json");
                return;
            }
            body["audio_path"] = audio_file.string();

            VidGenJobRequest request;
            std::string error_message;
            if (!parse_avatar_video_request(body, *runtime, request, error_message)) {
                res.status = 400;
                res.set_content(json({{"error", error_message}}).dump(), "application/json");
                return;
            }
            sd_vid_gen_params_t params = request.to_sd_vid_gen_params_t();
            apply_avatar_bsa(body, params);

            SDImageVec frames;
            sd_audio_t* generated_audio = nullptr;
            int frame_count = 0;
            {
                std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
                sd_image_t* raw_frames = nullptr;
                if (!generate_video(runtime->sd_ctx, &params, &raw_frames, &frame_count, &generated_audio)) {
                    free_sd_audio(generated_audio);
                    res.status = 500;
                    res.set_content(R"({"error":"avatar generation failed"})", "application/json");
                    return;
                }
                frames.adopt(raw_frames, frame_count);
                if (runtime->gpu_sharing != nullptr) {
                    runtime->gpu_sharing->diffusion_loaded.store(true);
                }
            }
            if (frames.count() <= 0) {
                free_sd_audio(generated_audio);
                res.status = 500;
                res.set_content(R"({"error":"avatar generation produced no frames"})", "application/json");
                return;
            }
            const auto video = create_video_from_sd_images_to_vector("webm",
                                                                       frames.data(),
                                                                       frames.count(),
                                                                       request.gen_params.fps,
                                                                       request.output_compression,
                                                                       generated_audio);
            free_sd_audio(generated_audio);
            if (video.empty()) {
                res.status = 500;
                res.set_content(R"({"error":"could not encode avatar video"})", "application/json");
                return;
            }
            res.set_header("X-Avatar-Segments", "1");
            res.set_header("X-Avatar-Offload", "false");
            res.set_header("X-Avatar-Render_Sec", "0");
            res.set_content(std::string(reinterpret_cast<const char*>(video.data()), video.size()), "video/webm");
        } catch (const json::parse_error& error) {
            res.status = 400;
            res.set_content(json({{"error", "invalid json"}, {"message", error.what()}}).dump(), "application/json");
        } catch (const std::exception& error) {
            res.status = 500;
            res.set_content(json({{"error", "server_error"}, {"message", error.what()}}).dump(), "application/json");
        }
    });
}
