#include "routes.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
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
    // `part:<name>` — the WAV rode as a binary part of this request rather than as ~40 MB of
    // base64 inside it (`docs/media-transport.md` §4/§5: a 3-minute drive track is the single
    // biggest inline payload on the avatar path).
    if (is_media_part_ref(value)) {
        const std::string* part = find_media_part(value);
        if (part == nullptr || part->empty()) {
            return false;
        }
        bytes.assign(part->begin(), part->end());
        return true;
    }
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

uint16_t read_u16_le(const uint8_t* data) {
    return static_cast<uint16_t>(data[0]) | (static_cast<uint16_t>(data[1]) << 8);
}

uint32_t read_u32_le(const uint8_t* data) {
    return static_cast<uint32_t>(data[0]) |
           (static_cast<uint32_t>(data[1]) << 8) |
           (static_cast<uint32_t>(data[2]) << 16) |
           (static_cast<uint32_t>(data[3]) << 24);
}

sd_audio_t* load_avatar_output_audio(const std::vector<uint8_t>& wav, int frame_count, int fps) {
    if (wav.size() < 12 || std::memcmp(wav.data(), "RIFF", 4) != 0 || std::memcmp(wav.data() + 8, "WAVE", 4) != 0) {
        return nullptr;
    }
    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    const uint8_t* samples = nullptr;
    size_t sample_bytes = 0;
    for (size_t offset = 12; offset + 8 <= wav.size();) {
        const uint32_t size = read_u32_le(wav.data() + offset + 4);
        const size_t content = offset + 8;
        if (content + size > wav.size()) {
            return nullptr;
        }
        if (std::memcmp(wav.data() + offset, "fmt ", 4) == 0 && size >= 16) {
            format = read_u16_le(wav.data() + content);
            channels = read_u16_le(wav.data() + content + 2);
            sample_rate = read_u32_le(wav.data() + content + 4);
            bits_per_sample = read_u16_le(wav.data() + content + 14);
        } else if (std::memcmp(wav.data() + offset, "data", 4) == 0) {
            samples = wav.data() + content;
            sample_bytes = size;
        }
        offset = content + size + (size & 1U);
    }
    if (samples == nullptr || channels == 0 || sample_rate == 0 ||
        !((format == 1 && (bits_per_sample == 16 || bits_per_sample == 32)) ||
          (format == 3 && bits_per_sample == 32))) {
        return nullptr;
    }
    const size_t bytes_per_sample = bits_per_sample / 8;
    const size_t total_frames = sample_bytes / (static_cast<size_t>(channels) * bytes_per_sample);
    const size_t count = std::min(total_frames,
                                  static_cast<size_t>(std::max(0, frame_count)) * sample_rate / std::max(1, fps));
    auto* audio = static_cast<sd_audio_t*>(calloc(1, sizeof(sd_audio_t)));
    if (audio == nullptr) {
        return nullptr;
    }
    audio->sample_rate = sample_rate;
    audio->channels = channels;
    audio->sample_count = count;
    const size_t value_count = count * channels;
    audio->data = value_count == 0 ? nullptr : static_cast<float*>(malloc(value_count * sizeof(float)));
    if (value_count != 0 && audio->data == nullptr) {
        free(audio);
        return nullptr;
    }
    for (size_t i = 0; i < value_count; ++i) {
        const uint8_t* source = samples + i * bytes_per_sample;
        if (format == 1 && bits_per_sample == 16) {
            audio->data[i] = static_cast<float>(static_cast<int16_t>(read_u16_le(source))) / 32768.f;
        } else if (format == 1) {
            audio->data[i] = static_cast<float>(static_cast<int32_t>(read_u32_le(source))) / 2147483648.f;
        } else {
            float value = 0.f;
            std::memcpy(&value, source, sizeof(value));
            audio->data[i] = value;
        }
    }
    return audio;
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
    // LongCat's established API exposes segment_frames/video_frames at the
    // top level.  Keep this explicit after generic parameter resolution: the
    // generic parser otherwise preserves the server's 65-frame default here,
    // while Avatar's audio projector is sized from the requested frame count.
    if (body.contains("video_frames") && body["video_frames"].is_number_integer()) {
        request.gen_params.video_frames = body["video_frames"].get<int>();
    }
    if (body.contains("fps") && body["fps"].is_number_integer()) {
        request.gen_params.fps      = body["fps"].get<int>();
        // Mark it explicit so resolve_and_validate() does not promote to the avatar's
        // native fps over the top of a caller who deliberately asked for something else.
        request.gen_params.fps_explicit = true;
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

    // `avatar_rendering` is set ONLY by the synchronous LongCat /generate route below. Every other
    // render path — /ltx/v1/generate, /wan/v1/generate, /sdcpp/v1/{img,vid}_gen — returns 202
    // immediately and runs on the async job worker, so that flag stays false for the whole render.
    //
    // The supervisor's in_flight() is `active_generation_requests + (child_busy ? 1 : 0)`, and
    // active_generation_requests drops back to 0 the moment the proxied submit returns its 202.
    // So with only avatar_rendering behind child_busy, an LTX render reported busy:false /
    // in_flight:0 for its entire duration — the service looked IDLE while it was rendering.
    // Koblem's GPU gate then evicts it to reclaim VRAM, and the next job/media poll lands on the
    // supervisor's deliberate "worker is unloaded" branch => the caller sees a mid-render 410.
    // Counting live async jobs closes that hole.
    svr.Get("/health", [runtime](const httplib::Request&, httplib::Response& res) {
        // `accepts_part_refs` is the deploy-order gate (`docs/media-transport.md` §9.4): koblem
        // may only send `part:<name>` media to an engine that says it understands it. Without a
        // flag, rolling an engine back while koblem is on the new path turns every marker into a
        // string that fails to decode — a corrupt request rather than a clean fallback.
        res.set_content(json({{"status", "ok"},
                              {"busy", avatar_rendering.load() || async_job_in_flight(runtime->async_job_manager)},
                              {"draining", runtime_is_draining(*runtime)},
                              {"accepts_part_refs", true},
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
            if (req.body.empty() && !req.is_multipart_form_data()) {
                res.status = 400;
                res.set_content(R"({"error":"empty body"})", "application/json");
                return;
            }
            // `part:<name>` media (§4). The avatar path decodes BOTH its inputs during the parse
            // below — the portrait through the image funnel, the drive WAV through
            // `decode_avatar_blob` — so the registration only has to span this handler.
            MediaPartTable media_parts = collect_media_parts(req);
            if (const std::string bad = first_media_part_hash_mismatch(media_parts); !bad.empty()) {
                res.status = 400;
                res.set_content(json({{"error", "part " + bad + " does not match its content hash"}}).dump(),
                                "application/json");
                return;
            }
            ScopedMediaParts parts_guard(&media_parts);
            json body;
            if (req.is_multipart_form_data()) {
                // A `request` part holds the document; the portrait and the WAV are binary parts
                // it names. Same shape as /ltx/v1/generate and /h3/v1/generate.
                std::string document;
                if (req.form.has_field("request")) {
                    document = req.form.get_field("request");
                } else if (req.form.has_file("request")) {
                    document = req.form.get_file("request").content;
                }
                if (document.empty()) {
                    res.status = 400;
                    res.set_content(R"({"error":"multipart requires a non-empty request part"})", "application/json");
                    return;
                }
                body = json::parse(document);
            } else {
                body = json::parse(req.body);
            }
            const std::string image = body.value("image", std::string());
            const std::string audio = body.value("audio", std::string());
            if (image.empty() || audio.empty()) {
                res.status = 400;
                res.set_content(R"({"error":"image and audio are required"})", "application/json");
                return;
            }
            if (!body.contains("init_image")) {
                // A `part:` reference and a data URI are both passed through untouched; only a
                // bare base64 payload needs the prefix the image funnel expects.
                body["init_image"] = (image.rfind("data:", 0) == 0 || is_media_part_ref(image))
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
            int segment_count = body.value("segments", 1);
            const int continuation_frames = body.value("cont_cond_frames", 13);
            // `cont_cond_frames` is a count of PIXEL (decoded) frames. The chain API wants
            // LATENT frames, and the VAE is 4x temporal, so the two must be converted — not
            // used interchangeably. Mirrors prod's avatar_render.cpp:
            //   13 pixel -> num_cond_latents = 1 + (13-1)/4 = 4 latent -> 1 + 3*4 = 13 pixel.
            // Treating 13 as latent instead expands the overlap to 1 + 12*4 = 49 pixel frames,
            // leaving only 4 new frames per segment (so a 2.25 s clip wants ~14 segments
            // instead of 2) and passing a 13 that the engine reads as latent — which is why
            // chained avatar renders failed outright while single-segment ones were fine.
            const int cond_vframes     = std::max(1, continuation_frames);
            const int num_cond_latents = 1 + (cond_vframes - 1) / 4;
            const int cond_decoded_v   = 1 + (num_cond_latents - 1) * 4;
            if (body.contains("duration_sec") && body["duration_sec"].is_number()) {
                const int overlap = cond_decoded_v;
                const int new_frames = std::max(1, params.video_frames - overlap);
                const int target_frames = std::max(1, static_cast<int>(std::round(body["duration_sec"].get<double>() * 25.0)));
                segment_count = target_frames <= params.video_frames
                                    ? 1
                                    : 1 + (target_frames - params.video_frames + new_frames - 1) / new_frames;
            }
            if (segment_count < 1 || continuation_frames < 1) {
                res.status = 400;
                res.set_content(R"({"error":"segments and cont_cond_frames must be positive"})", "application/json");
                return;
            }

            SDImageVec frames;
            sd_audio_t* generated_audio = nullptr;
            int frame_count = 0;
            {
                std::lock_guard<std::mutex> lock(*runtime->sd_ctx_mutex);
                sd_image_t* raw_frames = nullptr;
                bool generated = false;
                if (segment_count == 1) {
                    generated = generate_video(runtime->sd_ctx, &params, &raw_frames, &frame_count, &generated_audio);
                } else {
                    sd_vid_chain_params_t chain = {};
                    chain.n_segments = segment_count;
                    // LATENT frames (see the pixel->latent conversion above), not the raw
                    // pixel `cont_cond_frames` the request carries.
                    chain.cont_latent_frames = num_cond_latents;
                    generated = generate_video_chain(runtime->sd_ctx,
                                                       &params,
                                                       &chain,
                                                       &raw_frames,
                                                       &frame_count,
                                                       &generated_audio);
                }
                if (!generated) {
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
            if (segment_count > 1) {
                free_sd_audio(generated_audio);
                generated_audio = load_avatar_output_audio(audio_bytes, frames.count(), request.gen_params.fps);
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
            res.set_header("X-Avatar-Segments", std::to_string(segment_count));
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
