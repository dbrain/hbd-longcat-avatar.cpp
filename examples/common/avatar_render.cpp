// avatar_render.cpp — see avatar_render.h.
//
// Ported from examples/cli/main.cpp (VID_GEN branch, n_segments > 1 path
// + the single-segment path), with:
//   - environment toggles (LONGCAT_AUDIO_MOUTH_SCALE / LONGCAT_AUDIO_LOWPASS)
//     wired the same way the cli does (the avatar lib reads these per render);
//   - output encoded to the requested container via
//     create_video_from_sd_images_to_vector (so callers can stream the bytes
//     directly without saving to disk);
//   - LongCat's init_image is loaded here so callers can simply set
//     gen_params.init_image_path before calling.
//
// Caller MUST already hold the sd_ctx mutex.

#include "avatar_render.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "common/log.h"
#include "common/media_io.h"
#include "common/resource_owners.hpp"
#include "stable-diffusion.h"

// continuity_match_segment is a verbatim port of cli/main.cpp's helper. We
// only need it for chained renders, so it stays file-local here.
static void continuity_match_segment(sd_image_t* frames, int n, int cond_n, float strength) {
    if (frames == nullptr || n <= cond_n || cond_n <= 0) return;
    const int C = (int)frames[0].channel;
    if (C < 1) return;
    const size_t px_per_frame = (size_t)frames[0].width * frames[0].height;
    if (px_per_frame == 0) return;
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
    for (int f = cond_n; f < n; ++f) {
        uint8_t* d = frames[f].data;
        for (int c = 0; c < C; ++c) {
            double gain = cv[c] / gv[c];
            double bias = cm[c] - gm[c] * gain;
            gain = 1.0 + strength * (gain - 1.0);
            bias = strength * bias;
            for (size_t p = 0; p < px_per_frame; ++p) {
                double x = (double)d[p * C + c] * gain + bias;
                d[p * C + c] = (uint8_t)(x < 0 ? 0 : (x > 255 ? 255 : x + 0.5));
            }
        }
    }
}

bool render_avatar_to_video_bytes(sd_ctx_t* sd_ctx,
                                  SDGenerationParams& gen_params,
                                  const std::string& output_format,
                                  int output_quality,
                                  std::vector<uint8_t>& out_bytes,
                                  int& out_segments_rendered,
                                  std::string& error_message) {
    out_bytes.clear();
    out_segments_rendered = 0;
    error_message.clear();

    if (sd_ctx == nullptr) {
        error_message = "sd_ctx is null";
        return false;
    }

    // Load init image from path if not already populated. Mirrors cli/main.cpp.
    if (gen_params.init_image.get().data == nullptr && !gen_params.init_image_path.empty()) {
        int expected_width  = gen_params.width_and_height_are_set() ? gen_params.width  : 0;
        int expected_height = gen_params.width_and_height_are_set() ? gen_params.height : 0;
        if (!load_sd_image_from_file(gen_params.init_image.put(),
                                     gen_params.init_image_path.c_str(),
                                     expected_width,
                                     expected_height,
                                     3)) {
            error_message = "failed to load init_image from path: " + gen_params.init_image_path;
            return false;
        }
        gen_params.set_width_and_height_if_unset(gen_params.init_image.get().width,
                                                 gen_params.init_image.get().height);
    }

    // Build the C-API params blob *once* (same way the cli does).
    sd_vid_gen_params_t vid_gen_params = gen_params.to_sd_vid_gen_params_t();

    // LongCat-Avatar mouth-exaggeration / lowpass knobs ride through process
    // env vars — the avatar lib reads them per-render. Mirror the cli wiring.
    if (gen_params.audio_mouth_scale != 1.0f) {
        setenv("LONGCAT_AUDIO_MOUTH_SCALE",
               std::to_string(gen_params.audio_mouth_scale).c_str(), 1);
    }
    if (gen_params.audio_lowpass > 0.0f) {
        setenv("LONGCAT_AUDIO_LOWPASS",
               std::to_string(gen_params.audio_lowpass).c_str(), 1);
    }

    SDImageVec results;
    sd_audio_t* generated_audio = nullptr;
    int num_results = 0;

    const int n_segments = std::max(1, gen_params.segments);
    if (n_segments == 1) {
        // Fast path — single render, just like the non-chain CLI branch.
        sd_image_t* generated_video = nullptr;
        if (!generate_video(sd_ctx, &vid_gen_params, &generated_video, &num_results, &generated_audio)) {
            generated_video = nullptr;
        }
        results.adopt(generated_video, num_results);
    } else {
        // Multi-segment chained render. Port of cli/main.cpp's loop verbatim.
        const int cond_vframes     = std::max(1, gen_params.cont_cond_frames);
        const int seg_frames       = gen_params.video_frames;
        const int num_cond_latents = 1 + (cond_vframes - 1) / 4;
        const int cond_decoded_v   = 1 + (num_cond_latents - 1) * 4;
        if (cond_decoded_v >= seg_frames) {
            error_message = "cont_cond_frames overlap (" + std::to_string(cond_decoded_v) +
                            " decoded frames) must leave new frames in video_frames (" +
                            std::to_string(seg_frames) + ")";
            return false;
        }
        const int new_per_seg = seg_frames - cond_decoded_v;
        LOG_INFO("continuation: %d segments, overlap %d video frames = %d cond latents (%d decoded), %d new frames/seg",
                 n_segments, cond_vframes, num_cond_latents, cond_decoded_v, new_per_seg);

        sd_ctx_keep_diffusion_model_resident(sd_ctx, true);

        std::vector<float> stitched_audio;
        uint32_t audio_sr = 0;
        std::vector<sd_image_t> cond_tail_frames;
        std::vector<float> cond_latent_buf;
        bool raw_latent = std::getenv("LONGCAT_CONT_RAW_LATENT") != nullptr;
        std::vector<float> prev_latent;
        int prev_lw = 0, prev_lh = 0, prev_lt = 0, prev_lc = 0;

        bool use_ref_anchor = (std::getenv("LONGCAT_CONT_NO_REF_ANCHOR") == nullptr);
        std::vector<float> ref_anchor_latent;
        bool ref_anchor_ready = false;

        std::vector<sd_image_t> stitched;

        for (int seg = 0; seg < n_segments; ++seg) {
            if (seg > 0 && use_ref_anchor && ref_anchor_ready) {
                vid_gen_params.cont_ref_latent       = ref_anchor_latent.data();
                vid_gen_params.cont_ref_img_index    = gen_params.cont_ref_img_index;
                vid_gen_params.cont_mask_frame_range = gen_params.cont_mask_frame_range;
            } else {
                vid_gen_params.cont_ref_latent = nullptr;
            }
            if (seg == 0) {
                vid_gen_params.cont_latent        = nullptr;
                vid_gen_params.cont_latent_frames = 0;
                vid_gen_params.audio_frame_offset = 0;
            } else if (raw_latent) {
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
                int rlw = 0, rlh = 0, rlt = 0, rlc = 0;
                float* reenc = sd_ctx_encode_video_frames(
                    sd_ctx, cond_tail_frames.data(), (int)cond_tail_frames.size(),
                    gen_params.width, gen_params.height, &rlw, &rlh, &rlt, &rlc);
                if (reenc == nullptr) {
                    error_message = "continuation re-encode failed at segment " + std::to_string(seg);
                    for (auto& f : cond_tail_frames) free(f.data);
                    return false;
                }
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
                vid_gen_params.cont_latent        = cond_latent_buf.data();
                vid_gen_params.cont_latent_frames = keep;
                vid_gen_params.audio_frame_offset = seg * new_per_seg;
            }
            vid_gen_params.seed = (gen_params.seed < 0) ? gen_params.seed : gen_params.seed + seg;

            LOG_INFO("=== continuation segment %d/%d (audio_offset=%d frames) ===",
                     seg + 1, n_segments, vid_gen_params.audio_frame_offset);

            sd_image_t* seg_video = nullptr;
            int seg_count         = 0;
            sd_audio_t* seg_audio = nullptr;
            bool want_latent      = raw_latent || (use_ref_anchor && seg == 0);
            float* lat_out        = nullptr;
            int lw = 0, lh = 0, lt = 0, lc = 0;
            if (!generate_video_ex(sd_ctx, &vid_gen_params, &seg_video, &seg_count, &seg_audio,
                                   want_latent ? &lat_out : nullptr,
                                   want_latent ? &lw : nullptr, want_latent ? &lh : nullptr,
                                   want_latent ? &lt : nullptr, want_latent ? &lc : nullptr)) {
                free_sd_audio(seg_audio);
                free(seg_video);
                free(lat_out);
                for (auto& f : cond_tail_frames) free(f.data);
                error_message = "continuation segment " + std::to_string(seg + 1) + " failed";
                return false;
            }
            if (raw_latent && lat_out != nullptr) {
                prev_latent.assign(lat_out, lat_out + (size_t)lw * lh * lt * lc);
                prev_lw = lw; prev_lh = lh; prev_lt = lt; prev_lc = lc;
            }
            if (use_ref_anchor && seg == 0 && lat_out != nullptr && lt > 0) {
                size_t plane = (size_t)lw * lh;
                ref_anchor_latent.assign(plane * (size_t)lc, 0.f);
                for (int c = 0; c < lc; ++c) {
                    const float* src = lat_out + ((size_t)c * lt + 0) * plane;
                    std::memcpy(ref_anchor_latent.data() + (size_t)c * plane, src, plane * sizeof(float));
                }
                ref_anchor_ready = true;
            }
            if (lat_out != nullptr) free(lat_out);

            int drop = (seg == 0) ? 0 : cond_decoded_v;
            if (seg > 0 && std::getenv("LONGCAT_CONT_NO_EXPOSURE_MATCH") == nullptr) {
                float strength = 1.0f;
                if (const char* se = std::getenv("LONGCAT_CONT_EXPOSURE_STRENGTH"))
                    strength = (float)std::atof(se);
                continuity_match_segment(seg_video, seg_count, drop, strength);
            }
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
                    free(seg_video[i].data);
                } else {
                    stitched.push_back(seg_video[i]);
                }
            }
            free(seg_video);

            if (seg_audio != nullptr && seg_audio->data != nullptr) {
                audio_sr            = seg_audio->sample_rate;
                size_t drop_samples = (size_t)((double)drop / 25.0 * (double)audio_sr);
                if (drop_samples > seg_audio->sample_count) drop_samples = seg_audio->sample_count;
                stitched_audio.insert(stitched_audio.end(),
                                      seg_audio->data + drop_samples,
                                      seg_audio->data + seg_audio->sample_count);
            }
            free_sd_audio(seg_audio);
        }
        for (auto& f : cond_tail_frames) free(f.data);
        cond_tail_frames.clear();

        num_results = (int)stitched.size();
        // adopt() takes a malloc'd array; copy into one.
        sd_image_t* stitched_arr = (sd_image_t*)malloc(sizeof(sd_image_t) * num_results);
        std::memcpy(stitched_arr, stitched.data(), sizeof(sd_image_t) * num_results);
        results.adopt(stitched_arr, num_results);

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
    }

    num_results = results.count();
    if (num_results <= 0) {
        free_sd_audio(generated_audio);
        error_message = "generate_video returned no results";
        return false;
    }

    out_bytes = create_video_from_sd_images_to_vector(output_format,
                                                      results.data(),
                                                      num_results,
                                                      gen_params.fps,
                                                      output_quality,
                                                      generated_audio);
    free_sd_audio(generated_audio);

    if (out_bytes.empty()) {
        error_message = "video container encode failed";
        return false;
    }
    out_segments_rendered = std::max(1, gen_params.segments);
    return true;
}
