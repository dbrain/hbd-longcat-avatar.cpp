// avatar_render.h — LongCat-Avatar shared chaining helper.
//
// Both the cli (`sd-cli vid_gen`) and the native server
// (`longcat-avatar-server::/generate`) need to drive the same multi-segment
// avatar pipeline (segments > 1 -> chained renders with cont_latent +
// reference anchor + drift-sink re-encode + audio stitching). This header
// exposes that as a single entry point so the server doesn't have to
// duplicate ~250 lines of cli/main.cpp.
//
// `gen_params` is mutated (the call sets cont_* / audio_frame_offset / seed
// per segment); callers should pass a freshly-built copy from
// SDGenerationParams::from_json_str.
//
// Returns true on success. On failure, error_message is populated. Output
// is the encoded video container (webm/avi/webp depending on output_format).
//
// Thread-safety: caller must hold the sd_ctx mutex for the duration of the
// call. Internally sd_ctx_keep_diffusion_model_resident is flipped to true
// for chained renders so the DiT survives across segments.

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "common/common.h"

bool render_avatar_to_video_bytes(sd_ctx_t* sd_ctx,
                                  SDGenerationParams& gen_params,
                                  const std::string& output_format,
                                  int output_quality,
                                  std::vector<uint8_t>& out_bytes,
                                  int& out_segments_rendered,
                                  std::string& error_message);
