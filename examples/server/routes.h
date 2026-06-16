#pragma once

#include <string>

#include <json.hpp>

#include "httplib.h"
#include "runtime.h"

// Parse + validate a /sdcpp/v1/img_gen body into an ImgGenJobRequest (resolves
// loras, output format, and the optional base/edit variant). Exposed so the
// worker-isolation child can reuse the exact same parse before rendering.
bool parse_img_gen_request(const nlohmann::json& body,
                           ServerRuntime& runtime,
                           ImgGenJobRequest& request,
                           std::string& error_message);

void register_index_endpoints(httplib::Server& svr, const SDSvrParams& svr_params, const std::string& index_html);
void register_openai_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdapi_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
// /health + /v1/admin/{drain,unload,load} for the external GPU gate (mirrors
// routes_longcat.cpp). Registered alongside register_sdcpp_api_endpoints.
void register_sdcpp_admin_endpoints(httplib::Server& svr, ServerRuntime& rt);

// POST /ltx/v1/generate — LTXAV multi-segment video chain submit (async). Accepts the
// chain request (segments[] prompts + base gen params + inline base64 init_image) as JSON
// or multipart (with per-segment audio_<i> wav parts the parent writes to a shared /tmp dir
// for lip-sync). Enqueues a VidGen job carrying the chain request JSON; poll/cancel reuse
// the /sdcpp/v1/jobs/{id} endpoints. Registered in LTX_VIDEO_ISOLATION mode.
void register_ltx_video_endpoints(httplib::Server& svr, ServerRuntime& rt);
