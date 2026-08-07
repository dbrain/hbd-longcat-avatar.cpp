#pragma once

#include <string>

#include "common/common.h"
#include "httplib.h"
#include "runtime.h"

// Every binary part of a multipart request, by part name, ready to be stored on the job so a
// `part:<name>` media string can be resolved when the worker decodes it
// (`docs/media-transport.md` §4).
//
// Takes ALL file parts, including ones the route consumes itself (`audio_full`, `init_image`, …).
// They are copied either way, and a filter here would be a second list of names to keep in step
// with the route -- exactly the kind of drift the `part:` design exists to avoid. A request that
// names no parts collects nothing.
MediaPartTable collect_media_parts(const httplib::Request& req);

// Hash-named parts are self-verifying: `a_<first 16 hex of the sha256>` is what koblem's
// `stage_refs` emits, so the receiver can prove the bytes are the bytes that were meant
// (§9.3). Returns the name of the first part whose content does not match its name, or "" when
// every checkable part checks out. Parts whose names are not hash-shaped are not checked.
std::string first_media_part_hash_mismatch(const MediaPartTable& parts);

void register_index_endpoints(httplib::Server& svr, const SDSvrParams& svr_params, const std::string& index_html);
void register_openai_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdapi_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_wan_video_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_ltx_video_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_longcat_avatar_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_gpu_sharing_endpoints(httplib::Server& svr, ServerRuntime& rt);
