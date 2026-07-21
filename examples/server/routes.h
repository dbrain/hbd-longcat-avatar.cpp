#pragma once

#include <string>

#include "httplib.h"
#include "runtime.h"

void register_index_endpoints(httplib::Server& svr, const SDSvrParams& svr_params, const std::string& index_html);
void register_openai_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdapi_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_wan_video_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_ltx_video_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_longcat_avatar_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_gpu_sharing_endpoints(httplib::Server& svr, ServerRuntime& rt);
