#pragma once

#include <string>

#include "httplib.h"
#include "runtime.h"

void register_index_endpoints(httplib::Server& svr, const SDSvrParams& svr_params, const std::string& index_html);
void register_openai_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdapi_endpoints(httplib::Server& svr, ServerRuntime& rt);
void register_sdcpp_api_endpoints(httplib::Server& svr, ServerRuntime& rt);
// /health + /v1/admin/{drain,unload,load} for the external GPU gate (mirrors
// routes_longcat.cpp). Registered alongside register_sdcpp_api_endpoints.
void register_sdcpp_admin_endpoints(httplib::Server& svr, ServerRuntime& rt);
