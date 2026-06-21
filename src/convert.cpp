#include <cstring>
#include <map>
#include <mutex>
#include <regex>
#include <vector>

#include "core/ggml_extend.hpp"  // SDImatrixCollector (nvfp4-twolevel)
#include "gguf.h"                 // gguf_init_from_file (imatrix load)
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_loader.h"
#include "stable-diffusion.h"
#include "util.h"

#include "ggml_extend_backend.h"

static ggml_type get_export_tensor_type(ModelLoader& model_loader,
                                        const TensorStorage& tensor_storage,
                                        ggml_type type,
                                        const TensorTypeRules& tensor_type_rules) {
    const std::string& name = tensor_storage.name;
    ggml_type tensor_type   = tensor_storage.type;
    ggml_type dst_type      = type;

    for (const auto& tensor_type_rule : tensor_type_rules) {
        std::regex pattern(tensor_type_rule.first);
        if (std::regex_search(name, pattern)) {
            dst_type = tensor_type_rule.second;
            break;
        }
    }

    if (model_loader.tensor_should_be_converted(tensor_storage, dst_type)) {
        tensor_type = dst_type;
    }

    return tensor_type;
}

static bool load_tensors_for_export(ModelLoader& model_loader,
                                    ggml_context* ggml_ctx,
                                    ggml_type type,
                                    const TensorTypeRules& tensor_type_rules,
                                    std::vector<TensorWriteInfo>& tensors) {
    std::mutex tensor_mutex;
    auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
        const std::string& name = tensor_storage.name;
        ggml_type tensor_type   = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);

        std::lock_guard<std::mutex> lock(tensor_mutex);
        ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, tensor_type, tensor_storage.n_dims, tensor_storage.ne);
        if (tensor == nullptr) {
            LOG_ERROR("ggml_new_tensor failed");
            return false;
        }
        ggml_set_name(tensor, name.c_str());

        if (!tensor->data) {
            GGML_ASSERT(ggml_nelements(tensor) == 0);
            // Avoid crashing writers by setting a dummy pointer for zero-sized tensors.
            LOG_DEBUG("setting dummy pointer for zero-sized tensor %s", name.c_str());
            tensor->data = ggml_get_mem_buffer(ggml_ctx);
        }

        TensorWriteInfo write_info;
        write_info.tensor = tensor;
        write_info.n_dims = tensor_storage.n_dims;
        for (int i = 0; i < tensor_storage.n_dims; ++i) {
            write_info.ne[i] = tensor_storage.ne[i];
        }

        *dst_tensor = tensor;
        tensors.push_back(std::move(write_info));

        return true;
    };

    bool success = model_loader.load_tensors(on_new_tensor_cb);
    LOG_INFO("load tensors done");
    return success;
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    ModelLoader model_loader;

    if (!model_loader.init_from_file(input_path)) {
        LOG_ERROR("init model loader from file failed: '%s'", input_path);
        return false;
    }

    if (vae_path != nullptr && strlen(vae_path) > 0) {
        if (!model_loader.init_from_file(vae_path, "vae.")) {
            LOG_ERROR("init model loader from file failed: '%s'", vae_path);
            return false;
        }
    }
    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    ggml_type type             = (ggml_type)output_type;
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    auto backend    = sd_backend_cpu_init();
    size_t mem_size = 1 * 1024 * 1024;  // for padding
    mem_size += model_loader.get_tensor_storage_map().size() * ggml_tensor_overhead();
    // Rules-aware context sizing: get_params_mem_size() only knows the base
    // `type`, so a --tensor-type-rules entry that forces a tensor to a LARGER
    // type (e.g. keeping patchify_proj/proj_out/adaLN linears at f16 instead of
    // q4_K) would overflow the context and trip GGML_ASSERT(obj_new) at the tail
    // of the export. Sum each tensor's ACTUAL export size (rules applied).
    {
        size_t align = ggml_backend_get_alignment(backend);
        for (const auto& kv : model_loader.get_tensor_storage_map()) {
            TensorStorage ts = kv.second;
            ts.type          = get_export_tensor_type(model_loader, ts, type, type_rules);
            mem_size += ts.nbytes() + align;
        }
    }
    LOG_INFO("model tensors mem size: %.2fMB", mem_size / 1024.f / 1024.f);
    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});

    if (ggml_ctx == nullptr) {
        LOG_ERROR("ggml_init failed for converter");
        ggml_backend_free(backend);
        return false;
    }

    std::vector<TensorWriteInfo> tensors;
    bool success = load_tensors_for_export(model_loader, ggml_ctx, type, type_rules, tensors);
    ggml_backend_free(backend);

    std::string error;
    if (success) {
        if (output_is_safetensors) {
            success = write_safetensors_file(output_path, tensors, &error);
        } else {
            success = write_gguf_file(output_path, tensors, &error);
        }
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    ggml_free(ggml_ctx);
    return success;
}

// ===========================================================================
// nvfp4-twolevel: diffusion imatrix collection + imatrix-fed convert
// ===========================================================================

static std::string imx_strip_dit_prefix(const std::string& name) {
    static const std::string p1 = "model.diffusion_model.";
    static const std::string p2 = "diffusion_model.";
    if (name.rfind(p1, 0) == 0)
        return name.substr(p1.size());
    if (name.rfind(p2, 0) == 0)
        return name.substr(p2.size());
    return name;
}

void sd_imatrix_collect_begin(const char* name_filter) {
    auto& col = SDImatrixCollector::instance();
    col.set_name_filter(name_filter != nullptr ? std::string(name_filter) : std::string());
    col.enable(true);
    LOG_INFO("[imatrix] collection enabled (filter='%s')", name_filter ? name_filter : "");
}

bool sd_imatrix_collect_write_gguf(const char* out_path, int* n_written) {
    auto& col          = SDImatrixCollector::instance();
    const auto& entries = col.data();
    if (entries.empty()) {
        LOG_ERROR("[imatrix] no stats collected; nothing to write");
        if (n_written)
            *n_written = 0;
        return false;
    }

    // Build a CPU ggml context to hold one f32 1-D tensor per weight, named by
    // the prefix-stripped weight name. Value == mean per-column second moment
    // (sum(act^2) / n_tokens), which is the AWQ-style importance.
    size_t mem = ggml_tensor_overhead() * (entries.size() + 1);
    for (const auto& kv : entries) {
        mem += kv.second.sums.size() * sizeof(float) + 256;
    }
    ggml_context* ctx = ggml_init({mem, nullptr, false});
    if (ctx == nullptr) {
        LOG_ERROR("[imatrix] ggml_init failed");
        return false;
    }

    std::vector<TensorWriteInfo> tensors;
    int written = 0;
    for (const auto& kv : entries) {
        const std::string name = imx_strip_dit_prefix(kv.first);
        const auto& e          = kv.second;
        if (e.sums.empty() || e.n_tokens <= 0) {
            continue;
        }
        ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t)e.sums.size());
        ggml_set_name(t, name.c_str());
        float* dst = (float*)t->data;
        for (size_t i = 0; i < e.sums.size(); ++i) {
            dst[i] = (float)(e.sums[i] / (double)e.n_tokens);
        }
        TensorWriteInfo wi;
        wi.tensor  = t;
        wi.n_dims  = 1;
        wi.ne[0]   = (int64_t)e.sums.size();
        tensors.push_back(wi);
        written++;
    }

    std::string error;
    bool ok = write_gguf_file(out_path, tensors, &error);
    if (!ok) {
        LOG_ERROR("[imatrix] write failed: %s", error.c_str());
    } else {
        LOG_INFO("[imatrix] wrote %d tensors to %s", written, out_path);
    }
    ggml_free(ctx);
    if (n_written)
        *n_written = written;
    return ok;
}

// Load an imatrix gguf into a name->vector<float> map.
static bool load_imatrix_gguf(const std::string& path,
                              std::map<std::string, std::vector<float>>& out) {
    struct gguf_init_params gp = {/*no_alloc=*/false, /*ctx=*/nullptr};
    ggml_context* data_ctx     = nullptr;
    gp.ctx                     = &data_ctx;
    gguf_context* gctx         = gguf_init_from_file(path.c_str(), gp);
    if (gctx == nullptr) {
        LOG_ERROR("[imatrix] failed to open '%s'", path.c_str());
        return false;
    }
    const int n = (int)gguf_get_n_tensors(gctx);
    for (int i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(gctx, i);
        ggml_tensor* t   = ggml_get_tensor(data_ctx, name);
        if (t == nullptr || t->type != GGML_TYPE_F32) {
            continue;
        }
        const int64_t ne = ggml_nelements(t);
        std::vector<float> v((size_t)ne);
        memcpy(v.data(), t->data, (size_t)ne * sizeof(float));
        out[name] = std::move(v);
    }
    gguf_free(gctx);
    if (data_ctx)
        ggml_free(data_ctx);
    LOG_INFO("[imatrix] loaded %zu tensors from %s", out.size(), path.c_str());
    return !out.empty();
}

bool convert_with_imatrix(const char* input_path,
                          const char* vae_path,
                          const char* output_path,
                          enum sd_type_t output_type,
                          const char* tensor_type_rules,
                          bool convert_name,
                          const char* imatrix_path) {
    std::map<std::string, std::vector<float>> imatrix;
    bool have_imatrix = false;
    if (imatrix_path != nullptr && strlen(imatrix_path) > 0) {
        if (!load_imatrix_gguf(imatrix_path, imatrix)) {
            LOG_ERROR("convert: failed to load imatrix '%s'", imatrix_path);
            return false;
        }
        set_convert_imatrix(&imatrix);
        have_imatrix = true;
    }
    bool ok = convert(input_path, vae_path, output_path, output_type, tensor_type_rules, convert_name);
    if (have_imatrix) {
        set_convert_imatrix(nullptr);
    }
    return ok;
}
