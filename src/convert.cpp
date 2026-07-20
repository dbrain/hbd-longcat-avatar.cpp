#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <thread>
#include <vector>

#include "core/ggml_extend.hpp"  // SDImatrixCollector (nvfp4-twolevel)
#include "gguf.h"                 // gguf_init_from_file (imatrix load)
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/streaming_writer.h"
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

// ---------------------------------------------------------------------------
// Streaming / bounded-memory export (opt-in; the whole-ctx convert() below is
// left untouched as the default). Tensors are loaded + quantized + written to
// disk one at a time under a fixed memory budget across N worker threads, so a
// whole model never lives in one ggml_ctx. The per-tensor load routes through
// ModelLoader::load_tensor -> load_tensors -> convert_tensor, so the active
// g_convert_imatrix (set by convert_with_imatrix) applies automatically.
// ---------------------------------------------------------------------------
struct TensorExportInfo {
    TensorStorage storage;
    ggml_type type;
};

struct TensorExportJob {
    TensorExportInfo info;
    std::vector<uint8_t> data;
    std::string error;
    bool success = false;
};

static bool collect_tensors_for_export(ModelLoader& model_loader,
                                       ggml_type type,
                                       const TensorTypeRules& tensor_type_rules,
                                       std::vector<TensorExportInfo>& tensors) {
    tensors.clear();
    tensors.reserve(model_loader.get_tensor_storage_map().size());
    for (const auto& kv : model_loader.get_tensor_storage_map()) {
        const TensorStorage& tensor_storage = kv.second;
        TensorExportInfo info;
        info.storage = tensor_storage;
        info.type    = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);
        tensors.push_back(std::move(info));
    }
    LOG_INFO("collected %zu tensors for export", tensors.size());
    return true;
}

static size_t export_tensor_nbytes(const TensorExportInfo& info) {
    TensorStorage output_storage = info.storage;
    output_storage.type          = info.type;
    return static_cast<size_t>(output_storage.nbytes());
}

static TensorWritePlan tensor_write_plan_from_export_info(const TensorExportInfo& info) {
    TensorWritePlan plan;
    plan.name   = info.storage.name;
    plan.type   = info.type;
    plan.n_dims = info.storage.n_dims;
    for (int i = 0; i < SD_MAX_DIMS; i++) {
        plan.ne[i] = info.storage.ne[i];
    }
    return plan;
}

static std::vector<TensorWritePlan> tensor_write_plans_from_export_infos(const std::vector<TensorExportInfo>& tensors) {
    std::vector<TensorWritePlan> plans;
    plans.reserve(tensors.size());
    for (const TensorExportInfo& info : tensors) {
        plans.push_back(tensor_write_plan_from_export_info(info));
    }
    return plans;
}

static bool preallocate_output_file(const std::string& output_path, uint64_t file_size, std::string* error) {
    if (file_size == 0) {
        return true;
    }
    std::fstream file(output_path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file.is_open()) {
        if (error != nullptr) {
            *error = "failed to open output file '" + output_path + "' for preallocation";
        }
        return false;
    }
    file.seekp(static_cast<std::streamoff>(file_size - 1), std::ios::beg);
    file.put('\0');
    file.flush();
    if (!file) {
        if (error != nullptr) {
            *error = "failed to preallocate output file '" + output_path + "'";
        }
        return false;
    }
    return true;
}

static bool load_tensor_for_export(ModelLoader& model_loader, TensorExportJob& job) {
    size_t mem_size = 1 * 1024 * 1024;
    mem_size += ggml_tensor_overhead();
    TensorStorage output_storage = job.info.storage;
    output_storage.type          = job.info.type;
    mem_size += static_cast<size_t>(output_storage.nbytes());

    ggml_context* ggml_ctx = ggml_init({mem_size, nullptr, false});
    if (ggml_ctx == nullptr) {
        job.error = "ggml_init failed for tensor '" + job.info.storage.name + "'";
        return false;
    }
    ggml_tensor* tensor = ggml_new_tensor(ggml_ctx, job.info.type, job.info.storage.n_dims, job.info.storage.ne);
    if (tensor == nullptr) {
        ggml_free(ggml_ctx);
        job.error = "ggml_new_tensor failed for tensor '" + job.info.storage.name + "'";
        return false;
    }
    ggml_set_name(tensor, job.info.storage.name.c_str());

    const size_t tensor_nbytes = ggml_nbytes(tensor);
    if (tensor_nbytes > 0 && !model_loader.load_tensor(job.info.storage, tensor)) {
        ggml_free(ggml_ctx);
        job.error = "failed to load tensor '" + job.info.storage.name + "'";
        return false;
    }
    job.data.resize(tensor_nbytes);
    if (tensor_nbytes > 0) {
        memcpy(job.data.data(), tensor->data, tensor_nbytes);
    }
    ggml_free(ggml_ctx);
    return true;
}

static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::string& output_path,
                               const std::vector<TensorExportInfo>& tensors,
                               const StreamingModelWriter& writer,
                               int n_threads,
                               std::string* error) {
    n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    n_threads = std::max(1, n_threads);
    LOG_INFO("streaming convert with %d threads", n_threads);

    int64_t start_time     = ggml_time_ms();
    uint64_t bytes_written = 0;
    size_t tensors_written = 0;
    size_t next_tensor_index = 0;
    bool failed            = false;
    std::string failure;

    const size_t memory_budget = 1024ull * 1024ull * 1024ull;
    size_t reserved_bytes      = 0;

    std::mutex work_mutex;
    std::mutex progress_mutex;
    std::condition_variable memory_cv;
    std::vector<std::thread> workers;
    workers.reserve(n_threads);

    auto reserve_memory = [&](size_t bytes) -> bool {
        std::unique_lock<std::mutex> lock(work_mutex);
        memory_cv.wait(lock, [&]() {
            return failed || reserved_bytes == 0 || reserved_bytes + bytes <= memory_budget;
        });
        if (failed) {
            return false;
        }
        reserved_bytes += bytes;
        return true;
    };
    auto release_memory = [&](size_t bytes) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            reserved_bytes -= std::min(reserved_bytes, bytes);
        }
        memory_cv.notify_all();
    };
    auto fail = [&](const std::string& message) {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            if (!failed) {
                failed  = true;
                failure = message;
            }
        }
        memory_cv.notify_all();
    };

    for (int worker = 0; worker < n_threads; worker++) {
        workers.emplace_back([&]() {
            std::fstream output_file(output_path, std::ios::binary | std::ios::in | std::ios::out);
            if (!output_file.is_open()) {
                fail("failed to open output file '" + output_path + "' for tensor writing");
                return;
            }
            while (true) {
                size_t tensor_index = 0;
                {
                    std::lock_guard<std::mutex> lock(work_mutex);
                    if (failed || next_tensor_index >= tensors.size()) {
                        return;
                    }
                    tensor_index = next_tensor_index++;
                }
                const size_t tensor_bytes = export_tensor_nbytes(tensors[tensor_index]);
                if (!reserve_memory(tensor_bytes)) {
                    return;
                }
                TensorExportJob job;
                job.info = tensors[tensor_index];
                try {
                    job.success = load_tensor_for_export(model_loader, job);
                } catch (const std::exception& e) {
                    job.error   = e.what();
                    job.success = false;
                }
                if (!job.success) {
                    release_memory(tensor_bytes);
                    fail(job.error.empty() ? "streaming conversion failed" : job.error);
                    return;
                }
                std::string write_error;
                if (!writer.write_tensor(output_file,
                                         tensor_index,
                                         job.data.empty() ? nullptr : job.data.data(),
                                         job.data.size(),
                                         &write_error)) {
                    release_memory(tensor_bytes);
                    fail(write_error.empty() ? "streaming conversion write failed" : write_error);
                    return;
                }
                {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    bytes_written += job.data.size();
                    tensors_written++;
                    float elapsed_seconds = (ggml_time_ms() - start_time) / 1000.0f;
                    pretty_bytes_progress(static_cast<int>(tensors_written),
                                          static_cast<int>(tensors.size()),
                                          bytes_written,
                                          elapsed_seconds);
                }
                release_memory(tensor_bytes);
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    printf("\n");
    if (failed) {
        if (error != nullptr) {
            *error = failure;
        }
        return false;
    }
    LOG_INFO("streaming conversion completed, taking %.2fs", (ggml_time_ms() - start_time) / 1000.f);
    return true;
}

static bool write_model_file_streaming(ModelLoader& model_loader,
                                       const std::string& output_path,
                                       const std::vector<TensorExportInfo>& tensors,
                                       StreamingModelWriter& writer,
                                       int n_threads,
                                       std::string* error) {
    std::vector<TensorWritePlan> plans = tensor_write_plans_from_export_infos(tensors);
    if (!writer.write_metadata(output_path, plans, error)) {
        return false;
    }
    if (!preallocate_output_file(output_path, writer.file_size(), error)) {
        return false;
    }
    model_loader.process_model_files(false, false);  // pin mmap OFF before threaded per-tensor loads
    return stream_tensor_data(model_loader, output_path, tensors, writer, n_threads, error);
}

static bool export_loaded_model_streaming(ModelLoader& model_loader,
                                          const char* output_path,
                                          sd_type_t output_type,
                                          const char* tensor_type_rules,
                                          int n_threads) {
    ggml_type type             = (ggml_type)output_type;  // fork convention (no sd_type_to_ggml_type)
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    std::vector<TensorExportInfo> tensors;
    bool success = collect_tensors_for_export(model_loader, type, type_rules, tensors);
    std::string error;
    if (success) {
        std::unique_ptr<StreamingModelWriter> writer;
        if (output_is_safetensors) {
            writer = std::make_unique<SafetensorsStreamingWriter>();
        } else {
            writer = std::make_unique<GGUFStreamingWriter>();
        }
        success = write_model_file_streaming(model_loader, output_path, tensors, *writer, n_threads, &error);
    }
    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }
    return success;
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

// Opt-in streaming/bounded-memory convert of multiple components. Unlike convert()
// (which holds the whole model in one ggml_ctx), this loads+quantizes+writes each
// tensor individually under a fixed memory budget. The default convert() path is
// left untouched. The active g_convert_imatrix (set by a caller around this, e.g.
// convert_with_imatrix's set/clear bracket) applies automatically via load_tensor.
bool convert_with_components(const char* model_path,
                             const char* clip_l_path,
                             const char* clip_g_path,
                             const char* t5xxl_path,
                             const char* diffusion_model_path,
                             const char* vae_path,
                             const char* output_path,
                             sd_type_t output_type,
                             const char* tensor_type_rules,
                             bool convert_name,
                             int n_threads) {
    ModelLoader model_loader;
    bool loaded_any = false;
    auto init_path  = [&](const char* path, const char* prefix) -> bool {
        if (path == nullptr || strlen(path) == 0) {
            return true;
        }
        if (!model_loader.init_from_file(path, prefix)) {
            LOG_ERROR("init model loader from file failed: '%s'", path);
            return false;
        }
        loaded_any = true;
        return true;
    };

    if (!init_path(model_path, "") ||
        !init_path(clip_l_path, "text_encoders.clip_l.transformer.") ||
        !init_path(clip_g_path, "text_encoders.clip_g.transformer.") ||
        !init_path(t5xxl_path, "text_encoders.t5xxl.transformer.") ||
        !init_path(diffusion_model_path, "model.diffusion_model.") ||
        !init_path(vae_path, "vae.")) {
        return false;
    }
    if (!loaded_any) {
        LOG_ERROR("no input model path provided for convert");
        return false;
    }
    if (convert_name) {
        model_loader.convert_tensors_name();
    }
    return export_loaded_model_streaming(model_loader, output_path, output_type, tensor_type_rules, n_threads);
}
