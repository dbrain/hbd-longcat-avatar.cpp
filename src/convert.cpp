#include <inttypes.h>
#include <algorithm>
#include <cctype>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#include "core/util.h"
#include "model/diffusion/minimax_h3_qk_permute.hpp"
#include "model_io/gguf_io.h"
#include "model_io/safetensors_io.h"
#include "model_io/streaming_writer.h"
#include "model_loader.h"

// How a tensor's head-channel axis is rewritten on the way out.  See the H3 section below.
enum class QKPermuteMode {
    None,
    QKVRows,   // 2-D [in, 3*inner]: permute channels within each head of the q and k thirds
    NormElems  // 1-D [head_dim]: permute the channels themselves
};

struct TensorExportInfo {
    TensorStorage storage;
    ggml_type type;
    QKPermuteMode qk_permute = QKPermuteMode::None;
};

struct TensorExportJob {
    TensorExportInfo info;
    std::vector<uint8_t> data;
    std::string error;
    bool success = false;
};

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

/*======================================= MiniMax-H3 q/k RoPE permutation ==============================*/

// Rewrite the head-channel axis of the H3 DiT's q/k projections so the runtime can rope the full
// head in one call instead of slicing, roping and concatenating.  The maths, the proof that it is an
// exact identity, and the numeric verification live in model/diffusion/minimax_h3_qk_permute.hpp.
//
// Scope: `<prefix>blocks.<N>.attn.{qkv_proj,q_norm,k_norm}.weight` only.  `token_refiner.blocks.*`
// is deliberately excluded -- the refiner has no rotary embedding at all, so permuting it would be
// churn that buys nothing.  Nothing else in this file or any other model is touched.
struct H3QKPermutePlan {
    bool active = false;
    std::string prefix;  // includes the trailing '.', e.g. "model.diffusion_model."
    int64_t head_dim = 0;
    int64_t rot_dim  = 0;
    std::vector<int64_t> perm;  // out[i] = in[perm[i]]
};

static bool h3_block_attn_leaf(const std::string& name, const std::string& prefix, const char* leaf) {
    // <prefix>blocks.<digits>.attn.<leaf>, and NOT <prefix>token_refiner.blocks...
    const std::string head = prefix + "blocks.";
    if (!starts_with(name, head)) {
        return false;
    }
    size_t i = head.size();
    if (i >= name.size() || !isdigit(static_cast<unsigned char>(name[i]))) {
        return false;
    }
    while (i < name.size() && isdigit(static_cast<unsigned char>(name[i]))) {
        i++;
    }
    return name.compare(i, std::string::npos, std::string(".attn.") + leaf) == 0;
}

static H3QKPermutePlan h3_plan_qk_permutation(const ModelLoader& model_loader) {
    H3QKPermutePlan plan;
    const String2TensorStorage& map = model_loader.get_tensor_storage_map();

    const std::string marker = MiniMaxH3::qk_permuted_marker_name();
    const std::string anchor = "video_patch_proj.weight";
    std::string prefix;
    for (const auto& [name, _] : map) {
        if (name.size() > anchor.size() && ends_with(name, anchor)) {
            const std::string candidate = name.substr(0, name.size() - anchor.size());
            if (map.count(candidate + "audio_patch_proj.weight") != 0) {
                prefix = candidate;
                break;
            }
        }
    }
    if (prefix.empty()) {
        return plan;
    }
    if (map.count(prefix + marker) != 0) {
        LOG_INFO("minimax-h3: q/k head channels are already permuted, passing them through");
        return plan;
    }

    auto find = [&](const std::string& n) -> const TensorStorage* {
        auto it = map.find(prefix + n);
        return it == map.end() ? nullptr : &it->second;
    };
    const TensorStorage* q_norm = find("blocks.0.attn.q_norm.weight");
    const TensorStorage* qkv    = find("blocks.0.attn.qkv_proj.weight");
    if (q_norm == nullptr || q_norm->n_dims != 1 || qkv == nullptr || qkv->n_dims != 2) {
        LOG_WARN("minimax-h3: no blocks.0.attn q/k weights found, skipping the RoPE permutation");
        return plan;
    }
    plan.head_dim = q_norm->ne[0];
    // rope.inv_freq is a persistent buffer in the reference spelling; diffusers computes it instead,
    // so fall back to the reference's 16 rather than refusing.
    const TensorStorage* inv = find("rope.inv_freq");
    plan.rot_dim             = 2 * 3 * ((inv != nullptr && inv->n_dims == 1) ? inv->ne[0] : 16);

    if (!MiniMaxH3::qk_permutation_is_applicable(plan.head_dim, plan.rot_dim)) {
        LOG_WARN("minimax-h3: head_dim %" PRId64 " / rot_dim %" PRId64 " admits no q/k permutation, skipping",
                 plan.head_dim,
                 plan.rot_dim);
        return plan;
    }
    if (plan.rot_dim == plan.head_dim) {
        // Split-half over the whole head is already H3's own pairing; there is nothing to rewrite
        // and no slice/concat to remove.
        return plan;
    }
    if (qkv->ne[1] % (3 * plan.head_dim) != 0) {
        LOG_WARN("minimax-h3: qkv_proj out dim %" PRId64 " is not 3 * heads * %" PRId64 ", skipping the permutation",
                 qkv->ne[1],
                 plan.head_dim);
        return plan;
    }

    plan.perm   = MiniMaxH3::build_qk_head_permutation(plan.head_dim, plan.rot_dim);
    plan.prefix = prefix;
    plan.active = true;
    LOG_INFO("minimax-h3: permuting q/k head channels for full-width split-half RoPE (head_dim=%" PRId64
             ", rot_dim=%" PRId64 ")",
             plan.head_dim,
             plan.rot_dim);
    return plan;
}

// Add the marker the runtime keys off.  Written through the tensor storage map rather than appended
// to the export list so that load_tensor() -- which resolves by name against the map -- can find it.
static void h3_add_permuted_marker(ModelLoader& model_loader, const H3QKPermutePlan& plan) {
    String2TensorStorage& map     = model_loader.get_tensor_storage_map();
    const std::string anchor      = plan.prefix + "video_patch_proj.weight";
    const int64_t ne[SD_MAX_DIMS] = {1, 1, 1, 1, 1};

    TensorStorage marker(plan.prefix + MiniMaxH3::qk_permuted_marker_name(),
                         GGML_TYPE_F32,
                         ne,
                         1,
                         map.at(anchor).file_index,
                         0);
    marker.is_inline_f32 = true;
    marker.inline_f32    = 1.0f;
    map[marker.name]     = marker;
}

static QKPermuteMode h3_permute_mode(const std::string& name, const H3QKPermutePlan& plan) {
    if (!plan.active) {
        return QKPermuteMode::None;
    }
    if (h3_block_attn_leaf(name, plan.prefix, "qkv_proj.weight")) {
        return QKPermuteMode::QKVRows;
    }
    if (h3_block_attn_leaf(name, plan.prefix, "q_norm.weight") ||
        h3_block_attn_leaf(name, plan.prefix, "k_norm.weight")) {
        return QKPermuteMode::NormElems;
    }
    return QKPermuteMode::None;
}

// Apply the permutation to already-typed export bytes.
//
// QKVRows is a pure row shuffle along the OUTPUT axis, so it is exact for every type including a
// quantised one: a row is `ne[0]` elements, which is a whole number of quant blocks, and the
// quantiser works per row -- quantise-then-permute and permute-then-quantise are the same bytes.
// NormElems moves individual elements, which is only meaningful when a "block" is one element;
// collect_tensors_for_export keeps those two tiny 1-D tensors at their source type for exactly that
// reason (and because 4-bit-quantising a per-channel RMSNorm gain was never a good idea).
static bool h3_apply_permute(const TensorExportInfo& info, const H3QKPermutePlan& plan, std::vector<uint8_t>& data) {
    if (info.qk_permute == QKPermuteMode::None) {
        return true;
    }
    const int64_t head_dim           = plan.head_dim;
    const std::vector<int64_t>& perm = plan.perm;

    std::vector<uint8_t> scratch;

    if (info.qk_permute == QKPermuteMode::NormElems) {
        const size_t esz = ggml_type_size(info.type);
        if (ggml_blck_size(info.type) != 1 || info.storage.ne[0] != head_dim ||
            data.size() != static_cast<size_t>(head_dim) * esz) {
            LOG_ERROR("minimax-h3: cannot permute '%s' as %s", info.storage.name.c_str(), ggml_type_name(info.type));
            return false;
        }
        // One "row" is one element here, so the same helper does the job.
        MiniMaxH3::permute_head_rows(data.data(), esz, 1, perm, scratch);
        return true;
    }

    const size_t row_bytes = ggml_row_size(info.type, info.storage.ne[0]);
    const int64_t rows     = info.storage.nelements() / info.storage.ne[0];
    const int64_t inner    = rows / 3;
    if (row_bytes == 0 || rows % 3 != 0 || inner % head_dim != 0 ||
        data.size() != static_cast<size_t>(rows) * row_bytes) {
        LOG_ERROR("minimax-h3: cannot permute '%s' (%" PRId64 " rows of %zu bytes)",
                  info.storage.name.c_str(),
                  rows,
                  row_bytes);
        return false;
    }

    // q and k only; v keeps its channel order because out_proj consumes it.
    for (int64_t third = 0; third < 2; third++) {
        MiniMaxH3::permute_head_rows(data.data() + static_cast<size_t>(third * inner) * row_bytes,
                                     row_bytes,
                                     inner / head_dim,
                                     perm,
                                     scratch);
    }
    return true;
}

/*=====================================================================================================*/

static bool collect_tensors_for_export(ModelLoader& model_loader,
                                       ggml_type type,
                                       const TensorTypeRules& tensor_type_rules,
                                       const H3QKPermutePlan& h3_plan,
                                       std::vector<TensorExportInfo>& tensors) {
    tensors.clear();
    tensors.reserve(model_loader.get_tensor_storage_map().size());
    const std::string h3_marker = h3_plan.active ? h3_plan.prefix + MiniMaxH3::qk_permuted_marker_name() : "";
    for (const auto& kv : model_loader.get_tensor_storage_map()) {
        const TensorStorage& tensor_storage = kv.second;
        TensorExportInfo info;
        info.storage    = tensor_storage;
        info.type       = get_export_tensor_type(model_loader, tensor_storage, type, tensor_type_rules);
        info.qk_permute = h3_permute_mode(tensor_storage.name, h3_plan);
        if (info.qk_permute == QKPermuteMode::NormElems || tensor_storage.name == h3_marker) {
            // Both are per-element, not per-row: a quantised destination would make the permutation
            // meaningless for the norms and the 4-byte marker unreadable.
            info.type = tensor_storage.type;
        }
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

    // This portable fallback sets the final file size. A platform-specific
    // posix_fallocate/ftruncate path can replace it later.
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

static bool load_tensor_for_export(ModelLoader& model_loader, const H3QKPermutePlan& h3_plan, TensorExportJob& job) {
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

    if (!h3_apply_permute(job.info, h3_plan, job.data)) {
        job.error = "failed to permute q/k head channels for '" + job.info.storage.name + "'";
        return false;
    }
    return true;
}

static bool stream_tensor_data(ModelLoader& model_loader,
                               const std::string& output_path,
                               const std::vector<TensorExportInfo>& tensors,
                               const StreamingModelWriter& writer,
                               const H3QKPermutePlan& h3_plan,
                               int n_threads,
                               std::string* error) {
    n_threads = n_threads > 0 ? n_threads : sd_get_num_physical_cores();
    n_threads = std::max(1, n_threads);
    LOG_INFO("streaming convert with %d threads", n_threads);

    int64_t start_time       = ggml_time_ms();
    uint64_t bytes_written   = 0;
    size_t tensors_written   = 0;
    size_t next_tensor_index = 0;
    bool failed              = false;
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
                    job.success = load_tensor_for_export(model_loader, h3_plan, job);
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
                                       const H3QKPermutePlan& h3_plan,
                                       int n_threads,
                                       std::string* error) {
    std::vector<TensorWritePlan> plans = tensor_write_plans_from_export_infos(tensors);
    if (!writer.write_metadata(output_path, plans, error)) {
        return false;
    }
    if (!preallocate_output_file(output_path, writer.file_size(), error)) {
        return false;
    }
    model_loader.process_model_files(false, false);
    return stream_tensor_data(model_loader, output_path, tensors, writer, h3_plan, n_threads, error);
}

static bool init_convert_path(ModelLoader& model_loader, const char* path, const char* prefix, bool& loaded_any) {
    if (path == nullptr || strlen(path) == 0) {
        return true;
    }
    if (!model_loader.init_from_file(path, prefix)) {
        LOG_ERROR("init model loader from file failed: '%s'", path);
        return false;
    }
    loaded_any = true;
    return true;
}

static bool export_loaded_model(ModelLoader& model_loader,
                                const char* output_path,
                                sd_type_t output_type,
                                const char* tensor_type_rules,
                                int n_threads) {
    ggml_type type             = sd_type_to_ggml_type(output_type);
    bool output_is_safetensors = ends_with(output_path, ".safetensors");
    TensorTypeRules type_rules = parse_tensor_type_rules(tensor_type_rules);

    // MiniMax-H3 only, and only when the input is not already permuted; every other model gets an
    // inactive plan and an untouched export path.
    H3QKPermutePlan h3_plan = h3_plan_qk_permutation(model_loader);
    if (h3_plan.active) {
        h3_add_permuted_marker(model_loader, h3_plan);
    }

    std::vector<TensorExportInfo> tensors;
    bool success = collect_tensors_for_export(model_loader, type, type_rules, h3_plan, tensors);
    std::string error;
    if (success) {
        std::unique_ptr<StreamingModelWriter> writer;
        if (output_is_safetensors) {
            writer = std::make_unique<SafetensorsStreamingWriter>();
        } else {
            writer = std::make_unique<GGUFStreamingWriter>();
        }
        success = write_model_file_streaming(model_loader, output_path, tensors, *writer, h3_plan, n_threads, &error);
    }

    if (!success && !error.empty()) {
        LOG_ERROR("%s", error.c_str());
    }

    return success;
}

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

    if (!init_convert_path(model_loader, model_path, "", loaded_any) ||
        !init_convert_path(model_loader, clip_l_path, "text_encoders.clip_l.transformer.", loaded_any) ||
        !init_convert_path(model_loader, clip_g_path, "text_encoders.clip_g.transformer.", loaded_any) ||
        !init_convert_path(model_loader, t5xxl_path, "text_encoders.t5xxl.transformer.", loaded_any) ||
        !init_convert_path(model_loader, diffusion_model_path, "model.diffusion_model.", loaded_any) ||
        !init_convert_path(model_loader, vae_path, "vae.", loaded_any)) {
        return false;
    }

    if (!loaded_any) {
        LOG_ERROR("no input model path provided for convert");
        return false;
    }

    if (convert_name) {
        model_loader.convert_tensors_name();
    }

    return export_loaded_model(model_loader, output_path, output_type, tensor_type_rules, n_threads);
}

bool convert(const char* input_path,
             const char* vae_path,
             const char* output_path,
             sd_type_t output_type,
             const char* tensor_type_rules,
             bool convert_name) {
    return convert_with_components(input_path,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   vae_path,
                                   output_path,
                                   output_type,
                                   tensor_type_rules,
                                   convert_name,
                                   0);
}
