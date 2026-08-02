#ifndef __SD_MODEL_ADAPTER_LORA_HPP__
#define __SD_MODEL_ADAPTER_LORA_HPP__

#include <algorithm>
#include <mutex>
#include <string>
#include <vector>
#include "core/ggml_extend.hpp"
#include "model_loader.h"
#include "model_manager.h"

#define LORA_GRAPH_BASE_SIZE 10240

// ---- runtime-LoRA accumulate direction (see forward_with_lora) -------------------------------
// SD_LORA_ACC_BASE=1        emit the delta ADD as add_inplace(delta, base) instead of
//                           add_inplace(base, delta), so the BASE gemm is the one the CUDA
//                           MUL_MAT+ADD fusion folds into. OFF by default: this changes float
//                           accumulation order and which kernel is fused, so it is opt-in until
//                           measured per model.
// SD_LORA_ACC_BASE_SKIP=a,b comma-separated substrings matched against the patched Linear's
//                           prefix; a match keeps that module on the forward ordering. Exists
//                           because the reversal only pays where the base gemm has spare
//                           bandwidth (e.g. "mlp.down" measured at 96% of DRAM peak).
__STATIC_INLINE__ bool sd_lora_acc_base_enabled() {
    static const bool enabled = [] {
        const char* e = getenv("SD_LORA_ACC_BASE");
        return e != nullptr && e[0] != '\0' && std::atoi(e) != 0;
    }();
    return enabled;
}

__STATIC_INLINE__ const std::vector<std::string>& sd_lora_acc_base_skip() {
    static const std::vector<std::string> pats = [] {
        std::vector<std::string> out;
        const char* e = getenv("SD_LORA_ACC_BASE_SKIP");
        if (e == nullptr) {
            return out;
        }
        std::string s(e), cur;
        for (char c : s) {
            if (c == ',') {
                if (!cur.empty()) {
                    out.push_back(cur);
                }
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) {
            out.push_back(cur);
        }
        return out;
    }();
    return pats;
}

struct LoraModel : public GGMLRunner {
    std::string lora_id;
    float multiplier = 1.0f;
    std::unordered_map<std::string, ggml_tensor*> lora_tensors;
    std::map<ggml_tensor*, ggml_tensor*> original_tensor_to_final_tensor;
    std::set<std::string> applied_lora_tensors;
    std::set<std::string> skipped_incompatible_lora_tensors;
    std::set<std::string> warned_incompatible_model_tensors;
    std::string file_path;
    std::shared_ptr<ModelManager> model_manager;
    ggml_backend_t params_backend = nullptr;
    bool load_failed              = false;
    bool applied                  = false;
    bool tensor_preprocessed      = false;

    typedef std::function<bool(const std::string&)> filter_t;

    LoraModel(const std::string& lora_id,
              ggml_backend_t backend,
              ggml_backend_t params_backend_,
              const std::string& file_path          = "",
              std::string prefix                    = "",
              SDVersion version                     = VERSION_COUNT,
              std::shared_ptr<ModelManager> manager = std::make_shared<ModelManager>())
        : GGMLRunner(backend, manager), lora_id(lora_id), file_path(file_path), model_manager(std::move(manager)), params_backend(params_backend_) {
        prefix = "lora." + prefix;
        if (model_manager == nullptr || !model_manager->loader().init_from_file_and_convert_name(file_path, prefix, version)) {
            load_failed = true;
        }
    }

    std::string get_desc() override {
        return "lora";
    }

    bool load_from_file(int n_threads, filter_t filter = nullptr) {
        LOG_INFO("loading LoRA from '%s'", file_path.c_str());

        if (load_failed) {
            LOG_ERROR("init lora model loader from file failed: '%s'", file_path.c_str());
            return false;
        }

        std::unordered_map<std::string, TensorStorage> tensors_to_create;
        std::mutex lora_mutex;
        bool dry_run          = true;
        auto on_new_tensor_cb = [&](const TensorStorage& tensor_storage, ggml_tensor** dst_tensor) -> bool {
            if (dry_run) {
                const std::string& name = tensor_storage.name;

                if (filter && !filter(name)) {
                    return true;
                }

                {
                    std::lock_guard<std::mutex> lock(lora_mutex);
                    tensors_to_create[name] = tensor_storage;
                }
            } else {
                const std::string& name = tensor_storage.name;
                auto iter               = lora_tensors.find(name);
                if (iter != lora_tensors.end()) {
                    *dst_tensor = iter->second;
                }
            }
            return true;
        };

        if (model_manager != nullptr) {
            model_manager->set_n_threads(n_threads);
        }
        ModelLoader& model_loader = model_manager->loader();
        model_loader.load_tensors(on_new_tensor_cb);

        if (tensors_to_create.empty()) {
            return true;
        }

        for (const auto& pair : tensors_to_create) {
            const auto& name   = pair.first;
            const auto& ts     = pair.second;
            ggml_tensor* real  = ggml_new_tensor(params_ctx,
                                                 ts.type,
                                                 ts.n_dims,
                                                 ts.ne);
            lora_tensors[name] = real;
        }

        std::map<std::string, ggml_tensor*> tensors;
        for (const auto& pair : lora_tensors) {
            tensors[pair.first] = pair.second;
        }
        if (model_manager == nullptr ||
            !model_manager->register_param_tensors("LoRA",
                                                   std::move(tensors),
                                                   ModelManager::ResidencyMode::ParamBackend,
                                                   runtime_backend,
                                                   params_backend) ||
            !model_manager->validate_registered_tensors()) {
            LOG_ERROR("lora model manager registration failed");
            return false;
        }
        std::vector<ggml_tensor*> lora_params;
        lora_params.reserve(lora_tensors.size());
        for (const auto& pair : lora_tensors) {
            lora_params.push_back(pair.second);
        }
        if (!model_manager->prepare_params(lora_params)) {
            LOG_ERROR("lora model manager prepare params failed");
            return false;
        }

        LOG_DEBUG("finished loaded lora");
        return true;
    }

    void release_loaded_tensors() {
        runner_done();
        free_compute_buffer();
        model_manager.reset();
        free_params_ctx();
        alloc_params_ctx();
        model_manager  = std::make_shared<ModelManager>();
        weight_manager = model_manager;
        lora_tensors.clear();
        original_tensor_to_final_tensor.clear();
        applied_lora_tensors.clear();
        skipped_incompatible_lora_tensors.clear();
        warned_incompatible_model_tensors.clear();
        applied             = false;
        tensor_preprocessed = false;
    }

    static std::set<std::string> tensor_names(const std::map<std::string, ggml_tensor*>& model_tensors) {
        std::set<std::string> names;
        for (const auto& item : model_tensors) {
            names.insert(item.first);
        }
        return names;
    }

    void preprocess_lora_tensors(const std::set<std::string>& model_tensor_names) {
        if (tensor_preprocessed) {
            return;
        }
        tensor_preprocessed = true;
        // I really hate these hardcoded processes.
        if (model_tensor_names.find("cond_stage_model.1.transformer.text_model.encoder.layers.0.self_attn.in_proj.weight") != model_tensor_names.end()) {
            std::unordered_map<std::string, ggml_tensor*> new_lora_tensors;
            for (auto& [old_name, tensor] : lora_tensors) {
                std::string new_name = old_name;

                if (contains(new_name, "cond_stage_model.1.transformer.text_model.encoder.layers")) {
                    std::vector<std::pair<std::string, std::string>> qkv_name_map = {
                        {"self_attn.q_proj.weight", "self_attn.in_proj.weight"},
                        {"self_attn.q_proj.bias", "self_attn.in_proj.bias"},
                        {"self_attn.k_proj.weight", "self_attn.in_proj.weight.1"},
                        {"self_attn.k_proj.bias", "self_attn.in_proj.bias.1"},
                        {"self_attn.v_proj.weight", "self_attn.in_proj.weight.2"},
                        {"self_attn.v_proj.bias", "self_attn.in_proj.bias.2"},
                    };
                    for (auto kv : qkv_name_map) {
                        size_t pos = new_name.find(kv.first);
                        if (pos != std::string::npos) {
                            new_name.replace(pos, kv.first.size(), kv.second);
                        }
                    }
                }

                new_lora_tensors[new_name] = tensor;
            }

            lora_tensors = std::move(new_lora_tensors);
        }
    }

    ggml_tensor* get_lora_weight_diff(const std::string& model_tensor_name, ggml_context* ctx, ggml_backend_t backend) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }

            std::string lora_down_name = "lora." + key + ".lora_down";
            std::string lora_up_name   = "lora." + key + ".lora_up";
            std::string lora_mid_name  = "lora." + key + ".lora_mid";
            std::string scale_name     = "lora." + key + ".scale";
            std::string alpha_name     = "lora." + key + ".alpha";

            ggml_tensor* lora_up   = nullptr;
            ggml_tensor* lora_mid  = nullptr;
            ggml_tensor* lora_down = nullptr;

            auto iter = lora_tensors.find(lora_up_name);
            if (iter != lora_tensors.end()) {
                lora_up = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            iter = lora_tensors.find(lora_mid_name);
            if (iter != lora_tensors.end()) {
                lora_mid = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            iter = lora_tensors.find(lora_down_name);
            if (iter != lora_tensors.end()) {
                lora_down = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            if (lora_up == nullptr || lora_down == nullptr) {
                break;
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);

            if (lora_mid) {
                applied_lora_tensors.insert(lora_mid_name);
            }

            float scale_value = 1.0f;

            int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
            iter         = lora_tensors.find(scale_name);
            if (iter != lora_tensors.end()) {
                scale_value = ggml_ext_backend_tensor_get_f32(iter->second);
                applied_lora_tensors.insert(scale_name);
            } else {
                iter = lora_tensors.find(alpha_name);
                if (iter != lora_tensors.end()) {
                    float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                    scale_value = alpha / rank;
                    // LOG_DEBUG("rank %s %ld %.2f %.2f", alpha_name.c_str(), rank, alpha, scale_value);
                    applied_lora_tensors.insert(alpha_name);
                }
            }
            scale_value *= multiplier;

            auto curr_updown = ggml_ext_merge_lora(ctx, lora_down, lora_up, lora_mid);
            curr_updown      = ggml_ext_scale(ctx, curr_updown, scale_value, true);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }

            index++;
        }
        return updown;
    }

    ggml_tensor* get_raw_weight_diff(const std::string& model_tensor_name, ggml_context* ctx, ggml_backend_t backend) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }

            std::string diff_name = "lora." + key + ".diff";

            ggml_tensor* curr_updown = nullptr;

            auto iter = lora_tensors.find(diff_name);
            if (iter != lora_tensors.end()) {
                curr_updown = ggml_ext_cast_f32(ctx, backend, iter->second);
            } else {
                break;
            }

            applied_lora_tensors.insert(diff_name);

            float scale_value = 1.0f;
            scale_value *= multiplier;

            curr_updown = ggml_ext_scale(ctx, curr_updown, scale_value, true);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }

            index++;
        }
        return updown;
    }

    ggml_tensor* get_loha_weight_diff(const std::string& model_tensor_name, ggml_context* ctx, ggml_backend_t backend) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }
            std::string hada_1_down_name = "lora." + key + ".hada_w1_b";
            std::string hada_1_mid_name  = "lora." + key + ".hada_t1";
            std::string hada_1_up_name   = "lora." + key + ".hada_w1_a";
            std::string hada_2_down_name = "lora." + key + ".hada_w2_b";
            std::string hada_2_mid_name  = "lora." + key + ".hada_t2";
            std::string hada_2_up_name   = "lora." + key + ".hada_w2_a";
            std::string alpha_name       = "lora." + key + ".alpha";

            ggml_tensor* hada_1_mid  = nullptr;  // tau for tucker decomposition
            ggml_tensor* hada_1_up   = nullptr;
            ggml_tensor* hada_1_down = nullptr;

            ggml_tensor* hada_2_mid  = nullptr;  // tau for tucker decomposition
            ggml_tensor* hada_2_up   = nullptr;
            ggml_tensor* hada_2_down = nullptr;

            auto iter = lora_tensors.find(hada_1_down_name);
            if (iter != lora_tensors.end()) {
                hada_1_down = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            iter = lora_tensors.find(hada_1_up_name);
            if (iter != lora_tensors.end()) {
                hada_1_up = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            iter = lora_tensors.find(hada_1_mid_name);
            if (iter != lora_tensors.end()) {
                hada_1_mid = ggml_ext_cast_f32(ctx, backend, iter->second);
                if (hada_1_up != nullptr) {
                    hada_1_up = ggml_cont(ctx, ggml_transpose(ctx, hada_1_up));
                }
            }

            iter = lora_tensors.find(hada_2_down_name);
            if (iter != lora_tensors.end()) {
                hada_2_down = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            iter = lora_tensors.find(hada_2_up_name);
            if (iter != lora_tensors.end()) {
                hada_2_up = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            iter = lora_tensors.find(hada_2_mid_name);
            if (iter != lora_tensors.end()) {
                hada_2_mid = ggml_ext_cast_f32(ctx, backend, iter->second);
                if (hada_2_up != nullptr) {
                    hada_2_up = ggml_cont(ctx, ggml_transpose(ctx, hada_2_up));
                }
            }

            if (hada_1_up == nullptr || hada_1_down == nullptr || hada_2_up == nullptr || hada_2_down == nullptr) {
                break;
            }

            applied_lora_tensors.insert(hada_1_down_name);
            applied_lora_tensors.insert(hada_1_up_name);
            applied_lora_tensors.insert(hada_2_down_name);
            applied_lora_tensors.insert(hada_2_up_name);
            applied_lora_tensors.insert(alpha_name);

            if (hada_1_mid) {
                applied_lora_tensors.insert(hada_1_mid_name);
            }

            if (hada_2_mid) {
                applied_lora_tensors.insert(hada_2_mid_name);
            }

            float scale_value = 1.0f;

            // calc_scale
            // TODO: .dora_scale?
            int64_t rank = hada_1_down->ne[ggml_n_dims(hada_1_down) - 1];
            iter         = lora_tensors.find(alpha_name);
            if (iter != lora_tensors.end()) {
                float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                scale_value = alpha / rank;
                applied_lora_tensors.insert(alpha_name);
            }
            scale_value *= multiplier;

            ggml_tensor* updown_1 = ggml_ext_merge_lora(ctx, hada_1_down, hada_1_up, hada_1_mid);
            ggml_tensor* updown_2 = ggml_ext_merge_lora(ctx, hada_2_down, hada_2_up, hada_2_mid);
            auto curr_updown      = ggml_mul_inplace(ctx, updown_1, updown_2);
            curr_updown           = ggml_ext_scale(ctx, curr_updown, scale_value, true);
            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }
            index++;
        }
        return updown;
    }

    ggml_tensor* get_lokr_weight_diff(const std::string& model_tensor_name, ggml_context* ctx, ggml_backend_t backend) {
        ggml_tensor* updown = nullptr;
        int index           = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }
            std::string lokr_w1_name   = "lora." + key + ".lokr_w1";
            std::string lokr_w1_a_name = "lora." + key + ".lokr_w1_a";
            std::string lokr_w1_b_name = "lora." + key + ".lokr_w1_b";
            std::string lokr_w2_name   = "lora." + key + ".lokr_w2";
            std::string lokr_w2_a_name = "lora." + key + ".lokr_w2_a";
            std::string lokr_w2_b_name = "lora." + key + ".lokr_w2_b";
            std::string alpha_name     = "lora." + key + ".alpha";

            ggml_tensor* lokr_w1   = nullptr;
            ggml_tensor* lokr_w1_a = nullptr;
            ggml_tensor* lokr_w1_b = nullptr;
            ggml_tensor* lokr_w2   = nullptr;
            ggml_tensor* lokr_w2_a = nullptr;
            ggml_tensor* lokr_w2_b = nullptr;

            auto iter = lora_tensors.find(lokr_w1_name);
            if (iter != lora_tensors.end()) {
                lokr_w1 = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            iter = lora_tensors.find(lokr_w2_name);
            if (iter != lora_tensors.end()) {
                lokr_w2 = ggml_ext_cast_f32(ctx, backend, iter->second);
            }

            int64_t rank = 1;
            if (lokr_w1 == nullptr) {
                iter = lora_tensors.find(lokr_w1_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_a = ggml_ext_cast_f32(ctx, backend, iter->second);
                }

                iter = lora_tensors.find(lokr_w1_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_b = ggml_ext_cast_f32(ctx, backend, iter->second);
                }

                if (lokr_w1_a == nullptr || lokr_w1_b == nullptr) {
                    break;
                }

                rank = lokr_w1_b->ne[ggml_n_dims(lokr_w1_b) - 1];

                lokr_w1 = ggml_ext_merge_lora(ctx, lokr_w1_b, lokr_w1_a);
            }

            if (lokr_w2 == nullptr) {
                iter = lora_tensors.find(lokr_w2_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_a = ggml_ext_cast_f32(ctx, backend, iter->second);
                }

                iter = lora_tensors.find(lokr_w2_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_b = ggml_ext_cast_f32(ctx, backend, iter->second);
                }

                if (lokr_w2_a == nullptr || lokr_w2_b == nullptr) {
                    break;
                }

                rank = lokr_w2_b->ne[ggml_n_dims(lokr_w2_b) - 1];

                lokr_w2 = ggml_ext_merge_lora(ctx, lokr_w2_b, lokr_w2_a);
            }

            if (!lokr_w1_a) {
                applied_lora_tensors.insert(lokr_w1_name);
            } else {
                applied_lora_tensors.insert(lokr_w1_a_name);
                applied_lora_tensors.insert(lokr_w1_b_name);
            }

            if (!lokr_w2_a) {
                applied_lora_tensors.insert(lokr_w2_name);
            } else {
                applied_lora_tensors.insert(lokr_w2_a_name);
                applied_lora_tensors.insert(lokr_w2_b_name);
            }

            float scale_value = 1.0f;
            iter              = lora_tensors.find(alpha_name);
            if (iter != lora_tensors.end()) {
                float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                scale_value = alpha / rank;
                applied_lora_tensors.insert(alpha_name);
            }

            if (rank == 1) {
                scale_value = 1.0f;
            }

            scale_value *= multiplier;

            auto curr_updown = ggml_ext_kronecker(ctx, lokr_w1, lokr_w2);
            curr_updown      = ggml_ext_scale(ctx, curr_updown, scale_value, true);

            if (updown == nullptr) {
                updown = curr_updown;
            } else {
                updown = ggml_concat(ctx, updown, curr_updown, ggml_n_dims(updown) - 1);
            }
            index++;
        }
        return updown;
    }

    ggml_tensor* get_weight_diff(const std::string& model_tensor_name, ggml_backend_t backend, ggml_context* ctx, ggml_tensor* model_tensor, bool with_lora_and_lokr = true) {
        // lora
        ggml_tensor* diff = nullptr;
        if (with_lora_and_lokr) {
            diff = get_lora_weight_diff(model_tensor_name, ctx, backend);
        }
        // diff
        if (diff == nullptr) {
            diff = get_raw_weight_diff(model_tensor_name, ctx, backend);
        }
        // loha
        if (diff == nullptr) {
            diff = get_loha_weight_diff(model_tensor_name, ctx, backend);
        }
        // lokr
        if (diff == nullptr && with_lora_and_lokr) {
            diff = get_lokr_weight_diff(model_tensor_name, ctx, backend);
        }
        if (diff != nullptr) {
            if (ggml_nelements(diff) < ggml_nelements(model_tensor)) {
                if (ggml_n_dims(diff) == 2 && ggml_n_dims(model_tensor) == 2 && diff->ne[0] == model_tensor->ne[0]) {
                    LOG_WARN("pad for %s", model_tensor_name.c_str());
                    auto pad_tensor = ggml_ext_zeros(ctx, diff->ne[0], model_tensor->ne[1] - diff->ne[1], 1, 1);
                    diff            = ggml_concat(ctx, diff, pad_tensor, 1);
                }
            }

            if (ggml_nelements(diff) != ggml_nelements(model_tensor)) {
                const std::string lora_tensor_prefix = "lora." + model_tensor_name + ".";
                for (const auto& tensor_name : applied_lora_tensors) {
                    if (starts_with(tensor_name, lora_tensor_prefix)) {
                        skipped_incompatible_lora_tensors.insert(tensor_name);
                    }
                }
                if (warned_incompatible_model_tensors.insert(model_tensor_name).second) {
                    LOG_WARN("skip incompatible LoRA tensor |%s|: model shape = [%lld, %lld, %lld, %lld], LoRA shape = [%lld, %lld, %lld, %lld]",
                             model_tensor_name.c_str(),
                             static_cast<long long>(model_tensor->ne[0]),
                             static_cast<long long>(model_tensor->ne[1]),
                             static_cast<long long>(model_tensor->ne[2]),
                             static_cast<long long>(model_tensor->ne[3]),
                             static_cast<long long>(diff->ne[0]),
                             static_cast<long long>(diff->ne[1]),
                             static_cast<long long>(diff->ne[2]),
                             static_cast<long long>(diff->ne[3]));
                }
                return nullptr;
            }
            diff = ggml_reshape(ctx, diff, model_tensor);
        }
        return diff;
    }

    ggml_tensor* get_out_diff(ggml_context* ctx,
                              ggml_backend_t backend,
                              ggml_tensor* x,
                              ggml_tensor* model_weight,
                              WeightAdapter::ForwardParams forward_params,
                              const std::string& model_tensor_name) {
        ggml_tensor* out_diff = nullptr;
        int index             = 0;
        while (true) {
            std::string key;
            if (index == 0) {
                key = model_tensor_name;
            } else {
                key = model_tensor_name + "." + std::to_string(index);
            }
            bool is_conv2d = forward_params.op_type == WeightAdapter::ForwardParams::op_type_t::OP_CONV2D;

            std::string lokr_w1_name   = "lora." + key + ".lokr_w1";
            std::string lokr_w1_a_name = "lora." + key + ".lokr_w1_a";
            // if either of these is found, then we have a lokr lora
            auto iter   = lora_tensors.find(lokr_w1_name);
            auto iter_a = lora_tensors.find(lokr_w1_a_name);
            if (iter != lora_tensors.end() || iter_a != lora_tensors.end()) {
                std::string lokr_w1_b_name = "lora." + key + ".lokr_w1_b";
                std::string lokr_w2_name   = "lora." + key + ".lokr_w2";
                std::string lokr_w2_a_name = "lora." + key + ".lokr_w2_a";
                std::string lokr_w2_b_name = "lora." + key + ".lokr_w2_b";
                std::string alpha_name     = "lora." + key + ".alpha";

                ggml_tensor* lokr_w1   = nullptr;
                ggml_tensor* lokr_w1_a = nullptr;
                ggml_tensor* lokr_w1_b = nullptr;
                ggml_tensor* lokr_w2   = nullptr;
                ggml_tensor* lokr_w2_a = nullptr;
                ggml_tensor* lokr_w2_b = nullptr;

                if (iter != lora_tensors.end()) {
                    lokr_w1 = iter->second;
                }
                iter = iter_a;
                if (iter != lora_tensors.end()) {
                    lokr_w1_a = iter->second;
                }
                iter = lora_tensors.find(lokr_w1_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w1_b = iter->second;
                }

                iter = lora_tensors.find(lokr_w2_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2 = iter->second;
                    if (is_conv2d && lokr_w2->type != GGML_TYPE_F16) {
                        lokr_w2 = ggml_cast(ctx, lokr_w2, GGML_TYPE_F16);
                    }
                }
                iter = lora_tensors.find(lokr_w2_a_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_a = iter->second;
                    if (is_conv2d && lokr_w2_a->type != GGML_TYPE_F16) {
                        lokr_w2_a = ggml_cast(ctx, lokr_w2_a, GGML_TYPE_F16);
                    }
                }
                iter = lora_tensors.find(lokr_w2_b_name);
                if (iter != lora_tensors.end()) {
                    lokr_w2_b = iter->second;
                    if (is_conv2d && lokr_w2_b->type != GGML_TYPE_F16) {
                        lokr_w2_b = ggml_cast(ctx, lokr_w2_b, GGML_TYPE_F16);
                    }
                }

                int rank = 1;
                if (lokr_w1_b) {
                    rank = (int)lokr_w1_b->ne[ggml_n_dims(lokr_w1_b) - 1];
                }
                if (lokr_w2_b) {
                    rank = (int)lokr_w2_b->ne[ggml_n_dims(lokr_w2_b) - 1];
                }

                float scale_value = 1.0f;
                iter              = lora_tensors.find(alpha_name);
                if (iter != lora_tensors.end()) {
                    float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                    scale_value = alpha / rank;
                    applied_lora_tensors.insert(alpha_name);
                }

                if (rank == 1) {
                    scale_value = 1.0f;
                }
                scale_value *= multiplier;

                auto curr_out_diff = ggml_ext_lokr_forward(ctx, backend, x, lokr_w1, lokr_w1_a, lokr_w1_b, lokr_w2, lokr_w2_a, lokr_w2_b, is_conv2d, forward_params.conv2d, scale_value);
                if (out_diff == nullptr) {
                    out_diff = curr_out_diff;
                } else {
                    out_diff = ggml_concat(ctx, out_diff, curr_out_diff, 0);
                }

                if (lokr_w1)
                    applied_lora_tensors.insert(lokr_w1_name);
                if (lokr_w1_a)
                    applied_lora_tensors.insert(lokr_w1_a_name);
                if (lokr_w1_b)
                    applied_lora_tensors.insert(lokr_w1_b_name);
                if (lokr_w2)
                    applied_lora_tensors.insert(lokr_w2_name);
                if (lokr_w2_a)
                    applied_lora_tensors.insert(lokr_w2_a_name);
                if (lokr_w2_b)
                    applied_lora_tensors.insert(lokr_w2_b_name);
                applied_lora_tensors.insert(alpha_name);

                index++;
                continue;
            }

            // not a lokr, normal lora path

            std::string lora_down_name = "lora." + key + ".lora_down";
            std::string lora_up_name   = "lora." + key + ".lora_up";
            std::string lora_mid_name  = "lora." + key + ".lora_mid";
            std::string scale_name     = "lora." + key + ".scale";
            std::string alpha_name     = "lora." + key + ".alpha";

            ggml_tensor* lora_up   = nullptr;
            ggml_tensor* lora_mid  = nullptr;
            ggml_tensor* lora_down = nullptr;

            iter = lora_tensors.find(lora_up_name);
            if (iter != lora_tensors.end()) {
                lora_up = iter->second;
                if (is_conv2d && lora_up->type != GGML_TYPE_F16) {
                    lora_up = ggml_cast(ctx, lora_up, GGML_TYPE_F16);
                }
            }

            iter = lora_tensors.find(lora_mid_name);
            if (iter != lora_tensors.end()) {
                lora_mid = iter->second;
                if (is_conv2d && lora_mid->type != GGML_TYPE_F16) {
                    lora_mid = ggml_cast(ctx, lora_mid, GGML_TYPE_F16);
                }
            }

            iter = lora_tensors.find(lora_down_name);
            if (iter != lora_tensors.end()) {
                lora_down = iter->second;
                if (is_conv2d && lora_down->type != GGML_TYPE_F16) {
                    lora_down = ggml_cast(ctx, lora_down, GGML_TYPE_F16);
                }
            }

            if (lora_up == nullptr || lora_down == nullptr) {
                break;
            }

            if (!is_conv2d) {
                const int64_t down_in  = lora_down->ne[0];
                const int64_t down_out = lora_down->ne[1];
                const int64_t up_in    = lora_up->ne[0];
                const int64_t up_out   = lora_up->ne[1];

                bool compatible = down_in == model_weight->ne[0] &&
                                  up_out == model_weight->ne[1];
                if (lora_mid != nullptr) {
                    compatible = compatible &&
                                 lora_mid->ne[0] == down_out &&
                                 up_in == lora_mid->ne[1];
                } else {
                    compatible = compatible && up_in == down_out;
                }

                if (!compatible) {
                    skipped_incompatible_lora_tensors.insert(lora_down_name);
                    skipped_incompatible_lora_tensors.insert(lora_up_name);
                    skipped_incompatible_lora_tensors.insert(lora_mid_name);
                    skipped_incompatible_lora_tensors.insert(scale_name);
                    skipped_incompatible_lora_tensors.insert(alpha_name);
                    if (warned_incompatible_model_tensors.insert(model_tensor_name).second) {
                        LOG_WARN("skip incompatible LoRA tensor |%s|: model shape = [%lld, %lld], down shape = [%lld, %lld], up shape = [%lld, %lld]",
                                 model_tensor_name.c_str(),
                                 static_cast<long long>(model_weight->ne[0]),
                                 static_cast<long long>(model_weight->ne[1]),
                                 static_cast<long long>(down_in),
                                 static_cast<long long>(down_out),
                                 static_cast<long long>(up_in),
                                 static_cast<long long>(up_out));
                    }
                    index++;
                    continue;
                }
            }

            applied_lora_tensors.insert(lora_up_name);
            applied_lora_tensors.insert(lora_down_name);

            if (lora_mid) {
                applied_lora_tensors.insert(lora_mid_name);
            }

            float scale_value = 1.0f;

            int64_t rank = lora_down->ne[ggml_n_dims(lora_down) - 1];
            iter         = lora_tensors.find(scale_name);
            if (iter != lora_tensors.end()) {
                scale_value = ggml_ext_backend_tensor_get_f32(iter->second);
                applied_lora_tensors.insert(scale_name);
            } else {
                iter = lora_tensors.find(alpha_name);
                if (iter != lora_tensors.end()) {
                    float alpha = ggml_ext_backend_tensor_get_f32(iter->second);
                    scale_value = alpha / rank;
                    // LOG_DEBUG("rank %s %ld %.2f %.2f", alpha_name.c_str(), rank, alpha, scale_value);
                    applied_lora_tensors.insert(alpha_name);
                }
            }
            scale_value *= multiplier;

            ggml_tensor* lx;
            // Whether the LoRA strength has already been folded into the RANK-WIDE intermediate
            // below, so the full-width scale at the end can be skipped.
            bool scale_folded = false;
            if (!is_conv2d) {
                lx = ggml_ext_linear(ctx, x, lora_down, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
                if (lora_mid) {
                    lx = ggml_ext_linear(ctx, lx, lora_mid, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
                }
                // Apply the strength HERE, on the [tokens x rank] intermediate, instead of on the
                // [tokens x out] result after lora_up: rank is 32..128 against out 4096..16384,
                // so this touches ~1.6% of the elements for the same intended maths.
                //
                // MEASURED WIN (nsys, 1920x1088/145f, 2 steps, audio-reactive, rebuilt binary):
                // scale_f32 +4295 ms -> +46 ms, cutting the adapter's total GPU overhead from
                // +15841 ms to +11530 ms (-27%), every other kernel unchanged. The full-width
                // scale had been the 2nd-largest line in the profile, above the LoRA's own GEMMs.
                //
                // NUMERICS: **UNVALIDATED**. It commutes in exact arithmetic, but lora_up is Q8_0
                // so this GEMM takes ggml's MMQ route and quantises the ACTIVATION to q8_1 first;
                // scaling before that quantisation need not be bit-identical to scaling after.
                // An A/B at 768x448/25f measured mean |dLuma| 2.75/255 against the unpatched
                // build -- but the SAME binary run twice at that shape differs by 2.36, so that
                // comparison sits in the noise and proves nothing either way. The LoRA path is
                // NON-DETERMINISTIC at small shapes; 1920x1088/145f server renders of one seed
                // were previously bit-identical, so that is the shape to validate on, and the CLI
                // cannot reach it (VAE decode OOMs without the server's tiling). Needs a server
                // build. Do not claim neutrality — in either direction — until that A/B is run.
                //
                // An adapter with no `.alpha` and no `.scale` tensor, at multiplier 1.0, has
                // scale_value == 1.0 exactly, and this SCALE is then a pure identity: a
                // [rank x tokens] read plus a [rank x tokens] write, per module per step, for
                // nothing. krea2's rank-256 256-module adapter spends ~3.8 GB/step and 224 kernel
                // launches on it. Skipping the emission is a graph change only -- the surviving
                // arithmetic is untouched, so the non-unit case still gets the node in the SAME
                // rank-wide position as before. `lx` comes straight out of a matmul, so the
                // ggml_cont() inside ggml_ext_scale() was never doing anything here either.
                if (scale_value != 1.0f) {
                    lx = ggml_ext_scale(ctx, lx, scale_value, true);
                }
                scale_folded = true;
                lx = ggml_ext_linear(ctx, lx, lora_up, nullptr, forward_params.linear.force_prec_f32, forward_params.linear.scale);
            } else {  // OP_CONV2D
                lx = ggml_ext_conv_2d(ctx,
                                      x,
                                      lora_down,
                                      nullptr,
                                      forward_params.conv2d.s0,
                                      forward_params.conv2d.s1,
                                      forward_params.conv2d.p0,
                                      forward_params.conv2d.p1,
                                      forward_params.conv2d.d0,
                                      forward_params.conv2d.d1,
                                      forward_params.conv2d.direct,
                                      forward_params.conv2d.circular_x,
                                      forward_params.conv2d.circular_y,
                                      forward_params.conv2d.scale);
                if (lora_mid) {
                    lx = ggml_ext_conv_2d(ctx,
                                          lx,
                                          lora_mid,
                                          nullptr,
                                          1,
                                          1,
                                          0,
                                          0,
                                          1,
                                          1,
                                          forward_params.conv2d.direct,
                                          forward_params.conv2d.circular_x,
                                          forward_params.conv2d.circular_y,
                                          forward_params.conv2d.scale);
                }
                lx = ggml_ext_conv_2d(ctx,
                                      lx,
                                      lora_up,
                                      nullptr,
                                      1,
                                      1,
                                      0,
                                      0,
                                      1,
                                      1,
                                      forward_params.conv2d.direct,
                                      forward_params.conv2d.circular_x,
                                      forward_params.conv2d.circular_y,
                                      forward_params.conv2d.scale);
            }

            // Conv2d still scales at full width (its intermediate is not narrower, so there is
            // nothing to win); the linear path already folded it into the rank-wide tensor.
            auto curr_out_diff = scale_folded ? lx : ggml_ext_scale(ctx, lx, scale_value, true);

            if (out_diff == nullptr) {
                out_diff = curr_out_diff;
            } else {
                out_diff = ggml_concat(ctx, out_diff, curr_out_diff, 0);
            }

            index++;
        }
        return out_diff;
    }

    ggml_cgraph* build_lora_graph(const std::map<std::string, ggml_tensor*>& model_tensors,
                                  const std::set<std::string>& model_tensor_names,
                                  SDVersion version) {
        size_t lora_graph_size = LORA_GRAPH_BASE_SIZE + lora_tensors.size() * 10;
        ggml_cgraph* gf        = ggml_new_graph_custom(compute_ctx, lora_graph_size, false);

        preprocess_lora_tensors(model_tensor_names);

        original_tensor_to_final_tensor.clear();
        applied_lora_tensors.clear();

        for (auto it : model_tensors) {
            std::string model_tensor_name = it.first;
            ggml_tensor* model_tensor     = it.second;

            // lora
            ggml_tensor* diff = get_weight_diff(model_tensor_name, runtime_backend, compute_ctx, model_tensor);
            if (diff == nullptr) {
                continue;
            }

            ggml_tensor* original_tensor = model_tensor;
            if (!sd_backend_is_cpu(runtime_backend) && ggml_backend_buffer_is_host(original_tensor->buffer)) {
                model_tensor = ggml_dup_tensor(compute_ctx, model_tensor);
                set_backend_tensor_data(model_tensor, original_tensor->data);
            }

            ggml_tensor* final_tensor;
            if (model_tensor->type != GGML_TYPE_F32 && model_tensor->type != GGML_TYPE_F16) {
                final_tensor = ggml_ext_cast_f32(compute_ctx, runtime_backend, model_tensor);
                final_tensor = ggml_add_inplace(compute_ctx, final_tensor, diff);
                final_tensor = ggml_cpy(compute_ctx, final_tensor, model_tensor);
            } else {
                final_tensor = ggml_add_inplace(compute_ctx, model_tensor, diff);
            }
            ggml_build_forward_expand(gf, final_tensor);
            if (!sd_backend_is_cpu(runtime_backend) && ggml_backend_buffer_is_host(original_tensor->buffer)) {
                original_tensor_to_final_tensor[original_tensor] = final_tensor;
            }
        }
        return gf;
    }

    void apply(std::map<std::string, ggml_tensor*> model_tensors,
               const std::set<std::string>& model_tensor_names,
               SDVersion version,
               int n_threads,
               bool warn_unused = true) {
        auto get_graph = [&]() -> ggml_cgraph* {
            return build_lora_graph(model_tensors, model_tensor_names, version);
        };
        GGMLRunner::compute<float>(get_graph, n_threads, false, false, false, true);
        stat(!warn_unused);
        for (auto item : original_tensor_to_final_tensor) {
            ggml_tensor* original_tensor = item.first;
            ggml_tensor* final_tensor    = item.second;

            ggml_backend_tensor_copy(final_tensor, original_tensor);
        }
        original_tensor_to_final_tensor.clear();
        GGMLRunner::free_compute_buffer();
    }

    void apply(std::map<std::string, ggml_tensor*> model_tensors, SDVersion version, int n_threads, bool warn_unused = true) {
        apply(model_tensors, tensor_names(model_tensors), version, n_threads, warn_unused);
    }

    void stat(bool at_runntime = false) {
        size_t total_lora_tensors_count   = 0;
        size_t applied_lora_tensors_count = 0;
        size_t skipped_lora_tensors_count = 0;

        for (auto& kv : lora_tensors) {
            total_lora_tensors_count++;
            if (skipped_incompatible_lora_tensors.find(kv.first) != skipped_incompatible_lora_tensors.end()) {
                skipped_lora_tensors_count++;
            } else if (applied_lora_tensors.find(kv.first) == applied_lora_tensors.end()) {
                if (!at_runntime) {
                    LOG_WARN("unused lora tensor |%s|", kv.first.c_str());
                    print_ggml_tensor(kv.second, true);
                }
            } else {
                applied_lora_tensors_count++;
            }
        }
        /* Don't worry if this message shows up twice in the logs per LoRA,
         * this function is called once to calculate the required buffer size
         * and then again to actually generate a graph to be used */
        size_t compatible_lora_tensors_count = total_lora_tensors_count - skipped_lora_tensors_count;
        if (!at_runntime && applied_lora_tensors_count != compatible_lora_tensors_count) {
            LOG_WARN("Only (%lu / %lu) LoRA tensors have been applied, lora_file_path = %s",
                     applied_lora_tensors_count, compatible_lora_tensors_count, file_path.c_str());
        } else {
            LOG_INFO("(%lu / %lu) LoRA tensors have been applied, lora_file_path = %s",
                     applied_lora_tensors_count, compatible_lora_tensors_count, file_path.c_str());
        }
        if (skipped_lora_tensors_count > 0) {
            LOG_WARN("(%lu / %lu) incompatible LoRA tensors have been skipped, lora_file_path = %s",
                     skipped_lora_tensors_count, total_lora_tensors_count, file_path.c_str());
        }
    }
};

struct MultiLoraAdapter : public WeightAdapter {
protected:
    std::vector<std::shared_ptr<LoraModel>> lora_models;

public:
    explicit MultiLoraAdapter(const std::vector<std::shared_ptr<LoraModel>>& lora_models)
        : lora_models(lora_models) {
    }

    // Every tensor this adapter owns must be an NVFP4 matrix whose contraction dimension is a
    // whole number of 64-element FP4 blocks. That is exactly the precondition
    // ggml_cuda_nvfp4_cublaslt_shapes_ok() enforces (`a->ne[0] % 64 != 0` bails), and it is what
    // makes ggml_backend_cuda_device_supports_op() accept an F16 src1 against this src0.
    //
    // For a LoRA, ne[0] is the CONTRACTION dim of each factor: lora_down is [in_features, rank]
    // so ne[0] = in_features, lora_up is [rank, out_features] so ne[0] = RANK. Rank 256 and 64
    // pass; rank 32 can never pass, which is why ltx-video's adapters cannot take this route.
    //
    // Deliberately strict: a stray F32 `.alpha`/`.scale` scalar makes the whole adapter decline.
    // Those are read host-side at graph-build time and are never GEMM operands, so this is
    // stricter than strictly necessary — but the failure mode of a wrong `true` is nodes on the
    // CPU, and the cost of a wrong `false` is only that the F32 stream runs. Strict wins.
    bool supports_f16_activation() const override {
        bool saw_any = false;
        for (const auto& lora_model : lora_models) {
            for (const auto& kv : lora_model->lora_tensors) {
                const ggml_tensor* t = kv.second;
                if (t == nullptr) {
                    return false;
                }
                if (t->type != GGML_TYPE_NVFP4 || t->ne[0] % 64 != 0) {
                    return false;
                }
                saw_any = true;
            }
        }
        // An adapter that has not loaded its tensors yet must not be waved through.
        return saw_any;
    }

    // See WeightAdapter::cache_identity(). Identifies WHICH adapters are attached and with what
    // strength, for callers caching an adapter-dependent subgraph across renders.
    //
    // Keyed on WHAT the adapter is, not WHERE its bytes happen to live this request:
    //   * file_path + lora_id     which adapter, and under which request-level name
    //   * multiplier (raw bits)   strength; 0.0 and 1.0 are genuinely different results
    //   * every tensor's name, type and ne[0..3], in SORTED name order because lora_tensors is
    //     an unordered_map whose iteration order is not stable across rehashes.
    //
    // ⚠️ `t->data` is deliberately NOT hashed, and this is load-bearing rather than an omission.
    // apply_loras_at_runtime CLEARS AND RELOADS runtime_lora_models on every img_gen, so the
    // tensor allocations are fresh each request: hashing the pointers made this identity change
    // every render, which turned every cache lookup into a miss and silently reverted the whole
    // optimisation (MEASURED: 3 text-branch computes across 3 identical renders instead of 1).
    // The identity must be stable for the same adapter file at the same strength, and it is the
    // file path + tensor-set shape that carry that. A file mutated on disk mid-process under an
    // unchanged path is the one case this cannot see; it is not a mode this service has.
    //
    // Cost is a few thousand short string hashes once per render, against a text branch worth
    // ~2.5 TFLOP.
    //
    // Returns 0 ("do not cache") if ANY attached model is still loading or failed, because an
    // adapter whose tensors are not resident yet would otherwise hash as a stable identity and
    // then change underneath the entry.
    uint64_t cache_identity() const override {
        uint64_t h    = 1469598103934665603ull;
        auto mix_bytes = [&h](const void* data, size_t bytes) {
            const auto* p = static_cast<const unsigned char*>(data);
            for (size_t i = 0; i < bytes; ++i) {
                h = (h ^ p[i]) * 1099511628211ull;
            }
        };
        auto mix_str = [&](const std::string& s) {
            mix_bytes(s.data(), s.size());
            const unsigned char sep = 0x1f;  // keep "ab"+"c" from colliding with "a"+"bc"
            mix_bytes(&sep, 1);
        };

        const uint64_t n_models = lora_models.size();
        mix_bytes(&n_models, sizeof(n_models));
        for (const auto& lora_model : lora_models) {
            if (lora_model == nullptr || lora_model->load_failed || lora_model->lora_tensors.empty()) {
                return 0;  // unidentifiable right now -> caller must not cache
            }
            mix_str(lora_model->file_path);
            mix_str(lora_model->lora_id);
            mix_bytes(&lora_model->multiplier, sizeof(lora_model->multiplier));

            std::vector<const std::string*> names;
            names.reserve(lora_model->lora_tensors.size());
            for (const auto& kv : lora_model->lora_tensors) {
                names.push_back(&kv.first);
            }
            std::sort(names.begin(), names.end(),
                      [](const std::string* a, const std::string* b) { return *a < *b; });
            const uint64_t n_tensors = names.size();
            mix_bytes(&n_tensors, sizeof(n_tensors));
            for (const std::string* name : names) {
                mix_str(*name);
                const ggml_tensor* t = lora_model->lora_tensors.at(*name);
                if (t == nullptr) {
                    return 0;
                }
                const int type = static_cast<int>(t->type);
                mix_bytes(&type, sizeof(type));
                for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                    mix_bytes(&t->ne[d], sizeof(t->ne[d]));
                }
            }
        }
        // 0 is the "do not cache" sentinel, so a real identity must never be 0.
        return h == 0 ? 1 : h;
    }

    ggml_tensor* patch_weight(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* weight, const std::string& weight_name, bool with_lora_and_lokr) {
        for (auto& lora_model : lora_models) {
            ggml_tensor* diff = lora_model->get_weight_diff(weight_name, backend, ctx, weight, with_lora_and_lokr);
            if (diff == nullptr) {
                continue;
            }

            if (weight->type != GGML_TYPE_F32 && weight->type != GGML_TYPE_F16) {
                weight = ggml_ext_cast_f32(ctx, backend, weight);
            }
            weight = ggml_add(ctx, weight, diff);
        }
        return weight;
    }

    ggml_tensor* patch_weight(ggml_context* ctx, ggml_backend_t backend, ggml_tensor* weight, const std::string& weight_name) override {
        return patch_weight(ctx, backend, weight, weight_name, true);
    }

    ggml_tensor* forward_with_lora(ggml_context* ctx,
                                   ggml_backend_t backend,
                                   ggml_tensor* x,
                                   ggml_tensor* w,
                                   ggml_tensor* b,
                                   const std::string& prefix,
                                   WeightAdapter::ForwardParams forward_params,
                                   ggml_tensor* base_output_scale = nullptr) override {
        w = patch_weight(ctx, backend, w, prefix + "weight", false);
        if (b) {
            b = patch_weight(ctx, backend, b, prefix + "bias", false);
        }
        ggml_tensor* out;
        if (forward_params.op_type == ForwardParams::op_type_t::OP_LINEAR) {
            out = ggml_ext_linear(ctx, x, w, b, forward_params.linear.force_prec_f32, forward_params.linear.scale);
        } else {  // OP_CONV2D
            out = ggml_ext_conv_2d(ctx,
                                   x,
                                   w,
                                   b,
                                   forward_params.conv2d.s0,
                                   forward_params.conv2d.s1,
                                   forward_params.conv2d.p0,
                                   forward_params.conv2d.p1,
                                   forward_params.conv2d.d0,
                                   forward_params.conv2d.d1,
                                   forward_params.conv2d.direct,
                                   forward_params.conv2d.circular_x,
                                   forward_params.conv2d.circular_y,
                                   forward_params.conv2d.scale);
        }
        // ModelOpt's `.wglobal` belongs to the quantized base weight, not to
        // any runtime LoRA delta. Apply it before the delta loop below.
        if (base_output_scale != nullptr) {
            out = ggml_mul(ctx, out, base_output_scale);
        }
        // ---- delta accumulation, and WHICH GEMM pays for it --------------------------------
        //
        // Both orderings compute out + delta and move exactly the same bytes; what differs is
        // which kernel does the read-modify-write, because ggml_cuda_try_fuse_mul_mat_acc()
        // folds an in-place ADD into whichever MUL_MAT is its immediate graph predecessor.
        //
        //   FORWARD (default)   add_inplace(out, delta)
        //       cgraph order: base_gemm, lora_down, lora_up, ADD.  The fusable MUL_MAT is
        //       lora_up, so lora_up runs beta=1 and reads `out` back.
        //
        //   REVERSED            add_inplace(delta, out)
        //       cgraph order: lora_down, lora_up, base_gemm, ADD.  The fusable MUL_MAT is the
        //       BASE gemm, so lora_up writes its own [O x L] delta at beta=0 and the base gemm
        //       reads it back at beta=1.
        //
        // (The order above is not a hope: ggml_build_forward_expand does a left-to-right DFS
        // over src[], so src[0]'s subtree is emitted first and the node itself last. Swapping
        // the ADD's operands is therefore the whole reordering -- the base Linear is still
        // *constructed* first, it is just *visited* second.)
        //
        // WHY REVERSED CAN WIN. ncu on the krea2 edit shape: the LoRA `up` GEMM is at 78-85%
        // of DRAM peak and 13-17% of tensor peak -- bandwidth-bound at the roofline, and half
        // its traffic is the beta=1 read-back. The base Linear it pairs with is compute-bound
        // (SM 84-87%, DRAM 22-29%) and has bandwidth to spare, so it can absorb the same
        // read-modify-write behind its own math.
        //
        // Two extra effects, both in the same direction:
        //   * with a NON-NVFP4 adapter (the shipping Q8_0 one) the up GEMM cannot fuse at all
        //     -- Route A needs an NVFP4 src0 and Route B is off -- so today the delta ADD runs
        //     as a separate full-width binbcast. Reversed, the fusable node is the base gemm,
        //     whose weight IS NVFP4, so the ADD disappears outright: 5 full-width passes
        //     (W out, W diff, R+R+W add) become 3 (W diff, R+W base).
        //   * a delta chain that ends in a CONCAT or a SCALE (multi-tensor modules, non-unit
        //     strength) is not a MUL_MAT and never fused forward; reversed it is only the
        //     addend, so those modules become fusable too.
        //
        // Not free everywhere: where the BASE gemm is itself bandwidth-bound (mlp.down measured
        // at DRAM 96%) there is no slack to hide the read-back in and this is a wash. Hence the
        // name filter.
        //
        // SD_LORA_ACC_BASE=1        reverse every module      (opt-in; default is unchanged)
        // SD_LORA_ACC_BASE_SKIP=a,b substring blacklist on the Linear's prefix, so a module
        //                           type whose base partner has no bandwidth to spare can be
        //                           held on the forward ordering.
        // Reversal preconditions, each of which would otherwise turn a win into a loss or a
        // wrong answer:
        //   * base_output_scale makes `out` a MUL node, not a MUL_MAT -> nothing to fuse into,
        //     and we would give up the forward fusion for nothing;
        //   * the ADD must not be a broadcast. Forward needs the delta repeatable into `out`;
        //     reversed needs `out` repeatable into the delta. Only equal shapes satisfy both,
        //     and a broadcasting delta would silently change the result. Re-checked per module
        //     below, since get_out_diff() is what decides the shape.
        bool reverse = sd_lora_acc_base_enabled() && base_output_scale == nullptr;
        if (reverse) {
            for (const std::string& pat : sd_lora_acc_base_skip()) {
                if (!pat.empty() && prefix.find(pat) != std::string::npos) {
                    reverse = false;
                    break;
                }
            }
        }

        // `delta` is only used by the reversed path. The forward path is left EXACTLY as it
        // was -- one in-place ADD per lora_model, in load order -- so with the flag off this
        // whole block is bit-identical to the pre-change engine for any number of adapters.
        ggml_tensor* delta = nullptr;
        for (auto& lora_model : lora_models) {
            ggml_tensor* out_diff = lora_model->get_out_diff(ctx, backend, x, w, forward_params, prefix + "weight");
            if (out_diff == nullptr) {
                continue;
            }
            if (!reverse || !ggml_are_same_shape(out, out_diff)) {
                // Fold anything already accumulated into `delta` back onto `out` first, so a
                // module that declines reversal mid-list cannot drop a delta.
                if (delta != nullptr) {
                    out   = ggml_add_inplace(ctx, out, delta);
                    delta = nullptr;
                }
                out = ggml_add_inplace(ctx, out, out_diff);
                continue;
            }
            delta = delta == nullptr ? out_diff : ggml_add_inplace(ctx, delta, out_diff);
        }
        if (delta != nullptr) {
            out = ggml_add_inplace(ctx, delta, out);
        }
        return out;
    }

    size_t get_extra_graph_size() override {
        size_t lora_tensor_num = 0;
        for (auto& lora_model : lora_models) {
            lora_tensor_num += lora_model->lora_tensors.size();
        }
        return LORA_GRAPH_BASE_SIZE + lora_tensor_num * 10;
    }
};

#endif  // __SD_MODEL_ADAPTER_LORA_HPP__
