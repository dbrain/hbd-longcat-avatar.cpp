#ifndef __WAN_SHOTSTREAM_HPP__
#define __WAN_SHOTSTREAM_HPP__

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "ggml_extend.hpp"
#include "model/common/rope.hpp"
#include "model/diffusion/wan.hpp"

// Kling ShotStream's causal multi-shot T2V loop. Cache ownership deliberately
// stays in this runner rather than changing the upstream Wan model lifecycle.
namespace WAN_SHOTSTREAM {

static constexpr float WARPED_TIMESTEPS[] = {1000.0f, 957.929f, 888.889f, 737.589f};

struct ShotStreamConfig {
    int frames_per_chunk  = 3;
    int frames_per_shot   = 21;
    int context_frames    = 6;
    int condition_start   = 6;
    float timestep_shift  = 8.0f;
    float shot_rope_theta = 1.0f / 6.0f;
};

inline std::vector<float> build_pe(const WAN::WanConfig& config,
                                   int frames,
                                   int height,
                                   int width,
                                   int temporal_offset,
                                   int shot_index,
                                   float shot_rope_theta) {
    const auto [pt, ph, pw] = config.patch_size;
    auto ids = Rope::gen_vid_ids(frames, height, width, pt, ph, pw, 1, temporal_offset, 0, 0);
    auto pe = Rope::embed_nd(ids, 1, static_cast<float>(config.theta), config.axes_dim);
    if (shot_index == 0 || shot_rope_theta == 0.0f) {
        return pe;
    }
    const int channels = static_cast<int>(config.axes_dim_sum / 2);
    const int temporal_channels = config.axes_dim[0] / 2;
    const int stride = channels * 4;
    const float delta = shot_index * shot_rope_theta;
    const float c = std::cos(delta), s = std::sin(delta);
    for (size_t p = 0; p < pe.size() / stride; ++p) {
        for (int channel = 0; channel < temporal_channels; ++channel) {
            float* cell = &pe[p * stride + channel * 4];
            const float old_c = cell[0], old_s = cell[2];
            const float new_c = old_c * c - old_s * s;
            const float new_s = old_s * c + old_c * s;
            cell[0] = new_c; cell[1] = -new_s; cell[2] = new_s; cell[3] = new_c;
        }
    }
    return pe;
}

struct ShotStreamRunner : public GGMLRunner {
    WAN::WanConfig config;
    WAN::Wan wan;
    ShotStreamConfig shotstream;
    std::string desc = "ShotStream-1.3B";
    std::vector<sd::Tensor<ggml_fp16_t>> local_k, local_v, context_k, context_v;
    std::vector<float> pe_hold;
    sd::Tensor<float> timestep_hold;
    std::mt19937 rng{1234};

    ShotStreamRunner(ggml_backend_t backend,
                     const String2TensorStorage& tensors = {},
                     const std::string& prefix = "model.diffusion_model",
                     std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : GGMLRunner(backend, weight_manager),
          config(WAN::WanConfig::detect_from_weights(tensors, prefix)) {
        // ShotStream uses the merged Wan2.1 T2V-1.3B weight layout.
        config.model_type = "t2v";
        wan = WAN::Wan(config);
        wan.init(params_ctx, tensors, prefix);
    }

    std::string get_desc() override { return desc; }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors, const std::string prefix) {
        wan.get_param_tensors(tensors, prefix);
    }

    void reset_local_cache() { local_k.clear(); local_v.clear(); }
    void reset_context_cache() { context_k.clear(); context_v.clear(); }

    sd::Tensor<float> random_noise(int width, int height, int frames) {
        sd::Tensor<float> out({width, height, frames, 16});
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (int64_t i = 0; i < out.numel(); ++i) out.data()[i] = normal(rng);
        return out;
    }

    sd::Tensor<float> forward_block(int n_threads,
                                    int temporal_offset,
                                    int shot_index,
                                    const sd::Tensor<float>& x,
                                    float timestep,
                                    const sd::Tensor<float>& context,
                                    bool persist_local,
                                    bool persist_context) {
        const int frames = static_cast<int>(x.shape()[2]);
        const int height = static_cast<int>(x.shape()[1]);
        const int width = static_cast<int>(x.shape()[0]);
        const auto [pt, ph, pw] = config.patch_size;
        const int tokens = ((frames + pt / 2) / pt) * ((height + ph / 2) / ph) * ((width + pw / 2) / pw);
        const int heads = static_cast<int>(config.num_heads);
        const int head_dim = static_cast<int>(config.dim / config.num_heads);
        const bool retain_kv = persist_local || persist_context;
        std::vector<sd::Tensor<ggml_fp16_t>> new_k(config.num_layers), new_v(config.num_layers);
        std::vector<bool> got_k(config.num_layers), got_v(config.num_layers);

        auto get_graph = [&]() -> ggml_cgraph* {
            auto graph = new_graph_custom(WAN::WAN_GRAPH_SIZE);
            auto gx = make_input(x);
            auto gcontext = make_input(context);
            timestep_hold = sd::Tensor<float>::from_vector({timestep});
            auto gtimestep = make_input(timestep_hold);
            pe_hold = build_pe(config, frames, height, width, temporal_offset, shot_index, shotstream.shot_rope_theta);
            const int positions = static_cast<int>(pe_hold.size() / config.axes_dim_sum / 2);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, config.axes_dim_sum / 2, positions);
            set_backend_tensor_data(pe, pe_hold.data());

            std::vector<ggml_tensor*> local_k_input, local_v_input, context_k_input, context_v_input;
            if (!local_k.empty()) {
                local_k_input.resize(config.num_layers); local_v_input.resize(config.num_layers);
                for (int i = 0; i < config.num_layers; ++i) {
                    local_k_input[i] = make_input(local_k[i]);
                    local_v_input[i] = make_input(local_v[i]);
                }
            }
            if (!context_k.empty()) {
                context_k_input.resize(config.num_layers); context_v_input.resize(config.num_layers);
                for (int i = 0; i < config.num_layers; ++i) {
                    context_k_input[i] = make_input(context_k[i]);
                    context_v_input[i] = make_input(context_v[i]);
                }
            }
            auto runner_context = get_context();
            std::vector<ggml_tensor*> graph_k, graph_v;
            auto velocity = wan.forward_causal_block(&runner_context, gx, gtimestep, gcontext, pe,
                                                      local_k_input, local_v_input,
                                                      context_k_input, context_v_input, graph_k, graph_v);
            if (retain_kv) {
                for (int i = 0; i < config.num_layers; ++i) {
                    auto k = ggml_cast(runner_context.ggml_ctx, graph_k[i], GGML_TYPE_F16);
                    auto v = ggml_cast(runner_context.ggml_ctx, graph_v[i], GGML_TYPE_F16);
                    ggml_set_output(k); ggml_set_output(v);
                    ggml_build_forward_expand(graph, k); ggml_build_forward_expand(graph, v);
                    sd::ggml_graph_cut::mark_graph_cut(k, "shotstream.blocks." + std::to_string(i) + ".out", "k");
                    sd::ggml_graph_cut::mark_graph_cut(v, "shotstream.blocks." + std::to_string(i) + ".out", "v");
                }
            }
            ggml_set_output(velocity);
            ggml_build_forward_expand(graph, velocity);
            return graph;
        };

        if (retain_kv) {
            segment_readback_hook_ = [&](ggml_cgraph* graph) {
                for (int node = 0; node < ggml_graph_n_nodes(graph); ++node) {
                    auto tensor = ggml_graph_node(graph, node);
                    if (tensor == nullptr || tensor->buffer == nullptr) continue;
                    for (int layer = 0; layer < config.num_layers; ++layer) {
                        const auto group = "shotstream.blocks." + std::to_string(layer) + ".out";
                        if (!got_k[layer] && std::strcmp(tensor->name, sd::ggml_graph_cut::make_graph_cut_name(group, "k").c_str()) == 0) {
                            new_k[layer] = sd::Tensor<ggml_fp16_t>({head_dim, tokens, heads});
                            ggml_backend_tensor_get(tensor, new_k[layer].data(), 0, ggml_nbytes(tensor)); got_k[layer] = true;
                        }
                        if (!got_v[layer] && std::strcmp(tensor->name, sd::ggml_graph_cut::make_graph_cut_name(group, "v").c_str()) == 0) {
                            new_v[layer] = sd::Tensor<ggml_fp16_t>({head_dim, tokens, heads});
                            ggml_backend_tensor_get(tensor, new_v[layer].data(), 0, ggml_nbytes(tensor)); got_v[layer] = true;
                        }
                    }
                }
            };
        }
        auto result = GGMLRunner::compute<float>(get_graph, n_threads, true, false);
        segment_readback_hook_ = nullptr;
        if (!result.has_value() || result->empty()) return {};
        if (retain_kv && (!std::all_of(got_k.begin(), got_k.end(), [](bool value) { return value; }) ||
                          !std::all_of(got_v.begin(), got_v.end(), [](bool value) { return value; }))) return {};
        if (persist_context) { context_k = std::move(new_k); context_v = std::move(new_v); }
        if (persist_local) {
            if (local_k.empty()) { local_k = std::move(new_k); local_v = std::move(new_v); }
            else for (int i = 0; i < config.num_layers; ++i) {
                local_k[i] = sd::ops::concat(local_k[i], new_k[i], 1);
                local_v[i] = sd::ops::concat(local_v[i], new_v[i], 1);
            }
        }
        auto velocity = std::move(*result);
        velocity.reshape_({width, height, frames, 16});
        return velocity;
    }

    void prefill_context(int n_threads, const sd::Tensor<float>& latents, const sd::Tensor<float>& context) {
        reset_local_cache(); reset_context_cache();
        if (!latents.empty()) (void)forward_block(n_threads, 0, 0, latents, 0.0f, context, false, true);
    }

    sd::Tensor<float> run_shot(int n_threads, int shot_index, int width, int height, const sd::Tensor<float>& context) {
        reset_local_cache();
        sd::Tensor<float> out({width, height, shotstream.frames_per_shot, 16});
        for (int start = 0; start < shotstream.frames_per_shot; start += shotstream.frames_per_chunk) {
            auto x = random_noise(width, height, shotstream.frames_per_chunk);
            sd::Tensor<float> x0;
            for (float timestep : WARPED_TIMESTEPS) {
                auto flow = forward_block(n_threads, shotstream.condition_start + start, shot_index, x, timestep, context, false, false);
                if (flow.empty()) return {};
                const float sigma = timestep / 1000.0f;
                x0 = sd::Tensor<float>(x.shape());
                for (int64_t i = 0; i < x.numel(); ++i) x0.data()[i] = x.data()[i] - sigma * flow.data()[i];
                x = x0;
            }
            if (start + shotstream.frames_per_chunk < shotstream.frames_per_shot &&
                forward_block(n_threads, shotstream.condition_start + start, shot_index, x0, 0.0f, context, true, false).empty()) return {};
            sd::ops::slice_assign(&out, 2, start, start + shotstream.frames_per_chunk, x0);
        }
        return out;
    }
};

} // namespace WAN_SHOTSTREAM

#endif
