// Opt-in, final-prefix-token activation trace for the Qwen3 parity harness.
// Set RIG_QWEN3_SUBTRACE_DIR to an explicit directory.  This is deliberately
// graph-tap based: it observes the values consumed by the native forward, not
// values from a second diagnostic forward.
#pragma once

#include "m1_ggml.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Qwen3SubtraceTap {
    std::string name;
    ggml_tensor * tensor;
    std::vector<int64_t> shape; // NumPy C-order; ggml's ne0 is contiguous.
};

struct Qwen3Subtrace {
    std::string dir;
    std::vector<Qwen3SubtraceTap> taps;

    static const char * env_dir() {
        const char * dir = std::getenv("RIG_QWEN3_SUBTRACE_DIR");
        return dir && dir[0] ? dir : nullptr;
    }
    bool enabled() const { return !dir.empty(); }

    void add(ggml_context * ctx, const std::string & name, ggml_tensor * t,
             const std::vector<int64_t> & shape) {
        if (!enabled()) return;
        // All files are fp32, including BF16 diagnostic paths, so a Python
        // forward hook can load them directly without dtype special cases.
        if (t->type != GGML_TYPE_F32) t = ggml_cast(ctx, t, GGML_TYPE_F32);
        ggml_set_output(t);
        taps.push_back({name, t, shape});
    }

    void add_last_2d(ggml_context * ctx, const std::string & name, ggml_tensor * t) {
        // t is [channels, tokens]; the desired final prefix token is [channels].
        ggml_tensor * last = ggml_view_1d(ctx, t, t->ne[0], (size_t)(t->ne[1] - 1) * t->nb[1]);
        add(ctx, name, last, {t->ne[0]});
    }

    void add_last_heads(ggml_context * ctx, const std::string & name, ggml_tensor * t) {
        // ggml is [head_dim, heads, tokens].  Its contiguous storage is NumPy
        // C-order [heads, head_dim], exactly the layout seen by HF hooks.
        ggml_tensor * last = ggml_view_2d(ctx, t, t->ne[0], t->ne[1], t->nb[1],
                                           (size_t)(t->ne[2] - 1) * t->nb[2]);
        add(ctx, name, last, {t->ne[1], t->ne[0]});
    }

    // Scores are [keys, queries, heads].  Retain the final query as a
    // contiguous [heads, keys] NumPy array; a raw view would have the query
    // stride between heads and backend_tensor_get would serialize gaps.
    void add_final_query_scores(ggml_context * ctx, const std::string & name, ggml_tensor * scores) {
        ggml_tensor * last = ggml_view_3d(ctx, scores, scores->ne[0], 1, scores->ne[2],
                                           scores->nb[1], scores->nb[2],
                                           (size_t)(scores->ne[1] - 1) * scores->nb[1]);
        last = ggml_cont(ctx, ggml_permute(ctx, last, 0, 2, 1, 3)); // [keys, heads, 1]
        add_last_heads(ctx, name, last);
    }

    void add_to_graph(ggml_cgraph * gf) const {
        for (const auto & tap : taps) ggml_build_forward_expand(gf, tap.tensor);
    }

    static void write_npy(const std::string & path, const float * data,
                          const std::vector<int64_t> & shape) {
        std::string dims = "(";
        for (int64_t d : shape) dims += std::to_string(d) + ",";
        dims += ")";
        std::string hdr = "{'descr': '<f4', 'fortran_order': False, 'shape': " + dims + ", }";
        const size_t pad = (64 - ((10 + hdr.size() + 1) % 64)) % 64;
        hdr.append(pad, ' '); hdr.push_back('\n');
        const uint16_t hlen = (uint16_t) hdr.size();
        size_t n = 1; for (int64_t d : shape) n *= (size_t) d;
        std::ofstream out(path, std::ios::binary);
        if (!out) throw std::runtime_error("cannot write Qwen3 subtrace: " + path);
        out.write("\x93NUMPY", 6);
        const char ver[2] = {1, 0}; out.write(ver, 2);
        out.write(reinterpret_cast<const char *>(&hlen), sizeof(hlen));
        out.write(hdr.data(), (std::streamsize) hdr.size());
        out.write(reinterpret_cast<const char *>(data), (std::streamsize)(n * sizeof(float)));
        if (!out) throw std::runtime_error("failed writing Qwen3 subtrace: " + path);
    }

    void write(M1Harness & H) const {
        if (!enabled()) return;
        std::filesystem::create_directories(dir);
        for (const auto & tap : taps) {
            size_t n = 1; for (int64_t d : tap.shape) n *= (size_t) d;
            std::vector<float> host(n);
            ggml_backend_tensor_get(tap.tensor, host.data(), 0, n * sizeof(float));
            write_npy(dir + "/" + tap.name + ".npy", host.data(), tap.shape);
        }
    }
};
