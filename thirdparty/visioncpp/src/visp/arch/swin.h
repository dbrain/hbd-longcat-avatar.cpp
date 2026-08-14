#pragma once

#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::swin {

struct block_params {
    int n_heads = 6;
    int window_size = 7;
    int64_t w = 0;
    int64_t h = 0;
    int shift = 0;
};

struct layer_result {
    tensor x_out;
    int64_t w_out;
    int64_t h_out;
    tensor x_down;
    int64_t w_down;
    int64_t h_down;
};

void compute_relative_position_index(span<int32_t> dst, int window_size);
tensor_data create_relative_position_index(ggml_context* ctx, int window_size);
void compute_attention_mask(std::span<byte> out, int64_t w, int64_t h, int window_size);
tensor_data create_attention_mask(ggml_context* ctx, int64_t w, int64_t h, int window_size);

tensor mlp(model_ref m, tensor x);
tensor patch_merging(model_ref m, tensor x, int64_t w, int64_t h);
tensor window_partition(model_ref m, tensor x, int window);
tensor window_reverse(model_ref m, tensor x, int w, int h, int window);
tensor window_attention(model_ref m, tensor x, tensor mask, int num_heads, int window);
tensor block(model_ref m, tensor x, tensor mask, block_params const&);
layer_result layer(
    model_ref, tensor, int64_t w, int64_t h, swin_layer_t const&, int window_size, bool downsample);

} // namespace visp::swin