
#include "visp/arch/depth-anything.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/arch/dino.h"
#include "visp/ml.h"
#include "visp/nn.h"

namespace visp {
namespace dpt {

int32_t const bilinear_align_corners = int32_t(GGML_SCALE_MODE_BILINEAR) |
    GGML_SCALE_FLAG_ALIGN_CORNERS;

tensor residual_conv(model_ref m, tensor x) {
    tensor out = x;
    out = ggml_relu(m, out);
    out = conv_2d(m["convolution1"], out, 1, 1);
    out = ggml_relu(m, out);
    out = conv_2d(m["convolution2"], out, 1, 1);
    x = ggml_add_inplace(m, x, out);
    return named(m, x);
}

tensor feature_fusion(model_ref m, tensor x0, tensor x1, int64_t const* size) {
    tensor x = x0;
    if (x1) {
        tensor res = residual_conv(m["residual_layer1"], x1);
        x = ggml_add_inplace(m, x, res);
    }
    x = residual_conv(m["residual_layer2"], x);

    int const dim = m.flags & model_build_flag::cwhn ? 1 : 0;
    int64_t w = size ? size[dim + 0] : x->ne[dim + 0] * 2;
    int64_t h = size ? size[dim + 1] : x->ne[dim + 1] * 2;
    x = contiguous_2d_to_whcn(m, x);
    x = interpolate(m, x, {w, h}, bilinear_align_corners);
    x = whcn_to_contiguous_2d(m, x);

    x = conv_2d(m["projection"], x);
    return named(m, x);
}

tensor neck(model_ref m, span<tensor> features, int64_t patch_w, int64_t patch_h) {
    ASSERT(features.size() == 4);
    std::array<tensor, 4> layer;

    model_ref reassemble = m["reassemble_stage.layers"];
    for (int i = 0; i < 4; ++i) {
        tensor x = features[i];
        x = slice(m, x, {}, {1, x->ne[1]}, {}, {});
        x = ggml_reshape_4d(m, x, x->ne[0], patch_w, patch_h, x->ne[3]);

        model_ref proj = reassemble[i]["projection"];
        proj.flags |= model_build_flag::cwhn;
        x = conv_2d(proj, x); // 1x1 conv, keep CWHN layout and directly use mul_mat

        x = cwhn_to_contiguous_2d(m, x);
        switch (i) {
            case 0: x = conv_transpose_2d(reassemble[i]["resize"], x, 4); break;
            case 1: x = conv_transpose_2d(reassemble[i]["resize"], x, 2); break;
            case 3: x = conv_2d(reassemble[i]["resize"], x, 2, 1); break;
        }
        layer[i] = x;
    }

    model_ref convs = m["convs"];
    for (int i = 0; i < 4; ++i) {
        layer[i] = conv_2d(convs[i], layer[i], 1, 1);
    }

    model_ref fusion = m["fusion_stage.layers"];
    tensor fused;
    fused = feature_fusion(fusion[0], layer[3], nullptr, layer[2]->ne);
    fused = feature_fusion(fusion[1], fused, layer[2], layer[1]->ne);
    fused = feature_fusion(fusion[2], fused, layer[1], layer[0]->ne);
    fused = feature_fusion(fusion[3], fused, layer[0]);
    return fused;
}

tensor head(model_ref m, tensor x, int64_t w, int64_t h, float max_depth) {
    tensor out = conv_2d(m["conv1"], x, 1, 1);
    out = contiguous_2d_to_whcn(m, out);
    out = interpolate(m, out, {w, h}, bilinear_align_corners);
    out = whcn_to_contiguous_2d(m, out);

    out = conv_2d(m["conv2"], out, 1, 1);
    out = ggml_relu_inplace(m, out);
    out = conv_2d(m["conv3"], out);
    out = ggml_relu_inplace(m, out);

    if (max_depth != 1) {
        out = ggml_scale(m, out, max_depth);
    }
    return out;
}

} // namespace dpt

tensor depthany_predict(model_ref m, tensor image, depthany_params const& p) {
    auto [c, w, h, n] = nelements(image);
    int64_t w_patch = w / p.dino.patch_size;
    int64_t h_patch = h / p.dino.patch_size;

    auto features = dino_get_intermediate_layers(m["backbone"], image, p.feature_layers, p.dino);
    tensor fused = dpt::neck(m["neck"], features, w_patch, h_patch);
    tensor depth = dpt::head(m["head"], fused, w, h, p.max_depth);

    return compute_graph_output(m, depth);
}

i32x2 depthany_image_extent(i32x2 extent, depthany_params const& p) {
    int min_side = std::min(extent[0], extent[1]);
    int tgt_side = std::max(p.image_size, next_multiple(min_side, p.image_multiple));
    i32x2 target = extent * tgt_side / min_side;
    return next_multiple(target, p.image_multiple);
}

depthany_params depthany_detect_params(model_file const& file, i32x2 input_extent) {
    depthany_params p;
    p.dino = dino_detect_params(file);
    p.image_size = file.get_int("depthanything.image_size");
    file.get_array("depthanything.feature_layers", p.feature_layers);
    if (input_extent[0] > 0 && input_extent[1] > 0) {
        p.image_extent = depthany_image_extent(input_extent, p);
    }
    return p;
}

image_data depthany_process_input(image_view image, depthany_params const& p) {
    constexpr f32x4 mean = f32x4{0.485f, 0.456f, 0.406f, 0.f};
    constexpr f32x4 std = f32x4{0.229f, 0.224f, 0.225f, 1.f};

    image_data resized;
    if (image.extent != p.image_extent) {
        resized = image_scale(image, p.image_extent);
        image = image_view(resized);
    }
    return image_u8_to_f32(image, image_format::rgb_f32, -mean, 1.f / std);
}

image_data depthany_process_output(span<float const> data, i32x2 extent, depthany_params const& p) {
    image_view depth_output(p.image_extent, data);
    image_data normalized = image_normalize(depth_output);
    if (normalized.extent != extent) {
        return image_scale(normalized, extent);
    }
    return normalized;
}

} // namespace visp