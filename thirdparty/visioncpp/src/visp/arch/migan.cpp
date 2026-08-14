#include "visp/arch/migan.h"
#include "util/math.h"
#include "util/string.h"
#include "visp/image-impl.h"
#include "visp/nn.h"
#include "visp/vision.h"

#include <array>
#include <cmath>
#include <optional>

namespace visp {
namespace migan {

constexpr float sqrt2 = 1.4142135623f;

tensor lrelu_agc(model_ref m, tensor x, float alpha, float gain, float clamp) {
    x = ggml_leaky_relu(m, x, alpha, true);
    if (gain != 1) {
        x = ggml_scale_inplace(m, x, gain);
    }
    if (clamp != 0) {
        x = ggml_clamp(m, x, -clamp, clamp);
    }
    return named(m, x);
}

tensor downsample_2d(model_ref m, tensor x) {
    return conv_2d_depthwise(m["filter"], x, 2, 1);
}

tensor upsample_2d(model_ref m, tensor x) {
    tensor filter = m.weights("filter_const");
    if (m.flags & model_build_flag::cwhn) {
        filter = ggml_reshape_4d(m, filter, 1, filter->ne[0], filter->ne[1], 1);
    }

    auto [w, h, c, n] = nelements_whcn(m, x);
    x = interpolate(m, x, {w * 2, h * 2}, GGML_SCALE_MODE_NEAREST);
    x = ggml_mul_inplace(m, x, filter);
    x = conv_2d_depthwise(m["filter"], x, 1, 2); // 4x4 filter

    // remove padding from right and bottom
    if (m.flags & model_build_flag::cwhn) {
        x = slice(m, x, {}, {0, -1}, {0, -1}, {});
    } else {
        x = slice(m, x, {0, -1}, {0, -1}, {}, {});
    }
    x = ggml_cont(m, x); // required by subsequent ggml_scale for some reason
    return named(m, x);
}

tensor separable_conv_2d(model_ref m, tensor x, flags<conv> flags) {
    int kdim = (m.flags & model_build_flag::cwhn) ? 2 : 0; // to get kernel size
    int pad = int(m["conv1"].weights("weight")->ne[kdim] / 2);

    x = conv_2d_depthwise(m["conv1"], x, 1, pad);
    if (flags & conv::activation) {
        x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    }

    if (flags & conv::downsample) {
        x = downsample_2d(m["downsample"], x);
    }
    x = conv_2d(m["conv2"], x);
    if (flags & conv::upsample) {
        x = upsample_2d(m["upsample"], x);
    }

    if (flags & conv::noise) {
        tensor noise = m.weights("noise_const");
        noise = ggml_mul_inplace(m, noise, m.weights("noise_strength"));
        if (m.flags & model_build_flag::cwhn) {
            noise = ggml_reshape_4d(m, noise, 1, noise->ne[0], noise->ne[1], 1);
        }
        x = ggml_add_inplace(m, x, noise);
    }
    if (flags & conv::activation) {
        x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    }
    return named(m, x);
}

tensor from_rgb(model_ref m, tensor x) {
    x = cwhn_to_contiguous_2d(m, x);
    x = conv_2d(m["fromrgb"], x);
    x = lrelu_agc(m, x, 0.2f, sqrt2, 256);
    return named(m, x);
}

std::pair<tensor, tensor> encoder_block(model_ref m, tensor x, conv flag) {
    tensor feat = separable_conv_2d(m["conv1"], x, conv::activation);
    x = separable_conv_2d(m["conv2"], feat, conv::activation | flag);
    return {x, feat};
}

using Features = std::array<tensor, 9>;

std::pair<tensor, Features> encode(model_ref m, tensor x, int res) {
    ASSERT(res == int(x->ne[1]));
    int n = log2(res) - 1;
    ASSERT((1 << (n + 1)) == res);

    x = from_rgb(m[format<tensor_name>("b{}", res)], x);
    Features feats{};
    for (int i = 0; i < n - 1; ++i) {
        model_ref block = m[format<tensor_name>("b{}", res >> i)];
        std::tie(x, feats[i]) = encoder_block(block, x, conv::downsample);
    }
    std::tie(x, feats[n - 1]) = encoder_block(m["b4"], x);
    return {x, feats};
}

std::pair<tensor, tensor> synthesis_block(
    model_ref m, tensor x, tensor feat, tensor img, conv up_flag, conv noise_flag) {
    x = separable_conv_2d(m["conv1"], x, conv::activation | noise_flag | up_flag);
    x = ggml_add_inplace(m, x, feat);
    x = separable_conv_2d(m["conv2"], x, conv::activation | noise_flag);

    if (img) {
        img = upsample_2d(m["upsample"], img);
    }
    tensor y = conv_2d(m["torgb"], x);
    img = img ? ggml_add_inplace(m, img, y) : y;

    return {x, img};
}

tensor synthesis(model_ref m, tensor x_in, Features feats, int res) {
    int n = log2(res) - 1;
    ASSERT((1 << (n + 1)) == res);

    auto [x, img] = synthesis_block(m["b4"], x_in, feats[n - 1], nullptr);
    for (int i = n - 2; i >= 0; --i) {
        model_ref block = m[format<tensor_name>("b{}", res >> i)];
        std::tie(x, img) = synthesis_block(block, x, feats[i], img, conv::upsample, conv::noise);
    }
    img = contiguous_2d_to_cwhn(m, img);
    return img;
}

template <typename T>
void preprocess(
    image_source<T> img, image_source<uint8_t> mask, image_target<f32x4> dst, bool invert_mask) {

    for (int y = 0; y < img.extent[1]; ++y) {
        for (int x = 0; x < img.extent[0]; ++x) {
            i32x2 c = {x, y};
            float alpha = mask.load(c)[0];
            if (invert_mask) {
                alpha = 1.0f - alpha;
            }
            f32x4 color = alpha * (img.load(c) * 2.0f - f32x4(1.0f));
            dst.store(c, {alpha - 0.5f, color[0], color[1], color[2]});
        }
    }
}

} // namespace migan

tensor migan_generate(model_ref m, tensor image, migan_params const& p) {
    auto [x, feats] = migan::encode(m["encoder"], image, p.resolution);
    tensor result = migan::synthesis(m["synthesis"], x, feats, p.resolution);
    return compute_graph_output(m, result);
}

migan_params migan_detect_params(model_file const& f) {
    if (std::string_view arch = f.arch(); arch != "migan") {
        throw except("Architecture expected to be 'migan', but was '{}' ({})", arch, f.path);
    }
    migan_params p;
    p.resolution = f.get_int("migan.image_size");
    return p;
}

image_data migan_process_input(image_view image, image_view mask, migan_params const& p) {
    i32x2 res = {p.resolution, p.resolution};
    std::optional<image_data> resized_image;
    if (image.extent != res) {
        resized_image = image_scale(image, res);
        image = image_view(*resized_image);
    }
    std::optional<image_data> resized_mask;
    if (mask.extent != res) {
        resized_mask = image_scale(mask, res);
        mask = image_view(*resized_mask);
    }
    image_data result = image_alloc(res, image_format::rgba_f32);
    switch (n_channels(image)) {
        case 3: migan::preprocess<u8x3>(image, mask, result, p.invert_mask); break;
        case 4: migan::preprocess<u8x4>(image, mask, result, p.invert_mask); break;
        default: ASSERT(false, "Unsupported image format for migan image input");
    }
    return result;
}

image_data migan_process_output(std::span<float const> data, i32x2 extent, migan_params const& p) {
    i32x2 model_extent = {p.resolution, p.resolution};
    image_view image(model_extent, image_format::rgb_f32, data.data());
    image_data resized;
    if (model_extent != extent) {
        resized = image_scale(image, extent);
        image = image_view(resized);
    }
    return image_f32_to_u8(image, image_format::rgba_u8, 0.5f, 0.5f);
}

} // namespace visp