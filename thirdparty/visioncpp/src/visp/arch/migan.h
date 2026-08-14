#pragma once

#include "visp/image.h"
#include "visp/ml.h"
#include "visp/util.h"

#include <array>
#include <cmath>

namespace visp::migan {

enum class conv {
    none = 0,
    upsample = 1 << 0,
    downsample = 1 << 1,
    noise = 1 << 2,
    activation = 1 << 3,
};

constexpr flags<conv> operator|(conv lhs, conv rhs) {
    return flags<conv>(uint32_t(lhs) | uint32_t(rhs));
}

using features = std::array<tensor, 9>;

tensor lrelu_agc(model_ref m, tensor x, float alpha = 0.2f, float gain = 1, float clamp = 0);
tensor downsample_2d(model_ref m, tensor x);
tensor upsample_2d(model_ref m, tensor x);
tensor separable_conv_2d(model_ref m, tensor x, flags<conv> flags = {});
tensor from_rgb(model_ref m, tensor x);

std::pair<tensor, tensor> encoder_block(model_ref m, tensor x, conv flag = conv::none);
std::pair<tensor, features> encode(model_ref m, tensor x, int res);

std::pair<tensor, tensor> synthesis_block(
    model_ref m,
    tensor x,
    tensor feat,
    tensor img,
    conv up_flag = conv::none,
    conv noise_flag = conv::none);

tensor synthesis(model_ref m, tensor x_in, features feats, int res);

} // namespace visp::migan