#pragma once

#include "visp/image.h"
#include "visp/ml.h"
#include "visp/vision.h"

namespace visp::birefnet {

// Encoder

swin_result encode_concat(model_ref m, swin_result& xs, swin_result& xs_low);
swin_result encode(model_ref m, tensor x, swin_params const& p);

// Decoder

tensor deformable_conv_2d(model_ref m, tensor x, int stride = 1, int pad = 0);
tensor mean_2d(model_ref m, tensor x);
tensor global_avg_pool(model_ref m, tensor x);
tensor aspp_module_deformable(model_ref m, tensor x, int padding = 0);
tensor aspp_deformable(model_ref m, tensor x);
tensor basic_decoder_block(model_ref m, tensor x);
tensor simple_conv(model_ref m, tensor x);
tensor image_to_patches(model_ref m, tensor x, int64_t out_w, int64_t out_h);
tensor gdt_conv(model_ref m, tensor x);
tensor decode(model_ref m, tensor x, swin_result const& features);

} // namespace visp::birefnet