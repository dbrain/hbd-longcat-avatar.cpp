#pragma once

#include "visp/ml.h"

namespace visp::esrgan {

tensor upsample(model_ref m, tensor x);
tensor conv_block(model_ref m, tensor x);
tensor risidual_dense_block(model_ref m, tensor x);
tensor rrdb(model_ref m, tensor x);

} // namespace visp::esrgan