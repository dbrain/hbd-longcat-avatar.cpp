// CUDA-side PyTorch 2.7 exponential_ stream used by multinomial without
// replacement.  The implementation deliberately returns every logical value:
// masked logits still consume Philox values in the upstream flattened race.
#pragma once

#include <cstdint>
#include <vector>

extern "C" bool rig_torch_philox_exponentials(std::vector<float>& out,
                                               uint64_t seed,
                                               uint64_t* offset);
