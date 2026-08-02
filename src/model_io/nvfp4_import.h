#ifndef __SD_NVFP4_IMPORT_H__
#define __SD_NVFP4_IMPORT_H__

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "tensor_storage.h"

// Ingestion of NVFP4 checkpoints quantised by an EXTERNAL tool, as opposed to this
// engine's own converter.  Two on-disk formats are handled; both store the same numbers
// and differ only in tensor names and where the config lives:
//
//   ModelOpt        `<L>.weight`        U8      [out, in/2]   packed E2M1 nibbles
//   (hf_quant_config `<L>.weight_scale`  F8_E4M3 [out, in/16]  per-16-group scale
//    producer.name = `<L>.weight_scale_2` F32    scalar        per-TENSOR global
//    "modelopt")    `<L>.input_scale`   F32     scalar        static activation scale
//
//   llm-compressor  `<L>.weight_packed`       U8      [out, in/2]
//   (format         `<L>.weight_scale`        F8_E4M3 [out, in/16]
//    "nvfp4-pack-   `<L>.weight_global_scale` F32     scalar
//     quantized")
//
// Dequantisation is identical for both:
//     w[o,i] = E2M1(nibble) * E4M3(weight_scale[o, i/16]) * <global>
//
// The 4-bit codes and the FP8 group scales are byte-for-byte what ggml's block_nvfp4
// already stores (see the convention note at the top of ggml/src/ggml-cuda/
// nvfp4-cublaslt.cu: ggml's 2x kvalues_mxfp4 and its /2 UE4M3 decode cancel, so the
// stored bytes ARE the standard e4m3 * standard e2m1 product).  Import is therefore a
// pure RE-LAYOUT with no arithmetic and no loss:
//
//   * group scales   : verbatim byte copy, [out, in/16] row-major -> block_nvfp4::d[4].
//   * nibbles        : source packs consecutive elements (2j low, 2j+1 high); block_nvfp4
//                      packs split halves of each 16-group (j low, j+8 high).  This is
//                      exactly the inverse of repack_weight_kernel() in nvfp4-cublaslt.cu.
//   * per-tensor global: CANNOT be folded into the group scale -- that scale is a 3-bit-
//                      mantissa FP8 already saturated at 448 by construction, so the
//                      product would round and, for small groups, flush to zero.  It is
//                      emitted instead as the sibling F32 scalar `<L>.weight.wglobal`
//                      that this fork's UNFOLDED-NVFP4 path already consumes end to end
//                      (Linear::init_params -> ggml_cuda_nvfp4_register_weight_global,
//                      with a graph-level ggml_mul fallback when the GEMM cannot fold it).
//
// `input_scale` is dropped: the NVFP4 GEMM quantises activations dynamically per call
// (quant_act_kernel), so a static activation scale has no consumer here.
//
// `pre_quant_scale` (ModelOpt AWQ-style input smoothing, added by ComfyUI PR #15224) is
// NOT dropped and NOT supported -- it is a per-input-channel vector that must multiply
// the ACTIVATION before the matmul, and no such hook exists in this engine's Linear.
// Import fails loudly rather than produce a silently mis-scaled model.

// Rewrite the tensor list read from a safetensors file so that externally-quantised NVFP4
// linears appear as ordinary GGML_TYPE_NVFP4 weights plus `.wglobal` sidecars.
//
// Returns false only on a MALFORMED external-NVFP4 checkpoint (an error is logged); a file
// that simply is not one is left untouched and returns true.  Cheap to call on every
// safetensors file: it bails after scanning the already-parsed names unless a
// `weight_scale_2` / `weight_global_scale` entry is present.
bool nvfp4_import_rewrite_safetensors(const std::string& file_path,
                                      std::vector<TensorStorage>& tensor_storages);

// Reads `n` bytes at absolute file offset `off` into `buf`.
typedef std::function<bool(char* buf, size_t n, uint64_t off)> nvfp4_read_fn;

// Materialise the host-side bytes of an imported tensor: `dst_nbytes` of block_nvfp4 for a
// packed weight, or 4 bytes of F32 for a `.wglobal` sidecar.  `scratch` is reused across
// calls by the caller's worker thread.
bool nvfp4_import_materialize(const TensorStorage& tensor_storage,
                              size_t dst_nbytes,
                              const nvfp4_read_fn& read_at,
                              std::vector<uint8_t>& scratch,
                              std::vector<uint8_t>& out);

// Repack one weight matrix. `packed` is out*(in/2) bytes, `scales` is out*(in/16) bytes,
// `dst` is out*(in/64) block_nvfp4 (36 bytes each). `in` must be a multiple of 64.
void nvfp4_import_assemble_blocks(const uint8_t* packed,
                                  const uint8_t* scales,
                                  uint8_t* dst,
                                  int64_t in_features,
                                  int64_t out_features);

#endif  // __SD_NVFP4_IMPORT_H__
