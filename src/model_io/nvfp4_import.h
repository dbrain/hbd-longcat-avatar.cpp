#ifndef __SD_NVFP4_IMPORT_H__
#define __SD_NVFP4_IMPORT_H__

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "tensor_storage.h"

// Ingestion of NVFP4 checkpoints quantised by an EXTERNAL tool, as opposed to this
// engine's own converter.  THREE on-disk formats are handled.  All three store the same
// numbers, but they do NOT all store them in the same order -- see the ComfyUI note below,
// which is the one that bites.
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
//   ComfyUI         `<L>.weight`          U8      [out, in/2]     packed E2M1 nibbles
//   (`<L>.comfy_    `<L>.weight_scale`    F8_E4M3 [rows*, cols*]  per-16-group scale,
//     quant` U8      `<L>.weight_scale_2`  F32     scalar          cuBLAS-SWIZZLED
//     holds JSON     `<L>.comfy_quant`     U8      json bytes
//     {"format":                                  rows* = roundup(out, 128)
//      "nvfp4"})                                  cols* = roundup(in/16, 4)
//
// Dequantisation is identical for all three:
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
//
// 🔴 COMFYUI USES MODELOPT'S TENSOR NAMES AND A DIFFERENT BYTE ORDER.  `<L>.weight` +
// `<L>.weight_scale` + `<L>.weight_scale_2` is EXACTLY the ModelOpt triple, so the
// ModelOpt branch below matches a ComfyUI file on names alone and would import it without
// a single warning -- and get numeric garbage, because comfy-kitchen
// (TensorCoreNVFP4Layout -> quantize_nvfp4) differs in TWO places:
//
//   1. group scales are stored in the cuBLAS "blocked" swizzle, not row-major.  For one
//      real layer, reading them row-major gave a per-output-row L2 norm correlation of
//      0.013 against the bf16 the file was quantised from; de-swizzling gave 0.99994.
//      There is no error, no NaN, no shape mismatch -- just a wrong model.
//   2. nibble order is hi_first: element 2j is in the HIGH nibble, 2j+1 in the LOW.
//      ModelOpt and llm-compressor are the other way round.  This one is a permutation
//      within each 16-group, so it is invisible to every norm-based check.
//
// The dispatch therefore keys on `comfy_quant`, NOT on tensor names: a file with any
// `<L>.comfy_quant` tensor takes the ComfyUI branch, everything else takes the
// name-sniffing branch.  Format strings other than "nvfp4" are REFUSED, not guessed at.

// What a safetensors file's `comfy_quant` tensors say about it.
enum class ComfyQuantScan {
    None,         // no `.comfy_quant` tensor: not a ComfyUI-quantised checkpoint
    Nvfp4,        // every `.comfy_quant` is exactly {"format": "nvfp4"}
    Unsupported,  // at least one names a format/modifier this engine cannot load
    Malformed,    // header or a `comfy_quant` blob could not be read
};

// Parse the safetensors header and every `<L>.comfy_quant` JSON blob in it.
//
// Call this BEFORE read_safetensors_file(): an `int8_tensorwise` checkpoint stores I8
// weights, which that reader rejects with a bare "unsupported dtype 'I8'" that says
// nothing about why.  On Unsupported/Malformed, `*detail` (if non-null) is filled with a
// message naming the offending layer and the exact blob it carried.
ComfyQuantScan comfy_quant_scan_safetensors(const std::string& file_path, std::string* detail);

// Rewrite the tensor list read from a safetensors file so that externally-quantised NVFP4
// linears appear as ordinary GGML_TYPE_NVFP4 weights plus `.wglobal` sidecars.
//
// `comfy` must be the result of comfy_quant_scan_safetensors() on the same file; passing
// None for a file that does have `comfy_quant` tensors is the silent-corruption bug this
// header warns about, so the ModelOpt branch asserts the file has none.
//
// Returns false only on a MALFORMED external-NVFP4 checkpoint (an error is logged); a file
// that simply is not one is left untouched and returns true.  Cheap to call on every
// safetensors file: it bails after scanning the already-parsed names unless a
// `weight_scale_2` / `weight_global_scale` entry is present.
bool nvfp4_import_rewrite_safetensors(const std::string& file_path,
                                      std::vector<TensorStorage>& tensor_storages,
                                      ComfyQuantScan comfy = ComfyQuantScan::None);

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

// As above for a ComfyUI/comfy-kitchen export: `packed` holds element 2j in the HIGH
// nibble, and `scales` is roundup(out,128) * roundup(in/16,4) bytes in the cuBLAS blocked
// swizzle rather than out*(in/16) row-major.
void nvfp4_import_assemble_blocks_comfy(const uint8_t* packed,
                                        const uint8_t* scales,
                                        uint8_t* dst,
                                        int64_t in_features,
                                        int64_t out_features);

#endif  // __SD_NVFP4_IMPORT_H__
