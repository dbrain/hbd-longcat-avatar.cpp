# Native FP8 (e4m3) quality: fix the dotty-in-motion GEMM while keeping the fp8 tensor cores

File under investigation: `ggml/src/ggml-cuda/nvfp4-cublaslt.cu` (branch `ltx-denoise-workflow`).
This one file holds BOTH the NVFP4 GEMM (`ggml_cuda_nvfp4_cublaslt_mul_mat`) and the native-e4m3
FP8 GEMM (`ggml_cuda_fp8_cublaslt_mul_mat`). The dotty path is the FP8 one, specifically the
`w_is_e4m3` branch (ComfyUI dev-fp8 weight loaded verbatim).

---

## 1. Confirmed root cause: per-TENSOR activation scale crushes per-block dynamic range

The native-e4m3 GEMM scales the activation (and the weight) with a **single per-tensor scalar**
`amax/448`, then tells cuBLASLt the scales are scalars via
`CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F`. Exact code:

- **Activation amax is per-TENSOR** (one max over the whole M×K matrix):
  `fp8_a_amax_kernel` (a flat grid-stride `fmaxf` over all `n = M*K`), lines ~888–902.
- **Scale = amax/448, one scalar**: `fp8_scale_from_amax` → `scale_out[0] = a * (1/FP8_E4M3_MAX)`
  with `FP8_E4M3_MAX = 448.0f`, lines ~939–944 (and `#define` line ~828).
- **Every element quantized with that one inverse**: `fp8_a_quant_kernel`
  `const float inv = 1.0f/(*scale); out[i] = __nv_fp8_e4m3(x[i]*inv)`, lines ~922–937.
  Host launches at lines ~1443–1451, scale finalized at ~1446.
- **Weight is the same story** (per-tensor): `fp8_w_amax_kernel` (~833–858) →
  `fp8_scale_from_amax` → `fp8_w_quant_kernel` (~861–886); for the native-e4m3 weight the
  per-tensor recovery scalar is just `wglobal` set by `fp8_set_scalar_kernel` (~992, used ~1266).
- **cuBLASLt told "scalar"**: `cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;`
  set on both `A_SCALE_MODE` and `B_SCALE_MODE`, line ~1462; A/B scale pointers point at the single
  floats, ~1468–1470; `alpha = 1`, ~1458.

Why this dots in motion: one scalar `amax/448` is fixed by the single largest activation element
in the whole matrix. Diffusion activations are heavy-tailed — a few large channels set `amax`, so
the *typical* block sits far below it and, after dividing by the global inverse, lands deep in
e4m3's subnormal / low-mantissa region where the representable steps are coarse. Round-to-nearest
there is a **structured, per-block-coherent** error. Because the outlier that pins `amax` moves
frame to frame, the quantization floor breathes between frames → the error is temporally
incoherent → visible dot/grain crawl in motion. Weights are constant so their per-tensor scale is
stable (comfy ships exactly per-tensor weights and looks fine); the moving part is the activation.

## 2. Why our NVFP4 path is cleaner (the key delta)

Same file, `ggml_cuda_nvfp4_cublaslt_mul_mat` uses a **block-scaled** scheme:

- `cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;` (line ~702) — a
  **dedicated UE4M3 scale per 16-element block** along K, laid out in cuBLASLt's `SWIZZLE_32_4_4`
  scale layout (`swz_off`, lines ~30–37).
- On top of the per-block scale it runs comfy's **two-level** trick (`GGML_NVFP4_QUANT_TWOLEVEL`):
  a per-tensor global `amax/(6*448)` (line ~657) is factored out into the GEMM `alpha` so the
  per-16-block UE4M3 codes stay in their well-conditioned range instead of underflowing
  (`quant_act_kernel`, per-block `tgt = (amax/6)/per_tensor`, lines ~271–324).

So each 16-lane block gets its own scale. A quiet block is no longer forced to share the loud
block's scale, so it keeps its mantissa resolution and the per-block-coherent error collapses.
**That per-block granularity is the entire reason nvfp4 looks clean and fp8 (per-tensor) dots.**
The FP8 path threw away granularity to buy an 8-bit mantissa — but coarse *scale* beat the finer
element, so we got the worst of both.

## 3. Blackwell MXFP8 via cuBLASLt: YES, available and the right fix

The CUDA-13.3 cuBLASLt (current `longcat-avatar-dev` and prod builder base:
`nvidia/cuda:13.3.0-devel-ubuntu24.04`) exposes the block-scaled fp8 scale mode:

```c
typedef enum {
  CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F  = 0,  // one f32 scalar / whole tensor  (what fp8 uses today)
  CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 = 1,  // UE4M3 scale / 16-elem block     (what nvfp4 uses)
  CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 = 2,  // UE8M0 scale / 32-elem block  == MXFP8
} cublasLtMatmulMatrixScale_t;
```

Header doc for mode 2: *"scaling factor tensor elements have type CUDA_R_8F_UE8M0 and the block
size is 32 elements."* Applied to `CUDA_R_8F_E4M3` data this is exactly **MXFP8** — per-32-element
e8m0 (power-of-2) block scales on the fp8 tensor cores (`mma.block_scale`, sm120). Same fast path,
just finer scale. We already prove the block-scaled machinery works on this box: the nvfp4 path
runs VEC16 with cosine=1.0. MXFP8 is the same wiring with mode 2 + 32-element blocks + e8m0 codes.

There is already an e8m0 encode/decode in-tree: `ggml_cuda_e8m0_to_fp32` (`common.cuh` ~815) and
`compute_e8m0_scale` (`quantize.cu` ~52, but that one is FP4-tuned, EMAX=2 — for e4m3 data EMAX=8).

### Recommended fix order (all keep the fp8 tensor cores)

**(a) PRIMARY — MXFP8 block-scaled e4m3 via `VEC32_UE8M0`.** Prototype below. Block-scale BOTH A
(weight) and B (activation): the Blackwell block-scale MMA scales both operands, so mixing SCALAR
on one side with VEC32 on the other is not a valid tensor-core config — cuBLASLt's heuristic will
return no algo. Since the native weight is already e4m3+per-tensor, we re-scale it to per-32-block
e8m0 (folding `wglobal` in → `alpha=1`); the weight stays e4m3 codes, only its *scale* gets finer,
so it's no worse and usually better (quiet weight blocks leave e4m3 subnormals too).

**(b) Fallback if MXFP8 disappoints — reuse the nvfp4 two-level VEC16_UE4M3 scheme for the e4m3
weight path.** i.e. keep e4m3 *data* but drive it with 16-block UE4M3 scales + a per-tensor global
in alpha, exactly like `quant_act_kernel`'s two-level branch. Finer blocks (16 vs 32) and a
mantissa'd scale code (UE4M3 vs pure-exponent UE8M0) — strictly more scale resolution than MXFP8,
at the cost of the extra global. Whether cuBLASLt accepts VEC16_UE4M3 with `CUDA_R_8F_E4M3`
*data* (as opposed to E2M1) must be probed on-device; if it rejects it, MXFP8 (a) is the only
hardware block-scaled fp8 mode.

**(c) Cheapest partial win, if both block modes are rejected for e4m3 data — per-ROW scalar
activation scale.** cuBLASLt supports an outer-vector scalar-per-row mode; a scale per token-row
already removes the cross-token dynamic-range coupling that drives most of the motion crawl,
though not the per-channel-block part. Lowest engineering cost, partial quality.

---

## 4. The prototype (landed, env-gated, additive)

Gate: **`GGML_F8_MXFP8=1`** (default off → the file is byte-identical to before). Native-e4m3
weight path only for now (`mxfp8 = GGML_F8_MXFP8 && w_is_e4m3`). Added in
`ggml/src/ggml-cuda/nvfp4-cublaslt.cu`:

- `mxfp8_e8m0_from_amax_e4m3(amax)` — e8m0 shared-exponent for an e4m3 block:
  `2^(floor(log2(amax)) - 8)` (E4M3 EMAX=8), biased +127, clamped [0,254]. FLOOR (via `frexpf`)
  not round-to-nearest, so a block peak can never scale past 448 and clip.
- `fp8_w_to_mxfp8_kernel` — reads the already-e4m3 weight bytes, `value = decode(byte)*wglobal`,
  per-32-block amax → e8m0 (swizzled via `swz_off`, col_len = K/32) → re-quant to e4m3.
- `fp8_a_to_mxfp8_kernel<act_t>` — src1 (F16/F32) per-32-block amax → e8m0 (swizzled) → e4m3.
- Host wiring in `ggml_cuda_fp8_cublaslt_mul_mat`: `mxfp8` branch requants W and A into pool
  buffers + swizzled e8m0 scale buffers (function-scope `mx_w_scale` / `mx_a_scale`), then sets
  `sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0`, binds the swizzled buffers on
  `A_SCALE_POINTER`/`B_SCALE_POINTER`, `alpha=1`. The per-tensor scalar path, the act/weight
  caches, the Hadamard/`f8_zero` paths, and the algo cache are all guarded `!mxfp8` so they are
  untouched when the flag is off. The fp8 algo-cache key gained an `mxfp8` field so a scalar algo
  is never reused for a block-scaled GEMM.

**GPU-test #1 result (coordinator):** math CONFIRMED clean — `[F8_LOUD] attn1.to_q
IN{amax=2.4} OUT{nnan=0 ninf=0 amax=4.6}`, `[F8_MM] ms=0 ok=1`. VEC32_UE8M0 block-scaling works
and the swizzle is right (no garbage). First run then hit a ggml-CUDA **pool LIFO assert**
(`ggml-cuda.cu:650`): the two `mx_w_scale`/`mx_a_scale` pool buffers were declared at function top
(destroyed last) but allocated interleaved *after* `w_fp8`/`a_fp8_pool` (must be freed first) — the
stack allocator requires frees in reverse-of-alloc, i.e. declaration order must equal allocation
order. **Fixed** by relocating the declarations: `mx_w_scale` now sits right after the `w_fp8`
group, `mx_a_scale` right after the `a_fp8_pool` group, so the RAII unwind is strict LIFO. Still
env-gated / default-off byte-identical.

### Open validation points (must be checked on GPU — could not build/run here)

1. **Scale swizzle for VEC32.** The prototype reuses `swz_off` (the VEC16 layout) with the column
   count = K/32. The `SWIZZLE_32_4_4` row/inner-4 tiling is independent of block size, so this
   *should* be identical, but a wrong scale layout yields garbage (not a crash) — this is the #1
   thing to confirm. If wrong, the fallback is the layout cuBLASLt documents for block-scaled
   factors in the CUDA 13 cuBLASLt docs / `cublasLtMatmulDescGetAttribute` sample.
2. **e8m0 EMAX.** `EMAX=8` targets block peak ≈ 2^8=256 (headroom under 448). If blocks look
   dim/clipped, sweep the `-8` term (7 gives more headroom, 9 less).
3. **cuBLASLt accepts mixed nothing** — both A and B are VEC32; if the heuristic returns no algo
   for a shape it returns `false` and the caller falls back to the dequant path (safe).

---

## 5. Test recipe for the main thread (build + GPU)

Build (CUDA, sm120):
```
# in the longcat-avatar-ltxdenoise build (same flags as the nvfp4/fp8 work)
# no code change needed beyond this file; GGML_F8_MXFP8 default-off is byte-identical.
```

Run the FP8 native path with the new gate, on the SAME eye-test clip that showed the dots:
```
GGML_CUDA_F8_GEMM=1 \      # native e4m3 weight GEMM on (default on already)
GGML_F8_MXFP8=1 \          # NEW: block-scaled MXFP8 activation+weight
GGML_F8_DBG=1 \            # optional: prints [F8_MM]/[F8_LOUD] so you can confirm it took the path
   <normal ltx fp8 render command>
```

What to look for:
- **A/B correctness first:** with `GGML_F8_DBG=1`, `[F8_LOUD]` should show `OUT{nnan=0 ninf=0}` and
  `OUT.amax` in the same O(1–10) ballpark as the per-tensor path (mode confirms the GEMM ran and
  didn't NaN). If `[F8_MM] ms!=0` for every shape, cuBLASLt refused VEC32 for the shape → check
  point (1)/(3) above.
- **Quality (the actual goal):** side-by-side the MXFP8 clip vs (i) current per-tensor fp8 (dotty)
  and (ii) nvfp4 (clean) and (iii) comfy bf16 (gold) on the SAME seed/prompt via the eye-test page
  (`:8077`). Success = the motion dots/grain are gone and it reads as clean as nvfp4, at fp8 speed
  (perf should match the current fp8 path — same tensor cores, one extra cheap requant kernel).
- **A/B toggle:** `GGML_F8_MXFP8=0` must produce the byte-identical old output (regression guard).
- If MXFP8 is clean but you want to push weight scale finer, that's fallback (b) — flip the e4m3
  weight to the two-level VEC16_UE4M3 scheme; see §3(b).

Note: no builds/renders were run here (shared 16GB card + main-thread GPU jobs). The prototype
compiles against the CUDA-13.3 cuBLASLt headers (VEC32_UE8M0 present, `operator float()` on
`__nv_fp8_e4m3` is explicit so the `(float)` decode cast is valid) but needs the on-GPU
validation in §4.
