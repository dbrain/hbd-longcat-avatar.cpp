# HANDOFF — F16 VAE DECODE activation path (Wan2.2 VAE), env-gated

**Status:** implemented, NOT built, NOT committed. Main agent builds (docker cudnn builder) + GPU-validates.
**Gate:** `WAN_VAE_F16` (default OFF → F32 byte-identical). DECODE only; encode untouched.
**Hard dependency:** requires `GGML_CUDNN_CONV3D=1` (or `GGML_CUDNN_CONV=1`). Rationale below.

---

## Goal (recap)
The Wan2.2 VAE weights are already F16 (`longcat-wan-vae-f16.gguf`) but decode runs *activations*
in F32. Running the decode activation stream in F16:
1. **Unlocks 1×1 zero-seam decode**: the temporal-streaming path fits chunk-0 but chunk-1+ does
   the 4× temporal upsample → a ~**14 GB F32** intermediate → OOM. F16 halves it to ~7 GB
   (+~4 GB causal cache ≈ ~11 GB, fits ≤11.5).
2. **General decode speedup**: half the HBM traffic on the conv-boundary activations, the 2×
   spatial upscale, and the norm/silu/copy glue (the cuDNN conv GEMM is already F16 internally,
   so compute is unchanged; the win is the surrounding traffic + no per-conv F32↔F16 re-convert).

---

## F16 / F32-island map (proven from each op's ggml dtype support)

### F16-native — this is the win
| Op (decode) | Why F16 works |
|---|---|
| **CausalConv3d** (conv2, decoder conv1, every residual conv, **upsample3d.time_conv**, head_2) | Routed to cuDNN `GGML_OP_CONV_3D`. cuDNN already runs HALF internally; extended `conv3d-cudnn.cu` to accept F16 src/dst (templated layout transposes) + `ggml_conv_3d_direct` now emits an F16 result for an F16 input + supports_op relaxed. **time_conv + the temporal-doubling cont = the ~14 GB; now F16.** |
| **RMS_norm** | `rms_norm_f32<…,T>` is templated (norm.cu) → F16 ok; the `mul(h, gamma)` is `add/mul` src0=F16, src1=F32-gamma = the **safe** binbcast direction (F16,F32→F16, binbcast.cu:378). |
| **SiLU** | `ggml_cuda_op_unary` asserts F32-or-F16, dispatches half (unary.cu:146-150). |
| **ggml_upscale NEAREST** (2× spatial — a heavy tensor) | upscale was **F32-only** (asserted). Extended: templated `upscale_nearest<T>` + relaxed core/CUDA asserts for NEAREST. Pure gather/copy → no precision change. (bilinear/bicubic stay F32-only; VAE uses NEAREST.) |
| **pad / pad_circular** | pad.cu templated `<T>`, asserts F32-or-F16 (pad.cu:84). |
| **concat** (frame assembly + feat_cache) | concat.cu templated `<T>`, F16 case (concat.cu:260). |
| **residual adds / DupUp3D / cont / permute / reshape / slice / view** | binbcast F16,F16→F16; copies are dtype-generic. |
| **cross-graph causal feat_cache** | persisted as F16 (`persist_feat_map` conts F16); every chunk reloads & re-feeds it F16 → consistent across chunks (byte-copy cache buffer is type-agnostic). |

### F32 islands — unavoidable, justified
| Island | Why it MUST be F32 | Cost |
|---|---|---|
| **AttentionBlock** (decoder middle.1 — whole block) | Its Conv2d (to_qkv/proj) → `ggml_conv_2d` → **im2col asserts src1==F32 (im2col.cu:87)**; `ggml_mul_mat` emits F32; and `add(proj_out, identity)` would otherwise be the **unsafe** `add(F32 src0, F16 src1)` broadcast (binbcast.cu:261 stride assert — the same combo WAN_DIT_F16 excludes VACE for). Cast x→F32 at entry (so identity is F32), run the whole block F32 (= prod numerics exactly), cast result→F16 at exit. | At the **bottleneck** (smallest spatial) → cheap. |
| **resample.1 Conv2d input** (Resample, all up/down-sample modes) | Same `im2col src1==F32` requirement. Cast x→F32 right before `resample_1->forward`; output (mul_mat F32) restored to F16 at the function tail. | A **spatial-stage** F32 temporary of the conv input — *separate stage* from the 14 GB temporal peak, so it does not coincide. (Eliminable later by templating im2col's src1 read on dtype — see Follow-ups.) |
| conv **bias add** | `add(F16 conv-out, F32 bias)` = the **safe** binbcast direction (F16 src0). Not a real island — listed for completeness. | none |
| decode **entry latent / final pixels** | F32↔F16 cast boundaries (latent in, RGB out for host read-back). | 2 casts/graph |

### Precision-risk ops (owner eye-test gates)
- **head_2 (final CausalConv3d → RGB) in F16**: the F16 quantisation step in pixel range (~5e-4 near
  ±1, smaller near 0) is **below 8-bit** (1/255 = 3.9e-3), so F16 banding should be sub-perceptual —
  **but** the RGB output is exactly where it would first show. **Look at:** smooth gradients (skies,
  walls, skin, fades), low-contrast flats. If banding appears, the cheap fix is to keep just head_2's
  output F32 (cast x→F32 before head_2; the 14 GB is upstream so this costs little).
- **RMS_norm in F16**: kernel reduces mean-of-squares; the templated path reads F16 but the math is
  float-accumulated (low risk). Watch flat-area shimmer.
- **latent→F16 entry cast**: latent std≈0.76, within F16 range → negligible.

---

## Why `GGML_CUDNN_CONV3D` is a hard requirement
With cuDNN ON, every CausalConv3d routes to the cuDNN `GGML_OP_CONV_3D` path (now F16-capable).
With cuDNN OFF, CausalConv3d falls to `ggml_conv_3d` → **im2col_3d, which asserts src1==F32
(im2col.cu:548)** → abort on an F16 stream. The validation env sets it; the gate is documented as
requiring it. (If a conv shape exceeds the cuDNN workspace cap, the existing code already aborts —
not F16-specific. Set `GGML_CUDNN_CONV3D_WS_MB` large enough to admit the FAST plan, same as the
F32 path. The cuDNN plan/workspace is **identical** F16 vs F32 — io is HALF either way — so F16
introduces no new workspace pressure.)

---

## Files changed (all gated / default-path byte-identical)

**ggml core**
- `ggml/src/ggml.c`
  - `ggml_conv_3d_direct`: result dtype = `b->type==F16 ? F16 : F32` (was hardcoded F32). F32 input → unchanged.
  - `ggml_interpolate_impl`: allow F16 input for NEAREST mode (was `ASSERT(F32)`).
- `ggml/src/ggml-cuda/conv3d-cudnn.cu`
  - Templated the two NCDHW↔NDHWC-f16 layout transposes on the ggml-side dtype (`ncdhw_to_ndhwc_f16_tiled<Tin>`, `ndhwc_f16_to_ncdhw_tiled<Tout>`), dispatched by `input->type` / `dst->type`. cuDNN io stays HALF.
  - Relaxed the dtype gate: input & dst may each be F16 **or** F32.
- `ggml/src/ggml-cuda/upscale.cu`
  - `upscale_f32`→`upscale_nearest<T>` (+ `upscale_nearest_cuda<T>`), F16/F32 dispatch for NEAREST; bilinear/bicubic keep their F32 assert.
- `ggml/src/ggml-cuda/ggml-cuda.cu`
  - supports_op `GGML_OP_CONV_3D`: allow src[1] (activations) and dst to be F16 or F32 (was F32-only).

**model**
- `src/model/vae/wan_vae.hpp`
  - `wan_vae_f16_enabled()` gate (env `WAN_VAE_F16`).
  - `WanVAE::decode` + `WanVAE::decode_chunk`: cast latent→F16 at entry, decoded pixels→F32 before return.
  - `AttentionBlock::forward`: whole-block F32 island (cast in/out) — driven by `x->type==F16`, so auto-scoped to the F16 decode (encode/default untouched).
  - `Resample::forward`: F16 stream through time_conv/upscale/cont; F32 island around `resample.1`; restore F16 at the tail.

No change to: encode (`encode`/`encode_tail`/`encode_partial`/`encode_temporal_streaming`), the
spatial-tiled encode path, `build_graph_partial` (disabled), or any non-WAN model.

---

## Validation plan (main agent)

**Build:** `docker run --rm --gpus all -v <repo>:/src -v longcat-ccache:/root/.ccache -w /src
longcat-avatar-dev:builder-cudnn bash -lc "cmake --build /src/build -j --target sd-cli"`
(must be the **cudnn** builder image — the F16 conv path is `#ifdef GGML_CUDNN`).

**Run (the owner's #1 — 1×1 zero-seam temporal-streaming decode):**
```
VAE_TILE=1x1 LONGCAT_VAE_TEMPORAL_CHUNK=1 LONGCAT_VAE_ENCODE_REL_TILE=0.5 \
WAN_VAE_F16=1 GGML_CUDNN_CONV3D=1 GGML_CUDNN_CONV=1 DITF16=1 MAXV=10.0 \
./run_wan22_i2v_nvfp4.sh        # FR=25
```

**Pass criteria**
1. **No OOM.** The chunk-1+ temporal-upsample buffer ~14 GB → ~7 GB; peak ≤ 11.5 GB. Watch the
   per-chunk allocator high-water (the chunk-1 decode is where it previously died).
2. **Numerics:** decoded std ≈ 0.76, **nnan = 0** (the F16 path is range-safe; if nnan>0, suspect
   the head_2/RMS island — flip head_2 to F32 per the precision note).
3. **A/B baseline:** same command with `WAN_VAE_F16=0` (and a larger `MAXV` if it OOMs at 1×1, or
   the spatial-tiled decode) → compare the decoded video. Expect **no seams** (1×1) and no new
   banding vs the F32 decode.
4. **Eye-test (owner gates):** F16 banding/precision — focus on smooth gradients / skin / skies /
   low-contrast flats (head_2 is the first place F16 would show). Motion/temporal continuity across
   chunk boundaries should be unchanged (the causal feat_cache is F16 but consistent).

**Expected decode-time delta:** a modest speedup (conv-boundary + upscale + glue traffic halved;
cuDNN GEMM unchanged) — not a 2×. Report decode s/clip vs the F32 baseline.

**Default-path regression check:** one run with `WAN_VAE_F16` UNSET must be byte-identical to
pre-change (all edits are gated on the env or on `x->type==F16`, which only occurs under the env).

---

## Follow-ups (not done — would deepen the win)
- **Template im2col src1 on dtype** (im2col.cu:87/548) → removes the two F32 islands (resample.1 and
  the im2col_3d fallback), letting the spatial conv input stay F16 and dropping the cuDNN hard-dep.
- **Keep head_2 F32** if eye-test shows RGB banding (cheap; 14 GB is upstream).
- The non-streaming `decode()` also honors the gate (payoff #2 without 1×1), but builds all frames in
  one graph → higher peak; the streaming path is the OOM-relevant one.
