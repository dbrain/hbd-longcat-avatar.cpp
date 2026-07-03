# HANDOFF — Wan2.2 DiT: F16 residual stream + FUSE flags (LTX levers ported)

Goal: bring LTX-video's proven DiT glue/cast collapse to the Wan2.2 DiT. The nsys
profile of the cuDNN-attn Wan DiT shows ~16%+ in glue/copy/cast on top of cuDNN
flash attn (20.9%): `k_bin_bcast<op_add>` 5.6% (5754 calls — the residual adds),
`cpy_perm_transpose` 3.1%, `concat_cont` 2.4%, `k_bin_bcast<op_mul>` 2.0% (AdaLN
modulate), `convert_unary<half,float>` 1.6% (casts), `rms_norm` ~3.3%. The F16
residual stream + the fusion kernels collapse exactly that wall.

**Bottom line: the ggml-cuda side was already fully built for `LTX_DIT_F16` and
needs NO change. The only missing piece was the Wan *model* wiring (3 casts) +
the run-script forwarding. Implemented below, default-OFF / F32 byte-identical.**

---

## Per-flag status & proven root cause

Run env in question (all forwarded by `run_wan22_i2v_nvfp4.sh`):
`GGML_CUDA_F16_BCAST_FUSE`, `GGML_CUDA_BIAS_GELU_FUSE`, `GGML_CUDA_BIAS_RMS_FUSE`,
`GGML_CUDA_RMS_MOD_FUSE`, `LTX_DIT_F16`(→ Wan: `WAN_DIT_F16`), `GGML_CUDNN_ATTN_F16_OUT`.

### 1. `GGML_CUDA_BIAS_GELU_FUSE` — already supported, fires on Wan as-is
Generic graph fusion `ggml-cuda.cu:4141`: matches `ADD(matmul_out, bias)` + `GELU`
(contiguous, single-use, bias 1-D F32 over ne0). Wan FFN emits exactly this:
`ffn_0->forward` (Linear+bias) → `ggml_ext_gelu` (wan.hpp:472-473). No assert; the
detector simply returns 0 when it doesn't match. Works at F32 today; under the F16
stream `node->type` is F16 (allowed, line 4147) and the bias stays F32. **Supported,
no change.**

### 2. `GGML_CUDA_BIAS_RMS_FUSE` — already supported, fires on Wan as-is
Generic fusion `ggml-cuda.cu:4189`: folds `ADD(matmul,bias)` + `RMS_NORM` (the
pre-bias). Wan's qk-norm is `q_proj`(Linear+bias)→`norm_q`(RMSNorm) (wan.hpp:135-138)
— matches. Guarded; no assert when unmatched. **Supported, no change.**

### 3. `GGML_CUDA_RMS_MOD_FUSE` — structurally N/A to Wan's main modulate (not an error)
This is a *model-side* op (`ggml_rms_modulate`, ltxv.hpp:65-86) emitted only by LTX
(`modulate_fused`), which fuses `rms_norm(x)*(1+scale)+shift`. **Wan's norm1/norm2 are
`LayerNorm(dim,eps,false)` (no affine) → `ggml_norm`, NOT RMSNorm** (wan.hpp:411,424;
LayerNorm class ggml_extend.hpp:5192). So there is no rms+modulate pattern to fuse in
Wan, and Wan never calls `ggml_rms_modulate`. The env is read only in ltxv.hpp:68 — on
a Wan run it is simply inert. **Not applicable; harmless. No error.** (Wan's
modulate instead folds via flag #1's sibling, the F16/madd bcast fusion — see #4.)

### 4. `GGML_CUDA_F16_BCAST_FUSE` — already supported; fires on Wan **once the stream is F16**
Generic fusion `ggml-cuda.cu:4768`. Two relevant patterns, both present in Wan:
- AdaLN `x + x*scale + shift` (self-attn/ffn pre-modulate): `modulate_mul(y,es[1])`
  then `ggml_add(y, ·)` then `modulate_add(y,es[0])` (wan.hpp:456-457, 469-470) =
  `MUL(y,scale)→ADD(y,mul)→ADD(_,shift)`. The detector's flux2-AdaLN branch
  (lines 4817-4847) matches it (gate/shift are `[dim,1,1]` broadcast F32).
- gate-add `x + y*gate`: `ggml_add(x, modulate_mul(y,es[2]))` (wan.hpp:460,476) — the
  bcast MUL+ADD branch.
At **F32** these already fuse via the same-shape/`F16_BCAST` paths (no change vs today).
The `f16_bcast_fuse` env only *additionally* allows the big operand to be **F16**
(line 4778) — i.e. the win only materializes when the residual stream is F16. With an
F32 stream the flag is a no-op on these ops. **Supported; needs the F16 stream (#5) to pay off.**

### 5. `LTX_DIT_F16` → **Wan: `WAN_DIT_F16`** — was NOT wired for Wan (the real gap)
`LTX_DIT_F16` is read only in `ltxv.hpp:1738`; the Wan DiT had no equivalent, so the
residual stream stayed F32 and none of the F16 fusions/cast-savings could engage.
**This is the missing implementation, now added** (`WAN_DIT_F16`, see below). All the
ggml-cuda support it needs already exists (built for LTX):
- NVFP4 Linear F16-dst: `ggml_ext_linear` mm_dst gate (ggml_extend.hpp:1149), cuBLASLt
  FP4 GEMM takes F16 src1 + F16 dst (nvfp4-cublaslt.cu:473-475,650), supports_op
  (ggml-cuda.cu:6046). FP8-FFN GEMM likewise (nvfp4-cublaslt.cu:884-885,958).
- `ggml_norm` F16 (norm.cu:671), `rms_norm` F16 (templated T), `gelu` F16.
- binbcast F16/F32 combos: F16,F16→F16 and F16,F32→F16 (binbcast.cu:376-379) — so
  `x_F16 + mod_F32` and `LayerNorm·weight_F32` work.
- rms_norm+MUL fusion correctly **declines** to fuse when x=F16 but weight=F32
  (`ggml_cuda_can_fuse` type check, ggml-cuda.cu:3981-3984) → falls back to unfused
  F16 rms + F16×F32 binbcast mul. No assert.

### 6. `GGML_CUDNN_ATTN_F16_OUT` — was the assert; root cause proven, fixed by #5
Generic in `Rope::attention`'s wrapper (ggml_extend.hpp:1584-1594): for a maskless
attn with `d_head ∈ {64,128}` it retypes the cuDNN flash output F32→F16 (recomputing
nb[]). Wan self-attn is d_head = dim/num_heads = 5120/40 = **128**, maskless → it fires.

**The actual failure (verified, NOT "F32 modulate"):** with an **F32** residual stream,
the F16 attn output flows through `o_proj` (NVFP4+F16 → F16, ggml_extend.hpp:1149) and
the AdaLN gate, then hits the residual `x = ggml_add(x_F32, gate·attn_F16)`
(wan.hpp:460). `ggml_add` takes its result type from src0 → dst is **F32**, src0 F32,
src1 **F16**. The CUDA binbcast dispatcher (`ggml_cuda_op_bin_bcast`, binbcast.cu:374)
sees `src0==F32 && dst==F32` and picks the `<float,float,float>` instantiation —
**reading the F16 src1 as float**. `launch_bin_bcast_pack` then asserts
`nb10 % sizeof(src1_t)==0` → F16 src1 has `nb10=2`, `sizeof(float)=4`, `2 % 4 ≠ 0` →
**`GGML_ASSERT` fails at binbcast.cu:261**. (binbcast.cu:261 *is* the trip point the
memory note cited, but the cause is the missing F32+F16→F32 add combo, surfaced by an
F16 attn output added into an F32 stream — not a modulate dtype.)

**Fix = make the stream F16 (#5):** then `x` is F16, the add is `F16 + F16 → F16`
(binbcast.cu:376, supported), and F16_OUT composes cleanly. Hence `DITF16=1` in the run
script sets `GGML_CUDNN_ATTN_F16_OUT=1` for you; standalone F16_OUT remains unsafe and
must not be used without `WAN_DIT_F16`.

---

## Implementation (committed to working tree; not committed to git)

All self-gated, default-OFF, F32 byte-identical when `WAN_DIT_F16` is unset.

### `src/model/diffusion/wan.hpp`
1. **`WanModel::forward_orig`** — after patch-embed + permute (the residual `x` is now
   `[N, tokens, dim]`), cast `x` to F16 when `WAN_DIT_F16` is set **and**
   `config.vace_layers == 0`. `e0`/`context`/`pe` stay F32. Cast `x` back to F32 right
   before `head->forward` so the head/unpatchify/VAE tail is unchanged. The
   `vace_layers==0` guard excludes the VACE path, whose `ggml_add(c_F32, x_orig)` would
   otherwise mix F32 src0 + F16 src1 and hit the same binbcast.cu:261 assert (prod i2v/t2v
   is vace_layers==0; VACE+F16 is out of scope/untested).
2. **`WanSelfAttention::forward`** — under the F16 stream q/k/v Linears emit F16, but the
   fast fused RoPE (`ggml_rope_pe`) is F32-only (rope.hpp:953) and an F16 q would fall to
   the slow cont+repeat+mul+add chain. Upcast q/k → F32 before the reshape/RoPE
   (self-gated on `q->type==F16`, so the default F32 path is a no-op). v stays F16 (it is
   re-cast to F16 inside `ggml_ext_attention_ext`'s build_kqv). This mirrors LTX's
   "apply_hidden_rope q/k F32-cast". Cross-attention is untouched: its q (F16) goes
   straight to flash with no RoPE; k/v come from the F32 context and are cast to F16
   inside the wrapper — all consistent, output F16 under F16_OUT.

### `run_wan22_i2v_nvfp4.sh`
- New `DITF16=1` toggle → adds `WAN_DIT_F16=1 GGML_CUDNN_ATTN_F16_OUT=1`.
- Parent-shell env forward regex extended to also pass `WAN_*` (was `GGML_|LONGCAT_|LTX_`).
- Stale "F16_OUT trips binbcast … needs integration fix" comment corrected to the real
  root cause + that it's now safe under the F16 stream.

### Why the stream stays F16 without per-op plumbing
Every block op already has F16 support, and every residual `ggml_add` has the F16 `x`
as src0 so dst stays F16: norm1/norm2 `ggml_norm`(F16); modulate MUL/ADD (F16,F32→F16);
self_attn → o_proj F16 (or F32 without F16_OUT — still adds into F16 x fine); cross_attn
→ F16/F32; ffn_0/ffn_2 NVFP4 F16-dst, gelu F16. RMSNorm weight-mul declines to fuse and
runs as F16×F32 binbcast. No op needed a new dtype branch.

---

## Validation plan (main agent — build + GPU)

Build: `docker run --rm --gpus all -v <repo>:/src -v longcat-ccache:/root/.ccache -w /src
longcat-avatar-dev:builder-cudnn bash -lc "cmake --build /src/build -j --target sd-cli"`

Runs (GPU device=1 baked into the script; pause for owner between renders per project policy):
1. **Baseline (unchanged)**: `TAG=nvfp4_base ./run_wan22_i2v_nvfp4.sh` — confirms the F32
   path is byte-identical to before (no `WAN_DIT_F16`, FUSE stack on as today).
2. **F16 stream**: `TAG=nvfp4_ditf16 DITF16=1 ./run_wan22_i2v_nvfp4.sh` — the lever. This
   sets `WAN_DIT_F16=1 GGML_CUDNN_ATTN_F16_OUT=1` on top of the FUSE stack.
3. If #2 asserts/NaNs, isolate by layering: `DITF16=1 FP8_OFF=1` (drop FP8-FFN),
   then `DITF16=1 FUSE_OFF=1` (drop fusions — pure stream), then `DITF16=1 CUDNN_OFF=1`.
   This pinpoints whether any interaction (FP8-FFN F16 dst, a specific fusion) misbehaves.

Numerical gate (F16 is non-deterministic run-to-run — cuDNN/cuBLASLt algo pick — so the
bar is "close", NOT bit-exact): final latent std ≈ 0.76, `nnan=0` (use the existing
NaN/range scans). Eye-test the clip vs the `nvfp4_base` mp4 for motion/identity. Owner
judges quality (per project policy — don't declare a winner).

nsys expectations under DITF16=1 (re-profile to confirm the lever landed):
- `k_bin_bcast<op_add>` (5754-call residual-add bucket) and `k_bin_bcast<op_mul>`
  (AdaLN modulate) shrink hard — folded into `fused_madd_same` / `mul_add_bcast` and
  halved width (F16). `convert_unary<half,float>` casts (the cuDNN-out upcast) largely
  disappear (F16_OUT keeps it F16). Residual `cpy`/`concat` traffic ~halves (F16 width).
- cuDNN flash attn time ~unchanged (already F16 internally); q/k now carry one F32-cast
  each (small, expected — LTX pays the same).
- DiT compute should drop on the order of the glue+cast share (~10-16% of DiT) minus the
  q/k cast tax; peak VRAM should ease slightly (F16 residual tensors). Watch peak vs the
  `--max-vram 14` budget.

Risk points to watch (all gated; flagged honestly):
- FP8-FFN (`GGML_FP8_FFN=1`, default on) with an F16 activation+dst: the cuBLASLt FP8
  path *does* accept F16 (nvfp4-cublaslt.cu:884-885,958), but it's the least-exercised
  combo — `FP8_OFF=1` isolates it if the FFN looks wrong.
- Offload / cross-segment graph-cut now snapshots an F16 `x` at `wan.prelude`/`wan.blocks.*`.
  Bytes-only, should be fine, but confirm multi-segment continuity isn't disturbed.
- VACE is explicitly excluded (vace_layers==0 guard); do not expect F16 on VACE runs.
