# LongCat-Avatar.cpp — PERF HANDOFF (start here)

*Self-contained. You do NOT need to read PORT-PROGRESS.md (the "neverending story") to start —
everything you need for the next levers is here. Cross-reference PERF.md lap-23 + lap-21 §7 only for
detail. Written end of lap-23, 2026-05-27.*

---

## Tone / mandate (read this, it changes how you work)

**The #1 rule: never declare something "not a lever" / "at the floor" / "GPU-bound" without 100% proof.**
This entire VAE phase (−39% and counting) came off the back of a confident, WRONG verdict —
*"the DiT/VAE is all at the roofline, it's GPU-bound, nothing left."* It wasn't: im2col was at **7%** of
memory bandwidth (a kernel idling on int64 division), then 14% (uncoalesced reads), now CONT is a 3.4×
coalescing gap hiding behind "it's just a copy." Every "we're at the floor" claim in this codebase has,
so far, been disprovable with a microbenchmark. So:

1. **Prove the floor, don't assert it.** "It's compute-bound" needs a roofline number; "the kernel is
   optimal" needs a bandwidth/occupancy measurement; "won't help" needs an A/B. The wins hid exactly
   where someone said "won't help." You don't have to rewrite ggml (the owner says: ideally don't) —
   but you DO have to *measure before concluding*. Forking a kernel, adding a CUDA op, restructuring the
   model graph are all fair game when the measurement justifies them.
2. **Measure, don't predict.** Quote measured numbers, never forecasts (fastdiv was predicted ~5×,
   measured ~2× — we said so, and that miss is what exposed the next bottleneck). Label every
   measurement by step count (`--steps 1` vs `8`). Profilers serialize → absolute ms inflate, proportions
   are exact; say which you're quoting.
3. **Bit-exact or it doesn't ship.** Output is user-approved, must stay byte-identical unless the owner
   explicitly OKs a quality trade. Gate EVERY change (see Gates). The owner cross-checks with a fresh
   agent + codex — be rigorous and honest, and re-derive inherited claims rather than trusting them
   (the old "98% roofline everywhere" was wrong: flash 63%, im2col 7%).
4. **Don't bank early to be safe.** If a lever has measured headroom, push it to the floor. But when the
   *profile* says a lever has gone small (it moves as you win), say so and pivot — don't grind a 2% knob
   while a 40% one waits. Re-profile after each win; the bottleneck relocates.

ONE GPU (RTX 3060, sm_86, 12 GB). Stop prod acestep/tts/llama before heavy runs. Watch for orphaned
`longcat-avatar-iter` containers.

---

## What this is

From-scratch C++/ggml port of LongCat-Video-Avatar 1.5 (13.6B audio-driven video DiT + Wan VAE), runs
image+audio→talking-video on the 3060. Functionally complete, user-approved quality. We're in the
perf-optimization phase. Repo: `~/dev/longcat-avatar.cpp`, branch `longcat-avatar-port`. ggml is a
submodule (`ggml/`, branch `longcat-avatar-port`, fork of leejet/ggml) — commit kernel changes IN the
submodule, then bump the parent pointer.

**Single-clip wall (25f/8-step) ≈ 181 s:** DiT sampling ~140 s (77%), VAE decode ~17.7 s (~10%),
text/audio encode ~16.5 s (9%). **Use case is LONG CHAINED clips → VAE runs once PER SEGMENT, so VAE
wins compound ×N.** That's why we finish the VAE floor before the DiT.

---

## State: what just shipped (lap-24, committed, bit-exact)

CONT pixel-shuffle + DiT-attn permute copies coalesced (ggml `740156c2`, `cpy.cu`). The
slow strided `cpy_scalar` backing `ggml_cont(permute(...))` is replaced, for F32
cont-to-contiguous, by two kernels: **`cpy_perm_transpose`** (dim0 strided + one unit-stride
axis f → smem-tiled batched 2D transpose, 53–122 → 328–331 GB/s) and **`cpy_perm_coalesced`**
(dim0 unit-stride both sides, higher dims permuted → batch-index decomposed once per block,
kills the per-element int64 divide, 147–221 → 315–468 GB/s). Bit-exact by construction:
`LONGCAT_CONT_VERIFY` byte-compares vs scalar (6617 conts, 0 mismatch) + --steps-1/8 PSNR
99.00. **VAE decode 17.70→15.10 s (−14.7%); DiT sampling 139.9→137.5 s (−1.7%); cont-op
−62%.** CONT moved #1→#5 of the VAE graph. Env: `LONGCAT_CONT_NOTRANSP`, `LONGCAT_CONT_VERIFY`.

**⭐ THE BOTTLENECK MOVED AGAIN — VAE is now MUL_MAT 26.4% (cuBLAS conv-GEMM floor) /
IM2COL_3D 21.8% / CONCAT 11.4% / IM2COL-2D 11.3% / CONT 8.9%. → DO LEVER 2 NEXT** (conv-3d-direct
WMMA fuses away both IM2COLs = 33% + the materialize round-trip). CONCAT at 11.4% is newly the
#3 op and was never investigated — worth a CONT_PROF-style probe if lever 2 stalls.

## (lap-23, prior) what shipped

`im2col_3d` smem-halo **tiled kernel + half2 vectorized F16 store** (`ggml/src/ggml-cuda/im2col.cu`,
ggml `9966677a` + parent `03ba865`). The Wan-VAE convs are stride=1/dil=1/pad=0 (CausalConv3d pre-pads
externally) ⇒ every tap in-bounds, no boundary checks. A block stages the reused 3×3×3×ICh input window
in shared memory once (coalesced), threads write columns from smem (input re-read ~1.4× halo vs the old
~10× cache-line amplification); `half2` packs two F16 column entries into one 32-bit store (per-warp
write 64B→128B).

- **im2col_3d 15367→3701 ms (−76%, 4.15×); VAE decode 29.2→17.7 s (−39%); big conv call 50→239 GB/s
  (66% of peak).** Default tile `LONGCAT_IM2COL_TILE=8,8,8,4`. fastdiv kernel retained as universal
  fallback (`im2col_3d_tiled_try` returns false on non-unit stride/dil, non-zero pad, smem>48 KB).
- **DEAD-END, do not retry:** `VEC=8` (uint4 128-bit store) measured a wash vs half2 (smaller blockDim.x
  + bigger smem cancel the wider store).

---

## ⭐ THE BOTTLENECK MOVED — re-profiled VAE decode

`LONGCAT_OP_PROFILE` (serializes every op → absolute ms inflated but **proportions exact**; here
serialized total 17.2 s ≈ real 17.7 s, so proportions ≈ real wall):

**lap-24 (post-CONT-fix) VAE profile** (`LONGCAT_OP_PROFILE`, per VAE tile graph, % of tile):

| op | % | what it is |
|---|---|---|
| **MUL_MAT** | 26.4% | the conv GEMM — cuBLAS F16 tensor-core ceiling, **immovable as a GEMM** (but lever 2 fuses im2col INTO it) |
| **IM2COL_3D** | 21.8% | lap-23 territory — **lever 2 target** |
| CONCAT | 11.4% | newly the #3 op (was 9.6%); never probed |
| IM2COL (2D) | 11.3% | **lever 2 target** |
| CONT | 8.9% | was #1 ~22.8% pre-lap-24; now PERMT/COAL fast paths |
| PAD/RMS_NORM/ADD/MUL/UNARY | ~19% | |

(Pre-lap-24 CONT was ~4.0 s / 22.8% / #1. The DiT *sampling* step is a separate graph:
MUL_MAT 45% + FLASH_ATTN_EXT 33% — that's lever 3.)

---

## Lever 1 — CONT pixel-shuffle ✅ DONE (lap-24, see "State" above). Kept below for reference.

**Root cause is nailed** (via the committed `LONGCAT_CONT_PROF` probe in `ggml/src/ggml-cuda/cpy.cu`,
ggml `8e34af2`): the big high-res conts — `ne=[96,256,256,4]`, `[256,256,2,96]`, `[192,128,128,4]`,
`[384,64,64,2]` — run at **~95 GB/s vs ~328 GB/s for a contiguous memcpy (3.4× gap)**. They're
uncoalesced: the source's contiguous axis is dim1 (`nb01 == elsize`) but `ggml_cuda_cpy` writes
dim0-contiguous, so in `cpy_scalar` consecutive threads read `nb00` (~262144 elems) apart. The fast
smem-tiled `cpy_scalar_transpose` kernel EXISTS but `can_be_transposed` is too narrow — it requires
`ne[3]==1` AND a contiguous batch stride, and these are genuine 4D permutes (own dim2/dim3 strides), so
they fall through to the slow strided scalar path.

**Source of the conts:** Wan VAE `DupUp3D` / `Resample` / `patchify`/`unpatchify` in `src/wan.hpp`
(~L273–329, ~L975–1024) — each does 3–4 `ggml_cont(ggml_ext_torch_permute(...))` depth↔space chains;
~6503 strided cont calls per decode.

**Fix direction (your call which):**
- **(a) Generalize the tiled transpose-copy** — a smem-tiled `cpy` that coalesces whenever the source's
  contiguous axis ≠ dst dim0, batched over the higher dims with their real strides; broaden the
  `can_be_transposed` dispatch to route the 4D pixel-shuffle conts to it. Target ~95→~280 GB/s.
  Shared model-wide (DiT too) so it's bit-exact-or-bust. *Same playbook as the im2col win.*
- **(b) Custom fused depth↔space op** collapsing each 3–4-cont chain into ONE pass (1 read + 1 write
  instead of 3–4 round-trips). More work, bigger ceiling. Parked behind (a) unless (a) underdelivers.
- **(c)** Or something cleverer — restructure the model graph to avoid the shuffles, do them in F16,
  whatever measures best. Nothing sacred.

Use `LONGCAT_CONT_PROF` to confirm which shapes flip to the fast path and the new GB/s.

---

## Lever 2 — conv-3d-direct (Tier-2, bigger, ~15–20% VAE)

Fuse im2col INTO the conv matmul so the 27× expansion is never materialized (kills IM2COL_3D 18.4% +
IM2COL-2D 9.6% + the materialize→reload round-trip).

**Groundwork is already here:** `GGML_OP_CONV_3D` + `ggml_conv_3d_direct` exist in this ggml
(`ggml/src/ggml.c:4848`, CPU impl only); `ggml/src/ggml-cuda/conv2d.cu` is a CUDA direct-conv template.
So this is "write the CUDA kernel for the existing op + switch `CausalConv3d` (`src/wan.hpp`) to
`ggml_conv_3d_direct`" — **no new op to invent.**

**⚠ CRITICAL (this is why a naive port backfires):** `conv2d.cu` is a one-thread-per-output FMA loop on
**CUDA cores**. Porting it to 3D fuses away the ~5 s of im2col BUT runs the ~4 s of MUL_MAT compute
(currently at the cuBLAS F16 **tensor-core** ceiling) on CUDA cores → net LOSS. **It MUST be a WMMA
tensor-core implicit-GEMM.** Reference the owner's WMMA kernel `~/dev/qwen3-tts.cpp/ggml/src/ggml-cuda/
conv-1d-direct.cu` (computes the B-tile input index on-the-fly inside the GEMM load — exactly our case
in 1D). Its tiles (BM/BN/BK) are tuned for 1D vocoder shapes; **RETUNE for the VAE** (K = 27·IC ≈ 2592,
small spatial tiles, tiny batch). Write it FRESH against the current leejet/ggml base — the owner has
de-scoped the old "wait for a merged ggml base" concern: take from the reference if useful, else fresh.
**Prototype + measure ONE VAE conv shape before promising a number.**

**Real VAE conv GEMM dims (measured lap-24, `LONGCAT_IM2COL_PROF`, 25f decode — design the
WMMA tiling against THESE, biggest first):** the post-im2col matmul is `[M = OD·OH·OW] × [K =
27·IC] × [N = OC]` — **huge M, moderate K, small N** (tall-skinny):

| conv (dominant first) | im2col ms (25f) | M = OD·OH·OW | K = 27·IC | N = OC |
|---|---|---|---|---|
| IC=96, OD=4, OH=OW=256 | **1908** (51% of im2col) | 262144 | 2592 | ~96–192 |
| IC=192, OD=4, OH=OW=128 | 796 | 65536 | 5184 | ~192 |
| IC=384, OD=2, OH=OW=64 | 179 | 8192 | 10368 | ~384 |
| IC=384, OD=1, OH=OW=32 | 65 (560 calls) | 1024 | 10368 | ~384 |

⇒ **tile M generously (it's 8K–262K), loop K (2592–10368), N needs only 1–3 tiles of 64–128.**
The top two convs are 70% of im2col_3d; nail IC=96/OH=256 first. Prototype that one shape's
WMMA implicit-GEMM, microbench vs (im2col_3d + cuBLAS) on it, THEN promise a number.

---

## Lever 3 — DiT / sampling (AFTER the VAE, the real fish: 77% of clip wall)

lap-21 §7 pinned matmuls at ~93% cuBLAS and flash-attn at ~63% (fork-class). The owner's standing
instruction: **"roofline-bound ≠ no lever in the *pathway*."** Don't accept "it's GPU-bound" — re-derive
it. Look at op COUNT, fusion opportunities, redundant CONT/copies (the DiT step's CONT was 4.3% = 734 ms
over 1543 calls — a lever the same shape as VAE lever 1), precision, anything mis-dispatched. Re-profile
the DiT step graph fresh (`LONGCAT_OP_PROFILE`, the block with FLASH_ATTN_EXT present, ~10775 nodes:
MUL_MAT 45% / FLASH 33% / ADD 5.7% / CONT 4.3% / MUL 3.6% / SCALE 2.6% / NORM 1.5% / ROPE_PE 1.2% / …)
before claiming a target. Flash-attn is the single biggest under-saturated hot op (63% of roof) — an
FA2/FA3-style kernel for d=128/L=10920 on sm_86 is fork-class but the owner is up for it ("rewrite ggml").

---

## Method / environment

**Build:** `~/dev/kobbler/docker/longcat-avatar-dev/iter.sh build` (CUDA builder image, sm_86, ccache;
host has no CUDA, binaries run from the image). `iter.sh shell` for an interactive build/run shell.

**Run / serve:** `iter.sh cli -- <args>` one-shot; `ITER_PORT=8095 iter.sh serve` runs the GPU-polite
`avatar_server.py` (spawns sd-cli per request, GPU→~0 between renders). Clip viewer for eyeballing:
`python3 tools/serve_clips.py --dir models --port 8011` (host stdlib, no deps) → http://10.0.0.208:8011/.

**Standard render (hold constant):**
```
-M vid_gen -m models/longcat-avatar-1.5-dit-dmd-q4_k.gguf --t5xxl models/longcat-umt5-xxl-q8_0.gguf \
  --vae models/longcat-wan-vae-f16.gguf --audio-vae models/longcat-whisper-v3-encoder-f16.gguf \
  --init-img models/_testinputs/girl_480x832.png --audio models/_testinputs/speech_16k.wav \
  -p "a person talking" --cfg-scale 1.0 --video-frames 25 -W 480 -H 832 \
  --steps 8 --diffusion-fa --seed 42 --clip-on-cpu --max-vram 9
```
Use `--steps 1` for kernel sweeps — **VAE decode time is step-count-independent**, so a 1-step render
gives the full VAE op profile in ~60 s instead of ~180 s.

**Profilers (all env-gated; serialize the stream so absolute ms inflate but proportions/GB-s are exact):**
- `LONGCAT_IM2COL_PROF` — per-call im2col_3d shape + write BW (+ TILED/fastd path tag).
- `LONGCAT_CONT_PROF` — per-call cpy/cont path (memcpy/transpose/SCALAR) + BW. **Use for lever 1.**
- `LONGCAT_OP_PROFILE` — per-op-type breakdown per graph >1000 nodes (VAE tiles + DiT step).
- `tools/roofline_dit.cpp` — isolated matmul/flash microbench (build recipe in PERF.md lap-21 §7).

**GATES (mandatory, every change):**
- Bit-exact data-movement ops: `LONGCAT_IM2COL_VERIFY`-style byte-compare (re-run the old kernel into a
  temp buffer + `memcmp`) is the strongest, GPU-cheap, no render needed.
- Render gate: `--steps 1` with the change vs baseline (`LONGCAT_IM2COL_NOTILE` etc. force the old path),
  then `python3 tools/clip_compare.py <baseline.webm> <new.webm>` → must read **PSNR 99.00 dB / min 99.00**
  (clip_compare's bit-identical cap). Host has numpy+PIL+ffmpeg; the builder image does NOT (run
  clip_compare on the host).
- Coherence (for any non-bit-exact change the owner OKs): `clip_compare.py <clip.webm>` single-arg →
  ac16 should stay 0.83–0.84 flat across all frames (collapse/drift toward the end = the "watercolour
  melt" failure the owner has been burned by — check the LAST frames specifically).

**Quant/quality ladder, offload, chaining, the gallocr view-output fix** — all in PERF.md (laps 13–20)
if you need them; not relevant to the perf levers above.

---

## Knobs added this phase (all env-gated, default-safe)
`LONGCAT_IM2COL_TILE="CB,TOH,TOW,P"` (default 8,8,8,4) · `LONGCAT_IM2COL_NOTILE` · `LONGCAT_IM2COL_NOVEC2`
· `LONGCAT_IM2COL_VERIFY` · `LONGCAT_IM2COL_PROF` · `LONGCAT_CONT_PROF` · `LONGCAT_OP_PROFILE`
· `LONGCAT_CONT_NOTRANSP` (force old strided cpy_scalar for cont — lap-24 A/B) · `LONGCAT_CONT_VERIFY`
(byte-compare PERMT/COAL vs scalar reference).
