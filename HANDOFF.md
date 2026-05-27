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
IM2COL_3D 21.8% / CONCAT 11.4% / IM2COL-2D 11.3% / CONT 8.9%. **lap-25 UPDATE: lever 2 retiled +
fastdiv → 5.0× over the prototype but still 1.7× slower than im2col+cuBLAS (see Lever 2 below); cheap
pokes exhausted, remaining gap is fork-class GEMM tiling + halo gather → PIVOTED TO LEVER 3 (DiT, 77%
of clip wall).** CONCAT at 11.4% is still the unprobed #3 VAE op — worth a CONT_PROF-style probe if
DiT stalls and you come back to the VAE.

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

## Lever 2 — conv-3d-direct ⚠️ 7.1× FASTER THAN PROTOTYPE, 1.19× SLOWER THAN cuBLAS — occupancy-bound floor (lap-25)

**Status: correct, default-OFF behind `LONGCAT_CONV3D_DIRECT`. F16-accum is a precision trade
(owner sign-off pending). Decision: keep grinding only if the gather-occupancy wall (below) breaks.**
ggml head (lap-25 chain). **VAE decode 25f --steps 1: prototype 128.22 → 17.93 s (7.1×); baseline
im2col+cuBLAS 15.06 s ⇒ 1.19× slower.** 8-step real-quality A/B: 39.27 dB mean / 33.21 min, ac16
identical to baseline (0.832/0.834), no end-melt. Full clip +2.8 s (175.2 vs 172.3) — VAE is ~10% of
wall, **compounds ×N on chained clips**.

### The lever chain (all measured, all committed, all gated)
| step | VAE s | what / why |
|---|---|---|
| lap-24 prototype | 128.22 | B regathered ceil(oc/16)=6–24× (1D-ref grid copied with BM=16) |
| **retile** | 34.4 | one block covers MSUB oc-subtiles ⇒ B gathered once/patch-tile (the big one, 3.7×) |
| fastdiv | 27.2 | `fast_div_modulo` kills 6 raw int-divs/elem (lap-23 lesson) |
| hoist + warps | 25.8 | patch-decomp out of BK loop; tune |
| **NSUB=2** | 23.96 | each warp owns 2 patch-cols ⇒ weight frag reused 2× (−16% GEMM) |
| **F16 accum + split-K** | 19.5 | ⭐ the unlock — see ncu below |
| incremental k-decode | 18.39 | decompose k_outer once, increment (kx→ky→kz→ic) — no per-elem fastdiv |
| bounds-skip | **17.93** | s=1/d=1/p=0 ⇒ taps always in-bounds, skip 6 compares/elem |

Store bug found+fixed early: direct `store_matrix_sync` to a large-stride **global** dst gives WRONG
output (8.88 dB) — must stage through c_smem. Knobs: `LONGCAT_CONV3D_DIRECT` · `_MSUB` · `_NSUB` ·
`_KSPLIT` · `_BFILL` (timing probe) · build `-DC3_ACC_F32` to revert F16 accum.

### ⭐ ROOT CAUSE — ncu ground truth (NOT inference). This answers "why is it slower."
ncu on `conv3d_mma_kernel` (the inferred bfill-partition story was *wrong* — verify with the profiler):
- **DRAM 1.3–4.7%, tensor-core 8–18% — BOTH idle.** Not memory-bound, not compute-bound.
- **warps_active 16.7% → 43–47%** (after F16 accum). **register-limited**: 125 → 70–72 regs/thread ⇒
  4 → 7 blocks/SM. Stalls: `wait` + `long_scoreboard` (memory *latency*) + `barrier` — textbook
  under-occupancy. The kernel is **occupancy/latency-bound**, full stop.
- **F16 WMMA accumulators** (`c3_acc_t`) were the unlock: halved accumulator regs (8→4/frag),
  125→70 regs, occupancy 16.7%→43%, 23.96→19.5 s. **split-K** (`LONGCAT_CONV3D_KSPLIT`, auto) adds
  blocks because VAE tiling makes per-conv grids tiny (~50 blocks « 28 SMs); partials atomic-add into
  a pre-zeroed F32 dst. Together they doubled occupancy and throughput.

### Why it CANNOT cheaply reach parity (the fundamental wall — proven, not asserted)
**`LONGCAT_CONV3D_BFILL` (gather neutered) = 13.59 s — already BEATS baseline's 15.06 s.** So the
GEMM + store + all other VAE ops are *competitive*; the **entire** 1.19× gap is the ~4.3 s gather, and
the gather is **occupancy-bound, not work-bound** (DRAM idle, reads are ~95% L2 hits). The fused kernel
must hold the GEMM accumulators **live across the whole K-reduction including every gather phase**,
pinning ~72 regs/thread throughout → 7 blocks/SM → 47% occupancy → the gather's L2 latency + barriers
can't be hidden. The two-kernel im2col+cuBLAS baseline sidesteps this: im2col carries **no
accumulators** (≈100% occupancy, saturates), cuBLAS runs its GEMM separately. **Neither carries both
burdens; the fused kernel does — that's the structural cost of fusion on sm_86.**

**smem-halo gather (the obvious "reduce the 27× reads" lever) is disproven by arithmetic:** the kernel
is register-limited to 7 blocks/SM using only ~11 KB smem (headroom to ~14 KB). Adding a halo buffer
(~8–16 KB) pushes smem to ~20–27 KB/block ⇒ **smem caps occupancy to ~4 blocks/SM** — it trades the
gather latency for *less* occupancy, the already-binding constraint. Net-negative. (And the reads are
L2-bound, not DRAM-bound, so there's no bandwidth to recover.)

**What's NOT yet tried (the only candidates to actually break parity):** (a) a fundamentally different
tiling that doesn't hold accumulators across the gather (≈ splitting into two kernels = im2col+cuBLAS,
i.e. give up on fusion); (b) `cp.async` / pipelined global→smem to hide the gather latency *without*
extra registers (Ampere `__pipeline_memcpy_async` — does not consume the register file like a
software-pipelined double-buffer did; **this is the one lever with a real shot** and wasn't tried);
(c) larger VAE tiles so grids aren't starved (but split-K already covers most of that). Double-buffer
(register-pipelined), launch_bounds spilling, oc-split-across-warps, k-offset-smem all measured WORSE.

**ROI:** VAE ~10% of single-clip wall; whole lever-2 ceiling ~3.9 s/clip (compounds ×N chained). At
1.19× the kernel is a net LOSS per clip today (+2.8 s) — **do NOT flip the default ON** until it beats
baseline. Next real shot is `cp.async` (b); if that doesn't cross parity, the fused approach is
occupancy-bottlenecked on sm_86 and the honest call is to keep im2col+cuBLAS for the VAE.

### Reference scope (kept from original handoff)

The original Tier-2 framing (~15–20% VAE) below assumed the fusion would land near cuBLAS; the
prototype shows that's gated on real GEMM tiling, not just "fuse im2col in." Original notes:

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

**EXACT implicit-GEMM mapping (reverse-engineered lap-24 from `ggml_compute_forward_conv_3d_impl`,
`ops.cpp:6839`, + `ggml_conv_3d_direct`, `ggml.c:4848`).** It is structurally **identical to the
1D reference** (`~/dev/qwen3-tts.cpp/.../conv-1d-direct.cu`) — adapt that kernel, don't reinvent:
- **C[oc, patch] = Σ_K A[oc,K]·B[K,patch]** (matrix_a row-major, matrix_b col-major, F32 accum).
- **A = weight**, `ne=[KW,KH,KD, c·oc]`, contiguous ⇒ flat `A[oc,K] = w[(oc*c+ic)*KDKHKW + (kz*KH*KW+ky*KW+kx)]`
  i.e. K-index `= ic*KDKHKW + kz*KH*KW + ky*KW + kx`, and `ne[3]` packs as `j = oc*c + ic`. So A is
  already `[oc, K]` row-major contiguous — pass `w` straight in (F16, like the 1D ref's `w`).
- **N-dim "patch"** = `OD·OH·OW` per batch. n_local→ `od = n/(OH*OW); oh = (n%(OH*OW))/OW; ow = n%OW`.
- **B-gather** (replaces im2col temp): `sx = ow*s0+kx*d0-p0`, `sy = oh*s1+ky*d1-p1`, `sz = od*s2+kz*d2-p2`;
  `src[ sx*nb0 + sy*nb1 + sz*nb2 + (batch*c+ic)*nb3 ]` (0 if OOB). **Wan VAE: s=1,d=1,p=0 ⇒ all in-bounds**
  (CausalConv3d pre-pads externally — same fact lap-23 used for im2col).
- **C-store**: dst `ne=[OW,OH,OD, oc·n]`, contiguous ⇒ `dst[ patch_in_batch + (batch*oc + oc_idx)*OW*OH*OD ]`.
- grid `(ceil(patch/BN), ceil(oc/BM), n)`; ref's BM=16/BN=64/BK=16/4-warp tiling is a starting point —
  **retune** (oc is the small dim → maybe BM=16 fine; patch huge → BN can grow). No CUDA CONV_3D
  dispatch exists yet — add it to `ggml-cuda.cu` (CPU-only today). Wire `CausalConv3d` (`wan.hpp`) to
  `ggml_conv_3d_direct` behind an env flag for A/B. Precision: F16×F16→F32 accum like the current
  im2col+cuBLAS path ⇒ expect coherent but maybe NOT 99 dB bit-exact (lap-11-class accum change);
  gate with ac16 coherence + owner OK if PSNR <99.

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
· `LONGCAT_CONV3D_DIRECT` (lap-25: route 3D convs through the fused WMMA kernel — default-OFF, still
1.7× slower than im2col+cuBLAS) · `LONGCAT_CONV3D_MSUB=N` (force oc-subtiles/block for tiling sweeps)
· `LONGCAT_CONV3D_BFILL` (timing probe: constant-fill B to isolate GEMM from gather — WRONG output).
