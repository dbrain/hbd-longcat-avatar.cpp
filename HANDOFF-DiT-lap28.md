# LongCat-Avatar.cpp — DiT PERF HANDOFF (lap-28, the levers that are STILL left)

*Written end of lap-27 (2026-05-28). Lap-27 shipped TWO bit-exact perf wins +
one offload-correctness P0 fix that unblocks prod. Read this doc first; it
supersedes HANDOFF-DiT-lap27.md and PERF-DiT-lap26.md (kept for archaeology).*

---

## ⏱️ FIRST FIVE MINUTES — eye-test server should already be up

```
curl -sI http://10.0.0.208:8011/    # serve_clips.py should answer HTTP 200
# if dead:
cd ~/dev/longcat-avatar.cpp && nohup python3 tools/serve_clips.py --dir build --port 8011 \
  > /tmp/serve_clips.log 2>&1 & disown
```

http://10.0.0.208:8011/ is the **quality checkpoint**, NOT a stop-and-wait —
render an A/B clip after each win; the owner looks when convenient. Do not
pause the campaign waiting for eyeballs.

---

## 🔥 MOOD / MANDATE — read TWICE, do not skip

**NO QUITTING. NOTHING IS "TOO HARD."** Fork-class means hands dirty — NOT
"too hard." Forking a kernel, adding a ggml op, rewriting the allocator, even
porting a custom flash kernel: all fair game. **The owner is a relentless
steerer who treats "we're at the floor / too hard / good enough / I'm out of
ideas" as a CHALLENGE.** They will keep pushing. Preempt it: just do the hard
thing. If something isn't working and you can't explain why — KEEP PUSHING
UNTIL YOU CAN. The user is happy with multi-hour debug arcs as long as you
land on root cause (or document why the question itself was wrong).

**Every "floor" claim in this codebase has been DISPROVABLE with a profiler:**
- "im2col is at roofline" → was 7% of BW (lap-21)
- "VAE conv is GEMM-bound" → was 16.7% occupancy (lap-25)
- "MUL_MAT is floored" → was doing 7× redundant cond work (lap-26)
- **"reset_segment_runtime_tensors orphans the cache, ggml-alloc surgery"** → was
  actually the planner skipping non-subcut'd ops; one-line subcut fix (lap-27)

"Compute-bound" / "at floor" / "ggml-alloc bug" is the START of the
investigation, not the verdict. **Measure (ncu + bit-exact A/B + GPU readback
of actual buffer contents), never predict.** Wall-clock guesses lied all
through laps 21-27. Quote numbers with step-count and PSNR.

**Gates (mandatory, every change):**
- Bit-exact: `tools/clip_compare.py <base> <new>` reads **PSNR 99.00**.
- Quality trade (needs owner OK): ac16 0.83–0.84 flat across ALL frames incl.
  the LAST (the "watercolour melt" failure mode) + 2 seeds.
- The benchmark is **480×832, 25 frames, --steps 8** — wins MUST show there.
  320p is a fast draft only (untrained res); won't catch real regressions.

**ONE GPU** (RTX 3060 / sm_86 / 12 GB). Stop prod acestep / tts / llama before
heavy runs. `docker rm -f longcat-avatar-iter` strays. Never two GPU jobs at
once.

---

## What's DONE (lap-26 + lap-27)

| commit | what | wall delta |
|---|---|---|
| `03d477d` (lap-26) | cond-K/V cache (write+consume) | -8.1% resident sampling, PSNR 99 |
| `0248212` (lap-27.1) | fused LayerNorm+MUL+ADD across RESHAPE views | -1.13% resident sampling, PSNR 99 |
| `36f52e0` | default-on cond-cache (env: `LONGCAT_NO_COND_CACHE=1` to opt out) | (config) |
| `3782728` | `GGMLRunner::persistent_tensors_` registry | (infra) |
| `6cdb645` | FinalLayer modulate fusion (hygiene) | <0.05% |
| `f0b4bcc` (lap-27 P0) | **cond-cache + offload bit-exact** (subcut cpy writes) | unlocks **-8.8% offload**, PSNR 99 |

**Stacked totals @ 480/8-step:**
- Resident: 138 s no-cache → **124.77 s** = **-9.6%**
- Offload:  153.83 s no-cache → **140.25 s** = **-8.8%** (prod path)

Cond-cache is **default-on for BOTH paths**. No env knobs needed for prod.

---

## The lap-27 offload bug — root cause, for future "why does this keep happening"

The lap-26 doc + handoff-lap27 framed this as "ggml-alloc cross-buffer view
routing" / "bind_segment_cached_inputs mis-binds persistent leaves." That
framing was **wrong** — the actual bug was much simpler.

**Root cause:** `ggml_graph_cut.cpp:540-562` builds segments only around
subcut-marked tensors (`mark_graph_cut`) + the final output. The cond-cache's
`ggml_cpy` write nodes were never subcut'd and aren't on the residual's
ancestor chain, so they belonged to **no segment** → never executed →
`condkv_buf` stayed zero-initialized → consume reads zeros → PSNR collapses
to 12 dB. Resident mode runs the whole cgraph monolithically so the writes
executed fine; offload's segmented path skipped them entirely.

**Fix (4 lines):** in `self_attn`'s persist branch, mark each cpy write with
the SAME `post_self_attn` group name as the block's existing residual subcut.
The planner groups by name → writes become additional outputs of an existing
segment, no extra compute (the cast/scale chain shares ancestors with the
attn path).

**How it was caught:** GPU readback of `condkv_k[0]` first 8 halfs after step 0:
- Resident: `9557 977f 1b55 1817 a462 9c8f 971a 2253` (real data, FNV `f6b303741dda8361`)
- Offload:  `0000 0000 0000 0000 0000 0000 0000 0000` (zeros)

That readback was the moment of certainty. Before that, three rounds of
"surely the persistent_tensors_ fix does it" were wrong. **When a fix sounds
correct but the symptom persists: don't trust the theory, read the actual
bytes.**

**General lesson recorded:** if you add a side-effect op (cpy, set_rows,
custom write) to the graph that's not on the output's dependency chain,
**subcut it** with a known group name. Otherwise the offload planner drops it
silently.

---

## The levers that are LEFT (ranked, with effort + ROI honesty)

### 1. **Re-profile FIRST** (always, then this list is stale)

```bash
/tmp/render_bench.sh /src/build/_prof.webm "LONGCAT_OP_PROFILE=1 LONGCAT_COND_CACHE=1" --steps 2
```

Re-read the consume-step proportions. **The bottleneck moved 4× during the
VAE campaign and once more in lap-27** — don't trust this list until
re-measured.

### 2. **Audio modulate fusion** — small but it's RIGHT there, ~20 min

In lap-27.1, of the ~145 candidate modulate sites per consume step, **97
fused, ~47 didn't** (op-profile: NORM 242 → 145 calls; if all sites fused
we'd be ~98). The 47 unfired sites are the audio modulate (`modulate(ctx,
mod_norm_attn, ao, am[0], am[1], T_noise)` at `longcat_avatar.hpp:675`).

**Debug breadcrumb:** the audio path's topo pattern (from
`LONGCAT_DEBUG_NORM_FUSE=1`) is `NORM, MUL, ADD, MUL_MAT, ADD` — note **no
RESHAPE between NORM and MUL**, unlike the msa/mlp pattern `NORM, RESHAPE,
MUL, VIEW, CONT, ...`. The fusion check at `ggml-cuda.cu:4202` should fire on
the `NORM, MUL, ADD` adjacency directly (zero view-ops to walk past) — but
empirically doesn't.

Best guesses (PROVE one then act):
- `ggml_node_get_use_count` >1 on the audio NORM or MUL (some downstream
  capture I'm missing — the cast/cont chain after `ggml_ext_chunk` on `am[]`)
- shape check in `ggml_can_fuse_ext` rejects (ao's shape post-`audio_cross_attn`
  vs scale1's broadcast shape) — but msa/mlp uses the same shapes and fires
- the SCALE_BIAS for am[1] isn't being pre-expanded (the modulate's
  `ctx->gf` pre-expand fires per-call; it SHOULD work for audio too — verify)

Run `LONGCAT_DEBUG_NORM_FUSE=1` + dump segment-graph indices for the audio
NORM node and trace why my custom check rejects. Expected win **~0.3% wall**.

### 3. **gate_add MUL+ADD fusion** — ~1h, ~0.5% wall

`gate_add(x, y, gate)` at `longcat_avatar.hpp:159` is:
```
y = RESHAPE_4d(y); y = MUL(y, gate); y = RESHAPE_3d(y); return ADD(x, y);
```

2 per block × 48 = **96 chains per consume step**, currently unfused. Each
bounces a [10920×4096] F32 tensor (~178 MB) through HBM twice. Fusing the
`{MUL, ADD}` pair (skip the RESHAPE, same trick as lap-27.1) saves ~178 MB
per chain × 96 = **17 GB/step** = ~50 ms at 360 GB/s × 7 consume steps =
~350 ms = **~0.3% wall** (revise estimate via measurement).

**Recipe:**
1. New kernel in `ggml/src/ggml-cuda/binbcast.cu` or new file:
   `mul_add_bcast_f32(x, y, gate, dst, ne, strides, gate_broadcast_packed)`.
   Per element: `dst[i] = __fadd_rn(x[i], __fmul_rn(y[i], gate[broadcast(i)]))`.
   Force IEEE single-rounded ops (not FMA) — `-use_fast_math` collapses
   `mul+add` → `fma` which is NOT bit-exact vs the unfused chain (see lap-27.1's
   `__fmul_rn` / `__fadd_rn` trick in `norm.cu`).
2. Host wrapper `ggml_cuda_op_mul_add_bcast` in same file. Use ADD's tensor
   for output strides (post-RESHAPE 3D shape; same contiguous bytes).
3. Autofusion entry in `ggml-cuda.cu` mirroring the lap-27.1 norm-fuse:
   walk past RESHAPE between MUL and ADD via `trace_back`; verify ADD.src
   chains to MUL; use_count checks (==1) on MUL and any intermediate
   RESHAPE node; ggml_are_same_shape(mul_n, add_n).
4. **Bit-exact gate**: PSNR 99.00 vs lap27_base. If FMA contraction leaks
   (PSNR < 99), force `__fmul_rn` + `__fadd_rn`.

### 4. **F16-prescaled K/V plumbing for cond-cache consume** — ~1h, ~0.5%

In consume's `self_attn` (`longcat_avatar.hpp:282-291`), cached F16 cond k/v
get round-tripped through F32:
```
F16(stored) → ggml_cast→F32 → ggml_ext_scale(*256) → ggml_concat F32 →
  (inside wrapper) ggml_ext_scale(*kv_scale) → ggml_cast→F16 → flash
```
The `×256 then ×kv_scale=1/256` cancels exactly; same with `F16→F32→F16`.
Pure HBM waste — per consume self-attn: ~300 MB redundant HBM × 48 blocks ×
7 consume steps = ~100 GB total ≈ 280 ms wall.

**Recipe:**
1. Add `bool kv_prescaled_f16 = false` arg to `ggml_ext_attention_ext` in
   `ggml_extend.hpp:1288`. Default false → all existing callers unchanged.
2. Inside `build_kqv` (line ~1336): if `kv_prescaled_f16`, skip the `kv_scale`
   `ggml_ext_scale` AND the `ggml_cast(F16)` (k_in/v_in already F16, already
   prescaled). Keep the softmax `scale / kv_scale` and output `*1/kv_scale`
   passthrough (those are correct regardless).
3. In `self_attn`'s consume branch (line 282-291):
   - Drop the `ggml_cast(F32) + ggml_ext_scale(1/kv_scale)` round-trip on
     `k_cond`/`v_cond` — pass the stored F16 directly.
   - Pre-scale + cast the noise `k_rope`/`v` to F16(×kv_scale) BEFORE concat.
   - `ggml_concat` two F16 tensors → F16 k_full / v_full.
   - Pass `kv_prescaled_f16=true` to `ggml_ext_attention_ext`.

**Watch:** the wrapper's V handling does a permute+cont (line 1345-1346)
that must still work on F16. The flash kernel's existing F16 cast inside is
the no-op cast(F16→F16) we want to skip via the flag.

**Bit-exact gate**: PSNR 99 (the F16(k*kv_scale) round-trips bit-identical;
that's why kv_scale=1/256 was chosen — exact F16 exponent shift).

### 5. **Audio cross-attn KV caching** — measure FIRST, ~maybe 0.3%

Per `additional-levers.md`, text+audio cross-attn `kv_linear` re-projects
step-invariant K/V every step × 48 blocks. Audio kv_linear has M=32, text M=512
— SMALL. Previously estimated <1% wall. **Measure with op-profile before
building**. If `kv_linear` MUL_MATs at M≤512 total < 2% of consume → SKIP.
The audio_proj output windowing is already cross-step-invariant (already
hoisted out of the loop per `additional-levers.md`). Only the per-block
`audio_cross_attn.kv_linear` is still in-loop.

### 6. **Fused adaLN-modulate kernel (proper, not autofusion)** — ~half a day

The lap-27.1 autofusion catches NORM+MUL+ADD but the FULL modulate is:
`silu(t) → Linear(adaLN) → reshape+chunk → scale_bias → norm(x) → reshape+mul+add+reshape`.
The first half (silu/adaLN/chunk) is a SHARED prelude across all 6 chunks +
3 modulates per block. There's no autofusion for the silu→Linear chain.
Worth ~maybe 1% with custom kernel; not bit-exact-trivial because Linear
goes through MMQ. Lower priority than 2-4.

### 7. **FLASH FA2/3-class kernel surgery** — FORK-CLASS, multi-week, ~3-5% uncertain

The handoff #1 lever. lap-27 ncu teardown of the consume-step
`flash_attn_ext_f16<128,128,64,1,0,0>` MMA kernel:
- Compute (SM) Throughput: 31.7% (tensor pipe 68% idle)
- L1TEX stall: ~50% of avg 14.5-cycle CPI (smem→register ldmatrix dependency)
- Occupancy: 16.66% (theoretical=achieved, 2 blocks/SM capped by BOTH registers AND smem)
- Grid 4704, 128 threads/block
- Tile sweep proved `ncols1=64` optimal (smaller = more occupancy but loses K/V reuse → monotonic loss)

ggml's MMA kernel is already FA2-class (cp.async double-buffer, Q-in-reg).
Beating it needs:
- **Warp specialization** (one warp dedicated to K/V loads, others compute) —
  tighter cp.async/compute overlap than current double-buffer can give.
  Implementation requires `__nanosleep` + manual mbarrier on sm_86 (no
  TMA / async-bulk-tensor like Hopper).
- **OR per-head Q-batch persistence**: each block handles N Q-tiles instead
  of 1, amortizing K/V loads across them. Cuts K HBM traffic ÷N. Cost: N×
  Q/VKQ_C register state → drops to 1 block/SM. Tradeoff measurement TBD.
- **OR `nstages=3` triple-buffer**: more cp.async in flight → less long_scoreboard.
  Cost: more smem → drops occupancy. Likely net-neutral but worth measuring.

**Recipe to NOT skip:**
1. Build `tools/roofline_dit.cpp` (already exists; shape needs update for
   asymmetric Q=9360/K=10920 consume case).
2. Microbench JUST the flash kernel at the consume shape.
3. Pick ONE structural change (start with `nstages=3` since it's smallest).
4. Build, microbench, compare to ggml's baseline.
5. If gain > 0 with PSNR99 → integrate. Else: ncu-teardown to find why.

**Owner-explicit license: "fork-class means hands dirty, NOT too hard."**
You may write your own kernel and replace ggml's MMA. Don't declare floored
without an ncu stall-reason teardown of YOUR kernel vs ggml's, side by side.

### 8. **MUL_MAT lever** — proven floor for now, ONE remaining angle to test

MUL_MAT = 44.7% of consume = HUGE. lap-26 verdict: Q4_K MMQ is the Ampere
floor (Q3_K +7% slower, prec flag ignored). **Untested:** what if the
**activation** path is the bottleneck, not the weight read? The MMQ kernel
quantizes activations to Q8_1 each call. If the activation quant kernel
(`quantize_mmq_q8_1`) is suboptimal at M≈9360, there's headroom.

Per lap-26 ncu: `quantize_mmq_q8_1` is healthy (93% DRAM, 82% occupancy,
~0.7 ms). So this is probably nothing. But it's the only un-investigated
piece of MUL_MAT.

### 9. **Block-sparse attention (BSA)** — OWNER MUST OK, ~12.5% sparsity, large win uncertain

Per `additional-levers.md` IN #1, LongCat's reference disables BSA for the
avatar (off-distribution) but the paper's BSA mask keeps ~12.5% of attention
blocks. A windowed-static BSA (each cube attends neighbours + first/ref
frame) gets the algorithmic win without the dynamic top-k overhead. **Quality
trade, not bit-exact** — needs owner OK + the ac16 + last-frame coherence gate.

If you pursue it: static block mask (cube `[4,4,8]=128 tokens`) at FORWARD
time only. Reuses ggml flash_attn_ext mask path. Estimated wall: attention
33-36% × 8× FLOP reduction × kernel-efficiency-loss factor → realistic 10-15%
wall IF the dense kernel adapts. Bigger than #7 IF it works at quality. 

### 10. **DMD step count 4≈8** — OWNER REJECTED, do NOT re-pursue

The model is DMD-distilled at 8 steps. Reference default is 8. Owner has
explicitly rejected the 8→4 reduction as a sampling-loop knob. Do not raise.

### 11. **The dead-ends so you don't re-burn them**

From PERF-DiT-lap26.md + lap-21..25 + lap-27 findings — these are PROVEN
dead, do not re-attempt without a new measurement that contradicts:

- **MUL_MAT precision pf32 lever**: MMQ dispatch ignores `dst->op_params[0]`
  prec flag (`ggml-cuda.cu:2551-2607`). Q4_K always uses MMQ. dropping pf32
  saves nothing. (lap-26)
- **CUDA graphs**: ~5 ms launch-overhead amortization, but step is compute-
  bound → no benefit. (lap pre-26)
- **FP8 K/V storage**: sm_86 has no FP8 compute; bandwidth-only trick;
  consume isn't VRAM-bound. (lap pre-26)
- **Q3_K weights**: +7% SLOWER than Q4_K. (turboquant)
- **Flash `nbatch_fa` smaller tile sweep**: monotonic LOSS (proved bit-exact
  via `LONGCAT_FA_NCOLS1`). Bigger tile wins; ggml's auto-pick is optimal.
  (lap-26)
- **conv-3d-direct VAE kernel (lap-25)**: 1.19× SLOWER than im2col+cuBLAS;
  ncu-proven occupancy wall; cp.async/smem-halo all dead. KEEP im2col+cuBLAS.
- **FP32 LayerNorm → FP16-with-FP32-accum**: blocked by bit-exact mandate;
  norms are only ~1.5% of step.
- **Native m16n8k8 D=72 ViT FA kernel (lap-12 parked)**: ~12-20h CUDA surgery
  for ~2% wall. ROI awful. Park unless ANOTHER non-MMA-friendly head_dim ViT
  shows up to amortize the work. (lap-11)
- **Step count 4≈8**: owner rejected (above).

---

## Method (reproducibly)

**Build:** `~/dev/kobbler/docker/longcat-avatar-dev/iter.sh build`
(~30 s incremental in builder image, sm_86, ccache; host has no CUDA).

**Render (standard bench):** the `/tmp/render_bench.sh` helper this lap left
behind exists; or run sd-cli directly:
```
docker run --rm --gpus all -v /home/dbrain/dev/longcat-avatar.cpp:/src \
  -v /home/dbrain/dev/longcat-avatar.cpp/models:/models -w /src \
  longcat-avatar-dev:builder /src/build/bin/sd-cli \
    -M vid_gen -m /models/longcat-avatar-1.5-dit-dmd-q4_k.gguf \
    --t5xxl /models/longcat-umt5-xxl-q8_0.gguf \
    --vae /models/longcat-wan-vae-f16.gguf \
    --audio-vae /models/longcat-whisper-v3-encoder-f16.gguf \
    --init-img /models/_testinputs/girl_480x832.png \
    --audio /models/_testinputs/speech_16k.wav \
    -p "a person talking" --cfg-scale 1.0 --video-frames 25 -W 480 -H 832 \
    --diffusion-fa --seed 42 --clip-on-cpu --max-vram 9 --steps 8 \
    -o /src/build/<NAME>.webm
```
For prod-equivalent offload: add `--offload-to-cpu`.
For env knobs: add `-e KEY=VAL` to the docker run.

**Profilers (env-gated, default-off):**
- `LONGCAT_OP_PROFILE=1` — per-op-type wall breakdown per graph
- `LONGCAT_DEBUG_NORM_FUSE=1` — trace the {NORM, MUL, ADD} autofusion misses
- `LONGCAT_FA_NCOLS1=N` — flash MMA query-tile width override (bit-exact diag)
- `tools/roofline_dit.cpp` — isolated matmul/flash microbench (recipe in PERF.md
  lap-21 §7; shape needs update for asymmetric Q=9360/K=10920)
- ncu: container has `/usr/local/cuda/bin/ncu`. Add `--cap-add=SYS_ADMIN`
  (RmProfilingAdminOnly=1). Example: `--kernel-name regex:flash_attn_ext_f16
  --launch-skip 144 --launch-count 3 --section SpeedOfLight,WarpStateStats,
  ComputeWorkloadAnalysis,MemoryWorkloadAnalysis,Occupancy,LaunchStats`.

**GPU readback (the move that broke open the lap-27 offload bug):** in
`build_graph()`, with `LONGCAT_DEBUG_OFFLOAD=1`, the avatar runner already
logs condkv_buf identity + first-8-halfs + full FNV-1a64 of `condkv_k[0]`
each step. When in doubt about whether a buffer holds what you THINK it
holds: dump the bytes. The hash mismatch (resident `f6b303741dda8361` vs
offload zeros) was the closure on a multi-hour debug arc.

**Bit-exact gate:** `python3 tools/clip_compare.py <ref.webm> <new.webm>`
on host (builder image lacks numpy). Read PSNR 99.00 mean+min.

---

## Build status (lap-27 close)

- `feature/lap27` branch state: all 4 wins committed (top 4 commits in `git log`).
- ggml submodule `src/ggml-cuda/{norm.cu,norm.cuh,ggml-cuda.cu}` carries the
  norm-fusion kernel + the `LONGCAT_DEBUG_NORM_FUSE` env diag.
- Parent: `src/ggml_extend.hpp` carries `GGMLRunner::persistent_tensors_`
  registry; `src/longcat_avatar.hpp` carries the post-self-attn cpy subcut +
  modulate scale_bias pre-expand + cond-cache default-on logic.
- Resident bit-exact PSNR 99.00 vs lap-26 baseline. Offload bit-exact PSNR 99.00
  vs resident.

---

## ONE-LINER reminders the next agent will need

- Eye-test: http://10.0.0.208:8011/
- Standard bench: 480×832, 25f, --steps 8 (320p only for fast draft; not a gate)
- Owner is a relentless steerer. **No quitting.**
- Measure, do not predict. The op-profile is serialized so absolute ms are
  inflated; the PROPORTIONS are exact.
- Resident path's monolithic gallocr behaves DIFFERENTLY from offload's
  per-segment gallocr. The lap-27 offload bug recurs every time something
  side-effect-y joins the graph without a subcut. Watch this.
- "I can't explain why" + matching symptom on multiple tries = read the actual
  bytes (GPU readback). Don't trust the theory; trust the readback.
- Commit each win. Submodule-first, then bump parent. Bit-exact PSNR 99.00
  every time, OR get owner OK on the quality trade.
