# LongCat-Avatar.cpp — DiT PERF HANDOFF (lap-30, the levers that are STILL left)

*Written end of lap-29 (2026-05-28). Lap-29 shipped **ONE** bit-exact perf win
(**-8.5% wall, 155.11 → 142.03s mean (3 runs) @ 480/25f/--steps 8 RESIDENT**) by
bumping the FA MMA kernel's launch_bounds occupancy from 2 to 3 — a single-integer
change in `ggml/src/ggml-cuda/fattn-mma-f16.cuh`. Five other experiments were
attempted and all dead-ended (MMQ occupancy bump regressed +27%, Q_in_reg=false
nuked the win, audio cross-attn KV cache too small to measure + fails bit-exact at
F16, smaller FA ncols1 is 10× slower, occupancy=4 spills). Read this doc first;
it supersedes HANDOFF-DiT-lap29.md (kept for archaeology).*

---

## ⏱️ FIRST FIVE MINUTES — eye-test server should already be up

```
curl -sI http://10.0.0.208:8011/    # serve_clips.py should answer HTTP 200
# if dead:
cd ~/dev/longcat-avatar.cpp && nohup python3 tools/serve_clips.py --dir build --port 8011 \
  > /tmp/serve_clips.log 2>&1 & disown
```

http://10.0.0.208:8011/ — `lap29_occ3_final.webm` is the lap-29.1 reference clip
(142.15s, PSNR 99.00 vs lap-27 baseline). Owner reviews when convenient.

---

## 🔥 MOOD / MANDATE (unchanged from lap-29)

*"Go mental — get it faster than it should ever be on this hardware."* Owner is
explicit: *"custom kernel for wins is our bag."* Don't write off a lever because
"the kernel needs surgery" — that IS the work.

**Every "floor" claim disprovable with a profiler.** New evidence this lap: the
FA kernel was sold (HANDOFF-DiT-lap28 lever #7) as needing fork-class FA2/FA3
surgery — actual win came from a **one-integer launch_bounds change**. The lap-28
ncu numbers (31.7% SOL, 50%-CPI L1TEX stalls, 16.66% occupancy) all named the
exact same disease: not enough concurrent warps. The fix was telling nvcc to fit
3 blocks/SM instead of 2. Verify the cheap thing FIRST before scoping kernel forks.

**Gates (mandatory, every change):**
- Bit-exact: `tools/clip_compare.py <base> <new>` reads **PSNR 99.00**.
- Quality trade (needs owner OK): ac16 0.83–0.84 flat across all frames + last + 2 seeds.
- Standard bench: **480×832, 25 frames, --steps 8, RESIDENT, `--max-vram 9`** — wins MUST show there.

**ONE GPU** (RTX 3060 / sm_86 / 12 GB). Standard bench under `--max-vram 9`. Stop
prod acestep / tts / llama before heavy runs. `docker rm -f longcat-avatar-iter`
strays. Never two GPU jobs at once.

---

## What lap-29 SHIPPED

| commit | tag | what | wall (resident, 480/25f/--steps 8) |
|---|---|---|---|
| `72747ac` (parent) / `90670f7a` (ggml) | `kobbler-lap29.1-fa-occupancy-3-2026-05-28` | FA MMA `__launch_bounds__` occupancy 2→3 for DKQ=DV=128 ncols=64 (avatar's consume self-attn shape). nvcc fits 3 blocks/SM with zero register spills — pure free latency hiding for the L1TEX-stall-bound kernel. | 155.11s → **142.03s** mean of 3 runs = **−8.5%** (sampling 121.04 → 107.65s = −11.1%); PSNR 99.00 vs lap-27 reference, all 25 frames |

**Stacked cumulative @ 480/25f/--steps 8 resident:** 159.03s (lap-27 baseline) →
**142.03s** (lap-29.1) = **−10.7%** over 5 ship laps (28.1–28.5 + 29.1).

The 142.03s number is the "current shipped" baseline for lap-30 deltas. Current
shipped HEAD is `72747ac` (parent) bumping ggml to `90670f7a`, clip is
`build/lap29_occ3_final.webm`. PSNR 99.00 vs lap-27 baseline maintained.

**Production-quality clips in `build/`:**
- `lap28_lap27baseline.webm` — same-conditions lap-27 reference (159.03s, **the A/B reference**)
- `lap28_scale_cast.webm` — lap-28.5 (155.16s)
- `lap29_baseline.webm` — re-confirm of lap-28.5 baseline on lap-29 HEAD (155.11s)
- `lap29_occ3.webm`, `lap29_occ3_run2.webm`, `lap29_occ3_final.webm` — lap-29.1 (141.93 / 142.00 / 142.15s)

---

## What FAILED in lap-29 (write up so it isn't re-burned)

### ✗ MMQ launch_bounds occupancy 1→2 — **+27% regression**

Same lever as lap-29.1 applied to the MMQ kernel (`mmq.cuh:3537` Volta+ branch,
currently at `__launch_bounds__(nwarps*warp_size, 1)`). nvcc compiled cleanly,
no spill warnings, but bench: **142.15s → 180.81s = +27.2% regression**.
Reverted (no commit).

MMQ kernel is fundamentally different from FA: it's smem-heavy (X tile + Y tile
dequant buffers), so packing 2 blocks/SM either forces a smaller per-block tile
(less efficient) or hits smem-bank-conflict contention. The FA win generalized
poorly. **Rule: launch_bounds occupancy bumps work for L1TEX-stall-bound kernels
(FA), NOT for smem-throughput-bound kernels (MMQ). Profile before bumping.**

### ✗ FA Q_in_reg=true → false — **kills the lap-29.1 win**

With Q_in_reg=false the kernel re-loads Q from smem every K iter. Bench: **142.15s
→ 155.62s = back to pre-lap-29.1**. The two are linked: Q-in-registers is what
makes the 3-block-per-SM compile fit AND what makes the per-iter work small enough
that the higher occupancy is a net win. Reverted (no commit).

### ✗ FA `LONGCAT_FA_NCOLS1=32` — **10× regression**

Reducing the per-block Q-tile width from 64 to 32 doubles the number of FA blocks
launched (each handling half the Q rows). Each block still reads the full K
sequence → 2× K HBM traffic + 2× scheduling overhead. Bench: 1.85s/it (default
ncols1=64) → **19.34s/it (ncols1=32)** — measured during 2/8 iters before kill.
The lap-26 dev knob comment "smaller ncols1 → lower register pressure / higher
occupancy" had the directionality WRONG for this workload. (Occupancy is already
saturated at the launch_bounds limit, so smaller ncols1 only adds K-read overhead.)

### ✗ FA occupancy=4 — **+1.4% regression vs occupancy=3**

3 blocks/SM is the sweet spot on sm_86 for this shape. Bumping to 4 → nvcc must
cram more registers/block, slight register spills (no warnings but measurable),
wall +1.98s. Reverted.

### ✗ Audio cross-attn KV cache — **too small to measure + F16 fails bit-exact**

Implemented full lap-26-style runner-side persistent buffer + per-block cpy
writes at step 0/1, consume reads at step>1. F32 buffer (151 MiB) **OOMs under
--max-vram 9** (compute buffer wanted 1409 MiB, total push past 9 GB). Dropped to
F16 buffer (75 MiB) — fits, runs, but bench shows **+0.5s wall (within noise)**
and **PSNR mean 44 dB / min 38 dB** — the F16 precision drop in K visibly drifts
across the 25 frames. Reverted.

Audio cross-attn projection chain is just too small (768→4096 GEMM on M=192
tokens, called 48 blocks × 6 steps cache-hit ≈ 144ms savings = 0.1% wall, below
the bench noise floor). The handoff-quoted "~0.3-0.5% wall" was optimistic.
Lever is dead at this resolution; only worth revisiting if/when standard bench
gets faster (noise floor shrinks). Lap-26-pattern infrastructure proof-of-concept
worked though — text cross-attn cache could follow the same shape if needed.

### ✗ Reserve: other launch_bounds sites NOT tried

In the same audit I found `, 1)` launch_bounds on:
- `mmvq.cu:395, :601` — matvec quantized
- `mmf.cuh:49, :298` — matmul float
- `fattn-common.cuh:625, :678, :758, :864` — FA combine/reduce kernels (each ~µs/call, marginal)
- `softmax.cu:302` — parallelize_cols softmax
- `topk-moe.cu:80`, `ssm-scan.cu:19, :118` — not used by avatar

The MMQ regression evidence suggests these probably don't all respond. Worth
trying selectively (mmvq for VAE, mmf for F16 matmuls in the cross-attn
projections, fattn-common combine kernels), but each needs an ncu-based decision
or empirical bench-and-revert. **Rule reminder: bump only kernels that are
L1TEX-stall-bound, not smem-bound.**

---

## The levers that are LEFT for lap-30 (ranked, with current ROI honesty)

### 1. **Custom sparse-flash kernel** (BSA-aware) — still the biggest authorized upside

Unchanged from lap-29's framing — owner explicit license, BSA mask infra is
DONE (lap-28.4), kernel is the work. But now scaled differently: with FA share
dropped from 38.9% to 30.5% of consume-step wall, the BSA upside drops from
~8-15% to ~6-12% wall. Still the biggest available lever.

**Quality gate is still the blocker.** Owner verdict on radius={1,2} cube[4,6]
was NOT acceptable. lap-30 should render a quality-experiment matrix BEFORE
writing the kernel:
- Asymmetric h/w radius (e.g., rh=4 rw=2 — vertical motion is smaller)
- Multi-frame anchors (t=0 + t=last + every-Nth)
- Per-block density (early blocks dense, later sparse)
- Per-step density (first 2 steps dense, last 6 sparse)
- Smaller cubes (e.g., [2,3] or [4,4])

These are all env-knob experiments using the existing lap-28.4 mask plumbing
(`LONGCAT_BSA_CUBE_H`/`_W`/`_RADIUS`). Render each, owner-review, find a
quality-acceptable config, THEN write the sparse-flash kernel.

### 2. **`ggml_ext_attention_ext` non-flash → flash dispatch (audio cross-attn)**

Audio cross-attn currently uses non-flash (`flash=false`) because the per-frame
batching collapses N to 1 via the FA wrapper's output view. The non-flash path
materializes the [L_k=32, L_q=192, n_head*T_n] score tensor in F32 — cheap, but
running flash would skip materialization. The wrapper bug that forced non-flash
("FLASH path collapses N to 1" comment in `audio_cross_attn`) is fixable — just
needs the wrapper to return a non-collapsed output view for N>1. Then audio
attention runs on the now-faster FA kernel.

Estimated wall delta: ~0.1-0.3%. Small but real, and bit-exact.

### 3. **Cross-attn (TEXT) KV cache** — lap-26 pattern, bigger savings than audio

Text cross-attn has a bigger kv_linear (4096→4096 GEMM on L_ctx=512 tokens) +
permute + cont + k_norm chain, called 48 blocks × 7 redundant steps. Sized ~0.4%
wall. Same plumbing pattern as the dead audio attempt — runner persistent buffer,
per-block cpy writes at step 0/1, consume reads at step>1.

The key constraint that killed audio cache (F32 buffer OOMs at --max-vram 9):
text K/V is [head_dim=128, num_heads=16, n_ctx=512] = 1 MiB per tensor × 48 × 2
= 96 MiB F32 (vs audio's 151 MiB). Fits.

PSNR risk: same as audio's F16 attempt (~40dB) if cached F16; bit-exact if F32.
Try F32 first; if it fits AND bench shows real wall savings (above noise), ship.

### 4. **MMF / MMVQ launch_bounds audit** (after #2, #3)

`mmf.cuh:49, :298` and `mmvq.cu:395, :601` are at `, 1)` launch_bounds. mmf is
used for F16/BF16/F32 matmul fallbacks (the avatar's pf32-forced Linears may
route here for some shapes). Try bumping each to 2 with bench-and-revert.
**Skip if it spills or regresses** — MMQ taught us not all kernels respond.

### 5. **FA combine kernels** (`fattn-common.cuh:625, :678, :758, :864`)

Each FA call ends with a small combine/reduce pass over the multi-pass output.
Per-call wall is ~µs; bumping occupancy from 1 to 2 is cheap to try but
estimated <0.1% wall. Low ROI, only worth trying as part of a batch.

### 6. **FA `nbatch_K2` / `nbatch_V2` / `nbatch_fa` tuning** for our ncols=64 case

Lap-29.1 changed only `occupancy`. The other config knobs (`nbatch_K2=64`,
`nbatch_V2=64`, `nbatch_fa=64`) might also have headroom now that occupancy=3
freed register budget. Each is a sweep — try 32/96/128 for each, rebuild +
bench. Risk: regressions / spills. ~30 min per experiment.

### 7. **ncols=128 case** (per-head Q-batch persistence, handoff lap-29 #2b)

Each block handles 2× more Q rows = 0.5× K HBM reads + better Q-amortization.
Requires:
- New `GGML_CUDA_FATTN_MMA_CONFIG_CASE(128, 128, 128, ...)` entry
- Template instantiation in `fattn.cu` (the dispatch only goes up to 64/ncols2 now)
- Verify register budget fits (occupancy may need to drop back to 2)

~4-8h work. Realistic ~1-3% wall on top of lap-29.1. Try AFTER the cheap levers.

### 8. **Dead-ends (do not re-burn)**

ALL of these have been measured dead:
- MMQ occupancy bump (+27% regression) — see "What FAILED" above
- FA Q_in_reg=false (-8.5% revert) — see above
- FA NCOLS1<64 via env (10× regression) — see above
- FA occupancy=4 (+1.4% regression) — see above
- Audio cross-attn KV cache F16 (PSNR fail, no wall savings) — see above
- BSA via stock ggml flash (+5.7% regression) — see lap-29 handoff
- MUL_MAT precision pf32 lever (lap-26): MMQ dispatch ignores prec flag.
- CUDA graphs: ~5ms launch overhead, step is compute-bound.
- FP8 K/V storage: sm_86 has no FP8 compute.
- Q3_K weights: +7% slower than Q4_K.
- conv-3d-direct VAE kernel: 1.19× slower than im2col+cuBLAS.
- FP32 LayerNorm → FP16-with-FP32-accum: bit-exact blocker.
- Native m16n8k8 D=72 ViT FA kernel: 12-20h surgery for ~2% wall.

---

## Method (reproducibly)

**Build:** `~/dev/kobbler/docker/longcat-avatar-dev/iter.sh build` (~18-30s
incremental, sm_86, ccache).

**Standard bench (lap-30 reference):** 480x832, 25f, --steps 8, resident
(no --offload-to-cpu), `--diffusion-fa --seed 42 --clip-on-cpu --max-vram 9`.
Use `/tmp/render_bench.sh /src/build/<NAME>.webm "<env kvs>" --steps 8`.

**Bit-exact gate:** `python3 tools/clip_compare.py build/lap28_lap27baseline.webm
build/<NEW>.webm` — read PSNR 99.00 mean+min.

**Profile:** `LONGCAT_OP_PROFILE=1`. Post-lap-29.1 per-step breakdown:
- MUL_MAT 55.8% (7797 ms, 970 calls) — Q4_K MMQ, **floor on Ampere per project memory + lap-29.2 MMQ retune**
- FLASH_ATTN_EXT 30.5% (3908 ms, 144 calls) — already tuned (occ=3)
- ADD 4.0%, CONT 3.2%, SCALE 2.3%, CONCAT 2.0%, MUL 1.8%, ROPE_PE 1.5%, UNARY 1.0%
- SOFT_MAX 0.2%

MUL_MAT + FA = 86.3% — basically unchanged in ratio from pre-lap-29 (86.0%).
What changed: total compute time shrunk because FA share got faster.

---

## ONE-LINER reminders the next agent will need

- Eye-test: http://10.0.0.208:8011/
- Standard bench: 480×832, 25f, --steps 8 RESIDENT, --max-vram 9.
- New shipped baseline: **142.03s mean**, sampling **107.65s mean**.
- Owner mentality: *"go mental"* + *"custom kernel for wins is our bag."*
- BSA mask infra is DONE (lap-28.4) — kernel is the next big move IF quality OK.
- lap-29.1 (`lap29_occ3_final.webm`) is the current shipped baseline.
- Measure, do not predict. Quote step-count + PSNR with every number.
- Bump launch_bounds occupancy ONLY for L1TEX-stall-bound kernels. MMQ taught us
  this the expensive way (+27% regression when bumped).
- "I can't explain why" + matching symptom = read actual GPU bytes (lap-27
  pattern: FNV hash of buffer contents).
- Commit each win: submodule-first (if ggml touched), then bump parent.
  Bit-exact PSNR 99.00 every time, OR get owner OK on quality trade.
