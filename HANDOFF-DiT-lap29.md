# LongCat-Avatar.cpp — DiT PERF HANDOFF (lap-29, the levers that are STILL left)

*Written end of lap-28 (2026-05-28). Lap-28 shipped FOUR bit-exact perf wins
(cumulative **−2.43% wall, 159.03s → 155.16s** @ 480/25f/--steps 8 RESIDENT vs
lap-27 baseline) + landed a BSA mask prototype that proved sparse attention via
ggml's stock flash kernel is a wall-time dead end (needs a custom sparse-flash
kernel) AND owner eye-tested BSA radius={1,2} = not acceptable quality.
Read this doc first; it supersedes HANDOFF-DiT-lap28.md (kept for archaeology).*

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

**MENTALITY:** *"Go mental — get it faster than it should ever be on this
hardware."* Owner is explicit: "custom kernel for wins is our bag." Do not
write off a lever because "the kernel needs surgery" — that IS the work.
"Too hard basket" framings get pushed back on (rightly).

**Every "floor" claim in this codebase has been DISPROVABLE with a profiler:**
- "im2col is at roofline" → was 7% of BW (lap-21)
- "VAE conv is GEMM-bound" → was 16.7% occupancy (lap-25)
- "MUL_MAT is floored" → was doing 7× redundant cond work (lap-26)
- "ggml-alloc cross-buffer view bug" → was actually the planner skipping
  non-subcut'd ops (lap-27)
- "ggml flash applies mask post-QK so sparse can't help" → TRUE for stock
  kernel, but custom sparse-flash kernel is the next move (lap-28.4)

"Compute-bound" / "at floor" is the START of the investigation, not the
verdict. **Measure (ncu + bit-exact A/B + GPU readback of actual buffer
contents), never predict.** Wall-clock guesses lied all through laps 21-27.

**Gates (mandatory, every change):**
- Bit-exact: `tools/clip_compare.py <base> <new>` reads **PSNR 99.00**.
- Quality trade (needs owner OK): ac16 0.83–0.84 flat across ALL frames incl.
  the LAST + 2 seeds. **BSA at radius=1 visibly failed quality** (owner: image
  warps "like a coffee cup fluid spreading to the whole scene"). Radius=2
  rendered but not yet eye-tested.
- Standard bench: **480×832, 25 frames, --steps 8** — wins MUST show there.

**ONE GPU** (RTX 3060 / sm_86 / 12 GB). Stop prod acestep / tts / llama before
heavy runs. `docker rm -f longcat-avatar-iter` strays. Never two GPU jobs at
once.

---

## What lap-28 SHIPPED

| commit | tag | what | wall delta (resident, 480/25f/--steps 8) |
|---|---|---|---|
| `003eaf6` | `kobbler-lap28.1-modulate-shift-preexpand-2026-05-28` | pre-expand modulate's `shift` so NORM+MUL+ADD autofusion fires on all 290 prev-rejecting sites | 159.03s → 157.38s = **−1.04%** |
| `9663680` | `kobbler-lap28.2-kv-prescaled-f16-2026-05-28` | F16-prescaled cond-cache K/V consume (skip F16→F32→F16 round-trip); + ggml-cuda concat extended to F16 | 157.38s → 157.19s (mean of 2 runs) = **−0.12%** (within noise; infra positive) |
| `ecb4985` | `kobbler-lap28.3-gate-add-fusion-2026-05-28` | new ggml-cuda `mul_add_bcast` kernel — fused MUL+ADD across RESHAPE views for gate_add | 157.19s → 156.28s = **−0.58%** |
| `b3ca5a3` | `kobbler-lap28.4-bsa-mask-prototype-2026-05-28` | BSA mask infrastructure (env-gated `LONGCAT_BSA=1`); built but stock ggml flash applies mask post-QK so it's a wall regression (156.28s → 165.21s = **+5.7%**), kept as prototype for the custom kernel | (regression, env-default-off) |
| `4d07d11` | `kobbler-lap28.5-scale-cast-fusion-2026-05-28` | new ggml-cuda `scale_cast` kernel — fused SCALE→CPY(F32→F16) for kv_scale prescale (~672 fired pairs/render at avatar shape, ~306 MB bandwidth saved per pair) | 156.28s → 155.16s = **−0.72%** |

**Stacked cumulative @ 480/25f/--steps 8 resident:** 159.03s → **155.16s** = **−2.43%** (lap-28.1 + .2 + .3 + .5, all bit-exact PSNR 99.00; lap-28.4 BSA prototype is env-default-off and excluded from cumulative).

The 155.16s number is the "current shipped" baseline for lap-29 deltas. Current shipped HEAD is `4d07d11`, clip is `build/lap28_scale_cast.webm`. PSNR 99.00 vs lap-27 baseline maintained throughout (`build/lap28_lap27baseline.webm` is the same-conditions reference).

**Production-quality clips in `build/`** (don't confuse with `_*.webm` smoke/debug clips):
- `lap28_lap27baseline.webm` — same-conditions lap-27 reference (159.03s, **the A/B reference**)
- `lap28_audio_fuse.webm` — lap-28.1 output (157.38s, bit-exact 99 vs baseline)
- `lap28_kvprescaled.webm`, `lap28_kvprescaled_run2.webm` — lap-28.2 (157.12s, 157.26s)
- `lap28_gateadd.webm` — lap-28.3 (156.28s, bit-exact 99)
- `lap28_bsa_f16.webm` — lap-28.4 BSA radius=1 (165.21s, NOT acceptable quality per owner)
- `lap28_bsa_r2.webm` — lap-28.4 BSA radius=2 (165.04s, BETTER but NOT prod-acceptable per owner)
- `lap28_scale_cast.webm` — **lap-28.5 (current shipped HEAD, 155.16s, bit-exact 99)**

---

## What FAILED in lap-28 (write up so it isn't re-burned)

### BSA via stock ggml flash — **dead-end without custom kernel**

The lap-28.4 BSA infrastructure (mask construction, runner-side persistent
buffer, wrapper plumbing for F16 mask) is **complete and committed**. The mask
correctly encodes the cube-window + cond-frame anchor pattern. But stock
ggml's `ggml_flash_attn_ext` applies the mask AFTER the QK dot product (mask
values are added to scores pre-softmax) — it does **not** skip masked-out K/V
tiles. The compute still hits the full L_q×L_k FLOP count, so sparsity buys
nothing in wall.

Wall measurement (radius=1, cube [4,6], cond-frame anchor): **156.28s →
165.21s, +5.7% REGRESSION**. The regression comes from the mask layer-add
overhead, not from any compute savings.

QUALITY measurement (owner eye-test): both radius=1 and radius=2 are NOT
acceptable for production. Radius=1 (`lap28_bsa_f16.webm`) visibly warps "like
fluid motion applied to the whole scene" (a coffee cup in the background
moves like liquid). Radius=2 (`lap28_bsa_r2.webm`) is "better but not 'go
live with this to save 10%' acceptable — maybe usable per-request configurable."
Owner: "if it was insanely better performance and per-request configurable
- maybe usable - but definitely not a default."

Root-cause hypothesis: the avatar's DiT distill wasn't trained with sparse
attention. Any sparsity removes context the model expects, causing global
coherence drift on frames 2-6 (which lose anchors as we move away from frame 0).
The fix is either (a) MUCH more permissive sparsity (radius=3+, multi-frame
anchors) — but that shrinks the speedup to ~5%, OR (b) finetune the model on
the sparsity pattern.

**To realize BSA as a wall lever, write a custom sparse-flash kernel** that
iterates only over allowed K tiles per Q cube. The lap-28.4 mask infrastructure
is the input — the kernel just needs to consume it (or take the cube structure
directly). See lever §1 below.

### Profile-squeeze ROI is small

Lap-28.4 didn't ship a profile-squeeze lever, but mid-session analysis of the
op-profile (post-lap-28.3) showed:
- MUL_MAT 48.1% + FLASH_ATTN 38.4% = **86%** of consume-step wall
- All other ops together = 14%
- Even folding scale1 into the norm-fused kernel (eliminate ~145 SCALE_BIAS
  calls/step) only saves ~8ms/render — below the noise floor.

The remaining named handoff levers #5 (cross-attn KV cache, ~0.3%) and #6
(full adaLN-modulate kernel, ~1% but "not bit-exact-trivial") are similarly
small. **Big wins live in MUL_MAT or FLASH_ATTN attacks. Per the handoff,
MUL_MAT (Q4_K MMQ on Ampere) is the floor.** That leaves FLASH_ATTN.

---

## The levers that are LEFT (ranked, with current ROI honesty)

### 1. **Custom sparse-flash kernel** (BSA-aware) — biggest authorized upside

**Owner-explicit license:** *"custom kernel for wins is our bag."*

Self-attn FLASH_ATTN is 38.4% of consume-step wall = ~47s across 8 steps.
A BSA-style sparse kernel that actually skips masked-out K tiles can give an
algorithmic 3-6× reduction in attention compute (depending on cube/radius).
Realistic wall savings: **8-15%** stacked on top of lap-28.3's 156.28s.

The lap-28.4 mask construction is already in `LongCatAvatarRunner::ensure_bsa_mask`
— gated behind `LONGCAT_BSA=1`. The mask is F16 [L_k, L_q] contiguous, with
-INF for denied positions and 0 elsewhere. The runner-side persistent buffer
pattern (lap-26 `condkv_buf` template) keeps it across compute() calls.

**To make BSA a real wall lever, write a CUDA kernel that:**
- Inputs: Q [d_head, L_q, H] F16, K [d_head, L_k, H] F16, V [d_head, L_k, H]
  F16, mask [L_k, L_q] F16 (already built by runner), softmax scale.
- Output: [d_head, L_q, H] F32 attention result.
- For each Q tile, iterate only over K tiles where the corresponding mask
  block is NOT all -INF. Skip "dead" tiles (whole-block deny) — that's where
  the speedup lives.
- Block layout: keep ggml's flash MMA structure (ncols1=64 stays optimal per
  the lap-26 tile-sweep) but add the per-tile mask check.

**Recipe:**
1. Start from `ggml/src/ggml-cuda/fattn-mma-f16.cuh`. Add a new template
   parameter `bool sparse_mask`.
2. In the K iteration loop (the `mask_h2 += KQ_per_iter * stride_mask;` step),
   pre-load a "block alive" predicate by OR-reducing the mask bits over the
   current K tile.
3. If the predicate says "all denied", `continue` past the K tile entirely
   (skip the QK MMA + softmax accumulation for this tile).
4. Otherwise compute as normal, with the mask-add applied to scores.
5. Bit-exact (vs stock ggml flash with the same mask) is the gate IF owner OK's
   the BSA quality first. Without owner OK, this lever stays parked behind
   the env flag.

**Quality gate FIRST.** Show owner `lap28_bsa_r2.webm` (radius=2) and any
other cube/radius experiments. If owner OK at SOME configuration, the kernel
is worth writing (4-8h estimated). If NO configuration is acceptable, the BSA
direction is dead and lever #1 → lever #2.

**The lap-28.4 commit explicitly documents** that the mask infrastructure is
done so the kernel work isn't blocked on it. Just consume the mask.

### 2. **FA2/3-class generic kernel surgery** — bit-exact, harder

Per `HANDOFF-DiT-lap28.md` lever #7, the stock ggml MMA kernel on the consume
self-attn shape (Q=9360, K=10920, head_dim=128, num_heads=32) sits at:
- Compute (SM) Throughput: 31.7%   ⇒ tensor pipe 68% idle
- L1TEX stall: ~50% of avg 14.5-cycle CPI (smem→register ldmatrix dep)
- Occupancy: 16.66% (theoretical=achieved, capped by registers AND smem)

ggml's kernel is already FA2-class (cp.async double-buffer, Q-in-reg). Beating
it bit-exact is **fork-class** work but the headroom (31.7% SOL) is real.
Three sub-experiments:

**a) `nstages=3` triple-buffer.** Smallest change. Current static_assert in
`fattn-mma-f16.cuh:34` caps nstages at 2 — remove and add the third buffer
to smem pipeline. Likely net-neutral per the lap-28 handoff but **measure**.

**b) Per-head Q-batch persistence.** Each block handles N Q-tiles instead of
1, amortizing K/V loads across them. Cuts K HBM traffic ÷N. Cost: N× register
state → drops to 1 block/SM. Tradeoff TBD via measurement.

**c) Warp specialization** (one warp dedicated to K/V loads, others compute):
tighter cp.async/compute overlap than the current double-buffer. On sm_86
requires `__nanosleep` + manual mbarrier (no async-bulk-tensor like Hopper).

**Don't claim "kernel is floored" without an ncu stall-reason teardown of
YOUR kernel vs ggml's, side by side.**

### 3. **Audio cross-attn KV cache (lever #5)** — ~0.3-0.5% wall

Audio cross-attn's `kv_linear` (audio_dim=768 → 2×hidden=8192) re-projects
step-invariant K/V every step × 48 blocks. With M=32 audio tokens the
per-call work is small but it adds up: 48 × 7 = 336 redundant calls per
render = ~few hundred ms.

Cheap reward / real-ish plumbing — cross-graph K/V persistence (lap-26 condkv
pattern again, separate buffer). Skip if other levers land first.

### 4. **Full fused adaLN-modulate kernel (lever #6)** — ~1% wall, "not
bit-exact-trivial"

Per the handoff §6: the silu→adaLN_Linear→chunk→scale_bias prelude is
unfused. Linear goes through MMQ so bit-exact is hard. Half-day estimate.

### 5. **Profile-squeeze: SCALE/CONT reductions** — small but real

The post-lap-28.3 op-profile has SCALE at 676 calls (373ms) and CONT at 1211
calls (333ms) per consume step. Most are kv_scale + permute+cont in the attn
wrapper. **Reducing SCALE via scale_bias fold into norm-fused is ~8ms
savings — below noise.** CONT reductions need build_kqv restructuring and may
land ~1-2% if done carefully (the V permute+cont is in the hot path on every
attn call).

### 6. **DMD step count 4 ≠ 8** — OWNER REJECTED, do NOT re-pursue

The 8→4 step reduction stays rejected. Don't raise.

### 7. **Dead-ends so they aren't re-burned** (from lap-26 + lap-27 + lap-28)

- BSA via stock ggml flash → +5.7% regression (lap-28.4).
- MUL_MAT precision pf32 lever (lap-26): MMQ dispatch ignores prec flag.
- CUDA graphs: ~5ms launch overhead amortization, step is compute-bound.
- FP8 K/V storage: sm_86 has no FP8 compute.
- Q3_K weights: +7% slower than Q4_K.
- Flash `nbatch_fa` smaller tile sweep: monotonic LOSS, ggml's auto-pick wins.
- conv-3d-direct VAE kernel: 1.19× slower than im2col+cuBLAS, occupancy-bound.
- FP32 LayerNorm → FP16-with-FP32-accum: bit-exact blocker; norms only 1.5%.
- Native m16n8k8 D=72 ViT FA kernel: 12-20h surgery for ~2% wall, ROI awful.

---

## Method (reproducibly)

**Build:** `~/dev/kobbler/docker/longcat-avatar-dev/iter.sh build`
(~30s incremental, sm_86, ccache; host has no CUDA).

**Standard bench (lap-29 reference):** 480x832, 25f, --steps 8, resident
(no --offload-to-cpu), `--diffusion-fa --seed 42 --clip-on-cpu --max-vram 9`.
Use `/tmp/render_bench.sh /src/build/<NAME>.webm "<env kvs>" --steps 8`.

**Bit-exact gate:** `python3 tools/clip_compare.py build/lap28_lap27baseline.webm
build/<NEW>.webm` — read PSNR 99.00 mean+min.

**Profile:** `LONGCAT_OP_PROFILE=1` (post-lap-28.3 breakdown is in this file
under "What FAILED" → "Profile-squeeze ROI is small"). Profile post-lap-29
lever to confirm the next bottleneck moved.

**Quality A/B for BSA-style trades:** render at `--steps 8` then send to
owner via the eye-test page or directly. Owner specifically checks the
"watercolour melt" / "puddle warp" failure modes — last-frame coherence and
background stability.

---

## ONE-LINER reminders the next agent will need

- Eye-test: http://10.0.0.208:8011/
- Standard bench: 480×832, 25f, --steps 8 RESIDENT.
- Owner mentality: *"go mental, get it faster than it should ever be on this
  hardware"* + *"custom kernel for wins is our bag."* Don't write off levers
  because they need surgery — that IS the work.
- BSA mask infra is DONE (lap-28.4) — kernel is the next move IF quality OK.
- Lap-28.3 (`lap28_gateadd.webm`) is the current shipped-quality baseline.
- Measure, do not predict. Quote step-count + PSNR with every number.
- "I can't explain why" + matching symptom = read actual GPU bytes (lap-27
  pattern: FNV hash of buffer contents).
- Commit each win: submodule-first (if ggml touched), then bump parent.
  Bit-exact PSNR 99.00 every time, OR get owner OK on quality trade.
