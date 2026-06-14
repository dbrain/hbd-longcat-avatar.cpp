# HANDOFF — FULL-TUNE Wan2.2-VACE performance (RTX 3060)
_Worktree `/home/dbrain/dev/longcat-avatar-wan22` (branch `wan22-infinitetalk`, UNCOMMITTED). Detail trail:
`HANDOFF-wan22-PERF-VRAM-TUNING.md` (FINDINGS-A→K), memory `project_wan22_infinitetalk_3060`._

## MISSION (single goal)
**Wan2.2-VACE-FUN-A14B is the chosen engine for long-form directed video.** Make it as FAST as possible on
the 3060 — **match or beat LTX-2.3, ideally beat it.** Performance only. Continuation/video-quality testing
is DEFERRED (later pass) — but don't REGRESS visual output while tuning (A/B every change with a frame
eyeball + bit-exactness where claimed).

### The target to beat
LTX-2.3 prod = **1280×704, ~50 min for 27 s** of video. Compare on **render-seconds per second-of-video**
at a chosen production resolution. Current VACE @ 480×832 ≈ 58 min/27s (≈ LTX, but lower res) — the job is
to pull that WELL under 50 min and ideally push the res up. Pick a production res + report throughput in
those terms every time.

## BASELINE (measured this session, `run_vace_chain.sh`, 480×832, FR=21, distill 4-step, maxv6)
Per segment ≈ **127–158 s**. Stage breakdown (the map of where to dig):
| stage | VACE | i2v (same cfg) | VACE-specific penalty |
|--|--|--|--|
| **VAE encode (control context)** | **~30–32 s** | ~6 s | **+24 s** — re-encodes ref+inactive+reactive (3–4 full VAE passes, mostly GRAY frames) |
| DiT sampling (4 step) | 69–89 s (~22 s/it) | 55 s (13.85 s/it) | +15–34 s — control conditioning adds tokens → graph cut to **38 segments** (vs i2v's 5) |
| VAE decode | ~27 s | 18 s | +9 s |
| T5 (umT5 GPU) | 1–10 s | 1 s | run-to-run variance |
VACE-FUN expert = **9.87 GB** (vs i2v 8.15 GB) → at maxv6 it can't fully reside → more/smaller graph cuts.

## PRIORITIZED LEVERS (attack in this order — biggest VACE-specific waste first)
1. **★ The ~30 s VAE control-context encode (the #1 prize, VACE-only).** It runs `encode_first_stage`
   3–4×/segment on the control video (inactive + reactive + ref). For a continuation segment only K frames
   carry real content; the rest is `full(0.5)` gray — encoding gray every step is pure waste. Levers:
   (a) the latent backdoor already SKIPS re-encoding the K carried frames (`VACE_CONT_LATENT`) — extend that
   logic so the gray/non-content frames aren't VAE-encoded at all (synthesize their latent directly — gray
   0.5 → a known constant latent, no VAE pass); (b) batch the inactive+reactive encodes into one;
   (c) VAE-encode tiling tune (currently 0.25 = many tiles). Code: `src/stable-diffusion.cpp:5700-5770`
   (the VACE branch: `encode_first_stage(inactive)` @5730, ref @5704). Profile it (nsys/ncu) first.
2. **DiT max-vram re-sweep for the 9.87 GB expert.** maxv6 was tuned for the 8.15 GB i2v expert; the bigger
   VACE expert + control-token compute buffer needs a different budget to keep the active expert resident
   (fewer than 38 graph cuts). Sweep `--max-vram` 6/7/8/9 at FR=21; find where DiT drops toward the
   resident floor without OOM. Watch peak vs the sub-7.5 GB target (relax if a bigger budget wins big).
3. **VAE decode (~27 s).** Same tiling/`im2col`-pad levers as i2v (FINDINGS-11); 0.25 temporal tiling may be
   over-tiling. Sweet-spot sweep.
4. **VACE-specific DiT work (the 8 vace_layers + control tokens).** VACE's DiT is heavier than i2v purely
   because it does MORE (8 extra control layers per step + more tokens), through the SAME floored kernels.
   (a) **vace_layers step-invariance:** the control branch conditions on the (fixed) control video — profile
   whether its 8 layers must recompute EVERY denoise step or can be computed ONCE and cached across steps
   (big win if cacheable — 4 steps × 8 layers → 1× 8 layers). (b) **control-token reduction:** check whether
   the full control sequence must be in the DiT every step or only the active window. Both = less DiT work +
   smaller buffer (helps lever 2 reside). NOT covered by the i2v floor proof — profile fresh.
5. **Distill schedule / steps.** Already 4-step DMD (cfg 1). Confirm 4 is needed; 3 might hold quality.
6. **Token/res sweep** for the production-res decision (quality call — surface to user, don't silently cut).

## VACE vs i2v = SAME PATH, MORE WORK (verified from logs — important scoping of "do not chase")
Both = identical Wan arch (dim 2048, 40 layers, 16 heads, **head_dim 128**, q4_K weights) and the SAME DiT
kernels (`mul_mat_q<Q4_K,128>` + `flash_attn_ext_f16<…128…>`). VACE differs ONLY by doing MORE:
- **+8 `vace_layers`** (i2v=0, VACE=8) — a control branch, +20% transformer layers, run per denoise step.
- **+more tokens** — control context injected into the sequence: VACE L_q ≈ 1820/frame vs i2v 1560/frame
  (verified: VACE flash L_q=10920 @ FR=21 vs i2v L_q=6240 @ FR=13) → bigger matmul N + O(L²) attention.
⇒ VACE's ~22 s/it (vs i2v 13.85) is MORE WORK through floored kernels, NOT slower kernels.

## DO NOT RE-CHASE — but read the SCOPE carefully (the "at floor" is KERNEL-efficiency, transfers to VACE)
- **The matmul + flash-attn KERNELS are AT THE SILICON FLOOR** (ncu --set full on i2v: occupancy/latency-
  bound at theoretical-max occupancy, zero divergence, no unit saturated; mmq_x knob dead; nsys: 3µs launch
  = NOT launch-bound). This is a property of the shared q4_K/head_dim-128 kernels → **VALID FOR VACE (same
  kernels, verified).** Do NOT try to make each matmul/FA run faster.
- **HOWEVER — VACE's EXTRA work is NOT covered by that proof and IS a lever** (it's "more launches of floored
  kernels," reducible by doing less): the **8 vace_layers** (profile whether the control branch must run
  EVERY denoise step or can be computed once/cached — the control context is step-invariant), and the
  **extra control tokens** (lever 4). These are the VACE-specific DiT levers — fair game, unprofiled.
- Pinned-offload = one-shot loss. Kernel A/B (FORCE_MMQ/CUBLAS) = flat.
- **Pinned-offload (LONGCAT_DIT_NO_MMAP)** = net LOSS one-shot (unamortized cudaMallocHost). Only revisit for
  a WARM multi-segment worker (load once, amortize) — relevant if you build a resident VACE server.
- Kernel A/B (FORCE_MMQ/CUBLAS) = flat.

## RULES (carry over)
- Every "at floor"/"dead lever" claim ships a quantitative breakdown (which HW limiter, what was measured).
- Profile DEEP not basic: `ncu --set full` / nsys; both are IN the builder (nsys at
  `/opt/nvidia/nsight-compute/<VER>/host/target-linux-x64/nsys`; convert .qdstrm via host QdstrmImporter +
  `apt install libdw1`; `--cap-add SYS_ADMIN` for counters). Keep raw output in `perf_out/`, read summaries.
- A/B every change: wall + peak VRAM + per-stage + a frame eyeball. Run-to-run variance is REAL (GPU shared
  with prod + thermal) — repeat key numbers; don't trust a single run.
- Single GPU shared with PROD (gemma llama :8080 + ace :8088, worker-isolated to VRAM-0 when idle). Don't
  run two GPU jobs at once; coordinate. C++ builds fine on-box (docker builder); Rust does NOT build here.
- Drive heavy iterative GPU work from the MAIN loop, harness-tracked (`run_in_background:true`), never
  detached `&`. Clean up strays (`pgrep`/`docker kill`) before finishing.

## ENV / TOOLING
- Models in `models/`: `wan22-vace-fun-a14b-{low,high}-distill-q4_k.gguf` (9.87 GB ea), `longcat-wan-vae-f16`,
  `longcat-umt5-xxl-q8_0`. (i2v/s2v/IT models also present.) More on `10.0.0.151:~/dev/wan22-infinitetalk/models/`.
- Scripts: **`run_vace_chain.sh`** (2-seg continuation; the VACE harness — adapt for single-seg perf runs),
  `perf_a14b.sh` (env-knob sweep, i2v — clone for VACE), `profile_{decomp,nsys,ncu_full}.sh`. VACE env:
  `VACE_SAVE_LATENT`/`VACE_CONT_FRAMES=K`/`VACE_CONT_LATENT` + `--control-video <dir>`.
- Build: `docker run --rm --gpus all -v $PWD:/src -v longcat-avatar-iter-ccache:/root/.ccache -w /src
  longcat-avatar-dev:builder bash -lc "cmake -S /src -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON
  -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j\$(nproc) --target sd-cli"`.
- Eye-test page: `cd perf_out/eyetest && python3 -m http.server 8097 --bind 0.0.0.0 &` → http://10.0.0.208:8097/.
- Results log: append findings to THIS file + memory `project_wan22_infinitetalk_3060` continuously.

## SUCCESS = VACE per-segment wall pulled well below the current ~130 s (target the 30 s encode + the DiT
## graph-cut), throughput beating LTX-2.3's 50 min/27 s at the chosen production res, output not regressed.

---
# LAP LOG (2026-06-14, full-tune run)

## FINDINGS-L1 — gray control-context encode KILLED (lever #1, SHIPPED, bit-exact)
**Root cause confirmed (from existing seg logs, no new profile needed — structural waste):** the VACE
control-context encode is 3× `encode_first_stage`: ref (1 frame, ~2.5s) + inactive (~14.5s) + reactive
(~14.5s) = ~32s. With `control_frame_count=0` (fresh i2v shot, no `--control-video`) BOTH `inactive` and
`reactive` collapse to `full(0.5)` — byte-identical pure-gray tensors. VAE-encoding a constant is
deterministic + input-independent ⇒ pure waste to encode it (twice, every segment).
**VACE semantics (the user's Q):** gray 0.5 IS the "generate this / no signal here" placeholder. `inactive`
carries the kept/continuation context (mask=0 region — "replace with previous frames"); `reactive` carries
the to-be-generated control frames (mask=1 region). Each branch is gray wherever the OTHER branch holds the
signal. So gray-substitution is only ever applied to a "generate, no context" slot — never to real content.
**Fix (`stable-diffusion.cpp` VACE branch ~5726):** `vace_encode_ctx()` lambda — if the input is entirely
`0.5` (`is_const_gray`), return a cached encode keyed by (W,H,T): in-run dedup (inactive==reactive) +
optional cross-run disk cache `VACE_GRAY_CACHE_DIR` (so only the FIRST segment of a chain ever computes it).
`cache[gray]==encode(gray)` ⇒ bit-exact by construction. `VACE_NO_GRAY_FAST=1` reverts. Needs `<map>`/`<array>`.
**A/B (run_vace_grayab.sh, 480×832 FR=21 maxv6, base vs opt): 42/42 frames BYTE-IDENTICAL (0 differ).**
| stage | base encode | opt encode |
|--|--|--|
| seg1 (fresh, all-gray both) | 32.3s (2.49+14.43+14.46) | **17.3s** first-ever (gray computed once + reactive cache-hit + ref 1.9) → **~2s warm** (disk-cache hit) |
| seg2 (continuation) | 30.1s (14.6+14.4) | **15.6s** (reactive disk-cache hit=0; inactive=14.6s REAL tail encode, legitimate) |
**Realized:** fresh i2v shots (director's per-scene pattern) encode **32→~2s** warm; continuation segs **30→~15s**.
**Follow-on (deferred, riskier):** seg2 `inactive` still pays ~15s to encode [K real tail][gray]; its head 2
latent frames are OVERWRITTEN by the VACE_CONT_LATENT injection and its deep tail (latent frames 3,4,5) is
bit-identical to the gray cache — only the boundary frame (frame 2, sees real pixel @4) is real. Could encode
just the real+boundary window (~12 frames) + splice gray cache for the rest. Boundary-sensitive → defer until
the all-gray win is banked. NOT done.

## FINDINGS-L2 — max-vram sweep: marginal, DiT is OFFLOAD-BOUND for the 9.87GB expert (lever #2 ~wash)
`run_vace_maxv_sweep.sh` single fresh seg 480×832 FR=21, gray-cached so DiT dominates:
| max-vram | DiT hi+lo | graph cuts | peak VRAM | gen wall |
|--|--|--|--|--|
| 6 | 83.9s | 9 | 6153 | 129.3 (computed gray) |
| 7 | ~84 (47.2=noise) | 8 | 6503 | 119.5 |
| **8** | **83.4s** | **8** | **6513** | **116.7** ← knee |
| 9 | 98.8s | 6 | 9785 | 129.7 ← SLOWER + over-budget |
DiT flat ~42s/expert across 6/7/8 (within noise). The 9.87GB expert + ~3GB compute buf > 12GB ⇒ CANNOT fully
reside at any budget (8 cuts vs i2v's 5); mv9 pushes residency (6 cuts, 9785 peak) but is SLOWER — past the
knee the bigger resident set starves headroom. **Operating point = maxv7.3 (LTX-matched cap; DiT flat across
6/7/8 within noise, so match LTX's --max-vram for apples-to-apples when scaling res — user call 2026-06-14).**
Lever #2 = marginal,
NOT the win. DiT levers left = step-count (quality) or res/tokens. Confirms handoff: VACE DiT speed = less work.

## FINDINGS-L3 — vace_layers step-invariance DEAD (lever #4a, by code inspection, no GPU)
`VaceWanAttentionBlock::forward(c, x_orig, e0, …)` (wan.hpp:502): block 0 seeds `c = before_proj(c) + x_orig`
(wan.hpp:517) where x_orig = the patch-embedded NOISY latent of the CURRENT step; then every vace block runs
`WanAttentionBlock::forward(c, e0, …)` (wan.hpp:522) with e0 = the timestep modulation. BOTH change every
denoise step ⇒ the 8 vace blocks genuinely recompute per step. Only `vace_patch_embedding(vace_context)`
(wan.hpp:796) is step-invariant = a single cheap Conv3d. **No caching win. Lever #4a closed.**

## TOKENS (this run, FA2DBG): self-attn L_q=L_k=10920 (=7 latent-frames × 1560 tok/frame @480×832), cross-attn
## L_k=512 (text). VACE does NOT add tokens to the main x-stream vs i2v — extra cost = the 8 vace blocks only.

## FINDINGS-L4 — VAE decode tiling: overlap 0.5→0.25 = −40% (lever #3, SHIPPED, visually clean)
Decode-only harness (NEW `VACE_DECODE_LATENT` env — loads a VACE_SAVE_LATENT-banked latent, sets
`final_latent_prestripped` to skip the post-sampling ref/cont strips, SKIPS the 84s DiT → ~30s/config not
~111s). `run_vace_decode_sweep.sh` + mini, 480×832 6-latent-frame banked latent, maxv7.3:
| config | decode | tiles | peak VRAM |
|--|--|--|--|
| **base ov0.5 r0.25** (old) | **26.75s** | 42 | 6525 |
| **ov0.25 r0.25** ← WIN | **15.99s** | 25 | 6513 |
| ov0.25 r0.5 | 20.70s | 9 | 6537 |
| ov0.25 r0.5 no-temporal | 20.71s | 9 | 6527 |
| ov0.25 r1.0 / no-tiling (full-frame) | **OOM** (needs 18.5GB compute buf) | 1 | — |
**Root cause:** VACE never got the avatar lap-21 0.25-overlap default (it's gated on `is_longcat_avatar`);
VACE ran the stock 0.5 overlap = ~64% overcompute. Decode time ∝ total tile AREA, and OVERLAP dominates that
(bigger tiles at fixed overlap were SLOWER — rel0.5 = 9 tiles but 20.7s > rel0.25 = 25 tiles 16s — larger
tiles = bigger per-tile buffers, worse). Full-frame impossible on 12GB (18.5GB buffer). **Fix = pass
`--vae-tile-overlap 0.25`.** Frame eyeball base-vs-ov0.25 (SAME latent): visually identical, no seams.
**Decode 26.75→15.99s (−40%, −10.8s), VRAM-neutral.**
Overlap-floor mini-sweep: ov0.1=15.98s/25 tiles (no gain over 0.25 — tile count floors at 25 @rel0.25),
ov0.0=10.27s/16 tiles BUT **REJECTED — visible SEAMS** (border-column abs-diff spikes 9.9/5.9/11.6 at
x=120/240/360 = the 4×4 tile borders, vs median 0.57 = 17-20×; ov0.25/0.1 border peaks 0.1-0.3 ≈ median 0.14
= flat/clean). no-temporal-tiling = no change @rel0.25. **FINAL decode floor (clean) = ov0.25 = 15.98s.**

## BREAKDOWN now (warm, maxv7.3, ov0.25): DiT ~84s (79%) · decode ~16s (15%) · T5 ~4s · encode ~2s ≈ ~106s.
## (Step count is a RUNTIME quality dial, NOT a perf lever — user 2026-06-14. Hold 4-step fixed for perf.)

## FINDINGS-L5 — DiT nsys profile: kernels FLOORED (81.5%) but 14% GPU-IDLE = HtoD WEIGHT-STREAM STALL (LEVER)
`run_vace_nsys.sh` (1 hi+1 lo step, FR=21, maxv7.3, ov0.25) → qdstrm→nsys-rep (QdstrmImporter needs
`apt-get update && apt-get install libdw1`), sqlite queried on HOST python3 (no python in builder).
**DiT-window (43.3s span) kernel composition:** mul_mat_q 44.3% (16.48s) + flash_attn_ext 37.2% (13.82s) =
**81.5% the floored kernels (CONFIRMED for VACE, not just inherited from i2v)**; k_bin_bcast (standalone
add/mul ×1759) 5.3%, quantize_mmq_q8_1 2.2%, **fused madd `mul_add_bcast` 1.9% (×386 — fusion IS firing)**,
norm+rms_norm 2.3%, rope 1.4%, rest small. Glue ~18%, partly fused.
**THE LEVER (user's "offload=0ms cagey" instinct = CORRECT):** execute_graph `offload=0ms` is zero only
because a background prefetch thread (lap-33/34) does HtoD off the synchronous path. Measured: **23.45 GB
HtoD in 5.34s** in the DiT window = **4.4 GB/s = PAGEABLE bandwidth** (mmap weights; pinned PCIe ≈12 GB/s).
GPU **85.8% busy / 14.2% idle (6.15s)**; of that idle, **5.34s has an HtoD copy running during the gap →
the GPU stalls WAITING on weight streaming** (causal, not coincidence). Biggest gaps 1231/786/600ms = at
graph-cut segment boundaries (prefetch not far enough ahead). **If HtoD were fully hidden, DiT floor =
max(HtoD 5.3s, compute 37.2s) = 37.2s vs actual 43.3s ⇒ ~14% reclaimable (~6s/step-pair, ~12s over 4-step).**
**Fix candidates (lossless):** (a) pinned STAGING buffer in the prefetch (mmap→pinned→GPU) for ~12 GB/s HtoD
without the full-19.7GB cudaMallocHost that made LONGCAT_DIT_NO_MMAP a one-shot loss; (b) prefetch deeper
(2 segs ahead); (c) prefetch the FIRST segment during T5/encode. NEXT: read prefetch impl → pick fix.

## FINDINGS-L5b — CORRECTION: the 14% idle is NOT a reclaimable H2D stall (offload prefetch = NEUTRAL)
TESTED the existing overlap-prefetch thread (built for exactly this) — the LTX prod combo
LONGCAT_OFFLOAD_PREFETCH_THREAD=1 + NO_PREFETCH_POOL=1 (pool ON balloons peak 6.5→10.3GB! user's warning
confirmed — keep pool OFF) + SHARED_RESIDENT=1 + NO_OFFLOAD_PIPELINING=1. **Clean A/B (gray PRE-cached so no
confound), DiT sampling wall: off 82.4s · pf 82.4s · pfsr 81.6s = FLAT (within noise); all bit-exact 21/21.**
Per-segment it ADDED counted offload (0→~200ms/seg) for zero wall gain; baseline already shows offload=0ms
(H2D already efficiently overlapped by the driver). ⇒ **the FINDINGS-L5 "HtoD stall" hypothesis was WRONG:
the overlap thread that targets exactly that does NOT cut the wall, proving the 14% nsys idle is
SEGMENT-BOUNDARY SERIALIZATION (alloc/sync/warmup across the 9 graph-cut segments + the high→low expert
switch + cold first segment — the big 1231/786/600ms gaps), not weight streaming. My earlier "prefetch win"
was the gray-cache confound (first run computes the 14s gray latent, later runs hit cache).** Offload combo
NOT adopted. SHARED_RESIDENT ~1% = noise here (LTX's −5.2% was at 720p/16GB DiT, different regime).
**LESSON: pre-warm the gray cache before any A/B; the first run pays the one-time 14s gray compute.**
**DiT FLOOR CONFIRMED = ~82s @ 480×832 FR=21 4-step (kernels 81.5% + neutral offload + irreducible seg
serialization). No lossless DiT lever remains.** max-vram re-sweep already showed fewer segments (mv9) is
SLOWER (compute-buffer pressure), so the seg-serialization can't be cheaply cut either.

## ============ FINAL FAIR-FIGHT VERDICT (2026-06-14) ============
**Cumulative lossless wins (480×832 FR=21 4-step, all bit-exact/visually clean):**
- Encode (gray control-context): 32→~2s warm (lever #1).  Decode (overlap 0.5→0.25): 27→16s, −40% (lever #3).
- DiT 82s = silicon floor (unchanged — no lossless lever; offload prefetch neutral; vace not cacheable).
- **Warm fresh-shot wall: ~127s → ~102s (−20%).**  Stage split: enc ~2 + T5 ~2 + DiT 82 + dec 16.

**Throughput vs LTX-2.3 (LTX prod = 1280×704, 50min/27s = 111 render-sec/sec-of-video):**
| res | config | wall | sec-video | render-s/s-video | vs LTX |
|--|--|--|--|--|--|
| **480×832** (VACE native) | FR=21 maxv7.3 | ~102s | 1.31s (21f@16) | **~78** | **FASTER than LTX (0.70×)** |
| **1280×704** (LTX native) | FR=13 maxv7.3 | 291s | 0.81s (13f@16) | **~358** | ~3.2× slower |

**WHY 1280 is slower = VRAM CAPACITY, not kernel speed.** At 1280×704 a single DiT block's compute buffer =
8.5GB (FR=21, OOMs 12GB even at maxv2.5 → had to drop to FR=13, buffer 5.1GB). The 9.87GB expert + 5.1GB
buffer can't co-reside in 12GB, so ~half the expert PCIe-streams every step → DiT 265s (offload-bound, peak
only 8.5GB — capacity-locked, not headroom-locked). maxv 5→7.3 only bought 311→291s. **On a ≥24GB card both
experts + buffer reside → 1280 DiT would be compute-bound (≈3-4× faster). The 3060's 12GB is the limiter.**
**FAIR CONCLUSION: VACE is NOT "too slow" — at its native 480×832 it BEATS LTX throughput (78 vs 111) and
runs a stronger model (quality bet ≥ LTX@1280); it only loses at 1280×704 because the dual-9.87GB-expert MoE
doesn't fit 12GB. The right production move on the 3060 = render VACE at 480×832 (its lane), not 1280.**
Clip: perf_out/final1280/final_1280x704.mp4 (note: fresh-i2v from char.png + "young man" prompt shows
init-vs-prompt ghosting artifacts — a QUALITY-pass item, deferred; perf-only this lap).
**Operating config (480p prod): maxv7.3 + `--vae-tile-overlap 0.25` + VACE_GRAY_CACHE_DIR set, NO offload
prefetch. All changes UNCOMMITTED in worktree.**

---
# LAP LOG (2026-06-14, run #2 — VRAM-wall re-attack + the REAL 1280 cost)

## FINDINGS-L6 — 1280×704 is COMPUTE-BOUND, the "offload=0ms" was FAKE (user-flagged, confirmed)
Re-attacked the automation's premise (shrink per-block compute buffer so the 9.87GB expert RESIDES ⇒ 3× faster).
**The premise is disproven by direct measurement.** Mechanism corrections from reading the cutter + offload path:
- The graph cutter makes **42 base segments**, then `apply_max_vram_budget` **MERGES 42→25** to FILL the
  max_vram budget (7475MB). The 5.1GB "per-block" compute buffer is NOT a single atomic block — it's a MERGED
  multi-block segment. `--max-vram 2.5` "didn't shrink it" because a single FR=21 block already exceeds 2.5GB.
- Graph-cut compute-buffer segmentation ONLY exists in offload mode (`should_use_graph_cut_segmented_compute`
  requires `params_backend != runtime_backend`). Without --offload-to-cpu the buffer is monolithic (whole-graph).
- In offload mode each block's weights are read by exactly 1 segment ⇒ `shared_resident` (min 2 segs, clamped)
  can pin only the ≥2-seg global params (adaLN/embedders), NEVER the bulk 9.87GB expert. So "expert resident"
  is not reachable by shrinking the buffer; the expert inherently streams in offload mode.
**Measured (1280×704 FR=13 maxv7.3, gray-cached): DiT 296s (hi 143 + lo 153), decode 22s, gen 327s, peak 9.4GB.**
**OFFLOAD_PROFILE per high-noise step (73.2s):** execute_graph 60.8s [compute **60.5s** / H2D 0ms / alloc 0.15s]
+ segment_overhead 12.5s. Low step: compute 60.4s + overhead 8.2s. **compute is ROCK-STEADY 60.5s/step.**
- GPU util sampled during DiT ≈ 75% (100% mid-segment, multi-second idle at the high→low expert switch).
- compute 60.5s/step ≈ 4× the 480 per-step → genuine: 1280 has ~2.3× the tokens (L_q≈14080 vs 10920) through
  the SAME floored q4_K/FA-d128 kernels (i2v floor proof transfers). This is REAL work, only movable by res/steps.

## FINDINGS-L7 — the "0ms H2D" is a BUCKETING LIE; real stall = commit/H2D-wait (instrumented, user was right)
Added per-segment timing split to compute_with_graph_cuts (commit_ms / graphprep_ms / other) — ggml_extend.hpp
OFFLOAD_PROFILE line now prints `[commit/H2D-wait=.. graphprep=.. other=..]`. Built + reran:
**high step segment_overhead 5777ms = commit/H2D-wait 5465ms (95%) + graphprep 1ms + other 311ms.**
⇒ The prefetch thread does H2D off the execute_graph path (hence offload_H2D=0ms), but the WAIT for it
(`commit_prefetched_state`: wait_prefetch_job + event_wait, in the segloop OUTSIDE execute_graph) is the real
stall — ~5.5s/step steady (12.5s cold first step). graphprep is genuinely free (build_segment_graph = O(nodes)).
**This is the ONLY reclaimable DiT slice: ~6–8s/step steady (~10% of DiT, ~24–40s over 4 steps).** Lever =
faster/deeper prefetch (pinned staging mmap→pinned→GPU @12GB/s vs current pageable 4.4GB/s) OR fewer segments
(higher max_vram, costs peak VRAM). NOTE lap-34 removed pinned staging as "bought nothing" at 480/LTX (8 segs);
at 1280 (25 segs, bigger per-seg params) the regime differs — UNTESTED whether the mmap→pinned memcpy offsets it.

## VERDICT UPDATE (supersedes FINDINGS-L5b's "streaming wall" wording, sharpens FINAL VERDICT)
1280×704 is **compute-bound** (60.5s/step floored-kernel compute), NOT a streaming/capacity wall. The expert
streaming is ~fully hidden; only ~10% (the H2D-commit wait) is reclaimable and even that leaves 1280 ~3× LTX.
The 3× gap = genuine compute (2.3× tokens). In-block graph cuts would ADD segments ⇒ MORE commit-waits ⇒ SLOWER
(counterproductive for speed). Buffer-shrink's only real value = VRAM HEADROOM (peak 9.4→lower) to enable a
Q5/Q6 quality bump, NOT speed. Fastest watchable path stays 480×832 (78 vs 403 render-s/s-video, 5× faster).

---
# ★★★ FINDINGS-L8 — THE MURK WAS THE SAMPLER SCHEDULE, NOT Q4_K (root cause + fix, 2026-06-14)
User flagged the t2v smoke as "very murky" and suspected "too low steps / not driving the 4-step distill right."
CORRECT. Root cause found by reading the distill provenance + the scheduler code, confirmed by isolation A/B.

**Provenance:** wan22-vace-fun-a14b-*-distill-q4_k = lightx2v `Wan2.2-A14B-Moe-Distill-Lightx2v` LoRA folded onto
VACE-Fun (HANDOFF-3060-bringup.md:38). lightx2v model card: **4 steps (2 hi+2 lo), CFG off, denoising
timesteps [1000,750,500,250], scheduler shift 5.**

**The bug:** Wan2.2 falls through to the generic `DiscreteScheduler` (denoiser.hpp:29). For n=4 it emits
t=[999,666,333,0] then appends another 0 ⇒ sigmas [s999,s666,s333,0,0] = only **3 real denoise steps on the
WRONG grid** (vs lightx2v's 4 steps at [1000,750,500,250]). sd.cpp HAS the correct schedule
(`build_longcat_dmd_sigmas`, stable-diffusion.cpp:3524) but gates it to `sd_version_is_longcat_avatar` ONLY —
Wan2.2 never gets it. Net = under-denoised → murk. (`build_longcat_dmd_sigmas(4,1000,shift)` reproduces the
lightx2v grid EXACTLY: shift5→[1,0.9375,0.833,0.625,0], shift7→[1,0.9545,0.875,0.699,0].)

**The fix (NO rebuild):** inject the proper grid via the CLI `--sigmas` flag. Isolation (1280×704 t2v, seed42,
same prompt/model/steps):
- baseline (smoke, generic discrete grid, shift7): MURKY.
- C = `--sigmas "1.0,0.9545,0.875,0.699,0.0"` (proper grid, shift7): CRISP. Same shift as baseline, only grid
  changed ⇒ proves the grid/wasted-step was the culprit. Face/windows/car sharp.
- B = `--sigmas "1.0,0.9375,0.8333,0.625,0.0"` (exact lightx2v shift5): crisp but slightly softer than C.
⇒ At 1280, **shift 7 proper-grid (C) wins** (matches bringup note "shift res-coupled, try 7/11 at full res").
C's hi/lo handoff lands exactly on the 0.875 MoE boundary with --high-noise-steps 2 (clean 2+2). gen=125s, peak ~9.4GB.

**WINNING PROD CONFIG (1280 t2v):** `--steps 2 --high-noise-steps 2 --sigmas "1.0,0.9545,0.875,0.699,0.0"`
(drop --flow-shift; custom sigmas override it). before/after: perf_out/{mid,face}_before_after.png.
**FOLLOW-UP (code, optional):** extend the build_longcat_dmd_sigmas override to fire for Wan2.2 distill models
so --sigmas isn't needed each run (gating challenge: distinguish distill vs full-step VACE-Fun GGUF — maybe a
--distilled flag or detect folded LoRA). For now the --sigmas workaround is correct + sufficient.

# FINDINGS-L8b — res-coupled Wan distill schedule WIRED (code, uncommitted, build deferred)
Wired the L8 fix into a proper auto-schedule so --sigmas isn't needed per run, mirroring LTX's resolution-aware
LTX2Scheduler. stable-diffusion.cpp: new `wan_distill_res_shift(spatial_seq_len)` (linear token-anchored shift,
PROVISIONAL anchors 5.0@~480p seq6240 / 7.0@1280 seq14080, clamp[5,8]) + a resolve() `else if` after the
longcat-avatar DMD block, gated `WAN_DISTILL_SIGMAS=1` + sd_version_is_wan + no custom_sigmas. Replaces ONLY the
sigma grid (build_longcat_dmd_sigmas at the res-coupled shift) — does NOT clobber sample_steps/high_noise_sample_steps
(Wan2.2 is MoE; that split drives the high<->low expert switch). At 1280 seq_len=14080 -> shift exactly 7.0 ->
identical to the validated --sigmas "1.0,0.9545,0.875,0.699,0.0". BUILD DEFERRED (live render). TODO post-render:
build, validate WAN_DISTILL_SIGMAS=1@1280==C config, then multi-res shift A/B to calibrate the low anchor.

# FINDINGS-L8d — "dit dotty" speckle = under-resolved final denoise; FIX = more low-noise steps (2+4)
User flagged the 27s montage as "dit dotty" (fine speckle, esp. wet/dark detailed areas like the neon street).
Two stacked causes: (1) the stitched mp4 was 1.5 Mbps -> H.264 mosquito noise (fixed no-GPU: CRF-14 re-encode,
musicvideo_27s_hq.mp4); (2) GENUINE residual few-step diffusion grain in the frames — the distill spends only 2
low-noise steps on cleanup so flat/dark regions don't fully resolve.
**Test (run_speckle_test.sh ladder + run_speckle_neon.sh A/B): hold 2 HIGH steps, add LOW-noise cleanup steps,
keeping trained anchors [1000,750,500,250] + appended finer tail (125,62,...). Neon-street base 2+2 vs 2+4
(seed 42): 2+4 visibly SMOOTHER (reflections resolve as reflections, not dots) + crisper neon, NO off-distribution
artifacts. Cost +17% (159->176s/seg). 2h+5l/redrive8 also clean (diminishing returns).** Sampler verified correct
(euler, eta=0 deterministic — euler_a would ADD noise; distilled_guidance ignored by Wan; lightx2v specifies no
named solver, just the 4 fixed timesteps).
**WIRED: new build_wan_distill_sigmas(total,T,shift) = trained 4 anchors + halved low-noise tail for total>4
(replaces build_longcat_dmd_sigmas in the Wan distill resolve block — does NOT move trained high anchors). So
WAN_DISTILL_SIGMAS=1 --high-noise-steps 2 --steps 4 = the clean 2+4 grid automatically.** run_musicvideo_fixed.sh
default bumped to 2+4. Knob: --steps N controls low-step count (2+2 fast/dotty .. 2+4 clean .. 2+5 max).

# FINDINGS-L8e — lower-shift is NOT a free speckle fix (2+4 stands); lightx2v "4 steps"=4 total confirmed
User asked: any CLEAN speckle fix WITHOUT the 2+4 +17% cost? And: did lightx2v "4 steps" mean 4+4=8?
- **lightx2v "4 steps" = 4 TOTAL.** config: infer_steps 4, denoising_step_list [1000,750,500,250] (4 entries),
  boundary_step_index 2 ⇒ steps 0,1=high expert, 2,3=low expert = 2 high + 2 low. Our 2+2 already matches.
- **Lower-shift FREE lever TESTED (run_shift_grain.sh, neon, 4-step, shift 7/5/3 vs 2+4-paid):** shift 5 ≈ shift 7
  (still dotty); shift 3 grain a bit smoother BUT darker/weaker structure (a TRADE, + shift3 = softest in L8c sweep);
  **2+4@shift7 = cleanest, no structure penalty.** All 4-step ≈ same time (125-138s); 2+4 = 176s (+17%). ⇒ NO clean
  free lunch: few-step grain is the coarse 2-step final denoise; only real cure = more low-noise refinement (paid).
  Lower shift just redistributes the same 4 steps. hqdn3d post = user calls janky. **DECISION: 2+4 is the fix.**

# FINDINGS-L8f — "two-halves car" = HIGH-NOISE structure/coherence failure (not grain/motion); more-high test queued
User spotted some montage cars looking like "2 cars sticky-taped together." Located it: shot1 (vintage car), seeds
123 & 7 (seed 42 = coherent). Zoom (perf_out/coherence or /tmp/car123_zoom.png): tan front-half + maroon rear-half
fused into one incoherent vehicle, STATIC + SHARP ⇒ NOT grain (low-noise), NOT motion ghost (no smear) = object-
coherence failure in the COARSE-LAYOUT phase = the HIGH-noise expert's job. With only 2 high steps the structure can
lock incoherent; low expert then renders the wrong structure sharply. So unlike grain (wants LOW steps), THIS wants
more HIGH steps — user's "should both be 4" is well-motivated for coherence specifically. Caveat: partly SEED-dependent
(known few-step distill object-duplication) ⇒ more high reduces RATE not guarantees zero; rerolling a bad take = free fix.
**Queued run_coherence_test.sh (seed 123, base 2+2 vs 4high+2low vs 4high+4low; high grids bisect [1000..500] via
--sigmas since build_wan_distill_sigmas only appends LOW tail).** GPU busy — fires when free. If 4-high merges the car ⇒
add a high-step knob; else it's seed lottery (pick coherent takes / reroll). Roles recap: HIGH=structure/layout/motion,
LOW=detail/cleanup/grain.

# FINDINGS-L8g — coherence CONFIRMED: more HIGH steps fixes two-halves car (seed123 + seed7); STEP MENU
run_coherence_test.sh (seed 123) + run_coherence_seed7.sh: base 2+2 = two-halves car; **4 high + 2 low = ONE
coherent car (both seeds).** Mechanism (high expert = structure) confirmed, generalizes past one seed. So HIGH
steps = the coherence lever, LOW steps = the grain lever (orthogonal).
**TIMING (1280x704 FR=13, measured): 2+2=168s base | 2+4=+17% | 4+2=+15% | 4+4=234s=+39% (NOT +100% — extra
steps amortize fixed load/encode).** STEP MENU: 2+2=draft/seed-hunt; 2+4(+17%)=economy (grain clean, curate
seeds for coherence); 4+4(+39%)=FULL QUALITY (grain clean + coherent). Coherence also partly seed-dependent
(seed42 fine at 2+2) so 2+4 + reroll is viable.
**WIRED: build_wan_distill_sigmas refactored to (n_high,n_low) — HIGH region bisects (500,1000] (n_high evals:
2->[1000,750], 4->[1000,875,750,625]); LOW region [500,250]+halved tail. So WAN_DISTILL_SIGMAS=1 --high-noise-steps H
--steps L gives the right grid for ANY split, no manual --sigmas. Trained 2+2 == [1000,750,500,250] exactly.**

# FINDINGS-L8h — 3+3 is the SWEET SPOT (user idea): both grain+coherence at 2+4's cost
run_3x3.sh (WAN_DISTILL_SIGMAS 3 high+3 low, wired env, odd split OK): coherence (seed123 car) = SINGLE COHERENT
car (3 high enough, == 4 high); grain (neon zoom) = CLEAN (3 low ≈ 4 low). Cost: neon_3h3l 176s == add_2h4l 176s
(both 6 steps, +17%). ⇒ **3+3 DOMINATES 2+4: same +17% cost but fixes BOTH grain AND coherence (2+4 only grain).**
Updated step menu: 2+2=draft(both issues) | **3+3(+17%)=RECOMMENDED DEFAULT (grain+coherence clean)** | 4+4(+39%)=
marginal max. run_musicvideo_fixed.sh default bumped 2+4 -> 3+3 (HSTEPS/LSTEPS still override). build_wan_distill_sigmas
handles odd n_high/n_low (3 high=[1000,833,667], 3 low=[500,250,125]).

# FINDINGS-L9 — FR ceiling @1280 + the long-form frames/throughput tension (the real LTX-2.3 blocker)
The LTX-2.3 test = ONE continuous ~27s video; our FR=13 segs = 0.8s (net ~0.5s after K=5 continuation overlap)
=> ~54 segs. LTX ~90f/seg (~5-6 segs; 8x temporal VAE vs Wan 4x = structural 2x disadvantage). FR-ceiling probe
(run_fr_ceiling.sh, 1280x704 t2v 3+3): **FR=17 FIT (5 latent, peak 8.6GB, 461s) | FR=21 FIT (6 latent, peak
11.0GB, 610s) | FR=25 OOM (buf 8.5GB).** So FR=21 fits TODAY (old "FR=21 OOMs" was the init-img/control path);
FR=25 = the buffer-shrink frontier. BUT throughput WORSENS with frames (attention O(L^2) in tokens; tokens ∝
latent frames): render-s/s-video ~222(FR13)/435(FR17)/466(FR21) vs LTX 111. ⇒ longer segs = fewer seams/better
coherence but quadratically slower (WIDENS LTX gap). Strategic fork (handoff GOAL 1): (a)1280 short+continuation
(b)1280 long-seg FR21 (c)480p long-seg (~4x fewer tokens => more frames + faster; prior verdict 480 BEATS LTX ~78
vs 111). RECO: prove long-form quality at 480 first. Buffer-shrink (in-block cuts / F16 intermediates) = the
lever to push FR past the OOM IF the 1280-long-seg lane is chosen — for FRAMES not speed (1280 compute-bound).
See HANDOFF-CONTINUATIONS-3X3.md.
