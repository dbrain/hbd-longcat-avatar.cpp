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

# ============ LAP LOG (2026-06-14, run #3 — "is the 720p throughput path real?") ============

# FINDINGS-L10 — PHASE A1: off-the-shelf efficient-attention scan + the BIG in-fork discovery
**Headline: the fork-class barrier is ALREADY GONE — this fork ALREADY HAS a working, bit-exact,
block-SKIPPING sparse-FA CUDA kernel** (LongCat lap-31.2 "BSA bitmap"). The hard 80% (a ggml-CUDA FA kernel
that genuinely skips denied K-tiles, not just masks them) is DONE + validated. BUT its measured payoff on this
exact kernel+GPU was SMALL, which is the real verdict on whether the throughput path closes the LTX gap.

## The in-fork BSA (the thing to actually reason about — it dominates the off-the-shelf options)
- Files: `ggml/src/ggml-cuda/longcat-fa-bsa-bitmap.cuh` + `fattn.cu` (device bitmap symbols +
  `ggml_cuda_set_longcat_fa_bsa_bitmap()` host setter) + `fattn-mma-f16.cuh` (the ncols2==1 MMA-f16 kernel reads
  a per-(Q-tile,K-tile) all-deny bitmap from smem, SKIPS iter dispatch for fully-denied K-tiles = real FLOP cut).
  Runner side: `longcat_avatar.hpp` `ensure_bsa_mask()` builds a STATIC cube local-window mask
  (radius + self_frame + bookend, cube_h=4 cube_w=6) + the matching CPU bitmap; `stable-diffusion.cpp`
  threads `bsa_*` params. Gated `LONGCAT_BSA=1 LONGCAT_BSA_BITMAP=1`, default-off ⇒ dense/bit-exact.
- **MEASURED on the avatar (480×832, 25f, 8-step, RTX 3060 — the SAME kernel I'd reuse for VACE):**
  r=1+self_frame skips **50% of K-tiles** yet wall only **−1.98s on 139.88s = −1.4% wall** (−2.92s sampling
  = −2.7%). Bit-exact-to-its-own-baseline (it's a quality trade vs dense; the r=1 mask itself was the quality
  decision, PSNR-gated against the BSA baseline not dense). lap-31 close: the FA codepath is "closed" — empirical
  ceiling for more mask-driven wins ≈ 0.8s. ncols=128 (2× Q-rows/CTA) PROVEN DEAD (+1.6..+5.4s, K-HBM was never
  the bottleneck; ncols2==1 forces nstages=0 ⇒ no cp.async overlap).
- **WHY only −1.4% despite 50% K-skip:** (1) avatar seq is SHORT (480/25f) so FA is a small slice + per-tile
  bitmap overhead (smem load + per-iter bit check + branch) eats most of it; (2) the DiT step is mul_mat 44% +
  flash 37% + glue 18% (FINDINGS-L5) — **sparse attention touches ONLY the 37% flash slice; the 44% mul_mat
  FFN/proj GEMMs (q4_K silicon floor) are untouched.** Even a PERFECT 50%-skip removes ≤18% of the step.
- **Wiring BSA into VACE/Wan2.2 = BOUNDED, not from-scratch:** Wan self-attn (`wan.hpp:145`
  `Rope::attention(…, pe, mask)`) already plumbs a `mask` through `ggml_ext_attention_ext` — the SAME arg the
  avatar's BSA mask rides (`longcat_avatar.hpp:337`), same head_dim=128 ⇒ same ncols2==1 MMA-f16 path. To enable:
  build a cube mask + CPU bitmap for Wan's [T,H,W] token geometry (the avatar's `ensure_bsa_mask` is
  avatar-token-layout-specific) and `set_longcat_fa_bsa_bitmap()` before sampling, pass mask into the Wan
  self-attn call. Days (mask geometry + wiring + coherence gate), NOT weeks. The kernel is the hard part and it's done.

## Off-the-shelf scan (ranked — all are PyTorch/Triton/CUDA, NONE ggml-native; all attack only the 37% FA slice)
| method | claimed speedup | hardware | quality | port effort to sd.cpp/ggml | worth chasing? |
|--|--|--|--|--|--|
| **In-fork BSA bitmap** (already built) | **−1.4% wall measured @480/25f** (50% K-skip); unproven but bigger at long seq | Ampere ✅ (this kernel) | static r=1 mask, PSNR-gated trade | **wire into Wan: days** (kernel exists) | **MEASURE-ONLY** — cheapest realization, but ceiling is low |
| Radial Attention (MIT, NeurIPS'25) | 1.9× attn @native len, up to 3.7× for 4× longer video | CUDA (block-sparse backend) | STATIC O(n log n) energy-decay mask, ~lossless w/ LoRA | re-impl mask on in-fork bitmap = days; its kernel = weeks | concept yes (maps to in-fork bitmap), its kernel no |
| Sliding-Tile Attn / FastVideo (hao-ai-lab) | 2.8–17× over FA2 **but** big nums need FA3/ThunderKittens + finetune | **Hopper** for headline; Ampere weak | ~lossless training-free ~1.8× e2e Hunyuan | FA3 kernel, fork-class **weeks**; Ampere strips the win | NO (Hopper-tuned) |
| Sparse VideoGen 2 (SVG2) | 1.89× Wan2.1 / 2.3× Hunyuan | CUDA (dynamic + flash-kmeans kernel) | top-p dynamic, PSNR~26-30 | dynamic per-step top-k + kmeans kernel = **weeks** | NO (dynamic = heavy, our L3 showed per-step recompute) |
| SageAttention 2 (thu-ml) | 1.5–2.7× over FA2 | **Ampere-native sm80/86** ✅ (INT8 QK, FP16 V) | plug-and-play, ~lossless e2e | INT8-QK attn kernel into ggml FA = fork-class **weeks** | NO for the gain (still only the 37% slice; weeks for ~1.5×-of-37%) |
| lightx2v attn | (mostly STEP-distill, which we ALREADY use) + wraps Sage/sparse | — | — | n/a — we already run its 4-step distill | already adopted (the distill) |
| ComfyUI sparse nodes (KJ/WanVideoWrapper) | wrappers over Sage/Radial/STA | — | — | not a kernel source | NO (just wrappers) |

## A1 VERDICT — is the 720p throughput path worth chasing? **NO for closing the LTX gap; the kernel already exists so a MEASURE-ONLY confirm is cheap.**
Arithmetic ceiling: at 720p the DiT step is mul_mat 44% + flash 37% + glue 18%. EVERY method above (off-the-shelf
or in-fork) attacks ONLY the 37% flash slice. A perfect 2× attention ⇒ ≤18% DiT cut; the realistic in-fork
number was −1.4% wall. So no attention lever — not Radial, not Sage, not STA, not the in-fork BSA — can bend
720p VACE from ~2× LTX (FR=13) toward 1×. The 720p gap is dominated by (a) the 44% mul_mat FFN GEMMs at the q4_K
silicon floor and (b) the dual-9.87GB-expert MoE not fitting 12GB (weight-streaming, FINDINGS-L6/L7) — neither is
an attention problem. Confirms additional-levers.md's BSA framing was OPTIMISTIC ("attacks 33% of clip wall"):
the shipped avatar result was −1.4%. RECO: do NOT port any off-the-shelf kernel; the only justified BSA work is
wiring the EXISTING in-fork bitmap into Wan to MEASURE the 720p payoff empirically (it should beat 480/25f's
−1.4% since FA is a bigger fraction at 720p, but won't approach the 2–4× needed). The real near-LTX-720p levers
remain non-attention: a ≥24GB card (experts resident → L6 projects 3–4×) or the forbidden 480p pivot (already
beats LTX 78 vs 111).

# FINDINGS-L11 — PHASE A2 (throughput knee, MEASURED) + B3 (seam eye-test) — the mission verdict
Ran `run_cont_knee.sh` (NEW): 1 scene, 3 chained continuation segs, 1280×704, 3+3, maxv7.3, K=5, gray-cache
pre-warmed. Output perf_out/contknee/fr13/{seg0,1,2, chain_fr13.mp4, seams/}.

## A2 — continuation cost + the HONEST continuous-video throughput knee
**MEASURED seg times (FR=13 720p 3+3):** seg0 (t2v) **201.7s** · seg1 (cont) **209.2s** · seg2 (cont) **199.7s**.
⇒ **CONTINUATION OVERHEAD IS ~NOISE (+0..8s).** The control path is fully absorbed: gray-cache (L1) kills the
inactive/reactive gray encodes, and the real K=5-tail encode + the 8 vace blocks add only seconds. (seg1 decode
21.5s; DiT dominates as in t2v.) So at 720p, continuation ≈ t2v — the earlier "continuation = slow path" fear is
GONE post-L1.
**The honest continuous-27s number (counts BOTH the K-overlap waste AND all segs — NOT the naive t2v rate):**
- Net video/seg = (FR − K)/16 s. seg0 keeps all FR. 27s @16fps = 432 frames.
| FR | seg time | net frames/seg | segs for 27s | total render | render-s/s-video | vs LTX(111) |
|--|--|--|--|--|--|--|
| **13** | **~204s (MEAS)** | 8 (13−5) | **54** | **~11,000s (183min)** | **~408** | **3.7×** ← KNEE (best) |
| 17 | ~461s (L9 t2v; +~0 cont) | 12 | 36 | ~16,600s | ~615 | 5.5× |
| 21 | ~610s (L9 t2v; +~0 cont) | 16 | 27 | ~16,500s | ~613 | 5.5× |
(17/21 seg times = L9 t2v + the measured ~0 continuation overhead; not re-run on GPU — the FR ordering is
O(L²)-robust, not worth the burn.) **FR=13 minimizes total 27s render time = the knee.** Note these are ~2×
WORSE than L9's per-seg t2v rate (222) because the honest continuous number MUST count (a) the K-overlap waste
(render 13, keep 8 = 1.6× tax) and (b) all 54 segs. **Honest 720p continuous = ~3.7× LTX at the best FR.** BSA
(≤−10-15% DiT, attention-only) → ~3.2× at best. Not near LTX. K can't shrink (already the continuity carry, and
the seam already drifts at K=5 — see B3).

## B3 — SEAM EYE-TEST @ 3+3: structure CARRIES, but EXPOSURE/CONTRAST DRIFTS CUMULATIVELY (a real defect)
Inspected the visible joins (stitched g0012|g0013 = seam1, g0020|g0021 = seam2; seams/big/*.png) + per-frame
luma/std trace across the 29-frame chain.
- **GOOD: structure/identity/camera CARRY.** Both seams keep the same "Rosie's" neon corner-bar, the same vintage
  car, same composition, same slow-track camera. The VACE velocity continuation mechanism WORKS — subject holds.
- **BAD: a visible, MEASURABLE, CUMULATIVE brightness+contrast/grain JUMP at each seam.** Per-frame stats:
  seg0 g00→g12 luma 49.6→45.6 std ~32 (stable) · **SEAM1 g12→g13: luma 45.6→51.1, std 32.0→48.8 (+52%)** ·
  seg1 ~luma52/std52 · **SEAM2 g20→g21: luma 53.2→65.1 (+22%), std 52.5→64.6** · seg2 ~luma66/std66.
  ⇒ each continuation segment re-generates BRIGHTER + higher-contrast; by seg2 the neon is visibly blown out
  (luma 45→65, std 32→66 over just 2 seams). This is a per-generation distribution shift that COMPOUNDS — a
  54-seg / 27s chain would runaway into over-saturation/artifacting. The PRE-fix note "mild, not invisible" was
  optimistic at 3+3: it's visible AND cumulative.
- Same defect family as LTX mod-collapse / qwen3-tts cb0 drift. Likely fix = exposure/contrast AGC-normalize each
  continuation against the prior tail (cf. TTS LoudnessAGC), or tighten the latent-injection distribution match.
  That's a FIX (next pass), not this measurement.

## ============ MISSION VERDICT (2026-06-14, run #3) — "can VACE do long-form 720p near LTX?" = NO ============
Two INDEPENDENT blockers, both now measured:
1. **THROUGHPUT:** best operating point FR=13 = **~3.7× LTX** (408 vs 111 render-s/s-video, honest continuous
   number). The ONLY attention lever (BSA) — whose kernel ALREADY EXISTS in-fork — measured −1.4% on the avatar
   and is ceiling-bound to ≤−15% DiT here (attacks only the 37% flash slice). It CANNOT close 3.7×→1×. The gap is
   structural: 44% mul_mat q4_K GEMMs (silicon floor) + Wan 4× temporal VAE vs LTX 8× + dual-9.87GB-expert MoE
   not fitting 12GB (FINDINGS-L6/L7). None are attention problems.
2. **CONTINUITY:** even ignoring throughput, the 3+3 continuation has cumulative exposure/contrast drift that
   would runaway over 54 segs. Needs an AGC-style normalization fix before a clean continuous 27s video exists.
**RECO (unchanged from L5b/L7, now double-confirmed): 720p long-form on a 3060 stays 2–4× LTX; do NOT port any
off-the-shelf attention kernel (confirmed not worth chasing per the handoff's gate). The lanes that actually
reach/beat LTX are the forbidden 480p (78 vs 111, beats it) or a ≥24GB card (experts resident → 3–4×). If 720p
on the 3060 is mandatory, the next-highest-value work is the continuation EXPOSURE-DRIFT FIX (makes the long
video coherent) — NOT attention perf.** The in-fork BSA could be wired into Wan to get an empirical 720p number
(days: build cube mask+bitmap for Wan token geometry + wire + coherence-gate) but the arithmetic says it won't
change this verdict — surfaced to owner as an option, not done.

# FINDINGS-L12 — CORRECTION to B3: the seam contrast bump is a CONTINUATION BUG (latent-variance ratchet), NOT inherent drift
Owner flagged the continuation as "broken — every cut bumps contrast ridiculously, jumps weird, replays frames,
probably not even continuing." Investigated GPU-free (parsed the banked .bin latents + the predecode-latent logs).
**CONFIRMED: the drift is in the DIFFUSION LATENT, and it COMPOUNDS — a positive-feedback ratchet in the
continuation path, fixable.** Per-latent-frame std (predecode, frame0=head … frame3=tail-that-gets-carried):
- seg0 (fresh t2v, NO injection): 0.44 → 0.67 → 0.69 → **0.73**  (healthy; intra-seg ramp present but tolerable)
- seg1 (cont, injects seg0 tail): 0.63 → 0.72 → 0.90 → **0.91**
- seg2 (cont, injects seg1 tail): 0.82 → 0.87 → 0.98 → **0.99**
Banked whole-latent std: seg0 0.643 → seg1 0.799 → seg2 0.919 (range ±2.9 → ±3.8 → ±4.4). Decoded image std
tracked it 32→52→66. **Two stacked effects: (1) intra-segment ramp — the free-generated tail over-drives vs the
pinned/injected head (even seg0 ramps 0.44→0.73; property of the few-step distill + VACE gray-control on the
generated frames = weaker constraint → variance creep); (2) inter-segment COMPOUNDING — VACE_CONT_LATENT carries
the prior segment's TAIL (its HIGHEST-variance frames, slice Tprev−K..Tprev) as the next segment's injected head,
so the baseline RATCHETS up every cut (min std 0.44→0.63→0.82, max 0.73→0.91→0.99). Geometric ⇒ blown-out neon by
seg2, would saturate/explode over a 54-seg/27s chain.** Injection DID fire (logs: "injected 2 tail latent frames"
each cont seg) — it IS continuing; the "replays frames" perception = each seg adds only 8 NET frames (0.5s motion,
tiny) + the K=5 overlap reproduction + the contrast ratchet dominating the eye.
**THE FIX (the real B3 unblock — supersedes L11's "AGC the displayed exposure"): break the ratchet at the latent
level.** Renormalize the carried tail latent to a CANONICAL reference scale before injection (per-frame/per-channel
standardize cont_tail to ~seg0's head distribution) so every segment starts from the SAME baseline instead of the
prior elevated tail — AGC-for-latents, directly analogous to the TTS LoudnessAGC drift fix. Code site:
stable-diffusion.cpp ~5874-5907 (the VACE_CONT_LATENT block) — rescale cont_tail before `slice_assign`. Optionally
also standardize the whole video_latent before banking (6100) + before decode (fixes the displayed bump too).
Small change; NEEDS GPU to validate (owner GPU busy 2026-06-14 — deferred).

**IMPLEMENTED 2026-06-14 (UNCOMMITTED, UNBUILT — owner bench active, don't build/run yet):**
stable-diffusion.cpp ~5897 (VACE_CONT_LATENT else-branch): contrast-only AGC on the carried `cont_tail`
BEFORE reshape/slice_assign — compute global mean+std, apply a single uniform gain `g=target/std` about the
mean so it preserves brightness + per-channel ratios (no colour distortion), pulling the carried-tail std to a
canonical target. Opt-in (output NOT bit-exact): `VACE_CONT_AGC=1` enables, `VACE_CONT_AGC_TARGET` overrides
(default 0.65 = measured seg0 global latent std @ 3+3). Logs `VACE_CONT_AGC: carried-tail std X -> Y (gain g)`.
run_cont_knee.sh wired: `AGC=1 [AGC_TARGET=0.65] [TAG=agc] FR=13 bash run_cont_knee.sh` + it now prints
per-seg maxFrameStd (the ratchet metric) so the A/B shows the std flattening (expect: AGC-off seg0/1/2 maxStd
0.73/0.91/0.99 ratcheting; AGC-on ~flat ~0.65-0.75). TAG makes a distinct output dir+mp4 for clean A/B vs the
banked AGC-off baseline (perf_out/contknee/fr13/). VALIDATION PLAN: (1) build (docker builder, sd-cli target);
(2) `TAG=agc AGC=1 FR=13 SEGS=3 bash run_cont_knee.sh`; (3) compare chain_fr13_agc.mp4 vs chain_fr13.mp4 +
the maxFrameStd column + eyeball the seams (contrast should hold). If a per-segment sawtooth remains (intra-seg
ramp still visible), follow-up = per-frame output AGC before decode. Tune AGC_TARGET if 0.65 over/under-flattens.
Caveat still open: confirm banked latent == encode_first_stage space (grep found only spatial scale_factor, no
per-channel latents_mean/std — if Wan per-channel latent norm is missing from the VAE path that's the deeper root).

**VALIDATED 2026-06-14 (built + A/B run, GPU): input-AGC is MARGINAL, NOT a fix — and the result reframes the root cause.**
A/B chain FR=13 720p 3+3, AGC=1 target 0.65 vs the banked baseline. Per-seg maxFrameStd (latent) + decoded image std:
| seg | baseline latent std | AGC latent std | AGC gain applied | decoded img std (base→AGC) |
|--|--|--|--|--|
| seg0 (fresh) | 0.727 | 0.727 (untouched) | — | 32→32 (byte-identical) |
| seg1 (cont) | 0.910 | **0.892** | tail 0.708→0.650 (g0.919) | 52→51 |
| seg2 (cont) | 0.993 | **0.920** | tail 0.887→0.650 (g0.733) | 66→62 |
**Verdict: AGC fired correctly (logs confirm) and shaved the ratchet ~6% at seg2 (0.99→0.92) — but the blow-out
trajectory is intact** (latent std still 0.73→0.89→0.92 climbing; decoded contrast still 32→51→62, neon still
blows out). **KEY DIAGNOSTIC: pulling the conditioning input DOWN (0.71→0.65, 0.89→0.65) did NOT pull the output
down proportionally — the output rode up to ~0.89-0.92 regardless. ⇒ the per-segment over-drive is INTRINSIC to
the continuation generation, NOT a conditioning-scale feedback loop.** This kills the "loop-gain" hypothesis as
the primary driver; input-AGC can only ever shave the small input-coupled fraction. The drift is generation-
intrinsic (the few-step distill over-sharpens when continuing) — the same class as LTX mod-collapse, and input-
side regulation can't reach it. **Two harder fixes remain IF Wan is pursued: (a) OUTPUT-side AGC — normalize the
video_latent per-frame to seg0's profile BEFORE decode+bank (forces the displayed + carried statistics directly,
regardless of intrinsic cause; needs a per-frame target profile saved to a sidecar since segments run as separate
processes); (b) root-cause the intrinsic over-drive (gray-context discontinuity in `inactive` [real 0.65 head vs
gray 1.1 tail], VACE control-branch gain, or the continuation noise schedule).**
**STRATEGIC (the bigger point): Wan continuation has TWO compounding problems — this contrast drift AND tiny
per-segment motion (8 net frames = 0.5s/seg → "barely moves/replays"). LTX-2.3 generates the full ~27s in ONE
pass (no seams, no continuation drift, no chaining) + does native audio-lipsync — so it SIDESTEPS this entire mess.
⇒ Wan continuation tuning PARKED; the decision gate is the LTX dev-model (UD-Q4, downloaded models/ltx23-dev/)
warpy-character A/B. Only resume Wan/output-AGC if LTX dev can't do clean medium-shot characters.** AGC code stays
(opt-in, default-off, harmless); run_cont_knee.sh keeps AGC/TAG knobs. chain_fr13_agc.mp4 banked.

**PIXEL-PATH A/B 2026-06-14 (owner idea: "proper VACE outpainting" = drop the raw-latent backdoor, use pixel
re-encode) — the BEST continuation mechanism tested; latent backdoor was HURTING.** Ran `NOLATENT=1` (omit
VACE_CONT_LATENT, keep VACE_CONT_FRAMES + control video → pure pixel decode→re-encode extension). The VAE
roundtrip LAUNDERS the off-manifold latent drift the backdoor was re-injecting. Three-way (latent maxFrameStd):
| seg | latent backdoor | input-AGC | **pixel path** |
|--|--|--|--|
| seg0 | 0.727 | 0.727 | 0.727 |
| seg1 | 0.910 | 0.892 | **0.857** |
| seg2 | 0.993 | 0.920 | **0.907** |
But the DECODED-image delta is BIGGER than the latent metric shows (OOD latent decodes to disproportionate
blowup): decoded luma drift seg1→seg2 **pixel +14% (47.6→54.3) vs baseline +31% (51→67)**; decoded contrast std
**pixel 39→52 vs baseline 49→66**. Seam2 right-frame: pixel luma53/std49 vs baseline luma65/std65 — the blown-out
neon is GONE, the two sides match (seams/seam2.png). **Owner's "compression compounding/softening" worry did NOT
materialize: frames are lossless PNG (no codec in loop, mp4 only at end); and SHARPNESS (laplacian-var) actually
ROSE over segments (pixel g13 6.4→g28 10.8) — the diffusion regen + distill over-sharpen bias dominated any
VAE-roundtrip softening.** Continuity held (scene/car/composition carries). **VERDICT: pixel-based extension
drifts ~HALF as much as the latent backdoor, doesn't soften, gentler seams — it's strictly better. The latent
backdoor (a perf optimization saving ~15s/seg) was actively HURTING quality; drop it as the continuation default.**
STILL not flat (luma 47→54 climbing) — the few-step-distill intrinsic over-drive remains the floor → for a clean
54-seg chain, stack a LIGHT output-AGC on the pixel path (laundering + regularization), or run more steps. But
pixel-extension is the right base mechanism. **Root-cause meta across all 3 experiments: the drift is the few-step
distill over-driving on continuation — SAME root as LTX warping. Conditioning-mechanism fixes (latent scale,
pixel roundtrip) only shave it; the real cures are more-steps (dev model) or one-pass gen (LTX). LTX-dev test
still the gate; if Wan pursued, use NOLATENT pixel extension + light output-AGC.** chain_fr13_pixel.mp4 banked. **This makes long-form continuity a
fixable bug, not a fundamental limit — and it's broken at ANY res (shared code), so it's the #1 prerequisite for a
clean continuous video at 480 OR 720.** Caveat to check on validation: confirm the banked diffusion latent is
truly in encode_first_stage's (mu-mean)/std space (the injection comment asserts it; if Wan per-channel
latents_mean/std normalization is missing from the VAE path that could be the deeper root — grep found only spatial
scale_factor, no per-channel latent norm constants).
