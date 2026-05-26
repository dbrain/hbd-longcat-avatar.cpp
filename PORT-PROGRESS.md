# LongCat-Video-Avatar 1.5 → sd.cpp/ggml — PORT PROGRESS / HANDOFF

Living handoff for the C++ port. The product brief is `~/dev/longcat-video-ref/HANDOFF.md`
(START HERE). This file tracks the *implementation*. Goal: an offline ≤30s 480p
talking-avatar generator on the RTX 3060. **Phase goal right now: get it VRAM-friendly
enough to SEE a clip (functional), serve the clip in a little webserver, then do a tracked
VRAM+perf baseline run.** Perf/lean optimization comes AFTER a working baseline.

Build/run on this box is fine (the no-build rule is the kobbler Rust workspace only).
GPU is single (RTX 3060 12GB) — stop prod acestep/tts/llama before heavy runs.

## STATUS (update this section every session)

- 🟢 **FEATURE lap 15 (session 20): SEAM DRIFT FIX — decode→re-encode drift sink. The lap-14
  "multi-frame Wan-VAE encode is BROKEN" diagnosis was WRONG (it was a harness input-shape bug,
  not a code bug); the encode works at T>1. Switched chaining from raw-latent passthrough to
  decode→re-encode. 2×93f seam: VISIBLE JUMP 5.51× → acceptable 2.76×. Parent-only code, ggml
  submodule pristine `c3685f55`, branch green. See `PERF.md` lap 15.**
  - **VAE-ENCODE "BUG" WAS A HARNESS ARTIFACT.** The lap-14 `GGML_ASSERT(a->ne[2]==b->ne[2])`
    crash (blamed on `wan.hpp:1056` chunk-concat) reproduces ONLY when a **4D** `[W,H,T,C]` tensor
    is fed to `WanVAERunner::encode` — `_compute` then `unsqueeze(2)`s it into a folded-channel
    `[W,H,1,T*C]` shape, so by the first downsample Conv2d the channel dim is `C*T` (e.g. 480=96×5)
    ≠ the weight's IC. The real crash is in `Resample::forward → Conv2d → ggml_im2col` (`ggml.c:4458`),
    NOT the encode-loop concat. **Fix = feed a 5D `[W,H,T,C,1]` input** (the layout the production
    sampler latent / oracle bins already use). The Wan-VAE multi-frame temporal encode was always
    correct: `sd-vae-roundtrip` (5D input) **encode→decode round-trips at T=5 = 39.03 dB, T=13 = 39.61 dB**
    (both above the ~37.6 dB VAE-vs-input ceiling, healthy latents `[60,104,2/4,16]` nnan=0).
    **`src/wan.hpp` is UNCHANGED** (no code fix needed — it was never broken). New harness inputs:
    `models/_testinputs/vae_input_t{5,13}.bin` (5D).
  - **DRIFT SINK (the real fix):** new API `sd_ctx_encode_video_frames()` (`stable-diffusion.cpp` +
    `include/stable-diffusion.h`) VAE-encodes a stack of decoded RGB frames → diffusion latent
    (matches `generate_video_ex`'s `final_latent_out` / the next segment's `cont_latent` space).
    The CLI (`examples/cli/main.cpp`) now, after each segment, takes the LAST `cond_decoded_v` DECODED
    frames, re-encodes them, and feeds the trailing `num_cond_latents` latent frames as the next
    segment's cond — instead of the v1 raw diffusion-latent passthrough. The encode→decode round-trip
    re-regularizes the cond to the Wan-VAE manifold every chain step (the reference `generate_vc`'s
    drift sink). A/B escape hatch `LONGCAT_CONT_RAW_LATENT=1` restores v1 for measurement.
  - **SEAM Δ (2×93f Q4_K, seam @f93, lap-14 regime):** BEFORE (raw passthrough) **5.51× [VISIBLE JUMP]**
    → AFTER (re-encode) **2.76× [acceptable]** — the hard seam JUMP is gone. Identity carries (cosine
    0.9995). Clips: `models/_perf/chain2x93_{before_rawlatent,after_reencode}.webm` +
    deliverable `models/_perf/chained_q4k_fixed.webm`.
  - **RESIDUAL (honest):** the re-encode smooths the seam TRANSITION but does NOT fully kill the
    per-segment exposure drift — at 2×93f, seg1 still darkens (seg0 R98 → seg1 R75; per-seg color-drift
    sum ~39 in both). The darkening is a **cond→generated brightness STEP inside each segment** (the
    cond frames decode bright ~95-100, the 8-step-DMD-generated frames sit ~75-84), NOT a uniform
    per-segment offset. **Documented mitigations tried/blocked:** (a) "more steps" — BLOCKED: the DMD
    schedule is CLAMPED to the distilled 8 (`stable-diffusion.cpp:3460` `dmd_steps = (s>0 && s<8)?s:8`),
    so `--steps>8` is a no-op; raising it needs the distill-schedule logic changed (out of scope, brief
    says don't touch step default). (b) pixel-space per-segment uniform exposure anchor — MEASURED a
    WASH (the drift is a within-segment cond→gen step, not a uniform offset; correction was ~0.8/255,
    seam unchanged 2.76×) — REMOVED, not shipped (same verdict class as lap-14's latent renorm). The
    residual cond→gen brightness step is an 8-step-DMD generation limitation; the re-encode is the
    correct architectural fix and clears the VISIBLE-JUMP class. At shorter segments (33f) the drift is
    mild in both paths (the bug is a 93f-segment phenomenon).
  - **NOTE:** the avatar ctx loads the VAE encoder (not decode-only — the AI2V ref-image path needs it),
    so `sd_ctx_encode_video_frames` works in the normal chained run with no extra flags. Single-clip path
    (`--segments 1`) bypasses ALL chaining code → bit-identical to BEST (25f/99 dB UNTOUCHED).

- 🟡 **FEATURE lap 14 (session 19): CONTINUATION CHAINING — multi-segment clips END-TO-END,
  but COLOR DRIFTS across segments (seam-quality risk realized). Parent-only code, ggml
  submodule pristine `c3685f55`, branch green. Owner eyeball gate. [SUPERSEDED by lap 15 above —
  the drift is reduced via decode→re-encode; the "VAE encode broken" claim was a harness artifact.]**
  **WHAT SHIPPED (the LongCat headline feature — arbitrary-length avatar clips):**
  - **`--segments N` + `--cont-cond-frames K`** (CLI, `examples/cli/main.cpp` + `common.{h,cpp}`):
    renders N chained 93f segments and stitches them into one continuous clip + muxes the full
    audio. Validated: **3×93f → 253 frames / 10.08s clip** (`models/_perf/chained_multiseg*.webm`),
    audio-muxed, runs clean (no OOM at the 11.8 GiB resident peak). Runtime-driven, no rebuild.
  - **Latent-passthrough conditioning** (NOT pixel re-encode): `generate_video_ex()` hands the
    post-sampling diffusion latent back; the CLI feeds the prior segment's LAST `num_cond_latents`
    (= `1+(K-1)/4`) latent frames as the next segment's fixed cond (`sd_vid_gen_params_t.cont_latent`
    + `cont_latent_frames`; held via denoise-mask 0 / timestep 0 — generalizes the N==1 ai2v path).
    **WHY NOT the reference's decode→re-encode:** the multi-frame Wan-VAE *encode* is BROKEN in
    this port — `GGML_ASSERT(a->ne[2]==b->ne[2])` at the chunk-concat (`wan.hpp:1056`), reproduced
    on CPU via `sd-vae-roundtrip` at T=5 (the avatar only ever encoded T=1). Latent-passthrough
    sidesteps it AND is lossless (no VAE roundtrip).
  - **Audio window offset** (`build_proj_inputs(... frame_offset)`, `longcat_audio.hpp`): each
    segment windows its slice of the driving audio at global frame `seg*(seg_frames-cond_decoded)`,
    so lip-sync continues. Per-segment mux also offset (`stable-diffusion.cpp`).
  - **Resident chaining** (`sd_ctx_keep_diffusion_model_resident()`): back-to-back `generate_video`
    calls would free the DiT/VAE/whisper params (one-shot `free_params_immediately` design) → later
    segments rendered against FREED GPU memory (3 successive illegal-access crashes, each one
    layer deeper: DiT free, then TE free [+ text-cond cache], then whisper free, then VAE free).
    Now all kept resident across segments when chaining; text conditioning is computed once
    (segment 0) and CACHED (the GPU-TE umT5 is freed after seg 0 to fit the DiT, so segments 1+
    can't re-run it). **Fast preset: Q3_K + GPU-TE, 93f resident per segment.**
  - **SEAM TOOL** `tools/seam_analysis.py`: frame-to-frame |Δ| at the stitch points vs within-segment.
  **THE PROBLEM (honest report — v1 seams are NOT clean):** structure/identity/pose/scene + lip
  motion DO carry across all 3 segments (the hard part works — montage shows the same girl/scene),
  but a **global color/exposure drift COMPOUNDS per segment**: seg0 RGB ~125/119/97 → seg1 ~80/88/92
  → seg2 ~55/65/**176** (heavy blue cast). Seam Δ-vs-within-median: **seam1 @f93 = 2.95× (borderline
  acceptable), seam2 @f173 = 8.47× (VISIBLE JUMP)**. **Root cause:** each segment conditions on the
  prior segment's GENERATED latent; the reference's decode→re-encode roundtrip re-regularizes the
  latent to the VAE distribution every chain step, which the latent-passthrough skips → drift
  accumulates. The 8-step DMD schedule (vs reference 50-step `generate_vc`) likely amplifies it.
  **FIX ATTEMPT (per-channel latent stat-match, MEASURED — INSUFFICIENT):** renormalize each
  cond-latent tail to segment-0's per-channel mean/std (`LONGCAT_CONT_RENORM=1`, OPT-IN). 3×93f
  A/B (`chained_multiseg_norenorm.webm` vs `_renorm.webm`): renorm REDUCED the latent color-cast
  magnitude (seg2 blue channel B 176→104, much closer to seg0's ~97) but did NOT fix the visible
  chroma drift (the per-channel latent match is too coarse for the nonlinear VAE — seg1 still
  greenish, seg2 still teal in the decoded pixels) and slightly SHARPENED the per-seam brightness
  STEP (seam2 8.47×→12.16× by the Δ metric; within-seg median dropped 5.3→4.8). Net: marginal,
  not a fix. Left opt-in; default OFF.
  **THE REAL FIX (next session):** fix the multi-frame Wan-VAE encode bug (chunk-output ne[2]
  mismatch at `wan.hpp:1056`, reproduced on CPU via `sd-vae-roundtrip` at T=5) and condition on
  decode→re-encoded pixels — the reference's `generate_vc` does exactly this, and the VAE roundtrip
  re-regularizes the latent to the data manifold every chain step (the missing drift sink). OR
  raise `--steps` for chained renders (the 8-step DMD likely amplifies the drift vs the reference's
  50-step). Single-clip path / Q4_K reference / step-count all UNTOUCHED. CLI: `--segments 3 --cont-cond-frames 13` (default cond-frames 13 = the
  reference's `num_cond_frames`).
  **CLI USAGE (owner):** add `--segments N` (+ optional `--cont-cond-frames K`, default 13) to the
  normal avatar `vid_gen` command, e.g.
  `sd-cli -M vid_gen -m <q3_k dit> --t5xxl <umt5> --vae <wanvae> --audio-vae <whisper>
   --init-img girl.png --audio speech.wav -p "a person talking" --cfg-scale 1.0
   --video-frames 93 -W 480 -H 832 --steps 8 --segments 3 --cont-cond-frames 13
   --diffusion-fa --seed 42 --max-vram 9 -o chained.webm`.
  Clip length = `video_frames + (N-1)*(video_frames - cond_decoded)` frames @ 25 fps (93f×3, 13-frame
  overlap → 253f / 10.08 s). Each segment is ~19 min (915 s sample + 201 s decode) at 93f-resident
  Q3_K on the RTX 3060; cap `--segments` accordingly. `LONGCAT_NO_CONT_RENORM=1` disables drift control.

- 🟢 **PERF lap 13 (session 18): QUALITY-DROP LADDER (owner-directed) — 4 runtime-selectable rungs
  GPU-TE→Q4_0→Q3_K→Q2_K, each side-by-side clipped. HEADLINE: Q3_K (6.84 GB) & Q2_K (5.24 GB) UNLOCK
  NATIVE 93f RESIDENT (no offload) — the lap-08b/10/11 ~89 MiB blocker dissolves with a smaller gguf.
  Q2_K is the floor and STILL renders coherent (model is noise-tolerant). Q4_K + CPU-TE reference path
  UNTOUCHED. Parent-only, ggml submodule pristine `c3685f55`, branch green. See `PERF.md` lap 13 for the
  full ladder table.**
  Gate = eyeball-coherent (NOT 99 dB); PSNR-vs-BEST reported FYI only. Step-count fixed at 8.
  - **Rung 1 GPU-TE (Q4_K):** 25f peak 10515 MiB, coherent (PSNR-FYI 44.9 dB), 81f still offload. Clip
    `models/_perf/ladder1_gpute.mp4`.
  - **Rung 2 +Q4_0:** 17.0 s/step (−2.4%), coherent (30.8 dB). **93f-resident = NO** — Q4_0 is the same
    8.94 GB on disk as Q4_K, so same resident footprint, same OOM. Q4_0 = speed knob, NOT a VRAM win.
    Clip `ladder2_gpute_q4_0.mp4`.
  - **Rung 3 +Q3_K ★ VRAM PRIZE:** DiT 6.84 GB (−2.1 GB) → 25f peak **8513 MiB** (−2000), and **native
    93f renders RESIDENT** (peak 10269 MiB, full 8-step coherent clip `q3_k_93f_resident.webm`, 93 frames
    + audio, latent std 0.88 nnan=0). Slower per step (18.8 @ 25f). Resident-93f sampling ~114 s/step
    (monolithic) — ~30% slower than Q4_K-offload 88 s/step, but needs no offload machinery & lifts the
    resident frame-ceiling from ~81f to the native 93f. Clip `ladder3_gpute_q3_k.mp4`.
  - **Rung 4 +Q2_K ★ FLOOR, STILL COHERENT:** DiT 5.24 GB (−3.7 GB) → 25f peak **8169 MiB**, 93f-resident
    peak **8737 MiB** (~3.2 GB headroom). Eyeball-coherent — face intact, mouth tracks audio, faint grain
    only; latent healthy (std 0.92/0.81, nnan=0). **Did NOT visibly break.** Slowest (21.2 @ 25f). Clip
    `ladder4_gpute_q2_k.mp4`. Q2_K is ggml's smallest K-quant — no rung built past it (further down needs
    imatrix/IQ-quants, out of scope).
  - **PRESETS (runtime-selectable, no rebuild):** "quality" = Q4_K + `--clip-on-cpu` (bit-identical BEST,
    81f needs `--offload-to-cpu`); "fast" = **Q3_K + GPU-TE** (−2 GB, native 93f RESIDENT, coherent).
    Q2_K = smallest-footprint floor for chaining many resident segments.
  - **WHY IT WORKED:** laps 08b/10/11 proved native-93f-resident had no quality-neutral rope/pool fix
    (missed the card by ~89 MiB across two un-shareable allocators on top of the 8.5 GB Q4_K weights). The
    ladder sidesteps it: a Q3_K DiT frees ~2 GB of resident weight ≫ the 89 MiB shortfall, so the same
    unchanged reserve+pool now fit. The cumulative-VRAM payoff, delivered by an owner-authorized quality
    trade. **Build deps:** Q3_K/Q2_K ggufs built via `sd-cli -M convert --tensor-type-rules
    "model.diffusion_model.=q3_K|q2_K"` from the DMD-folded q8_0 intermediate (TMPDIR=$PWD/.convert-tmp,
    memory-capped docker, `--gpus all` required for libcuda even though quant is CPU-side). Tools:
    `tools/ladder_render.sh` + `tools/sidebyside.py`.

- 🟢 **PERF lap 12 (session 17): umT5 text-encode ON GPU (deferred DiT load) — NET −15 s / −7%, OPT-IN;
  + ROOFLINE microbench → VERDICT "AT the roofline, no hand-kernel swing"; + continuation-chaining
  SCOPED below. Parent-only (ggml submodule pristine `c3685f55`), branch green. See `PERF.md` lap 12.**
  **TASK 1 (shipped, opt-in):** solved lap-05's "GPU text-encode OOMs at load (umT5 6 GB + DiT 8.5 GB
  coexist)" by REORDERING the loads. New code (`src/stable-diffusion.cpp`): when the avatar runs with the
  TE on the **GPU** (no `--clip-on-cpu`) + `free_params_immediately` + no-mmap, the `model.diffusion_model.`
  tensors are split out of the load map, DiT alloc is deferred, and a `ModelLoader` copy is stashed.
  Flow = alloc+load TE (GPU) → VAE-encode ref → umT5 encode (GPU) → free TE → `finalize_deferred_dit_load()`
  allocs DiT on GPU + 2nd `load_tensors()` for the DiT weights → whisper-encode → sample. **umT5 + DiT never
  coexist.** **NET WALL (same-build A/B, 25f/8-step): CPU-TE 214.9 s → GPU-TE 199.9 s = −15.0 s (−7.0%)**
  (umT5 encode 16.9 → 0.34 s; deferred DiT load +2.75 s). **The change is BYTE-IDENTICAL on the default
  `--clip-on-cpu` path** (predecode latent mean 0.02622/std 0.89283 = BEST; `clip_compare` 99.00 dB / min
  99.00 dB all 25 frames vs BEST). GPU-TE output is COHERENT (ac16 0.838, cond frame 99 dB) but **44.9 dB
  vs BEST** — GPU-vs-CPU umT5 float rounding drifts the chaotic DMD trajectory (same class as F16-rope /
  Q4_0). **DISPOSITION: GPU-TE ships OPT-IN (drop `--clip-on-cpu`); default stays CPU-TE (BEST-matching).**
  Strictly better than prior (was a hard load-OOM). Clips `models/_perf/task1_{gpute,cpute}_25f.webm`.
  **TASK 2/3 (roofline, no kernel shipped):** `tools/roofline_dit.cpp` times the hot ops in isolation.
  **Q4_K matmuls are COMPUTE-bound (AI ~38,800 flop/B ≫ ~280 balance), at ~37 eff TFLOPS = ~98% of the
  cuBLAS-class achievable GEMM ceiling** — proven by F16-cuBLAS / Q8_0 / Q4_0 all landing in a 35–39 TFLOPS
  band at the same shape (the 101 INT8-TOPS marketing figure is NOT the applicable roof). **Flash-attn d=128
  = 63% of fp16-tensor peak, at its realistic softmax-bound ceiling** (correct MMA kernel, lap-08).
  **VERDICT: hot ops are AT the roofline; a hand kernel CANNOT win; the FFN w1+w3 fuse is sub-1% (quantified)
  → not pursued.** Per the campaign rule, no kernel written; scoped continuation-chaining instead (below).


### CONTINUATION-CHAINING SCOPE (lap 12, NOT BUILT — needs owner greenlight)

**Goal: arbitrary-length avatar clips by chaining 93f segments, each conditioned on the prior
segment's TAIL, instead of one offload-tax full render.** The reference path is `generate_vc`
(`~/dev/longcat-video-ref/run_demo_video_continuation.py` → `pipeline_longcat_video.py:generate_vc`,
the "video continuation" mode). The avatar already has the machinery; continuation is a small
generalization, NOT a new model path.

**HOW THE REFERENCE CONTINUES (read from `pipeline_longcat_video.py`):**
- `generate_vc(video=<prior frames>, num_cond_frames=13, num_frames=93, ...)`. The last
  `num_cond_frames` frames of the prior clip become the conditioning.
- `prepare_latents`: VAE-encode those cond frames → `num_cond_latents = 1 + (num_cond_frames-1) //
  vae_scale_factor_temporal` latents (13 frames → `1 + 12/4 = 4` cond latents), normalized, written to
  `latents[:, :, :num_cond_latents]`. The remaining `num_frames - cond` latents are noise.
- Denoise loop (the simple `use_kv_cache=False` branch, which is EXACTLY what the avatar already does):
  `timestep[:, :num_cond_latents] = 0` (cond latents are clean/fixed) and
  `latents[:, :, num_cond_latents:] = scheduler.step(noise_pred[:, :, num_cond_latents:], ...)` — only the
  non-cond latents are denoised; the cond latents are held. The DiT gets `num_cond_latents=N` so its
  self-attn runs the cond/noise two-pass split (cond tokens see only cond, noise tokens see all) over N
  cond frames instead of 1.
- After sampling, `torch.cat([cond_latents, latents])` → decode → the OUTPUT keeps only
  `output[num_cond_frames:]` (the cond frames came from the prior clip; only the newly-generated tail is
  appended). frame_index / 3D-RoPE positions are assigned over the FULL `num_frames` so the new frames get
  positions continuing past the cond frames (the avatar's `gen_vid_ids` already does `idx = t*hw + h*w + w`
  over the full T — no special continuation offset needed; the cond frames occupy positions 0..N-1).
- `use_kv_cache=True` (`_cache_clean_latents`) is a SPEED optimization (cache the clean cond latents' K/V
  once instead of recomputing every step). NOT required for correctness — the `False` branch is the avatar's
  current path. **Skip KV-cache for v1.**

**WHAT THE C++ ALREADY HAS (so this is small):**
- `LongCatAvatar::num_cond_latents` (`src/longcat_avatar.hpp:959`) + the cond/noise self-attn two-pass split
  + the text-cross-attn cond-zeroing already generalize to N>1 cond latents (today set to 1 for ai2v).
- The `denoise_mask` + `process_timesteps` machinery (`stable-diffusion.cpp:1685` / 2050–2054) already pins
  the cond latents (mask=0 → `noised_input = noised_input*mask + init_latent*(1-mask)`, timestep=0 on cond).
  Setting N>1 cond latents + a wider mask front is the same code path.
- `encode_first_stage` already VAE-encodes images → latents; the avatar's audio windowing already maps
  `T_video → T_latent`.

**THE C++ / CLI CHANGES NEEDED (concrete, in ROI order):**
1. **CLI: accept a prior-clip tail as conditioning.** Add `--cont-from <prev.webm>` (+ optional
   `--cont-cond-frames`, default 13) to the avatar `vid_gen` path. Decode the LAST `cond_frames` frames of
   `<prev.webm>` (reuse the existing ffmpeg/webm reader the muxer already links), resize to W×H, stack to a
   `[W,H,cond_frames,3]` tensor.
2. **prepare path (`prepare_video_generation_latents`, the `sd_version_is_longcat_avatar` branch):** when
   continuing, VAE-encode the cond-frame stack (not a single image) → `cond_frames`-frame latent →
   `num_cond_latents = 1 + (cond_frames-1)/4`; write to `init_latent[:, :, :num_cond_latents]`; set
   `denoise_mask = 0` over `[0:num_cond_latents]`, `1` elsewhere (generalizes today's single-cond mask).
   Set `avatar_model->avatar.num_cond_latents = num_cond_latents`.
3. **audio alignment:** the audio window must start at the cond-frames boundary so lip-sync continues from
   the prior segment's end — offset the `build_proj_inputs` window by `num_cond_latents*vae_scale` frames
   (the cond latents reuse the prior audio; new frames get the next audio slice). One index offset in the
   `generate_video` audio block (`stable-diffusion.cpp:5242+`).
4. **output trim + concat:** decode the full segment, then EMIT only `frames[num_cond_frames:]` (drop the
   re-rendered cond frames); a chaining wrapper (`tools/chain_segments.sh` or in `serve_clips`) concats
   segment_0 (full) + segment_1..N (tails) + muxes the full audio. The per-segment denoise loop is unchanged
   (it already only steps the non-cond latents via the mask).
5. **(later, optional) KV-cache the clean cond latents** (`_cache_clean_latents`) to skip recomputing the N
   cond frames' K/V every step — a per-step speed win once v1 works; needs a persistent cond-K/V buffer
   across the 8 steps (gallocr-aliasing care, cf. siglip2/lap-01 memo). Defer.

**RISKS / OPEN QUESTIONS:** (a) seam continuity — does a 4-cond-latent (13-frame) overlap give a seamless
join, or does the DMD-distilled 8-step schedule (vs the reference's 50-step `generate_vc`) soften the
continuation? Validate with a 2-segment chain + eyeball the seam. (b) the reference `generate_vc` uses
`num_inference_steps=50` + `guidance_scale=4.0` (CFG) whereas the avatar is distilled 8-step / cfg=1.0 —
the cond-latent conditioning math is identical, but the seam quality at 8 steps is unproven (eyeball call,
same class as `--steps 6`). (c) BSA is OFF for the avatar (lap-05) — keep it off; continuation is dense.
**Build only on owner greenlight; this is a feature, not a perf lever.** Est. ~1 focused session for v1
(steps 1–4), since the cond-latent split + mask + VAE-encode already exist.

- 🟢 **PERF lap 11 (session 16, 2026-05-26): RESIDENT-93f blocker FULLY ACCOUNTED — INFEASIBLE
  quality/speed-neutrally; 93f stays on offload. No code shipped; submodule pristine at `c3685f55`,
  parent `4b132c4`, branch green. See `PERF.md` lap 11.** Attacked the lap-08b/lap-10 "lever 1"
  (bound the MMQ Q8_1 VMM pool so native 93f fits resident). Instrumented the ggml-cuda VMM pool +
  every matmul scratch alloc (env-gated, all reverted) and read the exact resident-93f OOM. **Finding:
  the blocker is TWO un-shareable allocators, not one growing pool — (1) the gallocr compute reserve
  is a single contiguous `cudaMalloc` of exactly 3,044.26 MiB (the lap-08b/10 self-attn floor; FFN
  tiling does NOT move it, verified), and (2) the FIRST DiT runtime-pool alloc is a 292.5 MiB F16
  `[4096×37,440]` transient (37,440 = the 93f token count) — NOT the MMQ Q8_1 src1 scratch, NOT
  op_mul_mat, NOT batched-cuBLAS (all instrumented, none fired), so it's an F16 dequant/conv scratch
  invisible to gallocr.** Budget (measured): usable 11,909 MiB; DiT weights 8,539 + CUDA context ~122
  + VAE 242. **VAE-on-GPU 93f: the 3,044 reserve itself fails (free ~2,970, miss ~74 MiB). VAE-on-CPU
  (frees the 242) 93f: reserve PASSES, then the 292 MiB pool `cuMemCreate` fails (free 205, miss
  ~89 MiB)** → `8,661 + 3,044 + 292 = 11,997 > 11,909 by ~88 MiB`. **Max resident frame count measured
  (VAE-on-CPU, steps=1): 81f FITS (healthy latent, nnan=0); 91f/93f OOM on the pool grow.** Resident
  ceiling advanced **~40f → ~85f** this campaign (lap-07 −1.7 GiB + lap-10 −585 MiB + VAE-on-CPU −242),
  but native 93f misses by the last ~89 MiB. **Every quality/speed-neutral fix tested or measured-out:**
  `GGML_CUDA_NO_VMM` legacy `cudaMalloc` pool **still OOMs** (genuine shortfall, not VMM
  granularity/frag — 292.5→294 is only 1.5 MiB slack); routing pool scratch into gallocr is an
  architectural ggml-matmul rewrite (fork-class, out of scope); capping the pool below the op's genuine
  need fails the op / slows MMQ (violates the "25f stays ~140 s" gate); the 3,044 reserve has no
  quality-neutral rope-layer cut (lap-08b: F16 rope = 43 dB fail). DiT Q4_K + CUDA context are LOCKED.
  **DISPOSITION: 93f ships via `--offload-to-cpu` (~117 s/step, lap-07, unchanged + slightly faster
  post lap-10). 25f hot path re-verified PSNR 99.00 dB / min 99.00 dB on all 25 frames vs BEST with the
  diagnostic build (proves the instrumentation was inert). For a resident render longer than 81f there
  is no headroom — STOP rather than ship a regression or a math change.** WATCHDOG NOTE for future
  sessions: a `timeout`-killed `docker run --rm` leaves the CONTAINER running (client SIGKILL ≠
  container stop) — it orphaned an 11.7 GiB sd-cli once this lap; always `docker run --name X` +
  explicit `docker rm -f X`, and check `docker ps | grep -v kobbler` + `nvidia-smi` before reporting.

- 🟢 **PERF lap 10 (session 15, 2026-05-26): fused-RoPE op ROOT-CAUSED + made DEFAULT — banks −5% at
  ALL lengths, 99 dB bit-identical to BEST. See `PERF.md` lap 10. Branch green.**
  The lap-09 42 dB divergence was **NOT gallocr aliasing** (the handoff's hypothesis). A NEW minimal
  repro `tools/test_rope_pe_gallocr.cpp` builds the REAL self-attn topology (rope → cond/noise split →
  cont → flash-attn → proj) through the ACTUAL `ggml_gallocr` (the isolation test bypasses gallocr via
  `alloc_ctx_tensors`) and diffs chain vs fused: the lap-09 kernel was only **5.4e-5** off with matched
  means — a buffer clobber would be garbage/NaN, not a uniform tiny delta. Confirmed by reading
  `ggml-alloc.c` (ROPE_PE not in `ggml_op_can_inplace`; output allocated before inputs freed).
  **ROOT CAUSE = FMA contraction in `rope-pe.cu`**: nvcc fused the rope `a*c±b*s` into `fmaf`, skipping
  the intermediate-product rounding that the chain's separate `ggml_mul`+`ggml_add` does → ~5e-5/elem,
  compounded over 48 blocks × 8 DMD steps → chaotic-trajectory drift off the gate. **FIX:** `__fmul_rn`/
  `__fadd_rn`/`__fsub_rn` (un-contractable round-to-nearest) in the chain's exact term order →
  **max|chain−fused| = 0.0** in BOTH the isolation AND the new real-gallocr repro. **Made default-on**
  (`src/rope.hpp`; opt out `LONGCAT_NO_FUSED_ROPE=1`). **VALIDATION:** 25f render on the DEFAULT path
  = **PSNR 99.00 dB / min 99.00 dB on all 25 frames** vs BEST (predecode latent byte-identical to the
  env-ON run). **Speed: sampling 147 → 139.9 s (−4.9%), DiT 17.46 s/step (was 18.35), all lengths**;
  25f peak VRAM 10,513 MiB unchanged. **93f-resident:** the rope-buffer collapse drops the 93f resident
  compute reserve **3,629 → 3,044 MiB (−585)** but it still OOMs over the 8,539 MiB resident weights —
  the remaining blocker is the lap-08b MMQ-Q8_1-VMM-pool (lever 1), now the only thing between this and
  a resident 93f; 93f keeps shipping via `--offload-to-cpu`. Commits: ggml submodule (`rope-pe.cu`) +
  parent (`rope.hpp` + repro tool + docs), both buildable/green. Clips: `models/_perf/lap10_default_25f.webm`
  (default path) / `lap10_fusedrope_25f.webm` (env-ON).

- 🟢 **PERF lap 09 (session 14, 2026-05-26): FULL PIPELINE ACCOUNTING + custom fused-RoPE CUDA op
  `ggml_rope_pe` — op PROVEN bit-identical in isolation, gives −5% per DiT step when ON, but the
  full render FAILS the 99 dB gate (42 dB, graph/gallocr interaction, NOT a math bug). Op gated
  OPT-IN (`LONGCAT_FUSED_ROPE=1`); default = chain, re-verified 99 dB vs BEST. Branch green. See
  `PERF.md` lap 09 for the TIME + VRAM tables + the divergence analysis.**
  PHASE 1 (fresh `LONGCAT_PROFILE=1` 25f render, GPU idle 9 MiB so numbers are the model's own):
  **TIME** — model load ~13 s / ref-img VAE enc 1.9 s / **umT5 CPU 16.25 s (7%)** / whisper 1.0 s /
  **DiT sampling 147.3 s (63%, 18.35 s/step)** / **VAE tiled decode 54.1 s (23%, 10×5.34 s)** / total
  ~233 s. **VRAM** — 25f peak **10,513 MiB** = weights 8,781 (DiT 8,539 + VAE 242) + DiT compute 1,451
  + MMQ Q8_1 VMM pool ~280; 25f has ~1.4 GiB headroom. 93f-resident OOMs by ~530 MiB (compute floor
  3,629 = the six ~585 MiB RoPE-internal buffers + MMQ pool). **The rope buffers are the 93f peak, NOT
  the 25f peak** — so the fused op is a 93f-resident lever, not a 25f-VRAM one.
  PHASE 2 — new `GGML_OP_ROPE_PE` (CUDA-only; CPU never claims support, avatar runs DiT on a single
  CUDA backend). One kernel reads x+pe and writes q_rope directly (no cont+2×repeat+mul+add).
  `tools/test_rope_pe.cpp` proves **max|chain−fused|=1.2e-7 @ [128,32,257]** (exact math). Wired ON:
  DiT **18.35→17.42 s/step (−5.1%)**, sampling **147→139.9 s**, 25f peak unchanged. **BUT the 25f render
  drifts to 42 dB (fails the hard 99 dB gate)** even though the op output is identical — a downstream
  gallocr-aliasing-class interaction (op collapses ~10 rope nodes to 1, changing buffer liveness), NOT
  math. Reproducibility ruled out (old-chain rebuild = 99 dB vs BEST). **lap-10 cheap next step: rerun
  with `GGML_ALLOCATOR_DEBUG`, find the buffer that overlaps the fused q_rope/k_rope output, force it
  non-reusable (likely a 1-op fix). Then the −5% all-length + 93f-resident win lands.** Commits:
  ggml submodule + parent pointer (both buildable, green).

- 🟢 **PERF lap 08b (session 13, 2026-05-26): RESIDENT-93f investigation — NO quality-neutral win
  exists at the rope layer; no code shipped, tree green at `c78e86c`. See `PERF.md` lap 08b.**
  `GGML_ALLOCATOR_DEBUG` (with an op-type-augmented high-water dump) **re-localized the 3,629 MiB
  93f resident peak** and CORRECTED the lap-07 attribution: it is **six ~585 MiB tensors that are
  RoPE INTERNALS** (`apply_rope`'s `[d/2,L,nh*N,2]` input `ggml_cont` + the two full-size
  `ggml_repeat` buffers, per call — Q's set overlaps K's), NOT the q/k/v matmul outputs or residual
  x the handoff named. **Levers measured + reverted:** (1) **RoPE math in F16** for q/k/v cuts the
  peak 3,629→**2,983 (−646 MiB)** but FAILS the bit-identical gate (25f vs BEST mean **43 dB**/min
  35 dB — coherent ac16≈0.83 but a visibly different render, trajectory drift from F16 q·k logits
  over 48 blk × 8 steps) AND still runtime-OOMs on the **MMQ Q8_1 VMM pool** (needs ~300-400 MiB
  above weights 8,539 + compute; 8,539+2,983 leaves only ~387 of 11,909 usable) — so it buys neither
  quality nor a resident render; (2) **bit-identical low-VRAM F32 rope** (compute the two within-pair
  output halves at ~292 MiB + `ggml_concat`, no `[2,...]` repeat) is **25f BIT-IDENTICAL (99 dB,
  sampling 147 s)** but REGRESSES the *resident* peak to ~3,860 (the two F32 rope OUTPUTS + downstream
  `MUL_MAT` still floor it; gallocr packs the concats higher) — may help the SEGMENTED offload path
  (untested); (3) V-early-F16 / in-place muls / serialized repeats — all bit-identical, none in the
  peak set. **VERDICT: the ~530 MiB resident gap can't be cleared quality-neutrally at the rope
  layer.** Precise fix for owner greenlight: (a) bound/disable the MMQ activation-quant VMM pool
  growth (ggml-cuda; lowest risk), or (b) a custom F32 fused-RoPE CUDA op writing q_rope in one
  kernel over a single buffer (no cont+2×repeat+mul chain) — fork-class. 93f keeps shipping via
  `--offload-to-cpu` (~117 s/step, lap-07), unaffected. 25f hot path bit-identical, unchanged.

- 🟢 **PERF lap 08 (session 12, 2026-05-26): HOT-PATH KERNEL PROFILE — flash-attn d=128 is on the
  MMA tensor-core path, MUL_MAT-Q4_K is on the MMQ int8-tensor-core kernel; BOTH optimal for this
  shape on sm_86. NO code change shipped. See `PERF.md` lap 08.** Directly answered the campaign's
  central question (is flash-attn d=128 on MMA or a slow non-MMA fallback like the head_dim-72 VL
  TILE trap?): read `ggml_cuda_get_best_fattn_kernel` — d=128 + F16 k/v + `turing_mma_available`
  true on sm_86 + Q->ne[1]≫2 ⇒ falls through to **`BEST_FATTN_KERNEL_MMA_F16`** (the head_dim-72
  trap fires only because 72 is *excluded* from MMA; 128 is first-class). MMA tile config is also
  correct (mask null + gqa_ratio 1 ⇒ ncols2=1/ncols1=64, the large-batch config). MUL_MAT: Q4_K ×
  F32 at M~10920 ⇒ MMQ int8-tensor-core (the tuned llama.cpp kernel), confirmed via
  `ggml_cuda_mul_mat` + `ggml_cuda_should_use_mmq`. Fresh op profile on the real 12503-node DiT
  step: MUL_MAT 7662 ms (M>4096 bucket = 92.6%) + FLASH 5617 ms = **72.8%**, on the optimal kernels.
  **Levers scoped, none shipped:** custom d=128 flash / Q4_K MMQ kernel = the fork-class multi-day
  rewrite the campaign authorized, but prior cross-project memos (parakeet lap-7, qwen3 INT8-mma)
  repeatedly measured hand kernels at a fraction of ggml's tuned throughput — low prior, not pursued
  without a measured reason; **FFN w1+w3 fuse** (one [4096→22016] matmul, saves 1 Q8_1 activation
  re-quant/block, bit-exact) is the single best remaining MUL_MAT lever at ~1% but needs a
  *converter-side* fused-weight tensor to dodge the gallocr-aliasing trap (parked for a future
  session); **kv_scale=1/256 SCALE-fold** = sub-noise, recorded not implemented. 25f sampling 147 s
  (8-step), **bit-identical to BEST (99 dB all frames)**, branch green at `c0ea6d3` (no change).
  Re-take the profile with the env-gated `LONGCAT_OP_PROFILE` block (re-apply ~42 lines in
  `ggml/src/ggml-cuda/ggml-cuda.cu` around `ggml_cuda_compute_forward`, then revert — kept submodule
  pristine).

- 🟢 **PERF lap 07 (session 11, 2026-05-26): QKV-SPLIT — −1.7 GiB resident @ 93f AND −9.5% on the
  25f hot path, both bit-identical. See `PERF.md` lap 07.** `GGML_ALLOCATOR_DEBUG` re-localized the
  resident (no-offload, monolithic) 93f peak: after lap-06 removed the flash mask, the peak was the
  **fused `attn.qkv` Linear output (~1.75 GiB) + its `split_qkv` permute-CONT (~1.75 GiB)** stacked.
  Fix: split the qkv WEIGHT (out-dim rows; Q4_K rows are independently quantized → exact row-slice)
  into Wq/Wk/Wv and run three separate matmuls — no fused `[3*C]` buffer or its cont ever
  materializes (`get_weight()`/`get_bias()` accessors added to `Linear`; the split is in
  `self_attn`, `src/longcat_avatar.hpp`). **93f resident monolithic 5,323 → 3,629 MiB (−1,694);
  25f sampling 162.3 → 146.8 s (−9.5%, one fewer 1.75 GiB permute-cont/block × 48); 25f BIT-IDENTICAL
  to BEST (99 dB); 93f offload BIT-IDENTICAL to lap-06 native (99 dB all 93 frames), 120.7 → 117.4
  s/step, segs 49 → 33.** Also shipped FFN token-tiling (`LONGCAT_FFN_TILES`, auto >16k tok; 1 =
  single-shot/bit-identical on the 25f path) to bound the SwiGLU `[11008, n_token]` triple at full
  length. **Resident 93f still OOMs by ~530 MiB:** the 3,629 floor is four ~585 MiB `[4096, n_token]`
  F32 tensors (q_rope MUST be F32 — flash kernel asserts it — + residual x + k_rope/v). F16-casting
  k/v did NOT move the peak (they're downstream of the peak offset) and self-attn query-tiling made
  it WORSE (shared k/v stay live, concat grows) — both measured + reverted/opt-in. 93f ships via
  offload (~117 s/step); resident-fit would need a structural rework of the self-attn working set.
  Commit `2ce48ee` on `longcat-avatar-port` (code: `ggml_extend.hpp` + `longcat_avatar.hpp`).

- 🟢 **NATIVE FULL-LENGTH (93f / 3.72s) NOW RENDERS — the 12 GB wall is CLEARED (session 10,
  2026-05-26). See `PERF.md` lap 06.** The 93f self-attn VRAM peak was NOT the qkv
  double-buffer the lap-05 handoff flagged — `GGML_ALLOCATOR_DEBUG` localized it to a
  **synthesized flash kv-pad mask** (`build_kqv` in `ggml_extend.hpp` pads `L_k` to a
  multiple of 256 and, when `mask==nullptr`, materializes a `[L_k_pad × L_q]` F32 mask +
  F16 cast — **~5 GiB ×2 at 37k tokens**). FIX: opt-in `flash_skip_kv_pad` param on
  `ggml_ext_attention_ext` (default off, every other caller byte-identical); the avatar's
  three self-attn calls pass it (`src/longcat_avatar.hpp`). Modern `ggml_flash_attn_ext` +
  CUDA MMA handle unpadded `L_k` with `mask==nullptr` directly. **93f self-attn sub-segment
  12,008 → 5,511 MiB (−6,497).** Native 93f renders via `--offload-to-cpu`
  (`models/_perf/fulllen_93f_native.webm`, 8-step, sampling 965.6s = **120.7 s/step**, VAE
  decode 200.8s, coherent ac16≈0.82 all 93f, audio muxed). **25f quality gate BIT-IDENTICAL
  to BEST (PSNR 99 dB all frames)** — pure memory-layout change. Resident (no-offload) 93f
  still OOMs (monolithic compute buffer 5,323 MiB vs ~3.0 GiB free after the 8,781 MiB
  resident weights). Resident ceiling probed: 93f reserve 5,323 MiB OOM, 61f reserve 3,551 MiB
  OOM, 49f reserve 2,887 MiB fits-but-runtime-OOMs on the tight ~3.0 GiB margin → safe
  resident ceiling is ~40f. 93f (and 49–93f) ship via weight-offload — full-length is a
  turnaround knob, not the 25f hot path.
  Commit on `longcat-avatar-port` (code: `ggml_extend.hpp` + `longcat_avatar.hpp` only).

- 🟢 **FULL-LENGTH (81f / 3.24s) RENDERS LAND via graph-cut + CPU-offload (session 8,
  2026-05-26). See `PERF.md` lap 04.** The native 93-frame segment OOMs a 12 GB card —
  the monolithic DiT forward reserves a ~13.3 GiB activation buffer at ~37k tokens, over
  the card even with zero resident weights. Fix (`src/longcat_avatar.hpp`, committed):
  per-block + intra-block (self-attn / cross-attn / FFN) graph-cut boundaries mirroring
  `anima.hpp`, so with `--offload-to-cpu --max-vram 9` the GGMLRunner segmented path
  reserves one sub-segment's buffer at a time and streams its weights from CPU. **Fits up
  to 81 frames (peak 11603 MiB); 85f+ and the native 93f OOM** (the per-block self-attn
  sub-segment is an atomic 12.6 GiB at 93f — would need intra-attention temporal chunking).
  Marks are inert when weights are resident, so the 25f fast path is unchanged. Full-length
  A/B set in `models/_perf/fulllen_81f_{8,6,4}step.webm` (8-step wall 1218s / 6-step 967s
  −21% / 4-step 711s −42%, all with muxed audio). Owner's morning lip-sync call: 6-vs-4 at
  full length is the eyeball decision; default step count unchanged (8). Offload streaming
  makes full-length ~124 s/step (turnaround knob, not the hot path).
- 🟢 **PERF/VRAM lap 05 (session 9, 2026-05-26): re-opened the "compute-bound,
  exhausted" verdict the owner flagged as premature. See `PERF.md` lap 05.** Ran the
  four owner levers MEASURED. (1) step-invariant recompute = sub-noise, not worth
  (assessed, recorded). (2) glue-op sweep: THREE clean cuts SHIPPED (`scale_bias`
  fuse + redundant qkv-cont removal + audio silu reuse) — sampling 163.7→162.3s
  (−0.86%), **bit-identical to BEST (99 dB)**. (3) Q4_0 quant: **−2.4% faster** (158.4
  vs 162.3s 8-step) but quality NOT neutral (30.7 dB vs BEST) → OPT-IN gguf, default
  stays Q4_K (better than the Q3_K dead-end). (4) BSA SCOPED: **the avatar inference
  path runs DENSE by design** (ckpt `enable_bsa=False`, `proof_gen.py` forces it off,
  BSA is batch>1-only in the reference) — so BSA/windowed-attn for the avatar would be
  a quality CUT, NOT the native attention; do NOT implement it. The 93f VRAM wall is a
  pure dense-activation memory problem: the lever-2 cont cut narrowed the 93f self-attn
  sub-segment 12,675→12,008 MiB (**now misses the card by only ~107 MiB**, was ~770);
  one more quality-neutral activation cut (de-double-buffer the qkv_out↔split_qkv cont,
  or F16 q_rope/k_rope) likely clears native 93f. **NOTE: the older "safe-speed search
  CLOSED" / "no default-on speed lever" framing was OVERSTATED — glue + Q4_0 + the 93f
  activation lever all had real measured headroom left.** Commit `24d2d9c`.
- 🟢 **PERF/VRAM PHASE — DiT-sampling PROFILED (session 7,
  2026-05-26). See `PERF.md` lap 03/03b. NOTE: the older "generated frames still
  noise" status below is STALE — the current tree renders COHERENT talking avatars
  (ac16≈0.83 on all 25 frames, frame-autocorr `tools/clip_compare.py`).**
  - **lap 03: profiled one DiT step (the remaining 69% of wall).** Verdict: the step
    is COMPUTE-BOUND — MUL_MAT 38% + FLASH_ATTN_EXT 34% = 72% irreducible Q4_K
    compute; graph build/alloc/copy is ~5 ms/step. ⇒ **CUDA-graph reuse (the forecast
    marquee win) is DEAD** — no launch overhead to recover, and ggml's CUDA-graph
    path (compiled OFF anyway) keys on a per-build graph that would never warm up.
    No large safe default-on speed lever exists. Added two env-gated reusable
    profilers (`LONGCAT_PROFILE` phase timer in `ggml_extend.hpp`, committed; a
    per-op-type ggml-cuda aggregator, used+reverted to keep the submodule pristine).
    Shipped one safe micro-win: audio `gate_mul` (drop a pointless full-size zeros-add
    per block) — nodes 13705→13561, wall −0.4%, output bit-identical (99 dB).
  - **lap 03b: opt-in step-count A/B clips for the owner** — fresh 8/6/4-step renders
    (`models/_perf/lap03_{gate_mul,6steps,4steps}.webm`): 6-step −17% / 4-step −35%
    total wall, both structurally coherent (ac16 0.83). **Default stays 8; 6-vs-4
    lip-sync is the owner's eyeball call.** Remaining levers are all quality-sensitive
    (fewer steps, or a Q3_K/mixed quant ladder via the check_qkv oracle).
  - **Baseline → best: 768.7s → 237.8s total wall (3.2x faster), peak VRAM
    10535 → 10779 MiB (both fit the 12 GB 3060).** Standard render config
    (`--video-frames 25 -W 480 -H 832 --steps 8 --cfg-scale 1.0 --diffusion-fa
    --clip-on-cpu --max-vram 9`, no `--vae-on-cpu`).
  - **Lever 1 (correctness): fps defaults to 25 (native save_fps) when `--audio`
    is given; the input conditioning WAV is auto-muxed into the output container**
    (trimmed to video duration) so clips come out viewable WITH sound — no manual
    ffmpeg. Verified: output webm carries a pcm_s16le 16 kHz audio stream.
    (`common.cpp resolve_and_validate`, `stable-diffusion.cpp generate_video`.)
  - **Lever 2 (THE big one): GPU VAE decode via spatial tiling, default-on for the
    avatar.** The full-clip Wan-VAE temporal decode OOMs a 12 GB card (~11.9 GiB);
    spatial tiling bounds per-tile activations. **VAE decode 569.7s (CPU) → 54.0s
    (GPU-tiled), 10.5x.** `generate_video` enables tiling by default when the VAE
    is on GPU and tiling wasn't requested (`--vae-on-cpu` / explicit `--vae-tiling`
    take precedence). Quality vs CPU baseline: mean PSNR 40 dB (above the ~37.6 dB
    VAE-vs-input ceiling), ac16 identical → visually equivalent.
  - **Lever 4 (opt-in): `--steps` now drives the DMD schedule (default 8).
    `--steps 6` → 197.4s total (-17%), stays coherent, but a different render than
    8-step; left non-default (6-vs-8 lip-sync is a human eyeball call).**
  - **Dead ends (recorded in PERF.md):** full-clip GPU VAE (OOM), GPU text encode
    (load-time OOM — umT5 6GB + DiT 8.5GB coexist upfront), larger VAE tiles
    (slower: 48-tile decode 73.9s vs 32-tile 54s), **CUDA-graph reuse across the 8 DiT
    steps (lap 03 — profiled dead, 72% compute / ~5 ms overhead)**. DiT sampling
    (~164s) remains the biggest cost but is compute-bound; only quality-sensitive
    levers left (fewer steps / quant ladder).
  - Checkpoint clips in `models/_perf/`: `BEST_8step_gpuvae_25fps_sound.webm`
    (best; lap03 8-step is bit-identical to it), `lap00_baseline.webm`,
    `lap01_gpuvae_tiled.webm`, `lap02_6steps.webm`, and the lap-03b A/B set
    `lap03_gate_mul.webm` (8) / `lap03_6steps.webm` (6) / `lap03_4steps.webm` (4).
    Tools: `tools/clip_compare.py` (per-frame PSNR + ac_lag16 coherence gate).

- 🟢 **AUDIO GRAFT LANDED + VALIDATED END-TO-END (2026-05-25 session 5). whisper encoder +
  AudioProjModel + per-block audio cross-attn + CLI `--audio` all wired; audio render runs
  to a clip and MEANINGFULLY DIFFERS from the no-audio render in the lower-face region.**
  - **VALIDATION (the required render):** baseline `longcat_noaudio_25f.webm` vs audio
    `longcat_audio_25f.webm` (`--video-frames 25 -W 480 -H 832 --steps 8 --cfg-scale 1.0
    --clip-on-cpu --vae-on-cpu --diffusion-fa --max-vram 9`, audio `speech_16k.wav`).
    `tools/audio_diff.py` (ffmpeg→PNG→numpy lower-face ROI diff):
    - **frame 0 IDENTICAL (diff 0.000)** — the cond/ref-image frame correctly receives NO
      audio (cond frames excluded from audio cross-attn).
    - **generated frames 1-24 differ**: MEAN full-frame |Δ| 7.61, MEAN lower-face |Δ| 7.31
      (0-255 scale), MAX lower-face 9.36 @ frame 18. The diff GROWS away from the cond anchor
      (f01≈3.9 → f16-24≈9.0-9.3) — the expected signature of audio progressively driving the
      generated frames. Lower-face diff tracks full-frame (mouth region is modulated).
    - audio-conditioned latent is coherent/finite (overall std 0.89, per-frame 0.53→0.99,
      nnan=0); per-frame stds diverge from no-audio (frames 3-6: 0.97/0.99/0.99/0.99 vs
      0.955/0.987/0.977/0.970). **Human eyeball needed for actual lip-sync quality** — clips
      served at http://10.0.0.208:8011/ (longcat_audio_25f.webm + longcat_noaudio_25f.webm).
    - Sampling 164s (vs 148s no-audio: +11% for the per-block audio cross-attn). CPU VAE
      decode ~12 min/25-frame clip (the wall-clock bottleneck; unrelated to audio).
  - **NEW FILE `src/longcat_audio.hpp`**: (a) minimal WAV reader (PCM16/PCM32/float,
    mono/multi, resample→16k); (b) `WhisperMel` = WhisperFeatureExtractor-equivalent
    128-bin log-mel (n_fft 400, hop 160, Hann, slaney mel filterbank, log10, clamp
    max-8, (x+4)/4) on CPU; (c) `WhisperEncoderRunner` = whisper-large-v3 ENCODER
    (conv1 s1 + conv2 s2 + posembed + 32 pre-LN layers + final LN), captures ALL 33
    hidden states → `[1280, T_enc, 33]`; (d) host glue `build_full_audio_emb`
    (group 33 hs into 5 feats: mean[0:8],[8:16],[16:24],[24:32],hs[32]; ENC_FPS 50 →
    fps 25 linear interp align_corners) + `build_proj_inputs` (the ±2 window +
    DiT vae_scale=4 windowing → proj1 in 32000 / proj1_vf in 51200).
  - **`src/longcat_avatar.hpp`**: `AudioProjModel` forward (`audio_proj.*`, dual-window
    MLP → `[768,32,N_t]`) now invoked in the runner; per-block `audio_cross_attn`
    (per-frame block-diagonal: each noise frame's spatial tokens attend its 32 audio
    tokens; cond frames get none; reuse `mod_norm_attn`, gated by `audio_adaLN`,
    `pre_video_crs_attn_norm` on x, audio-side prenorm = Identity). `audio_first`/
    `audio_latter` set per request on the runner.
  - **`src/stable-diffusion.cpp`**: avatar reuses `--audio-vae`/`audio_vae_path` slot as
    the whisper-encoder gguf path → builds `WhisperEncoderRunner` (prefix `audio_encoder`)
    instead of LTX audio VAE; `generate_video` computes mel→whisper→window once before
    the denoise loop, sets the runner's audio inputs, frees the whisper buffer. CLI
    `--audio <wav>` plumbed through `sd_vid_gen_params_t.audio_path`.
  - **Verified spec facts**: avatar-1.5 uses **WHISPER** (not wav2vec2; that's v1.0);
    config.json: audio_window 5 / audio_block 5 / audio_channel 1280 / intermediate 512 /
    output_dim 768 / context_tokens 32 / vae_scale 4 / audio_prenorm False; save_fps 25 /
    audio_stride 1. The "audio_block=5" task hint is the GROUPED-FEATURE count (S dim),
    NOT a single intermediate layer tap — the reference taps ALL 33 hidden states and
    groups them into 5. Whisper run needs **`audio_vae_path` = whisper gguf** on CLI.
  - **mel VALIDATED**: `tools/check_mel.py` mirrors the C++ math in numpy and compares to
    `transformers.WhisperFeatureExtractor` on the test wav → **max abs diff 0.019, mean 5e-6**
    (essentially bit-identical; the 0.019 max is one edge frame from the global-max clamp
    differing because HF pads to 30s). The C++ mel frontend is correct.
  - **BUG FOUND+FIXED during validation**: audio_cross_attn's kv split used the wrong
    `ggml_permute` arg order (`0,2,3,1` instead of `0,3,1,2`) → graph-build reshape assert.
    `ggml_permute(a, p0,p1,p2,p3)` = "src dim i goes to dst position p_i"; mirror
    text_cross_attn's `0,3,1,2`. Fixed. Also the audio<->frame alignment uses a FIXED
    save_fps=25 (model constant), independent of output --fps.
  - **RUN INVOCATION**: pass the whisper gguf via `--audio-vae` and the wav via `--audio`:
    `sd-cli -M vid_gen -m <dit> --t5xxl <umt5> --vae <wanvae> --audio-vae <whisper>
     --init-img <png> --audio <wav> --video-frames 25 -W 480 -H 832 --steps 8 --cfg-scale 1.0
     --clip-on-cpu --vae-on-cpu --diffusion-fa --max-vram 9 -o out.webm`. Verified the whole
    audio pipeline runs: mel 400 frames → whisper [1280,200,33] → window first[32000,1]
    latter[51200,6] N_t=7=latent T. (CPU VAE decode of 25 frames ≈ 12 min.)
  - **OPEN ASSUMPTIONS / residual risk**: (1) hidden_states[32] post-final-LN assumption
    (HF returns last_hidden_state post-LN as hs[-1]); (2) the 8-entry latter-window emit
    order (first/middle/last) vs the torch concat — matched by inspection, not numerically
    diffed against torch (the ref venv lacks the avatar pipeline deps); (3) whisper encoder
    output not diffed against torch WhisperModel.encoder (graph built from HF spec). Validation
    render + audio-vs-no-audio lower-face diff results recorded once decode completes.


- 🟢 **DiT BLOCK MATH PROVEN CORRECT via an INDEPENDENT standalone oracle; the prior "patch_embed
  divergence" was a HARNESS BUG, not a C++ bug. Bug is NOT in the DiT body — it is in the
  generated-frame LATENT MAGNITUDE (std ~3.2 vs expected ~1) → still noise (2026-05-25 session 4).**

  **What was done (rigorous, sequenced C++ dump → torch oracle → diff):**
  - Instrumented `src/longcat_avatar.hpp` to dump (env `LONGCAT_DUMP_DIR`) the DiT inputs
    (`in_x`/`in_timestep`/`in_context`) and taps (patch_embed, t_embed, y_embed, block0/1,
    final_layer, output) plus block-0 sub-steps (modulation chunks, x_m, attn q/k/v pre/post-rope,
    attn_out, ffn gu/out, residual stages). Raw-f32 dumper added to `ggml_extend.hpp`'s
    `debug_tensors` loop (writes `<dir>/<tap>.bin`: int64 ndim + dims (ggml ne order) + f32). All
    taps gated behind `LONGCAT_DUMP_DIR` → zero graph ops in production.
  - Validation venv `.venv-oracle` (CPU torch 2.6 + diffusers/transformers; triton/flash/xformers/
    audio_process stubbed). `tools/dit_oracle.py` = full-model oracle (int8 DiT + DMD-lora);
    `tools/dit_oracle_block0.py` = block-0-only; `tools/check_rope.py`, `tools/check_qkv.py`,
    `tools/check_attn.py` = standalone numeric checks.

  **Forward-order diff (C++ vs torch, identical dumped inputs):**
  ```
  patch_embed cos 1.000  | t_embed cos 1.000 | y_embed cos 1.000   (embedders CORRECT)
  block0      cos 0.68   (full-model oracle)  <-- looked like first divergence
  ```
  This pointed at block 0, so block-0 was bisected to sub-steps. The modulation chunks
  (shift/scale/gate_msa+mlp) all matched cos 1.000 (incl. the genuinely-large scale_mlp mean ~14,
  gate_mlp std ~22 — real DMD behaviour). x_m_attn matched cos 1.000. But the full-model oracle
  reported attn_out cos 0.07 and block0 std 24.

  **THE FULL-MODEL ORACLE WAS WRONG.** A standalone re-derivation (`tools/check_qkv.py`) that loads
  ONLY block-0's weights from the int8 shard, folds the DMD-lora exactly (n_seperate=3 qkv,
  alpha_scale 0.5), and runs the reference block-0 math on the C++-dumped `x_m`, gives:
  ```
  q_prerope  cos 0.99981   v cos 0.99936          (qkv split + q/k_norm CORRECT)
  full self-attn (cond-split + 3D-RoPE + proj) -> attn_out  cos 0.99963
  ffn_gu     cos 0.99970   ffn_out cos 0.99951    (SwiGLU CORRECT; gu std ~600, out std ~3900 GENUINE)
  after_attn cos 0.99991 ; tap_block0 (after_ffn) cos 0.99983  (std ~322k = REAL residual blow-up)
  ```
  RoPE was separately confirmed via `tools/check_rope.py`: applying the verbatim reference
  `rope_3d` to the C++ pre-rope q reproduces the C++ post-rope q at cos 1.000 (and ≠ NeoX 0.44),
  so the interleaved 3D-RoPE + {44,42,42} axis split is correct.
  **⇒ The C++ DiT block (modulate / fused-qkv split / qk-RMSNorm / interleaved 3D-RoPE /
  num_cond_latents two-pass self-attn / text cross-attn / SwiGLU FFN / per-(B,T) gates) is
  numerically correct to cos 0.999+.** The large internal magnitudes (residual std → 1e5–1e6) are
  GENUINE model behaviour — so the prior sessions' ffn.w2 Q8_1-overflow guard (×1/256) and the
  flash-attn F16 v-overflow guard (kv_scale 1/256) are CORRECT and load-bearing, NOT masking a bug.
  The full-model `dit_oracle.py` reference values are unreliable (its `block_forward` monkeypatch /
  runtime-LoRA path under CPU autocast diverges from the converter-folded weights) — trust
  `check_qkv.py` (independent weight load), not `dit_oracle.py`, for the reference.

  **patch_embed was a REAL bug, already FIXED (prior session) and now CONFIRMED:** the token-flatten
  permute `ggml_ext_torch_permute(x, 2,0,1,3)` (line ~555) yields token order (T,h,w) w-innermost —
  oracle patch_embed cos 1.000. The prior session's "patch_embed cos 0.006 FIRST DIVERGENCE" report
  was from the buggy full-model oracle BEFORE the fix; the fix is in tree and validated.

  **ACTUAL REMAINING BUG = generated-frame latent magnitude.** Validation render
  (`--video-frames 13 --steps 8`, fixed patch_embed, `models/_testinputs/longcat_validate_13f.webm`):
  predecode latent frame 0 (cond) std 0.53 (correct, matches VAE latent scale), but generated
  frames 1+ std **~3.2** (should be ~1). Frame coherence: frame 0 ac_lag16=0.875 (faithful
  portrait); generated frames 03/06/12 ac_lag16 = 0.06/0.07/0.02 = STILL NOISE. So the DiT block
  math is right but the *output latent for noise frames is ~3–6× too large* → decodes to noise.
  **NEXT (next agent): the bug is downstream of the per-block math — candidates, cheapest first:**
  (1) the DMD/flow-match SAMPLER step in `stable-diffusion.cpp` (does the C++ sigma→x update match
  `generate_ai2v` + `FlowMatchEulerDiscreteScheduler` with `use_distill`? a wrong x0/velocity
  interpretation or a missing per-step rescale would over-scale the generated latent while leaving
  the masked cond frame intact); (2) the final_layer + unpatchify on the *generated* frames
  (validate `tap_output` vs reference across ALL frames — block-0 is proven, but accumulation over
  48 blocks or the final modulation could mis-scale); (3) verify the C++ feeds the correct per-step
  timestep to the noise frames (the dumped in_timestep noise value tracked the DMD sigma×1000, but
  confirm the sampler's noise/sigma blend). Build a reliable FULL-DiT reference by FIXING
  `dit_oracle.py` to match `check_qkv.py` (replace the runtime-LoRA path with the converter-fold,
  drop the `block_forward` monkeypatch in favour of the standalone math) OR diff the C++ `tap_output`
  against a 48-block standalone reproduction (needs the int8 shards streamed block-by-block to fit RAM
  — the full bf16 construct OOMs at ~27 GB on this 31 GB box; load per-block on demand).
  Tools (kept): `tools/{dit_oracle,dit_oracle_block0,check_rope,check_qkv,check_attn}.py`
  (`.venv-oracle` CPU torch). Re-dump anytime with `LONGCAT_DUMP_DIR=<dir>` on the sd-cli run
  (env-gated taps in `longcat_avatar.hpp::LongCatAvatar::forward`: in_x/in_timestep/in_context +
  patch_embed/t_embed/y_embed/block0/block1/final_layer/output — zero graph ops when unset; the
  scratch `models/_dump` was deleted). `check_qkv.py` is the RELIABLE standalone reference (loads
  block-0 from the int8 shard + folds the DMD lora; no full-model construct). Latest validation
  clip: `models/_testinputs/longcat_validate_13f.webm` (generated frames still noise, latent std ~3.2).

- 🟡 **FLASH-ATTN NaN FIXED + cond-split implemented at 480p (2026-05-25 session 2). Latent
  finite under `--diffusion-fa`; cond frame-0 still faithful (PSNR 36.4). GENERATED frames
  (1+) ARE STILL NOISE. Leads 1–3 worked through; the residual is a DiT-body numerical bug
  that structural stats can't localize → next step is a numerical reference harness (user-gated).**
  - **TWO real bugs found & fixed this session (both in `src/longcat_avatar.hpp`):**
    1. **FLASH-ATTN F16 OVERFLOW (was producing a 100%-NaN latent whenever `--diffusion-fa`
       was passed).** `ggml_ext_attention_ext`'s flash path casts k/v to **F16** before
       `ggml_flash_attn_ext`. q/k are RMS-normed (bounded ~1) but **v carries the large
       additively-growing residual-stream magnitude** (up to ~1e6 over 48 blocks) — casting
       it straight to F16 overflows (>65504 → inf → NaN). Same *class* as the ffn.w2 Q8_1
       overflow. **FIX:** pass `kv_scale = 1/256` to the self-attn (and, defensively, text
       cross-attn) `ggml_ext_attention_ext` calls when flash is on — it shrinks k/v before
       the F16 cast and rescales the output back (softmax is invariant to a uniform k scale;
       the v scale is undone on the output → mathematically identical, F16-safe). Non-flash
       path keeps F32 scores (`kv_scale==1`, no-op). **This is why the prior session's
       "PSNR 36.4" validation only held without flash — flash silently NaN'd the whole latent;
       the prior `--diffusion-fa REMOVED → still NaN` note predated the ffn.w2 fix.** Verified:
       5f & 13f under `--diffusion-fa` now give finite latents (`nnan=0`, std 0.94/1.08),
       bit-matching the non-flash latent stats.
    2. **num_cond_latents self-attn split (LEAD 1) — now implemented CORRECTLY at 480p.** The
       prior dense `[N,N]` additive mask (capped at n_token≤6000, so SKIPPED at 480p's ~10920
       tokens, and it would have forced the non-flash 15 GB-score path anyway) is **replaced by
       the reference's exact two-pass split** (`avatar/attention.py` `num_cond_latents==1`):
       RoPE the full q/k, then run `attn(q_cond × {k,v}_cond)` (cond tokens see only cond) and
       `attn(q_noise × {k,v}_full)` (noise tokens see all) and concat. No O(N²) memory, keeps
       flash-attn, works at any res. The text cross-attn cond-zeroing was already correct.
  - **RESULT: generated frames are STILL colorful noise.** Frame-coherence assessment (480×832,
    13f, 8-step DMD, `--diffusion-fa`, clip `models/_testinputs/longcat_avatar_faflashfix_13f.webm`;
    5f variant `..._5f.webm`):
    - **Frame 0 (held cond latent): PSNR 36.42 dB vs input** — sharp faithful portrait (unchanged,
      pinned by the latent denoise_mask; proves VAE + unpatchify spatial mapping correct).
    - **Generated frames 1+: NOISE.** Long-range structure metric `ac_lag16` (natural img ≈0.84,
      white noise ≈0): frame 0 = 0.84; generated frames decay 0.26 → 0.12 → 0.02 → ~0.0 as you
      move away from the cond anchor. (BEWARE `ac_lag1` ≈ 0.96 even on these noise frames — that's
      RGB-subpixel/codec low-pass, NOT real structure; the lag-16 metric and the eyeball both say
      noise.) FFT low-freq fraction drops 0.96 → ~0.5. Visually: frame 2 has a faint luminance
      gradient (residual conditioning) but no recognizable content; frames 3+ are full noise.
    - **The temporal decay from the cond frame** (frame 2 retains the most structure, collapsing
      to pure noise by frame 5) shows the conditioning IS partially propagating — the cond split,
      denoise_mask, and timestep-0 anchor are doing *something* — but the noise-frame denoising
      itself is wrong.
  - **Leads worked through (per the handoff order):**
    - **LEAD 1 (cond split skipped at 480p): ADDRESSED — was a real gap, now implemented
      correctly (two-pass). Did NOT by itself produce coherent frames.** (It did surface +
      fix the flash-attn NaN.)
    - **LEAD 2 (3D-RoPE token-order / position assignment): VERIFIED CORRECT by code reading.**
      Token flatten order after PatchEmbed3D is `(T,h,w)` with w innermost (`patch_embed`
      reshape) ≡ reference `rearrange(freqs,"T H W D -> (T H W) D")`. `gen_vid_ids` assigns
      positions `idx = t*h_len*w_len + h*w_len + w` — same order. Per-axis dim split
      `dim_t=44/dim_h=42/dim_w=42` = reference `head_dim-4*(hd//6) / 2*(hd//6) / 2*(hd//6)`,
      concatenated t→h→w at the same offsets. Interleaved (GPT-J) convention matches
      `repeat(...,r=2)`+`rotate_half` adjacent-pair. No discrepancy found.
    - **LEAD 3 (DMD sampling / sigma / flow-match): VERIFIED CONSISTENT.** `build_longcat_dmd_sigmas`
      sigmas match Python; FLOW `sigma_to_t = sigma*1000` feeds the DiT a 0–1000 timestep
      (reference `sigmas*num_train_timesteps`); per-frame `process_timesteps` sets ONLY the cond
      frame to t=0 (others get the step t), and the DiT broadcasts per-frame t correctly; the
      denoise-mask blend pins frame 0. Shares the in-tree Wan video sampler (which works), so the
      step math is unlikely to be the culprit.
    - LEAD 4 (audio cross-attn stubbed): still stubbed, correctly deprioritized — absent audio
      should give a still/coherent person, not noise.
  - **NEXT STEP (per handoff: STOP rather than thrash): a NUMERICAL REFERENCE HARNESS.** The
    symptom (cond anchor perfect, generated frames noise, partial near-anchor propagation) points
    to a DiT-body numeric/graph bug that survives all three structural leads — most plausibly in
    the adaLN per-(B,T)-frame modulation broadcast (`modulate`/`gate_add` reshape `n_token/T`
    over spatial tokens), the t-embed/per-frame-timestep expand, the SwiGLU/ffn, or the y_embedder
    text path — i.e. something that ONLY corrupts the noise frames while leaving the masked/pinned
    cond frame intact. Recreating the torch venv (heavy, user-gated) to dump reference DiT
    intermediate activations (post patch-embed, post block-0, post final-layer) and diff against
    the C++ taps (the `capture_tensor`/`debug_tensors` facility is still in tree) is the
    highest-ROI next move. Do NOT add more speculative C++ levers without that numerical anchor.
  - **Debug aids:** `[DBG predecode latent]` stats log stays in `decode_video_outputs`. The
    self-attn now exposes per-call `kv_scale` (= the flash F16 guard).
  - Best clips: `models/_testinputs/longcat_avatar_faflashfix_13f.webm` (target config),
    `..._5f.webm`. Prior nan-fix clip `..._nanfix_5f.webm` was non-flash-only.

- ✅ **DiT NaN FIXED (2026-05-25). Latent is finite; frame-0 resembles the input portrait.**
  - **First-NaN localized** by tapping every sub-op of blocks 0/1 (env-free, via the
    GGMLRunner `capture_tensor`/`debug_tensors` facility; the print loop in
    `ggml_extend.hpp` now emits mean/std/min/max/nnan stats instead of a full dump).
    Bisect result (step-0, q4_k): patch_embed/t_emb/y_emb/adaLN/self-attn/text-cross all
    FINITE. The residual stream grows **additively** (~+75k std per block — EXPECTED:
    `mod_norm_*` LayerNorms renormalize each block's modulation input, so per-block
    contribution is bounded while the residual accumulates to ~6e6 over depth; the
    reference survives this in **bf16**, which has F32 exponent range). The **first NaN
    appears in `ffn.w2`** of block 1: its input `silu(w1)*w3` is FINITE at ~1e5
    (`b1_11b_gate_times_up` max 98325, 0 NaN) but `ffn_w2->forward(...)` comes out 100% poisoned.
  - **ROOT CAUSE = F16 overflow in the Q8_1 activation quantization of the Q4_K×large-activation
    matmul.** ggml-cuda routes `Q4_K_weight × F32_activation` (batch>MMVQ_MAX) through **MMQ**,
    which quantizes the activation to **Q8_1** (per-32-block `d=amax/127` + a **sum field
    `s = (Σ q_i)·d` stored in F16**). With activation ~1e5, `s ≈ 32·127·(1e5/127) ≈ 3.2e6`
    **overflows F16 (>65504) → inf → NaN**. NOT a structural/shape/RoPE/eps/modulation bug —
    the adaLN bias (incl. scale_mlp) is ~0 and BIT-IDENTICAL between q4_k and q8_0 gguf, and the
    big `scale_mlp` (mean ~9.3 at t=1000) is genuine model behaviour the reference runs in bf16.
    `ggml_mul_mat_set_prec(F32)` does NOT help — it only affects cuBLAS accumulation, not the MMQ
    Q8_1 quant (and `GGML_CUDA_FORCE_CUBLAS` is compile-time-global, so not an option).
  - **FIX (`src/longcat_avatar.hpp`):** pre-scale `ffn.w2`'s input down by **1/256** and the output
    back up by 256 (`Linear(..., scale=1/256)` → `ggml_ext_linear` does `x*=s` before the matmul,
    `out*=1/s` after — mathematically identical, but the Q8_1 sum field lands at ~12.5k, safely
    in F16 range). Robust across all 48 blocks because the FFN-branch magnitude is gated by
    scale_mlp/the SwiGLU and does NOT grow with the additive residual depth. Also set
    `force_prec_f32=true` on every avatar Linear (F32 matmul accumulation, for bf16 parity — kept
    even though it isn't the load-bearing fix). Only `ffn.w2` sees an unbounded (non-renormed)
    activation; all other matmuls take renormalized/bounded inputs.
  - **VALIDATION** (`--video-frames 5 --steps 8 --t5xxl ... --vae ... --init-img girl_480x832.png
    --cfg-scale 1.0 --clip-on-cpu --vae-on-cpu --diffusion-fa --max-vram 9`):
    pre-decode latent **numel=199680 mean=0.0078 std=0.935 min=-5.20 max=5.78 nnan=0** (was 100% NaN).
    Decoded clip **frame 0 vs input: PSNR 36.41 dB** (px mean 108.2/std 93.7 vs input 109.4/93.6;
    VAE ceiling 37.59 dB) — a sharp, faithful portrait of the girl. Winning clip saved at
    `models/_testinputs/longcat_avatar_nanfix_5f.webm`. Generated frames 1+ are still incoherent
    (colorful noise) — that's the next correctness pass (RoPE/cond-quality + the stubbed audio
    cross-attn), explicitly out of scope for this milestone.
  - **Debug aids:** all in-graph `capture_tensor` taps were removed after bisecting; the
    `[DBG predecode latent]` stats log in `decode_video_outputs` stays. The `[DBG tap]` stats
    print in `ggml_extend.hpp`'s `debug_tensors` loop is a generic improvement (dormant unless a
    `capture_tensor` is added back) — re-add taps + rebuild to bisect future numeric issues.

- 🔴 **(SUPERSEDED) WHITE-CLIP ROOT CAUSE = DiT FORWARD EMITS ALL-NaN LATENT (2026-05-25). NOT the VAE.**
  Discriminated Hypothesis A vs B per the handoff. **Hypothesis B (VAE temporal decode / latent
  normalization) is RULED OUT; Hypothesis A (DiT emits a degenerate latent) is CONFIRMED.**
  - **VAE multi-frame temporal decode is CORRECT.** Torch oracle encodes the portrait as a 25-frame
    video → 7 latent frames (`tools/vae_oracle_multiframe.py`); feeding that **5D** latent
    (`models/_testinputs/oracle_latent_mf5.bin`, ne `[w,h,T,C,1]`) through sd.cpp's WanVAE decode
    produces all 25 video frames faithfully: `decode(oracle z)` shape `[480,832,25,3,1]` mean 0.4299
    std 0.3664, **frame-0 PSNR 37.59 dB vs input** (= torch ceiling), px mean 109.5/std 93.5 — a
    perfect portrait (`models/_testinputs/mf_recon_f0.png`). latents_mean/latents_std in
    `src/wan.hpp get_latents_mean_std` (16-ch Wan2.1 branch) match the LongCat VAE config.json
    BIT-EXACT, and encode→diffusion_to_vae round-trips. So the VAE + normalization path is sound.
    - GOTCHA found: the **standalone roundtrip harness** must be fed a **5D** latent. A 4D
      `[w,h,T,C]` latent trips `WanVAERunner::_compute`'s `if (z.dim()==4) input = z.unsqueeze(2)`
      which mangles a multi-frame latent (T merges into C) → `GGML_ASSERT(a->ne[2]==b->ne[2])` crash
      in the `middle.1` AttentionBlock 1×1 Conv2d (a(w)=[1,1,384,1152] vs b(x)=[60,104,2688,1],
      2688=384·7). This is a HARNESS-INPUT artifact only — **production always passes a 5D latent**
      `[W,H,T,C,B]` (from `generate_init_latent`), so the `unsqueeze` never fires in the real path
      and the bug is irrelevant to the white clip. (If you want the harness robust, fix the 4D
      unsqueeze-then-decode branch; not needed for the port.)
  - **DiT forward is the culprit.** Added a temp pre-`decode_first_stage` latent-stats log in
    `src/stable-diffusion.cpp::decode_video_outputs` (`[DBG predecode latent]`). A short
    `--video-frames 5 --steps 8` ai2v render gives a latent that is **100% NaN**: `numel=199680
    nnan=199680`, frame 0 mean=nan, frame 1 mean=nan. The all-NaN latent VAE-decodes to the
    saturated/white frames (and the NaN even poisons the held cond frame 0, because the
    denoise-mask blend `denoised*mask + init*(1-mask)` does `NaN*0 = NaN` for the mask=0 cond frame).
  - **It is NOT the missing text encoder.** The handoff's smoke command (and the original white
    render) omitted `--t5xxl`, so `get_learned_condition` returned in 0.00s and the DiT ran with a
    **null context** (`make_optional_input` returns nullptr on empty) — a guaranteed NaN. But
    re-running WITH `--t5xxl models/longcat-umt5-xxl-q8_0.gguf` (real context, `get_learned_condition`
    16.17s) STILL yields a 100%-NaN latent. So a valid context does not fix it — the NaN is intrinsic
    to the DiT compute graph. (NOTE for future runs: **always pass `--t5xxl`**; the DiT — like the Wan
    path — runs its y_embedder + text cross-attn unconditionally and has no null-context fallback.)
  - **STOPPING SHORT of rewriting the DiT, per handoff instruction.** This needs its own agent.
    Repro (short, ~3 min sample on the 3060): `iter.sh`-built `sd-cli -M vid_gen
    -m models/longcat-avatar-1.5-dit-dmd-q4_k.gguf --t5xxl models/longcat-umt5-xxl-q8_0.gguf
    --vae models/longcat-wan-vae-f16.gguf --init-img models/_testinputs/girl_480x832.png
    -W 480 -H 832 --video-frames 5 --steps 8 --cfg-scale 1.0 --clip-on-cpu --vae-on-cpu
    --diffusion-fa --max-vram 9 -p "a person talking"` → watch for `[DBG predecode latent] ... nnan=`.
    NEXT-AGENT LEADS (cheapest first): (1) bisect the NaN by tapping intermediate tensors
    (after patch_embed, after block 0, after final_layer) — the runner builds one graph, so add a
    debug early-return or a NaN-abort op; (2) test whether the **q8_0 intermediate** DiT
    (`models/longcat-avatar-1.5-dit-dmd-q8_0.gguf`) also NaNs — if q8 is clean, the q4_k requant
    (or the DMD-LoRA fold) produced NaN/inf weights → a CONVERTER fix, not a DiT fix; (3) scan the
    gguf weights for NaN/inf — DONE for the F16/F32 norms/biases (`uv run --with gguf` over
    `longcat-avatar-1.5-dit-dmd-q4_k.gguf`): **0 NaN/inf in all 1608 F16/F32 tensors** (the Q4_K
    weight blocks can't represent inf/NaN, so weights are very unlikely the cause); (4) flash-attn
    is NOT implicated — re-ran with `--diffusion-fa` REMOVED (non-flash, materialized scores) and the
    latent is STILL 100% NaN (`nnan=199680`). So the NaN is independent of the attention kernel
    (both paths NaN), pointing to a graph/numerics bug that fires regardless of attention. Suspect
    surfaces in `src/longcat_avatar.hpp` (in likely order): qk-RMSNorm over head_dim, the adaLN
    `scale+1` modulation reshape/broadcast (`modulate`/`gate_add` reshape `n_token/T`), the 3D-RoPE
    apply (interleaved), `final_layer` modulation, SwiGLU, or the DMD timestep/sigma feeding
    `ggml_ext_timestep_embedding`. Bisect by tapping intermediates (best ROI).
  - **Debug aids left in tree:** `tools/vae_oracle_multiframe.py` (multi-frame torch oracle +
    5D/4D `.bin` dump) and the `[DBG predecode latent]` stats log in
    `src/stable-diffusion.cpp::decode_video_outputs` (REMOVE before shipping; harmless, builds green).
    The throwaway ggml im2col debug print was reverted.
- ✅ **VAE BLOCKER RESOLVED (2026-05-25): Wan VAE round-trips faithfully.** Root cause was a
  **single converter bug in `tools/convert_wan_vae.py`**: the `RMS_norm` *gamma* (scale) tensors
  were emitted with their diffusers trailing-singleton shapes (resnet `[C,1,1,1]`, **attention
  `middle.1.norm.gamma` `[C,1,1]`**) instead of flat 1-D `[C]`. sd.cpp's `WAN::RMS_norm::init_params`
  allocates a 1-D `[C]` tensor, so the 3-D attention gamma landed as ggml ne `[1,1,C,1]` and the
  loader's shape check **rejected the whole VAE** (`tensor '...middle.1.norm.gamma' has wrong shape:
  got [1,1,384,1], expected [384,1,1,1]` → `load tensors from file failed`). The avatar pipeline then
  ran with an **unloaded / zero VAE** → the decode collapsed to a saturated/blank image. Fix: the
  converter now `arr.reshape(-1)` for every `.gamma` (one-line change). **The 5D→4D `CausalConv3d`
  weight reshape was CORRECT** (`[out,in,kt,kh,kw]→[out*in,kt,kh,kw]`, out-major) — it exactly matches
  ggml `ggml_conv_3d`'s expected `[OC*IC,...]` packing (OC outer / IC inner), verified by code reading
  *and* numerically. No C++ change needed — `src/wan.hpp` WanVAE/Encoder3d/Decoder3d/CausalConv3d are
  all correct.
  - **Validation (CPU, f16 VAE):** standalone `sd-vae-roundtrip` harness
    (`examples/vae-roundtrip/`) encodes `models/_testinputs/girl_480x832.png` and decodes both its
    own latent and the torch oracle's latent.
    - sd.cpp encode `mu`: mean −0.1018 / std 1.1370 ≡ torch oracle latent (mean −0.1019 / std 1.1368).
    - sd.cpp `decode(encode(png))` vs **torch oracle recon: PSNR 64.93 dB** (MSE ≈ 0; f16-vs-f32).
    - sd.cpp `decode(torch oracle latent)` vs torch oracle recon: **PSNR 67.44 dB**.
    - sd.cpp recon vs **input image: PSNR 37.59 dB — identical to torch's own 37.59 dB ceiling.**
    - Output is a faithful portrait (NOT blank): px mean 109.5 / std 93.5. PNG
      `models/_testinputs/sdcpp_recon_self.png`.
  - **Oracle + tooling (small CPU torch venv `.venv-vae`):** `tools/vae_oracle.py` (torch ground
    truth), `tools/dump_vae_bins.py` (PNG/latent → sd.cpp `.bin`), `tools/compare_vae_recon.py`
    (PSNR/MSE + PNG). VAE gguf regenerated: `models/longcat-wan-vae-f16.gguf` (194 tensors, 253.8 MB).
  - **NEXT:** the white-clip blocker is gone; re-render the full ai2v+DMD pipeline (the DiT/DMD/RoPE
    work was never the cause) and judge coherence with a *loaded* VAE.
- ✅ **Build**: `longcat-avatar.cpp` fork builds CUDA (sm_86) green via
  `kobbler/docker/longcat-avatar-dev/iter.sh`. sd-cli + sd-server run, Wan video path present.
- ✅ **DiT converter**: `tools/convert_longcat_avatar.py` (int8-dequant → gguf, verbatim
  names under `model.diffusion_model.`). Artifacts on SSD `models/`:
  - `longcat-avatar-1.5-dit-q8_0.gguf` (16.85 GB, requant intermediate)
  - `longcat-avatar-1.5-dit-q4_k.gguf` (**8.94 GB**, runtime target — fits 12GB with ~3GB
    headroom for activations + offloaded encoders). 8.893 GB is Q4_K weights; F16 norms/biases
    are only 10 MB (nothing to trim there). uniform Q4_K is the size floor.
- ✅ **Supporting converters** (umT5 / Wan VAE / whisper encoder): DONE + validated against
  sd.cpp's reconstructed name sets. Details in `tools/SUPPORTING-MODELS-NOTES.md`. Artifacts:
  - `models/longcat-umt5-xxl-q8_0.gguf` (6.04 GB, 242 tensors) — prefix-only rewrite
    (`text_encoders.t5xxl.transformer.`); loads via sd.cpp `T5CLIPEmbedder`/`T5Runner` with
    `is_umt5=true` (auto from the `text_encoders.t5xxl` match). Tool: `tools/convert_umt5.py`.
  - `models/longcat-wan-vae-f16.gguf` (253.8 MB, 194 tensors) — REQUIRED a diffusers
    `AutoencoderKLWan` → ComfyUI-Wan remap (sd.cpp's `WanVAERunner` wants the original layout;
    NOT stock-loadable). 5D convs reshaped `[out,in,kt,kh,kw]→[out*in,kt,kh,kw]` for sd.cpp's
    4D `CausalConv3d`. Load via `--vae` (lands as `first_stage_model.*`). Tool: `tools/convert_wan_vae.py`.
  - `models/longcat-whisper-v3-encoder-f16.gguf` (1274 MB, 487 tensors, 0 decoder) — HF names
    under `audio_encoder.` prefix. Tool: `tools/convert_whisper_encoder.py`. **Encoder config**:
    32 layers, d_model 1280, 20 heads (head_dim 64), ffn 5120, num_mel_bins 128, conv1 128→1280
    k3s1p1, conv2 1280→1280 k3 **s2** p1, embed_positions [1500,1280] (max_source_pos 1500),
    pre-LN, gelu, post `layer_norm`. `self_attn.k_proj` has NO bias. AudioProjModel taps
    `audio_block=5` intermediate encoder-layer hidden states. (Config also in gguf KV `whisper.encoder.*`.)
  - Full runtime set on SSD ≈ 16.5 GB (DiT Q4_K 8.94 + umT5 6.04 + VAE 0.25 + whisper 1.27);
    umT5 + whisper offload to CPU at runtime so VRAM ≈ DiT + activations + VAE.
- ✅ **Arch registration**: DONE + COMPILES GREEN. `VERSION_LONGCAT_AVATAR` in `src/model.h`
  (enum, `sd_version_is_longcat_avatar()`, `sd_version_is_dit()`), detection branch in
  `src/model.cpp` get_sd_version() (keyed on `...blocks.0.audio_cross_attn.q_linear.weight`), and
  version string "Longcat-Video-Avatar" in `src/stable-diffusion.cpp`. At runtime the version now
  instantiates the `LongCatAvatarModel` (see below) and runs a full vid_gen forward.
- ✅ **C++ avatar base DiT** (`src/longcat_avatar.hpp` + wrapper + instantiation): LANDED &
  RUNS (2026-05-25). **Builds green, loads all 1608 gguf DiT tensors (no missing/extra), and a
  full vid_gen forward runs end-to-end to a written `.webm`** (text-encode → 48-block DiT sample →
  Wan-VAE decode, EXIT=0). Smoke run: `--video-frames 5 -W 480 -H 832 --steps 2 --clip-on-cpu
  --vae-on-cpu --diffusion-fa --max-vram 9` → 156s wall (decode is 108s of that), peak DiT VRAM
  8539 MiB. As of 2026-05-25 the forward now also does DMD-distilled few-step + ai2v ref-image
  conditioning + correct interleaved RoPE (see below); coherence pending human eyeball.
  - `src/longcat_avatar.hpp` (namespace `LONGCAT_AVATAR`): `LongCatAvatarSingleStreamBlock`
    (self-attn fused-qkv + qk-RMSNorm + 3D RoPE gated by adaLN; UNGATED text cross-attn after
    affine `pre_crs_attn_norm`; SwiGLU ffn gated by adaLN; `mod_norm_attn`/`mod_norm_ffn` are
    non-affine LN), `FinalLayer`, `PatchEmbed3D`, `LongCatAvatar` body, `LongCatAvatarRunner`.
  - `src/diffusion_model.hpp`: `LongCatAvatarModel : public DiffusionModel` (mirrors WanModel,
    delegates to the runner; `get_adm_in_channels()` = 768).
  - `src/stable-diffusion.cpp`: instantiation branch in `init_diffusion_model()` (umT5 via
    `T5CLIPEmbedder(...,is_umt5=true)` + `LongCatAvatarModel`); plus the video/VAE plumbing the
    avatar shares with Wan — routed `sd_version_is_longcat_avatar(version)` into: WanVAERunner
    selection, FLOW_PRED, `default_flow_shift = 7.0` (FlowMatchEuler), `down_factor = 2`,
    `video_frames_to_latent_frames`/`latent_frames_to_video_frames` (4× temporal),
    `sd_version_supports_video_generation`.
  - **PatchEmbed3D**: gguf `x_embedder.proj.weight` is `[2,2,16,4096]` (Conv3d with temporal
    patch=1 → Conv2d-shaped `[pw,ph,in,out]`), NOT the generic `Conv3d` 4D `[kw,kh,kt,in*out]`
    layout — so a dedicated block allocates the weight in that exact shape and runs a per-frame
    `ggml_conv_2d`. Latent comes in as `[W,H,T,C]`; permuted to `[W,H,C,T]` for the conv.
  - **timestep broadcast**: sampler passes a single (per-batch) timestep; PyTorch expands to
    `[B,T]` over latent frames. We broadcast the t-embed across T (`ggml_repeat_4d`) in forward.
- ✅ **3D RoPE — VALIDATED & FIXED (2026-05-25)**: was `rope_interleaved=false`, now `true`.
  The avatar's `modules/rope_3d.py` builds freqs with `repeat(freqs, "... n -> ... (n r)", r=2)`
  (→ `[f0,f0,f1,f1,...]`) and rotates with `rotate_half` = `(... (d r) -> ... d r, r=2)` which
  pairs **adjacent** dims `(2i, 2i+1)`. That is the INTERLEAVED (GPT-J) convention — the earlier
  "GPT-NeoX/non-interleaved" note in this file was WRONG. Numerically validated in Python:
  the Wan helper's per-axis `omega[j]=theta^(-2j/dim)` angles are bit-identical (max abs diff
  1e-16) to the avatar's, across all three axes, with the `{44,42,42}` split = `dim_t=44/
  dim_h=42/dim_w=42`. Token flatten order `(T,h,w)` and positions `0..len-1` also match
  (`gen_vid_ids` ≡ avatar `linspace(...,endpoint=False)`). One-line fix in `self_attn`.
- 🟡 **AUDIO PATH — LOADED BUT STUBBED**: `audio_cross_attn.*`, `audio_adaLN_modulation.*`,
  `pre_video_crs_attn_norm.*`, and `audio_proj.*` tensors are all declared/loaded (so all 1608
  tensors resolve) but the audio cross-attn is NOT wired into the per-block forward
  (`TODO(audio)` in `LongCatAvatarSingleStreamBlock::forward`). `audio_proj` block dims are
  declared from the observed gguf shapes (proj1 `Linear(32000,512)`, proj1_vf `Linear(51200,512)`,
  proj2 `Linear(512,512)`, proj3 `Linear(512,24576)`, norm `LayerNorm(768)`) — but the
  AudioProjModel host-side token-prep + whisper encoder are not plumbed; the conditioner branch
  has a `TODO(audio)` and does not yet build the whisper audio encoder.
- ✅ **DMD few-step distillation — RESOLVED & FOLDED (2026-05-25)**: the int8 base is the
  NON-distilled model; the reference (`proof_gen.py`, `run_demo_avatar_single_audio_to_video.py`)
  loads `lora/dmd_lora.safetensors` (dim128/alpha64, multiplier 1.0) on top and runs 8 steps with
  `use_distill=True`. **Decision: FOLD at convert time** (not runtime LoRA). `tools/convert_longcat_avatar.py`
  gained `--dmd-lora`: it computes `W += multiplier * alpha_scale * (up @ down)` on the dequantized
  f16 weights before requant. `alpha_scale=0.5` is read from the file. LoRA targets all 7 Linear
  modules per block × 48 = 336 modules (`attn.qkv` n_seperate=3, `attn.proj`, `cross_attn.q_linear`,
  `cross_attn.kv_linear` n_seperate=2, `ffn.{w1,w2,w3}`); the `lora_up.blocks.{i}` splits are folded
  block-diagonally per output-row group. Names map 1:1 to our verbatim gguf tensor names. Verified:
  7 weights/block fold, shapes match exactly, delta is nonzero across all qkv sub-blocks. New artifact:
  `models/longcat-avatar-1.5-dit-dmd-q4_k.gguf`. **8-step DMD sigma schedule** wired in
  `stable-diffusion.cpp::build_longcat_dmd_sigmas()` (reproduces `get_timesteps_sigmas(avatar-v1.5,
  use_distill=True)` + `FlowMatchEulerDiscreteScheduler.set_timesteps` shift 7.0; sigmas
  `[1.0, 0.97998, 0.95449, 0.92094, 0.87478, 0.80728, 0.69916, 0.49799, 0.0]`, C++ matches Python).
  Auto-applied for the avatar version when no custom sigmas are given.
- ✅ **ai2v reference-image conditioning — WIRED (2026-05-25)**: mechanism = the reference portrait
  is VAE-encoded to **one** temporal cond latent (`num_cond_latents=1`), placed as the first latent
  frame (`init_latent[:,:,:1]`); generated frames follow. The cond frame is held FIXED across all
  denoise steps via a per-frame `denoise_mask` (0 for frame 0, 1 elsewhere) — identical to the
  Wan2.2-TI2V I2V pattern already in `prepare_video_generation_latents`. Added an
  `sd_version_is_longcat_avatar() && !start_image.empty()` branch there (encode → slice_assign →
  mask). `sample()`'s denoise-mask blend (`noised_input = x*mask + init_latent*(1-mask)` + the
  post-denoise blend) was extended to the avatar version. **Per-frame timestep[:,:1]=0** wired via
  `process_timesteps` (now also fires for the avatar): the cond frame's adaLN/attention sees t=0
  (clean anchor), matching `generate_ai2v`'s `timestep[:,:1]=0`. **CLI: use `--init-img <png>`**
  (NOT `--image` — that flag is metadata-inspect only and is silently ignored for vid_gen; the
  task's smoke command had this wrong). `--init-img` populates `vid_gen_params.init_image` →
  reaches the AI2V branch. NOT doing channel-concat / `num_ref_latents` continuation (that path
  is for avc video-continuation, not ai2v singletalk).
  - **num_cond_latents attention split (the actual DiT mechanism)**: from
    `modules/avatar/attention.py` — with `num_cond_latents>0` the SELF-attn restricts the
    cond-frame tokens to attend ONLY to themselves (generated tokens attend to all), and the
    TEXT cross-attn ZEROES the cond-frame output (ref frame gets no text). Both wired in
    `longcat_avatar.hpp`: the runner derives `num_cond_latents` from leading-zero per-frame
    timesteps (no new plumbing — `diffusion_model.hpp::LongCatAvatarModel::compute`), builds an
    additive self-attn bias `[Lk,Lq]` (-inf for cond-query→noise-key), and the block zeroes the
    cond rows of the text cross-attn output. The self-attn mask is O(n_token²) memory (~238 MiB
    f16 at 10920 tokens) so it is **capped at n_token ≤ 6000** — above that the self-attn split is
    skipped (subtle refinement) but the cheap text-cross-attn cond-zeroing still applies; the
    latent-space `denoise_mask` pins the cond frame either way. At 480×832/25f n_token≈10920 so the
    self-attn split is currently SKIPPED on the smoke config. **NOTE for next agent**: a dense
    additive mask FORCES the non-flash attention path, which materializes the full
    `[L_k, L_q, n_head]` score tensor (≈15 GB at 10920²×32) — infeasible on the 3060 at this res
    regardless of the mask's own size. A faithful self-attn cond-split at 480p would need a
    flash-attn-compatible block mask or a 2-pass (cond-only + full) attention like the reference's
    `q_cond` split. Deferred; not blocking coherence (the latent denoise_mask + timestep-0 anchor +
    text-cross zeroing already condition on the ref image).
- ⬜ **Remaining**: audio graft (whisper encoder + AudioProjModel + per-block audio cross-attn +
  host token prep — NEXT AGENT), CLI `--audio`, webserver, perf baseline. Audio cross-attn is still
  stubbed (TODO(audio)); this milestone's clip has NO lip-sync — it is a coherent text+image→video.

## INTEGRATION MAP — files to touch to register the new arch

(Verified against this tree. Mirror the existing **Wan** path everywhere.)

1. `src/model.h`:
   - SDVersion enum (~line 50): add `VERSION_LONGCAT_AVATAR` before `VERSION_COUNT`.
   - after `sd_version_is_longcat()` (~145): add `sd_version_is_longcat_avatar()`.
   - `sd_version_is_dit()` (~177-192): include the new version.
2. `src/model.cpp` `ModelLoader::get_sd_version()` (~412-606): add a detection branch keyed
   on a tensor UNIQUE to avatar — use `...blocks.0.audio_cross_attn.q_linear.weight`
   (no other arch has audio_cross_attn). Return `VERSION_LONGCAT_AVATAR`.
3. `src/diffusion_model.hpp`: add a `LongCatAvatarModel : public DiffusionModel` wrapper
   (mirror `WanModel`, ~line 349-419) delegating to a `LongCatAvatar::Runner`. Interface to
   implement: `get_desc`, `compute(int, const DiffusionParams&)`, `alloc/free_params_buffer`,
   `get_param_tensors`, `get_params_buffer_size`, `set_flash_attention_enabled`,
   `set_max_graph_vram_bytes`, `set_circular_axes`.
4. `src/stable-diffusion.cpp` `init_diffusion_model()` (~598-614, after the longcat block):
   add an `else if (sd_version_is_longcat_avatar(version))` branch instantiating the
   conditioner (umT5 via `T5CLIPEmbedder(..., is_umt5=true)`), the `LongCatAvatarModel`, and
   our audio encoder. ai2v needs a reference-image latent (VAE-encoded), NOT clip_vision.
5. `src/longcat_avatar.hpp` (NEW): the model. Mirror `src/wan.hpp` patterns (GGMLBlock,
   `Rope::attention`, the `WanRunner`/`GGMLRunner` scaffold). See spec below.
6. CLI: `examples/cli/main.cpp` already has `-M vid_gen`, `--video-frames`, `--fps`,
   `--flow-shift`. Need to ADD an `--audio <wav>` input flag and thread it to the runner.
   (LTX already has `--audio-vae`; look at how LTX audio input is plumbed for the pattern.)

### Runtime contract (from the Wan path)
- Sampler entrypoint: `StableDiffusion::sample(...)` in `src/stable-diffusion.cpp` (~1839).
  Per-step it calls `diffusion_model->compute(n_threads, DiffusionParams)`. `DiffusionParams`
  carries `x` (latent [N*C,T,H,W]), `timesteps`, `context` (text), `c_concat`/`ref_latents`
  (I2V cond), etc. CFG is applied by calling compute twice (cond/uncond) and blending.
- Sigmas come from the denoiser/scheduler; `sample()` takes a `std::vector<float> sigmas`.
  DMD 8-step = supply our own sigma list (FlowMatchEuler, shift 7.0; DMD distill → 8 steps).
- VID_GEN path: `generate_video()` (~5001). Encodes init/ref image via `encode_first_stage()`,
  builds embeds via text encoder (+clip vision for Wan-I2V), runs `sample()`, decodes via
  `decode_first_stage(..., decode_video=true)`, assembles frames → avi/webm/webp (media_io).
- VAE/TE/diffusion are naturally sequential; `offload_params_to_cpu` + `free_params_immediately`
  + `max_graph_vram_bytes` are the VRAM levers (keep umT5/whisper on CPU, run once, free).

## AVATAR BLOCK SPEC (verified from reference source + checkpoint)

DiT: `LongCatVideoAvatarTransformer3DModel`. hidden 4096, **48 blocks**, 32 heads (head_dim
128), caption 4096 (umT5), SwiGLU ffn inner 11008, patch [1,2,2], adaln_tembed 512, in/out 16ch.
audio: window 5 / block 5 / channel 1280 (whisper) / output_dim 768 / context_tokens 32 /
vae_scale 4 / audio_prenorm=FALSE / class_range 24. Scheduler FlowMatchEulerDiscrete shift 7.0.

Per-block forward (`avatar/longcat_video_dit_avatar.py` `LongCatAvatarSingleStreamBlock`):
```
shift_msa,scale_msa,gate_msa, shift_mlp,scale_mlp,gate_mlp = adaLN_modulation(t).chunk(6)  # [B,T,1,C]
x = x + gate_msa * attn( modulate(mod_norm_attn(x), shift_msa, scale_msa) )   # self-attn, qk-norm, 3D RoPE
x = x + cross_attn( pre_crs_attn_norm(x), y )                                 # TEXT cross-attn, UNGATED
# audio cross-attn (only for generated frames; first num_cond_latents ref-image frames get NONE):
a_shift,a_scale,a_gate = audio_adaLN_modulation(t[:,num_cond_latents:]).chunk(3)
ao_cond, ao_noise = audio_cross_attn( pre_video_crs_attn_norm(x), audio_hidden_states )  # audio_prenorm=false→Identity on audio side
ao_noise = modulate(mod_norm_attn(ao_noise), a_shift, a_scale)               # REUSES mod_norm_attn
add = a_gate * ao_noise ; if ao_cond: add = cat([ao_cond, add])
x = x + add
x = x + gate_mlp * ffn( modulate(mod_norm_ffn(x), shift_mlp, scale_mlp) )     # SwiGLU
```
- `mod_norm_attn`/`mod_norm_ffn` = LayerNorm fp32 **NO affine** (no weights in ckpt).
- `pre_crs_attn_norm`, `pre_video_crs_attn_norm` = affine LayerNorm fp32 (weights in ckpt).
- modulation is per-(B,T) frame, broadcast over the N//T spatial tokens.
- self-attn fused `qkv`; q_norm/k_norm are RMSNorm over head_dim 128.
- text cross_attn + audio_cross_attn both: q from x, kv from context; `q_linear`,`kv_linear`(2C),
  `proj`, q_norm/k_norm. audio kv_linear in-dim is **768** (audio), text is 4096.

3D RoPE: `modules/rope_3d.py` (+ avatar/rope_3d.py adds frame_index for continuation). head_dim
128 split across (T,H,W) axes. Patchified grid from latent [T,H,W] after patch [1,2,2].

Audio token prep (DiT.forward L420-441, do ONCE before denoise loop, host/CPU side): split
audio_embs into first-frame window + latter-frames windows, reshape with vae_scale=4 + the
window middle-index logic, → `audio_proj` (AudioProjModel) → `audio_hidden_states` [B,T,32,768].
AudioProjModel (`avatar/blocks.py`): proj1(first)/proj1_vf(later) → relu → proj2 → relu → proj3
→ reshape [.,32,768] → LayerNorm. Whisper encoder (1280) feeds it `audio_block`(5) stacked
layer hidden states per window.

## TENSOR NAMES (verified, 2241 src tensors → 1608 in gguf; 48 uniform blocks)
Globals: `x_embedder.proj`(Conv3d→reshaped 4D [4096,16,2,2]), `t_embedder.mlp.0/.2`,
`y_embedder.y_proj.0/.2`, `audio_proj.{proj1,proj1_vf,proj2,proj3,norm}`,
`final_layer.{adaLN_modulation.1, linear}` (linear is F16 skip-quant).
Per block: `adaLN_modulation.1`[24576,512], `attn.{qkv[12288,4096],proj,q_norm/k_norm[128]}`,
`cross_attn.{q_linear,kv_linear[8192,4096],proj,q/k_norm}`,
`audio_cross_attn.{q_linear,kv_linear[8192,768],proj,q/k_norm}`,
`audio_adaLN_modulation.1`[12288,512], `ffn.{w1,w3[11008,4096],w2[4096,11008]}`,
`pre_crs_attn_norm`(w+b), `pre_video_crs_attn_norm`(w+b). All under `model.diffusion_model.` in gguf.
ggml shapes are reversed (e.g. qkv ne=[4096,12288], audio kv ne=[768,8192], conv ne=[2,2,16,4096]).

## DMD LoRA
`/mnt/hdd/longcat/avatar-1.5/lora/dmd_lora.safetensors` (dim128/alpha64). 8-step distill.
Either fold into DiT weights at convert time OR load as a runtime LoRA (sd.cpp has `src/lora.hpp`).
NOTE: the int8 base is the DISTILL model already? Verify whether DMD is already baked into
base_model_int8 or must be applied. (The PyTorch path loaded dmd_lora separately on top of int8.)

## BUILD / RUN / CONVERT commands
```bash
cd ~/dev/kobbler/docker/longcat-avatar-dev
./iter.sh build                 # build sd-cli+sd-server (ccache; archs=86)
./iter.sh cli --help            # run sd-cli (GPU + /models mounted)
./iter.sh shell                 # interactive in builder (GPU + repo + /models)
# DiT convert (host, no torch):
cd ~/dev/longcat-avatar.cpp
TMPDIR=$PWD/.convert-tmp uv run --with numpy --with gguf python3 tools/convert_longcat_avatar.py \
  --src /mnt/hdd/longcat/avatar-1.5/base_model_int8 --out models/longcat-avatar-1.5-dit-q8_0.gguf --dtype q8_0
# requant to Q4_K via ggml (arch-agnostic convert mode — works pre-arch-registration):
./iter.sh cli -M convert -m /src/models/longcat-avatar-1.5-dit-q8_0.gguf \
  -o /src/models/longcat-avatar-1.5-dit-q4_k.gguf --tensor-type-rules "model.diffusion_model.=q4_K"
```
GOTCHA: gguf use_temp_file spools to /tmp = 16GB tmpfs → always set TMPDIR to the SSD.

## WEBSERVER (user wants to SEE the clip)
Original PyTorch agent had a Gradio app `~/dev/longcat-video-ref/proof_app.py` on :7860 and a
demo-clip static server on :8001 — but its venv is DELETED, so Gradio won't run. Simplest plan:
once we produce an output clip (.webm/.mp4), serve it with a tiny static http server
(`python3 -m http.server` from the output dir, or a 20-line FastAPI/`http.server` page) bound to
0.0.0.0 so the user can view at http://10.0.0.208:<port>. Then a tracked VRAM+perf re-run.

## REMAINING WORK (ordered for earliest visible signal)
1. Finish supporting converters (background agent) → umT5/VAE/whisper gguf on SSD.
2. Register arch (model.h/.cpp, diffusion_model.hpp, stable-diffusion.cpp) — must COMPILE.
3. Write `src/longcat_avatar.hpp`: WanParams-like config, the block, model, Runner. Mirror wan.hpp.
   Get it to LOAD the q4_k gguf (init_params over all tensor names) — first big checkpoint.
4. Implement forward: patch-embed → 48 blocks (self-attn+text-cross+ffn; 3D RoPE; adaLN) →
   final_layer → unpatchify. Validate it runs end-to-end producing a latent (even if quality off).
5. Audio graft: whisper encoder (ggml) + AudioProjModel + per-block audio_cross_attn + audio prep.
6. DMD 8-step sigma schedule; ai2v ref-image latent conditioning (cond latents prepended).
7. CLI `--audio`; wire generate_video for avatar; produce a clip.
8. Serve clip in webserver. Then tracked VRAM+perf baseline run (record peak VRAM, t/s, wall).

VALIDATION is hard (PyTorch venv deleted). Options: re-create a minimal torch venv to dump
reference intermediate tensors for a few layers, OR validate structurally + visually (coherent
frames). Start DENSE attention (skip block-sparse BSA). Single-speaker first (ignore class_range
multi-talk).
