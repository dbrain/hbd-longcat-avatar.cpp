# Mini-prompt — Wan2.2-VACE: crack the 1280 VRAM/offload wall, THEN judge model quality (autonomous)

You are tuning **Wan2.2-VACE-FUN-A14B** (stable-diffusion.cpp fork) on a single **RTX 3060 (12GB)** to render
long-form directed video. Two phases, IN ORDER.

**READ FIRST:** `HANDOFF-VACE-TUNING.md` (FINDINGS-L1..L5b + FINAL VERDICT — the full prior lap), memory
`project_wan22_infinitetalk_3060` + `feedback_wan22_overnight_quality_not_benchmark`. Worktree
`/home/dbrain/dev/longcat-avatar-wan22`, branch `wan22-infinitetalk`, UNCOMMITTED.

## STATE (already done, bit-exact — do NOT re-chase)
- Operating config: `--max-vram 7.3` + `--vae-tile-overlap 0.25` + `VACE_GRAY_CACHE_DIR` set. Steps = 4 DMD
  (2 high + 2 low) — DMD-native, do NOT treat step count as a perf lever.
- Wins: gray control-encode 32→~2s (lever #1, `vace_encode_ctx` in stable-diffusion.cpp ~5726, disk-cached);
  VAE decode 27→16s via overlap 0.5→0.25 (lever #3). Warm 480×832 fresh-shot ≈102s (was 127).
- **480×832 DiT = silicon floor (~82s): matmul+FA 81.5%, fusion ON, offload neutral. Don't touch kernels.**
- DEAD: vace_layers caching (block consumes per-step x_orig+e0); offload prefetch thread NEUTRAL for VACE
  (and pool-ON balloons VRAM — if ever used, MUST set `LONGCAT_NO_PREFETCH_POOL=1`); higher max-vram flat/worse.

## ★ PHASE 1 (DO FIRST) — make VRAM/offload less of a wall at 1280×704
**Problem:** at 1280×704 a SINGLE DiT block's compute buffer = 8.5GB (FR=21, OOMs 12GB) / 5.1GB (FR=13).
Active expert 9.87GB + buffer >12GB ⇒ the expert PCIe-STREAMS every step ⇒ DiT ~265s, gen ~291s (≈3.2×
LTX-2.3's 111 render-s/s-video). Peak only 8.5/12GB ⇒ this is a CAPACITY/streaming wall, not kernel speed.
**Goal: shrink the per-block buffer enough that the 9.87GB expert RESIDES (no streaming) ⇒ ~3× faster.**

Attack in order:
1. **Diagnose first (don't guess):** run `LONGCAT_VRAM_BREAKDOWN=1` on a 1280×704 FR=13 segment — dump the
   compute-buffer composition. Find what's in the 5.1GB; confirm nothing is held longer than needed (T5
   5.76GB should free before DiT; inactive expert should stream from host, not sit in VRAM). Keep raw output
   in a file, read the summary.
2. **★ Finer graph-cut granularity (the big lever, bit-exact, code change):** the cutter only splits at BLOCK
   boundaries — `mark_graph_cut` is called once per block (`wan.hpp` ~827, `"wan.blocks."+i`). So one block's
   buffer is ATOMIC (that's why `--max-vram 2.5` did NOT shrink it). Add cut-marks INSIDE the block (after
   self-attn, after cross-attn, after each FFN matmul) so the cutter can sub-segment a block → smaller buffer.
   If buffer drops under ~2GB, expert (9.87) + buffer fits 12GB → resident → compute-bound → big win. Pure
   execution-order change ⇒ must stay bit-exact (A/B frames vs current). Same for the 8 vace_blocks.
3. **Single fat tensor?** If the breakdown shows one dominant reducible F32 tensor (LTX MOD_COLLAPSE cut a
   buffer 62% this way; Wan timestep is scalar so that exact lever N/A, but look for an analog).
4. **F16 non-matmul intermediates (secondary, quality-risk):** activations are F32 (`wan.hpp` ~193 casts up);
   MMQ needs F32 matmul INPUT (proven dead-end elsewhere) so only non-matmul intermediates could be F16 —
   partial buffer cut, validate quality. Last resort.
**Win condition:** active expert resident at 1280×704 (DiT offload→~0, compute-bound). Re-measure DiT + peak
+ throughput vs LTX 111. A/B EVERY change (wall + peak VRAM + frame bit-exactness where claimed).

## PHASE 2 (after Phase 1) — the GOAL RUN: is the model any good at all?
Render the LTX music-video shot list as **t2v** (this is a QUALITY judgment, not a benchmark). Goal: faces
fuzzy/warping? motion natural? do chained segments connect? `run_vace_musicvideo.sh` is prepared (4 scenes,
ported Wan2.2 t2v prompts, chained, stitched).
- **t2v — NO `--init-img`.** char.png is an ANIME-GIRL ref; passing it as i2v init poisoned EVERY prior test
  (anime face / ghosting). t2v = prompt drives everything, no identity anchor (identity drift is FINE).
- **SMOKE TEST FIRST (mandatory):** render ONE short t2v segment at 1280×704, eyeball the frame — confirm it's
  clean and there is NO anime girl — BEFORE the long run. All prior tests were broken by the anime girl.
- Target: **~27 seconds of video @ 1280×704, distill 4-step** (matches LTX's 27s clip). 16fps ⇒ ~432 frames
  ⇒ chain ~30+ FR=13 segments (0.81s each) across the 4 scenes; stitch to ONE mp4 (host ffmpeg; none in
  builder). Use Phase-1's faster config. Send the mp4 to the user (SendUserFile).
- Q4_K DiT (only quant local). If faces are fuzzy, precision is the lever (Q5/Q6 — bigger VRAM, source from
  10.0.0.151 or convert; viable at 480, worse at 1280). Surface that finding; don't silently requant.

## RULES / TOOLING
- Drive heavy GPU from the MAIN loop, `run_in_background:true`, NEVER detached `&`. One GPU job at a time,
  shared with prod (worker-isolated, idle when not used) — coordinate, ask if mid-bench. Clean up strays:
  `docker ps --filter ancestor=longcat-avatar-dev:builder` then `docker kill`.
- C++ builds on-box via the docker builder; Rust does NOT. Build:
  `docker run --rm --gpus all -v $PWD:/src -v longcat-avatar-iter-ccache:/root/.ccache -w /src
   longcat-avatar-dev:builder bash -lc "cmake -S /src -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON
   -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j\$(nproc) --target sd-cli"`.
- nsys: bundled at `/opt/nvidia/nsight-compute/2025.2.1/host/target-linux-x64/nsys`; qdstrm→nsys-rep needs
  `apt-get update && apt-get install -y libdw1` then QdstrmImporter; query the .sqlite on HOST python3 (NO
  python in builder). ncu needs `--cap-add SYS_ADMIN`. No ffmpeg in builder — host has it.
- **GRAY-CACHE CONFOUND:** pre-warm `VACE_GRAY_CACHE_DIR` before ANY A/B — the FIRST run pays a one-time
  ~14s (480) / ~24s (1280) gray-latent compute; it poisoned 2 of my A/Bs (looked like a perf win, was the cache).
- Decode-only harness `VACE_DECODE_LATENT=<saved latent>` skips the 84s DiT (sweep decode tiling at ~30s/cfg).
- Models in `models/`: `wan22-vace-fun-a14b-{low,high}-distill-q4_k.gguf` (9.87GB), `longcat-wan-vae-f16`,
  `longcat-umt5-xxl-q8_0`. Only Q4_K DiT local; more on `10.0.0.151:~/dev/wan22-infinitetalk/models/`.
- Scripts: `run_vace_musicvideo.sh` (goal run), `run_vace_1280_fit.sh` (single-seg fit/timing), `run_vace_nsys.sh`,
  `run_vace_{maxv_sweep,decode_sweep,offload_combo}.sh`. Eye-test page :8097 (10.0.0.208).
- Bank findings into `HANDOFF-VACE-TUNING.md` + memory continuously. A/B every change; "at floor"/"dead"
  claims ship a metric breakdown.

## START: Phase 1 — run `LONGCAT_VRAM_BREAKDOWN=1` on 1280×704 FR=13 to see the 5.1GB buffer composition,
## then implement finer in-block graph-cut marks to shrink it so the 9.87GB expert resides. Report DiT +
## peak + throughput vs LTX after the win. THEN smoke-test t2v (no anime girl) and render the 27s quality clip.
