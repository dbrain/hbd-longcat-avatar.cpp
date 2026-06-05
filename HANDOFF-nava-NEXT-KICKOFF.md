# NAVA cpp — NEXT SESSION kickoff (full-auto, biggest win) — pick up 2026-06-05

You are a FRESH agent on a long, fully-autonomous run. Branch `nava-port` in
`/home/dbrain/dev/longcat-avatar.cpp`. 1× RTX 3060, **serial GPU (one job at a time)**. **RUN UNTIL
COMPLETE — don't hand back early.** Drive ALL build/render/profile from the MAIN LOOP (Agent-tool
subagents STALL on GPU/background jobs — use them only for read-only analysis/design/authoring).

## READ FIRST
1. `HANDOFF-nava-VRAM-MODULATION.md` — the primary task + a complete bit-exact implementation spec.
2. Skim `PERF-nava-DEEP-SESSION-2026-06-05.md` (what's been won), `bench/results/PROFILING-BIBLE-2026-06-05.md`
   (kernel map), `/mnt/hdd/nava/COMPARISON-GUIDE.md` (clip ↔ experiment map). Memory:
   project_nava_vram_modulation_lever, project_nava_deep_perf_2026_06_05.

## Build / run / review mechanics
```
export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
cmake --build build-nava --target nava -j8        # cpp builds are fine on-server (no-build rule is Rust-only)
```
Render config (q5_K is the locked prod quant): see the locked block in HANDOFF-nava-PERF-BIBLE-KICKOFF.md,
`--gguf models/nava-dit-q5_k.gguf`. Review: eye http://10.0.0.208:8097 (auto-lists /mnt/hdd/nava/cpp-runs,
each run needs clip.webm+meta.json — render with `--out-name <name>`). VRAM: sample
`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` DURING a real render (NOT profile mode).
ncu/nsys are bundled: `$TOOLCHAIN/nsight-compute/2024.1.1/...` (ncu needs sudo on this host). Probes:
`NAVA_DIT_VRAM_HIST=1` (DiT graph top tensors), `NAVA_DUMP_AUDIO_LATENT`/`NAVA_DUMP_WAV` (bit-exact A/B).

## TASK 1 (do first) — implement the modulation 2-timestep collapse (bit-exact VRAM + speed win)
Follow HANDOFF-nava-VRAM-MODULATION.md §SPEC verbatim (Strategy i). The DiT 1771 MB buffer is dominated
by redundant `[3072,6,5148]` AdaLN modulation tensors (2 unique timestep values). Expected: buffer
~800-1000 MB → DiT peak 6855 → ~6000-6150 MiB (q5_K), bit-exact, + a small DiT speedup.
**VALIDATE HARD:** same seed, current-vs-patched, audio-latent cos ≈ 1.0 AND ~0 waveform diff (compare
CUDA-vs-CUDA). The SingleBlock 4-segment case is the high-risk piece — check a single-block hidden-state
capture vs baseline before trusting the final latent. Commit when bit-exact + VRAM measured.

## TASK 2 (the owner's real goal) — 1280×704 NATIVE-RES, recommended-spec quant shootout
1280×704 is the model's NATIVE resolution; 25 steps is its preferred step count. The owner's hypothesis:
the Q4/Q5/Q8 quality ordering "shuffles" at off-spec settings (896×448, 10 steps) — sometimes Q5/Q8 sound
like the "bad" Q4 — and a STABLE "clearly better" quant should emerge at native res + recommended steps.
- First confirm Task 1's VRAM headroom makes 1280×704 fit (more tokens → bigger everything; the modulation
  fix helps high-res MORE since those tensors scale with token count). Measure peak VRAM at 1280×704.
- Render the quant ladder at **1280×704 × 25 steps**: q8 / q6_K / q5_K / q4_K (and q5k-audioq8 mix if useful).
  Dump audio latents; put every clip on :8097 (`--out-name q<quant>_native`) for the owner's EAR/EYE judgment.
  Report VRAM peak + wall per quant. Find the stable ranking.
- If 1280×704 OOMs even after Task 1: that's a finding — report how far over, and whether a lower frame
  count / the FFN-tile (the buffer's tier-2 floor) / q4_K closes the gap.

## QUALITY PHILOSOPHY (the owner's rule — follow exactly)
- Confirm everything with NUMBERS (VRAM peak, wall, audio-latent cos, bit-exact diffs).
- BUT **do not write off a quant/config on numbers alone — it needs an EAR test** (clip on :8097), UNLESS
  the numbers are *clearly, clearly* broken (NaN/garbage/OOM/crash). Audio latent-cos-vs-q8 is MISLEADING
  (it tracks trajectory drift, not perceptual quality — proven this session: both q4/q5 "diverge" from q8
  at 25 steps yet sound cleaner). Ears are the judge. Render the clip and let the owner listen.
- One thing at a time; serial GPU; fresh seed per A/B; eye-test every visual change.

## FACTS — don't re-derive (measured 2026-06-05)
- Quant affects VRAM, NOT speed (MMQ dequants all to int8 mma; ncu shows not bandwidth-bound). VRAM @896×448/13f:
  q4_K 6156 / q5_K 6855 / q6_K 7579 MiB. Wall ≈ flat across quants.
- Steps do NOT cost VRAM (weights + 1-forward buffer + VAE(frames/res)). VRAM scales with frames & resolution.
- Audio is the quant-sensitive stream. q5_K clean to ear @10/896; q4_K "off tone" @10/896 but q4_K@25 ≈
  q5_K@25 (latent cos 0.968) — q4_K's tone issue may be a low-step/low-res artifact → THE native-res test.
- DEAD (measured, don't retry): mmq occupancy (−10..16%), align-off (audio quality). Opt-in step-cache exists
  (NAVA_CACHE_THRESH) but trades audio refinement — leave off for quality runs.
- ggml submodule commits are LOCAL-ONLY (concat 0bcf0e83, madd 292516d5) → push dbrain/ggml before any deploy.

## DONE = Task 1 shipped+bit-exact+VRAM-measured, Task 2 quant ladder rendered at native res with clips on
## :8097 + a VRAM/wall table + a recommendation (pending owner ears), everything committed, memory+handoff
## updated. Don't write off the hard tier (video-VAE direct-conv3d, ~−10-15s/clip bit-exact) without flagging it.
