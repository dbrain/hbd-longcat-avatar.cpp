# NAVA performance learnings — cross-fork salvage guide (2026-06-06)

Branch `nava-port`. **Purpose:** NAVA (ernie-research 6.3B joint audio+video MMDiT, sd.cpp/ggml C++
port) is being retired for the continuity mission — but its perf work was substantial and a chunk of
it **translates to the other cpp GPU forks** (llama, siglip2, parakeet, acestep, longcat-avatar,
flux2). This is the pointer doc for agents salvaging that value. Read the §"TRANSLATES" section
first; it's the high-yield part. Hardware context for all absolute numbers: single **RTX 3060
(12 GB), sm_86**, prod quant **q5_k**, unless stated.

## Where NAVA's time/VRAM actually goes (so you know what's worth chasing)
Baseline 2 s clip (896×448, 25 frames era): **175.9 s** total = **DiT 95.6 s / video-VAE 46 s /
audio-VAE 29.7 s**; VRAM peak **7579 MiB, entirely in the DiT sampling phase**. After the wins below:
**136.8 s (−22%, zero quality loss)**. The DiT is a **proven latency/occupancy-bound floor**
(`mul_mat_q` runs at ~16.7% occupancy — it is NOT DRAM-bound; do not chase memory bandwidth there).
At 960×960/10-steps this session: peak **8.06 GB**, ~250 s/2 s-clip (sample ~157 s + VAE-decode ~89 s).

## ✅ TRANSLATES to other forks — start here
1. **ggml single-launch contiguous concat** — `ggml/src/ggml-cuda/concat.cu`, commit `0bcf0e83`,
   **now on `dbrain/ggml` master**. `concat_dispatch` was launching one kernel per `dst->ne[3]`
   slice → ~6.1M tiny launches on the Wan2.2 VAE decode (≈20% of decode, ~9.9 s). New
   `concat_T_cont_4d` folds the ne3 loop into one grid-stride kernel: **1 CONCAT op = 1 launch,
   bit-identical**. **Every concat-heavy graph benefits** (VAE/conv stacks, joint-attention concats).
   Any fork on `dbrain/ggml` master gets it for free on next build — just confirm the rebuild.
2. **ggml same-shape fused multiply-add** — `x + y*g [+ shift]`, commit `292516d5`, **now on master**.
   Generalizes the lap-28.3 gate_add fusion to the **no-broadcast** case (full same-shape contiguous
   F32, batch N=1) — fuses `ADD(x, MUL(a,b)) [+ ADD(_,shift)]` into one pass instead of 3 full-size
   kernels. **Bit-exact** (`__fmul_rn`/`__fadd_rn`, verified 0.0 end-to-end). Kill-switch
   `GGML_CUDA_NO_MADD_FUSE=1`. Helps any per-token AdaLN-style modulation (most DiTs at N=1).
3. **AdaLN 2-timestep modulation collapse** (the pattern, not the code) — commit `98e8d16`,
   `src/nava.hpp:898`, handoff `HANDOFF-nava-VRAM-MODULATION.md`. NAVA's DiT computed the AdaLN
   `modulation [3072,6,5148]` tensor (≈380 MB each, ~40 of them) on the **full noisy timestep axis**
   though only **2 timesteps are ever unique** (clean-anchor prefix `t=0` + `tval` for the rest).
   Computing modulation on the compact `{0, tval}` table then scattering = **bit-exact**, DiT buffer
   **1771→627 MB**, and it **double-dips into speed** (those modulation ADDs are also DiT-time mass).
   This unlocked 1280×704 native res (9453→7233 MiB). **Generic DiT pattern** — any DiT that
   broadcasts an AdaLN modulation buffer across a timestep axis with few unique values (longcat, flux
   DiTs) is a candidate. This is the single highest-yield port target.
4. **Profiling method** — `nsys`/`ncu` are bundled in `/mnt/hdd/3d/avatar-shootout/toolchain`
   (`ncu` needs `sudo`). The wins above were all found by **nsys-attributing wall time to kernels**
   then **ncu-checking occupancy/bound-type** before optimizing — not by guessing. Reuse this loop on
   any fork. Key reusable insight: a kernel that's *latency/occupancy-bound* (low occ, DRAM idle)
   is fixed by launch-count/occupancy work, NOT by bandwidth tricks.

## NAVA-specific wins (context — may inspire, won't drop-in)
- **Audio-VAE F16 conv_transpose WMMA** (−27 s): the LTX audio-VAE `ups.N` weights were stored F32 →
  fell to a naive kernel that was **33% of all GPU time**. Storing F16 + a WMMA path fixed it. Only
  relevant to forks using that LTX audio decoder, but the lesson (audit weight dtype vs the kernel it
  selects) is general.
- **DiT cond-frame cache** (~12% bit-exact, longcat path): cache the clean conditioning frame's
  contribution across steps. Applies where a clean anchor is re-attended every step.

## ❌ Dead-ends — do NOT re-try (measured, not guessed)
- **MMQ occupancy tuning on the DiT `mul_mat_q`** — measured dead; it's latency-bound at 16.7% occ,
  not throughput-bound. Bigger tiles/occupancy knobs did nothing.
- **align-guidance off / BWE off as perf levers** — each saves time (−31.7 s / −21 s) but **costs
  quality** (sharpness/detail jump); not free wins, don't ship them as optimizations.
- **Fused conv-3d** — occupancy-bound; **keep im2col + cuBLAS** (the fused kernel is slower).
- **Quant as a *speed* lever** — q4/q5/q6 change **VRAM only, not speed** (q4 6156 / q5 6855 /
  q6 7579 MiB). q5_k is locked prod. Don't expect quant to buy throughput.
- **Steps don't cost VRAM** (only time). Don't trade steps for memory.
- **Step-cache (TeaCache/EasyCache, `NAVA_CACHE_THRESH`)** — real speed (−43% @25 steps) but
  **trades audio fidelity**; opt-in only, and use `NAVA_CACHE_AUDIO_AWARE=1` if you must on a
  talking clip. Not a free general win.

## Pointers
- Deep-perf handoff + numbers: `HANDOFF-nava-PERF-BIBLE-KICKOFF.md`, `HANDOFF-nava-VRAM-MODULATION.md`,
  `HANDOFF-nava-AUDIO-ENCODER-SPEC.md`.
- ggml commits: `dbrain/ggml` master `0bcf0e83` (concat), `292516d5` (madd-fuse).
- DiT modulation collapse: `src/nava.hpp:898` (commit `98e8d16`).
- Reusable ggml-cuda autofusion catalog lives in the consolidated ggml notes (see the autofusion
  reference in project memory).
