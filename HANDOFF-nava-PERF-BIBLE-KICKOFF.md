# NAVA cpp — DEEP PERF SESSION kickoff (the "bible" baseline) — 2026-06-05

Branch `nava-port`. For a FRESH agent running a long, deep performance session. The owner's
framing: **"get as much as you can out as the bible for where time and VRAM is going, then we
rerun as we chop the heads off."** So: PHASE 1 = exhaustive profiling (ncu kernel-level +
VRAM-at-every-step) → a definitive map. PHASE 2 = iterate, re-profiling against this baseline.

This doc gives you the locked baseline numbers, the measured levers, the unprofiled targets,
and all run/build/profile mechanics so you start at full speed.

## Build / run (on-server, cpp builds are fine here — the no-build rule is Rust-only)
```
export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH
export LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib
cmake --build build-nava --target nava -j8          # binary: build-nava/bin/nava
# (also --target sd-cli for making k-quants)
```
1× RTX 3060 (12 GB), **SERIAL GPU — one job at a time**. After killing a GPU job, sleep 2-3s.
Review headless: eye http://10.0.0.208:8097 (auto-lists /mnt/hdd/nava/cpp-runs), ear
http://10.0.0.208:8099 (`tools/nava_audio_demo_add_row.py`).

LOCKED render config (all perf numbers below use it):
```
NAVA_VAE_TILE=16 build-nava/bin/nava render --cuda --gguf models/nava-dit-q6_k.gguf \
  --context /mnt/hdd/nava/peter_ctx.bin --neg-context /mnt/hdd/nava/vneg_now.bin \
  --image /mnt/hdd/nava/peter_896x448.bin --vae models/wan2.2-vae-48ch-f16.gguf \
  --audio-vae models/nava-ltx-audio-vae-f16.gguf --steps 10 --frames 13 \
  --width 896 --height 448 --seed 42 --out-name <name>
```
q6_K + flash-attention (default ON) + tile16. The binary already prints per-phase timing
(backbone loaded / VAE decode / audio VAE decode / sample / render done + meta.json wall_s,
s_per_step). `nvidia-smi --query-gpu=memory.used` sampled during a run gives the VRAM curve.

## BASELINE — where the 175.9s wall goes (MEASURED 2026-06-05, this config)
| phase | time | % | VRAM during phase |
|---|---|---|---|
| load DiT | 2.4s | 1% | ramps to peak |
| **DiT sampling** | **95.6s** | **54%** | **7579 MiB (the ceiling, flat)** |
| **video VAE decode** | **46.0s** | **26%** | 4719 MiB |
| **audio VAE decode** | **29.7s** | **17%** | 565–695 MiB |
| mux | ~1.5s | 1% | — |
| **wall** | **175.9s** | | **peak 7579 MiB** |
DiT = 9.55 s/step ×10 (3 forwards/step: cond + uncond + align-mmask). Video VAE decodes 49
pixel frames. Audio VAE decodes 2.05s audio in 29.7s (RTF ~14.5).

## VRAM profile (the whole story is the DiT phase)
The peak **7579 MiB sits ENTIRELY in DiT sampling** (flat for the full 96s): q6_K weights
~5581 MiB + FA activation ~2000 MiB. The instant the DiT frees, video VAE = 4719 MiB, audio VAE
= ~0.7 GB. **=> The VRAM budget is a DiT-phase problem ONLY. Both VAE phases have 4–11 GB of
unused headroom** (e.g. the video VAE could use a larger tile if that were faster — but tile16
is already both smaller AND faster than tile24: 46s vs 56s, measured prior).

## LEVERS — measured this session (seed for the bible)
| config | wall_s | Δ | s/step | note |
|---|---|---|---|---|
| baseline (q6_K+FA+tile16) | 175.9 | — | 9.56 | |
| `NAVA_AUDIO_DISABLE_BWE=1` | 154.9 | **−21.0s** | 9.56 | audio VAE 29.7→~9s; output stays 16 kHz. EAR-TEST quality. |
| `NAVA_NO_ALIGN_CFG=1` | 144.1 | **−31.7s** | 6.39 | drops the mmask forward (3→2/step, −33% DiT). Quality-load-bearing on HARD prompts (audio→noise divergence) — A/B by ear before adopting; fine on easy prompts. |
| both | ~123 (est) | −53s | 6.39 | |

## UNPROFILED — your Phase-1 targets (the deep dive)
1. **DiT kernels (54% of wall) — the main event.** ncu-profile the FFN/attention matmuls.
   q6_K ≈ q8 ≈ q4_K speed (no quant speedup measured) ⇒ likely MMQ/compute-bound, mirroring
   flux2 lap-6 & the turboquant laps. CONFIRM with counters: occupancy, DRAM%, issue-slot
   utilization, regs/smem per the dominant matmul. Is it occupancy-bound (like flux2's Q4_K
   matmul: 16% occ, 221 regs+58KB smem → 1 block/SM) or genuinely compute-bound? The joint
   self-attn is ~5k tokens, head_dim 128, flash-attention ON.
2. **Video VAE decode (46s, 26%) — SECOND biggest, totally unprofiled.** Wan2.2 48ch causal
   conv3d, tiled 16×16 latent / 0.25 overlap, 49 frames. Where does the 46s go — conv3d?
   tiling overlap recompute? Profile it. Big potential win, no quality risk (decode is exact).
3. **q5_K** — make it (`sd-cli --mode convert -m models/nava-dit-f16.gguf -o models/nava-dit-q5_k.gguf
   --tensor-type-rules ".*=q5_K"`; NOTE capital K), measure VRAM (~4.8 GB weights → peak ~6.8?)
   + audio latent cos vs q8 (q6_K holds 0.991; q4_K collapses to 0.754). Audio stream is the
   4-bit-sensitive one — q5_K is the open question for more headroom.
4. **Weight offload** — q8-exact under budget via the longcat prefetch-thread pattern
   (background H2D overlap; see project_longcat_vram_reorder_hunt in memory). Only if wanted;
   q6_K already fits+holds. This is the path to q8 quality at <7.5 GB.
5. **align_cfg quality value** — quantify what the mmask forward actually buys (it's +33% DiT).
   Render hard vs easy prompts with/without; if easy prompts don't need it, default-off them.

## ncu mechanics
`ncu` is at `$TOOLCHAIN/bin/ncu`. Hardware counters need profiling permission
(`NVreg_RestrictProfilingToAdminUsers`); if you get ERR_NVGPUCTRPERM, run ncu under sudo or set
the module param (native host — NOT the Docker `--cap-add SYS_ADMIN` path; that memory note is
Docker-only). Profile mode INFLATES VRAM/time massively — quote PROD-mode (graphs ON) cost in
any sizing, never profile-mode (see memory: project_llama_memory_shuffle_sizing). Target a
SINGLE step (`--steps 1`) and filter to the hot kernels (`-k regex` / `--launch-count`) so the
profile finishes. Dump to `bench/results/` like the other cpp projects.

## CONSTRAINTS (hard)
- **Quality bar:** audio latent cos ≥ 0.991 (q6_K level); NO quality loss from any perf change
  (no step cuts, no cfg changes that degrade, no requant below q6_K for the shared MMDiT).
  Latent cosine OVERSTATES the perceptual audio gap — also judge by ear (:8099).
- **VRAM ≤ 7.5 GB** (3060 budget w/ coexistence headroom). Baseline 7579 MiB is right at it.
- Eye-test every visual change (:8097). One thing at a time; serial GPU; fresh seed per A/B.

## INSTRUMENTATION already in the binary (examples/nava/main.cpp)
- per-phase wall timing (printed) + meta.json (wall_s, load_s, s_per_step).
- `NAVA_DUMP_LATENT` / `NAVA_DUMP_AUDIO_LATENT` (final latents, for cross-checks / cos).
- `NAVA_DUMP_TAIL=<dir>` (lossless tail frames), `NAVA_DUMP_TRAJ` / `NAVA_DUMP_AUDIO_TRAJ`
  (per-step trajectory dumps).
- env levers: `NAVA_NO_FLASH`, `NAVA_NO_ALIGN_CFG`, `NAVA_AUDIO_DISABLE_BWE`, `NAVA_VAE_TILE`,
  `NAVA_EULER`/`NAVA_AUDIO_EULER`, `NAVA_FREEZE_AUDIO`.
- Suggested Phase-1 add: a per-phase VRAM logger (peak via cudaMemGetInfo at phase boundaries)
  baked into the binary for exact "VRAM at every step" instead of nvidia-smi polling.

## Models (models/)
nava-dit-{f16,q8_0,q6_k,q4_k,q4k-audioq8,q4_0}.gguf, nava-dit-fp8fold-{q8_0,q8_audiof16}.gguf,
wan2.2-vae-48ch-f16.gguf, nava-ltx-audio-vae-f16.gguf, longcat-umt5-xxl-q8_0.gguf.
Scratch/logs in /mnt/hdd/nava/ (perf_baseline.log, perf_vram.csv this session).

## Prior perf context (read for the floor-finding playbook)
PERF-lap*.md (the flux2 DiT laps: FA, mmap, cuBLAS-vs-MMQ, ncu occupancy NEXTLAP, mmq tile),
HANDOFF-nava-SESSION-features-perf.md §3 (the q-quant + FA + tile table). Memory notes:
project_llama_turboquant_lap*, project_flux2_cpp (ncu occupancy playbook), project_longcat_vram_reorder_hunt
(offload prefetch-thread pattern), feedback_check_running_cmdline, feedback_one_thing_at_a_time.

## Status of the other tracks (this session, all committed)
- Track A continuation: DONE. `encode-video` + `NAVA_DUMP_TAIL` shipped (commit 1c39120);
  M=1 is the validated photoreal path (HANDOFF-nava-CONTINUATION-RESULTS.md).
- Track B audio-VAE encoder: DEFERRED w/ complete blueprint (HANDOFF-nava-AUDIO-ENCODER-SPEC.md).
- Warm-start continuation trick: deferred task. These come AFTER perf (faster iteration first).
