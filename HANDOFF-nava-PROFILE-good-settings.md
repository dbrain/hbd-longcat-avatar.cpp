# NAVA cpp — "good settings" profile (time + VRAM) + encode-prompt sweep — 2026-06-05

Branch `nava-port`. Profiled the owner's chosen prod config after the modulation collapse +
audio-aware cache landed. Goal: "no stone unturned" — find any remaining heavy hitters.

## GOOD SETTINGS (owner's go-to)
q5_k + **audio-aware step-cache** (`NAVA_CACHE_THRESH=0.15 NAVA_CACHE_AUDIO_AWARE=1`).
Steps/res are quality dials (owner: 25 steps reads better than 50 — 50 added unwanted head
movement; NAVA's "recommended 50" is NOT the perceptual optimum here). Profiled at the bench
config 896×448 / 13 frames (49 px) / 10 steps unless noted.

## TIME PROFILE (896×448, 10 steps, wall 138.6s)
| phase | time | % | status |
|---|---|---|---|
| load DiT | 3.7s | 3% | — |
| **DiT sampling** | **92.7s** | **67%** | at floor (MMQ matmul-bound; **cache inert ≤10 steps** — 0/10 skipped) |
| **video VAE decode** | **36.4s** | **26%** | conv3d-bound — see below |
| audio VAE decode | 3.1s | 2% | already optimized (was 29.7s → wmma conv_transpose) |

**Cache step-count dependence (measured):** skipped 0/10 @10 steps, 10/25 @25, 24/50 @50.
The cache only earns its keep at ≥25 steps; at 10 it's a no-op. Audio-aware vs no-cache latent
cos: 0.986 @25 (t15), 0.9955 @50 — near-lossless at the real step count.

## VRAM PROFILE (896×448) — weights dominate, buffers are not the lever
- DiT phase peak **5867 MiB** = q5_k weights ~5135 + compute buffer **627** + overhead.
- Video VAE phase **4829 MiB** (wan_vae compute buffer 3432). DiT freed before VAE → VAE never
  sets the peak at this res.
- Audio VAE ~775 MiB.
- **Weights set the peak**, so further DiT-buffer trimming won't move it. (At 1280×704 native:
  DiT peak 7233 compact / 7367 from-full; old pre-collapse baseline 9453 = over budget.)

## DiT compute buffer post-collapse (NAVA_DIT_VRAM_HIST) — collapse confirmed
Old top tensor (modulation `[3072,6,5148]` 379.6 MB) is GONE. New floor = FFN `[14336,5148]`
= 295 MB tensors (real, not dedupable). Buffer 1771 → **627 MB** holds. (My expansion adds
transient REPEAT/CONCAT graph mass but gallocr packs it; no net VRAM cost.)

## VIDEO VAE deep-dive (the #2 heavy hitter, 36.4s) — near floor
nsys kernel time (1 DiT step + full VAE decode):
- `im2col_3d_tiled` 6.56s (14.2%, #1 kernel overall) + `im2col` 2D 4.49s → **~11s im2col**
- conv F16 tensor-core gemms (`ampere_h16816gemm` + `h1688gemm`×3) ~13.1s
- `pad_f32` 2.26s (4230 launches)
**ncu on im2col_3d: Compute(SM) 77.4% / DRAM 47.2% → COMPUTE-bound, not idle bandwidth.**
So the im2col can't be fused away for free (gather work is real), matching longcat's
"keep im2col+cuBLAS" conclusion.
- **Tile sweep (measured, q5_k):** tile8 43.1s / **tile16 36.4s** / tile24 50.2s → tile16 is
  the confirmed optimum (clean U-shape). No free win from tiling.
- **`pad_f32` is NOT a quick fold** (I was wrong initially): CausalConv3d pads asymmetric
  causal-temporal (`lp2=2*pad, rp2=0`) + optional circular spatial; im2col only does symmetric
  zero-pad → can't absorb it without extending the ggml im2col_3d kernel. ~2s, low priority.
- **Only real lever = full fused/implicit tensor-core conv3d.** Rough gain: best case ~−8-10s
  (the im2col time) → ~−6-7% wall; realistic less; real risk of ~0 or negative (longcat
  precedent). High effort (custom bit-exact tensor-core conv3d kernel, days). Marginal EV.

## VERDICT: pipeline is near its floor at the good settings.
Banked wins this round: modulation collapse (VRAM), audio-VAE wmma (−27s), audio-aware cache
(−44% @50 near-lossless). DiT = proven matmul floor. VAE = conv3d floor (im2col compute-bound).
No cheap stones left; fused-conv3d is the only deep/speculative lever.

## umT5 encode-prompt sweep (the per-segment text-context cost the owner flagged)
`nava encode-prompt <umt5_q8_0.gguf> prompt.txt out.bin`. Speech changes every continuation
segment → this runs PER SEGMENT in the real product. Measured:
- **Cold: 14.2s wall, peak 6441 MiB.** Breakdown: ~12s = **disk read of the 6.0 GB q8_0 gguf**
  (`/dev/sdb2` ~500 MB/s); GPU H2D only 1.13s (nsys); the encode forward (512 tok, 1 pass) ~2s.
- **Warm (OS page cache hot): 2.9s** — `loading tensors 1.93s (read 0.52s, copy 0.65s)` + ~1s
  encode. So a per-segment reload of the SAME gguf is ~3s if RAM keeps it cached, NOT 14s.
- **The encode COMPUTE is cheap and fine — this is a model-LOAD/architecture problem, not a
  kernel problem.** Levers for the next agent (plumbing, not GPU work):
  1. **Load-once-encode-all**: encode every segment's prompt while umT5 is resident, then free,
     then loop DiT segments → 1× cold-load + ~1s/segment. Best for N segments.
  2. **Enable mmap** (loader currently "mmap disabled by caller") → lazy, page-cache-backed,
     reclaimable RAM; cheap warm reloads. Mirrors prod sd-server/longcat mmap pattern.
  3. umT5 can't co-reside with the ~7 GB DiT on 12 GB → lifecycle MUST be
     encode(s) → free → DiT → free → VAE (sequential). 6.4 GB is transient, freed before DiT.
- **How busy is the next agent?** NOT very, on perf — the forward is cheap. The work is the
  inline-encode + free + (batch | mmap) plumbing already noted in project_nava_continuation_m1
  ("umT5 context is PER-SEGMENT"). `nava render` already links `T5Embedder`; add a `--prompt`
  inline path with mmap + free. No kernel optimization needed.
