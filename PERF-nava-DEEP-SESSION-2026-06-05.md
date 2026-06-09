# NAVA cpp — DEEP PERF SESSION: results + floor proof — 2026-06-05

Branch `nava-port`. RTX 3060, serial GPU. Locked config: q6_K + FA + tile16, 896x448, 10 steps,
13 frames (49 pixel frames), seed 42. Picks up from `HANDOFF-nava-PERF-BIBLE-KICKOFF.md`.
Tooling: nsys 2024.1.1 + ncu (both bundled in the toolchain nsight-compute dir; ncu needs sudo
for perf counters — `NVreg_RestrictProfilingToAdminUsers` is set on this host).

## HEADLINE: wall 175.9s → 136.8s (−22.2%), zero quality loss

| build | wall | DiT (s/step) | video VAE | audio VAE | note |
|---|---|---|---|---|---|
| baseline (locked) | 175.9s | 9.55 | 46.0s | 29.7s | |
| + audio F16 conv_transpose | 148.5s | 9.55 | 46.0s | **2.66s** | wmma path; ear-transparent (owner-confirmed) |
| + single-launch concat | 138.7s | 9.60 | **35.6s** | 2.66s | bit-identical RGB |
| + same-shape madd fusion | **136.8s** | **9.42** | 35.6s | 2.66s | bit-exact (0.0 waveform diff e2e) |

Quality: audio F16 is ear-indistinguishable from F32 (owner A/B on :8099, rows 30 vs 31; SNR
34.9 dB / cos 0.99984). The other two changes are **bit-identical** (decoded-RGB stats match to
all digits; end-to-end waveform diff exactly 0.0). VRAM peak unchanged (7579 MiB, DiT-phase).
Eye-page clips (http://10.0.0.208:8097): `perfbible_baseline` (old) vs `madd_fuse` (all opts) —
visually identical, audio transparent.

## THE PROFILING BIBLE (kernel-level, every phase)

### nsys per-phase kernel attribution (baseline, 1-DiT-step trace)
The whole-pipeline GPU time was dominated by TWO non-DiT pathologies that the wall-time split
(DiT 54% / video 26% / audio 17%) hid:

1. **audio VAE `conv_transpose_1d` = 27.1s across 11 launches (33.5% of ALL GPU kernel time)** —
   the single biggest kernel in the pipeline. The 11 HiFiGAN `ups.N` upsamplers (6 base vocoder
   + 5 BWE) are stored **F32** in the GGUF → the naive triple-loop F32 kernel. An F16-capable
   smem-tiled wmma kernel already existed (from qwen3-tts) but only dispatches on F16 weights.
2. **video VAE `concat_T_cont` = 9.9s across 6.1 MILLION launches (20% of the decode)** — pure
   launch overhead. Root cause: `concat_dispatch` host-looped over `dst->ne[3]` (=1024 channels)
   launching one kernel per slice; 720 dim-2 concats/tile × ~850 ne3 × 10 tiles ≈ 6.1M.

### ncu DiT roofline (the main event — PROVEN not at roofline, but on the practical floor)
Both dominant DiT kernels are **latency-bound from low occupancy** (register+smem pressure →
1–2 blocks/SM), NOT compute- or memory-bound:

| kernel | Compute(SM)% | DRAM% | occupancy | limiter | verdict |
|---|---|---|---|---|---|
| `mul_mat_q` (Q6_K, 4.19ms FFN) | 51.8% | 34.4% | 16.7% (achieved=theo) | 242 regs + smem → 1 blk/SM | latency-bound |
| `flash_attn` (big, L_q≈5148) | 35% | 18% | 24.6% / theo 25% | regs + smem | latency-bound |

ncu's own diagnosis: *"Achieved compute/bandwidth below 60% of peak typically indicate LATENCY
issues… 2.00 theoretical warps per scheduler… below hardware."* This is the **identical MMQ/
flash signature flux2 lap-6 and the turboquant laps already characterized and drove to the 3060
floor** (cuBLAS measured *slower* than MMQ there; the mmq_x 128→64 occupancy experiment was left
as an unconfirmed NEXTLAP — high-risk, shared `mmq.cuh`, uncertain payoff since more tiles = more
work). The Q6_K weight mass (mul_mat_q 41.5s/render) is immovable without a quality-gated requant.

## THE OPTIMIZATIONS (heads chopped, by measured size)

### 1. Audio VAE: F16 conv_transpose → wmma (−27.1s, the biggest win)
`ConvTranspose1D::init_params` (src/ltx_audio_vae.h) hard-coded `GGML_TYPE_F32`. Forced the 11
`ups.N` weights to **F16** → routes to the existing tensor-core `conv_transpose_1d_mma_kernel`.
Audio VAE decode **29.7s → 2.66s** (conv_transpose 27.1s → 0.04s). Ear-transparent. Envs:
`NAVA_CT1D_F32=1` (legacy F32), `NAVA_CT1D_F16_BWE_ONLY=1` (72.6 dB hybrid: F32 base vocoder,
F16 BWE only — unneeded since full F16 passes). Commit `ae2d63c`.

### 2. Video VAE: single-launch contiguous concat (−10.4s, ggml-cuda core fix)
Folded the `ne3` host-loop in `concat_dispatch` into the kernel (`concat_T_cont_4d`, grid-stride
over the full 4D extent): one ggml CONCAT op == one launch (was up to 1024). Video VAE decode
**46.0s → 35.6s**. Bit-identical. **Benefits every concat-heavy graph across all forks** (VAE/conv
stacks, joint-attention concats). ggml `0bcf0e83`, parent `2f41549`.

### 3. DiT: same-shape fused multiply-add (−1.9% DiT)
Generalized the lap-28.3 gate_add fusion to the no-broadcast case: NAVA's per-token AdaLN
modulation `x + x*scale + shift` (batch N=1, gate is full `[d0,L,1]` not a `[d0,1,..]` broadcast)
ran as 3 full-size kernels. New `fused_madd_same` + detection fuses any single-use
`ADD(x, MUL(a,b)) [+ ADD(_,shift)]`. DiT **9.604 → 9.423 s/step** (clean ON/OFF A/B). Bit-exact
(0.0 e2e waveform diff). Small because ggml interleaves the video/audio streams, breaking the
MUL→ADD adjacency the detector needs; the rest of NAVA's k_bin_bcast are gateless residual /
time-embed adds with no fusable partner (mirrors flux2 lap-8's conclusion). Env-disable
`GGML_CUDA_NO_MADD_FUSE=1`. ggml `292516d5`, parent `d7e2824`.

## WHERE TIME GOES NOW (136.8s) — the new floor
| phase | time | % | status |
|---|---|---|---|
| DiT sampling | 94.2s | 69% | floor — mul_mat_q 41.5s + flash 19.4s latency/occupancy-bound (shared MMQ/flash, prior-lap floor); adds fused where fusable |
| video VAE decode | 35.6s | 26% | ~22s real conv3d (im2col+gemm) + ~8s glue (pad/cpy/k_bin_bcast). im2col is memory-bound lowering; direct-conv3d would need a new kernel |
| audio VAE decode | 2.66s | 2% | floor — conv_transpose now 0.04s; residual is im2col (depthwise BWE filters) + matvec |
| mux | ~1.5s | 1% | — |

## REMAINING HEADROOM (documented, not taken — diminishing returns / out of scope)
- **DiT mul_mat_q occupancy** (mmq_x 128→64): the only large lever left, but it's the shared
  `mmq.cuh` kernel that flux2/turboquant deliberately deferred (uncertain: more tiles = more
  work; risk to all forks). Not worth it this session.
- **align_cfg 3rd forward** (~31s of DiT): `NAVA_NO_ALIGN_CFG=1` is −33% DiT but quality-load-
  bearing on hard prompts (HANDOFF-nava-ALIGN-CFG-REVIEW.md) — a quality tradeoff, not a free win.
- **video VAE im2col→direct conv3d** (~9.5s im2col lowering): needs a fused direct-conv3d kernel
  (deep). The conv gemms (~12.8s) are tensor-core and near-optimal.
- **video VAE glue** (~8s pad/cpy/k_bin_bcast): fiddly fusions, a few % at best (lap-8 territory).
- **audio VAE im2col** (1.5s): minor.

## DEPLOY NOTE
The ggml submodule commits (`0bcf0e83` concat, `292516d5` madd) are **local-only**. Both are
bit-exact + env-gated + universally beneficial, but ggml is shared across all GPU forks
(llama/flux2/siglip2/…). **Push dbrain/ggml before any Docker deploy** (Docker builds fetch the
pinned submodule SHA — the parent gitlink currently points at unpushed commits). The
ltx_audio_vae.h + main.cpp + wan.hpp changes are in the longcat-avatar.cpp repo itself.

## REPRO / instrumentation added this session
- `NAVA_DUMP_WAV=<path>` — dump raw planar waveform (for SNR A/B). examples/nava/main.cpp.
- `NAVA_VAE_OP_HIST=1` — per-tile VAE graph op histogram (found the concat catastrophe). wan.hpp.
- nsys traces + ncu reports in `bench/results/` (nava_full_1step, nava_postfix_1step, ncu_mulmatq,
  ncu_fa) + `bench/results/PROFILING-BIBLE-2026-06-05.md` (the raw kernel tables).
