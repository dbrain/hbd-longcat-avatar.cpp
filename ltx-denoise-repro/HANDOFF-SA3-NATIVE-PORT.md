# LTX SageAttention3 Native Port — Current Handoff

> Related Relip work: [Lipdub / Relip speed and VRAM handoff](HANDOFF-LIPDUB-RELIP-SPEED-VRAM.md).
> The locked production policy is now `GGML_LTX_SA3=1` plus
> `GGML_LTX_SA3_POLICY=first`, not all-step SA3.

## Status

Native Blackwell SageAttention3 is working in ggml/CUDA for the eligible LTX
self-attention calls, but it is no longer the production default. The static
crossing A/B exposes FP4 "DiT dot"/background degradation with both FP32 and
F16 correction storage. Keep cuDNN as the quality default until LTX has a
model-specific selective-SA3 policy.

**Current priority:** make continuation appearance transfer functionally work.
SA3 performance is accepted; do not spend the next turn on performance or VRAM
unless it directly enables the appearance-transfer solution.

The implementation is in the nested `ggml` worktree on `sa3-native-port`.
Do not reset the root or nested worktree: both include unrelated user-owned
performance work and generated experiment artifacts.

## Default behavior and escape hatch

`GGML_SAGEATTENTION3=ON` compiles `fattn-sa3.cu`. Set `GGML_LTX_SA3=1` to
opt in to its native LTX self-attention contract:

- Blackwell GPU;
- B=1, H=32, D=128, contiguous self-attention;
- K/V F16; Q and output F16 or F32; no mask/bias/softcap.

With the variable unset (or set to `0`), the prior cuDNN route is the quality
default. This is deliberate: the upstream SA3 project does not guarantee
lossless acceleration for every model and its public API validates F16/BF16
QKV, while current LTX supplies F32 Q. Non-SA3 builds remain on their original
fallback path.

## Build

```bash
cd /home/dbrain/dev/longcat-avatar-ltxdenoise

docker run --rm \
  -v /home/dbrain/dev/longcat-avatar-ltxdenoise:/src \
  -w /src longcat-avatar-dev:builder-cudnn-ff \
  bash -lc 'cmake -S /src -B /src/build-sa3 \
    -DGGML_CUDA=ON -DGGML_CUDNN=ON -DGGML_SAGEATTENTION3=ON \
    -DCMAKE_CUDA_ARCHITECTURES=120 && \
  cmake --build /src/build-sa3 -j1 --target ggml-cuda && \
  cmake --build /src/build-sa3 -j12 --target sd-cli'
```

The `-j1` CUDA phase avoids compiler memory pressure in the SA3 CUTLASS TU.

## Produce the baked-default singing clip

The harness now defaults to `/src/build-sa3/bin/sd-cli`, so an SA3 build needs
no special environment variable:

```bash
cd /home/dbrain/dev/longcat-avatar-ltxdenoise

OUTDIR=$PWD/ltx-denoise-repro/_ablation_out/sa3_default_s3 \
OUT_NAME=sa3_default_s3.webm \
GPU=1 SEGMENTS=3 \
bash ltx-denoise-repro/run_singing_clip.sh
```

The locked recipe is 960x544 base -> x2 latent upscale/refine, 97 frames per
segment, 8 base steps, 3 refine steps, fixed seed 42, and original-song mux.
Use `GGML_LTX_SA3=0` on that command for a direct cuDNN comparison.

## Verified artifacts and numbers

Best quality/performance proof artifact:

```text
ltx-denoise-repro/_ablation_out/sa3_reuse_s3/sa3_reuse_s3.webm
```

- 3 segments, 243 stitched frames / 10.124 s;
- wall 577 s;
- peak SMI 11,766 MiB (10 MiB under 11.5 GiB / 11,776 MiB);
- normal Q/K/V data, no zero/clamp diagnostic behavior;
- seam frames inspected; user accepted visual quality subject to the known
  continuation identity issue.

The current adapter uses one padded F16 scratch buffer sequentially for Q, K,
V and output.  That recovered about 1 GiB versus the initial native port.

An attempted `MAXV=8.75` run was cancelled early: it expanded a base graph cut
from 6 to 50 segments, so it cannot satisfy the requirement of no material
speed regression.  The 11,766 MiB point is currently the right speed/VRAM
trade-off.

The chain peak is **not a cumulative per-segment leak**.  It first occurs in
segment 2's 32,640-token refine pass and segment 3 reaches the same shape and
high-water.  Segment 1's refine length is 26,624 padded tokens.  The larger
continuation shape raises SA3's live working set by about 427 MiB:

```text
                         L=26,624     L=32,640
FP32 delta_s correction     676 MiB      1,016 MiB
reused F16 scratch           208 MiB        255 MiB
FP4 Q/K/V + scales           176 MiB        215 MiB
SA3 live total             1,060 MiB      1,486 MiB
```

LTX's graph-cut persisted external inputs also grow from 1,297 to 1,585 MiB
(+288 MiB), and the graph cache grows from 207.6 to 255.5 MiB.  The CUDA VMM
pool retains its committed high-water between calls until an explicit trim;
that is reusable pool capacity, not accumulating live allocations, and cannot
lower the segment-2 peak.  SA3 no longer has avoidable duplicate full-sequence
buffers: Q, K, V and output share one scratch allocation.  The next real SA3
VRAM project, if the remaining 266 MiB matters, is an accuracy/performance
study of reducing or streaming the full FP32 `delta_s` correction tensor; the
upstream CUTLASS path currently consumes it as FP32.

### Experimental F16 delta correction

The locked singing repro defaults to F16 correction storage. It keeps the
correction GEMM's FP32 accumulation but stores `delta_s` in F16 and converts it
back to FP32 in the SA3 score accumulator. Set
`GGML_LTX_SA3_DELTA_F16=0` for the FP32 recovery/A-B path; other native SA3
callers remain FP32 unless they opt in.

Corrected candidate artifact:

```text
ltx-denoise-repro/_ablation_out/sa3_deltaf16_s3_fixed/sa3_deltaf16_s3_fixed.webm
wall 584 s; peak SMI 11,258 MiB
```

This clears the preferred 11,500 MiB target. A first F16 attempt had a TMA
packet-indexing bug and produced corrupted frames; ignore
`sa3_deltaf16_s3.webm`. The `*_fixed.webm` clip uses the corrected mapping and
passed a smoke plus sampled seam-frame sanity checks and was visually accepted.

## Attention performance evidence

`GGML_LTX_SA3_TIMING=1` enables CUDA-event instrumentation for diagnosis only;
it synchronizes each SA3 call and must not be used for headline wall time.

For exact long refine self-attention, B=1 H=32 L=26,520 D=128:

```text
native SA3 total:      69.431 ms average (warmup excluded)
  preprocessing:       15.219 ms
  CUTLASS MHA + output: 54.212 ms
prior cuDNN SDPA:      about 236.5 ms
```

This is 3.41x at the attention-call level and matches the upstream SA3 70.093
ms benchmark.  There is no obvious SA3-kernel speed gap remaining.  End-to-end
render speed cannot approach 3.4x because FFNs, cross-attention, graph cuts,
and VAE decode remain outside SA3.

Two rejected optimizations:

- Long-only SA3 dispatch: 594 s / 12,024 MiB.  cuDNN's continuation workspace
  was worse; retain SA3 for every eligible D128 LTX self-attention call.
- `cublasGemmStridedBatchedEx` for the 32 delta GEMMs: 582 s / 12,022 MiB.
  It reserves additional cuBLAS workspace; retain the per-head `cublasGemmEx`
  loop despite its extra host launches.

## Profile next, if more performance work is desired

The next performance task is system-level attribution, not SA3 kernel tuning.
Run a short fixed-seed SA3 profile first (profiling adds synchronization):

```bash
GPU=1 SEGMENTS=1 STEPS=1 REFSTEPS=0 \
GGML_LTX_SA3_TIMING=1 \
OUTDIR=$PWD/ltx-denoise-repro/_ablation_out/sa3_timing \
bash ltx-denoise-repro/run_singing_clip.sh
```

For Nsight Compute/System, use the harness `NCU=1` capability flag and set
`PROFILER` to the profiler command; restrict capture to the long-refine window.
Do not use `GGML_FP8_GEMM_PROFILE=1` for headline timing because it synchronizes
every GEMM.  A useful next question is how much wall time remains in FFN,
cross-attention, offload/copy, and VAE after SA3.

## Next functional target: continuation identity

The three segments render a different-looking singer at each boundary.  Treat
this as a continuation-data/refine-schedule problem, not an SA3 issue.

1. Trace the retained continuation latent/frame data through every base and
   refine step; prove that the correct overlap reaches each segment.
2. Establish a cuDNN-vs-SA3 fixed-seed A/B to show whether SA3 changes the
   identity behavior materially.
3. Sweep gentler refine schedules first.  The locked custom schedule starts at
   0.909375, which can re-roll the face.  Test e.g.
   `REFINE_SIGMAS="0.4,0.28,0.15,0.0"`, then lower starts, with the same seed.
4. Test `REFINE_CONST_SEED=1`; it was previously observed to reduce identity
   flash and should be measured with identical continuation inputs.

Do not change the SA3 default or its allocation scheme as part of identity
work unless an A/B proves a causal effect.

## Update — rejected refine-guide experiment (2026-07-10)

Rebuilding the stage-2 continuation guide mask and RoPE positions was tested
in `sa3_identity_guide_s2`. It did **not** remove the visible singer jump at
the seam, so the source change was reverted. Do not use that artifact as proof
of an identity fix.

The leading hypothesis is a base/refine hand-off mismatch: `chain_base_latent`
is captured before x2/refine in `src/stable-diffusion.cpp`, while the viewer
sees the independently high-sigma refined result. A first opt-in experiment
(`LTXAV_CHAIN_REFINED_TAIL=1`) tried to downsample the visible refined latent
back to the base grid for the next segment. It was also reverted: both attempts
logged `incompatible refined/base latent layouts` and fell back to the old base
hand-off. The outstanding work is to log the precise ranks/shapes and define a
valid video-only transport layout (the base transport carries packed audio at
this point); then run a controlled 2-segment A/B. Do not assume this hypothesis
is proven until that A/B visibly removes the jump.

The lower-risk functional controls remain the actual first sweep:
`REFINE_CONST_SEED=1` and progressively gentler `REFINE_SIGMAS` from the locked
`0.909375,0.725,0.421875,0.0`. Compare them with identical base continuation
inputs, fixed seed, and seam frames, before changing the default schedule.


The established profile still makes VAE decode the next largest performance
target after SA3, but 2x1/1x2 spatial tiles exceed the current VRAM envelope.
The promising constrained experiment is therefore to increase the decoder's
internal `LTX_VAE_CONV3D_WTILES/_HTILES` subdivision enough to make 2x1 fit;
this trades inner conv launches for half as many full spatial tile passes and
must be measured as a VRAM/time/quality three-way A/B.

## Update — dual-resolution continuation fixed identity (2026-07-10)

The identity jump was caused by a real two-resolution hand-off mismatch, not
native SA3: stage 1 continued from the low-resolution base latent tail, but
each new stage-2 refine had no access to the previous visible/refined identity.

`generate_video_chain()` now carries both valid states:

- the existing base-grid video-only tail continues stage 1;
- the last three refined high-resolution video-only latent frames are supplied
  to the next stage-2 pass as separate frozen LTX guide tokens at frame zero.

Audio is intentionally not transported in that second state. The next segment
retains its own driven audio latent. The relip two-stage branch is explicitly
excluded, so the production lipdub routing remains untouched.

This is default-on for hires chains. The redundant three low-resolution base-guide
frames are removed from stage 2 before the refined guide is attached; that is what
keeps the quality setting within the VRAM envelope. `LTXAV_CHAIN_HIRES_REFERENCE=0`
restores the old base-only refine behavior; `LTXAV_CHAIN_HIRES_REFERENCE_FRAMES=1..K`
and `LTX_REFINE_CONTEXT_FRAMES` remain diagnostic overrides. Three frames is the
locked quality point.

Validated artifacts:

- `extension_hiresref_sa3_f16_s2.webm`: full-resolution three-frame guide,
  strongest seam anchor but 12,904 MiB before context trimming.
- `extension_x2_hiresref3_ctx0_sa3_s2.webm`: full-resolution three-frame guide
  plus trimmed base-guide context; clean seam; 11,768 MiB peak (8 MiB below the
  11.5 GiB / 11,776 MiB envelope). This is the locked configuration.
- `extension_hiresref1_cudnn_s2.webm`: earlier one-frame cuDNN transport control,
  which confirms the mechanism works independently of SA3; 11,334 MiB peak.

The separate `run_ltx_relip.sh` production path intentionally runs the sibling
relip binary. That canonical baseline still completed the 25-frame 1280x704
two-stage lipdub smoke in 92.66s. The target worktree's own relip branch
segfaults before sampling even with cuDNN and without this feature; this is a
pre-existing local relip issue, documented separately in `LIPDUB-RELIP-FAILURE.md`.
