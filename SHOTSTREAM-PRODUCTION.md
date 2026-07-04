# ShotStream — Production Recipe (RTX 3060 12 GB / RTX 5060 Ti 16 GB)

**The go-to reference so we stop losing this.** Streaming multi-shot text→video,
Wan2.1-T2V-1.3B causal-distilled fork, C++/ggml port. Last verified 2026-07-05.

> **STATUS: BANKED, NOT PRODUCTIONISED (2026-07-05).** The pipeline is fast (see perf below) but the
> 4-step distilled model is **inherently low/bad-motion** — subjects barely move, and when they do it's
> distorted. It's a tech win (nvfp4 + cuDNN attn + custom kernels, all reusable on a better model), not a
> shippable toy. The **nvfp4 DiT is the live model** (f16 deleted). All perf learnings recorded below.
>
> **5060 Ti (nvfp4) = ~49 s/shot** (DiT 35 s + decode 14 s), DiT 827 MB, peak 9.7 GB — 1.9× the 3060.
> Levers: cuDNN SDPA attention (optimal, −26% DiT), nvfp4 FP4/FP8 GEMM (−1.8 GB, ~8%), `WAN_DIT_F16`
> (~6% conversion churn). Dead ends: FP8 flash attention (cuDNN F16 already faster), FP4 *activation*
> (the "worm/blob"). Add to the recipe for the 5060: `GGML_NVFP4_CUBLASLT=1 GGML_NVFP4_QUANT_TWOLEVEL=1
> GGML_FP8_FFN=1 GGML_FP8_LAYERS=blocks. WAN_DIT_F16=1`.

## Git provenance (both on master)
- **shotstream** (`dbrain/hbd-longcat-avatar.cpp`): `14c855b`
- **ggml submodule** (`dbrain/ggml`): `a1a27a9b`
- Build image: `longcat-avatar-dev:builder-cudnn-ff` (cudnn9 + cudnn-frontend + ffmpeg + nsys; arch `86;120` runs on both 3060 and 5060). Dockerfile: `kobbler/docker/longcat-avatar-dev/Dockerfile.cudnn`.

## Models (`/models/`)
| role | file |
|---|---|
| DiT | `shotstream-1.3b-dit-nvfp4.gguf` (866 MB — **the live model**; f16 deleted) |
| text encoder | `longcat-umt5-xxl-q8_0.gguf` (runs on CPU / host RAM) |
| VAE | `longcat-wan-vae-f16.gguf` |

To re-make the nvfp4 gguf from an f16 source (if ever needed): `sd-cli -M convert -m <f16>.gguf --type nvfp4 --tensor-type-rules 'modulation=f32,time_projection=f32,head\.=f32' -o <out>.gguf`.

## Output config
832×480, 16 fps, **7 chunks/shot** = 21 latent frames = **81 pixel frames/shot**.
4-step warped DMD schedule (sigmas `[1000, 957.929, 888.889, 737.589]`). Multi-shot chains via
dual KV cache (local 21-frame window + 6-frame global context) + per-chunk clean-rewrite.

## THE PRODUCTION ENV RECIPE
Run via `run_shotstream_chain_timed.sh` (which already sets the DiT/pipeline block below); pass the
VAE-decode block as `EXTRA_ENV` + `NO_VAE_TILE=1`:

```bash
TAG=prod SHOTS=2 DEVICE=0 NO_VAE_TILE=1 \
  EXTRA_ENV="WAN_VAE_F16=1 GGML_CUDNN_CONV=1 WAN_VAE_RMS_KERNEL=1" \
  ./run_shotstream_chain_timed.sh
```

### DiT / pipeline (default in the chain script — don't need to re-pass)
| flag | why |
|---|---|
| `SHOTSTREAM_NO_OFFLOAD=1` | DiT resident in VRAM (offload is a net loss here) |
| `SHOTSTREAM_INGRAPH_KV=1` | E5: in-graph K/V append, no GPU→host→GPU round-trip (default ON under NO_OFFLOAD) |
| `SHOTSTREAM_LASTCHUNK_REWRITE` (skip) | skip the last chunk's clean-rewrite — provably bit-identical (its K/V is never read) |
| `GGML_CUDNN_ATTN=1` | cuDNN SDPA for DiT self-attn (5060/sm120); 3060 uses the custom D=128 flash kernel |
| `GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1 GGML_CUDA_RMS_MOD_FUSE=1` | fused elementwise/norm kernels |
| `LONGCAT_FFN_TILE_TOKENS=4096` | FFN tiling |

### VAE decode (pass via EXTRA_ENV — this is the whole-frame fast path)
| flag | why |
|---|---|
| `--no-vae-tiling` (`NO_VAE_TILE=1`) | **whole-frame decode** — no tile seams, faster than the 4-tile path |
| `WAN_VAE_F16=1` | F16 VAE decode stream (matches reference bf16; output bit-close to F32) |
| `GGML_CUDNN_CONV=1` | route VAE 2D convs to our `conv2d-cudnn.cu` (im2col-free, tensor-core, F32-accum) |
| `GGML_CUDNN_CONV3D=1` | VAE 3D convs via cuDNN |
| `WAN_VAE_RMS_KERNEL=1` | channels-last RMS kernel — the copy-overhead fix (decode 29→21 s) |

### DO NOT set
- **`WAN_VAE_SLICE_NOCOPY`** — costs +100–330 MB VRAM for <1% speed. Left OFF.
- **temporal VAE tiling** — the reference is whole-frame; keep it off (whole-frame path).
- **VAE param offload** (`SHOTSTREAM_VAE_OFFLOAD`) — net negative.

## Performance (3060, 2-shot 21-latent)
| | value |
|---|---|
| DiT | ~74 s/shot — **floored** (Ampere attention occupancy + cuBLAS roofline; GPU ~96% util) |
| VAE decode | **21 s** (was 74 s tiled → 28.85 s cuDNN conv2d → 21 s RMS kernel = 3.5×; ~PyTorch parity) |
| **peak VRAM (2-shot)** | **11,499 MiB** — under the 11.5 GB (11,500 MiB) budget (deterministic; **1 MiB margin**) |

**Budget note:** the 11,499 peak is the whole-frame decode co-resident with the DiT params. It's a
deterministic gallocr reservation (stable run-to-run), but there's no headroom. To reclaim ~445 MB
(→ ~11 GB), smem-stage the RMS kernel (`ggml_rms_norm_channels`) — future work, not needed to meet budget.

## How we got here (so we don't re-derive it)
- Whole-frame decode fits via `WAN_VAE_F16` + a per-frame im2col that TSPLIT bounded, then **cuDNN conv2d**
  (`GGML_CUDNN_CONV`) made it im2col-free — our own `conv2d-cudnn.cu`, nothing from upstream ggml.
- Decode was **copy-bound**, not compute-bound (nsys: `cpy_scalar` = 25.7%, all from RMS doing 2× `ggml_cont`).
  Fixed with `GGML_OP_RMS_NORM_CHANNELS` (channels-last RMS, one coalesced pass) → cpy_scalar 25.7%→4.9%.
- Multi-shot fit via **pool-trim** at shot boundaries (`ggml_backend_cuda_trim_pools`) — the ggml-cuda VMM
  pool never shrinks on free, so the decode high-water sat committed into the next shot's DiT.
- `d_head=384` flash tile kernel (VAE mid-block attn) is engaged but perf/peak-neutral @832×480 — kept for higher-res.
- **Motion** ("ghosted/weird" fox) = inherent to the 4-step distilled model, NOT a port bug (temporal path
  matches the PyTorch reference line-by-line). Use a seed sweep for motion quality.
- **nsys profiling**: the builder image needs `cuda-nsight-systems-13-0` (baked into Dockerfile.cudnn now);
  the ncu-bundled nsys has a broken importer. Use `prof_shotstream_nsys.sh`.
