# Wan2.2-VACE continuation — PRODUCTION recipe (the one obvious place)

Single source of truth for the flags/env that make VACE continuation fit + run on the 16GB 5060 Ti.
`iter_seg2.sh` bakes these as defaults; `run_vace_*.sh` should source the same. Don't re-derive — read this.

## The VRAM story (measured with `LONGCAT_VRAM_BREAKDOWN=1`, per-compute-buffer `[VRAM]` lines)

At 1280×704 the peak is TWO compute buffers, not the weights or the TE:
- **TE (umT5 q8_0): 759 MB** — released after encode; never the problem.
- **VAE decode: was 10,941 MB** — the real ceiling. FIXED (see below) → **686 MB**.
- **DiT (Wan-VACE-14B): 6,529 MB (F32) → 5,584 MB @25f (F16)**; scales O(frames × spatial). At 65f ≈ 8,124 MB. This is the remaining wall for high frame counts at full res.

## The two engine fixes (this session, uncommitted — in `src/model/diffusion/wan.hpp`)

1. **VAE decode buffer: temporal streaming.** `--temporal-tiling` is a **silent no-op** for the Wan VAE (only LtxVAE overrides it). The real mechanism is env **`LONGCAT_VAE_TEMPORAL_CHUNK=1`** — decodes 1 latent frame per graph, seam-free (causal cache), composes with spatial tiling. Combined with a smaller spatial tile **`--vae-relative-tile-size 0.25x0.25`** the decode buffer drops **10,941 → 686 MB**. (Do NOT use `WAN_VAE_F16` — it trips `im2col.cu:548` F32 assert in this decode path and isn't needed once temporally chunked.)
2. **DiT F16 residual for VACE (`WAN_DIT_F16=1`).** i2v had it; VACE didn't (F16 attn output into F32 vace stream → `binbcast.cu:261`). Two self-gated fixes: upcast the F16 main stream at the vace `before_proj` add (`wan.hpp:~602`), and **cast the vace control stream `c`→F16** (`wan.hpp:~918`) so the vace-block Linears (nvfp4 weights) emit F16 too. Drops the DiT buffer + halves the main residual stream. Default (no `WAN_DIT_F16`) = byte-identical.

## Canonical PROD env (baked into iter_seg2.sh defaults)

```
# nvfp4 compute stack (Blackwell FP4/FP8 tensor cores) — same as run_wan22_i2v_nvfp4.sh:
GGML_NVFP4_CUBLASLT=1 GGML_NVFP4_QUANT_TWOLEVEL=1
GGML_FP8_FFN=1 GGML_FP8_LAYERS=blocks.
GGML_CUDNN_ATTN=1                       # cuDNN flash-attn (O(N) memory)
GGML_CUDA_F16_BCAST_FUSE=1 GGML_CUDA_BIAS_GELU_FUSE=1 GGML_CUDA_BIAS_RMS_FUSE=1 GGML_CUDA_RMS_MOD_FUSE=1
WAN_DISTILL_SIGMAS=1 WAN_DISTILL_SHIFT=5
# fit levers (this session):
WAN_DIT_F16=1                           # F16 residual stream (now VACE-safe)
LONGCAT_VAE_TEMPORAL_CHUNK=1            # VAE temporal streaming (the fix --temporal-tiling should be)
# CLI:
--vae-relative-tile-size 0.25x0.25 --vae-tiling --temporal-tiling
--offload-to-cpu --mmap --max-vram 10   # ~9.5-10.5 is prod; NOT a per-render toggle
--diffusion-fa --sampling-method euler_a --eta 1.0 --steps 2 --high-noise-steps 2 --flow-shift 5
```

Diagnostics: `LONGCAT_VRAM_BREAKDOWN=1` (forwarded by iter_seg2's FWD loop) → grep `[VRAM]` in `cont_<tag>/log`.

## Measured frame-count / resolution vs VRAM (max-vram 10, full prod stack)

| frames | res | ~sec | peak VRAM | ≤11.5GB |
|--------|-----|------|-----------|---------|
| 25 | 1280×704 | 1.5s | 9,955 MB | ✅ |
| 41 | 1280×704 | 2.5s | 12,107 MB | ✗ (just over) |
| 65 | 1280×704 | 4.0s | ~13,300 MB | ✗ (DiT buffer wall) |
| 65 | 1024×576 | 4.0s | 11,951 MB | ~ (0.45GB over) |
| 65 | 960×544  | 4.0s | 11,005 MB | ✅ |

**The rule:** the peak is the DiT activation buffer, which is **O(frames × spatial-tokens)**. At 1280×704 the ≤11.5GB ceiling is ~**33 frames** (2s). For the **64f (4s) target at ≤11.5GB, drop to ≤~960×544** (or 1024×576 + no-prefetch). To get 64f at full 1280×704≤11.5 would need deeper DiT F16 (the FP8-FFN intermediate + cuDNN attention scratch are the remaining non-F16 / overhead pieces — see the F16-islands report) — not done.

### Extra VRAM levers (if you need more headroom, e.g. 64f@1024 under 11.5)
- `LONGCAT_NO_OFFLOAD_PIPELINING=1 LONGCAT_NO_PREFETCH_POOL=1` — drops the double-buffered prefetched weight chunk (~1.9GB), at some speed cost. 64f@1024×576 → ~10GB.
- Lower `--max-vram` offloads more weights (frees VRAM for the buffer) — but the buffer, not the weights, is usually the ceiling, so this helps less than the above.

### DON'T
- `WAN_VAE_F16` — trips `im2col.cu:548` F32 assert in this decode path; unneeded once temporally chunked.
- `--temporal-tiling` alone — silent no-op for the Wan VAE (use `LONGCAT_VAE_TEMPORAL_CHUNK`).
- `CUDNN_OFF` (plain ggml flash) — asserts under the F16 residual; keep cuDNN attn.
- Treating `--max-vram` as the fit lever — it's not; shrink the buffers.

## Quality gate (owner eye-test, per no-declaring-winners)
`WAN_DIT_F16` for VACE changes the residual stream to F16 — validated to RENDER clean at 25f but the motion/quality eye-test vs the F32 path is the gate. The VAE temporal-chunk seams are structurally seam-free (causal cache) but eye-test the real artifact.
