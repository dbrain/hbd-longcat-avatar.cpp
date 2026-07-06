# LTX-2.3 quant sweep — results (overnight run)

All 1920×1088, dev+distill@0.5, seed 42, 8-step base + 3-step refine, MAXV=7/TBF=3/VWT=16.
Eye-test: http://10.0.0.208:8077/ltx_denoise/sweep.html

## Timing / VRAM (measured, solid)
| model | speed | peak VRAM | verdict |
|---|---|---|---|
| imatrix-fp4 (calibrated nvfp4) | 445s | 15.7 GB | **FREE vs RTN** — same speed+VRAM as baseline |
| old-nvfp4 (RTN, dev050) | 445s | 15.7 GB | baseline |
| q4km (dev UD-Q4_K_M folded) | 480s | 15.6 GB | **~8% SLOWER than nvfp4** — K-quant dequant path loses to tensor cores |
| comfy-fp8 (reference) | ~400s | — | comfy's own render |

Key: a properly imatrix-calibrated nvfp4 costs NOTHING over RTN → if cleaner in motion, it's a pure win.
Q4_K is NOT a speed win (slower than nvfp4 despite similar size).
Quality (imatrix vs RTN mush) = owner's motion A/B call.

## Tooling built (reusable)
- GGML_TYPE_F8_E4M3 weight type (ggml + convert `tools/import_ltx_fp8.py` + F8 cuBLASLt GEMM). Env: GGML_CUDA_F8_GEMM.
- imatrix pipeline: `IMATRIX-RUNBOOK.md`, `tools/fold_ltx_distill_bf16.py` (fold distill@0.5 into bf16), `-M vid_gen --collect-imatrix` + `-M convert --type nvfp4 --imatrix`. Output is a drop-in for nvfp4-CLEAN-dev050 (no wglobals, same flags).
- sweep infra: `sweep_render.sh` (OOM-retry, peak-VRAM capture), `sweep_page.sh`.
- models: /ltx2/nvfp4-imatrix-dev050.gguf, /ltx2/dev050-q4k.gguf (both folded dev+distill@0.5).

## fp8 — SHELVED (9 rounds)
F8_E4M3 type works mechanically (loads/runs/renders/banks latents). Output is NaN→white.
- Ruled out: Q-overflow (Q~10 fits F16), weight scale (fixed: A_scale=amax(bf16)/448), activation scale (standard amax/448), act-cache cross-stream race (serialized streams still white), BF16→F16 weight range (weights max 5.3, fit F16).
- LEADING (code-confirmed, runtime-UNconfirmed) mechanism: **F16 output-store overflow** — the fp8 GEMM accumulates F32 but stores F16 (out_dt=CUDA_R_16F); deep-block outputs exceed 65504 → inf on store → NaN. Nondeterminism (our 23dB nvfp4 issue) makes the first-NaN node wander. This likely ALSO explains the nvfp4 nondeterminism.
- Fix implemented (unvalidated): `GGML_F8_CLAMP_OUT=1` — F32 GEMM temp → clamp ±65504 → F16 store. Could not validate: the 26GB fp8 model + fix's F32 temp OOMs on the 16GB 5060 Ti for EVERY test render (LTX_DIT_F16=0 doubles VRAM too). **The blocker is hardware VRAM, not logic.**
- Revisit: bigger GPU, or a slimmer fp8 build. comfy-fp8 covers the quality reference column.
- fp8 is ~2x slower than nvfp4 anyway (26GB offload) → low practical value vs imatrix-fp4.
