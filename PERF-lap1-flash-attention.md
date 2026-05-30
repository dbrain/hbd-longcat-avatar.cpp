# lap-1: engage flash attention (2026-05-29)

**DiT 2.18→1.68 s/step (−23%), wall 19.6→15.6 s (−20%). One-line fix. Quality preserved.**

## bug
- `--diffusion-fa` logged on, but profile showed 0 fattn kernels — attention ran unfused (KQ GEMM→scale→soft_max→PV).
- cause: fork added empty no-op `DiffusionModelRunner::set_flash_attention_enabled(bool){}` (diffusion_model.hpp:114) that name-hides the working `GGMLRunner::set_flash_attention_enabled`. `FluxRunner` doesn't override → `flash_attn_enabled` stayed false. Hit every non-avatar model (avatar/hidream override).

## fix
- diffusion_model.hpp:114 → `{ GGMLRunner::set_flash_attention_enabled(enabled); }` (forward to base, matches param-method pattern at 119-123).
- FA2DBG debug logs left in ggml_extend.hpp ~1442 (DEBUG level). Confirmed `flash ENGAGED L_q=1536 L_k=1536 d_head=128` ×64/step.

## numbers (resident, klein-base Q4, Q4 enc, 512², 8-step, cfg5, seed42, warm)
| | lap0 | lap1 |
|--|--|--|
| DiT s/step | 2.18 | **1.68** |
| wall | 19.6s | **15.6s** |
| VRAM peak | 6831 | 6831 (encoder-bound) |
| diffusion plateau | 6695 | 6479 (−216) |

- profile: `flash_attn_ext_f16` now 7.1%; soft_max 13.5%→gone; scale 5.5%→0.2%; QK/PV GEMMs 10%→0.3%.
- matmuls now 66% (Q4_K 50.6 + Q5_K 14.8 + Q6_K 0.5) ← next perf target.
- quality: `gallery/_lap0_vs_lap1.png` identical knight; PSNR 31dB (fused-vs-unfused rounding over 8 steps, not a regression). goldens re-baselined FA-on (`goldens_lap0/` archived).

## next
- VRAM (peak encoder-bound): stream Qwen3-8B-Q4 encoder layers (6342MB resident 1.65s → ~1 layer). check offload per-layer granularity. Q4 = quality floor.
- perf: Q4_K/Q5_K MMQ = 66%. ncu the top mul_mat_q (mem vs compute bound) → MMQ tune / mma-vs-dp4a (M≫1). profiles/lap1.nsys-rep.
