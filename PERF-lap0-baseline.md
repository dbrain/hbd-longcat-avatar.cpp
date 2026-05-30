# lap-0: baseline + infra (2026-05-29)

FLUX.2-Klein on RTX 3060 12GB, "heavy bucket" beside prod LLM (time-share, not co-resident).
Fork of longcat-avatar.cpp on dbrain/ggml @048cba4d. Goal: lean VRAM + fast.

## infra
- repo `~/dev/flux2.cpp` (from longcat-avatar.cpp @a08224a). server target renamed → `flux2-server`.
- dev harness `~/dev/kobbler/docker/flux2-dev/` (mirrors longcat-avatar-dev). builder = CUDA 12.9.1 devel + nsys + ncu, sm_86.
  - `./iter.sh build` (sd-cli + flux2-server) / `serve` (resident :8095, klein-base+Q4enc+offload+fa) / `cli` / `shell` / `logs` / `stop` / `cycle`
- gallery `tools/gallery_server.py` :8096 — drives resident server, VRAM/timing/PSNR-vs-golden, Cancel button, civ presets. samples → `gallery/`.
- client `tools/flux_client.py` (submit/poll/cancel + VRAM sampler + log-scraped stage timings). shared by gallery + bench.
- bench `~/dev/bench/adapters/flux2.py` registered. lap-0 still CLI-based — TODO: move through resident server.
- models `models/` (~20GB): unet klein-base-9b-Q4_K_M (5.9G) + klein-9b-Q4_K_M (5.9G, edit), enc Qwen3-8B Q4_K_M (5.0G, floor) + Q8_0 (8.7G), vae full_encoder_small_decoder (249M).

## cancellation (shipped, validated)
- in-flight cooperative abort, 6 files. `sd_request_cancel/clear/is_cancel_requested` (stable-diffusion.h + util.cpp atomic); denoise lambda polls → `return {}` → sampler `{}` → clean fail. async worker: clear-before-publish + `generating_job_id` + Cancelled status. `/sdcpp/v1/jobs/{id}/cancel` fires only for active job; capability flags true.
- validated: cancel 30-step → cancelled in 1.6s (=1 step), worker stays warm, VRAM freed, next job ok.

## baseline (resident, klein-base Q4, Q4 enc, 512², 8-step, cfg5, euler, seed42)
| path | wall | DiT/step | cond | vae | VRAM peak |
|--|--|--|--|--|--|
| CLI cold | 45.4s | 2.19 | 1.88 | 0.38 | 6735 (+25.6s load) |
| resident warm | 19.6s | 2.18 | 1.65 | 0.40 | 6831 |
- resident kills the 25.6s CLI load; cold≈warm. **bench via resident server.**
- output bit-identical run-to-run (PSNR 999) → deterministic → PSNR gate reliable.
- civ set (caveman/roman/knight/musketeer/modern) all render clean ~19.6s.

## phase VRAM (idle 397)
- encode 0-1.65s: **6831 ← global peak** (Qwen3-8B-Q4 6342 + buffers)
- diffusion 1.65-19s: 6695 plateau, FLAT (UNet 5636 resident whole loop, no per-step paging) — DiT s/step is pure compute
- vae: brief 1248 buffer

## targets
- VRAM (peak encoder-bound): encoder streaming. then diffusion plateau (UNet + buffers).
- perf: DiT = 89% of wall. [lap-1: FA fused attention −23%]

## conventions
- per-lap docs `PERF-lapNN-*.md`, terse. memory stays clean.
- always bench resident server, seed 42, same config, PSNR-gate image. free prod LLM first (user does it).
