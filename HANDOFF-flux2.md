# flux2.cpp — handoff / index  (other HANDOFF*.md here are stale longcat clones, ignore)

FLUX.2-Klein image gen on RTX 3060, heavy-bucket beside prod LLM. Lap-based perf.
Per-lap detail in `PERF-lapNN-*.md` (terse). Project memory: `project_flux2_cpp`.

## run it
- build: `~/dev/kobbler/docker/flux2-dev/iter.sh build`  (sd-cli + sd-server, sm_86)
- NOTE: the server CMake target is the shared-fork-generic `sd-server` (was `flux2-server`); the built binary still serves the flux2 sdcpp async API.
- serve: `iter.sh serve` → resident server :8095 (klein-base + Q4 enc + offload + fa)
- gallery: `cd tools && python3 gallery_server.py` → :8096 (gen UI, Cancel, PSNR-vs-golden, civ presets)
- driver: `tools/flux_client.py` (submit/poll/cancel + VRAM + stage timings); bench `~/dev/bench/adapters/flux2.py`
- GPU is single — free prod llama-server first (user does it).

## state (2026-05-29)
- infra + cancellation shipped (lap-0). FA fix (lap-1, −23% DiT). mmap (lap-2) + Q4_K-embd (lap-3). all uncommitted.
- klein-base 512²/8-step/cfg5 → **~15.8s warm, 1.69 s/step, VRAM peak 6455 MiB, ~0.9 GB non-reclaimable host RAM**.
- deterministic output → PSNR gate reliable. goldens FA-on (lap0 archived gallery/goldens_lap0). md5 `6c0a783425ea` (laps 1-3 all bit-identical).
- profiles in `profiles/` (lap0 unfused, lap1 FA). serve now defaults `--mmap`.
- **lap-2 (mmap):** `--mmap` flag was just never passed (NOT a lap-32 conflict). Weights → reclaimable page cache. host RSS 13.3→3.6 GB, `available` 13→23 GB. +0.5s warm, bit-identical. one-time cold page-in ~34s (acceptable, that's the point).
- **lap-3 (Q4_K embd):** Qwen token_embd was force-F32 (2374 MB pinned) because `support_get_rows` didn't whitelist Q4_K — but CUDA get_rows supports it. One-line whitelist → embd stays Q4_K + mmappable. pinned host RAM 2.56 GB→130 MB (non-reclaimable total ~0.9 GB), and VRAM peak 6831→6455 (encoder no longer the wall; DiT plateau is). bit-identical.
- **perf — see PERF-lap4 (matmul NOT at floor).** whole-run "66% matmul" includes the Qwen encoder (~26% of it). DiT-phase-only: mul_mat_q 69%, FA 8%, layout-copies 6.7%, modulation-bcast 5.5%, norms+rope 6%, quantize 2.1%. GPU 99% busy (not launch-bound). Dominant = single-block linear1 (N=36864, M=1536), 48% of ALL GPU time, but only **~38% of int8 TC peak** (Q4_K dequant overhead). `ggml_cuda_should_use_mmq` picks MMQ unconditionally on sm_86 — never compares cuBLAS-F16. Pending (GPU free): (1) FREE `GGML_CUDA_FORCE_CUBLAS=1` A/B, (2) CFG-batch N=2 → M=3072. Quant-angle (Q5_0/Q4_0 DiT, faster dequant) = quality tradeoff, ask first.

## three optimization axes (VRAM + host-RAM first per user, then perf)
- **loaded host RAM ~13 GB** (unload-on-idle already exists fleet-wide — goal is to shrink the LOADED footprint, not free-when-idle). whole model held in RAM by --offload-to-cpu, mmap disabled → real/likely-pinned anonymous alloc. levers: (a) **mmap the offloaded weights** → file-backed reclaimable page cache instead of locked RSS; per-gen H2D is once (UNet paged in at diffusion start, resident whole loop ~flat plateau), so pageable-vs-pinned costs only ~+0.5-1s/gen for ~12 GB freed. investigate WHY offload disabled mmap (lap-32 pinned-buffer contiguity?). (b) encoder-streaming (below) also cuts loaded RAM. NOTE: flux2 in-process server likely lacks the worker-iso unload wiring qwen3-tts/parakeet have — port later if wanted, but secondary to footprint.
- **VRAM peak 6831 = encoder-bound**: Qwen3-8B-Q4 (6342MB) loads monolithic + resident for the 1.65s encode. lever = stream encoder layers via lap-32 per-segment offload (offload_ctx/partial_offload_pairs, ggml_extend.hpp) — currently DiT-only. deeper change. Q4 = quality floor. (also cuts host RAM.)
- diffusion plateau 6479 (UNet 5636 + ~840 buffers) = 2nd VRAM wall.
- **perf**: matmuls 66% (Q4_K 50.6 + Q5_K 14.8). next = `ncu` top `mul_mat_q<12,128>` mem-vs-compute → MMQ tune / mma-vs-dp4a (M=1536≫1 → mma should win). also edit-9b cfg1 = 1 fwd/step vs base 2.
- TODO: bench adapter still CLI-based → point at resident server.
- later: kobbler/koblem integration (flux heavy-bucket container + /image endpoint).

## gotchas
- ports 8090/8091 = prod koblem; flux 8095, gallery 8096.
- build/ root-owned; write outputs to gallery/ or /tmp.
- FA2DBG debug logs in ggml_extend.hpp ~1442 (DEBUG, harmless) — gate/drop before commit.
