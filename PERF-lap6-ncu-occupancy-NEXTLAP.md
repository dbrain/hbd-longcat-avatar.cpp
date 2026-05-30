# lap-6: ncu UNBLOCKED — matmul is occupancy/latency-bound, NOT at floor (2026-05-30)

**HANDOFF for fresh agent. Earlier "matmul at floor / compute-bound" was WRONG (it was FLOP-math
inference, ncu was blocked). Real ncu data: 16.67% occupancy, latency-bound, ~50% SM idle.**

## ncu now WORKS — the unlock was `--cap-add SYS_ADMIN` (ERR_NVGPUCTRPERM is just missing the cap)
```
docker run --rm --gpus all --cap-add SYS_ADMIN \
  -v /home/dbrain/dev/flux2.cpp:/src -v /home/dbrain/dev/flux2.cpp/models:/models -w /src flux2-dev:builder bash -lc '
ncu --target-processes all --kernel-name regex:"mul_mat_q" --launch-skip 430 --launch-count 2 \
    --section SpeedOfLight --section Occupancy --section WarpStateStats --section SchedulerStats --section LaunchStats \
    /src/build/bin/sd-cli --diffusion-model /models/unet/flux-2-klein-base-9b-Q4_K_M.gguf \
      --vae /models/vae/full_encoder_small_decoder.safetensors --llm /models/text_encoders/Qwen3-8B-Q4_K_M.gguf \
      --offload-to-cpu --mmap --diffusion-fa -p "a knight" -W 512 -H 512 --steps 2 --cfg-scale 5 --seed 42 -o /tmp/n.png'
```
(launch-skip ~430 lands on DiT single-block linear1 = the 48%-of-GPU matmul, NOT the encoder. Stop any
resident server first — ncu needs the GPU. ncu replays each kernel ~7 passes, slow but fine.)

## the diagnosis (dominant DiT matmul mul_mat_q<12,128,0> = Q4_K, grid 384, ~1.9ms)
| metric | value | reading |
|--|--|--|
| Achieved occupancy | **16.66%** (8/48 warps/SM) | starved |
| Theoretical occupancy | 16.67% | hard cap (not runtime imbalance) |
| Registers/thread | **221** → Block Limit Registers = **1 block/SM** | co-limiter |
| Dynamic shared mem/block | **58 KB** → Block Limit Shared Mem = **1 block/SM** | co-limiter |
| Compute (SM) throughput | ~48-52% | half idle |
| Memory throughput | ~41-44% (L1/L2) | — |
| **DRAM throughput** | **~12%** | NOT memory-bound → headroom to trade |
| Eligible warps/sched | **0.81 of 2.0** → 46% cycles issue NOTHING | latency-bound |
| Warp cyc/issued instr | 3.7 | — |

=> latency-bound: too few warps (reg+smem capped) to hide instruction latency. ~50% SM unused.

## PRIMARY NEXT LAP — shrink MMQ tile (mmq_x 128 -> 64) to raise occupancy
- root: `get_mmq_x_max_host` (ggml/src/ggml-cuda/mmq.cuh:109) returns **128** for turing_mma (sm_86).
  the per-launch selector (mmq.cuh:**4069** `for mmq_x=8..mmq_x_max`) greedily picks the LARGEST mmq_x
  that minimizes tile count (ncols=1536 → mmq_x=128, 12 tiles). **It optimizes fewest-tiles, NOT
  occupancy.** mmq_x=128 → 221 regs + 58KB smem → 1 block/SM → 16.67%.
- hypothesis: mmq_x=64 (or 48) ≈ halves smem/regs → 2 blocks/SM → ~33% occupancy → hides latency.
  we're latency-bound w/ 12% DRAM, so smaller tiles (more tiles, more DRAM) should still WIN.
- test (cheap, ~1h): hack `get_mmq_x_max_host` to `return 64;` (or cap the 4069 loop at 64), rebuild
  (`iter.sh build`, ~45s ccache), A/B DiT s/step via resident server. re-ncu to confirm occupancy ↑.
  NOT bit-exact (diff accumulation order) but quality-equivalent → PSNR-gate vs golden (md5 6c0a783425ea
  baseline image in gallery/goldens; ~31dB = noise floor is fine).
- if win, make it conditional (only when ncols large / arch sm_86) so LLM path unaffected; or just a
  flux2-local build flag. ALSO ncu the smaller-tile kernel's WarpStateStats to see the new bottleneck.
- NOT YET DONE: pull `--section WarpStateStats` detail (stall_long_sb vs stall_mma vs stall_wait) +
  `--section SourceCounters` to confirm WHAT the 46% no-eligible cycles stall on (dequant? mma? smem?).
  That tells you whether occupancy is really the fix or if it's a stall ggml can't avoid.

## SECONDARY NEXT LAP — encoder layers 28-36 are computed & DISCARDED (CONFIRMED, bit-exact win)
- flux2-klein uses Qwen hidden states from layers **{9,18,27}** only (conditioner.hpp:**2082**;
  concatenated → context_in_dim=12288=3×4096). final norm output NOT used.
- but the layer loop (llm.hpp:**1077** `for i<num_layers`) runs ALL layers, and the GGUF has **36**
  (qwen3.block_count=36, verified). so layers 28-36 (25% of encoder) compute then get thrown away.
- fix: early-stop the loop at `max(out_layers)` when not return_all_hidden_states. bit-exact (those
  layers don't feed the used outputs... CONFIRM: layer i+1 output depends on layer i, but we only
  need up to 27, so stop after collecting 27). saves ~25% encoder compute (~0.25s of 1.6s cond) +
  ~1GB less weight to offload/H2D (those layers' weights). FREE, no quality cost.
- bonus: could drop layers 28-36 from the GGUF entirely (smaller file + less host RAM) but early-stop
  is safer/reversible.

## HARD CONSTRAINTS (from user, 2026-05-30)
- **VRAM ≤ 7.5 GB.** currently 6.3GB (6455 MiB), ~1.2GB headroom. KILLS "keep weights GPU-resident to
  skip the 1.34s/gen offload H2D" (needs ~10GB). offload-to-cpu STAYS.
- **NO quality loss.** base flux.2 = 8steps×cfg5 is the quality floor; below that needs the flux.2-EDIT
  model (4×1, "ok stuff", editing-oriented) — that's the "speed-at-quality-cost" answer, NOT a tune.
  So OFF the table: fewer steps, CFG-interval, step/feature caching (EasyCache/CacheDIT), DiT requant.
  Only LOSSLESS / quality-equivalent levers count now.
- Q4 = encoder quant floor.

## RULED OUT (measured, don't redo)
- **cuBLAS-F16**: rebuilt -DGGML_CUDA_FORCE_CUBLAS, measured **+4.5% slower + 312 MiB** (32F & 16F
  accum). int8 MMQ wins. (env GGML_CUDA_FORCE_CUBLAS is a #ifdef, not getenv — no-op as env.)
- **CFG-batch M1536→3072**: tile math says neutral (no wave-quant, GPU 99% busy, weights resident).
  [NOTE: revisit IF the mmq_x lever changes occupancy math — batching might interact.]
- **feature caching**: works (EasyCache −11%@thr0.2 / −25%@thr0.4) but quality cost; block-level
  (cache-dit/dbcache) defaults warmup=8 so useless at 8 steps. quality-gated out by user.
- **keep-resident / no-offload**: VRAM-gated out (>7.5GB).

## misc
- gallery (tools/gallery_server.py, :8096) had a bug: psnr() returned np.float64 → `==999.0` gave
  np.bool_ → json.dumps 500 on any 2nd-gen-of-a-key. FIXED (float(round(...))). gallery now shows the
  step ladder + easycache cards. drive experiments THROUGH /gen (POST form) so they appear as cards.
- shipped lossless wins so far: lap-1 FA, lap-2 mmap, lap-3 Q4_K-embd (peak 6831→6455, host RAM
  13.3→0.9GB non-reclaimable, all bit-identical md5 6c0a783425ea).
- DiT-phase kernel mix (lap1.sqlite, FA-on): mul_mat_q 69% / FA 8% / layout-copies 6.7% /
  modulation-bcast 5.5% / norms+rope 6% / quantize 2.1%. after matmul, the ~12% glue (concat/copy/
  modulation) is the next-largest LOSSLESS target (graph fusion) — not yet ncu'd.
