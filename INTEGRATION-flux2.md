# flux2 perf — integration handoff (read me first)

This repo is the unified `stable-diffusion.cpp` fork serving **both** longcat-avatar (video)
and **FLUX.2-Klein** (image gen). This branch = avatar master (`c7d282b`) + 2 flux2 perf commits
on top. **Everything is bit-exact** (golden md5 `6c0a783425ea`) and **builds clean**. **Not pushed.**

## State
```
branch (= origin/master + 2, clean fast-forward):
  c750940  flux2: perf-lap writeups (PERF-lap0..11) + bench tooling
  4fcabbd  flux2: perf laps — FA fix, Q4_K-embd mmap, encoder early-stop, keep-resident A+C
  c7d282b  (origin/master) avatar: umT5 free + cooperative cancel
```
- `origin` = `git@github.com:dbrain/hbd-longcat-avatar.cpp.git`.
- Validated post-rebase: flux2 512²/8-step = wall 15.33s, cond 1.05s, dit 1.716/step, peak 6549 MiB,
  **md5 6c0a783425ea (bit-exact)**. Avatar code compiles in the same build (routes_longcat/worker_ipc).

## To push (whoever does it)
The 2 commits are a clean **fast-forward** on `c7d282b`. Push `master` (or a review branch first if you
want the avatar agent to bench before it lands). The local branch is still named `longcat-avatar-port`
(= master + perf) — rename if you want a cleaner name. **No `--force` needed; it's linear.**

## What's in the perf commit (4fcabbd) — all bit-exact, default-on unless noted
- **FA fix** — `DiffusionModelRunner::set_flash_attention_enabled` no longer name-hides the base setter
  (FA was silently off for non-avatar models). −23% DiT for flux2. **Avatar/hidream nested-runner
  wrappers still override → avatar behaviour unchanged.**
- **Q4_K-embd mmap** — `support_get_rows` whitelists Q4_K so the Qwen token_embd stays Q4_K + mmappable
  (was force-F32, 2.4 GB pinned). −376 MiB VRAM, bit-identical.
- **Encoder early-stop** — LLM text encoder stops at `max(out_layers)` (flux2-klein uses {9,18,27}/36).
- **Lever A** — keep text-encoder resident across the two CFG (cond+uncond) encodes (cond 1.72→1.04s).
- **Lever C** — keep the UNet resident across `batch_count` seeds (was reloading 5.6 GB/seed; 3-seed
  batch −10%, VRAM flat).
- `FLUX2_TEXT_MINLEN` env (default 512, gated `VERSION_FLUX2_KLEIN`) — opt-in text trim, **default-off**
  (quality tradeoff, see PERF-lap9).

## PROD runtime flags (NOT compiled-in — set these in the flux2 entrypoint)
`--offload-to-cpu --mmap --diffusion-fa`  →  FA on, mmap'd reclaimable weights, CPU-offload (≤7.5 GB).
The dev harness (`docker/flux2-dev/iter.sh serve`, in the kobbler repo) already passes these.

## For the avatar's "use mmap too"
mmap is just the `--mmap` flag on the avatar entrypoint (file-backed reclaimable weights vs locked RSS).
The Q4_K-embd whitelist + FA fix are already merged here and don't touch avatar output. Bench it.

## Held-back: shared ggml changes (NOT applied — see patches/)
`patches/ggml-flux2-lap8-modulate-fusion-and-lap7-mmq-scaffolding.patch` holds two ggml-side changes,
kept OUT of the build because `ggml/` is shared `dbrain/ggml` (all 6 GPU projects) and the gitlink stays
at `048cba4d` to match master:
- **lap-8 modulate shift-fusion** — folds flux AdaLN `x*scale+shift` into the longcat gate_add kernel.
  Bit-exact (−0.7% dit). Worth landing in dbrain/ggml.
- **lap-7 MMQ occupancy scaffolding** — `GGML_MMQ_X_CAP` env + `MMQ_EXPERIMENT_MIN_BLOCKS`/`MMQ_Y_OVERRIDE`
  defines, all **default-off / upstream**. Dead-end experiments (kept for reproducibility, see PERF-lap7).
To enable: apply the patch in `ggml/`, commit+push to `dbrain/ggml`, bump the gitlink. The build is
bit-exact WITHOUT it, so it's optional.

## Facts for the UI / next integrator (measured; full detail in PERF-lap10 + project memory)
- **Multi-seed:** `batch_count` (= `batch` in flux_client / gallery) renders seeds = seed, seed+1, …
  sequentially; shares the text-encode + (Lever C) the UNet load; VRAM peak flat. Default workflow = 3 seeds.
- **Resolution:** dims must be ×16; **non-square OK**; VRAM scales with pixels (512²=6.5 GB, 1024²=7.8 GB,
  1280²=8.9 GB). VAE decode is the high-res wall (OOM ~1536² w/o tiling); `vae_tiling` param caps it
  (2048²≈11.7 GB, card edge). klein is a ~1 MP model — quality artifacts above ~1024². For the 7.5 GB
  budget, practical max ≈ 1024×768.
- **Edit model:** `models/unet/flux-2-klein-9b-Q4_K_M.gguf`, recommended **cfg1 + 4 steps** (cfg1 = 1
  fwd/step → ~4× cheaper than base). Takes reference/init images.

## Don't re-investigate (closed with evidence, see PERF-lap4..11)
matmul is arithmetic-pipe-bound (not occupancy/latency) — no quality-neutral path; FA at occupancy
ceiling; cuBLAS +4.5%; CFG-batch neutral; mmq tile-shrink slower; UNet step-0 H2D pipelining infeasible
(segment-granularity); text-trim is a quality tradeoff. The lossless surface is exhausted.
