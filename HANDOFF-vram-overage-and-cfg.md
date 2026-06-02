# HANDOFF — lap-34 VRAM overage (URGENT) + CFG-in-request (2× prod cost)

Date: 2026-05-31. Author context is exhausted; this is for a fresh agent. Two independent problems
surfaced while shipping the offload prefetch-thread optimization (lap-33/34). **#1 is urgent and is
LIVE IN PROD RIGHT NOW.**

═══════════════════════════════════════════════════════════════════════════
## PROBLEM #1 (URGENT): offload peak VRAM is ~9.2–9.3 GB, ~1.5–1.8 GB OVER the `--max-vram 7.5` budget
═══════════════════════════════════════════════════════════════════════════

The prod avatar runs `--offload-to-cpu --max-vram 7.5` specifically to leave VRAM headroom for the
coexisting GPU peers (llama/flux2/tts). The whole point is staying ~7.5 GB. **lap-34 (currently
deployed) peaks 9287 MiB** in a standard render — blowing the budget by ~1.8 GB. This silently eats
the coexistence headroom and can OOM/contend with peers.

### Measured board-VRAM peaks (480×832, 25f, 8 steps, offload, --max-vram 7.5, --mmap):
| binary | peak MiB | note |
|--|--|--|
| OLD offload (pre-my-work, 3c09e5d-era) | **7339** | the budget was respected |
| lap-33 (prefetch thread + pinned ring) | ~7495 | +156 MiB, fine |
| **lap-34 (prefetch thread + buffer POOL) — DEPLOYED** | **9287** | **+1.8 GB — the regression** |
(resident/no-offload for reference = 11641 MiB.)

### Prime suspect: the lap-34 weight-buffer POOL (Task 2)
lap-34 added a GPU buffer pool (`pool_alloc_ctx_tensors` / `pool_return_partial_buffer` /
`free_prefetch_buffer_pool` in `src/ggml_extend.hpp`, class `GGMLRunner`) to reuse the per-segment
weight buffers instead of cudaMalloc/cudaFree each segment. It's **capped at 2** buffers and converges
on the LARGEST segment sizes → holds ~2 big segment weight buffers resident (~2× ~1 GB). That is almost
certainly the +1.8 GB. (lap-33, which alloc/freed per segment, was only +156 MiB.)
There is also an inherent cost: the prefetch overlap holds segment N+1's weight buffer while computing
segment N (one extra buffer) — but lap-33 shows that alone is small (+156 MiB). The POOL retaining
buffers is the big add.
NOTE: `--max-vram 7.5` only governs the graph-cut **compute/activation** buffer; the prefetch **weight**
buffers are allocated ON TOP of it, so they are not counted in the budget.

### FIRST TEST (decisive, ~1 render): does disabling the pool restore ~7.5 GB?
```
LONGCAT_NO_PREFETCH_POOL=1   # env, already wired into lap-34
```
Render with it set and poll board VRAM (harness below). EXPECT ~7.5 GB. If confirmed:
the pool's measured benefit was only **~0.7 s/step** (Task-2 in the lap-34 commit) — a terrible trade
for +1.8 GB of coexistence headroom. Options:
  - default the pool OFF (flip the gate), keep the prefetch THREAD (that's the real ~5%/2× win, ~budget-neutral), OR
  - redesign the pool to bound its VRAM (e.g. cap to 1 buffer, or size-bound it, or only pool the small segments), OR
  - lower `--max-vram` to compensate (hacky).
Also independently confirm the prefetch THREAD alone (pool off) stays ≈7.5 GB and keeps the speed win.

### IMMEDIATE PROD SAFETY (decide with owner)
Prod is deployed at lap-34 = 9.2 GB peak NOW. Until fixed, either:
  (a) add `LONGCAT_NO_PREFETCH_POOL=1` to `docker/longcat-avatar/Dockerfile` ENV + rebuild/redeploy, or
  (b) roll back: set compose `LONGCAT_REF` + Dockerfile ARG back to `3c09e5d` (last known-good 7.3 GB,
      no prefetch thread) and **rebuild** — the old `3c09e5d` image is GONE (overwritten by
      `f7ac11f230b2`/lap-34; not retained untagged), so an instant retag is NOT possible.

OWNER NOTE (2026-05-31): prod was deliberately left AT lap-34 (9.2 GB) for now — owner is handing the
fix to a fresh agent and did not want more changes from the exhausted session. The pool is the obvious
first thing to kill; owner's stated priority is VRAM ≪ speed (offload exists to spend speed for VRAM;
7.5 GB was already higher than desired — so do NOT trade VRAM for speed, and consider going BELOW 7.5).

═══════════════════════════════════════════════════════════════════════════
## PROBLEM #2: prod runs CFG=7 (2 forward passes/step = 2× compute); make it request-configurable, default 1
═══════════════════════════════════════════════════════════════════════════

All our perf numbers ("~118 s sampling / ~15 s/step") were measured with the HANDOFF bench harness which
forces `--cfg-scale 1` (1 forward pass). **Real prod `/generate` runs at cfg=7** → cond+uncond = 2 passes
→ exactly 2× compute. Measured (lap-34, 480×832/25f/8step):
| | cfg 1 (1 pass) | cfg 7 (prod default, 2 passes) |
|--|--|--|
| sampling | 118.1 s | 236.5 s |
| DiT s/step | 14.8 | 29.6 |
| full render | 156.2 s | 290.4 s |
| peak VRAM | 9287 MiB | 9287 MiB (CFG adds passes, not memory) |

### Why cfg=7
`sd_sample_params_init` (`src/stable-diffusion.cpp:3030`) defaults `guidance.txt_cfg = 7.0`. The avatar
`/generate` route (`examples/server/routes_longcat.cpp`) **does not map any cfg field from the request
body**, so every prod render uses 7.0. CFG runs the uncond pass only when cfg>1
(`src/stable-diffusion.cpp:2394` `if (!uncond.empty())`).

### The opportunity
The model is `longcat-avatar-1.5-dit-DMD` — DMD = distribution-matching **distilled**, which is trained
to run WITHOUT classifier-free guidance (cfg≈1). Running it at cfg=7 is likely both 2× slower AND
over-guided/wrong. **If cfg=1 looks as good (very likely for DMD), prod gets 2× faster for free.**

### Action
1. Expose cfg in the route: `routes_longcat.cpp` → read `body.value("cfg_scale", 1.0)` (also accept
   `cfg`/`guidance` aliases) and forward to the worker as `--cfg-scale`. **Default 1.0.**
2. Quality A/B cfg=1 vs cfg=7 (a side-by-side was rendered this session — see below). Use
   `tools/clip_compare.py <a> <b>` for a metric, but mainly eyeball the talking-avatar quality. Owner
   decides the default; DMD strongly suggests 1.
3. Watch out: koblem may send its own cfg; check the koblem→avatar wire shape (`AVATAR_SERVER_URL`,
   docker-compose.yml ~line 764) so the default actually takes effect end-to-end.

═══════════════════════════════════════════════════════════════════════════
## CONTEXT: what shipped this session (lap-33 → lap-34)
═══════════════════════════════════════════════════════════════════════════

Root cause found via nsys: under `--offload-to-cpu`, each DiT segment's weights were streamed H2D inline
on the main thread right before its compute → synchronous (pageable mmap source) with the GPU idle =
**0% H2D/compute overlap, ~1.6 s/step serial = the +15% offload tax** vs resident. (nsys harness:
`longcat-nsys:builder` image has nsys; analysis scripts at `/tmp/longcat_vram/an.py`,`verify.py`.)

FIX = **background prefetch thread** (`prefetch_worker_loop`/`dispatch_prefetch_job`/`wait_prefetch_job`
in `src/ggml_extend.hpp`): the worker stages segment N+1's weights while the main thread runs segment N's
compute. A pageable `cudaMemcpyAsync` blocks its CALLING thread, not the GPU compute stream — so on the
worker it overlaps compute. **Gotcha that CRASHED once:** the worker MUST
`ggml_backend_synchronize(copy_backend_)` before each `set_async` (drains the driver's finite pinned
bounce buffer); omitting it corrupts the copy stream → crash.
- lap-33 (commit e65ec9c): thread + a pinned staging ring. The ring turned out USELESS (ring=1 MB =
  118.85 s = same as 128 MB), so —
- lap-34 (commit **38e3f30**, fork master, PUSHED): removed the pinned ring (direct pageable copy on the
  worker, zero pinned RAM — RssFile 14.9 GB weights stay reclaimable, RssAnon ~0.7 GB unchanged) + added
  the weight-buffer POOL (← Problem #1).

Speed result at cfg=1: old offload 15.4 s/step → lap-34 14.8 s/step (−5%); the thread overlaps the H2D
(nsys H2D/compute overlap 0%→69%). Bit-exact (clip_compare PSNR 99.00 mean+min). At cfg=7 (prod) the
absolute saving is ~2× since it applies per pass. **BUT this came at +1.8 GB peak (Problem #1).**

### Commits / deployed state
- Fork `dbrain/hbd-longcat-avatar.cpp` @ **38e3f30** (master, pushed). Only `src/ggml_extend.hpp` changed.
- kobbler @ **805887bb** (main, pushed): `docker/longcat-avatar/Dockerfile` adds
  `ENV LONGCAT_OFFLOAD_PREFETCH_THREAD=1`; `docker-compose.yml` `LONGCAT_REF` default `3c09e5d→38e3f30`
  (the COMPOSE default is the operative ref for `compose build`, not the Dockerfile ARG — bump both).
- **PROD DEPLOYED**: `docker compose build longcat-avatar` (clone @38e3f30) + `up -d`. Container
  `kobbler-longcat-avatar-1` healthy. Env-gated: prefetch thread default OFF in code, ON via Dockerfile ENV.

### Env knobs (all on the lap-34 binary)
- `LONGCAT_OFFLOAD_PREFETCH_THREAD=1` — enable the prefetch thread (the win). Default OFF in code.
- `LONGCAT_NO_PREFETCH_POOL=1` — disable the weight-buffer pool (← test this for Problem #1).

═══════════════════════════════════════════════════════════════════════════
## HARNESS / HOW TO REPRODUCE
═══════════════════════════════════════════════════════════════════════════
- GPU is free / dedicated (single RTX 3060, 12 GB). cpp forks build on-server fine (the no-build rule is
  Rust-only). Never run two GPU jobs at once.
- Build: `cd /home/dbrain/dev/kobbler/docker/longcat-avatar-dev && ./iter.sh build` (~90 s; rebuilds
  `/home/dbrain/dev/longcat-avatar.cpp/build/bin/sd-cli` via the docker builder; host has no CUDA).
- Standard render (bench cfg=1; drop `--cfg-scale 1` to get prod cfg=7):
  `docker run --rm --gpus all -v /home/dbrain/dev/longcat-avatar.cpp:/src -v /tmp/out:/out -w /src -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 longcat-avatar-dev:builder /src/build/bin/sd-cli -M vid_gen -m models/longcat-avatar-1.5-dit-dmd-q4_k.gguf --t5xxl models/longcat-umt5-xxl-q8_0.gguf --vae models/longcat-wan-vae-f16.gguf --audio-vae models/longcat-whisper-v3-encoder-f16.gguf --init-img models/_testinputs/girl_480x832.png --audio models/_testinputs/speech_16k.wav -p "a person talking" --video-frames 25 -W 480 -H 832 --steps 8 --diffusion-fa --seed 42 --clip-on-cpu --offload-to-cpu --vae-tiling --mmap --max-vram 7.5 --cfg-scale 1 -o /out/x.webm`
  (output MUST go to a MOUNTED path, e.g. `-v /tmp/out:/out -o /out/...` — a container-local path is lost on `--rm`.)
- Peak board VRAM: poll `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` every ~0.3 s during the render; take max.
- Per-step time: the `N/8 - Xs/it` log lines (cumulative mean; 8/8 ≈ steady). Sampling/full: `sampling completed, taking Xs` / `generate_video completed in Ys`.
- Bit-exact gate (any change): `python3 tools/clip_compare.py <base.webm> <new.webm>` → PSNR 99.00 mean+min (run on HOST; builder image lacks python/numpy).
- nsys: `longcat-nsys:builder` image, `nsys profile -t cuda -s none ...` (add `--cap-add SYS_ADMIN`), `--steps 3` to keep traces small; export sqlite + `/tmp/longcat_vram/{an,verify}.py`.
- Memory/notes for the project live in `/home/dbrain/.claude/projects/-home-dbrain-dev-kobbler/memory/project_longcat_vram_reorder_hunt.md` and `MEMORY.md`.

### Side-by-side comparison web page (renders N configs → videos + VRAM/timing side by side)
Reusable script: **`kobbler/docker/longcat-avatar-dev/compare.sh`** (self-contained: renders each config
in the prod offload setup, polls peak board VRAM, parses timing from sd-cli logs, generates an HTML page
with the videos + a stats table + a VRAM-over-time sparkline with the 7.5 GB budget line, and serves it).
```
cd ~/dev/kobbler/docker/longcat-avatar-dev
./iter.sh build                                          # build sd-cli first
./compare.sh 8042 "cfg1:--cfg-scale 1" "cfg7:--cfg-scale 7"
# → open http://<this-host>:8042/   (this host = 10.0.0.208)
```
Each arg is `LABEL:EXTRA_CLI_ARGS` (the extra args are appended to the standard render). Output webm
goes to a MOUNTED dir (container-local paths vanish on `--rm` — that bit me). VRAM is board-level, so
don't run two GPU jobs at once. Stop the server with `pkill -f 'http.server 8042'`. To vary an ENV per
config (e.g. `LONGCAT_NO_PREFETCH_POOL`) instead of a CLI flag, edit the `-e` line / `DOCKER_ENV` in the
script or run two invocations into different `WEBROOT`s. (Was prototyped in `/tmp/longcat_cmp/` this
session — ephemeral; the durable copy is `compare.sh`.)

### Key code locations (`src/ggml_extend.hpp`, class `GGMLRunner`)
`ensure_prefetch_thread` (pool alloc + thread launch) · `prefetch_worker_loop` · `stage_and_dma` (the
per-tensor synchronize + pageable set_async) · `kick_off_prefetch` · `commit_prefetched_state` ·
`compute_with_graph_cuts` (the per-segment loop) · pool: `pool_alloc_ctx_tensors` /
`pool_return_partial_buffer` / `free_prefetch_buffer_pool`. CFG: `routes_longcat.cpp` (route),
`stable-diffusion.cpp:3030` (cfg default 7.0), `:2394` (uncond pass gate).
