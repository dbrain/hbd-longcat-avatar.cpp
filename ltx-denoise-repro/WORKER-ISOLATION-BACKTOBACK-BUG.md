# LTX-2.3 worker-isolation: back-to-back chain renders fail every other time

**Repo:** `longcat-avatar-ltxdenoise` @ `c7f019e` (branch `ltx-denoise-workflow`)
**Scope:** `examples/server/` — code analysis only, no build/GPU/deploy.

## Symptom (recap)

Engine runs worker-isolated (`LTX_VIDEO_ISOLATION` + `LONGCAT_AVATAR_WORKER_ISOLATION`):
a CUDA-free parent serves the async job API; a CUDA-owning child does the compute.
Submitting chain renders **back-to-back** through koblem, **every other** render fails with
`VIDGEN_CHAIN_RESP recv failed: peer closed cleanly` — the child dies ~114 s in, during the
hires/refine stage, with **no assert / CUDA error / OOM in the child log** (clean socket EOF =
child just gone). Regular alternation: render-after-a-success **fails**, render-after-a-failure
**succeeds**. The same failing render **succeeds when run alone after `POST /v1/admin/unload`**.

## Root cause

There is **one** `WorkerSession` for the whole process (`main.cpp:148`, held in `ctx.worker`,
wired into the async runtime at `main.cpp:223`). It is a **warm** worker: the child forks once,
loads the model, and keeps the DiT + VAE + whisper **resident across renders**
(`worker_session.cpp:410 sd_ctx_keep_diffusion_model_resident(true)`, and `:394`
`free_params_immediately` frees only umT5 — the DiT/VAE/whisper frees are gated off so they stay
warm). So the design is: **reuse the same child for every chain render.**

The async loop routes each VidGen chain job through `runtime.worker->render_video_chain(...)`
(`async_jobs.cpp:660-661`). That call hits the `ensure_loaded()` **fast path** which returns the
**existing warm child** unchanged when one is alive:

- `worker_session.cpp:68-71` — `if (pid_ > 0 && fd_ >= 0 && loaded_ && worker_gpu_ == gpu) return true;`

Crucially, **the success path never tears the child down.** `render_video_chain()` kills the child
**only** on an IPC error:

- `worker_session.cpp:301-306` — on `recv_frame` error / wrong frame type → `kill_worker_locked()`.
- On success (`worker_session.cpp:307-325`) it just returns; the child stays resident.
- The async loop (`async_jobs.cpp:656-693`) likewise never reaps the worker after a completed
  VidGen job.

So a **successful** chain render leaves a fully-resident CUDA child behind (DiT/VAE warm, GPU
primary context + offload/host buffers all live). The **next** chain render reuses that same warm
child (`ensure_loaded` fast path), and the child dies during hires — the retained resident set from
render N plus render N+1's hires/refine working set exceeds the budget. The clean EOF with **no CUDA
assert in the child log** is the tell: this is not a ggml GPU-OOM (that prints `GGML_ASSERT`); it is
a **silent SIGKILL** — the container/cgroup OOM-killer reaping the child as the hires-stage host
offload buffers stack on top of the already-resident set (cf. commit `c7f019e`: "MAXV=11 inflates
offload buffers to 15GB"), or an equivalent silent exit. Parent-side that surfaces as
`recv_frame → EofClean → VIDGEN_CHAIN_RESP recv failed: peer closed cleanly`.

This is a **temporal** collision inside one warm child across two renders — **not** two coexisting
CUDA children. With a single `WorkerSession` singleton and `ensure_loaded` reuse, there is never a
second live child from this path; the owner's "leftover resident state collides with the next
render" intuition is correct, but the colliding thing is the **retained resident state of the same
warm child**, not a second process.

## Why it alternates exactly (the toggle)

The toggle is: **success leaves the child warm; the death's error path kills it.**

- **R1:** `ensure_loaded` finds no child → spawns fresh (`worker_session.cpp:88-98`, log
  `worker: ready (pid=A)`). Fresh CUDA context → **succeeds**. Child A left **resident** (not reaped).
- **R2:** `ensure_loaded` fast-path **reuses warm child A** (no new fork). Hires stage on top of A's
  retained resident set → child A is **killed** → parent `recv_frame` EofClean →
  `worker_session.cpp:304 kill_worker_locked()` reaps A, `pid_ = -1`. Job **fails**.
- **R3:** `ensure_loaded` sees `pid_ <= 0` → spawns fresh child B (`worker: ready (pid=B)`). Fresh
  context → **succeeds**. B left resident.
- **R4:** reuse warm B → dies → reaped. …and so on.

So a new `worker: ready (pid=N)` line appears on **every other** render (R1, R3, R5 — the ones
following a failure), which matches the observed "new child per render" log. Success ⇒ resident
child ⇒ next render dies. Failure ⇒ `kill_worker_locked` ⇒ next render forks fresh ⇒ succeeds.

## Why `POST /v1/admin/unload` fixes it (the workaround)

`sdcpp_handle_unload` in the worker branch calls `rt.worker->shutdown()`
(`routes_sdcpp.cpp:746`), which is `kill_worker_locked()` — SIGKILL + `waitpid` + close fd +
`loaded_ = false` (`worker_session.cpp:49-61`). That is **exactly the same teardown the failure
path performs**, done deliberately: it reclaims true-0 VRAM (primary context + cuBLAS workspace +
cubin cache, not just weights). So an unload between renders forces the next render to
`ensure_loaded` → fork a **fresh** child, i.e. it manually converts every render into an "R1/R3"
fresh-fork render, which always succeeds. The normal completion path does **not** do this teardown,
which is the whole bug.

**Reliable workaround (already confirmed):** `POST /v1/admin/unload` between every chain render.

## Proposed fix (describe only — DO NOT apply; owner will review + rebuild)

Make each worker-isolated chain render **fork fresh**, i.e. reap the CUDA child at render
completion instead of leaving it warm — mirroring what the failure path and `/v1/admin/unload`
already do. Two equivalent insertion points:

- **Preferred (async loop):** in `async_jobs.cpp`, right after the VidGen chain result is captured
  (after line **669**, inside the `if (runtime.ltx_video_mode …)` block, once `r` is consumed into
  `ok`/`output_*`), unconditionally call `runtime.worker->shutdown();`. On failure the child is
  already dead (`shutdown()` is idempotent — `worker_session.cpp:49-61`); on success this tears down
  the resident child so the **next** job's `render_video_chain → ensure_loaded` re-forks cold. This
  scopes the change to the video-chain path and leaves FLUX img-gen isolation untouched.

- **Alternative (session-local):** in `worker_session.cpp:307-325`, after a successful
  `VIDGEN_CHAIN_RESP` unpack, call `kill_worker_locked()` before returning (guarded by a
  "recycle-after-chain" flag so it doesn't affect img-gen/avatar renders that legitimately want the
  warm worker).

**Trade-off to flag for the owner:** this reintroduces a **cold model reload per chain render** —
it discards the warm-weights win (`worker_session.cpp:410`) for the video path. That is acceptable
because (a) LTX chain renders are minutes long, so the reload is a small relative cost, and (b) the
warm second render is currently **fatally broken** anyway. The alternative — keeping the worker warm
and finding/freeing the specific retained resident buffer(s) that overflow during hires (e.g. hires
upscaler cache, VAE tiling scratch, or offload host buffers not released between
`generate_video_chain` calls) — is a deeper, untested investigation; the reap-per-render fix is the
minimal reliable structural fix that makes back-to-back renders work with no manual unload.

## Key file:line index

- `main.cpp:148,223` — single `WorkerSession` singleton, wired into the video-chain runtime.
- `main.cpp:201-245` — `LTX_VIDEO_ISOLATION`/`WAN_VIDEO_ISOLATION` branch (CUDA-free parent + async worker).
- `async_jobs.cpp:656-669` — VidGen chain routed through `worker->render_video_chain(...)`; **no reap after success**.
- `worker_session.cpp:68-71` — `ensure_loaded` fast-path reuses the warm child.
- `worker_session.cpp:301-306` — chain render kills the child **only on IPC error** (the "failure cleans up" step).
- `worker_session.cpp:307-325` — success path returns without teardown (child stays resident).
- `worker_session.cpp:394,410` — DiT/VAE/whisper kept resident across renders (warm-weights design).
- `worker_session.cpp:439` — `worker: ready (pid=N)` per fresh fork (appears every other render).
- `worker_session.cpp:49-61` — `kill_worker_locked` (SIGKILL + waitpid + close + loaded_=false).
- `routes_sdcpp.cpp:736-763` — `/v1/admin/unload` worker branch → `worker->shutdown()` (the workaround's teardown).
