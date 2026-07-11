# LTX-2.3 lip-dub (relip) failure — code analysis (no build / no GPU)

> Current continuation: [Lipdub / Relip speed and VRAM handoff](HANDOFF-LIPDUB-RELIP-SPEED-VRAM.md).
> It supersedes this document's historical "do not route" warning: the target
> branch Relip crash is fixed, while production speed/VRAM work remains.

## IMPLEMENTED (root cause #1 fix) — `examples/server/async_jobs.cpp`

Wired the `"model":"edit"` DiT variant swap into the video-chain path, mirroring the FLUX
image path. Added a block in `run_vid_chain_job()` right after `gen_params.apply_ltx_relip_env()`
(async_jobs.cpp:420-447), BEFORE the `generate_video_chain` `sd_ctx_mutex` lock block:

```cpp
std::string want_variant;  // empty = keep whatever DiT is loaded (base) — byte-identical
if (body.contains("model") && body["model"].is_string()) {
    want_variant = body["model"].get<std::string>();
    if (want_variant != "base" && want_variant != "edit") {
        error_message = "invalid model variant, must be \"base\" or \"edit\"";
        return false;
    }
    if (want_variant == "edit" && runtime.model_swap &&
        runtime.model_swap->edit_path.empty()) {
        error_message = "edit model not available (server started without --diffusion-model-edit)";
        return false;
    }
}
if (!ensure_variant_loaded(runtime, want_variant, error_message)) {
    return false;  // error_message set by ensure_variant_loaded
}
```

Why this location: `run_vid_chain_job` is invoked from BOTH the worker child's VIDGEN_CHAIN_REQ
handler (`worker_session.cpp:694`, the CUDA-owning child) and the in-process path
(`async_jobs.cpp:692`), so one edit covers both. `ensure_variant_loaded` takes
`*runtime.sd_ctx_mutex` internally, so it must precede (not nest inside) the render lock block.

Behaviour:
- **absent / `"base"` → byte-identical.** `ensure_variant_loaded("")` keeps the currently-loaded
  variant (base at fork) and hits its fast-path early-return (async_jobs.cpp:548,557) — no swap.
  `"base"` when base is resident hits the same fast path. Every non-relip t2v/i2v/chain/wan render
  is unaffected.
- **`"edit"` → loads `--diffusion-model-edit` (nvfp4-CLEAN-lipdub.gguf).** `variant_diff` fires →
  `sd_ctx_swap_diffusion_model(edit_path)` (async_jobs.cpp:561-575). The swap is generic (opens the
  new GGUF with the `model.diffusion_model.` prefix, `stable-diffusion.cpp:1872`), and the merged
  lipdub is a self-contained same-architecture GGUF, so NO extra dual-DiT flags are needed.

Subtlety noted (not a correctness issue): with `LTX_REAP_CHILD_AFTER_RENDER=1` (prod default) each
relip window re-forks a fresh child that loads BASE, then this swaps it to EDIT — one wasted base
load per window. Correct, minor perf cost; a future optimisation could fork the child directly on
the requested variant. Also unchanged: prod forces two-stage via `LTXAV_RELIP_TWOSTAGE=1` (env, not
the request field) — orthogonal to this swap.

Commit: see `git log` on `ltx-denoise-workflow` (message "ltx relip: honour {\"model\":\"edit\"}
DiT swap on the video-chain path"). Not built here (GPU-side rebuild/validation by the coordinator).

---


Repos: koblem (Rust orchestration) + longcat-avatar-ltxdenoise @ `ltx-denoise-workflow`.
Live repro that failed: `POST /api/v1/ltx-video/relip`, 4s 1280×704 source + 4s 16 kHz mono,
`{a2v_guidance:1.0, two_stage:false, seed:42, save_clip:true}`.

---

## ROOT CAUSE #1 (DEFINITE bug, HIGH confidence): the `"model":"edit"` DiT swap is NOT wired into the LTX video-chain path

koblem's relip request carries `"model":"edit"` to select the merged lipdub GGUF
(`LTX_DIFFUSION_EDIT=ltx2/nvfp4-CLEAN-lipdub.gguf`), and calls `POST /v1/admin/unload`
first so it "loads fresh":
- koblem: `api/src/routes/ltx_video.rs:729` (`"model":"edit"`), `:721` (`unload()` before the window loop).

**But the engine only honours `model_variant` on the FLUX *image* path — never on the LTX/Wan video-chain path:**
- The only callers of `ensure_variant_loaded()` (the only path to `sd_ctx_swap_diffusion_model`) are the **ImgGen** handlers:
  - `examples/server/worker_session.cpp:650` (IMG_GEN_REQ)
  - `examples/server/async_jobs.cpp:650` (in-process ImgGen)
- The LTX video path goes VIDGEN_CHAIN_REQ → `run_vid_chain_job()` (`examples/server/async_jobs.cpp:383`) and the worker child dispatch (`examples/server/worker_session.cpp:674`). **Neither reads `"model"` nor calls `ensure_variant_loaded`.**
- `SDGenerationParams::from_json_str` (`examples/common/common.cpp`) has **no** `value("model")` parse — the field is silently dropped.
- The child always forks with the **base** DiT: `worker_session.cpp:427-429` sets `model_swap.base_path = --diffusion-model`, `loaded_variant = "base"`.
- Git confirms it was never wired: `ensure_variant_loaded` was added in `f017154` **for FLUX.2 image dual-DiT only** (`git log -S ensure_variant_loaded -- async_jobs.cpp`); the VidGen branch has never called it.

**Consequence:** after koblem's `unload()`, the fresh child re-loads `nvfp4-CLEAN.gguf` (BASE),
NOT `nvfp4-CLEAN-lipdub.gguf` (EDIT). **The lipdub weights are never applied.** The whole koblem
"merged edit DiT via model-swap" design (koblem commit `a6f94e7`, and the compose comment at
`docker-compose.yml:928-931`) rests on a false assumption about the engine. This directly explains
"lip-dub was failing" — even a "successful" render is a plain base-model relip, not a lipdub.

Disambiguating log line: the engine log should show the child loading `nvfp4-CLEAN.gguf` and
**no** `img_gen: swapping DiT variant base -> edit` line (that log only exists in the img path,
async_jobs.cpp:573). If you see the base model name and no swap line, #1 is confirmed active.

**Fix:** wire the variant swap into the video path. In `run_vid_chain_job` (async_jobs.cpp:383),
parse `body.value("model", "")` and call `ensure_variant_loaded(runtime, want, err)` under
`*runtime.sd_ctx_mutex` **before** `generate_video_chain`, mirroring the img path. It must run in
the same process that owns `sd_ctx` — i.e. inside the worker child's VIDGEN_CHAIN_REQ handler
(`worker_session.cpp:674`, using `child_runtime`), exactly like IMG_GEN_REQ does at `:650`.
(Cheaper interim: start the ltx-video server with `--diffusion-model` pointed at the lipdub GGUF
for a dedicated relip instance, so "base" already == lipdub.)

---

## Item 4 (the VAE eviction I flagged): NOT the cause — it is gated OFF for this render

`LTXAV_VAE_LAZY` eviction lives at `src/stable-diffusion.cpp:8739` (pre-sample) and `:9200`
(pre-refine). **Both are gated `ltxav_vae_lazy && !relip_twostage`.**

Prod forces two-stage: `docker-compose.yml:879` `LTXAV_RELIP_TWOSTAGE=1`, and the engine reads
two-stage from that **env, not the request** (`stable-diffusion.cpp:8596-8629`, also `:6469`). So
koblem's `two_stage:false` is **ignored** and `relip_twostage=true` → **both `LTXAV_VAE_LAZY`
evictions are skipped.** Even on the (non-prod) single-stage path the eviction fires only *after*
the reference encode (`:6548`) and the LTX hires refine is a pure latent-space upscale
(`upscale_ltx_spatial_video_latent`, `:9023`) that never touches the VAE — so the reference encode
the relip needs is already done. The eviction is not what broke the single-stage relip encode.

---

## Ranked hard-failure candidates (for the actual "job FAILED") — capture the live log with breakdown to disambiguate

Since #1 renders base-model relip (may succeed-but-wrong, or may fault), the hard job failure is
most likely one of:

1. **[#1 above] Base model can't cleanly do the forced two-stage relip** — always active, the
   prime suspect. Disambiguate: base model name in log + absence of any swap line; error text
   surfaced by koblem `wait()` (koblem `ltx_video.rs:762-769`).

2. **New locked VAE decode recipe (landed TODAY 2026-07-07) regressing the two-stage relip decode.**
   compose `:847` `LTX_VAE_SPATIAL_TILES=2x2`, `:840` `LONGCAT_VAE_KEEP_RESIDENT=0`, `:854`
   `LTXAV_DIT_FREE_DURING_DECODE=1`, plus engine commits `d25e7e5` (offload eviction) and `c9cb8c9`
   ("reset stale streaming budget on residency release — fix 1080p chain seg-2 crash"). These were
   validated on t2v **chain**, not on the relip two-stage decode. Disambiguate: failure occurs at
   **decode** — look for `release_all_gpu_param_residency` / `offload params … to runtime backend`
   immediately followed by a CUDA OOM or the child dying:
   `VIDGEN_CHAIN_RESP recv failed: peer closed cleanly` (worker_session.cpp:302 / render_video_chain).

3. **env-forced two-stage vs request single-stage + the ÷64 requirement.** Two-stage hard-errors
   unless width & height are divisible by 64 (`stable-diffusion.cpp:8600`), but koblem's
   `target_dims`/`round32` only guarantees ÷32 (koblem `ltx_video.rs:round32`). 1280×704 *is* ÷64,
   so this does NOT explain the 1280×704 live test — but any other aspect (e.g. a portrait fit to
   ÷32-but-not-÷64) hard-fails with `LTXAV_RELIP_TWOSTAGE requires width/height divisible by 64`.
   Disambiguate: grep the log for that exact string.

Draining wedge (memory "service draining" 503) is **fixed**: `sdcpp_handle_unload` clears
`draining` on every branch (`routes_sdcpp.cpp:756/779/790`, commit `4b7205e`), and koblem's
`unload()` runs after acquiring the GPU slot (in_flight 0) so it takes the real unload branch.
Not the cause.

---

## Bottom line

The single provable, always-active root cause is **#1: the LTX video-chain path silently ignores
`"model":"edit"` and renders with the BASE DiT, so the lipdub weights are never loaded.** Fix by
wiring `ensure_variant_loaded` into the VIDGEN_CHAIN_REQ / `run_vid_chain_job` path (or point a
dedicated relip server's `--diffusion-model` at the lipdub GGUF). For the exact hard-crash on the
1280×704 live test, capture the engine log with breakdown on: the base-model name + the
decode-phase residency lines (candidate 2) will tell #1-vs-#2 apart immediately.

---

## Fresh repro evidence — 2026-07-10

This was rechecked while implementing the LTX continuation identity fix. The new continuation
transport is explicitly bypassed for `relip_twostage`, so these results are independent of that
work.

### Target worktree failure

Using `/home/dbrain/dev/longcat-avatar-ltxdenoise/build-sa3/bin/sd-cli`, 25 source frames at
1280×704, the lipdub IC-LoRA prompt, drive audio, `LTXAV_RELIP_TWOSTAGE=1`, and the established
low-memory relip settings (`MAXV=7.5`, `LTXAV_RELIP_ENCODE_TILE=0.25`, VAE 1×1):

- native SA3 segfaulted during the stage-1 DiT graph;
- `GGML_LTX_SA3=0` (cuDNN fallback) segfaulted at the same pre-sampling point;
- neither run reached stage 2 or video decode.

This rules out the new continuation-reference implementation as the direct cause. Artifacts:

```text
ltx-denoise-repro/_ablation_out/relip_hiresref_smoke/render.log
ltx-denoise-repro/_ablation_out/relip_hiresref_smoke_cudnn/render.log
```

### Working production baseline

The sibling production relip binary, `/home/dbrain/dev/longcat-avatar.cpp/build/bin/sd-cli`, ran
the same source/audio/LoRA/two-stage recipe with cuDNN successfully:

```text
generate_video completed in 92.66s
output: ltx-denoise-repro/_ablation_out/relip_baseline_sibling/relip.webm
```

This is the current safe production path. Do not route lipdub through the target worktree binary
until its two-stage relip crash is reproduced under a debugger and fixed. Start by diffing the
stage-1 relip graph/caching code against the sibling worktree; the target failure occurs after
`get_learned_condition` and during the first graph-cut execution.

---

## Resolution — 2026-07-11

The failure was the graph-cut compute allocator reuse introduced by commit `205eb40`. That change
kept the allocator live across graph-cut segments (`free_compute_buffer_immediately=false`). Relip
has changing stage-1 segment shapes; by the third group it could reuse incompatible allocator
state and crash before sampling. The same crash occurred with native SA3 disabled, which is why it
initially looked unrelated to the graph-cut lifetime change.

Restoring the previous per-segment lifetime at both graph-cut `execute_graph` call sites in
`src/core/ggml_extend.hpp` fixes the failure. With the current `build-sa3` binary, the exact
lipdub GGUF + IC-LoRA + two-stage recipe completed its full locked 8/3 default pass (25-frame
smoke) in 90.59 seconds, peak driver VRAM 13,486 MiB. The continuing three-segment singing
regression uses the same fix and locked `run_singing_clip.sh` environment.
