# Track B — "Who did it better" pass: runtime + diffusion blocks + upscaler

Scope: `git diff 19bdfe2 ea4e566` (upstream base→tip) vs current fork HEAD (`785ba0d`),
for the runtime denoiser/guidance, three diffusion blocks (unet/wan/z_image), and the
upscaler stack. Read-only analysis; no source edited.

Ground rules applied: fork's unique value preserved (single-GPU graph-cut VRAM/offload,
NVFP4/FP8 GEMM, LTX/Wan/audio video engine, SA3, own schedulers/denoisers). Upstream's
multi-GPU `ModelManager`/`weight_manager`/`RunnerWeightManager` is deliberately NOT adopted.
Strong bias to KEEP-FORK on any uncertainty.

---

## ADOPT (2 — both isolated, self-contained, deps verified present in fork)

### A1. `BetaScheduler` — new sigma scheduler  **(highest-value adopt)**
- **File:** `src/runtime/denoiser.hpp`
- **Rationale:** A proven Beta-distribution sigma schedule (ComfyUI's "beta"). Fork lacks it
  (`grep BetaScheduler` = 0). Fully self-contained: uses only `<cmath>` (`lgamma/exp/log/lround`),
  the fork's existing `SigmaScheduler` base (denoiser.hpp:23), `TIMESTEPS` (=1000, denoiser.hpp:20),
  and `t_to_sigma_t`. NO dependency on ModelManager or any new helper. Additive — touches no
  existing code path.
- **Verify:** fork already auto-detects/uses every symbol Beta needs; base class + `TIMESTEPS` confirmed.
- **Exact struct** (upstream `19bdfe2..ea4e566`, denoiser.hpp): the `struct BetaScheduler : SigmaScheduler`
  block — static `log_beta`, `incbeta` (Lentz continued fraction), `beta_cdf`, `beta_ppf` (Newton),
  and `get_sigmas(n, _, _, t_to_sigma)` with `alpha=beta=0.6`. Reproduced verbatim from the diff
  (lines 20–149 of the captured diff).
- **Insertion anchor (fork):** immediately after the fork's `struct KarrasScheduler` closes
  (fork denoiser.hpp:282 opens it; `struct SimpleScheduler` at :305) — same slot upstream used.
- **Plumbing (4 additive edits, all append-only, no reordering):**
  1. `include/stable-diffusion.h` — append `BETA_SCHEDULER,` after `LINEAR_QUADRATIC_SCHEDULER,`
     (line 73), before `SCHEDULER_COUNT`.
  2. `src/stable-diffusion.cpp` `scheduler_to_str[]` (ends line 4032 `"linear_quadratic",`) —
     append `"beta",` (order MUST match enum). `str_to_scheduler`/`sd_scheduler_name` are
     table-driven, so no further edits there.
  3. `src/runtime/denoiser.hpp` `Denoiser::get_sigmas` switch — add
     `case BETA_SCHEDULER: scheduler = std::make_shared<BetaScheduler>(); break;`
     next to `case KARRAS_SCHEDULER:` (fork line 656).
  4. (Optional) advertise in any CLI/help scheduler list.
- **Risk:** minimal. New enum value appended last (no ABI reordering of existing values); existing
  scheduler paths untouched. Owner GPU-validates output quality; failure mode is "new option unused."

### A2. `LogitNormalScheduler` — new sigma scheduler  *(secondary; lower value)*
- **File:** `src/runtime/denoiser.hpp`
- **Rationale:** Logit-normal timestep schedule (ideogram4-derived), resolution-aware mean shift.
  Fork lacks it. Self-contained: `<cmath>` + fork-present helpers `parse_key_value_args`,
  `parse_strict_float`, `parse_strict_bool` (all confirmed used elsewhere in fork). Constructed as
  `LogitNormalScheduler(image_seq_len, extra_sample_args)`.
- **Caveat lowering priority:** primarily beneficial for logit-normal-trained checkpoints; niche vs.
  the owner's LTX/Wan video focus. Adopt only if image-gen scheduler breadth is wanted.
- **Plumbing:** identical shape to A1 (append `LOGIT_NORMAL_SCHEDULER` enum + `"logit_normal"` str +
  switch case `std::make_shared<LogitNormalScheduler>(image_seq_len, extra_sample_args)`).
- **Risk:** minimal / additive. If only one is taken, take A1.

---

## KEEP-FORK

### Guidance (`guidance.cpp`, `guidance.h`)
- **`guidance_schedule` (per-step CFG scale override).** Upstream adds `parse_guidance_schedule[_from_spec]`
  and a defaulted `std::optional<float> scale_override` param on the `BaseGuidance::forward` virtual
  (CFG/APG/SLG). The parse+scale-plumb is self-contained *within* guidance.{h,cpp}, but it is INERT
  without sampler-loop wiring (compute `schedule[step]` and pass it as the override) — that wiring lives
  in the fork-owned sampler (`stable-diffusion.cpp`), not in the reviewed files. It is a feature (not a
  bugfix) and mutates the guidance virtual ABI. Cross-file + not isolated ⇒ KEEP-FORK. (Left as a clearly
  documented future option; the guidance math change itself is trivial `guidance_scale_ → scale.value_or(...)`.)
- No APG/NAG algorithmic fix here — the APG momentum/norm-threshold body is only whitespace-reflowed.

### Denoiser (`denoiser.hpp`) — everything except A1/A2
- **`FluxScheduler` / `Flux2Scheduler` (new SigmaScheduler structs + `flux_time_shift` free fn).**
  Fork already implements flux time-shift at the *denoiser* level: `FluxFlowDenoiser` (fork :872),
  `Flux2FlowDenoiser` (fork :885), and its own `flux_time_shift` (fork :868). Upstream's scheduler-struct
  form is a redundant alternate architecture. KEEP-FORK (equivalent present; grafting risks the flux path).
- **`noise_level_to_sigma` added as a PURE VIRTUAL on the `Denoiser` base.** Breaking: forces an
  implementation on *every* fork denoiser (LTX, Wan, audio, flux2, MiniT2I, …). Architectural/invasive. KEEP-FORK.
- **`SefiFlowDenoiser`** (dual sem/tex schedule) and **`MiniT2IFlowDenoiser`.** Model-specific; belong to the
  model-port track (`sefi_image.json`, `minit2i.json` in this dir). MiniT2I is ALREADY ported to the fork
  (HEAD 785ba0d; fork denoiser.hpp:920). KEEP-FORK / out of this pass's scope.
- Header includes (`<cstring>/<limits>/<map>`) only needed by the above — pull in only what A1/A2 require (none extra for Beta).

### `unet.hpp`
- **AnimateDiff motion-module support** (`enable_animatediff` detect, `AnimateDiffModel` wiring in
  in/mid/out blocks, `apply_motion_*` lambdas). A whole new subsystem depending on a new file
  `model/diffusion/animatediff.hpp` + `num_video_frames` plumbing. Fork's video path is LTX/Wan, not
  SD1.5 AnimateDiff. Large, non-isolated; belongs to the model-port track (`animatediff.json` present). KEEP-FORK.
- **Constructor + `compute<float>(...,false,false)` + `RunnerWeightManager` changes** — ModelManager
  plumbing. KEEP-FORK.

### `z_image.hpp`
- **ROCm precision fixes** (`out_proj/qkv_proj->set_force_prec_f32(true)`; `w2` force-f32 also on ROCm).
  ROCm-only numerical stability. Fork targets CUDA sm86/sm120 — no benefit. KEEP-FORK.
- **`Rope::RefIndexMode` replacing `bool increase_ref_index`, `ref_image_params.pass_to_dit`,
  ctor/`compute` signature changes, `ModelManager` in `test()`** — ref-image refactor + ModelManager
  plumbing. KEEP-FORK.

### `wan.hpp`
- 100% ModelManager/`RunnerWeightManager` constructor plumbing + `compute<float>(...,false,false)` +
  `test()` rewrite. No isolated engine change. KEEP-FORK (matches "be conservative on WAN").

### Upscaler stack
- **`esrgan.hpp` — `ESRGANConfig::detect_from_weights` refactor.** The fork's `ESRGAN` ctor ALREADY
  auto-detects scale (1/2/4), `num_block`, channels inline (fork esrgan.hpp:191–270). Upstream merely
  reorganizes that same detection into a config struct; no new capability, and it's entangled with the
  ModelManager ctor change. KEEP-FORK.
- **`ltx_latent_upscaler.hpp`.** Fork has its own (diverged) LTX temporal upscaler with its own
  `LatentUpsamplerConfig`/`temporal_upsample`/`rational_resampler` handling. Upstream's spatial
  `detect_from_weights` auto-config would have to be grafted onto a different file. Fork's specialized
  video territory. KEEP-FORK.
- **`upscaler.cpp` / `upscaler.h`.** ModelManager `register_param_tensors`/`ResidencyMode`, backend-init
  signature change, and a public C-API break on `upscale()` (now returns `bool` + `sd_image_t** images_out`
  batch array) + `new_upscaler_ctx()` losing `offload_params_to_cpu`. Tied to ModelManager and breaks fork
  callers. KEEP-FORK.

---

## Summary

Two ADOPT items, both new self-contained sigma schedulers in `src/runtime/denoiser.hpp` whose every
dependency (SigmaScheduler base, TIMESTEPS, the strict-parse helpers) is already present in the fork and
whose integration is purely additive (append one enum value + one string + one switch case each; no
reordering, no existing path touched). The highest-value one is **A1 `BetaScheduler`** — a zero-dependency,
`<cmath>`-only, proven Beta-distribution schedule the fork lacks, safe to graft with minimal risk. Everything
else is KEEP-FORK: the substantive upstream work is either ModelManager/weight_manager plumbing (deliberately
rejected), a breaking `noise_level_to_sigma` pure virtual, ROCm-only precision tweaks, redundant flux
schedulers the fork already covers at the denoiser level, or new subsystems (AnimateDiff, Sefi/MiniT2I)
that belong to the model-port track rather than a "who did it better" adopt.
