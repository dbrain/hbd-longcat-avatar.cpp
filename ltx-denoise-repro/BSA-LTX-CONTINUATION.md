# BSA on the LTX-2.3 continuation refine — map + enable path

Repo: `/home/dbrain/dev/longcat-avatar-ltxdenoise` (branch `ltx-denoise-workflow`).
Scope: can the shipped LongCat-Avatar Block-Sparse-Attention (BSA) machinery be engaged
on the LTX DiT self-attention during the continuation (base + hires refine) to cut the
O(N²) attention cost + VRAM?

**Bottom line: DID NOT WIRE (STOP + report).** The mask *apply* path is LTX-usable, but
BSA does **not** cut the refine VRAM — it *adds* ~127 MiB (the N² F16 mask) — and the only
real win (attention time) rides a **process-global CUDA bitmap** that would corrupt LTX's
*masked* text cross-attn unless carefully scoped, is sized for one specific FA template, and
needs different anchor semantics (first-3-frames, not LongCat's single cond frame). That is
squarely the ">~60 lines / risky" case, so this is a plan, not a commit.

---

## 1. Is the BSA mask-build + apply LTX-usable, or LongCat-only?

**APPLY path = fully generic / LTX-usable.**
The self-attention mask flows through the *shared* attention wrapper, not a LongCat-only one:

- `src/core/ggml_extend.hpp:1543` `ggml_ext_attention_ext(...)` is model-agnostic.
- `src/core/ggml_extend.hpp:1714-1749` handles an **F16 mask (BSA-style)**: it detects the
  F16 type, skips the `[L_q,L_k]→transpose` and the F32→F16 cast, and feeds it straight to
  `ggml_flash_attn_ext` at `:1767`. This branch does not care which model built the mask.
- LTX self-attn already routes here: `CrossAttention::forward` takes a `mask` param and
  passes it into `ggml_ext_attention_ext` at `src/model/diffusion/ltxv.hpp:820-830`.
- LTX video self-attn (`attn1`) is called with a `self_attention_mask` param at
  `src/model/diffusion/ltxv.hpp:1468` (AV block) and `:982` (plain block). Today that
  argument is hard-wired **`nullptr`** at the one live call site
  `src/model/diffusion/ltxv.hpp:2041` (`// self_attention_mask (unused on AV path)`).

So attaching a mask is a one-line change at `ltxv.hpp:2041` + a builder.

**BUILD path = LongCat-runner-only.**
The mask builder `ensure_bsa_mask(...)` is a **member of the LongCat avatar runner**
(`src/longcat_avatar.hpp:1369-1495`), populated from LongCat-only per-request fields
(`:1564-1569`, `:1959-1964`) and the `LONGCAT_BSA*` env overrides (`:1732-1737`). The
`sd_vid_gen_params_t.bsa_*` fields (`include/stable-diffusion.h:461-466`,
`src/stable-diffusion.cpp:3975-3980`) are threaded onto the model **only inside a
`sd_version_is_longcat_avatar` / `LongCatAvatarModel` guard**
(`src/stable-diffusion.cpp:6324-6333`). `ltxv.hpp` contains **zero** `bsa` references — the
LTX generate path never sees these params. There is no LTX mask builder.

**Where the LongCat mask is applied in the attention:** `src/longcat_avatar.hpp:333-339`
(`runner_ctx.bsa_mask` → `ctx->bsa_mask` arg of `ggml_ext_attention_ext`), reachable only
from the LongCat `cond_kv_cache` consume path (`:1711-1784`). The LTX DiT never enters that
code. The `GGMLRunnerContext::bsa_mask` field itself (`ggml_extend.hpp:2138-2142`) is generic
but is currently only *set* by the LongCat runner.

**The FA `ggml_flash_attn_ext` call at `ggml_extend.hpp:1767` accepts any F16 mask** — so the
mask reaches the kernel regardless of model. The *time win*, however, is a separate CUDA
mechanism (§ below), and that one is gated in ways that matter for LTX.

---

## 2. Knob semantics (exact)

Builder: `ensure_bsa_mask(n_noise, n_token, n_per_frame, h_len, w_len, cube_h, cube_w,
radius, self_frame_anchor, bookend_anchor)` — `src/longcat_avatar.hpp:1369-1495`.
Token layout is **t-outer / h-mid / w-inner** (`:1408`), mask tensor is F16
`[n_token(=L_k), n_noise(=L_q)]` (`:1386`). Allow/deny per (query qi, key ki), `:1417-1438`:

- **cube [cube_h, cube_w]** — spatial block size in *latent tokens*. Query token's spatial
  cube is `(q_h/cube_h, q_w/cube_w)`; same for key (`:1423-1431`).
- **radius R** — allowed iff `|q_cube_h−k_cube_h| ≤ R` **and** `|q_cube_w−k_cube_w| ≤ R`
  (`:1432-1433`). NB: **radius is a spatial-cube window, applied to ALL time frames equally
  — it is not a temporal window.** Cross-frame mixing is unrestricted except via the anchors.
- **cond/ref anchor (always on)** — `if (k_t==0) allow=true` (`:1434`). Pins **frame 0 only**.
- **self_frame_anchor** — `if (k_t==q_t) allow=true`: query at time T sees the *whole* of its
  own frame (`:1435`). Fixes edge-cube asymmetry / slow-rotation. **Default ON** in the preset.
- **bookend_anchor** — `if (k_t==T_last) allow=true`: every query also sees the **LAST**
  frame (`:1436`). Default OFF.

**Preset "r=1+self_frame" (the safe/tested config)** — baked into
`sd_vid_gen_params_init` (`stable-diffusion.cpp:3975-3980`) and the runner defaults
(`longcat_avatar.hpp:1564-1569`, `1959-1964`):
`radius=1, self_frame=1, bookend=0, cube_h=4, cube_w=6`. Enable via
`sd_vid_gen_params.bsa_enabled=1` (LongCat only) **or** env `LONGCAT_BSA=1` with overrides
`LONGCAT_BSA_RADIUS / _SELF_FRAME / _BOOKEND / _CUBE_H / _CUBE_W`
(`longcat_avatar.hpp:1732-1737`).

**Divisibility gate (the LOG_WARN).** `longcat_avatar.hpp:1741,1780`: BSA is silently
disabled unless `h_len % cube_h == 0` **and** `w_len % cube_w == 0`.
For the LTX **30×17** continuation latent (`w_len=30, h_len=17`):
- `h_len=17` is **prime** ⇒ `cube_h ∈ {1, 17}` only. `cube_h=17` = the whole height in one
  cube (no H sparsity); `cube_h=1` = per-row cubes. **The preset `cube_h=4` would trip the
  warn and disable BSA.**
- `w_len=30` ⇒ `cube_w ∈ {1,2,3,5,6,10,15,30}` (preset `cube_w=6` is fine).

**Bitmap (the actual time-saver).** `ensure_bsa_mask` also builds an all-deny **bitmap**
`[n_kwords, n_qtiles]` sized to a *hard-coded* FA template (`kNcols1=64, kNbatchFa=64,
DKQ=DV=128`, `:1392-1398`) and hands it to the CUDA FA kernel via
`ggml_cuda_set_longcat_fa_bsa_bitmap` (`:1765`). The kernel skips fully-denied K-tiles at
`ggml/src/ggml-cuda/fattn-mma-f16.cuh:1422-1430`. This bitmap skip — **not** the mask alone —
is what produced the measured **−1.98 s wall** on LongCat.

---

## 3. What it would take to engage BSA for the LTX continuation

### 3a. Is it reachable from the LTX generate path today? **No.**
Neither `sd_vid_gen_params.bsa_*` nor `LONGCAT_BSA*` reach `ltxv.hpp` (see §1). There is no
env/JSON that turns it on for LTX. New wiring is required.

### 3b. The critical finding: BSA does **not** cut the refine VRAM — it *adds* to it.
The refine peak (12977 MiB) is dominated by **resident DiT (5471)** + **compute_buf (3369,
linear in frames)** + overhead. `compute_buf` is the flash-attn *working set* (Q/K/V/out in
F16), which is **already O(N)** — flash never materializes the N² scores. A BSA mask is a
**new O(N²) F16 tensor**: for the 30×17×16 latent, `n_token = 510×16 = 8160`, so
`8160² × 2 B ≈ 127 MiB` **added** to compute_buf, and it re-enables the legacy `kv_pad`
mask-synthesis path unless guarded (`ggml_extend.hpp:804,1725-1728,1815-1817` — today LTX
self-attn passes `mask==nullptr` ⇒ `skip_kv_pad=true`, avoiding exactly this N² term). **So
enabling BSA raises the refine peak, it does not lower it.** BSA is a *time* lever, not a VRAM
lever. For VRAM the existing levers (spatial VAE tiling, offload eviction, `LTX_ATTN_KV_SCALE`)
are the right tools.

### 3c. The time win rides a fragile, process-global CUDA bitmap.
`ggml_cuda_set_longcat_fa_bsa_bitmap` sets **`__device__` globals** read by *every*
`ggml_flash_attn_ext` launch in the graph (`fattn-mma-f16.cuh:1370-1418`). It engages when
`ncols2==1` **and** `mask_h != nullptr` **and** `ktiles_ok`, where
`ktiles_ok = (n_ktiles_dev==0 || n_ktiles_dev==n_ktiles_call)` (`:1381-1383`). LongCat's
setter leaves `n_ktiles_dev==0` (no scoping) and is safe *only because LongCat's text
cross-attn is mask-free*, so the bitmap never matches it. **LTX text cross-attn passes a real
`attention_mask`** (`ltxv.hpp:1482,1510`, threaded to
`ggml_ext_attention_ext` at `:820-826`) ⇒ `mask_h != nullptr` ⇒ the self-attn bitmap would be
**mis-applied to LTX cross-attn and corrupt it**, unless we also set `n_ktiles` scoping via
`ggml_cuda_set_longcat_fa_bsa_mask_free(enabled, n_ktiles)` (`ggml/src/ggml-cuda/fattn.cu:36`)
so the shorter cross-attn tile count won't match. Additionally the bitmap constexpr config
(`ncols1=64, nbatch_fa=64`) is the **ampere** template — on the 5060 Ti (sm120) the LTX
self-attn may dispatch a different MMA config, mis-sizing the bitmap (the `n_kw<=64` guard
falls through quietly, i.e. *no* skip, but a config mismatch on tile geometry could skip wrong
tiles). This is real risk requiring a build+GPU A/B, which is out of scope here.

### 3d. Anchor semantics don't match — the bookend can NOT pin the keyframe anchors as-is.
The continuation seg-2 keyframe anchors are the **first 3 latent frames** (frame_idx 0,
13 gen + 3 prepended). The LongCat mask pins only **`k_t==0`** (one frame, `:1434`) and
`bookend` pins the **LAST** frame (`:1436`). Neither covers `k_t ∈ {0,1,2}`. Pinning the
3-frame continuity lifeline needs a **new "head-anchor N frames" rule**
(`if (k_t < n_anchor) allow=true`), i.e. a new builder — the LongCat one is not reusable
verbatim.

### 3e. Minimal wiring **plan** (if the owner still wants the time lever, accepting +VRAM)
Env-gated (`LTX_BSA=1`), dense/byte-identical when unset. ~70–90 lines across 3 files:

1. **`src/model/diffusion/ltxv.hpp` — new runner member** on `LTXAVRunner`
   (`:2099`), `ensure_ltx_bsa_mask(n_token, n_per_frame, h_len, w_len, cube_h, cube_w,
   radius, n_anchor)` — a copy of `longcat_avatar.hpp:1369-1495` **without** the cond/noise
   split (`L_q==L_k==n_token`, full self-attn) and with the anchor rule changed to
   `if (k_t < n_anchor) allow=true`. Default `n_anchor=3` (the continuation keyframe frames),
   `radius=1`, `self_frame=1`, `cube_h∈{1,17}`(!), `cube_w=6`. Persist the F16 mask on the
   runtime backend as a member (register as persistent tensor like the LongCat path).
2. **Thread into the block loop:** replace the `nullptr` at `ltxv.hpp:2041` with the built
   mask, gated on `getenv("LTX_BSA")` and on continuation (`frames>1 && n_anchor>0`). Only the
   **video** self-attn (`attn1`, `:1468`) should receive it; audio self-attn (`:1499`) stays
   `nullptr`.
3. **`src/model/diffusion/ltxv.hpp:804` — `CrossAttention::forward`:** change
   `const bool skip_kv_pad = (mask == nullptr);` →
   `const bool skip_kv_pad = (mask == nullptr) || (mask->type == GGML_TYPE_F16);`
   so the F16 BSA mask keeps the no-kv-pad flash path (an F32 cross-attn mask is unchanged).
   **Without this the BSA mask re-introduces the O(N²) kv-pad tensor.**
4. **Bitmap (optional, the only source of the time win):** build the bitmap like
   `longcat_avatar.hpp:1388-1475`, and set **both** the bitmap
   (`ggml_cuda_set_longcat_fa_bsa_bitmap`) **and** `n_ktiles` scoping
   (`ggml_cuda_set_longcat_fa_bsa_mask_free` / a new scoped setter) so LTX's masked cross-attn
   is excluded (§3c). **Clear both to null every step where the mask isn't in flight** (mirror
   `longcat_avatar.hpp:1769,1776`). This is the risky part; validate with the
   `[BSA-DBG]`/`[BSA]` logs (`fattn-mma-f16.cuh:1385`, `longcat_avatar.hpp:1488`) and a
   PSNR A/B before trusting it. Without the bitmap, BSA gives **no** compute win and only costs
   VRAM+quality — pointless.

### 3f. If wired, the exact knobs for the 30×17×16 continuation latent
- `LTX_BSA=1`
- `cube_w=6` (divides 30) — env `LTX_BSA_CUBE_W=6`
- `cube_h=17` (only 1 or 17 divide 17; use 17 = full-height cube, no H sparsity) or
  `cube_h=1` (per-row) — env `LTX_BSA_CUBE_H=17`. **`cube_h=4` is invalid → BSA auto-disables.**
- `radius=1`, `self_frame=1`
- `n_anchor=3` (pins latent frames 0–2, the prepended keyframe anchors — the continuity
  lifeline). The LongCat `bookend` flag can NOT express this; needs the new head-anchor rule.

**Expected effect:** with `cube_w=6, cube_h=17, r=1, self_frame=1`, every query keeps its own
full frame + the 3 anchor frames + a ±1-cube (±6-token-wide) spatial window across all frames.
Live K-tile fraction ≈ (3 anchor frames + self frame + neighbours)/16 frames ≈ **35–45 %**,
i.e. attention FLOPs cut ~55–65 % on the self-attn (self-attn is the bulk of DiT compute at
1080p). **Wall-time** win only realized *if* the bitmap kernel engages (§3c); the LongCat
analogue was −1.98 s. **Peak VRAM goes UP ~127 MiB** (the mask), not down. Quality: a lossy
trade (sparse attention, not bit-exact) — judge on an eye-test, do not lock.

---

## Recommendation
BSA is the wrong tool for the stated **VRAM** goal (it adds ~127 MiB and re-arms an N² kv-pad
term unless guarded). It is a viable **attention-time** lever, but only via the process-global
bitmap kernel, which needs `n_ktiles` scoping to avoid corrupting LTX's masked cross-attn, is
FA-template/GPU-sized, and needs a new first-3-frames anchor rule. All of that is build+GPU
work beyond a safe drop-in. **Held at the plan above; nothing committed.** If the owner wants
the time lever, implement §3e steps 1–4 behind `LTX_BSA=1` and A/B it on the target GPU.
