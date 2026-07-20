# BATCH 7 — remaining adopts + niche + examples + docs — apply-ready integration edits

Fork worktree: `/home/dbrain/dev/longcat-avatar-mergewt` (branch `upstream-merge-2026-07-19`, HEAD `0fd28e7`)
Upstream tip `ea4e566`, merge base `19bdfe2`. Roadmap "Batch 7" (PULL-IN-ROADMAP.md §Batch 7).
Every anchor below was `git show`/`git diff`/`Read`/`grep`-verified against the CURRENT fork tree.
**No source was edited; nothing built. Nothing here is GPU/checkpoint validated.**

The consistent theme of Batch 7: the *interesting* upstream hunks all live inside diffs that ALSO carry the
deliberate NON-PORT multi-GPU `ModelManager`/`RunnerWeightManager`/split-mode/disk/RPC/`ggml_backend_sched`
architecture. For every item this doc extracts ONLY the self-contained hunk and explicitly leaves the
architectural refactor (the `(backend, params_backend)`→`(backend, prefix, weight_manager)` ctor change, the
`compute<float>(get_graph, n_threads, false)`→`(…, false, false, false)` arity change, `get_param_tensors`
prefix-drop) untouched — the fork keeps its conventions.

---

## 0. TL;DR verdict (read first)

| # | Item | Verdict | Where |
|---|---|---|---|
| **1** | core: backend name-resolution robustness (`resolve_device_by_name`/`sd_backend_resolve_name`) | **ADOPT (narrow subset) — LOW VALUE / OPTIONAL.** The named symbols are 90% wrappers for the multi-GPU/disk/RPC/split arch (NON-PORT). The separable robustness core = registry-name resolution + empty-dev-name→reg-name fallback. Safe additive, but barely exercised on a single-CUDA box that already does exact+prefix name match. | §1 |
| **2** | masked flash-attn `ne[1]!=L_q` repeat fix (`ggml_extend.hpp`) | **FLAG — real correctness fix, NOT a blind adopt.** Fork's `ggml_ext_attention_ext` mask path diverged (F16 BSA special-case, conditional transpose, kv_pad synthesis). Fork lacks the fix; adopting must be ADAPTED to the fork branch + GPU-verified on Chroma/T5 **and** the LongCat BSA path. | §2 |
| **3** | `AlphaBlender::get_alpha` graph-side (`block.hpp` + `auto_encoder_kl.hpp`) | **ADOPT — clean small fix.** Fork's `get_alpha` is byte-identical to upstream pre-change; primitives present; math unchanged, moved graph-side (robust under offload/graph-cut). Affects live SVD paths. | §3 |
| **4** | niche PixelDiT 1.5 `pit_lq_inject` (`pid.hpp`) | **FLAG — niche, ~354-line gated additive port.** Whole PixelDiT-1.5 variant (gate_modules / replicate-pad / unpatchify-factor / per-token gate / rope-128). Gated on `pit_lq_inject` (default false → base PixelDiT-1 byte-identical). No checkpoint on-box. | §4 |
| **5** | niche Ideogram4 seamless/circular tiling (`rope.hpp` + `ideogram4.hpp`) | **ADOPT — clean, primitives already in fork.** Fork's rope already has `axis_wrap_dims` (in `rope()`) and the runner context already has `circular_x/y_enabled`. Only 3 small hunks needed; no new primitive. | §5 |
| **6** | examples: `--prompt-file`/`--negative-prompt-file`, `.ckpt` ext, A1111 `prepare_info_field` | **MIXED.** `.ckpt` ext + `--prompt-file`/`--negative-prompt-file` = **ADOPT (trivial CLI).** `prepare_info_field` = **ADOPT-optional/FLAG server** (self-contained fn, but call-site entangled with upstream's changed `generate_image` sig; fork keeps `sd_image_t*` return + koblem may own the A1111 surface). | §6 |
| **7** | docs/: take upstream wholesale | **ADOPT (21 files, trivial) + SKIP-already-present (12 files).** 21 upstream-touched docs the fork never touched → `git checkout ea4e566 -- <files>`. 12 both-touched docs are ALREADY byte-identical to upstream (fork copied them during pull-in) → no-op. | §7 |

**Bottom line:** Group the safe self-contained adopts — **§3 (get_alpha), §5 (Ideogram4 seamless), §6 `.ckpt`+`--prompt-file`, §7 docs (21 wholesale), §1 narrow robustness (optional)** — into ONE build+commit. Flag separately: **§2 (flash mask — GPU-verify), §4 (PixelDiT-1.5 niche), §6 `prepare_info_field` (server, adapt `generate_image`).** See §8.

---

## 1. `src/core/ggml_extend_backend.cpp` — backend name-resolution robustness (ADOPT narrow subset; OPTIONAL)

The upstream diff `git diff 19bdfe2 ea4e566 -- src/core/ggml_extend_backend.cpp` is **526 lines**, of which the
overwhelming majority is the multi-GPU/disk architecture the roadmap declares NON-PORT:
`split_device_list`/`primary_device_name`/`runtime_backends(module)` (multi-device lists via `&`), `split_mode`
/`split_buffer_type`/`SDSplitMode::ROW` (row-split), `is_disk_backend_token`/`params_backend_is_disk`
(disk residency), `add_rpc_devices` (distributed RPC backend), `sd_backend_graph_compute_with_eval_callback`
+ `sd_ggml_graph_view` (sched eval callback), `SDBackendManager::init(...)` signature change (drops the
`keep_*_on_cpu` bools for a `split_mode_spec`), and `SDBackendModule::DETECTOR`. **DO NOT port any of that** — it
is the exact architecture the fork replaces with graph-cut, and the fork's ADetailer port already deliberately
reuses the `UPSCALER` module slot instead of adding `DETECTOR` (verified: `ggml_extend_backend.h` has `UPSCALER`,
no `DETECTOR`).

The **separable robustness core** is three helpers + their use in the resolve/init/validate paths. The fork today
(verified `Read`, lines 104-302) does exact-name + prefix matching only, and returns `ggml_backend_dev_name(dev)`
raw (which can be empty for some backends).

### 1a. Empty-dev-name → registry-name fallback in `resolve_first_device_by_type`

Anchor (unique, fork `ggml_extend_backend.cpp:104-110`):

```cpp
static std::string resolve_first_device_by_type(enum ggml_backend_dev_type type) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (dev == nullptr) {
        return "";
    }
    return ggml_backend_dev_name(dev);
}
```

Replace with (mirrors upstream):

```cpp
static std::string resolve_first_device_by_type(enum ggml_backend_dev_type type) {
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(type);
    if (dev == nullptr) {
        return "";
    }
    const char* dev_name = ggml_backend_dev_name(dev);
    if (dev_name != nullptr && dev_name[0] != '\0') {
        return dev_name;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char* reg_name   = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
    return reg_name != nullptr ? reg_name : "";
}
```

### 1b. Registry-name resolution helper (resolve "cuda"/"vulkan"/"metal"→"mtl" by backend registry name)

Insert **after** `resolve_first_device_by_type` (the new helper; upstream adds it here). Anchor: the closing brace
of §1a's function, immediately before `static ggml_backend_buffer_t ggml_backend_tensor_buffer(` (fork line 112):

```cpp
static ggml_backend_dev_t resolve_first_device_by_registry_name(const std::string& name) {
    std::string lower = lower_copy(trim_copy(name));
    if (lower == "metal") {
        lower = "mtl";
    }
    if (lower.empty()) {
        return nullptr;
    }
    const size_t device_count = ggml_backend_dev_count();
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        if (reg == nullptr) {
            continue;
        }
        const char* reg_name = ggml_backend_reg_name(reg);
        if (reg_name != nullptr && lower_copy(reg_name) == lower) {
            return dev;
        }
    }
    return nullptr;
}
```

### 1c. Use the registry-name fallback in `sd_resolve_backend_name` (keep the fork's static name; do NOT rename)

> **Fork-convention decision:** upstream renames `sd_resolve_backend_name` → `sd_backend_resolve_name` and makes it
> non-static/exported, purely because the multi-GPU code in `stable-diffusion.cpp` calls it. The fork does not need
> the export. **Keep `sd_resolve_backend_name` static** and just add the fallback.

Anchor (unique, fork `ggml_extend_backend.cpp:265-283` — the tail of `sd_resolve_backend_name`, after the exact +
prefix loops, before the final `return "";`):

```cpp
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string dev_name   = ggml_backend_dev_name(dev);
        std::string dev_lower  = lower_copy(dev_name);
        if (dev_lower.rfind(lower, 0) == 0) {
            return dev_name;
        }
    }

    return "";
}
```

Replace with (insert one registry-name fallback before `return "";`):

```cpp
    for (size_t i = 0; i < device_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        std::string dev_name   = ggml_backend_dev_name(dev);
        std::string dev_lower  = lower_copy(dev_name);
        if (dev_lower.rfind(lower, 0) == 0) {
            return dev_name;
        }
    }

    if (ggml_backend_dev_t dev = resolve_first_device_by_registry_name(requested)) {
        const char* dev_name = ggml_backend_dev_name(dev);
        if (dev_name != nullptr && dev_name[0] != '\0') {
            return dev_name;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        const char* reg_name   = reg != nullptr ? ggml_backend_reg_name(reg) : nullptr;
        if (reg_name != nullptr) {
            return reg_name;
        }
    }

    return "";
}
```

> This lets a user write `--backend cuda` (registry name) rather than the exact device string. **Verified fork
> primitives:** `lower_copy`, `trim_copy`, `ggml_backend_dev_count/get`, `ggml_backend_dev_backend_reg`,
> `ggml_backend_reg_name` (all already used in this file). `backend_name_exists`/`init_named_backend`/
> `init_cached_backend` all call `sd_resolve_backend_name`, so they pick up the fallback for free — **no edits
> there** (do NOT adopt upstream's `init_named_backend` direct-`ggml_backend_dev_init` rewrite; it belongs to the
> multi-GPU path).

**Value assessment (be honest):** LOW. The fork already resolves exact + prefix device names, and the box uses
explicit device pins (`CVD` / per-request `gpu` field). Registry-name aliasing + the empty-name fallback are pure
robustness with essentially no on-box exercise. **Land it if grouping the safe batch is free; skip it without
regret.** `add_rpc_devices`/RPC is a genuine capability the fork docs advertise (fork copied `docs/rpc.md`, §7)
but has NO code for — that pairs with the multi-GPU/distributed decision, **not** Batch 7; flag to owner separately.

---

## 2. `src/core/ggml_extend.hpp` — masked flash-attn `ne[1]!=L_q` repeat fix (FLAG; ADAPT + GPU-verify)

Upstream's hunk (`git diff 19bdfe2 ea4e566 -- src/core/ggml_extend.hpp`, in `ggml_ext_attention_ext`) fixes a
real bug: `ggml_flash_attn_ext` expects the mask as `[n_kv, n_q, …]` and does **not** broadcast the query dim.
Callers that pass a per-key padding mask broadcast over queries (`[n_kv, 1, …]` — Chroma/T5) got a bare
`ggml_transpose` that produced `[1, n_kv, …]`, which the kernel misreads → **NaN/blank masked output**. Upstream
replaces the transpose with a query-dim materialization:

```cpp
if (mask_in->ne[1] != L_q) {
    mask_in = ggml_repeat(ctx, mask_in,
                          ggml_new_tensor_4d(ctx, mask_in->type, mask_in->ne[0], L_q, mask_in->ne[2], mask_in->ne[3]));
}
mask_in = ggml_cast(ctx, mask_in, GGML_TYPE_F16);
```

**The fork does NOT have this fix**, and its mask path has DIVERGED — this is why it is NOT a drop-in. Fork anchor
(unique, `ggml_extend.hpp:1745-1754`, the LongCat lap-28.4 conditional transpose):

```cpp
        if (mask_in != nullptr) {
            // LongCat lap-28.4: F16 callers (the BSA path) build the mask directly in
            // flash's expected [L_k, L_q] layout (ne[0]=L_k, ne[1]=L_q), contiguous.
            // The default wrapper convention is caller-provides-[L_q, L_k] which then
            // gets transposed → non-contiguous (fails flash's contiguity assert when
            // we skip the F32→F16 cast that previously materialized a contiguous F16
            // buffer). Skip the transpose for F16 masks (BSA-style).
            if (mask_in->type != GGML_TYPE_F16) {
                mask_in = ggml_transpose(ctx, mask_in);
            }
        } else {
```

Proposed **adapted** replace-with (keep the F16 BSA short-circuit exactly; on the non-F16 branch, replace the
transpose with the upstream query-materialization). The subsequent F16 cast at fork `ggml_extend.hpp:1778` is
left as-is (it already only casts non-F16 masks):

```cpp
        if (mask_in != nullptr) {
            // LongCat lap-28.4: F16 callers (the BSA path) build the mask directly in
            // flash's expected [L_k, L_q] layout (ne[0]=L_k, ne[1]=L_q), contiguous —
            // leave it untouched.
            if (mask_in->type != GGML_TYPE_F16) {
                // ggml_flash_attn_ext wants [n_kv, n_q, …] and does NOT broadcast the
                // query dim. Callers passing a per-key mask broadcast over queries
                // ([n_kv, 1, …] — Chroma/T5) must have the query dim materialized to
                // L_q, else the kernel misreads it → NaN/blank. (Replaces the old bare
                // transpose, which produced a [1, n_kv, …] mask for those callers.)
                if (mask_in->ne[1] != L_q) {
                    mask_in = ggml_repeat(ctx, mask_in,
                                          ggml_new_tensor_4d(ctx, mask_in->type,
                                                             mask_in->ne[0], L_q, mask_in->ne[2], mask_in->ne[3]));
                }
            }
        } else {
```

> **Why FLAG, not ADOPT:** (a) The fork's default mask convention is caller-provides-`[L_q, L_k]`; the old
> transpose flipped it to `[L_k, L_q]`. Upstream's convention is caller-provides-`[n_kv, n_q]` already. **The two
> forks disagree on the input mask layout**, so blindly dropping the repeat (which assumes upstream's layout) could
> mis-shape masks that today rely on the transpose. The correct adaptation depends on what layout each fork caller
> actually passes — must be traced per caller before landing. (b) The fork's kv_pad-synthesized masks are already
> `[L_k, L_q]` (`ne[1]=L_q`) so the `ne[1] != L_q` guard is a no-op for them — good. (c) No blind test: needs a
> **GPU A/B on a masked-flash model (Chroma or T5-conditioned) AND the LongCat BSA path** to confirm neither
> regresses. Do NOT adopt the two adjacent upstream hunks in the same region (`if
> (!ggml_backend_supports_op(backend, out)) return nullptr;` and the `ggml_view_3d`→`ggml_view_4d(…, N, …)`
> change) — those interact with the fork's diverged qtile/BSA/`supports_op` fallback block (`ggml_extend.hpp:1865-
> 1898`) and are out of scope for this fix. Also do NOT adopt the `ggml_ext_pad_ext`/`ggml_ext_conv_3d` `backend`-arg
> Metal fallbacks from the same file diff — CUDA box, and the qwen-layered doc already ruled them out.

---

## 3. `AlphaBlender::get_alpha` graph-side (ADOPT — clean, both files)

Upstream moves `get_alpha` from a **host tensor read** (`ggml_ext_backend_tensor_get_f32(params["mix_factor"])`
→ `sigmoid()` on a `float`) to a **graph-side** computation (`ggml_ext_cast_f32` + `ggml_sigmoid`), and rewrites
the blend as `x_temporal + (x_spatial - x_temporal) * alpha` (algebraically identical to
`alpha*x_spatial + (1-alpha)*x_temporal`). Benefit for the fork: no host readback of a param that may live on GPU
under graph-cut/offload. **Verified primitives present:** `ggml_ext_cast_f32` (`ggml_extend.hpp:1548`),
`ggml_sigmoid` (used in fork `lingbot_video.hpp:385`, `pid.hpp:481`, `ltxv.hpp:985`). Live consumers: SVD unet
(`block.hpp` AlphaBlender) + SVD video VAE decoder (`auto_encoder_kl.hpp`).

> **Port ONLY the `get_alpha` + forward hunks.** The same upstream file diffs also carry the NON-PORT
> `(backend, prefix, weight_manager)` ctor + `get_param_tensors(tensors)` prefix-drop +
> `compute<float>(…, false, false, false)` + `sd_version_uses_flux_vae`/sefi-slice refactors — **leave all of those.**

### 3a. `src/model/common/block.hpp`

Anchor (unique, fork `block.hpp:588-593`):

```cpp
    float get_alpha() {
        // image_only_indicator is always tensor([0.]) and since mix_factor.shape is [1,]
        // so learned_with_images is same as learned
        float alpha = ggml_ext_backend_tensor_get_f32(params["mix_factor"]);
        return sigmoid(alpha);
    }
```

Replace with:

```cpp
    ggml_tensor* get_alpha(GGMLRunnerContext* ctx) {
        // image_only_indicator is always tensor([0.]) and since mix_factor.shape is [1,]
        // so learned_with_images is same as learned
        auto mix_factor = ggml_ext_cast_f32(ctx->ggml_ctx, ctx->backend, params["mix_factor"]);
        return ggml_sigmoid(ctx->ggml_ctx, mix_factor);
    }
```

And the forward. Anchor (unique, fork `block.hpp:605-610`):

```cpp
        // image_only_indicator is always tensor([0.])
        float alpha = get_alpha();
        auto x      = ggml_add(ctx->ggml_ctx,
                               ggml_ext_scale(ctx->ggml_ctx, x_spatial, alpha),
                               ggml_ext_scale(ctx->ggml_ctx, x_temporal, 1.0f - alpha));
        return x;
```

Replace with:

```cpp
        // image_only_indicator is always tensor([0.])
        auto alpha = get_alpha(ctx);
        return ggml_add(ctx->ggml_ctx,
                        x_temporal,
                        ggml_mul(ctx->ggml_ctx,
                                 ggml_sub(ctx->ggml_ctx, x_spatial, x_temporal),
                                 alpha));
```

### 3b. `src/model/vae/auto_encoder_kl.hpp`

Anchor (unique, fork `auto_encoder_kl.hpp:216-219`):

```cpp
    float get_alpha() {
        float alpha = ggml_ext_backend_tensor_get_f32(params["mix_factor"]);
        return sigmoid(alpha);
    }
```

Replace with:

```cpp
    ggml_tensor* get_alpha(GGMLRunnerContext* ctx) {
        auto mix_factor = ggml_ext_cast_f32(ctx->ggml_ctx, ctx->backend, params["mix_factor"]);
        return ggml_sigmoid(ctx->ggml_ctx, mix_factor);
    }
```

And the forward. Anchor (unique, fork `auto_encoder_kl.hpp:253-256`):

```cpp
        float alpha = get_alpha();
        x           = ggml_add(ctx->ggml_ctx,
                               ggml_ext_scale(ctx->ggml_ctx, x, alpha),
                               ggml_ext_scale(ctx->ggml_ctx, x_mix, 1.0f - alpha));
```

Replace with:

```cpp
        auto alpha = get_alpha(ctx);
        x          = ggml_add(ctx->ggml_ctx,
                              x_mix,
                              ggml_mul(ctx->ggml_ctx,
                                       ggml_sub(ctx->ggml_ctx, x, x_mix),
                                       alpha));
```

> These two `forward`s already receive a `GGMLRunnerContext* ctx` (they call `ctx->ggml_ctx` throughout), so
> `get_alpha(ctx)` is in scope with no signature change. GPU-verify: an SVD img2vid render (the only path that
> constructs AlphaBlender) still produces identical frames.

---

## 4. `src/model/diffusion/pid.hpp` — PixelDiT 1.5 `pit_lq_inject` (FLAG — niche gated port)

Upstream's `git diff 19bdfe2 ea4e566 -- src/model/diffusion/pid.hpp` is **354 lines** and adds a whole
**PixelDiT-1.5** model variant on top of the fork's existing PixelDiT-1 (`pid.hpp` present, base config verified —
grep for `pit_lq_inject`/`lq_latent_unpatchify_factor`/`lq_gate_per_token` in the fork returns **nothing**). The
variant is detected by `lq_proj.pit_head.weight` (`config.pit_lq_inject`) and adds: `lq_latent_unpatchify_factor`,
`lq_replicate_padding`, `lq_gate_per_token`, per-`gate_modules` interval derivation, a 16-vs-128 latent-channel /
8-vs-16 down-factor / rope-128 branch, and matching forward-path changes (`lq_proj` gate injection, replicate
padding, unpatchify). It is **fully gated on `pit_lq_inject` (default false)** → base PixelDiT-1 stays
byte-identical.

Config-detection anchor (unique, fork `pid.hpp`, the `PixelDiTConfig` member block — grep `int64_t lq_latent_down_factor`):

```cpp
        int64_t lq_latent_down_factor  = 8;
        int64_t rope_ref_grid_h        = 64;
        int64_t rope_ref_grid_w        = 64;
```

Upstream inserts three new members before `rope_ref_grid_h` (`lq_latent_unpatchify_factor = 1`,
`lq_replicate_padding = false`, `lq_gate_per_token = false`, `pit_lq_inject = false`) and rewrites
`detect_from_weights` to key on `lq_prefix + "pit_head.weight"`, `gate_modules.*`, and the 16/32
`latent_proj.0.weight` channel split (see the full upstream hunk).

**Verdict FLAG-niche:** portable as a gated additive (same pattern as the qwen-layered §5/§7 adds), but it is a
substantial, self-contained model-variant port — config **plus** the forward-path gate/replicate/unpatchify
changes — with **no PixelDiT-1.5 checkpoint on-box** to validate the detection thresholds (16↔128 channels,
per-token gate `ne[1]==1`, rope-128 grid) or the axis semantics. **Do NOT fold into the safe batch.** Land as its
own milestone only if the owner wants PixelDiT-1.5 support; otherwise defer. Base PixelDiT-1 is unaffected either
way (all new behaviour is behind `pit_lq_inject`).

---

## 5. Ideogram4 seamless / circular tiling (ADOPT — primitives already in fork)

The roadmap worried this "needs rope `axis_wrap_dims`." **The fork's rope already has `axis_wrap_dims`** — its
`rope()` takes `const std::vector<int>& axis_wrap_dims = {}` (`rope.hpp:52`) and the fork's other circular-tiling
models (qwen_image/flux/lens/ernie) already thread it. The ONLY gap is that the fork's `embed_interleaved_mrope`
(Ideogram4's rope entry) doesn't yet accept/forward `axis_wrap_dims`, and `ideogram4.hpp` never wires the runner's
`circular_x/y_enabled`. Three small hunks, no new primitive.

> Port ONLY these three hunks. The `ideogram4.hpp` file diff also carries the NON-PORT
> `(backend, params_backend)`→`(backend, prefix, weight_manager)` ctor + `compute<float>(…, false, false, false)`
> arity change — **leave the fork's `(backend, params_backend)` ctor and 3-arg compute exactly as they are.**

### 5a. `src/model/common/rope.hpp` — thread `axis_wrap_dims` through `embed_interleaved_mrope`

Anchor (unique, fork `rope.hpp:252-256`, the signature):

```cpp
    __STATIC_INLINE__ std::vector<float> embed_interleaved_mrope(const std::vector<std::vector<float>>& ids,
                                                                 int bs,
                                                                 float theta,
                                                                 int head_dim,
                                                                 const std::vector<int>& mrope_section) {
```

Replace with (append the defaulted param):

```cpp
    __STATIC_INLINE__ std::vector<float> embed_interleaved_mrope(const std::vector<std::vector<float>>& ids,
                                                                 int bs,
                                                                 float theta,
                                                                 int head_dim,
                                                                 const std::vector<int>& mrope_section,
                                                                 const std::vector<std::vector<int>>& axis_wrap_dims = {}) {
```

And the per-axis `rope()` call. Anchor (unique, fork `rope.hpp:265-269`):

```cpp
        std::vector<std::vector<std::vector<float>>> axis_embs;
        axis_embs.reserve(3);
        for (int axis = 0; axis < 3; ++axis) {
            axis_embs.push_back(rope(trans_ids[axis], head_dim, theta));
        }
```

Replace with:

```cpp
        std::vector<std::vector<std::vector<float>>> axis_embs;
        axis_embs.reserve(3);
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<int> axis_wrap;
            if (axis < static_cast<int>(axis_wrap_dims.size())) {
                axis_wrap = axis_wrap_dims[axis];
            }
            axis_embs.push_back(rope(trans_ids[axis], head_dim, theta, axis_wrap));
        }
```

> The fork's `rope(ids, dim, theta, axis_wrap_dims={})` already accepts the 4th arg (verified `rope.hpp:52`, and
> the fork already calls `rope(trans_ids[i], axes_dim[i], axis_theta, axis_wrap_dims)` at `rope.hpp:210`) — the
> threading is exact.

### 5b. `src/model/diffusion/ideogram4.hpp` — `gen_ideogram4_pe` circular params + axis_wrap build

Anchor (unique, fork `ideogram4.hpp:148-172` — the whole helper). Add `circular_x/y` params and build the
`axis_wrap_dims` before the `embed_interleaved_mrope` call:

```cpp
    __STATIC_INLINE__ std::vector<float> gen_ideogram4_pe(int grid_h,
                                                          int grid_w,
                                                          int bs,
                                                          int context_len,
                                                          int head_dim,
                                                          int rope_theta,
                                                          const std::vector<int>& mrope_section) {
```

Replace the signature with (append two defaulted bools):

```cpp
    __STATIC_INLINE__ std::vector<float> gen_ideogram4_pe(int grid_h,
                                                          int grid_w,
                                                          int bs,
                                                          int context_len,
                                                          int head_dim,
                                                          int rope_theta,
                                                          const std::vector<int>& mrope_section,
                                                          bool circular_x = false,
                                                          bool circular_y = false) {
```

And the return. Anchor (unique, fork `ideogram4.hpp:172`):

```cpp
        return Rope::embed_interleaved_mrope(ids, bs, static_cast<float>(rope_theta), head_dim, mrope_section);
```

Replace with:

```cpp
        std::vector<std::vector<int>> axis_wrap_dims(3);
        if (circular_y || circular_x) {
            size_t total_len = static_cast<size_t>(bs) * (context_len + grid_h * grid_w);
            axis_wrap_dims[1].assign(total_len, 0);
            axis_wrap_dims[2].assign(total_len, 0);
            if (circular_y) {
                for (size_t idx = static_cast<size_t>(context_len); idx < total_len; ++idx) {
                    axis_wrap_dims[1][idx] = grid_h;
                }
            }
            if (circular_x) {
                for (size_t idx = static_cast<size_t>(context_len); idx < total_len; ++idx) {
                    axis_wrap_dims[2][idx] = grid_w;
                }
            }
        }

        return Rope::embed_interleaved_mrope(ids,
                                             bs,
                                             static_cast<float>(rope_theta),
                                             head_dim,
                                             mrope_section,
                                             axis_wrap_dims);
```

### 5c. `src/model/diffusion/ideogram4.hpp` — build_graph: pass `runner_ctx.circular_x/y_enabled`

Anchor (unique, fork `ideogram4.hpp:483-500` region — the `gen_ideogram4_pe` call currently sits **before**
`get_context()`):

```cpp
            pe_vec  = gen_ideogram4_pe(static_cast<int>(grid_h),
                                       static_cast<int>(grid_w),
                                       static_cast<int>(x->ne[3]),
                                       static_cast<int>(context_len),
                                       static_cast<int>(head_dim),
                                       static_cast<int>(config.rope_theta),
                                       config.mrope_section);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
```

Replace with (hoist `get_context()` above the call — matching upstream — and pass the circular flags):

```cpp
            auto runner_ctx = get_context();
            pe_vec          = gen_ideogram4_pe(static_cast<int>(grid_h),
                                               static_cast<int>(grid_w),
                                               static_cast<int>(x->ne[3]),
                                               static_cast<int>(context_len),
                                               static_cast<int>(head_dim),
                                               static_cast<int>(config.rope_theta),
                                               config.mrope_section,
                                               runner_ctx.circular_x_enabled,
                                               runner_ctx.circular_y_enabled);
            auto pe         = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, head_dim / 2, pos_len);
```

> **REQUIRED companion edit:** the fork currently constructs `auto runner_ctx = get_context();` *later* in the same
> method (fork `ideogram4.hpp:500`, just before the `active_model.forward(&runner_ctx, …)` call). After hoisting it
> above (as shown), **delete that now-duplicate later declaration** (change it to reuse the hoisted `runner_ctx`) —
> otherwise it redeclares. Verified: `get_context()`, `circular_x_enabled`/`circular_y_enabled` on
> `GGMLRunnerContext` are present and used across the fork (`dit.hpp:76`, `qwen_image.hpp:596`, `flux.hpp:1548`,
> `block.hpp:33`). GPU-verify: a normal (non-seamless) Ideogram4 render is unchanged (`circular_*` default false →
> empty `axis_wrap_dims` → identical PE), and a `--seamless`/circular render tiles without a seam.

---

## 6. `examples/` — `.ckpt` ext, `--prompt-file`/`--negative-prompt-file`, A1111 `prepare_info_field`

### 6a. `.ckpt` embedding/LoRA extension (ADOPT — trivial)

The fork's model-file extension allowlists omit `.ckpt`. Two live sites (verified fork `common.cpp`):

Anchor 1 (unique, fork `common.cpp:629`, embedding_map scan):

```cpp
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt"};
```

Anchor 2 (unique, fork `common.cpp:2127`, LoRA/model dir resolution):

```cpp
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt"};
```

Replace **both** with:

```cpp
    static const std::vector<std::string> valid_ext = {".gguf", ".safetensors", ".pt", ".ckpt"};
```

> `replace_all` is unsafe here — there is a **third** `valid_ext` at `common.cpp:2339` that is the *hires-upscaler*
> allowlist and already reads `{".gguf", ".safetensors", ".pt", ".pth"}` (different set). Leave 2339 alone (or add
> `.ckpt` to it too for parity — optional; upstream's `.ckpt` add targets embedding/LoRA/model, not upscalers).
> Edit 629 and 2127 by their surrounding context, not globally.

### 6b. `--prompt-file` / `--negative-prompt-file` (ADOPT — CLI)

Both bind fields (`prompt`, `negative_prompt`) that already exist on the fork's `SDGenerationParams` (verified: the
fork's `--prompt`/`--negative-prompt` string_options at `common.cpp:844/848`). Upstream adds two file-reading
lambdas + two `manual_options` entries in `SDGenerationParams::get_options()`.

First, add the include. Anchor (unique, fork `common.cpp:9`):

```cpp
#include <filesystem>
```

Insert after it:

```cpp
#include <fstream>
```

> Verified: `common.cpp` does **not** already include `<fstream>` (grep returned nothing); the lambdas use
> `std::ifstream` + `std::istreambuf_iterator`.

Then add the two lambdas + entries. Anchor (unique, fork `common.cpp:1560-1562`, the last entry of
`SDGenerationParams::get_options().manual_options`, `--vae-relative-tile-size`):

```cpp
        {"",
         "--vae-relative-tile-size",
         "relative tile size for vae tiling, format [X]x[Y], in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides --vae-tile-size)",
         on_relative_tile_size_arg},
```

Replace with (append the two new entries):

```cpp
        {"",
         "--vae-relative-tile-size",
         "relative tile size for vae tiling, format [X]x[Y], in fraction of image size if < 1, in number of tiles per dim if >=1 (overrides --vae-tile-size)",
         on_relative_tile_size_arg},
        {"",
         "--prompt-file",
         "path to the file containing the prompt to render",
         on_prompt_file_arg},
        {"",
         "--negative-prompt-file",
         "path to the file containing the negative prompt",
         on_negative_prompt_file_arg},
```

And define the two lambdas **above** the `options.manual_options = {` block. Anchor (unique, fork `common.cpp:1490`):

```cpp
    options.manual_options = {
        {"-s",
         "--seed",
```

Insert **before** that line:

```cpp
    auto on_prompt_file_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg = argv[index];
        std::ifstream f(arg, std::ios::binary);
        try {
            prompt = std::string(std::istreambuf_iterator<char>{f}, {});
        } catch (const std::ios_base::failure&) {
            f.setstate(std::ios_base::failbit);
        }
        if (f.fail()) {
            LOG_ERROR("error: failed to read prompt file '%s'\n", arg);
            return -1;
        }
        return 1;
    };

    auto on_negative_prompt_file_arg = [&](int argc, const char** argv, int index) {
        if (++index >= argc) {
            return -1;
        }
        const char* arg = argv[index];
        std::ifstream f(arg, std::ios::binary);
        try {
            negative_prompt = std::string(std::istreambuf_iterator<char>{f}, {});
        } catch (const std::ios_base::failure&) {
            f.setstate(std::ios_base::failbit);
        }
        if (f.fail()) {
            LOG_ERROR("error: failed to read negative prompt file '%s'\n", arg);
            return -1;
        }
        return 1;
    };

```

> The lambdas capture `[&]` (they mutate `SDGenerationParams::prompt`/`negative_prompt`, in scope inside
> `get_options()` — matches upstream and the fork's other `on_*_arg` lambdas). The fork's `manual_options` entry
> shape is `{short, long, help, callback}` (verified `common.cpp:597-623`), identical to upstream. **Do NOT** adopt
> upstream's adjacent `--sampling-method`/`--scheduler` help-string edits (they advertise `dpm++2m_sde`/`flux`/
> `logit_normal` schedulers whose fork enablement is out of scope for Batch 7).

### 6c. A1111 `prepare_info_field` (ADOPT-optional / FLAG — server, adapt call-site)

Upstream fills the previously-empty A1111 `info` field with a JSON infotext. The **function is self-contained** and
all its deps exist in the fork (`sd_sample_method_name`/`sd_scheduler_name` — `stable-diffusion.h:784/786`;
`sd_basename` — `examples/common/log.h:22`; `get_resolved_width`/`get_resolved_height`,
`sample_params.guidance.txt_cfg`, etc. on the fork's `SDGenerationParams`). Add it verbatim to
`examples/server/routes_sdapi.cpp` (before `register_sdapi_endpoints`) — copy upstream's
`prepare_info_field(const SDContextParams&, const SDGenerationParams&, bool img2img)` (see the upstream diff hunk).

Then fill the fork's empty stub. Anchor (unique, fork `routes_sdapi.cpp:309`):

```cpp
            out["info"]       = "";
```

Replace with:

```cpp
            out["info"] = prepare_info_field(*runtime->ctx_params, request.gen_params, img2img).dump();
```

> **Why FLAG, not clean ADOPT:** upstream's diff at this call-site is entangled with an unrelated **`generate_image`
> signature change** (upstream changed it to `bool generate_image(ctx, params, sd_image_t** out, int* n)`; the fork
> keeps `sd_image_t* generate_image(ctx, params)` — verified `stable-diffusion.h:815`) plus an `images_per_batch`
> seed-index rework. **Do NOT adopt those** — keep the fork's `generate_image` call and per-image loop exactly;
> only add the `prepare_info_field` fn + swap the `out["info"]` stub (as above). Confirm the fork's
> `runtime->ctx_params` / `request.gen_params` / `img2img` are in scope at fork `routes_sdapi.cpp:309` (the stub is
> already inside that handler). **Server/API surface may be koblem-owned — flag to owner** whether the C++ server's
> A1111 `info` field is even consumed; if koblem drives the real API this is cosmetic. Lower priority than 6a/6b.

---

## 7. `docs/` — take upstream wholesale (ADOPT 21 files + SKIP 12 already-present)

`git diff --stat 19bdfe2 ea4e566 -- docs/` = 33 files. Cross-referenced against what the **fork** actually touched
(`git diff --stat 19bdfe2 HEAD -- docs/` = 12 files):

### 7a. Upstream-touched, fork NEVER touched → **copy wholesale (21 files)**

```
docs/anima.md            docs/backend.md          docs/chroma.md           docs/chroma_radiance.md
docs/docker.md           docs/ernie_image.md      docs/flux.md             docs/flux2.md
docs/hidream_o1_image.md docs/ideogram4.md        docs/kontext.md          docs/lens.md
docs/longcat_image.md    docs/ltx2.md             docs/ovis_image.md       docs/performance.md
docs/pid.md              docs/qwen_image.md       docs/qwen_image_edit.md  docs/wan.md
docs/z_image.md
```

Apply (from the fork worktree):

```bash
git checkout ea4e566 -- docs/anima.md docs/backend.md docs/chroma.md docs/chroma_radiance.md \
  docs/docker.md docs/ernie_image.md docs/flux.md docs/flux2.md docs/hidream_o1_image.md \
  docs/ideogram4.md docs/kontext.md docs/lens.md docs/longcat_image.md docs/ltx2.md \
  docs/ovis_image.md docs/performance.md docs/pid.md docs/qwen_image.md docs/qwen_image_edit.md \
  docs/wan.md docs/z_image.md
```

> Pure docs, no code, no build impact. `docs/backend.md`/`docs/performance.md` describe upstream's multi-GPU
> backend spec — copying the DOC is harmless (it documents a superset spec; the fork simply doesn't implement the
> multi-GPU pieces). `docs/ideogram4.md`/`docs/pid.md` pair with §5/§4. Optional judgment call: if you prefer the
> docs not to advertise unported multi-GPU/RPC flags, hand-trim `backend.md` — but wholesale is the roadmap's
> instruction and is safe.

### 7b. Upstream-touched AND fork-touched → **already byte-identical to upstream, SKIP (12 files)**

```
docs/adetailer.md  docs/animatediff.md  docs/boogu_image.md  docs/edit.md
docs/hunyuan_video.md  docs/imatrix.md  docs/krea2.md  docs/lingbot_video.md
docs/minit2i.md  docs/pulid.md  docs/rpc.md  docs/sefi_image.md
```

**Evidence (proves already-present):** `for f in <these 12>; do git diff HEAD ea4e566 -- "$f" | wc -l; done`
returns **0 for every file** — the fork already copied upstream's version of each during earlier pull-in batches
(ADetailer/PuLID/sefi/minit2i/AnimateDiff). **No action.** (Note `docs/rpc.md` is here: the fork ships the RPC
*doc* but not the RPC *code* — see §1's RPC note; that gap is a separate owner decision, not a Batch 7 doc edit.)

---

## 8. Blockers, risks + ordered apply plan

**No genuine BLOCKER in Batch 7** — every item is either a self-contained additive adopt, an already-present skip,
or a niche/flagged item that can be deferred without blocking the rest.

**Risks recap:**
- **§2 (flash mask):** the one correctness fix with real risk — the fork's mask layout convention diverged from
  upstream, so the repeat must be traced against actual fork callers and A/B'd on Chroma/T5 **and** the LongCat BSA
  path. **Do not batch it.**
- **§4 (PixelDiT 1.5):** niche, ~354-line gated model-variant port, no checkpoint on-box. Defer unless wanted.
- **§6c (prepare_info_field):** server-side; call-site entangled with the fork's kept `generate_image` signature;
  possibly koblem-owned. Adopt the fn + stub-swap only; flag surface ownership.
- **§1 (backend robustness):** low value on a single-CUDA box; safe but marginal. Optional.
- **RISK to working paths from the safe batch: NONE by construction.** §3 is math-identical (graph-side move); §5
  is a no-op when `circular_*` default false (empty `axis_wrap_dims` ⇒ identical PE); §6a/§6b are additive CLI;
  §7 is docs only. None touches an existing hot path's numerics.

**Ordered apply plan:**

- **Stage 1 — ONE build + commit (safe self-contained adopts):**
  - §3 `AlphaBlender::get_alpha` graph-side (`block.hpp` + `auto_encoder_kl.hpp`, get_alpha hunks only).
  - §5 Ideogram4 seamless (`rope.hpp` `embed_interleaved_mrope` + `ideogram4.hpp` `gen_ideogram4_pe`/build_graph —
    **remember the duplicate-`runner_ctx` deletion in §5c**).
  - §6a `.ckpt` ext (edit `common.cpp:629` + `:2127`; leave `:2339`).
  - §6b `--prompt-file`/`--negative-prompt-file` (+ `#include <fstream>`).
  - §7a docs wholesale (`git checkout ea4e566 -- <21 files>`).
  - §1 backend robustness subset (OPTIONAL — include only if free; the static-name-preserving §1a-§1c).
  Build off-server. Verify: SVD img2vid unchanged (§3); normal Ideogram4 render unchanged (§5); `--prompt-file`
  loads a prompt; `.ckpt` embedding/LoRA now resolves. §7b docs are already present → confirm no diff, no commit
  needed for them.

- **Stage 2 — §6c A1111 `prepare_info_field` (opt-in, server):** add the fn + swap the `out["info"]` stub, keeping
  the fork's `generate_image`. Verify A1111 `/sdapi/v1/txt2img` returns a populated `info` JSON. Flag koblem
  ownership to owner. Can ride Stage 1's commit if server builds cleanly; otherwise separate.

- **Stage 3 — §2 flash-attn mask fix (separate, GPU-gated):** adapt the repeat to the fork's F16-BSA branch, trace
  fork mask-layout callers, then GPU A/B on Chroma/T5 masked flash AND the LongCat BSA path. Land alone.

- **Stage 4 — §4 PixelDiT 1.5 (separate milestone, checkpoint-gated):** port config + forward gate/replicate/
  unpatchify behind `pit_lq_inject`; needs a PixelDiT-1.5 checkpoint to validate. Defer unless owner wants it.

**Verification status:** all fork anchors (`ggml_extend_backend.cpp`, `ggml_extend.hpp`, `block.hpp`,
`auto_encoder_kl.hpp`, `pid.hpp`, `rope.hpp`, `ideogram4.hpp`, `common.cpp`, `routes_sdapi.cpp`, `docs/`) were
`Read`/`grep`/`git diff`-confirmed at the cited lines against the current worktree (HEAD `0fd28e7`). Confirmed
PRESENT in fork: `ggml_ext_cast_f32` (`ggml_extend.hpp:1548`), `ggml_sigmoid` (used in lingbot/pid/ltxv),
`axis_wrap_dims` in `rope()` (`rope.hpp:52`), `circular_x_enabled`/`circular_y_enabled` on GGMLRunnerContext,
`sd_basename`/`sd_sample_method_name`/`sd_scheduler_name`. Confirmed ABSENT (this batch adds): `resolve_device_by_name`/
`resolve_first_device_by_registry_name` (§1), the flash-attn `ne[1]!=L_q` repeat (§2), `pit_lq_inject` (§4),
`axis_wrap_dims` on `embed_interleaved_mrope` (§5), `--prompt-file`/`.ckpt`-in-valid_ext/`prepare_info_field` (§6).
Confirmed ALREADY-PRESENT (SKIP): 12 both-touched docs are byte-identical to `ea4e566` (§7b). Cannot build/compile
or GPU-run here (project rule: C++/CUDA engine builds happen on-server via the ADetailer/PuLID-style build+commit;
no checkpoints for Ideogram4-seamless/PixelDiT-1.5/masked-Chroma on-box).
