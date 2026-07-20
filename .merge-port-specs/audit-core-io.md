# Audit: core + I/O + infra — fork HEAD vs upstream leejet tip

Refs: merge base `19bdfe2` · upstream tip `ea4e566` · fork HEAD `c263e53` (branch
`upstream-merge-2026-07-19`). Method: for each file, `git diff 19bdfe2 ea4e566 -- <file>`
(upstream's evolution from the common base) classified against fork HEAD
(`git show HEAD:<file>`). Read-only; nothing edited or built.

**Categories:** ADOPT = self-contained, upstream ≥ fork, no ModelManager/weight_manager
coupling · ALREADY-HAVE = fork has an equivalent · KEEP-FORK = fork does it better / it is
the graph-cut/quant/video/offload core / architectural divergence · DROPPED-FEATURE = an
upstream capability with no fork equivalent and absent from our tree.

## Orientation — the shared-lineage split

The merge base `19bdfe2` (a real leejet commit, PR #1622) already contained
`SDBackendManager` (backend/params_backend convention) and `ggml_graph_cut`. So these are
**shared infra both sides inherited**, not fork inventions. From that base the two trees
diverged in *opposite architectural directions*:

- **Fork** froze the base's single-GPU `params_backend` + offload model and piled its unique
  value on top: NVFP4/FP8 imatrix, LongCat BSA/offload-ring, LTX/Wan video, cooperative
  cancel. (`ggml_extend.hpp` +2713/-67.)
- **Upstream** tore the offload model out and rebuilt `GGMLRunner` on a `RunnerWeightManager`
  + multi-device `ggml_backend_sched` + `layer_split_partition`, and grew `SDBackendManager`
  toward RPC / row+layer multi-GPU split / disk-backed params. (`ggml_extend.hpp` +681/-724;
  `ggml_extend_backend.*` +367; fork left both **untouched**.)

Consequence, and the core reassurance for the owner's worry: the large upstream churn in
`ggml_extend.hpp` and `ggml_extend_backend.*` is **the WeightManager/ModelManager/multi-GPU
line the steer explicitly says to KEEP-FORK** — it is not silently-dropped standalone value.
The genuinely self-contained upstream improvements are small and enumerated below; the real
DROPPED-FEATUREs are concentrated in `convert.cpp` + streaming writers and in new model
families in `model.h`.

---

## src/core/ggml_extend.hpp  (upstream +681/-724 · fork +2713/-67)

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 1 | Whole `GGMLRunner` memory model swapped: remove `params_backend`, `offload_all_params`/`offload_partial_params`/`restore_*`/`offload_resident_params`, `alloc_params_buffer`/`free_params_buffer`, `layer_registry_`, `compute_streaming_segments`; add `RunnerWeightManager` (weak_ptr), `prepare_execute_graph_weights`, RAII guards | **KEEP-FORK** | Architectural — this IS the offload/params_backend core the steer names. Fork keeps `offload_all_params`/`offload_partial_params`/`resident` (HEAD ggml_extend.hpp still defines them) + `layer_registry_`. Adopting = abandoning the fork's single-GPU offload runner. |
| 2 | Multi-device `ensure_sched`/`ggml_backend_sched`, `pin_multi_device_nodes`, `backend_for_weight`, `extra_runtime_backends`, `is_multi_device()`, `set_runtime_backends()` | **KEEP-FORK** | Multi-GPU tensor/layer split; against the single-GPU core. |
| 3 | `assign_graph_cut_layer_split_backends` + `resolve_graph_cut_layer_split_plan` + `layer_split_partition.h` | **KEEP-FORK** | Multi-GPU layer split; depends on `weight_manager` + `sd::partition_graph_cut_layer_split`. |
| 4 | `ggml_ext_attention_ext`: masked flash-attn fix — when `mask_in->ne[1] != L_q`, `ggml_repeat` the mask to materialize the query dim (bare `ggml_transpose` on a `[n_kv,1]` broadcast mask → `[1,n_kv]` → kernel misreads → NaN/blank); plus `if(!ggml_backend_supports_op(backend,out)) return nullptr;` guard | **ADOPT** (conditional) | Fork HEAD `ggml_extend.hpp:1753` still does the bare `mask_in = ggml_transpose(ctx, mask_in)` (only skipped for the F16/BSA path) with **no** `ne[1]!=L_q` repeat. Real correctness hardening for query-broadcast padding masks (Chroma/T5). Caveat: fork convention passes full-`L_q` masks, so it may not be hit today — verify no fork caller passes a `[n_kv,1]` mask before wiring. Highest-value correctness item in this file. |
| 5 | `ggml_ext_pad_ext` gains `backend` arg + Metal right-pad-only fallback (`ggml_pad_ext(0,lp+rp)` then `ggml_roll`), gated on `ggml_backend_supports_op` (#850) | **ADOPT** (low) | Fork `ggml_ext_pad_ext` (HEAD:1201) has no backend param / no roll fallback. Self-contained but **zero benefit on the fork's CUDA-only box** — Metal portability only. |
| 6 | `ggml_ext_conv_3d` gains `backend` arg + IM2COL_3D→`ggml_conv_3d_direct` fallback when backend lacks im2col_3d (#850) | **KEEP-FORK / ALREADY-HAVE-different** | Fork already routes to `ggml_conv_3d_direct` (HEAD:1342-1348) but for its own env-gated cuDNN/CUDA precision path, not backend-support detection. Upstream's variant is Metal portability (CUDA-irrelevant). |
| 7 | `ggml_tensor_to_sd_image`: drop `channels == 3` assert (allow non-3-channel export) | **ADOPT** (low) | Fork keeps `GGML_ASSERT(channels == 3 && ...)` at HEAD:521 and :545. Trivial relaxation; only matters if a path emits grayscale/4-ch. |
| 8 | `copy_cache_tensors_to_cache_buffer` early-return when cache empty; `execute_graph` `debug_tensors.empty()` skip; `rebuild_params_tensor_set` dirty-flag cache | **KEEP-FORK / minor** | Micro-opts entangled with the rewritten `execute_graph`/params-set lifecycle; not worth cherry-picking against the fork's diverged runner. |
| 9 | `kqv` `view_3d`→`view_4d` (adds N batch dim) in attention | **ALREADY-HAVE** | Fork already uses `ggml_view_4d` for qkv/kqv (HEAD:754, 1504-1506). |

## src/core/ggml_extend_backend.{cpp,h}  (upstream +367 · **fork UNCHANGED from base**)

Fork never touched these since base → every upstream change is conflict-free in principle,
but almost all serve the multi-backend direction.

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 10 | `init()` signature: drop `offload_params_to_cpu`/`keep_clip_on_cpu`/`keep_vae_on_cpu`/`keep_control_net_on_cpu`, add `split_mode_spec` | **KEEP-FORK** | Fork's callers still pass the four offload bools (base signature). Breaking change tied to multi-backend. |
| 11 | `SDSplitMode` LAYER/ROW, `runtime_backends()` (multi), `split_buffer_type` + `tensor_split`, per-module split validation | **KEEP-FORK** | Multi-GPU row/layer split; against single-GPU core. |
| 12 | RPC backend: `add_rpc_devices()` (`ggml_backend_rpc_add_server`) | **KEEP-FORK** | Multi-node distributed backend; against single-GPU core. Pairs with `SD_RPC` in CMake (#20). |
| 13 | `SDBackendModule::DETECTOR` (adetailer/yolo) enum + name | **DROPPED-FEATURE** | Fork has no detector module. Enum is trivial; the actual **adetailer detection pipeline** is a larger upstream feature living outside these files. Blast radius: new module wiring across loaders/pipeline. Effort here: ~5 lines; whole feature: large, out of this file set. Surface only. |
| 14 | `"disk"` params-backend token: `is_disk_backend_token`, `params_backend_is_disk`, `params_backend_follows_runtime` (stream weights from disk instead of resident) | **DROPPED-FEATURE** | Fork lacks disk-backed params. It is upstream's flavor of offload; complements the fork's graph-cut but is wired through the WeightManager path the fork doesn't have. Moderate port effort; conceptually overlaps fork's own offload so low urgency. |
| 15 | `sd_backend_graph_compute_with_eval_callback` + `sd_ggml_graph_view` (per-node eval callback splitting the graph) | **KEEP-FORK / low** | Debug/inspection + abort hook. Fork routes compute through graph-cut segments and already has cooperative cancel (`sd_request_cancel`, see #21). Self-contained but low value; wiring into the segmented path is non-trivial. |
| 16 | Backend name resolution robustness: `resolve_first_device_by_registry_name`, `resolve_device_by_name`, `backend_device_name`; `init_named_backend` tries device-by-name then registry-name; rename `sd_resolve_backend_name`→`sd_backend_resolve_name` (now exported) | **ADOPT** (moderate) | Self-contained: lets specs name backends by registry ("CUDA0", "metal"→"mtl") and device name, independent of multi-GPU. Fork lacks all of these (`sd_backend_resolve_name` ABSENT in fork). The rename is needed by upstream `MaxVramAssignment` but the resolver improvements stand alone. Improves `--backend`/`--params-backend` selection robustness. |

## src/core/ggml_graph_cut.{cpp,h}  (upstream +239 · fork +146)

Both sides diverged. Fork added: fusion `use_counts` population (484/484 NORM fusions were
silently disabled under offload), `saved_cut_leaves` gallocr-measurement fix, and the
`ring_per_block` LongCat offload-ring param cap — all KEEP-FORK core.

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 17 | `is_param_input`: also treat `tensor->view_src ∈ params_tensor_set` as a param input | **ADOPT** (small) | Fork's `is_param_input` has no `view_src` check (`view_src != nullptr` count = 0 in fork graph_cut.cpp). Self-contained robustness: views of params get correctly classified. Note fork *does* apply view_src canonicalization elsewhere, so verify it isn't already covered upstream of this call. |
| 18 | `is_segment_output_needed_after` + `make_segment_seed` change: when merging, keep a segment output only if a *later* segment consumes it (prune dead boundary outputs) | **ADOPT** (moderate) | Lands in `make_segment_seed`, which the fork left unmodified → applies cleanly. Reduces materialized boundary tensors on merged segments — a real VRAM win aligned with the fork's offload goals. Best-value graph_cut adopt; validate against the fork's `ring_per_block` merge path. |
| 19 | Measurement rewrite: `mark_measurement_external` (neutralize INPUT_PARAM **and** INPUT_PREVIOUS_CUT + `view_src` via `data=1` sentinel) + save/restore `output->flags` | **KEEP-FORK** (note breadth gap) | Fork solves the same gallocr-measurement hazard differently via `saved_cut_leaves` (nulls op/srcs for INPUT_PREVIOUS_CUT only). Fork's works for its cases; upstream additionally neutralizes INPUT_PARAM + view_src + output flags. If a future fork segment measures a param leaf with a live producer, revisit — otherwise keep fork's. |
| 20 | `MaxVramAssignment` (parse `--max-vram cuda0=8,cuda1=6`, per-backend byte budgets) | **KEEP-FORK** | Per-backend VRAM map = multi-GPU; against single-GPU core. Depends on `sd_backend_resolve_name` (#16). |
| —  | Upstream deletes `runtime_param_tensors` | n/a | Fork still uses `runtime_param_tensors` in its offload path — do not delete; consistent with KEEP-FORK #1. |

## src/core/util.{cpp,h}  (upstream +63 · fork +18) — disjoint additions

Fork added cooperative-cancel only; upstream added utilities. No conflict.

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 21 | Fork's `sd_request_cancel`/`sd_clear_cancel`/`sd_is_cancel_requested` (atomic flag) | **KEEP-FORK** | Fork-unique cooperative cancel for the video server. Not in upstream. |
| 22 | `sd_type_to_ggml_type(sd_type_t)` helper | **ADOPT** (trivial) | Fork ABSENT — fork casts `(ggml_type)output_type` directly in convert.cpp. One-liner; adopt if adopting streaming convert (#27) which uses it. |
| 23 | `sd_list_devices(char*,size_t)` — enumerate backend devices | **ADOPT** (low) / DROPPED | Fork ABSENT. Backs a `--list-devices` CLI helper. Self-contained; low value on a fixed rig. |
| 24 | Backend eval callback plumbing `sd_set/get_backend_eval_callback[_data]` | **KEEP-FORK / low** | Pairs with #15; low value given fork's segmented compute + cancel. |
| 25 | Download progress bar cosmetics (`'#'` fill, no `>` head for byte progress) | **ADOPT** (cosmetic) | Fork keeps single-style bar. Purely cosmetic. |

## src/convert.cpp  (upstream +268 · fork +150) — **primary DROPPED-FEATURE bundle**

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 26 | Fork's nvfp4-twolevel imatrix: `sd_imatrix_collect_begin/_write_gguf`, `load_imatrix_gguf`, `convert_with_imatrix`, rules-aware ctx sizing | **KEEP-FORK** | Fork-unique quant core (`SDImatrixCollector`). Woven into the *old* single-context convert path. |
| 27 | **Streaming conversion rewrite**: per-tensor load with a 1 GB memory budget + N worker threads, `preallocate_output_file`, `StreamingModelWriter` (`SafetensorsStreamingWriter`/`GGUFStreamingWriter`), and `convert_with_components(model/clip_l/clip_g/t5xxl/diffusion/vae, n_threads)` | **DROPPED-FEATURE** | Fork keeps the **old whole-model-in-one-ggml_ctx** convert. It lacks (a) bounded-memory conversion (fork holds the entire model in RAM — a real limit converting large DiTs on the box) and (b) multi-component convert. Blast radius: convert.cpp + `gguf_io` streaming writer (#28) + `safetensors_io` streaming writer (trackB) + `tensor_storage.h` `TensorWritePlan` (#31) + `model_loader.load_tensor()`. Partially scaffolded — `streaming_writer.h` base class is already copied into the fork (orphaned; see cross-ref). Effort: moderate; **must re-weave the fork's imatrix hook** into the streaming path. **Highest-value DROPPED-FEATURE.** |

## src/model_io/gguf_io.{cpp,h} + tensor_storage.h  (upstream +143/+28/+19 · **fork UNCHANGED**)

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 28 | `GGUFStreamingWriter` (metadata-then-seeked-tensor streaming via `gguf_write_to_file_ptr` + offset table) | **DROPPED-FEATURE** | Part of the #27 bundle. Fork's `gguf_io.h` has no `Streaming` class. Self-contained given `streaming_writer.h` + `TensorWritePlan`. |
| 31 | `tensor_storage.h`: `TensorWritePlan` struct (name/type/ne/n_dims + `nbytes()`) | **DROPPED-FEATURE** | Fork's `tensor_storage.h` has only `TensorWriteInfo` (base); `TensorWritePlan` ABSENT even though the fork's copied-in `streaming_writer.h` *references* it → that header is currently an **orphan that would not compile if built**. Confirms the streaming port is incomplete. Add this struct as step 1 of adopting #27. |

## CMakeLists.txt  (upstream +58 · fork +63) — disjoint

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 29 | `SD_RPC`→`GGML_RPC ON` + `-DSD_USE_RPC` | **KEEP-FORK** | Enables the RPC backend (#12); multi-node, against single-GPU core. |
| 30 | `GGML_MAX_NAME` 128→**160** | **ADOPT** (trivial) | Fork still at `-DGGML_MAX_NAME=128` (HEAD CMakeLists:372). The fork prefixes graph-cut tensor names with `ggml_runner_cut:<group>:<output>`; 160 gives headroom against silent truncation. One-line, low-risk. |
| 32 | zip as object lib (`$<TARGET_OBJECTS:zip>`) instead of `PUBLIC` link; full install/pkgconfig/cmake-package export (`stable-diffusion-config.cmake`, `.pc`) | **KEEP-FORK / decline** | Packaging for downstream consumers; fork ships via Docker, not `make install`. The `.pc`/config `.in` files aren't in the fork tree; adopting pulls unneeeded packaging. Low value. |
| — | Fork's Opus + libvpx/VP9 WebM detection, `SD_USE_CUDA` gate | **KEEP-FORK** | Fork-unique video-output deps + CPU-build link gate. Not in upstream. |

## .github/workflows/build.yml  (upstream +34/-41 · **fork UNCHANGED**)

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 33 | `Dockerfile*`→`docker/Dockerfile*` paths; collapse AVX matrix to a single `cpu` build using `GGML_BACKEND_DL=ON`+`GGML_CPU_ALL_VARIANTS=ON`; drop AVX512 probe; ROCm disk-cleanup reorder | **KEEP-FORK / decline** | Upstream CI targets its own release matrix + dynamic-backend packaging. Fork builds engines via its own Docker/`iter.sh` off this workflow. No fork-relevant build-flag worth taking; the meaningful flag change (`GGML_BACKEND_DL`/`CPU_ALL_VARIANTS`) is a packaging choice the fork does not use. |

## src/model.h  (upstream +74 · fork +45) — both add SDVersion enums

Overlap: fork already added KREA2, BOOGU_IMAGE, MINIT2I, HUNYUAN_VIDEO, LINGBOT_VIDEO (+ its
own LONGCAT_AVATAR, WAN_S2V) with matching `sd_version_is_*` predicates and `is_dit()` wiring.
**Note:** enum *ordering* differs between fork and upstream (SDVersion is detected at load, not
serialized, so this is compatibility-safe but will conflict textually on any future merge).

| # | Upstream change | Category | Fork anchor / rationale |
|---|---|---|---|
| 34 | `VERSION_KREA2`, `VERSION_BOOGU_IMAGE`, `VERSION_MINIT2I`, `VERSION_HUNYUAN_VIDEO`, `VERSION_LINGBOT_VIDEO` (+ predicates, `is_dit()`) | **ALREADY-HAVE** | Fork model.h added all five (HEAD:46-60, predicates :196-216). Ordering differs but semantics match. |
| 35 | `sd_version_uses_hunyuan_video_vae` | **ALREADY-HAVE** | Fork added identical helper (HEAD:224). |
| 36 | `VERSION_QWEN_IMAGE_LAYERED`, `VERSION_SEFI_IMAGE`, `VERSION_ESRGAN` (+ predicates; `sd_version_is_qwen_image` OR-extended; `uses_flux2_vae` extended for sefi) | **DROPPED-FEATURE** | Fork ABSENT (grep = 0). Three upstream model families with no fork enum. Enum stubs are trivial; **real support (arch + loader + VAE routing) lives outside this file set** — surfacing only. ESRGAN (classic upscaler) is the most self-contained of the three. |
| 37 | `sd_version_uses_flux_vae`, `sd_version_uses_wan_vae` convenience helpers | **DROPPED-FEATURE** (minor) | Fork ABSENT. Pure refactors of VAE-family routing used by model.cpp; low value unless adopting the upstream VAE-selection code that calls them. |

## Files with NO upstream change since base (accounted, nothing to do)

`src/core/tensor_ggml.hpp`, `examples/CMakeLists.txt`, `.gitmodules`, `.dockerignore` — empty
upstream diff. `.gitignore`: upstream added only `.claude/ CLAUDE.local.md .agents/ .codex/`
(agent-tooling ignores) — **ADOPT (trivial)** if desired; the fork already ignores far more.

---

## Counts (37 numbered upstream change-groups)

| Category | Count | Items |
|---|---|---|
| ADOPT | 10 | #4 (cond.), #5, #7, #16, #17, #18, #22, #23, #25, #30 (+ .gitignore trivial) |
| ALREADY-HAVE | 4 | #9, #34, #35, and view_4d |
| KEEP-FORK | 13 | #1, #2, #3, #6, #8, #10, #11, #12, #15, #19, #20, #24, #29, #32, #33 (offload/quant/video/multi-GPU/packaging/CI core) |
| DROPPED-FEATURE | 6 | #13 (DETECTOR/adetailer), #14 (disk params), #27 (streaming convert), #28 (GGUF streaming writer), #31 (TensorWritePlan), #36/#37 (qwen-layered / sefi / esrgan model families + vae helpers) |

*(#26, #21, and the fork-unique CMake/util rows are fork additions, not upstream changes —
listed for completeness, not counted.)*

## Bottom line

- **Highest-value ADOPT:** #16 — the **backend-name-resolution robustness** in
  `ggml_extend_backend.cpp` (`resolve_device_by_name` / `resolve_first_device_by_registry_name`
  + exported `sd_backend_resolve_name`). It is fully self-contained (independent of the
  multi-GPU machinery), the fork's file is byte-identical to base so it applies cleanly, and it
  hardens `--backend`/`--params-backend` selection by registry and device name. Runner-up on
  pure correctness: #4 (masked flash-attn `ne[1]!=L_q` repeat) — adopt if any fork caller can
  pass a query-broadcast mask; #18 (dead-output pruning in segment merge) is the best
  offload-aligned graph_cut adopt.

- **Highest-value DROPPED-FEATURE (and the one to actually surface to the owner):** #27 — the
  **streaming / bounded-memory model conversion** (`convert.cpp` rewrite + #28 GGUF streaming
  writer + #31 `TensorWritePlan` + the trackB safetensors writer). The fork's `convert` still
  materializes the *entire* model in one `ggml_ctx`, which caps the size of DiT it can quantize
  on-box; upstream converts within a 1 GB budget across N threads and adds multi-component
  convert. It is **partially scaffolded but incomplete**: `streaming_writer.h` was copied into
  the fork yet `TensorWritePlan` is missing, so that header is currently an orphan that would
  not compile — a concrete sign the port was started and dropped. Not silently lost from a
  "merged" file, but genuinely absent and worth finishing (re-weaving the fork's nvfp4 imatrix
  hook into the streaming path is the one non-mechanical step).

- **No silent drops in the KEEP-FORK churn:** the large `ggml_extend.hpp` (+681/-724) and
  `ggml_extend_backend.*` (+367) upstream deltas are the WeightManager / ModelManager /
  RPC / multi-GPU-split direction the steer explicitly says to decline — accounted for, not
  lost value.
