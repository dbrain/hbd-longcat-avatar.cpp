# longcat-avatar merge — SUPERSET PULL-IN ROADMAP (2026-07-20)

Owner mandate: the fork is a **performance/feature-add superset of upstream leejet**, NOT a replacement.
Every upstream model + user-facing feature must still work; the fork adds graph-cut VRAM/offload,
NVFP4/FP8 cuBLASLt GEMM, LTX/Wan/audio video engine, SA3 on top. If an upstream capability is missing,
that's a bug to fix — pull it in. The ONLY deliberate non-port is upstream's multi-GPU
ModelManager/weight_manager/backend_fit/layer_split *architecture* (the fork does the same single-GPU
job via graph-cut; true 2-GPU-per-model sharding is the one capability that'd need re-architecture, and
a single-3060/5060Ti box doesn't use it).

Refs: merge base `19bdfe2`, upstream tip `ea4e566`, fork branch `upstream-merge-2026-07-19`.
Full evidence: `.merge-port-specs/audit-core-io.md`, `audit-diffusion.md`, `audit-conditioner-dropped.md`,
`trackB-*.md`. Build: `bash .merge-port-specs/build.sh` (docker longcat-avatar-dev:builder, sm86;120+SA3).

## Base state (already done, all built + committed)
- ggml v0.13→v0.17 merge (531a2c72 submodule) + sd.cpp fork+upstream-ggml merge (6c0fdf2).
- Models ported onto fork runner: krea2 7502d89, boogu ca0d922, hunyuan f14b2b2, lingbot d279a0c, minit2i 785ba0d.
- Track B initial adopts 3ac6af4 (sharded safetensors, t5 detect, Beta scheduler, bounds hardening).

## PULL-IN PROGRESS (2026-07-20)
- Batch 1 self-contained adopts — DONE, commit 3e76178.
- Batch 2 RefImageParams resize + lingbot bugfix — DONE, commit c58a01d.
- Batch 3a AnimateDiff (graph eaf8f3d + glue 2b9d689) — DONE, builds. GPU-PENDING: frame-expansion no-op check.
- Batch 3b sefi_image — DONE, commit de2dd4a, builds. GPU-PENDING: dual-schedule render + ref-image path.
- Batch (qwen_image_layered A+B+C) — DONE, builds. Wired A model-plumbing (§1-4 enum/fold/str/detection),
  B addition_t_cond (§5 gated additive, default OFF), C wan_vae is_2D (§7 Conv2dBut3d + fully is_2D-gated
  Residual/Encoder/Decoder/WanVAE/Runner), conditioner vision gate (§9). Fixed QwenImageRunner uninitialised
  `version` member. §6 rope = NO edit (single-4D-forward design → layer temporal id 0). GPU-PENDING: no
  layered checkpoint on-box → Conv2dBut3d axis-swap orientation, decoder.conv1.weight ne[2]>3 detection,
  input_channels=4 all UNVERIFIED. Non-disturbance: every non-layered path byte-identical (is_2D=false).
- REMAINING (apply-docs / roadmap ready; wire like the above, one build+commit each):
  - qwen_image_layered ORCHESTRATION (§8, deferred sub-milestone) — get_image_channels(), qwen_image_layers
    request field, 5-D/looped init-latent + per-layer RGBA decode, fork-wide 3→4 channel output audit. No fork
    anchor (video-first divergence) + no checkpoint. Land as its own milestone.
  - ADetailer ENGINE CORE — DONE, builds. apply-doc `.merge-port-specs/adetailer-APPLY.md`. New detailer.{cpp,h}
    (CPU DSP verbatim), de-staled orphaned yolov8.h (ctor→(backend,params_backend), compute 3-bool→2-bool),
    load off ModelManager→ModelLoader::load_tensors+alloc_params_buffer (UPSCALER slot reuse; get_metadata absent
    →class labels degrade to class_N), inpaint via fork sd_image_t* generate_image + manual free, public C API in
    stable-diffusion.h. REMAINING (optional §8): CLI ad_* params + main.cpp apply_adetailer + resource_owners.hpp
    ADetailerCtxPtr; koblem/server field = Rust, owner. GPU-PENDING: no converted YOLOv8 .safetensors on-box.
  - PuLID — DONE, builds (all 4 stages). apply-doc `.merge-port-specs/pulid-APPLY.md`. Fork's flux.hpp was
    upstream-minus-pulid, so §4 = clean re-insertion (3 conventions: increase_ref_index bool, (backend,
    params_backend) ctor, 3-arg compute). Stage1 params/FluxDiffusionExtra/before_diffusion hook; Stage2 weights
    merge + registration + prepare plumbing; Stage3 new pulid_extension.cpp (verbatim) + denoise hook; Stage4
    flux.hpp CA-block build + double/single injection + threading. CLI (§9) fully wired (--pulid-weights/
    --pulid-id-embedding/--pulid-id-weight; fixed the positional sd_ctx_params_t init the new field shifted).
    GPU-PENDING: no PuLID checkpoint / id-gguf on-box → CA names, id shape [2048,32], VRAM of ~20 CA blocks
    UNVERIFIED. VERIFY non-PuLID flux stays bit-exact (pulid_enabled=false ⇒ no blocks/injection).
  - streaming/bounded-mem convert (Batch 6) — DONE, builds. apply-doc streaming-convert-APPLY.md. Added
    TensorWritePlan (un-orphaned streaming_writer.h) + ModelLoader::load_tensor (singular selective, n_threads=1) +
    GGUF/Safetensors StreamingWriter + convert.cpp streaming path (collect/load/stream per-tensor under 1GB budget)
    + public convert_with_components. OLD convert() UNTOUCHED (default byte-identical). imatrix re-weave = free
    (global g_convert_imatrix applies via load_tensor→convert_tensor). CLI --streaming flag NOT wired (koblem drives).
    GPU/convert-PENDING: no large model run → RSS bounding, quant round-trip, imatrix effect are Stage-4 CPU-convert
    validation. Streaming output NOT byte-identical to old writer (tensor order/header) — validate by round-trip.
  - Batch 7 remaining adopts + niche + CLI + docs-wholesale.
- HOLD (GPU/checkpoint-verify, do NOT blind-adopt): qwen3-vl vision split_patch_embed bias; qwen rope txt_id_start.

## PULL-IN BATCHES (this phase)

### Batch 1 — self-contained adopts  [DONE 3e76178]
- lora.hpp LoKr `lokr_w2_a_name` insert bug (line 615 dup) — FIXED.
- control.hpp `CONTROL_NET_GRAPH_SIZE = MAX_GRAPH_SIZE` — large SDXL ControlNet fit.
- vae.hpp image-encode tile factor 2.0 (video VAEs keep 1.30539) — ⚠ raises 3060 image-encode VRAM; GPU-validate.
- denoiser.hpp `LogitNormalScheduler` + LOGIT_NORMAL_SCHEDULER enum/str/case.

### Batch 2 — RefImageParams i2v resize (Scope A)  [NEXT] ★ fixes a regression I introduced
- Fork lingbot i2v branch compares AREA (h_bar*w_bar) vs LINEAR bounds (2*factor/128*factor) → caps ref
  image at ~60×60, starves the vision encoder. BUG.
- Port: `enum RefImageResizeMode{NONE,LONGEST_SIDE,AREA}` + `RefImageResizeParams` field on ConditionerParams
  + upstream `resize_image_dims()` helper + rewrite the 3 inline branches (qwen ~1982 / longcat ~2050 /
  lingbot ~2282) to use it. Keep fork's 2-arg `llm->encode_image`. Preserve qwen/longcat tuned 384²/560²;
  fix lingbot to AREA 4*factor²/16384*factor². Caller sets preset by SDVersion in stable-diffusion.cpp.
- Scope B (full RefImageParams: pass_to_vlm/pass_to_dit + Rope::RefIndexMode DECREASE + REF_IMAGE_PRESETS,
  threads every DiT) — DEFERRED (large, needs rope ref-index port; FIXED/INCREASE ref conditioning already works).

### Batch 3 — dropped models (scaffolding already in-tree; wire like Track A)
- SEFI_IMAGE (`sefi_image.hpp` + `.merge-port-specs/sefi_image.json` present) — flux-family, wire detection/dispatch.
- QWEN_IMAGE_LAYERED (`VERSION_QWEN_IMAGE_LAYERED`; needs wan_vae 2D `Conv2dBut3d` + layered decode + get_image_channels==4).
- AnimateDiff SD1.5 motion modules (`.merge-port-specs/animatediff.json`; unet.hpp AnimateDiffModel wiring + num_video_frames).

### Batch 4 — ADetailer  [dropped feature, port ~M]
- `detailer.{cpp,h}` + wire fork's orphaned `src/model/detector/yolov8.h` as a `GGMLRunner(backend,params_backend)`
  (drop its ModelManager). Pure-CPU detect/NMS/mask/sort verbatim; inpaint pass reuses sd_ctx img2img.
  Adds `ad_*` CLI/server params + `resource_owners.hpp` ADetailerCtxPtr.

### Batch 5 — PuLID  [dropped feature, port ~L, DiT-touching]
- FluxDiffusionExtra += `pulid_id`,`pulid_id_weight`; flux.hpp must consume/inject the ID tokens (the hard part —
  fork flux.hpp has zero pulid refs); generation_extension.h += `before_diffusion` hook + `create_pulid_extension`
  + `sd_pulid_params_t`; wire fork's orphaned `src/model/adapter/pulid.hpp` (ID encoder). CLI/server param surface.

### Batch 6 — streaming/bounded-memory model conversion  [dropped feature]
- convert.cpp holds whole model in one ggml_ctx → caps on-box DiT quant size. Finish the half-scaffolded
  streaming writer: add `TensorWritePlan` to tensor_storage.h (orphan today), re-weave fork's nvfp4 imatrix hook
  into the streaming path.

### Batch 7 — remaining self-contained adopts + niche + examples + docs
- core: backend name-resolution robustness (`resolve_device_by_name`/`sd_backend_resolve_name`, ggml_extend_backend.cpp);
  masked flash-attn ne[1]!=L_q repeat fix (ggml_extend.hpp ~1753); AlphaBlender::get_alpha graph-side.
- niche: PixelDiT 1.5 `pit_lq_inject` (pid.hpp); Ideogram4 seamless/circular tiling (needs rope axis_wrap_dims).
- examples: `--prompt-file`/`--negative-prompt-file`, A1111 `info` JSON `prepare_info_field`, `.ckpt` embedding/LoRA ext.
- docs/: take upstream wholesale (fork made zero doc edits).

## HOLD — needs GPU/checkpoint verification before adopting (could CORRUPT working paths)
- llm.hpp Qwen3-VL vision `split_patch_embed` + combined-bias rework — live path (hidream/qwen-image-edit/every
  i2v ref encode). Correct only if checkpoints store a single combined `visual.patch_embed.bias` vs fork's per-conv
  bias. VERIFY by dumping `visual.patch_embed.*` from a real Qwen3-VL checkpoint; adopt the rework as a unit only if combined.
- rope.hpp `gen_qwen_image` `txt_id_start=max(h,w)/2` — alters STANDARD qwen-image rope positions (fork reverted).
  Live numeric divergence; A/B before adopting.
- disk-backed params streaming (ModelManager Disk residency) — architectural; fork uses graph-cut. Only if a model
  can't fit even with graph-cut.

## THEN
GPU-validate (determinism + per-model smoke renders) → fast-forward master: ggml → push dbrain/ggml; sd.cpp →
push dbrain/hbd-longcat-avatar.cpp. i2v note: Scope-A resize lands the VLM ref path; Scope-B RoPE DECREASE deferred.
