# IMATRIX-RUNBOOK — calibrated nvfp4 for the LTX-2.3 22B **dev** DiT

Goal: produce `nvfp4-imatrix-dev050.gguf` — a **drop-in replacement** for
`nvfp4-CLEAN-dev050.gguf` that keeps the fast nvfp4 path (~16 GB, ~135 s) but uses
an **AWQ-style imatrix** (activation-importance) to steer nvfp4 rounding instead of
round-to-nearest (RTN). Same recipe (dev + distill-LoRA@0.5), same on-disk format,
only the quantizer changes → cleanest possible A/B for "does calibration recover the
RTN mush".

Legend: **[CPU]** = safe to run any time (no GPU). **[GPU]** = owner runs on the 5060 Ti.
Prepped by the investigation: fold script written+validated, all commands pinned.

---

## Verified facts (why this pipeline is correct)

- **Engine has the full calibrated-nvfp4 path already** (no code changes needed):
  - `-M convert --type nvfp4 --imatrix <f>` → `convert.cpp:convert_with_imatrix` →
    `model_loader.cpp:convert_tensor` → `ggml_quantize_chunk(GGML_TYPE_NVFP4, …, im)`.
    The imatrix is looked up per-tensor by `resolve_imatrix()` (exact name, then
    `diffusion_model.`-stripped) and fed as `quant_weights` into
    `quantize_row_nvfp4_impl` (`ggml-quants.c:421`), which does two-level +
    imatrix-weighted E2M1 code selection (`best_index_mxfp4_w`).
  - `-M vid_gen --collect-imatrix <out>` is **wired for VID_GEN** here
    (`examples/cli/main.cpp:891` begin + `:1266` write) — unlike longcat-avatar.cpp,
    no one-line patch is needed.
  - `--type nvfp4` resolves via ggml `type_name="nvfp4"` (`str_to_sd_type`,
    `stable-diffusion.cpp:3360`). ✓
- **Convert reads bf16 safetensors directly** (`ModelLoader::init_from_file` →
  `init_from_safetensors_file`). We do **NOT** need to download unsloth's
  `ltx-2.3-22b-dev-F16.gguf` (42 GB) — the 46 GB bf16 dev already on disk is the source.
- **Output format is byte-compatible with the RTN nvfp4** (`ggml-quants.c:384`:
  "kept identical to the RTN path"). Calibrated convert emits a **self-contained**
  block_nvfp4 gguf with **no `.wglobal` sidecars** ("legacy folded gguf" in the loader,
  `stable-diffusion.cpp:171/1411`) → loads & runs with the **same render flags**
  as dev050 (`GGML_NVFP4_CUBLASLT=1`, etc.). ✓
- **imatrix name matching**: collector writes bare stripped names
  (`transformer_blocks.N.attn1.to_q.weight`); convert from the safetensors strips
  `model.diffusion_model.` → identical bare key. ✓ (verified against real tensor lists.)

## Inputs on disk (confirmed present)

| what | path | note |
|---|---|---|
| bf16 dev DiT | `/mnt/ssd/models/ltx23-src/ltx-2.3-22b-dev.safetensors` | 46 GB, 5947 tensors |
| distill LoRA | `…/longcat-avatar.cpp/models/ltx2/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | rank=alpha=384 → lora scale 1.0; 1660 targets |
| current RTN nvfp4 | `…/longcat-avatar.cpp/models/ltx2/nvfp4-CLEAN-dev050.gguf` | 16.26 GB (the A/B baseline) |
| builder image | `longcat-avatar-dev:builder-cudnn-ff` | has `/src/build-cudnn/bin/sd-cli` |

Disk: `/mnt/ssd` = 56 GB free (tight); `/` (nvme) = **200 GB free** → write the
46 GB merged bf16 to nvme.

---

## Recommendation: **fold-first** (dev + distill@0.5 → bf16), then imatrix-quant

Why fold instead of raw-dev + runtime `--lora`:
- On nvfp4, a runtime LoRA forces **per-Linear dequant→f32 linear** for every DiT
  weight (`patch_weight`: dequant nvfp4→f32, add diff, run f32) — that **destroys the
  fast nvfp4 GEMM path** (the whole point: 16 GB / ~135 s). Folding keeps every DiT
  Linear native nvfp4.
- Folding makes a **true drop-in A/B** vs `nvfp4-CLEAN-dev050.gguf`: only RTN→imatrix
  differs. (fp8 tolerates runtime LoRA differently; for nvfp4-speed, fold.)

A zero-fold **fallback** exists (Step 2-alt below) if you want a quick probe.

---

## STEP 1 — [CPU] fold distill@0.5 into bf16 dev  → merged bf16 source

Script written & **dry-run validated** (1660/1660 lora targets matched, 0 unmatched):
`tools/fold_ltx_distill_bf16.py`.  Math: `W += 0.5 · (B @ A)` per target
(lora α/r = 384/384 = 1.0). Streaming per-tensor (peak RAM ≈ largest tensor, ~a few
hundred MB). Runtime: ~10–30 min (46 GB read + 46 GB write).

```bash
cd /home/dbrain/dev/longcat-avatar-ltxdenoise
python3 tools/fold_ltx_distill_bf16.py \
  /mnt/ssd/models/ltx23-src/ltx-2.3-22b-dev.safetensors \
  /home/dbrain/dev/longcat-avatar.cpp/models/ltx2/loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
  /home/dbrain/ltx-2.3-22b-dev050-bf16.safetensors 0.5
# (validate first, no write:  … 0.5 --dry-run)
```

Output: `/home/dbrain/ltx-2.3-22b-dev050-bf16.safetensors` (~46 GB, on nvme).
Left un-run by the investigation (46 GB write is the owner's to place); dry-run proven.

---

## STEP 2 — [GPU] collect the imatrix   (the ONE GPU step; ~2–7 min)

Collect on the **fast, already-folded** `nvfp4-CLEAN-dev050.gguf` (activation
distribution ≈ bf16; standard llama.cpp practice to collect on the quantized model —
much faster than a bf16 render, and every DiT Linear fires in a single render so one
pass is sufficient). Disable hires so only the base DiT contributes.

```bash
cd /home/dbrain/dev/longcat-avatar-ltxdenoise
LTX2=/home/dbrain/dev/longcat-avatar.cpp/models/ltx2
IMOUT=/home/dbrain/dev/longcat-avatar-ltxdenoise/ltx-denoise-repro/_imatrix
mkdir -p "$IMOUT"
PROMPT="Locked-off wide shot of a pedestrian crossing on a busy daytime city street, shot on a 35mm lens at eye level from the far kerb; the camera does not move. Cool overcast daylight flattens the scene and wet pavement holds pale reflections of the surrounding buildings. A woman in a red raincoat and jeans walks briskly from the left of the frame across the zebra crossing to the right, her gait natural and purposeful, a canvas bag swinging at her shoulder. Behind her, blurred traffic waits at the light and a few other pedestrians drift at the edges of the frame. She stays at a middle distance, small in the wide composition, never dominating the frame. Ambient city sound of idling engines, distant chatter, and footsteps on the crossing."

docker run --rm --gpus '"device=1"' \
  -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e LTX_DIT_F16=1 \
  -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1 \
  -v /home/dbrain/dev/longcat-avatar-ltxdenoise:/src -v "$LTX2":/ltx2 -w /src \
  longcat-avatar-dev:builder-cudnn-ff \
  stdbuf -oL -eL /src/build-cudnn/bin/sd-cli -M vid_gen \
  --diffusion-model /ltx2/nvfp4-CLEAN-dev050.gguf \
  --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors \
  --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
  --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf \
  --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  -p "$PROMPT" \
  -W 640 -H 352 --video-frames 121 --fps 24 \
  --sampling-method euler --steps 8 --cfg-scale 1.0 --diffusion-fa \
  --vae-tiling --offload-to-cpu --mmap --max-vram 11 -s 42 -v \
  --collect-imatrix /src/ltx-denoise-repro/_imatrix/ltx-dev050.imatrix \
  -o /src/ltx-denoise-repro/_imatrix/_collect.webm
```

Notes:
- Reduced res (640×352) + no hires keeps this fast; the imatrix is a coarse
  per-column 2nd-moment and is robust to res. Success log line:
  `imatrix: wrote N DiT Linear tensors to …ltx-dev050.imatrix`.
- **Optional stronger calibration** (multi-prompt): re-run with the s1 and s2 prompts
  (below) to separate `*.imatrix` files, then average per-column (n_tokens-weighted).
  One prompt is sufficient for a first calibrated quant; do this only if the single-prompt
  result under-recovers.
  - s1: "Wide cinematic shot on a rain-slicked city street at night… a man in his early
    thirties… walks steadily toward the camera…"
  - s2: "Medium-wide shot at a crowded outdoor night concert… a young woman… dances at
    the centre of the frame…"
  (full byte-exact text in `ltx-denoise-repro/run_ablation.sh` cases `1:t2v` / `2:t2v`.)

---

## STEP 3 — [CPU] quantize: imatrix-calibrated nvfp4   (no GPU; ~few min)

Convert is CPU-math (per longcat-avatar `avatar_nvfp4_2_quantize.sh`). Source = the
merged bf16 from Step 1; imatrix = Step 2.

```bash
cd /home/dbrain/dev/longcat-avatar-ltxdenoise
docker run --rm \
  -v /home/dbrain/dev/longcat-avatar-ltxdenoise:/src \
  -v /home/dbrain:/host -w /src \
  longcat-avatar-dev:builder-cudnn-ff \
  /src/build-cudnn/bin/sd-cli -M convert \
  -m /host/ltx-2.3-22b-dev050-bf16.safetensors \
  --type nvfp4 \
  --imatrix /src/ltx-denoise-repro/_imatrix/ltx-dev050.imatrix \
  -o /host/nvfp4-imatrix-dev050.gguf -v
# then place it next to the baseline:
#   mv /home/dbrain/nvfp4-imatrix-dev050.gguf \
#      /home/dbrain/dev/longcat-avatar.cpp/models/ltx2/nvfp4-imatrix-dev050.gguf
```

Expected: same tensor list/dims/format as `nvfp4-CLEAN-dev050.gguf`, ~16 GB. The
model_loader skip-list keeps norms/adaLN/bias high-precision automatically; only the
heavy Linears become nvfp4 — same set the imatrix covers.

### STEP 3-alt — [CPU] zero-fold quick probe (skips Step 1)
If you want a result without the 46 GB fold: quantize the **existing folded gguf**
in place (dequant nvfp4→f32→requant with imatrix). Compounds the ~13% RTN error
(inferior to Step 3) but needs no bf16 source:
```bash
  … -M convert -m /ltx2/nvfp4-CLEAN-dev050.gguf --type nvfp4 \
      --imatrix …/ltx-dev050.imatrix -o /host/nvfp4-imatrix-dev050-dq.gguf -v
```

---

## STEP 4 — [GPU] A/B eval  (RTN vs imatrix)

Same harness, only `DIT` swapped — the calibrated gguf uses identical flags:
```bash
cd /home/dbrain/dev/longcat-avatar-ltxdenoise/ltx-denoise-repro
DIT=nvfp4-imatrix-dev050.gguf bash run_parity_nvfp4.sh   # vs default nvfp4-CLEAN-dev050
```
(run_parity_nvfp4.sh reads `$DIT` from `…/longcat-avatar.cpp/models/ltx2`.)
Eye-test both on the LAN page; do not auto-declare a winner.

---

## Cost / blockers summary

- **GPU cost = one collection render** (Step 2): ~2–7 min at 640×352/8-step, no hires.
  Everything else (fold, quantize, A/B setup) is CPU.
- **No downloads needed** — bf16 dev + LoRA + baseline all on disk.
- **Disk**: write the 46 GB merged bf16 and the 16 GB output to **nvme `/home`**
  (200 GB free); `/mnt/ssd` is tight (56 GB).
- **No code changes** — vid_gen collect is already wired; `--type nvfp4 --imatrix`
  is implemented; output is drop-in.
- Fold script `tools/fold_ltx_distill_bf16.py` is **written and dry-run validated**
  (1660/1660 targets); the actual 46 GB write (Step 1) is left for the owner to place.
```
