# ComfyUI Setup — LTX-2.3 "Denoise-AI" oracle

Setup of the resident ComfyUI (`/home/dbrain/dev/comfy-docker`, image `comfyui-ltx:latest`) to run
the downloaded LTX-2.3 Denoise-AI workflows as a ground-truth reference for our C++ port.

**Status: SETUP ONLY — the container was NOT started and the GPU was NOT touched** (owner's work is
running). Everything below is verified statically (git-clone + grep of each node pack); the only thing
that cannot be verified without launching Comfy is that the graphs actually execute end-to-end.

Prepared 2026-07-05.

## 1. Node type → pack map

Union of node types across `v1_260318_image-to-video.json`, `v2_260329_dynamic.json` (reference),
`v3_260412_multi-image.json`. Each mapping was verified by cloning the pack at HEAD and grepping its
registrations.

### ComfyUI CORE (comfy_extras / nodes.py) — no custom pack needed
Verified against `comfyanonymous/ComfyUI` @ `7f287b7` (2026-07-04), the commit the Dockerfile now pins.
**These are the LTX-2.3 audio/video nodes — they are CORE, not the Lightricks pack** (this corrects the
task's assumption that they live in ComfyUI-LTXVideo):

| node type | core file |
|---|---|
| EmptyLTXVLatentVideo, LTXVConditioning, LTXVCropGuides, LTXVPreprocess, LTXVSeparateAVLatent, LTXVConcatAVLatent | comfy_extras/nodes_lt.py |
| LTXVAudioVAELoader, LTXVAudioVAEDecode, LTXAVTextEncoderLoader, LTXVEmptyLatentAudio | comfy_extras/nodes_lt_audio.py |
| LTXVLatentUpsampler | comfy_extras/nodes_lt_upsampler.py |
| LatentUpscaleModelLoader | comfy_extras/nodes_hunyuan.py |
| ResizeImageMaskNode | comfy_extras/nodes_replacements.py |
| CFGGuider, CLIPTextEncode, CheckpointLoaderSimple, KSamplerSelect, LoadImage, MarkdownNote, PreviewImage, PrimitiveBoolean, PrimitiveFloat, RandomNoise, SamplerCustomAdvanced, VAEDecode, VAEDecodeTiled | core nodes.py |

### Custom packs (baked, refreshed to HEAD on this rebuild)

| node type | pack | repo | verified |
|---|---|---|---|
| LTXVImgToVideoConditionOnly | ComfyUI-LTXVideo | Lightricks/ComfyUI-LTXVideo (@aceeae9, 2026-06-30) | yes |
| LTX2_NAG, LTX2AudioLatentNormalizingSampling | ComfyUI-KJNodes | kijai/ComfyUI-KJNodes (@e27a505, 2026-07-02) | yes |
| SimpleCalculatorKJ, INTConstant | ComfyUI-KJNodes | " | yes (py) |
| SetNode, GetNode | ComfyUI-KJNodes | " | yes (JS virtual nodes, web/js/setgetnodes.js) |
| LTXSequencer, MultiImageLoader *(v3 only)* | WhatDreamsCost-ComfyUI | WhatDreamsCost/WhatDreamsCost-ComfyUI | yes (ltx_sequencer.py, multi_image_loader.py) |

### Custom packs ADDED this session (were MISSING)

| node type | pack | repo | verified |
|---|---|---|---|
| Power Lora Loader (rgthree), Fast Groups Bypasser (rgthree) | rgthree-comfy | rgthree/rgthree-comfy | yes (get_name('Power Lora Loader') → "… (rgthree)") |
| VHS_VideoCombine | ComfyUI-VideoHelperSuite | Kosinkadink/ComfyUI-VideoHelperSuite | yes |
| CM_FloatToInt | ComfyMath | evanspearman/ComfyMath | yes (src/comfymath/convert.py) |
| FluxResolutionNode | ControlAltAI-Nodes | gseth/ControlAltAI-Nodes | yes (flux_resolution_cal_node.py, `"FluxResolutionNode"` mapping) |

**No unmapped node types remain.** Every class_type used by v1/v2/v3 resolves to core or an installed pack.

## 2. Dockerfile diff (`/home/dbrain/dev/comfy-docker/Dockerfile`)

Two changes:

1. **ComfyUI core pinned to a fresh commit** (was `git clone --depth 1` HEAD, cached ~Jun-23). The
   LTX-2.3 AV nodes are core, so a stale core would have been the real gap. Now:
   ```dockerfile
   RUN git clone https://github.com/comfyanonymous/ComfyUI.git && \
       git -C ComfyUI checkout 7f287b705e3daf430c206acb5f0f862dd86c2c7c
   ```
2. **Four packs appended** to the `custom_nodes` RUN block (same `git clone --depth 1` +
   `pip install -r requirements.txt || true` pattern):
   ```dockerfile
       git clone --depth 1 https://github.com/rgthree/rgthree-comfy.git && \
       (pip install -r rgthree-comfy/requirements.txt 2>/dev/null || true) && \
       git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
       (pip install -r ComfyUI-VideoHelperSuite/requirements.txt 2>/dev/null || true) && \
       git clone --depth 1 https://github.com/evanspearman/ComfyMath.git && \
       (pip install -r ComfyMath/requirements.txt 2>/dev/null || true) && \
       git clone --depth 1 https://github.com/gseth/ControlAltAI-Nodes.git && \
       (pip install -r ControlAltAI-Nodes/requirements.txt 2>/dev/null || true)
   ```
   (KJNodes / LTXVideo / WhatDreamsCost / GGUF clones are unchanged; editing the block re-clones all of
   them at HEAD, which is what pulls in KJNodes' LTX2_NAG and WhatDreamsCost's LTXSequencer.)

Rebuild command (CPU/network only — no GPU):
```bash
docker build -t comfyui-ltx:latest /home/dbrain/dev/comfy-docker/
```

## 3. Model symlink / copy map

Symlink targets are all under `/mnt/ssd/models/...` so they resolve inside the container (run_comfy.sh
mounts `/mnt/ssd/models` at the same path). `comfy-models/` is mounted at `/opt/ComfyUI/models`.

### From the background HF download (`/mnt/ssd/models/ltx23-comfy/_dl/`)
As of setup these were **still downloading** — the fp8 checkpoint was ~7.9 GB of ~large on disk. Links
were created anyway and resolve when the files land. **Verify these exist before running.**

| comfy-models path | → target (under _dl) | download state at setup |
|---|---|---|
| checkpoints/ltx-2.3-22b-dev-fp8.safetensors | ltx23fp8/ltx-2.3-22b-dev-fp8.safetensors | IN PROGRESS (.incomplete present) |
| text_encoders/gemma_3_12B_it_fp8_scaled.safetensors | comfyltx2/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors | not yet present |
| loras/ltx-2.3-22b-distilled-lora-384.safetensors | ltx23/ltx-2.3-22b-distilled-lora-384.safetensors | not yet present |
| latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors | ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors | not yet present |

### From models we already own (copied to `/mnt/ssd/models/ltx23-comfy/have/`, then symlinked) — PRESENT
| comfy-models path | → copied file (have/) | size |
|---|---|---|
| loras/ltx-2-19b-ic-lora-detailer.safetensors | ltx-2-19b-ic-lora-detailer.safetensors | 2.6 GB |
| latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors | ltx-2.3-spatial-upscaler-x2-1.1.safetensors | 1.0 GB |
| loras/ltx-2.3-22b-distilled-lora-384-1.1.safetensors | ltx-2.3-22b-distilled-lora-384-1.1.safetensors | 7.6 GB |

Note: the pre-existing `comfy-models` symlinks for `ltx-2.3-22b-distilled-lora-384.safetensors` and
`ltx-2.3-spatial-upscaler-x2-1.0.safetensors` pointed at `/refmodels/comfyref/…` paths that **did not
exist** (broken). They were repointed at the `_dl` downloads per the task. Other pre-existing links
(gemma gguf, audio/video VAE, embeddings-connectors, taeltx2) were left untouched.

Workflow JSONs installed into `comfy-docker/comfy-user-workflows/`: `v1_260318_image-to-video.json`,
`v2_260329_dynamic.json`, `v3_260412_multi-image.json`.

## 4. Launch command (WHEN GPU IS FREE)

```bash
/home/dbrain/dev/comfy-docker/run_comfy.sh
```
This launches the container detached on `:8188`, pinned to the RTX 5060 Ti
(`COMFY_GPU=GPU-bd93e020-…`), mounts `comfy-models` → `/opt/ComfyUI/models`,
`longcat-avatar.cpp/models/ltx2` → `/refmodels:ro`, `/mnt/ssd/models` → `/mnt/ssd/models:ro`, output to
`/mnt/ssd/comfy-out`. Set `COMFY_VRAM=--highvram` (16 GB card) if desired:
```bash
COMFY_VRAM=--highvram /home/dbrain/dev/comfy-docker/run_comfy.sh
```
Web UI: http://10.0.0.208:8188 (or localhost:8188 on the host).

## 5. "When GPU is free" checklist — run v2 (reference)

1. Confirm the GPU is idle (owner's work done) and the downloads finished:
   ```bash
   ls -lL /mnt/ssd/models/ltx23-comfy/_dl/ltx23fp8/ltx-2.3-22b-dev-fp8.safetensors \
          /mnt/ssd/models/ltx23-comfy/_dl/comfyltx2/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors \
          /mnt/ssd/models/ltx23-comfy/_dl/ltx23/ltx-2.3-22b-distilled-lora-384.safetensors \
          /mnt/ssd/models/ltx23-comfy/_dl/ltx23/ltx-2.3-spatial-upscaler-x2-1.0.safetensors
   ```
   All four must exist (no `.incomplete`, non-zero size) or the checkpoint/TE/LoRA loaders will fail.
2. `/home/dbrain/dev/comfy-docker/run_comfy.sh` — wait for "ComfyUI UP on :8188".
3. In the UI: **Workflow → Open** → `comfy-user-workflows/v2_260329_dynamic.json`. Confirm no red
   "missing node" banners (all packs are installed; if any appears, note the node type — it means a pack
   HEAD moved).
4. Set the **scenario-3** prompt (static camera / distant crossing, "★ run first") from
   `PROMPTS.md`, **byte-identical** to the C++ run, and **seed 42** (the ablation's fixed seed). For a
   like-for-like i2v comparison, load the same start still the C++ port used.
5. Render. Output lands in `/mnt/ssd/comfy-out` (and VHS_VideoCombine writes the mp4).
6. Compare against our port's clips on the eye-test page:
   http://10.0.0.208:8077/ltx_denoise/ (ablation.html side-by-side; index.html for input vetting).

## 6. Not verifiable without launching Comfy (flagged)

- Actual graph execution / that every node's inputs are satisfied at runtime (only static presence of
  each node type was verified).
- Model **filename expectations inside the workflow widgets** — the loaders reference names by string;
  the symlink basenames were matched to the task's spec but the JSON widget values should be eyeballed
  once loaded (esp. the fp8 checkpoint + gemma TE names).
- **Node version risks:** KJNodes `LTX2_NAG` has known upstream quirks — issue #535 (may be incompatible
  with the newer "Multimodal Guider" node) and #576 (works on master but mxfp8 can show subtitles). It is
  present and should load; behaviour under this exact graph is unverified. rgthree/VHS/ComfyMath/
  ControlAltAI are pinned only to "HEAD at build time" (not a fixed commit) — pin them for reproducibility
  if this becomes a prod oracle.
- **v3 (multi-image)** depends on WhatDreamsCost `LTXSequencer` + `MultiImageLoader`; both are present at
  HEAD but v3 was not the focus (v2 is the reference).
```
