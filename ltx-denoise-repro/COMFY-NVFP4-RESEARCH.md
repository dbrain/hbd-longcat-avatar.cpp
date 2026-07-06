# COMFY-NVFP4-RESEARCH — why comfy's dev-nvfp4 looks "almost worse than ours"

Read-only web + JSON research. No renders/builds. Scope: understand where comfy's nvfp4 checkpoint
comes from, whether the comfy workflow loads/uses it *correctly*, and why the owner's eye-test read
comfy-nvfp4 as looking **no better (or worse)** than our own C++/ggml nvfp4.

Distinction vs `FIDELITY-GAP.md`: that doc compared **OUR-nvfp4 vs COMFY-fp8**. This doc is the
**COMFY-nvfp4 vs OUR-nvfp4** question the owner just raised (both 4-bit; the shared-quant axis).

---

## TL;DR ranked answers

1. **The file is OFFICIAL Lightricks, not a community repack.** `ltx-2.3-22b-dev-nvfp4.safetensors`
   (21.7 GB) lives in the official `Lightricks/LTX-2.3-nvfp4` HF repo, trained by **Quantization-Aware
   Distillation (QAD)**. There was an *earlier* nvfp4 upload that users called visibly bad; an
   **updated QAD re-upload "mostly fixed" it**. **Action: verify the owner's local file is the current
   re-upload** (check HF commit date / size 21.7 GB), because a stale first-cut nvfp4 alone would
   explain "worse than ours."
2. **The comfy workflow uses the STANDARD community recipe** (UNETLoader/"Load Diffusion Model" for the
   nvfp4 DiT + `LoraLoaderModelOnly` distill-LoRA @0.5 applied in bf16 at runtime, CFG=1, distilled
   step counts). Loader wiring is *correct*; this is not a misconfigured graph.
3. **DIRECT HIT on the symptom:** the official nvfp4 repo's own discussion thread ("Weird Pattern
   Shift") reports a **"cloudy/smokey shift" + quality loss on nvfp4**, and a technical user pins the
   cause to **"the LoRA injection"** — a noise-resolution problem when a runtime LoRA is applied at
   **low step counts**. Fixes offered: **raise steps to 9+ on all stages**, add targeted noise after
   upscale, ramp LoRA block strengths. The owner's comfy graph runs only **8 base + 3 refine steps
   with the distill-LoRA @0.5** — *exactly* the low-step + runtime-LoRA-injection condition the
   community says produces the cloudy/smoky look. This is the leading explanation for "comfy nvfp4
   looks almost worse than ours."
4. **Fold-then-quantize (ours) vs quantize-then-runtime-LoRA (comfy) is expected to differ visibly** —
   and the community evidence points the *worse* way for comfy: the cloudy artifact is reported
   specifically on the runtime-injection path at low steps, which our baked-in fold does not have.

---

## 1. Provenance of `ltx-2.3-22b-dev-nvfp4.safetensors` (official? correct load?)

- **Official.** File is at `https://huggingface.co/Lightricks/LTX-2.3-nvfp4/blob/main/ltx-2.3-22b-dev-nvfp4.safetensors`
  (21.7 GB), a first-party Lightricks repo. The BF16 source is `Lightricks/LTX-2.3`, fp8 is
  `Lightricks/LTX-2.3-fp8`. The nvfp4 card states it is **"trained by Quantization Aware Distillation
  for improved accuracy"** and recommends training on the bf16 model. Not unsloth/drbaph/Hippotes —
  community repacks exist on CivitAI but the owner's filename matches the official one.
  Sources: HF repo above; `Lightricks/LTX-2.3-nvfp4` card; HF discussion `Lightricks/LTX-2.3/discussions/1`
  ("fp8??nvfp4 where?", closed when the official nvfp4 shipped).
- **Blackwell-only, needs cu130.** nvfp4 is the "official Blackwell / RTX 50xx path" with native FP4
  GEMM; it **falls back to slow paths on pre-Blackwell cards.** ComfyUI only dispatches native NVFP4
  GEMMs (CUDA 13 CUTLASS, 5th-gen FP4 tensor cores) when **PyTorch is built with CUDA 13.0 (cu130)**;
  nvfp4 loading "was stabilized in releases after early 2026."
  Sources: `blog.comfy.org/p/new-comfyui-optimizations-for-nvidia`; `docs.comfy.org/changelog`;
  `ltxworkflow.com/models`.
- **"Correct" load = built-in LTXVideo nodes.** The nvfp4 card points to the built-in ComfyUI
  LTXVideo nodes rather than any special third-party FP4 loader. There is **no separate companion
  scale file** — the per-block FP8 scales + global scale are packed inside the single safetensors;
  ComfyUI's native nvfp4 support reads them at load. (Note: the third-party `ComfyUI_bnb_nf4_fp4_Loaders`
  / `comfyui_nf4_loader` nodes that show up in searches are for **bitsandbytes NF4/FP4**, a *different*
  format — do NOT use those for this NVIDIA-NVFP4 file.)
  Sources: `Lightricks/LTX-2.3-nvfp4` card; `github.com/silveroxides/ComfyUI_bnb_nf4_fp4_Loaders`.

## 2. Is the comfy graph loading/using it the standard way? (JSON-confirmed)

Node graph of `wf_comfy_s3_t2v_nvfp4.json` (inspected):

- **`5001` UNETLoader** → `ltx-2.3-22b-dev-nvfp4.safetensors`, `weight_dtype:"default"` = the DiT.
- **`4922` LoraLoaderModelOnly** → model `[5001,0]`, `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`,
  `strength_model:0.5` = LoRA applied **at runtime, in bf16, on top of the nvfp4 base** (confirms the
  owner's stated "key difference").
- **`3940` CheckpointLoaderSimple** → `ltx-2.3-22b-dev-fp8.safetensors` — used **only for the VAE**
  (`[3940,2]`, feeds every decode: `3159`, `4970`, `4975`, `4995`) and the text-encoder/audio-VAE
  loaders (`4982`, `4010`). The nvfp4 DiT does *not* touch it. So VAE/text-encoder are identical
  between the two workflows; the **only** delta between `_nvfp4.json` and `_t2v.json` is the DiT source
  (node `5001` UNETLoader-nvfp4 → LoRA vs `[3940,0]` fp8-checkpoint-model → LoRA). Clean A/B.
- **Two-stage sampler (both workflows identical):**
  - Base: `4829` SamplerCustomAdvanced, `euler_ancestral_cfg_pp`, sigmas `4984` =
    `1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0` → **8 steps**, CFG=1 (`4828`),
    960×544, seed 42.
  - Upscale: `4975` LTXVLatentUpsampler (x2, `ltx-2.3-spatial-upscaler-x2-1.1`) → `4970` cond-only
    strength 1 → refine `4971` SamplerCustomAdvanced, `euler_cfg_pp`, sigmas `4985` =
    `0.85,0.725,0.4219,0.0` → **3 steps**, CFG=1. Tiled VAE decode `4995` (2×2, overlap 6).

**Verdict:** this **is** the standard community recipe (native-loader nvfp4 DiT + runtime bf16 distill
LoRA + distilled low-step CFG=1 + x2 latent-upscale refine). The graph is not misbuilt. Loader is the
right one for an NVIDIA-NVFP4 file. So the "worse" look is **not** a wrong-loader / missing-scale-file
bug — it's inherent to *this recipe on nvfp4* (see §3–4).

Confirming that this loader pairing is the norm: `LoraLoaderModelOnly` on the dev model is the
documented way to convert dev inference (many steps, CFG>1) into distilled 8-step CFG=1; nvfp4 goes in
`diffusion_models` and is loaded via "Load Diffusion Model"/UNETLoader, while the fp8 *checkpoint* uses
CheckpointLoaderSimple. Sources: search over `ltxworkflow.com/guide`, `github.com/wildminder/awesome-ltx2`,
`runcomfy.com/comfyui-workflows/ltx-2-3-ic-lora`.

## 3. Known comfy-nvfp4 quality pitfalls (what makes it look worse than it should)

- **THE big one — runtime-LoRA-injection cloudiness at low steps.** Official nvfp4 repo discussion
  `Lightricks/LTX-2.3-nvfp4/discussions/1` ("Weird Pattern Shift"): users report a **"cloudy/smokey
  shift"** and *"the speed gain is great but the quality loss is not worth it."* A technical commenter:
  *"this issue, with the cloudy/smokey shift, is **from the lora injection**"* — the injected LoRA
  perturbs noise that **doesn't fully resolve at low step counts**. Recommended fixes: **steps ≥9 on
  all stages**, add targeted noise to latents post-upscale, **ramp LoRA block strengths**. The owner's
  graph uses **8 base + 3 refine** steps with LoRA @0.5 — squarely in the failure regime. **This is the
  single most likely reason comfy-nvfp4 underperforms.** (Same thread: an updated nvfp4 upload "mostly
  fixed the image quality issue," reinforcing the §1 "verify you have the current file" action.)
- **nvfp4 is genuinely a step below fp8 in texture/edge fidelity.** Independent nvfp4-vs-fp8/fp16
  writeups consistently report on nvfp4: *fuzzier in-frame text, "mushy" edges on complex textures
  (foliage/patterns), softer high-contrast edges under fast motion, occasional warm color skew,* plus
  *micro-flicker / a temporal "pulse" every ~12–16 frames.* Community rule-of-thumb: **fp8/fp16 for
  deliverables, nvfp4 for ideation only.** (This axis is *shared* with our nvfp4 since both are 4-bit,
  so it does not by itself explain "comfy worse than ours" — but it does confirm nvfp4's softening.)
  Sources: `crepal.ai/.../blog-nvfp8-vs-nvfp4-ltx-2-comfyui`; `wavespeed.ai/blog/posts/blog-ltx-2-nvfp4-vs-nvfp8`;
  `nemovideo.com/blog/ltx-2-fp4-vs-fp8-rtx-guide` (blocked, title-level only).
- **Native-nvfp4 not actually engaged → silent upcast/mis-scale (checkable).** ComfyUI issue
  **#11864** ("Native NVFP4 (Blackwell) Loading Failure on RTX 5090 – Wan 2.2/Flux2Dev/LTX2"): with
  `weight_dtype:"default"`, "Load Diffusion Model" **upcast a `*_nvfp4_mixed` checkpoint to fp16
  (~28 GB)** or, on fp8 setting, to `float8_e4m3fn` (~14 GB) — *"model weight dtype
  torch.float8_e4m3fn, manual cast: torch.float16"* — instead of running native FP4.
  - *For VRAM/speed this is a regression.* **For numerical quality, dequantizing fp4→fp16 and computing
    in fp16 is actually the *highest-fidelity* way to run the 4-bit weights** — so an upcast, by
    itself, would make comfy look *better*, not worse. (analysis, not a claim from the issue)
  - *The real quality risk (label: possibility, needs a log check):* if the running comfy build /
    non-cu130 torch **doesn't recognize the nvfp4 scale tensors at all** and mis-dequantizes, output
    degrades. **Check the ComfyUI console at load** for an nvfp4/fp4 recognition line vs a bare
    fp16/fp8 cast, and confirm torch is **cu130** on the 50-series box. Source: issue #11864.
- **Upscaler/tiling and sigma schedule are NOT suspects here.** The x2 spatial upscaler, tiled decode,
  sampler and sigmas are byte-identical between the fp8 and nvfp4 workflows (JSON-confirmed), so they
  cannot account for an nvfp4-vs-our-nvfp4 delta.

## 4. Fold-then-quantize (ours) vs quantize-then-runtime-LoRA (comfy): expected to differ?

Yes — and the evidence points to comfy's path being the *worse* one here, which matches the owner:

- **Comfy (quantize base → bf16 LoRA at runtime):** LoRA delta stays exact (bf16), so *on paper* comfy
  is more faithful to the intended distill. **But** the community "cloudy/smokey shift" is reported
  *specifically* on this runtime-injection path, where the high-precision LoRA riding a 4-bit base
  leaves residual noise that **fails to resolve in the 8+3 low-step schedule** (§3). Net: exact LoRA,
  but a low-step injection artifact.
- **Ours (fold LoRA@0.5 into bf16 → quantize whole thing to nvfp4):** the LoRA delta gets 4-bit'd too
  (theoretically lossier LoRA), **but there is no runtime injection and no base-vs-LoRA precision
  mismatch** — the sampler sees a single self-consistent quantized operator, so the low-step
  noise-resolution problem does not arise the same way. This is a **plausible mechanistic reason our
  nvfp4 can look cleaner/less cloudy than comfy's nvfp4** at matched low steps.
- **Label:** this is community-report + mechanism reasoning, **not** a controlled A/B. To prove it,
  the cheap comfy-side test the community itself prescribes is **raise comfy to ≥9 steps per stage**
  (and/or ramp/lower LoRA strength); if comfy's cloudiness clears, the artifact was the
  runtime-LoRA-injection-at-low-steps, i.e. an artifact of *comfy's recipe*, not of nvfp4 or of our
  fold. (Also worth A/B on our side: our theoretical LoRA-quant loss predicts we should be slightly
  *flatter*, per FIDELITY-GAP §4 #3 — consistent, different symptom.)

---

## Concrete "why comfy-nvfp4 underperforms" checklist (most→least likely)

1. **Low-step + runtime-LoRA-injection cloudiness** (8 base + 3 refine, LoRA @0.5) — the exact
   community-reported "cloudy/smokey shift." Test: bump comfy to **≥9 steps all stages**. [HF nvfp4 disc #1]
2. **Stale first-cut nvfp4 file** — verify the local safetensors is the **updated QAD re-upload**
   (HF commit date / 21.7 GB). [HF nvfp4 disc #1 / repo]
3. **Native nvfp4 not actually engaged** (non-cu130 torch / old comfy) → possible mis-dequant of
   scales. Check load-time console + torch CUDA build. [ComfyUI #11864, blog.comfy.org]
4. **Intrinsic nvfp4 softening** (mushy edges/text, warm skew, ~12–16f pulse) — shared with our nvfp4,
   so not the differentiator, but sets the low ceiling. [crepal / wavespeed]

## Sources
- https://huggingface.co/Lightricks/LTX-2.3-nvfp4 (+ /blob/main/ltx-2.3-22b-dev-nvfp4.safetensors)
- https://huggingface.co/Lightricks/LTX-2.3-nvfp4/discussions/1  ("Weird Pattern Shift" — cloudy/smokey = LoRA injection at low steps)
- https://huggingface.co/Lightricks/LTX-2.3/discussions/1  ("fp8??nvfp4 where?")
- https://huggingface.co/Lightricks/LTX-2.3 · https://huggingface.co/Lightricks/LTX-2.3-fp8
- https://github.com/Comfy-Org/ComfyUI/issues/11864  (nvfp4 default → fp16/fp8 upcast, no native FP4)
- https://blog.comfy.org/p/new-comfyui-optimizations-for-nvidia · https://docs.comfy.org/changelog · https://docs.comfy.org/tutorials/video/ltx/ltx-2-3
- https://crepal.ai/blog/aivideo/blog-nvfp8-vs-nvfp4-ltx-2-comfyui/ · https://wavespeed.ai/blog/posts/blog-ltx-2-nvfp4-vs-nvfp8/ · https://www.nemovideo.com/blog/ltx-2-fp4-vs-fp8-rtx-guide
- https://ltxworkflow.com/models · https://ltxworkflow.com/models/ltx23-dev · https://github.com/wildminder/awesome-ltx2 · https://github.com/silveroxides/ComfyUI_bnb_nf4_fp4_Loaders (bnb NF4/FP4 — NOT this file)
