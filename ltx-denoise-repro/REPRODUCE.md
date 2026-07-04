# Reproduce the "Denoise-AI" LTX-2.3 workflow — definitive spec

Source of truth for the port. Every number below is verbatim-parsed from the three ComfyUI
graph JSONs in `./workflows/` (GetNode/SetNode virtual wires resolved, latent lineage traced).
Where this doc disagrees with `_CONTEXT.md`, **this doc wins** (see "Corrections" below).

---

## 1. Overview

### The 3 mechanisms (what makes faces clean on high-motion footage)
1. **Partial-strength distillation.** Base = the **non-distilled dev model** (`ltx-2.3-22b-dev-fp8`) with
   the distill LoRA applied at **partial strength (~0.48–0.8)**, never 1.0. Full distillation (strength 1.0 =
   our stock `nvfp4-CLEAN`) is what over-sharpens/mushes fast faces; running it "half-distilled" keeps more of
   the dev model's detail while still converging in a handful of steps.
2. **Low-res base → latent x2 spatial upscale → hi-res refine.** Generate the whole clip cheaply at ~0.3–0.5 MP,
   upsample the *latent* x2 with `ltx-2.3-spatial-upscaler-x2`, then run a short refine pass at ~2 MP. The weak
   partial-distillation is allowed to be weak because the structure is already fixed by the base pass; the refine
   pass only sharpens.
3. **NAG negative guidance (+ optional detailer LoRA).** CFG is **1.0 everywhere** (no classifier-free guidance).
   All negative-prompt steering comes from **LTX2_NAG** (normalized attention guidance), and it is wired **only on
   the first preview sub-stage**. A **detailer LoRA** exists but is toggled ON only in v1's final upscale; v2/v3
   ship it OFF.

### Are v1/v2/v3 the same pipeline?
Same **4-stage skeleton** in all three (base-preview → base-main → upscale-preview → upscale-refine, chained by
latent), same models, same NAG. They differ in tuning and in one structural add:

| | v1 `260318` image-to-video | v2 `260329` dynamic | v3 `260412` multi-image |
|---|---|---|---|
| sampler stages | **5** (adds a 5th "Optional refine", not saved) | 4 | 4 |
| conditioning node | `LTXVImgToVideoConditionOnly` (single ref image) | same | **`LTXSequencer` + `MultiImageLoader`** (keyframe multi-image) |
| extra LoRA | — | — | **`ltx2.3-transition.safetensors`@0.85** on S1+S3 |
| detailer LoRA | **ON** (S4@0.7, S5@0.8) | OFF everywhere | OFF everywhere |
| sampler algo | euler (all) | lcm/lcm/lcm/euler | lcm/euler/lcm/euler |
| final decode | `VAEDecodeTiled` | `VAEDecodeTiled` | plain `VAEDecode` |
| base MP (FluxRes) | 0.3 MP | 0.5 MP | 0.5 MP (+2.5 MP loader) |
| upscaler ver | **x2-1.0** | **x2-1.1** | x2-1.1 |

**Verdict:** v1/v2 are iterations of the **same single-image i2v flow**. **v3 is a genuinely different front-end**
— it swaps the single-image conditioner for a multi-keyframe `LTXSequencer` fed by a 2-image `MultiImageLoader`
plus a transition LoRA — but the sampler/upscale/audio backbone is identical to v1/v2.

**Reference for a single-image i2v port → use v2 (`260329_dynamic`).** It is the cleanest, most current
single-image flow: 4 stages (no dead 5th), upgraded x2-1.1 upscaler, base i2v-conditioning strength dialed to 0.7,
detailer dropped (removes the LTX-2 19B compat risk entirely), and it adds `LTX2AudioLatentNormalizingSampling`.
Pull the detailer idea from v1 only if the refine pass looks soft. Ignore v3 unless/until we want multi-keyframe.

---

## 2. Exact recipe (node-for-node)

**Global (identical across v1/v2/v3 unless noted):**
- Checkpoint: `ltx-2.3-22b-dev-fp8.safetensors` (dev / non-distilled). Video-VAE = same file; Audio-VAE via
  `LTXVAudioVAELoader('ltx-2.3-22b-dev-fp8.safetensors')`.
- Text encoder: `LTXAVTextEncoderLoader(gemma_3_12B_it_fp8_scaled.safetensors, ltx-2.3-22b-dev-fp8.safetensors, 'default')`.
- CFG = **1** on every `CFGGuider`. Guidance is via NAG only.
- `LTXVConditioning` frame_rate widget = **24** (all versions), fed a `PrimitiveFloat` "FPS" (see per-version fps).
- Steps of a stage = **len(sigmas) − 1**.
- Latent chain (all versions): **S1 → S2** (S2's `latent_image` = S1 sampler output, same latent, split schedule);
  S2 video latent → **`LTXVLatentUpsampler` (x2)** → **S3 → S4** (S4 continues from S3). Audio latent is carried
  alongside via `LTXVSeparateAVLatent`/`LTXVConcatAVLatent` at every hop.
- `NAG` wraps **only S1**. On S2–S4 the `CFGGuider.model` is the bare Power-Lora-Loader output
  (in v2/v3 wrapped by `LTX2AudioLatentNormalizingSampling`, see §Audio).

### v1 `260318` (all euler; **saved output = S4**; base FluxRes 0.3 MP → ~736×416, x2 → ~1472×832)

| Stage (group) | seed | sampler | sigmas (verbatim) | steps | model = dev-fp8 + LoRAs | guidance | latent in | saved? |
|---|---|---|---|---|---|---|---|---|
| **S1** Sampling 1 (Preview) | 72 | euler | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375` | 5 | distill@**0.5** | **NAG(14, 0.35, 2.5)** | fresh `Concat(i2v@1.0 ⊕ EmptyAudio)` | preview (no) |
| **S2** Sampling 2 (Main) | 72 | euler | `0.725, 0.421875, 0.0` | 2 | distill@**0.65** (detailer OFF) | cfg1 | **= S1 output** | preview (no) |
| — `LTXVLatentUpsampler` x2 on S2 video latent, `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` — |
| **S3** Sampling 3 (Upscale Preview) | 44 | euler | `0.85, 0.7250` | 1 | distill@**0.5** (detailer OFF) | cfg1 | `Concat(i2v@1.0(latent=upsampler) ⊕ S2 audio)` | preview (no) |
| **S4** Sampling 4 (Upscale) | 42 | euler | `0.421875, 0.0` | 1 | distill@**0.8** + detailer@**0.7** | cfg1 | **= S3 output** | **YES (VHS save_output=True, crf19)** |
| **S5** Optional refine | 42 | euler | `0.421875, 0.0` | 1 | distill@**0.8** + detailer@**0.8** | cfg1 | = S4 output | no |

### v2 `260329` dynamic — **RECOMMENDED REFERENCE** (saved output = S4; base FluxRes 0.5 MP → ~928×544, x2 → ~1856×1088)

| Stage | seed | sampler | sigmas (verbatim) | steps | model = dev-fp8 + LoRAs | guidance | latent in | saved? |
|---|---|---|---|---|---|---|---|---|
| **S1** Preview | 127 | **lcm** | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725` | 6 | distill@**0.48** | **NAG(14, 0.35, 2.5)** | fresh `Concat(i2v@**0.7** ⊕ EmptyAudio)` | no |
| **S2** Main | 72 | **lcm** | `0.725, 0.421875, 0.0` | 2 | distill@**0.5** | cfg1 | **= S1 output** | no |
| — `LTXVLatentUpsampler` x2, `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` — |
| **S3** Upscale Preview | 44 | **lcm** | `0.85, 0.7250` | 1 | distill@**0.5** | cfg1 | `Concat(i2v@1.0(latent=upsampler) ⊕ S2 audio)` | no |
| **S4** Upscale | 42 | **euler** | `0.7250, 0.421875, 0.0` | 2 | distill@**0.5** (detailer OFF) | cfg1 | **= S3 output** | **YES (crf14)** |

Detailer LoRA is present in every Power-Lora-Loader but **`on:false`** in v2 → not applied. i2v conditioning
strength on the base stage is **0.7** (v1 was 1.0).

### v3 `260412` multi-image (saved = S4; base 0.5 MP; multi-image prepared at 2.5 MP → cached 1088×1920 in `MultiImageLoader`)

| Stage | seed | sampler | sigmas (verbatim) | steps | model = dev-fp8 + LoRAs | guidance | conditioning |
|---|---|---|---|---|---|---|---|
| **S1** Preview | 164 | lcm | `1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725` | 6 | distill@**0.48** + **transition@0.85** | **NAG(15, 0.5, 2.5)** | `LTXSequencer` (keyframes) |
| **S2** Main | 73 | euler | `0.725, 0.421875, 0.0` | 2 | distill@**0.5** | cfg1 | continues S1 |
| — `LTXVLatentUpsampler` x2, x2-1.1 — |
| **S3** Upscale Preview | 44 | lcm | `0.85, 0.7250` | 1 | distill@**0.4** + **transition@0.85** | cfg1 | `LTXSequencer` |
| **S4** Upscale | 42 | euler | `0.7250, 0.421875, 0.0` | 2 | distill@**0.5** (detailer OFF) | cfg1 | `LTXSequencer` |

v3-only nodes: `MultiImageLoader` (loads 2 imgs: `%view%_00006_.png` + `Edit_00018_.png`), `LTXSequencer`
(keyframe placement: frame indices **0 / 125 / 260 / 270**, strengths **0.7 / 0.7 / 0.7 / 0.5**, `frames` mode,
fps 25), `LTXVCropGuides` (strips the guide/keyframe latents before decode), plain `VAEDecode`, transition LoRA.

### Collapsed "no previews" version (base → upscale → refine, drop inspection taps)
The preview `VHS_VideoCombine` on S1/S3 are inspection taps (`save_output=false`). **You cannot drop the S1/S3
*samplers*** — S2 continues from S1 and S4 continues from S3 (they are two halves of one split-sigma denoise).
What you drop is the S1 and S3 **decode+VideoCombine taps**. The load-bearing compute is all four samplers.
Minimal single-image pipeline (from v2):

```
base gen (dev-fp8 + distill@~0.5):
  pass A: NAG(14,0.35,2.5), sigmas 1.0→0.725 (lcm)          # was S1
  pass B: no NAG,           sigmas 0.725→0.0 (lcm)          # was S2, continues A's latent
latent x2 upscale (ltx-2.3-spatial-upscaler-x2-1.1)
refine (dev-fp8 + distill@~0.5):
  pass A: sigmas 0.85→0.725 (lcm)                           # was S3, re-noise upscaled latent + 1 step
  pass B: sigmas 0.725→0.0 (euler)                          # was S4, continues; SAVED
decode (VAEDecodeTiled 512/64/4096/8) + audio VAE decode → mux
```

---

## 3. Asset manifest

| File | Role | HF repo (from workflow "Model" note) | Size | On box? |
|---|---|---|---|---|
| `ltx-2.3-22b-dev-fp8.safetensors` | base ckpt + video-VAE + audio-VAE | `Lightricks/LTX-2.3-fp8` | ~22–24 GB (fp8) | dev fp4 exists; bf16 distilled on disk |
| `gemma_3_12B_it_fp8_scaled.safetensors` | text encoder | `Comfy-Org/ltx-2` → `split_files/text_encoders` | ~12 GB | we run Q4_K_XL Gemma (TE ≠ bottleneck) |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | partial-distill LoRA (rank-384) | `Lightricks/LTX-2.3-fp8` (loras) *(verify exact path)* | **7.61 GB** | downloading in bg |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` (v1) / **`-1.1`** (v2/v3) | latent x2 spatial upscaler | `Lightricks/LTX-2.3-fp8` (upscalers) *(verify)* | small (~sub-GB) | x2 upscalers on box |
| `ltx-2-19b-ic-lora-detailer.safetensors` | detailer IC-LoRA (v1 only, ON) | Lightricks LTX-2 (19B) *(verify)* | ? | **⚠ LTX-2 19B, base is 22B — dim-compat UNKNOWN** |
| `ltx2.3-transition.safetensors` | multi-image transition LoRA (v3 only) | Lightricks LTX-2.3 *(verify)* | ? | not needed for single-image port |

Exact HF sub-paths for the LoRAs/upscalers/detailer are not spelled out in the JSON (only the two "Model" note
links above). Confirm against `Lightricks/LTX-2.3-fp8` file tree before download.

---

## 4. MINIMIZE / A-B (ranked by quality-value ÷ maintenance-pain)

Ranking = keep the top, ablate the bottom first.

| Rank | Element | Keep/Drop | Why / cost if dropped | How to A/B |
|---|---|---|---|---|
| 1 (core) | **Partial-strength distill** (dev + distill@~0.5) | **KEEP** | This is the whole thesis — full-distill = the mushy-face poison. | fold distill at strength −0.5 onto `nvfp4-CLEAN` vs stock CLEAN, same clip. |
| 2 (core) | **Low-res base → x2 upscale → refine** | **KEEP** | Lets weak distillation converge cheaply + fixes structure before detail. | already have `--hires` + x2 upscaler; A/B is base-only vs base+hires. |
| 3 | **NAG on S1 only** (14, 0.35, 2.5) | keep, but **first thing to skip if NAG is unimplemented** | NAG only steers the *preview* sub-stage (kills captions/subtitles per the memo). Structure/faces come from the model, not NAG. Cost of dropping ≈ more on-screen text / weaker neg-prompt adherence, **not** mushy faces. | run S1 without NAG (plain CFGGuider@1, or small CFG on negative prompt) vs with. Cheap win: ship v1 of the port with **no NAG**, add later. |
| 4 | **LTX2AudioLatentNormalizingSampling** (v2/v3) | **DROP for first port** | All widget values are `1,…` = **identity / no-op** in these files. It's a per-step audio-latent noise scaler hook; at all-1 it does nothing. | leave out entirely; only matters if audio latents blow up. A/B only if audio artifacts appear. |
| 5 | **Detailer LoRA** (`ltx-2-19b-ic-lora-detailer`) | **DROP** | Already OFF in v2/v3. It's **LTX-2 19B vs our 22B base → dim-compat risk**, high pain, marginal value (only v1's final pass). | if refine looks soft, A/B refine pass with detailer@0.7 folded vs without — but only after confirming tensor-dim compat. |
| 6 | **5th "Optional refine" stage** (v1) | **DROP** | Not even saved in v1. Pure extra compute. | n/a — omit. |
| 7 | **lcm vs euler sampler** | tune later | v2 uses lcm on base/upscale-preview, euler on final; v1 all euler. Low pain to switch. | A/B lcm vs euler per stage once the rest works. |
| 8 | **Split-sigma S1+S2 / S3+S4** | can **collapse to single schedule** | The NAG-vs-no-NAG split forces two samplers per phase. If NAG is dropped (rank 3), S1+S2 collapse into **one** sampler with sigmas `1.0→0.0`; likewise S3+S4 into one `0.85→0.0`. Halves the sampler count. | A/B split (`1.0→0.725` then `0.725→0`) vs merged (`1.0→0`) — should be ~identical without NAG. |
| 9 | **Transition LoRA / LTXSequencer / MultiImage** (v3) | **DROP** for single-image | Only needed for multi-keyframe mode. | out of scope for i2v port. |

**Suggested minimal first port** (drops ranks 3–9): dev + distill@0.5, single base schedule `1.0→0.0`,
x2 latent upscale, single refine schedule `0.85→0.0` at distill@0.5, CFG1, no NAG, no detailer, no audio-norm.
Then add NAG back (rank 3) as the one quality lever most likely to matter for the "no on-screen text" goal.

---

## 5. Mapping to our stack

| Workflow element | Our equivalent | Status |
|---|---|---|
| `dev-fp8 + distill-lora@s` | **Fold distill LoRA at strength `(s−1.0)` onto `nvfp4-CLEAN.gguf`** (CLEAN == distilled@1.0). e.g. distill@0.5 → fold @ −0.5. Generalize `tools/fold_lipdub_lora.py` to take a strength arg. | **needs code** (strength-parameterized fold) |
| partial distill, alternative | Runtime LoRA `<lora:distill:−0.5>` on `nvfp4-CLEAN` if runtime negative-strength LoRA works | have runtime LoRA; **negative strength = verify** |
| low-res base → x2 upscale → refine | `--hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires-steps --hires-sigmas` (see `run_ltx_upscale_ab.sh`) | **have** |
| base custom sigma schedule | need a base-pass `--sigmas` equivalent (only `--hires-sigmas` exists today) | **needs code** |
| per-stage LoRA strength (base vs refine differ) | today `--hires` reuses same model+LoRA; needs per-phase LoRA/strength | **needs code** (or fold two variants) |
| `LTX2_NAG(scale, alpha, tau)` | not implemented (fork has CFG negative-prompt only) | **needs code** — lowest priority (S1-only); ship without first |
| `LTXVLatentUpsampler` + `ltx-2.3-spatial-upscaler-x2-1.1` | spatial+temporal x2 upscalers on box | **have** |
| `LTX2AudioLatentNormalizingSampling` | no-op at all-1 widgets | **skip** |
| `VAEDecodeTiled 512/64/4096/8` | our tiled VAE decode | **have** (map tile/overlap params) |
| native AV (`ConcatAVLatent`/`SeparateAVLatent` + audio VAE) | our LTX native-audio path | **have** |
| `FluxResolutionNode` (MP+ratio→dims/32) | pick res manually (~928×544 base, ~1856×1088 refine for 0.5 MP) | trivial |
| detailer / transition LoRA, `LTXSequencer`, `MultiImageLoader` | — | **out of scope** for single-image i2v |

---

## Appendix — corrections to `_CONTEXT.md` first-pass table

- **Stage count:** v1 has **5** sampler stages (S1 preview, S2 main, S3 upscale-preview, S4 upscale=SAVED,
  S5 optional-refine); v2/v3 have 4. The `_CONTEXT` 4-row table conflated S3/S4.
- **S3 sigmas** are `0.85, 0.7250` (a re-noise+1-step on the upscaled latent), **not** `0.421875, 0.0`. The
  `0.421875, 0.0` schedule belongs to **S4** (v1). `_CONTEXT` mis-assigned it.
- **S2 latent** does **not** come from a fresh `ConcatAVLatent` — it **continues from the S1 sampler output**
  (split-sigma). Only S1 starts fresh.
- **distill LoRA filename** = `ltx-2.3-22b-distilled-lora-384.safetensors` (no `-1.1` suffix in any JSON).
- **Upscaler file** = `ltx-2.3-spatial-upscaler-x2-**1.0**` in v1, `-**1.1**` in v2/v3 (`_CONTEXT` said 1.1 for all).
- **Distill strengths differ per version**: v1 = 0.5/0.65/0.5/0.8; v2 = 0.48/0.5/0.5/0.5; v3 = 0.48/0.5/0.4/0.5.
  The `_CONTEXT` values (0.5/0.65/0.8/0.8) match **v1 only**.
- **Detailer** is ON only in v1 (S4@0.7, S5@0.8); **OFF in all of v2 and v3**.
- **CFG=1 / NAG-only / NAG-on-S1-only / euler**: confirmed for v1. But **v2/v3 samplers are lcm on most stages**
  (v2: lcm/lcm/lcm/euler; v3: lcm/euler/lcm/euler), not all-euler.
- **NAG params vary**: v1 & v2 = (scale 14, alpha 0.35, tau 2.5); **v3 = (scale 15, alpha 0.5, tau 2.5)**.
  Memo guidance: "nag scale 13–15, nag alpha 0.4–0.5" to remove unwanted content (esp. subtitles/text).
- **Resolution/frames are link-driven, not the cached widget values.** `EmptyLTXVLatentVideo` shows a stale
  `544,960,121`; real dims come from `FluxResolutionNode` (v1 0.3 MP→~736×416; v2/v3 0.5 MP→~928×544; x2 refine
  ~1472×832 / ~1856×1088) and real frame count from `SimpleCalculatorKJ` (v1 `1+8*(round(a*b)/8)` @24fps×10s = **241**;
  v2 same @25×5 = **126**; v3 `1+8*round((a*b-1)/8)` @25×5 = **129**). fps: v1=24, v2=v3=25. `LTXVConditioning`
  frame_rate widget is 24 in all.
- **v3 is a different front-end**, not just a tweak: `LTXSequencer` + `MultiImageLoader` + transition LoRA replace
  the single-image `LTXVImgToVideoConditionOnly` conditioning path.
- **Base i2v conditioning strength**: v1 = 1.0; **v2 = 0.7** (`LTXVImgToVideoConditionOnly` first widget).
