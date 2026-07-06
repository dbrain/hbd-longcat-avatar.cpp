# FIDELITY-GAP — the remaining "comfy is way prettier" gap (nvfp4 vs ComfyUI dev-fp8)

CPU/analysis only. No GPU, no renders, no builds. Frames extracted with ffmpeg and eyeballed;
weights compared by header + byte hash; code paths read from the worktree.

Scope: the **striping + ghosting** artifacts are already fixed (VAE decode: `LTX_VAE_HEAD_F32` +
`LTX_VAE_TEMPORAL_BLEND`, RESULTS-PARITY.md). This doc is the **deeper fidelity/richness gap** the
owner still sees at matched resolution: *ours = flatter, softer, hazier, lower micro-contrast; comfy
= richer grade, finer grain, deeper contrast.*

Clips (both 1920×1088, 121f, 24fps, h264):
- OURS  `perf_out/ltx_denoise/clips/OURS_parity_nvfp4_1920.mp4`
- COMFY `perf_out/ltx_denoise/clips/COMFY_s3_t2v_2stage.mp4`
(Different seeds/compositions — compared quality characteristics, not layout.)

---

## 1. Gap characterization (frames 40 + 100, both scenes agree)

**Visual (Read'd PNGs):** COMFY is a graded, cinematic overcast look — deep near-black wet asphalt,
bright specular crosswalk reflections (high local dynamic range), saturated brick/red-coat/traffic-
light, crisp facade micro-texture, visible fine grain. OURS is milky and veiled — the whole frame
sits in a mid-grey band, storefront shadows are grey not black, buildings are pale/desaturated,
pavement is smooth with weak reflections and little grain, a soft haze over everything.

**Quantitative (ffmpeg `signalstats`, frame 40):**

| stat | OURS | COMFY | reading |
|---|---|---|---|
| YMIN | 16 | 10 | comfy reaches deeper black |
| **YLOW** (black floor) | **57** | **25** | **ours blacks lifted ~+30 codes** = the haze/flat |
| **YAVG** (mean) | **110** | **72** | ours ~+38 brighter overall (milky) |
| YHIGH | 166 | 135 | ours shifted up toward highlights |
| YMAX | 234 | 212 | — |
| **SATAVG** | **2.35** | **2.60** | **ours less saturated** = the muted grade |

So the gap is a **tonal + texture deficit**, not a composition/artifact issue: lifted black floor,
higher mean, compressed-upward histogram, lower saturation, less high-frequency micro-contrast.
This is a genuine per-pixel render difference (visible in the stills), not merely a container tag —
though note OURS is tagged `color_range=tv / bt470bg` while COMFY is untagged, worth normalizing at
encode time so a player-side range guess isn't compounding it (cheap, orthogonal, verify separately).

---

## 2. Two suspects KILLED by inspection (don't chase them)

- **VAE weights (dev vs distilled) — DEAD.** The VAE bundled in `ltx-2.3-22b-dev-fp8.safetensors`
  (comfy's decoder, node 3940[2] / 4995) is **byte-identical** to our
  `ltx-2.3-22b-distilled_video_vae.safetensors`: all 170 video-VAE tensors match name+shape and
  12/12 sampled decoder tensors are MD5-identical. The "extracted dev VAE" in comfy-models is just a
  **symlink to the distilled file**. LTX-2.3 dev and distilled **share the same video decoder** —
  comfy is not decoding with a richer VAE. **No dev-VAE swap to do.** Corollary: since the decoder is
  identical, *any* tonal difference at decode must come from **our decode-path (F16 activations /
  tiling)** or from the **latent the DiT hands the decoder** — which sharply narrows the search.
- **Refine schedule — already matched.** `run_parity_nvfp4.sh` uses comfy's exact 3-step
  `0.85,0.725,0.421875,0.0` and 8-step base sigmas. Not a gap.

## 3. One recipe claim is STALE (unlocks a cheap test)

PARITY-RECIPE.md says our engine has "plain euler (no cfg_pp / ancestral variants)." **False now.**
`str_to_sample_method` (src/stable-diffusion.cpp:3385) accepts **`euler_a`, `euler_cfg_pp`,
`euler_a_cfg_pp`**, and the samplers are implemented (denoiser.hpp:1917-1919). So comfy's
`euler_ancestral_cfg_pp` (base) is directly expressible as **`euler_a_cfg_pp`**. This is now a free
flag flip, not an "unmatchable delta."

---

## 4. Ranked cause table (for the flat / hazy / desaturated / low-micro-contrast gap)

| # | Delta (ours → comfy) | Likely contribution | Cost to test | How to test on our engine |
|---|---|---|---|---|
| 1 | **`LTX_VAE_DECODE_F16=1` → F32 decode** | **High** for haze/soft/micro-contrast, plausible for the black-lift. VAE is *proven identical* to comfy, so F32 decode is the ONLY path that can bit-match comfy's decode. The F16 cast happens on the **un-normalized latent at decode entry** (ltx_vae.hpp:1176), then per-channel mean/std renorm amplifies any DC rounding → can lift blacks + veil. Comfy decodes F32. | **Free** (remove one env) + VRAM | Remove `-e LTX_VAE_DECODE_F16=1` from PRODENV. F32 doubles the ~14.7 GB decode floor → raise spatial tiling to avoid OOM at 1920. |
| 2 | **nvfp4 4-bit DiT quant → higher precision** | **Med-High.** 4-bit weights lose high-freq / flatten the generated latent vs comfy's **fp8 (8-bit)**. Explains reduced texture + some flatness. | Med (have file, needs a CPU fold) | Use `nvfp4mixed`. `ltx2-distilled-nvfp4mixed-CLEAN.gguf` exists on `/mnt/ssd/models` but is distill@1.0 — negative-fold it to dev050 first (`fold_distill_lora.py --strength -0.5`, CPU), then `DIT=<that>.gguf`. |
| 3 | **Fold approximation: 316 bf16 adaln/modulation layers left at distill@1.0** | **Med.** Our negative-fold de-distilled only the 1344 nvfp4 Linears (COMPAT-REPORT.md). The adaln/timestep **modulation** path — which sets per-block activation **gain (scale/shift)** — is still fully distilled, i.e. *more distilled than comfy's @0.5* exactly on the gain path. Distillation flattens; so on modulation we are strictly flatter/lower-dynamic-range than comfy. A clean mechanistic fit for global flatness/desaturation. | Med (tool change, CPU) | Extend `fold_distill_lora.py` to ALSO fold the 316 bf16 targets (`delta=B@A` added to the bf16 tensor) at `--strength -0.5`, producing a fully-@0.5 model; A/B vs current dev050. |
| 4 | **Sampler `euler` → `euler_a_cfg_pp`** | **Low-Med** — specifically the "finer grain/texture." Ancestral re-injects noise each step; comfy's base is ancestral. cfg_pp≈inert at cfg=1, so the effect is the ancestral base-noise. Does NOT explain the black-lift/desaturation. | **Free** (flag) | `--sampling-method euler_a_cfg_pp` (was thought unsupported — it is). Caveat: one global flag, so refine also becomes ancestral whereas comfy refine=`euler_cfg_pp` (non-ancestral); minor. |
| 5 | dev-fp8 full model | Ceiling reference (folds #2+#3 together) | High (offload tax) | last resort; `ltx-2.3-22b-dev-fp8.safetensors` on box; heavy offload. |
| 6 | Text encoder Q4_K_XL vs gemma fp8_scaled | Low (adherence, not richness) | n/a | note only. |
| — | Color-range tag (`tv/bt470bg` vs untagged) | Possible compounding | Free | normalize encode range in the compare-clip build; verify not double-counted. |

---

## 5. Prioritized test plan (cheapest-high-impact first)

Baseline to beat = current `run_parity_nvfp4.sh`. Change **one thing at a time**, eye-test on
`:8077/ltx_denoise/`, re-shoot the same seed. Because the VAE is proven identical to comfy, tests 1
and (later) 2/3 cleanly separate **decode-path** vs **latent/DiT** causes.

**Test A — drop F16 decode (do this first; free flag, biggest cheap suspect).**
Edit `run_parity_nvfp4.sh` PRODENV (line ~67): **delete `-e LTX_VAE_DECODE_F16=1`**.
(Removal is required — the code enables on `getenv != nullptr`, so `=0` still enables it.)
F32 decode ~doubles the decode activation floor → to avoid OOM at 1920 either de-risk at speed res
or raise spatial tiles:
```bash
# de-risk first at 1280x704 (fast, won't OOM):
RES=speed VWT=8 VHT=4 bash run_parity_nvfp4.sh      # after deleting LTX_VAE_DECODE_F16=1
# then parity res with more spatial tiling for the F32 floor:
RES=parity MAXV=11 VWT=32 VHT=16 bash run_parity_nvfp4.sh
```
Expect: if this closes most of the haze/black-lift, the gap was the decode path — done cheaply.

**Test B — ancestral sampler (free flag; stacks on A's winner).**
In `run_parity_nvfp4.sh` change `--sampling-method euler` → **`--sampling-method euler_a_cfg_pp`**
(line ~91). Targets the "finer grain" specifically. (Single global flag; refine also goes ancestral —
acceptable, base pass dominates.)

**Test C — higher-precision quant (have the file; one CPU fold).**
Fold the mixed base to dev050, then point DIT at it:
```bash
python3 tools/fold_distill_lora.py --strength -0.5 \
  --base /mnt/ssd/models/ltx2-distilled-nvfp4mixed-CLEAN.gguf \
  --out  models/ltx2/nvfp4mixed-CLEAN-dev050.gguf          # CPU, no GPU
DIT=nvfp4mixed-CLEAN-dev050.gguf RES=speed bash run_parity_nvfp4.sh   # isolates quant precision
```
Isolates how much flatness/texture-loss is the 4-bit weight quant.

**Test D — de-distill the modulation path too (tool change; CPU).**
Extend `fold_distill_lora.py` to fold the 316 bf16 adaln/modulation targets at `-0.5` (currently
skipped), producing a *fully*-@0.5 model, and A/B vs dev050. Tests the "we're more-distilled than
comfy on the gain path → flat" hypothesis. Do only if A–C don't fully close the grade gap.

**Test E — dev-fp8 ceiling.** Only if A–D leave a gap; heavy offload, confirms the fp8 target.

**Order rationale:** A and B are free and target the two mechanisms with the tightest fit to the
symptom (decode-path haze/black-lift = A; grain = B). C and D require a CPU fold each and target the
latent-level flatness. E is the expensive ground-truth. Don't add NAG/detailer — comfy's graph runs
neither.

---

## 6. TL;DR (8 lines)

1. Gap is real and tonal, not artifact: OURS black floor YLOW 57 vs comfy 25, mean 110 vs 72, sat
   2.35 vs 2.60 — lifted blacks, milky/brighter, desaturated, softer, less micro-contrast/grain.
2. **VAE is NOT the cause** — the dev-fp8 VAE is byte-identical to our distilled VAE (12/12 decoder
   tensors match; the "dev VAE" file is a symlink). No dev-VAE swap exists to do.
3. Since the decoder is identical to comfy, the tonal gap must be **our F16 decode path** or the
   **latent the DiT produces** (4-bit quant + more-distilled modulation) — a narrow, testable set.
4. Top suspect: **`LTX_VAE_DECODE_F16=1`** — F16 cast on the un-normalized latent + per-channel
   renorm can lift blacks/haze/cut micro-contrast; comfy decodes F32. **Free to test.**
5. Second: **nvfp4 4-bit quant** vs comfy fp8 flattens texture (test nvfp4mixed, on box, one fold).
6. Third: **un-de-distilled adaln/modulation** (316 bf16 layers still at distill@1.0) = we're more
   distilled than comfy exactly on the per-block gain path → flatter (needs a tool tweak to fold).
7. Bonus finding: engine DOES support `euler_a_cfg_pp`/`euler_cfg_pp` (recipe doc is stale) — comfy's
   ancestral base sampler is a free flag flip, targets the "finer grain."
8. **Run first:** delete `-e LTX_VAE_DECODE_F16=1` and re-shoot at `RES=speed` (then parity with
   `VWT=32 VHT=16` for the F32 VRAM floor). Cheapest, highest-fit, cleanly isolates decode vs DiT.
