# PARITY-RECIPE — Comfy LTX-2.3 (scenario-3 t2v) vs our nvfp4 port

Source of truth = **the real working Comfy API graph** `comfy-docker/wf_comfy_s3_t2v.json`
(scenario-3 t2v). This graph is the *collapsed 2-sampler* form of the Denoise-AI ladder:
**one base pass (8-step) + one refine pass (3-step)**, single distill@0.5, **cfg=1, no NAG,
no detailer, no audio-norm**. (The 4-stage split in `REPRODUCE.md` is the un-collapsed v2;
the exported graph proves the collapse is what actually runs — good news, it maps 1:1 onto
our `--hires` ladder.)

Goal: closest nvfp4 (`nvfp4-CLEAN-dev050.gguf`) invocation **without** switching to dev-fp8,
matching comfy quality at **≥ our current 1280×704×193 speed**.

---

## 1. Delta table — Comfy graph vs our r2 "ladder"

Comfy col = parsed verbatim from `wf_comfy_s3_t2v.json` (node id in ⟨⟩).
Ours col = `run_ablation.sh` **row 2** (`s*_r2_ladder`, the ladder config).

| Parameter | Comfy (wf_comfy_s3_t2v.json) | Ours (r2 ladder) | Match? | Quality impact |
|---|---|---|---|---|
| **Base res** | **960×544** ⟨3059⟩ = 0.522 MP | **640×352** = 0.225 MP | ✗ **2.3× fewer px** | **★★★ highest** — distant-face detail is set here |
| **Final res** (after x2) | **1920×1088** = 2.09 MP ⟨4975 x2⟩ | **1280×704** = 0.90 MP | ✗ **2.3× fewer px** | **★★★ highest** — refine can only sharpen the pixels it has |
| Distill strength | dev + **0.5** ⟨4922⟩ | dev050 fold = **0.5** | ✓ **exact** | ★★★ (the thesis) — already matched |
| Upscaler ver | **x2-1.1** ⟨4974⟩ | x2-1.1 | ✓ exact | ★ |
| **Base sigmas** | 8-step `1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0` ⟨4984⟩ | **same** 8-step | ✓ **exact** | ★★ — already matched |
| **Refine sigmas** | 3-step **`0.85,0.725,0.4219,0.0`** ⟨4985⟩ | 2-step **`0.725,0.421875,0.0`** | ✗ | **★★ high** — comfy re-noises higher (0.85) + 1 extra refine step |
| cfg | **1** ⟨4964,4828⟩ | 1.0 | ✓ exact | — (guidance off in both) |
| **Sampler (base)** | **euler_ancestral_cfg_pp** ⟨4831⟩ | plain **euler** | ✗ | ★ — ancestral base-noise is the only real gap (cfg_pp≈inert at cfg1) |
| **Sampler (refine)** | **euler_cfg_pp** ⟨4976⟩ | plain **euler** | ✗ | ★ (cfg_pp near-inert at cfg=1) |
| NAG / detailer | **none** in graph | none (r2) | ✓ | — (don't add; graph proves neither is needed) |
| **VAE decode** | spatial **2×2 tiles, overlap 6**, last_frame_fix=false, **no temporal tiling** ⟨4995⟩ | 1×1 relative + **temporal_tile_frames=4** | ✗ | **★★ high (coherence)** — our temporal tiling risks motion seams; owned by **TILING-FIX.md** |
| **Frames** | **121** ⟨4988⟩ | 193 (FR default) | ✗ | ★ compute — see fps |
| **fps** | **24** ⟨4989⟩ | 48 | ✗ | ★★ compute — 48fps ≈ 1.6× the latent frames for the same ~4–5 s |

**Deltas that plausibly move quality most (in order):**
1. **Pixel budget** (base 0.225→0.522 MP, final 0.90→2.09 MP). Distant/high-motion faces are
   the stated poison; face fidelity is bounded by the pixels the DiT+VAE ever see. This is the
   single biggest lever — and the biggest speed cost (~2.3× pixel work).
2. **Refine sigma schedule** (2-step `0.725→0` vs 3-step `0.85→0`). Comfy re-noises the upscaled
   latent *higher* (0.85 vs 0.725) and gives it an extra step — the refine pass is doing more
   structural correction on the upscale, not just a light polish. Cheap to match (+1 hi-res step).
3. **Decode coherence** (temporal tiling). Comfy decodes spatial-only (2×2, overlap 6). Our
   `temporal_tile_frames=4` can seam on motion — exactly where "mush" is judged. (TILING-FIX.)

Lower impact: sampler variant (cfg_pp is near-inert at cfg=1; ancestral base-noise is a minor
texture difference), fps/frames (a *compute* delta, not directly a per-frame quality delta).

---

## 2. Closest nvfp4 invocation (no dev-fp8)

Turnkey script: **`ltx-denoise-repro/run_parity_nvfp4.sh`** (companion to this doc). It reproduces
the comfy graph on `nvfp4-CLEAN-dev050.gguf` on every axis our engine can express:

```bash
cd ltx-denoise-repro
# TRUE PARITY — comfy res 960x544 -> 1920x1088, 121f@24fps (~2.3x pixels vs our r2)
bash run_parity_nvfp4.sh                 # RES=parity (default)
# SPEED-MATCHED — keep OUR 640x352 -> 1280x704, but comfy's refine schedule + 24fps
RES=speed bash run_parity_nvfp4.sh
```

The core sd-cli call it emits (parity res):

```
sd-cli -M vid_gen --diffusion-model nvfp4-CLEAN-dev050.gguf
  -W 960 -H 544 --video-frames 121 --fps 24              # comfy base res + 121f@24
  --sampling-method euler --steps 8 --cfg-scale 1.0       # base 8-step, cfg1, euler
      (env)  LTX_CUSTOM_SIGMAS=1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0
  --hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 \
    --hires-steps 3 --hires-sigmas 0.85,0.725,0.421875,0.0   # comfy refine (3-step, starts 0.85)
  {{ VAE_DECODE_FLAGS — from TILING-FIX.md; comfy = spatial 2x2 overlap6, NO temporal }}
  --diffusion-fa --offload-to-cpu --mmap --max-vram 11 -s 42
```

**Exact matches:** model (dev050 == dev+distill@0.5), base res 960×544, x2-1.1 upscaler →
1920×1088, base 8-step sigmas, refine 3-step sigmas `0.85,0.725,0.421875,0.0`, cfg=1, 121f@24,
no NAG/detailer.

**Residual deltas our engine can't express (call them out at eval):**
- **Sampler variant** — comfy uses `euler_ancestral_cfg_pp` (base) / `euler_cfg_pp` (refine); we
  have plain `euler`. At **cfg=1 the `cfg_pp` (CFG++) term is near-inert**, so the only real
  difference is the **ancestral base-noise injection** on the base pass. Minor; note it, don't block.
- **VAE decode** — comfy is **spatial 2×2 / overlap 6 / no temporal tiling**. The script ships the
  current temporal-tiling flags as a *placeholder*; **swap in TILING-FIX.md's flags** before trusting
  the decode-coherence comparison (our temporal tiling is itself a suspected mush source).
- Base ⟨3059⟩ cached widget in the JSON is 960×544; the FluxRes-canonical 0.5 MP is ~928×544
  (both /32-clean). We match the graph's literal **960×544**; 928×544 is an equivalent alt.

### The res/speed tradeoff (be explicit)
- **parity** (960×544→1920×1088, 121f): ~2.09 MP final × 121f. Roughly **~1.4–1.5× our current
  r2 compute** (our 1280×704×193 ≈ 0.90 MP × 193). Best chance to match comfy face detail; costs
  speed + VRAM (watch the 11 GB cap at 1920×1088 — may need `--max-vram` headroom / MAXV bump).
- **speed** (640×352→1280×704, 121f): same pixels as our prod but **24fps halves the frames** →
  ~**37% faster** than our 193f baseline, while still gaining comfy's refine schedule + 24fps.
- The insight: **24fps frees ~50% of the temporal budget; spend it on resolution.** Going to comfy
  res *at 24fps* lands near our *current 48fps-low-res* speed — i.e. the big pixel-budget win comes
  close to free once you drop to 24fps (iff 24fps holds motion — that's the thing to eye-test).

---

## 3. Expected wins per delta + suggested A/B order

| Delta to apply | Expected quality win | Speed cost | Notes |
|---|---|---|---|
| Base+final res → comfy (960×544→1920×1088) | **Large** — more px = the distant-face fix | High (~2.3× px) | offset by 24fps below |
| 48→24 fps (193→121 frames) | Neutral→small (motion smoothness only) | **Negative cost (faster)** | the budget that pays for the res bump; also the "50%-less-work" goal |
| Refine 2-step→3-step, start 0.725→0.85 | Moderate — stronger upscale correction | Small (+1 hi-res step) | cheapest real quality lever |
| Decode: drop temporal tiling → spatial 2×2 | Moderate (motion-seam coherence) | ~neutral | from TILING-FIX.md |
| euler → ancestral base-noise | Small (texture) | ~neutral | only if engine gains the sampler |

**Suggested A/B order** (goal = comfy quality at ≥ our speed; change one thing, judge in motion):

1. **FIRST — the combined "free res bump":** `bash run_parity_nvfp4.sh` (parity, 960×544→1920×1088,
   3-step refine, **24fps/121f**) vs our r2 ladder (`s3_t2v_r2_ladder`, 640×352→1280×704, 48fps).
   This tests the top-2 quality deltas (res + refine schedule) with 24fps paying most of the speed
   bill. If 24fps holds motion and this cleans distant faces → **primary win, near speed-neutral.**
2. **If (1) is too slow / OOMs at 1920×1088:** `RES=speed bash run_parity_nvfp4.sh` — keep our
   1280×704 but adopt comfy's **3-step 0.85 refine schedule + 24fps**. Isolates the refine-schedule
   win at strictly-faster-than-prod speed. (Answers "how much of the win is res vs schedule?")
3. **Then decode:** re-run the winner with TILING-FIX.md's spatial-2×2/no-temporal decode flags vs
   the current temporal-tiling flags — isolates motion-seam coherence.
4. **Last (only if engine gains it):** ancestral base-noise sampler vs plain euler. Lowest value.

Do **not** add NAG or the detailer — the actual comfy graph runs neither; they're not part of the
quality it achieves here.
