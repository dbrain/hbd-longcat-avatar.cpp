# LTX-2.3 video render — VRAM + performance flow analysis

Analysis-only (read from code + existing render logs; **no GPU renders run**). Card = RTX 5060 Ti,
**15888 MiB** usable. Flow driver = `run_parity_nvfp4.sh`. Evidence = `_ablation_out/{verify_1920,
sweep_*_s3}/log`, `/tmp/vram_*` per-second samples, `perf_out/ltx_denoise/{sweep,fid}_manifest.tsv`.

---

## TL;DR (the answer to the owner's question)

**The ~15.6–15.8 GB peak is NOT co-resident models. It is the VAE decode stage in isolation.**
By the time the video VAE decodes, the DiT and gemma are already gone from VRAM. The peak is a single
buffer: the **VAE decode compute graph = 9819 MB** (the conv3d/im2col working set for the *final*
1920×1088 latent), plus ~1.4 GB of VAE params + the decoded RGB accumulator. It is set almost entirely
by the **final decode resolution** — 1920 vs 1280 is the dominant lever; frame count (97 vs 121) barely
moves the peak; MAXV/upscaler/gemma do not touch it.

The models are loaded/freed **sequentially**, so peak = max(single stage), not sum:

| Stage | What's resident (VRAM) | Plateau / peak (MiB) | Wall @1920×121 |
|---|---|---|---|
| Text encode (gemma-3-12b Q4_K_XL) | gemma streamed + 2.16 GB compute + 2.2 GB text-proj | ~4.7 GB spike | 8.9 s |
| **Base DiT sample** 960×544, 8 steps | DiT resident 5471 + compute 903 + cache 64 | **~7.3 GB** | 115 s |
| Latent spatial upscale ×2 (load upsampler 949) | upsampler 949 + compute 1216 (DiT still resident) | ~9–10 GB | 3 s |
| **Hires refine DiT** 1920×1088, 3 steps | DiT resident 5471 + compute **3389** + cache 255 | **~12.7 GB** | 141 s |
| Audio VAE decode | audio VAE 353 + compute 311 | ~0.7 GB (isolated) | 3.5 s |
| **Video VAE decode** 1920×1088×121 | VAE params 1385 + compute **9819** + RGB out ~3000 | **~15.7 GB ← TRUE PEAK** | 148 s |

Total ~422 s (`verify_1920`); sweep manifest shows 444–481 s incl. GPU-contention waits. All nvfp4/imatrix
1920 renders peak **15758 MiB**; q4km 15626. Only **~130 MiB headroom** at 1920 — a genuine OOM cliff.

---

## 1. The peak breakdown — co-resident vs sequential

**Sequential. Confirmed from `verify_1920/log` + `/tmp/vram_imatrix_s3` (per-second trace).**

The trace has three clean plateaus and then the peak, in this order:
`7294 → (refine) 12706 → (decode) 14074 with a 15758 spike → collapse to ~2000`.

At the **15758 MiB peak (VAE decode)**, resident is:
- **VAE decode compute buffer = 9819.47 MB** — logged 13× (one per temporal tile), `ltx_video_vae compute buffer size`. This is the whole-spatial conv3d/im2col working set for a TBF=3-frame tile of the *upscaled* 1920×1088 latent. **This one buffer is the peak.**
- **Video VAE params ≈ 1385 MB** (`ltx_video_vae params ... 1385.02 MB`), streamed resident for decode.
- **Decoded RGB accumulator ≈ 1.5–3 GB** (1920×1088×121×3; f16 compute with DF16, f32 assembled output).
- CUDA VMM pool reserve / fragmentation for the remainder.

**What is NOT resident at the peak (the important part):**
- **DiT — freed.** After the hires refine `sample()` the code calls
  `diffusion_model->free_params_buffer()` (`stable-diffusion.cpp:8973`, gated `free_params_immediately`),
  releasing the 5471 MB resident set *before* decode. The trace confirms it: the refine plateau (12706,
  DiT-resident) drops before the decode plateau climbs.
- **gemma-3-12b (12B TE) — never resident, and freed.** Under `--offload-to-cpu` gemma's weights live in
  RAM and are streamed per-segment via the lap-34 prefetch thread (VRAM cost = only the transient
  384/639/2160 MB compute buffers during the 8.9 s encode). `cond_stage_model->free_params_buffer()`
  runs after encode (`:7725/:7735`). Zero gemma VRAM by the time the DiT — let alone the VAE — runs.
- **Latent upsampler (949 MB) — freed.** It is a function-local runner in
  `upscale_ltx_spatial_video_latent()` (`:7991`); it is loaded (3 s), run, and destructed on return,
  before the refine and long before decode.
- **Audio VAE — decoded earlier and separately** (353 MB params + 311 MB compute, its own ~0.7 GB blip
  at 3.5 s), not co-resident with the video-decode peak.

So: **the upscaler + DiT + VAE + text-encoder are *never* all resident together.** The peak is a single
VAE-decode buffer whose size is a function of the **final (post-upscale) resolution**, not of model
co-residency. This is why 1280 vs 1920 is the whole ballgame.

---

## 2. Optimization levers, ranked (VRAM saved / perf cost)

Ranked by impact on the **true peak** (= VAE decode) and on wall time.

**1. Final decode resolution — the dominant peak lever.**
Peak ∝ decode-res pixels. 1280×704 = **0.43×** the pixels of 1920×1088, so the 9819 MB buffer drops to
~4–5 GB and the *decode ceases to be the peak* — the ~9–10 GB **sampling** plateau becomes the binding
constraint instead. 1280×704 also cuts the two biggest wall items (refine 141 s and decode 148 s) by
>50%: the fid manifest shows the **full 1280×704×121 pipeline = 135 s** vs 1920's ~444 s. This is by far
the best VRAM *and* perf lever.

**2. VAE decode tiling (TBF / VWT / VHT / DECODE_F16) — tune the 9819 MB buffer directly.**
Confirmed in `ltx_vae.hpp:1684` (temporal-blend) — the decode runs independent temporal tiles of
`LTX_VAE_TBLEND_FRAMES` (TBF) latent frames (bounds the feature-map floor), whole-spatial, with the
per-tile conv3d **im2col tiled `LTX_VAE_CONV3D_WTILES × _HTILES`** (VWT×VHT). So the buffer scales as
`decode_pixels × TBF / (VWT×VHT) × (f16?½:1)`.
- `TBF↓` (3→2) and `VWT/VHT↑` (16×8 → 20×10) shrink the buffer, buying the missing headroom at 1920, at
  a modest speed cost (more tiles / more conv invocations). Reclaims the 130 MB cliff.
- `LTX_VAE_DECODE_F16=1` (already on) halves the feature-map floor — keep it on.
- **Cost:** pure decode speed; no quality change (temporal-blend is seam-free by `TBLEND_OVERLAP`).

**3. MAXV (DiT offload depth) — a *free* speed lever at 1920, a peak lever at 1280.**
MAXV sets the DiT resident/segment-merge budget; higher = fewer segments = less offload = faster sampling.
- **At 1920** the sampling plateau (7–12.7 GB) sits *below* the 15.7 GB decode peak, so raising MAXV 7→11
  speeds the 256 s of base+refine sampling **without raising the true peak** (it hides under decode).
  Current sweeps at MAXV≈7 leave DiT-resident speed on the table for free — see §4.
- **At 1280** decode is cheap, so the sampling plateau *is* the peak → MAXV now trades VRAM for speed
  directly (MAXV=7 ≈ 7.5 GB peak, MAXV=9 ≈ 9.5 GB). Plenty of headroom either way on 16 GB.

**4. Refiner (hires) steps — the biggest tunable time sink after decode.**
The 3-step refine at full 1920 res = **141 s = 33 % of wall** (per-step it's *more* expensive than the
8-step 960 base, because it runs at 4× the tokens). Dropping 3→2 refine steps saves ~47 s. It does **not**
change the peak (peak is decode). Cost = some hires sharpness; A/B it. `NOHIRES=1` skips upscale+refine
entirely but then output is only base res (960×544) — not the product.

**5. Frame count 97 vs 121 — perf yes, peak barely.**
Decode is temporally tiled (TBF=3 latent frames/tile), so the **per-tile 9819 MB buffer is fixed
regardless of total frames** — 97f only shrinks the RGB accumulator (~0.4 GB f32 / ~0.2 GB f16). But 97/121
= **0.80×** fewer tiles and fewer sampling tokens → ~20 % less wall across base+refine+decode (422 s →
~340 s @1920; 135 s → ~108 s @1280). Use 97 for the product: free ~20 % speed, trivial VRAM.

**6. gemma free — already optimal.** Streamed + freed after encode; nothing to reclaim.

**7. The 1.5× "larger base" upscaler — NOT available, and would not lower the peak (see §3c).**

---

## 3. Three recommended recipes

All use `--offload-to-cpu --mmap`, `DF16=1`, `FR=97` (the 4 s product), gemma Q4_K_XL, DiT
`nvfp4-CLEAN-dev050.gguf`. Env is what you pass to `run_parity_nvfp4.sh`.

### (a) 1280×704 FAST — the primary/product path ✅ recommended default
```bash
RES=speed FR=97 MAXV=9 TBF=4 VWT=4 VHT=2 DF16=1 SAMP=euler \
  bash run_parity_nvfp4.sh        # base 640×352 → ×2 → 1280×704
```
- **Predicted peak ≈ 9–10 GB** (sampling-bound; decode only ~5–6 GB). ~6 GB of headroom — safe.
- **Predicted wall ≈ 105–110 s** @97f (fid baseline = 135 s @121f × 0.80). ~4× faster than 1920.
- VWT/VHT can be coarse (4×2) here because decode is small — that's the fid baseline and it's fast.

### (b) 1920×1088 QUALITY — the higher-fidelity option
```bash
RES=parity FR=97 MAXV=11 TBF=2 VWT=20 VHT=10 DF16=1 SAMP=euler \
  bash run_parity_nvfp4.sh        # base 960×544 → ×2 → 1920×1088
```
- **Predicted peak ≈ 14.5–15 GB.** vs the current 15.76 GB / 130 MB-cliff: `TBF=2` + `VWT=20/VHT=10`
  shrink the 9819 MB decode buffer enough to restore ~1 GB of headroom (the current TBF=3/16×8 config is
  the OOM cliff — see §4). Keep an eye via the nvidia-smi sampler on the first run.
- **MAXV=11** is free here (sampling plateau ≤ ~13 GB stays under the decode peak) and speeds the 256 s of
  sampling.
- **Predicted wall ≈ 330–350 s** @97f (422 s @121 × 0.80), or ~300 s if you also drop refine 3→2.

### (c) "Larger-base / lower-peak middle" — use RES=mid, **not** a 1.5× upscaler
There is **no 1.5× spatial upscaler on disk** — `models/ltx2/latent_upscale_models/` holds only
`ltx-2.3-spatial-upscaler-x2-1.1` (×2) and `ltx-2.3-temporal-upscaler-x2-1.0` (×2, temporal). The hires
path is hard-wired ×2 (`hires: scale: 2`), and the code **ignores** `hires.target_width/height` with a
`LTX latent spatial upsampler output is …; ignoring hires target` warning. So the 1.5× variant is not
runnable today. The honest middle recipe is the existing `RES=mid`:
```bash
RES=mid FR=97 MAXV=9 TBF=3 VWT=16 VHT=8 DF16=1 SAMP=euler \
  bash run_parity_nvfp4.sh        # base 832×480 → ×2 → 1664×960
```
- **Predicted peak ≈ 12–13 GB** (1664×960 = 0.765× the 1920 pixels → decode ~7.5 GB + params + out).
- **Predicted wall ≈ 230–260 s** @97f. Near-1920 sharpness at a comfortable, non-cliff peak.

> On the "1.5× keeps quality with a larger base at lower peak" idea: even if a 1.5× model existed, it
> would **not** lower the peak — the peak is set by the **final decode resolution**, not the upscale
> factor. A 1280→1.5×→1920 path still decodes at 1920 = same ~15 GB. A larger base *improves quality*
> (less upscale hallucination) at the cost of *more base-sampling compute*; it is orthogonal to peak.
> If the goal is "good quality at a lower peak," lower the **final** res → that is recipe (c).

---

## 4. Fixable VRAM waste / risks

1. **The 1920 decode leaves only ~130 MiB headroom (OOM cliff).** 15758 / 15888 MiB. Not "waste" but a
   real hazard under any co-tenant (prod `kobbler-ltx-video` shares the card — the sweep script literally
   waits for >14500 MiB free). **Fix:** ship 1920 with `TBF=2 VWT=20 VHT=10` (recipe b), which trims the
   9819 MB decode buffer to restore ~1 GB. Code site: the decode buffer is built in
   `ltx_vae.hpp` `decode_temporal_blend()` (`:1688`); levers already plumbed via `run_parity_nvfp4.sh`.

2. **MAXV under-utilizes VRAM during sampling at 1920 (free speed left on the table).** The base/refine
   plateaus (7.3 / 12.7 GB) sit well under the 15.7 GB decode peak, so the DiT-resident budget could be
   raised (MAXV 7→11) to cut offload overhead across 256 s of sampling **at zero cost to the true peak**.
   The sweeps ran ~MAXV 7 ("DiT keeps 7.00", `stable-diffusion.cpp:922`). Not a bug — an untaken lever.

3. **The latent upsampler reloads from disk (995 MB, no mmap) every render/segment.**
   `model_loader.cpp:874 NOT using mmap for …spatial-upscaler… (mmap disabled by caller)` — the upsampler
   is loaded fresh inside `upscale_ltx_spatial_video_latent()` (`:7991`) each call and destructed on
   return. Correct for peak VRAM (it's freed before decode), but for a **chained multi-segment** render
   it re-reads ~1 GB from disk per segment (~1 s each). Minor; only worth caching if you chain many
   short segments. Not a VRAM waste — a repeated-IO waste.

4. **No genuine co-residency waste found.** gemma freed after encode, DiT freed after refine, upsampler
   freed after upscale, audio VAE decoded separately. The lifecycle is already sequential-clean; the
   peak is intrinsic to decoding 1920×1088, not a leaked/held model.

---

## Optional single diagnostic render (only if you want to confirm the 1280 peak)

The 1280×704 peak is not in the manifests (fid runs didn't sample VRAM). One 97f run with the sampler
confirms recipe (a)'s predicted ~9–10 GB and that 1280 is sampling-bound (not decode-bound):

```bash
cd /home/dbrain/dev/longcat-avatar-ltxdenoise/ltx-denoise-repro
SMP=/tmp/vram_speed97; : > "$SMP"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 >>"$SMP"; sleep 1; done ) & SP=$!
RES=speed FR=97 MAXV=9 TBF=4 VWT=4 VHT=2 DF16=1 SAMP=euler TAG=diag_speed97 bash run_parity_nvfp4.sh
kill $SP; echo "PEAK: $(sort -n "$SMP" | tail -1) MiB"
```
Expected: peak ~9500–10500 MiB, wall ~105–110 s. If the peak plateau is early/flat (sampling) rather than
a late spike (decode), that confirms 1280 is sampling-bound and MAXV — not decode tiling — is its knob.
