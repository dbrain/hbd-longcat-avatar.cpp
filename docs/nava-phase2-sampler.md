# NAVA Phase-2 sampler spec (silent video, Euler flow-match, cond+uncond CFG)

Distilled from pipeline_nava.py + scheduler/flow_match.py. **Corrections applied** (the extractor
agent twice emitted Wan2.1 numbers): text dim is **4096** (umT5-xxl) → text_embedding Linear
[3072,4096]; the model `dim` is **3072**. EXACT latent grid + VAE down-factors: PIN from the
reference npz (`input_vid_latent` shape) + by reading the in-tree WanVAERunner — do NOT trust the
agent's H'=30/W'=52 vs /8-vs-/16 claim (it was self-flagged uncertain).

## Euler FlowMatchScheduler (the simple path — use this first @ 10 steps)
- `num_train_timesteps=1000`, `shift=5.0` (nava_640.yaml).
- `set_timesteps(N)`: `sigmas = linspace(sigma_max=1.0, sigma_min≈0.003/1.002, N)`; shift transform
  **`σ ← 5σ/(1+4σ)`**; `timesteps = σ·1000` (fed to the model as-is, in [0,1000]).
- step update (per latent): **`x ← x + v·(σ_next − σ_cur)`** (σ_next=0 at the final step).
- UniPC (FlowUniPCMultistepScheduler) = later/optional; Euler@10 is the target for the first clip.

## CFG (simplest correct subset)
- Two forwards per step, SAME timestep t for both.
- cond = forward with positive umT5 context; uncond = forward with negative-prompt context
  (curated negative text) OR zeros_like(context).
- combine: **`v = v_uncond + scale·(v_cond − v_uncond)`**, video **scale=3.0** (audio 2.0, Phase 3).
- DEFERRED: align_3d_cfg (+1 forward, modality-mask) and timbre_cfg (+1 forward, spk removed).
  The `slg_layer=11` skip-last-layers on the uncond path = optional opt; ignore for MVP.

## Per-step loop (silent video)
noise video latent (seeded randn, VAE-latent space, shape from reference) →
for t in timesteps[0..N): v_cond = DiT(latent, t, ctx_pos); v_uncond = DiT(latent, t, ctx_neg);
v = v_uncond + 3·(v_cond−v_uncond); latent += v·(σ_next−σ_cur).

## VAE decode (reuse in-tree WanVAERunner — already ported)
- local adapter scaling_factor=1.0, shift_factor=0.0 ⇒ NO latent renorm before decode (verify the
  in-tree wan VAE's own mean/std bake — longcat applies Wan2.2 48ch mean/std in WanVAERunner; CHECK
  whether NAVA's latent is already in that convention or needs the longcat normalization. PIN this
  when wiring — it's a prime "washed out / wrong contrast" suspect).
- reshape final video latent [Σ T·H'·W', 48] → per-sample [T,H',W',48] → permute→[1,48,T,H',W'] →
  WanVAE decode → [1,3,T,H·s,W·s] (s = VAE spatial up). pixels in [-1,1] → (x+1)/2·255 → u8 → webm.

## First-clip recipe
Reuse: T5Runner (umT5, existing), WanVAERunner.decode (existing). New: the Euler loop + CFG +
NavaRunner.forward wrapper that takes (latent, t, context) and returns video velocity. Render 640
bucket, short F, **2-step smoke first** then **10 steps**, SILENT (skip audio path) → eye-test :8097.

## Pin-from-reference (don't guess)
- exact video latent shape [48,F,H',W'] (the noise/denoise space) ← reference `input_vid_latent`.
- VAE spatial/temporal down factors ← in-tree wan VAE + the reference shapes.
- whether latent needs Wan2.2 mean/std normalization before decode ← read WanVAERunner.
