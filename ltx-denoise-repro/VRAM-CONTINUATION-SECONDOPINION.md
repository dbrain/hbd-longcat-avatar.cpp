# 1080p continuation VRAM — second-opinion brief (2026-07-07)

**Ask:** how to get a **97-frame, 1080p (1920×1088), multi-segment CONTINUATION** render to peak **≤11.5GB (11776 MiB)** VRAM. It currently **works** but peaks **~13GB** (12977 MiB steady, 13252 transient) — **~1.5GB over.** Single (non-continuation) 1080p renders already fit at ~11GB. We want an outside read on the levers before committing to a speed or quality trade.

## System / setup (so a fresh reader has the model)
- Engine: a C++/ggml fork ("longcat/LTX") running **LTX-2.3, a ~22B video DiT**, nvfp4-quantized weights. Repo `/home/dbrain/dev/longcat-avatar-ltxdenoise`, branch `ltx-denoise-workflow`.
- **The 22B DiT does not fit resident on the 16GB card** alongside the VAE + activations, so params are **offloaded to host (mmap) and streamed to GPU per-compute**. A graph-cut planner splits the 50-step compute into segments under a VRAM budget (`LTX_MAX_VRAM`, currently 7GB) and keeps a **shared-resident set** of hot params pinned on GPU across steps to avoid re-streaming.
- Pipeline per segment: **base sample** (960×544, 8 steps) → **latent spatial upscale ×2** (→1920×1088) → **hires refine** (3 steps) → **VAE decode** (spatial-tiled 2×2, feathered). Worker-isolated: a CUDA-free parent forks a CUDA child per render.
- Deployed recipe (env): `LTX_VAE_SPATIAL_TILES=2x2 SPATIAL_OVERLAP=4`, `LTX_VAE_DECODE_F16=1`, `LTX_DIT_F16=1` (residual stream + linears in F16 — **compute precision lever already pulled**), `LTXAV_VAE_LAZY=1` (evict VAE during DiT), `LTXAV_DIT_FREE_DURING_DECODE=1` (free DiT residency before VAE decode), `LTX_MAX_VRAM=7`. Attention = **dense flash-attention** (`ggml_flash_attn_ext`; cuDNN SDPA / native MMA / Blackwell sm120 MMA backends — all dense).

## The mechanism (well-established, please sanity-check)
A "97-frame" **continuation** segment is NOT a 97-frame render. `cont_latent_frames=K=3` prior latent frames (seg-1's tail) are **prepended** as the continuity anchor, so the DiT processes **16 latent frames** (13 for the 97-frame generation + 3 anchor), i.e. **121 pixel frames**. A single 97-frame render is **13 latent frames**.

The refine peak scales with **latent frame count** (≈ sequence length; flash-attn keeps *memory* O(N), so it's linear not quadratic):
```
single 97f (13 latent) refine:  resident 5471 + compute 2767 + overhead 1873 = 10527   ← fits
seg-2    (16 latent) refine:     resident 5471 + compute 3369 + overhead 3719 = 12977   ← +1.5GB over
delta = +3 latent frames = +602 compute + +1846 overhead ≈ +817 MiB per latent frame
```
Measured compute ratio 3369/2767 = 1.218 ≈ frame ratio 16/13 = 1.231 → **the extra cost is the extra frames, proportionally.** Overhead (flash-attn working buffers, ctx, cuDNN) *also* scales with sequence length, and is actually the bigger per-frame term (~615/frame vs ~200/frame compute).

**Open diagnostic (running now):** a **single 121-frame render** (16 latent, NO continuation machinery) at the same env — if it also peaks ~12977, it **confirms the VRAM is purely frame-count**, and "continuation" is a red herring (it's just a longer clip). Expected yes.

## Levers tried + results
| lever | idea | result |
|---|---|---|
| **DIT_F16** | F16 residual stream halves activation | **already on** (deployed). Lever exhausted. |
| **FIX 1** (commit d5254b0) | free DiT stream-scratch buffers before decode | worked on **decode** (12771→11009) but decode isn't the ceiling; **no effect on refine peak.** |
| **seam scratch-free** (734998e) | free scratch at seg-1→seg-2 seam + pre-refine | byte-identical, **did NOT move the refine peak** (the scratch it targets isn't what's resident during refine). |
| **FFN_TILE 4096→2048** | tile the FFN compute | **no effect** — the refine buffer is attention/sequence-bound, not FFN-bound. Reverted. |
| **MAXV 7→5** | shrink the graph-cut budget to shrink resident | **resident UNCHANGED at 5471** — the shared-resident set is structural (hot params read by ≥2 segments), not budget-bounded. **MAXV is not a resident lever.** |
| **BSA** (block-sparse attn, exists for LongCat) | sparsify the O(N²) attention | **speed lever, not VRAM** — flash-attn memory is already O(N); a BSA mask *adds* ~127MB (O(N²) F16 mask) + re-arms kv-pad. Wiring for LTX ≈ 90 lines, sm120-unverified. Noted for speed, not pursued. |

## The two remaining levers (each a real trade)
1. **K reduction** (`cont_latent_frames` 3→2→1): fewer anchor frames → fewer latent frames → less compute+overhead at ~817/frame. **K=3→1 (16→14 latent) ≈ −1634 → ~11.3GB, fits.** Cost: **less continuity anchor across the seam** (K=1 = ~1 frame of overlap) — risk of motion stutter / drift at the seam. **No speed loss, no resolution loss.** Testable with seam-stress prompts (continuous camera motion + persistent subject).
2. **free-DiT-in-refine** (stream the 5471 resident during the 3 refine steps instead of pinning): drops the resident → refine ~7-8GB, **fits easily.** Cost: refine ~2.5× slower (cold-stream 37→~90 s/it), **≈ +22min on an 8-seg 30s MV.** No quality/continuity loss.
3. (Quality option) **upscale ×2→×1.5** (1920×1088→1440×816): refine compute ~×0.56 ≈ −1.5GB. Cost: **face-quality drop** (owner already judged this "loud on faces").

## Questions for the second opinion
1. Is the frame-count mechanism right, or is there continuation-specific overhead we've mis-attributed? (The single-121f diagnostic tests this.)
2. Is there a lever we're missing on the **refine** that isn't speed/quality/continuity? e.g. can the flash-attn **overhead** (3719, the dominant per-frame term) be bounded independent of sequence length? Is the ~1846 refine overhead genuinely irreducible or is there a pool/workspace not being trimmed *during* the refine (FIX 1 only reached the decode)?
3. Could the refine run at a **lower resident set** than the full 5471 hot params (a partial pin — pin the top-N hottest, stream the rest) to trade a *little* speed for the ~1.5GB, cheaper than the full free-DiT? Does the graph-cut expose that knob?
4. Is **K=2** (15 latent, −817 → ~12.2GB) a reasonable half-measure combined with something small (e.g. the FIX-1 scratch-free actually applied to the refine pool)? What's the smallest total change that clears 11776?

## Reproduce
- Deployed engine `734998e` (+ compose in `/home/dbrain/dev/kobbler`, `LTX_REF` reconciled). koblem API `POST :8090/api/v1/ltx-video/generate` (Bearer), engine-direct `:8096`.
- 2-seg continuation JSON: `ltx-denoise-repro/../scratchpad/renders/cont2_1080.json`. Single-121f diagnostic: `single121.json`.
- Enable `LTX_VRAM_BREAKDOWN=1` → the `[VRAM] ltxav reserve: driver_used=… (compute_buf=… resident=…)` line is the per-phase attribution. `shared-resident set: N params (M MB)` line shows the pinned set. Refine steps log `x/3 - Ns/it`; base `x/8 - Ns/it`.

## UPDATE 2026-07-07 — root cause is stuck cuDNN plan memory (cross-project)
The continuation refine+decode +1.7GB/phase overhead is the persistent cuDNN SDPA/conv3d plan
memory, held by the cuDNN handle (NOT the ggml pool, NOT the fe::graph cache objects). Cache-clear
had no effect; conv3d-off crashes. This is a cross-project VRAM bug (any warm cuDNN worker).
Full investigation + fix directions for a reviewer: **CUDNN-STUCK-PLANS.md**.
Frame-fix (LTX_REFINE_CONTEXT_FRAMES) works but only lowers the refine, not the peak (the decode
is the ceiling). Quality fit-now lever: upscale x2→x1.75 (~-1.5GB, moderate face-softening).
