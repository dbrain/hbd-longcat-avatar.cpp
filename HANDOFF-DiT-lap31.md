# LongCat-Avatar.cpp — DiT PERF HANDOFF (lap-31 close)

*Written 2026-05-28 at lap-31 close. Two shipped wins (Stage 1 text x-attn cache, Stage 2 BSA bitmap) + one fully-proven dead end (Stage 3 ncols=128). Cumulative DiT step wall delta since lap-29.2: **−1.98s on the BSA opt-in path** (141.86→139.88s mean), bit-exact. Dense path stays 142.5s (unchanged). The non-FA codepath survey came back essentially empty for sub-multi-day work; lap-32 starting point is in **§ What's left** below.*

---

## ⏱️ FIRST FIVE MINUTES

```
curl -sI http://10.0.0.208:8011/    # serve_clips.py — restart if dead
/tmp/render_bench.sh /src/build/lap32_dense_baseline.webm "" --steps 8
# expect ~142.5s wall, ~107.8s sampling
```

Standard bench (480x832, 25f, --steps 8 RESIDENT, --max-vram 9, seed 42):

```
# dense (default):
/tmp/render_bench.sh /src/build/<NAME>.webm "" --steps 8

# BSA r=1+sf + bitmap (best opt-in perf config, lap-31.2):
/tmp/render_bench.sh /src/build/<NAME>.webm "LONGCAT_BSA=1 LONGCAT_BSA_RADIUS=1 LONGCAT_BSA_SELF_FRAME=1 LONGCAT_BSA_BITMAP=1" --steps 8

# Text x-attn cache (lap-31.1, dense path, opt-in, OOMs combined w/ BSA at --max-vram 9):
/tmp/render_bench.sh /src/build/<NAME>.webm "LONGCAT_XATTN_TEXT_CACHE=1" --steps 8
```

Bit-exact gates:
- Dense changes: PSNR 99.00 vs `build/lap28_lap27baseline.webm`
- BSA changes: PSNR 99.00 vs `build/lap29_bsa_r1_selfframe.webm`

---

## 🔥 MOOD / MANDATE

**Lap-31 shipped both pieces of the lap-30 §What's-left mission.** Path 1 (cross-CTA scan sharing) was killed by owner pre-lap-31. Path 2 (Stage 2 CPU bitmap) shipped at −1.98s. Path 3 (Stage 3 ncols=128) shipped DEAD-END — all 4 occupancy/path variants regress (root-cause analysis in § Dead-ends).

The cumulative empirical ceiling for further mask-driven wins at this scope is now ~0.8s (the remaining gap between Stage 2's −1.98s and the +2.81s lap-30 oracle ceiling). Not worth chasing without a fundamentally different lever class — see lap-30 §What's-left for the kill list of FA-kernel sub-1% candidates.

**Owner mentality (carried from lap-30):** *"no quitting unless 100% proven from all angles to not be a performance win."* Stage 3 dead-end is 100% proven dead (measured 4 variants, root cause clear). Stage 2's −1.98s is below the 2.81s ceiling because of bitmap-path overhead (smem load + per-iter bit check + branch) — closing the remaining 0.83s gap would require restructuring the K-tile iter dispatch, which the owner already killed as "not worth it" via the path-1 kill.

**Gates (mandatory, every change):**
- Bit-exact: `tools/clip_compare.py <base> <new>` → PSNR 99.00 mean+min.
- Standard bench: 480x832, 25f, --steps 8 RESIDENT, --max-vram 9. Wins MUST show there.
- For BSA changes compare vs `lap29_bsa_r1_selfframe.webm`.

**ONE GPU** (RTX 3060 / sm_86 / 12 GB). Stop prod acestep / tts / llama before heavy runs. `docker rm -f longcat-avatar-iter` strays. Never two GPU jobs at once.

---

## What lap-31 SHIPPED

### lap-31.1 — Text cross-attn K/V cache across DMD steps
Commit `ac008e0` (parent only; no ggml submodule bump).

`text_cross_attn.kv_linear(context)` is step-invariant (umT5 runs once before sampling), so the post-norm K and V tensors are byte-identical every step. Persist at sampler_step ≤1; consume at step >1 skips kv_linear + k_norm + permute+cont entirely. F16 prescaled by `kv_scale=1/256` matching lap-28.2 cond cache pattern; bit-exact end-to-end (exact F16 exponent shift).

**Wall delta:** −0.5s sampling (107.92 vs 108.45 baseline, ~0.5% sampling, 0.35% wall). Smaller than the 1.18s subagent projection — kv_linear at M=512 is cheaper than the FLOP estimate suggested.

**VRAM:** +384 MiB (`hidden_size=4096 × n_ctx=512 × N=1 × F16 × 48 layers × 2 K+V`).
*Note: the original commit message + this handoff line had wrong arithmetic
("227 MiB" assumed `hidden_size=2304`; actual avatar_params.hidden_size=4096
per longcat_avatar.hpp:935 → real cache size is 384 MiB).*
**Cannot combine with BSA at --max-vram 9** (DiT 8.5 GiB + compute peak 1.45 GiB +
cond_kv 1.2 GiB + BSA mask 195 MiB at 25f + xattn 384 MiB > 11.9 GiB usable →
OOM). The BSA-mask-1-bit-pack lap-32 candidate (lap-32 §VRAM survey) reclaims
~182 MiB and combined with xattn n_ctx-trim opens the triple-stack.

**Opt-in:** `LONGCAT_XATTN_TEXT_CACHE=1` env.

### lap-31.2 — CPU-precomputed BSA bitmap (Stage 2)
ggml commit `4e2a7e74` + parent commit `1224396`. Tags `kobbler-lap31.2-bsa-bitmap-2026-05-28` on both repos.

`ensure_bsa_mask` now also derives a per-(Q-tile, K-tile) all-deny bitmap from the same F16 mask data — bit `b` of word `w` for Q-tile `jt` is set iff at least one cell in the (jt × kb) cube is ALLOW. Bitmap is sized for the avatar's hot self-attn FA config (DKQ=DV=128, ncols=64, nbatch_fa=64): 147 Q-tiles × 6 words = 3.5 KiB at 480p. Uploaded once per render to a side tensor in the same BSA backend buffer.

The FA kernel (`flash_attn_ext_f16_process_tile`, ncols2==1 branch) loads the per-Q-tile slice into shared memory once per CTA, then performs a single-bit lookup per K-tile iter — skipping iter dispatch for fully-denied K-tiles. The last iter is always unconditional (it owns the output writeback); lap-29.2's in-iter sparse-tile-skip stays in place as the second-line filter.

**Wall delta:** −1.98s mean across two trials (139.71 + 140.04 → 139.88s mean), −1.40% wall vs BSA baseline. Sampling −2.92s vs dense (-2.69%). Live K-tile rate logged at **50.0%** (12578/25137 bits set) — way above lap-29.2's ~7% in-iter skip rate, which is where the bigger-than-ceiling-math win comes from.

**Bit-exact:** PSNR 99.00 mean+min vs `lap29_bsa_r1_selfframe.webm`. Dense path (LONGCAT_BSA unset) verified byte-identical to lap-29.2 (142.17 vs 142.67s within noise, PSNR 99.00 vs lap28).

**Opt-in:** `LONGCAT_BSA_BITMAP=1` env (requires `LONGCAT_BSA=1`).

**Plumbing receipt for next lap (DO NOT regress):**
- New `ggml/src/ggml-cuda/longcat-fa-bsa-bitmap.cuh` — extern __device__ declarations gated by `LONGCAT_FA_BSA_BITMAP_DEFINING_TU` so fattn.cu can include the same header chain and define
- `ggml-cuda.h::ggml_cuda_set_longcat_fa_bsa_bitmap()` — host setter using `cudaMemcpyToSymbol`
- fattn.cu owns the device-resident symbols (defines before pulling in fattn-mma-f16.cuh so the kernel template body sees them in this TU)
- Hardcoded CPU bitmap constants `kNcols1=64, kNbatchFa=64` in `ensure_bsa_mask` — **depend on the FA template config**; update if `ggml_cuda_fattn_mma_get_config_ampere(128,128,64)` ever changes
- Bitmap only engages when `ncols2==1 && mask_h != nullptr && bitmap_dev != nullptr` — every other FA caller byte-identical to lap-29.2

---

## What lap-31 PROVED DEAD (Stage 3)

### ncols=128 FA template for DKQ=DV=128 self-attn

The lap-30 §What's-left lever-3 ("Each block handles 2× more Q rows → halves K HBM reads per CTA").

**Numbers (4 variants, 480/25f/--steps 8 RESIDENT):**

| variant | wall | Δ baseline |
|---|---|---|
| dense baseline | 142.67s | — |
| BSA baseline | 141.86s | — |
| dense + ncols=128, occ=2 | 144.28s | **+1.6s** |
| dense + ncols=128, occ=1 | 148.04s | +5.4s |
| BSA + ncols=128, occ=2 | 146.45s | +4.6s |
| BSA + ncols=128, occ=1 | 145.39s | +3.5s |

**Three compounding root causes the handoff didn't account for:**
1. **K HBM was never the bottleneck.** ~480 MB/FA-call × 640 calls ≈ 307 GB total → 0.85s at HBM peak = 0.6% of wall. The "halve K HBM reads" thesis had no headroom to win against.
2. **`ncols2=1` forces `nstages=0`** in `get_nstages` (fattn-mma-f16.cuh:358). The avatar's `gqa_ratio=1` self-attn can never reach multi-stage cp.async pipelining, so the kernel can't actually overlap K loads with compute — the K-tile count savings stay theoretical.
3. **BSA sparse-tile-skip predicate fires less often at ncols=128.** Each Q-tile now spans 128 rows → higher chance "some Q row has an allowed K" → less full-tile skip. BSA path doubles down on the loss.

Bit-exact PSNR 99.00 in every variant — correctness confirmed, performance is the issue. All code reverted (ggml `fattn-mma-f16.cuh` config-case + extern DECL, `fattn.cu` dispatcher, `ncols1_128-ncols2_1.cu` template instance deleted).

**Carry to dead-end log:** don't pursue ncols=128 (or larger) on the avatar's DKQ=DV=128 ncols2=1 path until either the K HBM premise changes (it won't on this shape) or someone re-derives the multi-stage path without the ncols2>=2 gate.

---

## What's LEFT for lap-32

The cumulative empirical ceiling for further FA-kernel wins is now ~0.8s wall (gap between Stage 2's −1.98s and the lap-30 +2.81s oracle). Lap-29.2 + lap-31 closed the FA codepath at this scope.

Off-codepath candidates surveyed via opus 4.7 subagent (full survey in `additional-levers.md` triage):

### 1. **VAE decode** (15.1s) — ~1.5-2s headroom if F16 acts work
The VAE decoder body currently runs F32 activations through 30+ conv-3d + RMS_norm chain. Going F16 (with F32 RMS_norm accumulate for precision) would halve im2col_3d HBM traffic. **PSNR risk is the blocker, not perf.** Estimated 6-8h scoping + bench-burn cycle. Worth a discrete experiment in a separate branch with the PSNR gate as a hard stop.

### 2. **umT5 / CPU/GPU overlap** (16.4s on CPU) — VERIFIED 1-2s window only
The lap-31 subagent initially proposed this as a 10-14s win. Verified: the real overlap window is just the VAE-encode + AI2V setup (~1-2s), because the DiT step 0 needs the text embedding for `text_cross_attn` at block 0 (can't start sampling without it), and the DiT weight load is serialized after umT5 free (TE 6 GiB + DiT 8.5 GiB > 12 GiB VRAM, gated on `finalize_deferred_dit_load`). Net: 1-2s = 0.7-1.4% wall. Not worth multi-hour invest given the small window.

### 3. **Audio cross-attn K/V cache** — projected 0.1-0.2% wall
Mirror of lap-31.1 text cache pattern, but for `audio_cross_attn`. Audio path is non-flash batched per-frame, so cache would need F32 K/V (no kv_scale prescale). Sub-noise per subagent estimate — owner explicitly waved off ("fairly useless along with rest of unplayed additional-levers.md items"). Skip.

### 4. **The remaining 0.83s gap to the lap-30 ceiling** — diminishing
Stage 2 captured 70% of the +2.81s lap-30 oracle ceiling. The remaining 0.83s = bitmap-path overhead (smem load + per-iter bit check + branch). Closing this would require eliminating the per-iter branch via a compacted kb0_live[] list (the lap-30 v2 design with the in-kernel scan replaced by the CPU bitmap). Doable but only 0.5% wall — owner discretion.

### 5. **Considered + skipped from this lap's survey**
All these were measured/triaged dead per the additional-levers.md OUT list and the lap-30 dead-end log:
- CFG levers (N/A, avatar runs 1 DiT eval/step)
- INT8 tensor cores (Q4_K MMQ at floor)
- CUDA graphs reuse (5ms launch overhead, compute-bound)
- VAE generic profile (5 laps ahead per advisor)
- RoPE coalesce / pixel-shuffle / CONT (lap-24 done)
- FP32 LayerNorm → FP16 (PSNR-blocked)
- FP8 K/V (no Ampere FP8 compute)
- SwiGLU MM-MM-GLU autofusion (only fires at ne[1]==1, avatar runs M≈10000)
- Stage 3 ncols=128 (this lap)
- (b) kb0-compaction with in-kernel scan (lap-30)
- per-K-col-chunk skip (lap-30)
- MMF/MMVQ launch_bounds (lap-30)
- FA nbatch sweep (lap-30)

---

## Bench results (lap-31, all 480x832, 25f, --steps 8, --max-vram 9, seed 42)

| config | wall | sampling | notes |
|---|---|---|---|
| dense baseline (lap-29.2) | 142.67s | 108.45s | reference |
| BSA r=1+sf baseline (lap-29.2) | 141.86s | — | reference |
| **dense + xattn cache (lap-31.1)** | **142.07s mean** | **107.92s** | -0.5s, bit-exact, opt-in |
| **BSA r=1+sf + bitmap (lap-31.2)** | **139.88s mean** | **105.53s** | **−1.98s, bit-exact, opt-in** |
| dense + ncols=128 occ=2 (Stage 3) | 144.28s | — | DEAD, reverted |
| dense + ncols=128 occ=1 (Stage 3) | 148.04s | — | DEAD, reverted |
| BSA + ncols=128 occ=2 (Stage 3) | 146.45s | — | DEAD, reverted |
| BSA + ncols=128 occ=1 (Stage 3) | 145.39s | — | DEAD, reverted |

---

## ONE-LINER reminders for next lap

- Standard bench: 480x832, 25f, --steps 8 RESIDENT, --max-vram 9.
- Best opt-in perf config: `LONGCAT_BSA=1 LONGCAT_BSA_RADIUS=1 LONGCAT_BSA_SELF_FRAME=1 LONGCAT_BSA_BITMAP=1` (lap-31.2 path, 139.88s).
- Best dense config: `LONGCAT_XATTN_TEXT_CACHE=1` (lap-31.1, 142.07s) — DON'T combine with BSA at --max-vram 9.
- Tag `kobbler-lap31.2-bsa-bitmap-2026-05-28` (both parent + ggml) — the shipped tip.
- BSA bitmap live K-tile rate (logged in `[BSA] mask built` line) at r=1+sf: 50%. The lap-30 ceiling math (~2.81s) was pessimistic because the lap-29.2 in-iter skip rate (~7%) UNDERSAMPLED the actual all-deny structure — Stage 2 caught the full half-empty cubes.
- Owner waved off audio x-attn cache + remaining unplayed additional-levers.md items as "fairly useless" — DON'T burn cycles re-evaluating.
- Stage 3 (ncols=128) is 100% proven dead — DON'T re-try unless something fundamental changes (ncols2 != 1, or new ViT shape).
- ggml `__device__` symbols across TUs: use a header-with-extern-decl + LONGCAT_FA_BSA_BITMAP_DEFINING_TU guard pattern so the defining TU can include the same chain. nvcc otherwise treats `extern __device__` as a static definition (warns and creates per-TU copies).
- If `ggml_cuda_fattn_mma_get_config_ampere(128,128,64)` ever changes ncols1 or nbatch_fa from 64, the bitmap construction in `ensure_bsa_mask` (`kNcols1`/`kNbatchFa`) MUST be updated to match — otherwise OOB bitmap reads.
