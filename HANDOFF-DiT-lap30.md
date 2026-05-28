# LongCat-Avatar.cpp — DiT PERF HANDOFF (lap-30 close, lap-31 mission)

*Written end of lap-30 (2026-05-28). Lap-30 shipped **no perf changes** — the four levers explored (per-K-col-chunk skip, kb0-iter compaction, FA nbatch sweep, MMF/MMVQ launch_bounds) all measured dead or net-negative at the current scope. The honest finding is that the BSA r=1+self_frame path is **within ~1s of its empirical ceiling** for in-kernel mask-driven optimizations: the geometry of BSA's cube-near constraint plus K-tile boundaries means whole-tile-skip already catches almost everything skippable. Further wins on this codepath require either out-of-band mask preprocessing (multi-day plumbing) or stricter mask quality (LONGCAT_BSA_RADIUS=0 etc).*

*Lap-31 mission TBD — see `## What's left` below for the three candidate paths (cross-CTA scan sharing, CPU-precomputed BSA bitmap, ncols=128 case). User direction needed before kicking off.*

---

## ⏱️ FIRST FIVE MINUTES

```
curl -sI http://10.0.0.208:8011/    # serve_clips.py should answer HTTP 200
# if dead:
cd ~/dev/longcat-avatar.cpp && nohup python3 tools/serve_clips.py --dir build --port 8011 \
  > /tmp/serve_clips.log 2>&1 & disown
```

http://10.0.0.208:8011/ — `lap29_sparse_final.webm` is the dense baseline reference clip (PSNR 99.00 vs lap-27). The BSA quality reference is `lap29_bsa_r1_selfframe.webm`.

Standard bench command (unchanged from lap-29/30):
```
/tmp/render_bench.sh /src/build/<NAME>.webm "<env kvs>" --steps 8
# dense:  ""
# BSA:    "LONGCAT_BSA=1 LONGCAT_BSA_RADIUS=1 LONGCAT_BSA_SELF_FRAME=1"
```

Bit-exact gate:
```
cd ~/dev/longcat-avatar.cpp && python3 tools/clip_compare.py \
  build/lap28_lap27baseline.webm build/<NEW>.webm   # → PSNR 99.00 mean+min
```

**Current shipped baseline: 142.5s** (lap-29.2, same as lap-30 finish state, no code shipped).

---

## 🔥 MOOD / MANDATE — READ TWICE, DO NOT SKIP

**MENTALITY (carried from lap-30):** *"Go mental — get it faster than it should ever be on this hardware. Custom kernel for wins is our bag."* Owner explicit: *"no quitting unless 100% proven from all angles to not be a performance win."* The lap-30 dead-ends are 100%-proven dead at this scope — but the remaining live paths (cross-CTA scan sharing, CPU bitmap, ncols=128) are all multi-hour to multi-day. Owner's stated priority order for lap-31: **(2) launch_bounds / nbatch first, (4) cross-CTA scan sharing if stable, (1) CPU bitmap as final**. Lap-30 already exhausted (2). (4) and (1) are what's left.

**EVERY FLOOR CLAIM HAS BEEN DISPROVABLE:**
- "im2col is at roofline" → 7% BW (lap-21)
- "MUL_MAT is floored" → 7× redundant cond work (lap-26)
- "FA kernel needs fork-class FA2/FA3 surgery" → 1-int launch_bounds change (lap-29.1: −8.5%)
- "stock ggml flash applies mask post-QK so BSA can't help" → TRUE, but the cheap fix (whole-tile early-skip) only got us back to baseline — **the per-K-col fix DID land but at sub-noise gain because BSA geometry caps it (lap-30 finding)**

**Measure. Do not predict.** Lap-30 measured the kb0-compaction ceiling at **+2.8s wall = 2%** even with an oracle bitmap, because per-skipped-iter overhead is already low — the existing lap-29.2 whole-tile-skip path is near-optimal for BSA r=1+sf.

**Gates (mandatory, every change):**
- Bit-exact: `tools/clip_compare.py <base> <new>` reads PSNR 99.00 mean+min.
- Standard bench: **480×832, 25 frames, --steps 8, RESIDENT, `--max-vram 9`** — wins MUST show there.
- For BSA changes: compare vs `lap29_bsa_r1_selfframe.webm`.

**ONE GPU** (RTX 3060 / sm_86 / 12 GB). Stop prod acestep / tts / llama before heavy runs. `docker rm -f longcat-avatar-iter` strays. Never two GPU jobs at once.

---

## What lap-30 SHIPPED

**Nothing.** All four explored levers measured dead or net-negative. Code reverted to lap-29.2 state (commit `8d9b71f` parent / `11341ed5` ggml). Tag `kobbler-lap29.2-bsa-quality-knob-sparse-fa-2026-05-28` is the current production tip.

---

## What lap-30 PROVED DEAD (the dead-end log)

The handoff's headline mission was "per-K-col sparse-FA kernel for 12% wall via the 125.32s always-skip probe ceiling". That number was misread:

**The 125.32s always-skip probe was an OVER-skip measurement, not a reachable ceiling.** The probe `return`ed at every iter when mask is present, skipping body work for ALL iters — including the ~93% of iters whose body REALLY does need to run for correctness. The actual reachable ceiling for any "skip work inside or instead of the iter body" approach is the *body cost × actual skip rate*, which for BSA r=1+sf measures ~1.2s wall (already captured by lap-29.2's whole-tile-skip).

### (a) Per-K-col-chunk skip — measured dead

The mission's `(a) per-MMA-chunk skip` was implemented in `fattn-mma-f16.cuh::flash_attn_ext_f16_iter` — preprocessing scan over `tile_mask` in smem builds a per-warp `warp_kqc_compute_mask` bitmap (1 bit per K-position chunk of T_A_KQ::I=16 K-positions), then both Q_in_reg branches `continue` past `load_ldmatrix(K_A)` + `mma()` for chunks where this warp's Q-rows are all -INF.

**Numbers (480/25f/--steps 8, RESIDENT):**

| config | wall | what |
|---|---|---|
| dense baseline | 142.87s | reference |
| dense + (a) | 142.93s | +0.06s (noise) — preprocessing short-circuits cleanly on mask_h=nullptr |
| BSA r=1+sf baseline (lap-29.2) | 142.21s | reference (matches lap-29 prod) |
| BSA + (a) | 141.33s | -0.88s claimed but within ±1s noise |
| BSA + (a) force-skip ALL chunks (probe, output corrupt PSNR 8.74dB) | 139.07s | -2.7s wall = ceiling for "skip QK ldmatrix+MMA only" |

**Why (a) doesn't unlock the 12% target:**

The K-position chunk granularity (T_A_KQ::I = 16) is finer than the K-tile (nbatch_fa = 64). The avatar's geometry — 480×832 video → 6 latent frames × 1560 latent positions per frame; ncols1 = 64 Q-rows per Q-tile span ~0.04 of a frame; nbatch_fa = 64 K-positions per K-tile likewise — means each Q-tile and each K-tile are essentially within ONE latent frame. BSA's cube-near + anchor mask is determined at the per-frame level. So K-tile slices are **either fully-allowed or fully-denied per Q-tile**; the lap-29.2 whole-tile-skip already catches the fully-denied case. The chunks WITHIN a kept K-tile have ~16% chance of being fully denied for one warp's row slice — predicate-fire-rate matches that ratio, and chunk-skip savings are capped at `16% × 2.7s = 0.43s`. **The 2.7s force-skip-all ceiling was the bound; the predicate fires too rarely to approach it.**

Code reverted. Don't re-implement without first re-deriving the predicate-fire-rate from a different mask geometry.

### (b) kb0-iter compaction — measured dead at this scope

Implemented as preamble in `flash_attn_ext_f16_process_tile`: scan the full mask once per Q-tile into a per-CTA `__shared__ int kb0_live[]` list of K-tile indices that have ≥1 allowed cell. Then iterate that compacted list instead of contiguous `[kb0_start, kb0_stop)`. Iter signature extended with `kb0_next` for prefetch chain. Bit-exact (PSNR 99.00 confirmed on both v1 serial-scan and v2 parallel-scan variants).

**Numbers (480/25f/--steps 8, BSA r=1+sf):**

| config | wall | breakdown |
|---|---|---|
| baseline (lap-29.2 whole-tile-skip in iter) | 142.21s | reference |
| (b) v1 — serial per-kb0 scan, 145 __syncthreads | 157.50s | +15.74s regression |
| (b) v2 — parallel scan across warps + bitmap + compact | 149.90s | +7.69s regression |
| (b) v2 scan-ONLY isolation probe (preamble runs, dispatch uses uncompacted) | 152.71s | +10.95s = pure scan cost |
| (b) v2 compaction win component (152.71 - 149.90) | -2.81s | actual savings unlocked |

**Why (b) is structurally capped at ~2-3% wall:**

The compaction `skip rate` is real and substantial — **printf-instrumented at 58.2% of K-tiles skipped** for jt=0 (well above the ~7% effective skip rate the existing whole-tile-skip path was contributing). But the empirical savings are limited because:

1. **Per-skipped-iter overhead was already low.** lap-29.2's whole-tile-skip path (`cp.async wait + sync + sparse-tile-any-allowed predicate + prefetch chain + return`) costs ~33 μs per skipped iter, totaling ~2.8s wall across all skipped iters per render. Compaction eliminates that 2.8s. **That's the ceiling**, not the 16.44s probe-vs-BSA delta.

2. **The in-kernel scan to drive compaction costs more than it saves.** 4672 CTAs (146 Q-tiles × 32 heads) × 1.2 MB per-CTA mask scan = 5.6 GB mask HBM traffic per attn call. At 360 GB/s peak with ~33% effective bandwidth (measured) = ~11s wall per render. Net: -8s regression. **The redundant scan is the killer — heads share the same mask but can't share scan results without cross-CTA coordination or out-of-band precomputation.**

Code reverted. The compaction APPROACH is sound — the obstacle is the in-kernel scan cost, not the dispatch path. See **§ What's left** for the two paths that COULD unlock it (cross-CTA sharing or CPU bitmap).

### Lever #5 — FA nbatch sweep (DKQ=DV=128, ncols=64 cell)

| config change | wall (dense) | verdict |
|---|---|---|
| nbatch_V2 = 64 → 32 | 142.68s | noise (-0.31s) |
| nbatch_V2 = 64 → 16 | 144.34s | slight regression (+1.35s) |
| nbatch_fa = 64 → 32 | 143.50s | noise (+0.51s) |
| nbatch_fa = 64 → 128 + occupancy 3 → 2 | 197.27s | catastrophic (+54s) — halved occupancy unrecoverable |

`nbatch_K2` is fixed at DKQ/2 = 64 by the multi-stage pipeline static_assert. `nstages_target` is already at 2 (multi-stage). The DKQ=DV=128 ncols=64 cell at line 71 is at the local optimum after lap-29.1's occupancy 2→3 change. **No nbatch knob has headroom here.**

### Lever #3 — MMF/MMVQ launch_bounds 1 → 2

| change | wall (dense) | verdict |
|---|---|---|
| `mmf.cuh:49` `__launch_bounds__(.., 1)` → `(.., 2)` | 143.40s | noise (+0.41s) |
| `mmvq.cu:395` `__launch_bounds__(.., 1)` → `(.., 2)` | 143.33s | noise (+0.34s) |

Both MMF (float matmul) and MMVQ (vec-quantized matmul) are too cold on this codepath to register a launch_bounds bump. Q4_K_M model dispatches dominantly through MMQ (which the handoff already flagged as do-not-bump per lap-29 dead-end). `mmf.cuh:298` (mul_mat_f_ids, MoE variant) and `mmvq.cu:601` (mmvq_mmid, MoE variant) untouched — not used by the avatar's DiT. **No launch_bounds headroom in this group.**

---

## What's LEFT for lap-31 (ranked — owner pick: 3 then 2; **path 1 KILLED**)

The cumulative empirical ceiling for further in-kernel mask-skip work is **~2-3% wall** (kb0-compaction's 2.8s upper bound). Owner's lap-31 order: **(3) ncols=128 case FIRST** (orthogonal lever, no quality cost, ~1-3% wall), **THEN (2) CPU-precomputed BSA bitmap** (multi-day plumbing, ~2-3% wall — stacks on top of (3) since they touch different codepaths). Combined target: ~3-6% wall on top of the current 142.5s baseline.

### ~~1. (b) revival via cross-CTA scan sharing~~ — **KILLED by owner 2026-05-28**

Owner ruled out the cross-CTA spin-wait approach: "don't like the risk." Stability of spin-wait under CUDA's non-fair scheduler isn't provable at this scope. The kb0-compaction win (~2.8s) goes through path (2) instead. Leaving the design sketch below for the record only — DO NOT IMPLEMENT.

#### Original sketch (record-only):

Lap-30 measured the obstacle to (b) as the **5.6 GB/attn redundant mask scan** — 32 heads per Q-tile each independently scan the SAME mask data. If only the first CTA per Q-tile scans (others spin-wait on global flag, read result from HBM), redundancy drops 32× → ~175 MB/attn scan → ~487 μs/attn × 240 attn = ~117 ms total scan cost = effectively free. Plus the existing 2.8s compaction win. Net **+2-3% wall**.

**The risk** is the spin-wait synchronization. CUDA doesn't preempt running CTAs, so once the claimer is scheduled it WILL complete in finite time — but the scheduler is not fair, and if claimer + waiters land in different waves, waiters in earlier wave could spin a long time before claimer starts. Mitigate with a bounded spin-count then fallback to local scan. Owner authorized "go if confident stable".

**Sketch:**
```cuda
static __device__ uint64_t  g_qtile_key[MAX_QTILES] = {0};
static __device__ uint32_t  g_qtile_bitmap[MAX_QTILES][N_WORDS];

const uint64_t key = (uintptr_t)mask_h ^ ((uint64_t)stride_mask << 32);
const uint64_t SENTINEL = ~0ULL;

if (tid == 0) {
    uint64_t cur = atomicAdd(&g_qtile_key[jt], 0);
    if (cur == key) { read bitmap; }
    else {
        uint64_t old = atomicCAS(&g_qtile_key[jt], cur, SENTINEL);
        if (old == cur) {
            scan; write bitmap; __threadfence();
            atomicExch(&g_qtile_key[jt], key);
        } else {
            int spins = 0;
            while ((cur = atomicAdd(&g_qtile_key[jt], 0)) == SENTINEL) {
                if (++spins > 100000) { local_scan_fallback = true; break; }
            }
            if (cur == key && !local_scan_fallback) read bitmap;
            else local_scan_fallback = true;
        }
    }
}
```

**Implementation pitfalls to watch:**
- `g_qtile_key` persists across attn calls — keying on `mask_h ^ stride_mask` means same-mask FA calls hit cache (good). Different masks (audio cross-attn vs BSA self-attn) get different keys (correct). Two different BSA masks across renders would key the same if both have same `mask_h`/stride — should be fine since masks are constant per render.
- The fallback `local_scan` doubles back to the lap-30 v2 code (parallel scan + bitmap compact). Keep that code structurally available.
- MAX_QTILES sizing: the avatar's self-attn at 6 latent frames × 1560 = 9360 / nbatch_fa(64) = 146 Q-tiles. Audio cross-attn is much smaller. 256 leaves headroom; warn-and-fallback if jt ≥ MAX_QTILES.

### 2. **CPU-precomputed BSA bitmap** (multi-day — clean ceiling)

If (1) is flaky, fall back to building the per-(Q-tile, K-tile) all-deny bitmap CPU-side at BSA mask construction time. Since the BSA mask is built ONCE per render in `ensure_bsa_mask` (`src/longcat_avatar.hpp:1295-1359`) and is constant for the render's duration, the derived bitmap is also constant. Pre-compute it there, store as a side tensor in `runner_ctx.bsa_bitmap`, plumb through avatar code → new ggml op `LONGCAT_FA_BSA` (or extend `FLASH_ATTN_EXT` with optional bitmap input) → ggml-cuda dispatch passes bitmap pointer → kernel reads bitmap (no scan).

**Plumbing surface:**
- `ensure_bsa_mask`: derive bitmap (~50 lines, deterministic from cube_h/cube_w/radius/anchors/T/H/W/Q-tile-size).
- Add `runner_ctx.bsa_bitmap` field (3 lines).
- Either: (A) extend `ggml_flash_attn_ext` with optional 5th input (invasive, affects all FA consumers). (B) New `ggml_longcat_fa_bsa` op that wraps `ggml_flash_attn_ext` + bitmap (cleaner, opt-in for BSA path).

Estimated 8-16 hours plumbing + testing. Net **+2-3% wall**, but high-confidence and stable.

### 3. **ncols=128 case** (1-3% potential — multi-hour)

The handoff already listed this as lever #6. Each block handles 2× more Q rows = 0.5× K HBM reads per CTA. Requires new template instantiation in `fattn.cu` for `ncols=128` config + a new `mma_tile_sizes<DV, 128>` entry. ~4-8h work. Realistic 1-3% wall on top of lap-29.1. Try AFTER one of (1)/(2) lands compaction; this lever stacks orthogonally.

### 4. **Tighter BSA quality knob** (cheap experiment — quality cost)

Owner noted BSA is acceptable as a perf-vs-quality knob ("BSA isn't the target, performance is — BSA is the option if we need BSA to get wins"). Bench `LONGCAT_BSA_RADIUS=0` (only cube-co-located + anchors, no cube-near) to measure the wall savings from a tighter mask. If significant (>5%), the existing lap-29.2 whole-tile-skip path captures most of it for free — only the quality call from owner is needed. ~30 min experiment. Not yet run.

### 5. **Dead-ends — do not re-burn**

Carry forward from lap-29 handoff. ADDITIONS from lap-30:
- **MMQ launch_bounds occupancy 1→2** (lap-29): +27% regression. Don't bump.
- **MMF / MMVQ launch_bounds occupancy 1→2** (lap-30): noise only — both kernels too cold on Q4_K_M codepath.
- **FA nbatch_K2** is fixed at DKQ/2 for multi-stage. Can't sweep.
- **FA nbatch_fa = 128** drops occupancy 3→2 → catastrophic regression.
- **FA nbatch_V2 sweep** (16, 32) — all noise or slight regression at current shape.
- **(a) per-K-col-chunk skip via in-iter smem scan** — bit-exact, but predicate fires too rarely (~16% of chunks) at BSA r=1+sf geometry. Ceiling 2.7s = 1.9% wall, actual gain within noise. Don't re-implement unless mask geometry changes.
- **(b) v1 kb0-compaction with in-kernel mask scan** — net regression because per-CTA scan reads 1.2 MB × 4672 CTAs = 5.6 GB redundantly. Don't ship without out-of-band scan reduction (path 1 or 2 above).

---

## Method (reproducibly)

**Build:** `~/dev/kobbler/docker/longcat-avatar-dev/iter.sh build` (~18-30s incremental, sm_86, ccache).

**Standard bench:** 480×832, 25f, --steps 8, RESIDENT (no --offload-to-cpu), `--diffusion-fa --seed 42 --clip-on-cpu --max-vram 9`. Use `/tmp/render_bench.sh /src/build/<NAME>.webm "<env kvs>" --steps 8`.

**Bit-exact gate:** `python3 tools/clip_compare.py build/lap28_lap27baseline.webm build/<NEW>.webm` — read PSNR 99.00 mean+min. For BSA changes also compare vs `lap29_bsa_r1_selfframe.webm` (BSA reference).

**Profile (post-lap-29.1 / lap-30 finish, unchanged):** `LONGCAT_OP_PROFILE=1`. Per-step:
- MUL_MAT 55.3% (Q4_K MMQ — floored, do NOT bump)
- FLASH_ATTN_EXT 28.7% — main target, but lap-30 confirms only +2-3% wall headroom via (1)/(2)/(3) above
- ADD 4%, CONT 3%, SCALE 2%, CONCAT 2%, MUL 2%, ROPE_PE 1.5%, UNARY 1%

**To re-verify lap-30 measurements:**
- Always-skip probe ceiling: edit `ggml/src/ggml-cuda/fattn-mma-f16.cuh` nstages>1 branch, change `if (!any_allowed)` → `if (mask_h != nullptr)`. Build + bench BSA path. Wall ≈ 125s (NOT a reachable target — see § What lap-30 PROVED DEAD).
- (b) v2 compaction reference: see `git log --oneline kobbler-lap30-attempts-2026-05-28` (if tagged) or rebuild from this doc's sketch.

---

## ONE-LINER reminders the next agent will need

- Eye-test: http://10.0.0.208:8011/
- Standard bench: 480×832, 25f, --steps 8 RESIDENT, --max-vram 9.
- Current shipped baseline: **142.5s** (lap-29.2). Lap-30 shipped nothing.
- BSA quality is OWNER-OK'd at r=1+self_frame (mild camera-like movement); enable per-render via `LONGCAT_BSA=1 LONGCAT_BSA_SELF_FRAME=1 LONGCAT_BSA_RADIUS=1`.
- Cumulative empirical ceiling for further mask-driven wins: **~2-3% wall**. Anything bigger needs a different lever class (ncols=128 in #3, or off-codepath optimizations).
- Owner mentality: *"no quitting unless 100% proven from all angles to not be a performance win."* Lap-30 100%-proved per-K-col-chunk and in-kernel kb0-compaction. Lap-31 has three live paths (cross-CTA share, CPU bitmap, ncols=128) — owner direction needed before starting.
- Bump launch_bounds occupancy ONLY for L1TEX-stall-bound kernels. MMQ, MMF, MMVQ all measured DEAD or worse. Lap-29.1's FA occupancy bump remains the only valid case found.
- Commit each win: submodule-first (ggml), then bump parent. Bit-exact PSNR 99.00 every time, OR get owner OK on quality trade.
