# LongCat-Avatar.cpp — DiT PERF HANDOFF (lap-30, the levers that are STILL left)

*Written end of lap-29 (2026-05-28). Lap-29 shipped TWO things:*
*- **lap-29.1**: FA launch_bounds occupancy 2→3 = **−8.5% wall bit-exact** (155.11→142.03s mean, 480/25f/--steps 8 RESIDENT)*
*- **lap-29.2**: BSA self_frame/bookend anchors + sparse FA (whole-tile early-skip on -INF mask). Owner-validated quality config exists (r=1 + self_frame, "almost as good as no BSA fiddling, mild camera-like movement at 93f"), but whole-tile sparse skip only recovers the BSA mask-add overhead — BSA r=1+sf with sparse kernel matches dense baseline 142.5s, no net win. Always-skip probe shows ~17s more available with per-K-column skip.*

*Lap-30's PRIMARY MISSION: write the per-K-column sparse-FA kernel to unlock that remaining ~12% wall. **BSA without that perf win is a quality regression with no upside — do not declare BSA "shipped as configurable quality knob" until per-column skip lands.** Read this doc first; supersedes HANDOFF-DiT-lap29.md.*

---

## ⏱️ FIRST FIVE MINUTES

```
curl -sI http://10.0.0.208:8011/    # serve_clips.py should answer HTTP 200
# if dead:
cd ~/dev/longcat-avatar.cpp && nohup python3 tools/serve_clips.py --dir build --port 8011 \
  > /tmp/serve_clips.log 2>&1 & disown
```

http://10.0.0.208:8011/ — **`lap29_sparse_final.webm`** is the lap-29.2 dense baseline reference clip (142.99s, PSNR 99.00 vs lap-27). The BSA quality reference is **`lap29_bsa_r1_selfframe.webm`** (or the sparse-kernel bit-exact clone `lap29_sparse_final_bsa.webm`) — what owner OK'd as "mild camera movement, have as option for clips."

Standard bench command:
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

---

## 🔥 MOOD / MANDATE — READ TWICE, DO NOT SKIP

**MENTALITY:** *"Go mental — get it faster than it should ever be on this hardware. Custom kernel for wins is our bag."* Owner is explicit on this lap: *"let's multi-day per-column skip to get the 12%. no quitting unless 100% proven from all angles to not be a performance win."* The lever cost in days is NOT a reason to walk away — it's the work, not the excuse. "The kernel needs surgery" / "the MMA tile is atomic per-block" / "the K-col compaction needs gather/scatter" — these are the actual problems to solve.

**EVERY FLOOR CLAIM HAS BEEN DISPROVABLE:**
- "im2col is at roofline" → was 7% BW (lap-21)
- "MUL_MAT is floored" → was 7× redundant cond work (lap-26)
- "FA kernel needs fork-class FA2/FA3 surgery" → was a **one-integer launch_bounds change** (lap-29.1: −8.5% wall)
- "stock ggml flash applies mask post-QK so BSA can't help" → TRUE, but the cheap fix (whole-tile early-skip) only got us back to baseline — **the real fix is per-K-col skip, which requires inside-the-MMA-loop work** (lap-30 mission)

**Measure. Do not predict.** The always-skip probe (force `return` inside the iter when mask is present) measured the theoretical ceiling at **125.32s = -12% on top of lap-29.1**. That's the target. The honest path between current 142.5s and 125.3s is per-K-column skip, kernel work, several days.

**Gates (mandatory, every change):**
- Bit-exact: `tools/clip_compare.py <base> <new>` reads **PSNR 99.00 mean+min**.
- Quality trade (needs owner OK): BSA gate is owner-OK'd at r=1+self_frame.
- Standard bench: **480×832, 25 frames, --steps 8, RESIDENT, `--max-vram 9`** — wins MUST show there.

**ONE GPU** (RTX 3060 / sm_86 / 12 GB). Stop prod acestep / tts / llama before heavy runs. `docker rm -f longcat-avatar-iter` strays. Never two GPU jobs at once.

---

## What lap-29 SHIPPED

| commit | tag | what | wall (resident, 480/25f/--steps 8) |
|---|---|---|---|
| `72747ac` parent / `90670f7a` ggml | `kobbler-lap29.1-fa-occupancy-3-2026-05-28` | FA MMA `__launch_bounds__` occupancy 2→3 for DKQ=DV=128 ncols=64. nvcc fits 3 blocks/SM with zero register spills — free latency hiding for the L1TEX-stall-bound kernel. | 155.11s → **142.03s** mean of 3 = **−8.5%** (sampling 121.04 → 107.65s); PSNR 99.00 |
| `8d9b71f` parent / `11341ed5` ggml | `kobbler-lap29.2-bsa-quality-knob-sparse-fa-2026-05-28` | BSA self_frame + bookend anchors (env-gated) + sparse-FA whole-tile early-skip kernel. BSA r=1+sf becomes "free" vs dense, but no NET win — see Limitations below. | dense 142.99s (unchanged); BSA r=1+sf 142.49s (was 150.14s pre-sparse-kernel) |

**Cumulative since lap-27 baseline:** 159.03s → **142.49s** = **−10.4%** over 6 ship laps.

---

## 🎯 LAP-30 PRIMARY MISSION — per-K-column sparse FA kernel

**Target wall:** 142.5s → **~125s = −12%**. Measured via always-skip probe (see below). Bit-exact gate (PSNR 99.00) — skip math is exp(-INF)·anything = 0.

**Why whole-tile skip wasn't enough (lap-29.2 measurement):**

The lap-29.2 kernel checks "is ANY cell in the Q-tile × K-tile slice allowed?" and skips the iter if not. For our shape (Q tile = 64 Q rows spanning multiple spatial cubes, K tile = 64 K rows in one frame), the Q tile's UNION of allowed K positions is large enough that most K tiles have AT LEAST ONE allowed cell → can't whole-tile-skip. Measured:

| config | wall | what |
|---|---|---|
| dense (no mask) | 142.99s | baseline |
| BSA r=1+sf, sparse kernel (lap-29.2) | 142.49s | recovers mask-add overhead, no NET win |
| **always-skip probe** (force return when mask present) | **125.32s** | the 12% ceiling — perf upper bound |

The probe is wrong-output but proves the K-tile compute that CAN be elided. The 17s gap between real-skip and probe = compute on tiles with SOME allowed cells but mostly deny. Each such tile, the MMA does 64×64 work but most cells contribute exp(-INF)=0 to softmax → wasted work.

**The kernel work:**

Inside `flash_attn_ext_f16_iter` (`ggml/src/ggml-cuda/fattn-mma-f16.cuh`, the function I added the whole-tile skip to in lap-29.2), the K iteration runs MMA tiles of `T_A_KQ::I × T_A_KQ::J` (16 × 8 on Turing+). Per-K-column skip means: for each MMA tile chunk of K columns, check if ALL Q rows have that K range fully denied. If yes, skip the MMA call for that chunk; bookkeep `KQ_C[]` is initialized to 0 from previous untouched state (verify this assumption — the accumulator may need explicit zero-init if we're skipping a chunk that would otherwise contribute).

Three sub-approaches in order of incremental difficulty:

### (a) Per-MMA-chunk skip — simplest entry

Inside the K loop body at lines ~669-695 of `fattn-mma-f16.cuh`, before each `mma(KQ_C[...], K_A, Q_B[...])` call, check the mask slice for the K columns this MMA chunk covers (T_A_KQ::J = 8 K cols per chunk). If all -INF across the Q-tile's rows that the MMA tile's output covers → skip the load_ldmatrix + mma.

The K_A load_ldmatrix is from `tile_K + i_KQ_0*stride_tile_K + (k_KQ_0 - k0_start)`. So per-chunk check needs the mask block for (Q rows of this tile, K cols [k_KQ_0, k_KQ_0+8)).

Bit-exact: skipping the MMA doesn't update KQ_C[], so the partial sum is whatever was there. Need to verify KQ_C[] starts at 0 each iter (it does — declared per-iter on the stack).

Expected savings: 60-70% of K-tile compute on mask-present iters → meets the 12% target.

### (b) K-column compaction — biggest restructure

Preprocess the mask once per attention call into a "column survivors" index list. The K loop iterates only over surviving columns. Requires:
- New scratch buffer for compacted K column indices
- Modified K tile load to gather from non-contiguous K columns
- Modified mask load to gather correspondingly

More invasive but generalizes beyond BSA (causal attention, arbitrary user masks). The MMA structure stays intact — just sees a denser K input.

### (c) Pre-grouped K reordering — unlikely worth it

Sort/group K positions so allowed K's are contiguous. Avoids gather. But shuffles K's RoPE positions (the avatar uses 3D-RoPE on K — re-ordering breaks the position encoding). Probably DOA.

**Recommended start:** (a). Smallest change, highest confidence in correctness, single-file edit. Then measure. If (a) gets us to ~130s (most of the way), maybe (b) is unnecessary. If (a) only gets us to ~138s (the per-chunk reductions add overhead that eats the savings), pivot to (b).

**Verification ladder:**
1. Build with (a). Run dense bench — should still be 142.99s (no mask = no overhead path).
2. Run BSA r=1+sf bench. Expect 130-135s if (a) works.
3. PSNR check vs `lap29_bsa_r1_selfframe.webm` — must be 99.00.
4. PSNR check dense vs `lap28_lap27baseline.webm` — must be 99.00.
5. If both PSNR clean + wall drops, commit + tag `kobbler-lap30.1-sparse-fa-per-col-2026-MM-DD`.

---

## What FAILED in lap-29 (do not re-burn)

(see HANDOFF-DiT-lap29.md for the full list — most carried forward; new dead-ends from this lap):

- **MMQ launch_bounds occupancy 1→2**: +27% regression. MMQ is smem-bound, not L1TEX-stall-bound. **Rule: only bump launch_bounds for L1TEX-stall-bound kernels.**
- **FA Q_in_reg=false**: undoes lap-29.1 win entirely (back to 155.62s). Q-in-registers is load-bearing.
- **FA `LONGCAT_FA_NCOLS1=32`**: 10× slower. Smaller per-block Q-tile doubles K HBM traffic. The lap-26 dev knob comment claiming "smaller → higher occupancy" had directionality wrong.
- **FA occupancy=4**: +1.4% spill regression vs occupancy=3.
- **Audio cross-attn KV cache (lap-26 pattern)**: F32 buffer (151 MiB) OOMs at --max-vram 9; F16 buffer (75 MiB) fits, +0.5s wall (within noise), PSNR mean 44dB / min 38dB. Audio kv_linear chain too small to measure above noise floor.
- **BSA bookend anchor**: owner verdict, worse than self_frame.
- **BSA whole-tile sparse FA**: recovers mask-add overhead but no NET win vs dense. **See lap-30 mission for the per-K-column followup.**

---

## The levers that are LEFT for lap-30 (ranked)

### 1. **Per-K-column sparse FA kernel** — THE primary mission (12% ceiling)

See § Lap-30 Primary Mission above. Owner explicit: *"no quitting unless 100% proven from all angles to not be a performance win."*

### 2. **Text cross-attn KV cache** (lap-26 pattern, ~0.4% wall)

Same plumbing the audio attempt proved out. Text K/V is [head_dim=128, num_heads=16, n_ctx=512] = ~96 MiB F32 (fits under --max-vram 9, unlike audio's 151 MiB OOM). Cacheable since context is step-invariant. Lap-26 cond-kv code pattern in `src/longcat_avatar.hpp:1387-1408` (`ensure_condkv_cache`) and `:1631-1648` (build_graph cache writes) is the template.

### 3. **MMF / MMVQ launch_bounds audit** (after #1)

`mmf.cuh:49, :298` and `mmvq.cu:395, :601` are at `, 1)` launch_bounds. Try bumping each to 2 with bench-and-revert. **Skip if it spills or regresses** — MMQ taught the rule. Could be 1-3% each if applicable.

### 4. **FA combine kernels** (`fattn-common.cuh:625, :678, :758, :864`)

Each FA call ends with a small combine/reduce. Bumping occupancy is cheap; estimated <0.1% wall. Only worth a batch with #3.

### 5. **FA `nbatch_K2` / `nbatch_V2` / `nbatch_fa` tuning** for our DKQ=DV=128 ncols=64 case

Lap-29.1 only changed `occupancy`. Other config knobs in `fattn-mma-f16.cuh:62` (`nbatch_K2=64`, `nbatch_V2=64`, `nbatch_fa=64`) might have headroom now occupancy=3 freed register budget. Sweep 32/96/128 for each, rebuild + bench. Risk: regressions / spills. ~30 min per experiment.

### 6. **ncols=128 case** (per-head Q-batch persistence, handoff lap-29 #2b)

Each block handles 2× more Q rows = 0.5× K HBM reads. Requires new template instantiation in `fattn.cu`. ~4-8h work. Realistic ~1-3% wall on top of lap-29.1. Try AFTER the per-K-col kernel.

### 7. **Dead-ends do not re-burn (carry forward)**

ALL of these have been measured dead — see HANDOFF-DiT-lap29.md for the full list. New in lap-30: BSA whole-tile skip is recovered (lap-29.2) but is not the perf win — see §1 for the actual path.

---

## Method (reproducibly)

**Build:** `~/dev/kobbler/docker/longcat-avatar-dev/iter.sh build` (~18-30s incremental, sm_86, ccache).

**Standard bench:** 480x832, 25f, --steps 8, resident (no --offload-to-cpu), `--diffusion-fa --seed 42 --clip-on-cpu --max-vram 9`. Use `/tmp/render_bench.sh /src/build/<NAME>.webm "<env kvs>" --steps 8`.

**Bit-exact gate:** `python3 tools/clip_compare.py build/lap28_lap27baseline.webm build/<NEW>.webm` — read PSNR 99.00 mean+min. For BSA changes also compare vs `lap29_bsa_r1_selfframe.webm` (BSA reference).

**Profile:** `LONGCAT_OP_PROFILE=1`. Post-lap-29.1 per-step:
- MUL_MAT 55.8% (Q4_K MMQ — proven floored, do NOT bump)
- FLASH_ATTN_EXT 30.5% — the lap-30 target (per-K-col skip = ~12% more)
- ADD 4%, CONT 3%, SCALE 2%, CONCAT 2%, MUL 2%, ROPE_PE 1.5%, UNARY 1%

**Always-skip probe (re-establishing the ceiling):** Edit `ggml/src/ggml-cuda/fattn-mma-f16.cuh`, the nstages<=1 branch's `if (!any_allowed) return;` → `if (true) return;`. Build + bench BSA path. Output is wrong (garbage) but wall is the perf ceiling. Lap-29.2 measurement: 125.32s.

---

## ONE-LINER reminders the next agent will need

- Eye-test: http://10.0.0.208:8011/
- Standard bench: 480×832, 25f, --steps 8 RESIDENT, --max-vram 9.
- Current shipped baseline: **142.5s** (lap-29.2). Target: **~125s** via per-K-col sparse FA.
- BSA quality is OWNER-OK'd at r=1+self_frame (mild camera-like movement); enable per-render via `LONGCAT_BSA=1 LONGCAT_BSA_SELF_FRAME=1 LONGCAT_BSA_RADIUS=1`.
- BSA only ships as default-on IF the per-K-col kernel lands a measurable wall win.
- Owner mentality: *"no quitting unless 100% proven from all angles to not be a performance win."* Multi-day kernel work is fine — that IS the work.
- Bump launch_bounds occupancy ONLY for L1TEX-stall-bound kernels. MMQ taught the rule the expensive way (+27% regression).
- Commit each win: submodule-first (ggml), then bump parent. Bit-exact PSNR 99.00 every time, OR get owner OK on quality trade.
