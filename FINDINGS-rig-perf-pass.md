# FINDINGS — rigger performance pass (2026-06-17)

**Goal:** beams=10, fp32, **maxnew=2048** budget (kills the 2/10 runaways), **as fast as possible**,
**sub-7.5 GB VRAM**. Judge mathologically vs goldens, not vibes. Worktree `~/dev/longcat-sparse-spike`,
`$CP = tools/m1_ref/cpp_port`. RTX 3060 (11.9 GB). All numbers from the e2e's own instrumentation
(cudaMemGetInfo true peak, steady_clock timing) + `rig_score` + nsys.

## TL;DR
| metric | baseline (fp32 KV, per-beam prefix, sequential) | **final (f16 KV + shared-prefix + 2-seg batched decode)** |
|---|---|---|
| maxnew=2048 fits? | **NO — ~14 GB OOM** | **YES** |
| VRAM peak @2048 | ~14 GB (OOM) | **4513 MiB (4.4 GB)** (f16-KV + shared-prefix + single-buffer reorder; under 7.5 ✓) |
| VRAM peak @900 (measured) | 8455 MiB | 3915 MiB |
| decode speed (clean rig) | 5.2 tok/s | **~30–32 tok/s (≈6×)** |
| decode speed (long/malformed) | ~3 tok/s | 16.4 tok/s |
| runaways @2048 | n/a (couldn't run) | **0/10** (all seeds terminate) |
| quality | locked baseline | tightest/most Python-like distribution (see below) |

The model is small (**~425 M params, 1.7 GB fp32 on disk, ~1.7 GB resident** — embed_tokens stays
host-side). **VRAM was never the model — it was the KV cache:** 2·num_beams+1 = 21 caches × 589 MB
(fp32, max_seq 2570) = ~12.4 GB at maxnew=2048. The wins are all KV-cache-shaped.

## What "fp32" actually meant (the reframe)
`prec=fp32` was chosen because **bf16 *free-running* diverges into chaos** (22–89 joints) — a real
finding about *matmul precision* over hundreds of compounding steps. But it had been applied to two
independent axes: (1) the **KV-cache dtype** (the VRAM hog) and (2) **weight/matmul precision** (only
~1.7 GB, NOT the hog). Lever 1 below splits them.

## Lever 1 — f16 KV cache  (`qwen3_decode.hpp`)
K/V stored F16, matmul accumulates F32 (set_prec). MORE faithful than our old f32 — Python's KV is bf16,
and f16 has more mantissa than bf16. Toggle: `RIG_KV_F16=0` → f32 (A/B).
- VRAM @900: cache pool 6673→3397 MiB, peak 8455→5133 MiB.
- **Unblocks 2048** (fits) and **kills the runaway**: seed=9 (locked-baseline `cap` runaway) now
  terminates at 2048 (J=78, eos, only 705 tokens — it just needed budget to build its skeleton + emit eos).
- It is NOT bit-exact to fp32 KV (seed=1 @900: 62→70 joints) — expected: f16 rounding reshuffles which
  trajectory each RNG seed lands on. Judged by distribution, not per-seed.

## Lever 2 — shared-prefix KV  (`qwen3_decode.hpp`, threaded `pre`/`pre_len`)
The P=514-row mesh_cond+start prefix is identical across all beams → stored ONCE in a read-only `pre`
cache; each beam holds only its generated suffix; attention concats `pre[0,P) ++ suffix ++ new`.
**Bit-exact** (same KV bytes, deduplicated): seed=1 @900 reproduced J=70, gen=599, normscore −0.9855
to 4 decimals. Also drops 514 rows from every fork.
- **VRAM peak @2048: 7779 → 6435 MiB (6.28 GB).** Under 7.5 GB by both GiB and decimal interpretations.
- Pool: 2·num_beams suffix caches @ (maxnew+4) + 1 shared prefix @ P (the old +1 template is gone —
  no prefix to fork at step 0).

## Lever 3 — batched-beam decode  (`qwen3_batched.hpp`, `rig_beam_generate_batched.hpp`)
nsys on the sequential decode: launch-bound + matmuls running as matrix×**vector** (M=1, bandwidth-bound)
= 49% of GPU, because the 10 beams decode **sequentially**. All active beams are always at the same
position, so decode ALL beams in ONE forward (batch dim = beams): projections become M=10 matmuls
(tensor-core / `mul_mat_f`), launches drop ~8× per token. GQA via mul_mat broadcast (`i02=i12/n_rep`)
is identical to explicit repeat_kv. Default-on; `RIG_BEAM_SEQ=1` forces the sequential path.
- **3.2× faster: 5.2 → 16.6 tok/s** (clean rig; R3 134s → 24s @2048).
- NOT bit-identical to sequential (different GPU kernel for M=10 vs M=1 → different fp accumulation
  order). **Logit probe proves faithfulness:** step-0 beam-0 decode logits agree to ~1×10⁻⁴ relative
  (sum −10772.9 vs −10778.9, max 13.6461 vs 13.6437); the argmax flip (137→136) is on a near-tied pair
  (Δmax 0.0024) — that nudge → different token → divergent free-run = the documented hypersensitivity,
  NOT a bug (a bug gives garbage/NaN, not a Python-like distribution).

### Lever 3b — 2-segment broadcast attention (on top of batched) — KEPT (the winner)
nsys on materialized batched showed the matmuls are now efficient, but K/V *assembly* (prefix repeat +
concat, every step every layer) is the new ~48% bottleneck. The 2-segment attention reads the shared
prefix directly via mul_mat broadcast (beam=1→B, GQA nkv→nh; NO per-beam materialization, NO big K/V
concat) for one score block, computes the suffix score block separately, concats only the small score
vectors for a joint softmax, then sums the two value-matmuls. Output split is a sum-of-two-matmuls vs one
(~1e-6 fp diff on top of the kernel-dispatch diff; logit probe: sum −10779.8 vs seq −10772.9, still ~1e-4).
- **~1.85× over materialized batched on clean rigs: 16.6 → ~30–32 tok/s** (≈6× the sequential baseline),
  and 9.8 → 16.4 tok/s on long/malformed runs. Same 6407 MiB VRAM.
- **Distribution got TIGHTER, not worse:** J = [184,53,54,51,40,53,52,44,83,54] — 8/10 in 40–54 (squarely
  in Python's 42–56), 1 larger (83), 1 malformed (184). 10/10 terminate, 0 runaways. This is the closest
  match to Python @2048 of all three variants (sequential 23–81, materialized 23–76).

## Mathological validation — 10-seed sweep @ maxnew=2048 (golden gilly cond)
Sequential (bit-exact-to-f16 oracle): J = [70,188,51,23,24,49,81,57,78,56], **10/10 terminate, 0 runaways**,
all @ 6435 MiB. 1 malformed (seed2 J=188, sym 0.14).
Batched (materialized): J = [46,43,76,51,49,182,54,23,187,50], **10/10 terminate, 0 runaways**,
6407–6483 MiB, clean rigs rig_TOTAL 0.65–0.89, 2 malformed.
Python @2048 reference: [malformed,55,54,55,53,44,42,malformed,45,56] — 8 clean in 42–56 + 2 malformed.
- Ours: all terminate, 0 hard runaways (was 2/10 at maxnew=900), ~2 malformed like Python; clean spread
  wider than Python's tight 42–56 (fp/seed differences). Per the no-winners rule: owner judges rigs by
  RENDER (`web/rig_compare.html` / `skel_ab.html` :8013) + rig_score; these are observations.

## Profiles (nsys, RTX 3060)  — `$A/perf/rig_nsys*.nsys-rep`
- Sequential decode (maxnew=20): 199,863 kernel launches; mul_mat_vec M=1 = 49% GPU; launch overhead 23%.
- Batched decode (maxnew=30): launches/token ~8× lower; matmuls → cutlass tensor-core + mul_mat_f(M=10);
  new top costs = concat (42%) + cpy (18%) + repeat (6.6%) → motivated lever 3b (2-segment).
- ncu (kernel HW counters) BLOCKED on this host: `ERR_NVGPUCTRPERM` (perf counters admin-restricted;
  the `--cap-add SYS_ADMIN` fix is docker-only — native needs a root/module-param change). nsys (timeline,
  no restricted counters) covered the per-stage bottleneck identification + confirmed each fix.
  Real ncu binary = `$TOOL/nsight-compute/2024.1.1/ncu` (the `$TOOL/bin/ncu` wrapper is broken).

## Files touched
- `qwen3_decode.hpp` — f16 KV (RIG_KV_F16 toggle) + shared-prefix `pre`/`pre_len` threading.
- `qwen3_batched.hpp` (new) — batched cache + batched forward (2-segment attention).
- `rig_beam_generate_batched.hpp` (new) — batched beam loop (reuses all sequential host logic).
- `rig_beam_generate.hpp` — VRAM peak instrumentation + RIG_LOGIT_PROBE.
- `skintokens_e2e.cpp` — batched default beam path, RIG_BEAM_SEQ=1 fallback.
- Harness: `$A/perf/{sweep_2048.sh, batched_validate.sh}` + logs.

## Docker prod image + CUDA-graphs + ncu floor analysis (2026-06-17)
Host is CachyOS (GLIBC 2.43) → host-built binaries don't run in stock containers (same reason prod cpp
services build in-docker). Built a reusable image `rigprof:12.4` (`docker/rigprof/`: Dockerfile +
`build_in_container.sh` + `run.sh`) = `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` + cmake/build-essential
(CUDA 12.4 == host toolchain). In-container builds ggml (into `ggml/build-cuda-docker/`, cached in worktree)
+ `skintokens_e2e_docker` + `rig_score_docker`, self-consistent GLIBC. Gotchas: pin `/usr/bin/{gcc,g++}`
(image's `/usr/lib/ccache` wrappers are dangling) + `-I/usr/local/cuda/include` for the g++ link.
- **CUDA graphs**: host ggml is `GGML_CUDA_GRAPHS=OFF`; docker build flips it ON. **But graphs-ON ALONE
  does nothing here** (seed5 compute 21.88→22.15 ms, bit-identical J=40): our decode graph is rebuilt with
  GROWING klen every step → topology changes → ggml can't reuse the captured graph. Landing graphs needs a
  FIXED-SHAPE decode (pad KV to a stride + runtime mask, qwen3-tts kv_n_eff) so the topology is stable.
- **ncu (works in-docker via `--cap-add SYS_ADMIN`)**: the cutlass gemms run at **8–17% occupancy,
  31–40% Compute(SM), 12–23% DRAM — nothing saturated**; element-wise ops (silu/mul/add/rmsnorm) are
  80–102% occ / 85–90% mem-bound (fine). ⇒ decode is **latency/launch-bound + underutilized** (M=10 is
  small for tensor cores), NOT at compute floor — same signature as LTX/Wan ncu. nsys: GPU busy ~62% of
  wall ⇒ ~38% launch gaps. So the CUDA-graph ceiling ≈ recover that 38% (~1.5× → ~45 tok/s, ~9s clean rig).

## CUDA-graph lever = MEASURED DEAD (2026-06-17) — decode is compute/latency-bound, not gap-bound
Before building the bucketed fixed-shape + CUDA-graph rewrite, measured the per-step GPU-busy vs wall
(nsys kernel+mem sum, prefill-baseline subtraction: maxnew=60 minus maxnew=2, /58 steps):
- **per-step GPU-busy = 16.9 ms ≈ per-step compute WALL 16.6 ms ⇒ GPU ~100% busy, ~no launch gaps.**
  (The earlier "62% busy / 38% gaps" was the PRE-2seg version; 2-seg removed the big concat kernels.)
- CUDA graphs only recover launch GAPS → here the ceiling is ~1.0–1.1×. NOT worth a multi-hour, uncertain
  (set_rows + bucketing + double-buffer-two-graphs) rewrite. **Decision: do not build it.**
- ncu: kernels are BUSY but UNDERUTILIZED (gemms 8–17% occ, DRAM 12–23%, Compute 31–40% — nothing
  saturated) = LATENCY-bound at M=10 (beams=10 caps warp parallelism). This also means weight quant
  (Q8/Q4) won't speed decode (not DRAM-bound) — only saves VRAM, which we don't need. Same floor as LTX/Wan.
- The set_rows pattern from qwen3-tts IS portable (input-indexed write keeps topology fixed) — it just
  doesn't help a workload with no gaps. ~30 tok/s clean (≈6×) is the practical floor for beam-10 fp32
  autoregressive decode on the 3060.

## VRAM floor pass (2026-06-17) — current 6.4 GB breakdown was 73% KV double-buffer
At maxnew=2048: KV double-buffer 4.70 GB (73%) + fp32 weights 1.70 GB + shared prefix/transients ~0.06 GB.
- **#1 single-buffer in-place reorder (DONE, BIT-EXACT): −1.9 GB → 4513 MiB.** The 2nd full suffix
  buffer was pure reorder write-safety overhead. Replaced with ONE buffer + a 1-column temp; the per-step
  beam permutation is applied in place (skip-if-source-still-needed + 1 temp column breaks cycles; handles
  duplicate parents + drops). Bit-exact vs the double-buffer (seed1 normscore −2.1177/J184/gen1910,
  seed9 J83, seed5 J40 all reproduced). Speed unchanged (`reorder_inplace` in rig_beam_generate_batched.hpp).
- **#2 f16 weight storage (RIG_W_F16): −0.69 GB → 3825 MiB, but QUALITY REGRESSION → left OFF by default.**
  weight() stores big matmul weights F16 (norms stay F32), matmul still F32-accumulate (lin path). The
  10-seed sweep degraded vs fp32-weights: clean J spread 34–66 (vs tight 40–54), 2 malformed (vs 1), and
  2 lopsided/asymmetric clean rigs (seed5 J36 sym 0.29, seed7 J34 sym 0.52) — the milder bf16-style drift.
  Since we're already at 4.5 GB (far under 7.5), the 0.69 GB isn't worth worse rigs. Env-gated `RIG_W_F16`,
  default OFF; only use if VRAM becomes critical (e.g. bigger model/context).
- **Landed VRAM floor (practical): 6.4 → 4.5 GB via #1 alone (bit-exact, no quality cost).** Absolute floor
  (paged/trie KV dedup of sibling history) ~2.5 GB remains a major rearchitecture — not pursued.

## Not done / next levers (low ROI)
- weight quant (Q8/Q4): VRAM only (~0.9 GB), NOT a speed lever here (latency-bound, DRAM 12–23%); risks
  bf16-style drift. Skip unless VRAM becomes the constraint again.
- Fundamentally faster decode would need fewer/bigger kernels (op fusion) or more parallelism than beams=10
  allows — diminishing returns vs the 6× already banked.
- All code UNCOMMITTED in the worktree (commit on owner request). Docker image `rigprof:12.4` + `docker/rigprof/`.
