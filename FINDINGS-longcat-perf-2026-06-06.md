# longcat-avatar perf findings — 2026-06-06 (RTX 3060, prod config)

Bench: `sd-cli -M vid_gen`, 480×832, 25f, 8 steps, `--offload-to-cpu --vae-tiling --mmap
--max-vram 7.5 --diffusion-fa`, prefetch thread ON, prefetch pool OFF (prod). Harnesses in
`kobbler/docker/longcat-avatar-dev/`: `ab_ggml_bump.sh`, `mem_vram_trace.sh`, `nsys_concat.sh`,
`breakdown.sh`, `ncu_hot.sh`. nsys/ncu toolchain: `/mnt/hdd/3d/avatar-shootout/toolchain`.

## 1. SHIPPED-READY: ggml bump 4e3ee737 → 292516d5 (= dbrain/ggml master)
A/B (2 runs each, same source, clean GPU):
| metric | 4e3ee737 (prod pin) | 292516d5 (master) | Δ |
|---|---|---|---|
| DiT sampling | 239.1s | 239.1s | 0% |
| VAE decode | 19.33s | 18.21s | **−1.12s (−5.8%)** |
| full render | 293.7s | 292.8s | −0.3% |
| VRAM peak | 9523 MiB | 9523 MiB | 0 |
| host VmRSS/Anon/File | 15.15/0.69/14.50 GB | identical | 0 |

Mechanism (nsys): Wan-VAE dim-2 concat was a launch storm — `concat_T_cont<float,2>` **714,260
launches** (1.69s GPU) → master `concat_T_cont_4d` **2,550 launches** (0.83s GPU). Total concat
5.00s/716,690 → 3.95s/4,980. madd-fuse is a no-op for longcat (gate_add already broadcast-fused).
**Pure win / no loss.** To ship: bump ggml submodule in hbd-longcat-avatar.cpp master 4e3ee737→292516d5,
push, bump LONGCAT_REF, rebuild image. (Note: outputs are NOT byte-stable run-to-run — flash-attn
non-determinism — so webm md5 can't prove bit-exactness; concat fold is bit-identical by construction.)

## 2. HIGH-VALUE LEVER: autofusion is silently DISABLED in the offload path
**Finding:** ~19% of DiT GPU time runs as standalone elementwise/norm kernels the fusion suite is
designed to collapse, because `ggml_node_get_use_count()` returns **0** for every node in the
offload graph-cut sub-cgraphs, and every multi-op fusion gates on `use_count==1`.

Evidence (all from prod-config renders):
- `LONGCAT_DEBUG_NORM_FUSE=1`: **484/484 NORM ops rejected**, all `use_count NORM=0 MUL=0 (need ==1)`.
- nsys trace: **zero fused kernels** (no `norm_fused_add`, `mul_add_bcast`, `fused_madd`,
  `rms_norm_fused`); the residue is standalone: k_bin_bcast add 6.0%/19,760, mul 5.0%/13,924,
  rms_norm 2.9%/6,212, scale 2.5%/10,426, norm_f32 1.5%/3,856, silu 1.0%/4,616 = **~19% GPU**.

Root cause: `ggml_node_get_use_count` (ggml-impl.h:629) reads `cgraph->use_counts[]`, populated only
by `ggml_build_forward_expand` (ggml.c:7202-7215). The longcat graph-cut path (ggml_graph_cut.cpp,
`compute_with_graph_cuts`) executes sliced sub-cgraphs whose `use_counts` are not (re)populated → 0.
The matchers live in ggml-cuda.cu:4450 (RMS_NORM+MUL[+ADD]), 4484 (LayerNorm modulate view-skip),
~4600 (mul_add_bcast gate_add), 4694 (same-shape madd) — all gated on `use_count==1`.

Fix (careful, must stay bit-exact + segment-safe): repopulate / propagate `use_counts` for the
graph-cut sub-cgraphs so *within-segment* norm→mul→add chains fuse while genuine cross-segment-boundary
chains correctly don't. The `use_count==1` gate exists to prevent unsafe in-place fuse-overwrite when
the intermediate is read elsewhere — preserve that intent (don't just relax `==1` to `<=1`; a
boundary tensor read by the next segment is a real external use). Validate: NORM-FUSE-DBG shows
fusions firing + nsys shows fused kernels + PSNR-identical output. Estimated recoverable: the
redundant full-activation memory passes of norm/scale/shift/gate (these elementwise kernels are
memory-bound, unlike the occ-bound matmuls) — meaningful, measure with the A/B harness.

Secondary, independent lever: longcat FFN does `ggml_silu(w1) * ggml_mul(w3)` as 2 ops; ggml already
has fused `ggml_swiglu_split` (GGML_OP_GLU, CUDA kernel in unary.cu). Swap at longcat_avatar.hpp:822/830
→ one fused kernel (kills the SwiGLU silu+mul), bit-exact (same silu(a)*b math). Independent of the
use_counts fix.

### use_counts fix — IMPLEMENTED + VALIDATED 2026-06-06 (uncommitted, in working tree)
Fix is in `src/ggml_graph_cut.cpp::build_segment_graph` (NOT shared ggml — per-fork file). After adding
the segment's internal nodes, populate the fresh segment graph's `visited_hash_set` + `use_counts`:
(1) register each internal node, count 0; (2) count uses among segment-internal nodes only (scoped
mirror of ggml.c `ggml_visit_parents_graph`); (3) +1 for every `segment.output_node_indices` node
(external/boundary consumer — keeps that intermediate materialized). Preserves the `==1` gate's intent
(NOT relaxed to `<=1`) — boundary tensors get count≥2 and stay unfused. ~35 lines, uses
`ggml_hash_insert`/`ggml_hash_find`/`ggml_bitset_get` already reachable via `../ggml/src/ggml-impl.h`.

Validation (RTX 3060, prod config 480×832×25f 8steps, `--offload-to-cpu --diffusion-fa --max-vram 7.5`,
DiT genuinely segments 146→8/9):
- **NORM-FUSE-DBG**: before 484/484 NORM rejected `use_count=0`; after **968 NORM `uses=1`, ZERO rejects**.
- **nsys** (baseline `sd-cli.g292516d5` vs `sd-cli.fix`, 4 steps): baseline **FUSED kernels = (none)**;
  fix shows `mul_add_bcast_dim1_f32_kernel` (1056), `scale_cast_f32_to_f16` (1536), and the fused
  `norm_f32<…(1)(1)>` / `rms_norm_f32<256,(1)>` template variants now firing. Standalone residue
  collapses: k_bin_bcast mul 4.9%/8148→1.7%/3140, add 5.9%/11944→3.5%/8944, scale_f32 2.4%→1.8%.
- **Timing A/B** (2 runs each, FA on, 8 steps): DiT sampling **239.2s → 227.0s = −12.2s (−5.1%)**;
  VAE decode 18.22s unchanged; VRAM peak 9519 MiB unchanged (pool-on here; prod NO_PREFETCH_POOL=1 → 7575).
- **Bit-exact**: FA-off OOMs at 25f (13 GB attn-scores tensor, no FA) so md5-of-deterministic-render isn't
  available; instead all 4 FA-on webms are **pixel-identical, PSNR=inf** for every pair incl. base↔fix
  (webm container md5 differs run-to-run = muxer nondeterminism, NOT pixels). Both base & fix segmented,
  so the fix path was exercised → pixel-bit-exact, no quality change.

Binaries: `build/bin/sd-cli.fix` (with fix) vs `build/bin/sd-cli.g292516d5` (baseline, same ggml 292516d5).
Harnesses: `kobbler/docker/longcat-avatar-dev/{ab_usecounts_fix.sh,nsys_fix.sh}`.
**TO SHIP** (not done — needs user OK): commit the `ggml_graph_cut.cpp` change to hbd-longcat-avatar.cpp,
push, bump `LONGCAT_REF` in kobbler `docker/longcat-avatar/Dockerfile` + `docker-compose.yml`, rebuild +
redeploy the prod image. The fix is bit-exact + env-unconditional (always-on once shipped); flux2 gets
the same diff for parity but no measurable effect (its image DiT never segments — see flux2-dev HANDOFF).

## 3. AT THE FLOOR — do not chase (ncu-confirmed)
- mul_mat_q Q4_K (42% GPU): **achieved occupancy 16.66%**, Block Limit Registers=1 / Shared Mem=1,
  DRAM 39% SM 54%. Register+smem-bound — the MMQ floor (occupancy tuning measured-dead in prior laps).
- flash_attn (25% GPU): occupancy 25%, DRAM 2.5%, SM 47% — occ/latency-bound, not bandwidth.
- Offload solved: per-step OFFLOAD_PROFILE = 95% compute / alloc 524ms / overhead 217ms / **H2D 0ms**
  (lap-34 prefetch fully hides streaming). Per-segment compute-buffer cudaMalloc is not-poolable (prior lap).

## 4. VRAM / host-RAM breakdown
- VRAM peak (prod) **7575 MiB**, hard-capped by max_vram=7680 graph-cut budget (DiT→8-9 segments).
  Only droppable VRAM = the +1948 MiB prefetch *pool*, already off in prod (NO_PREFETCH_POOL=1; 9523→7575).
  Lowering max_vram adds segments (slower) — time/VRAM trade.
- Host VmRSS 15.15 GB = RssAnon **0.69 GB real** + RssFile **14.5 GB mmap (reclaimable)**.
  By file: DiT 8.94 + umT5 6.04 + audio 1.27 + vae 0.25 GB. umT5's 6 GB faults in at text-encode and
  sits resident/idle through all DiT sampling — only real drop candidate (madvise(DONTNEED) post-encode,
  −6 GB RssFile) but it's reclaimable cache already; the shipped umT5-free lever is mmap-exclusive.
  Host RAM is at the reclaimable-mmap optimum; nothing real to drop on a non-contended host.
