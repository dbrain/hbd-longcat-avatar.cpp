# Wan2.2 + InfiniteTalk — PERF/VRAM tuning handoff (RTX 3060)

**Mission:** make the Wan2.2-I2V-A14B + InfiniteTalk port good enough on the 3060 to render the
"music video" (directed singing shots + lip-synced song). The pipeline + CUDA build are DONE and the
quality is clean; this is a perf/VRAM-tuning job.
**PRIMARY GOAL = higher quality at the SAME or better speed** (not raw speed). Where a kernel/stage isn't
saturated, spend the slack on quality (higher-precision weights, less-lossy tiling), not just on going
faster. Port the LTX-2.3 levers we proved (FINDINGS-01..11 in `docker/longcat-avatar-dev/` of kobbler)
AND dig into the matmul itself (#1 below).

## Where everything lives
- **Worktree:** `/home/dbrain/dev/longcat-avatar-wan22` (branch `wan22-infinitetalk`, fork
  dbrain/hbd-longcat-avatar.cpp = stable-diffusion.cpp). Keeps prod LTX on `master` untouched. **All
  changes UNCOMMITTED** (user commits).
- **Models:** in worktree `models/` (~33GB, pulled from `10.0.0.151:~/dev/wan22-infinitetalk/models/`).
  `wan22-i2v-a14b-{low,high}-q4_k.gguf` (7.6GB ea), `longcat-umt5-xxl-q8_0.gguf` (5.6GB),
  `longcat-wan-vae-f16.gguf`, `infinitetalk-14b-q4_k.gguf` (10.7GB), `chinese-wav2vec2-base-f16.gguf`.
  Ref assets in `models/_drive/` (char.png = anime vtuber girl; song_16k.wav).
- **Build (host has NO CUDA — builds/runs in docker):**
  `docker run --rm --gpus all -v <worktree>:/src -v longcat-avatar-iter-ccache:/root/.ccache -w /src
   longcat-avatar-dev:builder bash -lc "cmake -S /src -B /src/build -DCMAKE_BUILD_TYPE=Release
   -DSD_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build /src/build -j\$(nproc)
   --target sd-cli sd-infinitetalk"`. Build is clean, NO guard fixes needed. **Drop `LONGCAT_NO_FUSED_ROPE` on GPU.**
- **ggml submodule @ 19727d01** (the LTX VAE-pad commit). Autofusion `use_counts` fix IS present.
- **Harness scripts (worktree root):** `perf_a14b.sh` (env-knob sweep, see below), `run_a14b.sh`,
  `run_it.sh`, `run_mvp.sh`, `gen_eyetest.sh` (eye-test page :8097). Results in `perf_out/sweep.csv`.

## perf_a14b.sh knobs
`LABEL=x SINGLE=0|1 OFFLOAD=0|1 CLIPCPU=0|1 VAECPU=0|1 TEMPORAL=0|1 TILE=0.5x0.5 MAXV=4.5 PINNED=0|1
 FR=21 GENV="GGML_CUDA_FORCE_MMQ=1 ..." ./perf_a14b.sh` → appends wall/peak/dit_s_it/buffers to sweep.csv.

## The reframe (READ THIS) — it's UNTUNED, not heavy
Wan-A14B DiT is **14B** (7.6GB/expert); LTX-2.3 DiT is **22B** (15.3GB). Apples-to-apples:

| | LTX-2.3 (tuned) | Wan-A14B (raw) |
|--|--|--|
| params | 22B | 14B |
| latent tokens L_q | 11960 | 9360 |
| res / frames | 720p / 97f | 480p / 21f |
| per-step | ~10.6 s/step (offload, **tuned**) | 12.4 s/it (resident, **raw**) |

LTX does **~2× the work/step** (params×tokens) yet is faster — because it has FINDINGS-01..11. So
Wan's slowness is missing tuning, not model weight. 12.4 s/it ⇒ only ~21 effective TOPS, below the
3060 ceiling. **CAVEAT: verify this headroom claim against the ncu result below before over-promising
— acestep/flux2 found their q4_k matmul occupancy-bound but AT FLOOR (mmq_x tuning was DEAD).**

## VRAM breakdown (A14B 480×832/21f, measured)
- **VAE compute buffer was the dominant allocation:** 4145MB encode / 2986MB decode > DiT 1889MB.
  Tight+temporal tiling (rel 0.25) cuts it to ~1553MB.
- **umT5 sits resident on GPU at 5.76GB** for the whole diffusion loop though it's needed once.
  `--clip-on-cpu` moves it off-GPU (frees 5.76GB) but CPU encode costs 16s.
- **Both MoE experts (15.2GB) alloc at ctx-init** → the resident-mode OOM cause (not single-model size).
- DiT: offload streaming **31 s/it** → resident **12.4 s/it** (FR=21) / **7.37 s/it** (FR=13).
- **81f single-shot OOMs** in BOTH modes (DiT activations grow with frames). Music video must
  **chain short 21-41f segments** (LTX director pattern), not render one long clip.

## Kernel A/B (done — NEGATIVE)
default 7.38 / FORCE_MMQ 7.36 / FORCE_CUBLAS 7.33 s/it — **identical**. Matmul kernel selection is NOT
a lever; default already optimal. Available knobs: `GGML_CUDA_{FORCE_MMQ,FORCE_CUBLAS,DISABLE_FUSION,
NO_MADD_FUSE,NO_FA,DISABLE_GRAPHS,NO_PINNED}`.

## ⭐ #1 PRIORITY — DiT matmul deep-dive (NOT closed; basic ncu only)
This is the single biggest cost center, so it gets the deepest dig — do **not** wave it away. Hot kernel =
`mul_mat_q<Q4_K(type 12), mmq_x=128, false>`. **Basic-set** metrics so far (5 launches, `perf_out/ncu_dit.log`):
- **Achieved Occupancy 16.66%** (8 warps/SM) — register-limited: **221 regs/thread → 1 block/SM**.
- **Compute(SM) 54.5% SoL, Memory 46%, DRAM 39.5%** — **neither unit saturated** ⇒ occupancy/latency-bound.

The same kernel *family* showed up in acestep/flux2 — that is a **HINT, not proof for Wan's shapes**. The
basic set + a sibling-project prior is NOT enough to call "at floor." Two things it already implies:
1. **Occupancy-bound, not compute/DRAM-bound** — the SM is warp-starved (1 block/SM), units idle on latency.
2. **⇒ QUALITY HEADROOM (the actual goal):** since no unit is saturated, there is likely slack to push
   **higher-precision weights (Q5_K / Q6_K / Q8_0) through ~the same latency budget** — buy quality for free.
   This is the north star (higher quality at same/better speed). The profile says it's plausible — **TEST IT**
   (re-quant the expert, re-ncu, compare wall + quality).

**REQUIRED deep breakdown (cheap — do it before any verdict):**
- `ncu --set full` / `--section ".*"` on the hot matmul: **WarpStateStats** (stall reasons: long-scoreboard=
  mem latency, MIO-throttle, barrier, wait...), **SchedulerStats** (issued/eligible/active warps),
  **InstructionStats** (inst mix — redundant dequant? int overhead? fp32↔fp16 converts?), **MemoryWorkload**
  (L1/L2 hit, bytes moved vs needed), **Source/SASS** counters (hot lines, per-instruction stalls).
- **Hunt for EXTRA WORK feeding/inside the matmul** (we have found this before): redundant dequant,
  padding waste (mmq_x=128 tile on small-N matmuls), unfused epilogue, transposes/layout copies, K-split
  overhead. List every matmul launch's N/K shape; flag any that waste the 128-wide tile.
- **Per-step time decomposition:** matmul vs FA-attention vs norm/rope/elementwise vs copy/transpose. No
  nsys in builder → `apt install nsys` in the builder, OR `ncu --launch-count <all> --metrics
  gpu__time_duration.sum` bucketed by kernel name, OR ggml per-op timing.
- Then quantify the LTX puzzle: **ncu an actual LTX DiT run the same way** (master worktree) — is LTX truly
  faster per-FLOP, or is its 238s TE(gemma-12B)+VAE-dominated? Settle it with data, not recollection.

## ⚠️ RULE — "at the floor" is a claim that must be EARNED
Every "at the floor" / "can't be tuned" / "dead lever" statement in this work MUST ship with a full
quantitative breakdown that justifies it **metric by metric**: which hardware limiter is actually hit, why
each contributing factor (occupancy, registers, each stall reason, mem traffic, instruction mix, tile
utilization) **cannot be reduced**, and what was tried + measured. No "at floor" by basic profile, by
intuition, or by analogy to another project. It is cheap to break the internal cost down — so do it, every
time, and write the justification next to the claim.

## PRIORITIZED LEVERS (the work)
**Lever #1 is the DiT matmul deep-dive above** (NOT closed — full breakdown required; primary payoff =
higher-precision weights at ~same latency). The levers below are the overhead-around-the-matmul, VRAM-fit,
and chaining wins that stack on top.
1. **Port `LONGCAT_DIT_NO_MMAP` pinned-offload (FINDINGS-07)** — already in tree (env-gated,
   `src/stable-diffusion.cpp:562`). Quickest first measurement: `PINNED=1 OFFLOAD=1 ./perf_a14b.sh`.
   For LTX it cut 14.4→10.6 s/step. UNTESTED on Wan.
2. **Free umT5 after text-encode** (code) — run it on GPU (fast, ~2s vs 16s on CPU) then release the
   5.76GB before the DiT loop, so resident DiT fits WITHOUT aggressive VAE tiling. ~14s + removes
   tiling pressure. Biggest clean win for the per-segment wall.
3. **Port LTX residency levers** — `LONGCAT_SHARED_RESIDENT` (FINDINGS-09, −5.2%) + cross-step
   residency (FINDINGS-10). Check if they already apply to the Wan DiT graph or need wiring.
4. **VAE tiling tune (FINDINGS-05/11)** — find the rel-size that fits resident DiT WITHOUT the
   over-tiling slowdown (0.25 was too many tiles → 38s; 0.5 OOM'd against resident DiT). Sweet spot
   ~0.33, or port the im2col spatial-pad tiled path (FINDINGS-11, LTX VAE 70.6→41.7s).
5. **Segment-chain director loop** — the actual music-video engine: per-segment prompts, latent-chain
   via VACE backdoor (`VACE_CONT_LATENT`/`VACE_CONT_FRAMES`) or IT_LATENT_CARRY. Mirror LTX
   `ltx_director.sh`. Use **full MoE** for faithful i2v (single-expert drops the init-image anchor —
   goes text-driven; see below).
6. **InfiniteTalk VRAM (separate, harder)** — see below.
7. **Token/res/frames tradeoff** — since DiT time ∝ tokens and the matmul is at floor, the ONLY raw-DiT
   lever is fewer tokens. Sweep 384×640 / 416×736 vs 480×832 and 13/21/41 frames; find the
   quality-floor res that still reads well, since the music video chains many segments.

## Gotchas / dead-ends
- **Single-expert (omit `--high-noise-diffusion-model`) = TEXT-DRIVEN** — drops the i2v init-image
  anchor (the high-noise expert locks early structure/identity). Fits resident + faster, but NOT
  faithful i2v. Use full MoE for identity/chaining; single only for fresh t2v-style gen.
- **InfiniteTalk:** lip-sync works at 256px. At 480p it OOMs — encode temporal-tiling was FIXED
  (`examples/infinitetalk/main.cpp:395`, env `IT_NO_ENCODE_TEMPORAL`/`IT_VAE_TILE_REL`) but it STILL
  OOMs because IT holds ~10.8GB GPU during the encode: its **DiT offload reserve + audio-embedding
  pass aren't freed before the VAE phase**. Needs VRAM-phase plumbing. Also IT loads its 10.7GB DiT
  into **anonymous RAM (no mmap)** → ~17GB hard RSS that clobbers prod gemma/ace on the 31GB box —
  wire mmap into the IT loader. (sd-cli mmaps and is gentle.)
- **ncu in docker needs `--cap-add SYS_ADMIN`** (else ERR_NVGPUCTRPERM). No nsys in the builder, ncu only.
- **GPU is shared with prod** (gemma llama-server :8080 + ace-server :8088, worker-isolated to VRAM-0
  when idle). Coordinate; A14B mmap is gentle, the IT anon-RAM load is not.
- **char.png is the anime vtuber girl**, not a man. For the music-video man, generate him (t2v/i2v) —
  the LTX "guy" was just LTX-generated, not a fixed asset.

## Per-segment wall floor (raw baseline, FR=21)
TE 2-16 + VAEenc 6-13 + DiT ~50 + VAEdec 18-25 ≈ ~100s. Goal: levers 1-4 should pull this well down
(target the DiT per-step + kill the TE/VAE overhead). See `perf_out/sweep.csv` for the full matrix.

---
## FINDINGS-A — DiT per-step kernel decomposition (ncu duration buckets, 2026-06-13)
Method: `profile_decomp.sh` — single-pass `gpu__time_duration.sum` over a full generate (single low
expert, 480x832, FR=13, 2 steps, resident, --clip-on-cpu, temporal VAE tile 0.25). Bucketed by
demangled kernel name; DiT phase isolated by ID-range between first/last matmul. Raw CSV:
`perf_out/decomp.ncu.csv` (105929 kernels, 47.16s total GPU under profiling). Ratios are what matter
(profiling inflates absolute time; unprofiled DiT = 7.37 s/it FR=13).

**DiT phase = 18.97s / 2 steps = 9.49s/step (profiled). Per-step breakdown:**
| kernel | % step | count/2steps | us/call | what |
|--|--|--|--|--|
| mul_mat_q (Q4_K) | **56.7** | 802 | 13407 | the hot matmul (qkv/o/ffn) |
| flash_attn_ext_f16 | **25.8** | 160 | 30625 | DiT self/cross attention |
| k_bin_bcast | 4.7 | 1290 | 688 | residual adds / elementwise |
| quantize_mmq_q8_1 | 2.1 | 802 | 495 | act→q8_1 requant per matmul |
| mul_add_bcast (AdaLN mod) | 2.0 | 321 | 1159 | modulation |
| rope_pe_f32 | 1.3 | 160 | 1555 | rope |
| norm_f32 + rms_norm_f32 | 2.6 | 561 | ~900 | norms |
| cpy/pad/concat/unary | ~3.5 | — | — | layout/glue |

**Matmul + attention = 82.5% of the DiT step.** Misc glue (norm/rope/adds/copies) ≈ 17.5%.
- The "extra work" hunt's first hit: `quantize_mmq_q8_1` (act requant) is real but only **2.1%** — 802
  launches (1 per matmul). Not the win it might be elsewhere.
- **Attention is the #2 cost (25.8%)** — 80 FA calls/step @ 30.6 us... wait 30.6 MS/call. FA is O(L²);
  at FR=21 it grows. FA is a genuine lever candidate, NOT just the matmul.

**Whole-run phase split (unprofiled, from ncu_dit.log baseline, FR=13 single-expert resident, ~80s):**
| phase | s | % | lever |
|--|--|--|--|
| umT5 text-encode (ON CPU) | 18.8 | 23 | **move to GPU = ~2s, save ~17s** (free-after-encode) |
| VAE encode | 8.5 | 11 | tiling tune / im2col-pad (FINDINGS-11) |
| DiT sampling (2 step) | 37.3 | 46 | matmul(57%)+FA(26%) |
| VAE decode | 15.7 | 19 | tiling tune |

⇒ Biggest *single* quick win = **umT5 on GPU instead of CPU** (~17s off an ~80s wall). Then DiT
matmul/FA, then VAE (30% of wall combined). NOTE the CPU-T5 path was only used because resident DiT +
GPU-T5 OOMs the 12GB card → the free-umT5-after-encode lever (run on GPU fast, release 5.76GB before DiT
loop) is the keystone that makes the resident path both fit AND drop the 17s.

---
## FINDINGS-B — fresh MoE baseline (offload, GPU-T5) 2026-06-13
`run_a14b.sh`: MoE (both experts) 480x832 FR=21, --offload-to-cpu --mmap --max-vram 4.5,
--vae-tiling 4x4, GPU umT5 (no --clip-on-cpu). **wall 118.7s, peak 6527 MiB.** Stage split:
- umT5 text-encode (GPU): **1.10s** (CPU path was 18.8s — GPU-T5 is the right default; free-after-encode already wired @ stable-diffusion.cpp:4733)
- VAE encode: ~5.7s
- DiT sampling 4 steps (high 46.33s=23.2/it + low 43.28s=21.6/it): **89.75s (78% of wall)**
- VAE decode: 18.15s (16%)

⇒ **DiT offload streaming (~22 s/it) vs resident (~12.4 s/it) = ~40s of pure weight-transfer overhead =
the #1 prize.** Experts run SEQUENTIALLY (separate sample() calls @ 6419 high / 6480 low) — each 7.6GB
expert fits resident alone in 12GB once umT5 is freed. Cheap knob attack: raise --max-vram so the active
expert stays resident across its 2 steps + PINNED=1 (LONGCAT_DIT_NO_MMAP) for faster H2D. Code lever (if
knobs fall short): true sequential expert residency via deferred-load + free-at-switch + swap_diffusion_model
(infra exists: avatar deferred-DiT-load @1480, flux2 swap @1535).

---
## FINDINGS-C — offload knob sweep (max-vram + pinned), FR=13 MoE 2026-06-13
| label | maxv | pin | wall_s | dit_s_it | peak_MiB |
|--|--|--|--|--|--|
| of_base | 4.5 | 0 | 58.8 | 8.68 | 6527 |
| of_max9 | 9 | 0 | 56.5 | 8.81 | 10727 |
| of_max9pin | 9 | 1 | 82.7 | 8.14 | 10741 |
| of_max10pin | 10.5 | 1 | 130.7 | 8.14 | 11795 |

- **PINNED (LONGCAT_DIT_NO_MMAP) = NET LOSS one-shot:** dit_s_it −6% (8.68→8.14) but wall +40% (the
  cudaMallocHost pinned-load of the DiT isn't amortized over only 4 steps; bigger max-vram = more pinned =
  worse). Would only pay on a WARM multi-segment worker (load once, amortize). DEAD for CLI / per-segment.
- **max-vram raise useless at FR=13** (8.81 vs 8.68): the FR=13 compute buffer is small enough that the
  graph-cut already keeps most weights resident; streaming isn't the bottleneck here. The ~22 s/it pain was
  at **FR=21** (bigger compute buffer eats the max-vram budget → forces streaming). ⇒ max-vram must be
  retested at FR=21, not FR=13. (Lesson: FR=13 is too cheap to expose the streaming-overhead lever.)
- **BUG found:** stable-diffusion.cpp:859 — high_noise expert get_param_tensors does NOT pass `&& !dit_no_mmap`
  (line 850 does for the low expert). So PINNED only ever affected the low expert. Moot given pinned=dead
  for CLI, but fix if the warm-worker pinned path is ever pursued.

---
## FINDINGS-D — nsys timeline (DiT is compute-bound, not launch/copy-bound) 2026-06-13
`profile_nsys.sh` low-overhead CUDA trace, single low expert FR=13 1 step resident. Convert qdstrm via
host QdstrmImporter (needs apt libdw1). Report: `perf_out/dit_nsys.nsys-rep/.sqlite`.

**CUDA API summary (CPU-side):** cudaLaunchKernel **median 3 us** (98778 calls whole-run; ~1500/DiT step ⇒
~5 ms launch overhead/step = NEGLIGIBLE vs ~7s). cudaMemcpyAsync 8.0s + cudaStreamSynchronize 16.8s are
**VAE-tiling**-dominated (resident DiT has no per-step weight H2D). ⇒ **DiT is GPU-compute-bound; CUDA-graphs
/ launch-fusion / fewer-kernels would buy ~nothing.** No structural/glue win available — it's real FLOPs.

**Per-step DiT GPU time (nsys, ~unprofiled):** mul_mat_q 3.82s (400 calls, avg 9.55ms, max 23.2ms=FFN) +
flash_attn_ext_f16 1.96s (80 calls, avg 24.6ms, **max 46.5ms**=full-L self-attn, min 3.6ms=cross-attn) +
quantize_mmq 0.19s + rope 0.12s + rms_norm 0.10s + glue. ≈ 7s total = matches 7.37 s/it resident.
mul_mat_q ~55%, **flash_attn ~28%** (NOT yet characterized — next), glue ~17% (cheap, many small kernels).

⇒ Only two in-DiT compute levers: (1) the Q4_K matmul (occupancy-bound, quality-headroom — parked play),
(2) **flash-attention 28%** (high variance, the long-L self-attn dominates; O(L²) so it grows with frames/res).
Both need ncu --set full to claim floor. Glue is irreducible small-kernel work, not a lever.

---
## FINDINGS-E — DiT matmul + FA floor verdict (ncu --set full) 2026-06-13
`profile_ncu_full.sh` (--set full, single low expert FR=13 1 step). Raw: `perf_out/ncu_full.log`.
12 genuine DiT-block-0 mul_mat_q instances + the FA kernel (same `flash_attn_ext_f16<128,128,64,1,0,0>`
template the DiT self-attn uses — regs/occupancy are L-independent so the proxy transfers).

### Q4_K matmul `mul_mat_q<12,128,0>` — AT FLOOR (metric-by-metric):
12 instances, dead-consistent. Big FFN matmul = grid 5292/1960, 31µs; projections grid 1960, 11.65µs.
| metric | value | meaning |
|--|--|--|
| Achieved occupancy | **16.67% = Theoretical 16.67%** | already at the kernel's max possible occupancy |
| Block Limit Registers | **1 block/SM** (221 regs) | reg-capped; also smem-capped (57.9KB dyn → 1 blk) |
| Compute (SM) | 54.5% | not saturated |
| Memory / DRAM | 46% / 39.5% | not saturated ⇒ NOT bandwidth-bound |
| Warp cyc/issued inst | 3.66 | healthy issue rate |
| No-eligible-warp | 45% | latency-bound (1 block/SM ⇒ scheduler starved) |
| Avg active threads/warp | **32.0 / 32** | zero divergence, zero predication waste |

**Why it can't be reduced:** occupancy is reg+smem-limited to 1 block/SM and ALREADY achieves theoretical;
no unit is saturated so it's latency-bound, not compute/DRAM-bound; the only structural knob (mmq_x tile
128→64, GGML_MMQ_X_CAP / MIN_BLOCKS=2) is PROVEN-DEAD on the identical kernel in acestep (+10–43% worse,
register spills) and flux2. Zero divergence ⇒ no warp-efficiency to recover. **= at floor for raw speed.**
The ONLY lever here is the NORTH-STAR quality play (parked): DRAM at 39.5% + SM 54.5% ⇒ headroom to push
Q5_K/Q6_K/Q8_0 weights through ~the same latency (Q8_0 may even be faster via the int8 MMA path).

### Flash-attention `flash_attn_ext_f16<128,...>` — latency-bound, LESS saturated than matmul:
168 regs → **25% theoretical occupancy** (= achieved); SM 37.6%, Mem 36.9%, DRAM **19%** (nothing
remotely saturated); **75% no-eligible-warp**, 0.32 eligible warps/sched of 2.99 active. Severely
latency/occupancy-bound. Same floor class — reg-limited ggml kernel, upstream-tuned, no quick knob. FA is
28% of the step and O(L²) so it grows with frames/res ⇒ the token/res lever (fewer tokens) is the only
raw-FA reducer.

### Extra-work hunt (per the rule):
- `quantize_mmq_q8_1` re-quantizes the F32 activation to Q8_1 once PER matmul (401×/step, 2.1%). q/k/v
  projections read the SAME x → 3× redundant quant; ffn gate+up likewise. **Fusing QKV (and gate+up) into
  single matmuls** would quantize x once + one bigger matmul. Bounded gain ~1–2% (quant savings; occupancy
  won't improve since reg-limited). Real but marginal — noted, not chased.
- Glue (cpy_perm_transpose, convert_unary f16↔f32, pad) = the attention wrapper's permute+cont+f16-cast +
  the kv_pad-to-256 (wasteful for cross-attn L_k≈30→256, but cross-attn is the cheap FA). All small.
- nsys already proved NOT launch/sync-bound (3µs median launch) ⇒ no CUDA-graph/fusion win.

**VERDICT: the DiT step is at the compute floor for raw speed on the 3060.** Matmul + FA are both
occupancy/latency-bound at their max achievable occupancy with zero divergence; no unit saturated; the
known structural knobs are proven dead. Further DiT speedup must come from FEWER TOKENS (res/frames) or
the offload→resident path (streaming overhead, not the kernels). Quality-at-same-speed (higher-precision
weights) is the one positive lever and it's the parked north-star play.

---
## FINDINGS-F — maxv7 = the DiT residency win via a KNOB (no code) 2026-06-13
`run_maxv7.sh` FR=21 MoE long-prompt, --offload-to-cpu --mmap --max-vram 7 (vs baseline maxv4.5):
| stage | maxv4.5 | maxv7 | Δ |
|--|--|--|--|
| DiT sampling (4 step) | 89.75s | **63.3s** | **−29.5%** (high 46.3→32.2, low 43.3→31.1) |
| per-step | ~22 s/it | ~16 s/it | resident single-expert floor = 12.4 s/it |
| VAE decode | 18.15s | 18.11s | — |
| wall | 118.7s | **99.4s** | **−16%** |
| peak VRAM | 6527 | 7629 MiB | +1.1GB (fits 12GB; leaves ~4GB for prod) |

At maxv7 the DiT graph cuts into **5 segments (vs 8)** and they run **offload=0ms (resident)** — the
big max-vram budget keeps the active 7.6GB expert mostly resident across its 2 steps. ⇒ **maxv7 IS the
sequential-expert-residency lever, achieved by a knob.** Closes ~60% of the offload→resident gap (22→16,
floor 12.4). Remaining 16-vs-12.4 = unavoidable cold cold upload of the 0.6GB that won't fit the 7GB budget
+ first-segment fill. **The sequential-residency CODE change is now low-value** (would buy ~3.5 s/it more
for real engineering); maxv7 is the practical sweet spot. peak 7629 MiB.
- T5 anomaly: get_learned_condition 8.24s here vs 1.10s in run_a14b (same prompt) — likely cold-cache/clock
  one-off; re-confirm. If it's real (T5 streamed off GPU at maxv7), pin T5 on GPU during encode = −7s.
- **New baseline = maxv7: 99.4s/segment, peak 7629 MiB.** Use --max-vram 7 going forward.

---
## FINDINGS-G — sub-7.5GB config + 1280x704 same-res-as-LTX comparison 2026-06-13
**Clean resident baseline (maxv6, 480x832 FR=21 MoE), repeated/de-noised:** wall 84.2s, DiT sampling
55.4s (high 27.74 + low 27.68 = 13.85 s/it), VAEdec 18.2s, T5 1.14s, **peak 6591 MiB = SUB-7.5GB**,
22/22 segments resident. maxv6.5: wall 84.6s, peak 7153. ⇒ **maxv6 is the sweet spot — full DiT residency
at 6591 MiB, no speed cost.** (The earlier maxv7 99.4s reading was contention-noise: its T5 was 8.2s vs
1.1s clean; maxv7 peak 7629 is OVER 7.5GB for no benefit. USE maxv6.) NOTE significant run-to-run variance
(GPU shared w/ prod + thermal) — repeat key numbers.

**Wan @ 1280x704 (LTX's res), 21f MoE offload maxv6, 0.25 temporal tiling:** wall **287.0s**, peak 6725 MiB.
- DiT sampling **184.8s** (high 92.33 + low 92.47 = **46.2 s/it** × 4) — at this res the compute buffer is
  4484MB, so expert(7.6)+buf(4.5)=12.1GB > 12GB ⇒ **resident is impossible, DiT forced to 40-segment offload
  streaming** = the residency win is LOST at high res (46 s/it vs 13.85 at low res).
- VAE enc+dec **~97s (34% of segment!)** — 0.25 tiling = 16 over-tiles + re-pad; UNOPTIMIZED at high res.
  FINDINGS-11 im2col-pad path (LTX VAE −41%) + coarser tiling = the big high-res lever (~−40s/seg).
- T5 1.12s.

**Apples-to-apples vs LTX-2.3 @ 1280x704 (~50min for 27s):**
27s chained 21f segments: @16fps ≈21 seg×287 = **~100 min**; @24fps ≈31×287 = **~148 min**. ⇒ **Wan @
1280x704 is currently ~2–3× SLOWER than LTX at the same res.** Root cause: (1) high-res DiT can't use the
residency win (OOM) so it's offload-bound; (2) VAE tiling unoptimized at high res. DiT KERNELS are at floor
(FINDINGS-E) so DiT won't get faster except fewer tokens; the realistic high-res claw-back is VAE (~−40s)
+ longer segments (amortize VAE/T5). Even optimized ≈ 80–120 min ⇒ Wan's speed edge is at LOWER res
(480x832 = ~35min/27s, betting quality ≥ LTX@1280). **Res is now a genuine product call (speed vs match-LTX-res).**

---
## FINDINGS-H — VACE chaining VALIDATED e2e (long-form pipeline) 2026-06-13
Pulled VACE-FUN distill models from .151 (`wan22-vace-fun-a14b-{low,high}-distill-q4_k.gguf`, 9.87GB ea).
`run_vace_chain.sh` 2-segment test @ 480x832 maxv6 FR=21 K=5:
- Loads as **"Wan2.x-VACE-14B"**, VACE code path engaged (stable-diffusion.cpp:5691).
- seg1: free i2v from char.png, **VACE_SAVE_LATENT banked `60x104x6x16`** latent (2.4MB). 162.6s.
- seg2: **VACE continuation fired** — "5 kept pixel frames (mask=0)" + "VACE_CONT_LATENT: injected 2 tail
  latent frames (of 6) into inactive head, bypassing pixel re-encode". 130.5s.
- **Seam quality: identity/style/outfit/framing CONTINUOUS across the seg1→seg2 cut** (perf_out/vace/seam.png) —
  no jump-cut/drift/color-shift. Latent backdoor works. Stitched 37-frame chain.mp4 sent to user.
- **Cost: VACE seg ≈ 130–160s vs i2v 84s** — VACE-FUN expert is 9.87GB (>i2v 8.15GB) so it can't fully
  reside at maxv6 (more streaming) + VACE does extra control-video VAE encodes (inactive/reactive context).
  VACE perf-tune = follow-on (same maxv/tiling levers, sweet spot shifts up for the bigger expert).
- **Throughput est:** 27s @16fps (432f, 21f segs w/ 5f overlap = 16 new/seg ⇒ 27 segs) × 130s ≈ **58 min**
  at 480x832 WITH smooth continuation — vs LTX 50min at 1280x704. Competitive, lower-res, proper chaining.

**Long-form chaining pipeline = DONE.** Remaining e2e piece = InfiniteTalk lip-sync (still OOMs: holds
~10.8GB during VAE encode — DiT offload-reserve + audio-embed pass not freed before VAE phase; + loads
10.7GB DiT into anon RAM, clobbers prod). Next task.

---
## FINDINGS-I — InfiniteTalk 480p OOM root-caused + architecture pivot 2026-06-13
**IT works at 256px** (run_it.sh W=256: rc=0, 64.1s, peak **11797 MiB**, 21f, prod-safe: RAM 28→14GB avail,
swap flat). **480p OOMs** because: the DiT is loaded into pinned host RAM (10254 MB) then
`move_params_to_runtime_backend` (ggml_extend.hpp:2405) **copies ALL 10.25GB resident onto CUDA0** — it
does NOT stream per-segment like sd-cli's graph-cut offload (the `IT_MAX_VRAM_GIB`/graph-cut only segments
the COMPUTE, not param residency). Peak @256px = DiT 10.25GB + VAE-enc buf 0.99GB + DiT-compute 0.62GB ≈
11.8GB (barely fits 12GB). @480p the VAE-enc buffer alone ~4GB ⇒ OOM. **Fix options:** (a) make IT use
true graph-cut streaming (params stay on CPU params_backend, stream per-segment) — best, no reload tax,
bigger change; (b) free the runtime DiT params (move back off CUDA0) before vae_encode_video + final decode
each window, reload for the diffusion step (~0.84s each) — localized, +~2.5s/window. **PAUSED pending the
architecture decision below** (IT may not be the main path).

## ⭐ ARCHITECTURE PIVOT — the real goal is SCRIPTED-ACTION + lip-sync (not talking-head)
User clarified: needs a character doing **directed/scripted actions + scene + camera** AND lip-syncing a
song. InfiniteTalk/longcat = talking-head only (no action control); VACE = action control, no audio.
**Key in-repo find: Wan2.2-S2V-14B is ALREADY PORTED** (`wan-s2v-14b-dit-dmd-q4_k.gguf` on .151, +
wav2vec2-xlsr53 + chinese-wav2vec2). CLI = ref-image + TEXT-PROMPT + audio → video; runner (wan_s2v.hpp:779)
has a **pose-conditioning input** not yet wired to a CLI flag. ⇒ S2V may be the SINGLE-MODEL answer
(identity + text-scripted action + pose + audio-lipsync in one pass). Other candidate paths: VACE(scripted)
→ cheap mouth-only lip-sync (LatentSync/MuseTalk, preserves action); VACE→IT/MultiTalk video-dub mode.
**Launched a research subagent** to rank these for the 12GB-3060/ggml constraint. Decide arch BEFORE more
IT/VACE perf work.

---
## FINDINGS-J — Wan2.2-S2V base VALIDATED (single-model lip-sync path) 2026-06-13
Research verdict (subagent): S2V = the single-model answer for scripted-action + singing lip-sync;
~90% ported (cond_encoder/pose path loads + wired into forward(), CLI feeds zeros). Pulled
`wan-s2v-14b-dit-dmd-q4_k.gguf` (9.4GB) + `wav2vec2-xlsr53-f16.gguf` (0.63GB) from .151. cae
(casual_audio_encoder) is BUNDLED in the s2v dit gguf (`model.diffusion_model.casual_audio_encoder.*`) →
`--audioenc` = the same s2v gguf. Built `sd-s2v` target (was missing; build clean).
**`run_s2v.sh` base test (ref+text+audio, NO pose), 480x832 21f --distilled (4-step):** rc=0,
**wall 96.8s, peak 7347 MiB (sub-7.5GB), 18 frames** (S2V temporal-compresses 21→18). graph-cut offload
5 segs, S2V_MAX_VRAM_GIB=6.5 default. **Lip-sync WORKS** (lipstrip frames 0/6/12/17 = distinct mouth
shapes tracking the song), identity held from char.png, natural head motion, clean quality. mp4 sent.
⇒ S2V foundation proven. **NEXT: wire `--pose-video`** for scripted body action (the user's core gap).
Pose path = VAE-encode skeleton-render frames (DWPose, generated upstream) → cond_states([W,H,T,16]) /
cond_block([W,H,nfb,16]); clone vace.hpp pixel→latent; pass instead of empty at s2v/main.cpp:568/625.

---
## FINDINGS-K — S2V action-range VERDICT: performer, not director 2026-06-13
`run_s2v_action.sh` (832x480 wide, 49f, explicit locomotion prompt "steps out of a convertible and walks
along a neon street toward a bar, singing, full-body wide tracking shot"): wall 237.5s, 46 frames.
**RESULT: S2V rendered a singing HEAD CLOSE-UP the entire 3s — ignored the locomotion prompt entirely.**
No car, no walking, no street/bar, no framing change (walkstrip.png = same close-up at 0/25/50/75/100%).
⇒ **S2V is an audio-driven PERFORMER, not a director.** Text does NOT drive locomotion/scene-action in
S2V (body motion needs a pose_video driver, and even that = "performer doing the pose", not scene
interaction). Confirms the user's "single scene + character" intuition.

### REVISED ARCHITECTURE (music video = i2v shot-list director; S2V is a specialized tool)
Action range is a Wan2.2-base property; S2V's audio fine-tune biases HARD to face-performance and loses
text-action. So:
- **Narrative/action shots** (drive, walk, open door, sit, eat) → **i2v/VACE text-prompted** (Wan2.2 base
  action range — needs its own probe; fine object-interaction like "eat chips + spit crumbs" is at the
  edge of ALL current text-to-video).
- **Static singing-performance shots** (close-up "sits at bar and sings") → **S2V one-pass** (clean
  lip-sync, validated FINDINGS-J). This is S2V's lane.
- **Singing DURING action** (sings while walking) → i2v/VACE generates the action, then **mouth-only
  lip-sync** (LatentSync / MuseTalk) repaints ONLY the mouth, PRESERVING the directed action. This was
  research-path #4; it's now the KEY piece for action+singing (the only way to add lips without
  regenerating—and losing—the i2v action). Caveat: PyTorch (sidecar, breaks pure-cpp), singing unverified.
**Pivot:** the i2v/VACE → mouth-only-lipsync path is the backbone (preserves action); S2V is a
special-case one-pass tool for static performance shots. InfiniteTalk/S2V do NOT solve action+singing.
Next experiments: (1) probe Wan2.2-I2V text action-range ("man walks to a bar and opens the door"). (2)
stand up LatentSync/MuseTalk sidecar, test singing lip-sync on an i2v action clip (preserve body?).

## FINDINGS-K addendum — S2V DROPPED (audio path broken + talking-head + no action)
The s2v base "lip-sync" was a FALSE POSITIVE: log shows ALL `casual_audio_encoder.*` tensors =
"unknown tensor ... in model file" (CAE weights never loaded) + "M1 forward/sampler ... NOT validated
against torch oracle". So the mouth motion was the model's generic unconditioned head animation, NOT
audio-driven. Combined with FINDINGS-K (no text action range) + S2V being a talking-head model = 3
disqualifiers. **S2V DROPPED.** (CAE non-load may be a fixable name-prefix bug — runner name
"model.diffusion_model.casual_audio_encoder" vs tensor "...casual_audio_encoder.encoder.*" — but not
worth fixing since it still can't direct action.)

## LOCKED ARCHITECTURE for the music video
1. **i2v/VACE shot-list director** = the narrative engine. Per-shot text prompts (drive/walk/open-door/
   sit/eat) + init-frame; VACE chaining (VALIDATED FINDINGS-H) for character/world continuity. Action
   fidelity = Wan2.2 base (probe next; fine interaction at edge of all video-AI).
2. **Singing lip-sync = dedicated MOUTH-ONLY model** (LatentSync / MuseTalk) applied to the i2v/VACE
   output — repaints ONLY the mouth, PRESERVES the directed body/scene/camera. PyTorch sidecar (off the
   ggml path); singing (vs speech) is the unverified risk to test. This is THE lip-sync mechanism (not
   S2V/InfiniteTalk, which regenerate motion / are talking-head-only).
Next: (a) probe Wan2.2-I2V text action-range; (b) stand up LatentSync/MuseTalk sidecar, test SINGING
lip-sync on an i2v action clip (does it preserve body + sync a sung vowel?).
