# PERF PLAYBOOK — image→rigged-3D native C++ pipeline
**2026-07-17.** Baseline: `$AS/_shootout_out/perf_baseline/{run.log,vram.tsv}`, RTX 3060 12GB, `--res 1536 --texsize 4096 --quad --tex-project --refine --model weights_gguf`, default cam, nice 15 / OMP=8.

Every claim below is tagged **[VERIFIED]** (measured, or arithmetic on logged numbers), **[PLAUSIBLE]** (code-read + consistent model, not run), **[SPECULATIVE]** (reasoned, unmeasured). **Six agents read the code; none was allowed to touch the GPU.** Nothing here is a before/after benchmark. Treat every "expected win" as a size, not a result.

---

## 0. READ THIS FIRST — the brief that spawned this doc was wrong in five places

| the inherited claim | the truth | evidence |
|---|---|---|
| "refine's DiT is 749s and is the bottleneck; dense decode ~280s" | **DiT 739.5s; dense decode 2502.3s.** The decode is the single biggest thing in the pipeline. The "280s" was an **octree-256** measurement. | `run.log:585-599` **[VERIFIED]** |
| "UNEXPLAINED: 668k verts vs the reference's 165k — it may BE the 3339s" | **Solved: `--us-octree`.** Prod default is 512; the eye-tested asset was 256. Not a bug, not a cost anomaly. | MC lattice pitch measured from both GLBs: 0.0016635 vs 0.0033285, ratio **2.0009** on all 3 axes, identical canonicalization **[VERIFIED]** |
| "VRAM peak 10751/12288, headroom 1537 MiB" | Card reports **11906 MiB usable**. Peak = **90.3%**, headroom **1155 MiB**. And the peak belongs to one 813s sub-stage, not "the refine". | `run.log:1-2`; `vram.tsv` **[VERIFIED]** |
| "`--moge` OOMs because coarse verts +20.6%" | The refine's ask is **byte-identical** with and without moge (N=8192, Tc=3970, octree=512, chunk=2048 in both logs). Coarse verts are produced **downstream** of M3b and cannot cause it. Real chain: FOV → N1 1227→1701 → M 12541→18674 → M². | `e2e_moge/run.log:24,39,54` vs `run.log:38,53` **[VERIFIED]** |
| "the tex DiT silently loads 4.9GB fp32 npy instead of a 2.59GB F16 gguf — inflates time AND VRAM peak" | **Half wrong.** `weights_gguf/` **is the fp32 dir** — 700/700 F32 tensors in every file. npy fallback yields F32 too, so it is **not a precision or VRAM bug**; it costs **load time** (640 HDD files, 5.17GB, measured 57.8s). The *real* precision story is bigger: **all four geometry DiTs run F32 and the F16 twins have sat in `weights_gguf_f16/` since June**, and the **UltraShape VAE has no gguf at all** — the 2502s decode runs fp32 npy. | parsed gguf headers; `run.log:4,54` **[VERIFIED]** |

---

## 1. THE BASELINE — quote this table

**TOTAL 4498.0s (75 min)** → verts=121437 faces=167340 J=33. **VRAM peak 10751 MiB / 11906 usable = 90.3%. Headroom 1155 MiB.**

| stage | time | % | VRAM in-window |
|---|---|---|---|
| **[1a/4] UltraShape refine** | **3339.0s** | **74.3%** | **10751 peak** |
| — VAE dense decode (G=513 → 135,005,697 q / 2048-chunk = 65,921 computes) | **2502.3s** | **55.6%** | **1389 flat** |
| — DiT sampling (50 steps ×2 CFG, N=8192) | 739.5s | 16.4% | **10751** |
| — conditioner | 13.3s | 0.3% | ~133 |
| — VAE transformer (hoisted, runs once) | 2.1s | — | 4213 |
| — voxelize + MC + floaters (CPU, **untimed**) | ~82s | 1.8% | idle |
| [1/4] geometry (all of it) | 893.1s | 19.9% | 8985 peak |
| — M3b DiT (21 fwd, M=12541) | 274.8s | 6.1% | 8985 |
| — tex DiT (cross, Ntok=4101, 12 fwd) | 181.8s | 4.0% | 8545 |
| — tex decoder (4.14M PBR voxels) | 64.3s | 1.4% | 5019 |
| — SS DiT (22 fwd, SEQ=4096) | 62.9s | 1.4% | |
| — M4 mesh (decoder ~54s + dual-grid 1.2s) | 55.4s | 1.2% | |
| — M3a / MC-REMESH / M2 DiT | 18.7 / 16.7 / 16.5s | 1.2% | |
| — DINOv3 @1024/@512, NAF @1024/@512, MoGe cam | 8.4 / 8.1 / 4.8 / 4.2 / 15.0s | 0.9% | |
| — **unattributed** | **176.5s** | **3.9%** | gguf loads, host post, GLB writes |
| [1a2/4] quad retopo (CPU shell-out) | 109.7s | 2.4% | 0 |
| [1/4] tex bake (reproject/shell) | 87.6s | 2.0% | 0 (host) |
| [3/4] rig (SkinTokens) | 40.6s | 0.9% | 5477 |
| [1b/4] tex project | 22.0s | 0.5% | 0 (host) |

**VRAM time histogram** (this is why "mean 4151 MiB" describes no actual moment):

| band | time | % | what |
|---|---|---|---|
| 0–1023 MiB | 536s | 12.1% | idle / CPU stages |
| **1024–2047** | **2516s** | **56.9%** | **the dense decode, flat at 1389** |
| 2048–8191 | 196s | 4.5% | |
| 8192–9215 | 450s | 10.2% | M3b DiT / M4 / tex decoder |
| **10240–11263** | **720s** | **16.3%** | **the refine DiT — the entire ceiling** |

**GPU idle: 550s (12.2%)** at util==0, incl. a contiguous **226s block at 135 MiB / 12W** (quad + bake + project — the CPU tail).

---

## 2. TL;DR — the five biggest levers

| # | lever | expected win | lossless? | effort | confidence |
|---|---|---|---|---|---|
| **1** | **Hierarchical/sparse VAE decode** — replace the brute-force 513³ sweep. **The isosurface is 0.34% of the grid** (measured by voxelizing `refined.glb`: 457,147 of 134M cells; cross-checked by surface area at 0.366%). We decode empty space 99.66% of the time, and `ultrashape_e2e.cpp:84` admits it: *"DENSE grid here; Python uses a hierarchical octree"*. | **−2000 to −2350s = up to −50% of the pipeline** | **unknown** — conservative by construction if done as a pyramid; a coarse-band-only version is **not** (measured: even a ±4 band misses 0.40% of isosurface cells → holes) | high | **[PLAUSIBLE]** — the 0.34% is measured; the "~22x / ~6M queries" is arithmetic |
| **2** | **`--us-octree 256`** — the anomaly. Prod default 512; **every eye-tested asset shipped at 256**. Decode ∝ G³. | **−2187s (−48.6%)** — but this is a **baseline correction**, not a saving. **VRAM win = zero.** | **lossy — owner judges** | zero (one flag) | **[VERIFIED]** (lattice pitch 2.0009×) |
| **3** | **Hoist `c_kv` + `ln_2(latents)` out of the decode chunk loop** — the tr-hoist stopped one function short. `tr_in` is a `const_tensor`; `us_cross_block` re-LNs it and re-projects K/V **65,921 times**. 34.4 of 146 GFLOP per chunk. | **−23.5% of the decode = ~−590s** | **lossless** (bit-exact by the same argument that made the tr-hoist bit-exact) | ~30-40 lines | **[VERIFIED]** arithmetic, **[PLAUSIBLE]** bit-exactness under ggml tiling |
| **4** | **Flash-attn in `us_cross_attn`** — skipped on a **stale comment**: `m1_ggml.hpp:411` says *"cross-attn's 5 kv tokens don't benefit"*. **This cross-attn has 8192 kv tokens.** The dense path materializes a **1.07 GiB** scores tensor per chunk → ~283 TiB of traffic → an **~865s bandwidth floor**. | **−800 to −1200s of the decode**; decode VRAM 1389 → ~350 MiB | **lossy** (f16 accum on an occupancy threshold) | medium — the flash path exists and is debugged; needs a mask + `--fast` | **[SPECULATIVE]** on magnitude |
| **5** | **F16 weights + `--fast` + flash on the geometry DiTs** — already A/B'd on 2026-06-20 and **parked, not rejected**: DiT stack 865.9→490.4s (−43%), VRAM 10133→6591 (−35%), geom chamfer **0.196% of bbox diag**, "visually indistinguishable". The park condition was *"re-evaluate against a CLEAN baseline once geom/texture quality is sorted"* — **that trigger has arguably fired.** | **−190s of DiT wall**; −3.5GB in the geometry window (does **not** lift the 10751 ceiling) | **lossy** — 0.196% mean chamfer, p95 0.48%, max 2.6% localized; SS coords diverge (N1 1250 vs 1248) so nothing downstream is bit-exact | zero (flags exist) | **[VERIFIED]** (measured June; predates today's tree) |

---

## 3. THE FREE WINS FIRST — flags, env vars, one-liners. Do these before writing any code.

Ordered by (win ÷ risk). **None of these needs a rebuild except where noted.**

| do this | win | lossless? | why it's free | tag |
|---|---|---|---|---|
| **`--us-chunk 16384`** | **−440 to −565s** | lossless-ish (see risk) | `RefineCfg::chunk = 2048` is commented **"(VRAM lever)"** — guarding VRAM that **isn't under any pressure** (decode sits at a flat 1389 MiB for 42 min with ~9.4GB idle). The standalone tool's own default is **16384** (`ultrashape_e2e.cpp:88`). Amortizes the redundant `c_kv` 8×: 23.5% → 3.6%. **This is also the cheapest possible test of the whole KV-recompute thesis — if it doesn't land near −20%, the FLOP model is wrong and levers 3/4 need re-sizing before anyone invests.** | **[VERIFIED]** arithmetic |
| **`PIXAL3D_ATTN_CAP_MB=512`** | peak **10751 → ~8200 MiB**; headroom 1155 → ~3700. Costs a few % of the 739.5s DiT. | **lossless** — the docstring says "bit-identical to the single-shot path" | `m1_ggml.hpp:483-486`'s docstring **names this exact use case**: *"Lower it (e.g. 256) … ~0.8GB less VRAM (lossless) … the lever for the 7.5GB budget on the 12GB card."* Converts the `--moge` crash into a run. | **[VERIFIED]** |
| **`ATL_FIT_RES=1`** | **~−20s** | lossy-ish (loses an incidental 1.16× supersample) | xatlas returned **4764×4756** despite `po.resolution=4096` — its texelsPerUnit overshot 16%/axis — then `tex_atlas.hpp:1108` area-downsamples to 4096×4089. **Raster, reproject, inpaint and pack all run at 1.353× the texels that survive.** Confirmed two ways: atlas area ratio 1.353; covered-texel ratio 12,955,246/9,577,970 = **1.352**. The fix is already written at `tex_atlas.hpp:745-759` and env-gated **off**. | **[VERIFIED]** |
| **Stage weights on NVMe** (`rsync /mnt/hdd/pixal3d/weights_gguf* → /`, repoint the symlink) | **~−300s cold (−6.7%)**, ~0 warm | **lossless** | `/mnt/hdd` = **WDC WD10EZEX, ROTA=1**, measured **78–107 MB/s** O_DIRECT. ~33GB of weights per cold run ≈ **330s = 7.3%**. Page cache holds **11.5%** of the 34.8GB gguf set right now; `slat_flow_1024`, `ss_flow`, `dit.gguf` are **0.0% resident** (mincore). `/` is a 990 PRO with 205GB free. **The prior perf pass flagged this as "do this FIRST" and it is still not done** — the symlink proves it. | **[VERIFIED]** (disk + residency measured; the *win* is bracketed, not pinned — see §8) |
| **`--model weights_gguf_f16 --fast`** | −190s DiT, −3.5GB geometry VRAM, −10GB of I/O | **lossy** | Both halves already exist. Also the only dir with `trellis2_tex_1024.gguf`, so it incidentally kills the 640-file npy fallback. **Warning: F16 *without* `--fast` is likely a net slowdown** — `ggml-cuda.cu:1720-1725` requires `GGML_PREC_DEFAULT` for `use_fp16`, and `lin()` forces `GGML_PREC_F32` unless FAST, so you get an F16→F32 **dequant into a pool buffer on every matmul**. The two are strictly coupled. | **[VERIFIED]** June A/B; **[PLAUSIBLE]** the dequant claim (code-read) |
| **Stop defaulting debug PNG dumps on with `--stage-dir`** | ~−3 to −5s | **lossless** | `image_to_rig.cpp:549` sets `pcfg.debug_dir = stage_dir` **unconditionally** → `tex_project.hpp:1826-1905` writes ~23MB of PNG (proj_base_color.png alone is 16.9MB). `TEXPROJ_DEBUG_DIR` already exists as an opt-in override. | **[PLAUSIBLE]** (the 4.6s is by subtraction) |
| **Gate `mesh_topology_stats` behind an env** | −1.85s, bit-exact | **lossless** | 29M unordered_map inserts over 9.66M faces, **purely to print `boundary=0 nonmanifold=0`** on a path whose watertightness is guaranteed by construction (Kuhn conforming-sign-field). Keep it in CI. | **[VERIFIED]** measured |
| **Log the env in the run header** | 0s | — | **We could not determine whether the baseline used `--fast`.** That is the single biggest interpretive gap in this document. One printf of `PIXAL3D_FAST / PIXAL3D_FLASH / PIXAL3D_ATTN_CAP_MB / PIXAL3D_GGUF_DIR` retires it forever. | **[VERIFIED]** gap |
| **Make the gguf fallback loud or fatal** | 0s | — | `m1_ggml.hpp:91-98` does `basename(dir)+".gguf"` and falls back to npy **silently**. It is currently misfiring on **five** weights: `trellis2_tex_1024`, `vae` (×2 — loaded twice), `conditioner`, `r1w_real`, `qwen3_w`. This bug class has now cost three separate agents an afternoon each. | **[VERIFIED]** |

**Free-win note on `--us-octree 256`:** it belongs in §2, not here, because it is not free — it is a genuine quality decision the owner has to make. But it is **zero effort**, and §6 argues the eye-test is cheap.

---

## 4. PER STAGE

### 4.1 UltraShape refine — 3339.0s (74.3%) — **its own section: §6**

### 4.2 Geometry DiTs — 893.1s (19.9%), VRAM 8985 peak

**What it is.** Three passes of the *same* 1.3B DiT (C=1536, 30 blocks, 12 heads × 128, MLP 1536→6144, adaLN + 3D sparse RoPE + cross-attn over 5 DINOv3 tokens). They differ **only in the token set**: SS = dense SEQ=4096 (FOV-invariant, verified: 62.9 vs 63.2s across both cam runs); M2 = N1=1227; M3b = M=12541. Self-attn is O(N²), so the token set *is* the cost.

**The cost law, fitted from two measured runs** (42° → M=12541 → 274.8s; MoGe 46.5° → M=18674 → 537.4s):

> **M3b(M) ≈ 0.00787·M + 1.12e-6·M²** → at M=12541: **98.7s linear + 176.1s attention. Attention is 64% of M3b.**
> tex(M) ≈ 0.00748·M + 5.60e-7·M², whose b′/b = 0.50 ≈ its forward ratio 12/21. **Two independent fits agree on structure**; a hand FLOP count predicts the MoGe ratio at 1.87 vs 1.96 measured (~5%). **[PLAUSIBLE]** — consistency checks, not proof.

Effective throughput: **4.2 TFLOPS for M3b AND 4.2 for SS DiT = 33% of the 3060's 12.7 fp32 peak.** Attention runs *worse* than its FLOP share (52% of FLOPs, 64% of time) — exactly what `m1_ggml.hpp:406`'s own nsys note predicts (38% of DiT GPU time in permute-copies + f16 casts, 12% in a materialized softmax).

| lever | win | lossless? | effort | risk | tag |
|---|---|---|---|---|---|
| **f16 + `--fast` + `PIXAL3D_FLASH=1`** (the parked June A/B) | −190s, −3.5GB | lossy | zero | 0.196% chamfer on a hard asset; owner judges | **[VERIFIED]** |
| **`NVIDIA_TF32_OVERRIDE=0` is hard-set at `image_to_rig.cpp:142`** — line 1 of `main()`, no flag, no comment. ggml *asks* for TF32 on every handle (`common.cuh:1502`); this env is the only veto. **It gates the refine too.** Add `--tf32`. | unquantified; bounded by the linear half (98.7s of M3b, ~2/3 of SS/M2) **plus the refine's GEMMs** | lossy | one line | `lin()`'s comment says tf32 "adds ~1e-2 noise that flips a few occupancy-threshold voxels" — someone hit this. **But TF32 (10-bit mantissa, fp32 range) is strictly more precise than the f16 path already measured at 0.196% and judged fine. If f16 passes, TF32 must.** | **[PLAUSIBLE]** — and see §8 gap 3 |
| **`PIXAL3D_FLASH` has no CLI flag and is welded to `PIXAL3D_FAST`** (`slat_dit_graph.hpp:64`, `tex_dit_cross.hpp:49` both gate on `pix_fast_prec() && getenv("PIXAL3D_FLASH")`). The flash path is **unreachable from the production driver**, and the `&&` means it can only ever be tested *bundled with* f16. Decoupling lets flash run with **F32 weights** — only Q·K/A·V drop to f16 with fp32 accum, a far smaller delta. | upper bound = 176.1s (M3b attn) + ~88s (tex) + ~20s (SS/M2) | lossy (small) | ~5 lines + one A/B | **This is the highest-value unmeasured experiment in geometry**, and it obeys the one-thing-at-a-time rule the June A/B violated | **[PLAUSIBLE]** |
| **`ss_dit_graph.hpp:73` has no flash path at all** — the branch is simply absent, unlike its two siblings. SEQ=4096 is already a 256-multiple → degenerate mask. | ~10-15s, **constant across all assets** (SS is FOV-invariant) | lossy | ~8 lines copied from the sibling | SS feeds `logits_to_coords` = an occupancy threshold. This is where the "flips a few voxels" concern is **most** live — the June A/B saw exactly that (N1 1250 vs 1248). | **[PLAUSIBLE]** |
| **The voxel budget is not adaptive.** Python's `max_num_tokens=49152` guard was **deliberately deleted** (`pixal3d_chain.hpp:290-294`: *"the C++ query-tiled attention removes that OOM guard, so we keep the full requested grid"*). That trade bought robustness and pays in M². Now there's a formula for it: capping M→10000 predicts −135s; M→8000 predicts −215s. | dialable | **lossy — THE quality knob** | low to add a top-K cap | M *is* the geometry's resolution. Thin separable features die first. | **[PLAUSIBLE]** |
| **`proj_linear(pin)` recomputed 21× identically** (`slat_dit_graph.hpp:114`, inside each of 30 blocks, every forward) — but `pin` takes exactly **two** values across the whole sampler. Also re-uploads 102.7 MB pageable H2D per forward = 2.16 GB/run, nine of them the same zero vector. | ~11s (0.25%) | **lossless** | moderate — needs const tensors + an index; real surgery for 11s | do it last, if at all | **[PLAUSIBLE]** |
| **27.9GB of weights off a spindle; 226s of the 899s window is GPU-idle (25.1%)** in gaps of 51/35/34/**77**/8s. DINOv3 is loaded **twice** (`pixal3d_chain.hpp:194` and `:304`). | bounded by 226s; **only the 77s gap is firm** (= the tex npy load, measured at 57.8s) | lossless | trivial to test | 27.9GB at 78 MB/s = 358s cold > the 226s observed → **the baseline was partly warm. A genuinely cold run is worse than this "ground truth".** | **[VERIFIED]** disk; **[SPECULATIVE]** attribution |

**Sampler audit — NEGATIVE RESULT, no flux2-klein-class bug here. [VERIFIED].** `pixal3d_chain.hpp:147-149`: ss {7.5, 0.7, 5.0, [0.6,1.0], 12}, shape {7.5, 0.5, 3.0, [0.6,1.0], 12}, tex {**1.0**, 0.0, 3.0, [0.6,0.9], 12}. These are TRELLIS-2's shipped values. **The tex DiT already runs CFG-off** (guidance==1.0 trips `geometry_e2e.hpp:130` → 12 forwards; log shows 0 `[cfg]`, 12 `[cond]`) — that optimisation is banked. CFG *does* cost the shape DiTs 1.75× (rescale_t=3.0 puts 9 of 12 steps' t inside [0.6,1.0] → 9×2+3 = **21 forwards**, exactly as logged). Dropping it → −117.8s on M3b; narrowing lo 0.6→0.75 → 19 forwards → −26s. **Both are real quality dials, not free wins. Owner judges.**

**Also dead: batching the vp/vn CFG pair into one forward.** Needs block-diagonal attention; full attention over 2M tokens would be *wrong* and 4× the cost. The GEMMs at M=12541 already saturate the card. **Don't chase it.**

### 4.3 Texture DiT + decoder — 246.1s (5.5%), VRAM 8545 / 5019 — **and it may be entirely dead work: see §7**

**Where the 181.8s goes** (M=12541, Ntok=4101, C=1536, NB=30, mlp 8192), per block per step: **self-attn QK+AV 966.2 G (42%)**, **MLP 631.2 G (27.5%)**, to_qkv 177.5 G, cross-attn QK+AV 316.0 G (13.8%), 3× to_out/to_q 177.6 G, cross to_kv 25.8 G. **= 68.8 TFLOP/step → 826 TFLOP for 12 steps → 4.54 TFLOPS effective.** **The 4101-token cross-attn the brief flagged is only ~15% of the step** — self-attn + MLP are the cost. Model validated: it predicts proj-mode at −11.5% vs a measured −13.3%. **[PLAUSIBLE]**

| lever | win | lossless? | tag |
|---|---|---|---|
| **F16 + FAST scoped to the tex DiT only.** The argument: FAST is dangerous for the *geometry* DiTs because tf32/f16 noise flips occupancy-threshold voxels. **The tex DiT has no occupancy threshold** — its output is a continuous PBR latent, and under `--tex-project` its base_color is overwritten anyway. So FAST is far safer here than anywhere else. Blocker: `pix_fast_prec()` is a **static bool read once**, so it cannot be toggled per-stage today. | ~181.8 → ~110-130s | lossy | **[PLAUSIBLE]** |
| **`--tex-dit proj`** — 157.7s vs 181.8s (−13.3%). Reframed by §7: if base_color is thrown away, cross-mode's *entire reason to exist* (the 4101-token cond that makes base_color faithful) is paying +24.1s for a discarded channel. The question collapses to "which gives better metal/rough". | −24.1s | lossy — owner judges, do **not** switch on our say-so | **[VERIFIED]** measured |
| Hoist the cross-attn KV projection (`tex_dit_cross.hpp:88` — `cin` is built once and re-uploaded unchanged every step; ck/cv are bit-identical across all 12) | ~1.9s (1.03%) + 185MB of redundant H2D | **lossless** | **[VERIFIED]** arithmetic |
| **Double npy read**: `m1_ggml.hpp:154` loads the *entire* array at graph-build time **purely to read `a.shape`**, discards it, then `alloc_and_upload` (`:280`) loads the same path again. `npy.hpp:81-82` is eager. **2 × 4.9GB per npy-backed model.** | ~0 on this box (51GB page cache absorbed it; **no unexplained idle window exists near tex** — the one gap t+574→647 is fully explained by M4 55.4s + MC-REMESH 16.7s = 72.1s). Matters on a cold cache. | lossless | **[VERIFIED]** code; **[VERIFIED]** the null result |
| **Prune the decoder's voxel set** — answered and **NO**: `svp_gpu.hpp:235-262` drives on `guide_subs[L]` = the **shape** subdivision masks (`pred_subdiv=false`), so its output is exactly the M4 dual-grid vert count (4,140,779; confirmed: `pbr_feats.bin` = 99,378,704 = 4140779×6×4). **Sealing the mesh will not drop the decoder's work at all.** Its cost is set by `resolution=1536`. | — | — | **[VERIFIED]** |

### 4.4 Mesh stages — M4 mesh 55.4s + MC-REMESH 16.7s + quad retopo 109.7s = 181.8s (4.0%), **zero VRAM**

Measured by re-running the **real production MC-REMESH** from the baseline's own occupancy cache and reproducing it **bit-exact** (4,831,250 v / 9,664,720 f; 16.14s @ OMP=4 vs 16.7s @ OMP=8). Every number here is on the real input.

**MC-REMESH 16.14s, phase-resolved [VERIFIED]:**
- `orient_consistent` **6.80s = 42%** — serial BFS, but the cost is **building ~14.5M individually heap-allocated `vector<int>`** (29M inserts). O(N), catastrophic constant.
- MC triple loop + seal + flood **~6.8s** — `remesh.hpp:673` is **single-threaded on a 12-core box**; the shared `vmap` serializes it.
- `mesh_topology_stats` **1.85s = 11% — pure QC printf** (see §3).
- `taubin_smooth` 0.73s (already OMP'd), `box_morph3` already OMP'd.
- **No O(N²) anywhere** in `remesh.hpp` / `quad_retopo.hpp` / `per_part_decimate.hpp`. The flood fill is an explicit-stack DFS. Nothing to find.

**Three plausible stories that measurement killed [VERIFIED]:**
1. *"The M4 dual-grid mesh is discarded"* — **true, and worth 1.2s, not 55s.** `flexible_dual_grid_to_mesh` measures **1.16-1.19s** at production scale. The other ~54s is the M4 ConvNeXt decoder, which is **required** (it produces coords1024 + subs). A 46× shrink.
2. *"The quadwild shell-out + OBJ round-trip is a glue gap worth attacking"* — **measured with verbatim copies of `qr::write_obj`/`qr::read_obj_poly` against the baseline's own byte-identical `/tmp/qr_stage` files: write 0.32s, read-back 0.71s, read result 0.04s = **1.1s of 109.7s = 1.0%**. `/tmp` is tmpfs — there is no disk I/O at all. Two `std::system()` forks are ~ms. **In-process linking wins ~1 second. Do not do it.**
3. *"`obj_decimate` forks a binary per part"* — **worth ZERO. It is not on the production path.** `ppd::per_part_decimate` only runs under `--part-retopo` (`image_to_rig.cpp:474`), and the baseline has no `[1b/4]` line. `--quad` also sets `decimate = 0` (`:464`).

**The real mesh finding:** quadwild's step1 log (`run.log:88`) — *"Before Remeshing - faces: 1336960 … After Iter 0 - faces: **42464**"*. **quadwild discards 96.8% of the refined mesh before it does any work.** See §6 and §7.

**Levers:**

| lever | win | lossless? | tag |
|---|---|---|---|
| Gate `mesh_topology_stats` | −1.85s | **lossless** | **[VERIFIED]** |
| `orient_consistent`: flat CSR (sort 29M (edge,face) pairs, index as CSR) instead of the hash-of-vectors | bounded by 6.8s; realistically ~4-5s | **lossless** | **[SPECULATIVE]** — not prototyped |
| **Ask whether `orient_consistent` is needed on the coarse mesh at all.** Its docstring justifies it by xatlas chart collapse — but in `--quad` the atlas is built on the **quad** mesh. The coarse mesh's only consumers are `coarse.glb` (cosmetic), the refine's voxelizer, and the bake's reproject shell (a BVH closest-point query — winding-agnostic). | **6.80s deletable outright** if the voxelizer has no winding dependence | unknown | **[SPECULATIVE]** — **VERIFY `us_vox::voxelize_mesh_inmem` FIRST.** If it does a winding-based inside test, dropping this **silently corrupts the refine's conditioning** and would look like a model problem. 30 min of reading. |
| Parallelize the MC main loop | bounded ~6.8s; maybe 3-4s after the merge pass | lossless (vertex *order* changes → geometry-comparable, not bit-comparable) | **not recommended** — 1-2 days for 3-4s of 4498s |

### 4.5 Texture bake — 87.6s (2.0%), **zero VRAM** (pure host)

**Accounted by measurement, not by guessing [VERIFIED]:** precluster 1.87s · ComputeCharts 1.40s · PackCharts 1.70s · welded normals 0.05s · DenseHash build 0.87s · rasterize ~0.77s (calibrated: `tex_project.hpp:861`'s `raster_uv` is a verbatim copy of the bake's loop and logs 0.57s for 9.58M texels; 0.57 × 12.955/9.578 = 0.77) · hole-scan 0.04s · inpaint 0.43s · pack+sRGB 0.54s · resize 0.94s = **8.61s. RESIDUAL ~79.0s** = `shell_attr` precompute + the per-texel reproject loop.

**Three of the reviewing agent's own hypotheses died here [VERIFIED]:** inpaint, the 68M `std::pow` sRGB calls, and the serial double-precision area-resize were each expected to be multi-second. They are **0.43s, 0.54s and 0.94s.**

**The bottleneck is `tex_atlas.hpp:998-1018`: 12.95M per-texel `DenseHash::sample()` closest-point queries against a 9,664,720-triangle shell.** Each scans 27 hash cells at ~100 tris/cell → **~10¹⁰ triangle tests**.

**Chart count is a quality problem, not a perf one [VERIFIED]:** the whole atlas/clustering chain (precluster + ComputeCharts + PackCharts + raster + inpaint) is **~6.2s of 87.6s (7%)**. Better clustering will not buy time. **Don't chase it for perf.**

| lever | win | lossless? | tag |
|---|---|---|---|
| **`ATL_FIT_RES=1`** (see §3) | ~−20s | lossy-ish | **[VERIFIED]** |
| **Per-vertex reproject instead of per-texel** — query at the **121,437 atlas vertices** and barycentrically interpolate inside the existing 0.77s raster loop. **12,955,246 → 121,437 = 107×.** Deletes the 4.83M-vert `shell_attr` precompute too. Precedent: mesh-attr **already** barycentrically interpolates per-vertex attrs — just of the shell's 4.83M verts. **Must** be gated on `tex_project=true`. | **~79s → ~1s. Bake 87.6 → ~10s.** *Alternative to `ATL_FIT_RES`, not additive.* | lossy | **[SPECULATIVE]** — the sampling margin is **thin**: atlas edge ≈ 1/145 of the model vs the PBR's own ~1/64 band limit (`image_to_rig.cpp:212-213`: *"carries ZERO detail… band-limited by design"*) = only **~2.3× oversampling**. Adequate, not a slam dunk. Eye-test required. |
| Skip the 3 dead channels post-sample when tex_project is on (`tex_atlas.hpp:1062, 1080-1092, 1112`) | **~1.3s, measured** | **lossless** | **[VERIFIED]** |

### 4.6 Tex project — 22.0s (0.5%), zero VRAM

Phases sum to 17.40s (`run.log:650`) — **~4.6s unaccounted** (debug PNGs + write-back + mask build).

**`fill3d` is 13.45s = 61% of the stage, filling 5,599,968 holes = 58.5% of all covered texels.** The fill is not slow — **it is being asked to invent over half the texture.** Front paints 23.0%, back 18.6%. Even *before* bg-reject, only **50.6%** of covered texels pass facing+depth for the two opposed cameras. **The sides are structurally unreachable — which is exactly the owner's stated problem area.**

**And the log WARNs on itself** (`run.log:634`): view 1's black threshold 0.05 **discarded 449,719 nonzero px (67.92%)** — *"threshold is eating DARK SUBJECT (boots/hair?) — lower it"*. **[VERIFIED]**

| lever | win | lossless? | tag |
|---|---|---|---|
| **Raise back-view coverage** — (a) the self-WARN'd threshold; (b) `--tex-view` is **already repeatable for arbitrary yaw** (`image_to_rig.cpp:280-290`) so side views need **no code**, only generated images. Halving the hole set roughly halves 13.45s **while replacing invented colour with real projected colour.** §7's pattern again. | ~−7s + a quality gain | lossy — owner judges | **[SPECULATIVE]** on magnitude |
| Debug dumps opt-in (§3) | ~−3-5s | lossless | **[PLAUSIBLE]** |

Risk on the threshold: lowering it risks re-admitting the black background that caused the "black streaks down the sides" (`image_to_rig.cpp:100-104`) — that is **what the threshold is for**.

### 4.7 Rig — 40.6s (0.9%), VRAM 5477 peak

**The brief's premise is wrong. [VERIFIED].** *"beams=20 is a 20× multiplier on the AR generate"* — no. `rig_pipeline.hpp:148` defaults to `rig_beam_generate_batched` (`RIG_BEAM_SEQ` is opt-in) and the log proves one batch: *"1 batched suffix @ 1538x20"*. The log gives the exact cost: `263 decodes × (1.91 + 38.66) ms = **10.67s**`. **The entire AR generate is 10.7s of the 40.6s rig = 0.24% of the pipeline.** beams=1 would be bandwidth-bound → maybe 2× per-step → **~5s = 0.1%**, in exchange for the exact runaway attractor already on record (`image_to_rig.cpp:149-153`: do_sample+beams=10 → J=178, maxfan=103, **rig_score 0.000** vs deterministic beams=20 → J=56, maxfan=5, rig_score 0.908). **Leave beams=20 alone.**

The rig's real cost is the other ~29s, and there is a provable cause: **`weights_gguf/r1w_real.gguf` and `weights_gguf/qwen3_w.gguf` do not exist anywhere on disk** (`find` over `$AS` confirms), and `image_to_rig.cpp:337` pins `PIXAL3D_GGUF_DIR` to the geometry dir and never restores it. So **R1+R3 load 1.8GB of fp32 npy across 311+ files off the HDD every run** and convert to BF16. Compute type is *already* BF16 (`rig_pipeline.hpp:44`), so **a BF16 gguf is numerically identical** to the current path — this is a **lossless** win. `pack_gguf.cpp` exists.

(R4 escapes the dir bug — `rig_pipeline.hpp:201` re-points at `skin_vae_gguf` — but then loads `skin_vae.gguf` **twice**, `Hc` at `:232` and `Hs` at `:255`, 465MB each. And `:230`/`:254` compute `ckv_embed` and `q_embed` with **byte-identical arguments**.)

| lever | win | lossless? | tag |
|---|---|---|---|
| **Pack `qwen3_w` + `r1w_real` as BF16 gguf** | unquantified; bounded by ~29s | **lossless** (BF16 either way) | **[VERIFIED]** the bug; **[SPECULATIVE]** the win |
| Batch the J=33 sequential skin decodes (`rig_pipeline.hpp:269-277` — each decodes all N=8192 queries against 384 cond latents, differing only in a tiny z) | unquantified | lossless | **[SPECULATIVE]** |
| Load `skin_vae.gguf` once; drop the duplicate embed | small | lossless | **[VERIFIED]** the duplication |
| **`prep_mesh_for_rig_inmem` / `sample_surface`** — **NEGATIVE RESULT: no waste found.** N=8192, M=512, FPS over 2048 candidates. O(10⁴) work in a 4498s pipeline; not even separately timed. **Don't spend effort here.** | 0 | — | **[VERIFIED]** |

---

## 5. VRAM — its own section

**The ceiling is 11906 MiB usable, not 12288.** Peak 10751 = **90.3%**. Headroom **1155 MiB (9.7%)**.

**VRAM binds in exactly ONE place, for 813 of 4498 seconds: the refine DiT.** Raw handover from `vram.tsv`, and it is unmistakable:

```
t+1601..1711   10745-10751 MiB, util 100%   ┐ refine DiT — PEAK 10751 at t+1715
t+1723            135 MiB, util   0%        ┘ Hd scope exit frees the 6.10GB dit.gguf
t+1731..4233     1389 MiB, util  99%        <- THE 2502s DENSE DECODE
                 (min == max == 1389 over 1,181 consecutive samples)
```

**Peak decomposition (arithmetic, and it closes) [VERIFIED]:**
- `dit.gguf` **F16, 6,104,041,056 B = 5817 MiB** (parsed header: 752 tensors, 320 F16, 3.05G F16 elems)
- self-attn scores `[8193, 8193, 16]` fp32 = **4.30 GB**, tiled to **~3072 MiB** by `PIXAL3D_ATTN_CAP_MB` (default 3072 → bq = 6141 → 2 tiles)
- CUDA ctx + activations + gallocr scratch ≈ **1.5 GB**
→ **≈ 10.7 GB ✅**

**Decode decomposition, 1389 MiB [VERIFIED]:** cross-attn scores `[8192, 2048, 16]` fp32 = **1024 MiB (74% of it)** + geo_decoder weights ~60 + `tr_cached` 33 + CUDA ctx ~250 ≈ **1367 ✅** — which **independently confirms** the decode is on the dense, non-flash attention path.

### What the headroom could buy — the honest answer

1. **The decode is the only place where VRAM is free AND time is being spent.** 2502.3s (55.6%) at 1389 MiB (11.7% of the card). **10.5 GB of the card idles for 42 minutes.** `--us-chunk` cashes this out today for zero code: CHUNK=8192 → ~4.3 GB scores → ~4.7 GB total, **still 6 GB below the DiT's own peak → zero additional peak VRAM.** *Everything else in this section is a rounding error next to this.*
2. **The peak is one tensor, and the fix already exists.** `ultrashape_dit.hpp:144,161` call `attention(...)` with `fa_mask` defaulted to `nullptr` → the debugged flash path at `m1_ggml.hpp:405-465` (q-pad fix, V-prescale NaN fix, ×256 kv-pad mask) is skipped → 4.30 GB materialized. **The UltraShape DiT just never got a mask wired.** Wiring it: **peak ~7.7 GB, headroom → ~4.2 GB.**
3. **`PIXAL3D_ATTN_CAP_MB=512` is the lowest-risk VRAM win on the board, needs no rebuild, and its docstring names this exact card.** Peak → ~8200 MiB. **Bit-identical.** Costs a few % of the 739.5s DiT. **Do this tonight.**
4. **Q8_0 the `dit.gguf`** (6.10GB → 3.05GB): peak → **~7700 MiB**, headroom → ~4200. Retires the whole OOM class. `[SPECULATIVE]` on quality — validate against the banked parity (`final_latents` cos ≥0.9999, grid cos 0.99995, Chamfer 0.03% bbox-diag).
5. **Raising `octree` is free on VRAM but NOT on time.** The decode is chunked, so its VRAM is set by CHUNK, not G. **"Free VRAM" does not mean "free quality" here.**
6. **Nothing needs offloading.** `M1Harness` picks CUDA **or** CPU (`m1_ggml.hpp:99-101`), never both; `sched_cpu` exists only under `PIXAL3D_SCHED` and its docstring says it's a flash-attn-NaN *correctness* workaround, not an offload path. With 64GB RAM / 95GB swap @ ~0 used, the host side is a non-issue. **The 12GB card is the constraint, and it is constrained by one avoidable tensor during 16% of the runtime.**
7. **Time and VRAM here are disjoint problems with disjoint fixes.** The sparse decode attacks 2502s at 1389 MiB. Flash-on-the-DiT attacks 10751 MiB at 739s. **Neither trades against the other.**

### The `--moge` OOM — restated correctly

The abort (`e2e_moge/run.log:72-73, 82, 107-114`) is `ggml_cuda_pool_vmm::alloc` → `ggml-cuda.cu:595` = `CU_CHECK(cuMemCreate(...))`, the VMM pool failing to commit pages, under `usr::refine`. `ggml_cuda_error` is `[[noreturn]]`, so this **is** the fatal abort (the later lines are just buffered stdout flushed at exit — one OOM event, not two).

**The refine asks for its normal ~10.75 of 11.9 GiB and the device could not serve it, because the process's larger upstream geometry stages left it unable to.** The refine is not the culprit; **it is merely the next allocator to ask.** We are not "20% from the cliff" — we are **9.7% from it on a normal run**, and *any* upstream stage that grows pushes a downstream stage over.

**[SPECULATIVE] — the residue holding the memory is NOT proven.** The baseline trace shows VRAM returning to ~133 MiB between stages, which argues **against** simple pool retention. Best candidate: fragmentation against the refine's **single 6.10GB contiguous weight buffer** (`ggml_backend_alloc_ctx_tensors`, `m1_ggml.hpp:272`) — the most fragmentation-fragile allocation in the pipeline, and exactly the one preceding the failing pool alloc. Every harness also gets its own `ggml_cuda_pool_vmm`, each reserving 32GB of VA (`ggml-cuda.cu:545`). **Resolving it needs `nvidia-smi --query-compute-apps` at 0.2s through the geometry→refine transition. But the levers don't depend on it: both #3 and #4 above give back 2.5-3GB regardless of mechanism.**

**No flux2-style memory-bound signature exists anywhere. [VERIFIED].** The decode runs **169.1W flat** (of ~170W TDP), the refine DiT 167W, M3b 116-168W. Nothing shows the "100% util at 57W" dequant signature. **Do not go looking for a flux2 repeat here.**

---

## 6. THE REFINE — 3339.0s (74.3%)

### 6.1 The anomaly: RESOLVED. Resolve it formally anyway — it is still the first thing to do.

**`inline_soldier1536` (and every `ab_fray`/`ab_seam`/`ab_texfix` A/B built from it) was decoded at octree 256. `perf_baseline` ran the code default, 512.** Five independent lines:

1. **The MC lattice pitch, measured directly from both GLBs** — 0.0016635 (baseline) vs 0.0033285 (shipped asset). **Ratio 2.0009 on all three axes**, identical canonicalization (axis min/max match to 6dp, so the ratio is scale-locked). **This is the load-bearing measurement.** The vert ratio (4.0438 ≈ (513/257)² = 3.985) is *consistent* but is precisely the "plausible ratio story" class this project has been burned by — it was deliberately not relied on.
2. **The bboxes are identical to 5e-5** — same object, same scale, same shape, **only the tessellation differs.** Only G can do that.
3. **The coarse input is byte-identical** — `ab_fray/A.log:7` loads `shell(4831250 v)`; `perf_baseline/run.log:584` feeds the same 4,831,250. Same input, same output shape, 4× the triangles.
4. **The decode time falls out** — `us_geo_decoder` is a **fixed-shape graph** (shape depends only on CHUNK=2048 and N=8192, **not** on G). G sets only the compute *count*: `ceil(G³/CHUNK)` = 65921 → 8288. 2502.3s/65921 = **37.96 ms/compute** × 8288 = **~315s** ≈ the memory's banked "~280s". **The banked 280s IS the octree-256 number.** Structural, not a fit.
5. Ruled out: `--decimate` (165296 v / 330644 f has F = 2V exactly — a closed MC mesh, not a decimation target, which would be a round 150000); `drop_frac` (removes 4%: 696364→668406); a different coarse mesh (see 3).

**But it is still inference:** no `run.log` exists for `inline_soldier1536`, `grep -r` finds nothing, `.bash_history` has nothing. **One `--us-octree 256` run settles it in ~20 min and should be the first thing done.**

### 6.2 The consequence the owner needs

`perf_baseline` spent **~2187 extra seconds (48.6% of the whole pipeline)** producing 4× the triangles of the asset already judged. And **quad retopo throws it away**: `run.log:602-604` takes 668,406 v → **83,702 v / 167,340 f**, and quadwild's *first action* crushes 1,336,960 faces → **42,464 (−96.8%)**. Both 256 (165k v) and 512 (668k v) sit far above the budget the pipeline immediately collapses to. **Nothing in the logs suggests anyone chose 512** — it is just `RefineCfg::octree = 512` (`ultrashape_refine.hpp:70`), whose own comment reads *"OCT=256 = the toy default"*.

**The counter-argument, and it is real — do NOT just cut the octree. [PLAUSIBLE]:** quadwild reports `Edge Size 0.00511892`; the model bbox (from texproj's fill grid) is 83×129×44 × 0.00651 → longest axis ~0.84 → an MC cell at G=513 is 0.84/512 = **0.00164**. **The quad mesh's edges are ~3.1 MC cells long.** Real oversampling — but only **~3× linear, not 10-100×**. At octree 256 an edge is **~1.6 cells = at Nyquist**, and quadwild also **projects back onto the dense surface** ("Getting Projection basis"), so the dense mesh is a live projection target. **Don't go below ~384.**

**This is exactly why the hierarchical decode is the right lever and a resolution cut is not: it keeps 513³ and skips only the 99.5% of cells that are empty.**

### 6.3 Why the decode is slow — three stacked causes

**(a) Brute force where the reference is hierarchical.** `ultrashape_e2e.cpp:84` says it out loud: *`int OCT = 512; // octree_resolution (DENSE grid here; Python uses a hierarchical octree)`*. **A known, unclosed port gap on the single most expensive stage in the pipeline.** `ultrashape_refine.hpp:261-282` evaluates all 135,005,697 corners to find an isosurface that touches almost none.

**The prize, measured two ways [VERIFIED]:** voxelizing `refined.glb` onto the 512³ grid → the isosurface occupies **457,147 cells = 0.341%**. Cross-check by surface area: 7.489 ÷ h² (h=2/512) → **490,808 = 0.366%**. Two methods, 7% apart. **99.66% of the 2502s decodes empty space or solid interior.**

The seed is free — the coarse mesh is already in RAM. **Measured band coverage [VERIFIED]:**

| band around coarse shell | cells | % of grid | speedup | isosurface coverage |
|---|---|---|---|---|
| ±0 | 822,088 | 0.613% | 163× | 42.18% |
| ±2 | 2,773,146 | 2.066% | 48.4× | 95.73% |
| ±3 | 3,567,783 | 2.658% | 37.6× | 98.79% |
| **±4** | **4,316,428** | **3.216%** | **31.1×** | **99.60%** |

**±4 still misses 0.4% → holes. A coarse-seeded band ALONE is NOT lossless. Do NOT ship band-only.** The safe construction is the reference's: hierarchical octree (129³ → subdivide sign-straddling → 257³ → 513³, conservative by construction), **unioned** with a ±2 band to catch sub-coarse-voxel-thin features (fingers, straps). Budget ≈ 2.15M + ~0.8M + ~3.0M ≈ **6M vs 135M = ~22×**.

**Minimal diff:** the loop already consumes a flat query list. Keep the 540MB host `logits` buffer (64GB RAM — free), pre-fill with ±BIG from the coarse mesh's sign, decode only the active set, scatter back, **MC unchanged** (undecoded cells are all-same-sign → emit nothing). **~30 lines around `ultrashape_refine.hpp:261-282` + an active-set builder.** Validate by asserting `boundary==0` on the MC output (the pipeline already prints this) and Chamfer vs a full dense decode at octree 256 (~315s — cheap).

**(b) Everything runs fp32 on CUDA cores. The tensor cores idle for 3339s.** Three separate things pin it: `lin()`/`attention()` force `GGML_PREC_F32` unless `PIXAL3D_FAST` (`m1_ggml.hpp:366,373`); `image_to_rig.cpp:142` sets `NVIDIA_TF32_OVERRIDE=0` **unconditionally**; `--fast` was (probably) not passed. **The UltraShape VAE has no gguf at all** — `ultrashape_goldens/gguf/` holds **`dit.gguf` only**; `vae` (1.3GB) and `conditioner` (1.2GB) fall back to fp32 npy, and **`vae` is loaded twice**. The header comment admits it (`ultrashape_refine.hpp:127-128`): *"cond/vae have no gguf there and fall back to npy"*. **The 2502s decode runs fp32 weights.** The loader infra is already there — only the file is missing.

**Arithmetic [VERIFIED]:** per chunk = 73.1 GMAC = 146 GFLOP; × 65,921 = **9.64 PFLOP in 2502.3s = 3.85 TFLOPS**. RTX 3060: FP32 **12.74**, FP16-tensor-core-w/-FP32-accum **51.2**. **30% of the slow path, 7.5% of the fast one.** Note `--fast` also sets `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1`, keeping FP32 accumulation — on Ampere, HMMA-with-FP32-accum still runs at full 51.2, so **most of the win survives the safe setting**. **A/B `--fast` alone before packing anything.**

**(c) The `c_kv` + `ln_2` recompute — the tr-hoist stopped one function short.** `us_cross_block` (`ultrashape_vae.hpp:140-147`) does `dn = layernorm(data)` and `us_cross_attn` does `lin(c_kv.weight, dn)` over `data = tr_in [1024, 8192]` — **all inside the per-chunk graph**, rebuilt and recomputed for every one of the 65,921 chunks, **bit-identical every time** (`tr_in` is a `const_tensor`, `ultrashape_refine.hpp:243/251`). `c_kv` = 8192×1024×2048 = **17.18 GMAC = 34.4 of 146 GFLOP = 23.5% of the decode ≈ 590s of pure redundancy**, plus a 67MB materialization and the `ln_2` pass. **`ultrashape_e2e.cpp:246-249` describes this exact bug for the 16-layer transformer and calls it "the dense-decode bottleneck" — the hoist was applied to the transformer and stopped one line short of `c_kv`/`ln_2`.**

### 6.4 The refine's small print

- **81.8s of the refine is invisible** (3339.0 − 13.3 − 739.5 − 2.1 − 2502.3): `us_vox::voxelize_mesh_inmem` over 4.83M verts, `us_dense_grid_queries` (**allocates a 135M×3 = 1.62 GB host array** — pure waste under a sparse decode), `us_mc::marching_cubes` over 135M cells, `drop_small_components` (union-find over 696k verts). **None is timed.** "Every little bit matters" cannot be honoured on a stage with no numbers. **4 `now_s()` pairs. Trivial.**
- **`us_fourier_embed` is single-threaded** (`ultrashape_vae.hpp:46-67`) — 2048×48 sin/cos per chunk, serialized with the GPU. **But it is a LUT, not transcendentals:** `us_dense_grid_queries` builds the grid from a `linspace` of **513 values**, so all 135M queries draw from **513 distinct coordinates per axis** — the table is 513×8×2 entries, computed once. **[SPECULATIVE] on the win:** util reads **99%** through the whole phase, which says the GPU is **not** starved. Either util is fooled at this granularity or the host work is faster than the estimate. **Flagged, not claimed.** `--us-chunk` amortizes it either way.
- **~198k heap allocations** — `qchunk`/`qe`/`got` are allocated **inside** the loop (`ultrashape_refine.hpp:271-280`). Hoist them.
- **The sampler is the outlier in the whole pipeline. [VERIFIED].** SS 22 fwd · M2 21 · M3b 21 · tex 12 (cfg-off) · **US refine: 50 steps, 100 forwards, ALL CFG, guidance 7.5 flat, no interval.** The geometry samplers have a CFG interval; the tex DiT skips CFG entirely; **the refine has neither.** `--us-steps 25` → ~−370s; porting the geometry's cfg-interval → ~−185s. **But 50 steps / guidance 7.5 IS the Python reference default** (`"Defaults = PRODUCTION (match the Python pipeline: guidance 7.5, octree 512)"`), so unlike the klein-edit case **there is no free win here — it is a genuine quality question. Owner judges.** Flagged only because it runs **4.2× the step count of its neighbours with zero CFG scheduling.**

---

## 7. QUALITY ↔ PERF INTERACTIONS

### 7.1 Where accuracy and perf are the SAME bug — look for more of these

| the bug | accuracy | perf | status |
|---|---|---|---|
| **`--tex-volume-direct`** — a chained double closest-point projection (texel→shell→volume) that **slides**: 9.2% of texels read a voxel >4 away. Replaced with a direct volume read. | bad texels **5.13% → 0.74%** | bake **107.1s → 27.7s = 3.9×** | **banked** (measured on `inline_soldier1536`; the 87.6s baseline is the **mesh-attr** path — `run.log:615` says `mode=mesh-attr`, no `volume-direct` line. **This A/B is not in the ground-truth baseline.**) |
| **`fill3d` invents 58.5% of the texture** because only 50.6% of covered texels are visible to the two opposed cameras, and view 1's black threshold discards 67.92% of its nonzero pixels (**the log WARNs on itself**). Raising coverage shrinks the hole set. | stops inventing texture on **exactly the owner's stated problem area (the sides)** | **~−7s of 13.45s** | **open** — the cheapest quality/perf twofer on the board |
| **The atlas is 1.353× oversized** and 35% of every texel's work is averaged away by a resize the code treats as **accidental** (`tex_project.hpp:854-859`). | neutral (loses an incidental 1.16× supersample) | **~−20s** | **open, `ATL_FIT_RES=1`** |
| **The rigged GLB drops metalRough entirely** — see 7.3 | **the owner's gold-buttons complaint literally cannot render** | **246.1s of dead GPU work** | **open — fork, see 7.3** |

**The pattern is real. Keep looking.** Three of the four above were found by asking "what does this stage's output actually reach?", not by profiling.

### 7.2 Where a perf lever costs quality — the owner's call, presented without a recommendation

| lever | the trade | size |
|---|---|---|
| **`--us-octree` 512→256** | 4× coarser tessellation. Mitigating: quad retopo collapses to 83,702 v / 167,340 f either way, and quadwild's Iter 0 crushes to 42,464 faces. Aggravating: at 256 the quad mesh's edges are **~1.6 MC cells = at Nyquist**, and quadwild **projects back onto the dense surface**. Failure mode: thin features (fingers, straps) — US-frame voxel pitch doubles to 0.0078. **Eye-test 256 vs 512 *through quad retopo*, not on `refined.glb`.** | **−2187s (−48.6%)** |
| **`--us-steps` 50→25 / add a CFG interval** | Both are the reference defaults. Genuine quality dials. | −370s / −185s |
| **The voxel budget M** (top-K cap) | **THE geometry resolution knob.** Thin separable features die first. Cost law: M→10000 = −135s; M→8000 = −215s. | dialable |
| **f16 / `--fast` / flash / TF32** | 0.196% mean-surface chamfer on a **hard** asset (thin twintails/fingers); p95 0.48%, max 2.6% localized. SS coords diverge (N1 1250 vs 1248), so **nothing downstream is bit-exact**. `lin()`'s comment: tf32 noise "flips a few occupancy-threshold voxels" — and the refine decoder's output **IS** an occupancy threshold at logit 0. **The risk lands exactly where the code says it hurts.** But: TF32 (10-bit mantissa, fp32 range) is **strictly more precise** than the f16 path already measured at 0.196% and judged "visually indistinguishable". | −190s (geom) + unquantified (refine) |
| **`mc_stride` 2→3/4** | **The brief's framing is FALSE in the `--quad` config.** *"fewer verts → less to decimate, atlas, bake, retopo, rig"* — no: quad retopo runs on the **refined** mesh; the atlas and rig run on the **quad** mesh; per-part decimate **does not run at all**. The coarse mesh's only consumers are MC-REMESH itself, `coarse.glb`, the refine's voxelizer (**409,600 surface samples → 33,748 occupied → 8,192 latent voxels = a 1,180× reduction; the refine literally cannot see coarse detail**), and the bake's reproject shell. **The knock-on is ~1/10 of what the brief implies.** Volume also inflates with stride — a real quality cost. Measured: stride 3 = 6.25s / 1,756,794 v (−63.6%); stride 4 = 2.35s / 883,138 v (−81.7%). | **~−20 to −40s total (0.5-0.9%)** — real, **not a headline** |
| **`REMESH_CLOSE_R`** | **TWO BRIEF CLAIMS REFUTED [VERIFIED].** (1) *"a double wall from a regionally-leaking flood fill"* — **the flood fill does not leak.** solid/wall = 6.245 at close_r=0 (a leak pins it at ~1.0), and `remesh.hpp:525-532` records that the emitted surface's signed volume matches solid×cellvol to 0.01% — i.e. the surface bounds the **filled body = ONE skin**. The measured sweep confirms the mechanism: the ratio **saturates** (6.2 → 8.2 → 11.5 → 11.8), which is a morphological closing eating concave crevices, **not a leak being fixed**. The "54.19% invisible from 100 directions" is better explained by **self-occlusion of a diagonal voxel staircase** than by a second wall. (2) *"bbox IDENTICAL"* — bbox yes, **silhouette NO**: at R=3, X +106, Y +24, Z +165. `remesh.hpp:459-460` asserts "delta must be 0"; **the invariant as stated is FALSE** (the comment's math proves a **bbox** bound, not a per-axis projection bound). **So shipping the seal is a feature-fusion quality call — the twintail/finger risk the code itself warns about — not a bug fix.** Measured: R=0: 4,831,250 v / 16.14s; R=3: 2,075,036 v / 9.72s. | −6.4s in MC-REMESH + a large slice of the bake's shell BVH (57% fewer shell faces) |
| **`--tex-dit proj` vs `cross`** | Reframed by 7.3: if base_color is discarded, cross-mode pays +24.1s for a dead channel and the question collapses to "which gives better metal/rough". | −24.1s |
| **Per-vertex reproject** | ~2.3× oversampling of the PBR's own band limit. **Adequate, not a slam dunk.** metal_rough is precisely where the owner complained (gold buttons) — though `tex_project.hpp:277` records that complaint as **registration**, not metalness. | −76s |
| **beams 20→1** | **DON'T.** 0.1% of the pipeline for a documented runaway attractor (rig_score 0.908 → 0.000). | ~−5s |

### 7.3 🔴 THE FORK — resolve this before touching the tex chain

**Under `--tex-project` + rig, nothing the tex DiT/decoder produce reaches the shipped GLB. [VERIFIED, code-traced].**

- `texproj::project_onto` **overwrites `bt.base_color`'s RGB in place** (`tex_project.hpp:14, :1201, :1818-1822`). The volume's colour is gone.
- `write_rigged_textured_glb` takes **only `base_png`**; its material hardcodes `"metallicFactor":0.0,"roughnessFactor":1.0` (`glb_rigged_textured.hpp:177`). **`bt.metal_rough` is never written.** (`proj_tex.glb` and the `--no-rig` path **do** carry it — only the deliverable drops it.)
- The same material is `"alphaMode":"OPAQUE"` (`:173`), so base_color's alpha — the **one** channel texproj preserves — is ignored by any renderer.

**⇒ tex DiT 181.8s + tex decoder 64.3s = 246.1s of GPU work produce zero bytes in `soldier_hq_rigged.glb`**, plus the 4.83M-vert `shell_attr` sampling and the bake's 12.96M-texel volume reads.

**And it is a QUALITY bug: the owner's gold-buttons metalness complaint cannot render — the shipped material is hardcoded non-metallic, fully rough.**

**Two mutually exclusive readings, and only the owner can pick:**
- **(a)** The tex chain is genuinely dead work under this recipe → gate the tex DiT + decoder off with `--no-tex` (keep the bake, for UVs). **−246.1s, bit-exact on the deliverable, lossless.**
- **(b)** `write_rigged_textured_glb` **should** be carrying metal_rough and the hardcoded material is a **shipping bug** → the fix is the *writer*, not the DiT, and **the gold-buttons complaint has a second cause nobody has looked at.**

**Do NOT gate the tex DiT off blind. Either way, something is wrong.**

---

## 8. DEAD ENDS / DON'T BOTHER — as valuable as the wins

| don't | why | tag |
|---|---|---|
| **In-process quadwild linking** (the "unclosed glue gap") | **Measured: the OBJ round-trip is 1.1s of 109.7s = 1.0%**, on tmpfs, with ~ms forks. The 109.7s is **all algorithm** and its breakdown is already in the log (`run.log:551-577`: step1 ~61s, step2 48.3s = smooth 35.9 + qfp 12.2, of which bimdf 10.06 over 28 calls). **~1 second for a day's work.** | **[VERIFIED]** |
| **`obj_decimate` fork-per-part** | **Not on the production path.** `--part-retopo` was off; `--quad` sets `decimate = 0`. **Worth exactly 0s.** | **[VERIFIED]** |
| **"The M4 dual-grid mesh is discarded"** | True, and worth **1.2s**, not 55s. The other ~54s is the required M4 decoder. **A plausible story that measurement shrank 46×.** | **[VERIFIED]** |
| **Better atlas clustering, for perf** | The whole chart chain is **6.2s of 87.6s (7%)**. It's a quality lever, not a perf one. (Note: the older memory's "xatlas ComputeCharts 131s CPU-serial" is **stale** — the precluster path made it 1.4s.) | **[VERIFIED]** |
| **inpaint / sRGB `std::pow` / the double-precision resize** | Expected multi-second each. **Measured 0.43s / 0.54s / 0.94s.** | **[VERIFIED]** |
| **`prep_mesh_for_rig_inmem` / `sample_surface`** | O(10⁴) work. Not even timed. | **[VERIFIED]** |
| **Lowering `beams`** | 0.1% of the pipeline for a documented rig_score collapse. | **[VERIFIED]** |
| **Batching the CFG vp/vn pair into one forward** | Needs block-diagonal attention; full attention over 2M tokens would be **wrong** and 4× the cost. The GEMMs already saturate. | **[VERIFIED]** |
| **`GGML_CUDNN_ATTN` / `GGML_CUDNN_CONV` / `GGML_CUDA_F16_BCAST_FUSE` / `GGML_CUDA_BIAS_GELU_FUSE` / cuBLASLt** (the kobbler fleet's levers) | **grep over `~/dev/longcat-sparse-spike/ggml` (HEAD 19727d01): zero hits for any cuDNN symbol.** This is a different, older fork. Those levers need a **rebase — a project, not a flag.** (`GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F` *does* exist, `ggml-cuda.cu:1673`, and is already set by `--fast`.) | **[VERIFIED]** |
| **CUDA graph capture** | Exists (`ggml-cuda.cu:3501`) and is **worthless here**: ~600 kernels/forward × ~5µs ≈ **3ms against a 13.1s forward**. | **[VERIFIED]** |
| **Overlapping the CPU tail** | The 226s block (quad → atlas → texproj) is a **true dependency chain**. The only GPU work that could overlap it is the tex DiT — which §7.3 says to **delete, not reschedule**. Total idle is 12%, mean util **86.7%** within the run window. **Don't chase overlap.** (The naive read of `vram.tsv` showing 80% idle is an artifact: the sampler kept running 4hrs past the job.) | **[VERIFIED]** |
| **A 540MB `--dump-logits` cache for the dev loop** | **Aimed at the wrong seam.** `--from-refined` already skips **both** geometry (893s) and refine (3339s), and the baseline **already wrote the whole cache** — every downstream experiment (quad+atlas+texproj+rig = 375s) is a cache hit **today**. The real gap is **inside** the refine: `--from-geo` re-runs the full 3339s. **Dump the post-sampling latents instead: 8192 × 64 floats = 2 MB**, making the 739.5s DiT a cache hit and leaving the 2502s decode freely iterable. **2 MB, not 540 MB.** | **[VERIFIED]** |
| **Optimising the host-side 540MB `logits` / 1.62GB grid / 165MB PBR arrays** | 64GB RAM, 95GB swap @ ~0 used. **Irrelevant.** (The 1.62GB grid array *does* become pure waste under a sparse decode — but as a side effect, not a target.) | **[VERIFIED]** |
| **Looking for a flux2-style memory-bound stage** | 169.1W flat at 99% util through the decode. **It is genuinely saturated.** | **[VERIFIED]** |

### 8.1 Open contradictions someone should resolve (10 min each, before re-deriving)

1. **The "tr-hoist gave a lossless ~5× decode (280s vs >1500s)" claim vs this run's 2502.3s.** Resolution: the 280s **is** the octree-256 number (2502.3 ÷ (513/257)³ = 315s ≈ 280s), and the hoist **is** in this build — `run.log` shows *"VAE transformer (16 layers over N=8192, once): 2.1s"*. **Both facts are true; the brief mixed configurations.** But confirm before sizing anything against it.
2. **Was `--fast` passed to the baseline?** Unknown. **This is the single biggest interpretive gap in this document** — it changes the meaning of every TFLOPS figure. One printf retires it.
3. **Does `attention()` tile the query dim, or materialize all 16 heads?** This sets the VRAM ceiling for `--us-chunk`. `m1_ggml.hpp:477-497` says it tiles (`PIXAL3D_ATTN_CAP_MB`, default 3072, "bit-identical to the single-shot path"), which would make even CHUNK=16384 safe — **but read it before picking a value.**
4. **Does `use_fp16` really require `GGML_PREC_DEFAULT` on this build?** (`ggml-cuda.cu:1720-1725`). If so, F16-without-`--fast` is a **net slowdown**. **One run settles it.**
5. **Is `NVIDIA_TF32_OVERRIDE=0` actually load-bearing?** The 4.2 TFLOPS ≈ 33% of FP32 peak says yes, but nobody confirmed cuBLAS honours `CUBLAS_TF32_TENSOR_OP_MATH` through ggml's `cublasGemmEx` + `CUBLAS_COMPUTE_32F` path, nor that `GGML_PREC_F32` doesn't independently veto it. **Note the corollary [PLAUSIBLE]: if `set_prec(GGML_PREC_F32)` only gates `use_fp16`, then for F32 weights it is a **no-op** and control reaches `cublasSgemm` on a TF32-enabled handle — meaning the geometry DiTs have been eating the tf32 noise the comment claims to prevent, all along.** Verify before acting either way.
6. **Is `orient_consistent` droppable on the coarse mesh?** 6.8s if yes; a **silent quality bug** if `us_vox` has a winding dependence. **30 min of reading. Do not act without it.**

### 8.2 What nobody measured

- **Zero GPU runs. Nothing here is a before/after.** A job was live; the rule was absolute.
- **No host/device split anywhere.** The 37.95 ms/chunk is wall-clock. **"3.85 TFLOPS achieved" is a lower bound on GPU efficiency** — if 30% of the wall is host-serial, the GPU is at 5.5 TFLOPS and the fp32-ceiling story is correspondingly weaker. **One nsys run over ~100 chunks resolves it, and it is the single most valuable missing measurement.**
- **The bake's ~79s residual is not split** between `shell_attr` (4.83M samples) and the 12.95M-texel loop. Attribution to the texel loop is query-count arithmetic (~10³ tri-tests vs 1 trilinear fetch), not a measurement.
- **The source tree drifted mid-analysis.** `tex_atlas.hpp` (08:50), `image_to_rig.cpp` (10:24), `tex_project.hpp` (10:39) and the binary (10:40) **all postdate the baseline run (03:51)**. The log's reproject line lacks the `fallback_r=` field current source prints. **Line numbers are current; timings are from an older build.**
- **The `geometry-unaccounted: 176.5s` bucket is not a real stage** — it's distributed across the window (gguf loads, host post, GLB writes). Its util/power row is meaningless. Everything downstream of `[1/4] geometry` aligns exactly to the logged boundaries and is trustworthy.
- **Page-cache state during the baseline is unknown.** 27.9GB at 78 MB/s = 358s cold > the 226s observed, which **proves** the run was partly warm but not how warm. **The NVMe win is bracketed, not pinned — and a genuinely cold run is worse than this "ground truth" baseline.**

---

## 9. SEQUENCING — a concrete ordered plan

**Phase 0 — settle the two questions that make everything else measurable (~1 hour, no code)**
1. **`--us-octree 256`, one run.** Confirms the anomaly formally, produces the missing `inline_soldier1536` provenance, and gives the honest baseline everything else should be measured against. **~2311s expected.**
2. **Add the env printf to the run header** (`PIXAL3D_FAST / _FLASH / _ATTN_CAP_MB / _GGUF_DIR`) + **make the gguf fallback loud**. Two lines. Retires a bug class that has now cost three agents an afternoon each.
3. **Read `attention()`** — settle the tiling question (gates step 5) — and **read `us_vox`** for a winding dependence (gates a 6.8s deletion).

**Phase 1 — the free wins, in this order (~an evening, no code)**

| # | do | expect |
|---|---|---|
| 4 | **`PIXAL3D_ATTN_CAP_MB=512`** | peak 10751 → ~8200; costs a few % of 739.5s. **Lossless. Do it first — it converts crashes into runs.** |
| 5 | **`--us-chunk 16384`** | **−440 to −565s.** *And it is the load-bearing test of the whole KV-recompute thesis — **if it doesn't land near −20%, stop and re-size levers 8/10 before investing in them.*** |
| 6 | **NVMe staging** (rsync + repoint) | ~−300s cold, deterministically, every run. **Still not done after two perf passes.** |
| 7 | **`ATL_FIT_RES=1`** | ~−20s |
| 8 | **Debug dumps opt-in**; **gate `mesh_topology_stats`** | ~−5 to −7s |

**Phase 2 — resolve the fork, then bank the lossless code wins (~a week)**

9. **§7.3: ask the owner which bug the dead tex chain is.** Either `--no-tex` (**−246.1s, bit-exact**) or fix `write_rigged_textured_glb` (**a shipping quality bug — the gold buttons cannot render**). **This is the biggest single decision in the doc after the octree.**
10. **Hoist `c_kv` + `ln_2` out of the decode loop** — **−590s at octree 512, lossless.** (Shrinks to ~75s at octree 256, ~25s after step 12 — **so do step 12 first, then re-measure.** If step 5 already landed the win, this is the permanent version.)
11. **Pack `qwen3_w` + `r1w_real` as BF16 gguf** — lossless, bounded by ~29s.
12. **Instrument the refine's invisible 81.8s** and the bake's ~79s residual. 4 `now_s()` pairs each. **"Every little bit matters" cannot be honoured on stages with no numbers.**

**Phase 3 — the big one (~2 weeks)**

13. **The hierarchical/sparse decode.** **−2000 to −2350s.** Structure: pyramid (129³ → 257³ → 513³, conservative by construction) **∪** a ±2 coarse-shell band (thin-feature safety net). **Never band-only — measured, it holes.** Gate on `boundary==0` + Chamfer vs a full dense decode at octree 256 (~315s — cheap). **Keeps 513³ — this is what makes the octree question moot rather than a compromise.**
14. **While you're there:** hoist the per-chunk allocations, LUT the fourier embed (513 distinct coords per axis), and drop the 1.62GB dense grid array (free under a sparse decode).

**Phase 4 — the precision A/Bs, one variable at a time (the June A/B violated this; don't repeat it)**

15. **`--fast` alone** on the refine (free, no repack). Gate on `[e2e] grid cos` / Chamfer.
16. **Decouple `PIXAL3D_FLASH` from `PIXAL3D_FAST`** (~5 lines) → **flash with F32 weights**. Only Q·K/A·V drop to f16 with fp32 accum — a far smaller delta than f16-everywhere, and **the highest-value unmeasured experiment in geometry**.
17. **Wire `fa_mask` into `ultrashape_dit.hpp:144,161`** — **the VRAM peak.** 10751 → ~7.7 GB. Unblocks `--moge` and any N > 8192. May need `PIXAL3D_FA_VSCALE` retuned (different absmax behaviour than pixal3d's DiT).
18. **Wire `fa_mask` into `us_cross_attn`** — **but only if step 13 didn't land.** After a sparse decode this is worth ~25s and may not be worth the numerical risk on an occupancy threshold.
19. **Pack `vae.gguf` F16** (the loader picks it up with zero code change) and **re-open `--model weights_gguf_f16 --fast`** against the clean baseline. **The June park condition has arguably fired.**
20. **`--tf32` as a flag** (never flip the default). Bounded above by the f16 result: if 0.196% chamfer passes, TF32 must.

**Phase 5 — the small stages, once the big numbers stop moving**
21. `orient_consistent` CSR (~4-5s) or its outright deletion (6.8s, pending step 3).
22. Per-vertex reproject (~−76s) — **supersedes `ATL_FIT_RES`, needs an eye-test.**
23. Batch the J=33 skin decodes; load `skin_vae.gguf` once; drop the duplicate DINOv3 harness and the duplicate embed.
24. `proj_linear` hoist (~11s, real surgery) — **last, if at all.**

**Owner-judgement items, parked outside the sequence (present the A/B, don't recommend):** `--us-octree`, `--us-steps` / the refine's missing CFG interval, the voxel-budget cap M, `--tex-dit proj` vs `cross`, `mc_stride`, `REMESH_CLOSE_R`, and every precision lever in Phase 4.

**Projected arc (arithmetic, not a promise):** 4498s → **~2300s** after Phase 1 at octree 512 (or ~2311s from step 1 alone, at 256) → **~1500-1700s** after Phase 2 → **~700-900s** after Phase 3 → **~500-700s** if the Phase 4 precision levers pass their eye-tests. **Every one of those numbers is a size, not a measurement.**
