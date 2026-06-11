# Spike #1 — 3D sparse convolution as a ggml op (go/no-go for the C++ 3D-asset port)

**Goal:** prove ggml can do efficient coordinate-based **sparse 3D convolution**
bit-exact vs the Pixal3D / TRELLIS.2 sparse VAE. This op is the single
existential risk for porting the image→rigged-3D pipeline to C++ — everything
else (DiT, LLM rigger, mesh-extraction) is known-portable. If this works, the
port is real; if not, pivot to a triplane/LRM architecture that avoids sparse
conv entirely.

**Worktree:** `/home/dbrain/dev/longcat-sparse-spike` — branch
`spike/sparse-conv-3d`, off `longcat-avatar.cpp@5e26fc5` (the stable-diffusion.cpp
visual fork: has ggml submodule, `src/core/ggml_extend.hpp` custom-op helpers,
`tools/`, gguf I/O). ⚠️ ggml submodule not yet checked out in this worktree —
`git submodule update --init ggml` before building (objects are local, offline OK).

**Model source:** `/mnt/hdd/3d/avatar-shootout/Pixal3D` (validated 7.9GB/3060).
Sparse code: `pixal3d/modules/sparse/conv/`. Default backend `flex_gemm`
(Triton), **swappable to `spconv`/`torchsparse` via `SPARSE_CONV_BACKEND`** — all
three are interchangeable impls of the same math, so we target documented
submanifold semantics, NOT the Triton kernel.

## Op semantics (from conv_spconv.py)
Cross-correlation, `out[p] = bias + Σ_k W[k]·in[p + (k-center)·dilation]`:
- **SubMConv3d** — stride 1, no pad. Output active set == input active set
  (submanifold). Only ACTIVE neighbours contribute (inactive = skipped, not
  zero-padded).
- **SparseConv3d** — strided/padded. New down-sampled active set; output site
  active if any kernel tap hits an active input. (spconv re-sorts strided output
  by batch; inverse conv un-sorts via cached map — only relevant for batch>1.)

## Canonical tensor layout (both numpy ref and C++ op use this — ONE convention)
- `coords` int32 `[N,4]` = (batch, z, y, x)
- `feats`  f32   `[N,Cin]`
- `weight` f32   `[K*K*K, Cin, Cout]`, kernel flattened z-major (kz,ky,kx)
- `bias`   f32   `[Cout]`

## Two golden sources (decouples C++ dev from the GPU/model) — `tools/sparse_spike/`
1. **`sparse_conv_ref.py`** — pure-numpy reference + synthetic generator.
   NO GPU / NO spconv / NO model. Emits `golden_ref/<case>/*.npy` + manifest.
   Cases: `subm_k3_small`, `subm_k3_dense`, `subm_k1`, `sparse_k3_s2`.
   Self-consistency verified (maxerr 0.0). **This is the C++ dev driver — works today.**
2. **`golden_hook.py`** — drop-in monkeypatch (GPU, run-once). Add to any
   Pixal3D decode: `import golden_hook; golden_hook.install('golden_model')`
   (with `SPARSE_CONV_BACKEND=spconv`). Dumps real-layer I/O in the SAME canonical
   layout; AUTO-CALIBRATES the spconv→canonical weight permutation on the first
   subm layer (brute-forces axis perms vs the numpy ref). Confirms our semantics
   == production. **GPU-gated — the only blocker.**

## GPU session 2026-06-11 — real-layer goldens + baseline CAPTURED (GPU now free)
One full Pixal3D decode (miku.png, res 1024, low_vram) with `golden_dump_runner.py`:
- **Only `flex_gemm` is installed** → every Pixal3D sparse conv is **submanifold,
  K=3³** (no strided sparse conv; downsampling is space-to-channel). Port = subm only.
- **9 distinct conv shapes** captured (the complete set), N from **1126** (deep,
  Ci1024→Co4096) to **382,533** (shallow, Ci64→Co64). ~40% sentinel (denser than
  synthetic). Goldens in `tools/sparse_spike/golden_model/` (gitignored, 2.1 GB).
- **Semantics CONFIRMED on real data**: our recomputed f64 ref vs flex_gemm's fp16
  output = **rel 3e-4 – 7e-4** (fp16/tf32 noise) across all 9. Weight layout
  `[Co,Kw,Kh,Kd,Ci]`→canonical verified. `finalize_model_goldens.py` (CPU).
- **flex_gemm BASELINE banked** (`golden_model/flexgemm_timing.json`) — the bar to
  beat. Warm (`min_ms`) heaviest: Ci64Co64 N=382k = **166ms**, Ci128Co512 N=89k =
  **139ms**; channel-heavy small-N ~3–40ms. (Row avg_ms inflated by 1st-call Triton
  autotune; use min_ms.) Also copied the 3060 autotune-cache (tile configs) for ref.
- **Everything for kernel dev is now CPU-only** (author + compile + validate vs these
  goldens). GPU needed again only to RUN/bench the CUDA kernel. `sparse_conv_test
  golden_model` validates the C++ op against the real layers.

## Status / next steps
- [x] Worktree + op-semantics analysis + canonical layout
- [x] numpy reference + synthetic goldens (GPU-free, self-consistent)
- [x] model golden hook (ready to fire at GPU time)
- [x] **C++ CPU op + test harness** — `sparse_conv.cpp` (+ `npy.hpp` reader),
      mirrors the numpy ref exactly. `g++ -O2 -std=c++17 sparse_conv.cpp -o
      sparse_conv_test && ./sparse_conv_test golden_ref` → **ALL PASS, maxerr
      0.000e+00, strided coords match (478 voxels)**. The math ports to C++. ✅
- [x] **FlexGEMM algorithm study** — full read of the installed `flex_gemm`
      source → **`PORT-SPEC-flexgemm-submanifold.md`** (the golden-example doc:
      rulebook build, implicit-GEMM, masked+splitk, VRAM math, ggml mapping,
      the rung ladder). Key finds: FlexGEMM is submanifold-ONLY (our validated
      hot path); cached rulebook + implicit (gather-fused) GEMM are the
      speed/VRAM levers; explicit im2col would blow VRAM (691MB vs 10.8MB @
      N=100k) so implicit is mandatory.
- [x] **Real-layer validation (CPU)** — C++ op vs all 9 real Pixal3D layers =
      **maxerr 0.000e+00** (rulebook matches captured neighbor_maps too). Up to
      N=382,533. Correctness 100% closed on production data.
- [x] **Rung-1 CUDA kernel DRAFTED** (GPU-free authoring) — `sparse_subm_conv.cu`:
      implicit-GEMM (gather-via-rulebook, fp32, no im2col), simple/correct, no
      tensor cores yet. CPU mirror `sparse_subm_conv_cpu.hpp` + `test_subm.cpp`
      validate the EXACT kernel math vs goldens → **maxrel ~1e-7** (predicts the
      CUDA kernel's accuracy before running; >> better than flex_gemm fp16 ~1e-3).
      ggml submodule checked out in worktree (build-ready). Architecture: sparse
      is a self-contained subsystem (own CUDA kernels + host rulebook cache),
      ggml core untouched — mirrors Pixal3D's modules/sparse split.
- [x] **Rung-1 kernel COMPILED (sm_86) — ready to run.** Found host nvcc 12.4 in
      the Pixal3D micromamba toolchain (`/mnt/hdd/3d/avatar-shootout/toolchain`,
      `-ccbin .../g++`). `test_subm.cpp`+`sparse_subm_conv.cu` compile clean →
      `test_subm_cuda` (standalone sm_86 binary). So the GPU session is ONE command:
      **`tools/sparse_spike/run_bench.sh`** → correctness vs real goldens (expect
      ~1e-7) + per-layer `ours vs flexgemm` timing. (Docker build is the canonical
      path for when the subsystem is integrated into longcat-avatar.cpp proper;
      the standalone toolchain binary is fine for the spike bench.)
- [x] **Rung 1 RUN on GPU — correct + precise + robust.** `sparse_subm_conv.cu`
      implicit fp32 kernel: **all 9 real layers maxrel ~2-6e-7**, synthetic edge
      cases (K=1, K=3, N 120→382533, Ci/Co 8→4096) all PASS. **This is the
      working/correct milestone — KEEP it as the precise oracle.**
- [x] **Perf intel banked (then PARKED)** — Rung-1.5 cuBLAS tf32 diagnostic
      (`bench_cublas.cu`): tensor cores close the compute gap (matched flex_gemm on
      the big GEMM) but explicit im2col = up to 2.64 GB/layer (over the 7.5 GB
      budget). All numbers + the Rung-2 plan (implicit MMA, masked, splitk, rulebook
      cache, fp16) in **`PERF-NOTES-sparse-conv.md`**. NOT the current direction.
- [ ] **DEFERRED to a dedicated perf run** (per [[feedback_correctness_before_perf]]):
      Rung-2 implicit tensor-core kernel etc. Precision/correctness comes first;
      perf (which loosens maxrel to ~1e-3) is its own session validated by E2E.
- [ ] **Next functional milestones** (correctness-first, not perf): GPU rulebook
      hashmap kernel; integrate the sparse subsystem into longcat-avatar.cpp (docker
      build); then more of the TRELLIS.2 pipeline toward E2E.
- [ ] **Rung 2 (GPU)** — implicit gather-fused CUDA kernel (reuse DiT wmma) →
      VRAM parity with FlexGEMM.
- [ ] **Rung 3 (GPU)** — masked + gray-code + splitk + rulebook cache → beat it.
- [ ] **[GPU]** golden_hook real-layer dump + flex_gemm perf/VRAM baseline.
- [ ] go/no-go decision.

Correctness risk is RETIRED (Rung 0 bit-exact + algorithm fully understood). The
remaining work is the CUDA kernel ladder (Rungs 1–3) — GPU-gated for build/perf,
but op design + host plumbing can be drafted GPU-free. See PORT-SPEC for detail.

## Reference split — correctness vs performance (IMPORTANT)
Overarching goal is ALWAYS **overall faster AND lighter on VRAM than the Python
impl** — both, not just speed.
- **Correctness reference = spconv semantics (any backend).** spconv /
  torchsparse / flex_gemm produce numerically identical output; we validate
  bit-exact against the documented submanifold math. spconv is just the most
  convenient golden source (documented + CPU-checkable layout).
- **Algorithm to PORT = FlexGEMM's, not spconv's.** FlexGEMM is fastest/lightest
  because of HOW it computes: `masked_implicit_gemm_splitk` — implicit GEMM over
  only occupied voxels (no explicit im2col gather → less memory traffic + less
  VRAM), hashmap rulebook (ratio 2.0). The CUDA kernel should reimplement THAT
  algorithm in ggml/CUDA, not spconv's explicit gather-GEMM. Study source:
  `github.com/JeffreyXiang/FlexGEMM` (Triton kernels) — a GPU-free reading task
  we can do before the kernel work.
- **Perf baseline to BEAT = flex_gemm on the 3060.** Bench against
  `SPARSE_CONV_BACKEND=flex_gemm`, capture per-layer time AND peak VRAM as the
  bar; our port must beat both.

## SKIMP LOG — deferred to the performance phase (don't lose these)
Things intentionally cut for the correctness spike, to revisit when chasing
"faster than Python":
1. **CPU reference op is naive** — per-call `unordered_map` rebuild, f64
   accumulate, O(N·K³) scalar gather. Correctness only; NOT a perf path.
2. **Rulebook / indice_key reuse** — spconv & FlexGEMM compute the neighbour map
   (rulebook) ONCE per active set and reuse it across every conv at that
   resolution (that's what `indice_key` does in conv_spconv.py). We rebuild the
   hash every call. **Caching the rulebook across layers is a top perf lever** —
   the sparse VAE runs many convs on the same coords.
3. **Tensor-core GEMM** — per-voxel matmuls should map to wmma/MMQ like the
   existing DiT kernels; FlexGEMM uses `masked_implicit_gemm_splitk` +
   hashmap-ratio 2.0 (`conv/config.py`). Study/match these in the CUDA phase.
4. **fp16/bf16 path** — production sparse VAE likely runs fp16; we validate at
   f32. Low-precision accumulate + tolerance is a perf-phase question
   (cf. spconv's bf16 KeyError → cast-to-f16 workaround noted in research).
5. **Gather/scatter coalescing & occupancy** — the FlexGEMM win is memory-layout
   (only occupied voxels, coalesced). Naive scatter will be bandwidth-bound.
6. **batch>1 strided re-sort** — conv_spconv re-sorts strided output by batch and
   the inverse conv un-sorts via a cached map. Single-image decode is batch=1, so
   skipped now; needed if we ever batch.
7. **dilation / asymmetric padding / non-cubic kernels** — only K∈{1,3}, sym pad,
   cubic tested. Enumerate real layers from the model dump before locking the op.
8. **ggml-graph integration** — the spike validates a standalone C++ kernel.
   Wrapping as a real `ggml_sparse_conv_3d` graph op (dynamic output shape vs
   gallocr static alloc) is deferred; it's engineering, not a go/no-go risk.
