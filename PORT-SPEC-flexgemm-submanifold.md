# Port spec: FlexGEMM submanifold sparse conv → ggml / sd.cpp / CUDA

Reference study of `flex_gemm` (JeffreyXiang, the TRELLIS.2 / Pixal3D sparse-conv
backend) so we port the **fast/light algorithm**, not a naive gather-GEMM. Goal:
beat the Python impl on **both speed and VRAM** on the RTX 3060. Source read from
the installed package in the Pixal3D venv (`.../site-packages/flex_gemm`).

Intended as the GOLDEN EXAMPLE for bringing a sparse-voxel 3D model under
ggml/sd.cpp — the patterns here (rulebook cache, implicit GEMM, gather-fused
matmul) generalize to the rest of the TRELLIS.2 sparse stack.

## Scope FlexGEMM actually covers (important)
`conv_flex_gemm.py` asserts **submanifold only** (stride=1, no padding). Strided
down/up-sampling convs in the VAE fall back to spconv/torchsparse. So:
- **FlexGEMM hot path = submanifold conv** — the op we already validated bit-exact
  in `tools/sparse_spike/sparse_conv.cpp`. This is what we port for speed.
- Strided `SparseConv3d` / `SparseInverseConv3d` → match spconv semantics (also
  validated, the `sparse_k3_s2` case); lower frequency, optimize later.

## Data model
- `feats`  : `[N, Ci]`  (fp16 in prod), contiguous
- `coords` : `[N, 4]` int32 = (batch, W, H, D)   (note: spconv/FlexGEMM use W,H,D;
  our CPU ref uses z,y,x — same thing, just be consistent per op)
- `weight` : FlexGEMM native `[Co, Kw, Kh, Kd, Ci]` → reshaped to **`[Co, V, Ci]`**
  for the kernel (V = Kw·Kh·Kd ≤ 32). Our CPU-ref canonical is `[V, Ci, Co]`;
  gguf conversion bridges the two.
- `neighbor_map` (the RULEBOOK) : `[N, V]` uint32, entry = index of the input
  voxel at `coord[n] + offset[v]`, or `0xffffffff` if that voxel is inactive.

## The algorithm = two phases

### Phase 1 — Rulebook build (ONCE per active set, then CACHED)
`SubMConv3dFunction._compute_neighbor_cache`:
1. `init_hashmap(shape, HASHMAP_RATIO·N)` — open-addressing GPU hashmap, size
   `2.0·N` (`HASHMAP_RATIO`). Keys = packed coords.
2. `hashmap_build_submanifold_conv_neighbour_map_cuda` — insert all N coords;
   then for each voxel n × each kernel offset v, probe `coord[n]+offset[v]` →
   `neighbor_map[n,v]` = its index or `0xffffffff`. (CUDA, net-new for us.)
3. **Masked variants only** — post-process the rulebook:
   - `..._post_process_..._1` → `gray_code` (space-filling order of voxels),
     `sorted_idx` (voxel permutation by gray code → spatial locality),
     `valid_signal_{i,o,seg}`.
   - `..._post_process_..._2(block_size=B1)` → per B1-block of (gray-sorted)
     voxels: `valid_kernel` = the list of offsets v that have ≥1 valid neighbor
     in that block, `valid_kernel_seg` = CSR segment offsets. **Lets the GEMM skip
     empty kernel taps for a whole block.** (V≤32 because the per-block kernel
     mask is a uint32 bitmask.)

**This rulebook is keyed by (kernel_size, dilation) and cached on the SparseTensor
(`indice_key` / `spatial_cache`).** The sparse VAE runs MANY submanifold convs per
resolution on the SAME coords → build once, reuse for every conv at that stage.
Missing this is the #1 perf own-goal (see SKIMP LOG).

### Phase 2 — Implicit GEMM forward
Conv-as-GEMM:  `out[n, co] = bias[co] + Σ_{v,ci} feats[ neighbor_map[n,v], ci ] · weight[co, v, ci]`
i.e. a `[N,Co] = A[N, V·Ci] · Bᵀ` matmul whose A rows are the gathered neighbour
features. Tiling: `B1`×`B2` output tile (N×Co), `BK` over Ci.

- **IMPLICIT** (the VRAM win): A is never materialized. Inside the K-loop the
  kernel loads `neighbor_map[n,v]` and gathers `feats[neighbor, ci-block]`
  on the fly (absent neighbour → masked to 0), then `tl.dot` (tensor-core, tf32
  acc fp32) against the weight block. The only extra buffer is `neighbor_map`
  (`4·N·V` bytes) — vs explicit im2col's `N·V·Ci` (see VRAM math below).
- **MASKED**: the K-loop runs only over `valid_kernel[valid_kernel_seg[blk] :]`
  — the non-empty offsets for this voxel block. Voxels are read through
  `sorted_idx` (gray-code) so a block shares valid offsets → coherent masks +
  cache locality.
- **SPLITK**: split the K reduction across `program_id(1)=SPLITK` blocks, each
  writing a partial `[SPLITK,N,Co]`, then sum. Raises occupancy when the B1×B2
  tile count < #SMs. SPLITK auto-picked from `get_num_sm()` (3060 = 28 SMs →
  splitk matters: small grids would leave SMs idle).
- Precision: `allow_tf32=True`, fp32 accumulate, output cast to feats dtype.

Autotune tile configs (cuda): `B1∈{32..128} B2∈{32..256} BK∈{32,64}`,
num_warps 2–8, stages 3–5 (`kernels/triton/spconv/config.py`).

## VRAM math — why we MUST reach implicit (not stop at explicit im2col)
For N active voxels, V=27 (K=3), Ci=128, fp16:
- `neighbor_map` (implicit, the FlexGEMM cost): `4·N·V` B → N=100k ⇒ **10.8 MB**.
- explicit im2col buffer (`[N·V, Ci]` fp16): `2·N·V·Ci` B → N=100k ⇒ **691 MB**.
A naive explicit-gemm port is correct and gets tensor cores for free, but the
im2col buffer alone would blow the "lighter than Python" goal. The implicit
(gather-fused) kernel is mandatory to win on VRAM.

## Port ladder (each rung independently shippable / benchmarkable)
- **Rung 0 — CPU reference** ✅ DONE. `sparse_conv.cpp`, bit-exact vs goldens.
- **Rung 1 — GPU MVP**: CUDA rulebook build (hashmap + probe) → ggml custom op
  that does explicit im2col gather → existing `ggml_mul_mat` (tensor cores free).
  Correct + fast-ish, but heavy im2col VRAM. Validates the rulebook + GPU path.
- **Rung 2 — Implicit (match FlexGEMM VRAM)**: fuse the gather into the matmul =
  custom CUDA tile kernel (B1×B2, K-loop over v×Ci, gather via neighbor_map,
  wmma/MMA dot). Reuse the wmma/MMQ helpers from the longcat/flux DiT kernels.
  Kills the im2col buffer → VRAM parity with FlexGEMM.
- **Rung 3 — Beat FlexGEMM**: + masked (per-block valid_kernel skip) + gray-code
  `sorted_idx` reorder + splitk (tuned to 28 SMs) + **rulebook cache across all
  convs at a resolution**. This is where speed is won on the 3060.

## ggml / sd.cpp mapping & reuse
- SparseTensor = a small C++ struct: `{ggml_tensor* feats[N,Ci]; ggml_tensor*
  coords[N,4] int32; spatial_shape; rulebook cache map<key, neighbor_map>}`.
  Mirrors `pixal3d/modules/sparse/basic.py:SparseTensor`.
- Rulebook build = a net-new CUDA kernel (no ggml hashmap op). Self-contained;
  output is a plain `ggml_tensor` uint32 `[N,V]` → cacheable like any tensor.
- Implicit GEMM = new ggml-cuda op `ggml_sparse_submanifold_conv`. Borrow the
  tile/wmma structure from the repo's existing tensor-core matmul (sd.cpp DiT
  path) — the novelty is only the A-operand gather + the masked K-loop.
- Graph integration: output `[N,Co]` shape is known (== N input voxels for
  submanifold), so gallocr static alloc is fine for the submanifold op (unlike
  strided, where output N changes — that one needs dynamic alloc or a 2-pass
  count-then-fill, deferred with the strided path).

## Correctness anchors (GPU-free)
- `explicit_gemm` branch in `submanifold_conv3d.py` (`torch.addmm` of im2col vs
  `weight.view(Co,V·Ci)`) is the exact, readable spec — matches our CPU ref.
- `_compute_neighbor_cache_torch` builds the rulebook with pure-torch
  `searchsorted` (no CUDA) for the explicit path. **We can cross-check our ref
  against FlexGEMM's own math on CPU** via `CUDA_VISIBLE_DEVICES=""` +
  ALGORITHM=explicit_gemm, if we want extra confidence before any GPU time.
  (Caveat: importing flex_gemm may create a CUDA context — gate with empty
  CUDA_VISIBLE_DEVICES so it stays off the busy GPU.)

## When GPU is needed (batched, single session)
1. `golden_hook` real-layer dump from a Pixal3D decode (`SPARSE_CONV_BACKEND=spconv`)
   — confirms semantics + resolves the weight permutation on real layers.
2. Bench `SPARSE_CONV_BACKEND=flex_gemm` per-layer time + peak VRAM = the bar.
3. Build + correctness + perf of the CUDA rulebook & implicit-GEMM kernels.
Everything else (CPU ref, op design, rung-1/2 host code) is GPU-free.
