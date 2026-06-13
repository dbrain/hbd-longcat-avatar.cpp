# HANDOFF B — Pixal3D perf/infra: VRAM spike, CPU poles, de-Python the bake

Continuation of the C++/ggml **Pixal3D** port (image → textured 3D GLB). This handoff is the
performance + infrastructure track (the texture *quality/size* track is `HANDOFF-A-model-smoothness.md`).
Work dir `tools/m1_ref/cpp_port/` on branch `spike/sparse-conv-3d`. Build `./build.sh <name> cuda`;
weights `/mnt/hdd/pixal3d/weights_gguf_f16`; always `NVIDIA_TF32_OVERRIDE=0`; `--fast` + `PIXAL3D_FLASH=1`.
Long GPU runs: harness-tracked background from the MAIN loop, NOT detached `&` (un-tracks). Sweep
`pgrep -x pixal3d` before handoff. No `Co-Authored-By`/AI mentions in commits.

Run recipe (see HANDOFF-A §2 for full detail):
`NVIDIA_TF32_OVERRIDE=0 PIXAL3D_FLASH=1 ./pixal3d --model /mnt/hdd/pixal3d/weights_gguf_f16 --image <png>
--out x.glb --remesh --tex --fast`. Miku img `../../sparse_spike/golden_stages/pre/preprocessed.png`;
turtle img `prep_test_matte.png` (heavy, ~520 s).

---
## 1. VRAM — target: absolute max 7.5 GB, ideally sub-4 GB
**Current (measured this session, RTX 3060):**
- **Miku peak = 6021 MiB** at the stage-5 **1024² window** (under 7.5 ✓; the sub-4 stretch needs this
  spike flattened). Everything else ≤ 4.2 GB (DINOv3@512 4185, SS DiT 4007 are the next tier).
- **Turtle peak = 6649 MiB** (under 7.5 ✓ — done; no work needed unless you also want it lower).

**The Miku 6021 spike = the 1024² stage.** It's one of two ops and the prior poll label lagged so it's
unconfirmed which: **DINOv3@1024 attention** vs **NAF@1024 im2col** (`naf_graph.hpp conv()`; already F16
under `--fast`). TASK:
1. **Instrument** which op — add per-phase `cudaMemGetInfo(&free,&total)` logging around the 1024²
   ops in the chain (`pixal3d_chain.hpp` stage-5 / `dinov3_graph.hpp` / `naf_graph.hpp`). pixal3d has no
   per-phase VRAM log yet.
2. If **NAF@1024 im2col**: **tile the conv spatially** — split the 1024² plane into overlapping tiles
   with a 1px reflect-pad halo (k3), im2col+matmul per tile, concat. Bounds the im2col scratch tensor.
3. If **DINOv3@1024 attention**: wire flash-attn there (flash already landed for the sparse DiTs via
   `PIXAL3D_FLASH`; extend to the DINOv3@1024 path).
4. **Validate after any change: N1 == 1120 (Miku) + mesh-IoU + render** (this path is correctness-
   sensitive — the SS occupancy is N1-exact-dense; don't perturb it). Each validation = one Miku E2E
   (~184 s). Miku's aligned dump is backed up (`miku_*.bin`) if you only need to re-bake textures.

## 2. Host-CPU poles (owner: "atlas still seems CPU bound", visible on the turtle)
Per-stage wall is printed by the chain as `(%.1fs)`. Remaining HOST-CPU poles:
- **M4 mesh-extract** (`sparse_vae.hpp` / `build_nmap` hashmap-per-level + host coord arithmetic):
  **~19.8 s Miku, ~63 s turtle** — the big one. Move the neighbour-map / coord growth to GPU (Morton /
  GPU hash). Flagged since lap-15, still open.
- **Texture bake raster + grid_sample**: in the C++ path (`tex_atlas.hpp` CPU triangle rasterizer +
  `tex_grid_sample.hpp` volume sample) these are CPU. nvdiffrast-style GPU raster + GPU grid_sample.
  (NOTE: the current *working* bake is a Python sidecar — see §3; whichever bake survives, its raster +
  sample + the snap-to-dense reproject should be GPU.)
- **xatlas unwrap** is CPU and SLOW (173 s on 200k, >20 min on dense) — see §3; this is the biggest infra
  question.

## 3. DE-PYTHON the texture bake (the REAL hard blocker — a crack-free C++ unwrap)
**Problem:** the lap-18 texture bake that produced the smooth, crack-free result is a **Python cumesh
sidecar** (`bake_uv.py`). The port's whole point is native C++/ggml — this must go. **But this is the
genuinely-hard blocker, not a wiring detail.**

**The crux = the UV unwrap, judged IN MODEL-VIEWER (not pyrender — pyrender hides seam cracks):**
- **cumesh** (GPU xatlas, Python) produces FEW CLEAN charts → no visible UV seams → **crack-free + crisp**.
  This is the only thing that currently looks right. It's the REFERENCE quality bar.
- **C++ xatlas** (bundled `thirdparty/xatlas`, `ComputeCharts`, `tex_reproject NO_PRECL=1`): fragments the
  NON-MANIFOLD mesh into ~37k tiny charts → tons of UV seams → **visible CRACKS all over in model-viewer**
  (looks like pyref — same defect). Also slow (173 s/200k, >20 min/dense) and the fragmented charts pack
  into a bloated atlas (4276² → big texture). **NOT acceptable as-is.** (An earlier note in this repo that
  "C++ xatlas matches cumesh quality" was WRONG — it was judged in pyrender, which hid the seam cracks.)
- **C++ precluster atlas** (`tex_atlas.hpp` precluster path): folds where the surface curves → teal +
  sliver STREAKS. Also not acceptable. (HANDOFF-A §7.)

**Reverse-engineered why cumesh wins (this session):** its `uv_unwrap` does its own GPU **"fast
clustering"** (~62k clusters) → feeds them to xatlas which parameterizes + packs → 6 clean charts,
crack-free, ~5 s. Same xatlas you have in C++; the gap is the clustering + a non-planar (conformal)
per-cluster UV. **Full detail + the prioritized leads are in `HANDOFF-A-model-smoothness.md` §1 + §5** —
that handoff owns the crack-free-unwrap task; this section is just the infra framing.

So a native C++ texture needs a **crack-free conformal unwrap** — the open research problem:
  1. **Manifold-repair the mesh, THEN xatlas** — the cracks come from non-manifold fragmentation. If the
     mesh is welded/repaired to (near-)manifold first, xatlas should produce few clean charts like cumesh.
     cumesh internally has `repair_non_manifold_edges` / `unify_face_orientations` / `remove_degenerate_faces`
     — replicate that in C++ (bundled meshopt has some; may need own weld). FINDINGS-15 tried a naive
     `make_manifold` (→ +253k boundary, fragmented worse) — needs a smarter repair. **This is the most
     promising native path.**
  2. **Seam-aware texture dilation** — aggressively pad/dilate each chart's gutter in the atlas so
     model-viewer's bilinear/mipmap can't sample across seams. May hide cracks even with many charts;
     cheap to try on the existing C++ xatlas bake (bump padding + inpaint iters in `tex_atlas.hpp`).
  3. **Port a better/GPU unwrapper to C++** (heaviest).
- The C++ bake MATH downstream of the unwrap is done + fine (`tex_atlas.hpp bake(reproject=true)` snap+
  volume + `glb_textured.hpp` sRGB output) — ONLY the unwrap is blocking.
- Lower-priority Python crutches to kill later: `estimate_camera.py` (MoGe cam), `preprocess_photo.py`
  (rembg → existing host service).

## 4. Quick reference
- `PERF-NOTES-pixal3d.md` LAP 17 = the VRAM re-profile (per-stage MiB table).
- `FINDINGS-15-remesh-conditioning-perf.md` §4 = ncu/nsys-in-docker setup (`--cap-add SYS_ADMIN` for
  hardware counters) + the M3b DiT kernel breakdown + flash-attn landing notes.
- VRAM poll trick (one-off peak): background `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
  at 0.3 s during the E2E, track max (GPU idle baseline ~247 MiB).
- Sub-4 next tier after the 1024² spike: DINOv3@512 (4185) + SS DiT (4007); then quant LAST (DiTs are
  F16; Q8 = first step down) — but the peak is ACTIVATION (im2col/attention), not DiT weights, so
  weight-quant won't move the peak.
