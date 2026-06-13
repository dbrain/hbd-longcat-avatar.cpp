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

## 3. DE-PYTHON the texture bake (the infra blocker)
**Problem:** the lap-18 texture bake that produced the smooth result is currently a **Python cumesh
sidecar** (`bake_uv.py`). The port's whole point is native C++/ggml — this must go.
- **Good news:** the **native C++ xatlas** unwrap (bundled `thirdparty/xatlas`, `ComputeCharts`, run via
  `tex_reproject NO_PRECL=1`) gives the **SAME smooth quality** (verified by render this session). So the
  native bake is quality-complete: C++ xatlas unwrap + the C++ snap+volume reproject (`tex_atlas.hpp`
  `bake(..., reproject=true)`) + sRGB baseColor/metalRough output (`glb_textured.hpp`).
- **The only gap = SPEED:** C++ xatlas `ComputeCharts` is pathologically slow on these meshes (173 s /
  200k; >20 min / dense — non-manifold edges blow up its half-edge segmentation). cumesh is fast because
  it's GPU xatlas. Options, in order of preference:
  1. **Accept the C++ xatlas cost** — texture bake is offline asset-gen; 173 s once may be fine. Wire the
     C++ bake (`tex_reproject` logic) into `pixal3d --tex` directly, drop `bake_uv.py`. Simplest path to
     "native + done".
  2. **Speed up the C++ unwrap** — reduce xatlas segmentation cost (chart options), or make the mesh more
     manifold first (weld/repair) so `ComputeCharts` isn't pathological, or port a GPU xatlas.
  3. The precluster atlas is NOT an option (folds → streaks; see HANDOFF-A §7).
- Also kill the other Python crutches eventually: `estimate_camera.py` (MoGe camera → warm host service
  or C++ port), `preprocess_photo.py` (rembg → existing host service). These are front-end, lower priority.

## 4. Quick reference
- `PERF-NOTES-pixal3d.md` LAP 17 = the VRAM re-profile (per-stage MiB table).
- `FINDINGS-15-remesh-conditioning-perf.md` §4 = ncu/nsys-in-docker setup (`--cap-add SYS_ADMIN` for
  hardware counters) + the M3b DiT kernel breakdown + flash-attn landing notes.
- VRAM poll trick (one-off peak): background `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`
  at 0.3 s during the E2E, track max (GPU idle baseline ~247 MiB).
- Sub-4 next tier after the 1024² spike: DINOv3@512 (4185) + SS DiT (4007); then quant LAST (DiTs are
  F16; Q8 = first step down) — but the peak is ACTIVATION (im2col/attention), not DiT weights, so
  weight-quant won't move the peak.
