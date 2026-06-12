# HANDOFF — Pixal3D lap-19 (after lap-18 TEXTURE QUALITY SOLVED)

Run FULLY AUTONOMOUS, owner intermittently around + sharp feedback (judge by RENDER vs the SOURCE
IMAGE, NOT pyref — pyref is geometry+rough-colour ref only and HAS teal-interior holes = a defect).
GPU+CPU free; C++ builds fine here; long GPU runs `run_in_background:true` from the MAIN loop and do
NOT detach with `&` inside the bg command (it un-tracks the job — bit me in lap-18). Sweep
`pgrep -x pixal3d` before handoff; no `pkill -f`/rm-globs. **Owner is watching token spend — be decisive,
don't re-explore solved ground.**

## ✅ LAP-18 DONE — texture quality (owner's #1), both assets
The lap-17 per-vertex COLOR_0 was washed (COLOR_0 is LINEAR → skips the sRGB decode → too bright) +
cracky (per-vertex on 200k facets) + dark twin-tail patches. **SOLVED** with a real UV PBR texture from
a CONFORMAL unwrap:

- **`tools/m1_ref/cpp_port/bake_uv.py`** (the production bake): **cumesh** (GPU xatlas) unwraps the crisp
  200k QEM mesh in ~5s, then nvdiffrast raster + flex_gemm grid_sample the PBR volume + cv2 inpaint →
  smooth sRGB baseColor + metallicRoughness. Run:
  `bake_uv.py --dump <dir> --mesh qem --texsize 2048 --out <glb>` on a `PIXAL3D_DUMP_BAKE=1` dump.
- **Why cumesh, not C++:** C++ xatlas ComputeCharts is pathologically slow (>20min on the 3.1M dense
  mesh; 173s + 37k fragmented charts on the 200k QEM → seams). My C++ precluster planar atlas FOLDS
  where the surface curves → teal-interior + sliver STREAKS (that was the whole streak problem, NOT mesh
  budget or sampling). cumesh's proper non-folding unwrap fixes both, fast.
- **Result:** non-folding unwrap ⇒ NO teal-interior (beats pyref); texture carries the detail ⇒ the crisp
  200k mesh stays SMOOTH (200k+cumesh ≈ the dense-161MB ceiling). Validated vs SOURCE on BOTH assets:
  - Miku `miku_qem_uv.glb` (~20MB) — compare.html `:8011` tab A (B=dense ceiling, C=pyref).
  - Turtle `turtle_qem_uv.glb` (~20MB) — compare_turtle.html `:8012`.
- **Render-judging gotcha:** pyrender ambient-0.18 is MISLEADINGLY dark+harsh and over-reads artifacts;
  the owner judges in model-viewer "neutral" IBL (bright/even). Use `_mv_render.py` (ambient 0.55 +
  6-light) to judge colour/smoothness. Colours are accepted ("as good as it gets, better than gloomy").
- **Dead ends (don't repeat):** C++ precluster atlas (folds), C++ xatlas on QEM (slow+fragments),
  per-vertex colour for web (needs ~1.5M verts to not crack; 400-800k still "nets"), tighter cones (no
  fix). All in this session's transcript.

Committed: `c62f28a` (reproject scaffold + dense-mesh dump) + `bake_uv.py` commit. The C++
`tex_reproject.{hpp,cpp}` + `tex_atlas` snap+volume reproject are committed but SUPERSEDED by cumesh.

## REMAINING (pick up here)

1. **Production invocation integration (small).** Today it's a 2-step: `pixal3d ... --remesh --tex --fast
   PIXAL3D_FORCE_UVATLAS=1 PIXAL3D_DUMP_BAKE=1` (C++, writes the dump) → `bake_uv.py --mesh qem` (Python
   sidecar, writes the textured glb). Either fold the `bake_uv.py` call into `pixal3d` (shell out after
   the dump) or document the 2-step as the texture path. (pyref is the same shape — a GPU-Python bake.)

2. **Miku sub-4GB VRAM (owner #2; task still open).** Turtle peak MEASURED = **6649 MiB (<10GB ✓, done)**.
   Miku peak still **6021 MiB** at the stage-5 1024² window (PERF-NOTES LAP 17). FIRST instrument which op
   (DINOv3@1024 attention vs NAF@1024 im2col) with per-phase `cudaMemGetInfo` (the poll label lagged),
   then tile the @1024 conv spatially (overlap +1px reflect halo, im2col+matmul per tile, concat) OR wire
   flash to DINOv3@1024. Validate N1==1120 + mesh IoU + render. Needs Miku E2E reruns (~3min each); the
   aligned Miku dump is backed up as `miku_*.bin` in the cpp_port dir (restore over `dump_*.bin` to
   re-bake Miku textures without a rerun).

3. **Optional: smaller texture.** 2K atlas → ~20MB glb; 1K (`--texsize 1024`) ≈ ~6MB if quality holds —
   re-bake + eyeball in model-viewer.

4. **CPU poles → GPU (owner: "atlas still CPU bound", task 6, lower priority).** With cumesh, the unwrap
   + raster + grid_sample are already GPU. Remaining host pole = M4 mesh-extract (`build_nmap`
   hashmap-per-level, ~19.8s Miku / 63s turtle). GPU/Morton neighbour-map.

## Build / run
`cd tools/m1_ref/cpp_port && ./build.sh <t> cuda`. Weights `/mnt/hdd/pixal3d/weights_gguf_f16`. Miku
input `tools/sparse_spike/golden_stages/pre/preprocessed.png`; turtle `prep_test_matte.png`. `--fast` +
`PIXAL3D_FLASH=1`; `NVIDIA_TF32_OVERRIDE=0`. Python venv `/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv`.
Branch `spike/sparse-conv-3d`. cpp_port/.gitignore excludes glb/ply/png/bin; compare*.html are gitignored.
