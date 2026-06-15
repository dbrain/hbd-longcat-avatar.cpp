# MINI-PROMPT — finish finger-preserving retopo (pixal3d), full-automation

Pick up the pixal3d retopo lap. **Read `HANDOFF-C-retopo-quadriflow.md` top "⭐ RESUME HERE" block first —
that's your spec.** Goal: image → pixal3d (SOTA gen, KEEP) → riggable quad retopo that PRESERVES detail
(esp. **fingers** — they're in the source) → bake baseColor+metalRough+normal → compressed GLB. Work
autonomously from the main loop; get an owner eyeball only at the final textured render.

## The story so far (so you don't repeat it)
Retopo via `marching_cubes_solid` (coarse) FUSED the fingers + looked "PS2" (render-verified). The fix:
feed QuadriFlow the **DETAILED** manifold mesh, not a coarse remesh. PolyGen (learned retopo) is NOT
open-weights — not an option. The detailed manifold mesh = the 8M marching-tet `miku_remesh_smooth.ply`;
`ply_decimate_obj … sloppy` decimates it (quality LockBorder stalls on the voxel lattice; sloppy works)
→ `/tmp/det.obj` (385k tris, keeps fingers) is ALREADY ON DISK, QF-ready.

## STEP 1 (do this first — it unblocks everything)
Machine is clear now. Run:
```
cd tools/m1_ref/cpp_port
QF=$(./build_quadriflow.sh -p)
"$QF" -i /tmp/det.obj -o /tmp/det_quad.obj -f 60000        # ~10-30s on a clear machine
/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python render_geo_detail.py /tmp/det_quad.obj /tmp/dq
```
Then **Read /tmp/dq_detail.png (hand/feet crops) and /tmp/dq_body.png** — did the fingers survive?
- **Completes + fingers present** → success path: `RETOPO_INSP=1 ./retopo_bake /tmp/det_quad.obj native_retopo2048.glb 2048`
  → `python _mv_render.py native_retopo2048.glb.insp.glb /tmp/tex.png` → Read it → scrutinize.
- **QF stalls even on a clear machine** → sloppy output is too non-manifold for QF. Then: (a) clean before
  QF (weld dup verts + drop zero-area tris + keep largest component), or (b) **vendor Instant Meshes**
  headless (more robust to messy AI meshes — likely the real fix), or (c) a manifold-repair lib.

## Then (priority order)
2. Sharpest fingers: if smoothed-8M softened them, sloppy-decimate the **dense** dual-grid instead
   (`dump_dense_*.bin`, 1.5M, crispest fingers) → QF.
3. Denoise the normal bake: `normal_bake.hpp` currently samples raw `dump_dense` → rainbow speckle.
   Sample the SMOOTHED mesh / smooth dense vertex normals → clean tangent-space normal map.
4. Full asset: retopo + baseColor + metalRough + clean normal → packed GLB → RENDER + scrutinize.
5. Later: LOD tiers (lower -f), AO bake, wire `--retopo` into the live pixal3d chain (GPU), SkinTokens.

## HARD RULES (the lesson from last lap)
- **RENDER AND LOOK before claiming anything works.** chart-count / validator-pass ≠ visual quality. The
  previous agent declared victory TWICE on un-rendered results (a blob, then fused fingers). Always render
  geometry (`render_mesh.py`, `render_geo_detail.py` for hand/feet crops) AND textured (`_mv_render.py`
  on the `RETOPO_INSP=1` uncompressed sidecar — trimesh can't read KTX2/meshopt) and Read the PNGs.
- Don't use `marching_cubes_solid` as the retopo input (fuses fingers).
- reproject(dense) is REQUIRED in `retopo_bake` (off-shell specks otherwise) — it's already wired.

## Environment / norms
- Drive from the MAIN LOOP (sub-agents stall on background jobs). C++/CUDA builds fine here (toolchain
  g++ at /mnt/hdd/3d/avatar-shootout/toolchain; no-build rule is Rust-only). `build_*.sh` patterns exist.
- **Machine is SHARED** — `cat /proc/loadavg` before heavy runs; if load is high (other agent), pause.
  Launch long jobs `run_in_background:true`, never detached `&`. Clean up stray procs before handoff.
- No multi-agent Workflow / deep-research for ordinary research. Plain git OK; commit when it works.
- Validation harness: `gltf_validator -o file.glb` (0 errors), `./meshopt_verify file.glb` (streams
  decode), `compare.html` :8011 (model-viewer). 4 model-viewer gotchas already fixed (see HANDOFF-B).

## Key files (tools/m1_ref/cpp_port/)
build_quadriflow.sh · coarse_obj.cpp · ply_decimate_obj.cpp (has `sloppy`) · retopo_probe.cpp ·
retopo_bake.cpp · normal_bake.hpp · glb_packed.hpp (normal-texture slot) · render_mesh.py ·
render_geo_detail.py · _mv_render.py. Artifacts: /tmp/det.obj, miku_remesh_smooth.ply, dump_dense_*.bin,
dump_pbr_*.bin, refs/stage5/head_coords.npy. Branch spike/sparse-conv-3d. Last commits: 61b8e94 c553935.
