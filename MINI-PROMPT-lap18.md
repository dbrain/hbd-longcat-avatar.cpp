Continue the C++/ggml Pixal3D (TRELLIS.2) port — FULL AUTONOMY, owner intermittently around + gives
sharp feedback (judge by RENDER vs pyref, never false-"perfect"). GPU+CPU free.

## THE JOB: make the TEXTURE QUALITY perfect (owner's #1). Shape is already DONE (crisp QEM remesh).
End of lap-17 the textured Miku (`pixal3d --remesh --tex --fast`, per-vertex COLOR_0,
`miku_qem_vcolor.glb` on compare.html :8011 tex tab A) is "a lot better but colours WASHED OUT and
still CRACKY like smashed porcelain glued back together." Both are inherent to per-vertex-colour-on-a-
faceted-mesh (averaging desaturates; per-vertex colour is discontinuous at facet edges → cracks; thin
tails get dark back-of-hair patches). **THE FIX = BVH-reproject the QEM mesh's texels onto the DENSE
dual-grid mesh (which has exact, smooth, full-saturation per-vertex PBR) → bake a real UV PBR TEXTURE**
(use the fast precluster atlas for layout; closest-POINT-on-triangle not nearest-vertex; front-face
reject for the thin tails; verify sRGB colour space for the wash). Concurrent: sub-4GB VRAM (peak 6021
MiB at the 1024² conv im2col — tile it / flash DINOv3@1024).

READ IN FULL FIRST: `HANDOFF-NEXT-lap18-texture-quality.md` (the plan + every dead-end + the OFFLINE
ITER HARNESS with ALIGNED data — `PIXAL3D_DUMP_BAKE` → `tex_bake_dump`; a Miku dump already exists, so
prototype the reproject with NO GPU run). Then FINDINGS-15 §1c/§5/§6/§6b, PERF-NOTES LAP 17, memory
project_3dgen_cpp_port. Build `cd tools/m1_ref/cpp_port && ./build.sh <t> cuda`; weights
`/mnt/hdd/pixal3d/weights_gguf_f16`; Miku input `tools/sparse_spike/golden_stages/pre/preprocessed.png`;
turtle `prep_test_matte.png` (heavy). Long GPU runs `run_in_background:true` from the MAIN loop
(sub-agents deadlock on bg completion). Branch `spike/sparse-conv-3d`, commit source as you go, no
Co-Authored-By. DoD: front-on render vs pyref = saturated, continuous (no cracks/patches), heels/
fingers/chin still crisp — THEN sub-4GB.
