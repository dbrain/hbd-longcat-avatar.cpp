Continue the C++/ggml port of Pixal3D (TRELLIS.2): `pixal3d --model <gguf> --image <png> --out <glb> [--tex]`.
The geometry+texture MATH and a big PERFORMANCE run are DONE+validated; **your job is FEATURE-COMPLETE
vs the Python library FIRST, then back to performance.** Run FULLY AUTONOMOUSLY (owner away; GPU+CPU
100% free): decide, do, golden-validate, document, continue. Only halt if genuinely hard-stuck.

START by reading IN FULL, in /home/dbrain/dev/longcat-sparse-spike/:
  **HANDOFF-NEXT-feature-complete.md** (★ the full plan: what's done, the feature gaps, the parked perf)
  · HANDOFF-NEXT.md (product-goal ladder) · CONFIGS-RESOLVED.md (tex decoder = out-6 PBR) · PERF-NOTES-pixal3d.md.
  Memory: project_3dgen_cpp_port, project_avatar_rig_path, feedback_correctness_before_perf,
  reference_subagent_background_stall.

THE WORK, in order:
PHASE 1 — FEATURE-COMPLETE (match the core Python pipeline):
  1. **FULL UV-ATLAS PBR TEXTURES** (the headline gap — replaces the interim per-vertex COLOR_0). The
     per-voxel 6-ch PBR (baseColor3/metallic/roughness/alpha) is already computed+validated (m6_tex_decode,
     maxabs 3.5e-6). MISSING = `o_voxel.postprocess.to_glb`: xatlas **uv_unwrap** + a **CUDA raster** of a
     2048² PBR atlas (sample each texel's surface point → nearest-voxel PBR) → glTF embedded
     **baseColorTexture + metallicRoughnessTexture** + per-vertex TEXCOORD_0. Capture the Python `to_glb`
     output on the golden mesh+PBR to validate (visual + attribute-sampling, not bit-exact). Largest net-new
     piece (xatlas + a CUDA rasterizer); scope it first — it's geometry→UV→raster→embed, no new model math.
  2. **Raw-photo front-end** (ingestion parity; CLI currently needs a preprocessed square matte + --fov):
     host-side rembg (BiRefNet RMBG-2.0) + crop, and either port MoGe (camera) or keep --fov as the cut-line
     (confirm which with the owner; cheapest = a small host rembg.py + --fov).
  3. **glb_writer JSON** → bundled thirdparty/json.hpp (nlohmann), alongside the richer UV-atlas glTF.
  4. **Parity sweep** vs pixal3d/pipelines/pixal3d_image_to_3d.py — confirm nothing else is unported.
PHASE 2 — BACK TO PERFORMANCE (only after feature-complete; re-profile, features shift the picture):
  parked levers in PERF-NOTES-pixal3d.md — spike conv kernel (~18s, bit-exact fp32 tiling), SS-DiT
  flash-attn, VRAM<7.5GB (measured peak 8.0GB@M3b — owner says fine for now) / imatrix-Q4 sub-4GB.

KEY FACTS: build `cd tools/m1_ref/cpp_port && ./build.sh <test> cuda`; weights live on **/mnt/hdd/pixal3d**
(symlinked back); the validated fast config = `pixal3d --fast` (needs weights_gguf_f16/); the **default path
stays bit-exact** (perf is opt-in). Live 3-up compare page: http://10.0.0.208:8011/compare.html.
GOTCHAS: validate vs the TRUE-fp32 oracle (not the bf16 golden), judge E2E by mesh agreement; persistent-
weights buffer (gallocr recompute→NaN); fp32 tanh-GELU; float64 t_seq; tex-decode oracle needs GPU VISIBLE.
C++/CUDA builds fine here; **NO Rust builds**. Long jobs `run_in_background:true` (NOT detached `&`);
sub-agents DEADLOCK on background completion → drive heavy runs from the main loop; no multi-agent Workflow
for research; no `pkill -f` / no rm-globs. Keep HANDOFF + PERF-NOTES + memory current; note where you stop.

DEFINITION OF DONE: a **fully-textured** `pixal3d --tex` — image → glTF with embedded UV-atlas PBR
(baseColor + metallicRoughness) matching the Python `to_glb` — plus a crisp parity statement (what, if
anything, still differs from the core Python pipeline). Then resume measurable performance gains.
