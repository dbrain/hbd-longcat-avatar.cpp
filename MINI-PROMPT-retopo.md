# MINI-PROMPT — auto quad-retopo (QuadriFlow) lap, full-automation

You are picking up the pixal3d 3D-asset pipeline. **Task: add an automated quad-retopology stage so the
output mesh is manifold + field-aligned (quad-derived), which gives clean low-res (2048/1024/512)
textures AND animation-ready deformation topology.** Work autonomously, build→validate in a loop, and
get an owner eyeball ONLY at the one visual gate (R2, below).

## Read these first (in order)
1. `HANDOFF-C-retopo-quadriflow.md` — the full scope: why, the QuadriFlow decision, where it slots, the
   R0–R6 ladder, reuse inventory, risks, Definition of Done. **This is your spec.**
2. `HANDOFF-B-quality-and-ports.md` — the in-process meshopt+KTX2 packer + bake recipe you build on, and
   the "Packed-tier visual verdict" (the 2048 teal-seam failure you are fixing).
3. `FINDINGS-A-resolution-cumesh-bake.md` — the bake recipe + DEAD levers (don't retry normal_offset/
   dilate / bigger-atlas-more-downsample).
4. Memory: `project_pixal3d_inprocess_glb_pack`, `project_3dgen_cpp_port`, `project_avatar_rig_path`
   (the SkinTokens rig stage this feeds).

## The job, in one line
`pixal3d --remesh` (manifold marching-tet, already exists) → **QuadriFlow** quad-retopo → triangulate →
real xatlas `ComputeCharts` (drop the 182k-chart precluster) → `tex_atlas` bake → `glb_packed` meshopt+KTX2.
Net-new stage; there is NO Python oracle (TRELLIS does no retopo) — judge by the quality gate, not parity.

## The ONE success gate (R2) — get owner eyeball here, nowhere else
Bake at **2048** on the retopo'd mesh and confirm the **teal hair-chart seams are gone** (the HANDOFF-B
failure), with chart count in the **hundreds** (not 182k). Surface it in `compare.html` (:8011) for the
owner. Everything before R2 is autonomous; don't ask for sign-off on R0/R1 mechanics.

## Working norms (this environment)
- **Build from the main loop, not sub-agents** — sub-agents stall on `run_in_background` job completion
  ([[reference_subagent_background_stall]]). Bounded fan-out READS are fine.
- **C++/CUDA builds fine on this host** (toolchain g++ 12.4 at `/mnt/hdd/3d/avatar-shootout/toolchain`).
  The no-build rule is **Rust-only**. Use `build.sh`; copy the `build_basisu.sh` bootstrap-on-demand +
  gitignore pattern for vendoring QuadriFlow (pinned ref, `BUILD_FREE_LICENSE=ON` to skip Boost).
- **GPU is shared** — it may be free or the owner may be benching; check `nvidia-smi` and don't run
  heavy GPU work concurrently with their testing. Launch long GPU runs `run_in_background:true`
  (harness-tracked), never detached `&`. Most of QuadriFlow + unwrap is CPU; only the bake/generation
  touch the GPU.
- **No multi-agent Workflow / deep-research for ordinary research** ([[feedback_no_workflow_for_basic_research]]) —
  a few WebSearch/WebFetch or one Explore sub-agent at most.
- **No `rm` globs**; chain build→run with `&&`; clean up stray procs/PIDs before handoff.
- **No Co-Authored-By / AI mention in commits.** Commit only when it works + the owner asks; you're on
  branch `spike/sparse-conv-3d`. Add files by explicit name (`.glb`/`.ktx2`/binaries are NOT gitignored).

## Validation harness (use every loop)
- `gltf_validator -o file.glb` (at `/tmp/gltf_validator`) → must be **0 errors**.
- `./meshopt_verify file.glb` → every meshopt stream must **DECODE OK** (catches what the validator
  can't — e.g. the missing-`count` bug from HANDOFF-B).
- `compare.html` on :8011 (already serving; `python3 -m http.server 8011`) — model-viewer A/B/C tabs.
  meshopt is wired via `meshoptDecoderLocation` (see the bottom module); KTX2 is on by default.
- Four model-viewer/glTF gotchas already fixed (meshopt `count`, integer accessor min/max, vertex codec
  v0, decoder location) — see HANDOFF-B; don't reintroduce them.

## Definition of Done (from HANDOFF-C)
1. `pixal3d --retopo <N> --tex --pack` → hundreds of charts, field-aligned topology.
2. Clean 2048 + 512/1024 LODs, teal seams gone, owner-confirmed in model-viewer.
3. `gltf_validator` 0 errors + `meshopt_verify` all-decode.
4. Retopo'd mesh skins/deforms cleanly under SkinTokens (R5 spot-check).
5. Deterministic, headless, C++-only, MIT-clean.

Start at R0 (vendor + build QuadriFlow headless), prove it runs on a test mesh, then climb the ladder.
Update `HANDOFF-C-retopo-quadriflow.md` with findings as you go.
