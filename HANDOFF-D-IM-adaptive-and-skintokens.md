# HANDOFF-D — IM-adaptive retopo (geometry polish) + start of SkinTokens rigging

Two threads, in priority order:

1. **IM-adaptive geometry** — the ONE remaining geometry lever for clean low-poly fingers. A bounded,
   fully-automated engineering project (no new model, no GPU research). Optional polish.
2. **SkinTokens rigging** — the next major *subsystem*. Geometry is now "good enough" to rig; the
   full porting spec already exists in `HANDOFF-RIGGING-skintokens.md` (R0–R7). This doc just hands
   off the **why-now** and the de-risk entry point.

Read first: `HANDOFF-C-retopo-quadriflow.md` (the retopo pipeline that landed), `V6-DENSE-RECIPE.md`
(the hero asset + colour/normal source-of-truth), memories `project_pixal3d_retopo_manifoldplus`,
`project_learned_retopo_eval`, `project_avatar_rig_path`, `project_3dgen_cpp_port`.

---

## 0. Where geometry stands (the detour is CLOSED)

The retopo question — "image → riggable quad mesh that preserves fingers" — has a working answer and
a closed dead-end set:

- **WORKS (prod path):** dense pixal3d mesh → **ManifoldPlus** (watertight manifold) → **Instant
  Meshes** (field-aligned quad) → **bake** baseColor/metalRough/**normal** from the V6 dense shell.
  Clean textured Miku on the compare page. Fingers that the quad mesh can't *carve* are recovered by
  the **normal map** baked off the dense shell — this already looks right.
- **HERO/static:** the **V6 dense bake** (`V6-DENSE-RECIPE.md`) is the cleanest asset we produce
  (owner: "the cleanest model I've seen") and the colour+normal source-of-truth for every LOD.
- **DEAD ENDS (don't re-litigate — see `project_learned_retopo_eval`):** every open learned-retopo
  model (MeshFlow, MeshMosaic/DeepMesh, MeshAnythingV2, LATO) is a low-poly artist-mesh generator
  with NO finger detail and/or needs unreleased preprocessing. QuadriFlow OOMs on our >1M-face
  non-manifold input and its `-adaptive` path has 3 `exit()` crash bugs. Commercial ZRemesher would
  work but is manual + paid.

**The ONLY open geometry lever** for carving (not normal-faking) fingers at low poly is making
Instant Meshes spend its quad budget where curvature is — that's thread #1 below. It is *polish*,
not a blocker: the normal-mapped path already ships acceptable fingers.

---

## 1. IM-adaptive retopo — the plan

### The problem
Our headless Instant Meshes batch path (`im_batch_main.cpp` → `batch_process(...)`) drives IM in its
**uniform** mode: one global target edge length, so the quad budget is spread evenly. Fingers (high
curvature, thin) get the same density as the torso (flat, low curvature) → fingers fuse / web at any
poly count cheap enough to ship. Cranking the global count to resolve fingers explodes torso polys
and defeats the "cheapest to render" goal.

IM's **interactive core is already curvature-adaptive** — it builds a per-vertex *sizing field* from
local curvature and aligns the quad density to it. The batch wrapper just doesn't expose it. So this
is a **plumbing job**, not an algorithm: surface the adaptive sizing field through the batch path,
driven by **one global aggressiveness knob**, fully automated (curvature is computed from the mesh,
no per-asset hand-tuning).

### The work (bounded)
- **Source of truth:** `thirdparty/instant-meshes/src/` — the field/sizing code (`hierarchy.*`,
  `field.*`, `meshstats.*`; the GUI exposes an "Adaptive" toggle + a scale/curvature slider). Trace
  what the GUI's adaptive checkbox flips vs. our `batch_process` call.
- **`batch_process` signature** (in `im_batch_main.cpp`): currently
  `(input, output, rosy, posy, scale, face_count, vertex_count, creaseAngle, extrinsic,
  align_to_boundaries, smooth_iter, knn_points, pure_quad, deterministic)`. There is **no adaptive
  flag** — add one (`bool adaptive`, plus a `float adaptivity`/curvature-weight in [0,1]).
- **Plumb it:** make the batch sizing-field construction follow the same curvature path the GUI uses
  when adaptive is on, instead of the constant target. Keep `deterministic=true` (reproducible).
- **One knob:** expose `adaptivity` as an env var (e.g. `IM_ADAPTIVITY`, default a sane mid value)
  so the pipeline driver picks it; NO per-region masks, NO manual painting. Curvature does the work.
- **Validate by RENDER (HARD RULE):** retopo the standard Miku at a fixed face budget with adaptivity
  0 / 0.5 / 1.0; render the **hand zoom** (`render_hand_zoom.py` / `batch_render.sh`) and put the
  variants on `compare.html` for the owner's eyes. Success = fingers separate at a face count where
  the uniform path webs them, with torso density visibly dropping to pay for it. Same total budget,
  better-spent.

### Cost / risk
- Pure C++, builds fine on-server (`build_instant_meshes.sh`; classic TBB 2020.2 via pixi,
  `-std=gnu++17 -Wno-changes-meaning`). No GPU, no model, no new dependency.
- Risk is bounded: worst case the adaptive field is unstable on our meshes → fall back to the
  uniform path that already ships. It cannot regress the working pipeline (new flag, default off).
- This is **optional polish**. If the owner is happy with normal-mapped fingers, skip straight to §2.

---

## 2. SkinTokens rigging — start here next

Geometry is good enough to rig. **Rigging is the north-star subsystem** (image → rigged,
animation-ready GLB). The full porting spec already exists and does not need rewriting:

> **`HANDOFF-RIGGING-skintokens.md`** — the R0–R7 milestone ladder, reuse inventory, gotchas, DoD.

What this hand-off adds is only the **entry decision**:

- **Port target:** SkinTokens / TokenRig (VAST-AI, **MIT**, arXiv 2602.04805). Geometry-agnostic
  (rigs any mesh → future-proofs "best geometry model varies"), MIT (clean for the commercial
  pivot), proven on the 3060 in Python at **3.2 GB / 95 s, 52 bones + per-finger**. Golden source:
  `/mnt/hdd/3d/avatar-shootout/SkinTokens/`.
- **Architecture:** VecSet mesh encoder (`3DShape2Vecset`, ~110M) + FSQ-CVAE skin codec (levels
  `[8,8,8,5,5,5]` = 64K) + **Qwen3-0.6B** AR transformer (reuse `m1_ggml.hpp` — our strongest area:
  qwen3-tts.cpp, llama-turboquant) + beam-search (beams=10) + host tokenizer / rig-GLB writer.
- **DE-RISK FIRST (R1 = the spike):** port the **VecSet mesh encoder** and validate its condition
  features vs an R0 golden dump to fp32-oracle tol *before* committing to the rest. It is the only
  real GPU unknown and the net-new primitive — and it doubles as groundwork for a future Hunyuan3D
  shape-VAE (same VecSet family). Mirror how sparse-conv was spiked before the pixal3d E2E port.
- **Then** R2 Qwen3-0.6B core (teacher-forced validation) → R3 beam decode → R4 FSQ skin decoder →
  R5 host tokenizer/grid → R6 rigged-GLB writer (`glb_writer.hpp` + glTF skin) → R7 E2E
  `pixalrig in.glb → rigged out.glb` + `pixal3d --rig`. Detail + per-stage validation in the R-ladder.

**Standing gotchas** (inherited, both threads): validate vs the TRUE-fp32 oracle not the bf16/tf32
golden; persistent-weights buffer / gallocr-recompute → NaN; fp32 tanh-GELU; `NVIDIA_TF32_OVERRIDE=0`;
long runs `run_in_background:true` (NOT detached `&`); **sub-agents deadlock on background-job
completion** → drive heavy runs from the main loop; **no multi-agent Workflow for research**; no
`pkill -f`; no rm-globs (delete named files only). C++/CUDA builds fine on-server; **NO Rust builds**.

---

## 3. Files this lap added (committed on `spike/sparse-conv-3d`)

In `tools/m1_ref/cpp_port/` (the retopo toolchain — all CPU, build via their own scripts):
- `obj_manifold_clean.cpp` — weld verts, drop non-manifold edges, keep largest component
- `obj_fill_holes.cpp` — trace boundary loops (handles bowtie junctions), fan-fill; closes all loops
- `obj_decimate.cpp` — meshopt quality QEM decimation of an OBJ
- `im_batch_main.cpp` — headless Instant Meshes batch CLI (the §1 adaptive work edits this)
- `build_instant_meshes.sh` — build headless IM (classic TBB 2020.2, GUI-free core)
- `render_hand_zoom.py`, `batch_render.sh` — finger-zoom render + variant batch for compare.html

Thirdparty (gitignored, bootstrapped by scripts): `thirdparty/ManifoldPlus/`,
`thirdparty/instant-meshes/`, `thirdparty/QuadriFlow/` (QF field-math.hpp patched: the two
`TravelField` `exit(0)` sites → return the coord; the `-adaptive` crash fix — kept for reference,
QF itself is a dead end for us).

`compare.html` (gitignored — references local GLB artifacts) carries the "Retopo: ManifoldPlus→IM"
tab with V6 dense as the quality ceiling. Eval infra for MeshMosaic/MeshFlow lives in
`kobbler/docker/meshmosaic-dev/` (containerized, weights on `/mnt/hdd/pixal3d/`).
