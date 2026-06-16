# MINI-PROMPT — automate the image→textured-lowpoly pipeline (texture stage + single coordinate frame)

You are continuing a 3D-asset pipeline on the RTX 3060 (12GB, currently free). **READ FIRST, in order:**
1. `~/dev/longcat-sparse-spike/HANDOFF-I-texture-automation.md`  ← your spec (commands, gotchas, the task)
2. `~/dev/longcat-sparse-spike/HANDOFF-H-geometry-eval-results.md`  ← how the geometry half was reached
3. memory: `project_pixal3d_dense_to_lowpoly`, `feedback_automate_no_manual_stitching`, `feedback_no_build_on_server`

**Context in one paragraph:** `image → pixal3d (geom+PBR) → UltraShape (clean/watertight) → P3-SAM (hands
isolate) → per-part region-adaptive QEM → 69k lowpoly with fingers` is BUILT and validated (geometry ✅,
1.04M→69k, fingers intact). pixal3d `--tex` generates a clean texture ✅. The ONE open thing: **baking that
texture onto the final 69k fails (97.8% inpaint) because the meshes are in different coordinate frames**
(pixal3d to_glb rotation + UltraShape re-normalization). It was hand-aligned as a probe and failed.

**HARD RULE:** everything **automated + reproducible** — no manual stitching, no hand-typed coordinate aligns,
no hand-written binary dumps. Python ONLY wraps the unported models (P3-SAM, UltraShape, both already
docker-ized); all mesh ops / decimate / normalize / texture-bake / matte are real committed stages (C++ via the
pixal3d port for prod; a committed driver to orchestrate). C++/docker build on-server fine; NO Rust builds.

**YOUR TASK:**
1. Solve the texture bake **automatically** via a **single canonical coordinate frame** carried end-to-end
   (HANDOFF-I §3 — option A preferred: an automated `normalize-to-canonical` stage at each boundary so
   pixal3d's PBR and the final mesh coincide; then `CUMESH_TARGET=0 tex_reproject` bakes correctly, <~5%
   inpaint, colours matching `miku_try_tex.glb`). No hand alignment.
2. Wire the full chain into ONE reproducible driver `tools/m1_ref/cpp_port/shootout/run_pipeline.sh <image.png>`
   → textured riggable lowpoly with fingers, fully automatic. Also make the **matte** step (§1.0) a committed
   stage (it's currently an inline heredoc).
3. Show the textured final 69k on `compare.html` (§5) — replace the failed `us_lowpoly_tex.glb` (viewer B) with
   the correct one. Verify colours by zooming.

**Reuse, don't rebuild:** docker images `p3sam` + `ultrashape` are built; weights fetched; stage scripts live in
`tools/m1_ref/cpp_port/shootout/` (`p3sam_run.sh`, `ultrashape_run.sh`, `per_part_decimate.py`,
`Dockerfile.*`). The patched repo files (P3-SAM `auto_mask*.py`, UltraShape `infer_dit_refine.py`) are mounted
at runtime — keep them (HANDOFF-I §4). Re-run pixal3d `--tex` (HANDOFF-I §1b) to regenerate a clean PBR dump
(the existing one was clobbered by the failed probe).

Launch heavy GPU runs `run_in_background:true` from the MAIN loop (sub-agents deadlock on bg completion).
Drive iterative work from the main loop. Validate by RENDERING/zooming, not from logs.
