# HANDOFF — image→rigged-3D: rungs 1-2 DONE, texture-upgrade = front-projection hybrid

**2026-07-16.** Repo `~/dev/longcat-sparse-spike` branch `spike/sparse-conv-3d`. $CP = `tools/m1_ref/cpp_port`,
$AS = `/mnt/hdd/3d/avatar-shootout`. **GPU = 3060 (index 0) is DEDICATED to this work** — `docker stop
kobbler-llama-server-1` and leave it stopped (owner disabled it); NEVER touch the 5060 (index 1, owner's other
work). Pin every run `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0`. No
pause-between-renders needed. C++/CUDA builds OK on-box; NO Rust/web builds. Nothing committed (owner rule).

## WHAT'S DONE + VALIDATED THIS SESSION

**Rung 1 (wire UltraShape into the inline driver) — DONE.** `image_to_rig.cpp` now runs the whole chain in ONE
native process, no docker/python:
`matte → pixal3d geom+tex → UltraShape refine → bbox-canon → reproject bake → auto-rig → colored rigged GLB`.
New code: `ultrashape_refine.hpp` (`usr::refine`, ported from ultrashape_e2e.cpp), in-mem voxelizer
(`us_voxelize.hpp voxelize_mesh_inmem`), `bbox_canon_onto()` (refined ±1 → pixal [-0.5,0.5] frame, mirrors
`RP_CANON_TO_DENSE`), floater-drop (`usr::drop_small_components`, env `USR_DROP_FRAC`). Flags added:
`--refine/--no-refine`, `--us-octree/-latents/-steps/-guidance/-chunk`, `--us-gguf/-dit-w/-vae-w/-cnd-w/-meta`,
`--stage-dir` (emits every intermediate GLB), `--from-geo`/`--from-refined` (resume), `--tex-snap-volume`.
Fixed 2 real bugs: stb double-impl (`us_image_proc.hpp`), and the BLACK-TEXTURE bug (refined mesh drifts
proportionally from coarse → direct PBR-volume sample misses → use `texatlas::bake` REPROJECT path against the
coarse mesh as a colored shell; default mode = mesh-attr / `RP_ATTR`). Details:
memory `project_image_to_rig_inline_ultrashape_wiring`.

**Rung 2 (quad retopo) — DONE + wired.** quadwild-bimdf built Gurobi-free at `~/dev/quadwild-bimdf/` (CMake
3.31.6 pinned + gcc16 `-fpermissive -Wno-template-body`; see `QUADWILD-INTEGRATION-RECIPE.md`). New header
`quad_retopo.hpp` (`qr::quad_retopo`, shell-out to the 2 CLIs, NO linking); `--quad`/`--quadwild-repo` flags;
stage `[1a2/4]` after refine, emits `quad.glb`. **FEED IT THE REFINED MESH (~150-165k v), NOT coarse (974k
stalls).** On the soldier: 165k refined → 55-60k 100% quads, watertight, ~40-84s. Memory
`project_quadwild_bimdf_rung2`.

**Rung 3 (eye-test) — page done.** `/home/dbrain/dev/puppy-eyetest/inline-3d/` on `:8077` (LAN IP 10.0.0.208),
fetches per-subject `stages.json`, shows source+matte thumbnails, exposure slider, flat toggle. Subjects:
`soldier1536` (default, max quality), `soldier` (res1024), `miku`, `toy2`. Rungs 4 (perf) + 5 (HYMotion
auto-animate) NOT started (deferred by plan).

**Binary:** `image_to_rig` rebuilt 17:24 (`./build.sh image_to_rig cuda`; the `.sframe` ld note is benign).
Folds in reproject + floater-drop + quad. P3-SAM GPU heads already in this build (`-DP3SAM_USE_CUDA`);
`--part-retopo` path wired but NOT yet exercised end-to-end (a TODO).

## REFERENCE ASSETS ON DISK (the max-quality run)
`$AS/_shootout_out/inline_soldier1536/` (res 1536 + texsize 4096, ~38min, peak VRAM 10.9GB — FITS):
`coarse.glb`(4.83M v) · `refined.glb`(165k) · `quad.glb`(55k quads) · `refined_tex4096.glb` (sharpest volume
tex) · `soldier1536_rigged.glb` (final: quad+rig, J=38) · `proj_front.glb` + `proj_front_vs_source.png` +
`proj_overlay_debug.png` (the FRONT-PROJECTION proof). res-1024 run in `inline_soldier/`. Caches:
`geocache_soldier1536`, `geocache_soldier` (skip 9-16min geometry via `--from-geo`). Source: `soldier.png`
(raw flux2), `soldier_matte.png` (RMBG cut). Toy soldier: red coat/gold buttons/black bearskin/navy/black
boots/skin face — colored + A-pose (A-pose avoids the T-pose limb-tip artifact pixal3d spawns).

## TEXTURE INVESTIGATION — THE CONCLUSION (this is what the next work is about)

**The texture is "soft/ugg" and that is INTRINSIC to TRELLIS-2, not a wrong setting.**
- Our tex stage IS TRELLIS.2's dedicated `Trellis2TexturingPipeline` (`app_texturing`), ported native (cos
  0.9998). TRELLIS.2 is field-free (geometry+PBR in one O-Voxel latent) → there is NO "better TRELLIS path"
  to switch to. Softness = the PBR is *generated* in a ~16×-downsampled latent (~64³) then upsampled →
  band-limited by design; it regenerates appearance, doesn't project sharp pixels. `texsize` (atlas only) and
  voxel-res levers are marginal/risky.
- **We ALREADY ran the model bake-off (June 2026, verdict in memory `project_image_to_rig_native_goal`
  lines 61-101):** Hunyuan3D-Paint 2.1 (`$AS/Hunyuan3D-2.1/`, weights `$AS/_weights/hunyuan3d-2.1/`, docker
  `hy3d21-tex`) + MV-Adapter (`$AS/MV-Adapter/`, `mvadapter-tex`) vs TRELLIS-2 on the same mesh/ref. **TRELLIS-2
  WON**; MV-Adapter "washed/muddy"; HY2.1 "competitive at fair 6-view/512 but TRELLIS.2 still wins" → HY2.1 +
  MV-Adapter PARKED, TRELLIS-2 ported. So re-porting Hunyuan-Paint = re-treading + biggest port on the board
  (dual-stream ref UNet, PoseRoPE voxel-index multiview attn, DINOv2-giant — multi-week, bit-parity risk).
  MVPainter = the only UNtested candidate (never on disk); cheaper diffusion port (ControlNet already ours) but
  same generative class. Full landscape: `RESEARCH-texture-upgrade-2026-07-16.md`.

**⇒ THE DECISION (owner-approved): FRONT-PROJECTION HYBRID.** Every generative painter regenerates appearance
(soft by design); the one thing that beats them all on the FRONT is the REAL input image. Front-projection was
PROVEN this session to give near-source detail (see `proj_front_vs_source.png` — left source, right projection,
near-identical: buttons/eyes/buckle crisp). Plan: **project the real input onto the front + Flux-inpaint the
back/occluded + bake ALL to a UV texture** (animation-safe). This is the cheapest new build AND the highest
possible front quality. Owner: "if we can make it stick" — the real risk is back/seam consistency.

## THE PLAN — build a `tex_project` bake mode (NEXT WORK)

Owner wants: **best quality, dedicated stage, hands-off (never touch up), UV-baked (animates with the rig).**
Everything needed is already ported (sd.cpp: SD-2.1 UNet, ControlNet, KL VAE, CLIP, Euler v-pred, RealESRGAN,
Flux inpaint — `~/dev/longcat-sparse-spike/src/model/...`; our `tex_reproject.cpp`/`tex_atlas.hpp` xatlas UV
bake = the back-projection stage).

**Step 1 — front-projection to UV, WITH OCCLUSION (fixes the "double face in the hat").** The prototype used a
NORMAL-GATE only (no depth test) → the face bleeds onto the hat/back-of-head and onto surfaces tucked behind
others. The exact camera (reverse-engineered, VERIFIED pixel-perfect; from `geometry_e2e.hpp ProjCam::project`,
constructed `pixal3d_chain.hpp:173` as `ProjCam(cam,dist,ms)`):
```
cam = 0.7332379 rad (miku fov), dist = 1.302156, ms = 1.0
t = tan(cam*0.5);  depth = dist - vz            // camera at (0,0,+dist) looking -z, up +y, right +x
u_norm = 0.5 + 0.5*vx/(t*depth)                 // * image_width  -> pixel col
v_norm = 0.5 - 0.5*vy/(t*depth)                 // * image_height -> pixel row  (y-flipped)
// refined.glb verts are ALREADY in this frame: marching-cubes world = (cell+0.5)/grid - 0.5, i.e. [-0.5,0.5].
// front-facing test = vertex_normal.z > 0.  fov square → resolution-independent.
```
Add a real **z-buffer visibility test** (rasterize mesh depth from the camera; only paint the frontmost surface
per pixel, + a depth epsilon; let strongly-convex caps like the bearskin defer to the back-fill). Bake the
sampled source colors into the **UV atlas** (sRGB baseColor), not per-vertex COLOR_0. ~48% of verts are
front-visible.

**Step 2 — Flux-inpaint the back/occluded UV.** Render/identify the un-projected UV regions (back, sides,
occluded) → Flux/SD inpaint conditioned on the geometry (render normal/depth of those views) + the input for
style → back-project into the same UV atlas. Reuse the sd.cpp Flux inpaint path.

**Step 3 — seam blend ("make it stick").** Blend the projected-front and generated-back across a normal.z
transition band; optionally an Im2SurfTex-style neural back-projection (arXiv 2502.14006) or a light multi-view
UV-consistency pass to kill seams/drift. THIS is the risk area — naive per-view Flux inpaint drifts.

**Integrate** as a `--tex-project` bake mode in `image_to_rig.cpp` (alongside the current reproject bake), run
on the refined-or-quad mesh, emit a `proj_tex.glb` intermediate; A/B on the eye-test vs the volume bake.
**Validate by RENDER** (owner judges; park quality levers, no declaring winners).

## RESUME CHECKLIST (after reboot)
1. `docker stop kobbler-llama-server-1` (if it auto-started) → free the 3060. Confirm GPU0 free, 5060 untouched.
2. Read this doc + `RESEARCH-texture-upgrade-2026-07-16.md` + memories `project_image_to_rig_inline_ultrashape_wiring`,
   `project_quadwild_bimdf_rung2`, `project_image_to_rig_native_goal`, `project_pixal3d_native_mesh_texturing`.
3. Eye-test still there? restart `python3 -m http.server` in `/home/dbrain/dev/puppy-eyetest` on :8077 if down.
4. Sanity: `cd $CP && ./image_to_rig` (usage prints all flags). Fast texture A/B loop =
   `--from-refined $AS/_shootout_out/inline_soldier1536 --no-rig --texsize N [--tex-snap-volume]` (CPU, ~seconds).
5. Start the `tex_project` bake mode (Step 1 above) — the front-projection prototype code the agent wrote is
   the reference; the camera math is exact. proj_front.glb + proj_overlay_debug.png show it aligns.

## OPEN / DEFERRED
- `--part-retopo` (P3-SAM segment → per-part decimate, GPU heads built) NOT yet run e2e — validate when useful.
- Floater-drop authored but its A/B (fresh refine with/without) not run.
- Rung 4 perf pass (DiT 749s@N8192 flash cut, P3-SAM encoder GPU port, HDD→NVMe staging) — AFTER functional.
- Rung 5 HYMotion auto-animate — deferred, don't start.
- Quad topology currently fan-triangulated for the tri-only bake/rig/glb; a quad-preserving GLB is polish.
</content>
