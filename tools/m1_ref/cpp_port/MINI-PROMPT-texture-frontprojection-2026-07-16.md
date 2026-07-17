# MINI-PROMPT — make the texture actually good (front-projection hybrid), then close the last glue gaps

You are continuing a long-running effort in `~/dev/longcat-sparse-spike` (branch `spike/sparse-conv-3d`).
$CP = `tools/m1_ref/cpp_port`, $AS = `/mnt/hdd/3d/avatar-shootout`.
**Mission:** the image→rigged-asset pipeline is DONE and native (rungs 1-2 landed: one C++ process does
matte → pixal3d → UltraShape refine → quadwild quad-retopo → bake → auto-rig → colored rigged GLB). The ONE
thing the owner is unhappy with is **TEXTURE QUALITY** ("still pretty ugg"). Fix that. Target = **best quality,
dedicated texture stage, hands-off (owner never touches anything up), UV-baked so it animates with the rig.**

## READ FIRST (trust these over anything older)
1. `$CP/HANDOFF-texture-frontprojection-2026-07-16.md` — full state + the plan + the EXACT camera math.
2. `$CP/RESEARCH-texture-upgrade-2026-07-16.md` — the verified model landscape + why we're not porting a painter.
3. Memories: `project_image_to_rig_inline_ultrashape_wiring` (rung 1 + texture decision),
   `project_quadwild_bimdf_rung2`, `project_image_to_rig_native_goal` (**the June-2026 texture bake-off
   verdict**), `project_pixal3d_native_mesh_texturing`. VERIFY memory/handoff claims against code — they have
   been stale before.

## THE KEY CONTEXT (do NOT re-litigate this)
- Our texture IS TRELLIS.2's dedicated native `app_texturing` PBR pipeline, ported (cos 0.9998). **The softness
  is INTRINSIC** — the PBR is generated in a ~16×-downsampled (~64³) latent then upsampled, so it's
  band-limited by design. There is NO better TRELLIS path (field-free O-Voxel = geometry+PBR in one latent).
  `texsize`/voxel-res/guidance levers are marginal (texsize 2048 ≈ 1024 at res1024; proven).
- **The owner ALREADY ran the model bake-off (June 2026):** Hunyuan3D-Paint 2.1 + MV-Adapter vs TRELLIS-2 on
  the same mesh/ref → **TRELLIS-2 WON** (HY2.1 competitive at fair 6-view/512 but lost; MV-Adapter
  washed/muddy). Both PARKED. So **do NOT go port a generative painter** — it's a re-tread AND the biggest port
  on the board. (MVPainter is the only untested candidate; only A/B it if the plan below fails.)
- **The lever nobody uses = the REAL input image.** Every generative painter *regenerates* appearance (soft).
  Front-projection of the actual source was PROVEN this session to give near-source detail — see
  `$AS/_shootout_out/inline_soldier1536/proj_front_vs_source.png` (left=source, right=projection, near-identical:
  buttons, eyes, buckle all crisp). No model beats the real pixels on the front.

## THE WORK — build a `--tex-project` bake mode (owner-approved plan)
Everything needed is already ported: sd.cpp has SD-2.1 UNet / ControlNet / KL VAE / CLIP / Euler v-pred /
RealESRGAN / **Flux inpaint** (`~/dev/longcat-sparse-spike/src/model/...`), and `tex_reproject.cpp` /
`tex_atlas.hpp` (xatlas UV bake) is exactly the back-projection stage needed.

1. **Front-project the real image into UV, WITH OCCLUSION.** The existing prototype (`proj_front.glb`) used a
   NORMAL-GATE only (no depth test) → the owner correctly spotted "a double face projected into the hat" and
   "drawing partially inside the model". **Add a real z-buffer visibility test**: rasterize mesh depth from the
   camera, paint only the frontmost surface per pixel (+ depth epsilon); let strongly-convex caps (the bearskin)
   defer to the back-fill rather than force-project. Bake into an **sRGB baseColor UV atlas** (NOT per-vertex
   COLOR_0). The camera is EXACT and verified pixel-perfect (from `geometry_e2e.hpp ProjCam::project`, built at
   `pixal3d_chain.hpp:173`):
   ```
   cam=0.7332379 rad, dist=1.302156, ms=1.0
   t = tan(cam*0.5); depth = dist - vz          // camera at (0,0,+dist) looking -z; up +y, right +x
   u_norm = 0.5 + 0.5*vx/(t*depth)              // * image_width  -> pixel col
   v_norm = 0.5 - 0.5*vy/(t*depth)              // * image_height -> pixel row (y-flipped)
   // refined.glb verts are ALREADY in this frame ([-0.5,0.5]); front-facing = vertex_normal.z > 0 (~48% of verts)
   ```
2. **Flux-inpaint the back/occluded UV.** Identify un-projected UV regions → Flux/SD inpaint conditioned on
   rendered normal/depth of those views (+ the input image for style) → back-project into the SAME UV atlas.
3. **Seam blend — this is the "make it stick" risk.** Blend projected-front ↔ generated-back across a normal.z
   band; consider an Im2SurfTex-style neural back-projection (arXiv 2502.14006) or a light UV-consistency pass.
   Naive per-view Flux inpaint DRIFTS — that's the thing to beat.
4. **Wire as `--tex-project` in `image_to_rig.cpp`** alongside the current reproject bake; emit a `proj_tex.glb`
   intermediate; **A/B on the eye-test vs the volume bake. Owner judges — park quality levers, don't declare
   winners.**

## THEN — close the last glue gaps (no models remain un-ported; these are the only non-inline bits)
- **Inline `obj_decimate`**: `ppd::per_part_decimate` forks `./obj_decimate` per part + OBJ round-trip
  (per_part_decimate.hpp:271). **meshoptimizer's simplifier is ALREADY linked into image_to_rig** → call QEM
  directly. Easy win.
- **quadwild**: currently a `std::system()` shell-out + OBJ round-trip (`quad_retopo.hpp`, honest first cut).
  Optional promote to in-process lib linking (QUADWILD-INTEGRATION-RECIPE.md §5b).
- **Retire `shootout/run_pipeline.sh`** — the legacy driver still shells to docker oracles + `per_part_decimate.py`
  + the Pixal3D venv. `image_to_rig` fully supersedes it; delete/redirect it so it stops trapping sessions.
- **MoGe weights**: npy dump only (no .pt/GGUF). Only matters for wild photos (`--fov`/default cam otherwise).
- **Validate `--part-retopo` e2e** (P3-SAM GPU heads are built in) + the floater-drop A/B (`USR_DROP_FRAC`).
- **THEN rung 4 perf** (handoff §6): DiT 749s@N8192 → lossless flash cut; P3-SAM encoder GPU port (64s CPU);
  HDD→NVMe weight staging. Lossless-first; park quality levers. **Rung 5 (HYMotion auto-animate) = DEFERRED,
  do not start.**

## ENVIRONMENT (verified 2026-07-16 post-reboot)
- **Host RAM upgraded to 64GB (62 total / ~45 free), swap 95GB @ 0 used → swap thrashing is no longer a
  concern.** CPU-offload headroom is now generous: big atlas bakes (texsize 4096/8192), the 4.83M-vert res1536
  reproject shell + DenseHash, and any ggml CPU-offload strategy are all comfortable. **The 12GB 3060 VRAM is
  still the only real constraint** (res1536 geometry peak 9.1GB, UltraShape refine DiT peak 10.9GB — both fit).
- **POST-REBOOT GOTCHAS:** (1) `kobbler-llama-server-1` **auto-starts** (`restart: unless-stopped`) and takes
  the 3060 — `docker stop` it again and leave it stopped. (2) The eye-test server dies with the box — restart:
  `cd /home/dbrain/dev/puppy-eyetest && nohup python3 -m http.server 8077 >/tmp/eyetest8077.log 2>&1 &`.

## WORKING RULES
- **GPU: the 3060 (index 0) is DEDICATED to you.** `docker stop kobbler-llama-server-1` and LEAVE it stopped.
  **NEVER touch the 5060 (index 1)** — it's the owner's other work (often busy). Prefix every run
  `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 NVIDIA_TF32_OVERRIDE=0`; confirm on `nvidia-smi -i 0`.
  No pause-between-renders needed. Peak VRAM at res1536+refine = 10.9GB / 12GB — fits.
- **Drive GPU runs from the MAIN loop; use sub-agents for authoring/compiling/CPU sweeps/research** to keep
  context clean. Sub-agents must not touch the GPU.
- **Builds:** `./build.sh image_to_rig cuda` (C++/CUDA on-box OK; NO Rust/web builds). `.sframe` ld note = benign.
- **Judge by RENDER on the eye-test**, never a number. `http://10.0.0.208:8077/inline-3d/` (subjects:
  soldier1536 / soldier / miku / toy2; serves from `/home/dbrain/dev/puppy-eyetest/inline-3d/`, restart
  `python3 -m http.server 8077` there if down). Owner judges quality; don't lock winners.
- **Commit nothing** unless asked (no AI mentions in messages).

## STARTING POINTS
- Binary `$CP/image_to_rig` (rebuilt 2026-07-16 17:24; `./image_to_rig` prints all flags).
- **Fast texture A/B loop (CPU, seconds, no GPU):**
  `./image_to_rig --model weights_gguf --image $AS/_shootout_out/soldier_matte.png --out /tmp/x.glb \
     --from-refined $AS/_shootout_out/inline_soldier1536 --no-quad --no-rig --texsize N [--tex-snap-volume]`
- Reference run `$AS/_shootout_out/inline_soldier1536/` (res1536+texsize4096): coarse(4.83M) / refined(165k) /
  quad(55k) / refined_tex4096 / soldier1536_rigged (J=38) / **proj_front.glb + proj_front_vs_source.png +
  proj_overlay_debug.png** (the front-projection proof). Geometry caches `geocache_soldier1536` (`--from-geo`
  skips the 16min diffusion). Source `soldier.png` / `soldier_matte.png`.
- Fresh subject: flux2 gen (3060-pinned, `scratchpad/gen_flux_3060.sh` pattern: POST `/sdcpp/v1/img_gen` with a
  `gpu` field = `GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6`, then `POST :8095/v1/admin/unload`) → RMBG
  `curl -X POST ":18898/remove?bg_mode=alpha" -F images=@x.png -F gpu=<3060-uuid>` → `./make_matte rgba.png matte.png`.
  Prefer **A-pose** subjects (T-pose makes pixal3d spawn stray bits at limb tips that UltraShape then welds in).
</content>
