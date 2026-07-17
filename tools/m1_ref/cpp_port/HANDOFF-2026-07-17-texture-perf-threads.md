# HANDOFF 2026-07-17 — image→rigged-3D: texture fixes, 4 debunked claims, and the perf baseline

Repo `~/dev/longcat-sparse-spike` branch `spike/sparse-conv-3d`. `$CP=tools/m1_ref/cpp_port`,
`$AS=/mnt/hdd/3d/avatar-shootout`. **GPU = 3060 (index 0) ONLY. NEVER touch the 5060 (owner's
ltx-video work). flux2 is OUT OF SCOPE (owner, 2026-07-17) — do not drive it.**
CPU courtesy: prefix heavy work `nice -n 15 OMP_NUM_THREADS=8` (12 cores; ltx-video-server keeps ni=0).
**Nothing is committed.** Owner judges quality by RENDER on `http://10.0.0.208:8077/`.

---
## 0. THE META-LESSON (read this first)

**Five confident claims died on contact with measurement tonight. Four were load-bearing.** Every one
was a plausible story nobody had falsified. **Two "proofs" were tautologies** (see §3). If you take one
thing from this doc: *measure the thing itself, not a statistic derived from it.*

| claimed | reality |
|---|---|
| "front-facing = `normal.z > 0`" (handoff) | **BACKWARDS** — was painting the model's INSIDE |
| "we have Flux inpaint" (handoff) | **code-only, ZERO weights** |
| "softness is intrinsic to TRELLIS-2" | **HALF FALSE** — our bake frays the volume's clean cuts |
| "cos 0.9998 = texturing is faithful" | **CIRCULAR** — validated a model against the goldens it was picked to match |
| "the flood fill makes it solid" | **leaks REGIONALLY** — only the bearskin HAT seals |

---
## 1. ✅ LANDED + ON THE EYE-TEST

- **`--tex-project`** (`$CP/tex_project.hpp`, new): textures by PROJECTING real images into the UV
  atlas. front = `--image` matte, back = `--tex-back` (flux2-generated, ON DISK at
  `$AS/_shootout_out/proj_v1/back_v1.png` — **no need to regenerate; flux2 is out of scope**).
  Z-buffered occlusion + grazing-angle confidence ramp + linear-light blend + **3D-aware hole fill**
  (97.8% of holes filled from the nearest painted texel in 3D, normal-gated) + Telea for gutters.
  Arbitrary yaw views via `--tex-view <deg> <img>` (bit-exact at 0/180).
  **Owner verdict: "pretty damn good… shoes and head is fixed"**, remaining gripes in §2.
- **`Cam::nsign`** — MEASURES the mesh's normal convention against the z-buffer (96.1% vs 3.9%).
  Never assume; `TEXPROJ_NORMAL_SIGN=1|-1` overrides.
- **`--tex-dit proj|cross`** (default cross = today) + `--tex-dit-w` — see §4, this is a REGRESSION FIX.
- **Full e2e in ONE C++ process**: matte → geometry → refine → quad → projected texture → auto-rig →
  rigged textured GLB (JOINTS_0/WEIGHTS_0/TEXCOORD_0 verified).
- **Animation page** `http://10.0.0.208:8077/anim/` — self-hosted three.js r160, offline-safe.
  **Wobble** (owner: *"really good - best animation ive seen from this"*, 68.4% of verts move).
  `ab.html` = stock / delta / **delta+base-pose** A/B on soldier + gilly.
- **MoGe verified** (`moge_cam_test` PASS, 0.79° of reference) — **but see §5, it OOMs.**

## 2. 🔧 BUILT, NOT YET A/B'd ON THE EYE-TEST (do this first — it's cheap)

Owner's outstanding texture gripes:
> *"the golden buttons… show as gold theyre just maybe not 'deep enough'… they sit a little off the bumps"*
> *"left and right side of model still has black streaks (down ears -> end of hands)"*

- **`TEXPROJ_FRONT_ALIGN=1`** (default ON) — the front DOES need the silhouette fit; my "it's exact by
  construction" was wrong. **BUT it only buys ~1.2px at the chest** (fitted scale 1.0094, translate
  (−6.00,−4.49); the several-px error lives at hands/soles, far from the fit's fixed point).
  **If the buttons are off by more than a pixel or two, no bbox fit will reach it — don't keep tuning.**
  Note the drift mostly PREDATES refine (coarse.glb is already 0.5% narrow / 0.8% short vs the matte).
- **`TEXPROJ_BG_REJECT=1`** (default ON) — black streaks = grazing texels bilinear-sampling the matte's
  BLACK BACKGROUND. Rejects samples landing outside an eroded subject mask; the 3D fill covers them.
  ⚠️ **background = what the image BORDER can flood-reach, NOT a brightness threshold** — the boots and
  bearskin are (0,0,0) too. A 0.05 threshold eats **11.14% of the subject, 63% of it the boots**
  (verified twice). Front default `TEXPROJ_FRONT_BG_THRESH=1/255` + `TEXPROJ_BG_FILL_HOLES=1`.
  **The back view has the same black boots — A/B its threshold at 0.004.**
- A/B: `TEXPROJ_FRONT_ALIGN=0 TEXPROJ_BG_REJECT=0` = baseline. Debug map `proj_bg_reject.png`
  (R=rejected, G=facing, B=hole): a correct run is a **thin red rim** with a **black green channel**.
  Falsifier: if rejected texels are NOT concentrated at grazing angles, the diagnosis is wrong — the
  code prints `[WARN: rejected texels are NOT grazing]` for exactly that. **Believe the WARN over me.**
- CPU-only probe (seconds, no GPU): `./texproj_probe <mesh.glb> <img.png> <yaw>`.

## 3. 🔬 THE DOUBLE-WALL THREAD — SETTLED after 3 agents and 2 tautologies

**Meshes ARE double-walled. The fill leaks REGIONALLY.** See memory
`project_meshes_are_double_walled_shells` (has the full 3-pass history).

| mesh | never-visible @100 dirs |
|---|---|
| **`coarse.glb`** (out of `marching_cubes_solid`) | **54.19%** ← worst AT THE SOURCE |
| `refined.glb` | **37.05%** ← refine is INNOCENT, it REDUCES it |
| quad / shipped | 37.03% / 37.38% |

```
80-96% height (bearskin HAT):        inside/chord = 1.000    -> FILLED, sealed
<=72% (head/torso/tunic/legs/boots): inside/chord = 0.02-0.26 -> SHELL, LEAKED
```
- *"The head is solid"* was a **misread — it's the HAT** (dense blob). Its fill dominates the aggregate.
- **`solid/wall = 6.245` is the shell's THICKNESS, not a fill**: 6.245/768 = **0.0081** = the measured
  chest wall (0.0078/0.0092); boot wall 0.0013 = **exactly 1 cell**.
- **BOTH earlier proofs were tautologies.** "wall vol ≈ signed vol" and "signed vol ≈ solid×cellv" are
  the SAME INTEGRAL for any closed mesh — MC bounds the solid set *by construction*. Agent B caught
  agent A's tautology **then committed the same one**.
- *"MC cannot emit a double wall by construction"* is **backwards** — it rules out **sealed** voids only.
  The void is **OPEN** (measured: 5 sealed cavities / 15 voxels ≈ 0) ⇒ `st==2` ⇒ MC emits its boundary.
- ⚠️ **The QC assert is calibrated on the false premise** (`solid/wall < 1.2` warns; reality 6.245 on a
  provably double-walled mesh). `remesh.hpp`'s comment claiming *"solid/wall = 1.000"* **does not reproduce**.

**➡️ NEXT ACTION: GPU-test `REMESH_CLOSE_R=3`** (already coded in `remesh.hpp`, dilate→fill→erode,
default 0=OFF) **and LOOK at it on the eye-test.** Predicted: coarse never-visible **54.19% → <10%**,
verts 4.83M→2.08M (**−57% = the 2nd wall vanishing**), volume +89% (**= the body FILLING, not detail
melting**), holes 50.4% → ~21.9%, ~37% of polys+atlas reclaimed. R=1 (−27%) does NOT fully seal.
**Blob-vs-body is an eye-test call, not an argument.**
⚠️ **The refine-OOM prize is FALSE** — refine is `num_latents=8192`/`octree=512`-bound, NOT vert-bound.
Expect <5% movement. The winding flip is INDEPENDENT and will survive.

## 4. 🚨 THE TEX-DiT REGRESSION (highest-value quality item — GIT-VERIFIED)

```
git show HEAD:$CP/pixal3d_chain.hpp → TEXFLOW_W = "weights_npy/slat_flow_imgshape2tex_1024"  ← PROJ = gilly's model, SHIPPED
git log -S'trellis2_tex_1024'       → EMPTY — NEVER COMMITTED
git status → pixal3d_chain.hpp = M   tex_dit_cross.hpp = ??   image_to_rig.cpp = ??
working tree → TEXFLOW_W = "weights_npy/trellis2_tex_1024"                                   ← CROSS = uncommitted WIP
```
**We SHIPPED gilly's proj-mode model. An UNCOMMITTED change swapped production onto cross-mode.
The owner's "why does ours look so bad when gilly's is decent" may simply be THAT REGRESSION —
the fix could be a revert.** `tex_dit_cross.hpp:2` says *"the model the tex_goldens were captured
from"* ⇒ someone swapped **production to match a test fixture**, and **the "cos 0.9998" is circular.**

**➡️ NEXT ACTION — the A/B (only `--tex-dit` differs):**
```bash
cd $CP
./image_to_rig --model /mnt/hdd/pixal3d/weights_gguf_f16 --image $AS/_shootout_out/soldier_matte.png \
  --tex-dit cross --res 1536 --texsize 4096 --seed 42 --no-rig --no-quad \
  --stage-dir $AS/_shootout_out/ab_texdit_cross --out $AS/_shootout_out/ab_texdit_cross/soldier.glb
#   ... --tex-dit proj  ... --stage-dir .../ab_texdit_proj  --out .../ab_texdit_proj/soldier.glb
```
- ⚠️ **MUST use `weights_gguf_f16`** (has BOTH). `weights_gguf` has **NO `trellis2_tex_1024.gguf`** ⇒
  cross **silently loads 4.9GB fp32 npy** ⇒ poisons the A/B **and inflates the live tex-DiT cost/VRAM.**
  **This is live in production right now.**
- ⚠️ **CANNOT use `--from-geo`/`--from-refined`** — those caches store `pbr_feats.bin`, which IS the tex
  DiT's output. They'd bypass the thing under test. Full chain both sides (~75min each).
  Determinism is safe (`tex_noise` is the 4th seed-42 draw, taken BEFORE the mode branch).

## 5. ⏱️ PERF BASELINE — HIGH QUALITY, MEASURED (owner's ask)

`$AS/_shootout_out/perf_baseline/` (run.log + vram.tsv @2s). res1536, texsize4096, quad+rig+tex-project,
default cam. **Total 4498.0s (75 min)** → verts=121437 faces=167340 J=33.

| stage | time | **share** |
|---|---|---|
| **`[1a/4]` UltraShape refine** | **3339.0s** | **74.3%** |
| `[1/4]` geometry | 893.1s | 19.9% |
| `[1a2/4]` quad retopo (CPU) | 109.7s | 2.4% |
| `[1/4]` tex bake (reproject/shell) | 87.6s | 2.0% |
| `[3/4]` rig | 40.6s | 0.9% |
| `[1b/4]` tex project | 22.0s | 0.5% |

Sub-stages: M3b DiT **274.8s** · tex DiT (cross, Ntok=4101) **181.8s** · tex decoder 64.3s · SS DiT
62.9s · M4 mesh 55.4s · MC-REMESH 16.7s.
**VRAM peak 10751 MiB = 87.5% of the card, 1537 MiB headroom, at t+1715s (during the refine).**
Mean-while-busy 4151 MiB.

**⇒ THE REFINE IS 3/4 OF THE PIPELINE AND OWNS THE VRAM CEILING. It is the whole perf story.**

**🔎 UNEXPLAINED — CHASE THIS FIRST:** this run's refine emitted **668,406 verts**; the reference
`inline_soldier1536/refined.glb` has **165,296**. **4x.** That may itself explain the 3339s. Diff the
`--us-*` params / decimate between the two runs BEFORE optimising anything.

**Other measured perf facts:**
- **`--moge` roughly DOUBLES both big DiTs and then OOMs.** MoGe est. 46.50° vs default 42.0° → wider
  FOV → more voxels → M3b DiT **274.8→537.4s**, tex DiT **181.8→334.8s**, coarse verts 4.83M→**5.83M
  (+20.6%)** → `usr::refine` **CUDA OOM**. MoGe's accuracy is a PERF question, not just correctness.
- **The honest OOM lever is `mc_stride`**, not a seal (code's own recipe targets ~10²-10³k faces;
  stride 2 emits **9.66M** = 10-100x that): stride 3 → 1,756,794 v (**−63.6%**, vol 0.015317);
  stride 4 → 883,138 (**−81.7%**, vol 0.019322). Volume inflates with stride = real quality tradeoff
  → **owner judges**.
- The **72s volume bake runs ONLY to obtain UVs** when `--tex-project` (projection core = 0.16s); we
  keep only `bt.uvs` + `bt.metal_rough` and throw the rest away.
- Inherited/unverified: UltraShape DiT 749s@N8192, P3-SAM encoder 64s CPU, HDD→NVMe staging.

## 6. 🎬 RIG / ANIMATION

- **SkinTokens does BOTH skeleton AND skin** from ONE Qwen3-0.6B token stream (skeleton before EOS,
  skin after). **UniRig is DEAD** (0 refs; same group, older model). J is mesh-dependent but
  DETERMINISTIC (beam=20) — **not flaky, don't chase**. Bones are unnamed `bone_N` printf.
- **Delta retarget** (`anim/tools/retarget_delta.py`): roundtrip 0.000000° PASS (stock 51-135° FAIL),
  rig-invariant 0.1° spread (stock 33.4°), un-mangles gilly's feet (135.8°→11.3°).
- **The owner diagnosed the soldier's crossing arms himself**: *"delta also assumes a specific
  position."* **Delta assumed every bind pose IS SMPL's T-pose.** Soldier is a **45° A-pose** →
  44.0° mean arm-rest mismatch → arms fold through the torso (100% of walk frames). Gilly's arms are
  fine because she's T-posed (0.9° off SMPL); her LEGS mismatch 50.5°.
  **FIX = a retarget base pose** (what Unreal/Unity ask a human for), α per chain, default
  `arm=1, leg=0`. Crossing **100% → 0%**. **α is a JUDGEMENT, not a derivation** — both rigs' quirks
  are real (gilly's legs track her own MESH to 2.5°, so **she IS bow-legged by design**; we are NOT
  imposing a pose).
  ⚠️ *"0.000000° roundtrip PASS"* was **necessary, not sufficient** — "preserve our rest exactly" IS
  "inject the rest mismatch into every frame". Same property, both symptoms.
- The **head lean is in the MOTION** (MoMask's walk leans 21.4° off SMPL rest) — not a bug. Source-data
  lever, owner's call.
- **`bonemap.derive_map` CRASHES on the soldier** (his arms hang low → fingertips below hips → the
  spine passes the "is it a leg?" test). `anim/tools/bonemap_v2.py` fixes it and reproduces stock's map
  **identically on gilly** (regression-safe).
- ⚠️ `derive_map_v2`'s L/R convention is internally inverted but **cancels exactly** (44° vs 140°) —
  **anyone who "fixes" it will silently double-flip.**

## 7. ⛔ PARKED / DEAD (do not redo)

- **Face/mouth animation = PARKED** (owner: *"eh skippable… probably more of a manual step"*). The
  canyon isn't carving (~15 lines at `ultrashape_refine.hpp:261→282` — carve the decoded `logits`
  AFTER the DiT, BEFORE MC; nothing generative runs after). It's that **the input has a CLOSED mouth
  ⇒ a carved cavity is permanently agape**, and LBS can't pinch it shut. MEASURED: nose/brows ARE real
  (0.43/0.40mm); **eyes+mouth are PAINT** (0.16mm, paint-vs-relief corr **+0.059** = decal); exactly 2
  depth layers. Locating the mouth is FREE (eyes-anchored valley, **0.27mm**, ~60 lines numpy, no LLM).
  quad mean edge 0.776mm = the feature-size floor.
- **Side-view generation = DEAD.** flux2-edit **re-draws the character in a default pose** (arms down
  vs our A-pose) — a pose change no bbox align can fix. Kept as
  `proj_v1/side_yaw90_REJECTED_pose_mismatch.png`. The 180° BACK view DID preserve the pose (that's why
  back works). **Front+back is the right API scope.**
- **Flux Fill / ControlNet inpaint**: ZERO weights on the box; Klein cannot inpaint. Don't budget it.
- **"Confetti atlas" theory**: dead — gilly's atlas is MORE shattered (27,651 vs our 4,403; identical
  charts/1k-faces). "Clean cuts" = clean COLOUR boundaries, not UV charts.
- **A morphological close as an OOM fix**: no. Use `mc_stride`.

## 8. 📋 SUGGESTED ORDER

1. **A/B §2** (front-align + bg-reject) — cheap, CPU-ish, closes the owner's last two texture gripes.
2. **A/B §4** (`--tex-dit proj` vs `cross`, `weights_gguf_f16`) — may be a REVERT-level quality win.
3. **GPU-test §3** (`REMESH_CLOSE_R=3`) + eye-test — −37% polys/atlas, −73% holes if it's not a blob.
4. **Chase the 668k-vs-165k refine anomaly (§5)** BEFORE optimising the refine.
5. Then the refine itself (74.3% of runtime), `mc_stride`, and the UV-only volume bake.
6. Winding flip at source in `usr::refine` (latent landmine — `doubleSided:true` masks it today).

**Memory files (full evidence, all written 2026-07-16/17):**
`project_trellis_softness_claim_is_half_false` · `project_meshes_are_double_walled_shells` ·
`project_ultrashape_refine_flips_winding` · `project_skintokens_is_the_whole_rig` ·
`project_flux_inpaint_claim_hollow` · `project_texture_frontprojection_hybrid` ·
`project_image_to_rig_perf_leads_2026_07`
