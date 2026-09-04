# Image → textured asset → generic rig: clean end-to-end runbook

This is the authoritative operating procedure for a **new** image-to-rig
delivery. It is deliberately fail-closed: a GLB is publishable only when its
texture provenance, skeleton structure, and real glTF skin deformation all
pass. Do not replace a rejected asset with a nicer-looking diagnostic.

## Deliverables and current eye test

The reference deliveries are the fresh native-only end-to-end runs. These are
what a new subject should reproduce, and they are the ones that pass every gate
including weight health (audited 2026-07-24):

- `_shootout_out/fresh_e2e_nativeonly_20260723/miku/` — rig score 0.910,
  J=37, 22/22 Mixamo core, symmetry 1.000, 25/37 influential joints,
  biggest joint 16.4% of skin mass, clean head-turn pose gate.
- `_shootout_out/fresh_e2e_nativeonly_20260723/gilly/` — rig score 0.908,
  J=56, 22/22 core, symmetry 0.982, whole-tree/component worst 3.032x.
- `_shootout_out/fresh_e2e_nativeonly_20260723/soldier/` — rig score 0.766,
  22/22 core, passes weight health. Its low coverage (0.221) and 46.3%
  single-joint share are **not defects**: 44.7% of that mesh is a rigid bearskin
  busby above the head joint, correctly bound to `mixamorig:Head`. The body rig
  is complete down to the toes (lowest joint within 5% of the mesh floor).

> **Do not use `miku_p3sam_attachment_recovery_e2e_v5_20260723/attachment_recovery_experimental.glb`.**
> It was previously listed here as the clean Miku delivery. It is not: 2 of 50
> joints carry material weight, one joint holds 58.8% of all skin mass, and its
> pose gate render flings the hair across the body and opens a hole in the
> head — while still "passing" the p999 stretch gate. Its own
> `recovery-status.txt` says `experimental-review-required` and
> `Do not publish automatically`. The fresh e2e Miku above supersedes it.

- Clean cache-first textured generic Gilly delivery (retained, still valid):
  `runbook_image_to_rig/gilly_modelready_cachefirst_generic_e2e_20260723/generic_rigged_branchsafe_verified.glb`
- Eye-test page:
  `http://10.0.0.208:8077/inline-3d/e2e-runbook.html`
- Direct clean-texture review:
  `http://10.0.0.208:8077/inline-3d/runbook.html?subject=miku_p3sam_attachment_recovery_e2e_v5_20260723&artifact=attachment_recovery_experimental.glb`

The page may contain flat-colour rig controls. Those prove skeleton and skin
behaviour only; they are never evidence of a texture delivery.

## Stage explorer (eye test every stage, not just the delivery)

A delivery-level review cannot tell you *which* stage went wrong. The stage
explorer runs the whole pipeline fresh, retains every intermediate, and puts
them behind a subject selector and a stage selector so a fault can be attributed
to the stage that produced it rather than the stage where it became visible.

```sh
CP=/home/dbrain/dev/longcat-sparse-spike/tools/m1_ref/cpp_port
ROOT=/mnt/hdd/3d/avatar-shootout/_shootout_out/stage_explorer_$(date +%Y%m%d)
"$CP/shootout/stage_explorer_e2e.sh"        "$ROOT"    # fresh e2e, all image subjects
"$CP/shootout/stage_explorer_postprocess.sh" "$ROOT"   # pose gate + exercise clip + stages.json
ln -sfn "$ROOT" /home/dbrain/dev/puppy-eyetest/inline-3d/stage_explorer
```

Then `http://10.0.0.208:8077/inline-3d/stage-explorer.html`.

Stages shown, in pipeline order: input matte → Pixal3D coarse seed → UltraShape
refined → Hero / high / medium / low textured → rigged hand-off → skeleton pose
gate → skeleton exercising. Every model stage can be viewed textured or forced
to untextured clay, so a texture cannot disguise a geometry fault. "compare all
stages" puts the whole chain side by side for one subject. The side panel
carries the run's wall clock, peak VRAM, per-stage wall time, geometry sizes and
the pose-gate verdict.

The exercise clip (`rig_exercise_anim.py`) swings every materially weighted
joint in turn, so the rig can be judged in motion. It is derived from the
published rig and never modifies it.

Two subject kinds share the page:

- **image-driven** (miku, gilly, soldier) run the full chain above.
- **mesh-sourced creatures** (moth, fairy, winged imp, fallenangel, angelriggy,
  tira, winged bird, giraffe) have no input image — they enter at the rig stage
  from an authored source mesh, so their stages are source mesh → rig → skeleton
  → exercise. Run them with `stage_explorer_mesh.sh` into the *same* root.

```sh
"$CP/shootout/stage_explorer_mesh.sh" "$ROOT"        # creature limitations sweep
"$CP/shootout/stage_explorer_postprocess.sh" "$ROOT" # covers both kinds
```

This sweep exists to show the **limitations**, so a subject whose rig gate
rejects it is kept and labelled `RIG REJECTED` in the subject selector rather
than dropped, and the raw pre-gate candidate is retained beside it. An empty rig
stage on this page is a finding, not a missing file.

## Storage and build hygiene

All generated meshes, atlases, pose images, logs, and fixtures belong below:

```sh
export IMAGE_TO_RIG_OUT_ROOT=/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig
```

Never use Docker builds as an exploratory step in this runbook. Native C++
binaries and the Python environments are prerequisites. Docker builder cache
lives on `/` and can exhaust it even though delivery artifacts live on
`/mnt/hdd`. Before a run, confirm both the output mount and root have space:

```sh
df -h / /mnt/hdd
docker system df
```

If `/` is low, stop before creating or editing evidence. Do not make a nested
image-on-builder chain to recover a failed run. Builder-cache cleanup is an
operator action; never prune active images, volumes, or containers from this
pipeline.

All model inference is reserved for the RTX 3060 (`GPU-3b9ac5cf-95c5-5c9e-de19-af33af4b27d6`, PCI `04:00.0`).
Entrypoints discover and validate that UUID before launching CUDA or Docker;
they must reject the RTX 5060 Ti rather than relying on a numeric device
ordinal. The 5060 is another agent's lane and is never an image-to-rig fallback.

## Inputs that are allowed to become a delivery

1. A cleaned GLB with real positions, normals, UVs, and an embedded base-colour
   image; or an explicitly labelled flat-colour articulation diagnostic.
2. FPS rig samples generated from that exact GLB by `mesh_sample_main`.
   `sampling_provenance.txt` binds samples to the source path, and the chain
   rejects stale samples.

   > **Known fragility (2026-07-24):** those samples are taken from the
   > *textured* LOD, whose vertex set depends on UV-seam splitting, so the rig
   > conditioning changes when the atlas route changes even though the surface
   > does not. Gilly passed its rig gate on the `direct` atlas route and failed
   > it on the (correct, faster) `clustered` route from byte-identical
   > `refined.glb`. Rig acceptance is therefore not reproducible across
   > texture-side changes. Evidence and the recommended fix — condition on
   > `refined.glb` and keep transferring onto the delivered LOD — are in
   > `~/handoffs/tools/m1_ref/cpp_port/FINDINGS-rig-conditioned-on-atlas-route-2026-07-24.md`.
3. For a textured image-to-rig delivery, a generated Pixal PBR cache retained
   from the same live geometry/material run. Its manifest binds input image,
   refined mesh, PBR feature, and PBR-coordinate hashes.

Standalone direct M6 re-inference is a diagnostic only. It has produced a
corrupt atlas and must not be promoted as a texture fallback.

## Canonical texture route

For a new image + cleaned mesh, use the top-level wrapper only with the
projection-conditioned PBR cache from `native_image_to_rig_from_image.sh`:

```sh
CP=/home/dbrain/dev/longcat-sparse-spike/tools/m1_ref/cpp_port
cd "$CP"
NATIVE_GENERATED_PBR_CACHE=/absolute/cache/from/the/same/image-run \
RIG_PROFILE=generic \
./shootout/native_image_to_rig.sh refined.glb same-frame-rgba.png "$IMAGE_TO_RIG_OUT_ROOT/my_subject" my_subject
```

The wrapper refuses an unproven cache by default. It runs one authoritative
native PBR projection, then rebakes that same material to every LOD rather
than independently re-inferring atlases.

### Asset tier contract

Every successful image-to-rig delivery emits these separate assets. The rig
is not the visual master: it is one validated consumer of the `high` tier.

| Asset | Geometry | Intended use |
| --- | --- | --- |
| `native_hero_textured.glb` | Full refined mesh (`NATIVE_HERO_FACES=0`) | Close-up/hero render; 8K atlas by default; not auto-rigged. |
| `native_high_textured.glb` | 300k faces | High-quality interactive or rig source. |
| `native_medium_textured.glb` | 150k faces | General game character. |
| `native_low_textured.glb` | 50k faces | Background/distant character. |
| `hymotion_rigged.glb` / `generic_rigged.glb` | The highest tier whose rig passes | Animation hand-off; never silently substitutes for the Hero. |

> **Fixed 2026-07-24 — the "highest tier whose rig passes" ladder had never
> actually run.** `native_image_to_rig.sh` selects a rig with
> `for level in high medium low; do try_rig "$level" && break || { ... }; done`,
> but `emit_rig_failure_summary` opened with
> `local name="$1" log="...${name}..."`. Bash expands every word of `local`
> *before* the assignments take effect, so `${name}` was read while still unset
> and died under `set -u`. That aborted the `|| { ... }` block before its
> "trying next LOD" line, so a rejected high-tier rig ended the run instead of
> falling back to medium or low. Split into two `local` statements.
>
> Consequence for older results: any subject recorded as "rig rejected" was only
> ever tried at the **high** tier. Miku is the clear case — its fresh high-tier
> decode is reproducibly rejected at 13/22 Mixamo core, missing the same nine
> bones on every beam at both widths (Spine, Neck, Head, the whole right arm
> chain, both hands, LeftToeBase), and the accepted Miku delivery in the runbook
> came from a **manually** run low-source rig plus high-mesh transfer. That
> manual workaround is what the ladder was supposed to automate.

The Hero keeps every refined face and is textured from the same retained PBR
cache. Its default `NATIVE_HERO_UNWRAP=production` uses bounded local charts;
set `NATIVE_HERO_UNWRAP=reference` only for a labelled direct-chart parity
A/B. `NATIVE_HERO_FACES`, `NATIVE_HERO_ATLAS`, and the high/medium/low atlas
variables are deliberate delivery knobs, not cache aliases. The manifest names
`production_texture=native_hero_textured.glb` and
`rig_texture=native_high_textured.glb` so the distinction survives hand-off.

### CUDA Hero atlas evidence

`texture_rebake_native` also has a CUDA build.  With
`NATIVE_ATLAS_CUDA=1 ATL_NATIVE_CUMESH=1`, it reserves the same 3060 UUID,
uses native CuMesh for bounded chart pre-clustering, and rasterises packed UV
triangles into position/normal/mask buffers on CUDA.  The existing native PBR
sampling, chart-local repair, and quality gates remain authoritative. The
rebake sidecar records the 3060 UUID, peak VRAM, CPU peak, stage log, route,
face count, and atlas size; `atlas_raster_cuda_complete` is required evidence
for this route.

On the fresh full-resolution 8K controls, this yielded Miku 127s (from 146s
CPU) and Soldier 92s (from 111s CPU), with zero unresolved surface texels and
the existing recovery gates passing. Gilly's 87s CUDA control is retained as a
quality-equivalent measurement, not claimed as a speedup over its 86s CPU
baseline. No default delivery may select the unbounded global direct-chart
route.

For a pre-existing textured mesh and fresh rig samples, use the lower-level
chain (this does not generate a new texture):

```sh
CP=/home/dbrain/dev/longcat-sparse-spike/tools/m1_ref/cpp_port
PY=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
SRC=/absolute/source_textured.glb
OUT=/mnt/hdd/3d/avatar-shootout/_shootout_out/fixtures/my_subject/generic_rigged.glb
RIG_IN="$(dirname "$OUT")/rig_inputs"

mkdir -p "$RIG_IN"
cd "$CP"
./mesh_sample_main "$SRC" "$RIG_IN"
RIG_PROFILE=generic RIG_SKIN_MODE=learned-smooth RIG_SKIN_SMOOTH_ROUNDS=16 \
PIXAL3D_GGUF_DIR=/mnt/hdd/3d/avatar-shootout/_weights/skin_vae_gguf \
R1W_SRC=/mnt/hdd/3d/avatar-shootout/rig_audit/r1w_real \
./rig_texture_chain.sh "$RIG_IN" "$SRC" /home/dbrain/models/3d/rig/qwen3_w "$OUT" 20
```

Use `RIG_ALLOW_FLAT_BASECOLOR=1` only for a recorded source diagnostic with no
usable image. It is not permission to publish an untextured generated asset.

## Generic rig policy

`RIG_PROFILE=generic` is for creatures, wings, tails, props, and other
non-humanoid body plans. Its node names are intentionally non-semantic and
complete: `skintokens:Root_*` and `skintokens:Joint_*`. Do not invent Arm,
Wing, Tail, or Mixamo labels from geometry.

The deterministic C++ decoder is the only production route. Upstream sampled
decoding is hard-disabled in the delivery wrappers: setting
`RIG_OFFICIAL_SAMPLED_FALLBACK=1` is an error. Likewise, Python may render or
read a pose gate but cannot repair a production skeleton or weight field.
Historical sampled artifacts remain diagnostic evidence only and must never be
added to a production review manifest.

Every generic delivery must satisfy all of the following:

- one rooted tree, size-aware fan limit `max(8, ceil(J/5))`, rig score >= 0.50;
- complete stable generic namespace;
- actual written glTF LBS audit over every materially weighted non-root joint;
- visible motion and worst whole-tree/component p999 stretch <= 6.0x;
- **weight health**: at least 4 joints carrying >= 8% of peak joint mass. A high
  single-joint share is a *review flag*, not a failure — the soldier's Head bone
  legitimately holds 46% of the mesh because 44.7% of that model sits above the
  head joint and is a rigid bearskin busby;
- texture/provenance gate when a texture is claimed.

Run the independent final gates explicitly when inspecting an output:

```sh
$PY rig_pose_smoke.py artifact.glb artifact.pose-gate.png \
  --generic-all-influential --show-skeleton --pose-gate
./rig_score artifact.glb 55
$PY rig_weight_health.py artifact.glb
```

**Always look at the pose-gate PNG.** The stretch gate is a percentile test and
can pass a visibly broken rig: `p999 <= 6.0` tolerates ~0.1% of vertices tearing
arbitrarily far, and the audit only exercises joints holding >= 8% of peak mass,
so a *more* degenerate weight field gets a *weaker* audit. That is why
`rig_weight_health.py` is now a required gate rather than advice — it is the
check that rejects the nominally-50-bone / effectively-2-bone case the stretch
percentiles waved through.

### Decoder acceptance criterion: structural, never token-identical (2026-07-24)

Do not attempt, request, or score native/Python token parity. There is no
unique Python answer to match: two correct upstream attention backends, on the
same GPU, runtime, weights and condition tensor, agree on only 25 of 249
generated tokens and first diverge at generated index 5. Beam search over a
near-tie amplifies any 1-ULP difference into a different rig, so partial token
agreement is never partial progress. Full evidence, harnesses and the
structural native-vs-Python A/B across the winged set:
`~/handoffs/tools/m1_ref/cpp_port/RESULTS-skintokens-parity-closed-2026-07-24.md`.

Two consequences for this runbook:

- The native grammar in `rig_grammar.hpp` deliberately does **not** match stock
  upstream. Upstream's decoding mask treats `branch` as a single
  three-coordinate joint while its own detokenizer consumes six (parent triple
  plus child triple), which makes `eos` legal halfway through a branch payload.
  Native implements the corrected six-coordinate form. Do not "fix" it back.
- Compare decoders on J / roots / maxfan / coverage / bilateral symmetry /
  worst orphaned-vertex distance over shared FPS samples:

```sh
~/handoffs/tools/m1_ref/cpp_port/rig_native_vs_python_ab.sh subject=/abs/source.glb
$PY ~/handoffs/tools/m1_ref/cpp_port/rig_ab_report.py subject
```

On the winged set (angel, moth, fairy, imp, fallenangel) the two decoders trade
places by subject rather than one dominating: fallenangel is a near-tie,
native articulates more on angel, Python more on moth, upstream Python
*collapses* on fairy (J=199, 94-child fan) where native returns a clean J=65
tree, and both collapse on the winged imp. Deterministic-beam collapse on a
hard creature is a model limitation; the structural gates below reject it, and
`generic-structural-select` walks the remaining beam hypotheses in score order.
It correctly refuses when no beam ever reaches eos, as happens on the imp.

### Current native decoder parity evidence (2026-07-23)

The Qwen shared-prefix prefill has an operation-local, causal GQA FA2
contract: it is selected only by
`ggml_flash_attn_ext_qwen_causal_gqa()` for the exact F32 Qwen shape
`[128,T,16]` / `[128,T,8]`.  It is not an environment toggle and ordinary
null-mask flash attention cannot select it.  Production deterministic decoding
uses an F32 KV cache for this path; fixed-seed native CUDA Philox sampled
diagnostics use their separately validated BF16 cache path.  The accepted
manifest records both `r3_kv_type` and `r3_attention_prefill`.

The operation-local FA2 reruns below used real native R1/R4 conditioning and
the real glTF pose gate.  They supersede earlier decoder-path claims for these
controls:

| Subject | Current result |
| --- | --- |
| Gilly v4 | accepted native generic delivery; 22/22 named core; whole-tree/component worst 2.408x |
| Soldier v8 | accepted native humanoid delivery; 22/22 named core; whole-tree/component worst 4.673x |
| Tira v12 | accepted native generic delivery; 22/22 named core; whole-tree 1.118x, all-influential 1.842x, component 5.390x |
| Miku v1 | deterministic direct native decode rejected: 15/22 core (missing Spine, Neck, right shoulder/arm/forearm/hand, and Head); no publication manifest |
| Giraffe v4 | deformation gate passes (whole-tree 1.439x, all-influential 2.262x, component 2.482x), but generic structural quality is 0.495 < 0.50; rejected, not promoted |

Moth and Angel remain blocked at the source-texture provenance gate: their
source GLBs refer to unavailable external textures.  Do not reuse a historic
rigged output as a substitute texture source.  A fresh accepted delivery needs
the original licensed texture bundle or a same-run retained PBR cache.

## Native-only skin policy

`RIG_COMPONENT_BRANCH_REPAIR=1` is hard-disabled in production. A raw native
candidate that fails the real-LBS gate remains rejected until an equivalent
C++ candidate generator can demonstrate a pass and preserve the same gate.
This keeps Python strictly validation-only and prevents an edited weight field
from being represented as native SkinTokens output.

## Accepted controls

| Control | Result |
| --- | --- |
| Clean textured Gilly J55 | generic namespace; whole-tree/component worst 3.032x; clean PBR atlas |
| Miku P3-SAM body-view probe | native label-mask view yields a full 22-bone C++ SkinTokens tree; the full-mesh native C++ attachment candidate remains diagnostic until it passes bone falsification and the real-LBS pose gate |
| Winged bird J42 | passed deterministic native C++ R1/R3/R4; stable generic namespace; 21 influential joints audited; component worst 2.850x |
| Fairy J64 | winged humanoid; component worst 4.314x |
| Winged imp J54 | articulated wing control; component worst 1.760x (flat-colour source diagnostic) |
| Moth sampled J36 | historical sampled diagnostic only; not publishable under the native-only policy |
| FallenAngel J41 | 32 influential joints; component worst 5.467x |
| Tira sampled J53 | textured tail-bearing control; branch repair plus one proven 168-face detached root/joint repair; final component worst 3.045x |
| AngelRiggy sampled J30 | raw skin rejected at 20.842x; seven bounded branch repairs pass at 1.665x (flat-colour diagnostic because source texture is external) |
| Textured giraffe J56 | deterministic native C++ R1/R3/R4; pathological 30-child head fan is locally normalized to 12/12, then native bone-local geometric skinning passes the 17-joint all-influential audit (whole-tree 4.966x; component 5.167x) |
| Soldier high J26 | deterministic native C++ R1/R3/R4; exact mirrored right-arm recovery plus native sampled-geometry head/collar normalization; 22/22 Mixamo core; arm pose p999 4.614x |
| Miku high texture / low-source J55 | deterministic native C++ R1/R3/R4 on the explicitly declared 50k low source, then native high-mesh transfer and 48-pass smoothing; 22/22 Mixamo core; high-mesh arm pose p999 1.891x |

## Quadruped control: giraffe

Native deterministic decoding produces J56 with a pathological 30-child head
fan. The native route retains the decoded joints but deterministically
reparents only that overflow to the nearest retained sibling, yielding the
same J56 tree at maxfan 12/12. Its learned R4 field still failed the actual
LBS audit because tiny head/tip micro-branches received material support.
The generic-only native C++ geometric candidate therefore blends the nearest
parent-child bones and leaves sub-4%-diagonal micro-branches unweighted. It
passes the complete all-influential GLB audit: p999 4.042x, worst whole-tree
4.966x, worst component 5.167x (limit 6.0x). The artifact is:

`/mnt/hdd/3d/avatar-shootout/_shootout_out/fixtures/skintokens_giraffe_generic_native_localfan_geometric_pruned_20260723/giraffe_generic_rigged.glb`

Status is **deformation-accepted, structurally rejected**, and the runbook
should say so rather than list it as accepted outright. Re-measured 2026-07-24:
it scores `TOTAL=0.373`, below the >= 0.50 structural gate, because the
symmetry term is 0.000 for this body plan. Its weight field is the healthiest
of the set (17/56 influential joints, biggest joint 18.4%), which is what the
geometric skinning route buys. Either the symmetry term needs a quadruped-aware
form or the giraffe stays a control; do not quietly promote it.

The earlier sampled/Python-repaired results remain historical diagnostics only.

## What to keep beside every result

Keep the source GLB, exact input image, texture cache manifest, sample
provenance, rig log, raw candidate, final candidate, score, pose-gate text,
pose image, and component-repair report (when present). Add only accepted
assets to the eye-test page, with a direct query URL and a truthful
`textured` or `flat-colour diagnostic` label.
