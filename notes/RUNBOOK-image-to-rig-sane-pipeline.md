# Image → textured asset → generic rig: clean end-to-end runbook

This is the authoritative operating procedure for a **new** image-to-rig
delivery. It is deliberately fail-closed: a GLB is publishable only when its
texture provenance, skeleton structure, and real glTF skin deformation all
pass. Do not replace a rejected asset with a nicer-looking diagnostic.

## Deliverables and current eye test

- Clean textured Miku V5 delivery:
  `runbook_image_to_rig/miku_p3sam_attachment_recovery_e2e_v5_20260723/attachment_recovery_experimental.glb`
- Clean cache-first textured generic Gilly delivery:
  `runbook_image_to_rig/gilly_modelready_cachefirst_generic_e2e_20260723/generic_rigged_branchsafe_verified.glb`
- Eye-test page:
  `http://10.0.0.208:8077/inline-3d/runbook.html`
- Direct clean-texture review:
  `http://10.0.0.208:8077/inline-3d/runbook.html?subject=miku_p3sam_attachment_recovery_e2e_v5_20260723&artifact=attachment_recovery_experimental.glb`

The page may contain flat-colour rig controls. Those prove skeleton and skin
behaviour only; they are never evidence of a texture delivery.

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

The deterministic decoder is always tried first. If it rejects both learned
and geometric deterministic paths, the generic wrapper may use the bounded,
reproducible upstream sampled fallback (`seed=0`, `beams=10`) with the
corrected parent+child branch grammar. This is not a seed lottery.

Every generic delivery must satisfy all of the following:

- one rooted tree, size-aware fan limit `max(8, ceil(J/5))`, rig score >= 0.50;
- complete stable generic namespace;
- actual written glTF LBS audit over every materially weighted non-root joint;
- visible motion and worst whole-tree/component p999 stretch <= 6.0x;
- texture/provenance gate when a texture is claimed.

Run the independent final gate explicitly when inspecting an output:

```sh
$PY rig_pose_smoke.py artifact.glb artifact.pose-gate.png \
  --generic-all-influential --show-skeleton --pose-gate
./rig_score artifact.glb 55
```

## Bounded generic skin repair

`RIG_COMPONENT_BRANCH_REPAIR=1` is the default only after a raw generic GLB
fails its real pose gate. It retains
`*.raw-before-component-repair.glb`, produces `*.component-repair.json`, and
then repeats the unchanged final gate.

It can rigidly attach only:

1. a disconnected component below 15% of faces whose material support spans a
   branched skeleton subtree of diameter >= 5; or
2. after that repair, a detached component below 1% of faces whose root plus
   one nearest material joint demonstrably exceeds the 6.0x component gate
   under a real 45-degree rotation.
3. after both of those, an authored raw-index-connected piece below 10% of
   faces, but only if the unchanged all-influential audit proves its own
   45-degree LBS p999 exceeds 6.0x and it contains an over-limit edge. It is
   rigidly attached to the nearest existing rest joint.

The aggregate repaired surface is capped at 40%. The report records component
faces, support joints, anchor, thresholds, and the triggering stretch. Raw
index topology is deliberately not used by the publication gate: the gate
position-welds UV splits. It is only bounded provenance for a demonstrated
spike in a separately authored touching piece. The repair does not solve
secondary motion; repaired attachments are intentionally rigid until such a
stage exists.

## Accepted controls

| Control | Result |
| --- | --- |
| Clean textured Gilly J55 | generic namespace; whole-tree/component worst 3.032x; clean PBR atlas |
| Miku V5 P3-SAM recovery | clean native texture; named arm smoke 1.602x; secondary attachment recovered |
| Winged bird J42 | 21 influential joints; component worst 2.850x |
| Fairy J64 | winged humanoid; component worst 4.314x |
| Winged imp J54 | articulated wing control; component worst 1.760x (flat-colour source diagnostic) |
| Moth sampled J36 | deterministic tree rejected; fixed upstream seed-0 fallback passed 24 influential joints, component worst 4.988x |
| FallenAngel J41 | 32 influential joints; component worst 5.467x |
| Tira sampled J53 | textured tail-bearing control; branch repair plus one proven 168-face detached root/joint repair; final component worst 3.045x |
| AngelRiggy sampled J30 | raw skin rejected at 20.842x; seven bounded branch repairs pass at 1.665x (flat-colour diagnostic because source texture is external) |
| Textured giraffe sampled J49 | quadruped control; raw skin rejected at 8.341x; two proven raw-index pose-spike pieces (260/21,636 faces) repaired; all 14 material joints pass, component worst 3.413x |

## Quadruped control: giraffe

Native deterministic decoding still produces J56/maxfan30 and is rejected. A
fixed seed-0 upstream sample produced a structurally valid J49 tree. Its raw
learned skin was rejected at 8.341x, not promoted from its plausible rest pose.
The bounded repair found only two independently index-connected pieces with
real LBS spike evidence (156 and 104 faces), attached them to their nearest
existing joint, and reran the unchanged all-joint gate at 3.413x. The accepted
textured delivery is
`fixtures/skintokens_giraffe_official_fallback_seed0_v2_rawtopologyrepair_20260723/generic_rigged.glb`.
The pose image visually confirms a coherent head and legs under the selected
articulation; a small authored rear accessory remains visibly separate rather
than being misrepresented as anatomical skin.

## What to keep beside every result

Keep the source GLB, exact input image, texture cache manifest, sample
provenance, rig log, raw candidate, final candidate, score, pose-gate text,
pose image, and component-repair report (when present). Add only accepted
assets to the eye-test page, with a direct query URL and a truthful
`textured` or `flat-colour diagnostic` label.
