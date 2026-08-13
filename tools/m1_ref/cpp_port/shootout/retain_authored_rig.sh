#!/usr/bin/env bash
# Retain a verified authored GLB rig when auto-rigging correctly rejects a
# model prediction. This is deliberately not a skeleton replacement: the
# output is byte-for-byte the supplied textured source container, accompanied
# by a fresh real-GLB articulation gate and provenance record.
#
# Usage:
#   retain_authored_rig.sh <source-rigged-textured.glb> <new-output.glb>
#                          [--allow-static-accessories]
#
# By default every mesh primitive must be skinned by the one retained skin.
# A sword, halo, or other intentional static accessory needs the explicit
# flag, leaving the exceptional policy visible in the provenance record.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:?usage: retain_authored_rig.sh <source-rigged-textured.glb> <new-output.glb> [--allow-static-accessories]}"
OUT="${2:?usage: retain_authored_rig.sh <source-rigged-textured.glb> <new-output.glb> [--allow-static-accessories]}"
ACCESSORIES="${3:-}"
[[ -f "$SRC" ]] || { echo "missing source GLB: $SRC" >&2; exit 2; }
[[ ! -e "$OUT" ]] || { echo "refusing to overwrite retained-rig output: $OUT" >&2; exit 2; }
case "$ACCESSORIES" in
  '') allow_static=0 ;;
  --allow-static-accessories) allow_static=1 ;;
  *) echo "unknown option: $ACCESSORIES" >&2; exit 2 ;;
esac
PY="${RIG_POSE_GATE_PYTHON:-/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "missing pose-gate Python: $PY" >&2; exit 2; }

# Keep this deliberately narrow so a source with multiple unrelated skins,
# missing vertex skin attributes, or no embedded material cannot be relabelled
# as a generic delivery. Accessor bounds/type checks then occur in the pose
# renderer against the copied, actual output container.
"$PY" - "$SRC" "$allow_static" <<'PY'
import json, struct, sys
path, allow_static = sys.argv[1], bool(int(sys.argv[2]))
raw = open(path, 'rb').read()
if len(raw) < 20 or raw[:4] != b'glTF':
    raise SystemExit('retain-source-rig: not a GLB v2')
json_len, json_kind = struct.unpack_from('<II', raw, 12)
if json_kind != 0x4E4F534A:
    raise SystemExit('retain-source-rig: missing JSON chunk')
doc = json.loads(raw[20:20 + json_len])
skins = doc.get('skins', [])
if len(skins) != 1 or len(skins[0].get('joints', [])) < 2:
    raise SystemExit('retain-source-rig: require exactly one nontrivial authored skin')
if not doc.get('images') or not doc.get('textures') or not doc.get('materials'):
    raise SystemExit('retain-source-rig: require an embedded textured source material')
skinned = static = 0
for node in doc.get('nodes', []):
    if 'mesh' not in node:
        continue
    for primitive in doc['meshes'][node['mesh']].get('primitives', []):
        attrs = primitive.get('attributes', {})
        if 'skin' in node:
            if not all(key in attrs for key in ('POSITION', 'JOINTS_0', 'WEIGHTS_0')) or 'indices' not in primitive:
                raise SystemExit('retain-source-rig: skinned primitive lacks POSITION/JOINTS_0/WEIGHTS_0/indices')
            if node['skin'] != 0:
                raise SystemExit('retain-source-rig: primitive refers to unexpected skin')
            skinned += 1
        else:
            static += 1
if not skinned:
    raise SystemExit('retain-source-rig: no skinned delivery primitive')
if static and not allow_static:
    raise SystemExit(f'retain-source-rig: {static} static mesh primitive(s); pass --allow-static-accessories after review')
print(f'retain-source-rig: structure PASS skin-joints={len(skins[0]["joints"])} skinned-primitives={skinned} static-primitives={static}')
PY

mkdir -p "$(dirname "$OUT")"
cp --reflink=auto -- "$SRC" "$OUT"
"$PY" "$CP/rig_pose_smoke.py" "$OUT" "${OUT%.glb}.pose-gate.png" \
  --generic-all-influential --show-skeleton --pose-gate
{
  echo 'schema_version=1'
  echo "mode=retain-authored-source-rig"
  echo "source_glb=$(readlink -f "$SRC")"
  echo "source_sha256=$(sha256sum "$SRC" | awk '{print $1}')"
  echo "output_sha256=$(sha256sum "$OUT" | awk '{print $1}')"
  echo "allow_static_accessories=$allow_static"
} >"${OUT%.glb}.retain-source-rig.txt"
echo "== retained authored rig -> $OUT =="
