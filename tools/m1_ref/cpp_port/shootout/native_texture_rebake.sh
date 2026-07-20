#!/usr/bin/env bash
# CPU-only native texture rebake from a saved texture_mesh_native PBR dump.
#
# Use after a native inference run to A/B atlas resolution or conservative cleanup without
# rerunning DINO / texture diffusion. This intentionally holds no GPU lock: it performs only
# xatlas, CPU rasterisation, and native GLB writing.
#
# Usage:
#   native_texture_rebake.sh <refined.glb> <native-dump-dir> <out.glb> [texture_rebake_native args...]
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MESH="${1:?usage: native_texture_rebake.sh <refined.glb> <native-dump-dir> <out.glb> [args...]}"
DUMP="${2:?need native PBR dump directory}"
OUT="${3:?need output glb}"
shift 3

BIN="$CP/texture_rebake_native"
[[ -x "$BIN" ]] || { echo "missing $BIN; build it first: cd $CP && ./build.sh texture_rebake_native" >&2; exit 2; }
[[ -f "$MESH" ]] || { echo "missing mesh: $MESH" >&2; exit 2; }
[[ -f "$DUMP/native_pbr_feats.npy" && -f "$DUMP/native_pbr_coords.npy" ]] || {
  echo "missing native PBR dump in $DUMP (run native_texture_run.sh with --dump-dir first)" >&2; exit 2;
}
mkdir -p "$(dirname "$OUT")"

# A rebake does not occupy the 3060, but it can still spend time in a difficult
# xatlas solve.  Give it the same inspectable, append-only phase record as the
# inference rung so the delivery manifest can say which adaptive chart route
# actually ran rather than guessing from the requested unwrap mode.
STATUS_FILE="${OUT}.run-status.txt"
STAGE_FILE="${TEX_STAGE_LOG:-${OUT}.stage-log.txt}"
export TEX_STAGE_LOG="$STAGE_FILE"
: >"$STAGE_FILE"
STARTED_AT="$(date -Is)"
START_SECONDS=$SECONDS
MESH_SHA256="$(sha256sum "$MESH" | awk '{print $1}')"
DUMP_FEATS_SHA256="$(sha256sum "$DUMP/native_pbr_feats.npy" | awk '{print $1}')"
DUMP_COORDS_SHA256="$(sha256sum "$DUMP/native_pbr_coords.npy" | awk '{print $1}')"
SOURCE_REVISION="$(git -C "$CP" rev-parse --verify HEAD 2>/dev/null || printf unknown)"

write_status() {
  local state="$1" rc="${2:-0}" stage_line=""
  shift 2
  [[ -f "$STAGE_FILE" ]] && stage_line="$(tail -n 1 "$STAGE_FILE" || true)"
  {
    printf 'launcher_state=%s\n' "$state"
    printf 'started_at=%s\nupdated_at=%s\nelapsed_seconds=%s\nexit_code=%s\n' \
      "$STARTED_AT" "$(date -Is)" "$((SECONDS-START_SECONDS))" "$rc"
    printf 'execution=CPU-only native re-atlas; no GPU reserved\n'
    printf 'mesh=%s\npbr_dump=%s\nout=%s\nstage_log=%s\n' "$MESH" "$DUMP" "$OUT" "$STAGE_FILE"
    printf 'mesh_sha256=%s\npbr_feats_sha256=%s\npbr_coords_sha256=%s\nsource_revision=%s\n' \
      "$MESH_SHA256" "$DUMP_FEATS_SHA256" "$DUMP_COORDS_SHA256" "$SOURCE_REVISION"
    [[ -z "$stage_line" ]] || printf 'native_stage=%s\n' "$stage_line"
    [[ "$state" != succeeded ]] || printf 'artifact_state=succeeded\n'
    [[ "$state" != failed ]] || printf 'artifact_state=failed\n'
    printf 'args='
    printf ' %q' "$@"
    printf '\n'
  } >"$STATUS_FILE"
}

write_status running 0 "$@"
echo "== native CPU texture rebake: $(basename "$MESH") -> $OUT =="
"$BIN" --mesh "$MESH" --pbr-dir "$DUMP" --out "$OUT" "$@" &
CHILD_PID=$!
while kill -0 "$CHILD_PID" 2>/dev/null; do
  sleep 5
  write_status running 0 "$@"
done
set +e
wait "$CHILD_PID"
CHILD_RC=$?
set -e
if (( CHILD_RC != 0 )); then
  write_status failed "$CHILD_RC" "$@"
  exit "$CHILD_RC"
fi
if [[ ! -s "$OUT" ]]; then
  echo "rebake produced no GLB: $OUT" >&2
  write_status failed 1 "$@"
  exit 1
fi
write_status succeeded 0 "$@"
