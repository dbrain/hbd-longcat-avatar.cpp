#!/usr/bin/env bash
# Verify the selected HIGH / MEDIUM / LOW native texture delivery as one unit.
#
# Usage:
#   verify_native_texture_delivery.sh <delivery-dir>
#
# This is intentionally read-only: it does not reserve a GPU or regenerate
# anything. It follows the authoritative texture_delivery.txt selection rather
# than guessing from filenames, then invokes the per-asset structural and
# final-atlas-recovery verifier for every selected LOD.
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?usage: verify_native_texture_delivery.sh <delivery-dir>}"
MANIFEST="$OUT/texture_delivery.txt"
[[ -d "$OUT" && -s "$MANIFEST" ]] || { echo "missing delivery directory or manifest: $OUT" >&2; exit 1; }

value() { awk -F= -v key="$1" '$1==key {print substr($0,length(key)+2); exit}' "$MANIFEST"; }
field() {
  local line="$1" key="$2"
  for word in $line; do [[ "$word" == "$key="* ]] && { printf '%s\n' "${word#*=}"; return; }; done
  return 1
}
hash_matches() {
  local label="$1" expected="$2" file="$3" actual
  [[ "$expected" =~ ^[0-9a-f]{64}$ ]] || { echo "REJECT: $label record lacks SHA-256: $MANIFEST" >&2; exit 1; }
  [[ -s "$file" ]] || { echo "REJECT: $label sidecar missing: $file" >&2; exit 1; }
  actual="$(sha256sum "$file" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "REJECT: $label hash mismatch: $file (manifest=$expected actual=$actual)" >&2; exit 1;
  }
}

SCHEMA="$(value schema_version)"
[[ "$SCHEMA" =~ ^[0-9]+$ ]] && (( SCHEMA >= 3 )) || {
  echo "REJECT: delivery manifest is not schema 3: $MANIFEST" >&2; exit 1;
}
PRODUCTION_HIGH="$(value production_texture)"
[[ -n "$PRODUCTION_HIGH" && -s "$OUT/$PRODUCTION_HIGH" ]] || {
  echo "REJECT: selected production high texture is missing: $OUT/$PRODUCTION_HIGH" >&2; exit 1;
}

SELECTED="$(value selected_production_lods)"
if [[ -n "$SELECTED" ]]; then
  HIGH_ID=high-production; MEDIUM_ID=medium-production; LOW_ID=low-production
else
  HIGH_ID=high; MEDIUM_ID=medium; LOW_ID=low
fi

for id in "$HIGH_ID" "$MEDIUM_ID" "$LOW_ID"; do
  line="$(awk -v key="lod=$id" '$1==key {print; exit}' "$MANIFEST")"
  [[ -n "$line" ]] || { echo "REJECT: manifest has no selected $id LOD record: $MANIFEST" >&2; exit 1; }
  file="$(field "$line" file)" || { echo "REJECT: $id record has no file: $MANIFEST" >&2; exit 1; }
  [[ -s "$OUT/$file" ]] || { echo "REJECT: selected $id GLB missing: $OUT/$file" >&2; exit 1; }
  hash_matches "$id GLB" "$(field "$line" sha256)" "$OUT/$file"
  hash_matches "$id atlas" "$(field "$line" atlas_sha256)" "$OUT/${file%.glb}_atlas.png"
  hash_matches "$id stage log" "$(field "$line" stage_log_sha256)" "$OUT/$file.stage-log.txt"
  hash_matches "$id texture QC" "$(field "$line" texture_qc_sha256)" "$OUT/$file.texture-qc.txt"
  status="$OUT/$file.run-status.txt"
  [[ -s "$status" ]] || { echo "REJECT: selected $id has no status: $status" >&2; exit 1; }
  if grep -q '^execution=CPU-only native re-atlas; no GPU reserved$' "$status"; then execution=cpu; else execution=gpu; fi
  "$CP/shootout/verify_native_texture_asset.sh" "$OUT/$file" --execution "$execution"
  [[ "$id" != "$HIGH_ID" || "$file" == "$PRODUCTION_HIGH" ]] || {
    echo "REJECT: production_texture=$PRODUCTION_HIGH does not match selected high=$file" >&2; exit 1;
  }
done

printf 'VERIFIED native texture delivery: %s (high=%s, medium/low selected from %s)\n' \
  "$OUT" "$PRODUCTION_HIGH" "$MANIFEST"
