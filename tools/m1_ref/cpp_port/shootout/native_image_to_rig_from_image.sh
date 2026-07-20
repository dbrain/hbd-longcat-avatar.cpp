#!/usr/bin/env bash
# Model-agnostic clean image -> native textured LODs -> Hymotion-ready rig.
#
# This is the entrypoint for a NEW subject.  It creates the costly clean/refined geometry cache
# once, keyed by the actual source image bytes, then hands that exact mesh to native_image_to_rig.sh.
# The latter produces native high/medium/low texture assets and only publishes a rig that passes its
# structural gate.  Every inference invocation is pinned to physical PCI GPU 0 (the RTX 3060).
#
# Usage:
#   native_image_to_rig_from_image.sh <RGBA cutout, black matte, or raw photo> <out-dir> [label]
#
# Optional:
#   NATIVE_HIGH_RESOLUTION=1024 NATIVE_HIGH_ATLAS=4096  hero-detail high A/B
#   (the generic runner always retains native_high_texture_dump for CPU LOD rebakes)
#   IMAGE_TO_RIG_REFRESH=1                              recompute even if this image's cache exists
#   IMAGE_TO_RIG_PROJECT=1                               create observed-view projection A/B
#   IMAGE_TO_RIG_TEX_BACK=/abs/back.png
#   IMAGE_TO_RIG_TEX_VIEWS='90=/abs/right.png -90=/abs/left.png'
#   IMAGE_TO_RIG_TEX_VIEWS_FILE=/abs/turnaround.tsv       robust 1--8 view manifest (yaw<TAB>absolute path)
#                                                          yaw 0/180 replace the front/back source; all other
#                                                          yaws are passed as --tex-view. Blank/# lines are ignored.
#   IMAGE_TO_RIG_INPUT_MODE=auto|matte                   auto: preserve a cutout/matte or RMBG a raw photo
#   MATTING_URL=http://localhost:18898                   native RMBG-2.0 service (raw-photo input only)
#   IMAGE_TO_RIG_PREPARE_ONLY=1                          emit/audit input.png, without geometry inference
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:?usage: native_image_to_rig_from_image.sh <cutout, matte, or raw image> <out-dir> [label]}"
OUT="${2:?need output directory}"
LABEL="${3:-$(basename "$OUT")}" 
OUT_ROOT="${IMAGE_TO_RIG_OUT_ROOT:-/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig}"
export IMAGE_TO_RIG_OUT_ROOT="$OUT_ROOT"

[[ -f "$IMAGE" ]] || { echo "missing image: $IMAGE" >&2; exit 2; }
for bin in image_to_rig mesh_topo make_matte; do
  [[ -x "$CP/$bin" ]] || { echo "missing executable: $bin (build it first)" >&2; exit 2; }
done
[[ -x "$CP/shootout/native_image_to_rig.sh" ]] || { echo "missing native image-to-rig driver" >&2; exit 2; }

# A cache name tied to file content prevents a stale Miku/Soldier-style cache from silently being
# reused for a different source image. The model-facing matte is recorded separately, so the eye
# page and optional projection A/B consume precisely the frame that made the geometry.
IMAGE_HASH="$(sha256sum "$IMAGE" | awk '{print substr($1,1,16)}')"
mkdir -p "$OUT" "$OUT_ROOT"
ln -sfn "$IMAGE" "$OUT/source_image"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export REMESH_CLOSE_R="${REMESH_CLOSE_R:-3}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: PCI GPU 0 is '$GPU_NAME', expected RTX 3060" >&2; exit 1; }
GPU_3060_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 | head -1)"
MATTING_URL="${MATTING_URL:-http://localhost:18898}"
INPUT_MODE="${IMAGE_TO_RIG_INPUT_MODE:-auto}"
[[ "$INPUT_MODE" == auto || "$INPUT_MODE" == matte ]] || { echo "IMAGE_TO_RIG_INPUT_MODE must be auto or matte" >&2; exit 2; }

# All geometry and texture stages use exactly one image frame. Preserve a supplied RGBA cutout,
# preserve a trusted black matte, and only ask RMBG to process a genuinely opaque photo. The service
# request names the physical 3060 UUID and takes the shared lock; preprocessing never spills onto
# the busy 5060.
INPUT_KIND="$("$CP/make_matte" --inspect-input "$IMAGE" | awk -F= '$1=="input_kind" {print $2}')"
[[ -n "$INPUT_KIND" ]] || { echo "could not classify input image: $IMAGE" >&2; exit 1; }
PIPELINE_IMAGE="$IMAGE"
case "$INPUT_MODE:$INPUT_KIND" in
  matte:*)
    echo "== $LABEL: use caller-supplied matte frame ($INPUT_KIND) =="
    ;;
  auto:rgba-cutout)
    PIPELINE_IMAGE="$OUT/input_matte.png"
    echo "== $LABEL: make deterministic matte from RGBA cutout =="
    "$CP/make_matte" "$IMAGE" "$PIPELINE_IMAGE"
    ;;
  auto:black-matte)
    echo "== $LABEL: preserve black matte frame =="
    ;;
  auto:opaque)
    CUTOUT="$OUT/source_cutout_rgba.png"
    PIPELINE_IMAGE="$OUT/input_matte.png"
    LOCK="$OUT_ROOT/.3060-image-to-rig.lock"
    echo "== $LABEL: RMBG raw photo on PCI GPU 0 / RTX 3060 =="
    (
      exec 9>"$LOCK"
      flock -n 9 || { echo "another image-to-rig job owns the 3060 lock" >&2; exit 75; }
      code="$(curl -sS -X POST "$MATTING_URL/remove?bg_mode=alpha" -F "images=@$IMAGE" -F "gpu=$GPU_3060_UUID" -o "$CUTOUT" -w "%{http_code}")"
      [[ "$code" == 200 ]] || { echo "matting service HTTP $code" >&2; exit 1; }
    )
    [[ -s "$CUTOUT" ]] || { echo "matting service did not produce an RGBA cutout" >&2; exit 1; }
    "$CP/make_matte" "$CUTOUT" "$PIPELINE_IMAGE"
    ;;
esac
[[ -f "$PIPELINE_IMAGE" ]] || { echo "missing model-facing matte: $PIPELINE_IMAGE" >&2; exit 1; }
MATTE_HASH="$(sha256sum "$PIPELINE_IMAGE" | awk '{print substr($1,1,16)}')"
# The cache key includes both the uploaded bytes and the exact model-facing matte bytes. This prevents
# a changed matting service, alpha cutout, or framing recipe from silently reusing geometry made from a
# different image frame.
CACHE="$OUT/cache_${IMAGE_HASH}_${MATTE_HASH}_n16384_seal3"
ln -sfn "$PIPELINE_IMAGE" "$OUT/input.png"
printf 'source_image=%s\nsource_sha256=%s\ninput_kind=%s\ninput_mode=%s\nmodel_image=%s\nmodel_image_sha256=%s\n' \
  "$IMAGE" "$IMAGE_HASH" "$INPUT_KIND" "$INPUT_MODE" "$PIPELINE_IMAGE" "$MATTE_HASH" >"$OUT/preprocess_source.txt"
if [[ "${IMAGE_TO_RIG_PREPARE_ONLY:-0}" != 0 ]]; then
  echo "== DONE: prepared model input $PIPELINE_IMAGE (no geometry/texture inference) =="
  exit 0
fi

# Keep this geometry cache deliberately separate from the final native texture outputs.  `image_to_rig`
# is used only for its native MoGe + geometry + UltraShape stages; its legacy texture/rig result is never
# promoted over the native textured assets produced below.
if [[ "${IMAGE_TO_RIG_REFRESH:-0}" != 0 || ! -f "$CACHE/refined.glb" || ! -f "$CACHE/pbr_feats.bin" || ! -f "$CACHE/pbr_coords.bin" ]]; then
  LOCK="$OUT_ROOT/.3060-image-to-rig.lock"
  mkdir -p "$CACHE" "$OUT_ROOT"
  echo "== $LABEL: create clean geometry cache ${CACHE##*/} on the RTX 3060 =="
  (
    exec 9>"$LOCK"
    flock -n 9 || { echo "another image-to-rig job owns the 3060 lock" >&2; exit 75; }
    "$CP/image_to_rig" --model /home/dbrain/models/3d/geo --image "$PIPELINE_IMAGE" --moge \
      --no-quad --us-latents 16384 --tex-dit cross --tex-volume-direct --tex-fallback-r 8 \
      --no-rig --stage-dir "$CACHE" --out "$CACHE/legacy_geometry_texture_ab.glb"
  )
else
  echo "== $LABEL: reuse image-keyed clean geometry cache ${CACHE##*/} =="
fi

[[ -f "$CACHE/refined.glb" ]] || { echo "geometry cache did not produce refined.glb" >&2; exit 1; }
NATIVE_RIG=1
if ! "$CP/shootout/native_image_to_rig.sh" "$CACHE/refined.glb" "$PIPELINE_IMAGE" "$OUT" "$LABEL"; then
  NATIVE_RIG=0
  # The clean native texture LODs may be valid even when the learned skeleton is not. Do not throw
  # them away or publish an anonymous/malformed rig: use the older validated rig path only as a
  # named, explicit animation fallback. This is how Gilly remains usable while its one-leg native
  # beam sample is rejected by the same structural falsifier that protects every other subject.
  for f in "$OUT/native_high_textured.glb" "$OUT/native_medium_textured.glb" "$OUT/native_low_textured.glb"; do
    [[ -f "$f" ]] || { echo "native pipeline failed before producing clean LODs; no rig fallback" >&2; exit 1; }
  done
  LOCK="$OUT_ROOT/.3060-image-to-rig.lock"
  mixamo_core_ok() {
    local file="$1" n
    local core=(Hips LeftUpLeg RightUpLeg Spine LeftLeg RightLeg Spine1 LeftFoot RightFoot Spine2
                LeftToeBase RightToeBase Neck LeftShoulder RightShoulder Head LeftArm RightArm
                LeftForeArm RightForeArm LeftHand RightHand)
    for n in "${core[@]}"; do grep -a -q "mixamorig:$n" "$file" || return 1; done
  }
  legacy_rig_ok() {
    local file="$1" report fan total
    report="$("$CP/rig_score" "$file" 55 2>&1 || true)"; printf '%s\n' "$report"
    [[ "$report" =~ maxfan=([0-9]+) ]] || return 1; fan="${BASH_REMATCH[1]}"
    [[ "$report" =~ TOTAL=([0-9.]+) ]] || return 1; total="${BASH_REMATCH[1]}"
    (( fan <= 7 )) && awk "BEGIN { exit !($total >= 0.50) }" && mixamo_core_ok "$file"
  }
  for level in high medium low; do
    case "$level" in
      high) faces=300000; tex=2048;; medium) faces=150000; tex=2048;; low) faces=50000; tex=2048;;
    esac
    candidate="$OUT/legacy_rig_fallback_${level}.glb"
    echo "== $LABEL: native rig rejected; try explicit legacy-rig fallback $level =="
    if (
      exec 9>"$LOCK"; flock -n 9 || exit 75
      "$CP/image_to_rig" --model /home/dbrain/models/3d/geo --image "$PIPELINE_IMAGE" --moge \
        --from-refined "$CACHE" --stage-dir "$OUT" --decimate "$faces" --texsize "$tex" \
        --bone-facing +z --out "$candidate"
    ) && legacy_rig_ok "$candidate"; then
      cp -f "$candidate" "$OUT/hymotion_rigged.glb"
      LEGACY_RIG_LEVEL="$level"
      break
    fi
  done
  [[ -n "${LEGACY_RIG_LEVEL:-}" ]] || { echo "no native or legacy rig candidate passed the structural gate" >&2; exit 1; }
  # native_image_to_rig.sh truthfully records that its own learned skeleton was rejected. Preserve that
  # fact and also record the final generic-run outcome, rather than letting the top-level delivery
  # manifest imply that no Hymotion hand-off exists after this explicitly named fallback succeeds.
  [[ -f "$OUT/texture_delivery.txt" ]] || { echo "native texture delivery manifest missing after rig rejection" >&2; exit 1; }
  {
    printf 'final_generic_rig_state=explicit-legacy-fallback\n'
    printf 'final_generic_rig_level=%s\n' "$LEGACY_RIG_LEVEL"
    printf 'final_hymotion_rigged=hymotion_rigged.glb sha256=%s\n' "$(sha256sum "$OUT/hymotion_rigged.glb" | awk '{print $1}')"
  } >>"$OUT/texture_delivery.txt"
  cat >"$OUT/stages.json" <<EOF
{"subject":"$LABEL · native textured run with explicit legacy-rig fallback","input":"input.png","stages":[
 {"file":"hymotion_rigged.glb","label":"Hymotion-ready · legacy rig fallback ($LEGACY_RIG_LEVEL)","note":"all native skeleton candidates failed structural naming; clean native texture LODs remain the production texture assets"},
 {"file":"native_high_textured.glb","label":"HIGH · native textured","note":"300k target faces · native Trellis generated texture"},
 {"file":"native_medium_textured.glb","label":"MEDIUM · native textured","note":"150k target faces · native Trellis generated texture"},
 {"file":"native_low_textured.glb","label":"LOW · native textured","note":"50k target faces · native Trellis generated texture"}
]}
EOF
fi

# Projection is deliberately an A/B, never a replacement for native_high_textured.glb: the native
# generated material remains responsible for every unobserved texel.  The cache's PBR is used only
# to build an independently inspectable observed-view hybrid when the caller has real extra views.
#
# IMAGE_TO_RIG_TEX_VIEWS is retained for shell convenience, but it cannot represent paths containing
# whitespace and is awkward for a real 4--8 camera turnaround.  The TSV manifest is the production
# contract: one canonical camera yaw and one absolute source path per line.  Normalising yaws before
# invoking image_to_rig prevents an accidental `180`/`-180` duplicate camera and makes the recorded
# evidence unambiguous.
PROJ_ARGS=()
PROJECT="${IMAGE_TO_RIG_PROJECT:-0}"
declare -A PROJ_PATH_BY_YAW=()
declare -A PROJ_ORIGIN_BY_YAW=()
declare -A PROJ_LABEL_BY_YAW=()
PROJ_COUNT=0

normalise_yaw() {
  awk -v yaw="$1" 'BEGIN {
    if (yaw !~ /^[-+]?[0-9]*([.][0-9]+)?$/) exit 1
    y = yaw + 0
    y = y - 360 * int(y / 360)
    if (y < 0) y += 360
    if (y >= 360) y -= 360
    if (y == 0) y = 0
    printf "%.6g", y
  }'
}

add_projection_view() {
  local raw_yaw="$1" path="$2" origin="$3" yaw
  yaw="$(normalise_yaw "$raw_yaw")" || { echo "bad projection yaw '$raw_yaw' from $origin" >&2; exit 2; }
  [[ "$path" = /* ]] || { echo "projection path must be absolute ($origin): $path" >&2; exit 2; }
  [[ -f "$path" ]] || { echo "missing projection image ($origin): $path" >&2; exit 2; }
  [[ -z "${PROJ_PATH_BY_YAW[$yaw]:-}" ]] || {
    echo "duplicate projection yaw $yaw: ${PROJ_ORIGIN_BY_YAW[$yaw]} and $origin" >&2; exit 2;
  }
  (( PROJ_COUNT < 8 )) || { echo "projection supports at most 8 cameras (including front/back)" >&2; exit 2; }
  PROJ_PATH_BY_YAW[$yaw]="$path"
  PROJ_ORIGIN_BY_YAW[$yaw]="$origin"
  if [[ "$yaw" == 0 ]]; then
    PROJ_LABEL_BY_YAW[$yaw]=front
  elif [[ "$yaw" == 180 ]]; then
    PROJ_LABEL_BY_YAW[$yaw]=back
  else
    PROJ_LABEL_BY_YAW[$yaw]="yaw$yaw"
  fi
  ((PROJ_COUNT += 1))
}

if [[ -n "${IMAGE_TO_RIG_TEX_FRONT:-}" ]]; then
  add_projection_view 0 "$IMAGE_TO_RIG_TEX_FRONT" IMAGE_TO_RIG_TEX_FRONT; PROJECT=1
fi
if [[ -n "${IMAGE_TO_RIG_TEX_BACK:-}" ]]; then
  add_projection_view 180 "$IMAGE_TO_RIG_TEX_BACK" IMAGE_TO_RIG_TEX_BACK; PROJECT=1
fi
if [[ -n "${IMAGE_TO_RIG_TEX_VIEWS:-}" ]]; then
  for view in $IMAGE_TO_RIG_TEX_VIEWS; do
    yaw="${view%%=*}"; path="${view#*=}"
    [[ "$yaw" != "$view" ]] || { echo "bad IMAGE_TO_RIG_TEX_VIEWS '$view' (expected yaw=/absolute/image.png)" >&2; exit 2; }
    add_projection_view "$yaw" "$path" IMAGE_TO_RIG_TEX_VIEWS; PROJECT=1
  done
fi
if [[ -n "${IMAGE_TO_RIG_TEX_VIEWS_FILE:-}" ]]; then
  VIEW_MANIFEST="$IMAGE_TO_RIG_TEX_VIEWS_FILE"
  [[ "$VIEW_MANIFEST" = /* && -f "$VIEW_MANIFEST" ]] || {
    echo "IMAGE_TO_RIG_TEX_VIEWS_FILE must name an existing absolute TSV manifest" >&2; exit 2;
  }
  line_no=0
  while IFS= read -r line || [[ -n "$line" ]]; do
    ((line_no += 1))
    [[ -z "${line//[[:space:]]/}" || "$line" == \#* ]] && continue
    IFS=$'\t' read -r yaw path extra <<<"$line"
    [[ -n "$yaw" && -n "$path" && -z "${extra:-}" ]] || {
      echo "bad $VIEW_MANIFEST:$line_no (expected yaw<TAB>/absolute/path.png)" >&2; exit 2;
    }
    add_projection_view "$yaw" "$path" "$VIEW_MANIFEST:$line_no"
    PROJECT=1
  done <"$VIEW_MANIFEST"
fi

# The normalised map is the one source of truth. The front defaults to the model-facing matte if
# no explicit yaw=0 source was supplied; it still counts as a camera for the eight-view bound.
if [[ -z "${PROJ_PATH_BY_YAW[0]:-}" ]]; then
  PROJ_PATH_BY_YAW[0]="$PIPELINE_IMAGE"
  PROJ_ORIGIN_BY_YAW[0]=model_input
  PROJ_LABEL_BY_YAW[0]=front
fi
TOTAL_PROJ_CAMERAS="${#PROJ_PATH_BY_YAW[@]}"
(( TOTAL_PROJ_CAMERAS <= 8 )) || { echo "projection supports at most 8 cameras including default front" >&2; exit 2; }
for yaw in $(printf '%s\n' "${!PROJ_PATH_BY_YAW[@]}" | LC_ALL=C sort -n); do
  path="${PROJ_PATH_BY_YAW[$yaw]}"
  case "$yaw" in
    0) PROJ_ARGS+=(--tex-front "$path");;
    180) PROJ_ARGS+=(--tex-back "$path");;
    *) PROJ_ARGS+=(--tex-view "$yaw" "$path");;
  esac
done
if [[ "$PROJECT" != 0 ]]; then
  LOCK="$OUT_ROOT/.3060-image-to-rig.lock"
  echo "== $LABEL: observed-view projection A/B (native texture remains default) =="
  (
    exec 9>"$LOCK"
    flock -n 9 || { echo "another image-to-rig job owns the 3060 lock" >&2; exit 75; }
    "$CP/image_to_rig" --model /home/dbrain/models/3d/geo --image "$PIPELINE_IMAGE" --moge \
      --from-refined "$CACHE" --stage-dir "$OUT" --decimate 300000 --texsize 2048 --no-rig \
      --tex-project-overlay "${PROJ_ARGS[@]}" --out "$OUT/high_hybrid_projected.glb"
  )
  {
    printf 'mode=observed-view hybrid A/B; native_high_textured.glb remains production default\n'
    printf 'camera_count=%s\n' "$TOTAL_PROJ_CAMERAS"
    printf 'blend=real observed pixels in linear light; z-buffer + eroded-subject-mask reject; native generated base retained for unobserved texels\n'
    printf 'manifest=%s\n' "${IMAGE_TO_RIG_TEX_VIEWS_FILE:-none}"
    for yaw in $(printf '%s\n' "${!PROJ_PATH_BY_YAW[@]}" | LC_ALL=C sort -n); do
      path="${PROJ_PATH_BY_YAW[$yaw]}"
      printf 'view yaw=%s label=%s path=%s sha256=%s origin=%s\n' \
        "$yaw" "${PROJ_LABEL_BY_YAW[$yaw]}" "$path" "$(sha256sum "$path" | awk '{print $1}')" \
        "${PROJ_ORIGIN_BY_YAW[$yaw]}"
    done
  } >"$OUT/projection_source.txt"
fi

cat >"$OUT/runbook_source.txt" <<EOF
source_image=$IMAGE
source_sha256=$IMAGE_HASH
model_image=$PIPELINE_IMAGE
input_kind=$INPUT_KIND
input_mode=$INPUT_MODE
geometry_cache=$CACHE
geometry_recipe=image_to_rig --moge --no-quad --us-latents 16384 --tex-dit cross --tex-volume-direct --tex-fallback-r 8 --no-rig
texture_recipe=native high Trellis material + CPU medium/low rebakes from native_high_texture_dump + structural rig gate
texture_delivery_manifest=$OUT/texture_delivery.txt
rig_mode=$([[ "$NATIVE_RIG" == 1 ]] && echo native || echo explicit-legacy-fallback)
gpu=PCI GPU 0 / RTX 3060 only
EOF
echo "== DONE: $OUT =="
