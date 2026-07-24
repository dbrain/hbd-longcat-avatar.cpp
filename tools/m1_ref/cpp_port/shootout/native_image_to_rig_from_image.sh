#!/usr/bin/env bash
# Model-agnostic clean image -> native textured LODs -> profile-validated rig.
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
#   NATIVE_HERO_FACES=0 NATIVE_HERO_ATLAS=8192          full refined-mesh hero asset (0 = no decimation)
#   NATIVE_HERO_UNWRAP=production|reference             bounded default or explicit chart-quality A/B
#   NATIVE_HIGH_RESOLUTION=1024 NATIVE_HIGH_ATLAS=4096  300k rig/delivery high tier A/B
#   (the generic runner retains the Pixal PBR cache for all CPU LOD rebakes)
#   IMAGE_TO_RIG_REFRESH=1                              recompute even if this image's cache exists
#   IMAGE_TO_RIG_US_STEPS=50|80                          explicit UltraShape geometry-detail recipe (80 is a labelled face/detail A/B)
#   IMAGE_TO_RIG_GEO_RESOLUTION=1024|1536                Pixal3D seed lattice; 1536 is the true high-geometry A/B
#   IMAGE_TO_RIG_SEED=N                                  Pixal diffusion-noise seed (default 42); non-default is cache-keyed
#   IMAGE_TO_RIG_GEOMETRY_CAMERA=moge|default             geometry camera contract; default-camera is an explicit historical A/B
#   IMAGE_TO_RIG_FIXED_CAMERA=angle/dist/scale             exact camera replay; overrides the camera estimator and is cache-keyed
#   IMAGE_TO_RIG_SKIP_ULTRASHAPE=1                       stop after the Pixal3D seed (coarse clay review)
#   IMAGE_TO_RIG_GEOMETRY_REVIEW_ONLY=1                  publish clay-stage aliases/gate, then stop before texture or rig
#   IMAGE_TO_RIG_CLAY_VERDICT=approved|rejected           record the visual clay decision for this exact cache;
#                                                          default is pending, which stops after clay review
#   IMAGE_TO_RIG_PROJECT=1                               create native-base observed-view texture A/B (CPU-only)
#   IMAGE_TO_RIG_TEX_BACK=/abs/back.png
#   IMAGE_TO_RIG_TEX_VIEWS='90=/abs/right.png -90=/abs/left.png'
#   IMAGE_TO_RIG_TEX_VIEWS_FILE=/abs/turnaround.tsv       robust 1--8 view manifest (yaw<TAB>absolute path)
#                                                          yaw 0/180 replace the front/back source; all other
#                                                          yaws are passed as --tex-view. Blank/# lines are ignored.
#   IMAGE_TO_RIG_PROJECT_PROMOTE=1                         validate a consistent 4--8-view native-base candidate;
#                                                          it remains an A/B pending visual promotion
#   IMAGE_TO_RIG_INPUT_MODE=auto|matte                   auto: preserve a cutout/matte or RMBG a raw photo
#   MATTING_URL=http://localhost:18898                   native RMBG-2.0 service (raw-photo input only)
#   IMAGE_TO_RIG_PREPARE_ONLY=1                          emit/audit input.png, without geometry inference
#   RIG_PROFILE=humanoid|generic                          semantic Mixamo avatar gate, or stable generic creature gate
set -euo pipefail

CP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${1:?usage: native_image_to_rig_from_image.sh <cutout, matte, or raw image> <out-dir> [label]}"
OUT="${2:?need output directory}"
LABEL="${3:-$(basename "$OUT")}" 
OUT_ROOT="${IMAGE_TO_RIG_OUT_ROOT:-/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig}"
ROOT_MIN_FREE_GIB="${IMAGE_TO_RIG_MIN_ROOT_FREE_GIB:-20}"
[[ "$ROOT_MIN_FREE_GIB" =~ ^[0-9]+$ ]] || { echo "IMAGE_TO_RIG_MIN_ROOT_FREE_GIB must be a non-negative integer" >&2; exit 2; }
root_free_kib="$(df -Pk / | awk 'NR==2 {print $4}')"
(( root_free_kib >= ROOT_MIN_FREE_GIB * 1024 * 1024 )) || {
  echo "refusing image-to-rig run: / has only $(( root_free_kib / 1024 / 1024 )) GiB free; require ${ROOT_MIN_FREE_GIB} GiB. Keep Docker builder cache and all generated artifacts off /." >&2
  exit 75
}
# The canonical native Pixal bundle lives on the local NVMe.  This temporary F16
# control bundle retains high-resolution shape/texture flow weights while the clean
# geometry/texture path is being verified.  The compact Q8 bundle remains the intended
# VRAM/performance production target and must be rerun through the same approved recipe
# before promotion.  Keep the bundle path in provenance so the comparison is explicit.
PIXAL3D_MODEL_DIR="${IMAGE_TO_RIG_MODEL_DIR:-/home/dbrain/models/3d/geo_f16_native}"
RIG_PROFILE="${RIG_PROFILE:-humanoid}"
export IMAGE_TO_RIG_OUT_ROOT="$OUT_ROOT"

case "$RIG_PROFILE" in
  humanoid|generic) ;;
  *) echo "RIG_PROFILE must be humanoid or generic (got $RIG_PROFILE)" >&2; exit 2 ;;
esac

[[ -f "$IMAGE" ]] || { echo "missing image: $IMAGE" >&2; exit 2; }
for f in dinov3 naf ss_flow ss_dec slat_flow_512 slat_flow_1024 shape_dec slat_flow_imgshape2tex_1024 tex_dec; do
  [[ -s "$PIXAL3D_MODEL_DIR/$f.gguf" ]] || { echo "missing required Pixal checkpoint: $PIXAL3D_MODEL_DIR/$f.gguf" >&2; exit 2; }
done
for bin in image_to_rig mesh_topo make_matte; do
  [[ -x "$CP/$bin" ]] || { echo "missing executable: $bin (build it first)" >&2; exit 2; }
done
for bin in native_image_to_rig.sh native_observed_atlas_project.sh verify_native_observed_projection.sh; do
  [[ -x "$CP/shootout/$bin" ]] || { echo "missing native image-to-rig component: $bin" >&2; exit 2; }
done
# Detached production runs write their C++ progress to a file.  Force line buffering so stage logs say
# what the 3060 is actually doing instead of appearing frozen until libc's file buffer fills.
IMAGE_TO_RIG_NICE_LEVEL="${IMAGE_TO_RIG_NICE_LEVEL:-10}"
[[ "$IMAGE_TO_RIG_NICE_LEVEL" =~ ^([0-9]|1[0-9])$ ]] || {
  echo "IMAGE_TO_RIG_NICE_LEVEL must be an integer in [0,19]" >&2; exit 2;
}
# Sparse-grid bookkeeping and the UltraShape cleanup have substantial host work even while the
# model is on the reserved 3060. Keep every production native child below interactive workloads.
IMAGE_TO_RIG_CMD=(nice -n "$IMAGE_TO_RIG_NICE_LEVEL" ionice -c 2 -n 7 stdbuf -oL -eL "$CP/image_to_rig")

# A cache name tied to file content prevents a stale Miku/Soldier-style cache from silently being
# reused for a different source image. The model-facing matte is recorded separately, so the eye
# page and optional projection A/B consume precisely the frame that made the geometry.
IMAGE_HASH="$(sha256sum "$IMAGE" | awk '{print substr($1,1,16)}')"
mkdir -p "$OUT" "$OUT_ROOT"
ln -sfn "$IMAGE" "$OUT/source_image"

# A detached run must leave one small, atomic, human-readable answer to
# "which stage is this actually in?".  The launcher log remains the detailed
# evidence; this sidecar makes an interrupted process distinguishable from a
# completed cache or a silent texture failure without guessing from file dates.
PIPELINE_STATUS="$OUT/pipeline-status.txt"
PIPELINE_STAGE=initialising
PIPELINE_COMPLETED=0
write_pipeline_status() {
  local state="$1" detail="${2:-}"
  {
    printf 'schema_version=1\n'
    printf 'state=%s\n' "$state"
    printf 'stage=%s\n' "$PIPELINE_STAGE"
    printf 'detail=%s\n' "$detail"
    printf 'updated_at=%s\n' "$(date -Is)"
    printf 'label=%s\n' "$LABEL"
    printf 'source_image=%s\n' "$IMAGE"
    printf 'source_sha256=%s\n' "$IMAGE_HASH"
    printf 'model_image=%s\n' "${PIPELINE_IMAGE:-pending}"
    printf 'geometry_cache=%s\n' "${CACHE:-pending}"
    printf 'geometry_recipe=%s\n' "${GEOMETRY_RECIPE:-pending}"
    printf 'pixal_seed=%s\n' "${IMAGE_TO_RIG_SEED:-42}"
    printf 'rig_profile=%s\n' "$RIG_PROFILE"
    printf 'gpu=PCI GPU 0 / %s\n' "${GPU_NAME:-pending}"
    printf 'scheduler=nice=%s; ionice=best-effort:7\n' "${IMAGE_TO_RIG_NICE_LEVEL:-10}"
    printf 'pixal_model_dir=%s\n' "$PIXAL3D_MODEL_DIR"
    printf '%s\n' 'pixal_checkpoint_bundle=local-nvme-geo-f16-native plus F32 .npy M3b/M6 quality override'
  } >"$PIPELINE_STATUS.tmp"
  mv -f "$PIPELINE_STATUS.tmp" "$PIPELINE_STATUS"
}
pipeline_on_exit() {
  local rc=$?
  trap - EXIT
  if (( PIPELINE_COMPLETED == 0 )); then
    write_pipeline_status failed "exit_code=$rc" || true
  fi
  return "$rc"
}
trap pipeline_on_exit EXIT

GPU_3060_UUID="${IMAGE_TO_RIG_GPU_3060_UUID:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {uuid=$1} END {print uuid}')}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_3060_UUID" 2>/dev/null | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: '$GPU_3060_UUID' is '$GPU_NAME', expected the reserved RTX 3060" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_3060_UUID"
# The frozen Python-boundary audit found that both high-resolution shortcuts
# materially change the M3b geometry sample.  Do not let a caller's shell
# environment turn either one back on and then reuse/publish it as "clean".
# PIXAL3D_FAST is presence-based in the graph (`getenv(...) != nullptr`), so
# even PIXAL3D_FAST=0 turns the lossy fast path on. Reject any inherited
# definition instead of treating it like the numerical 0/1 switches below.
if [[ -n "${PIXAL3D_FAST+x}" ]]; then
  echo "PIXAL3D_FAST is rejected by the clean-geometry contract: it is presence-based, so unset it rather than assigning 0" >&2
  exit 2
fi
for quality_env in USR_GEO_FLASH PIXAL3D_SS_BF16 PIXAL3D_SLAT_BF16 PIXAL3D_M6_BF16 PIXAL3D_SLAT_BF16_GEOMETRY NVIDIA_TF32_OVERRIDE; do
  quality_value="${!quality_env:-0}"
  [[ "$quality_value" == 0 ]] || {
    echo "$quality_env=$quality_value is rejected by the clean-geometry contract; use dense F32 M3b with no inherited fast mode or TF32" >&2
    exit 2
  }
done
export USR_GEO_FLASH=0
export PIXAL3D_SS_BF16=0
export PIXAL3D_SLAT_BF16=0
export PIXAL3D_M6_BF16=0
export PIXAL3D_SLAT_BF16_GEOMETRY=0
export NVIDIA_TF32_OVERRIDE=0
# F16 GGUF is fast but fails the frozen Python-boundary M3b sampler check.
# Keep F32 source tensors for the two quality-critical generative DiTs; all
# other sequential stages still use the local NVMe F16 GGUF bundle.
export PIXAL3D_QUALITY_F32_M3B=1
export PIXAL3D_QUALITY_F32_PROJ_TEX=1
export REMESH_CLOSE_R="${REMESH_CLOSE_R:-3}"
[[ "$REMESH_CLOSE_R" =~ ^[1-9][0-9]*$ ]] || {
  echo "REMESH_CLOSE_R must be a positive integer (the clean-production default is 3)" >&2; exit 2;
}
ULTRASHAPE_STEPS="${IMAGE_TO_RIG_US_STEPS:-50}"
[[ "$ULTRASHAPE_STEPS" =~ ^[1-9][0-9]*$ ]] && (( ULTRASHAPE_STEPS >= 40 && ULTRASHAPE_STEPS <= 100 )) || {
  echo "IMAGE_TO_RIG_US_STEPS must be an integer in [40,100] (50 production default; 80 detail A/B)" >&2; exit 2;
}
# UltraShape refine latent count. 16384 is the production default; 8192 is the
# chibi-ceiling perf A/B ([[project_ultrashape_dit_vram_moe_tiling]] measured
# N=8192 visually identical to 16384 on chibi subjects, only topologically
# cleaner at 16384). The value flows into the geometry recipe + cache identity
# below so an 8192 run can never reuse a 16384 refined.glb (or vice versa).
US_LATENTS="${IMAGE_TO_RIG_US_LATENTS:-16384}"
[[ "$US_LATENTS" =~ ^[1-9][0-9]*$ ]] && (( US_LATENTS == 8192 || US_LATENTS == 16384 || US_LATENTS == 32768 )) || {
  echo "IMAGE_TO_RIG_US_LATENTS must be 8192, 16384, or 32768 (16384 production default)" >&2; exit 2;
}
GEOMETRY_RESOLUTION="${IMAGE_TO_RIG_GEO_RESOLUTION:-1024}"
[[ "$GEOMETRY_RESOLUTION" == 1024 || "$GEOMETRY_RESOLUTION" == 1536 ]] || {
  echo "IMAGE_TO_RIG_GEO_RESOLUTION must be 1024 or 1536" >&2; exit 2;
}
# The Pixal sampler seed is a real reconstruction input, not a display-only
# randomizer.  Keep it in the cache and every handoff manifest so a clean
# seed-42 cache can never be mistaken for a different-noise A/B.
IMAGE_TO_RIG_SEED="${IMAGE_TO_RIG_SEED:-42}"
[[ "$IMAGE_TO_RIG_SEED" =~ ^[0-9]+$ ]] && (( IMAGE_TO_RIG_SEED <= 2147483647 )) || {
  echo "IMAGE_TO_RIG_SEED must be an integer in [0,2147483647]" >&2; exit 2;
}
GEOMETRY_CAMERA="${IMAGE_TO_RIG_GEOMETRY_CAMERA:-moge}"
[[ "$GEOMETRY_CAMERA" == moge || "$GEOMETRY_CAMERA" == default ]] || {
  echo "IMAGE_TO_RIG_GEOMETRY_CAMERA must be moge or default" >&2; exit 2;
}
FIXED_CAMERA="${IMAGE_TO_RIG_FIXED_CAMERA:-}"
FIXED_CAMERA_ANGLE=""
FIXED_CAMERA_DISTANCE=""
FIXED_CAMERA_SCALE=""
FIXED_CAMERA_KEY=""
if [[ -n "$FIXED_CAMERA" ]]; then
  IFS=/ read -r FIXED_CAMERA_ANGLE FIXED_CAMERA_DISTANCE FIXED_CAMERA_SCALE fixed_camera_extra <<<"$FIXED_CAMERA"
  [[ -n "$FIXED_CAMERA_ANGLE" && -n "$FIXED_CAMERA_DISTANCE" && -n "$FIXED_CAMERA_SCALE" && -z "${fixed_camera_extra:-}" ]] || {
    echo "IMAGE_TO_RIG_FIXED_CAMERA must be angle/dist/scale (for example 0.858537/1.09233/1.0)" >&2; exit 2;
  }
  for fixed_camera_value in "$FIXED_CAMERA_ANGLE" "$FIXED_CAMERA_DISTANCE" "$FIXED_CAMERA_SCALE"; do
    [[ "$fixed_camera_value" =~ ^[-+]?[0-9]+([.][0-9]+)?$ ]] || {
      echo "IMAGE_TO_RIG_FIXED_CAMERA values must be decimal numbers" >&2; exit 2;
    }
  done
  awk -v d="$FIXED_CAMERA_DISTANCE" -v s="$FIXED_CAMERA_SCALE" 'BEGIN { exit !(d > 0 && s > 0) }' || {
    echo "IMAGE_TO_RIG_FIXED_CAMERA distance and scale must be positive" >&2; exit 2;
  }
  FIXED_CAMERA_KEY="$(printf '%s' "$FIXED_CAMERA" | sha256sum | awk '{print substr($1,1,12)}')"
  GEOMETRY_CAMERA=fixed
fi
SKIP_ULTRASHAPE="${IMAGE_TO_RIG_SKIP_ULTRASHAPE:-0}"
[[ "$SKIP_ULTRASHAPE" == 0 || "$SKIP_ULTRASHAPE" == 1 ]] || {
  echo "IMAGE_TO_RIG_SKIP_ULTRASHAPE must be 0 or 1" >&2; exit 2;
}
GEOMETRY_REVIEW_ONLY="${IMAGE_TO_RIG_GEOMETRY_REVIEW_ONLY:-0}"
[[ "$GEOMETRY_REVIEW_ONLY" == 0 || "$GEOMETRY_REVIEW_ONLY" == 1 ]] || {
  echo "IMAGE_TO_RIG_GEOMETRY_REVIEW_ONLY must be 0 or 1" >&2; exit 2;
}
CLAY_VERDICT_REQUEST="${IMAGE_TO_RIG_CLAY_VERDICT:-pending}"
[[ "$CLAY_VERDICT_REQUEST" == pending || "$CLAY_VERDICT_REQUEST" == approved || "$CLAY_VERDICT_REQUEST" == rejected ]] || {
  echo "IMAGE_TO_RIG_CLAY_VERDICT must be pending, approved, or rejected" >&2; exit 2;
}
# This is deliberately a versioned *geometry* recipe rather than an informal
# environment setting.  A cache made with a different seal radius is a
# different reconstruction and must never be reused just because the source
# image happens to be identical.
# v11 extends the dense-F32/no-fast/no-TF32 contract with F32 source weights
# for the quality-critical M3b geometry and projection-conditioned M6 texture
# samplers. Every variation deliberately shares this new baseline so no
# earlier cache can be relabelled as the current clean path.
GEOMETRY_RECIPE_VERSION=11
ULTRASHAPE_RECIPE_SUFFIX=""
if (( ULTRASHAPE_STEPS != 50 )); then
  ULTRASHAPE_RECIPE_SUFFIX="-ussteps${ULTRASHAPE_STEPS}"
fi
PIXAL_RECIPE_SUFFIX="-dense-f32-m3b"
GEOMETRY_CACHE_SUFFIX="-dense-m3b"
if [[ "$IMAGE_TO_RIG_SEED" != 42 ]]; then
  PIXAL_RECIPE_SUFFIX+="-seed${IMAGE_TO_RIG_SEED}"
  GEOMETRY_CACHE_SUFFIX+="-seed${IMAGE_TO_RIG_SEED}"
fi
if [[ "$GEOMETRY_RESOLUTION" != 1024 ]]; then
  # Resolution is the semantic mesh seed, not a texture/LOD knob.  It must be
  # visible in both the recipe and cache name so a 1024 Minecraft-like seed is
  # never silently presented as the 1536 A/B.
  PIXAL_RECIPE_SUFFIX="-pixal${GEOMETRY_RESOLUTION}"
  PIXAL_RECIPE_SUFFIX+="-dense-f32-m3b"
  GEOMETRY_CACHE_SUFFIX+="-pixal${GEOMETRY_RESOLUTION}"
fi
if [[ "$SKIP_ULTRASHAPE" == 1 ]]; then
  PIXAL_RECIPE_SUFFIX+="-coarse-only"
  GEOMETRY_CACHE_SUFFIX+="-coarse-only"
fi
if [[ "$GEOMETRY_CAMERA" == fixed ]]; then
  # Camera projection conditions M3b and the projection-conditioned material
  # model.  The exact numeric replay must therefore be visible in the cache
  # identity, rather than inheriting a nearby MoGe/default cache.
  PIXAL_RECIPE_SUFFIX+="-fixedcam${FIXED_CAMERA_KEY}"
  GEOMETRY_CACHE_SUFFIX+="-fixedcam${FIXED_CAMERA_KEY}"
elif [[ "$GEOMETRY_CAMERA" == default ]]; then
  # The upstream/default camera contract is a distinct reconstruction: camera
  # projection feeds Pixal's high-resolution M3b conditioning, so this is not
  # merely a viewer/FOV choice. It is nevertheless a distinct v11 cache.
  PIXAL_RECIPE_SUFFIX+="-defaultcam"
  GEOMETRY_CACHE_SUFFIX+="-defaultcam"
fi
# The clean default retains Pixal's own projection-conditioned M6 PBR volume while the exact
# image/shape latents are live on the 3060. This avoids the unvalidated inverse mesh-to-shape encoder
# in the arbitrary-mesh texture tool. The retained volume is rebaked onto the refined mesh below;
# it is generated material, not an observed-image projection overlay. The old environment name stays
# as an explicit compatibility alias for existing callers.
if [[ -n "${IMAGE_TO_RIG_LEGACY_PBR_CACHE+x}" ]]; then
  LEGACY_PBR_CACHE="$IMAGE_TO_RIG_LEGACY_PBR_CACHE"
else
  LEGACY_PBR_CACHE="${IMAGE_TO_RIG_GENERATED_PBR_CACHE:-1}"
fi
[[ "$LEGACY_PBR_CACHE" == 0 || "$LEGACY_PBR_CACHE" == 1 ]] || {
  echo "IMAGE_TO_RIG_LEGACY_PBR_CACHE must be 0 or 1" >&2; exit 2;
}
if [[ "$LEGACY_PBR_CACHE" == 1 ]]; then
  GEOMETRY_RECIPE="${GEOMETRY_CAMERA}-f16-noquad-us${US_LATENTS}${ULTRASHAPE_RECIPE_SUFFIX}${PIXAL_RECIPE_SUFFIX}-pixal-proj-pbr-cache-close-r${REMESH_CLOSE_R}-v${GEOMETRY_RECIPE_VERSION}"
  GEOMETRY_CACHE_MODE=generated-pixal-pbr-cache-f16
else
  GEOMETRY_RECIPE="${GEOMETRY_CAMERA}-f16-noquad-us${US_LATENTS}${ULTRASHAPE_RECIPE_SUFFIX}${PIXAL_RECIPE_SUFFIX}-geometry-only-close-r${REMESH_CLOSE_R}-v${GEOMETRY_RECIPE_VERSION}"
  GEOMETRY_CACHE_MODE=geometry-only-f16
fi
if [[ "$GEOMETRY_REVIEW_ONLY" == 1 && "$LEGACY_PBR_CACHE" == 1 ]]; then
  echo "geometry-review-only requires IMAGE_TO_RIG_LEGACY_PBR_CACHE=0" >&2; exit 2;
fi
if [[ "$SKIP_ULTRASHAPE" == 1 && "$LEGACY_PBR_CACHE" == 1 ]]; then
  echo "IMAGE_TO_RIG_SKIP_ULTRASHAPE requires IMAGE_TO_RIG_LEGACY_PBR_CACHE=0" >&2; exit 2;
fi
MATTING_URL="${MATTING_URL:-http://localhost:18898}"
INPUT_MODE="${IMAGE_TO_RIG_INPUT_MODE:-auto}"
[[ "$INPUT_MODE" == auto || "$INPUT_MODE" == matte ]] || { echo "IMAGE_TO_RIG_INPUT_MODE must be auto or matte" >&2; exit 2; }
# Neural matte model for auto black-matte / opaque inputs.
#   birefnet     (default) - Python's EXACT ZhengPeng7/BiRefNet weights + transform, run on the
#                            3060 via the Pixal venv. Produces the composited-over-black RGBA the
#                            native has_alpha crop path consumes, reproducing Pixal3D preprocess_image
#                            up to the crop (which native owns). Measured stage-1 coord overlap 97.3%
#                            / z_s 0.94 vs Python, up from the geometric flood-fill's 63.6% / 0.42.
#   geometric    - legacy border-flood silhouette matte (make_matte). A labelled A/B ONLY: its
#                            silhouette differs from Python's BiRefNet and collapses SS-DiT stage-1.
#   rmbg-service - the shared docker matting service (RMBG-2.0 GGUF). SAME BiRefNet architecture,
#                            DIFFERENT weights; measured only ~91.7% stage-1 -- not production parity.
MATTE_MODEL="${IMAGE_TO_RIG_MATTE_MODEL:-birefnet}"
[[ "$MATTE_MODEL" == birefnet || "$MATTE_MODEL" == geometric || "$MATTE_MODEL" == rmbg-service ]] || {
  echo "IMAGE_TO_RIG_MATTE_MODEL must be birefnet, geometric, or rmbg-service" >&2; exit 2; }
BIREFNET_PYTHON="${BIREFNET_PYTHON:-/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python}"

# Neural soft matte: run Pixal's exact BiRefNet on the reserved 3060 (holding the shared lock) and
# write a composited-over-black RGBA cutout. Feeding this to image_to_rig WITHOUT --image-model-ready
# routes it through the has_alpha crop/composite path (image_io.hpp pixal_preprocess_black_matte),
# which owns the PIL-parity Lanczos resize + float-box square crop fixes. That is the production
# contract that reaches Python stage-1 parity; the geometric silhouette never did.
neural_birefnet_matte() {
  local src="$1" out_rgba="$2" lock="$OUT_ROOT/.3060-image-to-rig.lock"
  [[ -x "$BIREFNET_PYTHON" ]] || { echo "missing BiRefNet python (Pixal venv): $BIREFNET_PYTHON" >&2; exit 1; }
  [[ -f "$CP/shootout/birefnet_matte.py" ]] || { echo "missing birefnet_matte.py helper" >&2; exit 1; }
  (
    exec 9>"$lock"
    flock -n 9 || { echo "another image-to-rig job owns the 3060 lock" >&2; exit 75; }
    CUDA_VISIBLE_DEVICES="$GPU_3060_UUID" nice -n "$IMAGE_TO_RIG_NICE_LEVEL" \
      "$BIREFNET_PYTHON" "$CP/shootout/birefnet_matte.py" "$src" "$out_rgba" --device cuda
  )
  [[ -s "$out_rgba" ]] || { echo "BiRefNet matte produced no RGBA cutout: $out_rgba" >&2; exit 1; }
}

# All geometry and texture stages use exactly one image frame.  A true RGBA cutout keeps its own
# framing contract; black-composited and opaque inputs are neural-matted (default BiRefNet) into a
# composited-over-black RGBA and handed to the native has_alpha crop path -- the geometric border
# flood is retained only as an explicit A/B, since its silhouette diverges from Python's and
# chaotically collapses SS-DiT stage-1.  Every 3060 step names the physical UUID and takes the lock.
INPUT_KIND="$("$CP/make_matte" --inspect-input "$IMAGE" | awk -F= '$1=="input_kind" {print $2}')"
[[ -n "$INPUT_KIND" ]] || { echo "could not classify input image: $IMAGE" >&2; exit 1; }
PIPELINE_IMAGE="$IMAGE"
MODEL_FRAME_READY=0
case "$INPUT_MODE:$INPUT_KIND" in
  matte:*)
    echo "== $LABEL: use caller-supplied matte frame ($INPUT_KIND) =="
    ;;
  auto:rgba-cutout)
    PIPELINE_IMAGE="$OUT/input_matte.png"
    MODEL_FRAME_READY=1
    echo "== $LABEL: make deterministic matte from RGBA cutout =="
    "$CP/make_matte" --rgba "$IMAGE" "$PIPELINE_IMAGE"
    ;;
  auto:black-matte)
    if [[ "$MATTE_MODEL" == geometric ]]; then
      PIPELINE_IMAGE="$OUT/input_matte.png"
      MODEL_FRAME_READY=1
      echo "== $LABEL: [A/B] geometric black-matte framing (border-flood; NOT stage-1 parity) =="
      "$CP/make_matte" "$IMAGE" "$PIPELINE_IMAGE"
    elif [[ "$MATTE_MODEL" == rmbg-service ]]; then
      CUTOUT="$OUT/source_cutout_rgba.png"
      PIPELINE_IMAGE="$CUTOUT"
      MODEL_FRAME_READY=0
      echo "== $LABEL: [A/B] RMBG-2.0 service neural matte on the RTX 3060 (~91.7% stage-1) =="
      (
        exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
        flock -n 9 || { echo "another image-to-rig job owns the 3060 lock" >&2; exit 75; }
        code="$(curl -sS -X POST "$MATTING_URL/remove?bg_mode=alpha" -F "images=@$IMAGE" -F "gpu=$GPU_3060_UUID" -o "$CUTOUT" -w "%{http_code}")"
        [[ "$code" == 200 ]] || { echo "matting service HTTP $code" >&2; exit 1; }
      )
      [[ -s "$CUTOUT" ]] || { echo "matting service did not produce an RGBA cutout" >&2; exit 1; }
    else
      PIPELINE_IMAGE="$OUT/input_birefnet_rgba.png"
      MODEL_FRAME_READY=0
      echo "== $LABEL: BiRefNet neural soft matte on PCI GPU 0 / RTX 3060 (Python stage-1 parity) =="
      neural_birefnet_matte "$IMAGE" "$PIPELINE_IMAGE"
    fi
    ;;
  auto:opaque)
    if [[ "$MATTE_MODEL" == rmbg-service ]]; then
      CUTOUT="$OUT/source_cutout_rgba.png"
      PIPELINE_IMAGE="$CUTOUT"
      MODEL_FRAME_READY=0
      echo "== $LABEL: [A/B] RMBG-2.0 service raw-photo matte on the RTX 3060 =="
      (
        exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
        flock -n 9 || { echo "another image-to-rig job owns the 3060 lock" >&2; exit 75; }
        code="$(curl -sS -X POST "$MATTING_URL/remove?bg_mode=alpha" -F "images=@$IMAGE" -F "gpu=$GPU_3060_UUID" -o "$CUTOUT" -w "%{http_code}")"
        [[ "$code" == 200 ]] || { echo "matting service HTTP $code" >&2; exit 1; }
      )
      [[ -s "$CUTOUT" ]] || { echo "matting service did not produce an RGBA cutout" >&2; exit 1; }
    else
      [[ "$MATTE_MODEL" != geometric ]] || { echo "geometric matte cannot remove a real background; use birefnet or rmbg-service for opaque input" >&2; exit 2; }
      PIPELINE_IMAGE="$OUT/input_birefnet_rgba.png"
      MODEL_FRAME_READY=0
      echo "== $LABEL: BiRefNet raw-photo neural matte on PCI GPU 0 / RTX 3060 =="
      neural_birefnet_matte "$IMAGE" "$PIPELINE_IMAGE"
    fi
    ;;
esac
[[ -f "$PIPELINE_IMAGE" ]] || { echo "missing model-facing matte: $PIPELINE_IMAGE" >&2; exit 1; }
MODEL_FRAME_ARGS=()
if [[ "$MODEL_FRAME_READY" == 1 ]]; then
  # make_matte already produces Pixal's final black model frame. Reapplying
  # image_to_rig's border-crop here visibly changes the 512px DINO input and
  # causes the cascade to diverge before texture generation begins.
  MODEL_FRAME_ARGS+=(--image-model-ready)
fi
MATTE_HASH="$(sha256sum "$PIPELINE_IMAGE" | awk '{print substr($1,1,16)}')"
# The cache key includes both the uploaded bytes and the exact model-facing matte bytes. This prevents
# a changed matting service, alpha cutout, framing recipe, or clean-geometry
# recipe from silently reusing geometry made from a different image frame.
# The cache directory itself is part of the provenance contract.  In
# particular, geometry-only caches intentionally contain no usable legacy PBR
# volume, so they must never share a name with the opt-in legacy-PBR/projection
# cache for the same image.
CACHE="$OUT/cache_${IMAGE_HASH}_${MATTE_HASH}_n${US_LATENTS}_${GEOMETRY_CACHE_MODE}_seal${REMESH_CLOSE_R}_geom${GEOMETRY_RECIPE_VERSION}${GEOMETRY_CACHE_SUFFIX}"
ln -sfn "$PIPELINE_IMAGE" "$OUT/input.png"
printf 'source_image=%s\nsource_sha256=%s\ninput_kind=%s\ninput_mode=%s\nmodel_image=%s\nmodel_image_sha256=%s\n' \
  "$IMAGE" "$IMAGE_HASH" "$INPUT_KIND" "$INPUT_MODE" "$PIPELINE_IMAGE" "$MATTE_HASH" >"$OUT/preprocess_source.txt"
if [[ "${IMAGE_TO_RIG_PREPARE_ONLY:-0}" != 0 ]]; then
  PIPELINE_STAGE=prepared-input
  PIPELINE_COMPLETED=1
  write_pipeline_status succeeded 'prepare-only; no geometry or texture inference requested'
  echo "== DONE: prepared model input $PIPELINE_IMAGE (no geometry/texture inference) =="
  exit 0
fi

# Keep this geometry cache deliberately separate from the final native texture outputs.  `image_to_rig`
# is used only for its native MoGe + geometry + UltraShape stages; its legacy texture/rig result is never
# promoted over the native textured assets produced below.
cache_recipe_matches=0
if [[ -f "$CACHE/geometry_recipe.txt" ]] && grep -Fqx "geometry_recipe=$GEOMETRY_RECIPE" "$CACHE/geometry_recipe.txt"; then
  cache_recipe_matches=1
fi
cache_pbr_ready=1
if [[ "$LEGACY_PBR_CACHE" == 1 && ( ! -f "$CACHE/pbr_feats.bin" || ! -f "$CACHE/pbr_coords.bin" ) ]]; then
  cache_pbr_ready=0
fi
PIPELINE_STAGE=geometry-cache
write_pipeline_status running 'geometry cache create or reuse; native material inference has not started'
if [[ "${IMAGE_TO_RIG_REFRESH:-0}" != 0 || ! -f "$CACHE/refined.glb" || "$cache_pbr_ready" != 1 || "$cache_recipe_matches" != 1 ]]; then
  LOCK="$OUT_ROOT/.3060-image-to-rig.lock"
  mkdir -p "$CACHE" "$OUT_ROOT"
  echo "== $LABEL: create clean geometry cache ${CACHE##*/} on the RTX 3060 =="
  (
    exec 9>"$LOCK"
    flock -n 9 || { echo "another image-to-rig job owns the 3060 lock" >&2; exit 75; }
    # GGUF checkpoint paths are absolute, but the native graph also loads its
    # checked-in RoPE phase table from `weights_npy/...`.  Run from the C++
    # tool directory so a caller's shell cwd cannot turn a clean replay into a
    # missing-weight crash.
    cd "$CP"
    ULTRASHAPE_ARGS=()
    if [[ "$SKIP_ULTRASHAPE" == 1 ]]; then ULTRASHAPE_ARGS+=(--no-refine); fi
    CAMERA_ARGS=()
    if [[ "$GEOMETRY_CAMERA" == moge ]]; then
      CAMERA_ARGS+=(--moge)
    elif [[ "$GEOMETRY_CAMERA" == fixed ]]; then
      CAMERA_ARGS+=(--cam "$FIXED_CAMERA_ANGLE" "$FIXED_CAMERA_DISTANCE" "$FIXED_CAMERA_SCALE")
    fi
    if [[ "$LEGACY_PBR_CACHE" == 1 ]]; then
      "${IMAGE_TO_RIG_CMD[@]}" --model "$PIXAL3D_MODEL_DIR" --image "$PIPELINE_IMAGE" "${MODEL_FRAME_ARGS[@]}" "${CAMERA_ARGS[@]}" --seed "$IMAGE_TO_RIG_SEED" \
        --resolution "$GEOMETRY_RESOLUTION" --no-quad --us-latents "$US_LATENTS" --us-steps "$ULTRASHAPE_STEPS" "${ULTRASHAPE_ARGS[@]}" --tex-dit proj \
        --material-cache-only --stage-dir "$CACHE" --out "$CACHE/material_cache_unused.glb" 2>&1 | tee "$CACHE/geometry.log"
    else
      "${IMAGE_TO_RIG_CMD[@]}" --model "$PIXAL3D_MODEL_DIR" --image "$PIPELINE_IMAGE" "${MODEL_FRAME_ARGS[@]}" "${CAMERA_ARGS[@]}" --seed "$IMAGE_TO_RIG_SEED" \
        --resolution "$GEOMETRY_RESOLUTION" --no-quad --us-latents "$US_LATENTS" --us-steps "$ULTRASHAPE_STEPS" "${ULTRASHAPE_ARGS[@]}" --geometry-only --stage-dir "$CACHE" \
        --out "$CACHE/geometry_only_unused.glb" 2>&1 | tee "$CACHE/geometry.log"
    fi
  )
  printf 'geometry_recipe=%s\ngeometry_camera_request=%s\nfixed_camera=%s\nremesh_close_r=%s\npixal_resolution=%s\npixal_seed=%s\nultrashape_steps=%s\nultrashape_skipped=%s\ngeometry_recipe_version=%s\ngeometry_attention=dense-f32-m3b (USR_GEO_FLASH=0; experimental BF16 modes rejected)\ncritical_dit_weights=F32 source tensors for M3b geometry and M6 projection texture; F16 GGUF retained for other sequential stages\nlegacy_pbr_cache=%s\npixal_model_dir=%s\npixal_precision=hybrid F32-critical/F16-remaining checkpoint set\nscheduler=nice=%s; ionice=best-effort:7\n' \
    "$GEOMETRY_RECIPE" "$GEOMETRY_CAMERA" "${FIXED_CAMERA:-none}" "$REMESH_CLOSE_R" "$GEOMETRY_RESOLUTION" "$IMAGE_TO_RIG_SEED" "$ULTRASHAPE_STEPS" "$SKIP_ULTRASHAPE" "$GEOMETRY_RECIPE_VERSION" "$LEGACY_PBR_CACHE" "$PIXAL3D_MODEL_DIR" "$IMAGE_TO_RIG_NICE_LEVEL" >"$CACHE/geometry_recipe.txt"
else
  echo "== $LABEL: reuse image-keyed clean geometry cache ${CACHE##*/} =="
fi

[[ -f "$CACHE/refined.glb" ]] || { echo "geometry cache did not produce refined.glb" >&2; exit 1; }
# Stable eye-test aliases: texture/rig stages must be judged against the exact
# coarse and refined meshes that produced them, not an old cache with a
# subject-specific hash in its path.  These links are diagnostic-only and do
# not alter the geometry cache or production mesh.
ln -sfn "$CACHE/refined.glb" "$OUT/refined_geometry.glb"
[[ ! -f "$CACHE/coarse.glb" ]] || ln -sfn "$CACHE/coarse.glb" "$OUT/coarse_geometry.glb"
# The review page accepts a query-selected subject, so a new model under the
# shared output root is immediately inspectable without adding it to a static
# JavaScript list.  Keep an ordinary landing URL for callers that deliberately
# put their output elsewhere.
OUT_SUBJECT="$(basename "$OUT")"
EYE_TEST_URL='http://10.0.0.208:8077/inline-3d/runbook.html'
if [[ "$OUT_SUBJECT" =~ ^[A-Za-z0-9._-]+$ ]] && [[ "$(readlink -f "$OUT")" == "$(readlink -f "$OUT_ROOT")/"* ]]; then
  EYE_TEST_URL+="?subject=$OUT_SUBJECT&artifact=refined_geometry.glb"
fi
# Do this before native material inference: a closed, inspectable geometry is
# a prerequisite for a clean texture delivery, not a fact to discover after
# consuming the reserved GPU.  Nonmanifold edges are retained as diagnostic
# evidence (UltraShape output can include seam contacts); an actual boundary
# is a hard stop because it will make both bake and animation visibly fail.
GEOMETRY_TOPOLOGY="$("$CP/mesh_topo" "$CACHE/refined.glb")" || { echo "could not inspect refined geometry" >&2; exit 1; }
[[ "$GEOMETRY_TOPOLOGY" =~ open=([0-9]+) ]] || { echo "refined geometry topology report lacks open-edge count" >&2; exit 1; }
GEOMETRY_OPEN="${BASH_REMATCH[1]}"
GEOMETRY_GATE_RESULT=rejected
if (( GEOMETRY_OPEN == 0 )); then GEOMETRY_GATE_RESULT=passed; fi
{
  printf 'geometry_cache=%s\n' "$CACHE"
  printf 'geometry_recipe=%s\n' "$GEOMETRY_RECIPE"
  printf 'geometry_camera_request=%s\n' "$GEOMETRY_CAMERA"
  printf 'fixed_camera=%s\n' "${FIXED_CAMERA:-none}"
  printf '%s\n' 'clay_stage_aliases=coarse_geometry.glb (when available), refined_geometry.glb (exact texture source)'
  printf 'remesh_close_r=%s\n' "$REMESH_CLOSE_R"
  printf 'pixal_resolution=%s\n' "$GEOMETRY_RESOLUTION"
  printf 'pixal_seed=%s\n' "$IMAGE_TO_RIG_SEED"
  printf 'ultrashape_steps=%s\n' "$ULTRASHAPE_STEPS"
  printf 'ultrashape_skipped=%s\n' "$SKIP_ULTRASHAPE"
  printf 'pixal_model_dir=%s\n' "$PIXAL3D_MODEL_DIR"
  printf '%s\n' 'pixal_precision=hybrid F32-critical/F16-remaining checkpoint set'
  printf 'geometry_topology=%s\n' "$GEOMETRY_TOPOLOGY"
  printf '%s\n' 'geometry_gate=position-welded open=0 before native texture inference'
  printf 'geometry_gate_result=%s\n' "$GEOMETRY_GATE_RESULT"
  printf 'eye_test_url=%s\n' "$EYE_TEST_URL"
} >"$OUT/geometry_delivery.txt"
(( GEOMETRY_OPEN == 0 )) || { echo "REJECT: refined geometry has $GEOMETRY_OPEN open edges; refusing texture/rig stages" >&2; exit 1; }

# A closed mesh is necessary but cannot tell us whether a face is torn, a
# character is Minecraft-like, or UltraShape merely preserved the wrong
# silhouette.  Persist a visual decision tied to the exact cache + recipe, so
# a later source/framing/geometry change can never inherit approval from an
# old clay review.  This is deliberately before every material, LOD, rig, or
# projection action: none of those can repair a rejected clay mesh.
CLAY_GATE="$OUT/clay_gate.txt"
CLAY_GATE_RESULT=pending
if [[ -f "$CLAY_GATE" ]] \
  && grep -Fqx 'visual_verdict=approved' "$CLAY_GATE" \
  && grep -Fqx "geometry_cache=$CACHE" "$CLAY_GATE" \
  && grep -Fqx "geometry_recipe=$GEOMETRY_RECIPE" "$CLAY_GATE" \
  && grep -Fqx "model_image_sha256=$MATTE_HASH" "$CLAY_GATE"; then
  CLAY_GATE_RESULT=approved
fi
if [[ "$CLAY_VERDICT_REQUEST" != pending ]]; then
  CLAY_GATE_RESULT="$CLAY_VERDICT_REQUEST"
fi
{
  printf 'schema_version=1\n'
  printf 'visual_verdict=%s\n' "$CLAY_GATE_RESULT"
  printf 'geometry_cache=%s\n' "$CACHE"
  printf 'geometry_recipe=%s\n' "$GEOMETRY_RECIPE"
  printf 'pixal_seed=%s\n' "$IMAGE_TO_RIG_SEED"
  printf 'model_image_sha256=%s\n' "$MATTE_HASH"
  printf 'coarse_clay=coarse_geometry.glb\n'
  printf 'refined_clay=refined_geometry.glb\n'
  printf 'eye_test_url=%s\n' "$EYE_TEST_URL"
  printf 'mechanical_topology=%s\n' "$GEOMETRY_TOPOLOGY"
  printf 'rule=texture_lods_rig_and_projection_are_blocked_unless_visual_verdict=approved\n'
  printf 'updated_at=%s\n' "$(date -Is)"
} >"$CLAY_GATE.tmp"
mv -f "$CLAY_GATE.tmp" "$CLAY_GATE"

if [[ "$CLAY_GATE_RESULT" != approved ]]; then
  PIPELINE_STAGE=clay-review
  PIPELINE_COMPLETED=1
  if [[ "$CLAY_GATE_RESULT" == rejected ]]; then
    write_pipeline_status succeeded 'visual clay rejected; texture, LOD, rig, and projection intentionally not run'
    echo "== CLAY REJECTED: recorded $CLAY_GATE; texture/rig blocked for this cache =="
  else
    write_pipeline_status succeeded 'awaiting visual clay approval; texture, LOD, rig, and projection intentionally not run'
    echo "== CLAY REVIEW REQUIRED: inspect coarse_geometry.glb and refined_geometry.glb, then rerun with IMAGE_TO_RIG_CLAY_VERDICT=approved only if the refined clay passes =="
  fi
  exit 0
fi
if [[ "$GEOMETRY_REVIEW_ONLY" == 1 ]]; then
  cat >"$OUT/runbook_source.txt" <<EOF
source_image=$IMAGE
source_sha256=$IMAGE_HASH
model_image=$PIPELINE_IMAGE
input_kind=$INPUT_KIND
input_mode=$INPUT_MODE
geometry_cache=$CACHE
geometry_recipe=$GEOMETRY_RECIPE
geometry_camera_request=$GEOMETRY_CAMERA
fixed_camera=${FIXED_CAMERA:-none}
pixal_resolution=$GEOMETRY_RESOLUTION
pixal_seed=$IMAGE_TO_RIG_SEED
ultrashape_steps=$ULTRASHAPE_STEPS
ultrashape_skipped=$SKIP_ULTRASHAPE
pixal_model_dir=$PIXAL3D_MODEL_DIR
pixal_precision=hybrid F32-critical/F16-remaining checkpoint set
geometry_delivery_manifest=$OUT/geometry_delivery.txt
camera_provenance=$CACHE/camera_provenance.txt
texture_recipe=not run: geometry-review-only clay gate
rig_mode=not run: geometry-review-only clay gate
gpu=PCI GPU 0 / RTX 3060 only
EOF
  PIPELINE_STAGE=geometry-review-ready
  PIPELINE_COMPLETED=1
  write_pipeline_status succeeded 'geometry review-only; native texture and rig intentionally not run'
  echo "== DONE: geometry review aliases ready (Pixal3D ${GEOMETRY_RESOLUTION}, ultrashape_skipped=${SKIP_ULTRASHAPE}) =="
  exit 0
fi
PIPELINE_STAGE=native-texture-lods-rig
write_pipeline_status running 'geometry gate passed; native high texture, CPU LOD rebakes, then structural rig gate'
# The retained PBR volume is valid only for the exact model-facing image and
# refined mesh that generated it.  Persist that binding before handing the
# cache to the reusable texture/rig wrapper: a complete atlas recovery check
# cannot detect an old-but-plausible material cache from another frame.
# The voxel coordinates in that volume are indices on a grid-GEOMETRY_RESOLUTION lattice, and the
# rebake needs that number to map mesh positions back onto them.  Recording it here (schema 2) makes
# a lattice mismatch a manifest rejection instead of a silently unpainted atlas.
if [[ "$LEGACY_PBR_CACHE" == 1 ]]; then
  PBR_CACHE_MANIFEST="$CACHE/pbr_cache_manifest.txt"
  PBR_CACHE_MANIFEST_TMP="$PBR_CACHE_MANIFEST.tmp.$$"
  CACHE_LATTICE_BIN=""
  if [[ -s "$CACHE/resolution.bin" ]]; then
    CACHE_LATTICE_BIN="$(od -An -tu4 -j8 -N4 "$CACHE/resolution.bin" 2>/dev/null | tr -d ' \n')"
  fi
  if [[ -n "$CACHE_LATTICE_BIN" && "$CACHE_LATTICE_BIN" != "$GEOMETRY_RESOLUTION" ]]; then
    echo "geometry cache lattice disagreement: resolution.bin says $CACHE_LATTICE_BIN, this run requested $GEOMETRY_RESOLUTION" >&2
    exit 2
  fi
  {
    printf 'schema_version=2\n'
    printf 'pbr_lattice_resolution=%s\n' "$GEOMETRY_RESOLUTION"
    printf 'model_image_sha256=%s\n' "$(sha256sum "$PIPELINE_IMAGE" | awk '{print $1}')"
    printf 'refined_mesh_sha256=%s\n' "$(sha256sum "$CACHE/refined.glb" | awk '{print $1}')"
    printf 'pbr_feats_sha256=%s\n' "$(sha256sum "$CACHE/pbr_feats.bin" | awk '{print $1}')"
    printf 'pbr_coords_sha256=%s\n' "$(sha256sum "$CACHE/pbr_coords.bin" | awk '{print $1}')"
    printf 'camera_provenance_sha256=%s\n' "$(sha256sum "$CACHE/camera_provenance.txt" | awk '{print $1}')"
  } >"$PBR_CACHE_MANIFEST_TMP"
  mv -f "$PBR_CACHE_MANIFEST_TMP" "$PBR_CACHE_MANIFEST"
fi
NATIVE_RIG=1
RIG_MODE=native
if ! RIG_PROFILE="$RIG_PROFILE" NATIVE_TEXTURE_CAMERA_FILE="$CACHE/camera_provenance.txt" NATIVE_GENERATED_PBR_CACHE="$([[ "$LEGACY_PBR_CACHE" == 1 ]] && printf '%s' "$CACHE")" \
     "$CP/shootout/native_image_to_rig.sh" "$CACHE/refined.glb" "$PIPELINE_IMAGE" "$OUT" "$LABEL"; then
  NATIVE_RIG=0
  RIG_MODE=structural-gate-rejected
  # Texture LODs are valuable evidence, but a rejected native SkinTokens
  # result is never replaced by a legacy or sampled rig.  Publication remains
  # fail-closed until the native route clears the unchanged structural and
  # real-LBS gates.
  for f in "$OUT/native_hero_textured.glb" "$OUT/native_high_textured.glb" "$OUT/native_medium_textured.glb" "$OUT/native_low_textured.glb"; do
    [[ -f "$f" ]] || { echo "native pipeline failed before producing clean LODs" >&2; exit 1; }
  done
  echo "native $RIG_PROFILE rig rejected; retaining clean texture delivery with no rig fallback" >&2
fi

# Projection is deliberately an A/B, never a replacement for native_high_textured.glb. It starts from
# that exact native atlas and changes only texels genuinely observed by the supplied cameras; every
# unobserved texel remains generated native material. The camera provenance is from the geometry cache.
#
# IMAGE_TO_RIG_TEX_VIEWS is retained for shell convenience, but it cannot represent paths containing
# whitespace and is awkward for a real 4--8 camera turnaround.  The TSV manifest is the production
# contract: one canonical camera yaw and one absolute source path per line.  Normalising yaws before
# invoking image_to_rig prevents an accidental `180`/`-180` duplicate camera and makes the recorded
# evidence unambiguous.
NATIVE_PROJ_ARGS=()
PROJECT="${IMAGE_TO_RIG_PROJECT:-0}"
PROJECT_PROMOTE="${IMAGE_TO_RIG_PROJECT_PROMOTE:-0}"
[[ "$PROJECT_PROMOTE" == 0 || "$PROJECT_PROMOTE" == 1 ]] || { echo "IMAGE_TO_RIG_PROJECT_PROMOTE must be 0 or 1" >&2; exit 2; }
[[ "$PROJECT_PROMOTE" != 1 ]] || PROJECT=1
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
    0) NATIVE_PROJ_ARGS+=(--front "$path");;
    180) NATIVE_PROJ_ARGS+=(--back "$path");;
    *) NATIVE_PROJ_ARGS+=(--view "$yaw" "$path");;
  esac
done
if [[ "$PROJECT" != 0 ]]; then
  PIPELINE_STAGE=native-observed-view-projection-ab
  write_pipeline_status running 'native textured delivery exists; CPU-only native-base observed-image A/B'
  CAMERA_PROVENANCE="$CACHE/camera_provenance.txt"
  [[ -s "$CAMERA_PROVENANCE" ]] || {
    echo "native observed projection needs camera provenance from this geometry cache; rerun geometry with IMAGE_TO_RIG_REFRESH=1" >&2; exit 1;
  }
  echo "== $LABEL: native-base observed-view projection A/B (native texture remains default) =="
  RIGGED_PROJECT_ARGS=()
  if [[ -s "$OUT/hymotion_rigged.glb" ]]; then
    RIGGED_PROJECT_ARGS=(--rigged-source "$OUT/hymotion_rigged.glb")
  fi
  "$CP/shootout/native_observed_atlas_project.sh" "$OUT/native_high_textured.glb" "$CAMERA_PROVENANCE" "$OUT" \
    "${NATIVE_PROJ_ARGS[@]}" "${RIGGED_PROJECT_ARGS[@]}"
  NATIVE_PROJ_GLB="$OUT/native_high_textured_observed_projected_ab.glb"
  if [[ "$PROJECT_PROMOTE" == 1 ]]; then
    (( TOTAL_PROJ_CAMERAS >= 4 )) || { echo "promotion validation needs 4--8 observed cameras" >&2; exit 1; }
    "$CP/shootout/verify_native_observed_projection.sh" "$NATIVE_PROJ_GLB" --require-turnaround
    printf 'promotion_result=validated-candidate; native high remains production pending visual promotion\n' \
      >>"${NATIVE_PROJ_GLB%.glb}.projection-source.txt"
  else
    "$CP/shootout/verify_native_observed_projection.sh" "$NATIVE_PROJ_GLB"
  fi
fi

cat >"$OUT/runbook_source.txt" <<EOF
source_image=$IMAGE
source_sha256=$IMAGE_HASH
model_image=$PIPELINE_IMAGE
input_kind=$INPUT_KIND
input_mode=$INPUT_MODE
geometry_cache=$CACHE
geometry_recipe=$GEOMETRY_RECIPE
geometry_camera_request=$GEOMETRY_CAMERA
fixed_camera=${FIXED_CAMERA:-none}
pixal_resolution=$GEOMETRY_RESOLUTION
pixal_seed=$IMAGE_TO_RIG_SEED
ultrashape_steps=$ULTRASHAPE_STEPS
ultrashape_skipped=$SKIP_ULTRASHAPE
pixal_model_dir=$PIXAL3D_MODEL_DIR
pixal_precision=hybrid F32-critical/F16-remaining checkpoint set
geometry_delivery_manifest=$OUT/geometry_delivery.txt
camera_provenance=$CACHE/camera_provenance.txt
legacy_pbr_cache=$LEGACY_PBR_CACHE
texture_recipe=native Pixal projection-conditioned F16 material cache + CPU high/medium/low rebakes + structural rig gate
texture_delivery_manifest=$OUT/texture_delivery.txt
rig_mode=$RIG_MODE
gpu=PCI GPU 0 / RTX 3060 only
EOF
if [[ "$RIG_MODE" == structural-gate-rejected ]]; then
  PIPELINE_STAGE=rig-rejected
  write_pipeline_status failed 'geometry and native texture LODs completed; no structurally valid Hymotion rig was published'
  echo "== TEXTURE DELIVERY COMPLETE; RIG REJECTED: $OUT ==" >&2
  exit 1
fi
PIPELINE_STAGE=complete
PIPELINE_COMPLETED=1
write_pipeline_status succeeded 'geometry, native texture LODs, and published rig/projection stages completed'
echo "== DONE: $OUT =="
