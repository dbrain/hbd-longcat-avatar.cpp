#!/usr/bin/env bash
# rig_texture_chain.sh — ONE reproducible chain: textured mesh + sampled-point rig inputs
#   -> skintokens_e2e (REAL R1/R4 rig: R1->R3->R5->R4) -> combine_rig_tex_main (kNN skin-transfer + merge)
#   -> a single TEXTURED + SKINNED .glb (the avatar deliverable). No manual stitching.
#
# RIG = DETERMINISTIC beam (do_sample=False), beams=20, followed by independent topology and actual-LBS pose gates
# over completed hypotheses. This is the PRINCIPLED decode, not a selection
# hack: sampling (do_sample=True, temp=1.5) is what produces the occasional head-fan/runaway — it's chaos
# amplifying the (unavoidable) ggml-vs-PyTorch float drift in the forward. Turning sampling OFF removes the
# chaos entirely; the deterministic top-beam decode is structurally clean every time. beams=20 (not 10)
# because the forward drift pushes Python's winning hypothesis just outside the top-10 — at top-20 the C++
# recovers it: on gilly the parent tree is BIT-IDENTICAL to Python's deterministic J56 rig (maxfan 5),
# 6.6 GB VRAM, ~17 s, fully reproducible (no seed lottery, no fan-rejection). See FINDINGS-rig-forward-parity.md.
#
# Usage:
#   rig_texture_chain.sh <rig_input_dir> <textured_mesh.glb> <qwen3_w_dir> <out.glb> [beams=20]
#
#   <rig_input_dir>     dir with vertices.npy [Ns,3] + normals.npy [Ns,3] (FPS-sampled cond points; banked).
#   <textured_mesh.glb> full-res TEXTURED mesh to skin (POSITION/NORMAL/TEXCOORD_0 + baseColor).
#   <qwen3_w_dir>       rigger qwen3 weight dir.
#   <out.glb>           output textured+skinned GLB.
#   [beams]             beam width (default 20; wider recovers the richer Python-matching skeleton).
#
# Env: PIXAL3D_GGUF_DIR must point at the skin-VAE GGUFs (skin_vae_gguf).
#      R1W_SRC points at the mesh-condition encoder/output-projection NPYs.
set -euo pipefail
CP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RIG_IN="${1:?rig_input_dir}"; TEX_GLB="${2:?textured_mesh.glb}"; QW="${3:?qwen3_w_dir}"; OUT="${4:?out.glb}"
BEAMS="${5:-20}"
ROOT_MIN_FREE_GIB="${IMAGE_TO_RIG_MIN_ROOT_FREE_GIB:-20}"
[[ "$ROOT_MIN_FREE_GIB" =~ ^[0-9]+$ ]] || { echo "IMAGE_TO_RIG_MIN_ROOT_FREE_GIB must be a non-negative integer" >&2; exit 2; }
root_free_kib="$(df -Pk / | awk 'NR==2 {print $4}')"
(( root_free_kib >= ROOT_MIN_FREE_GIB * 1024 * 1024 )) || {
  echo "refusing rig run: / has only $(( root_free_kib / 1024 / 1024 )) GiB free; require ${ROOT_MIN_FREE_GIB} GiB. Keep Docker builder cache and generated artifacts off /." >&2
  exit 75
}
: "${PIXAL3D_GGUF_DIR:?set PIXAL3D_GGUF_DIR to the skin_vae_gguf dir}"
R1W_SRC="${R1W_SRC:-/home/dbrain/models/3d/rig/r1w_real}"
RIG_PROFILE="${RIG_PROFILE:-humanoid}"
RIG_RECOVER_APPENDAGE_TIPS="${RIG_RECOVER_APPENDAGE_TIPS:-0}"
RIG_ALLOW_FLAT_BASECOLOR="${RIG_ALLOW_FLAT_BASECOLOR:-0}"
RIG_POSE_GATE_PYTHON="${RIG_POSE_GATE_PYTHON:-/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python}"
RIG_SKIN_MODE="${RIG_SKIN_MODE:-learned}"
RIG_SKIN_SMOOTH_ROUNDS="${RIG_SKIN_SMOOTH_ROUNDS:-16}"
RIG_DEFER_POSE_GATE="${RIG_DEFER_POSE_GATE:-0}"
RIG_ALLOW_TRANSFER_MESH_MISMATCH="${RIG_ALLOW_TRANSFER_MESH_MISMATCH:-0}"
RIG_DECODE_MODE="${RIG_DECODE_MODE:-auto}"
RIG_SAMPLE_BEAMS="${RIG_SAMPLE_BEAMS:-10}"
RIG_SAMPLE_SEED="${RIG_SAMPLE_SEED:-0}"
RIG_GENERIC_COMPONENT_REPAIR="${RIG_GENERIC_COMPONENT_REPAIR:-1}"
RIG_PREFIX_PREFILL_MODE="${RIG_PREFIX_PREFILL_MODE:-shared-b1}"
# Cache dtype is part of the native decoder contract, not a global quality
# knob: deterministic F32-prefix FA2 needs F32 KV, while the exact native
# Philox sampled fallback uses its validated BF16 cache regime. Select it
# inside run_native_decode(), after the mode is known.
RIG_KV_TYPE="${RIG_KV_TYPE:-auto}"
# Python may render/read the final GLB pose gate, but production must never
# mutate a native skeleton or skin field through a Python repair pass.
RIG_COMPONENT_BRANCH_REPAIR="${RIG_COMPONENT_BRANCH_REPAIR:-0}"
[[ -d "$R1W_SRC" ]] || { echo "missing R1 weights: $R1W_SRC" >&2; exit 2; }
case "$RIG_PROFILE" in
  humanoid) SELECT_FLAG="structural-select" ;;
  generic)  SELECT_FLAG="generic-structural-select" ;;
  *) echo "RIG_PROFILE must be humanoid or generic (got $RIG_PROFILE)" >&2; exit 2 ;;
esac
[[ "$RIG_RECOVER_APPENDAGE_TIPS" == 0 || "$RIG_RECOVER_APPENDAGE_TIPS" == 1 ]] || {
  echo "RIG_RECOVER_APPENDAGE_TIPS must be 0 or 1" >&2; exit 2;
}
[[ "$RIG_ALLOW_FLAT_BASECOLOR" == 0 || "$RIG_ALLOW_FLAT_BASECOLOR" == 1 ]] || {
  echo "RIG_ALLOW_FLAT_BASECOLOR must be 0 or 1" >&2; exit 2;
}
[[ "$RIG_SKIN_MODE" == learned || "$RIG_SKIN_MODE" == learned-smooth || "$RIG_SKIN_MODE" == geometric ]] || {
  echo "RIG_SKIN_MODE must be learned, learned-smooth, or geometric (got $RIG_SKIN_MODE)" >&2; exit 2;
}
[[ "$RIG_SKIN_SMOOTH_ROUNDS" =~ ^[0-9]+$ ]] && (( RIG_SKIN_SMOOTH_ROUNDS >= 1 && RIG_SKIN_SMOOTH_ROUNDS <= 256 )) || {
  echo "RIG_SKIN_SMOOTH_ROUNDS must be an integer in [1,256]" >&2; exit 2;
}
[[ "$RIG_DEFER_POSE_GATE" == 0 || "$RIG_DEFER_POSE_GATE" == 1 ]] || {
  echo "RIG_DEFER_POSE_GATE must be 0 or 1" >&2; exit 2;
}
[[ "$RIG_ALLOW_TRANSFER_MESH_MISMATCH" == 0 || "$RIG_ALLOW_TRANSFER_MESH_MISMATCH" == 1 ]] || {
  echo "RIG_ALLOW_TRANSFER_MESH_MISMATCH must be 0 or 1" >&2; exit 2;
}
case "$RIG_DECODE_MODE" in
  auto|deterministic|sampled) ;;
  *) echo "RIG_DECODE_MODE must be auto, deterministic, or sampled (got $RIG_DECODE_MODE)" >&2; exit 2 ;;
esac
[[ "$RIG_SAMPLE_BEAMS" =~ ^[1-9][0-9]*$ ]] && (( RIG_SAMPLE_BEAMS <= 20 )) || {
  echo "RIG_SAMPLE_BEAMS must be an integer in [1,20]" >&2; exit 2;
}
[[ "$RIG_SAMPLE_SEED" =~ ^[0-9]+$ ]] || { echo "RIG_SAMPLE_SEED must be a non-negative integer" >&2; exit 2; }
[[ "$RIG_PREFIX_PREFILL_MODE" == shared-b1 ]] || {
  echo "RIG_PREFIX_PREFILL_MODE=$RIG_PREFIX_PREFILL_MODE is diagnostic-only; production chain requires shared-b1" >&2; exit 64;
}
[[ "$RIG_KV_TYPE" == auto ]] || {
  echo "production chain owns RIG_KV_TYPE per decode mode; leave it unset" >&2; exit 64;
}
[[ "$RIG_GENERIC_COMPONENT_REPAIR" == 0 || "$RIG_GENERIC_COMPONENT_REPAIR" == 1 ]] || {
  echo "RIG_GENERIC_COMPONENT_REPAIR must be 0 or 1" >&2; exit 2;
}
[[ "$RIG_COMPONENT_BRANCH_REPAIR" == 0 ]] || {
  echo "RIG_COMPONENT_BRANCH_REPAIR is disabled for production: Python may validate poses but may not generate or modify weights" >&2; exit 2;
}
[[ "$RIG_RECOVER_APPENDAGE_TIPS" == 0 || "$RIG_PROFILE" == humanoid ]] || {
  echo "RIG_RECOVER_APPENDAGE_TIPS requires RIG_PROFILE=humanoid" >&2; exit 2;
}
PROVENANCE="$RIG_IN/sampling_provenance.txt"
if [[ "$RIG_ALLOW_TRANSFER_MESH_MISMATCH" == 0 ]]; then
  [[ -s "$PROVENANCE" ]] || {
    echo "missing sampling provenance: regenerate rig inputs with mesh_sample_main for $TEX_GLB" >&2; exit 2;
  }
  sampled_mesh="$(sed -n 's/^mesh_source_path=//p' "$PROVENANCE" | head -1)"
  [[ -n "$sampled_mesh" && -e "$sampled_mesh" ]] || {
    echo "sampling provenance has no readable source mesh: $PROVENANCE" >&2; exit 2;
  }
  [[ "$(readlink -f "$sampled_mesh")" == "$(readlink -f "$TEX_GLB")" ]] || {
    echo "refusing stale rig samples: $PROVENANCE was sampled from $sampled_mesh, not $TEX_GLB" >&2; exit 2;
  }
else
  echo "== explicit cross-mesh skin transfer: provenance mismatch allowed; final post-transfer pose gate is required =="
fi
RIG_GLB="$(dirname "$OUT")/_rig_only.glb"
DUMP=/tmp/skintokens_e2e
# Keep model-weight links out of the mesh-input directory.  That directory is
# immutable provenance for the sampled mesh; writing an r1w sidecar there can
# leave stale/cyclic links across retries and makes a native run appear to
# depend on an old Python capture.
R1W="$DUMP/r1w"
LOG="${OUT%.glb}.rig.log"
MANIFEST="${OUT%.glb}.native-rig-manifest.txt"
DUMP_OUT="${OUT%.glb}.native-rig-dump"
SCORE_FILE="${OUT%.glb}.score.txt"
# Preserve the learned/raw result as evidence.  Generic-only component repair
# is an explicit second candidate after the *same* real-GLB pose gate rejects
# it; a clean normal result remains byte-for-byte the delivery asset.
RAW_OUT="${OUT%.glb}.${RIG_SKIN_MODE}.raw-before-component-repair.glb"
[[ ! -e "$OUT" && ! -e "$RAW_OUT" ]] || { echo "refusing to overwrite output or raw candidate" >&2; exit 2; }

# This uses CUDA for the SkinTokens decode. Resolve the physical 3060 UUID,
# rather than relying on a CUDA or Docker ordinal that can expose the 5060.
OUT_ROOT="${IMAGE_TO_RIG_OUT_ROOT:-/mnt/hdd/3d/avatar-shootout/_shootout_out/runbook_image_to_rig}"
GPU_3060_UUID="${IMAGE_TO_RIG_GPU_3060_UUID:-$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /RTX 3060/ {uuid=$1} END {print uuid}')}"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$GPU_3060_UUID" 2>/dev/null | head -1)"
[[ "$GPU_NAME" == *"RTX 3060"* ]] || { echo "refusing: '$GPU_3060_UUID' is '$GPU_NAME', expected the reserved RTX 3060" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$GPU_3060_UUID"
mkdir -p "$OUT_ROOT" "$(dirname "$OUT")" "$R1W"
exec 9>"$OUT_ROOT/.3060-image-to-rig.lock"
flock -n 9 || { echo "another image-to-rig job owns the 3060" >&2; exit 75; }

# R1's query points are particular to this mesh.  The learned weights stay
# read-only in R1W_SRC, while this run owns its small audit directory.
for f in "$R1W_SRC"/*.npy; do
    [[ -e "$f" ]] || continue
    case "$(basename "$f")" in
      sampled_pc.npy|sampled_feats.npy) continue ;; # query inputs are generated natively, never linked
    esac
    ln -sfn "$f" "$R1W/$(basename "$f")"
done
# R1 now derives its query points locally using the exact fixed-seed NumPy
# PCG64 choice + FPS path.  Do not link or copy Python-produced sampled_pc /
# sampled_feats artifacts into the native inference directory: doing so would
# both reintroduce a hidden assisted input and make repeated runs depend on a
# stale mesh-specific sidecar.

run_native_decode() {
  local kind="$1"
  local kv_type
  local -a decode_args
  case "$kind" in
    deterministic)
      decode_args=(beam nosample prec=fp32 beams="$BEAMS" maxnew=2048)
      kv_type=f32
      ;;
    sampled)
      # Fixed-seed native beam sampling mirrors the upstream recovery regime,
      # but stays entirely in C++ (including R1/R4) and is only considered
      # after the deterministic native beam has produced an auditable reject.
      decode_args=(beam prec=bf16 beams="$RIG_SAMPLE_BEAMS" seed="$RIG_SAMPLE_SEED" maxnew=1534)
      kv_type=bf16
      ;;
    *) return 2 ;;
  esac
  if RIG_KV_TYPE="$kv_type" "$CP/skintokens_e2e" "$RIG_IN" "$QW" "$RIG_GLB" cuda \
      r1=real cond=real r1w="$R1W" prefix_prefill=shared-b1 "$SELECT_FLAG" "${decode_args[@]}" 2>&1 | tee -a "$LOG"; then
    ACTUAL_DECODE_MODE="$kind"
    ACTUAL_DECODE_ARGS="${decode_args[*]}"
    ACTUAL_KV_TYPE="$kv_type"
  else
    return 1
  fi
}

echo "== [1/2] native auto-rig (REAL mesh conditioning, profile=$RIG_PROFILE, decode=$RIG_DECODE_MODE) =="
: >"$LOG"
if [[ "$RIG_DECODE_MODE" == deterministic ]]; then
  run_native_decode deterministic
elif [[ "$RIG_DECODE_MODE" == sampled ]]; then
  run_native_decode sampled
else
  if ! run_native_decode deterministic; then
    echo "== deterministic native beam rejected; trying fixed-seed native sampled beam (beams=$RIG_SAMPLE_BEAMS seed=$RIG_SAMPLE_SEED) ==" | tee -a "$LOG"
    run_native_decode sampled
  fi
fi
# A banked condition is the giraffe oracle, not a valid fallback for a new
# character.  Fail closed rather than silently emitting its skeleton/weights.
rg -Fq '[STAGE R1] using REAL mesh_cond' "$LOG" || { echo "FAIL: R1 fell back to banked conditioning" >&2; exit 1; }
rg -Fq '[STAGE R4] cond_latents (REAL, computed)' "$LOG" || { echo "FAIL: R4 fell back to banked conditioning" >&2; exit 1; }
"$CP/rig_score" "$RIG_GLB" 55 2>/dev/null | grep -iE 'maxfan|TOTAL' | head -1 || true

echo "== [2/2] combine textured mesh + rig -> textured+skinned GLB =="
COMBINE_ARGS=(--profile "$RIG_PROFILE")
COMBINE_ARGS+=(--skin-mode "$RIG_SKIN_MODE")
[[ "$RIG_SKIN_MODE" == learned-smooth ]] && COMBINE_ARGS+=(--skin-smooth-rounds "$RIG_SKIN_SMOOTH_ROUNDS")
[[ "$RIG_PROFILE" == generic && "$RIG_GENERIC_COMPONENT_REPAIR" == 1 ]] && COMBINE_ARGS+=(--generic-component-repair)
[[ "$RIG_RECOVER_APPENDAGE_TIPS" == 1 ]] && COMBINE_ARGS+=(--recover-appendage-tips)
[[ "$RIG_ALLOW_FLAT_BASECOLOR" == 1 ]] && COMBINE_ARGS+=(--allow-flat-basecolor)
"$CP/combine_rig_tex_main" "$TEX_GLB" "$DUMP" "$RAW_OUT" \
    --sampled "$RIG_IN/vertices.npy" \
    --skin    "$DUMP/gen_skin_pred.npy" \
    --joints  "$DUMP/gen_joints.npy" \
    --parents "$DUMP/gen_parents.npy" "${COMBINE_ARGS[@]}" 2>&1 | tee -a "$LOG"

if [[ "$RIG_RECOVER_APPENDAGE_TIPS" == 1 ]]; then
  appendage_report="$("$CP/rig_score" "$RAW_OUT" 55 2>&1 || true)"
  printf '%s\n' "$appendage_report" | tee -a "$LOG"
  [[ "$appendage_report" =~ maxfan=([0-9]+)/([0-9]+) ]] || { echo "FAIL: appendage score lacks fan allowance" >&2; exit 1; }
  appendage_fan="${BASH_REMATCH[1]}"; appendage_limit="${BASH_REMATCH[2]}"
  [[ "$appendage_report" =~ recovered_appendage_tips=([0-9]+) ]] || { echo "FAIL: appendage score lacks recovery count" >&2; exit 1; }
  appendage_tips="${BASH_REMATCH[1]}"
  [[ "$appendage_report" =~ TOTAL=([0-9.]+) ]] || { echo "FAIL: appendage score lacks total" >&2; exit 1; }
  appendage_total="${BASH_REMATCH[1]}"
  (( appendage_tips >= 2 && appendage_fan <= appendage_limit )) && awk "BEGIN { exit !($appendage_total >= 0.50) }" || {
    echo "FAIL: recovered-appendage rig quality gate (tips=$appendage_tips maxfan=$appendage_fan/$appendage_limit total=$appendage_total)" >&2; exit 1;
  }
fi

[[ -x "$RIG_POSE_GATE_PYTHON" ]] || {
  echo "FAIL: rig publication requires the EGL pose-gate Python at $RIG_POSE_GATE_PYTHON" >&2; exit 2;
}
# Rest-pose skeleton scores and semantic names cannot detect a weight field
# that rubber-bands an arm/head/wing when articulated. Gate the written GLB
# with its real inverse-bind matrices and weights. Humanoids use their named
# arm smoke; generic rigs stress every materially weighted joint in the
# geometry-only namespace, without inventing a Wing/Tail/Arm label. A single
# pleasant distal pose is insufficient publication evidence.
if [[ "$RIG_DEFER_POSE_GATE" == 0 ]]; then
  POSE_ARGS=(--show-skeleton --pose-gate)
  [[ "$RIG_PROFILE" == generic ]] && POSE_ARGS=(--generic-all-influential "${POSE_ARGS[@]}")
  if "$RIG_POSE_GATE_PYTHON" "$CP/rig_pose_smoke.py" "$RAW_OUT" "${RAW_OUT%.glb}.pose-gate.png" "${POSE_ARGS[@]}" \
      | tee "${RAW_OUT%.glb}.pose-gate.txt"; then
    cp --reflink=auto -- "$RAW_OUT" "$OUT"
  else
    echo "FAIL: raw rig failed pose gate" >&2
    exit 1
  fi
  # The final output, rather than only the retained raw candidate, is the
  # promotion evidence.  Re-render even after a pass-through copy so callers
  # always find the image beside the published GLB.
  "$RIG_POSE_GATE_PYTHON" "$CP/rig_pose_smoke.py" "$OUT" "${OUT%.glb}.pose-gate.png" "${POSE_ARGS[@]}" \
    | tee "${OUT%.glb}.pose-gate.txt"
else
  echo "== pose gate deferred: caller must gate its final post-processing GLB =="
  cp --reflink=auto -- "$RAW_OUT" "$OUT"
fi

if [[ "$RIG_PROFILE" == generic ]]; then
  generic_report="$("$CP/rig_score" "$OUT" 55 2>&1 || true)"
  printf '%s\n' "$generic_report" | tee -a "$LOG"
  [[ "$generic_report" =~ J=([0-9]+) ]] || { echo "FAIL: generic score lacks joint count" >&2; exit 1; }
  generic_joints="${BASH_REMATCH[1]}"
  [[ "$generic_report" =~ maxfan=([0-9]+) ]] || { echo "FAIL: generic score lacks maxfan" >&2; exit 1; }
  generic_fan="${BASH_REMATCH[1]}"
  [[ "$generic_report" == *"generic_namespace=yes"* ]] || {
    echo "FAIL: generic rig does not have a complete stable skintokens namespace" >&2; exit 1;
  }
  [[ "$generic_report" =~ TOTAL=([0-9.]+) ]] || { echo "FAIL: generic score lacks total" >&2; exit 1; }
  generic_total="${BASH_REMATCH[1]}"
  generic_fan_limit=$(( (generic_joints + 4) / 5 ))
  (( generic_fan_limit < 8 )) && generic_fan_limit=8
  (( generic_fan <= generic_fan_limit )) && awk "BEGIN { exit !($generic_total >= 0.50) }" || {
    echo "FAIL: generic rig quality gate (J=$generic_joints maxfan=$generic_fan limit=$generic_fan_limit total=$generic_total)" >&2; exit 1;
  }
fi

# Write the promotion record only after the written delivery GLB has passed
# every enabled gate.  The decoder's /tmp dump is shared between runs, so copy
# its native artifacts before another invocation can overwrite the evidence.
# A deferred pose gate is deliberately not a publishable result and therefore
# receives no accepted-native manifest.
if [[ "$RIG_DEFER_POSE_GATE" == 0 ]]; then
  [[ -n "${ACTUAL_DECODE_MODE:-}" && -n "${ACTUAL_DECODE_ARGS:-}" ]] || {
    echo "FAIL: native decoder mode was not recorded" >&2; exit 1;
  }
  for f in gen_tokens_beam.npy gen_joints.npy gen_parents.npy gen_skin_pred.npy; do
    [[ -s "$DUMP/$f" ]] || { echo "FAIL: native decode evidence missing: $DUMP/$f" >&2; exit 1; }
  done
  [[ -s "${RAW_OUT%.glb}.pose-gate.txt" && -s "${OUT%.glb}.pose-gate.txt" ]] || {
    echo "FAIL: native delivery lacks final pose-gate evidence" >&2; exit 1;
  }
  "$CP/rig_score" "$OUT" 55 >"$SCORE_FILE" 2>&1 || {
    echo "FAIL: cannot record final rig score: $OUT" >&2; exit 1;
  }
  [[ -s "$SCORE_FILE" ]] || { echo "FAIL: final rig score was empty" >&2; exit 1; }
  [[ ! -e "$DUMP_OUT" && ! -e "$MANIFEST" ]] || {
    echo "FAIL: refusing to overwrite native rig evidence sidecar" >&2; exit 2;
  }
  mkdir "$DUMP_OUT"
  for f in gen_tokens_beam.npy gen_joints.npy gen_parents.npy gen_skin_pred.npy; do
    cp --reflink=auto -- "$DUMP/$f" "$DUMP_OUT/$f"
  done
  if [[ -s "$PROVENANCE" ]]; then
    cp --reflink=auto -- "$PROVENANCE" "$DUMP_OUT/sampling_provenance.txt"
  fi

  source_root="$(cd "$CP/../../.." && pwd)"
  source_revision="$(git -C "$source_root" rev-parse HEAD 2>/dev/null || printf unknown)"
  source_dirty=unknown
  git -C "$source_root" diff --quiet --ignore-submodules -- 2>/dev/null && source_dirty=0 || source_dirty=1
  raw_sha="$(sha256sum "$RAW_OUT" | awk '{print $1}')"
  final_sha="$(sha256sum "$OUT" | awk '{print $1}')"
  component_repair_lines="$(rg -c '^  generic component repair:' "$LOG" 2>/dev/null || true)"
  component_repair_lines="${component_repair_lines:-0}"
  manifest_tmp="${MANIFEST}.tmp.$$"
  {
    printf 'schema_version=1\n'
    printf 'state=accepted-native\n'
    printf 'final_glb=%s\nfinal_glb_sha256=%s\n' "$(basename "$OUT")" "$final_sha"
    printf 'raw_glb=%s\nraw_glb_sha256=%s\nraw_equals_final=%s\n' \
      "$(basename "$RAW_OUT")" "$raw_sha" "$([[ "$raw_sha" == "$final_sha" ]] && printf 1 || printf 0)"
    printf 'source_mesh=%s\nsource_mesh_sha256=%s\n' "$(readlink -f "$TEX_GLB")" "$(sha256sum "$TEX_GLB" | awk '{print $1}')"
    printf 'rig_inputs=%s\n' "$(readlink -f "$RIG_IN")"
    if [[ -s "$PROVENANCE" ]]; then
      printf 'sampling_provenance=%s\nsampling_provenance_sha256=%s\n' \
        "$(basename "$PROVENANCE")" "$(sha256sum "$PROVENANCE" | awk '{print $1}')"
      sed -n -e 's/^\(normalization\|sample_count\|query_count\|seed\|guide\)=/sampling_\1=/p' "$PROVENANCE"
    else
      printf 'sampling_provenance=absent-explicit-cross-mesh-transfer\n'
    fi
    printf 'backend=cuda\ngpu_uuid=%s\ngpu_name=%s\n' "$GPU_3060_UUID" "$GPU_NAME"
    printf 'decoder=skintokens_e2e\nr1_condition=real\nr3_decode_mode=%s\nr3_decode_args=%s\nr3_bf16_activations=%s\nr3_kv_type=%s\nr3_prefix_prefill=shared-b1\nr3_prefix_effective_batch=1\nr3_prefix_production_eligible=yes\nr4_condition=real\n' \
      "$ACTUAL_DECODE_MODE" "$ACTUAL_DECODE_ARGS" "${RIG_BF16_ACTIVATIONS:-0}" "$ACTUAL_KV_TYPE"
    if [[ "$ACTUAL_DECODE_MODE" == deterministic ]]; then
      printf 'r3_attention_prefill=fa2-causal-qwen-gqa\n'
    else
      printf 'r3_attention_prefill=masked-f32-gqa\n'
    fi
    [[ "$ACTUAL_DECODE_MODE" == sampled ]] && printf 'r3_sampler=native-cuda-philox\n'
    printf 'rig_profile=%s\nskin_mode=%s\nskin_smooth_rounds=%s\n' "$RIG_PROFILE" "$RIG_SKIN_MODE" "$RIG_SKIN_SMOOTH_ROUNDS"
    printf 'generic_component_repair_requested=%s\ngeneric_component_repair_log_lines=%s\n' \
      "$RIG_GENERIC_COMPONENT_REPAIR" "$component_repair_lines"
    printf 'rig_log=%s\nrig_log_sha256=%s\nscore_file=%s\nscore_file_sha256=%s\n' \
      "$(basename "$LOG")" "$(sha256sum "$LOG" | awk '{print $1}')" \
      "$(basename "$SCORE_FILE")" "$(sha256sum "$SCORE_FILE" | awk '{print $1}')"
    printf 'raw_pose_gate=%s\nraw_pose_gate_sha256=%s\nfinal_pose_gate=%s\nfinal_pose_gate_sha256=%s\n' \
      "$(basename "${RAW_OUT%.glb}.pose-gate.txt")" "$(sha256sum "${RAW_OUT%.glb}.pose-gate.txt" | awk '{print $1}')" \
      "$(basename "${OUT%.glb}.pose-gate.txt")" "$(sha256sum "${OUT%.glb}.pose-gate.txt" | awk '{print $1}')"
    for f in "$DUMP_OUT"/*; do printf 'dump_%s_sha256=%s\n' "$(basename "$f")" "$(sha256sum "$f" | awk '{print $1}')"; done
    printf 'binary_skintokens_e2e_sha256=%s\nbinary_combine_rig_tex_main_sha256=%s\nbinary_rig_score_sha256=%s\npose_gate_script_sha256=%s\n' \
      "$(sha256sum "$CP/skintokens_e2e" | awk '{print $1}')" \
      "$(sha256sum "$CP/combine_rig_tex_main" | awk '{print $1}')" \
      "$(sha256sum "$CP/rig_score" | awk '{print $1}')" \
      "$(sha256sum "$CP/rig_pose_smoke.py" | awk '{print $1}')"
    printf 'source_revision=%s\nsource_dirty=%s\n' "$source_revision" "$source_dirty"
  } >"$manifest_tmp"
  mv -f -- "$manifest_tmp" "$MANIFEST"
  echo "== native rig evidence -> $MANIFEST =="
fi

echo "== DONE -> $OUT =="
