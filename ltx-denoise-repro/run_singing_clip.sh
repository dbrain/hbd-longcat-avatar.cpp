#!/usr/bin/env bash
# =============================================================================
# run_singing_clip.sh — self-contained repro of the LTX-2.3 "singing continuation"
# clip used to develop the audio-sync + identity-flash fixes (2026-07-08 session).
#
# WHAT IT DOES (the whole recipe, no discovery needed):
#   - Renders an N-segment LTX-2.3 video chain of a young man singing to camera
#     (continuous locked-off close-up prompt — isolates seams + shows lip-sync).
#   - Drives per-segment lip-sync with slices of song.wav cut at the SEGMENT ADVANCE
#     interval (F-DROP)/fps, NOT the naive segment interval — this is the fix that
#     makes the video lip-sync to a *continuous* song timeline (else ~1s of song is
#     skipped at every seam).
#   - Muxes the ORIGINAL continuous song onto the final stitch (discards the VAE
#     per-segment audio) => pristine audio, no seam click, no end-fuzz.
#
# WHY THE MAGIC NUMBERS: each continuation segment renders F=97 frames but the stitch
#   drops DROP (~24) overlap frames at the seam, so it only ADDS (F-DROP)=~73 frames
#   (~3.04s) to the final. DROP≈24 is a model constant (guide 17 + ~7 settle frames);
#   auto-trim wants ~23-24 uniformly, so pinning it generalises. Advance/mux math is
#   derived from F, DROP, fps below — change one and it all stays consistent.
#
# KNOBS (env vars, all optional):
#   SEGMENTS=2         number of chain segments (2 = fast, 1 seam; 8 = full song)
#   HIRES=1            1 = 1080p (x2 upscale + refine); 0 = base 960x544 (fast, no refine)
#   WIDTH=960 HEIGHT=544
#                      base render size. With HIRES=1 this is the pre-upscale size; with HIRES=0
#                      this is the final output size.
#   MAXV=9             --max-vram budget. 9 is the current locked speed/VRAM point; 10 was faster
#                      but exceeded the old <=11.5GB co-resident target in the single-segment sweep.
#   DROP=24            overlap frames dropped per seam (pins the advance; must match slicing)
#   STEPS=8 REFSTEPS=3 base / hires-refine sampling steps
#   SAMPLING_METHOD=euler_a
#                      locked sampler for base + hires/refine. Explicit so this prod example does
#                      not depend on model defaults.
#   REFINE_SIGMAS="0.909375,0.725,0.421875,0.0"
#                      hires refine sigma schedule. 0.909=official/sharp but re-rolls the
#                      face per segment (identity flash). LOWER sigma0 (e.g. 0.4,0.28,0.15,0.0)
#                      = gentler, more identity-stable, faster. This is the KNOB to sweep for
#                      the identity issue (also exposed per-request as hires.custom_sigmas).
#                      Set REFINE_SIGMAS="" to omit --hires-sigmas and use the generated hires
#                      schedule path; combine with USE_HIRES_STRENGTH=1 for two-stage LTX tests.
#   REFINE_DENOISING_STRENGTH=0.7
#                      highres denoising strength used by the generated schedule path.
#   USE_HIRES_STRENGTH=1
#                      diagnostic: in two-stage LTX, do not inject the official custom sigma
#                      vector, so REFINE_DENOISING_STRENGTH actually controls schedule trimming.
#   REFINE_CONST_SEED=1  pin refine noise to a constant across segments (halves the identity
#                      flash — the per-segment base+seg seed was ~half the cause; opt-in engine
#                      flag LTX_REFINE_CONST_SEED, needs the build with commit 7b13cf0+).
#   SKIP_AUDIO_DECODE=1 skip generated audio latent decode. This harness muxes the original
#                      continuous song at the end, so decoded per-segment model audio is unused.
#   VAE_SPATIAL_TILES=2x2
#                      video VAE decode spatial tiling. 2x2 is the locked conservative default;
#                      2x1/1x2 are quality-neutral candidates to validate for fewer tile passes
#                      if VRAM allows.
#   LONGCAT_OFFLOAD_PROFILE=1
#                      emit aggregated graph-cut offload/copy/compute timings. Adds syncs, so use
#                      for profiling rather than headline wall comparisons.
#   LONGCAT_PERSIST_GRAPH_INPUTS=1
#                      keep repeated graph-cut external inputs resident for the duration of each
#                      graph-cut call. Quality-neutral copy avoidance; set 0 to disable.
#   LONGCAT_ATTN_TILES=2
#                      split large self-attention noise queries into exact row tiles. Quality-neutral
#                      and validated flat across the 3-segment MAXV=9 path; set 1 to disable.
#   LONGCAT_FFN_TILE_TOKENS=8192
#                      FFN token tile size. 8192 reduces tile reassembly concat work versus 4096,
#                      with 3-segment peak unchanged at the current MAXV=9 envelope.
#   LTXAV_CHAIN_CUDNN_RESET=1
#                      A/B cuDNN plan-cache release between chain segments. Quality-neutral; may
#                      trade lower cross-segment VRAM for plan rebuild cost.
#   LTXAV_REFINE_MAX_VRAM=10
#                      use a different graph budget only for the hires/refine denoise. This keeps
#                      base sampling at MAXV while testing whether a fatter refine graph cut is a
#                      speed win without raising the whole render's peak VRAM.
#   OUT_NAME=singing_clip.webm     output filename
#   OUTDIR=<repo>/ltx-denoise-repro/_singing_out
#
# KNOWN OPEN ISSUE (see HANDOFF-seg2-reserve-FRESH.md in kobbler): at 1080p the refine
#   re-rolls the singer's identity/skin-tone per segment. REFINE_CONST_SEED=1 halves it;
#   the residual is nvfp4 non-determinism (a deterministic refine is real engine work — you
#   CANNOT force it via env, both GGML_CUDNN_ATTN=0 and GGML_NVFP4_CUBLASLT=0 crash this build).
#
# USAGE:
#   bash run_singing_clip.sh                                  # default 2-seg 1080p
#   SEGMENTS=8 bash run_singing_clip.sh                       # full 8-seg
#   HIRES=0 bash run_singing_clip.sh                          # fast base-res
#   REFINE_SIGMAS="0.4,0.28,0.15,0.0" REFINE_CONST_SEED=1 bash run_singing_clip.sh   # identity sweep
# =============================================================================
set -euo pipefail

WT=${WT:-/home/dbrain/dev/longcat-avatar-ltxdenoise}
LTX2=${LTX2:-/home/dbrain/dev/longcat-avatar.cpp/models/ltx2}
SONG=${SONG:-$LTX2/_drive/music_video/song.wav}        # 30s stereo 48k
BUILDER=${BUILDER:-longcat-avatar-dev:builder-cudnn-ff}
BIN=/src/build-cudnn/bin/sd-cli
DIT=${DIT:-nvfp4-imatrix-dev050.gguf}
FFMPEG=${FFMPEG:-linuxserver/ffmpeg}
GPU=${GPU:-1}                                          # device index (5060 Ti = 1)

SEGMENTS=${SEGMENTS:-2}; HIRES=${HIRES:-1}; DROP=${DROP:-24}
WIDTH=${WIDTH:-960}; HEIGHT=${HEIGHT:-544}
MAXV=${MAXV:-9}
STEPS=${STEPS:-8}; REFSTEPS=${REFSTEPS:-3}
SAMPLING_METHOD=${SAMPLING_METHOD:-euler_a}
if [ "${REFINE_SIGMAS+x}" != x ]; then
  REFINE_SIGMAS=0.909375,0.725,0.421875,0.0
fi
REFINE_DENOISING_STRENGTH=${REFINE_DENOISING_STRENGTH:-0.7}
USE_HIRES_STRENGTH=${USE_HIRES_STRENGTH:-0}
REFINE_CONST_SEED=${REFINE_CONST_SEED:-0}
SKIP_AUDIO_DECODE=${SKIP_AUDIO_DECODE:-1}
VAE_SPATIAL_TILES=${VAE_SPATIAL_TILES:-2x2}
LONGCAT_PERSIST_GRAPH_INPUTS=${LONGCAT_PERSIST_GRAPH_INPUTS:-1}
LONGCAT_ATTN_TILES=${LONGCAT_ATTN_TILES:-2}
OUT_NAME=${OUT_NAME:-singing_clip.webm}
OUTDIR=${OUTDIR:-$WT/ltx-denoise-repro/_singing_out}
F=97; FPS=24
mkdir -p "$OUTDIR"

PROMPT="Locked-off medium close-up portrait of a young man with tousled dark brown hair and light stubble, wearing a blue denim jacket over a white t-shirt, centred against a soft warm-lit plain studio background. He looks straight into the camera and sings along to the song with clear, expressive, exaggerated mouth movements, enunciating every word, lips and jaw opening and closing distinctly in time with the vocals. He stays centred and holds his position, only his head, mouth and shoulders moving as he sings. The camera is completely static and locked off — no camera movement, no zoom, no cuts, one continuous take."
printf '%s\n' "$PROMPT" > "$OUTDIR/prompt.txt"

# ---- advance-corrected driving slices + the continuous mux song --------------
read ADV WIN TOTAL < <(python3 -c "F=$F;D=$DROP;fps=$FPS;S=$SEGMENTS;print((F-D)/fps, F/fps, (F+(S-1)*(F-D))/fps)")
echo "[slice] F=$F DROP=$DROP fps=$FPS -> advance=${ADV}s window=${WIN}s total=${TOTAL}s (segments=$SEGMENTS)"
ADIR="$OUTDIR/adir"; rm -rf "$ADIR"; mkdir -p "$ADIR"
for n in $(seq 0 $((SEGMENTS-1))); do
  START=$(python3 -c "print($n*($F-$DROP)/$FPS)")
  docker run --rm -v "$(dirname "$SONG"):/s" -v "$ADIR:/o" --entrypoint ffmpeg "$FFMPEG" -y -v error \
    -i "/s/$(basename "$SONG")" -ss "$START" -t "$WIN" -ac 1 -ar 16000 "/o/aud_$n.wav" >/dev/null 2>&1
done
docker run --rm -v "$(dirname "$SONG"):/s" -v "$OUTDIR:/o" --entrypoint ffmpeg "$FFMPEG" -y -v error \
  -i "/s/$(basename "$SONG")" -ss 0 -t "$TOTAL" -ac 2 -ar 48000 "/o/mux_song.wav" >/dev/null 2>&1

# ---- hires args --------------------------------------------------------------
HIRES_ARGS=""
if [ "$HIRES" = 1 ]; then
  HIRES_ARGS="--hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires-upscalers-dir /ltx2/latent_upscale_models --hires-steps $REFSTEPS --hires-denoising-strength $REFINE_DENOISING_STRENGTH"
  [ -n "$REFINE_SIGMAS" ] && HIRES_ARGS="$HIRES_ARGS --hires-sigmas $REFINE_SIGMAS"
fi

# ---- prod engine env (the beat-comfy nvfp4/fp8/cuDNN stack + leak-fix eviction) ----
ENV=( -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1 -e LTX_DIT_F16=1
  -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks
  -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1
  -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=0 -e LONGCAT_FFN_TILE_TOKENS=8192 -e LONGCAT_ENCODE_MAX_VRAM=6.5
  -e LONGCAT_NO_PREFETCH_POOL=1 -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 -e LONGCAT_NO_OFFLOAD_PIPELINING=0 -e LONGCAT_DIT_NO_MMAP=0
  -e LTXAV_VAE_LAZY=1 -e LTXAV_DIT_FREE_DURING_DECODE=1 -e LONGCAT_VRAM_BREAKDOWN=1
  -e LTX_VAE_HEAD_F32=1 -e LTX_VAE_CONV3D_WTILES=16 -e LTX_VAE_CONV3D_HTILES=8 -e LTX_VAE_DECODE_F16=1
  -e LTX_VAE_SPATIAL_TILES=$VAE_SPATIAL_TILES -e LTX_VAE_SPATIAL_OVERLAP=4
  -e LTX_CUSTOM_SIGMAS=1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0
  -e LTXAV_CHAIN_OVERLAP_DROP=$DROP )
[ "$REFINE_CONST_SEED" != 0 ] && ENV+=( -e LTX_REFINE_CONST_SEED=$REFINE_CONST_SEED )
[ "$USE_HIRES_STRENGTH" != 0 ] && ENV+=( -e LTXAV_TWOSTAGE_USE_HIRES_STRENGTH=$USE_HIRES_STRENGTH )
[ "$SKIP_AUDIO_DECODE" != 0 ] && ENV+=( -e LTXAV_SKIP_AUDIO_DECODE=1 )
for passthrough in LONGCAT_OFFLOAD_PROFILE LONGCAT_PROFILE LONGCAT_CONCAT_PROFILE LONGCAT_CONCAT_PROFILE_TOP LONGCAT_CONT_PROF LONGCAT_ATTN_TILES LONGCAT_FFN_TILE_TOKENS LTX_ATTN_QTILE GGML_CUDNN_ATTN_BUCKET GGML_CUDNN_ATTN_BUCKETS GGML_CUDNN_OP_TRACE GGML_CUDNN_ATTN_EXEC_TRACE GGML_FP8_ATTN GGML_FP8_ATTN_BC GGML_FP8_ATTN_LDM LTXAV_CHAIN_CUDNN_RESET LTXAV_CHAIN_POOL_TRIM LTXAV_END_RENDER_RECLAIM LONGCAT_SHARED_RESIDENT_MAX_MB LTXAV_REFINE_MAX_VRAM LONGCAT_PERSIST_GRAPH_INPUTS; do
  if [ "${!passthrough+x}" = x ]; then
    ENV+=( -e "$passthrough=${!passthrough}" )
  fi
done

INNER="stdbuf -oL -eL $BIN -M vid_gen --diffusion-model /ltx2/$DIT \
  --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
  --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  --lora-model-dir /ltx2/loras \
  --ltx-chain-segments $SEGMENTS --ltx-chain-prompts /work/prompt.txt --ltx-chain-audio-dir /work/adir --cont-latent-frames 3 \
  -W $WIDTH -H $HEIGHT --video-frames $F --fps $FPS --steps $STEPS --sampling-method $SAMPLING_METHOD --cfg-scale 1.0 --diffusion-fa \
  $HIRES_ARGS --offload-to-cpu --mmap --max-vram $MAXV -s 42 -v -o /work/_raw_$OUT_NAME"

LOG="$OUTDIR/render.log"
VRAM_LOG="$OUTDIR/vram.log"
echo "[render] DIT=$DIT MAXV=$MAXV REFINE_MAXV=${LTXAV_REFINE_MAX_VRAM:-<same>} SEGMENTS=$SEGMENTS HIRES=$HIRES WIDTH=$WIDTH HEIGHT=$HEIGHT DROP=$DROP sampler=$SAMPLING_METHOD sigmas=${REFINE_SIGMAS:-<generated>} denoise_strength=$REFINE_DENOISING_STRENGTH use_hires_strength=$USE_HIRES_STRENGTH const_seed=$REFINE_CONST_SEED skip_audio_decode=$SKIP_AUDIO_DECODE vae_tiles=$VAE_SPATIAL_TILES persist_graph_inputs=$LONGCAT_PERSIST_GRAPH_INPUTS attn_tiles=$LONGCAT_ATTN_TILES"
rm -f "$LOG" "$VRAM_LOG"
: > "$VRAM_LOG"
( while :; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null >> "$VRAM_LOG"; sleep 1; done ) &
VRAM_PID=$!
T0=$(date +%s)
set +e
docker run --rm --gpus "\"device=$GPU\"" "${ENV[@]}" \
  -v "$WT:/src" -v "$LTX2:/ltx2" -v "$OUTDIR:/work" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src \
  "$BUILDER" bash -c "$INNER" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
kill "$VRAM_PID" 2>/dev/null || true
WALL=$((T1-T0))
PEAK_DRIVER=$(perl -ne 'while(/driver_used=(\d+) MB/g){$m=$1 if $1>$m} END{print $m || 0}' "$LOG")
PEAK_CUDNN=$(perl -ne 'while(/used ([0-9.]+) MB/g){$m=$1 if $1>$m} END{print $m || 0}' "$LOG")
PEAK_SMI=$(sort -n "$VRAM_LOG" 2>/dev/null | tail -1)
echo "[metrics] render_rc=$RC wall=${WALL}s peak_driver=${PEAK_DRIVER:-0}MB peak_cudnn=${PEAK_CUDNN:-0}MB peak_smi=${PEAK_SMI:-0}MiB log=$LOG"
if [ "$RC" != 0 ]; then
  exit "$RC"
fi

# ---- mux the ORIGINAL continuous song onto the stitched video ----------------
docker run --rm -v "$OUTDIR:/w" --entrypoint ffmpeg "$FFMPEG" -y -v error \
  -i "/w/_raw_$OUT_NAME" -i /w/mux_song.wav -map 0:v -map 1:a -c:v copy -c:a libopus -shortest "/w/$OUT_NAME" >/dev/null 2>&1
DUR=$(docker run --rm -v "$OUTDIR:/w" --entrypoint ffprobe "$FFMPEG" -v error -show_entries format=duration -of default=nw=1:nk=1 "/w/$OUT_NAME")
echo "[done] $OUTDIR/$OUT_NAME  (${DUR}s)  — original song muxed, in sync across seams"
echo "[view] copy to the LAN eye-test page: cp $OUTDIR/$OUT_NAME /home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise/sweep2/  then open http://10.0.0.208:8077/ltx_denoise/sweep2/$OUT_NAME"
