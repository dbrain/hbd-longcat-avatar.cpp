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
#   DROP=24            overlap frames dropped per seam (pins the advance; must match slicing)
#   STEPS=8 REFSTEPS=3 base / hires-refine sampling steps
#   REFINE_SIGMAS="0.909375,0.725,0.421875,0.0"
#                      hires refine sigma schedule. 0.909=official/sharp but re-rolls the
#                      face per segment (identity flash). LOWER sigma0 (e.g. 0.4,0.28,0.15,0.0)
#                      = gentler, more identity-stable, faster. This is the KNOB to sweep for
#                      the identity issue (also exposed per-request as hires.custom_sigmas).
#   REFINE_CONST_SEED=1  pin refine noise to a constant across segments (halves the identity
#                      flash — the per-segment base+seg seed was ~half the cause; opt-in engine
#                      flag LTX_REFINE_CONST_SEED, needs the build with commit 7b13cf0+).
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
DIT=${DIT:-nvfp4-CLEAN.gguf}
FFMPEG=${FFMPEG:-linuxserver/ffmpeg}
GPU=${GPU:-1}                                          # device index (5060 Ti = 1)

SEGMENTS=${SEGMENTS:-2}; HIRES=${HIRES:-1}; DROP=${DROP:-24}
STEPS=${STEPS:-8}; REFSTEPS=${REFSTEPS:-3}
REFINE_SIGMAS=${REFINE_SIGMAS:-0.909375,0.725,0.421875,0.0}
REFINE_CONST_SEED=${REFINE_CONST_SEED:-0}
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
[ "$HIRES" = 1 ] && HIRES_ARGS="--hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires-upscalers-dir /ltx2/latent_upscale_models --hires-steps $REFSTEPS --hires-sigmas $REFINE_SIGMAS"

# ---- prod engine env (the beat-comfy nvfp4/fp8/cuDNN stack + leak-fix eviction) ----
ENV=( -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1 -e LTX_DIT_F16=1
  -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1 -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks
  -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1
  -e LONGCAT_SHARED_RESIDENT=1 -e LONGCAT_VAE_KEEP_RESIDENT=0 -e LONGCAT_FFN_TILE_TOKENS=4096 -e LONGCAT_ENCODE_MAX_VRAM=6.5
  -e LONGCAT_NO_PREFETCH_POOL=1 -e LONGCAT_OFFLOAD_PREFETCH_THREAD=1 -e LONGCAT_NO_OFFLOAD_PIPELINING=0 -e LONGCAT_DIT_NO_MMAP=0
  -e LTXAV_VAE_LAZY=1 -e LTXAV_DIT_FREE_DURING_DECODE=1 -e LONGCAT_VRAM_BREAKDOWN=1
  -e LTX_VAE_HEAD_F32=1 -e LTX_VAE_CONV3D_WTILES=16 -e LTX_VAE_CONV3D_HTILES=8 -e LTX_VAE_DECODE_F16=1
  -e LTX_VAE_SPATIAL_TILES=2x2 -e LTX_VAE_SPATIAL_OVERLAP=4
  -e LTX_CUSTOM_SIGMAS=1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0
  -e LTXAV_CHAIN_OVERLAP_DROP=$DROP )
[ "$REFINE_CONST_SEED" != 0 ] && ENV+=( -e LTX_REFINE_CONST_SEED=$REFINE_CONST_SEED )

INNER="stdbuf -oL -eL $BIN -M vid_gen --diffusion-model /ltx2/$DIT \
  --vae /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors --audio-vae /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
  --llm /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  --lora-model-dir /ltx2/loras \
  --ltx-chain-segments $SEGMENTS --ltx-chain-prompts /work/prompt.txt --ltx-chain-audio-dir /work/adir --cont-latent-frames 3 \
  -W 960 -H 544 --video-frames $F --fps $FPS --steps $STEPS --cfg-scale 1.0 --diffusion-fa \
  $HIRES_ARGS --offload-to-cpu --mmap --max-vram 7 -s 42 -v -o /work/_raw_$OUT_NAME"

echo "[render] SEGMENTS=$SEGMENTS HIRES=$HIRES DROP=$DROP sigmas=$REFINE_SIGMAS const_seed=$REFINE_CONST_SEED"
docker run --rm --gpus "\"device=$GPU\"" "${ENV[@]}" \
  -v "$WT:/src" -v "$LTX2:/ltx2" -v "$OUTDIR:/work" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src \
  "$BUILDER" bash -c "$INNER"

# ---- mux the ORIGINAL continuous song onto the stitched video ----------------
docker run --rm -v "$OUTDIR:/w" --entrypoint ffmpeg "$FFMPEG" -y -v error \
  -i "/w/_raw_$OUT_NAME" -i /w/mux_song.wav -map 0:v -map 1:a -c:v copy -c:a libopus -shortest "/w/$OUT_NAME" >/dev/null 2>&1
DUR=$(docker run --rm -v "$OUTDIR:/w" --entrypoint ffprobe "$FFMPEG" -v error -show_entries format=duration -of default=nw=1:nk=1 "/w/$OUT_NAME")
echo "[done] $OUTDIR/$OUT_NAME  (${DUR}s)  — original song muxed, in sync across seams"
echo "[view] copy to the LAN eye-test page: cp $OUTDIR/$OUT_NAME /home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise/sweep2/  then open http://10.0.0.208:8077/ltx_denoise/sweep2/$OUT_NAME"
