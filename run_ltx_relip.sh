#!/usr/bin/env bash
# LTX-2.3 V2V LIPDUB RELIP — the REAL audio-attach: take an existing Wan2.2 video,
# inject the vocal, re-lip the mouth via the lipdub IC-LoRA. KEEPS the video+motion,
# only changes the lips (apply_ltxav_video_relip_reference: whole clip -> frozen
# timeline-aligned reference; lipdub LoRA + frozen drive-audio constrain the regen).
#
# Uses the SIBLING repo binary (~/dev/longcat-avatar.cpp) which HAS the relip path
# (control_frames). The wan22-worktree binary does NOT (prints "not implemented").
# Prod VAE recipe = 1x1 full-frame tiling (zero seams) + max-vram 7.5 so the DiT
# offloads to CPU before the full-frame VAE decode (else 12.8GB decode buffer OOMs).
#   SRC=cont_seg1  ./run_ltx_relip.sh      # SRC = a dir of Wan2.2 frames (8k+1 count)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"; REPO="$PWD"
BUILDER="longcat-avatar-dev:builder-cudnn-ff"        # ffmpeg present (harmless here; consistent)
LTXSRC="/home/dbrain/dev/longcat-avatar.cpp"          # binary WITH relip
LTX2="$LTXSRC/models/ltx2"                            # prod weights
SRC=${SRC:-cont_seg1}                                 # dir of Wan2.2 frames (relative to repo)
WAV=${WAV:-models/_drive/song_vocals_16k.wav}
MUXWAV=${MUXWAV:-models/_drive/song.wav}
W=${W:-1280}; H=${H:-704}; FR=${FR:-25}; FPS=${FPS:-24}; MAXV=${MAXV:-9.5}; VTILE=${VTILE:-1x1}; STEPS=${STEPS:-8}
TAG=${TAG:-relip}; OUT="$REPO/mvp_out/$TAG"; mkdir -p "$OUT"
PROMPT=${PROMPT:-"A person singing on a neon-lit city street at night, cinematic, photorealistic <lora:ltx-2.3-22b-ic-lora-lipdub-0.9:1.0>"}
CFG=${CFG:-1.0}                                       # >1 = enable CFG guidance (test if distilled responds)
NEG=${NEG:-}                                          # negative prompt for CFG (e.g. "static, closed mouth, silent")
A2V=${A2V:-1.0}                                       # A2V (audio->video) modality guidance scale; >1 drives lip-sync
A2VRAMP=${A2VRAMP:-1.0}                                # A2V schedule: scale fraction kept at last step (1.0=const, 0=ramp off late)
ENC=${ENC:-0.25}                                      # relip reference ENCODE spatial tile (1.0 = no tiling; ghost A/B)
REFS=${REFS:-1.0}                                     # reference_strength (1.0=frozen; <1 loosens ref: less ghost/clamp)
TT=${TT:-4}                                           # decode temporal-tile frames (PROD=4; 0=OFF whole-clip)
TOV=${TOV:-1}                                          # decode temporal-tile OVERLAP (PROD=1; >0 kills seam/ghost)
DIT=${DIT:-nvfp4-CLEAN.gguf}                           # PROD model (was q4k-distilled = slow, no accel stack)
# Prod compute/offload recipe — pulled verbatim from the running kobbler-ltx-video-1 container
# (docker exec ... env). nvfp4 cuBLASLt + FP8 FFN + F16 residual + cuDNN attn/conv3d + fusion
# glue + offload-overlap + shared/VAE residency + the distilled custom sigma schedule.
PRODENV=(
  -e GGML_NVFP4_CUBLASLT=1 -e GGML_NVFP4_QUANT_TWOLEVEL=1
  -e GGML_FP8_FFN=1 -e GGML_FP8_LAYERS=transformer_blocks
  -e LTX_DIT_F16=1
  -e GGML_CUDA_BIAS_GELU_FUSE=1 -e GGML_CUDA_BIAS_RMS_FUSE=1 -e GGML_CUDA_F16_BCAST_FUSE=1 -e GGML_CUDA_RMS_MOD_FUSE=1
  -e GGML_CUDNN_ATTN=1 -e GGML_CUDNN_ATTN_F16_OUT=1 -e GGML_CUDNN_CONV3D=1
  -e LONGCAT_NO_OFFLOAD_PIPELINING=${NO_OVERLAP:-0} -e LONGCAT_OFFLOAD_PREFETCH_THREAD=${PREFETCH:-1} -e LONGCAT_NO_PREFETCH_POOL=1
  -e LONGCAT_SHARED_RESIDENT=${SHARED_RESIDENT:-1} -e LONGCAT_VAE_KEEP_RESIDENT=${VAE_RESIDENT:-1}
  -e LONGCAT_FFN_TILE_TOKENS=4096 -e LONGCAT_ENCODE_MAX_VRAM=6.5 -e LONGCAT_DIT_NO_MMAP=0
  -e LTXAV_END_RENDER_RECLAIM=1 -e LTXAV_CHAIN_POOL_TRIM=1
  -e "LTX_CUSTOM_SIGMAS=${LTX_CUSTOM_SIGMAS:-1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875,0.0}"
)
if [ "$TT" = "0" ]; then TTARG=""; else TTARG="--temporal-tiling --extra-tiling-args temporal_tile_frames=$TT,temporal_tile_overlap=$TOV"; fi
if [ "${TWOSTAGE:-0}" = "1" ]; then HIRESARG="--hires --hires-upscalers-dir /ltx2/latent_upscale_models --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1"; else HIRESARG=""; fi
nf=$(ls "$SRC"/*.png 2>/dev/null | wc -l); echo "source frames: $nf in $SRC"

vf="$OUT/.vram"; echo 0 >"$vf"; ( while :; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null); [ -n "$u" ]&&[ "$u" -eq "$u" ] 2>/dev/null&&{ m=$(cat "$vf"); [ "$u" -gt "$m" ]&&echo "$u">"$vf"; }; sleep 0.4; done ) & sp=$!
t0=$(date +%s.%N)
docker run --rm --gpus "\"device=1\"" \
  "${PRODENV[@]}" \
  ${SAVELAT:+-e LTX_SAVE_LATENTS=/src/$SAVELAT} \
  -e LTXAV_A2V_GUIDANCE=$A2V -e LTXAV_A2V_RAMP_END=$A2VRAMP -e LTXAV_RELIP_ENCODE_TILE=$ENC -e LTXAV_RELIP_REF_STRENGTH=$REFS \
  -e LTXAV_RELIP_REF_DOWNSCALE=${REFDS:-1} -e LTXAV_FREE_TE_COMPUTE=${FREETE:-1} -e LTXAV_FREE_TE_PARAMS=${FREEPARAMS:-0} -e LTXAV_RELIP_REF_TSTRIDE=${TSTRIDE:-1} \
  -e LTXAV_RELIP_TWOSTAGE=${TWOSTAGE:-0} \
  -e LTXAV_RELIP_ENCODE_TFRAMES=${TFRAMES:-0} -e LTXAV_TWOSTAGE_FREE_STAGE1=${FREES1:-1} -e LTXAV_TWOSTAGE_FREE_UNUSED=${FREEUNUSED:-1} \
  -v "$LTXSRC:/ltxsrc" -v "$REPO:/src" -v "$LTX2:/ltx2" -v /mnt/ssd/models:/mnt/ssd/models:ro -w /src "$BUILDER" \
  stdbuf -oL -eL /ltxsrc/build/bin/sd-cli -M vid_gen \
  --diffusion-model       /ltx2/$DIT \
  --vae                   /ltx2/vae/ltx-2.3-22b-distilled_video_vae.safetensors \
  --audio-vae             /ltx2/vae/ltx-2.3-22b-distilled_audio_vae-ENC-f16.gguf \
  --llm                   /ltx2/gemma-3-12b-it-UD-Q4_K_XL.gguf \
  --embeddings-connectors /ltx2/text_encoders/ltx-2.3-22b-distilled_embeddings_connectors.safetensors \
  --lora-model-dir        /ltx2/loras \
  --control-video         "/src/$SRC" \
  --drive-audio           "/src/$WAV" \
  -p "$PROMPT" ${NEG:+-n "$NEG"} \
  ${SEED:+-s $SEED} \
  -W $W -H $H --video-frames $FR --fps $FPS \
  --sampling-method euler --steps $STEPS --cfg-scale $CFG --diffusion-fa \
  --vae-tiling --vae-relative-tile-size $VTILE \
  $TTARG \
  $HIRESARG \
  --offload-to-cpu ${MMAPARG:---mmap} --max-vram $MAXV -v \
  -o /src/mvp_out/$TAG/relip.webm > "$OUT/relip.log" 2>&1
rc=$?; t1=$(date +%s.%N); kill "$sp" 2>/dev/null
wall=$(awk "BEGIN{printf \"%.1f\",$t1-$t0}"); peak=$(cat "$vf" 2>/dev/null); rm -f "$vf"
echo ">> RELIP ${W}x${H} ${FR}f: ${wall}s peak ${peak}MiB rc=$rc"
ls "$OUT"/*.webm 2>/dev/null
if [ -f "$OUT/relip.webm" ]; then
  ffmpeg -y -i "$OUT/relip.webm" -i "/src/$MUXWAV" -map 0:v -map 1:a -c:v libx264 -crf 18 -pix_fmt yuv420p -c:a aac -shortest \
    "$REPO/tools/lightx2v-test/out/ltx_relip_$TAG.mp4" -loglevel error 2>/dev/null && echo "muxed -> ltx_relip_$TAG.mp4"
fi
grep -aiE "RELIP|relip|control_frames|reference|drive|lora|out of memory|error|snapping|saved" "$OUT/relip.log" 2>/dev/null|grep -aviE 0x|tail -12
