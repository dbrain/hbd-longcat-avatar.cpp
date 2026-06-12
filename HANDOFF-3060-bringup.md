# 3060 box bring-up — Wan2.2 + InfiniteTalk + VACE port

Branch: **`wan22-infinitetalk`** on `git@github.com:dbrain/hbd-longcat-avatar.cpp`.
All filesystem paths/assets referenced here live on **host `10.0.0.151`** (the HX370 laptop where
this was developed) under `~/dev/wan22-infinitetalk/`. The port is CPU-validated at 256px; the 3060
is where it goes to render at real resolution/speed.

## TL;DR
The pipeline logic is **backend-agnostic ggml** — everything carries to CUDA unchanged. The only
CPU-isms are (a) the `LONGCAT_NO_FUSED_ROPE=1` workaround you **drop** on GPU, and (b) the
`-DSD_CUDA=OFF` build flag you **flip on**. The converted GGUFs are ggml format and load on CUDA
directly — copy them, don't re-convert.

## 1. Get the code
```
git clone -b wan22-infinitetalk git@github.com:dbrain/hbd-longcat-avatar.cpp wan22-infinitetalk
cd wan22-infinitetalk && git submodule update --init --recursive
```

## 2. Build WITH CUDA (the laptop built CPU-only)
```
cmake -B build -DSD_CUDA=ON          # laptop used -DSD_CUDA=OFF -DSD_HIPBLAS=OFF
cmake --build build -j
```
The CPU-only build needed `SD_USE_CUDA` guards around a few calls; the CUDA build is the upstream
default and our changes never touched backend code, so it *should* compile clean — but this is
**unverified on actual CUDA hardware**, so expect to shake out a guard or two on first build.

## 3. Models — copy from 10.0.0.151 (don't re-convert; ~70GB of GGUFs already built)
`scp -r dbrain@10.0.0.151:~/dev/wan22-infinitetalk/models/ ./models/` (the raw `models/dl/` source
checkpoints are also there if you'd rather re-convert). Manifest of what's already converted:

| file | what |
|------|------|
| `infinitetalk-14b-q4_k.gguf` (10.7GB) | Wan2.1-I2V-14B + MeiGen InfiniteTalk graft + lightx2v distill, all folded |
| `wan22-i2v-a14b-{low,high}-q4_k.gguf` (8.1GB ea) | Wan2.2-I2V-A14B MoE experts (lightx2v 4-step distill) |
| `wan22-vace-fun-a14b-{low,high}-q4_k.gguf` (9.9GB ea) | VACE-Fun-A14B, **undistilled** |
| `wan22-vace-fun-a14b-{low,high}-distill-q4_k.gguf` | VACE-Fun-A14B with Wan2.2 distill folded (1053/1262 deltas; crisp 4-step) |
| `chinese-wav2vec2-base-f16.gguf` (189MB) | InfiniteTalk audio encoder |
| `longcat-umt5-xxl-q8_0.gguf`, `longcat-wan-vae-f16.gguf` | shared umT5 text enc + 16ch Wan2.1 VAE |

(`*.gguf`, `models*`, `*.bin`, `*.log`, `build*` are gitignored — they only exist on 10.0.0.151.)

## 4. CPU→GPU runtime deltas
- **Drop `LONGCAT_NO_FUSED_ROPE=1`** everywhere. It disables the fused RoPE because `ROPE_PE` is
  CUDA-only — on the 3060 you WANT the fused path (faster). Raw `sd-cli` set it on CPU; just omit it.
- Everything else (flags, env vars, prompts) is identical.

## 5. What this branch adds (all backend-agnostic — verify, then build on)
Key code, with the env flags that drive it:
- **P0 quality fix** (`examples/infinitetalk/main.cpp`): blank conditioning frames are filled `0.5`
  (neutral after the VAE's `[0,1]->[-1,1]` `scale_input`), not `0.0` (which becomes pixel -1 = black
  and drives a dark/green latent drift). This is THE fix that made lip-sync coherent. See
  `HANDOFF-wan22-P3-progress.md` "P0 RESOLVED".
- **VACE continuation mask** (`src/stable-diffusion.cpp`, `Wan2.x-VACE-14B` branch): env
  `VACE_CONT_FRAMES=K` sets mask=0 on the first K control frames (kept context) so the segment
  *continues* from them instead of treating all frames as control. Default 0 = unchanged.
- **VACE latent backdoor** (`src/stable-diffusion.cpp`): `VACE_SAVE_LATENT=<path>` banks a render's
  diffusion latent; `VACE_CONT_LATENT=<path>` (requires `VACE_CONT_FRAMES>0`) injects the prior
  segment's TAIL latents directly into the VACE `inactive` context, bypassing the lossy pixel
  decode->re-encode roundtrip. Runtime-verified. Measured ~+30-46% sharpness in the continuation vs
  the pixel path (over ONE seam; compounds per cut).
- **Zero-clip i2v fallback** (`src/stable-diffusion.cpp`): Wan2.1-I2V runs without a `--clip-vision`
  model (feeds zero clip_fea) instead of failing.
- **`IT_LATENT_CARRY`** (`examples/infinitetalk/main.cpp`): InfiniteTalk streaming carries the prior
  window's latent tail directly (no roundtrip). InfiniteTalk-specific.
- **Converters** (`tools/`): `convert_wan_dit.py --lora` (fold a distill LoRA per expert),
  `convert_infinitetalk_dit.py --graft` optional (graft-free base build, used by the discriminator).
- **Core port headers**: `src/infinitetalk.hpp` (audio graft + AudioProjModel + runner),
  `src/vace.hpp` (`build_vace_context` / `make_continuation_source`).
- **Analysis tools** (`tools/`): `frame_stats.py` (RGB/brightness/Gtint), `mouth_motion.py`
  (lip-sync ROI vs audio envelope), `seam_velocity.py` (continuation seam continuity),
  `sharpness.py` (per-segment Laplacian variance / degradation).

## 6. Proven on CPU @256px (re-validate at 720p on GPU)
- InfiniteTalk lip-sync WORKS (mouth tracks audio; `tools/mouth_motion.py` = 2.75x mouth-vs-control).
- VACE-Fun velocity continuation WORKS (subject keeps moving across the seam; mild, not invisible).
- VACE latent-direct continuation BEATS pixel re-encode (sharper, less compounding drift).
- Wan2.2 distill folds onto VACE-Fun and renders coherent 4-step output.
- Training buckets: 480 (~640² area, shift 7), 720 (~960² area incl. **1280×704**, shift 11).
  We rendered at shift 5 (lightx2v value) on CPU; shift is res-coupled — try 7/11 at full res.
- Wall-time math: 14B×4-step ≈ 0.64× LTX-2.3 (22B×8-step) compute. Quality TBD on GPU.

## 7. TODO (none GPU-specific — buildable anywhere, just faster on the 3060)
1. **N-segment chain loop for the general VACE path.** Tonight was 2-segment via shell scripts
   (`run_vace_latent_cont.sh`). The real engine: a resident-DiT chain loop (mirror the existing
   LTX `--ltx-chain-segments`/`--cont-latent-frames`/`--ltx-chain-prompts` machinery in
   `examples/cli/main.cpp` ~860+, but on VACE + the latent backdoor). Per-segment prompts =
   the "director" layer for the music-video use case.
2. **InfiniteTalk V2V dub pass.** Audio-sync over *generated* footage. The reference
   (`/tmp/infinitetalk-src/wan_multitalk.py`) takes `cond_video` (a video) + shot detection; our port
   only wired the image+audio path. This is the "dub speech onto any clip" stage.
3. **720p validation + shift 7/11 + GPU wall-time vs LTX-2.3-distilled.**

## Test recipes (drop LONGCAT_NO_FUSED_ROPE on GPU)
- VACE pixel-vs-latent A/B: `run_vace_latent_cont.sh` (bumps res/steps as you like).
- InfiniteTalk lip-sync: `sd-infinitetalk --dit infinitetalk-14b-q4_k.gguf --wav2vec
  chinese-wav2vec2-base-f16.gguf --vae longcat-wan-vae-f16.gguf --umt5 longcat-umt5-xxl-q8_0.gguf
  --image <face> --wav <speech> --frames 81 --height 480 --width 480 --steps 4 --distilled`.
- `mkffmpeg` in the `run_*.sh` scripts now uses `-pattern_type glob -i "$dir/*.png"` (was a broken
  `%*.png` glob that silently produced no mp4 — frames were always fine, only the muxing failed).
