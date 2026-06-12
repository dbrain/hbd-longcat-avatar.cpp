# Wan2.2 + InfiniteTalk port — P2 progress (worktree `wan22-infinitetalk`)

Continues `HANDOFF-wan22-P1-progress.md`. Branch `wan22-infinitetalk`, HEAD `5f597cd` (P0+P1). HX370 CPU.
**Mandate:** compile freely (`cmake --build` IS your loop) — only defer the slow multi-hour RENDERS until
code-confident. Don't pause to ask permission for routine calls. (memory: [[feedback_dont_ask_continue_or_handoff]])

## START HERE (next agent)
Goal = scripted promptable MUSIC VIDEO + injected song: Wan2.2-I2V-A14B makes the shots → InfiniteTalk V2V
dubs the song onto the singing characters. Read "BUILD PROGRESS" + "InfiniteTalk port SPEC" below first; the
memory notes [[gotchas_wan22_moe_prewired]] + [[reference_wan22_i2v_continuation]] + [[reference_infinitetalk_distill]] back them.
Build: `cmake -B build -DSD_CUDA=OFF -DSD_HIPBLAS=OFF && cmake --build build -j` (~80s). `LONGCAT_NO_FUSED_ROPE=1` on CPU.

DONE (verified): A14B converter + both q4_K experts (`models/wan22-i2v-a14b-{low,high}-q4_k.gguf`, 8.15GB ea);
wav2vec2-base variant in `wav2vec2.hpp` (compiles, config-verified); `tools/convert_infinitetalk_dit.py` merge
converter (validated vs real tensors); all InfiniteTalk weights in `models/dl/`.

DO NEXT, in order (write + compile each; render only at the end):
1. `tools/convert_infinitetalk_dit.py` is ready — RUN it to produce `models/infinitetalk-14b-f16.gguf`
   (base 7 shards + graft + lightx2v LoRA), then `sd-cli -M convert --type q4_K`. Verify it load-tests.
2. Write `src/infinitetalk.hpp` (tasks 5/6/7): MultiTalk DiT = reuse `wan.hpp` Wan2.1-I2V-14B blocks +
   per-block `audio_cross_attn` graft + `norm_x` + `AudioProjModel` + non-causal motion-frame streaming driver.
   Spec is below, every tensor name/dim verified. Compile against a stub, iterate.
3. `examples/infinitetalk/main.cpp` + CMakeLists (task 9). chinese-wav2vec2-base→GGUF converter (task 8 TODO).
4. THEN spend renders: P2 single A14B clip first (cheap floor), then the InfiniteTalk dub.
Task list is live (TaskList). `git -C ~/dev/wan22-infinitetalk status` — nothing committed yet (user commits).

## MAJOR FINDING — the handoff's P2 "MoE code task" does not exist
P1-progress said "implement the MoE high/low expert switch in wan.hpp (absent)." **It's already fully wired**,
one level up in the sampler (correct place — `WanRunner::compute` is single-step, loop is external):
- `stable-diffusion.cpp:3914-3922` splits sigmas at `moe_boundary` (default 0.875).
- `:6331-6378` samples high-noise expert → frees it → low-noise expert.
- Two `WAN::WanRunner` instances: `diffusion_model` (low) + `high_noise_diffusion_model` (high), separate
  files, prefixes `model.diffusion_model.` / `model.high_noise_diffusion_model.` (`:700-718`).
- CLI: `--diffusion-model` + `--high-noise-diffusion-model` + `--moe-boundary` (common.cpp:359/1096). Server+UI wired.
→ Single-clip Wan2.2-I2V-A14B is ~ZERO new C++. Just convert both experts → q4_K GGUF + run sd-cli with both.

## Weights (DOWNLOADED, `models/dl/wan22-i2v-a14b-moe-distill/`)
`lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v` — ships TWO separate experts (exact two-file shape the fork wants):
- `distill_models/high_noise_model/distill_model.safetensors` (28.58GB BF16)
- `distill_models/low_noise_model/distill_model.safetensors`  (28.58GB BF16)
- `loras/{high,low}_noise_model_rank64.safetensors` (0.63/0.74GB — the distill as LoRA; alt to the merged experts)
Reuse from P1: `longcat-umt5-xxl-q8_0.gguf` (T5), `longcat-wan-vae-f16.gguf` (16ch Wan2.1 VAE for A14B).

## Continuation reality (the architecture decision) — see memory `reference_wan22_i2v_continuation`
Wan2.2-I2V-A14B's ONLY documented/trained conditioning is single FIRST-frame I2V (c_concat, `:5516-5547`).
NO native continuation/extension/multi-frame/FLF (verified vs lightx2v card + Wan2.2 README).
- longcat `denoise_mask`/`cont_latent` head-pin + LTX-2.3 latent injection are OOD for A14B (trained into
  longcat/LTX, not A14B) = the "out-of-model-params backdoor" to avoid.
- Velocity: a single frame has none → motion resets at seams. Wan VAE 4× temporal (pix=(lat-1)*4+1).
- Real wan-native velocity-preserving paths: S2V FramePack motioner (in our s2v gguf, audio model) and
  **Wan-VACE `alibaba-pai/Wan2.2-VACE-Fun-A14B`** (TRY LATER, separate model, P2-out-of-scope).
**DECISION:** option 1 = single-frame c_concat chaining + long-clip bias (96GB never-OOM → fewer seams).
Characterize seam-velocity as a limitation. VACE-Fun later.

## DO NEXT (code-complete, defer runs)
1. **Convert** both A14B experts safetensors→f16 GGUF then quantize→q4_K. No local Wan-DiT py converter
   (only `tools/convert_wan_vae.py`); adapt `tools/convert_nava_dit.py` (handles BF16/FP8 decode) to the
   Wan DiT tensor scheme (`blocks.{i}.self_attn.{q,k,v,o}`, `.cross_attn.*`, `.ffn.{0,2}`, `.modulation`;
   top `patch_embedding`, `text_embedding.{0,2}`, `time_embedding.{0,2}`, `time_projection.1`, `head`).
   Then `sd-cli -M convert --type q4_K`. 3060 is unreachable from the agent session → convert LOCAL.
2. **Wan2.2-I2V scene-gen chain example** (option 1): drive `generate_video_ex` (gets c_concat+MoE+VAE free)
   per segment, outer loop feeds prior segment's last decoded frame as next `--image`. Long-clip bias.
3. **InfiniteTalk (P3) wiring** — the big new port. Wan2.1-I2V-14B base + audio graft (reuse wav2vec2.hpp +
   wan_s2v audio_injector pattern). Weights: `Wan-AI/Wan2.1-I2V-14B-480P`, `TencentGameMate/chinese-wav2vec2-base`,
   `MeiGen-AI/InfiniteTalk`, distill `lightx2v/Wan2.1-I2V-14B-480P-StepDistill-CfgDistill-Lightx2v`.
   See memory `reference_infinitetalk_distill`.

## CPU build (unchanged): `cmake -B build -DSD_CUDA=OFF -DSD_HIPBLAS=OFF && cmake --build build -j` (~80s).
Set `LONGCAT_NO_FUSED_ROPE=1` on CPU (ROPE_PE is CUDA-only); watch concat_T_cont_4d/madd-fuse on bigger configs.

## GOAL (locked 2026-06-12): scripted, prompt-driven MUSIC VIDEO with the user's own song injected.
= "endlessly promptable scenes + characters that lip-sync to my song." LongCat/S2V (avatar-class,
human-faces) can't do it. Path = **Wan2.2-I2V-A14B generates the promptable shots → InfiniteTalk V2V
dubs the song onto the singing characters**. Music videos CUT between shots → seamless velocity-continuation
(VACE/FramePack) is OPTIONAL polish, NOT required. InfiniteTalk (V2V) is the real unlock.

## BUILD PROGRESS (this session — code-complete-then-debug)
LANDED + VERIFIED:
- `tools/convert_wan_dit.py` — A14B distill safetensors→f16 GGUF (BF16 decode, patch_embedding 5D→4D
  squeeze, flat names). VERIFIED: low expert → `models/wan22-i2v-a14b-low-q4_k.gguf` (8.15GB) end-to-end;
  high expert converting in background (`logs/a14b-convert.log`).
- `src/wav2vec2.hpp` — chinese-wav2vec2-base variant: `ConvNorm{LAYER,GROUP,NONE}`, `Wav2Vec2Params::base()`
  (768/12/12/3072, post-LN, group feat-extract-norm), GroupNorm-first-conv + post-LN encoder. Parameterized
  so the S2V large/stable path is byte-identical (defaults unchanged). COMPILES (sd-s2v green); config
  verified vs the real config.json.
- `tools/convert_infinitetalk_dit.py` — merge: Wan2.1 base 7 shards + MeiGen graft + (optional) lightx2v
  rank64 LoRA fold → one f16 GGUF. WRITTEN, unverified (weights mid-download; LoRA key-prefix + alpha
  conventions need a real-file check).
DOWNLOADS (`models/dl/`, `logs/hf-infinitetalk-download.log`): A14B experts ✓, Wan2.1-I2V-14B-480P (66GB:
7 DiT shards + Wan2.1_VAE.pth + CLIP) ✓, chinese-wav2vec2-base ✓, MeiGen InfiniteTalk single (mid), lightx2v LoRA (queued).

## InfiniteTalk port SPEC (remaining: tasks 5,6,7,9 — write against the reference + test, NOT blind)
Reference downloaded to `/tmp/infinitetalk-src` during arch mapping (re-pull github.com/MeiGen-AI/InfiniteTalk if gone).
- **Base = Wan2.1-I2V-14B, ALREADY in `wan.hpp`**: WanRunner num_layers=40 + model_type "i2v" → desc
  "Wan2.1-I2V-14B", in_dim=36, `WanI2VCrossAttention` (k_img/v_img/norm_k_img + the 257-CLIP-token context
  split). REUSE these blocks. Single dense expert (Wan2.1 is NOT MoE — no high/low swap for P3).
- **Per-block graft (ALL 40 layers)**, inserted after text cross-attn, before FFN: `x = x + audio_cross_attn(norm_x(x), audio)`.
  Tensors (flat, confirmed from single/infinitetalk.safetensors = 330 FP32):
  `blocks.{i}.audio_cross_attn.q_linear[5120,5120]`, `.kv_linear[10240,768]` (FUSED k+v, kv_dim 768),
  `.proj[5120,5120]`, `blocks.{i}.norm_x[5120]` (affine LayerNorm). NO qk_norm. Single-speaker =
  plain per-frame cross-attn (rearrange visual "B (N_t·S) C → (B·N_t) S C"; q=q_linear(visual), k,v=split(kv_linear(audio));
  32 audio tokens/frame; proj). SKIP the RoPE-speaker-class + x_ref_attn_map (multi-speaker only).
- **AudioProjModel** (`audio_proj.*`): proj1[512,46080=5·12·768], proj1_vf[512,73728=8·12·768], proj2[512,512],
  proj3[24576=32·768,512], norm[768]. First video frame → proj1 (5-window); latter frames grouped by
  vae_scale=4 → proj1_vf (8-window, middle_index=2 first/mid/last concat); relu between projs; LN out → 32 tokens/frame @768.
- **Audio prep**: `Wav2Vec2EncoderRunner(..., Wav2Vec2Params::base())` → stack hidden_states[1:] (12 layers)
  → linear_interpolation 50→25fps → AudioProjModel. (chinese-wav2vec2-base GGUF converter still TODO: map HF
  Wav2Vec2Model names → the `audio_encoder.*` scheme wav2vec2.hpp expects.)
- **I2V conditioning** (per window): VAE-encode [cond_frame, zeros(80)] → 16ch + 4ch mask(1@frame0) = 20ch
  c_concat (in_dim 36); PLUS CLIP-H/14(cond_frame) → img_emb MLPProj[1280→5120] → 257 tokens prepended to umT5 context.
- **Streaming driver** (NON-causal; model on `WanS2VRunner::compute`, NOT compute_causal_block): 81-frame
  (4n+1) windows, motion_frame=9 (→ 1+(9-1)//4 = 3 latent frames); each diffusion step OVERWRITES the first
  3 latent frames with noised motion latents (continuity, not KV-cache); `audio_start_idx += 81-9` overlap;
  drop first 9 output frames on windows>0; audio tail reflect. CFG cond/drop-text/uncond. distill: 4-step
  LCM shift=5 cfg=1 (lightx2v LoRA pre-folded by the merge converter).
- **examples/infinitetalk/main.cpp**: load merged DiT + chinese-wav2vec2 + CLIP + Wan2.1 VAE + umT5; source-video
  + song in; webm out. `LONGCAT_NO_FUSED_ROPE=1` on CPU. Add to examples/CMakeLists.txt.
