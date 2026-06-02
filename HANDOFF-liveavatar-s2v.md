# LiveAvatar (Wan2.2-S2V-14B) → sd.cpp/ggml — PORT HANDOFF

## MAX-PERF campaign (2026-06-03) — FA win + KV root-cause

Goal: beat longcat (~104 s/clip @ 480x832) at the FAIR 480x832, causal path. All measured
at 480x832, 1 step, frames=21 (2 causal blocks, nfb=3), DiT-only (NOVAE), boy.jpg+speechA.
GPU = RTX 3060 12 GB; host = 31 GB RAM, weights on a ROTATIONAL HDD (`/mnt/hdd`).

LANDED + COMMITTED: **flash-attention in the causal KV-cache self-attn** (`src/wan.hpp`
`WanSelfAttention::forward_kv_cache`, commit 7fa1140). Was hardcoded `flash_attn=false` →
materialized the L_blk x L_k scores (5.5 GB, THE causal compute-buffer peak). Now uses the
runner flash flag, mask=nullptr (full attention over the rolling cache). Effect:
- compute buffer 5.5 GB → ~3.8 GB; offload plan 21 segments → 4 (@9.5 GB budget) / 7 (@6.0).
- S2V_NO_FLASH=1 keeps the exact FA-off path for A/B.

MEASURED (block-0 = prefill + 1 block-forward, 1 step; pure GPU compute ~8.6 s):
| config                         | budget | peak VRAM | block-0 wall | segments |
|--------------------------------|--------|-----------|--------------|----------|
| baseline (FA off, offload 6.5) | 6.5    | 8896 MiB  | 1748 s (!)   | 21       |
| SOLO (S2V_NO_OFFLOAD, FA off)  | n/a    | OOM       | —            | monolith |
| SOLO (S2V_NO_OFFLOAD, FA on)   | n/a    | OOM @prefill (needs 2.6 GB on top of 8.8 GB resident) | monolith |
| offload+FA (near-SOLO)         | 9.5    | 10710 MiB | 16.3 s       | 4        |
| offload+FA (COEXIST)           | 6.0    | 8118 MiB  | 16.3 s       | 7        |
(baseline 1748 s = HDD-mmap + swap thrash, NOT GPU — see below. block-0 warm wall is ~16 s.)

ROOT CAUSE of the remaining bottleneck (fully proven, NOT compute):
1. Offload DiT weights are loaded **pinned (RAM, unswappable) 8.8 GB**, NOT mmap. Good.
2. The rolling causal KV cache is held **HOST-side F32** (`cache_k/cache_v` in wan_s2v.hpp),
   read back per block + re-uploaded next block. At 480x832 it is ~12.8 GB F32 at 2 blocks.
3. Starting at block ≥1 the growing F32 host cache + readback transients push the 31 GB host
   into SWAP → the pinned-bounce H2D + cudaMalloc staging stall: a single segment jumped to
   `alloc=73 s copy_in=167 s` (compute stayed 1.6 s). THIS is the 21-55 s/segment "copy_in"
   the brief saw. Block-0 (cache empty) is clean at ~16 s; block-1 explodes.

VERDICT vs longcat at 480x832: block-0 compute is ~8.6 s (prefill+1 forward), warm wall 16 s.
At 1 step that is competitive, but the host-KV swap pathology makes any ≥2-block clip blow up.
NOT a GPU-compute loss — a host-memory architecture problem.

NEXT (highest leverage, scoped, NOT yet done — risk = needs slow multi-block render validation):
- **Resident KV** (Lever 1): keep cond K/V + rolling K/V as GPU-resident persistent tensors
  (mirror LongCatAvatarRunner `condkv_buf` / `register_persistent_tensor`, longcat_avatar.hpp
  ~1303-1556) instead of the host F32 round-trip. Kills the swap villain entirely. Pair with
  **F16 KV** (halves footprint + PCIe). cond K/V is block-invariant (already prefilled once,
  read 40/40 OK). Sizing: F16 KV for a full clip ~6.4 GB → with 8.8 GB resident weights it
  will NOT all fit on 12 GB, so the practical target is offload@high-budget + resident KV
  (weights mostly resident, KV resident, no host churn). The persistent+offload combo was
  flagged buggy for longcat's cond-cache (ggml-alloc view-chain) — validate carefully or do
  it in the no-offload path on short clips first.
- Then kernel laps on the FA self-attn (occupancy / mmq tile) — only after H2D/swap is gone.


Goal: port **LiveAvatar** (Quark-Vision) — an audio-driven talking-head model = stock
**Wan-AI/Wan2.2-S2V-14B** + a rank-128 DMD-distill **LoRA** (`liveavatar.safetensors`) — into this
fork (a `leejet/stable-diffusion.cpp` fork that already runs LongCat-Video-Avatar). Target Q4_K on a
single RTX 3060. Hope: ≈ longcat quality, faster (4-step distilled vs longcat's 8-step). Real-time is
"5×H800/45fps" upstream — 3060 will be far from real-time, the question is *how* far + is quality worth it.

## Status (2026-06-02)
- Weights downloaded to `/mnt/hdd/live-avatar/` (NOT in repo):
  - `Wan2.2-S2V-14B/` — DiT 4 shards (32.6 GB bf16) + `Wan2.1_VAE.pth` + `wav2vec2-large-xlsr-53-english/`
    + `config.json`. (umT5 `.pth` skipped — we already have it as gguf.)
  - `LiveAvatar/liveavatar.safetensors` — 1.35 GB LoRA.
  - `ref/LiveAvatar/`, `ref/Wan2.2/` — reference source repos (READ THESE for ground truth).
- Architecture fully mapped (below). No C++ written yet.

## Decision: M1 correctness (stock, no LoRA) BEFORE M2 product (LoRA + causal streaming)
- **M1** = stock non-causal Wan2.2-S2V-14B, full-step CFG, **drop_motion_frames=True** (single clip,
  no motion history → FramePack motioner is SKIPPED entirely). One short clip. Validates DiT + wav2vec2
  + VAE + GGUF-conversion with the simplest sampler (closest to our existing `WanRunner`) and ZERO
  LoRA/causal/streaming unknowns. Slow on 3060, one-time gate.
- **M2** = merge the LoRA, add the FramePack motioner + per-frame causal KV-cache blockwise streaming →
  4-step, infinite length. This is the "faster than longcat?" verdict.
- Rationale = user's own "minimum repro / one thing at a time" rules. M2-first would stack 4 unproven
  things at once.

## What we ALREADY have (do not re-port)
- **`src/wan.hpp`** — full stock Wan2.x DiT (`Wan`, `WanParams`, `WanAttentionBlock`, `WanSelfAttention`,
  `WanCrossAttention`, `Head`, `WanRunner`) + **Wan VAE** (`WanVAE`, `WanVAERunner`). This is leejet's
  upstream Wan2.1/2.2 T2V/I2V/VACE port (PR #778/#819). `WanParams` already documents the 14B config.
- GGUFs in `models/`: `longcat-umt5-xxl-q8_0.gguf` (umT5-XXL, **same text encoder S2V uses**),
  `longcat-wan-vae-f16.gguf` (**Wan2.1 VAE — same as S2V's `Wan2.1_VAE.pth`**, verified). Reusable as-is.
- `src/longcat_audio.hpp` — whisper-v3 encoder port = the **template** for the wav2vec2 encoder.
- `src/longcat_avatar.hpp` — sibling audio-into-Wan-DiT port; proves the audio-cross-attn pattern here.
- All ggml infra: Q4_K quant, 3D RoPE (`src/rope.hpp`), flash-attn, CPU-offload + mmap + lap-34 prefetch
  thread, avatar HTTP server + cooperative cancellation, `src/lora.hpp`.

## NOT in upstream sd.cpp (the genuinely new work)
- Checked leejet branches (master, wan2.2_5B_flf2v, z-image, qwen) + PRs: **no Wan-S2V / audio / speech**.
  Lots of LTX-2 audio-video + ACE-Step, but that's a different family — not reusable for S2V.

---

# ARCHITECTURE MAP (ground truth = `ref/Wan2.2/wan/modules/s2v/model_s2v.py`, non-causal stock)

## DiT config (Wan2.2-S2V-14B) — single DiT, NOT MoE
`dim=5120, num_heads=40, head_dim=128, num_layers=40, ffn_dim=13824, freq_dim=256, text_dim=4096,
text_len=512, in_dim=out_dim=16, patch_size=(1,2,2), eps=1e-6, qk_norm=True, cross_attn_norm=True`.
S2V extras (from `wan_s2v_14B.py` / `wan_s2v_14B_modified.py`):
`cond_dim=16, audio_dim(DiT side)=1024 for injector k/v, num_audio_token=4, enable_adain=True,
adain_mode="attn_norm", audio_inject_layers=[0,4,8,12,16,20,24,27,30,33,36,39] (12 of 40),
enable_framepack=True, enable_motioner=False, motion_frames=[17,5]-ish (73 px / lat), num_frames_per_block=3`.
RoPE: `freqs = cat[rope_params(1024, d-4*(d//6)=107), rope_params(1024, 21), rope_params(1024, 21)]`,
3D (temporal/H/W), `theta=10000` (stock uses 1024 max; causal variant uses 45000). d=128.

## Per-block = WanS2VAttentionBlock (`model_s2v.py:184-244`)
Extends stock `WanAttentionBlock`; ONLY diff is the 2-segment modulation + separate q/k/v self-attn.
Modulation `e` = `(self.modulation[1,6,dim].unsqueeze(2) + e0).chunk(6,dim=1)` → 6 × `[B,2,dim]`
(slot 0 = noisy@t, slot 1 = ref/motion@t=0; the "2" comes from `zero_timestep`). `seg_idx=original_seq_len`.
```
norm_x = LayerNorm_noaffine_fp32(x)
# 2-seg modulate: tokens[0:seg] use slot0, tokens[seg:] use slot1
norm_x = concat( norm_x[:, :seg]*(1+e[1][:,0])+e[0][:,0],  norm_x[:, seg:]*(1+e[1][:,1])+e[0][:,1] )
y = self_attn(norm_x)                      # sep q/k/v, RMSNorm q/k over head_dim, 3D RoPE
x = x + concat( y[:, :seg]*e[2][:,0], y[:, seg:]*e[2][:,1] )         # gated
x = x + cross_attn(norm3(x), context)      # text cross-attn (norm3 affine LN)
norm2_x = LayerNorm_noaffine_fp32(x); 2-seg modulate with e[4]/e[3]
y = ffn(norm2_x)                           # Linear(5120->13824), GELU(tanh), Linear(13824->5120), NO bias
x = x + 2-seg( y * e[5] )                  # gated
```
NOTE: for M1, slot0≠slot1 ONLY if zero_timestep (ref/motion get t=0). Confirm `zero_timestep` from the
pipeline; if False, both slots equal → collapses to stock single-modulation (then `WanAttentionBlock`
forward is bit-correct and we only swap fused-qkv→sep-qkv weight layout).

## Audio injection (`after_transformer_block`, `model_s2v.py:601-648`) — runs after each of the 12 blocks
```
audio_emb = merged_audio_emb        # [B, F, n_tok=5, dim=5120]  (per latent frame)
h = hidden[:, :original_seq_len]    # noisy tokens only, reshape "b (t n) c -> (b t) n c"  (t=F frames)
if enable_adain (attn_norm): h_n = injector_adain_layers[j](h, temb=audio_emb_global[:,0])  # AdaLN
else:                        h_n = injector_pre_norm_feat[j](h)                              # LN no-affine
out = injector[j](x=h_n, context=audio_emb_perframe)   # cross-attn, q:5120, k/v:1024->? ; see LoRA shapes
hidden[:, :original_seq_len] += rearrange(out, "(b t) n c -> b (t n) c")
```
Per-frame: each video frame's spatial tokens cross-attend to that frame's 5 audio tokens. KEY: audio
drives only the noisy tokens, frame-aligned.

## Head_S2V (`model_s2v.py:135-147`)
`e = (head.modulation[1,2,dim] + e_t.unsqueeze(1)).chunk(2); x = head.head( norm(x)*(1+e[1]) + e[0] )`
then unpatchify (patch (1,2,2), out 16ch).

## Top-level forward assembly (`model_s2v.py:650-857`) — M1 path
1. `audio_input` [B,25,1024,T_a]; prepend `motion_frames[0]` repeats of frame0; `casual_audio_encoder` →
   `merged_audio_emb = audio_emb[:, motion_frames[1]:]`.
2. `x = patch_embedding(latent) + cond_encoder(cond_states)`  (cond/pose Conv3d; for M1 cond_states can be
   zeros). flatten→[B,L_noisy,dim]. grid_sizes noisy = [[0,0,0],[F,H,W],[F,H,W]].
3. ref: `patch_embedding(ref_latent)` → 1 frame, grid [[30,0,0],[31,H,W],[1,H,W]]; concat to x.
4. `mask_input` = 0 for noisy, 1 for ref (and 2 for motion, set in inject_motion). `rope_precompute`.
5. `inject_motion(drop_motion_frames=True)` → **NO-OP for M1** (returns empty). M2: FramePack adds 576
   motion tokens (zip_frame_buckets [1,2,16]) with mask=2, grids at negative f.
6. `x = x + trainable_cond_mask(mask_input)`   (Embedding(3, dim))
7. time: `e = time_embedding(sinusoid(freq_dim, t))`; `e0 = time_projection(e).unflatten(6,dim)`;
   if zero_timestep append t=0 row, build e0 `[B,6,2,dim]`, `e0=[e0, original_seq_len]`; else
   `e0=e0.repeat 2`, `e0=[e0,0]`.
8. `context = text_embedding(pad umT5 to text_len=512)`.
9. for block in 40: `x = block(x, e0, seq_lens, grid_sizes, freqs, context); x = after_transformer_block(i,x)`
10. `x = x[:, :original_seq_len]; x = head(x, e); unpatchify`.

## Audio path (`audio_encoder.py` + `auxi_blocks.py`)
- Encoder = **wav2vec2-large-xlsr-53-english** (24 transformer layers, 1024-d, 16 heads, conv frontend
  7 layers stride→320 → 50 fps, `feat_extract_norm="layer"`, `do_stable_layer_norm=True`). Output ALL 25
  hidden states (input embed + 24).
- `linear_interpolation(50fps → video_rate=30fps, align_corners=True)`.
- `get_audio_embed_bucket_fps(fps=16, m=0)` → per-latent-frame embed `[25*1024]` (center frame only at m=0).
- `CausalAudioEncoder` (`auxi_blocks.py`): learnable `[1,25,1,1]` SiLU weights → weighted-sum over 25
  layers → `MotionEncoder_tc` causal Conv1d (replicate-pad, strides 1→2→2) → tokens `[B,F,5,dim]`
  (1 global + 4 local) (+ global branch if adain). VERIFY exact conv dims in auxi_blocks.py when porting.
- Injected into DiT at the 12 layers via `AudioInjector_WAN` cross-attn (above).

## Scheduler / sampling
- Flow-matching (`flow_match.py`): `x_next = x + model_out * (sigma_next - sigma_t)`. Stock = ~40 steps
  + CFG (guide_scale 4.5). Distilled (LoRA) = **4 steps, CFG=0**.
- M2 block-wise autoregressive: 3 latent frames/block, KV-cache + motion-frame carryover; first block t=0
  sink-prefill of ref/motion KV. Per-block audio slice `audio[..., blk*4*3:(blk+1)*4*3]`.
- VAE = Wan2.1 (4×8×8 spatial, ×4 temporal). ref = encode(ref_img repeated 5 frames)[:, :, 1:] → 1 latent
  frame. Decode `cat([motion_latents, clip_output])`, drop overlap.

## LoRA (`liveavatar.safetensors`) — 896 tensors, rank 128, bf16
On: `blocks.{0-39}.{self_attn,cross_attn}.{q,k,v,o}`, `blocks.{0-39}.ffn.{0,1}`,
`audio_injector.injector.{0-11}.{q,k,v,o}`. Merge `W += (lora_B @ lora_A) * scale` (scale = alpha/rank,
confirm alpha; PEFT default alpha often = rank → scale 1.0). Merge into bf16 base BEFORE gguf convert.

---

# C++ FILE PLAN
- `src/wan_s2v.hpp` (NEW) — `WanS2V` DiT: subclass/extend `Wan`/`WanAttentionBlock` from `wan.hpp`. Add
  `cond_encoder` (Conv3d), `trainable_cond_mask` (Embedding 3×dim), ref-token concat, 2-seg modulation,
  `audio_injector` (12 cross-attn blocks), `Head_S2V`. Defer FramePack motioner to M2 (drop motion).
- `src/wav2vec2.hpp` (NEW) — feature extractor (7 Conv1d + LN) + 24 pre-LN transformer layers, capture
  all 25 hidden states. Template = `longcat_audio.hpp` WhisperEncoder. Plus CausalAudioEncoder
  (weighted-sum + MotionEncoder causal convs).
- Conversion: extend `src/name_conversion.cpp` + a python convert (see how longcat gguf was made) for the
  new `audio_encoder.*`, `audio_injector.*`, `cond_encoder.*`, `trainable_cond_mask.*` tensors. Quant Q4_K
  DiT, f16 VAE/wav2vec2, q8 umT5.
- Sampler: M1 reuse `WanRunner`-style full-seq loop (40-step flow-match + CFG). M2 = causal blockwise
  (new loop, KV cache).
- Wire into server like longcat (`routes_longcat.cpp` analog) — M2.

# GGUF tensor naming (match these from the PyTorch names)
Top: `patch_embedding.{weight,bias}`, `text_embedding.{0,2}.{w,b}`, `time_embedding.{0,2}.{w,b}`,
`time_projection.1.{w,b}`, `head.{norm,head}.{w,b}`, `head.modulation`, `cond_encoder.{w,b}`,
`trainable_cond_mask.weight`. Per-block (×40): `blocks.{i}.{norm1,norm2,norm3}.{w,b}`,
`blocks.{i}.self_attn.{q,k,v,o}.{w,b}`, `blocks.{i}.self_attn.{norm_q,norm_k}.w`,
`blocks.{i}.cross_attn.{q,k,v,o}.{w,b}` (+norm_q/k), `blocks.{i}.ffn.{0,2}.w`, `blocks.{i}.modulation`.
Audio (×12): `audio_injector.injector.{j}.{q,k,v,o}.{w,b}`, `audio_injector.injector_pre_norm_feat.{j}.*`,
`audio_injector.injector_adain_layers.{j}.*`. wav2vec2: `audio_encoder.*` (feature_extractor.conv_layers,
encoder.layers, pos_conv_embed) + CausalAudioEncoder `casual_audio_encoder.*`.

# Reference paths
- Stock non-causal (M1 truth): `/mnt/hdd/live-avatar/ref/Wan2.2/wan/modules/s2v/model_s2v.py`
- Causal streaming (M2 truth): `/mnt/hdd/live-avatar/ref/LiveAvatar/liveavatar/models/wan/causal_model_s2v.py`,
  `causal_s2v_pipeline.py`, `scheduler.py`, `flow_match.py`, `minimal_inference/s2v_streaming_interact.py`,
  `infinite_inference_single_gpu.sh`.
- Audio: `.../s2v/audio_encoder.py`, `auxi_blocks.py`, `motioner.py` (M2 framepack).
- Configs: `ref/Wan2.2/wan/configs/wan_s2v_14B.py`, `ref/LiveAvatar/.../configs/wan_s2v_14B_modified.py`.

# Rules (from memory)
- cpp forks build fine on-server; GPU renders are ASK-FIRST + one-at-a-time (stop prod acestep/tts/llama/
  longcat first; watch orphaned dev containers). NO Rust builds on-server.
- Gate kernel/correctness changes with a fixed-seed render + PSNR; iterate on shortest input.
