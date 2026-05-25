# LongCat-Video-Avatar 1.5 — supporting sub-model GGUF conversion notes

Companion to `convert_longcat_avatar.py` (the DiT). Covers the THREE supporting
sub-models needed for end-to-end gen. All output GGUF is on the SSD under
`models/`. All converters are pure-numpy (no torch): safetensors header parse +
mmap, bf16->f32 by hand, reusing the `Safetensors` reader from the DiT converter.

GGUF gotcha: `use_temp_file=True` spools to `$TMPDIR` (a 16 GB tmpfs at `/tmp`).
Always run with `TMPDIR=/home/dbrain/dev/longcat-avatar.cpp/.convert-tmp` (on SSD)
or the tmpfs OOMs. The `.convert-tmp` dir is created by these runs.

The pure-numpy `gguf` (0.19) quantizer supports F16/Q8_0/Q4_0/Q5_0 only (no K-quants).

---

## 1. umT5-XXL text encoder  — CONVERTER WRITTEN (`convert_umt5.py`)

- Source: `/mnt/hdd/longcat/base/text_encoder/` (HF `UMT5EncoderModel`, 5 F32 shards
  + `model.safetensors.index.json`). config: d_model 4096, d_ff 10240, num_heads 64,
  d_kv 64 (inner_dim = 64*64 = 4096 = d_model), num_layers 24, vocab 256384,
  gated-gelu, per-layer relative position bias.

- **Stock-load? Almost — needs a prefix-only rewrite, not a structural one.**
  sd.cpp's `T5Runner` (src/t5.hpp) with `is_umt5=true` (set by the conditioner when a
  tensor matches `text_encoders.t5xxl`) already expects the *original* T5 layout and
  sets `vocab_size=256384`, `relative_attention=false` (so every block constructs with
  `using_relative_attention_bias=true`, matching umT5's per-block rel-bias). The stock
  HF umT5 tensor names are ALREADY in that original layout:
    `shared.weight`,
    `encoder.block.N.layer.0.SelfAttention.{q,k,v,o}.weight`,
    `encoder.block.N.layer.0.SelfAttention.relative_attention_bias.weight` (all 24 blocks),
    `encoder.block.N.layer.0.layer_norm.weight`,
    `encoder.block.N.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight`,
    `encoder.block.N.layer.1.layer_norm.weight`,
    `encoder.final_layer_norm.weight`.
  sd.cpp loads `--t5xxl` under prefix `text_encoders.t5xxl.transformer.` (no name
  conversion at load), then `convert_tensors_name()` runs `t5_name_map`, which only
  rewrites llama.cpp-style names (`enc.`/`blk.`) — a no-op on these already-original
  names. So we just prepend the prefix and copy encoder tensors through verbatim.

- Name decisions: prefix every tensor with `text_encoders.t5xxl.transformer.`. This is
  an encoder-only checkpoint (no decoder tensors). The only tensor that would be dropped
  is a training-only `encoder.embed_tokens.weight` duplicate (sd.cpp lists it in its
  ignore set) — **not present** in this checkpoint, so nothing was dropped.

- Default dtype Q8_0 for 2D weight matrices (row len 4096 % 32 == 0, incl.
  `shared.weight` [256384,4096]); norms (1D) and `relative_attention_bias` [32,64] kept
  F16. Text encoder is CPU-offloadable; Q8_0 is the size/quality default.

- Output: `models/longcat-umt5-xxl-q8_0.gguf`  — SEE "Validation" below.
  Tensor count expected = 242 (240 block + final_layer_norm + shared).

## 2. Wan2.1 VAE  — CONVERTER WRITTEN (`convert_wan_vae.py`)

- Source: `/mnt/hdd/longcat/base/vae/diffusion_pytorch_model.safetensors` (HF diffusers
  `AutoencoderKLWan`, 194 F32 tensors). config: base_dim 96, z_dim 16, dim_mult
  [1,2,4,4], num_res_blocks 2, temperal_downsample [F,T,T].

- **Stock-load? NO — a structural name remap is required.** sd.cpp's `WAN::WanVAE`
  (src/wan.hpp), built by `WanVAERunner` with prefix `first_stage_model`, uses the
  *original* Wan (ComfyUI) layout: `encoder.conv1`, `encoder.downsamples.N`,
  `encoder.middle.{0,1,2}`, `encoder.head.{0,2}`, residual blocks as
  `residual.{0(RMS),2(conv),3(RMS),6(conv)}` + `shortcut`, `conv1`/`conv2` (the two
  quant convs), and the analogous decoder `upsamples.N`. The LongCat checkpoint is the
  HF *diffusers* refactor (`down_blocks.N`/`up_blocks.B.resnets.J`/`mid_block`/
  `*.norm.gamma`/`quant_conv`/`post_quant_conv`). sd.cpp has NO diffusers->Wan-VAE
  converter — its `convert_first_stage_model_name` only does SD1-style VAE — so we remap
  here.

- Name-mapping decisions (diffusers -> sd.cpp wan, emitted WITHOUT prefix so the `--vae`
  load path's `vae.`->`first_stage_model.` rewrite + no-op SD1 conversion lands them as
  `first_stage_model.<wanname>`):
    - `quant_conv.*`      -> `conv1.*`            (encoder post / top-level)
    - `post_quant_conv.*` -> `conv2.*`            (decoder pre / top-level)
    - `{enc,dec}.conv_in.*`  -> `{enc,dec}.conv1.*`
    - `{enc,dec}.conv_out.*` -> `{enc,dec}.head.2.*`
    - `{enc,dec}.norm_out.gamma` -> `{enc,dec}.head.0.gamma`
    - residual block `conv1/norm1/conv2/norm2/conv_shortcut` ->
      `residual.2 / residual.0 / residual.6 / residual.3 / shortcut`
    - `mid_block.resnets.{0,1}` -> `middle.{0,2}`; `mid_block.attentions.0` -> `middle.1`
      (attn sub-names `norm.gamma`/`to_qkv`/`proj` already match sd.cpp).
    - encoder `down_blocks.N` is already the flat sd.cpp `downsamples.N` order
      (res,res,resample,res,res,resample,res,res,resample,res,res); `resample.1`/
      `time_conv` sub-names pass through unchanged.
    - decoder `up_blocks.B` -> flat `upsamples.idx` where idx = B*4 + (j or 3 for the
      upsampler), since each up_block = 3 residual blocks + (B<3 ? 1 Resample).

- Conv weights: PyTorch 5D `[out,in,kt,kh,kw]` reshaped to 4D `[out*in,kt,kh,kw]`
  (out-major) to match sd.cpp `CausalConv3d`'s 4D ggml param `[kw,kh,kt,in*out]`.
  2D attn convs (1x1 / 1x1x1) kept as-is. **VERIFIED CORRECT** (2026-05-25): ggml
  `ggml_conv_3d` reshapes the weight to `[OC, IC*KD*KH*KW]` with OC outer / IC inner,
  which is exactly what the row-major `[out,in]→[out*in]` flatten produces.

- **RMS_norm gamma MUST be flattened to 1-D `[C]`** (added 2026-05-25). diffusers ships
  these with trailing singletons (resnet `[C,1,1,1]`, attention `middle.1.norm.gamma`
  `[C,1,1]`); sd.cpp's `WAN::RMS_norm` allocates a 1-D `[C]` tensor, so an un-flattened
  3-D gamma lands as ggml ne `[1,1,C,1]` and the loader's shape check **rejects the entire
  VAE** → the avatar runs with an unloaded/zero VAE → a saturated/BLANK-WHITE decode. This
  was the one and only "white clip" bug; it had passed the name-set check (names matched) but
  not a numeric check. **Always validate the VAE numerically**, see below.

- F16 (VAE wants precision).
- Output: `models/longcat-wan-vae-f16.gguf`, **194 tensors, 253.8 MB**.

- **Numeric validation harness** (`examples/vae-roundtrip/` → `sd-vae-roundtrip` +
  `tools/{vae_oracle,dump_vae_bins,compare_vae_recon}.py`, torch venv `.venv-vae`):
  `decode(encode(girl_480x832.png))` reconstructs the portrait at **PSNR 64.93 dB vs the
  torch oracle** and **37.59 dB vs input** (= torch's own ceiling). Encode latent mean/std
  (−0.102 / 1.137) matches torch to 4 decimals.

## 3. whisper-large-v3 ENCODER ONLY  — CONVERTER WRITTEN (`convert_whisper_encoder.py`)

- Source: `/mnt/hdd/longcat/avatar-1.5/whisper-large-v3/model.safetensors` (HF
  `WhisperForConditionalGeneration`, 1259 F16 tensors: 487 encoder + 772 decoder).
- We use whisper ONLY as an audio feature extractor; the avatar `AudioProjModel`
  consumes stacked hidden states from `audio_block=5` intermediate ENCODER layers.
  All `model.decoder.*` (772) dropped; `proj_out.weight` also dropped if present (none).

- Name decisions: keep HF encoder names verbatim, strip the HF `model.encoder.` prefix
  and re-prefix with `audio_encoder.`, e.g. `audio_encoder.conv1.weight`,
  `audio_encoder.layers.5.self_attn.q_proj.weight`, `audio_encoder.layer_norm.weight`.
  Note whisper's `self_attn.k_proj` has NO bias (q/v/out_proj do).

- F16 (already F16 in source). Conv1d weights stay 3D `[k,in,out]` (ggml-reversed).

- **Encoder config (for the C++ side; also stored as GGUF KV `whisper.encoder.*`):**
    - num encoder layers (num_hidden_layers): **32**
    - d_model: **1280**
    - encoder_attention_heads (n_heads): **20**  -> head_dim **64**
    - encoder_ffn_dim: **5120**
    - num_mel_bins: **128**
    - conv1: Conv1d 128->1280, kernel 3, stride 1, pad 1
    - conv2: Conv1d 1280->1280, kernel 3, stride **2**, pad 1
    - embed_positions: [1500, 1280] sinusoidal (fixed), max_source_positions **1500**
    - post-encoder `layer_norm` applied after all 32 layers
    - activation: gelu; pre-LN transformer (self_attn_layer_norm, final_layer_norm)

- Output: `models/longcat-whisper-v3-encoder-f16.gguf`, **487 tensors, 1274.0 MB**.
  Validated: 0 decoder-named tensors; 32 layers × 15 each (+7 non-layer); conv/embed/
  layer shapes correct; all 9 config KV present.
