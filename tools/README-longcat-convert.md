# LongCat-Video-Avatar 1.5 → GGUF conversion

Tools for getting the reference checkpoint at `/mnt/hdd/longcat/` into GGUF that
`longcat-avatar.cpp` can load. **Write GGUF to the main SSD** (`models/`), never
back to `/mnt/hdd` (that's slow raw-safetensors staging only).

## DiT (the bespoke part) — `convert_longcat_avatar.py`

The reference DiT (`avatar-1.5/base_model_int8/`, 48 blocks, hidden 4096, 32 heads)
is **weight-only INT8, per-channel symmetric**: each Linear `X` is stored as
`X.weight_int8` (I8 `[out,in]`) + `X.weight_scale` (F32 `[out]`), de-quantized as
`w[o,i] = int8[o,i] * scale[o]`. Norms, biases, the Conv3d patch embed, and
`final_layer.linear` are BF16 (the quant config skips `final_layer.linear`).

sd.cpp can't read that custom int8+scale layout, so the converter de-quantizes to
a normal GGUF. Tensor names are kept **verbatim** under `--prefix`
(default `model.diffusion_model.`); the C++ avatar model loads by these exact names.
No torch — safetensors headers are parsed by hand and mmap'd; BF16 is widened to
F32 by hand. RAM-safe (`use_temp_file=True`, one tensor at a time over mmap).

```bash
cd ~/dev/longcat-avatar.cpp
uv run --with numpy --with gguf python3 tools/convert_longcat_avatar.py \
    --src /mnt/hdd/longcat/avatar-1.5/base_model_int8 \
    --out models/longcat-avatar-1.5-dit-f16.gguf \
    --dtype f16
# debug round-trip:  add  --max-blocks 1   (57 tensors, ~0.84 GB, ~6 s)
```

Validated: dequant is exact vs manual `int8·scale` (err = f16 rounding ~5e-6);
ggml shapes come out correctly reversed (e.g. `attn.qkv` ne=[4096,12288],
`audio_cross_attn.kv_linear` ne=[768,8192], conv `x_embedder.proj` ne=[2,2,16,4096]).
The Conv3d patch embed `[4096,16,1,2,2]` is reshaped to 4D `[4096,16,2,2]` (GGUF is
max-4D; the temporal patch dim is 1).

### Quantization ladder (Q4_K to fit 12 GB)

`gguf`'s pure-numpy quantizer only does F16/Q8_0/Q4_0/Q5_0 — **no K-quants**. So the
path to Q4_K/Q5_K is: convert to **F16 here**, then requantize with ggml's mature
K-quantizer via sd-cli convert mode:

```bash
# (intended — depends on the avatar arch being registered in the C++ loader first)
./build/bin/sd-cli -M convert -m models/longcat-avatar-1.5-dit-f16.gguf \
    -o models/longcat-avatar-1.5-dit-q4_k.gguf \
    --tensor-type-rules "model.diffusion_model.=q4_K"
```

F16 → Q4_K is higher quality than Q8_0 → Q4_K, so F16 is the canonical intermediate.
(int8 → f16 is exact; the bf16 base for a clean re-quant is NOT downloaded.)

## TODO (other sub-models, needed before end-to-end gen)

- **umT5-XXL** text encoder (`/mnt/hdd/longcat/base/text_encoder/`, 5 shards,
  caption_channels 4096) — sd.cpp already has T5/umT5; check if it loads the stock
  safetensors directly or needs a gguf.
- **Wan2.1 VAE** (`/mnt/hdd/longcat/base/vae/`) — stock Wan VAE; sd.cpp's existing
  Wan VAE conversion should apply with no LongCat-specific work.
- **whisper-large-v3 encoder** (`avatar-1.5/whisper-large-v3/model.safetensors`,
  hidden 1280) — feature extractor for the audio graft (NOT STT). Convert the
  **encoder only** (we never run the decoder); whisper.cpp's converter or a bespoke
  encoder-only gguf.
- **DMD LoRA** (`avatar-1.5/lora/dmd_lora.safetensors`, dim128/alpha64) — 8-step
  distillation. Either fold into the DiT weights at convert time or load as a LoRA.
