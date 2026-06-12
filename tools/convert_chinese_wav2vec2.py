#!/usr/bin/env python3
"""
chinese-wav2vec2-base (HF Wav2Vec2Model) -> GGUF for wav2vec2.hpp's audio_encoder.*.

Maps the HF state_dict (prefix `wav2vec2.`) onto the names the C++
Wav2Vec2EncoderRunner loads under prefix `audio_encoder.`:
  wav2vec2.feature_extractor.conv_layers.{i}.conv.{weight,bias}
  wav2vec2.feature_extractor.conv_layers.0.layer_norm.{weight,bias}  (GROUP norm, base)
  wav2vec2.feature_projection.layer_norm.{weight,bias}
  wav2vec2.feature_projection.projection.{weight,bias}
  wav2vec2.encoder.pos_conv_embed.conv.{weight_g,weight_v,bias}  -> FUSED conv.weight (+bias)
  wav2vec2.encoder.layer_norm.{weight,bias}
  wav2vec2.encoder.layers.{i}.{attention.{q,k,v,out}_proj,layer_norm,
                               feed_forward.{intermediate_dense,output_dense},final_layer_norm}.*
The pretraining heads (quantizer.*, project_*, masked_spec_embed) are dropped.

pos_conv uses weight-norm with dim=2 (HF Wav2Vec2PositionalConvEmbedding); the C++
loads a single fused conv.weight, so reconstruct w = weight_v * weight_g / ||weight_v||
with the norm taken over dims (0,1) per kernel tap. f16 out (encoder is tiny; no quant).

Usage:
  uv run --with numpy --with torch --with gguf python3 tools/convert_chinese_wav2vec2.py \
      --in models/dl/chinese-wav2vec2-base/pytorch_model.bin \
      --out models/chinese-wav2vec2-base-f16.gguf
"""
import argparse, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="HF pytorch_model.bin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    import gguf

    sd = torch.load(args.inp, map_location="cpu", weights_only=True)

    def np_of(t):
        return t.detach().to(torch.float32).cpu().numpy()

    out = {}  # audio_encoder.* -> np.ndarray
    pos_g = pos_v = pos_b = None
    dropped = 0
    for k, v in sd.items():
        if not k.startswith("wav2vec2."):
            dropped += 1
            continue
        sub = k[len("wav2vec2."):]
        # collect pos_conv weight-norm parts for fusion; everything else copies through.
        if sub == "encoder.pos_conv_embed.conv.weight_g":
            pos_g = np_of(v); continue
        if sub == "encoder.pos_conv_embed.conv.weight_v":
            pos_v = np_of(v); continue
        if sub == "encoder.pos_conv_embed.conv.bias":
            pos_b = np_of(v); continue
        out["audio_encoder." + sub] = np_of(v)

    # base wav2vec2 has conv_bias=False -> the feature-extractor convs ship NO bias,
    # but wav2vec2.hpp's Wav2Vec2ConvLayer always loads conv.bias. Emit zero biases so
    # the loader resolves them (a no-op add, matching the bias-free conv).
    n_zero_bias = 0
    for name in list(out.keys()):
        if name.endswith(".conv.weight") and "feature_extractor" in name:
            bname = name[:-len(".conv.weight")] + ".conv.bias"
            if bname not in out:
                out[bname] = np.zeros((out[name].shape[0],), dtype=np.float32)
                n_zero_bias += 1

    # fuse pos_conv weight-norm (dim=2: per-kernel-tap g; norm over (out,in/groups)).
    assert pos_g is not None and pos_v is not None and pos_b is not None, "pos_conv parts missing"
    # weight_v [out, in/groups, kernel], weight_g [1,1,kernel].
    norm = np.sqrt((pos_v ** 2).sum(axis=(0, 1), keepdims=True))  # [1,1,kernel]
    pos_w = pos_v * (pos_g / norm)                                # [out, in/groups, kernel]
    out["audio_encoder.encoder.pos_conv_embed.conv.weight"] = pos_w.astype(np.float32)
    out["audio_encoder.encoder.pos_conv_embed.conv.bias"]   = pos_b

    t0 = time.time()
    writer = gguf.GGUFWriter(args.out, arch="wav2vec2", use_temp_file=True)
    writer.add_description("chinese-wav2vec2-base (HF) -> audio_encoder.* f16")
    n_t = 0
    total = 0
    for name in sorted(out.keys()):
        arr = np.ascontiguousarray(out[name], dtype=np.float16)
        writer.add_tensor(name, arr)
        n_t += 1
        total += arr.size
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    sz = os.path.getsize(args.out) / 1e6
    print(f"wrote {args.out}: {n_t} tensors f16 ({dropped} non-wav2vec keys dropped, "
          f"{n_zero_bias} zero conv biases added), {total/1e6:.1f}M params, {sz:.1f}MB in {time.time()-t0:.1f}s")
    print(f"  pos_conv fused: weight_v{pos_v.shape} * weight_g{pos_g.shape} -> conv.weight{pos_w.shape}")


if __name__ == "__main__":
    main()
