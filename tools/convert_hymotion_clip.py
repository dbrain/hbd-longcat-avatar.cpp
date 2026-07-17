#!/usr/bin/env python3
"""
CLIP-L text encoder (openai/clip-vit-large-patch14)  ->  GGUF, for HY-Motion's `vtxt`.

HY-Motion's second text encoder. It contributes the *pooled* 768-d sentence vector
(`vtxt_input`) that is added to the timestep embedding to form the DiT's modulation
`adapter` -- the Flux "pooled CLIP vec" slot. See:

  hymotion/network/text_encoders/text_encoder.py:39-47   SENTENCE_EMB_LAYOUT["clipl"]
      module_path   = ckpts/clip-vit-large-patch14
      encoder class = CLIPTextModel          <-- NOT CLIPTextModelWithProjection
      pooling_mode  = "pooler_output"
      max_length    = 77

NO TORCH REQUIRED (same rule as convert_hymotion.py). A safetensors file is an 8-byte
header length + a JSON header + raw tensor bytes; numpy reads it directly.

WHAT IS EMITTED
---------------
Only `text_model.*` (196 tensors). The checkpoint on disk is the *full* `CLIPModel`
(592 tensors: text tower + vision tower + both projections + logit_scale) because the
HF repo ships CLIPModel and `CLIPTextModel.from_pretrained` just picks the text tower
out of it. The vision tower is dead weight for us; `text_projection` is dead weight
too, and worse than dead -- see below.

Names are emitted VERBATIM under the `text_model.` prefix, which is exactly what
src/model/te/clip.hpp's CLIPTextModel block tree expects:

    text_model.embeddings.token_embedding.weight        <-> CLIPEmbeddings
    text_model.embeddings.position_embedding.weight
    text_model.encoder.layers.N.self_attn.{q,k,v,out}_proj.{weight,bias}
    text_model.encoder.layers.N.{layer_norm1,layer_norm2}.{weight,bias}
    text_model.encoder.layers.N.mlp.{fc1,fc2}.{weight,bias}
    text_model.final_layer_norm.{weight,bias}

so the C++ side loads it with `init_from_file()` (NO name conversion) and prefix
"text_model". `embeddings.position_ids` is dropped: it is an I64 arange buffer, not a
parameter, and clip.hpp does not declare it.

*** THE TRAP THIS CONVERTER EXISTS TO AVOID ***

`text_projection` is DELIBERATELY NOT EMITTED. For CLIP-L, projection_dim == 768 ==
hidden_size, so applying the projection produces a tensor of *exactly the right shape*
and silently wrong values -- there is no shape check anywhere that can catch it. The
reference takes `pooler_output`, and HF's CLIPTextModel.pooler_output is
`final_layer_norm(last_hidden)[argmax(input_ids)]` with NO projection (the projection
lives only in CLIPTextModelWithProjection -- transformers modeling_clip.py:1410).
clip.hpp only builds `params["text_projection"]` when version == OPEN_CLIP_VIT_BIGG_14
(clip.hpp:253-256), so for OPENAI_CLIP_VIT_L_14 the pooled path is projection-free and
matches. Emitting the tensor could not make it load -- but not emitting it means the
mistake cannot even be staged.

Usage:
    uv run --with numpy --with gguf python3 tools/convert_hymotion_clip.py \
        --src /mnt/hdd/3d/avatar-shootout/_weights/hymotion/clip-vit-large-patch14/model.safetensors \
        --out models/hymotion-clip-l-f16.gguf --dtype f16

    # audit only, writes nothing:
    python3 tools/convert_hymotion_clip.py --src .../model.safetensors --list
"""
import argparse
import json
import os
import struct

import numpy as np

# ---------------------------------------------------------------------------
# torch-free safetensors reader
# ---------------------------------------------------------------------------

_ST_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "BF16": "bfloat16",
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


class SafetensorsReader:
    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            (hdr_len,) = struct.unpack("<Q", f.read(8))
            self.header = json.loads(f.read(hdr_len))
        self.data_start = 8 + hdr_len
        self.meta = self.header.pop("__metadata__", None)

    def names(self):
        return list(self.header.keys())

    def shape(self, name):
        return tuple(self.header[name]["shape"])

    def dtype(self, name):
        return self.header[name]["dtype"]

    def to_numpy(self, name):
        e = self.header[name]
        dt = e["dtype"]
        if dt not in _ST_DTYPES:
            raise NotImplementedError(f"unhandled safetensors dtype {dt} for {name}")
        start, end = e["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.data_start + start)
            raw = f.read(end - start)
        if _ST_DTYPES[dt] == "bfloat16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=_ST_DTYPES[dt])
        return arr.reshape(e["shape"])


# ---------------------------------------------------------------------------

TEXT_PREFIX = "text_model."
# I64 arange buffer, not a parameter. clip.hpp never declares it.
DROP_SUFFIXES = ("embeddings.position_ids",)


def derive_arch(r, names):
    """Recover the CLIP-L text hparams from the weights, rather than trusting a
    config.json that may not travel with the file."""
    a = {}
    tok = r.shape(TEXT_PREFIX + "embeddings.token_embedding.weight")
    a["vocab_size"], a["hidden_size"] = tok[0], tok[1]
    a["n_positions"] = r.shape(TEXT_PREFIX + "embeddings.position_embedding.weight")[0]
    a["intermediate_size"] = r.shape(TEXT_PREFIX + "encoder.layers.0.mlp.fc1.weight")[0]
    layers = set()
    for n in names:
        if n.startswith(TEXT_PREFIX + "encoder.layers."):
            layers.add(int(n[len(TEXT_PREFIX + "encoder.layers.") :].split(".")[0]))
    a["num_layers"] = len(layers)
    # not recoverable from shapes (head_dim is never materialised); CLIP-L is 12 heads
    a["num_heads"] = 12
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="path to clip-vit-large-patch14/model.safetensors")
    ap.add_argument("--out", help="output .gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16"], help="big-matrix dtype")
    ap.add_argument("--list", action="store_true", help="audit and exit, writing nothing")
    args = ap.parse_args()

    print(f"[clip-l] reading {args.src}")
    r = SafetensorsReader(args.src)
    all_names = r.names()

    text = [n for n in all_names if n.startswith(TEXT_PREFIX)]
    keep = [n for n in text if not any(n.endswith(s) for s in DROP_SUFFIXES)]
    dropped_buf = sorted(set(text) - set(keep))
    other = sorted({n.split(".")[0] for n in all_names if not n.startswith(TEXT_PREFIX)})

    print(f"[clip-l] {len(all_names)} tensors in file; {len(text)} under '{TEXT_PREFIX}'")
    print(f"[clip-l] not emitted (not the text tower): {other}")
    if dropped_buf:
        print(f"[clip-l] not emitted (buffers, not parameters): {dropped_buf}")

    arch = derive_arch(r, keep)
    print("[clip-l] arch derived from weights:")
    for k, v in arch.items():
        print(f"           {k:18s} = {v}")

    # Structural assertions -- these catch a wrong/renamed checkpoint for free.
    assert arch["hidden_size"] == 768, f"CLIP-L hidden must be 768 (== HY-Motion vtxt_input_dim), got {arch['hidden_size']}"
    assert arch["vocab_size"] == 49408, f"CLIP BPE vocab must be 49408, got {arch['vocab_size']}"
    assert arch["n_positions"] == 77, f"CLIP-L max_position_embeddings must be 77, got {arch['n_positions']}"
    assert arch["num_layers"] == 12, f"CLIP-L is 12 layers, got {arch['num_layers']}"
    assert arch["intermediate_size"] == 3072, f"CLIP-L mlp is 4x768=3072, got {arch['intermediate_size']}"
    assert len(keep) == 4 + arch["num_layers"] * 16, f"expected 4 + 12*16 = 196 text tensors, got {len(keep)}"
    # If this ever fires, the pooled path in clip.hpp would silently project.
    assert not any(n.startswith(TEXT_PREFIX) and "text_projection" in n for n in all_names), \
        "text_projection must not live under text_model.* -- see the header of this file"

    if args.list:
        for n in sorted(keep):
            print(f"  {n:64s} {r.shape(n)} {r.dtype(n)}")
        return

    if not args.out:
        ap.error("--out is required unless --list")

    import gguf
    from gguf import GGMLQuantizationType as QT

    w = gguf.GGUFWriter(args.out, "hymotion-clip-l")

    w.add_string("hymotion_clip.source", os.path.basename(os.path.dirname(os.path.abspath(args.src))))
    w.add_uint32("hymotion_clip.hidden_size", arch["hidden_size"])
    w.add_uint32("hymotion_clip.vocab_size", arch["vocab_size"])
    w.add_uint32("hymotion_clip.n_positions", arch["n_positions"])
    w.add_uint32("hymotion_clip.num_layers", arch["num_layers"])
    w.add_uint32("hymotion_clip.num_heads", arch["num_heads"])
    w.add_uint32("hymotion_clip.intermediate_size", arch["intermediate_size"])
    # from text_encoder.py:45-46 -- the C++ side must not invent these.
    w.add_string("hymotion_clip.pooling_mode", "pooler_output")
    w.add_uint32("hymotion_clip.max_length", 77)
    w.add_bool("hymotion_clip.text_projection", False)

    big = {"f32": QT.F32, "f16": QT.F16}[args.dtype]
    n_big = 0
    for name in sorted(keep):
        arr = r.to_numpy(name).astype(np.float32)
        # Quant rule mirrors convert_hymotion.py: only 2-D weight matrices whose last
        # dim is a multiple of 32. LayerNorm weights/biases and all biases stay F32.
        is_big = arr.ndim == 2 and arr.shape[-1] % 32 == 0 and name.endswith(".weight")
        # ... except the position embedding, which clip.hpp declares as a HARDCODED F32
        # (`position_wtype = GGML_TYPE_F32`, clip.hpp:140 -- unlike token_embedding,
        # whose type it reads back from the file via get_type()). Emitting it as F16
        # would force a dtype conversion at load time for a 236KB tensor. Not worth a
        # single byte of risk.
        if name.endswith("embeddings.position_embedding.weight"):
            is_big = False
        if is_big and big != QT.F32:
            w.add_tensor(name, arr.astype(np.float16))
            n_big += 1
        else:
            w.add_tensor(name, arr)

    print(f"[clip-l] writing {args.out}  ({len(keep)} tensors, {n_big} as {args.dtype})")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"[clip-l] done: {args.out}  {os.path.getsize(args.out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
