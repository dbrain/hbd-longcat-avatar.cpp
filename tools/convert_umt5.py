#!/usr/bin/env python3
"""
umT5-XXL text encoder (LongCat base/text_encoder, HF `UMT5EncoderModel`) -> GGUF
for longcat-avatar.cpp / ggml (sd.cpp T5Runner with is_umt5=true).

The stock HF umT5 checkpoint already uses the *original* T5 tensor layout that
sd.cpp's T5/T5Runner expects:
    shared.weight
    encoder.block.N.layer.0.SelfAttention.{q,k,v,o}.weight
    encoder.block.N.layer.0.SelfAttention.relative_attention_bias.weight  (every block, umT5)
    encoder.block.N.layer.0.layer_norm.weight
    encoder.block.N.layer.1.DenseReluDense.{wi_0,wi_1,wo}.weight
    encoder.block.N.layer.1.layer_norm.weight
    encoder.final_layer_norm.weight

sd.cpp loads `--t5xxl` under prefix `text_encoders.t5xxl.transformer.` and then runs
convert_tensor_name(); the t5_name_map there only rewrites llama.cpp-style names
(`enc.`/`blk.`) -> original, so it's a no-op on these already-original names. The
runner is built with is_umt5=true (vocab 256384, per-block relative_attention_bias).

So conversion = prepend `text_encoders.t5xxl.transformer.` to every tensor and copy
through. This is an *encoder-only* model (UMT5EncoderModel) — there are no decoder
tensors to drop. We skip `encoder.embed_tokens.weight` if present (sd.cpp lists it as
training-only / ignored; `shared.weight` is the embedding it actually uses).

Default Q8_0 for 2D weight matrices (text encoder; CPU-offloadable; ~5.6B params).
Norms (1D) and the relative_attention_bias [32,64] stay F16. `shared.weight`
[256384,4096] is quantized (row len 4096 % 32 == 0).

No torch: safetensors header parse + mmap across the 5 shards; this file is F32.

Usage:
    TMPDIR=/home/dbrain/dev/longcat-avatar.cpp/.convert-tmp \
    uv run --with numpy --with gguf python3 tools/convert_umt5.py \
        --src /mnt/hdd/longcat/base/text_encoder \
        --out models/longcat-umt5-xxl-q8_0.gguf --dtype q8_0
"""
import argparse, json, mmap, os, struct, sys, time
import numpy as np

_ST_DT = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Safetensors:
    def __init__(self, path):
        self.path = path
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return self.header.keys()

    def get(self, name):
        e = self.header[name]
        dt = e["dtype"]
        a, b = e["data_offsets"]
        buf = self.mm[self.data_start + a: self.data_start + b]
        if dt == "BF16":
            arr = _bf16_to_f32(np.frombuffer(buf, dtype=np.uint16))
        else:
            arr = np.frombuffer(buf, dtype=_ST_DT[dt])
        return arr.reshape(e["shape"])

    def close(self):
        self.mm.close()
        self.f.close()


def load_index(src):
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.isfile(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(src, v) for k, v in wm.items()}
    if src.endswith(".safetensors"):
        st = Safetensors(src)
        m = {k: src for k in st.keys()}
        st.close()
        return m
    raise SystemExit(f"no index.json and not a .safetensors file: {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="text_encoders.t5xxl.transformer.")
    ap.add_argument("--dtype", default="q8_0", choices=["f16", "q8_0", "q4_0", "q5_0"])
    args = ap.parse_args()

    import gguf
    from gguf import GGMLQuantizationType as QT
    from gguf import quants

    QMAP = {"f16": QT.F16, "q8_0": QT.Q8_0, "q4_0": QT.Q4_0, "q5_0": QT.Q5_0}
    big_qt = QMAP[args.dtype]

    name2shard = load_index(args.src)
    shards = {}
    def shard(path):
        if path not in shards:
            shards[path] = Safetensors(path)
        return shards[path]

    writer = gguf.GGUFWriter(args.out, arch="umt5-xxl", use_temp_file=True)
    writer.add_description("LongCat base umT5-XXL encoder (-> sd.cpp t5xxl names)")

    t0 = time.time()
    n_q = n_copy = 0
    total = 0
    skipped = []
    for name in sorted(name2shard):
        # encoder-only model; the only non-encoder tensor is `shared.weight` (kept).
        # drop training-only embed_tokens duplicate if present.
        if name.endswith("encoder.embed_tokens.weight"):
            skipped.append(name)
            continue
        arr = shard(name2shard[name]).get(name)
        total += arr.size
        out_name = args.prefix + name
        # quantize 2D weight matrices when row length % 32 == 0; keep rel-bias + norms f16
        is_relbias = name.endswith("relative_attention_bias.weight")
        if (big_qt != QT.F16 and arr.ndim == 2 and not is_relbias
                and arr.shape[-1] % 32 == 0):
            qd = quants.quantize(np.ascontiguousarray(arr, dtype=np.float32), big_qt)
            writer.add_tensor(out_name, qd, raw_dtype=big_qt)
            n_q += 1
        else:
            writer.add_tensor(out_name, np.ascontiguousarray(arr, dtype=np.float16))
            n_copy += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    for st in shards.values():
        st.close()

    sz = os.path.getsize(args.out) / 1e9
    print(f"\nwrote {args.out}")
    print(f"  tensors: {n_q} {args.dtype} + {n_copy} f16 = {n_q + n_copy}")
    if skipped:
        print(f"  skipped (training-only): {skipped}")
    print(f"  params: {total/1e9:.2f}B   file: {sz:.2f} GB   in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
