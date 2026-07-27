#!/usr/bin/env python3
"""
whisper-large-v3 ENCODER ONLY (LongCat avatar-1.5 audio feature extractor) -> GGUF
for longcat-avatar.cpp / ggml.

We use whisper purely as an audio feature extractor: the avatar AudioProjModel
consumes stacked hidden states from `audio_block=5` intermediate encoder layers.
The decoder is NEVER run, so all `model.decoder.*` tensors are dropped.

Source: HF `WhisperForConditionalGeneration` safetensors. Encoder tensors live under
`model.encoder.`:
    conv1.{weight,bias}            Conv1d  in=128(mel) out=1280  k=3 s=1 p=1
    conv2.{weight,bias}            Conv1d  in=1280 out=1280       k=3 s=2 p=1
    embed_positions.weight         [1500,1280]  (sinusoidal pos emb, fixed)
    layers.N.self_attn.{q,k,v,out}_proj.weight   (+ q/v/out bias; k has NO bias)
    layers.N.self_attn_layer_norm.{weight,bias}
    layers.N.fc1.{weight,bias}     1280->5120
    layers.N.fc2.{weight,bias}     5120->1280
    layers.N.final_layer_norm.{weight,bias}
    layer_norm.{weight,bias}       (post-encoder LN, applied after all layers)

Output names keep the HF encoder names verbatim under prefix `audio_encoder.`
(strip the `model.encoder.` HF prefix first), e.g.
    audio_encoder.conv1.weight
    audio_encoder.layers.5.self_attn.q_proj.weight
    audio_encoder.layer_norm.weight

F16 throughout (already F16 in source). Conv1d weights are 3D PyTorch
[out,in,k] -> ggml stores reversed [k,in,out]; we keep the 3D shape (no merge needed,
ggml is fine with 3D). The C++ side builds the encoder graph from these dims:
    n_layers=32, d_model=1280, n_heads=20 (head_dim=64), ffn_dim=5120,
    num_mel_bins=128, conv k=3 (conv2 stride 2), max_source_positions=1500.

No torch: safetensors header parse + mmap.

Usage:
    TMPDIR=/home/dbrain/dev/longcat-avatar.cpp/.convert-tmp \
    uv run --with numpy --with gguf python3 tools/convert_whisper_encoder.py \
        --src /mnt/hdd/longcat/avatar-1.5/whisper-large-v3/model.safetensors \
        --out models/longcat-whisper-v3-encoder-f16.gguf
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


HF_ENC_PREFIX = "model.encoder."
OUT_PREFIX = "audio_encoder."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default=OUT_PREFIX)
    args = ap.parse_args()

    import gguf

    st = Safetensors(args.src)
    names = sorted(st.keys())

    writer = gguf.GGUFWriter(args.out, arch="whisper-encoder", use_temp_file=True)
    writer.add_description("whisper-large-v3 ENCODER ONLY (LongCat avatar audio features, f16)")
    # stash the encoder config so the C++ side can read it back
    writer.add_uint32("whisper.encoder.n_layers", 32)
    writer.add_uint32("whisper.encoder.d_model", 1280)
    writer.add_uint32("whisper.encoder.n_heads", 20)
    writer.add_uint32("whisper.encoder.head_dim", 64)
    writer.add_uint32("whisper.encoder.ffn_dim", 5120)
    writer.add_uint32("whisper.encoder.num_mel_bins", 128)
    writer.add_uint32("whisper.encoder.conv_kernel", 3)
    writer.add_uint32("whisper.encoder.conv2_stride", 2)
    writer.add_uint32("whisper.encoder.max_source_positions", 1500)

    t0 = time.time()
    n_enc = n_dec = n_other = 0
    for name in names:
        if name.startswith("model.decoder.") or name == "proj_out.weight":
            n_dec += 1
            continue
        if not name.startswith(HF_ENC_PREFIX):
            # any stray non-encoder tensor (none expected) -> drop, count it
            n_other += 1
            continue
        arr = st.get(name)
        out_name = args.prefix + name[len(HF_ENC_PREFIX):]
        writer.add_tensor(out_name, np.ascontiguousarray(arr, dtype=np.float16))
        n_enc += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    st.close()

    sz = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}")
    print(f"  encoder tensors: {n_enc}   dropped decoder/proj: {n_dec}   other-dropped: {n_other}")
    print(f"  file: {sz:.1f} MB   in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
