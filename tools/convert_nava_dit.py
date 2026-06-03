#!/usr/bin/env python3
"""
NAVA 6.3B joint audio-video MMDiT  ->  GGUF converter (for nava.hpp / ggml).

The master checkpoint `/mnt/hdd/nava/NAVA.safetensors` stores the whole DiT as a
single safetensors file, **all tensors F32** (1052 tensors, ~6.297B params, ~25GB).
No bf16 decode, no int8+scale (unlike longcat), no shards. Output tensor names are
the NAVA names verbatim under `--prefix` (default `backbone.` is already the source
prefix, so default here is empty); the C++ nava model loads by exactly these names.

Quant rule (mirrors convert_longcat_avatar.py): 2-D weight matrices whose last dim
is a multiple of 32 -> requested big-quant (q8_0/q4_0/q5_0). Everything else
(norms, biases, modulation, conv patch-embeds incl. the 3-D audio Conv1d stack,
speaker null_token) -> F16. K-quants (Q4_K) aren't in the numpy `gguf` package:
convert to F16 here, then requantize with `sd-cli --mode convert --tensor-type-rules`.

The video patch_embedding Conv3d weight is [3072,48,1,2,2] (5-D, temporal=1); ggml
is max-4D so we squeeze the singleton temporal dim -> [3072,48,2,2].

Usage:
    uv run --with numpy --with gguf python3 tools/convert_nava_dit.py \
        --src /mnt/hdd/nava/NAVA.safetensors \
        --out models/nava-dit-f16.gguf --dtype f16
    # then for Q4_K:
    #   sd-cli --mode convert -m models/nava-dit-f16.gguf \
    #          -o models/nava-dit-q4_k.gguf --tensor-type-rules "...=q4_k"
    # debug: --max-blocks 1   (tiny gguf, fast round-trip)
"""
import argparse, json, mmap, os, struct, time
import numpy as np

_ST_DT = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
}


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Safetensors:
    """Lazy reader over one safetensors file (header parse + mmap)."""

    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return self.header.keys()

    def get(self, name: str) -> np.ndarray:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="NAVA.safetensors (single F32 file)")
    ap.add_argument("--out", required=True, help="output .gguf path (on the main SSD!)")
    ap.add_argument("--prefix", default="",
                    help="prefix prepended to every tensor name (NAVA names already "
                         "start with 'backbone.'; default empty keeps them verbatim)")
    ap.add_argument("--dtype", default="f16", choices=["f16", "q8_0", "q4_0", "q5_0"],
                    help="output type for 2D weight matrices (norms/bias/conv stay f16)")
    ap.add_argument("--max-blocks", type=int, default=-1,
                    help="debug: only convert double/single blocks [0,N); -1 = all")
    args = ap.parse_args()

    import gguf
    from gguf import GGMLQuantizationType as QT
    from gguf import quants

    QMAP = {"f16": QT.F16, "q8_0": QT.Q8_0, "q4_0": QT.Q4_0, "q5_0": QT.Q5_0}
    big_qt = QMAP[args.dtype]

    st = Safetensors(args.src)
    names = list(st.keys())

    def keep(name: str) -> bool:
        if args.max_blocks < 0:
            return True
        # NAVA block names look like backbone.double_blocks.{i}.* / backbone.single_blocks.{i}.*
        for tag in ("double_blocks.", "single_blocks."):
            if tag in name:
                i = int(name.split(tag, 1)[1].split(".", 1)[0])
                return i < args.max_blocks
        return True

    names = [n for n in names if keep(n)]

    writer = gguf.GGUFWriter(args.out, arch="nava", use_temp_file=True)
    writer.add_description("NAVA 6.3B joint audio-video MMDiT (F32 -> gguf)")

    t0 = time.time()
    n_q = n_copy = 0
    total_params = 0

    def add(name: str, arr: np.ndarray):
        nonlocal n_q, n_copy, total_params
        total_params += arr.size
        out_name = args.prefix + name
        # 2-D weight matrix, last dim % 32 == 0  ->  block-quant
        if arr.ndim == 2 and big_qt != QT.F16 and arr.shape[-1] % 32 == 0:
            qd = quants.quantize(np.ascontiguousarray(arr, dtype=np.float32), big_qt)
            writer.add_tensor(out_name, qd, raw_dtype=big_qt)
            n_q += 1
            return
        writer.add_tensor(out_name, np.ascontiguousarray(arr, dtype=np.float16))
        n_copy += 1

    for name in sorted(names):
        arr = st.get(name)
        # squeeze the singleton temporal dim of the video Conv3d patch embed
        # [3072,48,1,2,2] -> [3072,48,2,2] (ggml is max-4D).
        if name.endswith("patch_embedding.weight") and arr.ndim == 5:
            assert arr.shape[2] == 1, f"unexpected conv temporal dim: {arr.shape}"
            arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[3], arr.shape[4])
        add(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    st.close()

    dt = time.time() - t0
    sz = os.path.getsize(args.out) / 1e9
    print(f"\nwrote {args.out}")
    print(f"  tensors: {n_q} quantized({args.dtype}) + {n_copy} copy(f16) = {n_q + n_copy}")
    print(f"  params : {total_params/1e9:.2f}B   file: {sz:.2f} GB   in {dt:.1f}s")


if __name__ == "__main__":
    main()
