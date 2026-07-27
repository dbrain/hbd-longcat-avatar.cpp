#!/usr/bin/env python3
"""
LongCat-Video-Avatar 1.5 DiT  ->  GGUF converter (for longcat-avatar.cpp / ggml).

The reference checkpoint (`base_model_int8/`) stores the 48-block video DiT as
weight-only INT8, per-channel symmetric:  for each Linear `X`,
    X.weight_int8   I8  [out, in]
    X.weight_scale  F32 [out]
and the de-quantized weight is  w[o, i] = int8[o, i] * scale[o].
Everything else (norms, biases, the Conv3d patch embed, final_layer.linear) is
stored BF16 and copied through.

We can't read that custom int8+scale layout in sd.cpp directly, so this tool
de-quantizes to a normal GGUF. Output tensor names are the LongCat names verbatim
(under `--prefix`, default `model.diffusion_model.`); the C++ avatar model loads
by exactly these names.

K-quants (Q4_K/Q5_K) are not available in the pure-numpy `gguf` package, so the
intended pipeline is:  convert here to F16  ->  requantize to Q4_K with sd-cli
`--mode convert --tensor-type-rules`. F16 is exact w.r.t. the int8 source.

No torch: safetensors headers are parsed by hand and data is mmap'd.

Usage:
    uv run --with numpy --with gguf python3 tools/convert_longcat_avatar.py \
        --src /mnt/hdd/longcat/avatar-1.5/base_model_int8 \
        --out models/longcat-avatar-1.5-dit-f16.gguf \
        --dtype f16
    # debug: --max-blocks 1   (tiny gguf, fast round-trip validation)
"""
import argparse, json, mmap, os, struct, sys, time
import numpy as np

# ---- safetensors dtype -> numpy ------------------------------------------------
_ST_DT = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
    # BF16 handled specially (numpy has no native bf16)
}


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """uint16 bf16 bits -> float32 (exact; bf16 is just the high 16 bits of f32)."""
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Safetensors:
    """Lazy reader over one safetensors shard (header parse + mmap)."""

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


class DmdLora:
    """Folds the DMD distillation LoRA into base weights at dequant time.

    The reference (`modules/lora_utils.py`) applies, per target Linear:
        out = W @ x + multiplier * alpha_scale * (up @ (down @ x))
    so the effective weight is  W_eff = W + multiplier * alpha_scale * (up @ down).

    Module names in the safetensors are `lora___lorahyphen___<a>___lorahyphen___<b>...`
    which decode to the dotted module path (e.g. `blocks.0.attn.qkv`). They map 1:1
    to our gguf tensor base names (the LongCat names are verbatim).

    For fused outputs (qkv n_seperate=3, cross_attn.kv_linear n_seperate=2) the up
    is split into `lora_up.blocks.{i}.weight` each `[out/n, rank]`, and `down` is
    `[n*rank, in]`. Output-row-block i uses `up_i @ down[i*rank:(i+1)*rank, :]`.
    """

    def __init__(self, path: str, multiplier: float):
        self.st = Safetensors(path)
        self.multiplier = float(multiplier)
        self.modules = {}  # dotted module name -> dict of suffix->raw key
        for k in self.st.keys():
            base, _, suffix = k.partition(".")  # base = lora___...___qkv, suffix = lora_down.weight etc.
            mod = base.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
            self.modules.setdefault(mod, {})[suffix] = k
        self.applied = set()

    def has(self, module_name: str) -> bool:
        return module_name in self.modules

    def delta(self, module_name: str) -> np.ndarray:
        """Return the f32 weight delta [out, in] for a target module."""
        m = self.modules[module_name]
        alpha_scale = float(self.st.get(m["alpha_scale"]))  # 0.5 for dim128/alpha64
        down = self.st.get(m["lora_down.weight"]).astype(np.float32)  # [n*rank, in]
        up_keys = sorted(k for k in m if k.startswith("lora_up.blocks."))
        if up_keys:
            n = len(up_keys)
            rank = down.shape[0] // n
            blocks = []
            for i, uk in enumerate(sorted(up_keys, key=lambda s: int(s.split(".")[2]))):
                up_i = self.st.get(m[uk]).astype(np.float32)        # [out/n, rank]
                down_i = down[i * rank:(i + 1) * rank, :]           # [rank, in]
                blocks.append(up_i @ down_i)                        # [out/n, in]
            delta = np.concatenate(blocks, axis=0)                  # [out, in]
        else:
            up = self.st.get(m["lora_up.weight"]).astype(np.float32)  # [out, rank]
            delta = up @ down                                          # [out, in]
        self.applied.add(module_name)
        return (self.multiplier * alpha_scale) * delta

    def close(self):
        self.st.close()


def load_index(src: str):
    """Return {tensor_name: shard_path} across all shards (or a single file)."""
    idx = os.path.join(src, "quantized_model.safetensors.index.json")
    if os.path.isfile(idx):
        wm = json.load(open(idx))["weight_map"]
        return {k: os.path.join(src, v) for k, v in wm.items()}
    # single safetensors file fallback
    if src.endswith(".safetensors"):
        st = Safetensors(src)
        m = {k: src for k in st.keys()}
        st.close()
        return m
    raise SystemExit(f"no index.json and not a .safetensors file: {src}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="base_model_int8 dir (or single .safetensors)")
    ap.add_argument("--out", required=True, help="output .gguf path (on the main SSD!)")
    ap.add_argument("--prefix", default="model.diffusion_model.",
                    help="prefix prepended to every tensor name")
    ap.add_argument("--dtype", default="f16", choices=["f16", "q8_0", "q4_0", "q5_0"],
                    help="output type for 2D weight matrices (norms/bias/conv stay f16/f32)")
    ap.add_argument("--max-blocks", type=int, default=-1,
                    help="debug: only convert blocks [0, N); -1 = all")
    ap.add_argument("--dmd-lora", default=None,
                    help="path to dmd_lora.safetensors; if set, the DMD distillation "
                         "LoRA is FOLDED into the dequantized weights before requant "
                         "(W += multiplier * alpha_scale * (up @ down)). Required for "
                         "8-step distilled inference. alpha_scale is read from the "
                         "file (dim128/alpha64 => 0.5).")
    ap.add_argument("--dmd-multiplier", type=float, default=1.0,
                    help="LoRA multiplier (reference uses 1.0)")
    args = ap.parse_args()

    import gguf
    from gguf import GGMLQuantizationType as QT
    from gguf import quants

    QMAP = {"f16": QT.F16, "q8_0": QT.Q8_0, "q4_0": QT.Q4_0, "q5_0": QT.Q5_0}
    big_qt = QMAP[args.dtype]

    name2shard = load_index(args.src)
    # open each shard once
    shards = {}
    def shard(path):
        if path not in shards:
            shards[path] = Safetensors(path)
        return shards[path]

    # group the int8/scale pairs: base name -> dict of suffix->tensor name
    # a "Linear X" appears as X.weight_int8 + X.weight_scale; copy-through tensors
    # appear as plain names (X.weight, X.bias, norms ...).
    names = list(name2shard)

    def keep_block(name: str) -> bool:
        if args.max_blocks < 0 or not name.startswith("blocks."):
            return True
        b = int(name.split(".")[1])
        return b < args.max_blocks

    names = [n for n in names if keep_block(n)]
    int8_bases = {n[:-len(".weight_int8")] for n in names if n.endswith(".weight_int8")}
    scales = {n for n in names if n.endswith(".weight_scale")}
    plain = [n for n in names if not n.endswith(".weight_int8") and not n.endswith(".weight_scale")]

    dmd = DmdLora(args.dmd_lora, args.dmd_multiplier) if args.dmd_lora else None
    if dmd:
        print(f"DMD LoRA: {len(dmd.modules)} target modules, multiplier={args.dmd_multiplier}")

    writer = gguf.GGUFWriter(args.out, arch="longcat-avatar", use_temp_file=True)
    writer.add_description("LongCat-Video-Avatar 1.5 DiT (int8 de-quant -> gguf)")

    t0 = time.time()
    n_q = n_copy = 0
    total_params = 0

    def add(name: str, arr: np.ndarray, quantize: bool):
        nonlocal n_q, n_copy, total_params
        total_params += arr.size
        out_name = args.prefix + name
        if quantize and arr.ndim == 2 and big_qt != QT.F16:
            # k-block quants need the row length divisible by the block size; the
            # numpy-supported quants here (q8_0/q4_0/q5_0) use block 32. Fall back
            # to f16 if a row isn't a multiple of 32.
            if arr.shape[-1] % 32 == 0:
                # quants.quantize returns the byte-packed array; raw_dtype lets gguf
                # recover the logical shape. Do NOT pass raw_shape (it's read as the
                # byte-row length and mis-validates).
                qd = quants.quantize(np.ascontiguousarray(arr, dtype=np.float32), big_qt)
                writer.add_tensor(out_name, qd, raw_dtype=big_qt)
                n_q += 1
                return
        # copy-through as f16
        writer.add_tensor(out_name, np.ascontiguousarray(arr, dtype=np.float16))
        n_copy += 1

    # 1) de-quantized linears (+ optional DMD LoRA fold)
    n_folded = 0
    for base in sorted(int8_bases):
        w8 = shard(name2shard[base + ".weight_int8"]).get(base + ".weight_int8")
        sc = shard(name2shard[base + ".weight_scale"]).get(base + ".weight_scale")
        # per-output-channel symmetric: w[o,i] = int8[o,i] * scale[o]
        w = w8.astype(np.float32) * sc.reshape(-1, *([1] * (w8.ndim - 1)))
        if dmd is not None and dmd.has(base):
            delta = dmd.delta(base)  # [out, in]
            assert delta.shape == w.shape, f"DMD delta shape {delta.shape} != weight {w.shape} for {base}"
            w = w + delta
            n_folded += 1
        add(base + ".weight", w, quantize=True)
    if dmd is not None:
        unapplied = set(dmd.modules) - dmd.applied
        if unapplied:
            print(f"  WARNING: {len(unapplied)} LoRA modules had no matching base weight: "
                  f"{sorted(unapplied)[:4]}{'...' if len(unapplied) > 4 else ''}")
        print(f"  DMD LoRA folded into {n_folded} weights")
        dmd.close()

    # 2) copy-through tensors (norms, biases, conv patch embed, final_layer.linear, embeds)
    for name in sorted(plain):
        arr = shard(name2shard[name]).get(name)
        # GGUF/ggml is max-4D: squeeze the singleton temporal patch dim of the
        # Conv3d patch embed  [4096,16,1,2,2] -> [4096,16,2,2].
        if name.endswith("x_embedder.proj.weight") and arr.ndim == 5:
            assert arr.shape[2] == 1, f"unexpected conv temporal dim: {arr.shape}"
            arr = arr.reshape(arr.shape[0], arr.shape[1], arr.shape[3], arr.shape[4])
        add(name, arr, quantize=False)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    for st in shards.values():
        st.close()

    dt = time.time() - t0
    sz = os.path.getsize(args.out) / 1e9
    print(f"\nwrote {args.out}")
    print(f"  tensors: {n_q} quantized({args.dtype}) + {n_copy} copy(f16) = {n_q + n_copy}")
    print(f"  params : {total_params/1e9:.2f}B   file: {sz:.2f} GB   in {dt:.1f}s")


if __name__ == "__main__":
    main()
