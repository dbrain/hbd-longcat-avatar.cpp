#!/usr/bin/env python3
"""
Wan2.2-I2V-A14B (lightx2v Moe-Distill) DiT safetensors -> GGUF converter.

Each expert `distill_models/{high,low}_noise_model/distill_model.safetensors` is a single
BF16 safetensors with the FLAT diffusers Wan naming (blocks.{i}.self_attn/cross_attn/ffn/
modulation/norm3; top-level patch_embedding/text_embedding/time_embedding/time_projection.1/
head). 1095 tensors, all BF16. No fp8, no shards.

Emit FLAT names (default --prefix ""): the C++ loader (model_loader.cpp:280) PREPENDS the
runtime prefix (`model.diffusion_model.` for --diffusion-model, `model.high_noise_diffusion_model.`
for --high-noise-diffusion-model) when absent, so ONE flat gguf per expert loads as either leg.
The video Conv3d patch_embedding [5120,36,1,2,2] is squeezed to [5120,36,2,2] (ggml is max-4D,
temporal dim == 1).

K-quants aren't in the numpy `gguf` package -> write F16 here, then requantize with sd-cli:
    sd-cli -M convert -m <f16.gguf> -o <q4_k.gguf> --type q4_K

Usage:
    uv run --with numpy --with gguf python3 tools/convert_wan_dit.py \
        --src models/dl/wan22-i2v-a14b-moe-distill/distill_models/low_noise_model/distill_model.safetensors \
        --out models/wan22-i2v-a14b-low-f16.gguf
    # debug tiny round-trip: --max-blocks 1
"""
import argparse, json, mmap, os, struct, time
import numpy as np

_ST_DT = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64,
          "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Safetensors:
    def __init__(self, path: str):
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
        a, b = e["data_offsets"]
        buf = self.mm[self.data_start + a: self.data_start + b]
        if e["dtype"] == "BF16":
            arr = _bf16_to_f32(np.frombuffer(buf, dtype=np.uint16))
        else:
            arr = np.frombuffer(buf, dtype=_ST_DT[e["dtype"]])
        return arr.reshape(e["shape"])

    def close(self):
        self.mm.close()
        self.f.close()


def load_lora(path):
    """lightx2v/ComfyUI distill LoRA -> per-base-tensor deltas (lora_down/up, diff, diff_b),
    scale 1.0/no-alpha. Same format as the Wan2.1 InfiniteTalk distill. Returns
    {base_tensor_name: [('lora',down,up,scale) | ('add',arr), ...]}."""
    st = Safetensors(path)

    def base_of(stem):
        for pfx in ("diffusion_model.", "transformer."):
            if stem.startswith(pfx):
                return stem[len(pfx):]
        return stem

    deltas = {}
    for k in list(st.keys()):
        if k.endswith(".lora_down.weight"):
            up_k = k.replace("lora_down.weight", "lora_up.weight")
            if up_k not in st.header:
                continue
            tgt = base_of(k[:-len(".lora_down.weight")]) + ".weight"
            deltas.setdefault(tgt, []).append(
                ("lora", st.get(k).astype(np.float32), st.get(up_k).astype(np.float32), 1.0))
        elif k.endswith(".diff"):
            deltas.setdefault(base_of(k[:-len(".diff")]) + ".weight", []).append(("add", st.get(k).astype(np.float32)))
        elif k.endswith(".diff_b"):
            deltas.setdefault(base_of(k[:-len(".diff_b")]) + ".bias", []).append(("add", st.get(k).astype(np.float32)))
    st.close()
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="one expert's distill_model.safetensors")
    ap.add_argument("--out", required=True, help="output f16 .gguf (on the main SSD)")
    ap.add_argument("--lora", default="", help="optional distill LoRA (e.g. lightx2v rank-64 per-expert) to fold in")
    ap.add_argument("--prefix", default="", help="name prefix; default empty (loader prepends)")
    ap.add_argument("--max-blocks", type=int, default=-1, help="debug: only blocks [0,N); -1=all")
    args = ap.parse_args()

    import gguf

    lora = load_lora(args.lora) if args.lora else {}
    n_lora_applied = 0
    total_deltas = sum(len(v) for v in lora.values())
    if args.lora:
        print(f"  loaded {total_deltas} LoRA deltas ({len(lora)} targets) from {os.path.basename(args.lora)}")

    st = Safetensors(args.src)
    names = list(st.keys())
    if args.max_blocks >= 0:
        def keep(n: str) -> bool:
            if n.startswith("blocks.") or ".blocks." in n:
                i = int(n.split("blocks.", 1)[1].split(".", 1)[0])
                return i < args.max_blocks
            return True
        names = [n for n in names if keep(n)]

    writer = gguf.GGUFWriter(args.out, arch="wan", use_temp_file=True)
    writer.add_description("Wan2.2-I2V-A14B distill (bf16 -> f16 gguf)")

    t0 = time.time()
    n_t = 0
    total = 0
    for name in sorted(names):
        arr = st.get(name).astype(np.float32)
        if name in lora:
            for d in lora[name]:
                if d[0] == "lora":
                    _, down, up, scale = d
                    if arr.ndim == 2 and arr.shape == (up.shape[0], down.shape[1]):
                        arr = arr + (up @ down) * scale
                        n_lora_applied += 1
                    else:
                        print(f"  LoRA shape mismatch, skip {name}: W{arr.shape} up{up.shape} down{down.shape}")
                elif d[1].shape == arr.shape:
                    arr = arr + d[1]
                    n_lora_applied += 1
                else:
                    print(f"  LoRA diff shape mismatch, skip {name}: {arr.shape} vs {d[1].shape}")
        # video Conv3d patch embed [out,in,kt=1,kh,kw]. The C++ Conv3d stores its weight as
        # ggml ne [kw,kh,kt, in*out] (ggml_ext_conv_3d: OC=ne[3]/IC, IC-fastest/OC-slowest),
        # i.e. numpy [out*in, kt, kh, kw] with the merged dim out-major. Merge out&in, KEEP kt.
        if name.endswith("patch_embedding.weight") and arr.ndim == 5:
            o, i, kt, kh, kw = arr.shape
            assert kt == 1, f"unexpected conv temporal dim: {arr.shape}"
            arr = arr.reshape(o * i, kt, kh, kw)
        writer.add_tensor(args.prefix + name, np.ascontiguousarray(arr, dtype=np.float16))
        n_t += 1
        total += arr.size

    if args.lora and n_lora_applied != total_deltas:
        print(f"  LoRA WARN: applied {n_lora_applied}/{total_deltas} deltas (unmatched -> NOT folded)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    st.close()

    sz = os.path.getsize(args.out) / 1e9
    print(f"wrote {args.out}: {n_t} tensors f16 ({n_lora_applied} lora-folded), "
          f"{total/1e9:.2f}B params, {sz:.2f}GB in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
