#!/usr/bin/env python3
"""
Wan2.1 VAE (LongCat base, HF diffusers `AutoencoderKLWan`) -> GGUF for
longcat-avatar.cpp / ggml (sd.cpp WanVAERunner).

The stock LongCat VAE (`base/vae/diffusion_pytorch_model.safetensors`) is in the
HuggingFace *diffusers* layout (`encoder.down_blocks.N`, `decoder.up_blocks.N`,
`mid_block.resnets`, `*.norm.gamma`, ...). sd.cpp's WAN::WanVAE expects the
*original* Wan (ComfyUI) layout (`encoder.downsamples.N`, `decoder.upsamples.N`,
`middle.{0,1,2}`, `head.{0,2}`, residual blocks as `residual.{0,2,3,6}` + `shortcut`).
sd.cpp has NO diffusers->wan-vae name converter (only an SD1 VAE one), so we remap
here and emit names sd.cpp will request.

WanVAERunner is constructed with prefix "first_stage_model" and asks for
  first_stage_model.<wanname>
When this gguf is passed via `--vae`, sd.cpp prepends `vae.` then
convert_first_stage_model_name() rewrites `vae.`->`first_stage_model.` and runs
convert_diffusers_vae_to_original_sd1() (a no-op on Wan names — none of its SD1
substrings appear). So we emit the *bare* wan names (no prefix) here.

F16 (VAE wants precision). Conv weights are PyTorch 5D [out,in,kt,kh,kw]; sd.cpp's
CausalConv3d param is 4D ggml [kw,kh,kt,in*out], so we merge [out,in]->[out*in]
(out-major, matching PyTorch row-major flatten) and keep [out*in,kt,kh,kw], which
ggml stores reversed as [kw,kh,kt,out*in]. 2D attn convs (1x1, 1x1x1) stay as-is.

No torch: safetensors header parse + mmap; bf16->f32 by hand (this file is F32).

Usage:
    TMPDIR=/home/dbrain/dev/longcat-avatar.cpp/.convert-tmp \
    uv run --with numpy --with gguf python3 tools/convert_wan_vae.py \
        --src /mnt/hdd/longcat/base/vae/diffusion_pytorch_model.safetensors \
        --out models/longcat-wan-vae-f16.gguf
"""
import argparse, json, mmap, os, re, struct, sys, time
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


# ----- residual block: diffusers (conv1/norm1/conv2/norm2/conv_shortcut) ->
#       sd.cpp wan (residual.0=RMS_norm, residual.2=conv, residual.3=RMS_norm,
#       residual.6=conv, shortcut=conv).
_RES_SUFFIX = {
    "norm1.gamma":          "residual.0.gamma",
    "conv1.weight":         "residual.2.weight",
    "conv1.bias":           "residual.2.bias",
    "norm2.gamma":          "residual.3.gamma",
    "conv2.weight":         "residual.6.weight",
    "conv2.bias":           "residual.6.bias",
    "conv_shortcut.weight": "shortcut.weight",
    "conv_shortcut.bias":   "shortcut.bias",
}


def map_res_suffix(suffix):
    if suffix not in _RES_SUFFIX:
        raise KeyError(f"unknown residual-block suffix: {suffix}")
    return _RES_SUFFIX[suffix]


def map_name(name):
    """diffusers AutoencoderKLWan name -> sd.cpp wan name (no prefix). None=drop."""
    # top-level quant convs
    if name.startswith("quant_conv."):
        return "conv1." + name[len("quant_conv."):]          # encoder post-conv
    if name.startswith("post_quant_conv."):
        return "conv2." + name[len("post_quant_conv."):]     # decoder pre-conv

    # encoder / decoder split
    m = re.match(r"^(encoder|decoder)\.(.*)$", name)
    if not m:
        raise KeyError(f"unexpected top-level name: {name}")
    side, rest = m.group(1), m.group(2)

    # conv_in -> conv1 ; conv_out -> head.2 ; norm_out -> head.0
    if rest.startswith("conv_in."):
        return f"{side}.conv1." + rest[len("conv_in."):]
    if rest.startswith("conv_out."):
        return f"{side}.head.2." + rest[len("conv_out."):]
    if rest.startswith("norm_out."):
        # norm_out.gamma -> head.0.gamma
        return f"{side}.head.0." + rest[len("norm_out."):]

    # mid_block: resnets.0->middle.0, attentions.0->middle.1, resnets.1->middle.2
    mm = re.match(r"^mid_block\.(resnets|attentions)\.(\d+)\.(.*)$", rest)
    if mm:
        kind, idx, suffix = mm.group(1), int(mm.group(2)), mm.group(3)
        if kind == "resnets":
            mid = {0: "middle.0", 1: "middle.2"}[idx]
            return f"{side}.{mid}." + map_res_suffix(suffix)
        else:  # attentions.0 -> middle.1 ; attn sub-names already match sd.cpp
            # diffusers: norm.gamma, to_qkv.{w,b}, proj.{w,b}
            return f"{side}.middle.1." + suffix

    # encoder down_blocks.N / decoder up_blocks.B
    if side == "encoder":
        dm = re.match(r"^down_blocks\.(\d+)\.(.*)$", rest)
        if not dm:
            raise KeyError(f"unexpected encoder name: {rest}")
        idx, suffix = int(dm.group(1)), dm.group(2)
        # diffusers already flattens encoder downsamples into down_blocks.N
        # (res/res/resample/res/res/resample/...), exactly sd.cpp's downsamples.N.
        if suffix.startswith("resample.") or suffix.startswith("time_conv."):
            return f"encoder.downsamples.{idx}.{suffix}"   # Resample sub-names match
        return f"encoder.downsamples.{idx}." + map_res_suffix(suffix)

    # decoder: up_blocks.B.resnets.J / up_blocks.B.upsamplers.0.*
    um = re.match(r"^up_blocks\.(\d+)\.(.*)$", rest)
    if not um:
        raise KeyError(f"unexpected decoder name: {rest}")
    b, suffix = int(um.group(1)), um.group(2)
    # sd.cpp decoder upsamples flat index: each up_block contributes
    # (num_res_blocks+1)=3 residual blocks, then up_blocks 0,1,2 add 1 Resample.
    # flat_base[B] = B*3 + (#resamples before this block) = B*3 + B = B*4
    flat_base = b * (3 + 1)
    rm = re.match(r"^resnets\.(\d+)\.(.*)$", suffix)
    if rm:
        j, rsuf = int(rm.group(1)), rm.group(2)
        return f"decoder.upsamples.{flat_base + j}." + map_res_suffix(rsuf)
    pm = re.match(r"^upsamplers\.0\.(.*)$", suffix)
    if pm:
        psuf = pm.group(1)  # resample.1.* or time_conv.*  -> sd.cpp Resample sub-names
        return f"decoder.upsamples.{flat_base + 3}.{psuf}"
    raise KeyError(f"unexpected decoder up_block suffix: {suffix}")


def is_conv5d(name):
    return name.endswith(".weight") and (".conv" in name or "time_conv" in name
                                         or name.startswith(("quant_conv", "post_quant_conv")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import gguf
    from gguf import GGMLQuantizationType as QT

    st = Safetensors(args.src)
    names = sorted(st.keys())

    writer = gguf.GGUFWriter(args.out, arch="wan-vae", use_temp_file=True)
    writer.add_description("LongCat base Wan2.1 VAE (diffusers -> sd.cpp wan names, f16)")

    t0 = time.time()
    n = 0
    out_seen = {}
    for src_name in names:
        arr = st.get(src_name)
        dst = map_name(src_name)
        if dst is None:
            continue
        # 5D conv weight [out,in,kt,kh,kw] -> 4D [out*in,kt,kh,kw] for sd.cpp CausalConv3d
        if arr.ndim == 5:
            o, i, kt, kh, kw = arr.shape
            arr = arr.reshape(o * i, kt, kh, kw)
        # RMS_norm gamma: diffusers ships these with trailing singleton dims
        # (resnet [C,1,1,1], attn [C,1,1]). sd.cpp's RMS_norm allocates a 1-D
        # [C] tensor, so flatten to 1-D — otherwise the ggml ne (reversed) is
        # [1,1,C,1] / [1,1,1,C] and the shape check rejects the load.
        if dst.endswith(".gamma"):
            arr = arr.reshape(-1)
        arr = np.ascontiguousarray(arr, dtype=np.float16)
        if dst in out_seen:
            raise SystemExit(f"name collision: {dst} (from {src_name} and {out_seen[dst]})")
        out_seen[dst] = src_name
        writer.add_tensor(dst, arr)
        n += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    st.close()

    sz = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}")
    print(f"  tensors: {n}   file: {sz:.1f} MB   in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
