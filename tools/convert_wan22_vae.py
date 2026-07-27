#!/usr/bin/env python3
"""
Wan2.2 48-channel video VAE (ORIGINAL ComfyUI/Wan layout `.pth`, NOT diffusers)
-> GGUF for longcat-avatar.cpp / ggml (sd.cpp WAN::WanVAE with wan2_2=true,
constructed as VERSION_WAN2_2_TI2V).

NAVA uses this exact VAE (`/mnt/hdd/nava/Wan2.2-TI2V-5B/Wan2.2_VAE.pth`). Its
state-dict names are ALREADY the original Wan layout sd.cpp expects
(`encoder.conv1`, `decoder.conv1/conv2`, `*.residual.{0,2,3,6}`, `*.shortcut`,
`*.resample.1`, `*.time_conv`, `middle.{0,1,2}`, `head.{0,2}`, `*.norm.gamma`),
including the DOUBLE-nested up/down-sampling stages
  encoder.downsamples.N.downsamples.M.<sub>
  decoder.upsamples.N.upsamples.M.<sub>
which sd.cpp's wan2_2 Encoder3d/Decoder3d ALSO use verbatim (the outer
`downsamples.N`/`upsamples.N` is a Down_/Up_ResidualBlock that itself contains
`downsamples.M`/`upsamples.M` — see src/wan.hpp Up_ResidualBlock). So names pass
through UNCHANGED; only the conv-weight pack + gamma-flatten transforms apply.
(Contrast tools/convert_wan_vae.py, which remaps the *diffusers* Wan2.1 layout.)

5D conv weight [out,in,kt,kh,kw] -> 4D [out*in,kt,kh,kw] (out-major), which ggml
stores reversed as [kw,kh,kt,out*in] — exactly sd.cpp's CausalConv3d 4D param.
`*.gamma` (RMS_norm) flattened to 1-D [C]. F16 output (VAE wants precision).

Usage (no torch needed at runtime; uses the NAVA venv only to read the .pth):
    /mnt/hdd/nava/.venv/bin/python tools/convert_wan22_vae.py \
        --src /mnt/hdd/nava/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
        --out models/wan2.2-vae-48ch-f16.gguf
"""
import argparse, os, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch  # only for reading the .pth
    import gguf

    sd = torch.load(args.src, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit("unexpected .pth structure (not a state-dict)")

    writer = gguf.GGUFWriter(args.out, arch="wan-vae", use_temp_file=True)
    writer.add_description("Wan2.2 48ch video VAE (NAVA; original layout -> sd.cpp wan names, f16)")

    t0 = time.time()
    n = 0
    out_seen = {}
    for src_name in sorted(sd.keys()):
        t = sd[src_name]
        arr = t.detach().to(torch.float32).cpu().numpy()
        dst = src_name  # verbatim (sd.cpp wan2_2 keeps the original nested names)
        if arr.ndim == 5:
            o, i, kt, kh, kw = arr.shape
            arr = arr.reshape(o * i, kt, kh, kw)
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

    sz = os.path.getsize(args.out) / 1e6
    print(f"\nwrote {args.out}")
    print(f"  tensors: {n}   file: {sz:.1f} MB   in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
