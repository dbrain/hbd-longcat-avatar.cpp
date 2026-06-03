#!/usr/bin/env python3
"""
NAVA LTX-2.3 audio VAE (`*_audio_vae.safetensors`, 1329 tensors)
-> GGUF for longcat-avatar.cpp / ggml in-tree LTXV audio VAE decoder
(`src/ltx_audio_vae.h`, namespace LTXV).

WHY VERBATIM NAMES (no prefix transform, no remap)
--------------------------------------------------
`LTXAudioVAE` (ltx_audio_vae.h) constructs its sub-blocks with the *full*
state-dict prefixes already baked in:
    blocks["audio_vae.decoder"]     = AudioDecoder(...)
    blocks["vocoder.vocoder"]       = Vocoder(...)
    blocks["vocoder.bwe_generator"] = Vocoder(..., bwe)          (when has_bwe)
and `init_params()` registers the stats / mel_stft params under their full
names too:
    "audio_vae.per_channel_statistics.mean-of-means"
    "audio_vae.per_channel_statistics.std-of-means"
    "vocoder.mel_stft.mel_basis"
    "vocoder.mel_stft.stft_fn.forward_basis"
    "vocoder.mel_stft.stft_fn.inverse_basis"
The runner is created with prefix="" (LTXAudioVAERunner default, and
load_from_file_and_test passes ""), so the lookup keys are EXACTLY the
safetensors keys. `detect_from_weights()` also greps the map for literal
"audio_vae.decoder.conv_in.conv.weight", "audio_vae.decoder.up.N.block.M...",
"vocoder.vocoder.ups.N.weight", etc. => keep every name byte-for-byte.

WHY NO CONV REPACK / NO AXIS REVERSAL
-------------------------------------
ModelLoader (src/model.cpp::load_tensors) does NOT permute/reverse conv axes;
it only converts dtype and shape-checks tensor_storage.ne (the gguf dims, which
gguf stores reversed from the numpy/torch shape) against the block's allocated
ggml ne. The block allocations line up with the torch layout once gguf reverses:
  * Conv2d  (decoder convs)  torch [OC,IC,kh,kw] -> ggml ne [kw,kh,IC,OC]
            ggml_extend Conv2d allocates [kw,kh,IC,OC].                 MATCH
  * Conv1D  (vocoder convs)  torch [OC,IC,k]     -> ggml ne [k,IC,OC,1]
            Conv1D::init_params allocates 4d(k,IC,OC,1).                MATCH
  * ConvTranspose1d (ups.*)  torch [IC,OC,k]     -> ggml ne [k,OC,IC,1]
            ConvTranspose1D::init_params allocates 4d(k,OC,IC,1).       MATCH
This VAE has NO 5D convs and NO RMS `.gamma` tensors (it uses PixelNorm2D /
SnakeBeta, not Wan RMS_norm), so the wan converter's 5D-pack + gamma-flatten
steps are intentionally absent here. Pure passthrough: name verbatim, dtype only.

DTYPE POLICY (mirrors what ltx_audio_vae.h actually allocates)
--------------------------------------------------------------
* Conv2d / Conv1D weights (ndim 3 or 4)         -> F16  (audio_conv_weight_type)
* ConvTranspose1d `ups.*.weight`                -> F32  (ConvTranspose1D allocs F32)
* downsample.lowpass.filter (3D, F16-allocatable)-> F16
* upsample.filter (3D)                          -> F32  (Activation1D allocs F32)
* mel_stft.* (mel_basis 2D / forward+inverse 3D)-> F32  (init_params allocs F32)
* every 1D tensor (bias / SnakeBeta alpha,beta /
  per_channel_statistics mean,std)              -> F32  (all 1D allocs are F32)
The loader upcasts/ downcasts on mismatch anyway, but matching avoids any
surprise and keeps stats/biases lossless. (Wan converter kept stats/gamma /
bias safe as f32 too.)

INCLUDED groups:  audio_vae.decoder (56), audio_vae.per_channel_statistics (2),
                  vocoder.vocoder (667), vocoder.bwe_generator (557),
                  vocoder.mel_stft (3)            => 1285 tensors.
SKIPPED:          audio_vae.encoder (44)  -- decode-only path.

TWO-STEP (like convert_wan22_vae.py): the NAVA venv has torch+safetensors but
NOT gguf. So step 1 (this file, run with the NAVA python) reads the safetensors
and dumps a typed /tmp npz; step 2 re-execs itself via `uv run --with gguf` to
pack the npz -> gguf. The re-exec is automatic; you only run the one command
below.

USAGE (do NOT need gguf in the NAVA venv):
    /mnt/hdd/nava/.venv/bin/python tools/convert_ltx_audio_vae.py \
        --src /mnt/hdd/nava/params/LTX2/ltx-2.3-22b-dev_audio_vae.safetensors \
        --out models/nava-ltx-audio-vae-f16.gguf
"""
import argparse
import os
import sys
import tempfile
import time


SKIP_PREFIX = "audio_vae.encoder"  # decode-only: encoder not needed

# Names (or suffixes) that must stay F32 regardless of ndim.
F32_MEL_STFT_PREFIX = "vocoder.mel_stft."
F32_UPSAMPLE_FILTER_SUFFIX = ".upsample.filter"  # Activation1D upsample.filter -> F32


def want_f32(name: str, ndim: int) -> bool:
    # All 1D tensors (bias, SnakeBeta alpha/beta, per_channel_statistics) -> F32.
    if ndim == 1:
        return True
    # mel_stft mel_basis (2D) + forward/inverse basis (3D) -> F32.
    if name.startswith(F32_MEL_STFT_PREFIX):
        return True
    # Activation1D upsample.filter (3D) is allocated F32 by the loader.
    if name.endswith(F32_UPSAMPLE_FILTER_SUFFIX):
        return True
    # ConvTranspose1d upsampling weights `*.ups.N.weight` allocated F32.
    # (vocoder.vocoder.ups.N.weight / vocoder.bwe_generator.ups.N.weight)
    parts = name.split(".")
    if len(parts) >= 3 and parts[-3] == "ups" and parts[-1] == "weight":
        return True
    return False


def step1_dump_npz(src: str, npz_path: str) -> int:
    """Read safetensors with the NAVA venv (torch) -> typed .npz of fp16/fp32 arrays.

    Source is bf16; safetensors' `np` framework can't decode bf16, so we read
    via the `pt` (torch) framework and upcast bf16 -> fp32 in torch first.
    """
    import numpy as np
    import torch
    from safetensors import safe_open

    arrays = {}
    n = 0
    with safe_open(src, framework="pt") as f:
        keys = sorted(f.keys())
        for name in keys:
            if name.startswith(SKIP_PREFIX):
                continue
            t = f.get_tensor(name)  # torch tensor (bf16)
            # Upcast to fp32 in torch (numpy has no bf16); handles bf16/f16/f32.
            arr = t.detach().to(dtype=torch.float32).cpu().numpy()
            ndim = arr.ndim
            # No 5D convs and no `.gamma` in this VAE => no repack / no flatten.
            assert ndim != 5, f"unexpected 5D tensor {name}"
            assert not name.endswith(".gamma"), f"unexpected RMS gamma {name}"
            target = np.float32 if want_f32(name, ndim) else np.float16
            arr = np.ascontiguousarray(arr, dtype=target)
            arrays[name] = arr
            n += 1
    np.savez(npz_path, **arrays)
    return n


def step2_pack_gguf(npz_path: str, out: str) -> int:
    """Pack the typed npz into a gguf (needs the `gguf` package; run via uv)."""
    import numpy as np
    import gguf

    data = np.load(npz_path)
    writer = gguf.GGUFWriter(out, arch="ltx-audio-vae", use_temp_file=True)
    writer.add_description(
        "NAVA LTX-2.3 audio VAE decoder (verbatim names -> LTXV; weights f16, stats/bias/ups/mel_stft f32)"
    )

    n = 0
    seen = set()
    for name in sorted(data.files):
        arr = np.ascontiguousarray(data[name])
        if name in seen:
            raise SystemExit(f"name collision: {name}")
        seen.add(name)
        writer.add_tensor(name, arr)
        n += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--npz", default=None, help="intermediate npz path (default: temp)")
    ap.add_argument(
        "--_pack_only",
        action="store_true",
        help="internal: only run the gguf-pack step (used by the uv re-exec)",
    )
    args = ap.parse_args()

    # gguf available? -> we can pack in-process. Otherwise re-exec the pack step
    # under `uv run --with gguf` exactly like convert_wan22_vae's two-step note.
    have_gguf = _module_available("gguf")

    if args._pack_only:
        assert have_gguf, "_pack_only requires gguf"
        npz_path = args.npz
        assert npz_path and os.path.exists(npz_path), "missing --npz for pack step"
        t0 = time.time()
        n = step2_pack_gguf(npz_path, args.out)
        sz = os.path.getsize(args.out) / 1e6
        print(f"\nwrote {args.out}")
        print(f"  tensors: {n}   file: {sz:.1f} MB   pack in {time.time() - t0:.1f}s")
        return

    # Step 1: dump npz (needs torch/safetensors; runs under the NAVA venv).
    npz_path = args.npz or os.path.join(
        tempfile.gettempdir(), "ltx_audio_vae_convert.npz"
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    n1 = step1_dump_npz(args.src, npz_path)
    print(f"step1: dumped {n1} tensors -> {npz_path}  ({time.time() - t0:.1f}s)")

    # Step 2: pack.
    if have_gguf:
        n2 = step2_pack_gguf(npz_path, args.out)
        sz = os.path.getsize(args.out) / 1e6
        print(f"\nwrote {args.out}")
        print(f"  tensors: {n2}   file: {sz:.1f} MB   total {time.time() - t0:.1f}s")
    else:
        import subprocess

        print("gguf not importable in this interpreter; re-exec pack via `uv run --with gguf`")
        cmd = [
            "uv", "run", "--with", "gguf", "--with", "numpy",
            "python", os.path.abspath(__file__),
            "--_pack_only", "--src", args.src, "--out", args.out, "--npz", npz_path,
        ]
        subprocess.run(cmd, check=True)


def _module_available(mod: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(mod) is not None


if __name__ == "__main__":
    main()
