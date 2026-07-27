#!/usr/bin/env python3
"""Precompute a NAVA speaker embedding for voice cloning.

Mirrors NAVA's data-loader spk path (nava_src/vae/local_audio_vae.py:238-245):
  wav -> resample 16k -> mono mean -> clip 30s -> ReDimNet('M', ft_mix) -> [192].
Writes the cpp .bin format consumed by `nava render --spk-emb`:
  int32 n_dims,name_len,type(0=f32); int32 dims[n_dims] (ne order); name; f32 payload.

Usage: nava_spk_extract.py <ref.wav> <out_spk.bin>
Run with the NAVA venv: /mnt/hdd/nava/.venv/bin/python tools/nava_spk_extract.py ...
"""
import os
import struct
import sys

import numpy as np
import torch
import torchaudio

SR = 16000


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    wav_path, out_path = sys.argv[1], sys.argv[2]

    hub_dir = os.path.expanduser("~/.cache/torch/hub/IDRnD_ReDimNet_master")
    src = "local" if os.path.isdir(hub_dir) else "github"
    repo = hub_dir if src == "local" else "IDRnD/ReDimNet"
    spk_model = torch.hub.load(
        repo, "ReDimNet", source=src,
        model_name="M", train_type="ft_mix", dataset="vb2+vox2+cnc",
        trust_repo=True,
    ).eval()

    wav, sr = torchaudio.load(wav_path)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=SR)
    spk_wav = wav.mean(dim=0, keepdim=True)
    spk_len = int(30.0 * SR)
    if spk_wav.shape[-1] > spk_len:
        spk_wav = spk_wav[..., :spk_len]
    with torch.no_grad():
        emb = spk_model(spk_wav).cpu().float().numpy().reshape(-1)
    assert emb.shape[0] == 192, emb.shape

    name = b"spk_emb"
    with open(out_path, "wb") as f:
        f.write(struct.pack("<iii", 1, len(name), 0))
        f.write(struct.pack("<i", 192))
        f.write(name)
        f.write(emb.astype("<f4").tobytes())
    print(f"wrote {out_path}  norm={float(np.linalg.norm(emb)):.4f}  "
          f"mean={emb.mean():.4f}  emb[:5]={np.round(emb[:5],4).tolist()}")


if __name__ == "__main__":
    main()
