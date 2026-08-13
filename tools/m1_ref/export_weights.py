#!/usr/bin/env python3
"""
Export Pixal3D Stage-1 model weights (safetensors) -> fp32 .npz for the numpy/C++
reference port. CPU only, no GPU. Also prints the tensor key list + shapes so the
reference modules know exact names.

Run: CUDA_VISIBLE_DEVICES="" <Pixal3D>/.venv/bin/python export_weights.py
"""
import os, sys, json
import numpy as np
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'weights')
os.makedirs(OUT, exist_ok=True)

SNAP = ('/home/dbrain/.cache/huggingface/hub/models--TencentARC--Pixal3D/'
        'snapshots/0b31f9160aa400719af409098bff7936a932f726/ckpts')
DINO = ('/home/dbrain/.cache/huggingface/hub/'
        'models--camenduru--dinov3-vitl16-pretrain-lvd1689m/'
        'snapshots/3c276edd87d6f6e569ff0c4400e086807d0f3881')


def find_safetensors(d):
    for f in os.listdir(d):
        if f.endswith('.safetensors'):
            return os.path.join(d, f)
    return None


def export(name, st_path, list_keys=True):
    print(f"\n===== {name}  ({os.path.basename(st_path)}) =====")
    tensors = {}
    with safe_open(st_path, framework='numpy') as f:
        keys = list(f.keys())
        for k in keys:
            t = f.get_tensor(k)
            # safetensors numpy backend returns the raw dtype; cast bf16/fp16 -> fp32.
            # numpy has no bf16; safetensors numpy yields uint16 for bf16 -> handle via torch fallback.
            tensors[k] = t
    np.savez(os.path.join(OUT, name + '.npz'), **{k: v.astype(np.float32) if v.dtype != np.float32 else v
                                                  for k, v in tensors.items()})
    if list_keys:
        meta = {k: list(v.shape) + [str(v.dtype)] for k, v in tensors.items()}
        json.dump(meta, open(os.path.join(OUT, name + '_keys.json'), 'w'), indent=2)
        for k in keys[:8]:
            print(f"  {k}: {tensors[k].shape} {tensors[k].dtype}")
        print(f"  ... {len(keys)} tensors total -> {name}.npz + {name}_keys.json")


def export_torch(name, st_path):
    """bf16 path: safetensors numpy backend can't bf16. Use torch to load+cast fp32."""
    import torch
    from safetensors.torch import load_file
    print(f"\n===== {name}  ({os.path.basename(st_path)}) [torch bf16->fp32] =====")
    sd = load_file(st_path, device='cpu')
    tensors = {}
    for k, v in sd.items():
        if torch.is_complex(v):
            # preserve complex (e.g. rope_phases) as a trailing [...,2] real/imag axis
            r = torch.view_as_real(v).float().numpy()
            tensors[k] = r
            print(f"  [complex] {k} -> {r.shape} (last axis = real,imag)")
        else:
            tensors[k] = v.float().numpy()
    np.savez(os.path.join(OUT, name + '.npz'), **tensors)
    meta = {k: list(v.shape) + [str(v.dtype)] for k, v in tensors.items()}
    json.dump(meta, open(os.path.join(OUT, name + '_keys.json'), 'w'), indent=2)
    ks = list(tensors.keys())
    for k in ks[:8]:
        print(f"  {k}: {tensors[k].shape} {tensors[k].dtype}")
    print(f"  ... {len(ks)} tensors total -> {name}.npz + {name}_keys.json")


def main():
    import sys
    which = sys.argv[1:] or ['ss_flow', 'ss_dec', 'dinov3']
    if 'ss_flow' in which:
        export_torch('ss_flow', os.path.join(SNAP, 'ss_flow_img_dit_1_3B_64_bf16.safetensors'))
    if 'ss_dec' in which:
        export_torch('ss_dec', os.path.join(SNAP, 'ss_dec_conv3d_16l8_fp16.safetensors'))
    if 'dinov3' in which:
        export_torch('dinov3', find_safetensors(DINO))
    # --- M2: Shape SLat LR flow (grid 32, proj_in 2048, in/out 32) ---
    if 'slat_flow_512' in which:
        export_torch('slat_flow_512', os.path.join(SNAP, 'slat_flow_img2shape_dit_1_3B_512_bf16.safetensors'))
    # --- M3b: Shape SLat HR flow (grid 64; same arch as 512) ---
    if 'slat_flow_1024' in which:
        export_torch('slat_flow_1024', os.path.join(SNAP, 'slat_flow_img2shape_dit_1_3B_1024_bf16.safetensors'))
    # --- M6: Tex SLat flow (in_ch 64 = 32 noise || 32 shape_slat; same DiT arch, CFG-off) ---
    if 'slat_flow_imgshape2tex_1024' in which:
        export_torch('slat_flow_imgshape2tex_1024',
                     os.path.join(SNAP, 'slat_flow_imgshape2tex_dit_1_3B_1024_bf16.safetensors'))
    # NOTE: tex_dec (tex_dec_next_dc_f16c32_fp16) is a sparse-VAE decoder like shape_dec — its
    # SparseConv3d weights need the spike [27,Cin,Cout] transpose, so export it via the
    # stage3a_capture.py path (mirror shape_dec export), NOT plain export_torch.
    print("\n[export_weights] done ->", OUT)


if __name__ == '__main__':
    main()
