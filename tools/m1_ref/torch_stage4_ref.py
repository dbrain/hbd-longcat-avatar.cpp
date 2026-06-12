#!/usr/bin/env python3
"""
fp32 closure for M6 Stage-4 Texture: real ElasticSLatFlowModel(tex, in_ch 64) + real
FlowEulerGuidanceIntervalSampler in fp32, golden stage4 cond (== stage3b cond) + the
RE-normalized golden shape_slat as concat_cond + the seed-42-reproduced TEXTURE noise
(the 4th continuous draw) -> denorm (tex_slat_normalization) -> tex_slat[M,32].

in_ch 64 = 32 noise || 32 shape_slat (re-normalized (x-mean)/std). CFG OFF (guidance_strength
1.0, rescale 0.0), interval [0.6,0.9], rescale_t 3.0, 12 steps. Noise stream (one global
seed-42): stage1 randn(1,8,16^3); stage2 randn(1126,32); stage3b randn(4734,32); tex randn(M,32).

CPU only, fp32. Run:
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=12 \
    /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python torch_stage4_ref.py
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(HERE, "weights")
GOLD = "/home/dbrain/dev/longcat-sparse-spike/tools/sparse_spike/golden_stages"
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, "/mnt/hdd/3d/avatar-shootout/Pixal3D")

import torch


def build_model():
    from pixal3d.models.structured_latent_flow import ElasticSLatFlowModel
    ElasticSLatFlowModel.initialize_weights = lambda self: None
    m = ElasticSLatFlowModel(
        resolution=64, in_channels=64, out_channels=32, model_channels=1536,
        cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
        pe_mode="rope", share_mod=True, qk_rms_norm=True, qk_rms_norm_cross=True,
        image_attn_mode="proj", proj_in_channels=2048, dtype="float32",
    )
    z = np.load(os.path.join(WEIGHTS_DIR, "slat_flow_imgshape2tex_1024.npz"))
    sd = m.state_dict()
    for k in sd.keys():
        if k in z.files:
            sd[k] = torch.from_numpy(z[k].astype(np.float32))
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def main():
    torch.set_grad_enabled(False)
    from pixal3d.pipelines.samplers.flow_euler import FlowEulerGuidanceIntervalSampler
    import pixal3d.modules.sparse as sp

    coords = np.load(os.path.join(GOLD, "stage4_cond", "proj_coords.npy")).astype(np.int32)  # [M,4]
    M = coords.shape[0]
    g = np.load(os.path.join(GOLD, "stage4_cond", "global.npy")).astype(np.float32)
    proj = np.load(os.path.join(GOLD, "stage4_cond", "proj_feats.npy")).astype(np.float32)   # [M,2048]
    N2 = int(np.load(os.path.join(GOLD, "stage2_cond", "proj_coords.npy")).shape[0])         # 1126
    M3 = int(np.load(os.path.join(GOLD, "stage3b_cond", "proj_coords.npy")).shape[0])        # 4734
    print(f"[torch_s4] M={M} N2={N2} M3={M3} coords{coords.shape} proj{proj.shape}", flush=True)

    pj = json.load(open(os.path.join(GOLD, "configs", "pipeline.json")))
    ss_norm = pj["args"]["shape_slat_normalization"]
    ss_std = np.asarray(ss_norm["std"], np.float32)[None]; ss_mean = np.asarray(ss_norm["mean"], np.float32)[None]
    tx_norm = pj["args"]["tex_slat_normalization"]
    tx_std = np.asarray(tx_norm["std"], np.float32)[None]; tx_mean = np.asarray(tx_norm["mean"], np.float32)[None]

    # concat_cond = RE-normalized golden shape_slat (denorm M3b output)
    shape_slat = np.load(os.path.join(GOLD, "stage3b_out", "shape_slat_feats.npy")).astype(np.float32)  # [M,32] denorm
    shape_renorm = (shape_slat - ss_mean) / ss_std

    m = build_model()
    ct = torch.from_numpy(coords)
    proj_st = sp.SparseTensor(feats=torch.from_numpy(proj), coords=ct)
    cond = {"global": torch.from_numpy(g), "proj": proj_st}
    neg = {"global": torch.zeros_like(cond["global"]),
           "proj": sp.SparseTensor(feats=torch.zeros_like(proj_st.feats), coords=ct)}
    concat_cond = sp.SparseTensor(feats=torch.from_numpy(shape_renorm), coords=ct)

    # reproduce seed-42 TEXTURE noise = 4th continuous draw
    torch.manual_seed(42)
    _ = torch.randn(1, 8, 16, 16, 16)   # stage1
    _ = torch.randn(N2, 32)             # stage2
    _ = torch.randn(M3, 32)             # stage3b
    noise_feats = torch.randn(M, m.in_channels - shape_slat.shape[1])  # tex [M,32]
    np.save(os.path.join(HERE, "cpp_port", "refs", "tex_noise.npy"), noise_feats.numpy().astype(np.float32))
    noise = sp.SparseTensor(feats=noise_feats, coords=ct)

    sampler = FlowEulerGuidanceIntervalSampler(sigma_min=1e-5)
    print("[torch_s4] sampling (real tex model, fp32, CFG off, 12 steps)...", flush=True)
    out = sampler.sample(
        m, noise, cond, neg_cond=neg, concat_cond=concat_cond, steps=12, rescale_t=3.0,
        guidance_strength=1.0, guidance_rescale=0.0, guidance_interval=(0.6, 0.9), verbose=True,
    )
    tex_raw = out.samples.feats.numpy().astype(np.float32)
    tex = tex_raw * tx_std + tx_mean

    refs = os.path.join(HERE, "cpp_port", "refs")
    np.save(os.path.join(refs, "torch_tex_slat_fp32.npy"), tex_raw)
    np.save(os.path.join(refs, "torch_tex_slat_denorm_fp32.npy"), tex)
    print(f"[torch_s4] saved tex_noise + torch_tex_slat_{{fp32,denorm_fp32}} ({tex.shape})", flush=True)

    gold = np.load(os.path.join(GOLD, "stage4_out", "tex_slat_feats.npy")).astype(np.float32)
    d = np.abs(tex - gold); a = tex.reshape(-1); b = gold.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    print(f"\n[torch_s4] tex_slat(fp32 denorm) vs golden(bf16): maxabs={d.max():.3e} "
          f"meanabs={d.mean():.3e} cosine={cos:.6f}")
    print("[torch_s4] NOISE CONFIRMED" if d.mean() < 0.4 else "[torch_s4] WARNING: divergence")


if __name__ == "__main__":
    main()
