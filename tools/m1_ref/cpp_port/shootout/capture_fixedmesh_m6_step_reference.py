#!/usr/bin/env python3
"""Capture the official M6 model input/output at every frozen flow step.

This is a diagnostic companion to capture_fixed_mesh_pixal_m6_reference.py.
It reuses that script's immutable sampler boundary, runs no geometry generation or
bake, and records the Python M6's 32-channel input/prediction at each Euler step.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PIXAL3D_ROOT = Path("/mnt/hdd/3d/avatar-shootout/Pixal3D")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True, type=Path,
                    help="immutable m6_sampler_oracle directory")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    required = ("shape_slat_coords.npy", "shape_slat_feats.npy", "cond_global.npy",
                "cond_proj.npy", "tex_noise.npy", "tex_slat.npy")
    if any(not (args.oracle / name).is_file() for name in required):
        raise SystemExit("oracle directory is incomplete")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite diagnostic: {args.out}")

    import os
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    import sys
    sys.path.insert(0, str(PIXAL3D_ROOT))
    import numpy as np
    import torch
    from pixal3d.modules.sparse import SparseTensor
    import inference
    # SparseMultiHeadAttention keeps a module-local reference to this function.
    # Wrapping it lets the diagnostic save the exact Q/K/V passed to PyTorch
    # SDPA, after RMSNorm and RoPE, without changing the official operation.
    import pixal3d.modules.sparse.attention.modules as sparse_attention_modules

    coords = np.load(args.oracle / "shape_slat_coords.npy").astype("<i4", copy=False)
    shape_feats = np.load(args.oracle / "shape_slat_feats.npy").astype("<f4", copy=False)
    global_feats = np.load(args.oracle / "cond_global.npy").astype("<f4", copy=False)
    proj_feats = np.load(args.oracle / "cond_proj.npy").astype("<f4", copy=False)
    noise = np.load(args.oracle / "tex_noise.npy").astype("<f4", copy=False)
    expected = np.load(args.oracle / "tex_slat.npy").astype("<f4", copy=False)
    m = len(coords)
    if coords.shape != (m, 4) or shape_feats.shape != (m, 32) or proj_feats.shape != (m, 2048) or noise.shape != (m, 32):
        raise SystemExit("unexpected frozen M6 oracle shapes")

    pipe = inference.init_pipeline("TencentARC/Pixal3D", device="cuda", low_vram=True)
    shape = SparseTensor(feats=torch.from_numpy(shape_feats).to("cuda"),
                         coords=torch.from_numpy(coords).to("cuda"))
    proj = SparseTensor(feats=torch.from_numpy(proj_feats).to("cuda"),
                        coords=torch.from_numpy(coords).to("cuda"))
    cond = {
        "cond": {"global": torch.from_numpy(global_feats).to("cuda"), "proj": proj},
        "neg_cond": {"global": torch.zeros_like(torch.from_numpy(global_feats).to("cuda")),
                     "proj": SparseTensor(feats=torch.zeros_like(proj.feats), coords=proj.coords)},
    }
    flow = pipe.models["tex_slat_flow_model_1024"]
    inputs, preds, block0, torso_input, self_attn0, self_attn_preout, self_attn_qkv, self_attn_qkv_input, tmods, self_attn_norm1, cross_attn0, cross_attn_preout, proj_linear0, mlp0, mlp0_input, mlp0_linear0, mlp0_hidden, self_attn_sdpa_qkv, cross_attn_sdpa_qkv = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    def hook(_module, call_args, output):
        x = call_args[0]
        inputs.append(x.feats[:, :32].detach().float().cpu().numpy().copy())
        preds.append(output.feats.detach().float().cpu().numpy().copy())

    handle = flow.register_forward_hook(hook)
    if not hasattr(flow, "blocks") or not len(flow.blocks):
        raise RuntimeError("M6 flow model exposes no transformer blocks")

    def block_hook(_module, _call_args, output):
        if not block0:
            torso_input.append(_call_args[0].feats.detach().float().cpu().numpy().copy())
            tmods.append(_call_args[1].detach().float().cpu().numpy().copy())
            block0.append(output.feats.detach().float().cpu().numpy().copy())

    block_handle = flow.blocks[0].register_forward_hook(block_hook)
    def self_attn_norm1_hook(_module, _call_args, output):
        if not self_attn_norm1:
            self_attn_norm1.append(output.detach().float().cpu().numpy().copy())

    self_norm1_handle = flow.blocks[0].norm1.register_forward_hook(self_attn_norm1_hook)
    def self_attn_hook(_module, _call_args, output):
        if not self_attn0:
            self_attn0.append(output.feats.detach().float().cpu().numpy().copy())

    self_handle = flow.blocks[0].self_attn.register_forward_hook(self_attn_hook)
    def self_attn_preout_hook(_module, call_args):
        if not self_attn_preout:
            self_attn_preout.append(call_args[0].detach().float().cpu().numpy().copy())

    self_preout_handle = flow.blocks[0].self_attn.to_out.register_forward_pre_hook(self_attn_preout_hook)
    def self_attn_qkv_hook(_module, _call_args, output):
        if not self_attn_qkv:
            self_attn_qkv.append(output.detach().float().cpu().numpy().copy())

    self_qkv_handle = flow.blocks[0].self_attn.to_qkv.register_forward_hook(self_attn_qkv_hook)
    def self_attn_qkv_input_hook(_module, call_args):
        if not self_attn_qkv_input:
            self_attn_qkv_input.append(call_args[0].detach().float().cpu().numpy().copy())

    self_qkv_pre_handle = flow.blocks[0].self_attn.to_qkv.register_forward_pre_hook(self_attn_qkv_input_hook)
    def sparse_output_hook(dst):
        def hook(_module, _call_args, output):
            if not dst:
                dst.append(output.feats.detach().float().cpu().numpy().copy())
        return hook

    cross_handle = flow.blocks[0].cross_attn.cross_attn_block.register_forward_hook(sparse_output_hook(cross_attn0))
    def cross_attn_preout_hook(_module, call_args):
        if not cross_attn_preout:
            feats = call_args[0].feats if hasattr(call_args[0], "feats") else call_args[0]
            cross_attn_preout.append(feats.detach().float().cpu().numpy().copy())

    cross_preout_handle = flow.blocks[0].cross_attn.cross_attn_block.to_out.register_forward_pre_hook(cross_attn_preout_hook)
    mlp_handle = flow.blocks[0].mlp.register_forward_hook(sparse_output_hook(mlp0))
    def proj_linear_hook(_module, _call_args, output):
        if not proj_linear0:
            proj_linear0.append(output.detach().float().cpu().numpy().copy())

    proj_handle = flow.blocks[0].cross_attn.proj_linear.register_forward_hook(proj_linear_hook)
    def mlp_input_hook(_module, call_args):
        if not mlp0_input:
            mlp0_input.append(call_args[0].feats.detach().float().cpu().numpy().copy())

    mlp_pre_handle = flow.blocks[0].mlp.mlp[0].register_forward_pre_hook(mlp_input_hook)
    mlp_linear0_handle = flow.blocks[0].mlp.mlp[0].register_forward_hook(sparse_output_hook(mlp0_linear0))
    mlp_hidden_handle = flow.blocks[0].mlp.mlp[1].register_forward_hook(sparse_output_hook(mlp0_hidden))
    real_sparse_attention = sparse_attention_modules.sparse_scaled_dot_product_attention

    def capture_sparse_attention(*call_args, **call_kwargs):
        # The first attention call is block 0 self-attention.  Its packed
        # VarLenTensor feats are [M, 3, heads, head_dim].
        if not self_attn_sdpa_qkv and len(call_args) == 1:
            self_attn_sdpa_qkv.append(call_args[0].feats.detach().float().cpu().numpy().copy())
        # The next attention call in block 0 is cross-attention.  Its three
        # operands are already Q/K RMS-normalized and RoPE-free, exactly the
        # tensors passed to PyTorch SDPA.  Capture them for a kernel-only
        # replay rather than inferring from projection outputs.
        elif not cross_attn_sdpa_qkv and len(call_args) == 3:
            def packed(arg):
                feats = arg.feats if hasattr(arg, "feats") else arg
                # Sparse q is [M,H,D]; dense global k/v arrives as [1,5,H,D].
                # The attention implementation flattens its leading batch/key
                # axes, so save that exact [tokens,H,D] representation.
                if feats.ndim == 4:
                    feats = feats.reshape(-1, *feats.shape[-2:])
                return feats.detach().float().cpu().numpy().copy()
            cross_attn_sdpa_qkv.append(tuple(packed(arg) for arg in call_args))
        return real_sparse_attention(*call_args, **call_kwargs)

    sparse_attention_modules.sparse_scaled_dot_product_attention = capture_sparse_attention
    real_randn = torch.randn

    def frozen_randn(*shape_args, **kwargs):
        if len(shape_args) >= 2 and tuple(shape_args[:2]) == (m, 32):
            return torch.from_numpy(noise.copy())
        return real_randn(*shape_args, **kwargs)

    torch.randn = frozen_randn
    try:
        with torch.no_grad():
            final = pipe.sample_tex_slat(flow_model=flow, shape_slat=shape, cond=cond,
                                         sampler_params={"steps": 12, "guidance_strength": 1.0,
                                                         "guidance_rescale": 0.0, "rescale_t": 3.0})
    finally:
        torch.randn = real_randn
        handle.remove()
        block_handle.remove()
        self_norm1_handle.remove()
        self_handle.remove()
        self_preout_handle.remove()
        self_qkv_handle.remove()
        self_qkv_pre_handle.remove()
        cross_handle.remove()
        cross_preout_handle.remove()
        mlp_handle.remove()
        proj_handle.remove()
        mlp_pre_handle.remove()
        mlp_linear0_handle.remove()
        mlp_hidden_handle.remove()
        sparse_attention_modules.sparse_scaled_dot_product_attention = real_sparse_attention
    if len(inputs) != 12 or len(preds) != 12 or len(block0) != 1 or len(torso_input) != 1 or len(tmods) != 1 or len(self_attn_norm1) != 1 or len(self_attn0) != 1 or len(self_attn_preout) != 1 or len(self_attn_qkv) != 1 or len(self_attn_qkv_input) != 1 or len(cross_attn0) != 1 or len(cross_attn_preout) != 1 or len(proj_linear0) != 1 or len(mlp0) != 1 or len(mlp0_input) != 1 or len(mlp0_linear0) != 1 or len(mlp0_hidden) != 1 or len(self_attn_sdpa_qkv) != 1 or len(cross_attn_sdpa_qkv) != 1:
        raise RuntimeError(f"expected 12 M6 calls, got inputs={len(inputs)} preds={len(preds)}")
    args.out.mkdir(parents=True)
    for i, (x, v) in enumerate(zip(inputs, preds)):
        np.save(args.out / f"python_m6_x_{i:02d}.npy", x)
        np.save(args.out / f"python_m6_pred_v_{i:02d}.npy", v)
    np.save(args.out / "python_m6_block_00_step_00.npy", block0[0])
    np.save(args.out / "python_m6_torso_input_step_00.npy", torso_input[0])
    np.save(args.out / "python_m6_tmod_step_00.npy", tmods[0])
    np.save(args.out / "python_m6_self_attn_norm1_step_00.npy", self_attn_norm1[0])
    np.save(args.out / "python_m6_self_attn_00_step_00.npy", self_attn0[0])
    np.save(args.out / "python_m6_self_attn_preout_step_00.npy", self_attn_preout[0])
    np.save(args.out / "python_m6_self_attn_qkv_step_00.npy", self_attn_qkv[0])
    np.save(args.out / "python_m6_self_attn_qkv_input_step_00.npy", self_attn_qkv_input[0])
    np.save(args.out / "python_m6_self_attn_sdpa_q_step_00.npy", self_attn_sdpa_qkv[0][:, 0])
    np.save(args.out / "python_m6_self_attn_sdpa_k_step_00.npy", self_attn_sdpa_qkv[0][:, 1])
    np.save(args.out / "python_m6_self_attn_sdpa_v_step_00.npy", self_attn_sdpa_qkv[0][:, 2])
    np.save(args.out / "python_m6_cross_attn_sdpa_q_step_00.npy", cross_attn_sdpa_qkv[0][0])
    np.save(args.out / "python_m6_cross_attn_sdpa_k_step_00.npy", cross_attn_sdpa_qkv[0][1])
    np.save(args.out / "python_m6_cross_attn_sdpa_v_step_00.npy", cross_attn_sdpa_qkv[0][2])
    np.save(args.out / "python_m6_cross_attn_00_step_00.npy", cross_attn0[0])
    np.save(args.out / "python_m6_cross_attn_preout_step_00.npy", cross_attn_preout[0])
    np.save(args.out / "python_m6_proj_linear_00_step_00.npy", proj_linear0[0])
    np.save(args.out / "python_m6_mlp_00_step_00.npy", mlp0[0])
    np.save(args.out / "python_m6_mlp_00_input_step_00.npy", mlp0_input[0])
    np.save(args.out / "python_m6_mlp_00_linear0_step_00.npy", mlp0_linear0[0])
    np.save(args.out / "python_m6_mlp_00_hidden_step_00.npy", mlp0_hidden[0])
    final_feats = final.feats.detach().float().cpu().numpy()
    np.save(args.out / "python_m6_final.npy", final_feats)
    err = np.max(np.abs(final_feats - expected))
    print(f"captured M6 steps={len(preds)} M={m} final_maxabs_vs_oracle={err:.9g}", flush=True)


if __name__ == "__main__":
    main()
