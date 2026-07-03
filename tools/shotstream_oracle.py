#!/usr/bin/env python3
# ruff: noqa
"""
ShotStream numerical ORACLE — single-chunk (3 latent frames) goldens.

Purpose: emit .npy goldens that a C++/ggml ShotStream port can diff against,
op-for-op, for one denoising chunk (3 latent frames @ 480x832).

PATH TAKEN: FALLBACK (faithful reimplementation) — see summary at bottom / README.
  Why not the full official pipeline: the checkpoint (models/shotstream_merged.pt)
  contains ONLY the DiT weights (generator.model.*). The UMT5-XXL text encoder and
  Wan2.1 VAE weights are NOT present, and CausalWanModel.from_pretrained() wants an
  HF-style `wan_models/Wan2.1-T2V-1.3B/` config dir (absent) plus a flex_attention
  torch.compile at import. So we cannot run the real 30-block forward end-to-end here.
  Instead we dump, with FIXED seeds and the REAL block-0 weights:
    - the exact FlowMatchScheduler warped sigmas/timesteps (faithful reimpl),
    - the dynamic RoPE angle table + composed per-token complex freqs (theta=1/6),
    - the chunk input noise latent,
    - a real-weights block-0 self-attention forward (q/k/v/o + WanRMSNorm + dyn-RoPE + SDPA),
    - the sampler math (flow->x0 = xt - sigma*flow, and renoise) on a SYNTHETIC flow.
  Everything is bit-reproducible from the fixed seeds; the C++ side feeds the SAME
  fixed inputs (dumped here) and diffs its outputs.

All faithful to /tmp/shotstream-ref:
  utils/scheduler.py         FlowMatchScheduler.set_timesteps / add_noise
  model/base.py:479-484      warp_denoising_step
  wan/modules/model.py       rope_params_angle, WanRMSNorm
  wan/modules/causal_model.py  causal_rope_apply_dynamic, CausalWanSelfAttention.forward
  utils/wan_wrapper.py       _convert_flow_pred_to_x0
"""
import os
import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "..", "models", "shotstream_merged.pt")
OUT = os.path.join(HERE, "shotstream_goldens")
os.makedirs(OUT, exist_ok=True)

# ---- fixed geometry (from REFERENCE-shotstream-inference.md) ----
DIM, NUM_HEADS, HEAD_DIM, NLAYERS = 1536, 12, 128, 30
FFN_DIM = 8960
PATCH = (1, 2, 2)
LAT_C, LAT_H, LAT_W = 16, 60, 104           # per-frame latent
GRID_F, GRID_H, GRID_W = 3, 30, 52          # 3-frame chunk after (1,2,2) patchify
FRAME_SEQ = 1560                             # 30*52 tokens per latent frame
CHUNK_TOK = GRID_F * GRID_H * GRID_W         # 4680
CTX_FRAMES = 6                               # global cache -> condition_start_frame = 6
THETA = 1.0 / 6.0                            # RoPE shot-phase
SHIFT = 8.0
NOMINAL_STEPS = [1000, 740, 500, 260]

saved = {}
def dump(name, arr):
    a = np.ascontiguousarray(np.asarray(arr))
    np.save(os.path.join(OUT, name + ".npy"), a)
    saved[name] = (a.shape, str(a.dtype))
    print(f"  saved {name:34s} {str(a.shape):22s} {a.dtype}")


# =====================================================================
# 1. FlowMatchScheduler warped sigmas / timesteps  (faithful reimpl)
#    utils/scheduler.py set_timesteps(1000, training=True), shift=8,
#    sigma_min=0.0, extra_one_step=True  +  base.py warp_denoising_step.
# =====================================================================
def flowmatch_sigmas(num=1000, shift=SHIFT, sigma_min=0.0, sigma_max=1.0):
    sigma_start = sigma_min + (sigma_max - sigma_min) * 1.0   # denoising_strength=1
    sig = torch.linspace(sigma_start, sigma_min, num + 1)[:-1]  # extra_one_step
    sig = shift * sig / (1 + (shift - 1) * sig)
    ts = sig * 1000.0
    return sig, ts

print("[1] scheduler")
sigmas_grid, ts_grid = flowmatch_sigmas()
# warp: timesteps_ext = cat(timesteps, [0]); pick indices 1000 - nominal
ts_ext = torch.cat([ts_grid, torch.tensor([0.0])])            # len 1001
idx = [1000 - s for s in NOMINAL_STEPS]                       # [0, 260, 500, 740]
warped_ts = ts_ext[idx].double()                              # [1000, 957.929, 888.889, 737.589]
warped_sigmas = (warped_ts / 1000.0).double()                 # sigma = t/1000 (exact grid members)
print("  warped timesteps:", [round(float(x), 3) for x in warped_ts])
print("  warped sigmas   :", [round(float(x), 6) for x in warped_sigmas])
dump("warped_timesteps", warped_ts.numpy())
dump("warped_sigmas", warped_sigmas.numpy())
dump("scheduler_sigmas_grid", sigmas_grid.double().numpy())   # full 1000-pt grid (reference)
dump("scheduler_timesteps_grid", ts_grid.double().numpy())


# =====================================================================
# 2. Dynamic RoPE  (theta=1/6)
#    rope_params_angle + causal_rope_apply_dynamic (wan/modules/*)
# =====================================================================
def rope_params_angle(max_seq_len, dim, theta=10000.0):
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    return freqs  # [max_seq_len, dim//2], raw angles (fp64)

def build_freqs_dynamic(d=HEAD_DIM):
    # cat([angle(d-4*(d//6)), angle(2*(d//6)), angle(2*(d//6))], dim=1)
    return torch.cat([
        rope_params_angle(1024, d - 4 * (d // 6)),   # temporal: 44 -> 22 cols
        rope_params_angle(1024, 2 * (d // 6)),       # H: 42 -> 21 cols
        rope_params_angle(1024, 2 * (d // 6)),       # W: 42 -> 21 cols
    ], dim=1)                                        # [1024, 64]

def composed_freqs_complex(freqs_dyn, f, h, w, start_frame, shot_flags, theta=THETA):
    """Reproduce the per-token complex multiplier of causal_rope_apply_dynamic."""
    c = HEAD_DIM // 2  # 64
    parts = torch.split(freqs_dyn, [c - 2 * (c // 3), c // 3, c // 3], dim=1)  # [22,21,21]
    sf = torch.tensor(shot_flags, dtype=torch.float64)
    temporal = (parts[0][start_frame:start_frame + f].view(f, 1, 1, -1)
                + sf.view(f, 1, 1, -1) * theta).expand(f, h, w, -1)
    hh = parts[1][:h].view(1, h, 1, -1).expand(f, h, w, -1)
    ww = parts[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ang = torch.cat([temporal, hh, ww], dim=-1).reshape(f * h * w, 1, -1)  # [seq,1,64]
    return torch.polar(torch.ones_like(ang), ang)  # complex128 [seq,1,64]

print("[2] rope")
freqs_dyn = build_freqs_dynamic()
dump("rope_freqs_dynamic_angle", freqs_dyn.numpy())            # [1024,64] raw angle table
dump("rope_theta", np.array(THETA, dtype=np.float64))
# chunk 0 of shot 0: start_frame = current(0)+condition(6) = 6, shot_flags=[0,0,0]
fc_shot0 = composed_freqs_complex(freqs_dyn, GRID_F, GRID_H, GRID_W, 6, [0, 0, 0])
# chunk of shot 1 (exercises theta phase): start_frame=6, shot_flags=[1,1,1]
fc_shot1 = composed_freqs_complex(freqs_dyn, GRID_F, GRID_H, GRID_W, 6, [1, 1, 1])
def cplx2ri(z):  # [seq,1,64] complex -> [seq,64,2] float64
    z = z.squeeze(1)
    return torch.stack([z.real, z.imag], dim=-1)
dump("rope_freqs_chunk_shot0", cplx2ri(fc_shot0).numpy())      # [4680,64,2]
dump("rope_freqs_chunk_shot1", cplx2ri(fc_shot1).numpy())      # [4680,64,2]


# =====================================================================
# 3. Chunk input noise latent (fixed seed) — the sampler's x for chunk 0
#    real pipeline: noise = randn([1,21,16,60,104]); chunk = noise[:, :3]
# =====================================================================
print("[3] chunk noise")
g = torch.Generator().manual_seed(42)
shot_noise = torch.randn(1, 21, LAT_C, LAT_H, LAT_W, generator=g)  # full shot
chunk_noise = shot_noise[:, :3].contiguous()                       # [1,3,16,60,104]
dump("chunk_input_noise_latent", chunk_noise.float().numpy())


# =====================================================================
# 4. Block-0 self-attention forward with REAL weights (CPU/f32, deterministic)
#    Faithful to CausalWanSelfAttention.forward (no-context sub-op, dyn-RoPE).
# =====================================================================
print("[4] block-0 self-attn (real weights, cpu/f32)")
ck = torch.load(CKPT, map_location="cpu", weights_only=True, mmap=True)
sd = ck["generator"]
P = "model.blocks.0.self_attn."
def w(name): return sd[P + name].float().clone()

qW, qB = w("q.weight"), w("q.bias")
kW, kB = w("k.weight"), w("k.bias")
vW, vB = w("v.weight"), w("v.bias")
oW, oB = w("o.weight"), w("o.bias")
nq, nk = w("norm_q.weight"), w("norm_k.weight")

def wan_rmsnorm(x, weight, eps=1e-6):     # WanRMSNorm, eps=1e-6 (set in self-attn)
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)).type_as(x) * weight

def apply_dyn_rope(x, freqs_complex):
    # x: [1, seq, n, d]; freqs_complex: [seq,1,64] complex128
    seq, n = x.shape[1], x.shape[2]
    xi = torch.view_as_complex(x[0, :seq].to(torch.float64).reshape(seq, n, -1, 2))
    xi = torch.view_as_real(xi * freqs_complex).flatten(2)   # [seq, n, d]
    return xi.unsqueeze(0).type_as(x)

# fixed self-attn input (stands in for the modulated normed hidden state)
gx = torch.Generator().manual_seed(7)
x_in = torch.randn(1, CHUNK_TOK, DIM, generator=gx)          # [1,4680,1536]
dump("block0_selfattn_input", x_in.numpy())

q = wan_rmsnorm(F.linear(x_in, qW, qB), nq).view(1, CHUNK_TOK, NUM_HEADS, HEAD_DIM)
k = wan_rmsnorm(F.linear(x_in, kW, kB), nk).view(1, CHUNK_TOK, NUM_HEADS, HEAD_DIM)
v = F.linear(x_in, vW, vB).view(1, CHUNK_TOK, NUM_HEADS, HEAD_DIM)
dump("block0_q_prerope", q.numpy())
dump("block0_k_prerope", k.numpy())
dump("block0_v", v.numpy())

rq = apply_dyn_rope(q, fc_shot0)
rk = apply_dyn_rope(k, fc_shot0)
dump("block0_roped_q", rq.numpy())
dump("block0_roped_k", rk.numpy())

# SDPA over the chunk (self-attention, no mask, is_causal=False)
attn = F.scaled_dot_product_attention(
    rq.transpose(2, 1), rk.transpose(2, 1), v.transpose(2, 1),
    attn_mask=None, dropout_p=0.0, is_causal=False).transpose(2, 1)   # [1,4680,12,128]
attn = attn.reshape(1, CHUNK_TOK, DIM)
dump("block0_selfattn_preo", attn.numpy())
out = F.linear(attn, oW, oB)
dump("block0_selfattn_out", out.numpy())


# =====================================================================
# 5. Sampler math per step: flow->x0 and renoise
#    x0 = xt - sigma*flow (wan_wrapper._convert_flow_pred_to_x0, fp64)
#    renoise: x_{i+1} = (1-sigma_{i+1})*x0 + sigma_{i+1}*noise (add_noise)
#    NOTE: flow here is SYNTHETIC (fixed seed) — the real DiT flow needs the
#    full 30-block model + text encoder, which is not runnable in this env.
#    This validates the sampler ARITHMETIC op-for-op, not the DiT weights.
# =====================================================================
print("[5] sampler math (synthetic flow)")
xt = chunk_noise.double()                                    # step-0 x = chunk noise
flows, x0s, renoised = [], [], []
gf = torch.Generator().manual_seed(123)
gn = torch.Generator().manual_seed(456)
for i in range(4):
    flow = torch.randn(chunk_noise.shape, generator=gf).double()
    sig = warped_sigmas[i]
    x0 = xt - sig * flow                                     # x0 = xt - sigma*flow
    flows.append(flow); x0s.append(x0)
    if i < 3:
        sig_next = warped_sigmas[i + 1]
        noise = torch.randn(chunk_noise.shape, generator=gn).double()
        xt = (1 - sig_next) * x0 + sig_next * noise          # add_noise -> next xt
        renoised.append(xt.clone())
dump("sampler_synth_flow_steps", torch.stack(flows).float().numpy())   # [4,1,3,16,60,104]
dump("sampler_x0_steps", torch.stack(x0s).float().numpy())             # [4,...]
dump("sampler_renoised_steps", torch.stack(renoised).float().numpy())  # [3,...]
dump("sampler_final_x0", x0s[-1].float().numpy())                      # final chunk x0

# ---- manifest ----
import json
with open(os.path.join(OUT, "MANIFEST.json"), "w") as f:
    json.dump({k: {"shape": list(v[0]), "dtype": v[1]} for k, v in saved.items()},
              f, indent=2)
print(f"\nDONE. {len(saved)} goldens -> {OUT}")
print("warped_timesteps =", [round(float(x), 3) for x in warped_ts])
print("warped_sigmas    =", [round(float(x), 6) for x in warped_sigmas])
print("theta            =", THETA)
