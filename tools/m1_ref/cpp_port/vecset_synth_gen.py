#!/usr/bin/env python
# vecset_synth_gen.py — CPU-ONLY synthetic parity oracle for the ggml VecSet encoder (vecset_encoder.hpp).
# NOT the R0 golden (that's the full proven run on the real mesh, GPU/bf16/flash — a STOP point).
# This instantiates the EXACT CrossAttentionEncoder submodules on CPU in fp32 with random weights and
# random points, runs the same math the ggml graph builds (input_proj -> cross_attn -> self_attn x8 ->
# ln_post, given a FIXED sampled set), and dumps weights+inputs+output as .npy. It de-risks the
# hand-written ggml graph (esp. the QKV head-slicing) numerically WITHOUT touching the GPU or the FPS/rng
# sampling port. Run from the SkinTokens repo so `src...` imports resolve:
#   cd /mnt/hdd/3d/avatar-shootout/SkinTokens && CUDA_VISIBLE_DEVICES="" .venv/bin/python \
#       <thisdir>/vecset_synth_gen.py <outdir>
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # keep the GPU OUT of this run entirely
import numpy as np, torch
# SkinTokens' use_flash3 singleton queries the GPU name at import; stub it so the import works on
# a CPU-only run (a 3060 isn't an H100 anyway -> flash3 path stays off). No GPU compute happens.
torch.cuda.get_device_name = lambda *a, **k: "cpu"
torch.cuda.is_available = lambda *a, **k: False
torch.manual_seed(0); np.random.seed(0)

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vecset_synth"
os.makedirs(OUT, exist_ok=True)

from src.model.michelangelo.models.modules.embedder import FourierEmbedder
from src.model.michelangelo.models.tsal.sal_perceiver import CrossAttentionEncoder

# exact checkpoint config (model_config.mesh_encoder)
W, H, L, NF, PF = 512, 8, 8, 8, 3
fe = FourierEmbedder(num_freqs=NF, include_pi=False)  # include_input True (default)
enc = CrossAttentionEncoder(
    device="cpu", dtype=torch.float32, num_latents=256, fourier_embedder=fe,
    point_feats=PF, width=W, heads=H, layers=L, init_scale=0.25 * (1.0 / W) ** 0.5,
    qkv_bias=False, flash=False, use_ln_post=True, query_method=False,
    use_full_input=True, token_num=512, no_query=True,
).eval()

N, Q = 2048, 512  # input points (kv) / sampled queries
pc = (torch.rand(1, N, 3) * 2 - 1).float()
feats = torch.randn(1, N, PF).float(); feats = feats / feats.norm(dim=-1, keepdim=True)
spc = (torch.rand(1, Q, 3) * 2 - 1).float()
sfeats = torch.randn(1, Q, PF).float(); sfeats = sfeats / sfeats.norm(dim=-1, keepdim=True)

with torch.no_grad():
    # mirror CrossAttentionEncoder._forward (query_method=False) FROM the fixed sampled set onward,
    # i.e. exactly what build_vecset_encoder() computes (no FPS/rng — those are injected).
    data = enc.input_proj(torch.cat([fe(pc), feats], dim=-1))            # [1,N,W]
    sd   = enc.input_proj(torch.cat([fe(spc), sfeats], dim=-1))          # [1,Q,W]
    lat = enc.cross_attn(sd, data)
    lat = enc.self_attn(lat)
    lat = enc.ln_post(lat)                                               # [1,Q,W]

def save(name, t): np.save(os.path.join(OUT, name + ".npy"), np.ascontiguousarray(t))
# inputs the C++ harness loads ([n,3], [Q,W])
save("pc",            pc[0].numpy())
save("feats",         feats[0].numpy())
save("sampled_pc",    spc[0].numpy())
save("sampled_feats", sfeats[0].numpy())
save("latents",       lat[0].numpy())
# weights with the C++ key prefix (mesh_encoder.encoder.*) — npy weight-load path
sd_enc = enc.state_dict()
for k, v in sd_enc.items():
    save("mesh_encoder.encoder." + k, v.float().numpy())
print(f"[synth] wrote {len(sd_enc)} weights + 5 arrays to {OUT}")
print(f"[synth] latents shape {tuple(lat[0].shape)}  mean {lat.mean():.5f}  std {lat.std():.5f}")
