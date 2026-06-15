#!/usr/bin/env python
# capture_qwen3_r2.py — R2 golden for the TokenRig Qwen3-0.6B AR core. Teacher-forced fp32 oracle:
# feed the EXACT inputs_embeds (cat[learned_mesh_cond, embed(output_ids)]) the generate() path uses,
# run the transformer in fp32/eager (TF32 off), dump (inputs_embeds, logits). Also dumps the 311
# transformer weights as fp32 npy (H.weight npy-mode) + config json. Validates the 28-layer Qwen3
# stack (GQA + RoPE + qk-norm + SwiGLU) in isolation. Run from the repo AFTER capture_skintokens_e2e:
#   cd /mnt/hdd/3d/avatar-shootout/SkinTokens && NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=$PWD \
#       .venv/bin/python <thisdir>/capture_qwen3_r2.py /tmp/qwen3_r2 /tmp/qwen3_w /tmp/skintokens_e2e
import os, sys, json
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
import numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

OUT  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen3_r2"
WDIR = sys.argv[2] if len(sys.argv) > 2 else "/tmp/qwen3_w"
E2E  = sys.argv[3] if len(sys.argv) > 3 else "/tmp/skintokens_e2e"
CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
os.makedirs(OUT, exist_ok=True); os.makedirs(WDIR, exist_ok=True)

from src.server.spec import get_model
print("[r2] loading model ..", flush=True)
model = get_model(CKPT).eval()
tr = model.transformer
# fp32 + eager attention (the TRUE-fp32 oracle, not bf16/flash)
tr = tr.float()
tr.config._attn_implementation = "eager"
dev = "cuda"

# rebuild the teacher-forcing input from the e2e golden: inputs_embeds = cat[cond, embed(output_ids)]
out_ids = torch.tensor(np.load(f"{E2E}/output_ids.npy"), dtype=torch.long, device=dev)        # [T]
cond    = torch.tensor(np.load(f"{E2E}/learned_mesh_cond.npy"), dtype=torch.float32, device=dev)  # [1,Lc,896]
if cond.dim() == 2: cond = cond.unsqueeze(0)
emb = tr.get_input_embeddings()(out_ids.unsqueeze(0)).float()                                   # [1,T,896]
inputs_embeds = torch.cat([cond, emb], dim=1).contiguous()                                      # [1,Lc+T,896]
S = inputs_embeds.shape[1]
pos = torch.arange(S, device=dev).unsqueeze(0)
print(f"[r2] Lc={cond.shape[1]} T={out_ids.shape[0]} S={S} hidden={cond.shape[-1]}", flush=True)

with torch.no_grad():
    logits = tr(inputs_embeds=inputs_embeds, position_ids=pos, use_cache=False).logits.float()  # [1,S,vocab]
print(f"[r2] logits {tuple(logits.shape)}  argmax(last)={int(logits[0,-1].argmax())}  "
      f"mean={logits.mean():.4f} std={logits.std():.4f}", flush=True)

def save(d, n, a): np.save(os.path.join(d, n + ".npy"), np.ascontiguousarray(np.asarray(a)))
save(OUT, "inputs_embeds", inputs_embeds[0].detach().cpu().numpy())   # [S,896]
save(OUT, "logits",        logits[0].detach().cpu().numpy())          # [S,vocab]
save(OUT, "output_ids",    out_ids.detach().cpu().numpy())

# dump transformer weights (fp32 npy). keys == state_dict (H.weight loads "<key>.npy").
sd = tr.state_dict()
n = 0
for k, v in sd.items():
    save(WDIR, "transformer." + k, v.detach().float().cpu().numpy()); n += 1
print(f"[r2] dumped {n} transformer weights to {WDIR}", flush=True)

cfg = tr.config
meta = dict(hidden_size=cfg.hidden_size, n_layers=cfg.num_hidden_layers,
            n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim, intermediate=cfg.intermediate_size,
            rms_eps=float(cfg.rms_norm_eps), rope_theta=1000000.0, vocab=cfg.vocab_size,
            Lc=int(cond.shape[1]), S=int(S))
json.dump(meta, open(os.path.join(OUT, "config.json"), "w"), indent=1)
print("[r2] config:", json.dumps(meta), flush=True)
