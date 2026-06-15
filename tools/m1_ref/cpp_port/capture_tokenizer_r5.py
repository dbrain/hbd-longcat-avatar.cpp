#!/usr/bin/env python
# capture_tokenizer_r5.py — R5 golden: the host skeleton de-tokenizer. Loads the tokenizer from the
# checkpoint config, runs detokenize() on the saved output_ids (deterministic), and dumps the joints +
# parents golden + a flat spec.txt (num_discrete, continuous_range, all special token ids) the C++
# detok reads. Pure host (no model forward). Run from the repo:
#   cd /mnt/hdd/3d/avatar-shootout/SkinTokens && PYTHONPATH=$PWD .venv/bin/python \
#       <thisdir>/capture_tokenizer_r5.py /tmp/skintokens_e2e
import os, sys, json
import numpy as np, torch
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/skintokens_e2e"
CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"

# tokenizer_config lives in the ckpt hyper_parameters
d = torch.load(CKPT, map_location="cpu", weights_only=False)
tcfg = d["hyper_parameters"]["tokenizer_config"]
from src.tokenizer.parse import get_tokenizer
tk = get_tokenizer(**tcfg)

out_ids = np.load(f"{OUT}/output_ids.npy").astype(np.int64)
# detokenize runs on the skeleton portion only (up to & including the first EOS); skin tokens follow.
eos_pos = int(np.where(out_ids == tk.token_id_eos)[0][0])
skeleton_tokens = out_ids[:eos_pos + 1]
print(f"[r5] first EOS at {eos_pos}; skeleton_tokens={len(skeleton_tokens)} of {len(out_ids)} total")
det = tk.detokenize(ids=skeleton_tokens)
joints = np.asarray(det.joints, np.float32)
parents = [int(x) for x in det.parents]
np.save(f"{OUT}/detok_joints.npy", np.ascontiguousarray(joints))
np.save(f"{OUT}/detok_parents.npy", np.ascontiguousarray(np.array(parents, np.int64)))
print(f"[r5] detok: joints {joints.shape}  parents[:8]={parents[:8]}  n={len(parents)}")

# flat spec the C++ reads (key value per line)
lo, hi = tk.continuous_range
spec = {
    "num_discrete": int(tk.num_discrete),
    "cont_lo": float(lo), "cont_hi": float(hi),
    "bos": int(tk.token_id_bos), "eos": int(tk.token_id_eos), "pad": int(tk.token_id_pad),
    "branch": int(tk.token_id_branch), "spring": int(tk.token_id_spring),
    "cls_none": int(tk.token_id_cls_none),
    "vocab_size": int(tk.vocab_size),
}
cls_ids = sorted(int(v) for v in tk.cls_token_id.values())
parts_ids = sorted(int(v) for v in tk.parts_token_id.values())
with open(f"{OUT}/tok_spec.txt", "w") as f:
    for k, v in spec.items():
        f.write(f"{k} {v}\n")
    f.write("cls_ids " + " ".join(str(x) for x in cls_ids) + "\n")
    f.write("parts_ids " + " ".join(str(x) for x in parts_ids) + "\n")
print("[r5] spec:", json.dumps(spec), "cls_ids", cls_ids, "parts_ids", parts_ids)
print("[r5] wrote detok_joints/detok_parents/tok_spec.txt to", OUT)
