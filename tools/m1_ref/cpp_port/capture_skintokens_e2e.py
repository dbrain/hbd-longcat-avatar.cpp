#!/usr/bin/env python
# capture_skintokens_e2e.py — run the REAL SkinTokens pipeline headless (no bpy; the core
# encode->generate->decode path takes vertices+normals directly) and dump every stage boundary as the
# porting oracle for R2..R7. Also exports a rigged GLB (trimesh, for an eyeball). Run from the repo:
#   cd /mnt/hdd/3d/avatar-shootout/SkinTokens && NVIDIA_TF32_OVERRIDE=0 PYTHONPATH=$PWD \
#       .venv/bin/python <thisdir>/capture_skintokens_e2e.py <outdir> [mesh.glb]
import os, sys, json
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
import numpy as np, torch, trimesh
torch.manual_seed(0); np.random.seed(0)

OUT  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/skintokens_e2e"
MESH = sys.argv[2] if len(sys.argv) > 2 else "examples/giraffe.glb"
CKPT = "experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt"
os.makedirs(OUT, exist_ok=True)
dev = "cuda"

from src.server.spec import get_model
print("[e2e] loading model ..", flush=True)
model = get_model(CKPT).eval()
print("[e2e] model loaded. tokenizer.vocab=%d vae.vocab=%d vocab_size=%d eos=%d tokens_per_skin=%d"
      % (model.tokenizer.vocab_size, model.vae.vocab_size, model.vocab_size, model.eos, model.tokens_per_skin), flush=True)

# input mesh -> normalize to [-1,1]^3 -> sample surface points + normals
scene = trimesh.load(MESH, process=False)
mesh = trimesh.util.concatenate([g for g in scene.geometry.values()]) if hasattr(scene, "geometry") else scene
v = np.asarray(mesh.vertices, np.float64); c = (v.max(0)+v.min(0))/2
mesh.vertices = (v - c) / np.abs(v - c).max()
N = 8192
pts, fid = trimesh.sample.sample_surface(mesh, N)
nrm = mesh.face_normals[fid]
vertices = torch.tensor(np.asarray(pts), dtype=torch.float32, device=dev)
normals  = torch.tensor(np.asarray(nrm), dtype=torch.float32, device=dev)

# ---- monkeypatch stage boundaries to capture intermediates ----
import src.model.tokenrig as TR
cap = {}
_orig_encode_mesh_cond = TR.encode_mesh_cond
def _emc(mesh_encoder, output_proj, tokens_skin_cond, batch):
    out = _orig_encode_mesh_cond(mesh_encoder, output_proj, tokens_skin_cond, batch)
    cap["learned_mesh_cond"] = out.detach().float().cpu().numpy()
    return out
TR.encode_mesh_cond = _emc

print("[e2e] available cls names:", list(model.tokenizer.cls_token_id.keys()), flush=True)
CLS = os.environ.get("RIG_CLS", "")   # "" / unknown -> token_id_cls_none (free skeleton)
gk = dict(max_length=2048, top_k=10, top_p=0.95, temperature=1.5, repetition_penalty=2.0,
          num_return_sequences=1, num_beams=10, do_sample=True)
print(f"[e2e] generate (beams=10, cls={CLS!r}) ..", flush=True)
with torch.no_grad():
    res = model.generate(vertices, normals, cls=CLS, return_decode_dict=False, **gk)

out_ids = res.output_ids.detach().cpu().numpy().astype(np.int64)
det = res.detokenize_output
print(f"[e2e] DONE: {len(out_ids)} tokens; joints={getattr(det,'joints',None) is not None and len(det.joints)}; "
      f"skin_pred={tuple(res.skin_pred.shape) if res.skin_pred is not None else None}", flush=True)

# ---- dump golden ----
def save(n, a): np.save(os.path.join(OUT, n + ".npy"), np.ascontiguousarray(np.asarray(a)))
save("vertices", vertices.cpu().numpy()); save("normals", normals.cpu().numpy())
save("output_ids", out_ids)
save("input_ids", res.input_ids.detach().cpu().numpy().astype(np.int64))
if "learned_mesh_cond" in cap: save("learned_mesh_cond", cap["learned_mesh_cond"])
save("cond_latents", res.cond_latents.detach().float().cpu().numpy())
if res.skin_pred is not None: save("skin_pred", res.skin_pred.detach().float().cpu().numpy())
meta = dict(vocab_size=int(model.vocab_size), eos=int(model.eos),
            tokenizer_vocab=int(model.tokenizer.vocab_size), vae_vocab=int(model.vae.vocab_size),
            tokens_per_skin=int(model.tokens_per_skin), tokens_skin_cond=int(model.tokens_skin_cond),
            n_tokens=int(len(out_ids)))
if det is not None:
    if getattr(det, "joints", None) is not None:
        j = np.asarray(det.joints); save("joints", j); meta["n_joints"] = int(len(j))
    if getattr(det, "parents", None) is not None:
        meta["parents"] = [int(x) if x is not None else -1 for x in det.parents]
    if getattr(det, "joint_names", None) is not None:
        meta["joint_names"] = list(det.joint_names)
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("[e2e] meta:", json.dumps({k:v for k,v in meta.items() if k not in ("parents","joint_names")}), flush=True)
print("[e2e] wrote golden to", OUT, flush=True)
