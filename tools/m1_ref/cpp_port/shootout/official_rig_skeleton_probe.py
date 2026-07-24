#!/usr/bin/env python3
"""Run the upstream SkinTokens skeleton decoder on an exact sampled mesh.

This is a parity discriminator, not a delivery exporter.  A structurally bad
tree from the native port could still be a port/conditioning error; the same
tree from this untouched upstream model establishes a model limitation before
we design a fallback.  It saves only IDs and a compact structural report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch


def install_complete_branch_grammar(tokenizer) -> None:
    """Repair upstream's branch mask: branch encodes parent(3)+child(3)."""
    def next_tokens(self, ids):
        if len(ids) == 0:
            return [self.token_id_bos]
        state = "bos"
        for token in ids:
            token = int(token)
            if state == "bos": state = "head"
            elif state == "head": state = "j2" if token < self.num_discrete else ("part" if token == self.token_id_cls_none or token in self.cls_token_id.values() else "j1")
            elif state == "part": state = "j2" if token < self.num_discrete else "part"
            elif state == "j1": state = "j2"
            elif state == "j2": state = "j3"
            elif state == "j3": state = "choice"
            elif state == "choice": state = "bp1" if token == self.token_id_branch else ("j2" if token < self.num_discrete else "j1")
            elif state == "bp1": state = "bp2"
            elif state == "bp2": state = "bp3"
            elif state == "bp3": state = "bc1"
            elif state == "bc1": state = "bc2"
            elif state == "bc2": state = "bc3"
            elif state == "bc3": state = "choice"
        coords = list(range(self.num_discrete))
        parts = [self.token_id_spring, *self.parts_token_id.values()]
        if state == "bos": return [self.token_id_bos]
        if state == "head": return [self.token_id_cls_none, *self.cls_token_id.values(), *parts, *coords]
        if state == "part": return [*parts, *coords, self.token_id_eos]
        if state in {"j1", "j2", "j3", "bp1", "bp2", "bp3", "bc1", "bc2", "bc3"}: return coords
        if state == "choice": return [*coords, *parts, self.token_id_branch, self.token_id_eos]
        raise AssertionError(state)

    def count_bones(self, ids):
        state, count = "bos", 0
        for token in ids:
            token = int(token)
            if state == "bos": state = "head"
            elif state == "head": state = "j2" if token < self.num_discrete else ("part" if token == self.token_id_cls_none or token in self.cls_token_id.values() else "j1")
            elif state == "part": state = "j2" if token < self.num_discrete else "part"
            elif state == "j1": state = "j2"
            elif state == "j2": state = "j3"
            elif state == "j3": count += 1; state = "choice"
            elif state == "choice": state = "bp1" if token == self.token_id_branch else ("j2" if token < self.num_discrete else "j1")
            elif state == "bp1": state = "bp2"
            elif state == "bp2": state = "bp3"
            elif state == "bp3": state = "bc1"
            elif state == "bc1": state = "bc2"
            elif state == "bc2": state = "bc3"
            elif state == "bc3": count += 1; state = "choice"
            if token == self.token_id_eos:
                break
        return count

    tokenizer.next_posible_token = types.MethodType(next_tokens, tokenizer)
    tokenizer.bones_in_sequence = types.MethodType(count_bones, tokenizer)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", type=Path, help="directory containing vertices.npy and normals.npy")
    parser.add_argument("out", type=Path, help="new diagnostic directory")
    parser.add_argument("--skintokens-root", type=Path, default=Path("/mnt/hdd/3d/avatar-shootout/SkinTokens"))
    parser.add_argument("--beams", type=int, default=20)
    parser.add_argument("--sample", action="store_true", help="use official stochastic decoding parameters")
    parser.add_argument("--complete-branch-grammar", action="store_true",
                        help="repair upstream's incomplete branch mask before decoding")
    parser.add_argument("--with-skin", action="store_true",
                        help="also decode and save the upstream skin field for a gated transfer")
    parser.add_argument("--trace-first-logits", action="store_true",
                        help="validation-only: save the first AR next-token logits before generation")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.beams < 1 or args.beams > 20:
        raise SystemExit("--beams must be in [1,20]")
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite diagnostic output: {args.out}")
    vertices = np.load(args.samples / "vertices.npy").astype(np.float32, copy=False)
    normals = np.load(args.samples / "normals.npy").astype(np.float32, copy=False)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or normals.shape != vertices.shape:
        raise SystemExit("samples must be matching [N,3] float arrays")

    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    os.environ.setdefault("XFORMERS_IGNORE_FLASH_VERSION_CHECK", "1")
    root = args.skintokens_root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    from src.server.spec import get_model
    from src.model.tokenrig import encode_mesh_cond
    from src.model.utils import fps

    model = get_model("experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt").eval()
    if args.complete_branch_grammar:
        install_complete_branch_grammar(model.tokenizer)
    # Upstream TokenRig calls the condition VAE with ``seed=None``. Its
    # `_sample_features` then constructs a fresh NumPy RNG, so the same AR
    # skeleton can decode different skin fields across processes. Bind that
    # otherwise implicit condition-sampling seed to this diagnostic seed.
    original_encode = model.vae.model._encode
    def seeded_encode(*call_args, **call_kwargs):
        call_kwargs.setdefault("seed", args.seed)
        return original_encode(*call_args, **call_kwargs)
    model.vae.model._encode = seeded_encode
    if args.trace_first_logits:
        # This is a parity boundary only: it does not select, mutate, or emit a
        # rig.  It is the exact pre-generation prefix consumed by HF generate:
        # mesh condition followed by BOS and cls-none.
        args.out.mkdir(parents=True, exist_ok=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            cond = torch.cat([torch.tensor(vertices, dtype=torch.float32, device="cuda"),
                              torch.tensor(normals, dtype=torch.float32, device="cuda")], dim=-1)
            _, raw_latents, _, raw_pre_pc = model.mesh_encoder.encode_latents(
                pc=cond[:, :3].unsqueeze(0), feats=cond[:, 3:].unsqueeze(0))
            np.save(args.out / "python_mesh_encoder_latents.npy",
                    raw_latents[0].detach().float().cpu().numpy())
            np.save(args.out / "python_mesh_encoder_pre_points.npy",
                    raw_pre_pc[0].detach().float().cpu().numpy())
            # Mirror CrossAttentionEncoder._forward's deterministic eval FPS
            # query boundary so the native VecSet port can compare inputs
            # before any attention math.
            query_xyz, query_feats = raw_pre_pc[:, :, :3], raw_pre_pc[:, :, 3:]
            count = query_xyz.shape[1]
            batch = torch.repeat_interleave(torch.arange(1, device=cond.device), count)
            chosen = fps(query_xyz.reshape(count, 3), batch, ratio=0.25, random_start=False)
            np.save(args.out / "python_mesh_encoder_query_pc.npy",
                    query_xyz.reshape(count, 3)[chosen].detach().float().cpu().numpy())
            np.save(args.out / "python_mesh_encoder_query_feats.npy",
                    query_feats.reshape(count, 3)[chosen].detach().float().cpu().numpy())
            learned = encode_mesh_cond(model.mesh_encoder, model.output_proj,
                                       model.tokens_skin_cond,
                                       {"vertices": cond[:, :3], "normals": cond[:, 3:]})
            np.save(args.out / "python_mesh_cond.npy", learned[0].detach().float().cpu().numpy())
            start = torch.tensor(model.make_start_tokens(device=cond.device, cls=[""])[0],
                                 device=cond.device).unsqueeze(0)
            embeds = torch.cat([learned, model.transformer.get_input_embeddings()(start)], dim=1)
            first_logits = model.transformer(inputs_embeds=embeds).logits[0, -1]
        np.save(args.out / "python_first_next_logits.npy", first_logits.detach().float().cpu().numpy())
    kwargs: dict[str, object] = dict(max_length=2048, repetition_penalty=2.0,
                                     num_return_sequences=1, num_beams=args.beams,
                                     do_sample=args.sample)
    if args.sample:
        kwargs.update(top_k=5, top_p=0.95, temperature=1.0)
    # Loading the upstream package can construct helper modules and consume
    # RNG state. Seed immediately before generation so a fixed fallback seed
    # actually describes the sampled sequence and decoded skin field.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    with torch.no_grad():
        result = model.generate(
            torch.tensor(vertices, dtype=torch.float32, device="cuda"),
            torch.tensor(normals, dtype=torch.float32, device="cuda"),
            cls="", return_decode_dict=not args.with_skin, **kwargs,
        )
    ids = result.output_ids.detach().cpu().numpy().astype(np.int64)
    eos = int(model.tokenizer.eos)
    eos_at = np.where(ids == eos)[0]
    report: dict[str, object] = {
        "schema_version": 1, "mode": "official-sampling" if args.sample else "official-deterministic",
        "branch_grammar": "complete-parent-plus-child" if args.complete_branch_grammar else "upstream-incomplete",
        "seed": args.seed, "condition_sampling_seed": args.seed, "beams": args.beams, "sample_count": int(len(vertices)),
        "eos_token": eos, "token_count": int(len(ids)),
        "eos_index": int(eos_at[0]) if len(eos_at) else None,
    }
    decoded = result.detokenize_output if args.with_skin else None
    if len(eos_at):
        try:
            if decoded is None:
                decoded = model.tokenizer.detokenize(ids=ids[: eos_at[0] + 1])
            parents = np.asarray(decoded.parents, dtype=np.int64)
            children = np.bincount(parents[parents >= 0], minlength=len(parents))
            report.update(joints=int(len(decoded.joints)), roots=int(np.sum(parents < 0)),
                          maxfan=int(children.max(initial=0)), status="detokenized")
        except Exception as exc:  # preserved as evidence rather than silently normalized
            report.update(status="detokenize-error", error=f"{type(exc).__name__}: {exc}")
    else:
        report["status"] = "no-eos"
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "output_ids.npy", ids)
    if args.with_skin:
        if decoded is None or result.skin_pred is None:
            raise RuntimeError("upstream skin decode returned no skeleton/skin field")
        skin = result.skin_pred.detach().float().cpu().numpy()
        if skin.shape != (len(vertices), len(decoded.joints)) or not np.isfinite(skin).all() or (skin < 0).any():
            raise RuntimeError(f"invalid upstream skin field: {skin.shape}")
        np.save(args.out / "vertices.npy", vertices)
        np.save(args.out / "gen_joints.npy", np.asarray(decoded.joints, dtype=np.float32))
        np.save(args.out / "gen_parents.npy", np.asarray(decoded.parents, dtype=np.int64))
        np.save(args.out / "gen_skin_pred.npy", skin.astype(np.float32, copy=False))
        report.update(skin_shape=list(skin.shape), skin_row_sum_min=float(skin.sum(axis=1).min()),
                      skin_row_sum_max=float(skin.sum(axis=1).max()))
    (args.out / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
