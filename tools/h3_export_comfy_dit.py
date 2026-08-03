#!/usr/bin/env python3
"""Export our MiniMax-H3 NVFP4 DiT GGUF as a ComfyUI-loadable NVFP4 safetensors file.

WHY
---
We have defects in our H3 port (audio tempo, a sequence-length rho cliff, video detail) and
we have only ever compared our OUTPUT against ComfyUI's output.  That comparison cannot
tell "our engine is wrong" from "our conversion/quantisation is wrong".  Running OUR
weights inside ComfyUI splits it in one experiment:

    good output in ComfyUI  -> the weights and the conversion are fine, the bug is OURS (engine)
    same defects in ComfyUI -> the bug is in the CONVERSION, and the engine is exonerated

ComfyUI has no GGUF support, so the file has to be re-expressed in the layout its own
`{"format": "nvfp4"}` loader reads.  Nothing here re-quantises: the 4-bit codes and the
E4M3 group scales are moved, not recomputed.

WHAT IT REWRITES, AND WHY EACH ONE IS LOAD-BEARING
--------------------------------------------------
1. NVFP4 BYTE LAYOUT.  ggml `block_nvfp4` -> comfy's (weight, weight_scale, weight_scale_2,
   comfy_quant) quadruple: cuBLAS-swizzled scale plane, `hi_first` nibble order.  See
   h3_comfy_nvfp4_layout.py; both conventions fail SILENTLY when wrong.

2. THE RoPE HEAD-CHANNEL PERMUTATION IS UNDONE.  Our converter permutes the q and k head
   channels of every `blocks.N.attn.qkv_proj.weight` (and of q_norm/k_norm) so that
   full-width split-half RoPE reproduces H3's partial pairing -- see
   src/model/diffusion/minimax_h3_qk_permute.hpp.  ComfyUI does NOT do that: it calls
   `rms_rope_split_half(..., rot_dim=96)` on the raw channel order (comfy/ldm/minimax/
   model.py:166).  Shipping permuted weights would leave every attention score wrong with
   nothing to see in any shape or norm.
   MEASURED: our q_norm vs comfy's q_norm is max|diff| 0.586 as-is and EXACTLY 0 with the
   permutation undone; block-0 qkv row-norm correlation 0.824 -> 0.999990.

3. THE FUSED-QKV DE-INTERLEAVE IS *NOT* UNDONE.  The raw MiniMax checkpoint stores qkv
   per-head interleaved and our converter de-interleaves it to [q_all; k_all; v_all].
   ComfyUI does `.split(heads * head_dim, dim=-1)` (model.py:158), i.e. it expects the
   CONTIGUOUS form too, so our layout is already right.
   MEASURED on the token refiner (which takes the de-interleave and not the RoPE
   permutation, so it isolates this): row-norm correlation vs comfy 0.999981 as-is,
   0.028 if re-interleaved.

4. THE TWO MARKER TENSORS ARE DROPPED.  `attn.qkv_deinterleaved` and `rope.qk_permuted`
   describe rewrites this exporter has just undone or that comfy does not need; leaving
   them in would put unknown keys in comfy's state dict.

5. F32 NORM GAINS GO BACK TO BF16 where that is bit-exact.  Our GGUF upcasts them; comfy's
   own checkpoint stores BF16.  The write is refused unless the value round-trips exactly,
   so this can never lose anything (VERIFIED: 0 mantissa bits below bf16 in every one).

WHAT IS DELIBERATELY LEFT ALONE
-------------------------------
The rank-8 AdaLN factorisation.  Ours is derived from our own bf16 weights rather than
copied from Comfy-Org's (tools/h3_prune_adaln.py), so `adaln_t_table` and every
`adaln_proj.linear.weight` differ from theirs numerically -- but they are consistent WITH
EACH OTHER, and comfy just evaluates `linear(table[t])`, so the modulation it computes is
ours.  They stay F32 (comfy's `adaln_dtype` is float32 in the curve form anyway; their
file's F16 is the lossy one).

USAGE
-----
    python3 tools/h3_export_comfy_dit.py \
        /home/dbrain/models/h3/h3-fl2va-dit-nvfp4-pruned.gguf \
        /mnt/ssd/h3-staging/comfy-ref/models/diffusion_models/ours_h3_fl2va_pruned_nvfp4.safetensors

    --dequant-refiner              emit token_refiner.* as BF16 rather than NVFP4, matching
                                   the structure of Comfy-Org's own checkpoint.  Lossless (a
                                   dequantised NVFP4 value needs at most 7 significant bits
                                   and bf16 has 8), but it changes the COMPUTE: comfy's nvfp4
                                   GEMM also quantises the activations, a bf16 GEMM does not.
                                   Use only as a fallback if comfy refuses a quantised refiner.
    --full-precision-mm            add {"full_precision_matrix_mult": true} to every layer,
                                   which makes comfy dequantise and run a normal matmul.
                                   Use if the fp4 tensor-core GEMM is unavailable on the GPU.
    --verify                       after writing, re-read the file: replay comfy's own
                                   model_detection on it, and round-trip sampled NVFP4 layers
                                   back to block_nvfp4 and compare to the source GGUF bytes.

`condition_proj.weight` is ALWAYS emitted BF16 and that is not a preference -- see the
comment on leave_unquantised().
"""
import argparse
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h3_comfy_nvfp4_layout as L  # noqa: E402
from h3_prune_adaln import GGML_TYPES, gguf_read_header, nbytes  # noqa: E402

# Tensors our converter stamps to describe rewrites.  Neither is a weight.
MARKERS = ("attn.qkv_deinterleaved", "rope.qk_permuted")

HEAD_DIM = 128
ROT_DIM = 96          # 2 * 3 * len(rope.inv_freq); read back from the file and re-checked


def qk_head_permutation(head_dim, rot_dim):
    """MiniMaxH3::build_qk_head_permutation -- ours[i] = original[perm[i]]."""
    half, rot_half, pass_half = head_dim // 2, rot_dim // 2, (head_dim - rot_dim) // 2
    perm = np.empty(head_dim, dtype=np.int64)
    for c in range(rot_half):
        perm[c] = c
        perm[half + c] = rot_half + c
    for j in range(pass_half):
        perm[rot_half + j] = rot_dim + j
        perm[half + rot_half + j] = rot_dim + pass_half + j
    return perm


def is_block_attn(name, leaf):
    """`blocks.<N>.attn.<leaf>` -- the DiT blocks only, never the token refiner (it has no
    rotary embedding, so the RoPE permutation never touched it)."""
    return name.startswith("blocks.") and name.endswith(".attn." + leaf)


# --------------------------------------------------------------------------- GGUF access
class GGUF:
    def __init__(self, path):
        self.f, _, _, _, tl, self.data_start, _ = gguf_read_header(path)
        self.order = [t["name"] for t in tl]
        self.t = {t["name"]: t for t in tl}

    def type_name(self, name):
        return GGML_TYPES[self.t[name]["type"]][0]

    def torch_shape(self, name):
        # GGUF dims are fastest-axis-first; a torch shape is the reverse.
        return [int(d) for d in reversed(self.t[name]["dims"])]

    def raw(self, name):
        t = self.t[name]
        self.f.seek(self.data_start + t["off"])
        n = nbytes(t["dims"], t["type"])
        buf = self.f.read(n)
        if len(buf) != n:
            raise SystemExit(f"{name}: short read ({len(buf)} of {n})")
        return buf


# --------------------------------------------------------------- safetensors streaming writer
ST_DTYPE_ITEMSIZE = {"F32": 4, "F16": 2, "BF16": 2, "U8": 1, "F8_E4M3": 1}


class SafetensorsWriter:
    """Two-pass writer: plan every (name, dtype, shape, nbytes) first, emit bytes second.

    Entries are ordered so that each tensor begins at an offset that is a multiple of its
    element size, without inserting gaps (the format requires the data blocks to be
    contiguous): everything whose length is a multiple of 8 goes first, then the 4-byte F32
    scalars, then the 1-byte-aligned JSON blobs.
    """

    def __init__(self):
        self.entries = []   # (name, dtype, shape, nbytes, producer)

    def add(self, name, dtype, shape, nbytes_, producer):
        self.entries.append((name, dtype, shape, nbytes_, producer))

    def write(self, path, metadata=None, progress=None):
        def rank(e):
            return 0 if e[3] % 8 == 0 else (1 if e[3] % 4 == 0 else 2)
        ordered = sorted(range(len(self.entries)), key=lambda i: (rank(self.entries[i]), i))

        header, off = {}, 0
        for i in ordered:
            name, dtype, shape, nb, _ = self.entries[i]
            if off % ST_DTYPE_ITEMSIZE[dtype]:
                raise SystemExit(f"{name}: offset {off} is not aligned for {dtype}")
            header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nb]}
            off += nb
        if metadata:
            header["__metadata__"] = metadata

        blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
        blob += b" " * ((-len(blob)) % 8)          # keep the data section 8-byte aligned

        tmp = path + ".partial"
        with open(tmp, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            written = 0
            for n, i in enumerate(ordered):
                name, dtype, shape, nb, producer = self.entries[i]
                data = producer()
                if len(data) != nb:
                    raise SystemExit(f"{name}: produced {len(data)} bytes, planned {nb}")
                f.write(data)
                written += nb
                if progress:
                    progress(n + 1, len(ordered), name, written, off)
            if written != off:
                raise SystemExit(f"wrote {written} bytes, planned {off}")
        os.replace(tmp, path)
        return 8 + len(blob) + off


# ------------------------------------------------------------------------------ producers
def bf16_bytes(f32):
    """Round-to-nearest-even f32 -> bf16.  Only called where the value is already exact."""
    u = np.asarray(f32, dtype=np.float32).view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return rounded.tobytes()


def bf16_is_exact(f32):
    return not np.any(np.asarray(f32, dtype=np.float32).view(np.uint32) & 0xFFFF)


def f32_from_gguf(g, name):
    t = g.type_name(name)
    raw = g.raw(name)
    if t == "f32":
        return np.frombuffer(raw, "<f4")
    if t == "bf16":
        return (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
    if t == "f16":
        return np.frombuffer(raw, "<f2").astype(np.float32)
    raise SystemExit(f"{name}: {t} is not a plain dtype")


# ------------------------------------------------------------------------- verification
COMFY_REF_DIT = ("/mnt/ssd/h3-staging/comfy-ref/models/diffusion_models/"
                 "minimax_h3_fl2va_pruned_int8_convrot.safetensors")


def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return h, 8 + n


def st_read(path, base, entry):
    with open(path, "rb") as f:
        f.seek(base + entry["data_offsets"][0])
        return f.read(entry["data_offsets"][1] - entry["data_offsets"][0])


def replay_model_detection(hdr):
    """comfy/model_detection.py's MiniMax-H3 branch, verbatim, against a header.

    This is the check that would have caught the condition_proj trap: every field here is
    read off a TENSOR SHAPE, and a packed NVFP4 weight reports half its logical input dim.
    """
    def shape(n):
        return hdr[n]["shape"]

    def count(prefix):
        i = 0
        while any(k.startswith(f"{prefix}{i}.") for k in hdr):
            i += 1
        return i

    cfg = {
        "num_layers": count("blocks."),
        "token_refiner_num_layers": count("token_refiner.blocks."),
        "hidden_size": shape("video_patch_proj.weight")[0],
        "latents_dim": shape("final_layer.video_out.weight")[0] // 4,
        "audio_latents_dim": shape("final_layer.audio_out.weight")[0],
        "attention_head_dim": shape("blocks.0.attn.q_norm.weight")[0],
        "ffn_hidden_size": shape("blocks.0.mlp.fc1.weight")[0] // 2,
        "text_dim": shape("condition_proj.weight")[1],
        "rope_inv_freq_len": shape("rope.inv_freq")[0],
    }
    cfg["num_attention_heads"] = shape("blocks.0.attn.qkv_proj.weight")[0] // (3 * cfg["attention_head_dim"])
    if "adaln_t_table" in hdr:
        t = shape("adaln_t_table")
        cfg["adaln_curve_grid"], cfg["time_embed_dim"] = t[0], t[1]
    return cfg


def verify(out_path, gguf_path, sample=4):
    print("\n" + "=" * 78)
    print("VERIFY")
    print("=" * 78)
    hdr, base = st_header(out_path)
    g = GGUF(gguf_path)
    ok = True

    # --- 1. comfy's own config detection, ours vs their reference checkpoint
    mine = replay_model_detection(hdr)
    print("  model_detection replayed on our export:")
    ref = None
    if os.path.exists(COMFY_REF_DIT):
        ref = replay_model_detection(st_header(COMFY_REF_DIT)[0])
    for k in sorted(mine):
        mark = ""
        if ref is not None:
            mark = "  ok" if ref.get(k) == mine[k] else f"  <== MISMATCH, comfy ref has {ref.get(k)}"
            ok &= ref.get(k) == mine[k]
        print(f"    {k:<26} {mine[k]}{mark}")

    # --- 2. the name set
    if ref is not None:
        rh = st_header(COMFY_REF_DIT)[0]
        extra = sorted(set(hdr) - set(rh))
        missing = sorted(set(rh) - set(hdr))
        # their int8 layers carry no weight_scale_2/comfy_quant shaped like ours; only
        # report names whose BASE is absent, which is what would actually break a load
        base_missing = [m for m in missing if not m.endswith((".weight_scale", ".weight_scale_2",
                                                              ".comfy_quant"))]
        print(f"\n  names: {len(hdr)} ours vs {len(rh)} comfy-ref; "
              f"{len(extra)} extra, {len(missing)} absent "
              f"({len(base_missing)} of them are real tensors)")
        if base_missing:
            print(f"    MISSING REAL TENSORS: {base_missing[:8]}")
            ok = False

    # --- 3. NVFP4 round-trip against the source GGUF bytes
    quant = sorted(k[: -len(".comfy_quant")] for k in hdr if k.endswith(".comfy_quant"))
    picks = [quant[i * len(quant) // sample] for i in range(sample)]
    permuted = "rope.qk_permuted" in g.t
    head_dim = int(g.torch_shape("blocks.0.attn.q_norm.weight")[0])
    rot_dim = 2 * 3 * int(g.torch_shape("rope.inv_freq")[0])
    perm = qk_head_permutation(head_dim, rot_dim)
    print("\n  NVFP4 round-trip (export -> block_nvfp4 -> compare to the GGUF):")
    for b in picks:
        e = hdr[b + ".weight"]
        out_f, in_f = e["shape"][0], e["shape"][1] * 2
        packed = np.frombuffer(st_read(out_path, base, e), np.uint8)
        plane = np.frombuffer(st_read(out_path, base, hdr[b + ".weight_scale"]), np.uint8)
        plane = plane.reshape(hdr[b + ".weight_scale"]["shape"])
        s2 = np.frombuffer(st_read(out_path, base, hdr[b + ".weight_scale_2"]), "<f4")[0]
        conf = json.loads(st_read(out_path, base, hdr[b + ".comfy_quant"]).decode())

        back = L.comfy_to_ggml(packed, plane, in_f, out_f)
        src = np.frombuffer(g.raw(b + ".weight"), np.uint8)
        if permuted and is_block_attn(b + ".weight", "qkv_proj.weight"):
            row = src.size // out_f
            s = src.reshape(out_f, row).copy()
            inner = out_f // 3
            heads = inner // head_dim
            for third in range(2):
                sl = slice(third * inner, (third + 1) * inner)
                blk = s[sl].reshape(heads, head_dim, row)
                fixed = np.empty_like(blk)
                fixed[:, perm] = blk
                s[sl] = fixed.reshape(inner, row)
            src = s.reshape(-1)
            note = "  (RoPE permutation undone on both sides)"
        else:
            note = ""
        same = bool(np.array_equal(src, back))
        ok &= same and s2 == np.float32(1.0)
        print(f"    {b:<44} [{out_f},{in_f}] {conf}")
        print(f"      bytes identical to source GGUF: {same}   scale_2={s2}{note}")

    # --- 4. an unquantised tensor, byte-exact against comfy's own checkpoint
    if ref is not None:
        rh, rbase = st_header(COMFY_REF_DIT), None
        rh, rbase = rh[0], rh[1]
        for name in ("blocks.0.attn.q_norm.weight", "blocks.17.norm1.weight"):
            if name not in hdr or name not in rh:
                continue
            a_ = st_read(out_path, base, hdr[name])
            b_ = st_read(COMFY_REF_DIT, rbase, rh[name])
            same = a_ == b_ and hdr[name]["dtype"] == rh[name]["dtype"]
            print(f"\n  {name}: byte-identical to Comfy-Org's checkpoint: {same} "
                  f"({hdr[name]['dtype']} vs {rh[name]['dtype']})")
            ok &= same
    print("\n  VERIFY:", "PASS" if ok else "FAIL")
    return ok


# ----------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--dequant-refiner", action="store_true",
                    help="emit token_refiner.* as BF16, as Comfy-Org's own checkpoint does, "
                         "instead of NVFP4 (condition_proj is always BF16 -- see the code)")
    ap.add_argument("--full-precision-mm", action="store_true",
                    help='add "full_precision_matrix_mult": true to every quantised layer')
    ap.add_argument("--max-blocks", type=int, default=0, help="smoke test: keep only the first N DiT blocks")
    ap.add_argument("--dry-run", action="store_true", help="plan and report, write nothing")
    ap.add_argument("--verify", action="store_true", help="re-read and check the written file")
    ap.add_argument("--verify-only", action="store_true", help="skip the write, check an existing file")
    a = ap.parse_args()

    if a.verify_only:
        raise SystemExit(0 if verify(a.out, a.gguf) else 1)

    g = GGUF(a.gguf)
    names = [n for n in g.order if n not in MARKERS]
    if a.max_blocks:
        # Smoke-test escape hatch: a few DiT blocks exercise every code path in seconds.
        # The result is NOT loadable by ComfyUI (num_layers would be wrong).
        names = [n for n in names
                 if not n.startswith("blocks.") or int(n.split(".")[1]) < a.max_blocks]
    dropped = [n for n in g.order if n in MARKERS]

    # rot_dim comes from the file, not from a constant: a different inv_freq length would
    # mean a different permutation, and guessing it is exactly the silent-failure mode.
    rot_dim = 2 * 3 * int(g.torch_shape("rope.inv_freq")[0]) if "rope.inv_freq" in g.t else ROT_DIM
    head_dim = int(g.torch_shape("blocks.0.attn.q_norm.weight")[0])
    perm = qk_head_permutation(head_dim, rot_dim)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(head_dim)          # original = ours[inv_perm]

    permuted = "rope.qk_permuted" in g.t
    deinterleaved = "attn.qkv_deinterleaved" in g.t
    print(f"source     : {a.gguf}")
    print(f"markers    : qk_permuted={permuted}  qkv_deinterleaved={deinterleaved}")
    print(f"geometry   : head_dim={head_dim} rot_dim={rot_dim}")
    if not deinterleaved:
        raise SystemExit(
            "REFUSING: this GGUF carries no `attn.qkv_deinterleaved` marker, so its fused-QKV "
            "row order is UNKNOWN. ComfyUI expects [q_all; k_all; v_all]; shipping a per-head "
            "interleaved tensor would be silently wrong. Re-convert with "
            "MINIMAX_H3_QKV_DEINTERLEAVE=1.")
    if not permuted:
        print("NOTE: no `rope.qk_permuted` marker -- nothing to undo, q/k pass through.")

    quant_conf = {"format": "nvfp4"}
    if a.full_precision_mm:
        quant_conf["full_precision_matrix_mult"] = True
    conf_blob = json.dumps(quant_conf).encode("utf-8")

    def leave_unquantised(name):
        # 🔴 condition_proj is NOT optional.  comfy's model_detection reads the model's
        # text_dim off `condition_proj.weight.shape[1]` (comfy/model_detection.py:374) -- the
        # INPUT dim -- and an NVFP4 weight stores two elements per byte, so a packed
        # [5376, 2560] would build a Linear(2560, 5376) and the whole DiT would come out
        # mis-shaped.  It is the one tensor whose logical shape comfy cannot recover.  Every
        # other H3 config field is read off an OUTPUT dim or off a tensor we never quantise.
        if name == "condition_proj.weight":
            return True
        return a.dequant_refiner and name.startswith("token_refiner.")

    w = SafetensorsWriter()
    stats = {"nvfp4": 0, "bf16_downcast": 0, "verbatim": 0, "dequantised": 0}

    for name in names:
        t = g.type_name(name)
        shape = g.torch_shape(name)

        # -------------------------------------------------- NVFP4 linears
        if t == "nvfp4":
            out_f, in_f = int(shape[0]), int(shape[1])
            if in_f % L.QK_NVFP4:
                raise SystemExit(f"{name}: in_features {in_f} is not a multiple of {L.QK_NVFP4}")
            need_unpermute = permuted and is_block_attn(name, "qkv_proj.weight")

            def load_blocks(name=name, out_f=out_f, in_f=in_f, need_unpermute=need_unpermute):
                b = np.frombuffer(g.raw(name), np.uint8).reshape(
                    out_f, (in_f // L.QK_NVFP4) * L.NVFP4_BLOCK_BYTES)
                if need_unpermute:
                    b = b.copy()
                    inner = out_f // 3
                    heads = inner // head_dim
                    for third in range(2):        # q and k; v feeds out_proj and never moved
                        s = slice(third * inner, (third + 1) * inner)
                        blk = b[s].reshape(heads, head_dim, -1)
                        fixed = np.empty_like(blk)
                        fixed[:, perm] = blk       # invert ours[i] = original[perm[i]]
                        b[s] = fixed.reshape(inner, -1)
                return b.reshape(-1)

            if leave_unquantised(name):
                def produce(load=load_blocks, in_f=in_f, out_f=out_f, name=name):
                    f32 = L.dequantize_ggml(load(), in_f, out_f)
                    if not bf16_is_exact(f32):
                        raise SystemExit(f"{name}: dequantised NVFP4 is not bf16-exact; refusing")
                    return bf16_bytes(f32)
                w.add(name, "BF16", [out_f, in_f], out_f * in_f * 2, produce)
                stats["dequantised"] += 1
                continue

            plane_rows, plane_cols = L.scale_plane_shape(in_f, out_f)
            cache = {}

            def convert(load=load_blocks, in_f=in_f, out_f=out_f, cache=cache):
                if not cache:
                    packed, plane, _ = L.ggml_to_comfy(load(), in_f, out_f)
                    cache["packed"] = packed.tobytes()
                    cache["plane"] = plane.tobytes()
                return cache

            base = name[: -len(".weight")]
            w.add(name, "U8", [out_f, in_f // 2], out_f * (in_f // 2),
                  lambda c=convert: c().pop("packed"))
            w.add(base + ".weight_scale", "F8_E4M3", [plane_rows, plane_cols],
                  plane_rows * plane_cols, lambda c=convert: c().pop("plane"))
            w.add(base + ".weight_scale_2", "F32", [], 4,
                  lambda: np.float32(1.0).tobytes())
            w.add(base + ".comfy_quant", "U8", [len(conf_blob)], len(conf_blob),
                  lambda: conf_blob)
            stats["nvfp4"] += 1
            continue

        # -------------------------------------------------- q_norm / k_norm: undo the permutation
        if permuted and (is_block_attn(name, "q_norm.weight") or is_block_attn(name, "k_norm.weight")):
            v = f32_from_gguf(g, name)
            orig = np.empty_like(v)
            orig[perm] = v
            if bf16_is_exact(orig):
                w.add(name, "BF16", shape, orig.size * 2, lambda o=orig: bf16_bytes(o))
                stats["bf16_downcast"] += 1
            else:
                w.add(name, "F32", shape, orig.size * 4, lambda o=orig: o.astype("<f4").tobytes())
                stats["verbatim"] += 1
            continue

        # -------------------------------------------------- RMSNorm gains: back to bf16
        # Our GGUF upcast these to F32; comfy's own checkpoint stores BF16.  The downcast is
        # refused unless it is bit-exact, so it can only ever restore the original bytes.
        if t == "f32" and name.endswith(("norm.weight", "norm1.weight", "norm2.weight")):
            v = f32_from_gguf(g, name)
            if bf16_is_exact(v):
                w.add(name, "BF16", shape, v.size * 2, lambda o=v: bf16_bytes(o))
                stats["bf16_downcast"] += 1
                continue

        # -------------------------------------------------- everything else, verbatim
        dtype = {"f32": "F32", "f16": "F16", "bf16": "BF16"}.get(t)
        if dtype is None:
            raise SystemExit(f"{name}: unhandled ggml type {t}")
        n = int(np.prod(shape)) if shape else 1
        w.add(name, dtype, shape, n * ST_DTYPE_ITEMSIZE[dtype], lambda nm=name: g.raw(nm))
        stats["verbatim"] += 1

    print(f"dropped    : {dropped}")
    print(f"tensors    : {stats['nvfp4']} nvfp4 linears -> 4 entries each, "
          f"{stats['dequantised']} dequantised to bf16, "
          f"{stats['bf16_downcast']} norms downcast to bf16, {stats['verbatim']} verbatim")
    print(f"quant conf : {quant_conf}")
    total = sum(e[3] for e in w.entries)
    print(f"entries    : {len(w.entries)}   payload {total / 1e9:.2f} GB")
    if a.dry_run:
        return

    meta = {"exporter": "tools/h3_export_comfy_dit.py",
            "source_gguf": os.path.basename(a.gguf),
            "note": "OUR weights in ComfyUI layout; RoPE q/k head permutation undone, "
                    "fused QKV left contiguous"}

    def prog(i, n, name, done, plan):
        if i % 40 == 0 or i == n:
            print(f"  [{i:>4}/{n}] {done / 1e9:6.2f}/{plan / 1e9:.2f} GB  {name}", flush=True)

    size = w.write(a.out, metadata=meta, progress=prog)
    print(f"wrote {a.out}  ({size / 1e9:.2f} GB)")

    if a.verify:
        raise SystemExit(0 if verify(a.out, a.gguf) else 1)


if __name__ == "__main__":
    main()
