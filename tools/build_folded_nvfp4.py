#!/usr/bin/env python3
"""Build a FOLDED nvfp4 gguf by merging a LoRA into the bf16 checkpoint and quantising ONCE.

THIS IS "ROUTE B", AND IT IS NOT THE SAME THING AS tools/fold_generic_lora.py
----------------------------------------------------------------------------
  route A  fold_generic_lora.py   adds the delta onto an ALREADY-QUANTISED gguf, whose per-16
                                  block scales are frozen. Round-to-nearest then discards every
                                  sub-half-step component: measured projection ~0.36, i.e. two
                                  thirds of the LoRA is lost, unevenly across layers. It also
                                  cannot touch tensors the base stores as BF16 (for the fal
                                  audio-reactive LoRA that is 288 of its 1632 modules).

  route B  THIS TOOL              merges in bf16 and re-derives the block scales from the merged
                                  weights, so the quantiser is solving for W+dW instead of being
                                  forced onto W's grid. Measured projection 0.995-0.996 at every
                                  multiplier tried, ~93% of elements moving. It folds ALL modules
                                  because BF16/F32 tensors simply stay BF16/F32 with the delta in.

Structure is cloned from a SRC gguf (tensor list, dims, per-tensor OUTPUT TYPE, KV) exactly as
tools/import_ltx_nvfp4.py does, so the loader sees a byte-compatible layout. Per-tensor `.wglobal`
siblings are RECOMPUTED (amax/(6*448)) — a rebuild legitimately changes them, which is why
verify_fold.py's C5 does not apply here; use tools/verify_folded_build.py instead.

ADAPTER FORMATS
---------------
  LoRA  <stem>.lora_A.weight / .lora_B.weight        dW = mult * (B @ A)
  LoKr  <stem>.lokr_w1[_a/_b] / .lokr_w2[_a/_b]      dW = mult * scale * kron(w1, w2)
        Scale mirrors the engine EXACTLY (model/adapter/lora.hpp:494-505): rank starts at 1 and
        is only set when a factor is DECOMPOSED, so a full w1 + full w2 pair always lands on
        scale = 1.0 regardless of the stored alpha. ai-toolkit writes alpha ~ 1e10 for full LoKr
        precisely because it is meant to be ignored — do not divide by it.

--lora is OPTIONAL. With no adapter this is a pure requantise of the bf16 through the SRC
template, which is the BETTER way to build a plain base gguf: measured 0.9956-0.9962 against
`-M convert`'s 0.9918. Build the base and its folded variants the same way so an A/B differs
only by the delta.

Usage:
  # folded variant
  build_folded_nvfp4.py --src <template.gguf> --bf16 <base.safetensors> \
                        --lora <lora.safetensors> --mult 1.5 --out <out.gguf>
  # plain requantise (no adapter), bare-named checkpoint (e.g. Krea2's turbo.safetensors)
  build_folded_nvfp4.py --src <template.gguf> --bf16 <base.safetensors> \
                        --base-prefix "" --out <out.gguf>
"""
import argparse, json, os, struct, sys
import numpy as np

GT_F32, GT_BF16, GT_NVFP4 = 0, 30, 40
# Defaults suit the LTX DiTs (checkpoint names carry `model.diffusion_model.`, adapters
# `diffusion_model.`). Krea2's turbo.safetensors is BARE — pass --base-prefix "".
BASE_PFX, LORA_PFX = "model.diffusion_model.", "diffusion_model."
E2M1_MAX, E4M3_MAX, E4M3_MIN_SUB = 6.0, 448.0, 2.0 ** -9


def st_open(p):
    f = open(p, "rb"); n = struct.unpack("<Q", f.read(8))[0]
    h = json.loads(f.read(n)); h.pop("__metadata__", None); return f, 8 + n, h


def st_raw(f, base, h, k):
    m = h[k]; a, b = m["data_offsets"]; f.seek(base + a)
    return np.frombuffer(f.read(b - a), dtype=np.uint8), m["dtype"], m["shape"]


def bf16_to_f32(u8):
    return (u8.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype(np.uint16)


def e4m3_dec_byte(u8):
    u = u8.astype(np.int32); e = (u >> 3) & 0xF; m = u & 0x7
    v = np.where(e == 0, np.ldexp(m.astype(np.float64), -9), np.ldexp(1.0 + m / 8.0, e - 7))
    return np.where((u & 0x7F) == 0, 0.0, v)


_pb = np.arange(128, dtype=np.uint8); _pv = e4m3_dec_byte(_pb)
_o = np.argsort(_pv); _sv = _pv[_o]; _sb = _pb[_o]


def e4m3_enc_pos(x):
    x = np.minimum(np.asarray(x, dtype=np.float64), E4M3_MAX)
    i = np.clip(np.searchsorted(_sv, x), 0, len(_sv) - 1); lo = np.clip(i - 1, 0, len(_sv) - 1)
    return _sb[np.where(np.abs(_sv[i] - x) < np.abs(_sv[lo] - x), i, lo)].astype(np.uint8)


LUTm = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_nearest(t):
    s = (t < 0).astype(np.uint8); m = np.abs(t)
    i = np.clip(np.searchsorted(LUTm, m), 0, 7); lo = np.clip(i - 1, 0, 7)
    return (s << 3) | np.where(np.abs(LUTm[i] - m) < np.abs(LUTm[lo] - m), i, lo).astype(np.uint8)


def quant_nvfp4_unfolded(W, flat=False):
    """W [out,in] f64 -> (packed ggml block bytes, wglobal).

    UNFOLDED (ModelOpt) convention: per-tensor wglobal = amax/(6*448), per-16 e4m3 block scale
    expressed in UNITS OF wglobal, carried as a `<weight>.wglobal` sibling.

    FLAT (ggml's own quantize_row_nvfp4_ref) convention: wg == 1, so the block scale is the
    absolute ue4m3(amax_sub/6) and there is no sibling at all.

    ★ THE CHOICE IS NOT COSMETIC — IT IS A CORRECTNESS REQUIREMENT.
    Only the cuBLASLt FP4 GEMM folds a wglobal into the GEMM alpha. Any NVFP4 mul_mat that the
    CUDA backend routes to MMQ / dequant-cuBLAS / CPU multiplies by 1.0 instead, so every
    unfolded tensor that misses cuBLASLt comes out scaled wrong by a per-tensor constant. The
    graph-level compensation in ggml_ext_linear cannot save you: it decides at GRAPH BUILD time
    whether to elide the explicit multiply, while MMQ-vs-cuBLASLt is a RUNTIME shape decision it
    cannot see. Krea2 routes 144 of its NVFP4 matmuls through MMQ per step -> saturated colour
    patches. Emit UNFOLDED only when you have proven every NVFP4 matmul hits cuBLASLt."""
    out, inn = W.shape
    amax = float(np.abs(W).max())
    wg = 1.0 if flat else ((amax / (E2M1_MAX * E4M3_MAX)) if amax > 0 else 1.0)
    b16 = W.reshape(out, inn // 16, 16)
    sc = np.clip(np.abs(b16).max(axis=2) / (E2M1_MAX * wg), E4M3_MIN_SUB, E4M3_MAX)
    byte = e4m3_enc_pos(sc)                                     # [out, nb16]
    eff = np.repeat(e4m3_dec_byte(byte), 16, axis=1) * wg        # true-unit step
    code = e2m1_nearest(np.where(eff > 0, W / eff, 0.0))         # [out, inn] nibbles
    nblk = inn // 64
    oc = code.reshape(out, nblk, 64)
    qs = np.empty((out, nblk, 32), dtype=np.uint8)
    for s in range(4):                                           # low nibble = first 8 of the 16
        sub = oc[:, :, s * 16:s * 16 + 16]
        qs[:, :, s * 8:s * 8 + 8] = sub[:, :, 0:8] | (sub[:, :, 8:16] << 4)
    db = byte.reshape(out, nblk, 4).astype(np.uint8)
    return np.concatenate([db, qs], axis=2).tobytes(), wg


def adapter_stems(h):
    """{stem: kind} for every module in an adapter safetensors. kind is 'lora' or 'lokr'."""
    out = {}
    for k in h:
        if k.endswith(".lora_A.weight"):
            out[k[:-len(".lora_A.weight")]] = "lora"
        elif k.endswith(".lokr_w1"):
            out[k[:-len(".lokr_w1")]] = "lokr"
        elif k.endswith(".lokr_w1_a"):
            out[k[:-len(".lokr_w1_a")]] = "lokr"
    return out


def _st_f32(f, base, h, key):
    u, dt, sh = st_raw(f, base, h, key)
    x = bf16_to_f32(u) if dt == "BF16" else u.view(np.float32)
    return x.reshape(sh).astype(np.float64) if sh else x.astype(np.float64)


def lokr_delta(f, base, h, stem):
    """dW for one LoKr module, WITHOUT the multiplier. Mirrors get_lokr_weight_diff():
    rank defaults to 1 and only a DECOMPOSED factor sets it, so full w1 x full w2 => scale 1.0
    and the (deliberately absurd) stored alpha is never applied."""
    rank = 1
    if stem + ".lokr_w1" in h:
        w1 = _st_f32(f, base, h, stem + ".lokr_w1")
    else:
        w1_a, w1_b = _st_f32(f, base, h, stem + ".lokr_w1_a"), _st_f32(f, base, h, stem + ".lokr_w1_b")
        rank = w1_b.shape[-1]
        w1 = w1_b @ w1_a
    if stem + ".lokr_w2" in h:
        w2 = _st_f32(f, base, h, stem + ".lokr_w2")
    else:
        w2_a, w2_b = _st_f32(f, base, h, stem + ".lokr_w2_a"), _st_f32(f, base, h, stem + ".lokr_w2_b")
        rank = w2_b.shape[-1]
        w2 = w2_b @ w2_a
    scale = 1.0
    if rank != 1 and stem + ".alpha" in h:
        scale = float(_st_f32(f, base, h, stem + ".alpha").reshape(-1)[0]) / rank
    return scale * np.kron(w1, w2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="template gguf (tensor list/dims/types/KV)")
    ap.add_argument("--bf16", required=True, help="bf16 base safetensors")
    # Repeatable and PAIRWISE: --lora A --mult 2.0 --lora B --mult 1.5 stacks both, which is how
    # a "omninft-ar"-style variant is built (OmniNFT@2.0 + audio-reactive@1.5). Stacking in bf16
    # is exact — the deltas simply sum before the single quantisation.
    # OPTIONAL: with none, this is a straight requantise of --bf16 through the template.
    ap.add_argument("--lora", action="append", default=[])
    ap.add_argument("--mult", action="append", type=float, default=[])
    ap.add_argument("--base-prefix", default=BASE_PFX,
                    help='prefix the CHECKPOINT puts on the template\'s bare names '
                         f'(default "{BASE_PFX}"; pass "" for a bare checkpoint like Krea2)')
    ap.add_argument("--lora-prefix", default=LORA_PFX,
                    help=f'prefix the ADAPTER puts on those names (default "{LORA_PFX}")')
    ap.add_argument("--flat", dest="flat", action="store_true", default=None,
                    help="force FLAT nvfp4 (no .wglobal siblings); default follows the template")
    ap.add_argument("--unfolded", dest="flat", action="store_false",
                    help="force UNFOLDED nvfp4 (+ .wglobal siblings) — only if every NVFP4 "
                         "matmul is proven to hit the cuBLASLt FP4 GEMM")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len(a.lora) != len(a.mult):
        sys.exit(f"--lora given {len(a.lora)}x but --mult {len(a.mult)}x; they must pair up")
    base_pfx, lora_pfx = a.base_prefix, a.lora_prefix

    fb, bb, hb = st_open(a.bf16)
    loras = []
    for path, mult in zip(a.lora, a.mult):
        fl_, bl_, hl_ = st_open(path)
        st = adapter_stems(hl_)
        kinds = sorted(set(st.values()))
        loras.append((os.path.basename(path), fl_, bl_, hl_, st, mult))
        print(f"  lora {os.path.basename(path)} @ {mult}  ({len(st)} modules, {'+'.join(kinds)})")
    stems = set().union(*(set(l[4]) for l in loras)) if loras else set()
    if not loras:
        print("  no adapter: straight requantise of --bf16 through the template")

    # ---- parse SRC gguf: KV verbatim, tensor list minus our own .wglobal siblings ----
    f = open(a.src, "rb")
    magic = f.read(4); ver = struct.unpack("<I", f.read(4))[0]
    nt = struct.unpack("<Q", f.read(8))[0]; nkv = struct.unpack("<Q", f.read(8))[0]
    kv_start = f.tell()

    def rs():
        n = struct.unpack("<Q", f.read(8))[0]; return f.read(n)

    def sv(t):
        if t in (0, 1, 7): f.read(1)
        elif t in (2, 3): f.read(2)
        elif t in (4, 5, 6): f.read(4)
        elif t in (10, 11, 12): f.read(8)
        elif t == 8: rs()
        elif t == 9:
            et = struct.unpack("<I", f.read(4))[0]
            nn = struct.unpack("<Q", f.read(8))[0]
            [(rs() if et == 8 else sv(et)) for _ in range(nn)]

    align = 32
    for _ in range(nkv):
        kl = struct.unpack("<Q", f.read(8))[0]; k = f.read(kl)
        t = struct.unpack("<I", f.read(4))[0]
        if k == b"general.alignment":
            align = struct.unpack("<I", f.read(4))[0]
        else:
            sv(t)
    kv_end = f.tell(); f.seek(kv_start); KV = f.read(kv_end - kv_start)

    infos = []
    src_has_wglobal = False
    for _ in range(nt):
        kl = struct.unpack("<Q", f.read(8))[0]; name = f.read(kl).decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack("<I", f.read(4))[0]; struct.unpack("<Q", f.read(8))[0]
        if name.endswith(".wglobal"):
            src_has_wglobal = True
            continue                      # ours; regenerated below
        infos.append((name, dims, tt))
    f.close()

    # Clone the template's CONVENTION, not just its tensor list. A template with no .wglobal
    # siblings is a flat ggml build, and emitting unfolded weights against it silently breaks
    # every matmul the backend routes away from cuBLASLt (see quant_nvfp4_unfolded).
    flat = (not src_has_wglobal) if a.flat is None else a.flat
    print(f"  convention: {'FLAT (no wglobal siblings)' if flat else 'UNFOLDED (+ wglobal siblings)'}"
          f"  [template {'has' if src_has_wglobal else 'has no'} wglobal]")

    def nbytes_of(tt, dims):
        ne = int(np.prod(dims))
        if tt == GT_F32: return ne * 4
        if tt == GT_BF16: return ne * 2
        if tt == GT_NVFP4: return (ne // dims[0]) * (dims[0] // 64) * 36
        raise Exception(f"unexpected type {tt}")

    plan = [(n, d, t) for (n, d, t) in infos]
    nvnames = [n for (n, d, t) in plan if t == GT_NVFP4]
    n_lora_hit = sum(1 for (n, d, t) in plan
                     if n.endswith(".weight") and lora_pfx + n[:-len(".weight")] in stems)
    print(f"src: {len(plan)} tensors ({len(nvnames)} nvfp4) | lora stems {len(stems)} | "
          f"lora-matched tensors {n_lora_hit} | mult {a.mult}")

    o = open(a.out, "wb")
    o.write(magic); o.write(struct.pack("<I", ver))
    n_extra = 0 if flat else len(nvnames)
    o.write(struct.pack("<Q", len(plan) + n_extra)); o.write(struct.pack("<Q", nkv))
    o.write(KV)
    off = 0
    hdr = []
    for (n, d, t) in plan:
        hdr.append((n, d, t, off)); nb = nbytes_of(t, d); off += nb + ((-nb) % align)
    if not flat:
        for n in nvnames:
            hdr.append((n + ".wglobal", [1], GT_F32, off)); off += 4 + ((-4) % align)
    for (n, d, t, oo) in hdr:
        nbk = n.encode(); o.write(struct.pack("<Q", len(nbk))); o.write(nbk)
        o.write(struct.pack("<I", len(d)))
        for x in d: o.write(struct.pack("<Q", x))
        o.write(struct.pack("<I", t)); o.write(struct.pack("<Q", oo))
    cur = o.tell(); o.write(b"\x00" * ((-cur) % align))

    wglobals = {}
    folded = 0
    applied = {l[0]: set() for l in loras}
    for i, (n, d, t) in enumerate(plan):
        oname = base_pfx + n
        u, dt, sh = st_raw(fb, bb, hb, oname)
        x = bf16_to_f32(u) if dt == "BF16" else u.view(np.float32)
        x = x.reshape(sh).astype(np.float64)
        stem = lora_pfx + n[:-len(".weight")] if n.endswith(".weight") else None
        hit = False
        for (nm_, fl_, bl_, hl_, st_, mult) in loras:
            if stem not in st_:
                continue
            if st_[stem] == "lokr":
                dW = lokr_delta(fl_, bl_, hl_, stem)
            else:
                A = _st_f32(fl_, bl_, hl_, stem + ".lora_A.weight")
                B = _st_f32(fl_, bl_, hl_, stem + ".lora_B.weight")
                dW = B @ A
            assert dW.shape == x.shape, f"{n}: delta {dW.shape} vs base {x.shape}"
            x = x + mult * dW
            applied[nm_].add(stem)
            hit = True
        if hit:
            folded += 1
        if t == GT_NVFP4:
            assert d[0] == x.shape[1] and d[1] == x.shape[0], f"{n}: dims {d} vs {x.shape}"
            data, wg = quant_nvfp4_unfolded(x, flat=flat)
            wglobals[n] = wg
        elif t == GT_BF16:
            data = f32_to_bf16(x).tobytes()
        else:
            data = np.ascontiguousarray(x, dtype=np.float32).tobytes()
        nb = nbytes_of(t, d); assert len(data) == nb, f"{n}: {len(data)} vs {nb}"
        o.write(data); o.write(b"\x00" * ((-len(data)) % align))
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(plan)}  (folded {folded})", flush=True)
    if not flat:
        for n in nvnames:
            o.write(struct.pack("<f", wglobals[n])); o.write(b"\x00" * ((-4) % align))
    o.close()
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes); folded {folded} tensors @ {a.mult}")
    # An adapter whose modules match NOTHING produces a file byte-identical to the plain
    # requantise and looks like a successful build. That exact failure has already cost this
    # tree a round of "the LoRA didn't do much" — make it loud.
    for (nm_, _f, _b, _h, st_, _m) in loras:
        missed = set(st_) - applied[nm_]
        if missed:
            print(f"!! {nm_}: {len(missed)}/{len(st_)} modules matched NO template tensor, e.g. "
                  f"{sorted(missed)[:3]} -- check --lora-prefix/--base-prefix", file=sys.stderr)
            if len(missed) == len(st_):
                sys.exit(f"!! {nm_} folded NOTHING; refusing to pass this off as a folded build")
    print("run tools/verify_folded_build.py on this output before deploying.")


if __name__ == "__main__":
    main()
