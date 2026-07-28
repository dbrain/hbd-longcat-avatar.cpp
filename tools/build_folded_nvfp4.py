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

Usage:
  build_folded_nvfp4.py --src <template.gguf> --bf16 <base.safetensors> \
                        --lora <lora.safetensors> --mult 1.5 --out <out.gguf>
"""
import argparse, json, os, struct, sys
import numpy as np

GT_F32, GT_BF16, GT_NVFP4 = 0, 30, 40
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


def quant_nvfp4_unfolded(W):
    """W [out,in] f64 -> (packed ggml block bytes, wglobal). ModelOpt UNFOLDED convention:
    per-tensor wglobal = amax/(6*448), per-16 e4m3 block scale expressed in units of wglobal."""
    out, inn = W.shape
    amax = float(np.abs(W).max())
    wg = (amax / (E2M1_MAX * E4M3_MAX)) if amax > 0 else 1.0
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="template gguf (tensor list/dims/types/KV)")
    ap.add_argument("--bf16", required=True, help="bf16 base safetensors")
    # Repeatable and PAIRWISE: --lora A --mult 2.0 --lora B --mult 1.5 stacks both, which is how
    # a "omninft-ar"-style variant is built (OmniNFT@2.0 + audio-reactive@1.5). Stacking in bf16
    # is exact — the deltas simply sum before the single quantisation.
    ap.add_argument("--lora", action="append", required=True)
    ap.add_argument("--mult", action="append", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if len(a.lora) != len(a.mult):
        sys.exit(f"--lora given {len(a.lora)}x but --mult {len(a.mult)}x; they must pair up")

    fb, bb, hb = st_open(a.bf16)
    loras = []
    for path, mult in zip(a.lora, a.mult):
        fl_, bl_, hl_ = st_open(path)
        st = {k[:-len(".lora_A.weight")] for k in hl_ if k.endswith(".lora_A.weight")}
        loras.append((os.path.basename(path), fl_, bl_, hl_, st, mult))
        print(f"  lora {os.path.basename(path)} @ {mult}  ({len(st)} modules)")
    stems = set().union(*(l[4] for l in loras))

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
    for _ in range(nt):
        kl = struct.unpack("<Q", f.read(8))[0]; name = f.read(kl).decode()
        nd = struct.unpack("<I", f.read(4))[0]
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        tt = struct.unpack("<I", f.read(4))[0]; struct.unpack("<Q", f.read(8))[0]
        if name.endswith(".wglobal"):
            continue                      # ours; regenerated below
        infos.append((name, dims, tt))
    f.close()

    def nbytes_of(tt, dims):
        ne = int(np.prod(dims))
        if tt == GT_F32: return ne * 4
        if tt == GT_BF16: return ne * 2
        if tt == GT_NVFP4: return (ne // dims[0]) * (dims[0] // 64) * 36
        raise Exception(f"unexpected type {tt}")

    plan = [(n, d, t) for (n, d, t) in infos]
    nvnames = [n for (n, d, t) in plan if t == GT_NVFP4]
    n_lora_hit = sum(1 for (n, d, t) in plan
                     if LORA_PFX + n[:-len(".weight")] in stems)
    print(f"src: {len(plan)} tensors ({len(nvnames)} nvfp4) | lora stems {len(stems)} | "
          f"lora-matched tensors {n_lora_hit} | mult {a.mult}")

    o = open(a.out, "wb")
    o.write(magic); o.write(struct.pack("<I", ver))
    o.write(struct.pack("<Q", len(plan) + len(nvnames))); o.write(struct.pack("<Q", nkv))
    o.write(KV)
    off = 0
    hdr = []
    for (n, d, t) in plan:
        hdr.append((n, d, t, off)); nb = nbytes_of(t, d); off += nb + ((-nb) % align)
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
    for i, (n, d, t) in enumerate(plan):
        oname = BASE_PFX + n
        u, dt, sh = st_raw(fb, bb, hb, oname)
        x = bf16_to_f32(u) if dt == "BF16" else u.view(np.float32)
        x = x.reshape(sh).astype(np.float64)
        stem = LORA_PFX + n[:-len(".weight")] if n.endswith(".weight") else None
        hit = False
        for (_nm, fl_, bl_, hl_, st_, mult) in loras:
            if stem not in st_:
                continue
            A = bf16_to_f32(st_raw(fl_, bl_, hl_, stem + ".lora_A.weight")[0]).reshape(
                hl_[stem + ".lora_A.weight"]["shape"]).astype(np.float32)
            B = bf16_to_f32(st_raw(fl_, bl_, hl_, stem + ".lora_B.weight")[0]).reshape(
                hl_[stem + ".lora_B.weight"]["shape"]).astype(np.float32)
            x = x + mult * (B.astype(np.float64) @ A.astype(np.float64))
            hit = True
        if hit:
            folded += 1
        if t == GT_NVFP4:
            assert d[0] == x.shape[1] and d[1] == x.shape[0], f"{n}: dims {d} vs {x.shape}"
            data, wg = quant_nvfp4_unfolded(x)
            wglobals[n] = wg
        elif t == GT_BF16:
            data = f32_to_bf16(x).tobytes()
        else:
            data = np.ascontiguousarray(x, dtype=np.float32).tobytes()
        nb = nbytes_of(t, d); assert len(data) == nb, f"{n}: {len(data)} vs {nb}"
        o.write(data); o.write(b"\x00" * ((-len(data)) % align))
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{len(plan)}  (folded {folded})", flush=True)
    for n in nvnames:
        o.write(struct.pack("<f", wglobals[n])); o.write(b"\x00" * ((-4) % align))
    o.close()
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes); folded {folded} tensors @ {a.mult}")
    print("run tools/verify_folded_build.py on this output before deploying.")


if __name__ == "__main__":
    main()
