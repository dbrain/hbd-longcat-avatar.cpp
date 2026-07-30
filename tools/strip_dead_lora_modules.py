#!/usr/bin/env python3
"""Drop the LoRA modules whose `lora_B` never left its zero init.

WHY THIS IS FREE
----------------
A LoRA module contributes `delta = mult * (B @ A)`. If B is exactly zero the product is
exactly zero for every multiplier, so removing the pair is BIT-EXACT — not "close enough".
That is not a hypothesis about training dynamics: it is checked here, per tensor, with
`np.abs(B).max() == 0.0`, and a module keeps its A/B unless that test passes.

WHY IT IS WORTH DOING
---------------------
Upstream LTX-2.3 trainers commonly instantiate adapters over EVERY Linear in the block —
including the audio branch (`audio_attn1/2`, `audio_to_video_attn`, `video_to_audio_attn`,
`audio_ff`) — and then train on a video-only dataset, so those Bs stay at zero. They still
cost disk, VRAM and fold time:

  faceid-closeup-rt   864 dead of 1344  ->  1.31 GB shrank to 695 MB
  VBVR-I2V-390K       768 dead of 1248  ->  554 MB of BF16 becomes 480 live modules
  ltx2.3-transition    96 dead of  576  ->  the two `audio_ff` families only

Fold-at-load walks the module list, so a dead module is also ~a fold-time no-op that still
pays a base-tensor lookup and a launch. Stripping keeps the `(N/N) applied` line honest:
after this, every module the engine reports is a module that actually moves a weight.

OUTPUT is a BF16 safetensors in the SAME (engine-native `diffusion_model.*`) convention as
the input, so it drops straight into convert_lora_q8.py. Keep the upstream file: it is the
archival original, and verify_lora_q8.py's L1 structure pass compares against whatever you
hand it, so verify the STRIPPED file against the STRIPPED source.

Usage:
  strip_dead_lora_modules.py <in.safetensors> <out.safetensors> [--dry-run]
"""
import argparse
import collections
import json
import os
import struct

import numpy as np

A_SUFFIX = '.lora_A.weight'
B_SUFFIX = '.lora_B.weight'


def st_read(path):
    f = open(path, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
    meta = h.pop('__metadata__', None)
    return f, 8 + n, h, meta


def raw_of(f, base, m):
    a, b = m['data_offsets']
    f.seek(base + a)
    return f.read(b - a)


def is_zero(m, raw):
    """Exact-zero test on the RAW bytes, per dtype. No dequantisation tolerance involved."""
    if m['dtype'] == 'BF16' or m['dtype'] == 'F16':
        u = np.frombuffer(raw, dtype=np.uint16)
        # +0.0 and -0.0 are both zero for our purposes: B@A is exactly 0 either way.
        return bool(np.all((u & 0x7FFF) == 0))
    if m['dtype'] == 'F32':
        return bool(np.all(np.frombuffer(raw, dtype=np.float32) == 0.0))
    raise SystemExit(f"unsupported dtype {m['dtype']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('out')
    ap.add_argument('--dry-run', action='store_true', help='report the split, write nothing')
    a = ap.parse_args()

    f, base, h, meta = st_read(a.src)

    mods, loose = collections.defaultdict(dict), []
    for k in h:
        if k.endswith(A_SUFFIX):
            mods[k[: -len(A_SUFFIX)]]['A'] = k
        elif k.endswith(B_SUFFIX):
            mods[k[: -len(B_SUFFIX)]]['B'] = k
        else:
            loose.append(k)   # .alpha and friends — carried through untouched

    dead, live = [], []
    for stem, kv in mods.items():
        if 'A' not in kv or 'B' not in kv:
            live.append(stem)          # half a module is not something to reason about — keep it
            continue
        (dead if is_zero(h[kv['B']], raw_of(f, base, h[kv['B']])) else live).append(stem)

    def family(s):
        return '.'.join(p for p in s.split('.') if not p.isdigit())

    print(f"{a.src}: {len(mods)} modules  ({len(live)} live, {len(dead)} dead), {len(loose)} loose")
    for tag, group in (('DEAD', dead), ('LIVE', live)):
        for fam, n in sorted(collections.Counter(family(s) for s in group).items()):
            print(f"  {tag} {n:4d}  {fam}")
    if loose:
        print(f"  KEPT (not A/B): {loose[:6]}{' …' if len(loose) > 6 else ''}")

    if a.dry_run:
        return
    if not dead:
        print("nothing dead — write skipped, use the source file as-is")
        return

    keep = [k for stem in live for k in (mods[stem].get('A'), mods[stem].get('B')) if k]
    keep += loose

    out_header, blobs, off = {}, [], 0
    for k in sorted(keep):
        m = h[k]
        raw = raw_of(f, base, m)
        out_header[k] = {'dtype': m['dtype'], 'shape': m['shape'],
                         'data_offsets': [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    if meta is not None:
        out_header['__metadata__'] = meta

    hdr = json.dumps(out_header).encode()
    hdr += b' ' * ((-len(hdr)) % 8)
    with open(a.out, 'wb') as o:
        o.write(struct.pack('<Q', len(hdr)))
        o.write(hdr)
        for d in blobs:
            o.write(d)
    print(f"wrote {a.out} ({os.path.getsize(a.out)} bytes, "
          f"{os.path.getsize(a.out) / os.path.getsize(a.src) * 100:.1f}% of source, "
          f"{len(keep)} tensors)")


if __name__ == '__main__':
    main()
