#!/usr/bin/env python3
# Compare a dumped tap (LONGCAT_DUMP_DIR .bin) between off (full) and on (noise-only),
# slicing off's cond-prefix so shapes match. Localizes the cond-cache consume bug.
import sys, struct, numpy as np

def load(p):
    with open(p, 'rb') as f:
        ndim = struct.unpack('<q', f.read(8))[0]
        dims = [struct.unpack('<q', f.read(8))[0] for _ in range(ndim)]
        data = np.frombuffer(f.read(), dtype='<f4')
    # ggml ne order, fastest-varying first (dims[0]); reshape reversed for numpy row-major
    return data.reshape(dims[::-1]), dims

off, doff = load(sys.argv[1])   # full:  reversed dims -> (..., n_tok_full, C)
on,  don  = load(sys.argv[2])   # noise: (..., n_noise, C)
ncond = int(sys.argv[3]) if len(sys.argv) > 3 else (doff[1]-don[1] if len(doff)>1 else 0)
print(f"off dims(ne)={doff} on dims(ne)={don} ncond={ncond}")
# token dim is axis -2 in numpy (since C fastest -> last axis)
off_noise = off[..., ncond:, :] if off.ndim >= 2 else off[ncond:]
print(f"off_noise shape={off_noise.shape} on shape={on.shape}")
if off_noise.shape != on.shape:
    print("SHAPE MISMATCH"); sys.exit(1)
d = np.abs(off_noise - on)
denom = np.abs(off_noise).mean() + 1e-9
print(f"max|off_noise-on|={d.max():.6g} mean={d.mean():.6g} rel_mean={d.mean()/denom:.6g}")
print("VERDICT:", "MATCH (bug is downstream of this tap)" if d.max() < 1e-3 else "DIVERGES HERE")
# per-frame breakdown: arg5 = n_per_frame (token dim is axis -2)
npf = int(sys.argv[4]) if len(sys.argv) > 4 else 0
if npf > 0 and on.ndim >= 2 and on.shape[-2] % npf == 0:
    nfr = on.shape[-2] // npf
    dr = d.reshape(d.shape[:-2] + (nfr, npf, d.shape[-1]))
    for f in range(nfr):
        print(f"  frame {f}: max={dr[...,f,:,:].max():.6g} mean={dr[...,f,:,:].mean():.6g}")
