#!/usr/bin/env python3
"""
Offline (CPU, GPU-free) finalizer for the real-layer goldens captured by
golden_hook / golden_dump_runner.

For each captured layer it:
  1. recomputes a BIT-EXACT f64 golden (out_feats.npy) from the saved inputs via
     the FlexGEMM rulebook + implicit-GEMM dataflow (vectorized);
  2. writes neighbor_map.npy;
  3. compares the recompute against flex_gemm's own fp16 output (model_out.npy)
     → confirms our semantics + weight layout match production within fp16/tf32
     tolerance, on REAL layer data.

After this runs, `sparse_conv_test golden_model` validates the C++ op against real
Pixal3D layers exactly the same way it does the synthetic cases.
"""
import argparse
import glob
import json
import os
import numpy as np
import sparse_conv_ref as ref

SENTINEL = ref.SENTINEL


def finalize(d):
    coords = np.load(d + '/in_coords.npy')
    feats = np.load(d + '/in_feats.npy').astype(np.float32)
    weight = np.load(d + '/weight.npy').astype(np.float32)      # [V,Cin,Cout]
    bias = np.load(d + '/bias.npy').astype(np.float32)
    man = json.load(open(d + '/manifest.json'))
    K = man['kernel_size']
    V, Cin, Cout = weight.shape
    N = coords.shape[0]

    nmap = ref.build_neighbor_map(coords, K)                    # [N,V] uint32
    valid = nmap != SENTINEL
    idx = np.where(valid, nmap, 0).astype(np.int64)
    wf = weight.astype(np.float64)
    out = np.zeros((N, Cout), np.float64)
    for v in range(V):
        g = feats[idx[:, v]].astype(np.float64)
        g[~valid[:, v]] = 0.0
        out += g @ wf[v]
    out += bias.astype(np.float64)
    golden = out.astype(np.float32)

    np.save(d + '/out_feats.npy', golden)
    np.save(d + '/out_coords.npy', coords.copy())              # submanifold: out coords == in
    np.save(d + '/neighbor_map.npy', nmap)

    rpt = {'N': int(N), 'V': int(V), 'Cin': int(Cin), 'Cout': int(Cout),
           'sentinel_frac': float(valid.size and (~valid).sum() / valid.size)}
    if os.path.exists(d + '/model_out.npy'):
        m = np.load(d + '/model_out.npy').astype(np.float32)
        if m.shape == golden.shape:
            aerr = float(np.abs(golden - m).max())
            denom = float(np.abs(golden).max()) or 1.0
            rpt.update(model_maxabs=aerr, model_maxrel=aerr / denom,
                       verdict='OK (fp16 tol)' if aerr / denom < 0.05 else 'CHECK')
        else:
            rpt.update(verdict='shape-mismatch', model_shape=list(m.shape))
    man['finalize'] = rpt
    json.dump(man, open(d + '/manifest.json', 'w'), indent=2)
    print(f"  {os.path.basename(d):28s} N={N:<7d} Ci={Cin:<4d} Co={Cout:<4d} "
          f"sent={rpt['sentinel_frac']*100:4.1f}%  "
          f"{'rel=%.2e %s' % (rpt['model_maxrel'], rpt['verdict']) if 'model_maxrel' in rpt else rpt.get('verdict','')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default=os.path.join(os.path.dirname(__file__), 'golden_model'))
    args = ap.parse_args()
    dirs = sorted(d for d in glob.glob(args.dir + '/*') if os.path.isfile(d + '/manifest.json'))
    print(f"finalize {len(dirs)} captured layers in {args.dir}")
    for d in dirs:
        try:
            finalize(d)
        except Exception as e:
            print(f"  {os.path.basename(d)}: FAILED {e}")
    print("done — now: ./sparse_conv_test golden_model")


if __name__ == '__main__':
    main()
