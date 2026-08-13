#!/usr/bin/env python3
"""
Pure-numpy reference for 3D sparse convolution (spconv-compatible semantics),
plus a synthetic golden generator.  NO GPU, NO spconv, NO model required.

This is the GPU-free driver for the C++ ggml sparse-conv op: the C++ op is
developed and bit-exact-tested against the goldens this script emits, long
before the real model is ever run.  The real-model hook (dump_model_golden.py)
later confirms these semantics match spconv/flex_gemm in production.

Two ops, matching pixal3d/modules/sparse/conv/conv_spconv.py:
  * SubMConv3d   : stride 1, no padding. Output coords == input coords
                   (submanifold — sparsity preserved). Only ACTIVE input
                   neighbours contribute (inactive sites are skipped, not
                   zero-padded).
  * SparseConv3d : strided/padded. Generates a new (down-sampled) active set;
                   an output site is active if ANY kernel tap lands on an
                   active input site.

Convention (DEEP-LEARNING CROSS-CORRELATION, what spconv implements):
    out[p, :] = bias + Σ_k  W[k] · in[ p + (k - center) ]
  with kernel index k = (kz,ky,kx) ∈ [0,K)³, center = (K-1)//2, dilation d
  scaling the offset.  Canonical layouts used by BOTH this ref and the C++ op:
    coords  : int32  [N,4]  = (batch, z, y, x)
    feats   : f32    [N,Cin]
    weight  : f32    [K*K*K, Cin, Cout]   kernel flattened z-major (kz,ky,kx)
    bias    : f32    [Cout]
The real-model dumper permutes spconv's native weight into this layout so the
C++ op only ever sees ONE convention.
"""
import argparse
import json
import os
import numpy as np

# ----------------------------------------------------------------------------
# kernel offset table (z-major, matches weight flatten order)
# ----------------------------------------------------------------------------
def kernel_offsets(K, dilation):
    c = (K - 1) // 2
    offs = []
    for kz in range(K):
        for ky in range(K):
            for kx in range(K):
                offs.append(((kz - c) * dilation,
                             (ky - c) * dilation,
                             (kx - c) * dilation))
    return offs  # length K*K*K, index == weight first dim


def _coord_key(c):
    # pack (b,z,y,x) int32 into a python int for hashing; assumes coords < 2^20
    b, z, y, x = (int(v) for v in c)
    return (((b << 20 | (z & 0xFFFFF)) << 20 | (y & 0xFFFFF)) << 20) | (x & 0xFFFFF)


SENTINEL = 0xFFFFFFFF  # FlexGEMM "absent neighbour" marker (uint32)


# ----------------------------------------------------------------------------
# Rulebook (neighbor map) — FlexGEMM's data structure, computed once per active
# set and reused across every submanifold conv at that resolution.
# neighbor_map[n, v] = index of the input voxel at coord[n]+offset[v], else SENTINEL.
# Offset order == kernel_offsets() (z-major), == weight first dim.
# ----------------------------------------------------------------------------
def build_neighbor_map(coords, kernel_size, dilation=1):
    N = coords.shape[0]
    offs = kernel_offsets(kernel_size, dilation)
    V = len(offs)
    index = {_coord_key(coords[i]): i for i in range(N)}
    nmap = np.full((N, V), SENTINEL, dtype=np.uint32)
    for i in range(N):
        b, z, y, x = (int(v) for v in coords[i])
        for k, (dz, dy, dx) in enumerate(offs):
            j = index.get(_coord_key((b, z + dz, y + dy, x + dx)))
            if j is not None:
                nmap[i, k] = j
    return nmap


def submconv3d_via_rulebook(feats, weight, bias, nmap):
    """Implicit-GEMM dataflow: gather neighbours via the rulebook, accumulate.
    Exactly mirrors what the CUDA kernel will do (no im2col buffer)."""
    N = feats.shape[0]
    V, Cin, Cout = weight.shape
    ff = feats.astype(np.float64)
    wf = weight.astype(np.float64)
    out = np.zeros((N, Cout), dtype=np.float64)
    for i in range(N):
        acc = np.zeros(Cout, dtype=np.float64)
        for v in range(V):
            j = int(nmap[i, v])
            if j != SENTINEL:
                acc += ff[j] @ wf[v]
        out[i] = acc
    if bias is not None:
        out += bias.astype(np.float64)
    return out.astype(np.float32)


# ----------------------------------------------------------------------------
# SubMConv3d  (stride 1, output set == input set)
# ----------------------------------------------------------------------------
def submconv3d(coords, feats, weight, bias, kernel_size, dilation=1):
    N, Cin = feats.shape
    KKK, Cin_w, Cout = weight.shape
    assert Cin_w == Cin
    assert KKK == kernel_size ** 3
    offs = kernel_offsets(kernel_size, dilation)

    index = {_coord_key(coords[i]): i for i in range(N)}
    out = np.zeros((N, Cout), dtype=np.float64)  # accumulate in f64, cast later
    wf = weight.astype(np.float64)
    ff = feats.astype(np.float64)

    for i in range(N):
        b, z, y, x = (int(v) for v in coords[i])
        acc = np.zeros(Cout, dtype=np.float64)
        for k, (dz, dy, dx) in enumerate(offs):
            j = index.get(_coord_key((b, z + dz, y + dy, x + dx)))
            if j is not None:
                acc += ff[j] @ wf[k]      # [Cin] @ [Cin,Cout] -> [Cout]
        out[i] = acc
    if bias is not None:
        out += bias.astype(np.float64)
    return out.astype(np.float32)  # output coords == input coords


# ----------------------------------------------------------------------------
# SparseConv3d (strided / padded, new down-sampled active set)
# ----------------------------------------------------------------------------
def sparseconv3d(coords, feats, weight, bias, kernel_size, stride, padding):
    N, Cin = feats.shape
    KKK, _, Cout = weight.shape
    offs = kernel_offsets(kernel_size, 1)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if padding is None:
        padding = (0, 0, 0)
    elif isinstance(padding, int):
        padding = (padding, padding, padding)

    index = {_coord_key(coords[i]): i for i in range(N)}
    ff = feats.astype(np.float64)
    wf = weight.astype(np.float64)

    # An output site o (in down-sampled grid) gathers input sites
    #   in_pos = o*stride + (k - center) - padding   ... standard sparse conv.
    # Build active output set: any input p maps to the output sites that cover it.
    out_acc = {}  # out_key -> (out_coord tuple, np.array[Cout])
    c = (kernel_size - 1) // 2
    for i in range(N):
        b, z, y, x = (int(v) for v in coords[i])
        for k, (kz, ky, kx) in enumerate(
                [(a + c, bb + c, cc + c) for (a, bb, cc) in offs]):  # back to [0,K)
            # input p contributes to output o where p = o*stride + (k - center) - pad
            #   => o = (p + pad - (k - center)) / stride   must be integral & >=0
            nz = z + padding[0] - (kz - c)
            ny = y + padding[1] - (ky - c)
            nx = x + padding[2] - (kx - c)
            if nz % stride[0] or ny % stride[1] or nx % stride[2]:
                continue
            oz, oy, ox = nz // stride[0], ny // stride[1], nx // stride[2]
            if oz < 0 or oy < 0 or ox < 0:
                continue
            key = _coord_key((b, oz, oy, ox))
            if key not in out_acc:
                out_acc[key] = [(b, oz, oy, ox), np.zeros(Cout, dtype=np.float64)]
            out_acc[key][1] += ff[i] @ wf[k]

    # deterministic order: sort by (b,z,y,x)
    items = sorted(out_acc.values(), key=lambda t: t[0])
    out_coords = np.array([it[0] for it in items], dtype=np.int32)
    out_feats = np.stack([it[1] for it in items]).astype(np.float64)
    if bias is not None:
        out_feats += bias.astype(np.float64)
    return out_coords, out_feats.astype(np.float32)


# ----------------------------------------------------------------------------
# synthetic golden generator
# ----------------------------------------------------------------------------
def gen_case(out_dir, name, *, grid=16, n_active=200, Cin=8, Cout=16,
             kernel_size=3, op='subm', stride=2, padding=None, seed=0):
    rng = np.random.default_rng(seed)
    # random distinct active voxels in a single batch
    all_xyz = rng.choice(grid ** 3, size=n_active, replace=False)
    zyx = np.stack([all_xyz // (grid * grid),
                    (all_xyz // grid) % grid,
                    all_xyz % grid], axis=1).astype(np.int32)
    coords = np.concatenate([np.zeros((n_active, 1), np.int32), zyx], axis=1)
    coords = coords[np.lexsort((coords[:, 3], coords[:, 2], coords[:, 1]))]
    feats = rng.standard_normal((n_active, Cin)).astype(np.float32)
    weight = (rng.standard_normal((kernel_size ** 3, Cin, Cout))
              * (1.0 / np.sqrt(Cin * kernel_size ** 3))).astype(np.float32)
    bias = (rng.standard_normal(Cout) * 0.1).astype(np.float32)

    meta = dict(name=name, op=op, grid=grid, kernel_size=kernel_size,
                Cin=Cin, Cout=Cout, n_active=int(n_active), seed=seed)
    nmap = None
    if op == 'subm':
        out_feats = submconv3d(coords, feats, weight, bias, kernel_size)
        out_coords = coords.copy()
        meta.update(stride=1, padding=0)
        # FlexGEMM dataflow cross-check: rulebook + implicit GEMM == direct math
        nmap = build_neighbor_map(coords, kernel_size)
        out_rb = submconv3d_via_rulebook(feats, weight, bias, nmap)
        rb_err = float(np.abs(out_rb - out_feats).max())
        assert rb_err == 0.0, f"rulebook path mismatch {rb_err}"
        meta['rulebook_selfcheck'] = 'pass'
    else:
        out_coords, out_feats = sparseconv3d(coords, feats, weight, bias,
                                             kernel_size, stride, padding)
        meta.update(stride=stride, padding=padding if padding else 0,
                    n_active_out=int(out_coords.shape[0]))

    case_dir = os.path.join(out_dir, name)
    os.makedirs(case_dir, exist_ok=True)
    np.save(os.path.join(case_dir, 'in_coords.npy'), coords)
    np.save(os.path.join(case_dir, 'in_feats.npy'), feats)
    np.save(os.path.join(case_dir, 'weight.npy'), weight)
    np.save(os.path.join(case_dir, 'bias.npy'), bias)
    np.save(os.path.join(case_dir, 'out_coords.npy'), out_coords)
    np.save(os.path.join(case_dir, 'out_feats.npy'), out_feats)
    if nmap is not None:
        np.save(os.path.join(case_dir, 'neighbor_map.npy'), nmap)
    with open(os.path.join(case_dir, 'manifest.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[{name}] op={op} in={coords.shape[0]} out={out_coords.shape[0]} "
          f"Cin={Cin} Cout={Cout} K={kernel_size} -> {case_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'golden_ref'))
    args = ap.parse_args()
    # a spread of cases the C++ op must reproduce bit-for-bit (fp tolerance)
    gen_case(args.out, 'subm_k3_small', op='subm', grid=12, n_active=120,
             Cin=8, Cout=16, kernel_size=3, seed=1)
    gen_case(args.out, 'subm_k3_dense', op='subm', grid=8, n_active=300,
             Cin=16, Cout=16, kernel_size=3, seed=2)
    gen_case(args.out, 'subm_k1', op='subm', grid=16, n_active=200,
             Cin=8, Cout=32, kernel_size=1, seed=3)
    gen_case(args.out, 'sparse_k3_s2', op='sparse', grid=16, n_active=250,
             Cin=8, Cout=16, kernel_size=3, stride=2, padding=1, seed=4)
    print("done.")


if __name__ == '__main__':
    main()
