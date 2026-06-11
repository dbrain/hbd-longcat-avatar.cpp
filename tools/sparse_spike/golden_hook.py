"""
Drop-in golden capture for Pixal3D / TRELLIS.2 submanifold sparse convs.
GPU-time, run-once, FLEX_GEMM backend (the only one installed).

songgen-style: monkeypatch SparseConv3d.forward, save the real {coords, feats,
weight, bias, model_out} for the first occurrence of each layer SHAPE, then
early-exit. Capture is fast (just tensor saves) — the bit-exact f64 golden is
recomputed OFFLINE by finalize_model_goldens.py (so no slow python conv on GPU).

flex_gemm specifics (vs spconv): the SparseConv3d module stores weight/bias
DIRECTLY on itself (no self.conv); weight layout is [Co,Kw,Kh,Kd,Ci]; input
feats/coords via the universal x.feats / x.coords accessors. We permute weight to
the canonical [V,Cin,Cout] used by sparse_conv_ref / sparse_conv.cpp.
model_out is flex_gemm's fp16/tf32 result — a TOLERANCE reference (semantics
check), not bit-exact.

USAGE: see golden_dump_runner.py.
"""
import json
import os
import numpy as np


class _StopCapture(Exception):
    pass


_STATE = {'out_dir': 'golden_model', 'max_per_key': 1, 'stop_after': 0,
          'seen': {}, 'count': 0, 'installed': False, 'timing': {}}


def _np(t):
    import torch
    return t.detach().to('cpu', dtype=torch.float32).numpy()


def install(out_dir='golden_model', max_per_key=1, stop_after=0):
    """stop_after>0 → raise _StopCapture once that many DISTINCT shapes are saved
    (caught by the runner) to avoid a full multi-minute decode."""
    if _STATE['installed']:
        return
    _STATE.update(out_dir=out_dir, max_per_key=max_per_key, stop_after=stop_after)
    os.makedirs(out_dir, exist_ok=True)
    from pixal3d.modules.sparse.conv import conv as conv_mod

    orig = conv_mod.SparseConv3d.forward

    import torch

    def hooked(self, x):
        w = getattr(self, 'weight', None)
        # time EVERY flex_gemm submanifold call (rough warm baseline = min over calls)
        if w is not None and w.dim() == 5:
            torch.cuda.synchronize()
            _t0 = torch.cuda.Event(enable_timing=True); _t1 = torch.cuda.Event(enable_timing=True)
            _t0.record()
            out = orig(self, x)
            _t1.record(); torch.cuda.synchronize()
            ms = _t0.elapsed_time(_t1)
            Co, Kw, Kh, Kd, Ci = w.shape
            tk = f"subm_K{Kw}{Kh}{Kd}_Ci{Ci}_Co{Co}_N{int(x.coords.shape[0])}"
            e = _STATE['timing'].setdefault(tk, {'count': 0, 'total_ms': 0.0, 'min_ms': 1e9})
            e['count'] += 1; e['total_ms'] += ms; e['min_ms'] = min(e['min_ms'], ms)
        else:
            out = orig(self, x)
        try:
            if w is None or w.dim() != 5:
                return out  # not the flex_gemm submanifold module we expect
            Co, Kw, Kh, Kd, Ci = w.shape
            key = f"subm_K{Kw}{Kh}{Kd}_Ci{Ci}_Co{Co}"
            n = _STATE['seen'].get(key, 0)
            if n < _STATE['max_per_key']:
                _STATE['seen'][key] = n + 1
                d = os.path.join(_STATE['out_dir'], f"{key}_{n}")
                os.makedirs(d, exist_ok=True)
                # canonical weight [V,Cin,Cout] = permute [Co,Kw,Kh,Kd,Ci] -> [Kw,Kh,Kd,Ci,Co] -> reshape
                wc = _np(w).transpose(1, 2, 3, 4, 0).reshape(Kw * Kh * Kd, Ci, Co)
                np.save(d + '/in_coords.npy', _np(x.coords).astype(np.int32))
                np.save(d + '/in_feats.npy', _np(x.feats).astype(np.float32))
                np.save(d + '/weight.npy', wc.astype(np.float32))
                b = getattr(self, 'bias', None)
                np.save(d + '/bias.npy',
                        (_np(b) if b is not None else np.zeros(Co, np.float32)).astype(np.float32))
                np.save(d + '/model_out.npy', _np(out.feats).astype(np.float32))
                dil = getattr(self, 'dilation', (1, 1, 1))
                json.dump(dict(name=f"{key}_{n}", op='subm', kernel_size=int(Kw),
                               Cin=int(Ci), Cout=int(Co), N=int(x.coords.shape[0]),
                               dilation=int(dil[0]) if hasattr(dil, '__len__') else int(dil),
                               weight_layout='flex_gemm[Co,Kw,Kh,Kd,Ci]->canonical[V,Cin,Cout]',
                               model_dtype=str(out.feats.dtype)),
                          open(d + '/manifest.json', 'w'), indent=2)
                _STATE['count'] += 1
                print(f"[golden_hook] {key}: N={x.coords.shape[0]} Ci={Ci} Co={Co} "
                      f"({_STATE['count']} saved)")
                if _STATE['stop_after'] and _STATE['count'] >= _STATE['stop_after']:
                    raise _StopCapture()
        except _StopCapture:
            raise
        except Exception as e:
            print(f"[golden_hook] capture skipped: {e}")
        return out

    conv_mod.SparseConv3d.forward = hooked

    import atexit

    def _dump_timing():
        t = _STATE['timing']
        if not t:
            return
        for k in t:
            t[k]['avg_ms'] = t[k]['total_ms'] / max(1, t[k]['count'])
        tot = sum(v['total_ms'] for v in t.values())
        summary = {'note': 'flex_gemm submanifold per-shape timing (3060). min_ms ~ warm '
                           '(excludes first-call autotune). BASELINE TO BEAT.',
                   'total_ms_all_calls': tot, 'n_shapes': len(t),
                   'n_calls': sum(v['count'] for v in t.values()), 'per_shape': t}
        json.dump(summary, open(os.path.join(_STATE['out_dir'], 'flexgemm_timing.json'), 'w'), indent=2)
        print(f"[golden_hook] timing: {len(t)} shapes, {summary['n_calls']} calls, "
              f"{tot:.1f} ms total submanifold-conv -> flexgemm_timing.json")

    atexit.register(_dump_timing)
    _STATE['installed'] = True
    print(f"[golden_hook] installed -> {out_dir} "
          f"(backend={os.environ.get('SPARSE_CONV_BACKEND', 'flex_gemm')}, "
          f"stop_after={stop_after})")
