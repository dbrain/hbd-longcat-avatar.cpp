#!/usr/bin/env python3
"""
ShotStream (Wan2.1-T2V-1.3B, self-forcing distilled) .pt  ->  GGUF (f16), no torch.

KlingTeam/ShotStream ships `shotstream_merged.pt` = torch.save of {'generator': OrderedDict}
where the DiT lives under the `generator.model.` prefix, all F32, FLAT Wan naming
(patch_embedding / text_embedding / time_embedding / time_projection.1 / blocks.{i}.
self_attn|cross_attn|ffn|modulation|norm3 / head). 825 tensors, dim=1536, ffn=8960, 30 blocks.

A .pt is a zip: `data.pkl` (the pickled tree) + `data/{key}` raw little-endian storage blobs.
We stub the unpickler to capture (flat_name -> storage_key, dtype, offset, shape) WITHOUT torch,
then read each blob directly. Mirrors tools/convert_wan_dit.py for the write side
(strip prefix, squeeze 5-D patch_embedding, f16, arch="wan"); requantize later with
`sd-cli -M convert -m <f16.gguf> -o <q8.gguf> --type q8_0` if desired.

Usage:
    uv run --with numpy --with gguf python3 tools/convert_shotstream_pt.py \
        --src models/shotstream_merged.pt --out models/shotstream-1.3b-dit-f16.gguf
"""
import argparse, os, pickle, time, zipfile
import numpy as np

STORAGE_NP = {
    'FloatStorage': np.float32, 'HalfStorage': np.float16, 'DoubleStorage': np.float64,
    'LongStorage': np.int64, 'IntStorage': np.int32, 'ShortStorage': np.int16,
    'ByteStorage': np.uint8, 'CharStorage': np.int8, 'BoolStorage': np.bool_,
}
STRIP = 'generator.model.'


class Meta:
    __slots__ = ('key', 'npdt', 'off', 'shape', 'stride')
    def __init__(self, key, npdt, off, shape, stride):
        self.key, self.npdt, self.off = key, npdt, off
        self.shape, self.stride = tuple(shape), tuple(stride)


class FakeStorage:
    __slots__ = ('key', 'npdt', 'numel')
    def __init__(self, key, npdt, numel):
        self.key, self.npdt, self.numel = key, npdt, numel


def _rebuild_tensor_v2(storage, off, size, stride, *a):
    return Meta(storage.key, storage.npdt, off, size, stride)

def _rebuild_from_type_v2(func, new_type, args, state=None, *a):
    return func(*args)


class U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == 'torch._utils' and n == '_rebuild_tensor_v2': return _rebuild_tensor_v2
        if m == 'torch._tensor' and n == '_rebuild_from_type_v2': return _rebuild_from_type_v2
        if m == 'collections' and n == 'OrderedDict':
            import collections; return collections.OrderedDict
        if n.endswith('Storage'):
            c = type(n, (), {}); c._sname = n; return c
        return type(n, (), {'__init__': lambda s, *a, **k: None})

    def persistent_load(self, pid):
        # ('storage', storage_cls, key, location, numel)
        cls, key, numel = pid[1], pid[2], pid[4]
        sname = getattr(cls, '_sname', getattr(cls, '__name__', ''))
        return FakeStorage(str(key), STORAGE_NP.get(sname, np.float32), numel)


def walk(o, p=''):
    if isinstance(o, Meta):
        yield p, o; return
    if isinstance(o, dict):
        for k, v in o.items():
            yield from walk(v, f"{p}.{k}" if p else str(k))


def c_stride(shape):
    st, acc = [], 1
    for d in reversed(shape):
        st.append(acc); acc *= d
    return tuple(reversed(st))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max-blocks', type=int, default=-1, help='debug: only blocks [0,N)')
    args = ap.parse_args()
    import gguf

    z = zipfile.ZipFile(args.src)
    pkl = next(n for n in z.namelist() if n.endswith('data.pkl'))
    root = pkl[:-len('data.pkl')]  # e.g. 'shotstream_merged/'
    tree = U(z.open(pkl)).load()
    tensors = dict(walk(tree))
    print(f"read {len(tensors)} tensors from {os.path.basename(args.src)}")

    writer = gguf.GGUFWriter(args.out, arch='wan', use_temp_file=True)
    writer.add_description('KlingTeam ShotStream (Wan2.1-T2V-1.3B distilled) f32 -> f16 gguf')

    t0, n_t, total, skipped = time.time(), 0, 0, 0
    for name in sorted(tensors):
        m = tensors[name]
        flat = name[len(STRIP):] if name.startswith(STRIP) else name
        if args.max_blocks >= 0 and flat.startswith('blocks.'):
            if int(flat.split('.', 2)[1]) >= args.max_blocks:
                skipped += 1; continue
        nel = int(np.prod(m.shape)) if m.shape else 1
        if m.stride != c_stride(m.shape) or m.off != 0:
            # non-contiguous / offset view: gather via full storage then as_strided-equivalent
            raise SystemExit(f"non-contiguous tensor {flat}: off={m.off} stride={m.stride} "
                             f"shape={m.shape} (unexpected for a saved state_dict)")
        raw = z.read(f"{root}data/{m.key}")
        arr = np.frombuffer(raw, dtype=m.npdt)[m.off:m.off + nel].reshape(m.shape)
        if flat.endswith('patch_embedding.weight') and arr.ndim == 5:
            o, i, kt, kh, kw = arr.shape
            assert kt == 1, f"unexpected conv temporal dim: {arr.shape}"
            arr = arr.reshape(o * i, kt, kh, kw)
        writer.add_tensor(flat, np.ascontiguousarray(arr, dtype=np.float16))
        n_t += 1; total += arr.size

    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    z.close()
    sz = os.path.getsize(args.out) / 1e9
    print(f"wrote {args.out}: {n_t} tensors f16 ({skipped} skipped), "
          f"{total/1e9:.2f}B params, {sz:.2f}GB in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
