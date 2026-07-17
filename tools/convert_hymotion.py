#!/usr/bin/env python3
"""
HY-Motion 1.0 (Tencent) text->humanoid-motion MMDiT  ->  GGUF converter (for hymotion.hpp / ggml).

Source is a `torch.save` checkpoint (`latest.ckpt`, a zip of a pickle + raw storages),
**all tensors F32**: 428 tensors / 1,042,878,811 params (1.0429B) for HY-Motion-1.0,
matching the advertised "1.0B". The Lite variant is 0.461B.

  https://huggingface.co/tencent/HY-Motion-1.0    (Tencent Hunyuan Community licence)

NO TORCH REQUIRED. This box has no torch and a torch install is ~3GB; the ckpt is a
plain zip and we unpickle it with a stub Unpickler that only needs the tensor
metadata + raw storage bytes. That also keeps the converter honest: it is a *dumb
copy*, not a re-interpretation.

TENSOR NAMES ARE KEPT VERBATIM (minus the `motion_transformer.` prefix). Deliberate:
a renaming converter is a silent-failure surface, and the C++ side (hymotion.hpp)
declares blocks under the HY names so the mapping is checkable by eye. The
pipeline-level parameters (null/game feats, mean, std) live at the ckpt top level and
are emitted under a `pipe.` prefix.

What the C++ side needs to know is written into the GGUF KV metadata, so hymotion.hpp
never hardcodes a hyperparameter that could drift from the ckpt.

Quant rule (mirrors convert_nava_dit.py / convert_longcat_avatar.py): 2-D weight
matrices whose last dim is a multiple of 32 -> requested big-quant. Everything else
(norms, biases, modulation, mean/std, null feats) -> F32/F16. Note the model is only
1.04B; at F16 it is ~2.1GB, which already fits the 3060 with room to spare, so F16 is
the default and quantising is not expected to be necessary.

Usage:
    uv run --with numpy --with gguf python3 tools/convert_hymotion.py \
        --src /mnt/hdd/3d/avatar-shootout/_weights/hymotion/HY-Motion-1.0/HY-Motion-1.0/latest.ckpt \
        --out models/hymotion-1.0-f16.gguf --dtype f16

    # Lite (0.46B). NOTE: Lite seeds its RNG on the GPU (random_generator_on_gpu:
    # true) so it is NOT reproducible across machines -- use the 1.0B for goldens.
    uv run --with numpy --with gguf python3 tools/convert_hymotion.py \
        --src .../HY-Motion-1.0-Lite/latest.ckpt --out models/hymotion-lite-f16.gguf

    # audit only, writes nothing:
    python3 tools/convert_hymotion.py --src .../latest.ckpt --list
"""
import argparse
import collections
import io
import os
import pickle
import sys
import zipfile

import numpy as np

# ---------------------------------------------------------------------------
# torch-free .ckpt reader
# ---------------------------------------------------------------------------

_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": "bfloat16",
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "BoolStorage": np.bool_,
}


class _Storage:
    __slots__ = ("key", "dtype_name", "numel")

    def __init__(self, key, dtype_name, numel):
        self.key, self.dtype_name, self.numel = key, dtype_name, numel


class LazyTensor:
    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset = storage, offset
        self.shape, self.stride = tuple(size), tuple(stride)

    @property
    def dtype(self):
        return self.storage.dtype_name

    @property
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *a, **k):
    return LazyTensor(storage, storage_offset, size, stride)


def _rebuild_parameter(data, *a, **k):
    return data


class _Stub:
    def __init__(self, module, name):
        self.__module__, self.__name__ = module, name

    def __call__(self, *a, **k):
        return None


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch._utils":
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_parameter":
                return _rebuild_parameter
        if module == "collections" and name == "OrderedDict":
            # must be the real OrderedDict: a plain dict has no __dict__ and the
            # pickle BUILD opcode then fails.
            return collections.OrderedDict
        return _Stub(module, name)

    def persistent_load(self, pid):
        assert pid[0] == "storage", pid
        storage_type, key, _location, numel = pid[1], pid[2], pid[3], pid[4]
        return _Storage(str(key), getattr(storage_type, "__name__", str(storage_type)), numel)


class CkptReader:
    def __init__(self, path):
        self.zf = zipfile.ZipFile(path)
        pkl = [n for n in self.zf.namelist() if n.endswith("data.pkl")]
        assert len(pkl) == 1, f"expected exactly one data.pkl, got {pkl}"
        self.prefix = pkl[0][: -len("data.pkl")]
        with self.zf.open(pkl[0]) as f:
            self.obj = _Unpickler(io.BytesIO(f.read())).load()

    def to_numpy(self, t):
        with self.zf.open(f"{self.prefix}data/{t.storage.key}") as f:
            raw = f.read()
        dn = t.storage.dtype_name
        if dn not in _DTYPES:
            raise NotImplementedError(f"unhandled storage dtype {dn}")
        npdt = _DTYPES[dn]
        if npdt == "bfloat16":
            arr = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(raw, dtype=npdt)
        arr = arr[t.offset : t.offset + t.numel]
        exp, acc = [], 1
        for s in reversed(t.shape):
            exp.append(acc)
            acc *= s
        if t.stride != tuple(reversed(exp)):
            raise NotImplementedError(f"non-contiguous tensor: stride {t.stride} shape {t.shape}")
        return arr.reshape(t.shape)


def walk(obj, prefix=""):
    if isinstance(obj, LazyTensor):
        yield prefix, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}.{i}" if prefix else str(i))


# ---------------------------------------------------------------------------
# name mapping
# ---------------------------------------------------------------------------

MT = "motion_transformer."
PIPE_KEYS = {
    "null_vtxt_feat",
    "null_ctxt_input",
    "special_game_vtxt_feat",
    "special_game_ctxt_feat",
    "mean",
    "std",
}


def out_name(src):
    """ckpt state_dict key -> gguf tensor name."""
    if src.startswith(MT):
        return src[len(MT) :]
    if src in PIPE_KEYS:
        return "pipe." + src
    return None  # dropped (optimizer state etc.)


# ---------------------------------------------------------------------------


def derive_arch(names, shapes):
    """Recover hyperparameters from the weights themselves rather than trusting a
    config.yml that may not travel with the ckpt. Cross-checked against config.yml
    by the caller."""
    a = {}
    a["input_dim"] = shapes["input_encoder.weight"][1]          # 201
    a["feat_dim"] = shapes["input_encoder.weight"][0]           # 1280
    a["ctxt_input_dim"] = shapes["ctxt_encoder.weight"][1]      # 4096
    a["vtxt_input_dim"] = shapes["vtxt_encoder.linears.0.weight"][1]  # 768
    a["output_dim"] = shapes["final_layer.linear.weight"][0]    # 201
    nd = len({n.split(".")[1] for n in names if n.startswith("double_blocks.")})
    ns = len({n.split(".")[1] for n in names if n.startswith("single_blocks.")})
    a["double_blocks"] = nd
    a["single_blocks"] = ns
    a["num_layers"] = nd + ns
    # mlp_hidden = feat_dim * mlp_ratio
    a["mlp_hidden"] = shapes["double_blocks.0.motion_mlp.fc1.weight"][0]
    # head_dim from the qk-norm weight (RMSNorm over head_dim)
    a["head_dim"] = shapes["double_blocks.0.motion_q_norm.weight"][0]
    a["num_heads"] = a["feat_dim"] // a["head_dim"]
    nr = len({n.split(".")[3] for n in names if n.startswith("text_refiner.individual_token_refiner.blocks.")})
    a["refiner_blocks"] = nr
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="path to latest.ckpt")
    ap.add_argument("--out", help="output .gguf")
    ap.add_argument("--dtype", default="f16", choices=["f32", "f16", "q8_0"], help="big-matrix dtype")
    ap.add_argument("--list", action="store_true", help="audit the ckpt and exit, writing nothing")
    args = ap.parse_args()

    print(f"[hymotion] reading {args.src}")
    r = CkptReader(args.src)
    root = r.obj
    if isinstance(root, dict) and "model_state_dict" in root:
        sd = root["model_state_dict"]
        print(f"[hymotion] ckpt epoch={root.get('epoch')} global_step={root.get('global_step')}")
    else:
        sd = root

    items = [(n, t) for n, t in walk(sd)]
    total = sum(t.numel for _, t in items)
    print(f"[hymotion] {len(items)} tensors, {total:,} params ({total/1e9:.4f}B), dtypes={ {t.dtype for _,t in items} }")

    mapped, dropped = [], []
    for n, t in items:
        o = out_name(n)
        (mapped if o else dropped).append((o or n, t))
    if dropped:
        print(f"[hymotion] WARNING: dropping {len(dropped)} unmapped tensors: {[d[0] for d in dropped][:8]}")

    names = [n for n, _ in mapped]
    shapes = {n: t.shape for n, t in mapped}
    arch = derive_arch(names, shapes)
    print("[hymotion] arch derived from weights:")
    for k, v in arch.items():
        print(f"             {k:16s} = {v}")

    # Structural assertions. These are the checks that catch a wrong ckpt or a
    # silently-changed upstream, and they cost nothing.
    assert arch["input_dim"] == 201, f"expected input_dim 201 (3 transl + 6 root_rot6d + 21*6 body_rot6d + 22*3 pos), got {arch['input_dim']}"
    assert arch["output_dim"] == arch["input_dim"], "final_layer must map back to input_dim"
    assert arch["num_layers"] % 3 == 0, "num_layers must be divisible by 3"
    assert arch["double_blocks"] == arch["num_layers"] // 3, "double_blocks must be num_layers//3"
    assert arch["single_blocks"] == arch["num_layers"] - arch["num_layers"] // 3
    assert arch["feat_dim"] % arch["num_heads"] == 0
    assert arch["mlp_hidden"] == arch["feat_dim"] * 4, "mlp_ratio is expected to be 4.0"
    assert arch["ctxt_input_dim"] == 4096, "Qwen3-8B hidden size"
    assert arch["vtxt_input_dim"] == 768, "CLIP-L pooled width"
    assert arch["refiner_blocks"] == 2, "SingleTokenRefiner is 2 blocks"

    if args.list:
        for n, t in sorted(mapped):
            print(f"  {n:70s} {tuple(t.shape)}")
        return

    if not args.out:
        ap.error("--out is required unless --list")

    import gguf
    from gguf import GGMLQuantizationType as QT
    from gguf import quants

    w = gguf.GGUFWriter(args.out, "hymotion")

    # ---- KV metadata: everything the C++ side needs, taken from the weights ----
    w.add_string("hymotion.source", os.path.basename(os.path.dirname(os.path.abspath(args.src))))
    w.add_uint32("hymotion.input_dim", arch["input_dim"])
    w.add_uint32("hymotion.feat_dim", arch["feat_dim"])
    w.add_uint32("hymotion.ctxt_input_dim", arch["ctxt_input_dim"])
    w.add_uint32("hymotion.vtxt_input_dim", arch["vtxt_input_dim"])
    w.add_uint32("hymotion.num_layers", arch["num_layers"])
    w.add_uint32("hymotion.double_blocks", arch["double_blocks"])
    w.add_uint32("hymotion.single_blocks", arch["single_blocks"])
    w.add_uint32("hymotion.num_heads", arch["num_heads"])
    w.add_uint32("hymotion.head_dim", arch["head_dim"])
    w.add_uint32("hymotion.refiner_blocks", arch["refiner_blocks"])
    w.add_float32("hymotion.mlp_ratio", arch["mlp_hidden"] / arch["feat_dim"])
    # --- from config.yml; these are NOT recoverable from the weights ---
    w.add_float32("hymotion.time_factor", 1000.0)       # config.yml: time_factor
    w.add_uint32("hymotion.train_frames", 360)          # config.yml: train_frames (12s @ 30fps)
    w.add_uint32("hymotion.fps", 30)                    # config.yml: output_mesh_fps
    w.add_float32("hymotion.cfg_scale", 5.0)            # config.yml: test_cfg.text_guidance_scale
    w.add_uint32("hymotion.steps", 50)                  # config.yml: infer_noise_scheduler_cfg.validation_steps
    w.add_uint32("hymotion.max_length_llm", 128)        # config.yml: text_encoder_cfg.max_length_llm
    w.add_string("hymotion.mask_mode", "narrowband")    # config.yml: mask_mode
    # narrowband_length is NOT in config.yml -- it is the code default 2.0 (seconds),
    # scaled by 30 in HunyuanMotionMMDiT.__init__ -> a +/-60 frame attention band.
    w.add_uint32("hymotion.narrowband_window", 60)
    # apply_rope_to_single_branch: false  ->  RoPE is applied to the FULL concatenated
    # [motion, text] stream, NOT to the motion branch alone. The flag name reads
    # backwards; see hymotion.hpp.
    w.add_bool("hymotion.rope_on_concat", True)
    w.add_float32("hymotion.rope_theta", 10000.0)
    w.add_bool("hymotion.enable_ctxt_null_feat", True)  # config.yml
    w.add_uint32("hymotion.num_joints", 22)             # SMPL-H body, no hands
    w.add_string("hymotion.rot6d_layout", "interleaved_columns")

    big = {"f32": QT.F32, "f16": QT.F16, "q8_0": QT.Q8_0}[args.dtype]

    n_big = 0
    for name, t in sorted(mapped):
        arr = r.to_numpy(t).astype(np.float32)
        # squeeze the (1,1,D) pipeline feats to (D,)
        if name.startswith("pipe.") and arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 1:
            arr = arr.reshape(-1)
        # Quant rule: only 2-D weight matrices with last dim %32==0. Norms, biases,
        # modulation linears, mean/std and the null feats stay F32/F16.
        is_big = arr.ndim == 2 and arr.shape[-1] % 32 == 0 and name.endswith(".weight")
        if name.startswith("pipe.") or name.endswith("mean") or name.endswith("std"):
            is_big = False
        if is_big and big != QT.F32:
            if big == QT.Q8_0:
                data = quants.quantize(arr, QT.Q8_0)
                w.add_tensor(name, data, raw_shape=arr.shape, raw_dtype=QT.Q8_0)
            else:
                w.add_tensor(name, arr.astype(np.float16))
            n_big += 1
        else:
            # keep the small/sensitive tensors in F32
            w.add_tensor(name, arr)

    print(f"[hymotion] writing {args.out}  ({len(mapped)} tensors, {n_big} as {args.dtype})")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    sz = os.path.getsize(args.out)
    print(f"[hymotion] done: {args.out}  {sz/1e6:.1f} MB")


if __name__ == "__main__":
    main()
