#!/usr/bin/env python3
"""
InfiniteTalk (MultiTalk on Wan2.1-I2V-14B) DiT -> GGUF merge converter.

Merges, in one pass, into a single f16 GGUF (flat names; the C++ loader prepends
`model.diffusion_model.`):
  1. the Wan2.1-I2V-14B base (7 diffusers shards: blocks.{i}.{self_attn,cross_attn
     incl. k_img/v_img/norm_k_img, ffn, modulation, norm3}; patch_embedding[5120,36,1,2,2];
     text_embedding/time_embedding/time_projection.1; img_emb.proj.{0,1,3,4}; head),
  2. the MeiGen InfiniteTalk graft `single/infinitetalk.safetensors` (330 FP32 tensors:
     blocks.{i}.audio_cross_attn.{q_linear,kv_linear,proj}.{weight,bias},
     blocks.{i}.norm_x.{weight,bias}, audio_proj.{proj1,proj1_vf,proj2,proj3,norm}.*),
  3. OPTIONAL lightx2v rank-64 step/cfg-distill LoRA, folded into the matching base
     linears (W += (up @ down) * alpha/rank) so the merged GGUF runs 4-step / cfg=1.

Base + graft tensor sets are disjoint; on any collision the graft wins (warns). The
patch_embedding Conv3d [5120,36,1,2,2] is squeezed to [5120,36,2,2] (ggml max-4D).
K-quants aren't in numpy gguf -> write f16, then sd-cli -M convert --type q4_K.

Usage:
  uv run --with numpy --with gguf python3 tools/convert_infinitetalk_dit.py \
      --base models/dl/wan21-i2v-14b-480p/diffusion_pytorch_model-0000{1..7}-of-00007.safetensors \
      --graft models/dl/infinitetalk/single/infinitetalk.safetensors \
      --lora models/dl/wan21-distill-lora/loras/Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors \
      --out models/infinitetalk-14b-f16.gguf
"""
import argparse, json, mmap, os, struct, time
import numpy as np

_ST_DT = {"F64": np.float64, "F32": np.float32, "F16": np.float16, "I64": np.int64,
          "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


def _bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Safetensors:
    def __init__(self, path: str):
        self.path = path
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.header.pop("__metadata__", None)
        self.data_start = 8 + n
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def keys(self):
        return self.header.keys()

    def get(self, name: str) -> np.ndarray:
        e = self.header[name]
        a, b = e["data_offsets"]
        buf = self.mm[self.data_start + a: self.data_start + b]
        if e["dtype"] == "BF16":
            arr = _bf16_to_f32(np.frombuffer(buf, dtype=np.uint16))
        else:
            arr = np.frombuffer(buf, dtype=_ST_DT[e["dtype"]])
        return arr.reshape(e["shape"])

    def close(self):
        self.mm.close()
        self.f.close()


def load_lora(path):
    """Parse a lightx2v/ComfyUI distill LoRA into per-base-tensor deltas. lightx2v ships
    THREE delta kinds (verified on the rank64 file) and NO .alpha (-> scale 1.0):
      <m>.lora_down.weight + <m>.lora_up.weight -> low-rank delta on <m>.weight
      <m>.diff                                  -> full weight delta on <m>.weight
      <m>.diff_b                                -> full bias delta   on <m>.bias
    Returns {base_tensor_name: [('lora',down,up,scale) | ('add',arr), ...]}."""
    st = Safetensors(path)
    keys = list(st.keys())

    def base_of(stem):
        for pfx in ("diffusion_model.", "transformer."):
            if stem.startswith(pfx):
                return stem[len(pfx):]
        return stem

    deltas = {}
    for k in keys:
        if k.endswith(".lora_down.weight"):
            up_k = k.replace("lora_down.weight", "lora_up.weight")
            if up_k not in st.header:
                print(f"  LoRA WARN: no up for {k}"); continue
            tgt = base_of(k[:-len(".lora_down.weight")]) + ".weight"
            deltas.setdefault(tgt, []).append(
                ("lora", st.get(k).astype(np.float32), st.get(up_k).astype(np.float32), 1.0))
        elif k.endswith(".diff"):
            tgt = base_of(k[:-len(".diff")]) + ".weight"
            deltas.setdefault(tgt, []).append(("add", st.get(k).astype(np.float32)))
        elif k.endswith(".diff_b"):
            tgt = base_of(k[:-len(".diff_b")]) + ".bias"
            deltas.setdefault(tgt, []).append(("add", st.get(k).astype(np.float32)))
    st.close()
    return deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", nargs="+", required=True, help="Wan2.1-I2V-14B diffusers shards")
    ap.add_argument("--graft", default="", help="MeiGen single/infinitetalk.safetensors (omit for a "
                    "graft-free base Wan2.1-I2V build — sd-cli then detects it as Wan2.1-I2V-14B)")
    ap.add_argument("--lora", default="", help="optional lightx2v rank-64 distill LoRA to fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    import gguf

    # collect every tensor: base shards first, graft overrides on collision.
    srcs = {}  # name -> (Safetensors, is_graft)
    shards = [Safetensors(p) for p in args.base]
    for sh in shards:
        for k in sh.keys():
            srcs[k] = (sh, False)
    graft = None
    if args.graft:
        graft = Safetensors(args.graft)
        for k in graft.keys():
            if k in srcs:
                print(f"  collision (graft wins): {k}")
            srcs[k] = (graft, True)
    else:
        print("  NO graft (base Wan2.1-I2V only)")

    lora = load_lora(args.lora) if args.lora else {}
    total_deltas = sum(len(v) for v in lora.values())
    if args.lora:
        print(f"  loaded {total_deltas} LoRA deltas ({len(lora)} targets) from {os.path.basename(args.lora)}")

    writer = gguf.GGUFWriter(args.out, arch="wan", use_temp_file=True)
    writer.add_description("InfiniteTalk (MultiTalk / Wan2.1-I2V-14B) merged (bf16+graft+lora -> f16)")

    t0 = time.time()
    n_t = 0
    total = 0
    n_lora_applied = 0
    for name in sorted(srcs.keys()):
        sh, _ = srcs[name]
        arr = sh.get(name).astype(np.float32)
        # fold LoRA/diff deltas into the matching base tensor if present
        if name in lora:
            for d in lora[name]:
                if d[0] == "lora":
                    _, down, up, scale = d
                    if arr.ndim == 2 and arr.shape == (up.shape[0], down.shape[1]):
                        arr = arr + (up @ down) * scale
                        n_lora_applied += 1
                    else:
                        print(f"  LoRA shape mismatch, skip {name}: W{arr.shape} up{up.shape} down{down.shape}")
                else:  # 'add' = diff (weight) or diff_b (bias) full delta
                    if d[1].shape == arr.shape:
                        arr = arr + d[1]
                        n_lora_applied += 1
                    else:
                        print(f"  LoRA diff shape mismatch, skip {name}: {arr.shape} vs {d[1].shape}")
        # video Conv3d patch embed [out,in,kt=1,kh,kw]. The C++ Conv3d stores its weight
        # as ggml ne [kw,kh,kt, in*out] (ggml_ext_conv_3d: OC=ne[3]/IC, IC-fastest/OC-slowest),
        # i.e. numpy [out*in, kt, kh, kw] with the merged dim out-major. Merge out&in, KEEP kt.
        if name.endswith("patch_embedding.weight") and arr.ndim == 5:
            o, i, kt, kh, kw = arr.shape
            assert kt == 1, f"unexpected conv temporal dim: {arr.shape}"
            arr = arr.reshape(o * i, kt, kh, kw)
        writer.add_tensor(args.prefix + name, np.ascontiguousarray(arr, dtype=np.float16))
        n_t += 1
        total += arr.size

    if args.lora and n_lora_applied != total_deltas:
        print(f"  LoRA WARN: applied {n_lora_applied}/{total_deltas} deltas (unmatched keys -> NOT folded)")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    for sh in shards:
        sh.close()
    if graft is not None:
        graft.close()

    sz = os.path.getsize(args.out) / 1e9
    print(f"wrote {args.out}: {n_t} tensors f16 ({n_lora_applied} lora-folded), "
          f"{total/1e9:.2f}B params, {sz:.2f}GB in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
