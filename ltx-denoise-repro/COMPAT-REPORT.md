# LTX-2.3 LoRA compatibility report — distill-384 negative fold + detailer

CPU-only header inspection (no tensor data read, no GPU). Base = `models/ltx2/nvfp4-CLEAN.gguf`
(== LTX-2.3-22b **dev + distill-lora@1.0** in nvfp4; 4444 tensors = 2694 F32 + 406 BF16 +
**1344 nvfp4** type-40 Linears; 48 transformer blocks, hidden 4096, ffn 16384).

---

## 1. Distill LoRA — `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (7.61 GB)

Path: `models/ltx2/loras/_hf_ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors`

| property | value |
|---|---|
| tensor count | 3320 (= **1660 lora A/B pairs**) |
| dtype | BF16 (all tensors) |
| metadata | `model_version=2.3.0`, `lora_rank=384`, `lora_alpha=384`, LTX-2 Community License |
| naming convention | `diffusion_model.<module>.lora_A.weight` / `.lora_B.weight` (PEFT-style; **identical to the lipdub IC-LoRA**) |
| orientation | `A = [rank, in]`, `B = [out, rank]`, `delta[out,in] = B @ A` (verified vs gguf dims) |
| **rank (per target, mixed)** | **384** ×1360, **256** ×8, **128** ×4, **32** ×288 — rank is capped at each layer's `in` dim, so it is **NOT a single constant**. The script reads rank from `A.shape[0]` per tensor. |

**Targets vs nvfp4-CLEAN Linear set:**

| bucket | count | note |
|---|---|---|
| lora targets hitting an **nvfp4** (type-40) Linear | **1344** | attn1 (self qkv+out), attn2 (cross qkv+out), ff.net.0.proj, ff.net.2 across all 48 blocks + patchify/proj_out trunk Linears |
| lora targets hitting a **non-nvfp4** tensor (BF16/F32) | 316 | adaln_single / timestep_embedder / audio_* / av_ca_* modulation + audio-path layers |
| lora targets mapping to **no** gguf tensor | 0 | |
| nvfp4 Linears with **no** lora target | **0** | |

**Bijection verdict on the fold set: YES — clean.** Every one of the 1344 nvfp4 Linears in CLEAN has
exactly one matching distill-LoRA target, and no nvfp4 Linear is left unmapped. The rank-384 targets
are exactly the 1344 nvfp4 Linears (rank histogram over the *folded* set = `{384: 1344}`); the
rank-32/128/256 targets are all on the skipped non-nvfp4 modulation/audio layers.

**Coverage limitation (by design, matches the lipdub precedent):** the 316 non-nvfp4 targets
(adaln modulation, timestep embedders, the audio branch) are **not** folded — those base tensors are
copied verbatim, so their share of the distillation delta stays baked in. Only the load-bearing
attn/ffn video Linears (where the face mush lives) are de-distilled. The eye-test gates whether the
residual modulation distillation matters; if it does, those bf16/f32 layers would need a separate fold.

### Negative-fold math (the cheap win)
`CLEAN == dev + distill@1.0`  ⇒  `dev + distill@s == distilled-1.1 + distill-lora@(s−1.0)`.
So `fold_distill_lora.py --strength -0.35` yields **effective distill 0.65** (the Denoise-AI S2 base
stage) with no 46 GB dev download and no whole-model re-quant — same nvfp4 format / param count /
kernels ⇒ same VRAM + per-step speed as prod. `--strength -1.0` should recover ~pure dev
(assumption still to verify on GPU: that `distilled-1.1 == dev + this-lora@1.0`).

---

## 2. Detailer LoRA — `ltx-2-19b-ic-lora-detailer.safetensors`

**Located at:** `https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer` (official Lightricks
repo; the only file is `ltx-2-19b-ic-lora-detailer.safetensors`). Header fetched via HTTP Range
request (first 8 bytes = header len 156216, then that many bytes of JSON — no tensor data pulled).
Note: it is published **only** as an LTX-2 **19B** IC-LoRA; there is no `LTX-2.3-22b` detailer in the
Lightricks IC-LoRA family (which does have LipDub, Union/Motion/Depth/Canny/Pose control, HDR,
Colorization, Deblur, etc.).

| property | value |
|---|---|
| tensor count | 960 (= **480 lora A/B pairs**) |
| dtype | BF16 |
| rank | **256** (uniform) |
| naming | `diffusion_model.transformer_blocks.<i>.<module>.lora_{A,B}.weight` (same PEFT convention) |
| transformer blocks | **48** (idx 0–47) |
| targets | attn1.{to_q,to_k,to_v,to_out.0}, attn2.{to_q,to_k,to_v,to_out.0}, ff.net.0.proj, ff.net.2 — **10 modules × 48 blocks** (video transformer trunk only; no audio/adaln targets) |

**Dimensional check against 22B LTX-2.3 (nvfp4-CLEAN):**

| bucket | count |
|---|---|
| detailer targets mapping to a CLEAN nvfp4 Linear | **480 / 480** |
| **dim mismatches** (A.in≠gguf.in or B.out≠gguf.out) | **0** |
| targets with no gguf match | 0 |

Every detailer target lands on an nvfp4 Linear with in=4096 / out∈{4096,16384} exactly matching our
22B base. Despite the "19b" label, the LTX-2 19B and LTX-2.3 22B share the **same video DiT trunk
geometry** (48 blocks × 4096 hidden × 16384 ffn × 4096 cross-attn context); the 19B↔22B parameter
delta lives outside the trunk (audio branch / added components) — which the detailer does not touch.

### VERDICT: **COMPATIBLE** (dimensionally) with our 22B LTX-2.3 base.
It can be folded by the same `fold_distill_lora.py` (`--lora <detailer> --strength 0.7/0.8`) or applied
at runtime as `<lora:ltx-2-19b-ic-lora-detailer:0.8>`; the 480 attn/ffn targets are a strict subset of
the 1344 nvfp4 Linears with matching shapes and rank-256 A/B. Caveat: dimensional compatibility ≠
numerical/semantic quality — whether the 19B-trained detailer delta actually improves 22B output is a
GPU eye-test question, and the Denoise-AI workflow only enables it on the **hires/upscale** stages
(S3/S4), never on the base stages. It should NOT be folded into the base model; keep it a runtime
hires-only LoRA.

---

## 3. Dry-run coverage output — `fold_distill_lora.py`

Command (run from `/home/dbrain/dev/longcat-avatar.cpp`):

```
$ python3 tools/fold_distill_lora.py --strength -0.35 --dry-run
========================================================================
base   : models/ltx2/nvfp4-CLEAN.gguf
lora   : models/ltx2/loras/_hf_ltx23/ltx-2.3-22b-distilled-lora-384-1.1.safetensors
prefix : diffusion_model.
strength requested : -0.35   (delta = strength * B@A)
  -> effective distill fraction ~= 0.650  (CLEAN==distill@1.0)
------------------------------------------------------------------------
gguf tensors total            : 4444
gguf nvfp4 (type 40) Linears  : 1344
lora targets (A/B pairs)      : 1660
nvfp4 Linears FOLDED          : 1344
nvfp4 Linears with NO lora    : 0
lora targets on NON-nvfp4     : 316  (bf16/f32, copied verbatim, NOT de-distilled)
lora targets on NO gguf tensor: 0
nvfp4 fold is a clean bijection: True
  -- non-nvfp4 lora targets skipped (sample):
        adaln_single.emb.timestep_embedder.linear_1.weight  (ggufType 0)
        adaln_single.emb.timestep_embedder.linear_2.weight  (ggufType 0)
        adaln_single.linear.weight  (ggufType 0)
        audio_adaln_single.emb.timestep_embedder.linear_1.weight  (ggufType 0)
        ... +304 more
========================================================================
rank histogram over folded set: {384: 1344}
shape validation errors: 0
[dry-run] mapping valid. Would fold 1344 nvfp4 Linears at strength=-0.35. No output written.
```

To actually produce the S2-base de-distilled model (heavy; not run here):

```
python3 tools/fold_distill_lora.py --strength -0.35 \
    --out models/ltx2/nvfp4-CLEAN-distill065.gguf
```
