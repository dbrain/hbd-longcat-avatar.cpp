# SCOPE — HY-Motion 1.0 native port + integration

**Date:** 2026-07-17 · **Status:** SCOPING ONLY — nothing built, nothing wired, no weights downloaded, GPU untouched.
**Rule honoured:** not wired into `image_to_rig`. Design only.

Every claim tagged **[VERIFIED]** (I read the primary source), **[INFERRED]** (arithmetic/reasoning from
verified facts), **[UNVERIFIED]** (nobody has measured it — including me).

Evidence root: `/tmp/claude-1000/-home-dbrain-dev-kobbler/5ca9a63c-1620-4a12-b7a3-c4e90bf8c2d1/scratchpad/hymotion/repo`
(shallow clone of `https://github.com/Tencent-Hunyuan/HY-Motion-1.0`, code only, 40MB, no weights).

---

## 0. CORRECTIONS TO THE BRIEF — read this first

The brief asked me to say loudly where it was wrong. It was wrong in five places. **The decisive claim,
however, is correct** — verified twice, independently.

| # | Brief said | Reality | Severity |
|---|---|---|---|
| 1 | "**rot6d LOCAL ROTATIONS + root translation**" | ✅ **CORRECT — VERIFIED TWICE** (paper + code). See §1. The whole scope stands. | — |
| 2 | "SMPL-X? SMPL-H? HumanML3D-22?" — open question | **SMPL-H, 22 joints, explicitly "without hands"** [VERIFIED]. Not SMPL-X. | resolved |
| 3 | "A **Q4 GGUF of that encoder already exists next to the ckpts**" | ❌ **FALSE.** The HF repo contains *only* 2 ckpts + assets. No encoder, no GGUF. [VERIFIED — full file listing, §4]. What's true: `Qwen/Qwen3-8B-GGUF` exists as a **separate official Qwen repo**. And `load_in_4bit=True` appears in the repo — but on the *optional prompt rewriter*, a different model. Two facts got fused into one wrong one. | **substance survives, location wrong** |
| 4 | "the 26GB README figure is **fp32** TEXT ENCODERS" | ❌ **Wrong label, right number.** Code loads Qwen3-8B at `torch_dtype=torch.bfloat16` [VERIFIED, `text_encoder.py:104`]. 16.4GB *is* bf16 (8.2B × 2B). fp32 would be 32.8GB. The *ckpts* are fp32 (verified by arithmetic, §2). | cosmetic — conclusion unaffected |
| 5 | "SSAE … judged by **Tencent's VLM**" | ❌ **FALSE.** Judged by **Google Gemini** — `gemini-3-pro-preview` in the repo script, "Gemini-2.5-Pro" in the paper [VERIFIED]. Third-party judge, and the eval prompts + script are **open in the repo** (`ssae/`). This is *less* circular than the brief assumed — it's reproducible by anyone. Still Tencent's metric design on Tencent's prompt set, so don't treat 78.6 as gospel — but the `cos 0.9998` analogy is unfair to it. | **upgrade in credibility** |
| 6 | "limitations concede trouble with 'highly detailed or complex instructions'" | ✅ Correct — it's in the **paper**, not the README [VERIFIED]. But the README carries a *sharper* limitation the brief missed, and it lands directly on the owner's own example. See §6. | brief under-sold the risk |

**One more the brief didn't anticipate:** the reference python **will not run on a 12GB 3060 as shipped**
(needs ~20.6GB resident). Even the "cheap PoC" needs a small patch. §7.

---

## 1. THE DECISIVE PROPERTY — VERIFIED TWICE ✅

The brief's central bet is **confirmed**, from two independent sources that agree exactly.

**Source A — the paper** (`arxiv.org/abs/2512.23464`), verbatim [VERIFIED]:

> "each frame is represented as a vector **𝒇∈ℝ²⁰¹**, comprising the global root translation **𝒕∈ℝ³**,
> the global body orientation **𝒓∈ℝ⁶**, the local joint rotations **𝒋ᵣ∈ℝ²¹ˣ⁶**, and the local joint
> positions **𝒋ₚ∈ℝ²²ˣ³**"
> "All rotational parameters adhere to the **continuous 6D representation**"
> "we employ the skeleton definition of **SMPL-H [Romero2017] (22 joints without hands)**"

**Source B — the code** (`hymotion/pipeline/motion_diffusion.py:219-254`, `_decode_o6dp`) [VERIFIED].
I decoded the 201-dim layout from the slicing arithmetic *before* reading the paper. They match exactly:

```
latent[..., 0:3]     → transl            root translation            3
latent[..., 3:9]     → root_rot6d        global body orientation     6   (rot6d)
latent[..., 9:135]   → body_rot6d        21 local joint rotations  126   (21 × 6, rot6d)
latent[..., 135:201] → (IGNORED by decoder)  22 joint positions     66   (22 × 3)
                                                              total = 201  ✓
```

`config.yml: input_dim: 201` [VERIFIED]. `3 + 6 + 126 + 66 = 201` ✓.

**So: rot6d local rotations + root translation. Confirmed. The scope rests on solid ground.**

Two bonus findings the brief didn't have:

- **The positions are carried but the decoder throws them away** [VERIFIED — `_decode_o6dp` slices
  `0:3`, `3:9`, `9:135` and never touches `135:201`]. They're auxiliary supervision. But **they're right
  there in the output tensor**, which matters enormously for the PoC (§7) — it means we can feed our
  *existing* position-based retargeter with zero new code.
- **Rotations recover TWIST; positions cannot.** This is the concrete, non-hand-wavy quality upgrade,
  and it's stronger than the brief argued. Our `retarget_delta.py` computes
  `D_s(t) = quat_between(u_s, v_s(t))` from two *directions* [VERIFIED — `retarget_delta.py:37-40`].
  `quat_between` of two vectors has **no roll component by construction** — it's the minimal-arc
  rotation. Forearm pronation, head roll, spine twist are **mathematically unrepresentable** from
  positions alone. HY-Motion hands us those DOF directly. [INFERRED, but it's geometry, not opinion.]

**Why our retargeter's head pointed down** — brief's causal story is corroborated. `retarget_delta.py:80-81`
verbatim [VERIFIED]:
> "The .npy motions are **JOINT POSITIONS ONLY (T,22,3)** -- MoMask/HumanML3D output, **no rotations
> anywhere**. So 'SMPL's rest orientation per joint' is not in the data"

That is the upstream cause, in our own source comments, in our own words.

### 1a. ⚠️ HANDS — the answer is NO, definitively

**HY-Motion 1.0 does not generate hands.** Triple-confirmed:
1. Paper: "22 joints **without hands**" [VERIFIED]
2. Arithmetic: 201 forces 22. A 52-joint variant would need `3 + 6 + 51×6 + 52×3 = 471` [INFERRED]
3. Code: `decode_motion_from_latent` **hardcodes `num_joints=22`** at the call site
   (`motion_diffusion.py:214`) [VERIFIED]

And the smoking gun — `body_model.py:279-283` [VERIFIED]:
```python
if rot_mats.shape[1] == 22:
    ...
    rot_mats = torch.cat([rot_mats, eye], dim=1)   # (B, 22 + 30, 3, 3)
```
**It pads 30 identity rotations for the fingers.** Tencent's own renderer produces dead hands. The
brief's feared failure mode is not a risk — it is the shipped behaviour.

**Curiosity worth logging:** there *is* a `nj == 52` branch with hand-splitting logic
(`motion_diffusion.py:243-246`), plus a 52-joint `joint_names.json` and an FBX asset literally named
`boy_Rigging_**smplx**_tex.fbx` [VERIFIED]. All **dead code for this checkpoint** — `mean.shape` is
asserted `(1, 201)` (`motion_diffusion.py:165`), which forces 22. Strong hint an internal hand-capable
variant exists. **Do not bank on a 1.1 release.** [UNVERIFIED — speculation.]

### 1b. My view on the hands question (the brief asked for one)

**Freeze the finger bones. Do not strip them. Do not chase a hand-capable model.** Five reasons:

1. **The geometry can't articulate anyway.** The owner's mesh has a thumb + mitten — 2 lobes, no digits.
   Skinning weights on a mitten will *smear*, not flex. Animated fingers inside blob geometry is exactly
   the "weird things" he predicted. The model refusing to move them is the *correct* result.
2. **Frozen fingers on a mitten are invisible.** A mitten at bind pose looks like a mitten. That's the
   target look. The failure mode is nil.
3. **Stripping is irreversible; freezing is free.** A better mesh later, or a hand-capable model later,
   would want those bones back. Cost of keeping: ~30 identity quats per frame in a JSON. Nothing.
4. **The rig earned its fingers** and SkinTokens' J is emergent — stripping means hand-editing an
   emergent, per-mesh-varying output (28/29/33/38/61). That's a new failure surface for zero gain.
5. **No model in this class does convincing fingers.** Swapping models to buy hands would trade a
   verified-good body for unverified-mediocre hands. Bad trade.

**One-line requirement this creates:** the retargeter must *explicitly write rest-pose quats* for
unmapped bones, not leave them undefined. Needs a 1-line check against `retarget_delta.py`. [UNVERIFIED
— I did not confirm current behaviour for unmapped bones.]

---

## 2. WHAT IT IS — architecture [VERIFIED from config.yml + source]

**Flow-matching MMDiT.** Not diffusion, not AR, **no VQ, no VAE, no motion tokenizer** — continuous flow
matching directly on the raw 201-dim feature. That's a large simplification vs MoMask.

| | HY-Motion-1.0 | Lite |
|---|---|---|
| ckpt bytes | 4171.70 MB | 1843.54 MB |
| **params (÷4 ⇒ fp32)** | **1.043 B** ✓ "1.0B" | **0.461 B** ✓ "0.46B" |
| `feat_dim` | 1280 | 1024 |
| `num_layers` | 27 | 18 |
| → double-stream | **9** (`27//3`) | 6 |
| → single-stream | **18** (`27−9`) | 12 |
| `num_heads` | 20 (head_dim 64) | 16 (head_dim 64) |
| `mlp_ratio` | 4.0 | 4.0 |
| RNG | **CPU** (`random_generator_on_gpu: false`) | **GPU** ← non-reproducible |

The param arithmetic cross-checking the advertised counts to 3 significant figures is a good sign the
ckpts are plain fp32 with no surprises. [INFERRED, but tight.]

**🔑 It is the Flux / HunyuanVideo family.** `MMDoubleStreamBlock` + `MMSingleStreamBlock`, `ModulateDiT`,
`apply_gate`, `token_refiner`, RMSNorm, RoPE, qk-norm [VERIFIED — `hymotion_mmdit.py`]. The conditioning
wiring is a near 1:1 structural match to Flux:

| | Flux | HY-Motion |
|---|---|---|
| pooled vec | timestep + **pooled CLIP** | `adapter = timestep_feat + vtxt_feat` (**pooled CLIP-L**, 768) |
| token stream | **T5-XXL** hidden states | **Qwen3-8B** hidden states (4096) |
| main stream | image patches (2D RoPE) | motion frames (**1D temporal RoPE**) |

**We already have Flux/Flux2 ported.** This is the single biggest cost lever in the whole scope.

**Ops inventory** [VERIFIED — grepped `hymotion/network/`]: `nn.Linear`, `LayerNorm`, `RMSNorm`, `SiLU`,
`GELU`(+tanh), RoPE, softmax attention with an **additive dense float mask**, Gram-Schmidt rot6d→matrix.
**Every one of these already exists in our ggml stack. Zero novel ops.** [INFERRED — from the brief's
inventory of ported infra + the grep.]

**Sequence length is tiny:** motion 360 + text 128 = **488 tokens**. Attention is 488² ≈ 238k — trivial.
**Flash-attn is pointless here**; the vanilla path is fine. Removes a whole dependency.

Custom bits (the only real port work):
- `mask_mode: narrowband` — an additive `[B,1,488,488]` float mask; T→M attention explicitly disabled
  (`base[:, :, motion_len:, :motion_len] = -inf`) [VERIFIED — `_build_dmm_attn_mask_shared`]
- `token_refiner` — HunyuanVideo-style text refiner over the Qwen3 stream
- optional `start_token`, optional `long_skip_connection`
- the fixed-length trick: **always denoises 360 frames and crops** (`trajectory[-1][:, :length]`)
  [VERIFIED — `motion_diffusion.py:569-581`]. Latent shape is constant regardless of requested duration.

**Sampler:** `odeint(fn, y0, t, method="euler")`, `t = linspace(0, 1, 51)` → **plain fixed-step Euler, 50
steps** [VERIFIED]. CFG scale 5.0, batch-doubled. 50 × 2 = 100 forward passes of a 1B model over 488
tokens. We have Euler.

**Undocumented lever — a "Game" domain token.** `enable_special_game_feat: true`;
`_maybe_inject_source_token(..., trigger_sources={"Taobao", "Game"})` [VERIFIED — code path exists].
A domain/style conditioning token biasing toward game-style animation. Given the owner is making game
assets, this is potentially interesting. **[UNVERIFIED — no idea what it does to output quality; not
mentioned in paper or README.]**

**Specs:** **30 fps** (`output_mesh_fps: 30`), **360 frames = 12s max** (`train_frames: 360`; paper:
"sequences longer than 12 seconds were segmented"). CFG 5.0. English (or Chinese via the rewriter).
[all VERIFIED]

**Determinism:** the **1.0B model seeds on CPU** → reproducible across machines. **Lite seeds on GPU** →
not. **Use 1.0B for any golden capture.** [VERIFIED from config; INFERRED for the portability claim.]

---

## 3. WEIGHTS [VERIFIED — HF API, `?blobs=true`]

`tencent/HY-Motion-1.0` — **`gated: False`** ✓ ungated. **Total 6.05 GB** ✓ (brief's figure correct).

```
4171.70 MB  HY-Motion-1.0/latest.ckpt          ← 1.043B fp32
1843.54 MB  HY-Motion-1.0-Lite/latest.ckpt     ← 0.461B fp32
  16.49 MB  assets/wooden_models/boy_Rigging_smplx_tex.fbx
  14.08 MB  assets/.../Boy_lambert4_BaseColor.png
   ~0.03 MB  config.yml ×2, LICENSE.txt, README.md, config.json
```
**Not downloaded** — verified fetchable via API metadata only, per instruction.

**⚠️ The encoders are NOT in this repo** — this is where the brief's "Q4 GGUF next to the ckpts" fails.
Required separately (`ckpts/README.md`) [VERIFIED]:
- `Qwen/Qwen3-8B` — **16.40 GB**, ungated, **Apache-2.0** ← *the elephant*
- `openai/clip-vit-large-patch14` — ~1.7GB
- `Text2MotionPrompter/Text2MotionPrompter` — **optional** rewriter/duration estimator (vLLM)

**Real total ≈ 24 GB, not 6.05 GB.** The 6.05GB is only the DiTs.

**Q4 GGUF reality:** `Qwen/Qwen3-8B-GGUF` (official, 138k downloads) and `unsloth/Qwen3-8B-GGUF` (225k)
both exist and are ungated [VERIFIED via HF API]. So a Q4 GGUF *is* obtainable — just from Qwen, not
from Tencent. **And §5 argues we shouldn't quantize the encoder at all.**

### Licence [VERIFIED — read `LICENSE.txt`, 81 lines]

`tencent-hunyuan-community`. Line 3, verbatim:
> "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE **EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA**"

Line 17: "**Territory**" = worldwide **excluding** EU, UK, South Korea. **Australia is inside the
Territory** — the owner is fine. Royalty-free, derivative works and Model Derivatives permitted, AUP
applies, exclusive jurisdiction **Hong Kong**.

**Two clauses worth knowing** [VERIFIED]:
- §5(c): must not use/distribute/display the Works **or their Output** outside the Territory. A publicly
  reachable `:8077` page technically displays Output. Personal LAN use: fine. Worth remembering if
  anything is ever published.
- §3: derivative distribution carries attribution + the use restrictions downstream.

Qwen3-8B is **Apache-2.0** — no such constraints. CLIP-L is permissive.

---

## 4. THE PORT COST — the main deliverable

Honest days. My estimates; nobody has tried this.

### 4.1 Text encoder — Qwen3-8B → `ctxt [128, 4096]` · **2-3 days** · *NOT "nearly free"*

The brief hoped this was ~free because we have Qwen3 ported. **Partly. Here's the honest picture.**

Our `qwen3_forward.hpp` (131 LOC) *is* architecturally the right shape — GQA, RMSNorm, SwiGLU, RoPE, and
it **does have q_norm/k_norm** [VERIFIED — `qwen3_forward.hpp:103-104`], which Qwen3-8B needs. But it's
bespoke to SkinTokens: `hidden 896, 28 layers, vocab 33036` [VERIFIED — header comment], a *shrunken
custom vocab*. Qwen3-8B is `hidden 4096, 36 layers, vocab 151936`. Config fields are defaults so it's
parameterisable, but it has no GGUF weight path and no Qwen3 BPE tokenizer.

**Recommendation: don't reuse our bespoke Qwen3 — use llama.cpp** (`~/dev/llama.cpp` present
[VERIFIED]). It already supports Qwen3-8B GGUF natively, tokenizer included. What we'd write:
- Qwen3 chat template with **`enable_thinking=False`** [VERIFIED — `text_encoder.py:124`], the fixed
  system prompt (`PROMPT_TEMPLATE_ENCODE_HUMAN_MOTION`, verbatim in repo)
- `crop_start` — locate where user text begins in the templated string and slice the system-prompt
  tokens off the hidden states; `max_length_llm = 128 + crop_start` [VERIFIED — `_compute_crop_start`]
- last-hidden-state extraction: `llama_get_embeddings_ith` with `LLAMA_POOLING_TYPE_NONE`

**⚠️ Trap to guard:** HF returns `llm_outputs.hidden_states[-1]` [VERIFIED — `text_encoder.py:155`].
Whether that is pre- or post-final-`norm`, and whether llama.cpp's embedding output matches it, is a
**classic off-by-one-layer bug**. This is the #1 place a port silently produces plausible-but-wrong
conditioning. **Must be golden-tested at the tensor level, not eyeballed.** [UNVERIFIED — I did not
check llama.cpp's exact semantics.]

Note it loads `AutoModelForCausalLM` — we only need hidden states, so **the `lm_head` (151936 × 4096 ≈
622M params ≈ 1.2GB bf16) is pure dead weight.** Dropping it is free. [INFERRED]

### 4.2 CLIP-L pooled → `vtxt [1, 768]` · **1 day**
We have CLIP ported. Need `pooler_output` semantics (final_layer_norm → EOS-token pooling), max_length
77. Small. [VERIFIED — `SENTENCE_EMB_LAYOUT`, `pooling_mode: pooler_output`.]

### 4.3 MMDiT backbone · **5-8 days** ← the real work
~1.3k LOC of python total (`hymotion_mmdit.py` 636 + `attention.py` 110 + `encoders.py` 121 +
`token_refiner.py` 192 + `positional_encoding.py` 174 + `modulate_layers.py` 49 + `bricks.py` 46)
[VERIFIED — `wc -l`]. A small surface, and Flux gives us the block skeleton.

New vs Flux: 1D temporal RoPE; the narrowband additive mask; token_refiner; vtxt MLP encoder; optional
long-skip + start token; fixed-360 latent with masked crop.

### 4.4 Sampler · **0.5 day** — fixed-step Euler, 50 steps, CFG 5.0. We have it.

### 4.5 Decode + post-processing · **2-3 days** ← *lumpier than it looks*
- rot6d → matrix (Gram-Schmidt): trivial
- **slerp smoothing σ=1.0** — Gaussian-weighted **Markley quaternion average**
  (`wavg_quaternion_markley`) = **eigenvector of a 4×4 accumulator matrix** → needs a small symmetric
  eigensolve in C++. Not hard, but it's *real* numerical code, not a memcpy. [VERIFIED — `motion_process.py:29`]
- **Savitzky-Golay** on translation, window 11, polyorder 5 → fixed FIR coefficients, precomputable
- **ground-align via body-model FK** — **we can skip this.** It exists to plant *their wooden mannequin*
  on the floor. We retarget onto our own rig and would ground-align against our own mesh. [INFERRED]

### 4.6 Norm stats · **0.5 day** — ship `stats/Mean.npy` + `Std.npy` (both `(201,)`), bake into GGUF.
Note the decoder guards `std < 1e-3 → 0` [VERIFIED — `motion_diffusion.py:209-211`]; replicate exactly.

### 4.7 Motion tokenizer / VQ · **0 days — DOES NOT EXIST.** Continuous flow matching. Nothing to port.

### 4.8 Tokenizers · **0 days** — Qwen3 BPE from llama.cpp, CLIP BPE we have.

### 4.9 Prompt rewriter · **skip initially, or 1 day to substitute ours**
Optional. `--disable_rewrite --disable_duration_est`. But it does two real jobs: **estimates duration**
and cleans the prompt. **The full prompt template is verbatim in the repo**
(`REWRITE_AND_INFER_TIME_PROMPT_FORMAT`) [VERIFIED] — so we can point **our existing LLM stack** at it
for ~1 day rather than pull another 8B + vLLM. Nice reuse.
⚠️ **Confound to control:** Tencent's headline numbers presumably use the rewriter. Disabling it and
then judging quality tests *a different system* than the one that scored 78.6. Say so out loud when the
owner judges. [INFERRED]

### Port total — **11-17 days** for the model, honestly

| Component | Days | Confidence |
|---|---|---|
| Qwen3-8B encoder via llama.cpp | 2-3 | med — hidden-state parity is the risk |
| CLIP-L pooled | 1 | high |
| MMDiT backbone (Flux-derived) | 5-8 | med |
| Euler sampler | 0.5 | high |
| Decode + Markley/savgol | 2-3 | med |
| Norm stats + packaging | 0.5 | high |
| Prompt rewriter (ours) | 1 | high — optional |
| **Total** | **12-18** | **±50%. Nobody has tried.** |

Excludes retargeter work (§6) and the PoC (§7).

---

## 5. VRAM PLAN for 12GB — **the brief's "#1 risk" is not a risk** ✅

### The decisive finding [VERIFIED — read the sampling loop]

**The text conditioning is computed ONCE, before the ODE solve, and Qwen3 is never re-invoked during
denoising.** `motion_diffusion.py:509` calls `encode_text`; the closure `fn` at :551-565 consumes only
the precomputed `ctxt_input` / `vtxt_input` tensors; `odeint` runs at :580. **The encoder is free after
encoding.** The brief's open question — "nobody checked whether Qwen3 is re-invoked during denoising" —
is now checked. **It is not.**

**Even better:** `sample(..., hidden_state_dict=None)` — if you *pass* `hidden_state_dict`, `encode_text`
is skipped entirely (`:508`). **The conditioning can be computed in a completely separate process** and
handed in. The reference implementation already has the seam we need. [VERIFIED]

### Why the reference needs 26GB [INFERRED — arithmetic, cross-checks the README]
```
Qwen3-8B          bf16   16.4 GB   ← resident, never freed (no offload code anywhere in the repo)
DiT 1.0B          fp32    4.2 GB
CLIP-L            fp32    ~0.5 GB
activations, torch overhead, num_seeds=4 batching
                        ≈ 24-26 GB   ✓ matches README's 26GB
```
I grepped the entire repo for `empty_cache|offload|del |.to('cpu')|load_in_4bit` — **the only hit is
`load_in_4bit=True` on the optional prompt rewriter** (`prompt_rewrite.py:277`) [VERIFIED]. **Nobody
wrote any offloading.** 26GB is not a floor; it is an absence of engineering.

### The plan — sequential, two stages

**Option A — CPU encode (recommended). Honest peak ≈ 3 GB.**
The encode is **one forward pass over ≤128 tokens**. On CPU via llama.cpp that's ~1-2s. It never needs
to touch the GPU at all.
```
Stage A (CPU):  Qwen3-8B Q8/f16 + CLIP-L  →  ctxt[128,4096] + vtxt[1,768]     GPU: 0 GB
Stage B (GPU):  DiT f16 2.1GB + acts <0.5GB (batch 2 CFG × 488 tok × 1280)    GPU: ~2.5-3 GB
                                                                        PEAK: ~3 GB / 12 GB
```

**Option B — GPU encode, sequential. Honest peak ≈ 9 GB.**
Qwen3-8B Q8 ~8.7GB + CLIP 0.25GB → encode → free → DiT f16 2.1GB. Peak ~9GB. Fits, with less headroom.

**Recommendation: Option A.** It removes the encoder from the VRAM budget *entirely* and, critically,
**sidesteps the quantization-perturbs-conditioning risk** — run the encoder at f16/bf16 on CPU where
memory is cheap, and never find out the hard way whether Q4 hidden states break a DiT trained on bf16
ones. That risk was real and [UNVERIFIED]; Option A makes it moot rather than measuring it.

**Answering the brief's questions directly:**
- *Can the encoder be freed after encoding?* **Yes — verified.** One-shot, precomputable, separable process.
- *Q4/Q8 the encoder?* **Don't.** Put it on CPU instead. Free, and strictly safer.
- *Sequential load?* **Yes — that's the design.**
- *Honest peak?* **~3 GB (CPU encode) / ~9 GB (GPU encode).** Both fit 12GB. The brief's "~5-6GB
  arithmetic, not a measurement" was the right ballpark and the right caveat — it is **still not a
  measurement** [UNVERIFIED], but the structural facts underneath it are now verified, which is what
  the estimate was actually resting on.

**Framing confirmed:** motion gen runs *separately* from asset gen, so the 3060 should be empty
(asset pipeline peaks 10751/12288 MiB — they must not co-reside). Sequential by design, not by luck.
✅ Correct framing. **VRAM is not the kill risk. §6 has the real one.**

---

## 6. INTEGRATION DESIGN (design only)

### 6.1 Factoring — **the owner's instinct is right. Confirmed.**
Separate binary `text_to_motion`, mirroring `image_to_rig`. **Not** inside it. Three reasons:
1. **Cardinality.** Asset = once per character. Motion = per-character **×** per-prompt. N:1. Coupling
   forces a 6GB model load per animation, or makes the asset pipeline carry motion's VRAM forever.
2. **VRAM.** Asset pipeline peaks 10751/12288 MiB. They cannot co-reside on a 12GB card. Separate ⇒
   sequential ⇒ both fit.
3. **The owner said so** — "avoid it being part of the workflow."

### 6.2 The chain
```
prompt
  └─► [C++] Qwen3-8B (CPU, llama.cpp) + CLIP-L  ──►  ctxt[128,4096], vtxt[1,768]
        └─► [C++] MMDiT + Euler×50, CFG 5.0     ──►  latent [T, 201]
              └─► [C++] denorm + decode          ──►  rot6d [T,22,6] + transl [T,3]  @30fps
                    └─► [python, for now] retarget_delta.py + RETARGET BASE POSE + bonemap_v2.py
                          └─► clip JSON {name, fps, bones[bone_N], quats[T][n][4] xyzw LOCAL, root_pos}
                                └─► three.js player :8077   ──►  the owner's eye
```

### 6.3 🔑 Our retargeter's skeleton **already is** HY-Motion's skeleton
`retarget_delta.py:86-89`, verbatim [VERIFIED]:
> "ASSUMPTION 1: SMPL-X's first 22 body joints are the same joints, in the same order, as
> SMPL/HumanML3D's 22 (pelvis..wrists). True by construction -- SMPL-X is body-compatible with SMPL"

and it builds rest from `Jrest = (J_regressor @ v_template)[:22]` of `SMPLX_NEUTRAL_2020.npz`.

SMPL / SMPL-H / SMPL-X **share the same first 22 body joints**. HY-Motion is SMPL-H's 22. **Our proven
retargeter's rest skeleton is already exactly HY-Motion's skeleton** — the assumption it's built on is
the one HY-Motion satisfies. No adapter. That is a genuinely lucky alignment and it de-risks §7 to
near-zero. [VERIFIED for our code + paper; INFERRED that no per-joint offset lurks — **worth a
structural check** (feed rest → expect identity), see §7.]

### 6.4 What must be C++, and what shouldn't
- **C++:** the DiT, encoders, sampler, decode. This is the heavy, hot, GPU path. That's the port.
- **Python, deliberately, for now:** `retarget_delta.py`, `bonemap_v2.py`. **Recommend porting these
  LAST or not at all initially.** They're pure numpy geometry — no GPU, no model, cold path,
  milliseconds per clip, and **proven** (0.000000° roundtrip; 0.1° rig-invariant spread; gilly's feet
  135.8°→11.3°). The "native C++/ggml, remove python/docker" rule exists to kill *heavy* python — a
  24KB numpy script that runs once per clip is not what that rule is aimed at. Rewriting proven
  geometry from scratch is how you re-earn bugs you already paid for.
  **Flagging this as a deliberate, revisitable call — not an oversight.** If the owner wants a single
  static binary end-to-end, add **3-5 days** and do it *after* the model is proven.

### 6.5 THE NAMING QUESTION — the owner asked directly. **CONFIRMED. He's not limiting himself.**

> *"as far as our rigging / retargetting - i thought we did mixamo/SMPL-X/everything? if not can we?
> i guess i dont want to limit us."*

**The hub-and-spoke read is correct, and it's stronger than the brief argued — because the hub already
exists and is already load-bearing in shipped code.**

`retarget_delta.py` *already* pins SMPL-22 as its rest basis (§6.3). SMPL-22 isn't a new choice to make;
it is the **de-facto hub today**, in proven code. So:

```
                        bone_N  (our rig, J emergent: 28/29/33/38/61, bones UNNAMED)
                          │
                    bonemap_v2.py   ◄── derive the semantic map ONCE (topology → SMPL-22)
                          │
                     ┌────┴─── THE HUB: SMPL-22 (pelvis..wrists) ───┐
                     │                                              │
       ┌─────────────┼─────────────┬──────────────┐          ┌──────┴───────┐
       ▼             ▼             ▼              ▼          ▼              ▼
   mixamo.yaml    ue5.yaml     vroid.yaml     (future)   HY-Motion     AMASS / CMU /
  mixamorig:*    UE5 names    VRM names                  (SMPL-H 22)   Mixamo library
   ── OUTPUT NAMINGS (leaves) ──                          ── INPUT SOURCES (leaves) ──
```

**One hub, many leaves — in both directions.** Naming is an *output projection*, sources are an *input
projection*, and both pivot on the same map. **Nothing is locked in.**

**Cost to support all three namings: ~1-2 days.** All three configs already exist on disk
(`skeleton_template/configs/skeleton/{mixamo,ue5,vroid}.yaml` [VERIFIED]), and `rename_to_mixamo.py`
(9.2KB) + `bonemap.py` (5.9KB) + `MIXAMO_REPORT.md` are already written and **proven on gilly (61
joints)** [VERIFIED files exist; brief's proof claim not independently re-run — **[UNVERIFIED]**, and
note it has **never been run on the soldier**]. The work is refactoring them to emit *from the one map*
rather than re-deriving per target. That's it.

**⚠️ Known landmine:** stock `derive_map` **crashes on the soldier** — his arms hang low, fingertips fall
below hips, and the spine subtree passes the "is it a leg?" test. `bonemap_v2.py` exists to fix this.
Any naming work inherits this bug. Don't rediscover it. [from brief — **[UNVERIFIED]** by me.]

**The owner's strategic point is right, and it's the best argument in this document.** Standard naming +
an SMPL-22 hub means **HY-Motion becomes a bonus, not a dependency**:
- **AMASS is natively SMPL-H** — *exactly* HY-Motion's skeleton. ~11k mocap sequences, same 22 joints,
  same pipe, zero new code. [INFERRED — AMASS/SMPL-H alignment is well-established; not re-verified here.]
- **Mixamo's library** arrives through `mixamo.yaml`, same hub.
- Tencent almost certainly chose SMPL-H *because* AMASS is SMPL-H.

⇒ **The retargeter pays for itself even if HY-Motion is judged bad.** That is a strong reason to build
the hub *first* and treat the model as an experiment hanging off it. It converts a bet on one model into
infrastructure that survives the bet losing.

**α (arm=1 / leg=0) stays a human judgement.** Nothing geometric picks it; gilly's bow legs tracking her
own mesh to 2.5° is a real property, not an error. The hub doesn't change this. The owner decides.
[from brief — **[UNVERIFIED]** by me, but the reasoning is sound and I have no evidence against it.]

---

## 7. RISKS / KILL CRITERIA

| # | Risk | Verdict |
|---|---|---|
| 1 | **HOI: "swat a fly away"** | 🔴 **THE REAL KILL RISK.** Sharper than the brief knew. README limitations, verbatim: **"❌ Environment & Camera: Descriptions of objects, scenes"** — *a fly is an object*. Paper: "**Human-Object Interaction (HOI)**: our current dataset primarily focuses on body kinematics **without explicit object geometry**... may struggle to generate physically accurate interactions with external objects" [VERIFIED, both]. The owner's own example is **explicitly documented as out of scope**. Not a coin flip — **the docs say no**. |
| 2 | **"do a barrel roll"** | 🟡 Plausible. Body-only acrobatic, no object, no environment; "game" + "sports" are named training categories. But paper concedes "highly detailed or complex instructions" struggle [VERIFIED]. **[UNVERIFIED]** — only the eye test decides. |
| 3 | **No hands** | 🟢 **Confirmed fact, NOT a kill.** Mitten geometry makes it moot. §1b. Downgrade from kill criterion to *known and accepted*. |
| 4 | **VRAM** | 🟢 **Resolved by design.** ~3GB (CPU encode). §5. Not a risk. |
| 5 | **Licence** | 🟢 Australia in Territory. Fine. §3. Note §5(c) if ever published publicly. |
| 6 | **Q4 encoder perturbs conditioning** | 🟢 **Designed away** — CPU f16 encode. Never arises. §5. |
| 7 | **⚠️ Non-humanoid characters** | 🔴 **"❌ Non-humanoid Characters: Animations for animals or non-human creatures"** [VERIFIED — README]. The eyetest dir is literally `puppy-eyetest/`. **If the target characters are animals, HY-Motion is dead on arrival.** gilly and the soldier read as humanoid (hips/spine/arms/legs, SMPL-22 maps to them). **But I did not verify what the characters actually are, and I'm not assuming. OWNER: please confirm.** [UNVERIFIED — flagging loudly rather than guessing, given the brief's track record of inverted assumptions.] |
| 8 | **No loops / no in-place** | 🟡 "❌ Special Modes: Seamless loop or in-place animations" [VERIFIED]. Looping idles are out. Matters if he wants idle cycles. |
| 9 | **12s cap** | 🟢 360 frames @30fps. Paper segments longer sequences into clips. Chaining = our problem, not the model's. |
| 10 | **Reference won't run on a 12GB 3060** | 🟡 **New finding.** 20.6GB resident as shipped. **Even the PoC needs the sequential patch.** ~20 lines. §8. |
| 11 | **Multi-person** | 🟢 "❌ Multi-person Interactions" [VERIFIED]. Single character. Fine. |

**Kill criteria, stated plainly:**
- **KILL** if the characters are non-humanoid (#7) — the model refuses that class by construction.
- **KILL** if the eye test on the owner's own prompts fails and in-distribution controls also fail.
- **DON'T KILL** on "swat a fly" alone — it's documented-unsupported, so failing it is *the model
  behaving as advertised*, not the model being bad. Judge it on prompts it claims to serve.
- **NOT kill criteria:** hands, VRAM, licence. All three resolved.

---

## 8. THE CHEAP PROOF-OF-CONCEPT — do this BEFORE any port

**The ordering in the brief is right and I want to underline it: porting a model whose output the owner
doesn't like is the expensive mistake.** Everything in §4 (12-18 days) is contingent on ~1 day of
eye test. Do the day first.

**And it's cheaper than the brief hoped, because of one finding:** the model *already emits `keypoints3d`
(T,22,3) positions* — **exactly the format our existing retargeter already eats** (it consumes
MoMask/HumanML3D `(T,22,3)` positions today, §1). **The PoC needs almost no new code.**

### Steps (~1 day)

1. **Fetch** (~24GB, one-time, off the GPU): HY-Motion 6.05GB + Qwen3-8B 16.4GB + CLIP-L 1.7GB.
   Skip Text2MotionPrompter.
2. **⚠️ Patch for 12GB — mandatory, ~20 lines.** The reference will OOM as shipped (#10). Cleanest:
   encode on CPU, pass `hidden_state_dict` into `sample()` — **the seam already exists**
   (`motion_diffusion.py:508`) [VERIFIED]. Alternative: `del pipeline.text_encoder;
   torch.cuda.empty_cache()` between encode and denoise.
   Run with `--disable_rewrite --disable_duration_est --num_seeds=1`, **use the 1.0B model** (CPU RNG ⇒
   reproducible; Lite is GPU-RNG ⇒ isn't), ≤30 words, ≤5s.
3. **Prompts — three buckets, deliberately:**
   - the owner's: **"do a barrel roll"**, **"swat a fly away"**
   - **in-distribution controls** from Tencent's own list ("A person stands up from the chair, then
     stretches their arms", "A person walks unsteadily, then slowly sits down")
   - 1-2 game-ish ones, since that's the actual use case
   **The controls are the point.** Without them, a bad "swat a fly" is uninterpretable — you cannot tell
   *"the model is bad"* from *"the prompt is documented as out-of-scope"* (#1). With them, you can.
4. **Retarget — take the free path first.** Feed the model's `(T,22,3)` positions to **`retarget_delta.py`
   unchanged**. Its SMPL-22 rest basis already matches (§6.3).
   ⚠️ **Caveat:** `keypoints3d` comes from Tencent's **wooden-mannequin FK** (`WoodenMesh`), not
   necessarily SMPL rest [VERIFIED — `body_model.py`]. Safer and still ~30-50 lines: do FK ourselves from
   `rot6d` + the `SMPLX_NEUTRAL_2020.npz` rest we already use, giving positions *and* rotations. That
   also structurally validates the 201 layout for free (step 6).
   **Why positions first, when rotations are the whole point?** Because it isolates the question.
   *"Is the motion any good?"* must be answered before *"is our rotation retarget better?"* — if the
   motion is bad, nothing downstream matters, and the position path is already proven. Rotation-native
   retarget (recovering twist, §1) is the **second** experiment, ~1-2 days, only if the eye test passes.
5. **Ship to `:8077`**, alongside an existing MoMask clip for reference. **Owner judges. No winner
   declared.** Follow the pause-between-renders discipline.
6. **Bank goldens while you're there** — the run is already paid for.

### Guarding the goldens — this project's goldens have been wrong twice

Concrete guards, in order of strength:

1. **Structural checks beat numeric ones.** This is the lesson of the `cos 0.9998` burn. I already ran
   one *for free* in this scope: the paper says `201 = 3 + 6 + 21×6 + 22×3`, and the code's slice
   arithmetic independently produces the same partition. **Two sources, derived separately, agreeing on
   structure** — that's worth more than any cosine. Do the same in code: **feed the SMPL-H rest pose and
   assert decoded rot6d ≈ identity.** If the layout is misread, that fails loudly. A cosine wouldn't.
2. **Never trust cos-sim. Use max-abs-error per element**, plus the shape/layout assertions.
3. **Bank at multiple depths, not just the output** — (a) `ctxt[128,4096]` + `vtxt[1,768]` from the
   encoder ← *most likely to be silently wrong* (§4.1's off-by-one-layer trap), (b) seeded `y0` noise,
   (c) block-0 and final-block activations, (d) final `rot6d`/`transl`. A single end-to-end golden tells
   you *that* you're wrong, never *where*.
4. **Exploit CPU RNG.** The 1.0B model's `random_generator_on_gpu: false` [VERIFIED] means goldens are
   reproducible across machines and drivers. **This is why goldens must come from 1.0B, not Lite.**
5. **The eye test is the real golden.** A tensor can match and the animation still look wrong. Ship to
   `:8077` — the owner's eye is the acceptance criterion, per project rule.

### PoC cost
**~1 day**, reusing: the existing retargeter, the existing player, the existing `:8077` page. New code:
~20-line VRAM patch + ~30-50 lines of FK. **Nothing is ported until the owner has looked at it.**

---

## 9. RECOMMENDED ORDER

1. **Answer #7 (non-humanoid).** One question to the owner. It's free and it can kill the whole thing.
2. **PoC (§8), ~1 day.** Renders on `:8077`. Owner judges.
3. **If the eye test passes** → build the **SMPL-22 hub + naming leaves** (§6.5, ~1-2 days). **This is
   worth doing even if HY-Motion is rejected** — it unlocks AMASS/Mixamo through the same retargeter.
   It's the only item here that pays off regardless.
4. **Then** the port (§4, 12-18 days), encoder-parity-first (§4.1 is the riskiest and cheapest to test).
5. **Rotation-native retarget** (twist recovery, §1) — 1-2 days, high value, do it early in the port.
6. **Port the retargeter to C++** — last, or never. §6.4.

---

## 10. WHAT I COULD NOT CHECK

Stated plainly, because the brief asked for honesty over polish.

- **Never touched the GPU.** No run, no measurement. **Every performance and VRAM number here is
  arithmetic** [INFERRED], not measurement. §5's ~3GB is a *plan*, not a result.
- **Did not download any weights.** Verified fetchable via HF API metadata only.
- **Never ran the reference.** No golden, no output, no image. The whole quality question is open.
- **Did not verify llama.cpp's hidden-state semantics** vs HF's `hidden_states[-1]` — §4.1's trap
  remains theoretical, and it's the port's most likely silent failure.
- **Did not re-verify the brief's claims about our own proven work** — the 0.000000° roundtrip, the
  0.1° rig-invariant spread, gilly's feet, the 44° A-pose mismatch, the `derive_map` soldier crash.
  I verified the **files exist** and read enough of `retarget_delta.py` to confirm its *design* and its
  SMPL-22 assumption (which §6.3 leans on hard). The **numbers** are [UNVERIFIED] by me.
- **Did not confirm what gilly and the soldier actually are** (risk #7). Given this brief's stated record
  of inverted assumptions — "gilly has fingers, the soldier doesn't" proved exactly backwards — **I am
  not guessing on the one open question that can kill the project.**
- **The "Game" token**: verified the code path exists; **no idea** what it does to output.
- **AMASS/SMPL-H alignment** (§6.5): asserted from general knowledge, not re-verified tonight.
- **Param-count table**: derived from ckpt bytes ÷ 4 and `config.yml`. Not from loading the ckpt.
- **SSAE 78.6**: still Tencent's metric on Tencent's prompt set. But it's **Gemini-judged and the eval
  harness is open** — more credible than the brief allowed, less than a neutral benchmark. Don't cite it
  as proof; don't dismiss it as circular either.
