# HANDOFF — Wan2.2 DiT post-attention glue audit (transpose / cast / concat / coalesced-copy)

Scope: code analysis only (no build / no GPU). Profiled config = Wan2.2 DiT, 25f @ 1280×704,
post-cuDNN-flash-attn + `WAN_DIT_F16` residual stream, 1 step. Goal (owner's bar): *is everything
cycling through HBM 100% needed, or is some of it redundant?* Per-block ops run ×40 blocks ×N steps,
so each removable op is `5519/40 ≈ 138` calls' worth over the profile window.

Files traced:
- `src/model/diffusion/wan.hpp` — block forward (`WanAttentionBlock::forward`, `WanSelfAttention`,
  `WanT2VCrossAttention`/`WanI2VCrossAttention`, `modulate_*`, `unpatchify`, `forward_orig`).
- `src/core/ggml_extend.hpp` — `ggml_ext_attention_ext` / `build_kqv` (the cuDNN/flash wrapper),
  `ggml_ext_linear`, `ggml_ext_chunk`, `ggml_ext_torch_permute`.
- `src/model/common/rope.hpp` — `Rope::attention` + `apply_rope` (fused `ggml_rope_pe`).
- `ggml/src/ggml-cuda/cpy.cu` + `concat.cu` — which ggml op each profiled kernel maps to.
- `src/model/diffusion/ltxv.hpp` — the prior cast-reduction work, for cross-checking decisions.

---

## 0. Kernel → ggml-op mapping (so the trace is grounded, not guessed)

From `ggml/src/ggml-cuda/cpy.cu` and `concat.cu`:

| Profiled kernel | ggml op that emits it | Notes |
|---|---|---|
| `cpy_perm_transpose<float>` (5519) | `ggml_cont`/`ggml_dup` of an **F32** tensor whose unit-stride run is on an axis `≠0` (a real transpose), `cpy.cu:542-554,595` | F32-only fast path. **The fact this is `<float>` and #2 overall means the heavy attention-layout conts are still running in F32, not F16** — i.e. the `WAN_DIT_F16` stream did *not* lower them, because q/k are cast back to F32 for RoPE (see §1). |
| `cpy_perm_coalesced<float>` (≈2500) | `ggml_cont` of an **F32** permuted tensor whose dim0 *is* unit-stride (higher dims permuted), `cpy.cu:556-558,613` | Same family, the non-transpose permute case. |
| `convert_unary<__half,float>` (3251) | `ggml_cast(x, F16)` where `x` is F32 — i.e. **F32→F16** (template is `<dst,src>`), the `cpy_scalar_*<...,half>` / convert path | These are the **K/V→F16 casts for flash** in `build_kqv` (`ggml_extend.hpp:1516,1528`), unchanged by the residual-stream dtype because the flash kernel always wants F16 K/V. |
| `concat_T_cont_4d<float,2>` (3075) | `ggml_concat` along **dim 2 or 3** (`concat.cu:126` is the `dim>=2` fallthrough) | **Not emitted by the standard t2v/i2v DiT block** — see §3. |

---

## 1. `cpy_perm_transpose<float>` (5519) + `cpy_perm_coalesced<float>` (≈2500) — attention layout conts

### Origin trace (per block)

Self-attn (`WanSelfAttention::forward` → `Rope::attention` → `ggml_ext_attention_ext` with
`skip_reshape=true`):
- q,k: `q_proj/k_proj` → `norm_q/k` → reshape_4d `[d_head, n_head, L, N]` → **`apply_rope`**.
  `apply_rope` calls the *fused* `ggml_rope_pe` (`rope.hpp:963`), whose **output is already
  `[d_head, L, n_head*N]`** — the n_head↔L transpose is fused *inside* the rope kernel, so q/k pay
  **zero** explicit cont here. (`skip_reshape=true` at `rope.hpp:1014` also skips the wrapper's own
  q/k reshape+permute+cont at `ggml_extend.hpp:1476-1481`.)
- v: does **not** go through rope. It arrives `[d_head, n_head, L, N]` and flash wants
  `[d_head, L, n_head, N]`, so `build_kqv` does `ggml_ext_cont(ggml_permute(v,0,2,1,3))`
  (`ggml_extend.hpp:1519`). **1 big transpose-cont/block** (L≈37k).

Cross-attn (`WanT2VCrossAttention::forward` → `ggml_ext_attention_ext` with `skip_reshape=false`):
- q permute+cont (`:1476`) — big (`n_token≈37k`).
- k permute+cont (`:1480`) — small (context L=512).
- v permute+cont (`:1519`) — small.
- **3 conts/block**, of which q is big.

Plus, once per *forward* (not per block): `unpatchify` (`wan.hpp:740-746`) does **4×
`ggml_ext_cont(ggml_ext_torch_permute(...))`** on the full output tensor, and `forward_orig`'s
patch-embed permute (`:782`). These are F32 and large but amortised over the whole step.

### Classification

| Cont | Class | Justification |
|---|---|---|
| self-attn **v** permute-cont (`:1519`) | **(c) NEEDED** *(but see (b) below)* | v's memory layout is `[d_head, n_head, L]`; flash requires `[d_head, L, n_head]`. L and n_head are not adjacent in memory, so a transpose-copy is unavoidable **with the current op set**. q/k get this transpose *for free* because it's fused into `ggml_rope_pe`; v has no rope so it can't piggy-back. |
| cross-attn q/k/v permute-cont (`:1476/1480/1519`) | **(c) NEEDED** | Standard BLHD→BHLD layout for flash; k/v are small (context=512). |
| `unpatchify` 4× cont (`:740-746`) | **(c) NEEDED** | Genuine pixel-shuffle reshape; once per forward, not per-block. |

### (b) REDUCIBLE design — fuse the v-transpose (documented, NOT implemented)

The self-attn v-cont is the single biggest *per-block* F32 transpose that is **not** load-bearing math —
it only exists because v skips rope. Two ways to remove it, both for the main agent to weigh (needs a
kernel and GPU validation, so left as a proposal):

1. **A v-only fused transpose-into-F16**: `build_kqv` already immediately casts v→F16 after the cont.
   A single `cpy_perm_transpose` variant that writes F16 directly (`<float>` in, `<half>` out) would
   fuse the transpose **and** the `convert_unary<half,float>` (§2) into one kernel — halving both the
   v-transpose traffic (read F32, write F16 = 0.75× bytes) and removing one of the 3251 casts/path.
   `cpy.cu` already has the `cpy_scalar_*<half,float>` machinery; the perm-transpose path is currently
   F32→F32 only (`cpy.cu:595`).
2. **Emit v in `[d_head, L, n_head]` from `v_proj`**: would need a strided/transposed GEMM epilogue
   (not available in the ggml mul_mat path). Not worth it vs. option 1.

Note the **F32-ness** is the real waste here: under `WAN_DIT_F16` the residual stream is F16 but v is
re-materialised through an F32 transpose. Option 1 is the clean fix.

---

## 2. `convert_unary<__half,float>` (3251) — the K/V→F16 flash casts

### Origin trace (per attention call, ×2 attentions/block for t2v, ×3 for i2v)

`build_kqv` (`ggml_extend.hpp`):
- `k_in = ggml_cast(k_in, F16)` (was `:1516`)
- `v_in = ggml_cast(v_in, F16)` (was `:1528`)

These fire on **both** the F32 and F16 residual streams — the flash/cuDNN kernel always consumes F16
K/V (F32 accumulate is internal). That's why the profile flagged them as *unchanged by the F16 stream*.

### Classification + WHAT I CHANGED

- On the **F32 prod stream**: K/V arrive F32 → the cast is a genuine F32→F16 conversion = **(c) NEEDED**
  (flash hard-requires F16 K/V). Untouched.
- On the **`WAN_DIT_F16` / NVFP4 stream**: the q/k/v Linears emit **F16** (`ggml_ext_linear` mm_dst gate,
  `ggml_extend.hpp:1149`), and the K rope output / V `ggml_ext_cont` keep that F16. So `ggml_cast(_, F16)`
  on an already-F16 tensor = a **redundant F16→F16 copy** — and `ggml_cast` **never short-circuits on a
  matching type** (verified in `ggml.c`: it unconditionally creates a `GGML_OP_CPY` node). That's a
  full-width copy of K and of V, every attention call, for zero value = **(a) REDUNDANT**.

**IMPLEMENTED** (`ggml_extend.hpp`, `build_kqv`): guarded both casts with `if (type != GGML_TYPE_F16)`.

```cpp
if (k_in->type != GGML_TYPE_F16) { k_in = ggml_cast(ctx, k_in, GGML_TYPE_F16); }
...
if (v_in->type != GGML_TYPE_F16) { v_in = ggml_cast(ctx, v_in, GGML_TYPE_F16); }
```

**Proof of output-identity / why it's clearly-safe:**
- Skipping `ggml_cast(F16→F16)` returns the *same bytes* (cast to the same type is the identity map),
  so any downstream read is identical.
- Contiguity is preserved: when F16, `k_in` is the contiguous fused-rope output (or the cross-attn
  `ggml_ext_cont` output at `:1480`), and `v_in` is the `ggml_ext_cont(permute(...))` output two lines
  up — both already contiguous, which is exactly what the cast would have produced. Flash gets a
  contiguous F16 either way.
- **Self-gating**: the guard only changes behaviour when the input is *already* F16 (the F16/NVFP4
  path). The default **F32 prod stream still casts F32→F16 unconditionally → byte-identical**.
- Benefits **LTX too**: `ltxv.hpp`'s attention goes through the same `build_kqv`, and `LTX_DIT_F16`
  has the same already-F16 K/V — this redundant re-cast was never guarded there either.

This removes 2 full-width K/V copies per attention call on the F16/NVFP4 path (on the F32 path these
casts are the real `convert_unary<half,float>` work and stay).

---

## 3. `concat_T_cont_4d<float,2>` (3075) — NOT in the standard DiT block

### Origin trace

I traced every `ggml_concat` reachable from `Wan::forward_orig` and the per-block path. The **standard
t2v/i2v DiT block has no concat**. The only concats in `Wan`:
- `wan.hpp:1063` `ggml_concat(x, c_concat, 3)` — i2v conditioning channels, **once per forward**.
- `wan.hpp:827` `ggml_concat(context_img, context, 1)` — i2v context, **once per forward** (dim 1).
- `wan.hpp:918` `ggml_concat(x, time_dim_concat, 2)` — once per forward, only if `time_dim_concat`.

None of these is per-block, so they cannot total 3075. The dim-2/3 concat at this volume comes from
**outside** `wan.hpp`. The two live candidates, both confirmed by grep:

1. **The LiveAvatar / S2V streaming KV-cache path** (`src/longcat_avatar.hpp:324,331`,
   `wan.hpp:forward_kv_cache:216-221`): `v_full = ggml_concat(v_cond, v_noise, 2)` (dim 2) +
   `k_full = ggml_concat(k_cond, k_noise, 1)` — **per block, per consume step** (48 blocks × ~7 steps).
   Given the profile is the *avatar* shape (25f, audio-driven), this is the most likely source.
   → **(c) NEEDED**: concatenating the persisted cond/rolling KV cache with the current block's K/V is
   load-bearing (it *is* the attention input). The fork already optimised the surrounding round-trip
   (lap-28.2: F16-with-F16 concat, no F32 bounce — `longcat_avatar.hpp:305-332`).
2. The **audio encoder** per-frame stacking (`longcat_audio.hpp:518`, `wav2vec2.hpp:390`,
   `conditioner.hpp:2150`) — `ggml_concat(stacked, h, 2)` in a per-frame loop. Also (c) NEEDED.

**Recommendation:** before optimising "the concat", re-run nsys with the kernel's *launch backtrace*
(`nsys profile --cudabacktrace` or the ggml node name) to confirm which of (1)/(2) it is. I did **not**
implement anything here — there is no redundant concat in the DiT block, and the avatar/audio concats
are structurally required. If it's the KV-cache concat, the only further lever is residency (keep the
cond cache pre-concatenated across steps), which is a scheduler change, not a glue removal.

---

## 4. Other per-block glue inspected (and why I left it)

| Op | Where | Class | Reason |
|---|---|---|---|
| q/k **F16→F32** cast for fused RoPE | `wan.hpp:150-153` | **(b) REDUCIBLE but DON'T** | Fused `ggml_rope_pe` is F32-only (`rope.hpp:953`). An F16 rope path was already **measured on LTX = +16% DiT compute, reverted** (`ltxv.hpp:584-587`): the F16 q/k slow the downstream reshape/cont+flash more than the saved cast. This is a validated keep, not a redundancy. **Self-gated** (`if type==F16`) so F32 prod is untouched. |
| `ggml_ext_chunk(e, 6, 1)` modulation chunks | `wan.hpp:458` (block) + `:580` (head) | **(b) REDUCIBLE, left as proposal** | `ggml_ext_chunk` uses **unconditional `ggml_cont`** (`ggml_extend.hpp:785`), so for the N=1/T=1 case (where each `[dim,1,1]` chunk view is *already contiguous*) it launches 6 (+2) **no-op copies/block**. Tiny tensors (~20 KB), so negligible HBM — **not** one of the big kernels. A safe fix exists (`ggml_cont`→`ggml_ext_cont`, which short-circuits contiguous) **but `ggml_ext_chunk` is shared across every model and a caller doing an in-place op on a now-aliased view could corrupt the parent**, so it is NOT "clearly safe" — left as a documented proposal. In the T>1 per-frame case the chunk view *is* non-contiguous and genuinely needs the cont. |
| `modulate_mul`/`modulate_add` reshape_4d/3d | `wan.hpp:379-405` | **(c) NEEDED** | Pure views (no copy) for N=1/T=1; for T>1 they implement the per-frame broadcast. |
| final `kqv = ggml_ext_cont(...)` | `ggml_extend.hpp:1665` | **(c) NEEDED, already optimal** | Uses `ggml_ext_cont` which is a no-op when the flash output view is already contiguous. |
| cuDNN F16-out retype (`:1589-1593`) | `ggml_extend.hpp` | **(c) NEEDED, metadata-only** | No kernel — just rewrites `type`/`nb[]` so the F16 flash output feeds `o_proj` with no upcast. |

---

## Summary

**Changed (clearly-safe, output-identical, F32 prod byte-identical):**
- `src/core/ggml_extend.hpp` `build_kqv` — guarded the K and V `ggml_cast(_, F16)` with
  `if (type != GGML_TYPE_F16)`. Removes a redundant full-width F16→F16 copy of K and V per attention
  call on the `WAN_DIT_F16` / NVFP4 stream (and the same for `LTX_DIT_F16`, which shares this code).
  Proven redundant because `ggml_cast` to a matching type is the identity map and never short-circuits;
  the inputs are already contiguous F16. The F32 default path still casts F32→F16 unchanged.

**Left as documented proposals (need a kernel + GPU validation — for the main agent):**
- §1(b) **fuse the self-attn v-transpose into an F32→F16 `cpy_perm_transpose<half,float>`** — kills the
  biggest per-block F32 transpose-cont *and* folds the v→F16 cast into it. The single highest-value
  remaining lever in the post-attention glue.
- §4 `ggml_ext_chunk` no-op-cont removal — real but tiny, and risky to do in the shared helper.

**Found NOT redundant (justified):**
- The q/k→F32 RoPE round-trip — validated keep (LTX measured F16-rope as +16%, reverted).
- The K/V→F16 flash casts on the **F32** stream — flash hard-requires F16 K/V.
- The `concat_T_cont_4d<float,2>` (3075) is **not in the DiT block** — it originates in the avatar
  KV-cache / audio-encoder concats, which are structurally load-bearing. Re-profile with a launch
  backtrace to confirm which, before spending effort there.

No build, no commit (per instructions).
