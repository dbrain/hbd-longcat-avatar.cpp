# ShotStream — Ground-Truth Inference Reference (for C++/ggml port)

**Scope:** Faithful spec of the *inference* algorithm of KlingTeam **ShotStream**
(streaming multi-shot T2V, finetuned from **Wan2.1-T2V-1.3B**), for an op-for-op
C++/ggml port. Derived from the official repo + paper + HF config.

**Sources**
- Paper: *ShotStream: Streaming Multi-Shot Video Generation for Interactive Storytelling*,
  Luo et al., **ECCV 2026**, arXiv **2603.25746** (MMLab CUHK + Kling Team, Kuaishou).
- Code: `github.com/KlingAIResearch/ShotStream` — cloned to `/tmp/shotstream-ref`
  (all `file:line` cites below are that tree). Built on CausVid + Self-Forcing + LongLive.
- Config: `huggingface.co/KlingTeam/ShotStream` (`shotstream.yaml`, `default_config.yaml`).

**The canonical inference entrypoint is `Inference_Causal.py` → `CausalInferenceArPipeline`
(`pipeline/causal_inference_ar.py`).** That AR pipeline is the streaming multi-shot loop and
is the authoritative reference for the port. (`pipeline/causal_inference.py` is a single-pass
teacher-forced variant that consumes a GT video for context; `interactive_causal_inference.py`
adds prompt-switching. Neither is the shipped multi-shot path.)

---

## 0. TL;DR numbers

| Quantity | Value | Source |
|---|---|---|
| Base model | Wan2.1-T2V-1.3B (causal fork) | `wan/configs/wan_t2v_1_3B.py`; `shotstream.yaml` |
| Transformer dim / heads / head_dim / layers | 1536 / 12 / 128 / 30 | `wan_t2v_1_3B.py:19-24`; hardcoded `causal_inference_ar.py:22-24` |
| FFN dim | 8960 (GELU-tanh) | `wan_t2v_1_3B.py:20`; `causal_model.py:456-458` |
| Patch size (t,h,w) | (1, 2, 2) | `wan_t2v_1_3B.py:18` |
| VAE stride (t,h,w) | (4, 8, 8), z_dim=16 | `wan_t2v_1_3B.py:16`; `wan_wrapper.py:78` |
| Resolution (fixed) | 480 × 832 (H×W) | `causal_inference_ar.py:133`; `default_config.yaml` |
| Tokens per latent frame (`frame_seq_length`) | **1560** = 30 × 52 | `causal_inference_ar.py:26` (30=60/2, 52=104/2) |
| **Chunk = `num_frame_per_block`** | **3 latent frames** | `shotstream.yaml`; `causal_inference_ar.py:30,181-182` |
| **Latent frames per shot** | **21** (= 7 chunks) | `causal_inference_ar.py:174` (`torch.randn([1,21,...])`) |
| **Pixel frames per shot** | **81** (= (21−1)·4+1), 16 fps ≈ 5.06 s | VAE temporal 4×; `Inference_Causal.py:141` fps=16 |
| **Local cache** | **21 latent frames = 7 chunks** | `causal_inference_ar.py:205` (`num_output_frames*1560`=32760) |
| **Global/context cache** | **6 latent frames = 2 chunks** | `max_context_frames:6` `shotstream.yaml`; `causal_inference_ar.py:209-210` |
| Denoising steps (nominal list) | `[1000, 740, 500, 260]` (**4-step**) | `shotstream.yaml`; `causal_inference_ar.py:265` |
| Denoising steps (WARPED, actual t used) | **[1000.0, 957.93, 888.89, 737.59]** | `model/base.py:479-484` + FlowMatchScheduler (computed) |
| Flow-matching shift | **8.0** | `shotstream.yaml` `timestep_shift`; `wan_wrapper.py:132,149-152` |
| CFG (guidance_scale) | 3.0 in yaml but **NOT used at inference** (used only in DMD training) | `shotstream.yaml`; see §5 |
| `sink_size` / `context_noise` | 0 / 0 | `shotstream.yaml` (no key → 0); `default_config.yaml:context_noise:0` |
| RoPE shot-phase θ | **1/6** rad per shot index | `causal_model.py:70`; `model.py:77` (`theta=1/6`) |
| dtype | bf16 (model/compute); RoPE math in fp64 | `Inference_Causal.py:92`; `causal_model.py:83` |
| Reported perf | ~16 FPS, single NVIDIA H200 | paper Abstract/§5.1/Table 1; README |

> **Port-critical simplification:** with the shipped config the **KV-cache
> rolling/eviction and attention-sink paths are dead code** (see §2.4). Only
> *direct-insert* (local cache) and *context-insert* (global cache) fire, and the
> local window never truncates. You do **not** need to implement the ring-buffer
> roll logic to match ShotStream's public checkpoint.

---

## 1. Chunking, shots, and the VAE

### 1.1 Shot geometry
- A **shot** is a fixed **81 pixel frames** → VAE-encoded (temporal stride 4, `1+4k` rule)
  to **21 latent frames**. The generator always allocates `noise = randn([1, 21, 16, 60, 104])`
  per shot (`causal_inference_ar.py:174`). Latent H=60=480/8, W=104=832/8.
- A shot's 21 latent frames are generated as **7 blocks/chunks of 3 latent frames**
  (`num_blocks = 21 // 3 = 7`, `causal_inference_ar.py:181-182,258`).
- The number of shots comes from the test JSON/CSV (`shot_num_from_caption`, `frame_number`
  ranges); each range is 81 frames wide (`demo/testdata/testset.csv`). The *pixel* content of
  the test video is **not** used by the AR pipeline — only the shot count and captions.

### 1.2 VAE (Wan2.1 3D VAE, `wan/modules/vae.py`, wrapper `utils/wan_wrapper.py`)
- **Latent normalization** is baked into encode/decode via per-channel mean/std
  (`wan_wrapper.py:63-72`): `z = (raw - mean) * (1/std)` on encode; inverse on decode.
  16 channels; the mean/std vectors are the standard Wan2.1 constants.
- **Context/history encode is per-frame:** each history pixel frame is encoded as an
  independent 1-frame clip `[c,1,h,w]` → exactly **1 latent frame** each
  (`wan_wrapper.py:81-99`, loop over frames; `encode_to_latent` called on `f c 1 h w`).
  So 6 history pixel frames → 6 context latents.
- **Shot decode is whole-shot:** `output[1,21,16,60,104]` → `decode_to_pixel(use_cache=False)`
  → 81 pixel frames, then `(x*0.5+0.5).clamp(0,1)` (`causal_inference_ar.py:334-340`).
- **History round-trips through pixels:** decoded pixel frames of prior shots are stored
  (`output_images_list`), then **re-encoded one frame at a time** to build the next shot's
  context (§4). The port must reproduce the pixel→latent round-trip, not cache latents directly.

---

## 2. Dual-cache memory — exact spec

Two *separate* per-layer KV stores, both allocated fresh **each shot**
(`_initialize_kv_cache`, `_initialize_context_kv_cache`, called inside the shot loop).
There are 30 dicts each (one per transformer block). Head layout `[B, tokens, 12, 128]`.

### 2.1 `kv_cache1_context` = **GLOBAL cache** (inter-shot consistency)
- **Stores:** self-attention **K,V of the 6 history/context latent frames**, RoPE-applied,
  computed at **timestep 0 (clean)**. Size = `6 * 1560 = 9360` tokens/layer
  (`causal_inference_ar.py:209-210,372-391`).
- **Written once per shot**, *before* the denoising loop, by a single generator call with
  `kv_cache=None, kv_cache_context=self.kv_cache1_context` → the `context_insert` branch
  (`causal_model.py:385-410`, applied by `_apply_cache_updates_context`, `causal_model.py:770-782`).
  It overwrites the whole buffer: `k[:] = roped_key; v[:] = v`.
- **Never updated during denoising.** Every query chunk in the shot attends to it read-only.
- **Roll/evict policy:** none within a shot. At the shot boundary it is *rebuilt from scratch*
  from freshly re-sampled+re-encoded history (§4). This is the paper's "global cache updated
  with generated content while the local cache is reset."
- Paper mapping: **"global cache of 2 chunks"** = 6 latent frames (2×3). ✅

### 2.2 `kv_cache1` = **LOCAL cache** (intra-shot continuity)
- **Stores:** self-attention **K,V of already-generated chunks of the current shot**,
  RoPE-applied. Allocated size = `num_output_frames * 1560 = 21*1560 = 32760` tokens/layer
  (`causal_inference_ar.py:205,344-370`) = exactly one shot = **7 chunks**. ✅ (paper "local cache of 7 chunks")
- **Update:** chunk *b* writes into fixed slot `[b*4680 : (b+1)*4680]` (`4680 = 3*1560`),
  via `direct_insert` (`causal_model.py:291-322`, applied by `_apply_cache_updates`,
  `causal_model.py:710-768`). Pointers `global_end_index`/`local_end_index` advance by 4680/chunk.
- **Roll/evict policy:** **none for the shipped config** (see §2.4). Reset to zeros each shot.

### 2.3 `crossattn_cache` (text KV, not temporal)
- Per-layer cache of the text-embedding K,V (`_initialize_crossattn_cache`, alloc `[B,512,12,128]`).
  Filled lazily on first block call each shot (`is_init`), reused across all chunks/steps of the shot.
  With multi-caption it holds `[num_shots_so_far, 512, 12, 128]` (§4.3). Reset each shot.

### 2.4 Why rolling/eviction is DEAD CODE here (important for the port)
`CausalWanModel` is built with `local_attn_size=-1, sink_size=0` (`base.py:462-465` →
`from_pretrained(..., local_attn_size=-1, sink_size=0)`; there is no `model_kwargs` in the yaml).
At inference `_set_all_modules_max_attention_size(21)` sets every `self_attn.max_attention_size = 21*1560 = 32760`
**but leaves each `self_attn.local_attn_size == -1`** (it only sets `max_attention_size`,
`causal_inference_ar.py:409-443`). Consequences in `CausalWanSelfAttention.forward`:
- The roll branch guard `if self.local_attn_size != -1 and …` (`causal_model.py:239`) is **never true** → **no `roll_and_insert`, no eviction**.
- `sink_size == 0` → the sink path (`causal_model.py:326-360`) is skipped; the **`else` branch (361-383)** always runs.
- `window_start = max(0, local_end_index - 32760)` is **always 0** within a shot (`local_end_index ≤ 32760`) → the local window is never truncated; a chunk sees *all* prior chunks of its shot.

So the effective per-shot attention set for a query chunk is simply **`[all 6 context frames] ++ [all already-generated frames of this shot] ++ [this chunk itself]`**, ≤ 6+21 = 27 latent frames.

### 2.5 Cache-update pseudocode (default path only)
```
# Called at the END of every generator forward, once, after all 30 blocks ran.

# GLOBAL cache: only when filling context (kv_cache is None, kv_cache_context given)
for layer in 0..29:
    kv_cache_context[layer].k[:] = roped_key_context[layer]   # [1, 9360, 12, 128]
    kv_cache_context[layer].v[:] = value_context[layer]

# LOCAL cache: during the denoising loop (kv_cache given). sink=0, local_attn_size=-1:
for layer in 0..29:
    cur_end   = current_start + 4680          # current_start = block_index*3*1560
    loc_end   = kv_cache[layer].local_end_index + cur_end - kv_cache[layer].global_end_index
    loc_start = loc_end - 4680                # == block_index*4680
    kv_cache[layer].k[:, loc_start:loc_end] = roped_key_current_chunk   # [1,4680,12,128]
    kv_cache[layer].v[:, loc_start:loc_end] = value_current_chunk
    kv_cache[layer].global_end_index = cur_end
    kv_cache[layer].local_end_index  = loc_end
```
Note: each of the 4 denoising steps of a chunk re-clones the cache into a temp buffer, writes
the *current-step* K,V into the temp, attends, then commits to the real cache. The final commit
for a chunk is the **clean rewrite** (§3.3), so the persisted local-cache entry for a chunk is its
**clean (t=0) K,V** — the CausVid/Self-Forcing "clean-context KV" trick.

---

## 3. Causal attention + RoPE — exact spec

### 3.1 Where in the block
Only **self-attention** is causal/cached. Block order per `CausalWanAttentionBlock.forward`
(`causal_model.py:463-538`): `AdaLN-modulate → self_attn(+dual KV cache) → residual →
cross_attn(to text) → FFN(GELU-tanh) → residual`, with 6-way AdaLN modulation
`e = modulation + time_embed` (`causal_model.py:495`). Cross-attention is standard Wan T2V
(frames→text), **not** causal, not temporally cached (§4.3).

### 3.2 The attention op (no explicit mask — structural causality)
Default branch `causal_model.py:361-383`:
```
q      = roped_query(current chunk)            # [1, 4680, 12, 128]
k_cat  = concat([ k_context , k_local ], dim=tokens)   # context FIRST, then local
v_cat  = concat([ v_context , v_local ], dim=tokens)
#   k_context = kv_cache_context.k              (6 frames, 9360 tok)
#   k_local   = temp_k[:, 0 : local_end_index]  (this shot so far, incl. current chunk)
x = attention(q, k_cat, v_cat)                 # plain SDPA / flash-attn, attn_mask=None, is_causal=False
```
`attention()` (`wan/modules/attention.py:150-185`) is **full attention with NO mask**
(`attn_mask=None, is_causal=False`). Causality is enforced *structurally*: the local cache only
contains past+current chunks (future chunks aren't written yet). Granularity is **block-causal**:
- **Within a chunk** (3 latent frames = 4680 tokens): fully bidirectional.
- **Across chunks**: a chunk attends to all earlier chunks of its shot + all 6 context frames; earlier chunks never attend forward (they were computed when later chunks didn't exist in cache).

(The *training* forward `_forward_train` builds an explicit block-causal boolean mask with a
hardcoded sink=3 and condition-visibility rules, `causal_model.py:1037-1060`. That mask is NOT
used at inference — inference relies on cache contents instead. Port the inference path.)

### 3.3 The 3 generator calls per chunk (denoise + clean rewrite)
`causal_inference_ar.py:259-328`. For each of the 7 chunks:
1. **Denoise loop** over the 4 warped timesteps. For step `i<3`: forward with
   `kv_cache=kv_cache1, kv_cache_context=kv_cache1_context` → `denoised_pred`; then re-noise to the
   next timestep: `noisy = scheduler.add_noise(denoised_pred, randn, next_t)`
   (`causal_inference_ar.py:287-293`). For step `i==3`: final forward → `denoised_pred` (the chunk output).
2. **Record** `output[:, chunk] = denoised_pred` (`:310`).
3. **Clean rewrite:** one more forward with `noisy=denoised_pred, timestep=context_noise(=0)`,
   `kv_cache=kv_cache1` (`:313-325`) → overwrites this chunk's local-cache slot with **clean** K,V.
4. `current_start_frame += 3`.

### 3.4 RoPE — dynamic shot-phase (change_rope=True)
Uses `causal_rope_apply_dynamic` (`causal_model.py:70-100`) with angle table
`freqs_dynamic = rope_params_angle(...)` (`model.py:38-45`, raw angles, θ_base=10000; converted to
`torch.polar` after adding shot phase). Head_dim 128 → complex dim c=64, split **[22, 21, 21]** for
(temporal, height, width) (`causal_model.py:75`).

Per the paper (§4.2): for latent `z_t` in shot `k`, temporal rotation **`Θ_t = φ·t + k·θ`**.
In code, the temporal angle for a token at temporal index `p` in shot `k`:
```
angle_temporal[j] = p * (1 / 10000^(2j/ d_temporal))  +  k * (1/6)      for j in 0..21
```
i.e. the shot index `k` (`shot_flags_for_rope`) adds a **uniform phase of k/6 rad** to *all 22*
temporal channels (`causal_model.py:86`: `freqs[0][start:start+f] + shot_flags*theta`), θ=1/6.
Height/width RoPE are unchanged.

**Position assignment (per shot — bounded, does NOT grow globally):**
- Context frames: temporal positions **0..5** (`start_frame=0` in the context_insert call,
  `causal_model.py:396`), each with its *own* shot's phase `k·θ` (e.g. shot-1 build: flags `[0,0,0,0,0,0]`;
  shot-2 build: `[0,0,0,1,1,1]`).
- Current generated chunk: `start_frame = current_start_frame + condition_start_frame`
  where `condition_start_frame = 9360/1560 = 6` (`causal_model.py:196-198,210`). So generated
  frames occupy temporal positions **6 .. 26** across the shot's 7 chunks, all with the *current*
  shot's phase `k=latent_gen_iter` (flags `[k,k,k]`).

Key port note: because caches reset each shot and `current_start_frame` resets to 0, **absolute
RoPE positions are bounded to [0,26] every shot** — shots are separated in *phase* (k·θ), not by an
ever-growing position index. `shot_flags_for_rope` is the per-frame shot-index vector; build it as
`[context frame shot-ids...] + [k]*21` (`causal_inference_ar.py:104-177`).

*(Optional variant `use_wo_rope_cache=True`, default False: store un-RoPE'd K in cache and re-apply
RoPE at read time, clamping the generated-frame position to `≤ condition_start+18`. Not the default
inference path — implement only if you need it.)*

---

## 4. Next-shot conditioning — exact spec

Per shot `latent_gen_iter` (`causal_inference_ar.py:82-255`):

### 4.1 Dynamic history-frame sampling (builds the global cache)
- `max_context_frames = 6` frames are split across all prior shots
  (`causal_inference_ar.py:87-101`): `base = 6 // num_prior_shots`, remainder added to the *last*
  shots. e.g. 1 prior → `[6]`; 2 prior → `[3,3]`; 5 prior → `[1,1,1,1,2]`. Total is asserted == 6.
- Within each prior shot, indices are `torch.linspace(start,end,steps=count).round()`
  (`:114-120`) over that shot's *pixel* frames in `output_images_list`.
- **Shot 0 has no history:** `condition_frames = zeros([6,480,832,3])` and, after encode,
  `condition_latents` is explicitly **zeroed** (`:131-133,160-162`). Shot 0 is effectively a
  text-only self-forcing T2V of one shot.

### 4.2 Encoding + injection
- Sampled pixel frames → per-frame VAE encode → **6 clean context latents** `[1,6,16,60,104]` (§1.2).
- Injected **not** by frame-concat into the DiT input, but by being **written into the global KV
  cache at timestep 0** (§2.1) and attended to (read-only) by every query chunk (§3.2). i.e.
  conditioning = *clean KV in the global cache*, prepended to the key sequence.
- (The "FrameConcat" in class/paper naming refers to the **teacher/training** design where context
  latents are concatenated along the frame axis, `pipeline/causal_inference.py`; the shipped causal
  AR generator realizes the same conditioning via the global cache instead of literal concat.)

### 4.3 Text conditioning (multi-caption)
- Prompt for shot i = `global_caption + shots_captions[i]` (string concat, `causal_inference_ar.py:165-171`).
  `shotN` local captions are prefixed `"shotN:"` (`dataset.py:167`).
- With `multi_caption=True`, the text encoder is fed the **list of captions for shots 0..k**
  → `prompt_embeds [k+1, 512, 4096]` (`causal_inference_ar.py:165-188`).
- In cross-attention (`model.py:206-286`), when `context.shape[0] > 1`, **each latent frame selects
  its own shot's caption** via `k = k[shot_flags_for_rope]; v = v[shot_flags_for_rope]`
  (`model.py:278-279`). So a generated chunk (flags `[k,k,k]`) attends to shot k's caption; context
  frames attend to their originating shot's caption. Text encoder = UMT5-XXL, seq_len 512
  (`wan_wrapper.py:20-36`). Negative prompt exists in config but is **not applied at inference**
  (no CFG; §5).

---

## 5. Sampler / distillation — exact spec

### 5.1 Scheduler
`FlowMatchScheduler(shift=8.0, sigma_min=0.0, extra_one_step=True)`, `set_timesteps(1000, training=True)`
(`wan_wrapper.py:149-152`). Rectified-flow forward: `x_t = (1−σ)·x0 + σ·noise`
(`scheduler.py:161-180`, `add_noise`). σ grid built as
`σ_raw = linspace(1.0, 0.0, 1001)[:-1]`, then **shifted** `σ = 8σ/(1+7σ)`, `t = 1000·σ`
(`scheduler.py:118-135`).

### 5.2 The 4 few-step timesteps (WARPED — this is what the model sees)
Nominal `denoising_step_list = [1000, 740, 500, 260]`, then `warp_denoising_step=True`
(`base.py:479-484`):
```
timesteps_ext = concat(scheduler.timesteps, [0])          # length 1001
denoising_step_list = timesteps_ext[ 1000 - [1000,740,500,260] ]
                    = timesteps_ext[ [0, 260, 500, 740] ]
```
Computed values (verified numerically, shift=8):

| step i | nominal | index | **actual t** | **σ** |
|---|---|---|---|---|
| 0 | 1000 | 0   | **1000.000** | 1.00000 |
| 1 | 740  | 260 | **957.929**  | 0.95793 |
| 2 | 500  | 500 | **888.889**  | 0.88889 |
| 3 | 260  | 740 | **737.589**  | 0.73759 |

The loop is monotonically *descending* in t (high-noise regime only). Between steps the x0 estimate
is re-noised to the next t with `add_noise` using **fresh** `randn` (`causal_inference_ar.py:287-293`).

### 5.3 Prediction & x0 conversion
Model predicts **flow** (`v = noise − x0`). x0 recovered as `x0 = x_t − σ_t · flow`
(`wan_wrapper.py:184-208`, `_convert_flow_pred_to_x0`; done in fp64). `denoised_pred` returned by the
generator wrapper *is already x0* (`wan_wrapper.py:325-334`). Per-frame timestep tensor is
`[B, num_frames]` all set to the same current t (`causal_inference_ar.py:268-271`).

### 5.4 CFG
**No classifier-free guidance at inference.** The generator is called once per step (single forward,
no uncond branch); `guidance_scale=3.0` in the yaml is consumed only inside DMD *training*
(`model/dmd*.py`). Do **not** implement CFG in the port's sampling loop.

### 5.5 Distillation lineage (context; not run at inference)
Two-stage self-forcing **DMD** (Distribution Matching Distillation), student = 4-step causal
generator, teacher ≈ many-step bidirectional Wan next-shot model. Pipeline (README + `tools/train`):
1. **Bidirectional next-shot teacher** (`1_basemodel`).
2. **Causal init** from teacher via ODE-regression on ~5K teacher ODE pairs (`Teacher_Ode_Sample.py`,
   `2_ode_init`, CausVid-style).
3. **Stage 2.1 — intra-shot** self-forcing DMD (`3_dmd.yaml`, trainer `score_distillation_frameconcat`).
4. **Stage 2.2 — inter-shot** self-forcing DMD on multi-shot rollouts (`4_dmd_long.yaml`, trainer
   `score_distillation_frameconcat_stream`), LoRA rank/alpha 256.
Both DMD stages: `num_frame_per_block=3`, warped `[1000,740,500,260]`, shift 8, gs 3,
`context_noise=0` — i.e. **the inference recipe matches the distillation recipe** (as required for
self-forcing consistency). The shipped weights are `shotstream_merged.pt` (LoRA merged).

---

## 6. Full streaming inference loop (faithful pseudocode)

```
# Entry: Inference_Causal.py -> CausalInferenceArPipeline.inference(batch)
# Fixed: chunk=3, frames/shot=21, blocks/shot=7, ctx=6, res 480x832, 30 layers, 12 heads x128
# warped_ts = [1000.0, 957.929, 888.889, 737.589]

shots            = unique(batch.shot_flag)          # e.g. 5 or 6 shots
output_pixels    = []                               # decoded pixel frames of finished shots
shot_flags_out   = []                               # per finished pixel-frame -> shot id

for k in range(len(shots)):                         # ---- OUTER: one iteration per shot ----
    # 1) build history context frames (global cache source)
    if k == 0:
        cond_pix = zeros([6, 480,832,3])            # no history
    else:
        counts   = split_evenly(6, over=k prior shots)     # e.g. k=2 -> [3,3]
        idxs     = concat_over_prior_shots(linspace(start,end,counts_j).round())
        cond_pix = pixels_of_prior_shots[idxs]      # 6 frames sampled from output_pixels
    cond_lat = [ VAE.encode(frame[c,1,h,w]) for frame in cond_pix ]   # 6 clean latents [1,6,16,60,104]
    if k == 0: cond_lat = zeros_like(cond_lat)

    # 2) text: per-shot captions 0..k ; each frame later picks its own via shot_flags
    prompts   = [ global_cap + local_cap[i] for i in 0..k ]           # multi_caption
    text_kv   = UMT5(prompts)                                         # [k+1,512,4096]

    # 3) rope shot-ids: context frames keep their shot id, generated frames = k
    shot_flags = [shot_id_of(each cond frame)] + [k]*21               # len 6+21

    # 4) fresh caches for this shot
    kv_local   = zeros(30 x [1, 21*1560, 12,128])   # LOCAL  = 7 chunks
    kv_ctx     = zeros(30 x [1,  6*1560, 12,128])   # GLOBAL = 2 chunks
    cross_kv   = empty(30)                           # text kv, lazy init

    # 5) fill GLOBAL cache once (context_insert), clean t=0, positions 0..5, phase per-shot
    generator(cond_lat, text_kv, t=0, kv_cache=None, kv_cache_context=kv_ctx,
              crossattn_cache=cross_kv, shot_flags=shot_flags[:6], current_start=0)

    noise  = randn([1,21,16,60,104])
    output = zeros([1,21,16,60,104])
    start  = 0
    for b in range(7):                              # ---- INNER: 7 chunks of 3 latent frames ----
        x = noise[:, start:start+3]
        for i, t in enumerate(warped_ts):           # ---- 4-step causal denoise ----
            _, x0 = generator(x, text_kv, t=t, kv_cache=kv_local, kv_cache_context=kv_ctx,
                              crossattn_cache=cross_kv, shot_flags=shot_flags[6+start : 6+start+3],
                              current_start=start*1560)     # writes noisy K,V into local slot b
            if i < 3:
                x = add_noise(x0, randn_like(x0), warped_ts[i+1])   # renoise to next t
        output[:, start:start+3] = x0               # chunk output (x0 at last step)
        # clean rewrite: persist CLEAN K,V of this chunk into local cache
        generator(x0, text_kv, t=0, kv_cache=kv_local, kv_cache_context=kv_ctx,
                  crossattn_cache=cross_kv, shot_flags=shot_flags[6+start:6+start+3],
                  current_start=start*1560)
        start += 3

    # 6) decode this shot to pixels, append; becomes history for later shots
    pix = VAE.decode(output)                         # 21 latent -> 81 pixel frames
    output_pixels.append(pix);  shot_flags_out += [k]*81

video = concat(output_pixels); video = (video*0.5+0.5).clamp(0,1)     # write @ 16 fps
```

Inside each `generator(...)` self-attention (default path), for the current chunk query:
```
q      = rope_dyn(Wq·x, pos = 6 + start .. , phase = k/6)
k_cur  = rope_dyn(Wk·x, ...)              # written into kv_local[b*4680 : (b+1)*4680]
k_cat  = concat(kv_ctx.k[:9360], kv_local.k[:(b+1)*4680])   # ctx first, then local
v_cat  = concat(kv_ctx.v,        kv_local.v[:(b+1)*4680])
attn   = softmax(q·k_catᵀ/√128) · v_cat  # NO mask (full attention over the concatenated keys)
```

---

## 7. Ambiguities, tiebreakers, and gaps

**Repo is the tiebreaker (paper under-specifies):**
- Paper says "global cache of 2 chunks / local cache of 7 chunks" and "chunk of 3 latent frames"
  (§5.1) — repo confirms exactly: ctx=6 latents (`max_context_frames`), local=21 latents
  (`num_output_frames`), chunk=`num_frame_per_block=3`.
- Paper's RoPE "`Θ_t = φt + kθ`" (§4.2) gives no θ value; **repo pins θ=1/6** (`model.py:77`,
  `causal_model.py:70`) and shows it's added uniformly to all 22 temporal channels.
- Paper implies KV caching for efficiency but doesn't give the eviction rule; **repo shows the
  shipped config never evicts** (§2.4) — the roll/sink code is inert. Port the simple path.
- Paper's "conditioning" is described abstractly; **repo shows it's clean-KV in a separate global
  cache at t=0**, plus multi-caption per-frame cross-attn routing (`k[shot_flags]`), not literal
  input frame-concat (that's the teacher).
- 4-step schedule: paper says "4-step"; repo pins the **warped** timesteps
  `[1000.0, 957.93, 888.89, 737.59]` (not the nominal `[1000,740,500,260]`). Use the warped values.

**Paper is the tiebreaker / adds context the repo doesn't:** perf (16 FPS, sub-second latency,
H200), the conceptual global/local roles, and that the recipe stems from a bidirectional teacher +
2-stage DMD. None of that changes the inference ops.

**Could NOT fully determine / verify (flagged):**
- Exact **Wan2.1 VAE** internal architecture (encoder/decoder conv stack, temporal causal padding)
  was not read line-by-line here (`wan/modules/vae.py`, 683 lines) — the port needs the standard
  Wan2.1 3D-VAE, stride (4,8,8), z_dim 16, with the mean/std in `wan_wrapper.py:63-70`. Treat the
  existing Wan2.1 VAE port as the reference; only the *per-frame encode* / *whole-shot decode* call
  pattern (§1.2) is ShotStream-specific.
- `is_recompute` logic (`causal_model.py:238,264,300`) only matters if a committed chunk is
  re-run with `current_start>0 & current_end≤global_end`; the AR loop never does this, so it stays
  False. Not needed for the port.
- The paper's arXiv HTML was parsed by a summarizer; exact §-numbers above (4.1/4.2/5.1) are as
  reported by that fetch and should be spot-checked against the PDF if precise citation is needed —
  but **all numeric/algorithmic claims here are grounded in the repo**, which is authoritative.
- `shotstream.yaml` sets `data_path/output_folder: None`; the runnable values come from
  `tools/inference/causal_fewsteps.sh` (`testset.csv`, `demo/infer/`) and CLI overrides.
```

---

## 8. Quick op-inventory for the ggml port

Per transformer block (30×), inference self-attn path needs:
`Linear q/k/v/o (1536→1536)`, `RMSNorm(q), RMSNorm(k)` (WanRMSNorm, fp32 reduce), **dynamic complex
RoPE** on q/k (temporal 22 pairs + shot phase, H 21, W 21), **maskless SDPA** over concat(ctx,local)
keys, `WanLayerNorm` (no affine) ×3, 6-way AdaLN modulation from `time_projection`, T2V cross-attn
(q from x, k/v from text, per-frame caption gather), FFN `1536→8960 →(GELU tanh)→ 1536`.
Plus: `patch_embedding` Conv3d k=s=(1,2,2), `time_embedding` (sinusoidal freq_dim 256 → SiLU MLP),
`text_embedding` (4096→1536 GELU-tanh MLP), `CausalHead` (LayerNorm+2-way AdaLN+Linear→ unpatchify).
All bf16 except RoPE/x0-conversion in fp64. Flow→x0 = `x_t − σ_t·flow`.
