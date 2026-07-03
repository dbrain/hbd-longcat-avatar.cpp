# HANDOFF — Port KlingTeam ShotStream (streaming multi-shot) into the C++/ggml Wan engine

Worktree: `/home/dbrain/dev/longcat-avatar-shotstream` (branch `shotstream`, fork of stable-diffusion.cpp off `wan22-infinitetalk`).
Status: DiT converted (`tools/convert_shotstream_pt.py`) and loads via `WanRunner` (`src/model/diffusion/wan.hpp`), but it runs through the **standard bidirectional Wan T2V sampler** (`-M vid_gen`). That is NOT ShotStream behaviour. This document specifies the causal-streaming + dual-cache inference port.

Sources: paper `arxiv.org/html/2603.25746v1`; repo `github.com/KlingAIResearch/ShotStream` (`pipeline/causal_inference_ar.py`, `wan/modules/causal_model_change_rope.py`, `wan/configs/wan_t2v_1_3B.py`); HF configs `shotstream.yaml` / `default_config.yaml`.

> Planning doc only. No code was changed and no GPU was run (a 3060 render is in progress — leave the GPU alone).

---

## 1. Algorithm spec (exact)

### 1.1 Base model (unchanged from vanilla Wan2.1-T2V-1.3B)
Detected by `WanConfig` at `wan.hpp:1148-1177`: **30 blocks, dim 1536, ffn 8960, 12 heads, head_dim 128, in_dim=out_dim=16, patch (1,2,2), freq_dim 256, text_len 512, eps 1e-6, theta 10000, axes_dim {44,42,42}**. umT5-xxl text encoder, Wan 3D causal VAE (stride **(4,8,8)** — 4× temporal, 8× spatial, from `wan_t2v_1_3B.py`). ShotStream only changes the *inference procedure* + adds RoPE-phase / cache logic; the weights are stock flat-Wan naming (confirmed by the converter's `generator.model.` strip).

### 1.2 Chunk / streaming granularity
- **`num_frame_per_block = 3`** latent frames per chunk (`shotstream.yaml`). At VAE 4× temporal that is ~12 pixel frames/chunk (first chunk maps `(T-1)/4+1`). At 16 fps a chunk is ~0.75 s → sub-second latency target.
- **`frame_seq_length = 1560`** tokens per latent frame, hard-coded in `causal_inference_ar.py`. This is `(H/8/2)·(W/8/2) = 30·52` at 480×832. Resolution-general: `frame_seq_length = (H/16)·(W/16)` (patch halves each spatial dim after the 8× VAE).
- Output shot loop (`causal_inference_ar.py`):
  ```
  num_blocks = num_output_frames // num_frame_per_block
  current_start_frame = 0
  for current_num_frames in [num_frame_per_block]*num_blocks:   # =3
      noisy_input = noise[:, current_start_frame : current_start_frame+3]
      for index, current_timestep in enumerate(denoising_step_list):   # 4-step DMD
          timestep = ones([b,3]) * current_timestep
          _, denoised_pred = generator(noisy_input, cond, timestep,
                                       kv_cache=kv_cache1, kv_cache_context=kv_cache1_context,
                                       crossattn_cache=crossattn_cache,
                                       current_start=current_start_frame*frame_seq_length, ...)
          if index < len-1:
              next_t = denoising_step_list[index+1]
              noisy_input = scheduler.add_noise(denoised_pred, randn_like, next_t)  # RE-NOISE
      output[:, current_start_frame:+3] = denoised_pred
      current_start_frame += 3
  ```

### 1.3 Few-step DMD schedule
- **`denoising_step_list = [1000, 740, 500, 260]`** (4 steps), **`warp_denoising_step = True`**, **`timestep_shift = 8.0`**, `num_train_timestep = 1000` (`shotstream.yaml`).
- Each chunk is denoised **independently** with these 4 steps; between steps the predicted `x0` is **re-noised** to the next timestep via `scheduler.add_noise` (flow-match), NOT euler-integrated. This is the DMD/self-forcing pattern.
- `warp_denoising_step`: the raw integer timesteps are remapped through the shifted flow schedule `t' = shift·t/1000 / (1 + (shift-1)·t/1000) · 1000` (shift=8) before use — i.e. the `[1000,740,500,260]` are indices into the *unshifted* 1000-step grid, then shifted. **The exact warp/sigma mapping must be read from `wan/utils/fm_solvers*.py` / `utils/scheduler.py` (not fully extracted here) — see Risks.**
- Guidance: distilled student is single-forward per step (`guidance_scale=3.0` in the yaml is the *teacher* guidance baked into distillation; inference is effectively **cfg=1**, no uncond pass). The engine's `--cfg-scale 1` path already matches.

### 1.4 Dual-cache memory (the core new mechanism)
Two separate per-layer KV caches (each a list of 30 dicts `{k,v,global_end_index,local_end_index}`), tensors shaped `[batch, kv_cache_size, num_heads=12, d=128]`:

| Cache | Repo name | Size | Contents | Roll policy |
|---|---|---|---|---|
| **Local** (intra-shot) | `kv_cache1` | `local_attn_size · frame_seq_length` = **21·1560 = 32760 tokens = 7 chunks × 3 frames** (`local_attn_size=21`, `causal_inference_ar.py`; `max_attention_size = 32760 if max_local==-1`) | RoPE'd K + raw V of **already-finalized** chunks in the current shot | sliding window: evict oldest, keep `sink_size` earliest frames pinned |
| **Global / context** (inter-shot) | `kv_cache1_context` | `condition_frame_numbers · frame_seq_length` (paper: **2 chunks** of sparse historical frames; `max_context_frames`=6 in `shotstream.yaml`, default 10 in code) | RoPE'd K + raw V of **VAE-encoded historical frames** from prior shots, prefilled ONCE per shot at `context_noise=0` (clean) | rebuilt at each shot boundary from newly-sampled condition frames |

- **Local cache roll/evict** (`causal_model_change_rope.py`, inference branch):
  ```
  num_evicted = num_new_tokens + local_end_index - kv_cache_size
  num_rolled  = local_end_index - num_evicted - sink_tokens
  temp_k[:, sink:sink+num_rolled] = temp_k[:, sink+num_evicted : sink+num_evicted+num_rolled].clone()
  ```
  `sink_tokens = sink_size · frame_seq_length` (earliest frames never evicted). During attention:
  ```
  local_budget = max_attention_size - sink_tokens
  local_start  = max(sink_tokens, local_end_index - local_budget)
  k_cat = cat([k_sink, temp_k[:, local_start:local_end_index]], dim=1)
  ```
- **Context prefill** (once per shot, `causal_inference_ar.py`): a single generator pass over `condition_latents` with `timestep=context_noise(=0)`, `kv_cache=None`, `kv_cache_context=kv_cache1_context`, `current_start=0`, `shot_flags_for_rope[:condition_latents.shape[1]]`. Fills the global cache; never re-noised.
- **Cross-attn (text) cache**: `crossattn_cache[block] = {k:zeros[b,512,12,128], v:…, is_init:False}`; text K/V computed once per shot (512 padded tokens), reused across all chunks/steps of the shot.

### 1.5 Attention mask / who-attends-to-whom
Training mask (`causal_model_change_rope._forward_train`), which the inference cache windowing *realises implicitly*:
```
mask[:, cond_seq : cond_seq + (i+1)*3*fs] = i+1     # block i tokens tagged i+1
mask = (rowval >= colval); mask[..., :cond_seq] = True
```
⇒ **block-causal: BIDIRECTIONAL within a chunk (all 3 frames share tag i+1), CAUSAL across chunks (block i sees blocks ≤ i), and ALL chunks always attend the condition/global tokens.** At inference there is **no explicit mask** — causality is enforced by the cache only containing past+current tokens. So each chunk's self-attention is dense over `[global_cache ++ local_window ++ its_own_fresh_KV]`.

### 1.6 RoPE / temporal positions
- Per-chunk offset: `current_start_frame = current_start // frame_seq_length`; `causal_rope_apply(q, grid, freqs, start_frame=current_start_frame)`. Context frames use `start_frame + condition_start_frame`.
- **`change_rope=True` shot phase shift** (paper §): for latent `z_t` in shot `k`, temporal angle `Θ_t = φ·t + k·θ` — a discrete jump `θ` at each shot boundary that decouples global (historical) vs local (current-shot) context. Carried via `shot_flags_for_rope`. **The concrete `θ` value + `causal_rope_apply` are in `causal_model_change_rope.py` (only partially extracted) — read before implementing (Risk).**

### 1.7 Ambiguities to resolve from the repo before coding
1. Exact warp/shift sigma mapping for `[1000,740,500,260]` (`fm_solvers*.py`).
2. Exact `add_noise` re-noise formula between DMD steps.
3. Whether the local cache is written **every** DMD step or **only the final** step (the S2V C++ driver snapshots/restores so only the final step persists — almost certainly correct here too; confirm against the `current_step`/cache-write gate in `_forward_inference`).
4. `θ` phase value and `sink_size` / `condition_frame_numbers` defaults (paper says global=2 chunks; code default `max_context_frames=10`).
5. `condition_indices` sampling (`dynamic_sample_frames`: `⌊f_context/S_hist⌋` frames per historical shot).

---

## 2. Gap analysis (current code vs ShotStream)

Current `-M vid_gen` flow, all bidirectional/whole-sequence:
- Entry `generate_video_ex` (`stable-diffusion.cpp:7367`) → `prepare_video_generation_latents` (`:5761`) builds ONE latent `[Wl,Hl,T,16]` for **all** frames → `sample()` (`:2407`) runs the full sigma schedule (`plan.sigmas`) over the whole clip → per step the denoise lambda (`:2503`) calls `WanRunner::compute` (`wan.hpp:1305`).
- `WanRunner::build_graph` (`wan.hpp:1232`) builds **one full-sequence RoPE** via `Rope::gen_wan_pe` (`:1250`, `rope.hpp:723`) and calls `Wan::forward` → `forward_orig` (`wan.hpp:836`).
- `forward_orig` loops all 30 blocks (`:1019`), each `WanAttentionBlock::forward` (`:509`) whose self-attn is `WanSelfAttention::forward` (`:119`) — **dense, bidirectional, mask=nullptr** (the `flash_skip_kv_pad` path `:223-227`). No chunking, no cache, no causal windowing, no per-shot state.

What ShotStream needs, mapped to what already exists:

| ShotStream need | Exists? | Where |
|---|---|---|
| Causal self-attn with rolling KV cache `[prev ++ cur ++ cond]` | **YES (reusable)** | `WanSelfAttention::forward_kv_cache` (`wan.hpp:238-307`) — attends over concat of prev/cur/cond, exports `new_kc/new_vc`. Currently mask=nullptr = full attention over the cache = exactly right. |
| Context/global cache prefill (K/V only, no attn output) | **YES (reusable)** | `WanSelfAttention::prefill_cond_kv` (`wan.hpp:312-324`) |
| Per-chunk block forward threading per-layer prev/cond caches + exporting new K/V | **YES (as S2V analog)** | `WanS2V::forward_causal_block` (`wan_s2v.hpp:795`) + `WanS2VAttentionBlock::forward_causal` (`:210`) — but these are S2V-specific blocks; mirror onto plain `WanAttentionBlock`. |
| Host-side rolling cache + window eviction (F16, graph-cut offload) | **YES (as S2V analog)** | `WanS2VRunner::compute_causal_block` (`wan_s2v.hpp:1210`), `cache_k/cache_v` (`:1185`), `S2V_KV_WINDOW_BLOCKS` slice (`:1401-1412`), `cond_k_cache`/`prefill_cond_host` (`:1191/:1073`). |
| Per-chunk RoPE offset | **YES (reusable)** | `Rope::gen_vid_ids(t,h,w,pt,ph,pw,1,t_offset,0,0)` used by `WanS2V::build_pe_frames` (`wan_s2v.hpp:1160`); anchor positions `Rope::gen_vid_ids_ref` (`rope.hpp:746`). |
| Chunked outer loop + per-chunk denoise + cache-snapshot-between-steps | **YES (as S2V analog)** | `examples/s2v/main.cpp:576-643` (the exact streaming skeleton). |
| Next-shot / history-frame conditioning (VAE-encode tail, carry across shots, re-encode) | **PARTIAL (reusable)** | `generate_video_chain` (`stable-diffusion.cpp:8219`): host-side latent tail carryover (`cont_buf`, `vp.cont_latent`/`cont_latent_frames`, `:8384-8398`) + `LONGCAT_CONT_REENCODE` decode→re-encode (`:8449`). Reuse for shot-to-shot context frames. |
| DMD 4-step re-noise schedule (`add_noise` between steps) | **NO** | new; S2V uses euler velocity integration (`examples/s2v/main.cpp:621-629`). |
| `change_rope` shot phase shift `Θ=φt+kθ` | **NO** | new small helper on the RoPE id generation. |
| Multi-caption per-shot prompt switch + crossattn cache reset | **PARTIAL** | text pre-encode exists (`sd_ctx_precompute_chain_text_conds`, `:8283`); need per-shot swap + cache reset. |

Net: the **primitives exist** (`forward_kv_cache`, `prefill_cond_kv`, host rolling cache pattern, RoPE offset, chain carryover). The port is **wiring them into the plain-Wan blocks + a new streaming driver + the DMD schedule + shot state**, not building new subsystems.

---

## 3. Implementation plan (staged, each stage independently testable)

Create a new runner header `src/wan_shotstream.hpp` (namespace `WAN_SHOTSTREAM`) modeled on `wan_s2v.hpp` but composing the stock `WAN::Wan` blocks, and a driver `examples/shotstream/main.cpp` cloned from `examples/s2v/main.cpp`. Keep everything behind a new `-M vid_gen` sub-path / example so `vid_gen` stays byte-identical.

### Stage A — causal self-attn on plain Wan blocks (single chunk, empty caches)
Goal: one 3-frame chunk through a causal block forward that routes self-attn through `forward_kv_cache`, no history.

(a) In `wan.hpp`, add `WanAttentionBlock::forward_causal(ctx, x, e0, pe_block, context, prev_kc, prev_vc, cond_kc, cond_vc, new_kc, new_vc)` — a copy of `forward` (`:509-572`) whose self-attn line (`:551`) calls the existing `self_attn->forward_kv_cache(...)` (`wan.hpp:238`) instead of `forward`, threading the per-layer caches and exporting `new_kc/new_vc`. Everything else (norm1, adaLN modulate `es[*]`, cross-attn, ffn, gates) is unchanged. Modulation `e` here is per-chunk `[N,6,dim]` (3 frames share one timestep, so the `T`-broadcast path in `modulate_add/mul` `:443-469` is a no-op — simpler than S2V's per-frame `e0_2`).

(b) Add `WAN::Wan::forward_causal_block(ctx, x_chunk, timestep, context, pe_block, prev_kc[], prev_vc[], cond_kc[], cond_vc[], new_kc[], new_vc[])` mirroring `WanS2V::forward_causal_block` (`wan_s2v.hpp:795-870`): patch-embed the chunk, build `e/e0`, text-embed context, loop 30 blocks calling `forward_causal`, collect per-layer new K/V, head+unpatchify to velocity/x0. Mark each block's residual + K/V as graph-cut outputs (`mark_graph_cut`, as `wan_s2v.hpp:1316-1327`) so the offload/segmented compute path applies.

(c) Add `WAN_SHOTSTREAM::ShotStreamRunner::compute_causal_block(...)` mirroring `wan_s2v.hpp:1210-1414`: host-side per-layer `cache_k/cache_v` (F16), `make_input` the prev/cond caches, run via `GGMLRunner::compute<float>` with the per-segment readback hook (`:1355`), append new K/V, apply the window slice.

Test: run block b=0 with empty caches, compare the denoised x0 to a torch oracle chunk-0 (`_forward_inference`, empty cache) — cosine ≥ 0.999. Also sanity-check vs the dense `vid_gen` on a 3-frame clip (should be close since chunk 0 has no causal history).

### Stage B — rolling local cache across chunks (single shot, no context)
(a) Driver: clone the `examples/s2v/main.cpp:576-643` skeleton: `causal_reset_cache()` → `for b in blocks`: `t_off=b·3`, slice `xb`, **snapshot cache before the chunk**, then `for i in 4 DMD steps`: **restore snapshot**, `compute_causal_block(t_off, xb, t=denoising_step_list[i])`, re-noise `xb` to `denoising_step_list[i+1]` (Stage-C schedule; for now a placeholder euler is fine to smoke the cache). The snapshot/restore (`main.cpp:619-623`) guarantees only the FINAL step's K/V persists into the local cache.
(b) `build_pe_frames` per chunk via `Rope::gen_vid_ids(...,t_offset=t_off,...)` (`wan_s2v.hpp:1160`).
(c) Window = 7 chunks: reuse the `S2V_KV_WINDOW_BLOCKS`-style slice (`wan_s2v.hpp:1401-1412`) with default **7** (= `local_attn_size/num_frame_per_block = 21/3`). Add `sink_size` pinning of the earliest `sink_size·1560` tokens.

Test: multi-chunk single shot; compare decoded frames + per-chunk denoised latents to a torch multi-chunk run (cache on). Watch for drift after chunk 8 (window eviction boundary).

### Stage C — DMD few-step schedule (exact)
Replace the placeholder step with the real DMD loop: map `[1000,740,500,260]` through `warp_denoising_step` + `timestep_shift=8` to sigmas (port `fm_solvers*.py`/`scheduler.py`), predict x0 each step, `add_noise(x0, randn, next_t)` between steps. This is a small standalone function; unit-test the sigma grid against a torch dump of `scheduler.timesteps`/`sigmas` before wiring.

Test: single-chunk 4-step output must match torch bit-close on the same seed/noise.

### Stage D — global/context cache + next-shot conditioning
(a) `prefill_context_host(...)` mirroring `WanS2VRunner::prefill_cond_host` (`wan_s2v.hpp:1073-1155`) but over N historical latent frames at `timestep=0` (uses `prefill_cond_kv`, `wan.hpp:312`), populating `cond_k_cache/cond_v_cache` (the global cache). Grid positions via `gen_vid_ids_ref` (`rope.hpp:746`) or the context RoPE offset.
(b) Cross-shot frame carryover: reuse `generate_video_chain`'s host-side tail capture (`stable-diffusion.cpp:8289-8398`) — take the previous shot's decoded/last frames, **VAE-encode** them (reuse `LONGCAT_CONT_REENCODE` re-encode path, `:8449`) to `condition_latents`, sample `condition_indices` (dynamic), prefill the global cache at each shot start; **reset the local cache** (`causal_reset_cache`) at the boundary.
(c) `condition_frame_numbers`/global = 2 chunks (confirm vs `max_context_frames`).

Test: 2-shot generation; verify inter-shot identity/scene coherence and that the boundary is clean.

### Stage E — shot RoPE phase + multi-caption + crossattn cache
(a) `change_rope`: add shot index `k` phase `Θ=φt+kθ` into the id generation (small addend on the temporal id before `embed_nd`, `rope.hpp:733`). Gate behind a flag; default matches single-shot.
(b) Multi-caption: per-shot prompt → re-encode umT5 (reuse `sd_ctx_precompute_chain_text_conds`, `:8283`), reset the crossattn (text) cache at the shot boundary. Optionally add a text-K/V cache to `WanT2VCrossAttention` (`wan.hpp:340`) with an `is_init` flag to skip recompute within a shot (perf only; recompute is numerically identical).
(c) Interactive/streaming prompt input can come later.

### Stage F — CLI/config + perf
- CLI: add a `--shotstream` sub-mode (or `examples/shotstream/main.cpp` like `examples/s2v/`) exposing `--num-frame-per-block 3`, `--local-attn-size 21`, `--sink-size`, `--context-frames`, `--denoising-steps 1000,740,500,260`, `--timestep-shift 8`, `--shots N`, per-shot prompt list, `--cfg 1`. Config surface can also read `shotstream.yaml` keys directly.
- Perf comes free from the S2V-inherited machinery: F16 host cache (`wan_s2v.hpp:1185`), graph-cut CPU-weight offload (`mark_graph_cut`), window bounding. The `WAN_DIT_F16` residual-stream lever (`wan.hpp:869-893`) and cuDNN flash apply unchanged.

---

## 4. Validation (cheap → expensive)

1. **Torch oracle, chunk-by-chunk** (the gate): clone the official repo, run `Inference_Causal.py` with `shotstream.yaml` on a fixed seed, and dump: the DMD `sigmas`/`timesteps`, the RoPE `freqs`, per-block `kv_cache` K/V after chunk 0 and chunk 8, `denoised_pred` per chunk, and the final latent. Diff our tensors op-for-op. This localizes drift (schedule vs cache-roll vs RoPE) — critical because multi-chunk failures are otherwise invisible until decode.
2. **Stage-A single-chunk parity**: cosine ≥ 0.999 on chunk-0 denoised latent vs oracle (empty cache).
3. **Stage-B/C multi-chunk**: per-chunk latent cosine vs oracle; then decode and compare frames (SSIM) — expect graceful degradation only past the 7-chunk window, matching torch.
4. **End-to-end**: diff decoded mp4 vs the repo's `demo/data/sample/0001.mp4` at matched seed/prompt; owner eye-test for multi-shot coherence (the ultimate gate, per project practice).
5. **VAE**: reuse the `S2V_REF_ROUNDTRIP` diagnostic pattern (`examples/s2v/main.cpp:399`) to isolate VAE encode/decode of history frames from DiT errors. Wan-VAE temporal streaming already exists (`WAN_VAE_*`, `LONGCAT_VAE_TEMPORAL_CHUNK`).

---

## 5. Risks / unknowns + rough effort

| Stage | Effort | Risk |
|---|---|---|
| A (causal block on plain Wan) | ~1 day | Low — `forward_kv_cache` already proven in S2V; just re-target from S2V blocks to `WanAttentionBlock`. |
| B (rolling local cache/window) | ~1 day | Low-Med — window/sink math; S2V `S2V_KV_WINDOW_BLOCKS` used W=1, ShotStream needs W=7 → host RAM 7× the S2V-measured footprint; F16 cache + window should still fit but validate (S2V noted swap churn past W=1 on 12 GB/31 GB host). |
| C (DMD warp/shift schedule) | ~1-2 days | **Med-High** — exact `warp_denoising_step`+`timestep_shift=8` sigma mapping and the `add_noise` re-noise formula not fully extracted; must port `fm_solvers*.py`/`scheduler.py` and unit-test vs a torch dump. |
| D (global cache + next-shot) | ~2-3 days | Med — chain carryover exists but `condition_indices` dynamic sampling + global-cache prefill grid positions + "global=2 chunks vs max_context_frames=10" discrepancy need repo confirmation; Wan-VAE per-chunk causal encode of history. |
| E (change_rope + multi-caption) | ~1-2 days | Med — `θ` phase value + `causal_rope_apply` only partially extracted; wrong phase silently degrades cross-shot decoupling. |
| F (CLI/perf) | ~1 day | Low — inherits S2V offload/F16 machinery. |

### Single biggest risk
**The DMD few-step schedule exactness combined with autoregressive error accumulation.** The engine already diverges numerically from torch (F16 rolling cache `wan.hpp:271-286`, cuDNN flash, ggml GEMM), and ShotStream is an autoregressive rollout where each chunk conditions on the previous chunks' cached K/V — so any error in the `[1000,740,500,260]` warp/shift→sigma mapping, the inter-step `add_noise`, or the "which step writes the cache" timing **compounds across chunks into visible drift/degradation that a single-chunk parity test will not catch.** Mitigation is non-negotiable: pin the schedule and cache-write timing against a **chunk-by-chunk torch oracle** (Validation #1) before trusting any multi-chunk output, and keep the local KV cache in F32 for the parity runs (drop to F16 only after multi-chunk parity holds).
