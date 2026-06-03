# NAVA → sd.cpp port — implementation plan & running log

Branch `nava-port` (off `master`). Goal: port NAVA (6.3B joint audio-video MMDiT,
`ernie-research/NAVA`) into this sd.cpp tree, quant to Q4_K, run under the longcat-avatar
server pattern, then perf (VRAM/RAM-churn first, then speed). Source-of-truth docs live in
`~/dev/NAVA/{HANDOFF-cpp-port.md, NAVA.md, HANDOFF-av-chaining.md}` — read those for the
PyTorch recipe + capability findings. This doc is the cpp side: arch spec, decisions, log.

## Use case (target)
"longcat-avatar 1.5 but actionable + ideally faster": image+prompt → talking head with
**promptable motion** ("wave", "jump for joy"), natural movement, lip-synced **synthesized**
speech (NAVA generates the audio — it is not a dubber), repeatable voice via `spk_wavs`.
Anime-style chars are a main use case (examples show animated chars, so plausibly OK).
PyTorch eval already hit all three goals on the 3060 (see NAVA.md results table).

## What we reuse vs build (verified against this tree + NAVA source)
| Component | Source here | Action |
|---|---|---|
| Wan2.2 video VAE (48-ch) | `src/wan.hpp` `WanVAE`/`WanVAERunner` (tiled decode) | **reuse as-is** |
| umT5-xxl text enc (512 ctx, 4096-d) | `src/t5.hpp` `T5Runner`, `tools/convert_umt5.py` | **reuse as-is** |
| RoPE / AdaLN / attn primitives | `src/wan.hpp` (`WanSelfAttention`, `modulate_*`), `src/rope.hpp` | **reuse, extend** |
| Server + worker isolation + cancel | `examples/server/` (fork+IPC, /unload, /generate) | **reuse, add nava route** |
| GGUF convert pattern | `tools/convert_longcat_avatar.py` | **mirror → `convert_nava_dit.py`** |
| **Dual-stream MMDiT (10 double + 20 single)** | — | **NEW — `src/nava.hpp` (the bulk)** |
| **LTX audio VAE decode + vocoder** | — | **NEW (Phase 3)** |
| ReDimNet speaker enc (192-d) | — | **stub (precomputed vec), port later if needed** |
| Joint flow-match / UniPC sampler + 4-branch CFG | — | **NEW (Phase 2)** |

NB the `wan-s2v-port` branch has a parallel Wan2.2-S2V port (wav2vec2 audio cross-attn,
causal KV streaming). NAVA does **not** use wav2vec2 conditioning — its audio is a joint
diffusion stream — so s2v is reference-only, not a dependency.

## Architecture spec (verified from `configs/model/dit/NAVA_6B.json` + `model_mm.py` + weights)
- dim **3072**, ffn_dim **14336**, num_heads **24**, head_dim **128** (c=64 complex), eps 1e-6.
- **30 layers = 10 double-stream + 20 single-stream.** patch_size **[1,2,2]**.
- video latent **48-ch**, audio latent **128-ch** (config + weights `head_audio [128,3072]`,
  `patch_embedding_audio [3072,128,7]` both confirm 128 — the `t2v.py:419 randn(...,48)` is a
  red herring). text 512 ctx, 4096-d in. qk_norm RMSNorm. cross_attn_norm. AdaLN-Zero (6-way
  modulation via `time_projection [18432,3072]` = 6×3072).
- **RoPE 3D for video, 1D for audio.** `rope_params(max=1024, dim, theta=10000, freqs_scaling)`
  returns `[1024, dim/2]` complex. head_dim 128 → c=64 complex. **CORRECTED (was backwards):**
  `self.freqs` (video) = `cat[rope_params(1024,44), rope_params(1024,42), rope_params(1024,42)]`
  = `[1024,64]` complex, split `[22,21,21]` (T,H,W) = `[44,42,42]` real, **UNSCALED**.
  `self.freqs_audio = rope_params(1024,44, freqs_scaling=0.24)` = `[1024,22]` complex →
  audio 1D rotates only the **first 22 complex dims (44 real) scaled by 0.24**, remaining 42
  complex (84 real) **pass through identity**. Pairing is **interleaved** (`view_as_complex`
  of `reshape(...,-1,2)` = adjacent pairs) → maps to ggml `apply_rope`/`ggml_rope_pe`
  interleaved path. PORT TRICK: precompute the JOINT `pe` host-side (video tokens: 64 complex
  cos/sin from freqs at the token's f,h,w grid indices, ordered f-major then h then w; audio
  tokens: 22 rotated@0.24 + 42 identity), concat `[vid_pe ++ aud_pe]`, then ONE rope+attention
  over `cat[q_vid,q_audio]` = exactly `rope_apply_joint`. B=1 unpadded ⇒ joint-attn gather/
  scatter is identity, NO mask needed.
- **Weights are F32** (not bf16) — 1052 tensors, 6.297B params, straight requant.

### Double blocks (×10, "Hierarchical Alignment")
- Separate video/audio: self_attn{q,k,v,o}(+norm_q/k), cross_attn{q,k,v,o}(+`_audio` k/v),
  modulation / modulation_audio. **SHARED ffn + norm3** across both streams (no ffn_audio in
  weights). Joint self-attn = concat[video_tokens ++ audio_tokens] → one attention → split.
  Per-stream text cross-attn. qk_norm on q,k (not v). AdaLN: y = norm(x)*(1+scale)+shift,
  gated residual with gate. 6 mod vectors: [shift,scale,gate]×2 (self-attn, ffn).
### Single blocks (×20, "Unified Fusion")
- ONE shared self_attn/cross_attn/ffn/norm3/modulation applied to the concatenated
  [video ++ audio] sequence (unified). Same AdaLN/qk_norm/joint-attn structure.
### Edges
- Patchify video: Conv3d 48→3072 k[1,2,2] s[1,2,2]; squeeze temporal=1 → ggml Conv2d-style.
- Patchify audio: ChannelLastConv1d 128→3072 k7 + SiLU, then ConvMLP (SwiGLU, w1/w2/w3 [8192,3072,7]).
- Text embed: Linear 4096→3072 + GELU + Linear 3072→3072. Time: sinusoid(256)→MLP→3072, proj→18432.
- Speaker (timbre-in-context): `SpkToken` Linear 192→3072 + LayerNorm, spliced into text context
  at `<S>`/`<extra_id_2>` positions (`model_mm.py:1515`). Stub with precomputed 192-d initially.
- Head: per-stream Linear 3072→(192 video=48×2×2 / 128 audio) + 2-way mod, unpatchify.

### Sampler / CFG (`pipeline_nava.py`)
- Flow-match (`FlowMatchScheduler`, Euler) or UniPC (`FlowUniPCMultistepScheduler`). ~25-50 steps.
- **Up to 4 sequential forwards/step**: cond, uncond(neg), align-3d (modality-mask), timbre
  (spk removed). Sequential ⇒ multiply *time* not memory. Guidance combine (cfg scales
  video≈3 / audio≈2 / align≈3 / timbre≈3): video & audio updated separately, see B2 in agent
  spec / `pipeline_nava.py:539-556`. I2V: splice first-frame latent at token0, that token gets
  per-token timestep t=0 (clean anchor, `model_mm.py:1459-1463`).
- I2V resolution buckets (W/H ignored for I2V): **640→832×480**, **960→1280×704**. Audio 25 tok/s.

## Phases (each gated on GPU validation; one GPU job at a time, minimal clips)
0. **Scaffold** — branch, this doc, `convert_nava_dit.py`, eye-test page, bf16 PyTorch ref
   + saved latents (GPU, gated). [in progress]
1. **DiT forward, F16, match ref** — `src/nava.hpp` + `examples/nava` single-step harness;
   diff intermediate tensors vs PyTorch (save per-block outputs); then full forward.
2. **Sampler + I2V** — scheduler + 4-branch CFG + clean anchor; reuse Wan VAE decode; match a
   full ~50-step clip (640 bucket, short) PSNR-tight in F16.
3. **Audio path** — LTX audio VAE decode + vocoder → stereo PCM; mux Opus webm (silent-webm
   lesson); ReDimNet stub→port. Match audio.
4. **Quant + perf** — OWNER CALL: **Q4_K is the start AND end** (skip the ladder). Q8_0 ONLY as
   a desperation/debug step if Q4_K fails the saturation/brightness canary. Then VRAM/RAM-churn
   lowering (under-the-slab, no quality drop, minimise swap), then speed (FA/fusion/occupancy/
   offload pipelining à la longcat lap-34 / qwen3-tts laps).
5. **Worker isolation + koblem** — fork+IPC idle-VRAM-0, /unload, cooperative cancel; koblem
   heavy bucket; prompt-rewrite via prod 9B (English→Chinese dense caption, keep `<S>` English).
   Deferred: AV-chaining (K-frame clamp, audio-driving=S2V) per HANDOFF-av-chaining.md.

## Validation method
PyTorch bf16 ref (fixed image+prompt+seed) via `/mnt/hdd/nava/run_nava.sh` with
the 24GB master + `--save_vid_latent`. Match cpp **F16** first (PSNR-tight), THEN quant.
**Saturation/brightness drift = the quant canary** (washed-out vs F16 ⇒ a matmul is unfaithful).
STEPS POLICY (owner): the eye-test clip = **10 steps** (python ref was clean at 10). Use a
**2-step smoke on a tiny clip first** to catch gross errors fast/cheap. Probe how low steps can
go to separate "our bug" from "model just needs more steps" — but 10 is the clean target.

## Rules / environment (from owner)
- GPU/CPU serial — one video job at a time (2 = OOM). Minimise GPU time: shortest clip / fewest
  steps / lowest res that proves the point. Minimise swap churn (live version must not swap).
- cpp builds on this box are **fine** (the no-build rule is Rust/kobbler-only). Don't touch the
  prod kobbler stack; **ask before heavy GPU jobs**.
- Eye-test page: server over the run outputs (clips + VRAM + host-RAM + timing), manual refresh
  + auto-refresh toggle. Owner is headless/CLI — this is the only way they see results.

## Paths
- Weights `/mnt/hdd/nava/`: `NAVA.safetensors` (F32 master = GGUF source), `Wan2.2-TI2V-5B/`
  (Wan VAE + umT5 — dups of longcat's), `params/LTX2/` (audio VAE). PyTorch repo `~/dev/NAVA`
  (patched), venv `/mnt/hdd/nava/.venv`. PyTorch harness + clips `/mnt/hdd/nava/`, viewer :8094.
- This port: GGUFs → `models/`, dev harness → `docker/`/`tools/`, eye-test page → see Phase 0.

## Wiring contract (GROUND TRUTH from NAVA.safetensors header — all F32)
Names verbatim under `backbone.` (converter keeps them; cpp loads by these). norm1/norm2
absent ⇒ param-free LN (AdaLN-only). norm3 has weight+bias ⇒ affine LN = cross_attn_norm.
qk-norm weights are `[3072]` ⇒ RMSNorm over full dim (applied pre-head-reshape).
```
TOP-LEVEL (32):
  patch_embedding.{weight[3072,48,1,2,2]→squeeze→[3072,48,2,2], bias[3072]}   # Conv3d video
  patch_embedding_audio.0.{weight[3072,128,7],bias}                            # Conv1d stem k7
  patch_embedding_audio.2.{w1[8192,3072,7], w2[3072,8192,7], w3[8192,3072,7]}  # SwiGLU ConvMLP
  text_embedding.0.{w[3072,4096],b}  text_embedding.2.{w[3072,3072],b}         # MLP+GELU
  time_embedding.0.{w[3072,256],b}   time_embedding.2.{w[3072,3072],b}         # sinusoid(256)→MLP
  time_projection.1.{w[18432,3072],b}                                          # →6×3072 AdaLN
  speaker_embedding.net.0.{w[192],b[192]}  net.1.{w[3072,192],b}  net.3.{w[3072,3072],b}
  speaker_embedding.{null_token[1,3072], out_norm.{w[3072],b}}                 # SpkToken 192→3072
  head.{head.w[192,3072],head.b[192], modulation[1,2,3072]}                    # video unpatch 48*2*2
  head_audio.{head.w[128,3072],head.b[128], modulation[1,2,3072]}              # audio unpatch 128
DOUBLE block (×10, 48 tensors): per-stream video + _audio twins; SHARED ffn+norm3:
  self_attn.{q,k,v,o}(+ .bias) + .{q,k,v,o}_audio ; norm_q,norm_k (+ _audio) [3072] RMSNorm
  cross_attn.{q,k,v,o}(+bias) + _audio ; norm_q,norm_k (+ _audio)            # per-stream text x-attn
  ffn.0[14336,3072]+b, ffn.2[3072,14336]+b                                   # SHARED (no_split_norm_ffn)
  norm3.{w,b}[3072]  modulation.modulation[1,6,3072]  modulation_audio.modulation[1,6,3072]
SINGLE block (×20, 27 tensors): ONE unified set over concat[video++audio]:
  self_attn.{q,k,v,o}+norm_q,norm_k ; cross_attn.{q,k,v,o}+norm_q,norm_k ; ffn.0/.2 ; norm3 ; modulation[1,6,3072]
```
AdaLN: e0 = time_projection(time_emb) reshaped [.,6,3072] + block.modulation → chunk 6 =
[shift1,scale1,gate1, shift2,scale2,gate2]. y=LN_paramfree(x)*(1+scale)+shift; x += gate*sublayer.
With per-token timestep (clean anchor) e0 is per-token [B,L,6,3072]. Config (nava_640.yaml):
align_3d_cfg+timbre_cfg ON, UniPC, cfg video3/audio2/align3/timbre3, i2v_mode_prob 0.5.

## VERIFIED FORWARD SPEC (source-traced model_mm.py — THE implementation bible)
B=1, both modalities, no padding, num_double_final_layers=0. Compute is bf16 in PyTorch;
match in F16. Helpers: WanLayerNorm = LayerNorm eps1e-6, **affine=False** (norm1/norm2) or
**True** (norm3). WanRMSNorm(x)=x*rsqrt(mean(x²)+eps)*weight, eps1e-6, over full dim 3072.
GELU tanh. SwiGLU ConvMLP: `w2(silu(w1(x))*w3(x))` (Conv1d k7 p3, channels-last).
ModulationAdd(e)= e + param[1,1,6,D]. Head modulation = param[1,1,2,D] + e_timeemb.unsqueeze.

PREPARE (per stream, builds x/e/e0/context):
- VIDEO patchify: Conv3d(48→3072,k[1,2,2],s[1,2,2]) on [1,48,F,H,W] → [1,3072,F,H/2,W/2];
  grid_vid=(F,H/2,W/2); flatten+transpose → x_vid [1,L_vid,3072], L_vid=F·(H/2)·(W/2).
- AUDIO patchify: ChannelLastConv1d(128→3072,k7,p3)→SiLU→ConvMLP(SwiGLU) on [1,Lraw,128]
  → x_audio [1,L_audio,3072]; grid_audio=(L_audio,).
- TIME: per-token timestep t_tok[seq_len]. I2V clean anchor (`first_frame_is_clean`, VIDEO
  ONLY): t_tok=t everywhere EXCEPT first frame's H'·W' tokens (=grid[1:].prod) set to 0.
  Audio: t_tok=t for all. e = time_embedding(sinusoidal(256,t_tok)) [1,seq,3072];
  e0 = time_projection(e) → [1,seq,6,3072]. (time_embedding=Lin256→3072,SiLU,Lin3072→3072;
  time_projection=SiLU,Lin3072→18432.)
- CONTEXT: text_embedding(umT5 padded to 512) = Lin4096→3072,GELUtanh,Lin3072→3072 → [1,512,3072].
  context_lens=None (attend all 512). spk: speaker_embedding(spk[N,192]) (net: Lin192→? actually
  net.0=elementwise[192],net.1=Lin192→3072,SiLU,net.3=Lin3072→3072; out_norm LN; null_token for
  zero-spk) → splice into context rows at spk_pos (<extra_id_2> positions). With spk set, the
  SHARED context for ALL blocks = the AUDIO context (with spk spliced); else the video context.
  (Same prompt encoded once; only the spk splice differs ⇒ effectively one [1,512,3072] context.)

MAIN: x = cat([x_vid, x_audio], dim=1). e_vid=e0_vid, e_audio=e0_audio. Loop 10 double then
20 single (then 0 final). Then x_vid=x[:,:L_vid], x_audio=x[:,L_vid:]; head(x_vid,e_vid_TIME),
head_audio(x_audio,e_audio_TIME); unpatchify.

DOUBLE block (no_split_norm_ffn=True ⇒ norm1/norm2/norm3/ffn SHARED across streams):
  ev = modulation(e_vid).chunk(6,dim=2)  # 6×[1,L_vid,1,3072]; ea = modulation_audio(e_audio).chunk(6)
  xv=x[:,:L_vid]; xa=x[:,L_vid:]
  # self-attn (joint):
  xv_n = norm1(xv)*(1+ev[1])+ev[0];  xa_n = norm1(xa)*(1+ea[1])+ea[0]
  (yv,ya) = self_attn(xv_n, xa_n):    # WanDoubleStreamSelfAttention
     qv,kv,vv = norm_q(q(xv_n))→heads, norm_k(k(xv_n))→heads, v(xv_n)→heads   # qk RMSNorm over 3072 pre-reshape
     qa,ka,va = norm_q_audio(q_audio(xa_n))…, …, v_audio(xa_n)
     q=cat[qv,qa]; k=cat[kv,ka]; v=cat[vv,va]   # over seq
     q=apply_jointpe(q); k=apply_jointpe(k); attn = softmax(qkᵀ/√128)v  (full, no mask)
     yv = o(attn[:, :L_vid].flatten); ya = o_audio(attn[:, L_vid:].flatten)
  xv = xv + yv*ev[2];  xa = xa + ya*ea[2]
  # cross-attn (per-stream, to text context) + FFN, UNGATED cross-attn:
  xv = xv + cross_attn_v(norm3(xv), context):  q=norm_q(q(·)),k=norm_k(k(ctx)),v=v(ctx),attn,o
  xa = xa + cross_attn_a(norm3(xa), context):  q_audio/k_audio/v_audio/o_audio (+norm_q/k_audio)
  xv = xv + ffn(norm2(xv)*(1+ev[4])+ev[3]) * ev[5]     # ffn = Lin3072→14336,GELUtanh,Lin14336→3072
  xa = xa + ffn(norm2(xa)*(1+ea[4])+ea[3]) * ea[5]
  x = cat([xv,xa],1)

SINGLE block (unified, single param set over concat):
  e = cat([e_vid,e_audio],dim=1); e = modulation(e).chunk(6)
  y = self_attn(norm1(x)*(1+e[1])+e[0]):  ONE q/k/v/o (WanSelfAttention), but rope is STILL
     joint (3d on first L_vid tokens via freqs_vid, 1d on rest via freqs_audio) → same joint pe.
  x = x + y*e[2]
  x = x + cross_attn(norm3(x), context)
  x = x + ffn(norm2(x)*(1+e[4])+e[3]) * e[5]

HEAD(x,e_time): em = (param[1,1,2,D] + e_time.unsqueeze(2)).chunk(2); x = head_lin(norm(x)*(1+em[1])+em[0])
  video head_lin: 3072→192; unpatchify: [L_vid,192]→view(F,H',W',1,2,2,48) einsum fhwpqrc->cfphqwr
    →[48,F,H'·2,W'·2]=[48,F,H,W]. audio head_lin: 3072→128; → [128,L_audio].
OUTPUT velocity: video [48,F,H,W], audio [128,L_audio] → scheduler.

## Running log
- 2026-06-03: Phase 0. Branch `nava-port` off master. DELIVERED: HANDOFF-nava.md (this) +
  `tools/convert_nava_dit.py` (smoke-validated) + `tools/nava_eyetest_server.py` (LIVE :8097,
  reads /mnt/hdd/nava/cpp-runs/<name>/{clip.webm,meta.json}). Full forward REVERSE-ENGINEERED
  from model_mm.py source → wiring contract + VERIFIED FORWARD SPEC sections above (the bible;
  corrected RoPE: 0.24 is AUDIO-only, video unscaled; joint-pe host-precompute trick).
  GGUF: `models/nava-dit-f16.gguf` built (1052 tensors, 6.30B, 12.59GB, F16) — Phase-1 load target.
  NEXT (Phase 0 tail, GPU greenlit "best possible"=bf16): write corrected tensor-dump script
  (workflow's draft had WRONG dims 2048/16/20 — ignore; truth 3072/48/128; hooks on
  backbone.{double,single}_blocks[i] + head/head_audio are the useful part) → run ONE bf16
  forward (group-offload to fit 12GB) at 832×480/13f/1-2 steps, save inputs+per-block+velocity
  to /mnt/hdd/nava/cpp-runs/_ref/ref_tensors.npz. THEN Phase 1: author src/nava.hpp +
  examples/nava (single-forward harness) and diff per-block vs ref (F16, PSNR-tight).
  Reuse map + ggml code patterns captured (Linear/RMSNorm/LayerNorm/Conv3d via GGMLBlock,
  modulate_add/mul, Rope::attention + apply_rope interleaved + ggml_rope_pe, WanSelfAttention
  template, WanVAERunner decode + T5Runner reused as-is). Build on this box is OK (cpp-only).
- 2026-06-03b: Two bg agents dispatched: (1) `~/dev/NAVA/dump_ref_tensors.py` — UNRELIABLE,
  it GUESSED the pipeline API (`pipe.model(audio_context=…,video_input=…)`) which won't match
  model_nava.py; group_offload only a placeholder. DO NOT run as-is. (2) src/nava.hpp +
  examples/nava authoring+compile (running). Validation glue WRITTEN: `tools/nava_npz_to_bin.py`
  (npz→per-key f32 .bin + manifest) + `tools/nava_tensor_diff.py` (per-tensor PSNR/maxerr,
  flags FIRST diverging block).
  REFERENCE PLAN (corrected): model build is ambiguous (init_fusion_score_model_ovi builds an
  OVI FusionModel from ovi/configs/.../{video,audio}.json, NOT a bare WanAVModel from NAVA_6B;
  pipe.model.backbone is the WanAVModel). ⇒ DON'T reconstruct the build. Robust path: ride the
  PROVEN inference_nava.py path (run_nava.sh) and inject a CLASS-LEVEL monkeypatch of
  WanAVModel.forward (model_mm.py) via a 2-line edit guarded by env NAVA_DUMP_REF: on 1st call
  register hooks on self.{double,single}_blocks + head/head_audio, capture the REAL args
  (vid,audio,t,vid_context,audio_context,seq_lens,spk,first_frame_is_clean), call orig, dump
  npz, sys.exit(0). Use a TINY data row (small image/short prompt) at the 640 bucket, steps=1.
  Weights: the F32 master needs bf16 + --group_offload --offload_group_size 1 to fit 12GB (or
  run that single forward CPU-bf16 to dodge GPU/offload entirely — backbone-only ~12.6GB RAM).
  NOTE: reference is NOT blocking until nava.hpp compiles+runs on dummy inputs (shape sanity);
  numeric diff comes after. Validation = cpp harness reads npz→bin inputs, emits per-block .bin,
  nava_tensor_diff vs ref.
- 2026-06-03c: ★ PHASE-1 STRUCTURAL MILESTONE HIT. `src/nava.hpp` + `examples/nava` authored,
  COMPILES CLEAN, runs a full forward on dummy inputs: 10 double + 20 single blocks all
  [3072,160], velocity_video [192,128] (patched, pre-unpatchify), velocity_audio [128,32],
  ZERO NaNs, 1043/1052 tensors load (9 = speaker stub). Reviewed by hand: gen_nava_joint_pe is
  EXACTLY correct (omega theta^(-2j/dim), dims [44,42,42], audio 22pairs×0.24+identity, f-major,
  [cos,-sin,sin,cos]); DoubleBlock forward matches the VERIFIED FORWARD SPEC precisely (chunk6,
  param-free norm1, x*(1+scale)+shift, per-stream QKV→concat→joint rope+attn→split→per-stream
  out→gated; ungated per-stream cross-attn to shared context; shared modulated+gated FFN).
  Structure: NavaParams, AttnProj (qkv_suffix/out_suffix/cross_suffix, "" + "_audio"), DoubleBlock,
  SingleBlock, NavaHead, Nava, NavaRunner.
  BUILD: links CUDA-only (alloc_params_buffer → ggml_backend_cuda_host_buffer_type; vae-roundtrip
  same). Built in fresh `build-nava/` with /mnt/hdd/3d/avatar-shootout/toolchain (cmake4.3.3+nvcc
  sm_86); repo `build/` is a stale in-container cache — DON'T reuse. Binary build-nava/bin/nava.
  CPU-harness concessions (NOT model truth; native on a CUDA service build): harness sets
  LONGCAT_NO_FUSED_ROPE=1 (fused ggml_rope_pe is CUDA-only) + allocates modulation/head-mod params
  F32 (CPU rejects f32⊕f16; GGUF has them F16).
  STUBS/TODO (Phase-1 fidelity): (a) speaker_embedding NOT loaded/spliced; (b) I2V per-token clean
  anchor NOT done (e_vid==e_audio==e0, uniform t); (c) heads output PATCHED video velocity
  [48·2·2,L_vid] (unpatchify deferred to Phase 2). ⇒ KEY: these stubs EXACTLY match a TEXT-mode /
  UNIFORM-t / NO-spk reference, so the FIRST numeric match should use that config (spk_embed=None,
  first_frame_is_clean=False) — stubs don't block it; add spk + I2V anchor after the base matches.
  patch_embedding weight packing (ic- vs oc-fastest in the [kw,kh,1,IC*OC] reshape) is the one
  UNVERIFIED layout assumption → prime suspect if video diverges.
  .bin I/O: nava harness uses sd.cpp `load_tensor_from_file_as_tensor` (HEADER: int32 n_dims,
  name_len, type=0(F32), dims[] ggml-ne-order, name; then f32 ne0-fastest). `nava_npz_to_bin.py`
  currently writes RAW headerless+manifest → MUST switch to that header format (or adjust harness)
  before feeding real inputs. Harness expects (ne-order): video.bin [W,H,F,48], audio.bin [128,Lraw],
  context.bin [4096,512] (raw umT5; harness runs text_embedding) OR [3072,512] post-embed, timestep.bin [1].
  NEXT: generate the text-mode/uniform-t/no-spk reference (ride inference_nava or load backbone via
  AudioVideoPipeline builder; dump tiny seeded forward) → fix bin format → diff per-block.
- 2026-06-03d: PHASE-1 NUMERIC VALIDATION (in progress). Reference dumper BUILT + WORKING path found.
  * `~/dev/NAVA/nava_dump_ref.py` (NEW) — class-level monkeypatch of WanAVModel.forward, guarded by
    env NAVA_DUMP_REF=1, installed via a 3-line edit in inference_nava.py right after
    `set_rope_params()` (`nava_dump_ref.install(pipe.model.backbone, pipe)`). On the 1st forward it
    registers forward-hooks on double_blocks[i]/single_blocks[i] (capture each block OUTPUT
    [1,L_total,3072] as `double_block_N`/`single_block_N`), wraps prepare_transformer_block_kwargs to
    capture post-text_embedding `context_vid` [1,512,3072] + per-token `e` + AdaLN `e0`, captures the
    raw input args (input_vid [48,F,H,W], input_audio [L,128], input_t, input_vid_context_raw) + meta
    (vid/audio_seq_len, first_frame_is_clean, spk), velocity_video_unpatchified [48,F,H,W] +
    velocity_audio [128,L], saves float32(inputs)+float16(block outputs) to
    /mnt/hdd/nava/cpp-runs/_ref/ref_tensors.npz + meta.json, then sys.exit(0).
  * CONFIG matches the cpp stubs EXACTLY: smoke.jsonl (NO image_path, NO spk_wavs) ⇒ spk_embed=None
    (verified: pipeline_nava spk_embs from batch only) + is_i2v=False ⇒ first_frame_is_clean=False ⇒
    uniform-t, context=context_vid (no spk splice). B=1 single sample ⇒ vid_seq_len==L_vid, NO padding
    (verified: max_seq_len_video = max over the one sample = exact token count).
  * FIT: bf16 master (NAVA.safetensors, 25GB) **CANNOT** GPU-resident on the 12GB 3060 — and
    `--group_offload` does NOT help because inference_nava.py L481 `pipe.to(device)` moves the WHOLE
    backbone to CUDA BEFORE `apply_group_offload` (L503) runs ⇒ hard CUDA OOM at .to(device). (That's
    why run_nava.sh uses the fp8 6.9GB ckpt.) WORKING fit = **CPU bf16**: added `NAVA_FORCE_CPU=1`
    guard (device=cpu, disables group/t5 offload — keep GPU VISIBLE, since t5.py:476 has a class-default
    `device=torch.cuda.current_device()` that errors if CUDA hidden) + `del state_dict; gc.collect()`
    right after load_state_dict (reclaim 25GB) + dumper frees pipe.text_model (~11GB umT5) at forward
    entry. Even so the box (31GB) is RAM-tight for a 256x256/9f clip (model 12.6GB + activations + 30
    captured taps swap-thrashed) ⇒ use a **tiny clip 160x160/5f** + fp16 captured-tap store to fit RAM.
    Runner: `/mnt/hdd/nava/dump_ref_cpu.sh`.
  * VALIDATION GLUE FIXED: `tools/nava_npz_to_bin.py` rewritten to emit the EXACT harness header
    (int32 n_dims, name_len, ggml_type=0, dims[] ggml-ne-order, name, f32 ne0-fastest): video.bin
    [W,H,F,48] / audio.bin [128,Lraw] / context.bin [3072,512] (post-embed, harness passes through) /
    timestep.bin [1] — all share byte-layout with the npz C-contiguous arrays (no transpose; strides
    line up). `tools/nava_tensor_diff.py` extended to read the cpp LONGCAT_DUMP_DIR header (int64 ndim
    + dims + f32, reversed to numpy) AND normalize tap names `double_blocks.N`→`double_block_N`.
    Driver: `tools/nava_run_diff.sh` (npz→bin → `LONGCAT_DUMP_DIR=out ./build-nava/bin/nava gguf in out`
    → diff). cpp per-block taps already wired (capture_tensor on each block output → get_compute_graph
    expands every debug tensor → dumped when LONGCAT_DUMP_DIR set). build-nava/bin/nava is current.
  * BUG FOUND + FIXED in the dumper: on CPU the model's internal `amp.autocast('cuda')` blocks are
    no-ops, and patch_embedding (Conv3d/Conv1d) runs OUTSIDE them → `RuntimeError: Input type (float)
    and bias type (c10::BFloat16) should be the same` at the very first patchify. FIX: wrap the
    orig_forward call in `torch.autocast(device_type="cpu", dtype=torch.bfloat16)` (gated on
    NAVA_FORCE_CPU). With that, the forward runs past patchify and through the blocks. (The GPU path
    never hit this — CUDA autocast was active.)
  * STATUS @ writeup: with the autocast fix the CPU forward RUNS (no crash, umT5 freed, no hard swap at
    160x160/5f) but is still SLOW on this 31GB box: the resident bf16 backbone (~12.6GB) doesn't fit
    page-cache alongside the rest, so each of the 30 blocks re-reads its weights from the slow /mnt/hdd
    HDD (read_bytes monotonic, ~0.5-1 GB/block) AND the joint video+audio attention is heavy at bf16 on
    CPU. The forward demonstrably advances block-by-block but the full 30-block forward + heads + npz
    save takes a long wall-time here. **THE WIN NEEDED = more RAM / faster disk**: run `dump_ref_cpu.sh`
    on a box where the 12.6GB model fits page-cache (>=48GB, or model on SSD/tmpfs), then it finishes in
    minutes and `tools/nava_run_diff.sh` does the per-block diff. ALL glue is done + verified-by-
    inspection; only the actual numbers are pending the reference npz. The bf16 master is the right
    reference (fp8 ckpt = coarser, last resort). DO NOT re-derive the pipeline build, the bin format, or
    the autocast fix — all solved here.
  * OUTCOME (confirmed): the 160x160/5f CPU bf16 run was **OOM-KILLED (rc=137)** — the kernel OOM-killer
    fired during the joint-attention activation peak (RSS+swap > 31GB+swap). So the dump does NOT
    complete on this 31GB box even at a tiny clip with every RAM lever applied (CPU, del state_dict,
    freed umT5, fp16 taps). **HARD CONCLUSION: the Phase-1 reference dump needs a bigger box.**
    Recommended: a >=48-64GB machine (12.6GB bf16 model + attention activations + caches fit), OR get the
    model GPU-resident on a >=24GB GPU (no swap; forward in seconds). On such a box:
    `bash /mnt/hdd/nava/dump_ref_cpu.sh` (CPU; raise the clip back to e.g. 384x384/13f), then
    `bash /home/dbrain/dev/longcat-avatar.cpp/tools/nava_run_diff.sh` for the per-block PSNR table.
    Everything except the actual reference numbers is built, fixed, and verified-by-inspection.
- 2026-06-03e: ★★ DiT NUMERICALLY VERIFIED ✅ (SUPERSEDES the "needs a bigger box" note above).
  The OOM was the full PIPELINE (umT5 11GB + VAEs + model 12.6GB + state_dict 12.6GB ≈ 38GB).
  FIX = STANDALONE backbone-only dumper ~/dev/NAVA/nava_dump_standalone.py: meta-init
  WanAVModel(**NAVA_6B.json, text_dim=4096, num_double_final_layers=0, add_spk_emb=True,
  no_split_norm_ffn=True) → load ONLY backbone.* cast bf16 via load_state_dict(assign=True)
  (missing=0 unexpected=0 — matches ckpt exactly) → NO umT5/VAE ⇒ ~12.6GB resident, NO swap.
  Seeded-random inputs (text-mode/uniform-t/no-spk), tiny F=2/H=16/W=16 → L_vid=128, L_audio=32.
  Needed a CPU SDPA fallback in NAVA attention.py:flash_attention (hard-asserts cuda; repo SDPA
  patch missed it). Load 230s + forward 38s, no thrash.
  RESULT (tools/nava_run_diff.sh, npz_to_bin called with --raw-context): all 10 double + 20 single
  blocks PSNR 62–78 dB; head_audio 45.2 dB; head_video 39.7 dB (maxAbs 0.09 — minor head-AdaLN
  imperfection, FOLLOW-UP to tighten; body is exact so it's the head Linear+mod, not structural).
  Confirms patch_embedding IC/OC packing + joint-pe + AdaLN + qk-norm + cross-attn + shared-FFN.
  cpp harness accepts raw [4096,512] context (runs text_embedding). build-nava/bin/nava current.
  Reproduce: python ~/dev/NAVA/nava_dump_standalone.py ; then npz_to_bin(--raw-context) → nava → diff.
  NEXT (Phase 2): Euler flow-match sampler + cond/uncond CFG (docs/nava-phase2-sampler.md) +
  WanVAERunner decode → SILENT clip (2-step smoke → 10 steps) → eye-test :8097; + tighten head_video AdaLN.
- 2026-06-03f: ★ PHASE-2 SILENT T2V RENDER PATH — COMPILES CLEAN + CPU SMOKE END-TO-END ✅.
  examples/nava/main.cpp gained a `render` subcommand (single-forward mode unchanged):
    nava render --prompt "..." --steps N --frames F --width W --height H --seed S
                --cfg 3.0 --shift 5.0 --vae <gguf> [--context <bin>] [--cuda]
                --out-name NAME --runs-dir DIR
  PIPELINE: seeded PhiloxRNG noise in VAE-latent space [W/16,H/16,F,48] → Euler
  flow-match loop (FlowMatchSched: sigmas=linspace(1,0.003/1.002,N); shift σ←5σ/(1+4σ);
  t=σ·1000; latent += v·(σ_next−σ_cur), σ_next=0 final) with TWO DiT forwards/step
  (cond ctx + uncond=zeros_like) combined v=v_uncond+3·(v_cond−v_uncond) → host-side
  unpatchify [192,L_vid]→[W/16,H/16,F,48] (einsum fhwpqrc→cfphqwr) → diffusion_to_vae_latents
  → WanVAERunner(VERSION_WAN2_2_TI2V) decode → frames → silent VP8 webm + meta.json.
  Verified-against-source: scheduler (flow_match.py), CFG combine (pipeline_nava.py:541 MVP
  path), latent grid (t2v.py:277 h=H//16,w=W//16; DiT patch /2 more), uncond=zeros
  (pipeline_nava.py:402).
  ★ VAE LATENT NORM — RESOLVED (the "washed-out" suspect): NAVA's LocalVideoVAEAdapter has
  scaling_factor=1.0/shift_factor=0.0 (NO pipeline renorm), and the Wan2.2 video VAE applies
  z←z*std+mean INSIDE its decode (vae2_2.py:812-818, self.scale=[mean,1/std]). The in-tree
  WanVAERunner mirrors this EXACTLY: diffusion_to_vae_latents = latents*std/scale_factor+mean
  with scale_factor=1.0, and its 48ch mean/std constants are BIT-IDENTICAL to NAVA's
  vae2_2.py:906-1014 (checked element-by-element). decode() then maps [-1,1]→[0,1]
  (scale_input=true). ⇒ DECISION: latent (diffusion convention) → diffusion_to_vae_latents
  → decode(decode_video=true). No extra renorm. Latent passed as 5D [W,H,F,48,1] so
  get_latents_mean_std reads channel at ne dim 3 (4D would be mis-read as image [W,H,C,1]).
  ★ VAE GGUF: models/longcat-wan-vae-f16.gguf is the longcat Wan2.1 *16ch* VAE — WRONG for
  NAVA (decoder.conv1 is [384] not [1024]). NAVA needs the Wan2.2 *48ch* VAE. Converted
  /mnt/hdd/nava/Wan2.2-TI2V-5B/Wan2.2_VAE.pth (original ComfyUI layout, names ALREADY match
  sd.cpp wan2_2 incl. the double-nested up/downsamples.N.upsamples.M) → models/wan2.2-vae-48ch-f16.gguf
  (196 tensors, 1.4GB, f16). New converter tools/convert_wan22_vae.py (verbatim names +
  5D-conv pack + gamma-flatten); ran via NAVA-venv torch → /tmp npz → uv gguf (venv has no gguf/pip).
  CPU SMOKE (no GPU): `nava render --steps 2 --frames 2 --width 64 --height 64 --seed 42
  --vae models/wan2.2-vae-48ch-f16.gguf` → loop+decode+webm ran end-to-end, ZERO NaNs.
  ffprobe: VP8 64x64 yuv420p 5 packets (=(2-1)*4+1) 0.167s; meta.json valid (eye-test schema).
  DUMMY cond context (no umT5) ⇒ output is garbage-but-valid (correctness NOT chased on CPU).
  BUILD: build-nava (toolchain /mnt/hdd/3d/avatar-shootout/toolchain, SD_CUDA+WEBM+WEBP ON),
  CMakeLists adds ../common/media_io.cpp+log.cpp; Opus intentionally NOT linked (silent path).
  CPU CONCESSIONS (NOT model truth): LONGCAT_NO_FUSED_ROPE=1 (set by harness) + F32 mod/head
  params (in nava.hpp) — same as Phase-1. PhiloxRNG noise (torch-cuda-randn-alike).
  STUB still present (Phase-1): video head AdaLN 39.7 dB (minor), speaker/I2V not wired (text-mode
  only) — none affect the silent-T2V path.
  HUMAN GPU RENDER (real clip): `nava render --cuda --steps 10 --frames 13 --width 832 --height 480
  --seed 42 --gguf models/nava-dit-f16.gguf --vae models/wan2.2-vae-48ch-f16.gguf
  --context <umT5.bin> --out-name <name>` → /mnt/hdd/nava/cpp-runs/<name>/{clip.webm,meta.json}.
  Expected VRAM ≈ DiT 12.6GB f16 resident + VAE ~1GB + ~0.2GB compute (fits 24GB; on 12GB needs
  quant/offload — Phase-4). REQUIRED: --context = raw umT5 [4096,512] (DiT runs text_embedding)
  or post-embed [3072,512], dumped from NAVA's own text_model (guarantees tokenizer parity incl.
  the special-token format; cpp umT5 tokenization is deliberately NOT reimplemented). Without
  --context the render uses a dummy ctx (smoke only). NEXT: dump a real umT5 context.bin for a
  prompt → 10-step 832×480 GPU render → eye-test :8097; tighten head_video AdaLN.

## Driving this with Claude Workflows (for the next session — owner asked)
Most of this port IS machine-verifiable, so prefer a deterministic `Workflow` script over ad-hoc
background agents (which run open-loop — that caused a 2h flail). Structure:
- **Parallel stages** (`parallel`/`pipeline`): spec extraction (reading PyTorch), code drafting,
  adversarial review, the per-component reads. Fan these out — they're context-heavy + independent.
- **Sequential stages** for GPU: one job at a time (2 = OOM). Put every GPU render/forward in a
  SINGLE sequential `agent()` call, never inside a concurrent `parallel()`. Kill stale procs first.
- **Machine gates (no human needed):** compile-pass; forward runs with no NaN; per-block PSNR ≥ ~40
  dB vs the PyTorch reference (tools/nava_tensor_diff.py); VRAM under the 7.5GB slab; saturation/
  brightness canary on quant (Q4_K vs F16). Encode these as `if`/retry-until-pass IN the script.
- **Hard timeouts / early-bail** on every GPU stage (the open-loop agents lacked this → 2h waste).
  A standalone reference forward should be MINUTES; if a stage exceeds its budget, bail + report.
- **The ONLY human gate = final aesthetic judgment** of the rendered clip (lip-sync/motion/anime
  face). Everything up to "numerically matches ref + renders clean under VRAM budget" is automatable.
- Reference-gen pattern that WORKS (don't re-derive): standalone backbone-only dumper
  (~/dev/NAVA/nava_dump_standalone.py) — no umT5/VAE, meta-init + assign-bf16 load, CPU SDPA. The
  full-pipeline path OOMs (38GB). Validate F16 first (done, 62-78dB), then Q4_K.
- 2026-06-03g: ★★★ FIRST REAL CLIP RENDERED (Phase-2 silent text-to-video, GPU). Path: Q4_0 DiT
  (models/nava-dit-q4_0.gguf, 4.31GB, numpy converter — sd-cli/K-quant deferred to Ph4) + 48ch
  Wan2.2 VAE (models/wan2.2-vae-48ch-f16.gguf) + precomputed umT5 context (NAVA's own encoder,
  ~/dev/NAVA/dump_umt5_ctx.py → prompt_ctx.bin [4096,512]; cpp tokenization intentionally not
  reimplemented). Render = examples/nava `render` subcmd (Euler flow-match, cond+uncond CFG=3,
  WanVAERunner decode). Enabled VAE tiling in main.cpp (24x24 latent tiles, spatial-only; wan VAE
  is causal-conv3d so temporal already streamed) → full 832x480 bucket fits 12GB (peak 7.6GB, 6
  tiles). RESULTS on eye-test :8097: nava_smoke2 (384²,2step=flat std0.028, under-denoised as
  expected), nava_10step_384 (384²,10step, latent std0.89 healthy — proved no bug), nava_clip01_832
  (832x480, 5 latent→17 pixel frames, 10 step, rgb std0.174, 52.6s wall / 3.05 s/step / VAE ~20s).
  Run cmd: LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib build-nava/bin/nava render --cuda
  --gguf models/nava-dit-q4_0.gguf --vae models/wan2.2-vae-48ch-f16.gguf --context <ctx.bin>
  --steps 10 --frames N --width 832 --height 480 --out-name X --runs-dir /mnt/hdd/nava/cpp-runs.
  NOTE confirms Wan2.2 VAE is 16x-spatial/4x-temporal (latent=pixel/16). AWAITING owner eye-test of
  nava_clip01_832 (coherence / promptable motion). CAVEATS: silent (Phase 3), text-mode not I2V
  (Phase 2 follow-up), Q4_0 not Q4_K, head_video AdaLN 39.7dB (minor). Prompt (zh): "一个女孩对着
  镜头微笑，轻轻挥手，背景是柔和的灯光，镜头缓慢推近。"
  HANDOFF POINT (as agreed): next session takes Phase 3 (audio) + I2V anchor + Q4_K + tiling tune +
  perf, driven as a self-verifying Workflow (see "Driving this with Claude Workflows" section).

## ✅ I2V clean-latent anchor — DONE + WORKING + VALIDATED 2026-06-04
Image→talking-head works (`nava_I2V_face` on eye-test :8097: frame0 = the input face, later frames
animate the same man talking; std bounded, no explosion). Per-token clean-anchor timestep matches
PyTorch `first_frame_is_clean=True` (blocks 100–120 dB, heads 77–85 dB) and text-mode regression is
bit-identical. Implementation:
- `src/nava.hpp`: `time_embed` already produces per-token e0[dim,6,L]/e[dim,L] from a per-token t.
  `chunk6` + SingleBlock inline-chunk + `NavaHead` permute modulation chunks to [dim,L,1] (no-op
  broadcast when L=1 → text-mode unchanged). Runner: `per_token = e0->ne[2]>1`; slice e0/e_time per
  stream (video frame-0 tokens carry t=0).
- `examples/nava/main.cpp` `render`: `--image <RGB bin>` (preprocess any image with
  `tools/nava_prep_image.py <img> 832 480 out.bin` → ggml ne [W,H,1,3], [0,1]; done in Python to
  dodge stb linking). VAE-encode (decode_only=false) BEFORE the DiT loads, `vae_to_diffusion_latents`,
  splice at frame 0, re-splice after every Euler step; per-token clean-anchor t each step.
- Validation tooling: `~/dev/NAVA/nava_dump_i2v_ref.py` (PyTorch F32 ref), harness `NAVA_I2V=1`.
Run: add `--image /mnt/hdd/nava/i2v_face.bin` to the text-mode render cmd. NEXT: Phase-3 audio.

## (HISTORICAL) NEXT: I2V clean-latent anchor (precise plan, scoped 2026-06-04)
Text-mode T2V works now. I2V is NAVA's real use case. The clean anchor pins frame 0 so motion is
image-conditioned. Steps (all in cpp; validate the std stays bounded + eye-test, like text-mode):
1. **Image → first-frame latent.** Load image (wire stb_image / common media_io), resize to the
   bucket (832×480), VAE-**encode** via `WanVAERunner` with `decode_graph=false` (`ae.encode`,
   src/wan.hpp:1029/1210) → [48,1,H_lat,W_lat]. Convert VAE→diffusion latent (INVERSE of
   `diffusion_to_vae_latents`: `(z-mean)*scale/std`).
2. **Splice** that latent into the noise `latent` at frame index 0 (ggml ne [W,H,F,48], frame dim=2).
3. **Per-token timestep (the anchor).** Source: model_mm.py:1459-1463. Currently `time_embed` takes a
   SCALAR t → e0[dim,6,1], e[dim,1] broadcast over all tokens. Change to per-token: build a
   t_tok[L_total] where the first `h_grid*w_grid` VIDEO tokens (frame 0) = 0, all other video + all
   audio = t. Then e = time_embedding(sinusoidal(t_tok)) → [dim, L], e0 = time_projection → [dim,6,L].
   Propagate: DoubleBlock/SingleBlock modulation already `ggml_mul/add` with e0 broadcast [dim,6,1] over
   tokens — switch to per-token [dim,6,L] (video slice for e_vid, audio slice for e_audio; the block
   splits at L_vid). NavaHead uses e_time[dim,1]→ make it [dim,L] (per-token). This is the invasive bit.
4. Keep audio per-token t = t everywhere (audio is never clean-anchored).
5. **Validate** ideally vs a PyTorch I2V reference (adapt nava_dump_standalone.py with
   first_frame_is_clean=True + a real image latent); else eye-test + bounded-std sanity.
Owner's product target: image + audio-sync + promptable actions (the kid's vtuber). Phase-3 audio
(LTX audio VAE + vocoder, currently silent) and Q4_K + perf come after.

## ✅ PHASE-2 INCOHERENCE — RESOLVED 2026-06-04 (see HANDOFF-nava-DIVERGENCE.md top)
It was a **dummy-context path bug** in `run_render`: `load_or_dummy(".", o.context)` built a relative
`".//abs/path"` that never exists → silently used the sin-based dummy context → unconditioned garbage
+ biased CFG → std runaway/dark. The DiT forward was bit-faithful to PyTorch F32 all along (blocks
96–122 dB). Fix: load `ctx_pos` directly (working tree, examples/nava/main.cpp). 832×480 q8 now renders
a coherent talking man (eye-test :8097 `nava_FIXED_832`). NEXT: I2V anchor, Phase-3 audio, Q4_K, perf.

## ★ (HISTORICAL) PHASE-2 BUG: clip renders but is INCOHERENT (owner eye-test 2026-06-03)
Owner viewed nava_clip01_832: structured (rgb std 0.174, not flat, no NaN) but VISUALLY MEANINGLESS
(no coherent subject/scene). CRITICAL FRAME: the DiT forward is NUMERICALLY VERIFIED per-block
(62-78 dB vs PyTorch) — so the bug is NOT in nava.hpp block math. It's in something the per-block
diff did NOT cover (that diff used RANDOM context + matched inputs both sides; it proves the OP, not
that real-prompt sampling yields coherent content). Ranked suspects:
  1. SAMPLER (UNVALIDATED vs PyTorch) — HIGH. The Euler flow-match schedule / timestep mapping
     (σ·1000) / step sign+direction / CFG combine in examples/nava/main.cpp render are hand-ported
     from docs/nava-phase2-sampler.md and never diffed against pipeline_nava.py's actual loop.
     A wrong sigma schedule or t-scaling denoises at the wrong noise levels → noise-like content.
  2. REAL-PROMPT CONDITIONING — HIGH. (a) umT5 context dump (~/dev/NAVA/dump_umt5_ctx.py) correctness:
     tokenization, the <S>/<extra_id_2> handling, padding to 512, and whether [4096,512] raw is what
     the cpp text_embedding+cross-attn expect. (b) PROMPT FORMAT: the README/handoff say NAVA needs a
     CHINESE DENSE CAPTION via the rewriter template (pe_src/prompts/rewrite_template.txt) — my prompt
     was a SHORT zh sentence, NOT a dense rewritten caption. "Raw/sparse prompts underperform" per the
     PyTorch notes. May be a big part of the vagueness (not a code bug).
  3. TEXT-MODE vs I2V — HIGH/STRUCTURAL. NAVA's PyTorch evals (NAVA.md table: peter_talk, action,
     etc.) are ALL I2AV (image-conditioned). Pure TEXT-to-video (no first-frame anchor) may be
     INHERENTLY weak for NAVA (it's an avatar/I2AV model). The I2V path (first-frame latent splice +
     per-token clean-anchor, currently STUBBED in nava.hpp) might be REQUIRED for coherence.
  4. noise init scale / diffusion_to_vae_latents / VAE mean-std application — MEDIUM.
  5. head_video AdaLN 39.7dB — LOW (minor, wouldn't cause total incoherence).
DECISIVE DIAGNOSTIC (do FIRST, disentangles code-bug from model-behaviour): render a PyTorch
reference clip at IDENTICAL settings (text-mode, this prompt, seed 42, 10 steps, 832x480) via the
real pipeline OR the standalone backbone + a ported Euler loop. THEN:
  - If PyTorch ALSO renders garbage at these settings → it's NOT our bug: switch to I2V (image anchor)
    + a dense rewritten Chinese caption (wire the prod-9B rewriter), re-eval. NAVA likely needs both.
  - If PyTorch renders COHERENT → our sampler/context has a bug: dump PyTorch per-step latents and
    diff vs cpp per-step latents (the cpp already taps; add a per-step latent dump). First diverging
    step localizes sampler-math vs context. Then fix in examples/nava/main.cpp / dump_umt5_ctx.py.
Suggest doing this as a self-verifying Workflow (parallel: PyTorch-ref-clip + sampler-code-audit vs
pipeline_nava.py; then sequential per-step latent diff; machine gate = per-step PSNR). Then proceed
to I2V anchor + audio (Phase 3) + Q4_K + perf.

## ★★★ 2026-06-03h SESSION — BUG ROOT-CAUSED (not yet fixed). FRESH AGENT START HERE.
The Phase-2 incoherence was hunted hard. NET RESULT: it is a CPP BUG (not model behaviour, not
recipe), narrowed to **a small SYSTEMATIC per-forward velocity error — worst in the HEAD
(head_video ~40 dB vs blocks 62-82 dB) — that COMPOUNDS over the iterated sampler** until the
latent std runs away (0.97→1.86 over 10 steps) and the clip is incoherent. The single-forward
validation (62-78 dB) never caught it because it ran at ONE fixed t with RANDOM, non-co-denoised
audio — neither condition the sampler actually hits. THE TASK: find & eliminate that head error
(and check the audio-coupling), then re-render. **Do NOT pivot to I2V to mask it (owner: "what's
the point in moving on if we're leaving bugs behind").**

### DECISIVE diagnostics that establish "cpp bug, head-localized" (trust these, don't redo)
- **PyTorch text-mode @ 832x480 = FULLY COHERENT** (run_nava.sh, smoke.jsonl dense caption, no
  --is_i2v): a man in a suit talking in a study. So text-mode is viable; NAVA does NOT require I2V
  for coherence; the bug is OURS. (clip /mnt/hdd/nava/clips/textmode_diag.mp4)
- **PyTorch with the cpp's EXACT recipe = STILL COHERENT.** Made nava_run_euler.yaml
  (scheduler_unipc:false + align_3d_cfg:false → Euler + 2-branch CFG, the cpp's recipe) → ran
  run_nava.sh → coherent man (clip euler_recipe.mp4). PROVES the recipe (Euler/2-branch/no-UniPC/
  no-align) is sufficient; the gap is the cpp IMPLEMENTATION, not the recipe.
- **Per-step latent diff (cpp Q8 GPU vs PyTorch bf16 standalone backbone, IDENTICAL noise + real
  context + identical Euler recipe)**: diverge from STEP 0 (latent PSNR 47.8 dB) and degrade every
  step → 12.5 dB by step 9; cpp std EXPLODES 0.97→1.86 while PyTorch stays bounded 0.98→0.80. No
  single broken op — a per-forward velocity error that the joint-AV feedback amplifies. Tooling:
  ~/dev/NAVA/nava_sampler_compare.py + the NAVA_DUMP_TRAJ gate in main.cpp (dumps vid_noise/
  aud_noise/vid_step_NN.bin). cpp dumps at /mnt/hdd/nava/traj_cpp/.
- **Timestep sweep of the per-block diff (t=1000,770,559,364,150,500, random ctx)**: per-block PSNR
  is FLAT (min ~61-63 dB at single_block_12/13 for every t; head_video 39.5-40.6 dB at every t).
  NO collapse at low t → the forward is timestep-FAITHFUL → the error is NOT time-embedding/
  time_projection/AdaLN-modulation (those were source-verified equivalent: ggml timestep_embedding
  cos-first/sin-second + 10000^(-j/128) matches model_mm sinusoidal_embedding_1d). Tooling:
  ~/dev/NAVA/nava_dump_tsweep.py + /tmp/tsweep_diff.sh; outputs /mnt/hdd/nava/cpp-runs/_tsweep/t*/.
- **Real-context single-forward diff** (dense_ctx fed as both vid+aud context, F=2 grid): ALL blocks
  68-91 dB, heads 44/53 dB — even BETTER than random. So the forward op is faithful WITH the real
  umT5 context too. Tooling: ~/dev/NAVA/nava_dump_realctx.py; out /mnt/hdd/nava/cpp-runs/_ref_realctx/.
- **The zeros-vs-negprompt asymmetry (the tell):** cpp ZEROS-uncond → stable (std 0.62) warm-blur,
  RIGHT colour (context conditions low-freq), NO subject. cpp NEG-prompt → diverges (std 1.85-1.94)
  dark void/blue, no subject. EXPLANATION: NAVA needs neg-prompt for a subject; with neg-prompt
  v_cond≈v_uncond so the 3·(v_cond−v_uncond) guidance is a SMALL difference where the ~10% head
  error does NOT cancel and dominates → runaway. With zeros the difference is LARGE so the same
  error is negligible → stable-but-weak. cfg sweep (1.5/2.0/2.5/3.0 + neg-prompt) → std
  0.47/0.92/1.46/1.90: lower cfg only trades divergence for dark/under-denoised, NO cfg yields a
  subject → the error CORRUPTS, not just amplifies. This is why the handoff's "head AdaLN 39.7 dB,
  LOW priority" was WRONG — head fidelity is LOAD-BEARING for text-mode.

### EXHAUSTIVELY RULED OUT this session (do NOT re-investigate)
- QUANT — Q8_0 diverges IDENTICALLY to Q4_0 (std 1.66 both); and PyTorch fp8 (coarser than Q8) is
  COHERENT → precision is not it. (Made models/nava-dit-q8_0.gguf via convert_nava_dit.py --dtype
  q8_0 through `uv run --with gguf --with numpy` — the NAVA venv lacks gguf.)
- VAE DECODE — cross-decoded the cpp's final latent through PyTorch's OWN wan2.2 VAE
  (~/dev/NAVA/cross_decode_cpp_latent.py): produced the IDENTICAL banding → cpp latent is the
  problem, VAE is faithful.
- UNPATCHIFY (main.cpp unpatchify_video) — bit-exact (maxerr 0.0) vs PyTorch einsum
  'fhwpqrc->cfphqwr', verified in numpy on real head_video.
- RoPE construction (gen_nava_joint_pe) — symmetric H/W (identical freqs), f-major, correct.
- CFG mechanism — cfg=1 (no guidance) diverges same as cfg=3 → not the CFG combine.
- SLG (slg_layer=11) — a NO-OP: WanAVModel.forward (model_mm.py:829/1016) absorbs slg_layer via
  **kwargs; the real backbone is WanAVModel (model_nava.py:196), NOT fusion.py's FusionModel.
- Sigma schedule — APPLIED the extra_one_step fix (main.cpp FlowMatchSched: linspace(1,σmin,N+1)[:-1]
  i.e. divide by n not n-1, matching PyTorch FlowMatchScheduler extra_one_step). No change to
  coherence. (Keep the fix; it IS the correct schedule.)
- Context padding — truncating the umT5 context to its real token length (dense 344 / neg 153,
  vs zero-pad-to-512) changed NOTHING. (dense_ctx_real.bin / neg_ctx_real.bin in _ctx/.)
- Timestep-conditioning / time embedding — see sweep above, flat across t.
- OOD RESOLUTION (was a red herring for the banding): square 256²/384²/512² give blocky/banded
  artifacts (NAVA trained on 832x480 & 1280x704 buckets); at 832x480 the artifacts VANISH (clean
  field). Always test at 832x480 (or 1280x704, the base sweet spot).

### FIXED / CHANGED this session (keep these)
- **AUDIO JOINT-DENOISE (a real bug fix).** The render fed a DEGENERATE 1-token, 0.02-magnitude
  dummy audio stream into the JOINT self-attention every step → catastrophic divergence (std 1.66+).
  Owner's hypothesis. FIX in main.cpp: (a) size audio_len to the video duration (≈25 tok/s,
  fps 24): `max(8, round(((F-1)*4+1)/24*25))`; (b) init audio as proper std-1 Gaussian noise;
  (c) CO-DENOISE audio in lockstep (Euler, cfg_audio=2.0) so the joint attention always sees an
  audio stream at the matching noise level. Required nava.hpp `compute_va()` (returns BOTH video +
  audio velocity from ONE forward via a pad-128→192 + concat-after-L_vid + host-side split;
  `return_joint` member gates build_graph). This fixed the catastrophic divergence (std now a
  healthy denoising curve) but NOT the residual ~10% head error → still no subject.
- examples/nava/unipc.hpp — full UniPC (FlowUniPCMultistepScheduler) port, **VALIDATED 2.15e-6 max
  abs err vs PyTorch** scheduler (unit test: `nava unipc-test <dir>` + ~/dev/NAVA/unipc_ref.py).
  NOT wired into render. NOTE the set_timesteps double-shift subtlety documented in the header.
  UniPC is NOT required for coherence (PyTorch Euler is coherent) — it's a quality/speed upgrade.
- NAVA_DUMP_LATENT + NAVA_DUMP_TRAJ env gates in main.cpp run_render (latent / per-step dumps).

### UPDATE (2026-06-03i): compute_va RULED OUT — bit-identical to compute()
Added a check in run_single_forward (main.cpp): runs compute() AND compute_va() on the same inputs,
diffs the video velocity → **maxAbsDiff=0, meanAbsDiff=0** (CPU F16, real-context _ref/bin inputs).
So the joint pad/concat/split path is faithful; the video velocity the render uses == the validated
one. The per-forward error is therefore the SHARED forward path (head + accumulated blocks), NOT the
joint wrapper. Two background sub-agents were dispatched to fix this and BOTH STALLED (sub-agents are
NOT woken on run_in_background completion → they deadlock; one also wasted a no---cuda CPU render).
LESSON: drive heavy iterative fixes from the MAIN loop or a FRESH top-level session, not a sub-agent;
if a sub-agent is used it must run every job FOREGROUND/blocking.

### THE BUG TO FIX (ranked suspects for the ~10% per-forward velocity error)
1. **HEAD MODULATION / precision.** head_video is the worst tap (~40 dB; blocks 62-82). Render uses
   F16 head/modulation params from the gguf; the CPU validation harness forces F32 (a documented
   concession) → the VALIDATED 40 dB may even FLATTER the render. Compare the cpp head output to
   PyTorch head_video element-wise AT the render's conditions; check: modulation param dtype (F16 vs
   F32) effect; the param-free LayerNorm (eps, affine=False) vs PyTorch; the head Linear; whether
   `em = modparam + e_time` broadcast/chunk order is exactly PyTorch's. nava.hpp NavaHead.forward
   (~line 495). The cpp velocity_video_patched.bin is the int32 write_bin header (NOT the int64
   LONGCAT_DUMP tap format — that bit a measurement this session).
2. **AUDIO CO-DENOISE COUPLING.** The audio velocity (head_audio ~45 dB) feeds back through joint
   attention every step; an audio error compounds into video. The cpp uses the VIDEO neg-prompt for
   the AUDIO uncond — PyTorch uses a SEPARATE audio negative prompt ("机械音、闷糊、回音、失真、
   电流声、爆音、杂音"). Dump it + use it for the audio stream; and validate compute_va's audio
   velocity == compute()'s (the joint pad/concat/split path is unvalidated).
3. Accumulated block error amplified by the head LayerNorm — quantify whether 40 dB is intrinsic
   head error or just the last block's accumulated error normalized.

### KEY INFRA / COMMANDS
- Build: `export PATH=/mnt/hdd/3d/avatar-shootout/toolchain/bin:$PATH; export
  LD_LIBRARY_PATH=/mnt/hdd/3d/avatar-shootout/toolchain/lib; cmake --build build-nava --target nava -j8`.
- Render (832x480, the working bucket): `LD_LIBRARY_PATH=...toolchain/lib build-nava/bin/nava render
  --cuda --gguf models/nava-dit-q8_0.gguf --vae models/wan2.2-vae-48ch-f16.gguf
  --context /mnt/hdd/nava/cpp-runs/_ctx/dense_ctx.bin
  --neg-context /mnt/hdd/nava/cpp-runs/_ctx/neg_ctx.bin --steps 10 --frames 5 --width 832
  --height 480 --seed 42 --cfg 3.0 --out-name X --runs-dir /mnt/hdd/nava/cpp-runs`. (Prints per-step
  latent std — watch it stay bounded ~<1 vs run away to 1.8+.) Frame check: ffmpeg -y -i clip.webm
  -frames:v 1 out.png ; Read the png.
- Per-block diff: ~/dev/NAVA/nava_dump_realctx.py (or nava_dump_tsweep.py for t-param) → npz; then
  `python tools/nava_npz_to_bin.py <npz> <bin> --raw-context` (nava_run_diff.sh OMITS --raw-context,
  add it) → `LONGCAT_DUMP_DIR=<out> build-nava/bin/nava models/nava-dit-f16.gguf <bin> <out>` →
  `python tools/nava_tensor_diff.py <npz> <out>`. PSNR = 20log10(peak)−10log10(mse).
- Per-step sampler diff: ~/dev/NAVA/nava_sampler_compare.py vs NAVA_DUMP_TRAJ cpp dumps.
- MEMORY: the PyTorch standalone backbone load is ~13GB / ~240s; the full nava_dump* peak ~27GB on
  this 31GB box — run ONE at a time, never alongside a cpp F16-harness CPU load (12.6GB) → OOM.
  Use /mnt/hdd/nava/.venv/bin/python with SETUPTOOLS_USE_DISTUTILS=stdlib; do NOT hide CUDA
  (CUDA_VISIBLE_DEVICES="" breaks t5's class-default torch.cuda.current_device()).
- Contexts already dumped in _ctx/: dense_ctx.bin (344 tok, smoke prompt), neg_ctx.bin (153 tok,
  video_negative_prompt), *_real.bin (unpadded). Dump more via ~/dev/NAVA/dump_two_ctx.py.

DEFERRED once coherent: wire UniPC (optional); audio VAE decode (Phase 3, audio latent already
co-denoised); I2V anchor + the REAL I2AV target (vtuber: image + audio-sync + promptable actions);
Q4_K (Q4_0 used as expedient; quant is NOT the bug); perf.
