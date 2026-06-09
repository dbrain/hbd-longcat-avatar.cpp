# NAVA cpp port — NEXT TASKS (post-decode-parity roadmap)

Self-contained roadmap for a fresh agent (assume NO shared memory). Pick this up once
the decode-parity work (`HANDOFF-nava-DECODE-PARITY.md`) is wrapped. That doc has the
full environment details — build, run/render commands, the python reference + decode
oracle, the eye/ear review servers (:8097 video, :8099 audio), and the low-memory/OOM
survival notes. READ IT FIRST for the mechanics; this doc is the what-and-why of the
remaining work.

## Context in one paragraph
C++ port of NAVA (ernie-research, 6.3B joint audio+video MMDiT) at
`/home/dbrain/dev/longcat-avatar.cpp` branch `nava-port` (sd.cpp/ggml stack; python
reference `/home/dbrain/dev/NAVA`). It renders joint audio+video (T2V/I2V + speech)
from a prompt. The garbled-speech bug (umT5 `<extra_id_2>` sentinel) is FIXED and
committed; decode-parity (cpp audio-VAE vs python) is the in-flight task. Everything
below is what's still needed for python feature-parity + production deployment as a
koblem GPU engine (the kid's-vtuber/avatar use case). 1× RTX 3060 (12 GB), serial GPU.

Rough priority order for "production-ready avatar": (1) finish decode parity, then
(2) production server + koblem integration, (3) voice clone, (4) VRAM≤7.5 GB + Q4_K +
offload, (5) clip continuation, (6) minor parity items.

---
## TASK 1 — Production server + koblem/kobbler integration  [BIGGEST GAP]
Today NAVA is CLI-only (`nava render ...`, one-shot). Every other cpp GPU service
(flux2, acestep, longcat-avatar) ships as a RESIDENT HTTP server wired into koblem.
Mirror that. Reference implementations to copy patterns from (same repo / sibling repos):
- `src/` sd-server / longcat-avatar server, acestep server — for the HTTP `/generate`,
  `/unload`, idle-watchdog, worker-isolation (fork+IPC → idle VRAM true-0), and
  cooperative cancellation (client-disconnect → abort) patterns.
- koblem (`../koblem`) + koblibs (`../koblibs`): kob-gpu-gate `HeavyKind`,
  `acquire_for_<engine>`, engine registration, a UI page. Search for how `flux2` /
  `acestep` / `longcat-avatar` (HeavyKind variants) are registered and gated.

Subtasks:
1. **HTTP server mode** for the nava binary (resident; `/generate` takes prompt + params
   → returns webm; `/unload`; health). Currently render lives in `examples/nava/main.cpp
   run_render`; factor the pipeline into a server entrypoint.
2. **umT5 ↔ DiT VRAM hand-off** INSIDE the server: umT5 (5.7 GB) and DiT (7 GB)+VAEs
   can't co-reside in 12 GB. Today it's two processes (encode-prompt, then render). The
   server must encode the prompt (load umT5 → context → free) then load DiT+VAEs and
   render — or hold umT5 resident with offload. Decide + implement.
3. **Worker isolation** (fork child does the GPU work; parent stays CUDA-free so idle
   VRAM → 0). Copy acestep/flux2 `*_WORKER_ISOLATION` pattern.
4. **Cooperative cancellation** (reader thread + atomic cancel + watchdog; client
   disconnect → 499, worker stays warm). Copy qwen3-tts/longcat pattern.
5. **koblem engine wiring**: HeavyKind, acquire/preempt (unload on other-engine
   preempt), UI page, request schema (prompt, negatives, image, frames/fps, seed, cfg,
   spk ref, etc.).
6. **Docker image + deploy**: model dir (ggufs), `*_REF` bump pattern, compose entry.
   Bake the standard negatives (currently hand-encoded `vneg_now.bin`/`aneg_now.bin`;
   the canonical strings are in `pipeline_nava.py` sample(): `video_negative_prompt`
   (use_mmdit_model=true variant) + `audio_negative_prompt`).
Acceptance: a warm server that idles at ~0 VRAM, renders peter end-to-end over HTTP,
cancels on disconnect, and is reachable through koblem like the other engines.

## TASK 2 — Voice clone (spk / timbre)  [main functional parity gap]
cpp stubs the speaker path (`src/nava.hpp:614` "speaker_embedding STUBBED"). It's a
cluster, all present in python (`pipeline_nava.py` + `model_mm.py`):
1. **Speaker encoder** (ReDimNet, `IDRnD_ReDimNet`) waveform → spk_emb [1,192]. Port or
   bundle it (python loads via torch.hub cache `~/.cache/torch/hub/IDRnD_ReDimNet_master`;
   the LTX audio VAE's `init_ltx_vae`/adapter also has an encode path that produces
   spk_embs — see `nava_src/vae/local_audio_vae.py` encode()). A cpp speaker encoder +
   the audio-VAE ENCODER are the new model paths needed (cpp currently has decode only).
2. **spk-token splice**: with use_speech_special_token=false the data loader inserts
   `<extra_id_2>` after `<S>` (already handled in encode-prompt); the spk EMBEDDING is
   projected by `backbone.speaker_embedding.net.*` (these tensors exist in the gguf but
   are currently "unknown tensor"/unused) and spliced at the spk-token position
   (`spk_pos`).
3. **Context-merge flip**: when spk_embed is NOT None, `model_mm.py:1644` uses
   `context_audio` (not context_vid) for the audio stream → the audio uncond must then
   use the AUDIO negative (the existing `NAVA_SEPARATE_AUDIO_NEG` branch in
   `examples/nava/main.cpp` is for exactly this; currently default-off because spk=None).
4. **timbre_cfg**: a 4th amplified CFG term (`pipeline_nava.py:551-554`):
   `eps += timbre_align_guidance_scale*(cond − timbre_uncond)` where timbre_uncond is a
   forward with spk_embs=None. Wire `effective_timbre = timbre_cfg and spk_embs is not
   None` + the extra forward + combine.
Validate against a python clone render (python+clone formant/ear vs no-clone).
Acceptance: cpp renders peter in a reference speaker's timbre, matching python+clone.

## TASK 3 — VRAM ≤ 7.5 GB + Q4_K + offload tuning  [perf/prod]
Same budget the avatar/flux2 services hit. Today render uses q8 DiT (~7 GB) + Wan VAE
(1.3 GB) + audio VAE — peaks above 7.5.
1. **Q4_K DiT** gguf (have q4_0; make/validate q4_K) — check quality vs q8 (audio is
   sensitive; q8 was validated cos 0.9999, q4 unknown). Convert via the same
   `tools/convert_nava_dit.py` path.
2. **Deep VRAM/compute profiling** (nsys/ncu in the GPU builder; `--cap-add SYS_ADMIN`
   for hw counters). Find the peaks (DiT sampling vs VAE decode/tiling).
3. **Offload** whatever's needed to hold ≤7.5 GB peak (weight/activation offload, VAE
   tiling already exists). Then perf-tune the offload (overlap H2D with compute — see the
   longcat-avatar prefetch-thread pattern). Target: ≤7.5 GB peak with minimal slowdown.
Acceptance: full render peak VRAM ≤ 7.5 GB, quality held, throughput documented.

## TASK 4 — Clip continuation  [net-new; python has NONE — adapt longcat-avatar chaining]
For long videos / consistent motion across segments. Two parts:
1. **Video continuation**: feed N prior frames as clean-anchor context (generalize the
   existing i2v "first_frame_is_clean" single-anchor path to N anchor frames; per-token
   timestep=0 for the clean frames). Reference the longcat-avatar.cpp segment-chaining.
2. **Audio continuation across segments** (so we DON'T fall back to a separate TTS):
   needs the **LTX audio-VAE ENCODER** wired (cpp has decode only) — encode the prior
   segment's waveform → audio latent → condition the next segment's audio stream. NOTE:
   this is the SAME encoder dependency as voice-clone (Task 2) — wire it once.
Acceptance: a 2+ segment render with visually consistent motion and continuous (non-
restarting) speech across the seam.

## TASK 5 — minor parity (only if needed)
- Other modalities: python supports `audio`/`video`/`image` combos + audio-only /
  video-only (`pipeline_nava.py` modality string). cpp is joint-AV+i2v only. Add if a
  use case needs it.
- Per-request CFG knobs: python exposes separate video/image/audio guidance + align +
  timbre scales; cpp hardcodes some (`cfg_audio=2.0`, etc., `main.cpp:718`). Expose via
  the server request schema.

## Already DONE / don't redo
- Garbled speech: umT5 `<extra_id_2>` sentinel + trailing-space metaspace fix in
  encode-prompt (cpp tokens == HF, 0 diffs). Committed.
- DiT forward / scheduler / align_3d_cfg 3-way CFG: validated faithful (audio velocity
  cos 0.9999 all timesteps; q8≈fp8). Don't re-chase.
- Decode parity: in flight (separate handoff).
