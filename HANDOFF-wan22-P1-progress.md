# Wan2.2 + InfiniteTalk port — P1 progress (worktree `wan22-infinitetalk`)

Continues `HANDOFF-wan22-infinitetalk.md`. Branch `wan22-infinitetalk` off master `a49e662`. HX370 CPU guinea pig.

## P0 (DONE)
CPU build green (`cmake -B build -DSD_CUDA=OFF -DSD_HIPBLAS=OFF && cmake --build build -j`, ~80s, 0 err →
sd-cli/sd-server/nava/sd-s2v). NAVA/LTX CPU smoke rendered a webm. Fix: master called ggml-cuda symbols
unguarded — re-added `add_compile_definitions(SD_USE_CUDA)` (CUDA/HIP/MUSA) + `#ifdef SD_USE_CUDA` on 5 sites
(ggml_extend.hpp ×2, longcat_avatar.hpp ×3). vae-roundtrip is stale (removed API) → EXCLUDE_FROM_ALL.

## P1 — Wan2.2-S2V onto the worktree (✅ COMPLETE)
GATE MET: render3 (no env var, baked CPU fixes) wrote build/s2v-runs/smoke/{frame_0000-0009.png, s2v.mp4
(h264 192² + aac), s2v_final_latent.bin}. Talking-head renders end-to-end on HX370 CPU, M1 non-causal
distilled 4-step. (M2 --causal path compiles + is wired but not yet runtime-tested on CPU.)

Cherry-picked `origin/wan-s2v-port` (merge-base `b804f0b`, 13 commits; master 98 ahead + restructured
`src/` → `src/model/...`). Done by hand (clean cherry-pick impossible due to restructure):

**New files (extracted verbatim, includes fixed for master layout):**
- `src/wan_s2v.hpp` (1419L), `src/wav2vec2.hpp` (782L) — compiled clean against master, ZERO API drift.
- `examples/s2v/{main.cpp 777L, CMakeLists.txt}` — fixed includes (`model/te/t5.hpp`, `model/diffusion/wan.hpp`,
  added `model/vae/wan_vae.hpp` for WAN::WanVAERunner), added `src/core` include dir, replaced removed
  `ggml_backend_cpu_init()`→`sd_backend_cpu_init()` ×3.

**Deltas applied to master files:**
- `src/model/diffusion/wan.hpp`: WAN_GRAPH_SIZE 10240→20480; appended `forward_kv_cache` + `prefill_cond_kv`
  to WanSelfAttention (API matched master exactly — Rope::apply_rope, allow_fused_rope, ggml_ext_attention_ext).
- `src/model.h`: VERSION_WAN_S2V (appended last before VERSION_COUNT). `src/stable-diffusion.cpp`: matching
  "Wan 2.2 S2V (LiveAvatar)" string (last). `src/model_loader.cpp`: S2V detection (audio_injector.injector.0.q
  / trainable_cond_mask). `src/core/ggml_extend.hpp`: segment_readback_hook_ member + call in execute_graph.
- `examples/CMakeLists.txt`: add_subdirectory(s2v).
- **DEFERRED** `src/convert.cpp` delta (+15, conversion-only ctx sizing) — not needed to render pre-made GGUFs.
- **NOT pulled** the branch's ggml submodule pointer (kept master's newer ggml `19727d01`).

**Models** (local `models/`, pulled from 3060): wan-s2v-14b-dit-dmd-q4_k (9.4GB), wav2vec2-xlsr53-f16,
longcat-umt5-xxl-q8_0, longcat-wan-vae-f16, ref_singer.png, speechA_1s.wav.

**Load test PASSED** (`--load-only`): DiT 1242 tensors (8.8GB, detected "Wan2.2-S2V-14B"), wav2vec2 420,
casual_audio_encoder 12 (bundled in DiT gguf → `--audioenc` = same dit). flash_attn=1.

**CPU runtime fixes (BAKED into examples/s2v/main.cpp):**
1. The fork's `ggml_rope_pe` (ROPE_PE) op is CUDA-only → CPU `ggml_graph_plan` aborts
   `op not implemented: ROPE_PE`. Fixed: `if (cpu) setenv("LONGCAT_NO_FUSED_ROPE","1",0)` (decomposed rope).
   Watch for more CUDA-only ops on CPU (concat_T_cont_4d, madd-fuse) on bigger configs — none hit at 192²/13f.
2. The example never created `--out`; it relied on the harness `mkdir -p`. Fixed: `system("mkdir -p out_dir")`
   up front (else PNG writes silently no-op + ffmpeg redirect fails).

**Render gate: COMPUTE PROVEN.** render2 (with env var) ran end-to-end on CPU EXIT 0 in **867s (~14.5 min)**:
umT5 encode → ref VAE encode → audio (wav2vec2+cae) → DiT 4 distilled steps (flash-attn on CPU) → Wan VAE
decode 56s → **valid decoded RGB [192,192,13,3], mean 0.47 std 0.27** (real frames, no NaN), 10 video frames,
"M1 forward/sampler completed." Only the file-save no-op'd (missing dir — now fixed). Re-running with the
baked binary (no env var) to capture the artifact: `./build/bin/sd-s2v --dit models/wan-s2v-14b-dit-dmd-q4_k.gguf
--wav2vec models/wav2vec2-xlsr53-f16.gguf --audioenc <same dit> --vae models/longcat-wan-vae-f16.gguf --umt5
models/longcat-umt5-xxl-q8_0.gguf --ref-image models/ref_singer.png --prompt "a man singing, photorealistic"
--wav models/speechA_1s.wav --out build/s2v-runs/smoke --height 192 --width 192 --frames 13 --distilled --cpu`.

## START HERE (next agent) — kickoff
- Worktree `~/dev/wan22-infinitetalk`, branch `wan22-infinitetalk`. Build: `cmake -B build -DSD_CUDA=OFF
  -DSD_HIPBLAS=OFF && cmake --build build -j` (~80s, 0 err). Models in `models/`. All P0+P1 changes are
  UNCOMMITTED (13 files): `git -C ~/dev/wan22-infinitetalk diff --stat` + untracked wan_s2v.hpp/wav2vec2.hpp/examples/s2v/.
- CPU mandate: render with `--cpu`; fork custom ops are CUDA-only (ROPE_PE handled; watch for concat_T_cont_4d/
  madd-fuse on bigger configs). Bit-exact mandate = CPU output is the oracle (don't compare to CUDA).
- Reproduce P1: the s2v render command in the P1 section above (no env var needed now).

### Open validation (cheap, optional first): M2 `--causal` S2V on CPU
The causal streaming path (forward_kv_cache + segment_readback_hook_, `wan_s2v.hpp:266/334/1142+`) compiles +
is wired but UNTESTED at runtime on CPU. Run the P1 command + `--causal --nfb 3` to validate before P2.

### P2 — Wan2.2 distilled scene-gen (swap LTX→Wan2.2). See memory `reference_infinitetalk_distill.md`.
1. Pull `lightx2v/Wan2.2-I2V-A14B-Moe-Distill-Lightx2v` (pre-merged 4-step DiT, ~28GB BF16 / ~15GB FP8) →
   convert to GGUF (template: 3060 `/mnt/hdd/live-avatar/convert/convert_wan_s2v_dit.py`). Reuse our T5/VAE.
   Run params: 4 steps (2 high + 2 low expert), cfg=1.0, shift=5.0, Euler.
2. **MoE GAP:** `wan.hpp` has NO high/low expert switch (the A14B distill is 2 experts). Implement the
   per-sigma-boundary expert swap — the gating P2 code task.
3. Parameterize NAVA's chain (`examples/nava/main.cpp` NavaRunner factory ~line 1083) to drive `WAN::WanRunner`
   (`wan.hpp:889-1066`, I2V via c_concat); reuse warm-start/anchor seam (~1296-1390) + FlowMatchSched; drop audio.
   Gate: multi-segment Wan2.2 scene clip, faces clean, seam-free.

### P3 — InfiniteTalk V2V dub. It's Wan2.1-I2V-based (not 2.2). Reuse our wav2vec2 + audio_injector;
new work = input-video conditioning + sparse-frame keyframe anchoring. Distill = lightx2v Wan2.1-480P StepDistill.
Details + weight links in memory `reference_infinitetalk_distill.md`.

### DEFERRED items (don't lose): convert.cpp delta (+15, conversion-only ctx sizing — needed when re-running
convert scripts); HIP/gfx1150 build (P0 stretch, untried — custom kernels may not hipify, CPU is the fallback).
