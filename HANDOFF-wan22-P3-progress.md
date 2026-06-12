# Wan2.2 + InfiniteTalk port — P3 progress (worktree `wan22-infinitetalk`)

Continues `HANDOFF-wan22-P2-progress.md` (read its "InfiniteTalk port SPEC" for the line-precise
tensor map). Branch `wan22-infinitetalk`. HX370 CPU. Nothing committed (user commits).
Build: `cmake -B build -DSD_CUDA=OFF -DSD_HIPBLAS=OFF && cmake --build build -j`. `LONGCAT_NO_FUSED_ROPE=1` on CPU.

## START HERE — the InfiniteTalk port is CODE-COMPLETE + RUNTIME-VALIDATED
The whole pipeline (song → wav2vec2 → AudioProjModel → per-block audio graft → Wan2.1-I2V-14B DiT
→ VAE) compiles, loads, and RENDERS end-to-end. A tiny CPU smoke (128×128, 5 frames, 1 step, 1
window) ran clean: `sd-infinitetalk ... --frames 5 --height 128 --width 128 --steps 1
--motion-frame 1 --max-windows 1 --distilled --cpu` → 5 PNGs + mp4, no NaN/abort, sane stats
(full_audio_emb [768,12,77], audio_emb [768,32,2]). The remaining work is the slow FULL-QUALITY
renders + a torch-oracle correctness pass.

## DONE this session
- **Models** (all in `models/`):
  - `infinitetalk-14b-q4_k.gguf` (10.7GB) — base Wan2.1-I2V-14B 7-shard + MeiGen graft + lightx2v
    rank64 distill, all 1262 LoRA deltas folded; 1633 tensors. **Load-tested green.**
  - `chinese-wav2vec2-base-f16.gguf` (189MB) — `tools/convert_chinese_wav2vec2.py` (HF →
    `audio_encoder.*`, weight-norm pos_conv fused, zero conv biases for the conv_bias=False base).
  - Reuse: `longcat-umt5-xxl-q8_0.gguf` (T5), `longcat-wan-vae-f16.gguf` (16ch Wan2.1 VAE).
- **Code**:
  - `src/infinitetalk.hpp` — `AudioCrossAttention` (fused kv_linear[2*dim,768] split k=first/v=last,
    NO qk_norm), `InfiniteTalkAttentionBlock` (= WanAttentionBlock + graft after text-xattn before
    FFN, all 40 layers), `AudioProjModel`, `build_audio_proj_inputs` (host first/mid/last window
    gather), `InfiniteTalkRunner` (compute + compute_audio_embedding).
  - `examples/infinitetalk/main.cpp` + CMake — streaming driver (81-frame windows, motion-frame
    pinning, 3-way/2-way/distilled CFG), CLIP-H/VAE/umT5/wav2vec wiring. Target `sd-infinitetalk`.
- **Bug fixed (was latent in BOTH converters):** `patch_embedding.weight` must merge `out*in`
  (keep kt=1) → numpy `[out*in,1,kh,kw]`, NOT squeeze to `[out,in,kh,kw]` — the C++ Conv3d wants
  ggml ne `[kw,kh,kt,in*out]` (`ggml_ext_conv_3d`: OC=ne[3]/IC). Fixed `convert_wan_dit.py` +
  `convert_infinitetalk_dit.py`. (memory `gotchas_wan_patch_embedding_conv3d`.)
- **A14B experts** re-converting now with the fixed `convert_wan_dit.py` (the old
  `wan22-i2v-a14b-{low,high}-q4_k.gguf` had the squeeze bug → would fail Conv3d load). `logs/a14b-reconvert.log`.

## ✅ P0 RESOLVED (2026-06-12) — blank cond-frame fill was BLACK, must be NEUTRAL
**Root cause:** `examples/infinitetalk/main.cpp` built the c_concat conditioning video with the
non-cond ("blank") frames filled at `0.0f` in [0,1]. `vae->encode()` applies `scale_input`
([0,1]→[-1,1], `vae.hpp:139`), so `0.0` became pixel **-1 (pure black)** → the conditioning latent
for generated frames was negative-mean / high-std (`y_vae f1-5 mean≈-0.28 std≈1.5`) → the DiT
predicted a +biased velocity → monotonic NEGATIVE latent drift → dark+green decode. The reference
(`wan_multitalk.py:571`) uses `torch.zeros` but its images are in **[-1,1]** (zeros = neutral gray);
sd-cli's i2v path uses `full(0.5)` (also neutral). **Fix: fill blanks with `0.5f`** (→ 0.0 in [-1,1]).
One line: `main.cpp:485` `0.0f`→`0.5f`.

**How it was pinpointed (the decisive discriminator, per the handoff plan):** converted a GRAFT-FREE
base Wan2.1-I2V+lora GGUF (`tools/convert_infinitetalk_dit.py` now has optional `--graft`;
`models/wan21-i2v-base-lora-f16.gguf`) and ran it through sd-cli's KNOWN-GOOD i2v path
(`logs/it-base-sdcli.log`, needed two small sd-cli patches: only build clip_vision when
`--clip-vision` is given, and feed zero clip_fea for Wan2.1-I2V when none loaded). Result: base
weights render CLEAN (final_latent f1-5 mean≈0, std grows 0.51→0.83) ⟹ **weights + distill-fold +
shared sampler EXONERATED**; bug was in my example. Static-diffed my `InfiniteTalk::forward` vs
`WAN::Wan::forward_orig` (identical), sigma schedule + ts=sigma*1000 vs sd-cli `DiscreteFlowDenoiser`
(identical), euler sign vs `wan_multitalk.py` (identical), then the `y_vae` per-frame dump exposed
the negative-mean blank latents → traced to the `0.0` fill + `scale_input`.

**VERIFIED (`it_fix/`, seed 42, same 256px/21f/4-step config):** final_latent f1-5 now mean≈0
std 0.72→0.87 (matches healthy base). Pixels: brightness FLAT 115.8→115.6 (was collapsing
115→57), Gtint FLAT +4.5→+3.4 (was climbing +4.5→+12.7), f0→f1 d|Δ|=2.0 (was 38.9 cliff). Dark
drift + green cast GONE. Tool `tools/frame_stats.py` does the per-frame RGB/brightness/Gtint compare.
Residual: inter-frame motion is small (d|Δ|~0.6-2 vs A14B's ~4-6) — expected for a cfg=1 portrait
talking-head; revisit only if a real audio-dub render looks static.

## ✅ LIP-SYNC VERIFIED + TRAINING RES (2026-06-12)
Audio graft proven to drive the mouth. Test: `models/lenna_face.jpg` (face fills frame) + real
`models/jfk_speech.wav` (JFK, 11s 16k mono) at **448×448** (training-res-adjacent), distilled 4-step,
45f/1.8s, `it_lipsync/`. Mouth opens/closes/changes shape with speech: f0 closed, f11 open "ah",
f34 rounded "oo", f15 (quiet) closed. `tools/mouth_motion.py`: mouth-ROI motion = **2.75× control**
(forehead/bg), loud passages → high mouth motion, quiet → low (corr +0.17 — undersold by frame-diff
vs same-frame-RMS phase noise; visual + ROI ratio are conclusive). Latents healthy at 448 too
(std 0.55→0.85). **The 256px tests were ~6× below training res → mouth ~15px, unresolvable; that
masked the lip motion.**
**Training buckets (repo `ASPECT_RATIO_627`/`_960`):** 480-bucket ≈640² area (640×640, 576×704,
512×768…), shift **7**; 720-bucket ≈960² area, includes **1280×704** (the LTX-test res), shift **11**.
We've been on shift 5 (lightx2v-distill value) — shift is res-coupled, revisit at full res. Output
aspect follows the input image.

## AUDIT FINDINGS — first MVP render (256px/21f/4-step distilled, CPU, NO clip)
Output is COHERENT + photorealistic (man + train scene; `it_mvp/`) — proves the whole stack
loads/denoises/decodes correctly. BUT two quality defects to chase:
  1. **NO MOTION** — essentially a still frame across all 21 (6 latent) frames; the subject doesn't move.
  2. **Green-moss decay** — crisp at frame 0, progressively darker → green cast → blur by frame 20.
**DISCRIMINATOR RESULT (rung 3, A14B via sd-cli — the independent known-good Wan path):** A14B renders
CLEAN — crisp, coherent, NO green cast, and it MOVES (pose changes f1→f12, inter-frame mean|Δ|≈4-6,
brightness stable 114→110). `it_a14b/clip.avi`. ⟹ **all SHARED code is good** (VAE encode/decode,
sampler, CPU build, converter + patch_embedding fix, q4_k experts). **The InfiniteTalk defect is in
`infinitetalk.hpp`, NOT shared code.**

**PINPOINTED (instrumented render `it_instr`, per-frame latent/velocity dumps):** the collapse is in
LATENT space and is a biased velocity, NOT a VAE/decode/euler/graft issue:
  - final_latent f0 (anchor) mean=-0.01 std=0.50 healthy; generated f1→f5 drift monotonically to
    mean=-0.17 (NEGATIVE), std~0.5. (A14B healthy: mean~0, std grows to 0.79.)
  - v@step f1-5 have POSITIVE mean (+0.06→+0.16); euler `x += v*(σ_next-σ_t)` (dt<0) ⟹ drives latent
    NEGATIVE ⟹ dark/green decode. Sign/convention VERIFIED correct (matches multitalk.py + A14B).
  - c_concat / y_vae sane + match the reference recipe; graft RULED OUT (off = darker).
  ⟹ the DiT predicts a biased velocity for GENERATED frames from OOD conditioning. c_concat matches +
  text is real ⟹ **prime cause = `clip_fea` (Wan2.1-I2V is TRAINED with 257 CLIP tokens; zeros = OOD)**.
  **CLIP RULED OUT (tested 2026-06-12):** `tools/precompute_wan_clip.py` runs the Wan visual tower →
  real clip_fea (std 1.01, NOT zeros), fed via `--clip-fea models/ref_singer_clipfea.bin`. Render
  `it_realclip`: final_latent f1-5 STILL drift -0.09→-0.18 (≈ unchanged), velocity still +biased, pixels
  still collapse. So clip starvation is NOT the cause. (clip path IS live — tiny latent delta — just not it.)

  **STATUS: graft RULED OUT, clip RULED OUT, VAE-decode RULED OUT (f0 decodes clean), euler-sign VERIFIED.**
  Remaining suspects for the generated-frame +velocity→-drift (a genuine multi-hypothesis dig — HANDOFF point):
  1. **DiT-weights vs my-example-code** — DECISIVE next test: run the InfiniteTalk GGUF through sd-cli's
     KNOWN-GOOD i2v path (`sd-cli -M vid_gen --diffusion-model models/infinitetalk-14b-q4_k.gguf` + vae +
     t5xxl + `--init-img` + `LONGCAT_NO_FUSED_ROPE=1`; sd-cli ignores the graft tensors → runs base Wan2.1-I2V).
     Clean → bug is in MY example (sampler/c_concat/motion-pin); dark-drift → the DiT WEIGHTS (lightx2v
     distill-fold math in convert_infinitetalk_dit.py, or the merge). (Caveat: sd-cli i2v wants clip_vision
     w/ the same `visual.*` naming → may need the name-map; or test the base A14B-style without clip.)
  2. **distill-fold correctness** — convert_infinitetalk_dit.py folds lightx2v `lora_down/up`+`diff`+`diff_b`
     with scale 1.0/no-alpha. If the alpha/scale is wrong, generation is biased. Compare a NON-distilled
     merge (drop --lora) render: if the drift vanishes, the fold is the bug.
  3. **my sampler vs sd-cli** — the distilled_sigmas(4,shift=5) schedule / ts=sigma*1000 / cfg. Drift is
     MONOTONIC with frame-distance from the anchor → smells temporal (RoPE t_len, or conditioning reach),
     not a global schedule issue.
  4. c_concat mask/channel semantics (match multitalk.py:544-581, but double-check vs sd-cli's i2v concat).
  Instrumentation is in `examples/infinitetalk/main.cpp` (dump_per_frame, IT_NO_GRAFT) — reuse it.

**SHARPENED SYMPTOM (rung 1 numbers):** it's not a dead temporal axis. latent-frame 0 (the pinned clean
cond frame) is bright+crisp (mean~115); the transition 0→1 is a CLIFF (mean|Δ|=56.7, brightness 115→63);
generated latent-frames 1..5 then change ~4-5/frame (≈ A14B's motion magnitude) but sit dark/green
(mean ~57-67). So the GENERATED frames are produced in a SHIFTED (dark/green, channel-biased) distribution
vs the anchor — a conditioning/normalization defect, not "no motion."
A/B RESULTS (256px, seed 42, same config):
  - **GRAFT EXONERATED.** graft-OFF (`IT_NO_GRAFT=1`) made it WORSE — near-black (mean 14-23) vs graft-ON
    dark-green (mean 57-67, G highest). So the graft ADDS energy (green tint); it is NOT the collapse cause.
    The dark collapse of generated frames happens WITH OR WITHOUT audio → it's the BASE conditioning path.
  - **PRIME SUSPECT NOW: `clip_fea`=zeros.** rung 1 + the A/B both ran with NO CLIP image tokens (rung 2,
    which supplies them, had crashed). Base Wan2.1-I2V leans on the 257 CLIP image tokens for the appearance
    anchor; zeros → generated frames have no anchor → collapse dark (frame 0 survives only because it's pinned).
    P0 and P1 are likely the SAME bug. **clip_preprocess crash FIXED** (it wants ne [W,H,3,1]; example passed
    [W,H,1,3] — reshape, not permute, since T=1). Decisive with-CLIP render running (`logs/it-clip2.log`,
    `it_clip/`): if frames brighten + lose the green/black, P0 is resolved by image conditioning.
  - if WITH-CLIP still collapses → next: hyp-vae-space (`(mu-mean)/std` on c_concat y / motion latents), then
    the c_concat mask/channel order. (A14B uses same VAE + is clean, so VAE decode itself is fine.)
  - hyp-temporal RULED OUT — generated frames vary ~4-5/frame.
  - **CLIP IS INERT (proven):** with-CLIP render is BYTE-IDENTICAL (md5) to the zeros run → clip_fea makes
    zero difference. clip_preprocess crash fixed + CLIP "loads", but the Wan `.pth` uses NON-STANDARD naming
    `visual.{cls_embedding,pos_embedding,pre_norm,patch_embedding,head,...}` (392 vision tensors) that
    name_conversion.cpp does NOT map to the C++ CLIPVisionModelProjection (HF) scheme → vision tower loads
    PARTIALLY → degenerate (~0) output. So the P0 "does image conditioning fix the collapse" test is STILL
    UNANSWERED — we've never fed working clip_fea.
  - **TO RESOLVE P0, get working clip_fea, two paths:** (a) write a Wan-CLIP `visual.*` → C++ vision name-map
    (involved; verify the ViT-H vision arch matches), or (b) PRECOMPUTE clip_fea in Python (load the .pth,
    run the vision tower, save [1280,257] .bin) and feed `--clip-fea <bin>` (the example already supports it).
    (b) is the faster sidestep. THEN re-render: if collapse clears → P0 was clip starvation; if not → hyp-vae-space.

## PROOF MATRIX (MVP runs, 256px CPU, 2026-06-12)
| rung | what | result |
|------|------|--------|
| 1 | InfiniteTalk, no clip | ⚠️ coherent imagery, but generated frames collapse dark/green (P0 bug, infinitetalk.hpp) |
| 2 | InfiniteTalk + CLIP-H | ❌ crash in `clip_preprocess` — CLIP MODEL loads (520 tensors, partial name-map) but my example passes the wrong image-tensor shape to clip_preprocess (`clip_of` lambda). Fixable in main.cpp. |
| 3 | A14B I2V MoE (sd-cli) | ✅ clean, coherent, MOVES (it_a14b/clip.avi). P2 generator proven; shared code (VAE/sampler/build/convert) good |
| 4 | InfiniteTalk streaming 2-win | ✅ machinery works — 2 windows, motion-frame carry (1→5), audio advance (0→16), 37 frames; quality inherits the P0 green bug |
| 5 | VACE-Fun-A14B | ✅ both experts convert (q4_k 9.87GB ea, flat-Wan + patch_embedding fix held) + render coherent (it_vace/clip.avi); vace_blocks + 96ch vace_patch_embedding branch live |

PRIORITIZED FIX LIST:
- **[P0] InfiniteTalk generated-frame dark/green distribution shift** (see AUDIT FINDINGS). Top suspect:
  audio-graft bias. First A/B: `audio_emb`=zeros — green gone ⟹ graft (check fused-kv split / proj / scale).
- **[P1] CLIP-H `clip_preprocess` crash** — fix the image-tensor shape in the example `clip_of` lambda
  (clip_preprocess expects a specific layout; we pass `[W,H,1,3]`). Plus the partial name-map (some CLIP
  tensors fail to load) — extend name_conversion or precompute clip_fea.
- **[P2] oracle items** (VAE space, motion-pin, kv split, distill fold) once P0/P1 clear.

## DO NEXT
1. **A14B cheap-floor render** (the GGUFs finish reconverting via `logs/a14b-reconvert.log`):
   `sd-cli` with `--diffusion-model models/wan22-i2v-a14b-low-q4_k.gguf
   --high-noise-diffusion-model models/wan22-i2v-a14b-high-q4_k.gguf --moe-boundary 0.875` + umT5 +
   `longcat-wan-vae-f16.gguf`. Single promptable I2V clip — validates the P2 scene generator.
2. **InfiniteTalk full dub render** — scale the smoke up: `sd-infinitetalk --dit
   models/infinitetalk-14b-q4_k.gguf --wav2vec models/chinese-wav2vec2-base-f16.gguf --vae
   models/longcat-wan-vae-f16.gguf --umt5 models/longcat-umt5-xxl-q8_0.gguf [--clip-vision
   models/dl/wan21-i2v-14b-480p/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth] --image
   <png> --wav <song.wav> --prompt "..." --frames 81 --height 480 --width 480 --distilled --cpu`.
   480×480×21 latent on CPU is multi-hour; consider the 3060 for the real run.
2b. **VACE continuation path (seamless takes, optional)** — `src/vace.hpp` (`build_vace_context` →
   96ch, compiles) + DiT support already in `wan.hpp`. Full wiring sketch + model convert + test loop
   in **`VACE-NOTES.md`**. Needs `Wan2.2-VACE-Fun-A14B` downloaded + converted; distill-fold unverified.
3. **Oracle validation** (correctness, deferred — pipeline RUNS but is not bit-checked):
   - **CLIP-H path** — the `--clip-vision` open-clip-xlm-roberta `.pth` name-conversion is untested;
     without it `clip_fea` falls back to zeros (degrades identity). Verify FrozenCLIPVisionEmbedder
     loads it and produces [1280,257], or precompute offline and pass `--clip-fea <bin>`.
   - **VAE latent space** — c_concat `y` and the motion latents are run through `vae_mu_to_diffusion`
     ((mu-mean)/std, the s2v convention). Confirm InfiniteTalk's pipeline uses the same space.
   - **Motion-frame pin** — pinned CLEAN each step (matches the reference's operative
     `latent[:,:m]=motion_latents` at multitalk.py:711/773; the P2 SPEC said "noised" but the
     reference overwrites clean — the noised re-inject at :768-771 is overwritten by :773).
   - **kv_linear split / window gather** — spot-check against a torch dump of audio_embedding.

## KEY FILES
`src/infinitetalk.hpp`, `examples/infinitetalk/main.cpp`, `tools/convert_infinitetalk_dit.py`,
`tools/convert_chinese_wav2vec2.py`, `tools/convert_wan_dit.py`. Reference: `/tmp/infinitetalk-src`
(re-pull github.com/MeiGen-AI/InfiniteTalk if gone). Smoke output: `it_smoke/`.
