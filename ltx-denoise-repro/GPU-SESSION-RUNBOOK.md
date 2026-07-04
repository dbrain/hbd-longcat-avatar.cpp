# GPU-SESSION RUNBOOK — LTX-2.3 "Denoise-AI workflow" port

Everything below is prepped CPU-side and waiting for GPU/build time. This is the single
actionable sequence; the other docs are the detail (see **Pointers** at the bottom).

## State at handoff (all CPU work done)
- **Worktree**: `/home/dbrain/dev/longcat-avatar-ltxdenoise` (branch `ltx-denoise-workflow`).
- **Folded models ready** (byte-identical nvfp4 layout to `nvfp4-CLEAN.gguf` ⇒ same VRAM + per-step speed):
  - `models/ltx2/nvfp4-CLEAN-dev050.gguf`  = dev + distill@0.50 (matches workflow v2, the clean reference)
  - `models/ltx2/nvfp4-CLEAN-dev065.gguf`  = dev + distill@0.65 (matches workflow v1 base stage)
  - Caveat: the fold de-distills the 1344 nvfp4 Linears; 316 bf16 modulation/adaln/audio tensors are
    left at full-distill (copied verbatim). Likely minor; eye-test gates. Can extend the fold to bf16 later.
- **C++ BUILT ✅** (8 files, +404/−41): `--hires-lora`, NAG, base `--sigmas`. **Compiled clean** (no source
  fixups) into `build-cudnn/bin/sd-cli` (arch 120/Blackwell, cuDNN+FA+NCCL, Release). All new CLI flags
  verified present in the binary: `--nag-scale/-alpha/-tau/-until-sigma`, `--hires-lora`, `--sigmas`.
  NOTE: binary is root-owned (built in the `longcat-avatar-dev:builder-cudnn` container) — run it via that
  container (mount worktree at `/src`, `/src/build-cudnn/bin/sd-cli`), same as `iter_seg2.sh`.

## STEP 1 — BUILD ✅ DONE (compiled clean; the compile needed no GPU)
Rebuild command (worktree source, ccache-backed, incremental) if you touch the C++ again:
```
docker run --rm -v /home/dbrain/dev/longcat-avatar-ltxdenoise:/src \
  -v longcat-avatar-iter-ccache:/root/.ccache -w /src longcat-avatar-dev:builder-cudnn \
  bash -lc "cmake --build /src/build-cudnn -j\$(nproc) --target sd-cli"
```
(Worktree submodules were initialized — `git submodule update --init --recursive` — required once.)
The runtime **BUILD-VERIFICATION CHECKLIST** in `CPP-CHANGES.md` still applies at EVAL time (these are
semantic, not compile, risks — they surface as wrong output, not build errors):
1. **NAG scale convention**: agent used `scale` as the extrapolation coeff; ComfyUI likely uses `scale−1`
   ⇒ workflow "14" ≈ our **`--nag-scale 13`**. Confirm at eval.
2. **pos/neg context length must match** (neg reuses the positive connector RoPE pe sizing).
3. **per-token L2 axis / blend space** in `CrossAttention::forward` (verify `to_out.0` output shape
   `[query_dim, tokens, batch]` before trusting `ggml_sum_rows`).
NAG-off is gated byte-identical (`nag_context==null` short-circuit), so a broken NAG can't taint the
non-NAG A/B runs — safe to build and run steps 2.1–2.2 even if NAG needs more work.

## STEP 2 — RUN THE ABLATION (turnkey — `run_ablation.sh` does everything)
The whole ladder (baseline → fold → +ladder → +hires-lora/detailer → +NAG → 24fps) is encapsulated in
`run_ablation.sh`: fixed prompt+seed per scenario, real `sd-cli` invocation (worktree binary in the
`builder-cudnn-ff` runtime), AV-webm → mp4 (keeps audio), auto-surfaced to the eye-test page + a results
table. GPU-idle preflight + flock built in. Copy-paste:

```
cd ltx-denoise-repro
# --- t2v, scenario 3 (static-cam distant crossing = purest poison) — the first run ---
bash run_ablation.sh 3 t2v "0 1 2 3"          # baseline, fold, +ladder, +hires-lora/detailer
# --- t2v with supplied audio (music) ---
AUDIO=/home/dbrain/dev/longcat-avatar.cpp/models/ltx2/_drive/song.wav bash run_ablation.sh 3 t2v "2 3"
# --- i2v: FIRST flux the distant start frame, then animate it ---
bash gen_i2v_stills.sh                          # writes models/ltx2/_inputs/i2v_start_{crossing,crowd}_*.png
INIT=/home/dbrain/dev/longcat-avatar.cpp/models/ltx2/_inputs/i2v_start_crossing_seed42.png \
  bash run_ablation.sh 3 i2v "0 1 2 3"
# scenario 1 i2v uses the neon still by default (no gen_i2v needed): bash run_ablation.sh 1 i2v "0 1 2 3"
# --- after eyeballing 0-3, pick the winner model then add NAG + the 24fps test ---
NAGMODEL=nvfp4-CLEAN-dev050.gguf bash run_ablation.sh 3 t2v 4        # NAG (+ sweep: edit --nag-scale 8/13/16)
WINMODEL=nvfp4-CLEAN-dev050.gguf FR=97 bash run_ablation.sh 3 t2v 5  # 24fps, half the frames = 50%-less-work test
```
Row semantics (each isolates ONE lever): 0 current prod · 1 de-distill fold only · 2 +low-res→x2→refine
ladder · 3 +hires-lora(distill@0.8)+detailer on refine · 4 +NAG(scale13, S1-only sigma gate) · 5 24fps.
**Speech audio = `voice_teen.wav`, music = `song.wav`** (voice_16k dropped; auto-resampled to 16k mono).
**No verdict on LTX until row 4 (NAG) is in the eval.** NAG scale: workflow "14" ≈ our `--nag-scale 13`.

## STEP 3 — JUDGE ON THE PAGE
`run_ablation.sh` auto-builds **http://10.0.0.208:8077/ltx_denoise/ablation.html** (clips side-by-side +
wall/VRAM table). Judge in motion (mush shows in motion, not stills): distant-face fidelity, contrast-on-
motion, does 24fps hold. Win = clean **and** not slower than baseline (ideally faster — fewer full-res steps)
and usable faster than the wan2.2+lipdub fallback. Input vetting page: `…/ltx_denoise/index.html`.

## Usage modes (all wired in `run_ablation.sh` / the driver)
t2v (native AV), t2v+`AUDIO=`(supplied `--drive-audio`), i2v (`INIT=`), and chain continuation (native
`--ltx-chain-segments N --ltx-chain-prompts <file> --ltx-chain-audio-dir <dir>` — see `run_denoise_workflow.sh`).

## Pointers
- **`run_ablation.sh`** — THE turnkey runner (all rows, auto-surface). Start here on GPU day.
- **`gen_i2v_stills.sh`** — flux.2 the busy/distant i2v start frames (scenarios 2 & 3). GPU.
- `PROMPTS.md` — LTX-2.3 prompting cheat-sheet + the 3 fixed pain-point prompts (t2v+i2v).
- `REPRODUCE.md` — node-exact spec of all 3 workflows + minimize/ablation rationale (v2 = reference).
- `CPP-CHANGES.md` — every C++ edit + flags + BUILD-VERIFICATION CHECKLIST.
- `CPP-IMPLEMENTATION-PLAN.md` — the design the edits were built from.
- `COMPAT-REPORT.md` — distill-LoRA bijection + detailer compat facts.
- `run_denoise_workflow.sh` — earlier hand-driver (superseded by run_ablation.sh; kept for the chain mode).

## Pre-GPU status (what's still cooking)
- Folded models `nvfp4-CLEAN-dev050.gguf` (rows 1–2) + `dev065.gguf` (row 3) are **re-folding cleanly now**
  (~1 h each, CPU — the first run got RAM-starved by the concurrent build and was scrapped). Rows 0–2 only
  need dev050; row 3 needs dev065. `run_ablation.sh` skips any row whose model isn't present yet, so you can
  start rows 0–2 the moment dev050 lands even if dev065 is still folding.
- LoRAs for row 3 are symlinked into `models/ltx2/loras/` (distill-384-1.1 + detailer) for `--lora-model-dir`.
- `tools/fold_distill_lora.py` — the strength-parameterized fold (re-run for other strengths).
