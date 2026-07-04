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
- **C++ implemented but UNBUILT** (8 files, +404/−41 — `git diff`): `--hires-lora`, NAG, and the flags below.
  Untested — expect compile fixups from the checklist in `CPP-CHANGES.md`.

## STEP 1 — BUILD (compile the C++ change; no GPU needed for the compile itself)
Build the way prod does — docker container, source mounted at `/src`, into `build-cudnn`
(CUDA arch **120**/Blackwell, `GGML_CUDNN=ON`, `GGML_CUDA_FA=ON`, `GGML_CUDA_NCCL=ON`, Release).
Point the build at THIS worktree's source. Then work the **BUILD-VERIFICATION CHECKLIST** in
`CPP-CHANGES.md` — the top 3 risks to expect:
1. **NAG scale convention**: agent used `scale` as the extrapolation coeff; ComfyUI likely uses `scale−1`
   ⇒ workflow "14" ≈ our **`--nag-scale 13`**. Confirm at eval.
2. **pos/neg context length must match** (neg reuses the positive connector RoPE pe sizing).
3. **per-token L2 axis / blend space** in `CrossAttention::forward` (verify `to_out.0` output shape
   `[query_dim, tokens, batch]` before trusting `ggml_sum_rows`).
NAG-off is gated byte-identical (`nag_context==null` short-circuit), so a broken NAG can't taint the
non-NAG A/B runs — safe to build and run steps 2.1–2.2 even if NAG needs more work.

## STEP 2 — THE ABLATION LADDER (busy clip, hold everything else fixed)
**Fixed clip** = the "LTX poison" case: distant characters + high motion, **1280×704, 193 frames**.
Same init image / prompt / seed across all runs. Driver = `run_denoise_workflow.sh` (update its model +
flags per row; it already has the base→x2→refine ladder, audio, chain, and eye-test surfacing wired).

| # | Run | Model | Key flags | Needs |
|---|---|---|---|---|
| 0 | **Baseline** (current prod) | `nvfp4-CLEAN.gguf` | current single-pass, **48 fps** | build-free |
| 1 | **Fold only** | `nvfp4-CLEAN-dev050.gguf` | same single-pass, 48 fps | build-free |
| 2 | **+ ladder** (low-res base→x2→refine) | dev050 | `--sigmas` base + `--hires --hires-upscaler ltx-2.3-spatial-upscaler-x2-1.1 --hires-sigmas 0.421875,0.0` | build-free |
| 3 | **+ per-phase LoRA + detailer** | dev065 base | `--hires-lora "ltx-2.3-22b-distilled-lora-384-1.1:0.15,<detailer>:0.8"` (bumps refine toward 0.8 + adds detailer) | **build** |
| 4 | **+ NAG** | winner of 2/3 | `--nag-scale 13 --nag-alpha 0.35 --nag-tau 2.5 --nag-until-sigma 0.9` (and a scale sweep 8/13/16) | **build** |
| 5 | **24 fps** re-run of the best | winner | drop to `--fps 24` (workflow's own default) | as above |

Notes:
- Row 3's `--hires-lora` strength is **incremental over the folded base**: dev065 already bakes 0.65, so
  adding the distill LoRA at +0.15 on the refine pass ≈ effective 0.8 there (verify sign/convention on build).
- Rows 2–5 each isolate one lever ⇒ we read off its exact contribution. **No verdict on LTX until row 4/5.**
- Detailer file: `ltx-2-19b-ic-lora-detailer` (Agent B confirmed 480/480 dim-compatible with our 22B trunk).
- fps=24 default comes straight from the workflow (all 3 versions run 24). Row 5 is the "50% less work" test.

## STEP 3 — SURFACE + JUDGE
Each clip → webm→mp4 (keep audio) → eye-test dir → regen page (driver does this;
pattern from `gen_eyetest.sh`, owner's LAN page). Compare in motion (mush shows in motion, not stills):
distant-character faces, contrast-on-motion, and whether 24 fps holds up. Also log wall-time per run —
the win condition is "clean **and** not slower than baseline" (ideally faster: fewer full-res steps),
and "usable faster than the wan2.2+lipdub fallback".

## Usage modes (all wired in the driver, `--mode`)
`i2v_audio` (`--init-img` + `--drive-audio <16kHz wav>` + `--audio-vae ...-ENC-f16.gguf`),
`t2v_audio`, `t2v_genaudio` (omit `--drive-audio` → native AV), `chain` (native
`--ltx-chain-segments N --ltx-chain-prompts <file> --ltx-chain-audio-dir <dir>` for same-character continuation).

## Pointers
- `REPRODUCE.md` — node-exact spec of all 3 workflows + minimize/ablation rationale (v2 = reference).
- `CPP-CHANGES.md` — every C++ edit + flags + BUILD-VERIFICATION CHECKLIST.
- `CPP-IMPLEMENTATION-PLAN.md` — the design the edits were built from.
- `COMPAT-REPORT.md` — distill-LoRA bijection + detailer compat facts.
- `run_denoise_workflow.sh` — the driver (update model/flags per ablation row).
- `tools/fold_distill_lora.py` — the strength-parameterized fold (re-run for other strengths).
