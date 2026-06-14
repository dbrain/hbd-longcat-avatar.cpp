# Mini-prompt — VACE 720p long-form: how close to LTX-2.3, and is the throughput path real? (autonomous)

You're continuing **Wan2.2-VACE-FUN-A14B** (stable-diffusion.cpp fork) on an **RTX 3060 (12GB)**, full autonomy.
**READ FIRST:** `HANDOFF-CONTINUATIONS-3X3.md` (mission + GOAL-1/2 + measured FR/throughput numbers) and
`HANDOFF-VACE-TUNING.md` (FINDINGS-L1..L9). Worktree `~/dev/longcat-avatar-wan22`, branch `wan22-infinitetalk`.
Memory: `project_wan22_infinitetalk_3060`.

**STATE (done + committed 74ba08c..9a633be — do NOT re-chase):** the sampler-schedule "murk" is FIXED and the
quality config is settled. Use `WAN_DISTILL_SIGMAS=1 --high-noise-steps 3 --steps 3` (the 3+3 DMD grid, fixed
shift 7; HIGH steps=coherence, LOW=grain) + `--max-vram 7.3 --vae-tile-overlap 0.25` + `VACE_GRAY_CACHE_DIR` +
`--sampling-method euler --high-noise-sampling-method euler` + `--cfg-scale 1`. DiT kernels are at the silicon
floor (L6/L7) and 3+3 is the quality floor — the throughput gap will NOT come from steps or kernel tuning.

**MISSION (fixed): can VACE do long-form directed video at 720p (1280×704) anywhere near LTX-2.3 throughput?**
LTX = **111 render-s/s-video**, ONE continuous ~27s clip. The test is a SINGLE continuous (continuation-chained)
video, NOT a montage. **720p is the target — NO 480p pivot.** Current 720p: ~2× LTX (FR=13) .. 4.2× (FR=21); the
wall is **O(L²) attention** (tokens ∝ latent frames, so long segments are quadratically slow). FR ceiling already
measured: FR=21 fits (peak 11GB), FR=25 OOMs (L9).

## PHASE A — is the throughput path worth chasing? (FIRST; research + cheap measurement)
1. **Off-the-shelf scan (research, NO GPU — spawn a research subagent).** Before ANY fork-class work, find whether
   a DOCUMENTED efficient/sparse attention for Wan2.2 video exists and is portable to our ggml/flash-attn path on
   a 3060. Cover at least: Sliding-Tile Attention (STA / FastVideo), Sparse VideoGen (SVG/SVG2), Radial Attention,
   SageAttention (quant attn), the Wan repo's own attention options, ComfyUI Wan sparse-attn nodes, lightx2v's
   attention. For each output: claimed 720p speedup · quality cost · port effort to sd.cpp/ggml · reference impl ·
   **"worth chasing? y/n"**. Deliver a ranked table. This decides whether near-LTX 720p is plausible at all.
2. **720p long-form throughput knee (GPU).** Measure continuation seg time at 3+3 across FR=13/17/21 (control path
   adds buffer + tokens vs t2v) × seam count for a 27s clip → the FR that minimizes TOTAL 27s render time while
   holding coherence. Report render-s/s-video vs LTX 111 per FR.

## PHASE B — seam validation + the one continuous video
3. **Seam eye-test at 3+3** (same runs as A2): one scene as a 2–3 seg continuation chain (`run_vace_musicvideo.sh`,
   already set to 3+3), inspect the JOIN frames (K-1,K,K+1): subject/motion carry? grain/brightness jump? The
   headline unknown is whether the seam holds with the fixed schedule (PRE-fix it was "mild, not invisible").
4. **If a Phase-A lever is worth it**, prototype the cheapest one (likely STA static local-window mask) behind an
   env flag; A/B speed + quality (coherence gate: faces/motion/seam; bit-exact NOT required — it's a quality trade,
   log the delta vs dense). Then render the ONE continuous ~27s 720p video and report render-s/s-video vs LTX.

## RULES
- GPU shared with prod; ONE job at a time; drive from the MAIN loop `run_in_background:true`, NEVER `&`; clean
  strays (`docker ps --filter ancestor=longcat-avatar-dev:builder` → `docker kill`).
- C++ builds on box via the docker builder; Rust does NOT. Build: `docker run --rm --gpus all -v $PWD:/src
  -v longcat-avatar-iter-ccache:/root/.ccache -w /src longcat-avatar-dev:builder bash -lc "cmake --build build
  -j\$(nproc) --target sd-cli"`.
- Pre-warm `VACE_GRAY_CACHE_DIR` before any A/B (first run pays a one-time gray-latent compute — it has faked
  "wins"). Eye-test page: `gen_eyetest_clips.sh` → http://10.0.0.208:8098/.
- A/B every change (wall + peak VRAM + frame/seam eyeball). Bank findings into `HANDOFF-VACE-TUNING.md` + memory
  continuously. Models in `models/`; full config in HANDOFF-CONTINUATIONS-3X3.md.

## START: Phase A1 — research whether there's an OFF-THE-SHELF, documented efficient-attention path for Wan2.2 at
## 720p (ranked table: speedup / quality / port effort / worth-chasing), THEN A2 measure the continuation
## throughput knee at FR=13/17/21. Report the table + numbers BEFORE committing to any fork-class attention port —
## we want to know it's worth chasing first.
