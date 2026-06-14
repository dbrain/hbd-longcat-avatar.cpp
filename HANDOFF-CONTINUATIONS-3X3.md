# HANDOFF — Make VACE continuations usable (real music video, not montage) @ 3+3
_Worktree `~/dev/longcat-avatar-wan22`, branch `wan22-infinitetalk`. Prior findings: `HANDOFF-VACE-TUNING.md`
(L1–L8h). Memory: `project_wan22_infinitetalk_3060`._

## WHAT'S DONE (committed this session)
- **Schedule fix WIRED + COMMITTED** (`src/stable-diffusion.cpp`): `WAN_DISTILL_SIGMAS=1` makes the Wan2.2
  lightx2v distill use the correct DMD sigma grid via `build_wan_distill_sigmas(n_high, n_low)` at fixed
  shift 7 (`wan_distill_shift`, env `WAN_DISTILL_SHIFT` to override). Without it, the generic DiscreteScheduler
  under-denoises (`t=[999,666,333,0]`, a wasted step on the wrong grid) → the "murk". This was THE big quality
  win (FINDINGS-L8). `ggml_extend.hpp` got the offload-profile split (commit/H2D-wait vs graphprep).
- **STEP MENU (1280×704 FR=13, two orthogonal levers):** HIGH steps = structure/object-coherence; LOW steps =
  detail/grain. `2+2`=draft (dotty + "two-halves car" lottery) · **`3+3` (+17%) = DEFAULT, fixes grain AND
  coherence** · `4+4` (+39%) = marginal max. Invoke: `WAN_DISTILL_SIGMAS=1 --high-noise-steps 3 --steps 3`.
- **Quality VALIDATED at 3+3 t2v:** crisp faces, clean grain, coherent objects, NO anime girl. A 27s clip exists
  (`perf_out/mv27/musicvideo_27s.mp4`) — but it's a **MONTAGE** (8 independent shots × 4 seeds, hard-cut), NOT a
  continuous music video. That's the gap this handoff closes.
- **Perf:** pure t2v ≈ 150–180s/seg @ 3+3 (1280×704 FR=13), ~1.4–1.7× LTX. 1280 is COMPUTE-bound, not a
  streaming wall (FINDINGS-L6/L7; in-block graph cuts would HURT — don't).
- **Operating config:** `--max-vram 7.3 --vae-tile-overlap 0.25` + `VACE_GRAY_CACHE_DIR` + `WAN_DISTILL_SIGMAS=1`
  + `--sampling-method euler --high-noise-sampling-method euler` (eta=0 deterministic — do NOT use euler_a, it
  injects noise). cfg-scale 1 (distill is CFG-free). `--diffusion-fa --offload-to-cpu --mmap`.

## THE PIVOT — make continuations usable
The montage is random hard-cuts. Goal: a REAL music video where subject/motion/identity **carries across the
seam** so each scene flows as one continuous shot (cuts only between scenes).

### Continuation mechanism (already built, validated PRE-fix)
- Script: **`run_vace_musicvideo.sh`** (4-scene chained script: seg0 = t2v, segN = continuation from prior tail).
  **Already updated to `WAN_DISTILL_SIGMAS=1 --high-noise-steps 3 --steps 3`** (verify the COMMON array + the
  per-call ENVV both carry it before trusting a run).
- Envs: `VACE_CONT_FRAMES=K` (carry K=5 tail frames) + `VACE_CONT_LATENT=<prior seg banked latent .bin>` +
  `--control-video <dir of the K tail PNGs>`. seg0 banks its tail latent via `VACE_SAVE_LATENT=<path.bin>`.
- How it works: the prior seg's last K frames become the VACE control video, and the banked tail latent is
  injected so the new seg continues the motion/identity (velocity continuity across the seam).
- Memory note (PRE-fix): "VACE-FUN velocity continuation WORKS — subject keeps moving across the seam; **mild,
  not invisible**." Seam quality at the FIXED 3+3 schedule is UNTESTED — that's the headline unknown.

### THE TASK (a perf run mixed with a "seam eye-test at 3+3")
1. **Seam eye-test at 3+3 (do first).** Render ONE scene as a 2–3 segment continuation chain at 3+3. Stitch,
   then look specifically at the **join frames** (around frame K: K-1, K, K+1): does subject/camera/motion carry
   cleanly? Any grain/coherence/brightness JUMP at the boundary (the control-encode × 3+3 interaction is new)?
   Compare seam-on vs a hard-cut of the same shots.
2. **Perf at 3+3.** Continuation segs use the heavier VACE control path (control-video encode + more tokens).
   PRE-fix that was ~327s/seg vs ~150 t2v — but the gray-cache + gray-control-encode kill (FINDINGS-L1, 32→~2s)
   apply; real continuation cost ≈ control-encode of the K REAL tail frames (~15s) + DiT(3+3) + decode. MEASURE
   the continuation seg time at 3+3 and report seg breakdown.
3. **Build the real ~27s music video** = the 4-scene shot list as **chained continuations within each scene** +
   **hard cuts between scenes**. Each scene flows; scenes cut. Stitch to one mp4. Judge intra-scene continuity +
   the cuts. (This replaces the montage as the deliverable.)
4. **Eye-test page:** add the seam/continuation clips (`gen_eyetest_clips.sh` → http://10.0.0.208:8098/).

### GOTCHAS / KNOWN
- Continuation = the SLOW path; goal here is QUALITY (does the seam hold at 3+3), not speed — don't perf-optimize
  before the seam is validated.
- `VACE_CONT_LATENT`: bank seg0 with `VACE_SAVE_LATENT`, feed to seg1; latent dims must match (W,H,T) → keep
  res/FR constant across a chain.
- 3+3 applies to BOTH seg0 (t2v) and continuation segs.
- GPU shared with prod; ONE job at a time; drive from main loop `run_in_background:true`; clean up strays
  (`docker ps --filter ancestor=longcat-avatar-dev:builder` → `docker kill`). Pre-warm `VACE_GRAY_CACHE_DIR`
  before any A/B (first run pays one-time gray compute — it has faked "wins" before).
- C++ builds on-box via the docker builder; Rust does NOT build here.

### MODELS / BUILD
- `models/`: `wan22-vace-fun-a14b-{low,high}-distill-q4_k.gguf` (9.87GB ea), `longcat-wan-vae-f16`,
  `longcat-umt5-xxl-q8_0`.
- Build: `docker run --rm --gpus all -v $PWD:/src -v longcat-avatar-iter-ccache:/root/.ccache -w /src
  longcat-avatar-dev:builder bash -lc "cmake --build build -j\$(nproc) --target sd-cli"`.

## START: run ONE scene of `run_vace_musicvideo.sh` as a 2–3 seg continuation chain at 3+3, stitch, eye-test the
## SEAM (join frames). If the seam holds → render the full 4-scene continuation music video. Report seam quality
## + continuation seg time. The schedule/step work is DONE and committed; this phase is continuity + perf only.
