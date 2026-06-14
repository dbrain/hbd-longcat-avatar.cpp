# Mini-prompt — Wan2.2/InfiniteTalk perf+quality tuning (autonomous)

You are tuning the Wan2.2-I2V-A14B + InfiniteTalk cpp port on an RTX 3060 so it can render a chained
"music video" (directed singing shots + lip-synced song). Build + pipeline are DONE; quality is clean.
**Read `HANDOFF-wan22-PERF-VRAM-TUNING.md` (same dir) first — it has setup, models, build cmd, the
measured findings, the ncu data, and the prioritized levers. Memory `project_wan22_infinitetalk_3060`
has the cross-session state.** Worktree: `/home/dbrain/dev/longcat-avatar-wan22`.

## North star
**Higher quality at the SAME or better speed — NOT raw speed.** When a kernel/stage isn't saturated,
spend the slack on quality (higher-precision weights Q5/Q6/Q8, less-lossy VAE tiling), not just speed.

## Non-negotiable rules
1. **Every "at the floor" / "can't be tuned" / "dead lever" claim MUST ship a full quantitative
   breakdown** justifying it metric-by-metric: which HW limiter is hit, why each factor (occupancy,
   regs, EACH stall reason, mem traffic, instruction mix, tile utilization) can't be reduced, what was
   tried + measured. No "at floor" by basic profile, intuition, or analogy to another project (acestep/
   flux2 are HINTS, not proof). Breaking down internal cost is cheap — do it every time.
2. **Profile deep, not basic.** `ncu --set full`/`--section ".*"` (WarpStateStats stall reasons,
   SchedulerStats, InstructionStats, MemoryWorkload, Source/SASS). ncu in docker needs `--cap-add
   SYS_ADMIN`. No nsys in builder → `apt install` it in the builder image or bucket ncu kernel durations.
3. **Hunt for EXTRA WORK** (we have found redundant work in matmul kernels before): redundant dequant,
   fp32↔fp16 converts, padding/tile waste (mmq_x=128 on small-N matmuls), unfused epilogues, transposes,
   layout copies. List every matmul launch's N/K shape.
4. **Prove every change** with A/B: wall + peak VRAM + per-stage timing + a quality eyeball (render a
   frame/clip, compare). Bit-exactness where claimed. Keep raw profiler output in FILES; only read
   summaries (context economy).

## Operating procedure (autonomous)
- Drive heavy iterative GPU work from the MAIN loop (sub-agents stall on background-job completion).
  Launch renders/profiles harness-tracked (`run_in_background:true`), never detached `&`.
- **Single GPU, shared with PROD** (gemma llama-server :8080 + ace-server :8088, worker-isolated to
  VRAM-0 when idle). Don't run two GPU jobs at once. sd-cli (`--mmap`) is RAM-gentle; sd-infinitetalk
  loads 10.7GB into ANON RAM (~17GB RSS) and clobbers prod — watch `free`/swap, keep it short or fix mmap.
- C++ cpp forks BUILD FINE here (docker builder image). Rust does NOT build here. Plain git OK.
- Bank findings continuously into `HANDOFF-wan22-PERF-VRAM-TUNING.md` + memory `project_wan22_infinitetalk_3060`.
  Don't stop early / don't ask "continue?" for routine calls — keep going until the lever is proven or
  refuted with a breakdown. Clean up strays (`pgrep`/`docker kill`) before finishing.

## Work loop (start at #1)
1. **DiT matmul deep-dive** (`mul_mat_q<Q4_K,mmq_x=128>`, occ 16.66%/221 regs, compute 54%/DRAM 39% =
   NOT saturated). Full `--set full` breakdown → stall reasons + extra-work hunt. THEN the quality play:
   re-quant the expert to **Q5_K/Q6_K/Q8_0**, re-ncu + measure wall — does higher precision fit the same
   latency budget? (Likely yes if occupancy-bound.) Document the breakdown either way.
2. Decompose per-step (matmul vs FA vs norm/rope vs copy) + ncu an actual LTX DiT run the same way to
   settle why LTX-22B looks faster (TE/VAE-dominated total, or genuinely fewer DiT tokens?).
3. Then the stacking overhead/VRAM levers in the handoff: pinned-offload (FINDINGS-07, in-tree, untested),
   free-umT5-after-encode, VAE-tiling tune (FINDINGS-05/11), residency (09/10), token/res sweep,
   InfiniteTalk VRAM (offload-reserve-free + mmap), segment-chain director.

Tooling: `perf_a14b.sh` (env-knob sweep), `run_{a14b,it,mvp}.sh`, `gen_eyetest.sh` (:8097). Results →
`perf_out/sweep.csv`. All worktree changes UNCOMMITTED (user commits).
