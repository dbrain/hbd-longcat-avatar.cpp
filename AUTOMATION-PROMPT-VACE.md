# Mini-prompt — FULL-TUNE Wan2.2-VACE performance (autonomous)

You are making the **Wan2.2-VACE-FUN-A14B** cpp port (stable-diffusion.cpp fork) as FAST as possible on a
single **RTX 3060 (12GB)**, so it can render long-form directed video. **Goal: match or beat LTX-2.3
performance — ideally beat it.** Performance ONLY this pass; continuation/quality testing is deferred (but
don't regress visual output — A/B every change with a frame eyeball).

**READ FIRST: `HANDOFF-VACE-TUNING.md`** (same dir) — it has the measured stage breakdown, the prioritized
levers, what's already proven at-floor (don't re-chase), the rules, and all env/build/tooling. Memory
`project_wan22_infinitetalk_3060` + `HANDOFF-wan22-PERF-VRAM-TUNING.md` (FINDINGS-A→K) have the full
history. Worktree `/home/dbrain/dev/longcat-avatar-wan22`, branch `wan22-infinitetalk` (UNCOMMITTED).

## The number to beat
LTX-2.3 = 1280×704, ~50 min / 27 s of video. Compare in **render-sec per sec-of-video** at a chosen
production resolution. VACE baseline ≈ 127–158 s/segment @ 480×832 (≈ LTX, lower res). Pull it well under.

## Where the time goes (attack biggest VACE-specific waste first)
1. **★ ~30 s VAE control-context encode** (VACE-only; i2v pays ~6 s). It re-encodes ref+inactive+reactive
   3–4×/segment, mostly GRAY frames. Synthesize gray-frame latents directly instead of VAE-encoding them;
   batch the encodes; tune encode tiling. Code: `src/stable-diffusion.cpp:5700–5770`. **Profile it first.**
2. **DiT max-vram re-sweep** for the 9.87 GB VACE expert (maxv6 was tuned for the smaller i2v expert →
   currently 38 graph cuts vs 5). Sweep `--max-vram` 6/7/8/9 @ FR=21; find the resident knee.
3. **VAE decode (~27 s)** tiling sweet-spot. 4. **Control-token reduction** (fewer DiT tokens). 5. Confirm
   4-step is needed. 6. Token/res for the production-res call (surface quality tradeoffs to the user).

## DO NOT re-chase (proven this session, FINDINGS-E): the DiT matmul + flash-attn kernels are AT THE SILICON
FLOOR (occupancy/latency-bound, zero divergence, mmq_x dead, NOT launch-bound). The only DiT levers are
fewer tokens or the offload→resident budget. Pinned-offload is a one-shot loss. Kernel A/B is flat.

## Operating procedure (autonomous — don't ask "continue?", keep going until each lever is proven/refuted)
- Drive heavy GPU work from the MAIN loop, harness-tracked (`run_in_background:true`), NEVER detached `&`.
- Single GPU shared with PROD (worker-isolated, idle when not in use) — never two GPU jobs at once; coordinate.
- Profile DEEP: `ncu --set full` / nsys (both in the builder; `--cap-add SYS_ADMIN`; convert nsys .qdstrm via
  host QdstrmImporter + `apt install libdw1`). Keep raw profiler output in FILES; read only summaries.
- A/B EVERY change: wall + peak VRAM + per-stage timing + a frame eyeball; bit-exactness where claimed.
  Run-to-run variance is REAL — repeat key numbers. Every "at-floor"/"dead" claim ships a metric breakdown.
- C++ builds fine on-box (docker builder image); Rust does NOT. Clean up strays (`pgrep`/`docker kill`)
  before finishing. Bank findings continuously into `HANDOFF-VACE-TUNING.md` + the memory file. Eye-test
  page at :8097 for the user (10.0.0.208).

## Start at lever #1: profile the VACE VAE control-context encode (nsys + the 3–4 encode_first_stage calls),
## prove how much is gray-frame waste, then kill it. Then the max-vram re-sweep. Report throughput vs LTX
## after each win.
