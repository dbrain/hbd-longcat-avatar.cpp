# Lipdub / Relip — Speed and VRAM Handoff

## Read these first

1. [Relip failure and correctness history](LIPDUB-RELIP-FAILURE.md)
2. [SA3 native port and production quality policy](HANDOFF-SA3-NATIVE-PORT.md)
3. [Locked singing production recipe](run_singing_clip.sh)
4. [Current Relip harness](run_relip_current.sh)

Do not reset this worktree or `ggml/`: both contain user-owned, uncommitted work.

## Locked production Relip recipe

The target branch's graph-cut allocator lifetime fix is required. In
`src/core/ggml_extend.hpp`, both graph-cut `execute_graph()` calls must pass
`free_compute_buffer_immediately=true`. The old cross-segment reuse crashes
two-stage Relip before sampling.

Run the exact repro:

```bash
OUTDIR=$PWD/ltx-denoise-repro/_relip_current_locked_defaults \
FR=97 bash ltx-denoise-repro/run_relip_current.sh
```

It now uses the measured 11.5 GiB envelope by default:

- source: `/home/dbrain/dev/longcat-avatar-wan22/perf_out/prod_eyetest/lipdub_source.webm`;
- lipdub DiT: `nvfp4-CLEAN-lipdub.gguf` + IC-LoRA;
- 97 frames at 24 fps, original `run_singing_clip.sh` song muxed over output;
- forced two-stage Relip: stage 1 640×352, then 1280×704 refine;
- separable reference: spatial downscale 2×, all adjacent early latent reference
  frames retained with the final two causal frames omitted (11 of 13);
- 8 GiB graph-cut budget, 3,000 MiB hot resident set, synchronous partial
  offload, and the text/VAE residency release paths;
- the currently locked runtime stack, including SA3 first-step hybrid policy.

The checked-in harness now passes that policy explicitly. The exact full render
completed in **200 s wall / 183.84 s engine / 11,770 MiB peak SMI / 11,729.1 MB
peak cuDNN**, retaining the graph-cut safety fix. This clears the 11.5 GiB
(11,776 MiB) target and is within the 187–195 s ordinary 97-frame segment
engine timing. The mid-clip A/B against the 12-frame separable control preserved
both presenters' identities and open-mouth articulation; artifacts are
`_relip_ds2_t11_max8_ref85_cap3g_nopipe/relip_song.webm` and
`_relip_compare/t11_mid.png` (left = 12-frame control, right = locked 11).

Verified artifact: [Relip + singing eye page](/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise/relip_regression/index.html).

## Previous full-reference cost profile

The pre-policy-lock 4.041 s Relip render succeeds, but costs:

```text
wall: 256 s
peak nvidia-smi: 14,552 MiB
engine generate_video: 241.40 s
```

The log is `ltx-denoise-repro/_relip_current_locked_defaults/render.log`.
The largest measured costs in that run are:

```text
stage-1 reference encode (640×352):              13.603 s
stage-2 reference image/audio conditioning:      42.923 s
stage-2 decode:                                  29.516 s
stage-2 DiT graph-cut plan: 50 -> 48 segments
stage-2 sequence: 11,440 target + 11,440 ref = 22,880 tokens
```

The stage-2 separable reference doubles the attention context. This is why
Relip is much heavier than the ordinary singing two-stage path. Do not claim a
speed/VRAM win from an isolated kernel benchmark; measure the complete 97-frame
two-stage render and keep the source/seed/settings fixed.

## Production safety / SA3

All-step SA3 produces dotted/fuzzy LTX backgrounds on the static-crossing
control. The verified policy is:

```text
GGML_LTX_SA3=1
GGML_LTX_SA3_POLICY=first
```

This uses cuDNN only at the first base/refine denoise step, then SA3. It is the
locked default in `run_singing_clip.sh`; cuDNN-only remains
`GGML_LTX_SA3=0`. The crossing A/B is on the
[SA3 comparison page](/home/dbrain/dev/longcat-avatar-wan22/perf_out/ltx_denoise/sa3_crossing_compare.html).

For Relip, verify the server/container receives both variables explicitly.
The generic engine default is intentionally cuDNN unless SA3 is enabled.

## Server edit-model proof

On 2026-07-11 the live worker accepted `POST /ltx/v1/generate` job
`job_6a51918b_00000000` with `model:"edit"`, V2V control frames and the IC-LoRA.
Its log proved the intended model selection:

```text
img_gen: swapping DiT variant base -> edit
load /models/ltx2/nvfp4-CLEAN-lipdub.gguf
swap_diffusion_model: loaded DiT from '/models/ltx2/nvfp4-CLEAN-lipdub.gguf'
```

The worker was rebuilt and redeployed from this worktree, then completed the
two-window V2V smoke job `job_6a5196b8_00000000` on 2026-07-11. It submitted
50 consecutive control frames with `relip_control_frame_counts:[25,25]`, two
audio parts and `model:"edit"`; the result is a 50-frame WebM. Its server log
records both independent source windows and both completed renders:

```text
run_vid_chain_job: chained V2V relip uses 2 distinct source windows (50 control frames)
generate_video_chain seg 1: V2V relip source window (25 frames)
generate_video completed in 83.90s
generate_video_chain seg 2: V2V relip source window (25 frames)
generate_video completed in 77.14s
```

This verifies the allocator fix in the live worker and confirms that chained
Relip consumes a distinct V2V source window for every segment rather than
replaying segment 0. The local `build-sa3/bin/sd-server` compiles the same
implementation.

## Next investigations, in order

1. **Profile stage-2 Relip conditioning.** The 42.9 s reference conditioning
   is the largest non-DiT cost. Split it into reference VAE encode, position
   construction, and audio work before changing algorithms.
2. **Reduce stage-2 reference-token cost without changing the visual control.**
   The current reference grid is exactly full-resolution (11,440 tokens) and
   doubles the stage-2 context. Test the existing `relip_ref_downscale` /
   `relip_ref_tstride` route one variable at a time, then compare mouth sync
   and identity to the eye-page baseline.
3. **Find a safe graph-cut budget.** Stage 2 becomes 48 segments under the
   9 GiB graph budget. A Relip-only `LTXAV_REFINE_MAX_VRAM` increase may reduce
   launch/offload overhead, but only retain it if whole-render peak remains
   acceptable on the 16 GiB card.
4. **Keep the allocator fix.** Do not reintroduce per-render graph-cut compute
   allocator reuse unless it is proven safe for dynamically shaped Relip graphs.
5. **Validate the actual server path.** The CLI harness proves engine behavior,
   not server orchestration. Confirm the live `model:"edit"` request loads
   `nvfp4-CLEAN-lipdub.gguf` and does not silently use base.

## Success criteria for the next agent

- A new Relip artifact uses the exact source/song/seed and passes visual mouth
  sync and identity comparison against the current eye-page artifact.
- It reports wall time and both driver/cuDNN VRAM from the render log.
- It retains the graph-cut correctness fix and SA3 first-step policy.
- Any server change has a real server-job log proving the EDIT/lipdub model was
  selected.
