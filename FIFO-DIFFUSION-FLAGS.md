# FIFO-Diffusion / unified-refine — branch reference

Status: **SHELVED (2026-07-21).** This branch explores FIFO-Diffusion sampling and a
"unified refine" chain mode for LTX-2.3 video. It is **not merged to master and not
deployed.** The decision was to prefer generating the base at high resolution
(`native1080`) with **no refine** for long varying-scene clips, because the refine
consistency problem this branch chased is not solved by the sampler topology (see
"Why shelved" below). Everything here is behind value-gated env flags — with all flags
unset the build is behaviorally identical to `master`.

All flags are value-gated: they engage only when set to exactly `1` (unless noted).
Unset / any other value = off = prod behavior.

## FIFO diagonal sampler (base and refine `SD::sample()`)
| Flag | Effect |
|------|--------|
| `LTXAV_FIFO=1` | Engage the FIFO diagonal sampler (per-frame sigma queue) in both base and refine passes. |
| `LTXAV_FIFO_BLOCKS=<n>` | Latent-partitioning block count (default 1). Reduces the intra-window noise-level gap. |
| `LTXAV_FIFO_QUEUE=<n>` | Queue depth override (default derived from step count). |
| `LTXAV_FIFO_WINDOW=<n>` | Bounded sliding-window length in latent frames. `0` = full-timeline (unbounded VRAM). |
| `LTXAV_FIFO_CONTEXT=<n>` | Backward clean-context anchor frames kept in the window (default = steps; `-1` = auto). Added in 90e83ca to fix per-frame seed re-roll. |
| `LTXAV_FIFO_LOOKAHEAD=<n>` | Lookahead-denoising frames (use cleaner future frames to help earlier ones). |
| `LTXAV_FIFO_AUDIO=freeze` | Audio handling in the diagonal (freeze vs lockstep). |
| `LTXAV_FIFO_STATS=1` | Verbose per-window latent stats logging (debug only). |

## Unified-refine chain mode
| Flag | Effect |
|------|--------|
| `LTXAV_CHAIN_UNIFIED_REFINE=1` | Chain mode: generate all N segment BASES → assemble one latent timeline (drop K overlap) → ONE refine over the whole timeline. Off = per-segment base+refine (prod). |
| `LTXAV_REFINE_FROM_LATENT=<path>` | Skip base gen + assembly; load a banked assembled base (`<save_dir>/unified_assembled_base.bin`) and run only the refine. The "generate base now, refine later" path. |
| `LTXAV_FIFO_REFINE=1` | Escape hatch: force the FIFO diagonal for the unified REFINE pass. Default the unified path DISABLES FIFO for its whole run (base+refine) — see below. |

## Continuation-guide placement (pre-existing, used by the chain)
| Flag | Effect |
|------|--------|
| `LTXAV_CONT_LEGACY_HEAD=1` | Place the continuation guide at the segment head (mask 0, frozen) instead of the default keyframe-append (past-RoPE warm-up + stitcher auto-trim). |
| `LTXAV_CONT_OVERLAP_MASK=<0..1>` | Overlap denoise mask: `0` = frozen (carry motion, may stall); `>0` = let the overlap re-denoise (motion evolves). |

## Why shelved (do not re-chase without reading this)
- **FIFO diagonal ERASES evolving content.** At staggered per-frame sigmas, a later frame's only *clean* attention anchors are earlier (already-refined) frames, so an SDEdit refine denoises later content back toward the timeline's start. Proven by a green-donkey A/B: a donkey entering in segment 2 is erased with FIFO on, persists with FIFO off. Full-timeline window and lower σ0 (0.6) both still erased — it's the diagonal itself, not the window or renoise strength. Hence `25a5690` makes the unified path disable `LTXAV_FIFO` for its whole run (opt back in with `LTXAV_FIFO_REFINE=1`).
- **FIFO is a GENERATION sampler with a coherence/low-dynamics bias.** It suits *extending one continuous shot*, not refining a timeline whose content changes.
- **The real blocker (refine reroll) is a DETERMINISM problem, not a topology one.** Every upscale/refine reinvents high-freq detail differently per window/loop (worst in low-attention regions). MultiDiffusion/temporal-windows preserve content but reroll; FIFO is consistent but suppresses/smears. Neither wins both. The untested lever is **shared/deterministic refine noise** (`LTXAV_SHARED_REFINE_NOISE` / the `2x_sharednoise` harness variant) + uniform-sigma tiling — try that FIRST if refine consistency is ever revisited.

## Commit map (this branch, off master `46c5892`)
- `0932e7b` FIFO diagonal PoC (per-frame sigma).
- `e5e248a` fix garbage output — VP-normalized flow-euler step + audio lockstep.
- `1d88774` VRAM-bounded sliding window.
- `90e83ca` backward clean-context anchor (fixes per-frame seed re-roll).
- `8039863` unified-refine chain mode (all-base → assemble → one refine).
- `25a5690` carry real audio through the join + disable FIFO for the whole unified path.
- `67c9991` preserve packed audio across the continuation guide-frame strip (audio-shear bugfix; no-op for the normal chain, which strips audio channels on continuation).
- `8a24c4f` free the pre-trim packed latent before the audio repack (alloc hygiene).
