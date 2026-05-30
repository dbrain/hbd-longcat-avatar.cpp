# lap-3: keep token_embd Q4_K (mmappable) instead of F32 (2026-05-29)

**Pinned host RAM 2.56 GB → 130 MB. VRAM peak 6831→6455 (−376, encoder no longer the wall). Bit-identical.**

## what
- after lap-2, 2374 MB still sat in non-reclaimable RssShmem. Cause: Qwen3-8B-Q4_K_M
  `token_embd.weight` [4096, 151936] is **Q4_K**, but `Embedding::init_params`
  (ggml_extend.hpp) force-converts to F32 when `!support_get_rows(wtype)`.
  `support_get_rows` whitelisted only F16/Q8_0/Q5_{0,1}/Q4_{0,1} — not Q4_K.
- F32 conversion: 622M elems × 4 B = exactly 2374 MB, and (type != storage) → not mmappable.
- but CUDA `get_rows` **already supports Q4_K** (ggml-cuda/getrows.cu:221,
  `dequantize_q4_K`). The F32 workaround was stale.

## fix (one line)
- ggml_extend.hpp `support_get_rows`: add `GGML_TYPE_Q4_K` to allow_types.
- embedding now built Q4_K → mmappable. get_rows uses the same `dequantize_q4_K` as the
  old load-time F32 dequant → looked-up rows **bit-identical** (md5 unchanged).

## numbers (resident, klein-base Q4, Q4 enc, 512², 8-step, cfg5, seed42, warm)
| | baseline | lap-2 | lap-3 |
|--|--|--|--|
| params RAM (non-mmap) | 12097 MB | 2492 MB | **118 MB** (vae only) |
| mmapped | — | 9604 MB | 9938 MB |
| RssShmem (pinned) | 12.4 GB | 2.56 GB | **130 MB** |
| pre-gen RSS | 13.3 GB | 3.58 GB | **805 MB** |
| non-reclaimable (Anon+Shmem) | ~13 GB | ~2.8 GB | **~0.9 GB** |
| `available` (free -g) | 13 GB | 23 GB | **25 GB** |
| VRAM peak | 6831 | 6811 | **6455** |
| warm wall | 15.57s | 16.07s | 15.82s |
| md5 | 6c0a783425ea | = | = (bit-identical) |

## why VRAM dropped too
- F32 embedding (2374 MB) was copied to GPU during encode → encode was the global peak.
  Q4_K (350 MB) drops the encode footprint below the DiT diffusion plateau, so peak is
  now the **DiT plateau (~6455)**. Encoder is no longer the VRAM bottleneck.

## next
- VRAM wall is now the DiT plateau (UNet 5636 + ~820 buffers). perf axis: matmuls 66%.
- Q6_K embeddings (other models) still convert — CUDA get_rows lacks Q6_K. our token_embd
  is Q4_K so N/A here; add Q6_K get_rows kernel only if a Q6_K-embd model needs it.
