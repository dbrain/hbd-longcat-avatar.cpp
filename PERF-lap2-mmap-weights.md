# lap-2: mmap the offloaded weights (2026-05-29)

**Host RAM: locked 13.3 GB RSS → reclaimable. `available` 13→23 GB. Bit-identical, +0.5s warm.**

## what
- `--offload-to-cpu` keeps the 12 GB model in host RAM; mmap was simply never enabled
  (handoff hypothesis "offload disabled mmap" was WRONG — `enable_mmap` just defaults
  false and the serve cmd didn't pass `--mmap`). No lap-32 pinned-buffer conflict.
- with offload, params backend is CPU → `module_can_mmap` true → tensors mmap fine.
- fix: add `--mmap` to the serve flags (now default in iter.sh).

## mechanism
- baseline: 12.4 GB held as **RssShmem** (lap-32.1 pinned cudaMallocHost host buffer) —
  NOT reclaimable. baseline `available` = 13 GB.
- mmap: weights become file-backed (RssFile = page cache, reclaimable). DiT fully
  mmapped (`flux all params already mmap-allocated, no separate buffer`); total params
  RAM 12097→2492 MB. After touch, RssFile 10.6 GB is reclaimable → `available` 23 GB.

## numbers (resident, klein-base Q4, Q4 enc, 512², 8-step, cfg5, seed42)
| | baseline | lap-2 |
|--|--|--|
| pre-gen RSS | 13.3 GB | 3.58 GB |
| RssShmem (pinned, non-reclaimable) | 12.4 GB | 2.56 GB |
| `available` (free -g) | 13 GB | 23 GB |
| warm wall | 15.57s | 16.07s (+0.5s) |
| DiT s/step | 1.676 | 1.70 |
| VRAM peak | 6831 | 6811 |
| md5 | 6c0a783425ea | 6c0a783425ea (bit-identical) |

- one-time cold page-in (first gen after eviction): ~34s wall (pages 12 GB from disk
  @ ~350 MB/s). acceptable for heavy-bucket time-share — that's the point (reclaimable).
- residual 2.56 GB RssShmem = encoder token_embd force-F32 → fixed in lap-3.

## next
- lap-3: token_embd F32 conversion (2374 MB) breaks mmap → keep Q4_K.
