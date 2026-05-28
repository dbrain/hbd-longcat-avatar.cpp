# HANDOFF — Native `longcat-avatar-server` (GO-LIVE-PLAN §3.4.a)

**Status:** the Python supervisor (`tools/avatar_server.py`) has been
extended to cover the GO-LIVE-PLAN surface and is what koblem currently talks
to. The native server replacement is **NOT in this commit** — it's the next
quality-of-life lever (eliminate the per-render model-load tax). Everything
below is what it would take to land.

## What's already in `avatar_server.py` (so the native server doesn't have to
re-invent it):

- `/health` returns `{status, busy, draining, loaded, in_flight,
  gpu_mem_used_mib, renders_done, last_render_sec, uptime_sec}` — already the
  shape `kob-gpu-gate` expects.
- `POST /v1/admin/{drain,load,unload}` — admin trio matching the sibling
  services (qwen3-tts.cpp / siglip2.cpp / parakeet.cpp).
- `POST /generate` supports the full knob set (`prompt`, `steps`,
  `segment_frames` default **53**, `cont_cond_frames`, `segments`,
  `duration_sec`, `resolution`, `quant`, `cfg_scale`, `seed`,
  `audio_mouth_scale`, `audio_lowpass_hz`, `offload` default **true**,
  `vae_tiling`) plus the **opt-in `bsa` block** and a free-form **`debug_env`**
  override (LONGCAT_* / GGML_* / CUDA_* only).
- GPU-clear-when-idle is solved by construction: spawn-per-render → idle
  VRAM 149 MiB on this host (verified 2026-05-29 with a 25f/4-step render —
  peak 8.7 GB, idle return 149 MiB on completion).
- Force-unload (`POST /v1/admin/unload {"force": true}`) SIGKILLs the
  in-flight `sd-cli`; idempotent on idle.

## What the native server is for

The Python supervisor pays the **model-load tax every render** (~3–5 s on
this host for the 9 GB GGUF set). With a warm-weights native server that's
free across consecutive renders, then dropped only on `/v1/admin/unload`.

## Implementation outline

`examples/server/` already ships a working sd.cpp HTTP framework
(`main.cpp` + `runtime.cpp` + `routes_*.cpp`, httplib + AsyncJobManager). It
even handles `/sdcpp/v1/vid_gen` end-to-end via `SDGenerationParams::from_json_str`
→ `to_sd_vid_gen_params_t`. **The gap is three things, in order:**

### 1. Extend `SDGenerationParams::from_json_str` (examples/common/common.cpp:1627)

The avatar-specific fields exist on the struct but are NOT loaded from JSON:

- `audio_path` (string) — driving wav
- `audio_mouth_scale` (float, default 1.0)
- `audio_lowpass` (float, default 0)
- `segments` (int, default 1)
- `cont_cond_frames` (int, default 13)

Add `load_if_exists(...)` for each (~10 lines).

### 2. Write `examples/server/routes_longcat.cpp`

Mirror `routes_sdcpp.cpp::vid_gen` but with two additions:

- Accept JSON-with-base64 (current koblem path) AND multipart
  (`request` + `image` + `audio` parts) — see §10.5 of GO-LIVE-PLAN. Mirror
  `koblem/api/src/routes/avatar.rs::parse_avatar_multipart` (which is the
  inverse-direction parser).
- Materialise the image + audio to disk, set them on the
  `SDGenerationParams`, set the BSA env block before kicking off the job,
  then `submit_vid_gen` exactly like `routes_sdcpp.cpp:472` does.

Wire in `examples/server/CMakeLists.txt` next to `routes_sdcpp.cpp`.

### 3. Rename the binary target

`add_executable(sd-server …)` → `add_executable(longcat-avatar-server …)`
so the Docker `COPY --from=builder` is unambiguous.

### 4. Worker-subprocess isolation (matches the sibling pattern)

The shape to copy is **qwen3-tts.cpp/src/server.cpp** — read it verbatim:

- L552 `fork()` on first generate request when
  `LONGCAT_AVATAR_WORKER_ISOLATION=1`. Parent stays CUDA-free; child opens
  the CUDA context + loads weights.
- L856–880 — request marshalling to child over a Unix-domain socket.
  Request blob = image bytes + audio bytes + knobs JSON.
- L1152–1158 — `POST /v1/admin/unload` handler SIGKILLs the child. Kernel
  exit tears down the primary CUDA context → ALL VRAM reclaimed (no phantom
  cuBLAS workspace / cubin residue).
- Idempotent on idle (matches the Python supervisor's
  `{status: "idle"}` shape).

### 5. Dockerfile + compose

Switch `kobbler/docker/longcat-avatar/Dockerfile`:

```dockerfile
# Stage 2 runtime: drop python3, COPY --from=builder /out/longcat-avatar-server
ENTRYPOINT ["/usr/local/bin/longcat-avatar-server", \
    "--host", "0.0.0.0", "--port", "8080", \
    "-m", "/models/longcat-avatar-1.5-dit-dmd-q4_k.gguf", \
    "--t5xxl", "/models/longcat-umt5-xxl-q8_0.gguf", \
    "--vae", "/models/longcat-wan-vae-f16.gguf", \
    "--audio-vae", "/models/longcat-whisper-v3-encoder-f16.gguf", \
    "--diffusion-fa", "--clip-on-cpu", "--offload-to-cpu", "--vae-tiling"]
```

Add `LONGCAT_AVATAR_WORKER_ISOLATION=1` to the compose `environment` block.

Delete `tools/avatar_server.py` once parity is verified end-to-end.

## What koblem and the gate already expect

- `koblem/api/src/avatar.rs` POSTs `/generate` with a JSON body containing
  base64 `image`/`audio` + the full knob set. The route already sets
  `return: "bytes"` so the supervisor streams the WebM back inline. The
  native server's `/generate` should respond with the same content-type
  semantics: `Accept: video/webm` → body bytes, no `Accept` header → JSON
  envelope (or a `?return=file` query). Headers `X-Avatar-*` (segments,
  render_sec, offload) are how koblem reads metadata — keep them.
- `kob-gpu-gate::HeavyService { unload_path: "/v1/admin/unload",
  drain_path: "/v1/admin/drain" }` is the registered shape — both endpoints
  must remain.
- `/v1/admin/load` is optional pre-warm (the Python supervisor returns
  `{status: "ok", note: "supervisor pre-loads nothing"}`).

## Risks

- httplib + multipart parsing is doable but cookie-cutter — see
  `parakeet.cpp/examples/server/server.cpp` for the closest reference.
- The fork-IPC layer is the highest-risk piece; the qwen3-tts.cpp shape
  works, port it directly.
- DiT graph-cut caching across renders — `runtime.cpp` already keeps
  `sd_ctx` resident; the cached `wan_vae build cached graph cut plan` line
  in current logs (`docker logs kobbler-longcat-avatar-1`) shows the graph
  is already cached when sd-cli stays warm. Warm-weights wins ~3–5 s per
  render.

## Don't get clever

The DiT perf campaign (HANDOFF-DiT-*.md, PERF.md) is the authoritative
source for any kernel-level change. This work is **wiring only** — do not
touch the DiT.
