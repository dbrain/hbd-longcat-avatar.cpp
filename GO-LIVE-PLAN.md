# LongCat-Avatar — Go-Live Plan

*Written 2026-05-29 from a cold read of the current tree + the kobbler/koblem
ecosystem. Scope = wiring an already-feature-complete avatar engine into the
koblem stack as a peer to `music`. **Not** a perf doc — that's `PERF.md` /
`HANDOFF.md` / `HANDOFF-DiT-*.md`, which stay authoritative for the kernel
work and **must not be disturbed**. This doc is "what's left between today's
`docker compose --profile longcat-avatar up` working-but-orphan service and a
button labelled **Avatar** in koblem-web that the kid can press."*

---

## 1. What's already live (no work needed)

- **The engine.** `sd-cli vid_gen` produces a coherent, lip-synced, audio-driven
  talking-avatar clip on the 3060. See `HANDOFF.md` §"What this is" — image +
  audio → video, single clip or chained, user-approved quality.
- **The stopgap supervisor.** `tools/avatar_server.py` is a thin Python HTTP
  wrapper that spawns `sd-cli` per render and exits → **GPU-clear-when-idle
  is solved by construction**, at the cost of paying the ~3–5 s model-load
  every render. Endpoints: `GET /health`, `POST /generate`, `POST /unload`,
  `GET /`. Idle VRAM ~9 MiB; busy peak ≈ 5.7 GiB (offload) or 10.8 GiB
  (resident). Sequential (one render at a time; concurrent `/generate` → 429).
  **This is the wrong long-term shape** — every sibling C++/ggml service
  (`qwen3-tts.cpp`, `siglip2.cpp`, `parakeet.cpp`) ships a **native C++
  server with worker-subprocess isolation** instead, and we should match
  them. See §3.4 for the migration; `avatar_server.py` retires once the
  native server is at parity.
- **Dev iteration.** `kobbler/docker/longcat-avatar-dev/iter.sh` (builder image,
  ccache, `iter.sh serve` → :8095). Perf work continues here, untouched.
- **Prod Dockerfile + compose entry.** `kobbler/docker/longcat-avatar/Dockerfile`
  builds a runtime image from the local `longcat-avatar.cpp` checkout (the fork
  isn't on a registry — same `additional_contexts` pattern as siglip2).
  `kobbler/docker-compose.yml` has the `longcat-avatar` service behind the
  `longcat-avatar` profile, port 8809→8080, models mounted from
  `${LONGCAT_AVATAR_MODELS_DIR:-…/longcat-avatar.cpp/models}`.
- **Models.** Q4_K DiT (8.94 GB) + umT5-XXL Q8 + Wan-VAE F16 + whisper-v3
  encoder F16, all on the main SSD under `models/`.

So the engine, the supervisor, the image, and the compose entry are done. What
ships live needs **(a)** GPU coordination with the LLM/TTS stack,
**(b)** koblem-side orchestration (route + GPU lock + storage), **(c)** a koblem-web
UI mirroring `Music`, **(d)** per-request knob exposure (incl. opt-in BSA),
and **(e)** the LLM going from "always resident" to "drainable on demand."

---

## 2. The "old 53f sweet spot" numbers (for the defaults)

The brief asked to dig these out — they're in `PERF.md` lap 22 §"Resident
frame ceiling PINNED" and live above the lap 24/lever-3 perf work:

| segment N | new / seg | resident s-step | offload s-step | VAE | **resident s / new (VRAM)** | **offload s / new (VRAM)** |
|---|---|---|---|---|---|---|
| 37f | 24 | 28.5 | 30.9 | 64.2 s | 13.6 (~11.3 GB) | 14.7 (~5.7 GB) |
| 45f | 32 | 37.4 | — | 78.6 s | 12.0 | 12.7 |
| **53f** | **40** | **47.1** | **50.0** | **91.8 s** | **11.7 (10.8 GB)** | **12.3 (5.3 GB)** |
| 93f | 80 | OOM | 117 | — | — | 13.5 (~6.2 GB) |

- **53f is the sweet spot.** Bigger segments (61f+) OOM resident / balloon
  the offload tax (+23 % at 93f, plus N²-ish self-attn). Smaller segments
  pay a "redo tax" — every continuation reprocesses `cont_cond_frames=13`
  frames, so smaller N spends a bigger fraction on the overlap.
- **Ship offload-only.** Resident mode is a perf-bench knob, not a prod
  config — the juggling (avatar takes the whole card → no light peers
  beside it) isn't worth saving ~5 % wall on a workflow already
  measured in minutes. The resident column above is left in for context
  with PERF.md; **the API defaults to and the UI assumes
  `offload=true`**. (The DiT campaign is closing the ~20 s offload tax
  — 160 s today vs 140 s resident — independently.)
- **Numbers move as the DiT campaign lands wins.** As of lap 31.2 the dense
  DiT step is ~142.5 s for 25f / 8-step (BSA-bitmap opt-in: 139.88 s). Per
  s-step values above will shrink proportionally — the *ranking* of 37/45/**53**/93
  is stable.

**Defaults the plan derives from this:** `segment_frames=53`, `cont_cond_frames=13`,
`steps=8`, **`offload=true`** (no `auto`; resident is not a prod option),
DiT GGUF = `q4_k`. Avatar peak ≈ 5.3 GB → coexists with the LLM-or-Flux
heavy slot's footprint **only** when no other heavy is resident (see §10.1)
— a render still drains the prior heavy.

---

## 3. The GPU-coordination story

### 3.1 Today's gate

`koblibs/kob-gpu-gate` owns four consumer slots with priorities **TTS > Music >
Vision > STT**, plus a tagged `set_tts_services()` registry for multiple TTS
engines. Music is "shared read-lock + background mutex + evict every peer on
acquire" — the same pattern Avatar needs.

The gate today **does not know about the LLM**. The LLM is always-on,
permanently resident, and treated as "the immovable 6.5 GB occupant
everything else has to fit around" (see compose, `llama-server` has no
`/unload` and no idle-unload). siglip2/tts-qwen3/parakeet-cpp coexist with it
via worker isolation + small footprints.

### 3.2 What live needs

**Two NEW service categories:**

- **`HeavyGpu`** — Avatar **or** LLM, mutually exclusive. Either is fine
  resident alongside the light peers; both at once won't fit (LLM 6.5 GB +
  avatar peak 5.3–10.8 GB ≥ 12 GB). Avatar > LLM in priority but **doesn't
  hard-kill**: it asks llama-server to drain in-flight, then unloads.
- **LLM-as-gate-participant.** The LLM stops being implicit-always-on. It
  gains an `acquire_for_llm()` slot, a `drain → unload` path, and lazy-load
  (cold start) on next request.

**Updated priority (final):**

```
TTS (interactive)
  > Avatar (heavy, takes the GPU solo; minutes-long)
  > Music (heavy-ish, 6–18s renders, current behaviour)
  > LLM (heavy, but the background occupant by default)
  > Vision
  > STT
```

TTS preempting avatar matches the existing music-preempt contract (drop the
render, return 503). The workflow doesn't generate interactive TTS during an
avatar render — TTS runs *before* (script → audio) and then the avatar
consumes that audio, so the preempt window is small in practice. **If user
ergonomics later prove this wrong**, the trivial alternative is to demote
avatar above TTS — but only if the kid is generating clips while
something else is reading aloud, which doesn't fit the usage.

### 3.3 The avatar-acquires-LLM-drains dance

When Avatar acquires:

1. Set a `llm_draining` flag and stop forwarding new koblem→llama requests:
   queue them or 503 them with `Retry-After`. (koblem's `llama.rs` chat path is
   the one place that needs to consult this.)
2. Wait *up to* a short ceiling (e.g. 20 s, configurable) for in-flight chat
   completions to finish naturally. The streaming chat token rate is high
   enough that any single in-flight request finishes well inside that window
   for normal max_tokens. If the ceiling expires, signal SIGTERM (let llama
   close its sockets), then SIGKILL.
3. Issue `/v1/admin/unload` to the llama supervisor (see §3.4): this is the
   "shut down llama-server and free VRAM" call.
4. Now do the existing music-style eviction: unload TTS / STT / Vision.
5. Acquire the avatar background slot and serve the render.
6. On guard-drop: clear `llm_draining` (chat resumes; lazy-loaded on next
   request).

When koblem-api gets a chat request while Avatar holds the slot:
- Return 503 with `Retry-After` proportional to the avatar's typical wall
  (e.g. `last_render_sec` from `/health`, clamped to a few minutes). The user
  is not browsing chat while the kid is rendering — this is the right
  failure mode.

### 3.4 Native worker-isolation: avatar AND llama-server

The right shape for both is the **qwen3-tts.cpp pattern**, not a Python
supervisor on top. Three siblings already ship this:

| service | binary | isolation trigger | unload endpoint |
|---|---|---|---|
| qwen3-tts.cpp | `qwen3-tts-server` (`src/server.cpp`) | env `QWEN3_TTS_WORKER_ISOLATION=1`, `fork()` @ L552 | `POST /v1/admin/unload` @ L1158 (SIGKILL the worker) |
| siglip2.cpp | `siglip2_server` (`src/siglip2_server.cpp` + `worker_ipc.cpp` + `worker_session.cpp`) | env `SIGLIP2_WORKER_ISOLATION=1` | `POST /v1/admin/unload` |
| parakeet.cpp | `parakeet-server` (`examples/server/server.cpp`) | env `PARAKEET_WORKER_ISOLATION=1` | `POST /unload` |

Pattern: parent process binds the HTTP port and stays CUDA-free; on first
GPU-touching request the parent `fork()`s a child that opens the CUDA
context + loads weights; subsequent requests proxy through to the child
via a Unix-domain or pipe IPC; `/v1/admin/unload` SIGKILLs the child →
**kernel exit tears down the primary CUDA context → ALL VRAM reclaimed**,
no phantom `cuBLAS` workspace / cubin cache residue. Cold reload on next
request, ~1–5 s depending on model size.

That's what longcat-avatar.cpp and the koblem-llama fork need.

#### 3.4.a — Native `longcat-avatar-server` (retires `avatar_server.py`)

longcat-avatar.cpp already has a sd.cpp-inherited HTTP framework at
`examples/server/` (`main.cpp`, `runtime.cpp`, `routes_*.cpp`,
`async_jobs.cpp`). The gap vs the sibling pattern is three things:

1. **No `vid_gen` route.** sd-server today handles `txt2img`/`img2img`; add
   a `routes_longcat.cpp` with `POST /generate` that dispatches the
   avatar pipeline (image + audio + knobs → WebM bytes) — mirror every
   knob `avatar_server.py::build_argv` translates today, plus the BSA
   block from §4.2. Mostly a wiring exercise: `runtime.cpp` already owns
   the model-load lifecycle and the sd_ctx; add a video-output path.
2. **No worker-subprocess isolation.** Add a `LONGCAT_AVATAR_WORKER_ISOLATION=1`
   gate that forks on first generate request, parent proxies, child holds
   sd_ctx + CUDA. Take the qwen3-tts.cpp shape verbatim — `fork()` in
   `main.cpp`, an IPC channel for the request blob (image bytes + audio
   bytes + knobs JSON), the child streams back WebM bytes. Owner approval
   is the in-place template; don't re-derive.
3. **No admin endpoints.** Add `POST /v1/admin/unload` (SIGKILL the worker
   → `last_token_at` clear, `loaded=false`), `POST /v1/admin/drain` (stop
   accepting new generate requests, finish in-flight), `POST /v1/admin/load`
   (optional pre-warm). `GET /health` already exists in the inherited
   framework — augment its body with `{loaded, busy, last_render_sec,
   in_flight}` to match the kob-gpu-gate health-probe shape.

Once the native server is at functional parity with `avatar_server.py`
(image + audio in, WebM out, all knobs honored), retire the Python
supervisor: delete `tools/avatar_server.py`, drop the python3 install from
`docker/longcat-avatar/Dockerfile`, switch `ENTRYPOINT` to the new binary.

**Wins vs the Python supervisor:**
- Warm child = no per-render model-reload cost (the Python wrapper paid
  ~3–5 s every render to re-load 9 GB; native keeps weights resident across
  renders, drops VRAM only on `/v1/admin/unload`).
- Single language. The owner's bench/eyeball scripts already speak C++ /
  HTTP; no shell-out-to-python in the render path.
- Same idle-VRAM contract (unload SIGKILLs the child → 0 MiB; matches
  siglip2 / qwen3-tts measured floors).

#### 3.4.b — In-fork `LLAMA_WORKER_ISOLATION` on koblem-llama

llama-server today has neither worker isolation nor an unload endpoint —
it holds 6.5 GB until process exit. Same fix, in-fork: extend the
koblem-llama (`hbd-llama-cpp-turboquant`) server to support
`LLAMA_WORKER_ISOLATION=1` + `POST /v1/admin/{unload,drain,load}`.

llama.cpp's server is bigger than qwen3-tts's so this is the largest
chunk of work in this section, but the structure is identical: the
parent stays CUDA-free, holds the HTTP port and the in-flight counter;
on first chat request `fork()`s a child that loads the GGUF + spec-mtp
draft + mmproj; admin endpoints SIGKILL the child.

The two non-trivial llama-server-specific bits:
- **Streaming responses across the IPC boundary.** Chat completions
  stream (`text/event-stream`) — the parent has to relay the child's
  SSE frames out the inbound HTTP socket without re-encoding. A
  dup'd-fd or splice approach (parent forwards the child's write-end
  fd) is the clean solution; alternative is a per-request named pipe.
- **`/v1/admin/drain` semantics under streams.** "Finish in-flight"
  means the currently-streaming request runs to its `[DONE]` event;
  draining only blocks *new* requests. Existing keep-alive
  connections that haven't sent a request yet are fine to keep open.

Cold reload: ~5 s warm-page-cache, < 2 s if `/models` already paged.
First chat request after an avatar render eats this — acceptable.

#### 3.4.c — Gate registration

With both services exposing the unified shape, register both on the
gate:

```rust
gate.set_service_urls(
    Some(stt_url),
    None,
    Some(vision_url),
);
gate.set_tts_services(vec![ TtsService { tag: "qwen3", … } ]);
gate.set_avatar_service(AvatarService {
    url: "http://longcat-avatar:8080",
    unload_path: "/v1/admin/unload",
    drain_path: "/v1/admin/drain",
});
gate.set_llm_service(LlmService {
    url: "http://llama-server:8080",
    unload_path: "/v1/admin/unload",
    drain_path: "/v1/admin/drain",
});
```

…and the avatar-acquires-LLM-drains sequence in §3.3 calls
`drain_path` → wait → `unload_path` instead of the Python-supervisor
endpoints I sketched earlier.

### 3.5 Concrete gate API additions (`koblibs/kob-gpu-gate`)

Add a third registered service kind (`Llm`) and a fourth acquire slot
(`acquire_for_avatar`). Sketch (additive — no breakage to existing callers):

```rust
pub struct LlmService {
    pub url: String,           // http://llama-server:8080
    pub unload_path: String,   // /v1/admin/unload
    pub drain_path: String,    // /v1/admin/drain — pre-step
}

impl GpuLock {
    pub fn set_llm_service(&self, llm: Option<LlmService>);

    /// Acquire for an LLM chat request. Blocks if Avatar holds the heavy
    /// slot. Returns None if a higher-priority consumer is pending.
    pub async fn acquire_for_llm(&self) -> Option<GpuLlmGuard>;

    /// Acquire for an Avatar render. Drains LLM (call /v1/admin/drain →
    /// wait for in-flight → /v1/admin/unload), then evicts all light peers.
    /// Returns None if TTS preempted.
    pub async fn acquire_for_avatar(&self) -> Option<GpuAvatarGuard>;

    /// Called by avatar handler between segments / between steps.
    pub fn is_avatar_pending(&self) -> bool;
}
```

Internally:
- The existing `background` mutex is enough to serialise Avatar against
  Music/Vision/STT. Avatar takes it like Music does, then drains LLM, then
  evicts.
- `cancel` token (the TTS-preempt mechanism) extends: Avatar handler polls
  `is_tts_requested()` between segments and aborts cleanly (same as music
  yielding mid-render).
- LLM and Avatar **share a new heavy-slot mutex** (above `background` in the
  ordering) so they serialise wrt each other without holding the background
  mutex hostage. LLM acquires the heavy slot (cheap, just permission to use
  the GPU), keeps it for the chat call, releases. Avatar acquires the heavy
  slot, drains LLM, then takes background + write-locks light peers' eviction.
- Update `koblem-llama` health-check + lazy-load wiring so the gate eviction
  doesn't trip the docker healthcheck (supervisor stays up; child cycles).

Existing tests in `kob-gpu-gate/src/lib.rs` (`music_evicts_all_peers`,
`tts_preempts_music`, …) port directly to the avatar slot with new fixtures —
mirror them.

### 3.6 Compose changes

In `kobbler/docker-compose.yml`:
- **Un-profile-gate `longcat-avatar`.** Today it's `profiles: ["longcat-avatar"]`
  so it doesn't auto-up. For prod it stays opt-in via env if you want, but
  the more koblem-native pattern is always-on with idle = ~9 MiB (matches
  acestep). The gate makes it safe to run alongside the LLM because Avatar
  is *idle* when not rendering. **Recommendation:** drop the profile.
- **Repoint koblem-api → llama supervisor.** Same `LLAMA_SERVER_URL` env, but
  the URL now points at the supervisor's port. Internally the supervisor
  proxies to llama-server.
- **Bind the supervisor source.** Add an `AVATAR_SERVER_URL` (or similar)
  env on `koblem-api` pointing at `http://longcat-avatar:8080`.
- Add koblem-api volume `${AVATAR_OUT_HOST:-/mnt/hdd/avatars}:/var/lib/koblem/avatars`
  to persist rendered clips (mirror SONGS_DIR).
- llama-server `command:` block stays — the supervisor's child is started
  with these args.

---

## 4. Per-request knob exposure (incl. BSA)

### 4.1 What `avatar_server.py` already exposes

The supervisor already passes through the request-shape knobs in `build_argv`:
prompt, steps, audio_mouth_scale, audio_lowpass_hz, segment_frames,
duration_sec, cont_cond_frames, segments, resolution, quant, cfg_scale, seed,
offload, vae_tiling, return. **All of these need to ride through koblem
unchanged.** Defaults are sane (see §4.3 for one bump).

### 4.2 What's missing — BSA and the DiT-perf-campaign knobs

The BSA / Stage-1 / Stage-2 levers are **env-gated process-level knobs** today
(see `src/longcat_avatar.hpp:1715`+ and `HANDOFF-DiT-lap31.md`):
`LONGCAT_BSA`, `LONGCAT_BSA_RADIUS`, `LONGCAT_BSA_SELF_FRAME`, `LONGCAT_BSA_BOOKEND`,
`LONGCAT_BSA_CUBE_H`, `LONGCAT_BSA_CUBE_W`, `LONGCAT_BSA_BITMAP`. The handoff
docs are clear:

- Dense (BSA off) is the production-default quality bar. **Do not switch the
  default to BSA** — owner eyeballed r=1 unacceptable, r=2 not prod-acceptable;
  only `LONGCAT_BSA=1 LONGCAT_BSA_RADIUS=1 LONGCAT_BSA_SELF_FRAME=1
  LONGCAT_BSA_BITMAP=1` is *bit-exact* (`HANDOFF-DiT-lap31.md` Stage 2)
  because the bitmap reconstructs the exact dense result. That stack is
  the only opt-in worth surfacing today.

**Plan:**
- Extend the supervisor's `/generate` JSON with an optional `bsa` block:
  ```json
  "bsa": {
    "enable": true,           // default false (dense)
    "radius": 1,              // 1 = the bit-exact bitmap config
    "self_frame": true,
    "bookend": null,
    "bitmap": true            // gates the new fast bit-exact path
  }
  ```
  When present, the supervisor sets the matching `LONGCAT_BSA*` env vars
  on the spawned `sd-cli` process (per-render — these are env-gated at
  startup, but the supervisor spawns fresh per render, so they ARE
  per-request by construction). Absent → dense, matching today's default.
- Also surface the existing `LONGCAT_IM2COL_TILE` / `LONGCAT_CONT_*` /
  `LONGCAT_OP_PROFILE` debug toggles via `debug_env: {…}` for the playground
  path — they're useful for the perf campaign without rebuilding the image.
  Keep them under a `debug_env` namespace so callers know they're advisory
  and may change.

### 4.3 Defaults to bump

`avatar_server.py` currently defaults `segment_frames=49`. The PERF.md
sweet-spot is 53f. **Change the default to 53.** No backward-compat concern;
the request explicitly overrides it anyway.

(One-liner: `int(req.get("segment_frames", req.get("video_frames", 53)))`.)

### 4.4 koblem-client surface

Add to `koblibs/koblem-client/src/types.rs` (mirrors the music block at the
bottom of that file):

```rust
pub struct AvatarBsaConfig {
    pub enable: bool,
    pub radius: Option<u8>,
    pub self_frame: Option<bool>,
    pub bookend: Option<u32>,
    pub bitmap: Option<bool>,
}

pub struct AvatarGenerateRequest {
    // Inputs
    pub image_source: AvatarImageSource,   // upload | library_clip_id | url
    pub audio_source: AvatarAudioSource,   // upload | tts | url
    pub prompt: Option<String>,            // default "a person talking"

    // Geometry / chaining
    pub resolution: Option<String>,        // "480x832"
    pub duration_sec: Option<f64>,         // overrides segment count
    pub segments: Option<u32>,
    pub segment_frames: Option<u32>,       // default 53
    pub cont_cond_frames: Option<u32>,     // default 13

    // Quality / speed
    pub steps: Option<u32>,                // default 8
    pub quant: Option<String>,             // "q4_k" | "q4_0"
    pub cfg_scale: Option<f64>,
    pub seed: Option<i64>,
    pub audio_mouth_scale: Option<f64>,
    pub audio_lowpass_hz: Option<u32>,
    pub offload: Option<AvatarOffload>,    // auto | true | false
    pub vae_tiling: Option<bool>,
    pub bsa: Option<AvatarBsaConfig>,      // opt-in only
    pub save_clip: Option<bool>,           // default true
}

pub enum AvatarAudioSource {
    Upload,                                // multipart file
    Tts {                                  // koblem runs TTS first
        script: String,
        voice: Option<String>,
        engine: Option<String>,            // qwen3 | chatterbox
        // …mirror TtsQwen3GenerateRequest essentials
    },
    Url(String),
    LibraryClipId(String),                 // saved TTS output
}

pub enum AvatarImageSource {
    Upload, Url(String), LibraryAssetId(String),
}

pub struct AvatarGenerateResponse {
    pub clip: AvatarClip,                  // url, mime, bytes, duration_ms
    pub segments_rendered: u32,
    pub render_ms: u64,
    pub gate_wait_ms: u64,
    pub script_used: Option<String>,       // if LLM wrote it
    pub gpu_log_id: Option<i64>,
}
```

Same `MusicProps`-shaped `AvatarProps { healthy, default_segment_frames, …}`
for the UI dropdown.

---

## 5. koblem-side orchestration

### 5.1 Files to land

```
koblem/api/src/avatar.rs                 # client to the longcat-avatar service
koblem/api/src/routes/avatar.rs          # the HTTP routes (mirrors routes/music.rs)
koblem/web/src/lib/Avatar.svelte         # the page (mirrors Music.svelte)
koblem/web/src/lib/client.ts             # add the avatar fetchers
koblem/web/src/lib/Sidebar.svelte        # nav entry
koblem/web/src/lib/MobileMenu.svelte     # nav entry
```

The split mirrors `music.rs` 1:1 — write `avatar.rs` (HTTP client to
`http://longcat-avatar:8080`) + `routes/avatar.rs` (HTTP surface +
acquire / drive / store / release pipeline).

### 5.2 Endpoint set

```
POST /api/v1/avatar/generate            — JSON | multipart, returns clip
POST /api/v1/avatar/script-from-llm     — ungated 9B writes a script from a brief
GET  /api/v1/avatar/props               — healthy + defaults for the UI
GET  /api/v1/avatar/file/{name}         — auth-gated serve from AVATARS_DIR
GET  /api/v1/avatar/clips               — list past renders (paginated)
```

### 5.3 The pipeline (per request)

The avatar-with-TTS pipeline is the headline workflow ("upload an image,
have TTS make me audio from a script (optionally LLM generated) and pressing
go should have longcat-avatar produce the clip"):

```text
0. Parse + materialise image (multipart upload OR library asset OR URL).

1. Resolve the script.
   - If AudioSource = Tts { script, … }: use it as-is.
   - If AudioSource = Tts { script_from_brief, … }: call the main 9B
     (ungated — no GPU lock yet; same pattern as music's lyrics path).
     System prompt: "you write a short spoken script for an animated
     character avatar. <= N seconds at normal speech rate. No stage
     directions, no markdown. Tone: <user brief>."

2. Render TTS audio.
   - acquire_for_tts_as("qwen3") (or chatterbox per settings).
   - call POST /v1/audio/speech with the resolved script.
   - drop the TTS guard.
   - Persist the WAV / MP3 to a tmp path; keep duration handy.

3. Acquire Avatar.
   - state.gpu.acquire_for_avatar() (drains LLM, evicts light peers).
   - 503 + retry-after if higher priority (TTS) pending.

4. Drive avatar.
   - POST /generate to longcat-avatar with image + audio + knobs + bsa.
   - The supervisor renders sequentially; long polling here is fine (the
     reqwest client should have no read-timeout, since renders are minutes).
   - Poll `is_tts_requested()` not needed — the avatar supervisor is single-
     shot per request. Mid-render preempt would mean SIGKILLing sd-cli and
     dropping the work. Acceptable but not the path used today; the simpler
     contract is "TTS preempt = next-acquire blocked, this render finishes."
     (Music handler yields mid-render; Avatar wall is minutes, so worth
     the tradeoff — discussed in the priority table above.)

5. Save the clip.
   - Stream the WebM body out of the supervisor (or use return:"file" then
     copy from /tmp/avatar-out — bytes mode is simpler).
   - Write to ${AVATARS_DIR}/<stamp>-<seed>.webm.
   - Drop the Avatar guard.

6. Persist the log row.
   - gpu_logs entry mirroring music's row shape — service="avatar",
     endpoint="generate", duration_ms, gate_wait_ms, request_meta (knobs +
     script), response_meta (clip url, segments, bytes), status, sample_id
     (clip path).

7. Respond with AvatarGenerateResponse pointing at /api/v1/avatar/file/<name>.
```

### 5.4 Notes on the script-from-LLM step

- The brief mentions "TTS make me audio from a script (optionally LLM
  generated)" — that's exactly the music `/suggest` shape but for spoken
  script. Re-use the LLM creative-sampling DB key (`avatar_llm_sampling`)
  for tweakability without a UI knob.
- The system prompt should constrain length to "~N seconds @ normal pace"
  derived from the duration the user picked (or duration_sec=null = "any").
  Same `WORDS_PER_SEC` heuristic as music can apply (≈ 2.5 wps for speech,
  not music — keep these tuned independently).

---

## 6. The `/Avatar` koblem-web page (UI shape)

Mirror `Music.svelte`'s pattern: simple default + progressive disclosure.

**Visible by default (the kid-press-and-go flow):**
- Image picker — upload OR pick a saved character (library_asset_id).
- Script box — free-form text + an "✨ Write me a script" button that calls
  `POST /api/v1/avatar/script-from-llm { brief, duration_sec? }` (the
  "music's suggest" analog). Pre-fill, editable.
- TTS voice picker — defaults from the current TTS settings; only show the
  cross-engine dropdown under Advanced.
- Duration slider — 2 s ↔ 30 s (single-clip ≤53f maps to ~2 s, chaining handles
  longer; the page just sets `duration_sec`, the API picks segments).
- **Go** button. While rendering: progress (poll `/health` busy/last_render_sec;
  derive % from segment count × s/new-frame).

**Hidden behind an Advanced panel (the "Music"-style expert disclosure):**
- prompt (default "a person talking")
- segment_frames (default 53, the sweet spot from PERF.md)
- cont_cond_frames (default 13)
- steps (default 8 — 6 = −17 % wall, opt-in; 4 = visibly faster, opt-in)
- quant (q4_k default, q4_0 opt-in)
- cfg_scale (default 1.0)
- seed (default random; set for reproducibility)
- audio_mouth_scale (default 1.0)
- audio_lowpass_hz (default 0)
- resolution (default 480×832)
- offload (default **on** — prod assumes offload; "off" is a perf-bench knob, not a regular toggle)
- vae_tiling (default on)

**Hidden behind a "Perf knobs" sub-panel of Advanced (non-default, opt-in):**
- BSA: enable + radius (1) + self_frame (on) + bitmap (on). Tooltip:
  "Faster render, bit-exact under the bitmap config — opt-in until owner
  approves as default."
- Debug env override (free-form key=value list — for the perf playground).

This matches the user's stated preference ("expose all of the knobs but hide
the 'non default' stuff in a nice way like Music").

### 6.1 Saved-character library

Bonus, low-effort: an `avatar_assets` table (or just a bucket under
`AVATARS_DIR/_assets/`) that the picker reads. The first time someone uploads
a reference portrait, save it for re-use — the kid doesn't want to re-upload
the same image every run. Two columns: id, original_filename, path, created_at.
GET endpoint `/api/v1/avatar/assets` returns the list; DELETE for cleanup.

---

## 6.5. Per-service disable (benchmarking switches)

A standing pain point: when benchmarking something that might eat the GPU
(perf-lap clip, fresh sd-cli run, micro-bench of a competing model), it's
annoying to `docker compose stop X Y Z` and remember to bring them back up.
What's wanted is an **administrative disable** per service, surfaced through
koblem, that cleanly evicts that service's VRAM and rejects new requests
with a 503 until re-enabled.

This is independent of the priority/preempt logic in §3 — disable is an
*operator switch*, not a contention rule. It composes:

| state | what happens to a new request |
|---|---|
| disabled | 503 `{"error": "<service> is administratively disabled"}` + `Retry-After: 0` |
| enabled, preempted | 503 (existing behaviour) `Retry-After: <short>` |
| enabled, idle, gate-permitted | served |

### 6.5.1 Where the switches live

Extend `kob-gpu-gate` with per-service `enabled` flags (atomic bool per kind),
DB-persisted under a `gpu_service_states` table. The gate's existing
`set_enabled(bool)` controls the master preempt switch; this is a parallel
finer-grain knob.

```rust
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ServiceKind { Llm, Tts, Avatar, Music, Vision, Stt }

impl GpuLock {
    pub async fn set_service_enabled(&self, kind: ServiceKind, enabled: bool);
    pub fn is_service_enabled(&self, kind: ServiceKind) -> bool;
    pub fn service_states(&self) -> Vec<(ServiceKind, bool)>;  // for /api/v1/gpu/services
}
```

The acquire paths early-return `None` when disabled:

```rust
pub async fn acquire_for_avatar(&self) -> Option<GpuAvatarGuard> {
    if !self.is_service_enabled(ServiceKind::Avatar) { return None; }
    // … existing path
}
```

**Disable side-effect (the benchmarking lever):** on the `enabled→disabled`
edge, the gate fires the service's unload URL (same path as eviction) so
VRAM clears within seconds. Re-enabling does nothing immediate — the next
request lazy-loads.

For the LLM, "disable" = call the llama supervisor's `/v1/admin/drain` + wait
+ `/v1/admin/unload` (§3.4). Avatar's "disable" while a render is in flight =
SIGKILL the in-flight sd-cli (`POST /unload {force:true}` — already in
`avatar_server.py`) and clear the busy flag.

### 6.5.2 The koblem surface

```
GET    /api/v1/gpu/services            list { name, enabled, busy, vram_mib }
POST   /api/v1/gpu/services/{name}     body: { enabled: bool }
POST   /api/v1/gpu/services/disable-all body: { except?: [name, …] }
POST   /api/v1/gpu/services/enable-all
```

All under the existing auth gate. The list endpoint extends what
`routes/gpu_status.rs` already shows (state + live VRAM). The state is
**process-state in the gate AND persisted to DB** — restart-stable, so an
operator can take the LLM down for an overnight bench and the morning
reboot doesn't undo it.

### 6.5.3 Route-handler translation (the "sane rejection")

Every gate-consulting route is already wired to handle `None` from acquire —
they return 503 with the preempt message today. Differentiate the cause so
the UI can show a distinct status:

```rust
let guard = match state.gpu.acquire_for_avatar().await {
    Some(g) => g,
    None if !state.gpu.is_service_enabled(ServiceKind::Avatar) => {
        return err(StatusCode::SERVICE_UNAVAILABLE,
                   "avatar is administratively disabled");
    }
    None => return err(StatusCode::SERVICE_UNAVAILABLE,
                       "GPU busy with higher-priority TTS — retry shortly"),
};
```

Same pattern in `llama.rs`, `tts*.rs`, `music.rs`, `stt.rs`, `vision.rs`.

Background workers (kobbler's STT/Vision audiobook jobs, kobload's review
queue → koblem chat completions) all already 503-tolerant — they retry on
their NATS schedules. Disabling LLM during a bench just parks the review
queue until re-enabled; no data loss.

### 6.5.4 The UI (tiny addition)

Existing `routes/gpu_status.rs` has a status page — extend it with a row per
service:

```
[●] LLM            6532 MiB   [Disable]
[●] Avatar           9 MiB   [Disable]
[○] TTS (qwen3)      0 MiB   [Disable]   (lazy)
[●] STT              0 MiB   [Disable]   (lazy)
[●] Vision (siglip2) 0 MiB   [Disable]   (lazy)
[●] Music (acestep) 121 MiB   [Disable]

[ Disable all ]   [ Enable all ]   [ Disable all except LLM ]
```

`●` = enabled, `○` = disabled. The "disable all except LLM" shortcut is the
"I'm benching avatar/tts/etc., keep the chat working" preset; "Disable all"
is the "I'm benching the LLM itself, get out of the way" preset.

Click → POST → page re-fetches state. No reload, no docker.

### 6.5.5 Bonus: idle-unload uniform knob

Today every service has its own `IDLE_UNLOAD_SECONDS` env (tts-qwen3=300,
parakeet=300, siglip2=300, chatterbox=0, vision=0, llama=∞). Wire a single
DB setting `gpu_idle_unload_seconds` that overrides them all via the gate
(periodic background task calls each service's `/unload` after N seconds
since `gate.last_release_ms[kind]`). Useful for benchmarking-adjacent: "drop
the idle floor to 30 s during this session" without touching compose.

This is optional polish on top of the disable feature — same plumbing
(gate knows the services), separate setting key.

---

## 7. Sequencing — concrete landing order

This is the recommended push order, smallest reversible blast-radius first.
Each step is independently testable; no step blocks on the next.

| # | step | where | unblocks |
|---|---|---|---|
| 1 | Bump `segment_frames` default 49→53 in `avatar_server.py` (interim — keeps the Python path coherent until #2 lands) | longcat-avatar.cpp | matches PERF.md sweet spot, no behaviour change vs explicit override |
| 2 | Land native `longcat-avatar-server`: add `routes_longcat.cpp` w/ `POST /generate` (full knob set incl. BSA block from §4.2) in `examples/server/`. Keep `avatar_server.py` alongside until #3 hits parity (§3.4.a) | longcat-avatar.cpp | native HTTP service, weights stay warm across renders |
| 3 | Add `LONGCAT_AVATAR_WORKER_ISOLATION=1` + `POST /v1/admin/{unload,drain,load}` in the native server (mirror `qwen3-tts.cpp/src/server.cpp` fork+SIGKILL pattern); update prod Dockerfile to use the native binary; retire `tools/avatar_server.py` (§3.4.a) | longcat-avatar.cpp + kobbler/docker/longcat-avatar/Dockerfile | parity with sibling C++/ggml services |
| 4 | Drop `profiles: ["longcat-avatar"]` from compose, mount AVATARS volume | kobbler | service starts on `docker compose up`, idle = ~9 MiB |
| 5 | Add `LLAMA_WORKER_ISOLATION=1` + `POST /v1/admin/{unload,drain,load}` to the koblem-llama fork (in-fork extension of llama.cpp's server; SSE-relay across IPC; §3.4.b). Update compose env, healthcheck | koblem-llama fork + kobbler compose | LLM gains drain/unload semantics, matches sibling pattern (no Python supervisor) |
| 6 | Extend `kob-gpu-gate` with `acquire_for_llm` + `acquire_for_avatar`, `LlmService`/`AvatarService` registry, heavy-slot mutex (§3.4.c) | koblibs/kob-gpu-gate | gate vocabulary for the orchestration |
| 7 | Switch koblem's `llama.rs` to acquire LLM through the gate (chat completions wrap with `acquire_for_llm`) | koblem | LLM goes from immortal to coordinated |
| 8 | Add `AvatarGenerateRequest`, `MusicGenerateRequest`-shaped wire types | koblibs/koblem-client | shared types for the route + page |
| 9 | Land `koblem/api/src/avatar.rs` + `routes/avatar.rs` (mirror music) | koblem | API surface |
| 10 | Land `koblem/web/src/lib/Avatar.svelte` + nav entries + client.ts | koblem/web | UI |
| 11 | Per-service disable: gate flags + DB table + `/api/v1/gpu/services` endpoints + status-page rows (§6.5) | koblibs/kob-gpu-gate + koblem | benchmarking ergonomics, lands independently of #1–10 |
| 12 | Avatar assets table (optional, ship #1–10 first) | koblem migration | saved characters |
| 13 | Optional: unified `gpu_idle_unload_seconds` gate-driven idle-unload (§6.5.5) | koblibs/kob-gpu-gate | session-level idle tuning |

Steps 1–3 are independent and ship immediately. Step 4 is the only step that
touches the LLM hot path; it's worth a careful read of how koblem holds an
in-flight chat stream when the supervisor starts draining (the supervisor
should drain on the **first new request after drain-flag flip**, never mid-
stream a running response).

---

## 8. Things explicitly NOT in scope here

- **Anything in the DiT perf campaign.** `HANDOFF-DiT.md`,
  `HANDOFF-DiT-lap*.md`, `PERF.md` post-lap-22 — untouched. The 53f sweet
  spot section above is just *quoting* `PERF.md` lap 22 for default-picking.
- **VAE / cont / im2col kernel work** — also untouched.
- **Quality A/Bs** for BSA r=2 / `--steps 6` / Q4_0 — owner ear-tests gate
  any default flips. This plan keeps the safe choices everywhere.
- **The pt LLM A/B path** (`tts-qwen3-pt`-style sibling) for the LLM — out
  of scope; only relevant if the new supervisor changes prod behaviour.
- **Continuation chaining quality fixes** — already in `PERF.md` lap 15+;
  the supervisor passes `--segments`/`--cont-cond-frames` through and that's
  enough for go-live.

---

## 9. Acceptance — "we shipped" looks like

- `docker compose up -d` brings the full stack including `longcat-avatar`
  always-on, idle.
- `nvidia-smi` while idle shows: LLM resident (~6.5 GB), Avatar idle (~9 MiB),
  TTS lazy (0 MiB or 1.1 GiB if recently used), STT lazy, Vision lazy.
- Kid opens koblem-web → `/avatar` → picks her character → types or "✨"s a
  brief → picks a voice → clicks Go.
- Server-side: TTS renders the script (LLM is fine, the chat path is
  ungated for non-blocking work) → koblem-api acquires Avatar → llama-server
  drains in-flight chat → unloads → light peers evict → `sd-cli` spawns and
  renders → clip lands at `/api/v1/avatar/file/<name>` → Avatar guard drops
  → llama-server lazy-loads on the next chat request.
- Wall: at 53f, 8-step, single segment, ≈ 12 s/new-frame → a 3-second clip
  in ~30–40 s (sampling + VAE + I/O); a 10-second clip via chaining ~2 min.
  See `PERF.md` Table 2.
- An interrupting TTS request returns 503 from `/api/v1/avatar/generate` with
  `Retry-After` (and the in-flight render keeps going — TTS contract is
  "next acquire blocked").
- All knobs in §4.4 land via the API; BSA stays default-off; the page
  hides non-defaults behind Advanced.

---

## 10. Future-proofing + loose ends

Things that aren't urgent but will bite if the doc doesn't say them out loud.

### 10.1 Heavy slot is a *set*, not a pair (Flux.2 Klein etc.)

§3.2 grouped LLM + Avatar as mutually-exclusive "heavy" consumers. The
owner's next mission is wiring **Flux.2 Klein** image-gen as a third one
(another diffusion model, GPU-eating). Design `kob-gpu-gate` so the
heavy slot accepts an **N-entry registry** from day one:

```rust
pub struct HeavyService { pub kind: HeavyKind, pub url: String,
                          pub unload_path: String, pub drain_path: String }
pub enum HeavyKind { Llm, Avatar, Flux, /* future */ }

impl GpuLock {
    pub fn register_heavy(&self, svc: HeavyService);
    pub async fn acquire_heavy(&self, kind: HeavyKind) -> Option<GpuHeavyGuard>;
}
```

`acquire_heavy(kind)` takes the heavy mutex, sees who currently holds the
slot (`current_heavy: Option<HeavyKind>`), and if it's a different kind:
drain → unload → claim. **No code change** when Flux.2 Klein lands —
just `gate.register_heavy(HeavyService { kind: Flux, … })` and the priority
table extends. Same `/v1/admin/{unload,drain,load}` contract on the Flux
side (the §3.4 work doubles as the blueprint for Flux's container — when
the owner ports a fluxgen.cpp, mirror longcat-avatar-server's main.cpp).

Priority for now: **TTS > Avatar > Flux > Music > LLM > Vision > STT**.
Avatar and Flux are both "kid presses Go and waits a minute" workflows;
Avatar ranking above is arbitrary — flip via a single DB setting if Flux
turns out to be the dominant flow. (Or wire it as a "most-recent-request
wins among heavy generates" if neither is more interactive.)

**VRAM math (12 GB card, all heavies ship in their coexist-friendly mode):**

| heavy | peak | free for light peers |
|---|---|---|
| LLM (Qwen3.5-9B Q4_K) | ~6.5 GB | ~5.3 GB |
| Avatar 53f offload | ~5.3 GB | ~6.5 GB |
| Flux.2 Klein (est. Q4) | ~4–6 GB | ~6 GB |

All three are "there can be only one" but each leaves room for the light
peers (siglip2 / tts-qwen3 / parakeet) to swap in and out exactly the way
they do beside the LLM today. The gate doesn't need to think about
footprints — `acquire_heavy(kind)` always drains the prior heavy; the
light-peer eviction rules in §3.3 stay unchanged. **Avatar resident mode
is not a prod option** (§2) — keep it as a perf-bench flag only.

### 10.2 Cancel-from-UI

Kid clicks Go, regrets it 3 s in. The native avatar server already has
`POST /unload {force:true}` (inherited from `avatar_server.py`) → SIGKILL
the in-flight `sd-cli`. Expose it as `DELETE /api/v1/avatar/jobs/current`
(or `POST .../cancel`) on koblem; the UI shows a Cancel button while
`busy=true`. Same for Flux later. Music currently has no cancel —
optional to backport.

### 10.3 AVATARS_DIR retention

WebM clips are big (single 25f clip ≈ 1–3 MB; chained 10s clip ≈ 10 MB).
With no retention policy `/mnt/hdd/avatars` grows forever. Two options:
**(a)** a `gpu_logs`-style rolling retention job (keep last N or last M
days, mirror koblem's existing `gpu_logs` retention pattern), or
**(b)** mark each clip with its `gpu_log_id` and cascade-delete when the
log row rolls off. (b) is the cleaner story — re-uses an existing
sweeper. Same idea applies to SONGS_DIR; both deferrable until the kid
has actually produced enough output to notice.

### 10.4 Worker-crash contract

If the avatar/llama worker child crashes mid-render (OOM, segfault,
SIGKILL from outside), the parent observes `SIGCHLD` → marks
`busy=false`, `loaded=false`, returns 502 from any in-flight request,
clears the kob-gpu-gate heavy slot. Cold restart on next request. State
this in `examples/server/main.cpp`'s comment block so it doesn't decay.
qwen3-tts.cpp's handler is the reference — port verbatim.

### 10.5 Request shape on the native server

`avatar_server.py` accepts JSON only (image/audio as base64 or URL). The
native server should accept **both** JSON-with-base64 (for koblem's
internal calls — koblem already materialises bytes server-side) and
multipart (for direct curl debugging). Mirror koblem's music handler's
content-type branch (`/music/generate`) — JSON when no parts, multipart
otherwise. ~30 lines, saves the next debugger from base64-ing a 200 KB
PNG by hand.

### 10.6 UI render states

The avatar pipeline has visible-to-user stages:

1. `script` — LLM writing (sub-second, optional)
2. `tts` — synth (1–4 s)
3. `draining-llm` — gate evicting LLM (1–5 s) ← *Avatar-specific*
4. `loading-avatar` — first request after unload (3–5 s) ← *Avatar-specific*
5. `rendering` — sd-cli wall (segments × s/new-frame)
6. `muxing` — ffmpeg WebM + audio (<1 s)
7. `done`

State 5 is the long one (~minutes); states 3–4 are confusing if not
labelled (the kid will think it crashed). Surface them as a `phase`
field on a `/api/v1/avatar/jobs/current` GET endpoint or on an SSE
stream → the page shows "draining LLM…" / "rendering segment 2/4…" /
"saving…". Cheap polish that prevents the "is it stuck?" question.

**Early visual feedback (pairs with §10.2 cancel).** "Is it on the rails?"
is the real question — at 12 s/new-frame the kid is staring at a spinner
for ~2 minutes on a 10 s chained clip. Two preview tiers, cheap-first:

- **Per-segment preview (Tier 1, biggest win).** Chained renders decode
  each segment to a complete frame batch before moving on (per `PERF.md`
  lap-14/15). After segment 0's VAE decode finishes, the native server
  has a complete viewable WebM for that segment — emit it on the SSE
  stream as `{phase:"segment_preview", segment:0, total:4, url:"…"}` and
  the UI auto-plays it in a `<video autoplay loop muted>` while
  segments 1..N continue. **Zero added wall** (the decode was happening
  anyway), just earlier delivery. By the time segment 1 finishes its
  ~50 s sampling pass, the kid has been watching segment 0 loop for a
  minute and has already decided to let it cook or cancel. This catches
  the lap-15-class "character broken / wrong identity / watercolour
  melt" failures at the earliest possible moment.

- **First-frame thumbnail (Tier 2, single-clip case).** Single-clip
  renders (no chaining) have no Tier-1 win — there's only one segment.
  But after sampling finishes and before the full multi-frame VAE
  decode runs, you can VAE-decode **just frame 0** to a JPEG and emit
  `{phase:"frame_preview", url:"…"}`. Single-frame VAE skips the
  temporal extent that dominates the 91.8 s 53f VAE cost — ~2–3 s
  extra wall on a 60–120 s render is a worthwhile trade for an
  early kill switch. Optional, gate behind a `preview:true` request
  knob if the wall hit matters.

- **Per-step live preview (Tier 3, not on the table).** Wan-VAE has no
  TAESD-style tiny-decoder counterpart, and running the full VAE every
  step would dominate the render. Stop at Tier 1+2.

Wiring: native avatar server emits SSE on the existing `/generate`
response when `Accept: text/event-stream` (or `stream:true` in the
request body); koblem proxies the stream through. Preview URLs point at
job-scoped paths under `${AVATARS_DIR}/_previews/<job_id>/{seg0,frame0}.webm`
that the AVATARS_DIR retention sweep (§10.3) cleans up alongside the
final clip. Browser side: the UI's render-progress panel shows the
preview inline; the existing Cancel button (§10.2) kills the in-flight
sd-cli the moment the kid sees it drift.

### 10.7 Tests to land alongside the gate work

Mirror the existing music tests in `kob-gpu-gate/src/lib.rs`:

- `avatar_evicts_light_peers` — TTS + STT + Vision URLs all see POST /unload.
- `avatar_drains_then_unloads_llm` — captures `/v1/admin/drain` THEN
  `/v1/admin/unload` calls in order against a fake LLM service.
- `tts_preempts_avatar` — avatar mid-render returns its cancel-and-drop
  shape when TTS acquires.
- `flux_and_avatar_serialise` — two heavy services, second blocks until
  first drops. (Use the N-way `HeavyKind` registry — exercises §10.1.)
- `disabled_avatar_returns_none` — `set_service_enabled(Avatar, false)`
  makes `acquire_for_avatar()` return None and fires the unload URL.

These are the contract — if they pass the orchestration works.

### 10.8 720p (and other resolutions) — known-incoming knob

Today's default is 480×832. The owner has flagged 720p as a future toggle
to play with — VRAM impact under offload is unknown but expected to be
significant (activations scale with token count, and the DiT segment-graph
budget already dominates peak). Path: render at 720p with `--steps 1` to
get the activation curve, see if it fits the offload budget, then bench
walls. **No work required now** — `resolution` is already a per-request
string knob in `avatar_server.py::build_argv`, so the day someone wants to
try it the call is `{"resolution":"720x1280"}` and the only question is
whether the segment_frames default needs lowering at 720p (likely yes —
N²-ish self-attn means even 37f at 720p may exceed 53f at 480p). Worth
flagging here so when peak goes up under the new resolution it doesn't
read as a regression.

### 10.9 Where dev happens vs where deploy happens

**This box is the prod GPU host AND the dev box** — that's the point: the
agent runs against the *real* RTX 3060 with the real LLM / TTS / STT / Vision
stack so load/unload + gate orchestration is verified end-to-end against
actual VRAM, not a mock. But **the Rust workspaces are NOT built here**
(owner constraint — heavy Rust builds eat the box). The dev / deploy split
is asymmetric per component:

| component | edit + build + test on this box? | how it ships |
|---|---|---|
| longcat-avatar.cpp (native server, §3.4.a) | **yes** — full edit + build + run + test against GPU. Dockerfile sources from host checkout (`LONGCAT_AVATAR_SRC`). | local-only; `docker compose build longcat-avatar` here on each iter |
| koblem-llama fork (§3.4.b) | **yes** — via the `llama-server-dev` profile + `LLAMA_SRC=$HOME/dev/llama.cpp-turboquant`. Full end-to-end on dev port 8093. | owner triggers registry rebuild of `koblem-llama:latest` off-device when the agent flags ready |
| `kob-gpu-gate` / `koblem-client` (koblibs) | **edit + `cargo check` / `cargo test` only** (plain cargo, no Docker). Run gate tests here — they don't need the full koblem stack to verify against. | `git commit && git push` to `dbrain/koblibs` on a feature branch; koblem rebuild picks up the new SHA off-device |
| `koblem-api` / `koblem-web` (Rust + SvelteKit) | **no — do NOT `docker compose build` here.** Edit, commit, push to feature branch. | owner builds + pushes `docker.oldug.com/koblem:latest` + `koblem-web:latest` off-device, then `docker compose pull` on this box |
| `kobbler` (api + web) | **no** per the standing "never build kobbler workspace here" rule. Edit, commit, push. | owner builds + pushes off-device |

**So the agent's actual deliverable shape on this box:**

1. Land + verify §3.4.a (native `longcat-avatar-server`) end-to-end against
   the real GPU. Confirm unload-then-reload cycles return VRAM to floor.
2. Land + verify §3.4.b (`LLAMA_WORKER_ISOLATION`) on the dev iter
   container (port 8093). Confirm chat → drain → unload → next-chat
   reload, all with the actual prod model GGUF.
3. Edit koblibs (`kob-gpu-gate` slot additions, `koblem-client` types) +
   `cargo test` the new gate behaviour locally. Push to a feature branch.
4. Edit `koblem/api` + `koblem/web` for the routes + UI on feature
   branches. **DO NOT** docker-build them here. Push.
5. Edit any kobbler bits needed (probably none for this work — kobbler
   doesn't touch avatar; only compose changes there). Push.
6. Hand off to owner with: "koblibs SHA, koblem feature branch, kobbler
   compose diff, longcat-avatar.cpp commit. Build + push the registry
   images, then `docker compose pull && docker compose up -d` here."

Owner does the registry-push half off-device; this box `docker compose pull`s
the new tags and the gate + avatar wiring goes live. **No registry push
ever happens from this box.**

This makes the GPU verification real (load/unload tested against actual
9 GB + 6.5 GB + 5.3 GB peaks, not mocks) without paying the heavy-Rust-build
tax on the prod host.

### 10.10 Source delivery (local-only builds, no registry push required)

**The longcat-avatar.cpp fork is NOT on a registry.** It doesn't need to be
— the prod Docker build already sources from the local checkout via the
compose `context:` line:

```yaml
longcat-avatar:
  build:
    context: ${LONGCAT_AVATAR_SRC:-/home/dbrain/dev/longcat-avatar.cpp}
    dockerfile: /home/dbrain/dev/kobbler/docker/longcat-avatar/Dockerfile
```

So the loop is just `docker compose build longcat-avatar && docker compose
up -d longcat-avatar` — no `git push`, no private registry, no SHA pin.
Same shape siglip2 uses in dev (`SIGLIP2_SRC` `additional_contexts`).

This matters for §3.4.a: the native `longcat-avatar-server` work lands as
ordinary edits to `examples/server/` in the host checkout; `docker compose
build` picks them up. Don't introduce a registry dependency.

**For the koblem-llama fork (§3.4.b),** prod runs the *pushed*
`docker.oldug.com/koblem-llama:latest` image (rebuilt out of band when the
owner bumps it). Iteration is in `docker/llama-server-dev/` with
`LLAMA_SRC=$HOME/dev/llama.cpp-turboquant` and the `llama-dev` profile —
identical local-only pattern. So the §3.4.b WORKER_ISOLATION + admin
endpoints land first in `llama-server-dev` (verified end-to-end with kobbler
compose pointed at port 8093), then the owner triggers the registry rebuild
of `koblem-llama:latest` once it's stable. **No registry push is on the
agent's critical path** — flag for owner when ready.

**koblibs (`kob-gpu-gate`, `koblem-client`):** per the project CLAUDE.md,
plain-cargo libs that Docker builds via git. So changes here DO need
`git commit && git push` to `dbrain/koblibs` before the koblem image
rebuild can see them. Order of operations for landing the gate + types
work (§7 steps 6–8): koblibs commit/push → koblem rebuild+push → kobbler
compose pulls (pull_policy: always) → up.

**koblem + koblem-web:** pushed registry images (`docker.oldug.com/koblem`,
`docker.oldug.com/koblem-web`). Same as koblem-llama — push happens at
owner-trigger, not on the agent's critical path. Build locally, validate
via `docker compose build koblem-api koblem-web && docker compose up -d`,
flag for owner when ready to push.

**Summary of "what gets pushed":**

| component | prod source | iteration shape |
|---|---|---|
| longcat-avatar.cpp | local host checkout (`LONGCAT_AVATAR_SRC`) | edit + `docker compose build longcat-avatar` |
| koblem-llama fork | pushed image (owner-triggered rebuild) | `llama-server-dev` profile + LLAMA_SRC |
| koblibs | git push → cargo dep | commit + push, then rebuild koblem |
| koblem (api + web) | pushed image (owner-triggered rebuild) | build locally + validate, flag for owner |
| kobbler | built in-place from this checkout | normal `cargo` / `make` flow |

### 10.11 Build wiring

The native server lives in `examples/server/`; add a new
`examples/server/routes_longcat.cpp` + wire into the existing
`examples/server/CMakeLists.txt` target alongside `routes_sdcpp.cpp`.
Stable-diffusion.cpp's build already produces `sd-server`; rename or
fork the target to `longcat-avatar-server` so the Docker image's
`COPY --from=builder` picks up the right binary unambiguously. Don't
break `sd-cli` — it's still useful for one-shot CLI renders + the
perf-lap bench harness.

---

## 11. References (read these before touching anything)

- This doc — read §10 (future-proofing + loose ends) AFTER §1–9, before
  opening any file: it's where the "yes but what about Flux / cancel /
  retention / drain UX / crash contract" questions get answered without
  re-litigating.
- `HANDOFF.md` — top of the perf phase, sets the floor-vs-prove-the-floor tone.
- `HANDOFF-DiT.md` — DiT campaign overview.
- `HANDOFF-DiT-lap31.md` — current head of the perf work; BSA bitmap details.
- `PERF.md` lap 22 §"Resident frame ceiling PINNED" — the 53f table this
  plan derives defaults from.
- `kobbler/docker/longcat-avatar/{Dockerfile,README.md}` — the prod image
  and its GPU-clear-when-idle contract.
- `kobbler/docker-compose.yml` (lines ~516–565) — current avatar service entry.
- `koblibs/kob-gpu-gate/src/lib.rs` — the existing gate, the pattern Avatar
  extends. `acquire_for_music` is the closest blueprint.
- `koblem/api/src/{music.rs,routes/music.rs}` — the orchestration pattern
  Avatar mirrors 1:1.
- `koblibs/koblem-client/src/types.rs` — music types start ~line 1180; new
  avatar types go after them.
- `tools/avatar_server.py` — the *stopgap* Python supervisor; retired once
  the native server (§3.4.a) hits parity.
- **`qwen3-tts.cpp/src/server.cpp`** — the worker-isolation blueprint.
  L552 `fork()`, L856–880 `QWEN3_TTS_WORKER_ISOLATION` gate, L1152–1158
  `POST /v1/admin/unload` handler. Read this before writing 3.4.a or 3.4.b.
- `siglip2.cpp/src/{siglip2_server.cpp, worker_ipc.cpp, worker_session.cpp}`
  — second worker-isolation reference, slightly different IPC shape.
- `parakeet.cpp/examples/server/server.cpp` — third reference; useful for
  the "`/unload` returns clean even when no model is loaded" idempotency
  pattern (compose's healthcheck depends on this).
- `longcat-avatar.cpp/examples/server/` — the existing sd.cpp-inherited
  HTTP framework that §3.4.a extends (don't start from scratch).
