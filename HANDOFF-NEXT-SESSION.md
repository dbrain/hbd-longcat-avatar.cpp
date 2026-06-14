# HANDOFF — Wan2.2 music-video pipeline (next session)
_Worktree: `/home/dbrain/dev/longcat-avatar-wan22` (branch `wan22-infinitetalk`). All changes UNCOMMITTED.
Detail in `HANDOFF-wan22-PERF-VRAM-TUNING.md` (FINDINGS-A→K) + memory `project_wan22_infinitetalk_3060`._

## THE GOAL (clarified this session)
A long-form **directed "music video"**: a character does **scripted multi-scene actions** (drives a
convertible singing → gets out → walks to a bar singing/dancing → opens the door → enters → sits → sings
while eating chips) **with lip-sync**. NOT a talking head. Self-hosted, single **RTX 3060 (12GB)**, all
C++/ggml ports of the stable-diffusion.cpp fork.

## LOCKED ARCHITECTURE (decided, with evidence)
1. **i2v / VACE shot-list director = the narrative engine.** Break the script into shots; each shot = a
   text-prompted segment + init frame; **VACE chaining** carries character/world continuity across cuts.
   The *action* in a shot = whatever **Wan2.2 base renders from text** (the real ceiling).
2. **Singing lip-sync = LatentSync** (mouth-only latent-diffusion), applied ON TOP of the i2v/VACE output —
   repaints only the mouth, **preserves the directed body/scene/camera**. PyTorch sidecar (off the ggml
   path). User chose it over MuseTalk (256px, softer) and InfiniteTalk-V2V (re-animates whole body).

### What's DROPPED and why (don't re-litigate)
- **S2V** — talking-head model; (a) ignores text action (sang in place on a "walk to bar" prompt), (b) the
  port's `casual_audio_encoder` weights DON'T load ("unknown tensor", M1-stage) so it doesn't even lip-sync.
  3 strikes. (CAE non-load may be a fixable name-prefix bug but not worth it.)
- **InfiniteTalk** — talking-head only, no action control; also 480p-OOMs (DiT 10.25GB fully resident on
  CUDA0 — root-caused, fix paused). Its V2V dub re-derives whole-body motion (fights directed action).

## CURRENT FOCUS (what the user asked for next) — PROVE VACE
Before building the lip-sync layer, prove the FOUNDATION:
1. **Scripted** — does VACE render directed ACTION from text (turn, walk, reach, open door)? Probe with
   action prompts; also test VACE control inputs (it's VACE-FUN: ref-image + control-video + mask).
2. **Clean chaining** — the current 2-seg chain (`perf_out/vace/chain.mp4`) is "a bit messy" per the user.
   Improve continuity: more overlap frames (K), tune the latent backdoor (only 2 latent frames carry now
   due to /4 temporal compression), consistent seed, smaller prompt drift between segments. Eyeball seams.
3. **Performant** — VACE seg = 130–160s vs i2v 84s (the VACE-FUN expert is 9.87GB > i2v 8.15GB, can't fully
   reside at maxv6 → more streaming). Re-sweep `--max-vram` for the bigger expert; same levers as i2v.

Tooling: `run_vace_chain.sh` (2-seg continuation; models = `wan22-vace-fun-a14b-{low,high}-distill-q4_k.gguf`
in `models/`). VACE env backdoor: `VACE_SAVE_LATENT` / `VACE_CONT_FRAMES=K` / `VACE_CONT_LATENT` +
`--control-video <tail-dir>` (code at src/stable-diffusion.cpp:5690-5770).

## THEN — LatentSync (ready to go)
- Repo cloned: `/home/dbrain/dev/LatentSync`. Weights DOWNLOADED: `checkpoints/latentsync_unet.pt` (5.1GB) +
  `checkpoints/whisper/tiny.pt`. Inference: `python -m scripts.inference --unet_config_path <cfg>
  --inference_ckpt_path checkpoints/latentsync_unet.pt --video_path <in.mp4> --audio_path <wav>
  --video_out_path <out.mp4> --inference_steps 20 --guidance_scale 1.5 --enable_deepcache`.
- **VRAM:** v1.6/`stage2_512.yaml` (512px) needs **18GB — WON'T fit the 3060.** Use `stage2.yaml` or
  `stage2_efficient.yaml` (**256px, ~8GB**, fits) — but the 1.6 unet is 512-trained, so may need the
  **LatentSync-1.5** unet for 256, OR test whether the 1.6 unet runs at 256. Needs a PyTorch docker env
  (torch 2.5.1 cu121 + requirements.txt: diffusers, mediapipe, insightface, onnxruntime-gpu, face-alignment,
  DeepCache). NOT built yet.
- **Validate:** does it sync SINGING (sustained vowels) not just speech, and preserve the body? (unknown).

## PERF / ENV FACTS (proven this session)
- **DiT is at the silicon floor** (ncu --set full): Q4_K matmul + flash-attn both occupancy/latency-bound at
  theoretical-max occupancy, zero divergence, no unit saturated; mmq_x knob proven-dead. Compute-bound, NOT
  launch/copy-bound (nsys: 3µs launch). Only raw-DiT lever left = fewer tokens. Quality-at-same-speed
  (Q5/Q6/Q8 weights) is the one positive lever — PARKED.
- **maxv6 = the i2v sweet spot:** 480x832 FR=21 MoE, `--max-vram 6` → full DiT residency, wall **84s/seg,
  peak 6591 MiB (sub-7.5GB)**. (maxv7 was contention-noise; use 6.) Run-to-run variance is real (GPU shared
  w/ prod + thermal) — repeat key numbers.
- **1280x704 (LTX's res):** 287s/seg, ~2-3× slower than LTX (high-res DiT can't reside → offload-bound; VAE
  untuned ~97s). Wan's edge is LOWER res. (LTX is fully tuned; Wan only has maxv6.)
- **Profiling tooling:** ncu + nsys both in the builder (`/opt/nvidia/nsight-compute/<VER>/host/
  target-linux-x64/nsys`; convert .qdstrm via host QdstrmImporter + `apt install libdw1`). `--cap-add
  SYS_ADMIN` for counters. Scripts: profile_{decomp,nsys,ncu_full}.sh, perf_a14b.sh, run_{a14b,maxv7,s2v,
  vace_chain}.sh, sweep_*.sh. Raw output in `perf_out/`.
- **Eye-test page:** `perf_out/eyetest/` served at **http://10.0.0.208:8097/** (python http.server, may need
  restart next session: `cd perf_out/eyetest && python3 -m http.server 8097 --bind 0.0.0.0 &`).
- **Models in worktree `models/`:** wan22-i2v-a14b-{low,high}-q4_k, wan22-vace-fun-a14b-{low,high}-distill-q4_k,
  wan-s2v-14b-dit-dmd-q4_k, infinitetalk-14b-q4_k, longcat-{umt5-xxl-q8_0,wan-vae-f16}, wav2vec2-xlsr53,
  chinese-wav2vec2-base. More on `10.0.0.151:~/dev/wan22-infinitetalk/models/`.
- **Build:** `docker run --rm --gpus all -v $PWD:/src -v longcat-avatar-iter-ccache:/root/.ccache -w /src
  longcat-avatar-dev:builder bash -lc "cmake -S /src -B build -DCMAKE_BUILD_TYPE=Release -DSD_CUDA=ON
  -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=86 && cmake --build build -j\$(nproc) --target sd-cli
  sd-s2v sd-infinitetalk"`. C++ builds fine on-box; coordinate GPU with prod (worker-isolated, idle).

## OPEN QUESTIONS for next session
- VACE action-range: how much scripted action does Wan2.2 actually render from text? (the make-or-break for
  "directed" — fine interaction like "eat chips + spit crumbs" is at the edge of all current video AI.)
- Best chaining recipe for clean seams (overlap K, latent-carry, seed/prompt discipline).
- LatentSync on the 3060: which config fits + does it sync singing while preserving the body.
