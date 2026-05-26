#!/usr/bin/env python3
"""
LongCat-Video-Avatar production HTTP server (GPU-polite supervisor).

This is the PRODUCTION wrapper that gives the avatar port the same API +
"fully cleared from GPU when not in use" experience as the sibling services
(qwen3-tts.cpp / siglip2.cpp / parakeet.cpp).

DESIGN — subprocess-per-render is the GPU-clear pattern.
  The avatar `sd-cli vid_gen` binary loads its weights (~9 GB DiT + VAE +
  whisper + umT5), renders the clip, frees everything and EXITS. So between
  renders there is NO process holding GPU memory — VRAM returns to ~0 (just the
  ~9 MiB idle floor) by construction. This is exactly the sibling "kill the
  worker → reclaim ALL GPU instantly" isolation pattern, except the worker is
  the whole `sd-cli` process and it self-terminates after each job. The owner's
  resident LLM gets the full 12 GB back the moment a render finishes.

  We serialize renders with a global lock (ONE heavy GPU job at a time, per the
  brief). A render in flight holds the GPU; idle holds nothing.

ENDPOINTS (JSON in, video out):
  GET  /health           -> {"status":"ok", busy, gpu_mem_used_mib, ...}
  POST /generate         -> renders a clip; returns the video bytes (or a path).
                            Single-clip i2v (segments=1) AND chained long-form
                            (segments>1 or duration_sec) both via this endpoint.
  POST /unload           -> explicit idle-clear. No-op when idle (nothing is
                            resident); kills an in-flight render if force=true.
  GET  /                 -> tiny HTML status / usage page.

KNOBS (per-request JSON, [default] in brackets):
  image            (required) ref portrait — path | data-URI | base64 | http(s) url
  audio            (required) driving wav  — path | data-URI | base64 | http(s) url
  prompt           ["a person talking"]
  steps            [8]        DMD steps (8 is the distilled count; <8 = faster/softer)
  audio_mouth_scale[1.0]      <1 milder mouth, >1 more exaggerated
  audio_lowpass_hz [0]        0=off; ~3000 softens viseme transients
  segment_frames   [49]       frames per segment (also accepts video_frames)
  duration_sec     [null]     target seconds @25fps -> auto segments (overrides segment count)
  cont_cond_frames [13]       overlap frames between chained segments
  segments         [1]        explicit segment count (ignored if duration_sec set)
  resolution       ["480x832"] WxH
  quant            ["q4_k"]    q4_k | q4_0 (model file picked from MODELS_DIR)
  cfg_scale        [1.0]
  seed             [42]        <0 = random
  offload          ["auto"]    auto | true | false  (auto: offload for >~40f or chains)
  vae_tiling       [true]
  return           ["file"]    "file" -> {video_path,...} | "bytes" -> raw video body

Run (inside the CUDA builder image — it carries libcuda):
  python3 tools/avatar_server.py --port 8080 --models /models --bin build/bin/sd-cli

Zero pip dependencies — pure Python 3 stdlib.
"""
import argparse
import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Config (CLI-overridable)
# ─────────────────────────────────────────────────────────────────────────────
CFG = {
    "bin": "build/bin/sd-cli",
    "models": "models",
    "outdir": "/tmp/avatar-out",
    "host": "0.0.0.0",
    "port": 8080,
    "max_vram": 9,          # --max-vram budget (GiB) for the offload graph-cut path
    "diffusion_fa": True,   # --diffusion-fa (flash attn) always on for the avatar
    "clip_on_cpu": True,    # CPU umT5 = the BEST-matching bit-identical text-encode path
    # model filenames per quant rung (relative to --models)
    "model_files": {
        "q4_k": "longcat-avatar-1.5-dit-dmd-q4_k.gguf",
        "q4_0": "longcat-avatar-1.5-dit-dmd-q4_0.gguf",
    },
    "t5xxl": "longcat-umt5-xxl-q8_0.gguf",
    "vae": "longcat-wan-vae-f16.gguf",
    "audio_vae": "longcat-whisper-v3-encoder-f16.gguf",
}

# ONE heavy GPU job at a time. Held for the entire render.
RENDER_LOCK = threading.Lock()
STATE = {
    "busy": False,
    "current_pid": None,
    "current_proc": None,
    "last_render_sec": None,
    "renders_done": 0,
    "started_at": time.time(),
}
STATE_LOCK = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def gpu_mem_used_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def _materialize(spec: str, dest: Path) -> Path:
    """Turn an image/audio spec (path | data-URI | base64 | http url) into a file."""
    if spec is None:
        raise ValueError("missing input")
    # existing path on disk (short, no scheme)
    if len(spec) < 4096 and not spec.startswith(("http://", "https://", "data:")):
        try:
            if Path(spec).exists():
                return Path(spec)
        except (OSError, ValueError):
            pass
    # http(s) url
    if spec.startswith("http://") or spec.startswith("https://"):
        with urllib.request.urlopen(spec, timeout=30) as r:
            dest.write_bytes(r.read())
        return dest
    # data URI
    if spec.startswith("data:"):
        spec = spec.split(",", 1)[1]
    # base64 blob
    dest.write_bytes(base64.b64decode(spec))
    return dest


def _frames_for_duration(duration_sec, segment_frames, cont_cond_frames):
    """clip_len = seg + (N-1)*(seg - cond_decoded); cond_decoded ~= cont_cond_frames.
    Solve for the smallest N covering duration_sec @ 25fps."""
    target = max(1, round(duration_sec * 25))
    new_per_seg = max(1, segment_frames - cont_cond_frames)
    if target <= segment_frames:
        return 1
    n = 1 + (target - segment_frames + new_per_seg - 1) // new_per_seg
    return max(1, n)


def build_argv(req: dict, image_path: Path, audio_path: Path, out_path: Path):
    """Map the JSON request -> sd-cli argv. All knobs honored; sane defaults."""
    quant = str(req.get("quant", "q4_k")).lower()
    if quant not in CFG["model_files"]:
        raise ValueError(f"unknown quant {quant!r}, must be one of {list(CFG['model_files'])}")
    models = Path(CFG["models"])
    model_path = models / CFG["model_files"][quant]

    res = str(req.get("resolution", "480x832")).lower().replace(" ", "")
    m = re.fullmatch(r"(\d+)x(\d+)", res)
    if not m:
        raise ValueError(f"bad resolution {res!r}, expected WxH e.g. 480x832")
    width, height = int(m.group(1)), int(m.group(2))

    steps = int(req.get("steps", 8))
    cont_cond_frames = int(req.get("cont_cond_frames", 13))
    segment_frames = int(req.get("segment_frames", req.get("video_frames", 49)))

    if req.get("duration_sec"):
        segments = _frames_for_duration(float(req["duration_sec"]), segment_frames, cont_cond_frames)
    else:
        segments = int(req.get("segments", 1))
    segments = max(1, segments)

    seed = int(req.get("seed", 42))
    cfg_scale = float(req.get("cfg_scale", 1.0))
    mouth = float(req.get("audio_mouth_scale", 1.0))
    lowpass = float(req.get("audio_lowpass_hz", req.get("audio_lowpass", 0)))
    prompt = str(req.get("prompt", "a person talking"))

    # offload decision: auto = offload for long single segments (>40f resident
    # ceiling) or any chain (chains keep DiT+VAE+whisper resident, peak ~11.8G —
    # the brief's min-VRAM story is offload). true/false force it.
    off = req.get("offload", "auto")
    if isinstance(off, bool):
        offload = off
    elif str(off).lower() in ("true", "1", "yes", "on"):
        offload = True
    elif str(off).lower() in ("false", "0", "no", "off"):
        offload = False
    else:  # auto
        offload = (segments > 1) or (segment_frames > 40)

    vae_tiling = req.get("vae_tiling", True)
    if isinstance(vae_tiling, str):
        vae_tiling = vae_tiling.lower() in ("true", "1", "yes", "on")

    argv = [
        CFG["bin"],
        "-M", "vid_gen",
        "-m", str(model_path),
        "--t5xxl", str(models / CFG["t5xxl"]),
        "--vae", str(models / CFG["vae"]),
        "--audio-vae", str(models / CFG["audio_vae"]),
        "--init-img", str(image_path),
        "--audio", str(audio_path),
        "-p", prompt,
        "--cfg-scale", str(cfg_scale),
        "--video-frames", str(segment_frames),
        "-W", str(width),
        "-H", str(height),
        "--steps", str(steps),
        "--seed", str(seed),
        "--max-vram", str(CFG["max_vram"]),
        "-o", str(out_path),
    ]
    if segments > 1:
        argv += ["--segments", str(segments), "--cont-cond-frames", str(cont_cond_frames)]
    if abs(mouth - 1.0) > 1e-9:
        argv += ["--audio-mouth-scale", str(mouth)]
    if lowpass > 0:
        argv += ["--audio-lowpass", str(lowpass)]
    if CFG["diffusion_fa"]:
        argv += ["--diffusion-fa"]
    if CFG["clip_on_cpu"]:
        argv += ["--clip-on-cpu"]
    if offload:
        argv += ["--offload-to-cpu"]
    if vae_tiling:
        argv += ["--vae-tiling"]

    meta = {
        "quant": quant, "resolution": f"{width}x{height}", "steps": steps,
        "segment_frames": segment_frames, "segments": segments,
        "cont_cond_frames": cont_cond_frames, "seed": seed,
        "cfg_scale": cfg_scale, "audio_mouth_scale": mouth,
        "audio_lowpass_hz": lowpass, "offload": offload, "vae_tiling": vae_tiling,
        "prompt": prompt,
    }
    return argv, meta


def run_render(req: dict):
    """Acquire the GPU lock, run sd-cli, return (video_path, meta). Raises on failure."""
    if not RENDER_LOCK.acquire(blocking=False):
        raise RuntimeError("busy: a render is already in flight (one GPU job at a time)")
    job_dir = Path(tempfile.mkdtemp(prefix="avatar-", dir=CFG["outdir"]))
    try:
        img = _materialize(req.get("image"), job_dir / "ref.png")
        aud = _materialize(req.get("audio"), job_dir / "speech.wav")
        out_path = job_dir / "out.webm"
        argv, meta = build_argv(req, img, aud, out_path)

        with STATE_LOCK:
            STATE["busy"] = True
        t0 = time.time()
        print(f">> render start: {' '.join(argv)}", flush=True)
        # cwd = repo root so build/bin/sd-cli + relative model paths resolve.
        proc = subprocess.Popen(
            argv, cwd=CFG["repo"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        with STATE_LOCK:
            STATE["current_pid"] = proc.pid
            STATE["current_proc"] = proc
        log_tail = []
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_tail.append(line.rstrip("\n"))
            if len(log_tail) > 80:
                log_tail.pop(0)
        rc = proc.wait()
        dt = time.time() - t0
        with STATE_LOCK:
            STATE["busy"] = False
            STATE["current_pid"] = None
            STATE["current_proc"] = None
            STATE["last_render_sec"] = round(dt, 1)
            STATE["renders_done"] += 1
        if rc != 0:
            raise RuntimeError(f"sd-cli exited {rc}\n" + "\n".join(log_tail[-30:]))
        if not out_path.exists():
            raise RuntimeError("render produced no output file\n" + "\n".join(log_tail[-30:]))
        meta["render_sec"] = round(dt, 1)
        meta["video_path"] = str(out_path)
        meta["video_bytes"] = out_path.stat().st_size
        print(f">> render done in {dt:.1f}s -> {out_path} ({meta['video_bytes']} bytes)", flush=True)
        return out_path, meta, job_dir
    except Exception:
        with STATE_LOCK:
            STATE["busy"] = False
            STATE["current_pid"] = None
            STATE["current_proc"] = None
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    finally:
        RENDER_LOCK.release()


# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/health"):
            with STATE_LOCK:
                s = dict(STATE)
            s.pop("current_proc", None)
            self._json(200, {
                "status": "ok",
                "busy": s["busy"],
                "gpu_mem_used_mib": gpu_mem_used_mib(),
                "renders_done": s["renders_done"],
                "last_render_sec": s["last_render_sec"],
                "uptime_sec": round(time.time() - s["started_at"], 1),
            })
            return
        if self.path == "/" or self.path.startswith("/index"):
            page = (
                "<!doctype html><meta charset=utf-8><title>LongCat-Avatar server</title>"
                "<body style='background:#111;color:#ddd;font-family:system-ui;margin:32px;max-width:760px'>"
                "<h2>LongCat-Video-Avatar server</h2>"
                "<p>GPU-polite talking-avatar generator. GPU returns to ~0 between renders "
                "(subprocess-per-render isolation). One render at a time.</p>"
                "<pre style='background:#000;padding:14px;border-radius:8px;white-space:pre-wrap'>"
                "GET  /health           status + live GPU MiB\n"
                "POST /generate         {image, audio, prompt, steps, ...} -> video\n"
                "POST /unload           explicit idle-clear (no-op when idle)\n\n"
                "curl -s localhost:8080/generate -H 'Content-Type: application/json' \\\n"
                "  -d '{\"image\":\"&lt;b64&gt;\",\"audio\":\"&lt;b64&gt;\",\"steps\":8,\"return\":\"file\"}'\n"
                "</pre></body>"
            )
            self._bytes(200, page.encode(), "text/html")
            return
        self._json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            req = json.loads(raw or b"{}")
        except Exception as e:
            self._json(400, {"error": f"bad json: {e}"})
            return

        if self.path.startswith("/unload"):
            force = bool(req.get("force", False))
            with STATE_LOCK:
                proc = STATE.get("current_proc")
                busy = STATE["busy"]
            if busy and force and proc is not None:
                proc.send_signal(signal.SIGKILL)
                self._json(200, {"status": "killed in-flight render; GPU will clear"})
                return
            # idle: nothing resident — already cleared by construction.
            self._json(200, {
                "status": "idle" if not busy else "busy (pass force=true to kill)",
                "gpu_mem_used_mib": gpu_mem_used_mib(),
            })
            return

        if self.path.startswith("/generate"):
            if not req.get("image") or not req.get("audio"):
                self._json(400, {"error": "image and audio are required"})
                return
            ret = str(req.get("return", "file")).lower()
            try:
                out_path, meta, job_dir = run_render(req)
            except RuntimeError as e:
                msg = str(e)
                code = 429 if msg.startswith("busy") else 500
                self._json(code, {"error": msg})
                return
            except Exception as e:
                self._json(400, {"error": str(e)})
                return
            if ret == "bytes":
                body = out_path.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/webm")
                self.send_header("Content-Length", str(len(body)))
                for k, v in meta.items():
                    if isinstance(v, (int, float, str, bool)):
                        self.send_header(f"X-Avatar-{k}", str(v))
                self.end_headers()
                self.wfile.write(body)
                shutil.rmtree(job_dir, ignore_errors=True)
            else:
                self._json(200, {"status": "ok", **meta})
            return

        self._json(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser(description="LongCat-Avatar GPU-polite HTTP server")
    ap.add_argument("--host", default=CFG["host"])
    ap.add_argument("--port", type=int, default=CFG["port"])
    ap.add_argument("--bin", default=CFG["bin"], help="path to sd-cli")
    ap.add_argument("--models", default=CFG["models"], help="dir with the GGUF set")
    ap.add_argument("--repo", default=os.getcwd(), help="repo root (cwd for sd-cli)")
    ap.add_argument("--outdir", default=CFG["outdir"])
    ap.add_argument("--max-vram", type=int, default=CFG["max_vram"])
    args = ap.parse_args()

    CFG["bin"] = args.bin
    CFG["models"] = args.models
    CFG["repo"] = args.repo
    CFG["outdir"] = args.outdir
    CFG["max_vram"] = args.max_vram
    Path(CFG["outdir"]).mkdir(parents=True, exist_ok=True)

    print(f">> LongCat-Avatar server on http://{args.host}:{args.port}", flush=True)
    print(f">> bin={CFG['bin']} models={CFG['models']} repo={CFG['repo']}", flush=True)
    print(">> GPU-clear: subprocess-per-render — idle VRAM ~0, model reloads on next request", flush=True)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
