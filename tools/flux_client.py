"""Shared client for the flux2-server sdcpp async API.

Submits an img_gen job, samples peak VRAM (nvidia-smi) for the run, polls to
completion, and scrapes the server container logs for per-stage timings
(text-encode / DiT sampling / VAE decode) which the job JSON does not carry.
Single-job-at-a-time serialization on the server means the last stage lines in
the log window belong to the job we just ran.

Used by tools/gallery_server.py and ~/dev/bench/adapters/flux2.py.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import threading
import time
import urllib.request

DEFAULT_BASE = "http://localhost:8095"
LOG_CONTAINER = "flux2-iter"


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True).strip().splitlines()
        return int(out[0])
    except Exception:
        return None


class VramSampler(threading.Thread):
    def __init__(self, interval=0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.idle = gpu_used_mib() or 0
        self.peak = self.idle
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            v = gpu_used_mib()
            if v is not None:
                self.peak = max(self.peak, v)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=2)


def _post(base, path, body):
    r = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=30))


def _get(base, path):
    return json.load(urllib.request.urlopen(base + path, timeout=30))


def submit(body, base=DEFAULT_BASE):
    return _post(base, "/sdcpp/v1/img_gen", body)


def poll(job_id, base=DEFAULT_BASE):
    return _get(base, f"/sdcpp/v1/jobs/{job_id}")


def cancel(job_id, base=DEFAULT_BASE):
    r = urllib.request.Request(base + f"/sdcpp/v1/jobs/{job_id}/cancel", data=b"",
                               headers={"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(r, timeout=30)
        return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        return e.code, None


def _docker_logs_since(container, since_s):
    try:
        return subprocess.check_output(
            ["docker", "logs", "--since", f"{int(since_s)}s", container],
            text=True, stderr=subprocess.STDOUT)
    except Exception:
        return ""


def build_body(prompt, *, width=512, height=512, seed=42, steps=8, cfg=5.0,
               sampler="euler", negative="", batch=1, vae_tiling=False):
    body = {
        "prompt": prompt, "negative_prompt": negative,
        "width": width, "height": height, "seed": seed, "batch_count": batch,
        "sample_params": {
            "sample_method": sampler, "sample_steps": steps,
            "guidance": {"txt_cfg": cfg},
        },
    }
    if vae_tiling:
        body["vae_tiling_params"] = {"enabled": True}
    return body


def run(body, base=DEFAULT_BASE, container=LOG_CONTAINER, poll_interval=0.25, on_submit=None):
    """Submit + wait. Returns dict with status, timings, vram, image_bytes.

    on_submit(job_id) is called right after submission (lets a caller record the
    id so it can cancel the in-flight render)."""
    vram = VramSampler()
    vram.start()
    t0 = time.time()
    sub = submit(body, base)
    jid = sub["id"]
    if on_submit:
        on_submit(jid)
    status = sub.get("status", "queued")
    while status not in ("completed", "failed", "cancelled"):
        time.sleep(poll_interval)
        j = poll(jid, base)
        status = j["status"]
    wall = time.time() - t0
    vram.stop()
    j = poll(jid, base)

    log = _docker_logs_since(container, wall + 5)

    def grab(pat):
        ms = re.findall(pat, log)
        return float(ms[-1]) if ms else None
    cond_s = grab(r"get_learned_condition completed, taking ([\d.]+)s")
    dit_s = grab(r"sampling completed, taking ([\d.]+)s")
    vae_s = grab(r"decode_first_stage completed, taking ([\d.]+)s")

    img_bytes = None
    if status == "completed" and j.get("result"):
        imgs = j["result"].get("images") or []
        if imgs:
            img_bytes = base64.b64decode(imgs[0]["b64_json"])

    compute = sum(x for x in (cond_s, dit_s, vae_s) if x)
    steps = body["sample_params"]["sample_steps"]
    return {
        "id": jid, "status": status, "wall_s": round(wall, 2),
        "cond_s": cond_s, "dit_s": dit_s, "vae_s": vae_s,
        "dit_s_per_step": round(dit_s / steps, 3) if dit_s else None,
        "overhead_s": round(wall - compute, 2) if compute else None,
        "vram_idle_mib": vram.idle, "vram_peak_mib": vram.peak,
        "vram_service_mib": vram.peak - vram.idle,
        "error": j.get("error"),
        "image_bytes": img_bytes,
    }
