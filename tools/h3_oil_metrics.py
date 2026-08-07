#!/usr/bin/env python3
"""Pixel-domain diagnostics for temporal oil/warp defects in H3 clips.

These metrics are diagnostics, not a substitute for watching native consecutive
frames.  They intentionally operate on decoded RGB pixels and do not inspect
codec motion vectors or bitstream block metadata.
"""

import argparse
import json
import math
import subprocess

import numpy as np


def probe(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames",
        "-of", "json", path,
    ])
    stream = json.loads(out)["streams"][0]
    return int(stream["width"]), int(stream["height"]), int(stream.get("nb_frames") or 0)


def decode(path, width):
    source_w, source_h, _ = probe(path)
    height = max(2, int(round(source_h * width / source_w / 2)) * 2)
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", path, "-an",
        "-vf", f"scale={width}:{height}:flags=area", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ])
    frame_bytes = width * height * 3
    if not raw or len(raw) % frame_bytes:
        raise SystemExit(f"{path}: incomplete rawvideo decode ({len(raw)} bytes)")
    rgb = np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3).astype(np.float32) / 255.0
    return rgb


def gradients(x):
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, 1:-1] = 0.5 * (x[:, 2:] - x[:, :-2])
    gy[1:-1, :] = 0.5 * (x[2:, :] - x[:-2, :])
    return gx, gy, np.hypot(gx, gy)


def laplacian(x):
    out = np.zeros_like(x)
    out[1:-1, 1:-1] = (
        x[1:-1, :-2] + x[1:-1, 2:] + x[:-2, 1:-1] + x[2:, 1:-1]
        - 4 * x[1:-1, 1:-1]
    )
    return out


def corr(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b / denom) if denom > 1e-20 else 0.0


def local_motion(prev, cur, patch=12, radius=3):
    """Block-match luma and return previous frame warped into current coordinates."""
    h, w = cur.shape
    warped = prev.copy()
    vectors = []
    for y in range(radius, h - patch - radius + 1, patch):
        for x in range(radius, w - patch - radius + 1, patch):
            target = cur[y:y + patch, x:x + patch]
            best = (float("inf"), 0, 0)
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    source = prev[y + dy:y + dy + patch, x + dx:x + dx + patch]
                    error = float(np.mean(np.abs(target - source)))
                    if error < best[0]:
                        best = (error, dy, dx)
            _, dy, dx = best
            warped[y:y + patch, x:x + patch] = prev[
                y + dy:y + dy + patch, x + dx:x + dx + patch
            ]
            vectors.append((y, x, dy, dx))
    return warped, vectors


def warp_with_vectors(prev, vectors, patch=12):
    warped = prev.copy()
    for y, x, dy, dx in vectors:
        warped[y:y + patch, x:x + patch] = prev[
            y + dy:y + dy + patch, x + dx:x + dx + patch
        ]
    return warped


def line_coherence(y):
    values = []
    long_fraction = []
    for frame in y:
        gx, gy, mag = gradients(frame)
        threshold = np.percentile(mag, 75)
        mask = mag >= threshold
        # Structure-tensor coherence over spatial tiles: long, stable edges are
        # anisotropic; isotropic melted texture approaches zero.
        tile_values = []
        for yy in range(0, frame.shape[0] - 15, 16):
            for xx in range(0, frame.shape[1] - 15, 16):
                sx = gx[yy:yy + 16, xx:xx + 16]
                sy = gy[yy:yy + 16, xx:xx + 16]
                a = float(np.sum(sx * sx))
                b = float(np.sum(sy * sy))
                c = float(np.sum(sx * sy))
                tile_values.append(math.sqrt((a - b) ** 2 + 4 * c * c) / (a + b + 1e-12))
        values.append(float(np.mean(tile_values)))

        # Fraction of strong edges supported by similarly oriented neighbours.
        angle = np.arctan2(gy, gx)
        support = np.zeros_like(frame, dtype=np.int16)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1)):
            shifted_mask = np.roll(mask, (dy, dx), axis=(0, 1))
            delta = np.abs(np.angle(np.exp(1j * (angle - np.roll(angle, (dy, dx), axis=(0, 1))))))
            support += shifted_mask & (delta < 0.22)
        long_fraction.append(float(np.mean(support[mask] >= 3)) if np.any(mask) else 0.0)
    return values, long_fraction


def grid_energy(y, period):
    fixed, maximum = [], []
    for frame in y:
        gx, gy, _ = gradients(frame)
        phase_energy = []
        for phase in range(period):
            vertical = np.mean(np.abs(gx[:, phase::period]))
            horizontal = np.mean(np.abs(gy[phase::period, :]))
            phase_energy.append(0.5 * (vertical + horizontal))
        median = float(np.median(phase_energy)) + 1e-12
        fixed.append(float(phase_energy[0] / median))
        maximum.append(float(max(phase_energy) / median))
    return {"fixed_phase_ratio": float(np.mean(fixed)), "max_phase_ratio": float(np.mean(maximum))}


def summarize(path, analysis_width):
    rgb = decode(path, analysis_width)
    y = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    cb = (rgb[..., 2] - y) / 1.8556
    cr = (rgb[..., 0] - y) / 1.5748
    chroma = np.stack((cb, cr), axis=-1)

    detail = {"laplacian_rms": [], "gradient_p90": []}
    for frame in y:
        _, _, mag = gradients(frame)
        detail["laplacian_rms"].append(float(np.sqrt(np.mean(laplacian(frame) ** 2))))
        detail["gradient_p90"].append(float(np.percentile(mag, 90)))

    multiscale = {}
    for scale in (1, 2, 4):
        ys = y[:, ::scale, ::scale]
        energy = [float(np.sqrt(np.mean(laplacian(f) ** 2))) for f in ys]
        multiscale[str(scale)] = {
            "mean": float(np.mean(energy)),
            "temporal_cv": float(np.std(energy) / (np.mean(energy) + 1e-12)),
            "last_first_ratio": float(np.mean(energy[-max(1, len(energy)//4):])
                                      / (np.mean(energy[:max(1, len(energy)//4)]) + 1e-12)),
        }

    warp_luma, warp_edge, chroma_error, vector_roughness, chroma_error_maps = [], [], [], [], []
    for i in range(1, len(y)):
        prev_warp, vectors = local_motion(y[i - 1], y[i])
        prev_rgb_warp = warp_with_vectors(chroma[i - 1], vectors)
        _, _, cur_edge = gradients(y[i])
        _, _, prev_edge = gradients(prev_warp)
        moving_edges = cur_edge >= np.percentile(cur_edge, 70)
        base = float(np.mean(np.abs(y[i] - y[i - 1]))) + 1e-12
        warp_luma.append(float(np.mean(np.abs(y[i] - prev_warp)) / base))
        warp_edge.append(float(np.mean(np.abs(cur_edge[moving_edges] - prev_edge[moving_edges]))
                               / (np.mean(cur_edge[moving_edges]) + 1e-12)))
        cerror = chroma[i] - prev_rgb_warp
        chroma_error.append(float(np.sqrt(np.mean(cerror ** 2))))
        chroma_error_maps.append(cerror)
        vec = np.asarray([(dy, dx) for _, _, dy, dx in vectors], np.float32)
        vector_roughness.append(float(np.mean(np.std(vec, axis=0)) / (np.mean(np.linalg.norm(vec, axis=1)) + 1.0)))

    chroma_persistence = [
        corr(chroma_error_maps[i - 1], chroma_error_maps[i])
        for i in range(1, len(chroma_error_maps))
    ]
    coherence, long_edges = line_coherence(y)

    return {
        "path": path,
        "frames": int(len(rgb)),
        "analysis_resolution": [int(rgb.shape[2]), int(rgb.shape[1])],
        "temporal_warp": {
            "motion_compensated_luma_ratio_mean": float(np.mean(warp_luma)),
            "moving_edge_residual_mean": float(np.mean(warp_edge)),
            "motion_vector_roughness_mean": float(np.mean(vector_roughness)),
        },
        "detail_retention": {
            "laplacian_rms_mean": float(np.mean(detail["laplacian_rms"])),
            "laplacian_temporal_cv": float(np.std(detail["laplacian_rms"])
                                           / (np.mean(detail["laplacian_rms"]) + 1e-12)),
            "gradient_p90_mean": float(np.mean(detail["gradient_p90"])),
            "multiscale": multiscale,
        },
        "temporal_chroma": {
            "motion_compensated_error_mean": float(np.mean(chroma_error)),
            "error_persistence_lag1": float(np.mean(chroma_persistence)),
        },
        "edge_line_coherence": {
            "structure_tensor_mean": float(np.mean(coherence)),
            "supported_strong_edge_fraction": float(np.mean(long_edges)),
            "temporal_cv": float(np.std(coherence) / (np.mean(coherence) + 1e-12)),
        },
        "pixel_grid_energy": {
            "period_8": grid_energy(y, 8),
            "period_16": grid_energy(y, 16),
            "period_32": grid_energy(y, 32),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("clips", nargs="+")
    parser.add_argument("--analysis-width", type=int, default=216)
    parser.add_argument("--json-out")
    args = parser.parse_args()
    results = [summarize(path, args.analysis_width) for path in args.clips]
    text = json.dumps(results, indent=2)
    if args.json_out:
        with open(args.json_out, "w") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
