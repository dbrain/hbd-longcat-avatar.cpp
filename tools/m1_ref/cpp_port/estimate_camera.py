#!/usr/bin/env python3
# Auto-camera front-end for the pixal3d CLI (A3) — host MoGe-2 cut-line.
#
# proj-mode conditioning back-projects image features onto the voxel grid using the CAMERA
# (FOV camera_angle_x + distance). A wrong FOV lands features on the wrong voxels -> distorted
# geometry. Pixal3D estimates the camera PER IMAGE with MoGe-2 (Ruicheng/moge-2-vitl, a monocular
# geometry model): predict intrinsics -> camera_angle_x = 2·atan(W/(2·fx)), then distance via
# distance_from_fov (pure host math). For "upload anything -> 3D" we need this; a fixed --fov only
# works for assets shot like the training data.
#
# This keeps the C++ on the ggml path (same host cut-line as rembg in preprocess_photo.py).
# Usage:
#   /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python estimate_camera.py matte.png
#     -> prints "--cam <angle_rad> <distance> <mesh_scale>"  (paste into the pixal3d call)
#   ... estimate_camera.py matte.png --json        -> {"camera_angle_x":..,"distance":..,"mesh_scale":..}
#   ... estimate_camera.py matte.png --run --model weights_gguf_f16 --out m.glb --tex --fast
#     -> estimate the camera AND exec pixal3d with the right --cam (one-shot upload->GLB).
#
# The math mirrors inference.py get_camera_params_wild_moge / distance_from_fov / compute_f_pixels
# EXACTLY (validated against the Python reference on the same image).
import os, sys, math, json, argparse
import numpy as np
import torch
from PIL import Image

MOGE_MODEL_NAME = "Ruicheng/moge-2-vitl"
IMAGE_RESOLUTION = 512   # MoGe distance math works in the model's 512 render space (inference.py default)


def compute_f_pixels(camera_angle_x: float, resolution: int) -> float:
    focal_length = 16.0 / math.tan(camera_angle_x / 2.0)
    return float(focal_length * resolution / 32.0)


def distance_from_fov(camera_angle_x, grid_point, target_point, mesh_scale, image_resolution):
    # grid_point @ R^T with R = [[1,0,0],[0,0,-1],[0,1,0]]
    gx, gy, gz = grid_point
    xw = gx
    yw = -gz
    zw = gy
    xw, yw, zw = xw / mesh_scale / 2, yw / mesh_scale / 2, zw / mesh_scale / 2
    xt, yt = float(target_point[0]), float(target_point[1])
    f_pixels = compute_f_pixels(camera_angle_x, image_resolution)
    x_ndc = xt - image_resolution / 2.0
    distance_x = f_pixels * xw / x_ndc - yw
    return float(distance_x)


def estimate(image_path, mesh_scale=1.0, extend_pixel=0, image_resolution=IMAGE_RESOLUTION, device="cuda"):
    from moge.model.v2 import MoGeModel
    pil = Image.open(image_path).convert("RGB")
    width, height = pil.size
    arr = np.asarray(pil).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).to(device)
    model = MoGeModel.from_pretrained(MOGE_MODEL_NAME).to(device).eval()
    with torch.no_grad():
        out = model.infer(t)
    intr = out["intrinsics"].squeeze().cpu().numpy()
    fx = float(intr[0, 0]) * width
    camera_angle_x = 2.0 * math.atan(width / (2.0 * fx))
    distance = distance_from_fov(
        camera_angle_x, [-1.0, 0.0, 0.0],
        [0 - extend_pixel, image_resolution - 1 + extend_pixel],
        mesh_scale, image_resolution)
    del model
    torch.cuda.empty_cache()
    return {"camera_angle_x": camera_angle_x, "distance": distance, "mesh_scale": mesh_scale}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Estimate pixal3d camera (FOV+distance) from an image via MoGe-2.")
    ap.add_argument("image")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of the --cam arg string")
    ap.add_argument("--mesh-scale", type=float, default=1.0)
    ap.add_argument("--run", action="store_true", help="exec ./pixal3d with the estimated --cam")
    ap.add_argument("--pixal3d", default="./pixal3d")
    args, rest = ap.parse_known_args()
    cp = estimate(args.image, mesh_scale=args.mesh_scale)
    if args.json:
        print(json.dumps(cp))
    else:
        sys.stderr.write(f"[cam] MoGe-2: fov={math.degrees(cp['camera_angle_x']):.2f}deg "
                         f"({cp['camera_angle_x']:.4f} rad) distance={cp['distance']:.4f} scale={cp['mesh_scale']:.3f}\n")
        print(f"--cam {cp['camera_angle_x']:.6f} {cp['distance']:.6f} {cp['mesh_scale']:.6f}")
    if args.run:
        cam = ["--cam", f"{cp['camera_angle_x']:.6f}", f"{cp['distance']:.6f}", f"{cp['mesh_scale']:.6f}"]
        cmd = [args.pixal3d, "--image", args.image] + cam + rest
        sys.stderr.write("[cam] exec: " + " ".join(cmd) + "\n")
        os.execvp(cmd[0], cmd)
