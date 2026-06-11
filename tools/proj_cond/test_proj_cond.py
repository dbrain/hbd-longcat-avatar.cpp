#!/usr/bin/env python3
"""
GPU-FREE validation: numpy proj_cond_ref  vs  the REAL Pixal3D torch ProjGrid.

The projection geometry + bilinear grid_sample are pure tensor math, so we validate
the entire net-new proj-cond op on CPU (CUDA hidden) WITHOUT the model/GPU — only the
DINOv3 feature map would need GPU, and we stand in a synthetic random map here. Mirrors
the sparse-conv spike methodology (numpy ref bit-exact vs the production op).

The torch functions below are copied VERBATIM from
  pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py
(project_points_to_image_batch, sample_features, ProjGrid) so we test against the exact
production math, with no pixal3d package import side effects.

Run:  CUDA_VISIBLE_DEVICES="" <Pixal3D>/.venv/bin/python test_proj_cond.py
"""
import os
import sys
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
import proj_cond_ref as ref


# ----------------------------------------------------------------------------- #
# VERBATIM torch reference (from image_conditioned_proj.py) — the production math
# ----------------------------------------------------------------------------- #
def project_points_to_image_batch(points_3d, transform_matrix, camera_angle_x, resolution=518):
    device = points_3d.device
    B = transform_matrix.shape[0]
    if points_3d.dim() == 2:
        points_3d_batch = points_3d.unsqueeze(0).expand(B, -1, -1)
    else:
        points_3d_batch = points_3d
    N = points_3d_batch.shape[1]
    ones = torch.ones(B, N, 1, device=device, dtype=points_3d_batch.dtype)
    points_homogeneous = torch.cat([points_3d_batch, ones], dim=-1)
    world_to_camera = torch.linalg.inv(transform_matrix.float()).to(transform_matrix.dtype)
    points_camera = torch.bmm(points_homogeneous, world_to_camera.transpose(-2, -1))[..., :3]
    x_cam = points_camera[..., 0]; y_cam = points_camera[..., 1]; z_cam = points_camera[..., 2]
    depth = -z_cam
    focal_length = 16.0 / torch.tan(camera_angle_x / 2.0)
    focal_length_pixels = focal_length * resolution / 32.0
    focal_length_pixels = focal_length_pixels.unsqueeze(1)
    x_ndc = focal_length_pixels * x_cam / (-z_cam + 1e-8)
    y_ndc = focal_length_pixels * y_cam / (-z_cam + 1e-8)
    x_pixel = x_ndc + resolution / 2.0
    y_pixel = -y_ndc + resolution / 2.0
    valid_mask = ((x_pixel >= 0) & (x_pixel < resolution) &
                  (y_pixel >= 0) & (y_pixel < resolution) & (depth > 0))
    points_2d = torch.stack([x_pixel, y_pixel], dim=-1)
    return points_2d, depth, valid_mask


def sample_features(fmap, queries_ndc):
    B, C, H, W = fmap.shape
    Bq, K, _ = queries_ndc.shape
    grid = queries_ndc.view(B, K, 1, 2)
    feat = F.grid_sample(fmap, grid, mode='bilinear', align_corners=False, padding_mode='border')
    return feat.squeeze(-1)


def torch_proj_grid_forward(fmap_bhwc, camera_angle_x, distance, mesh_scale,
                            grid_resolution, image_resolution):
    """ProjGrid.forward, B=1, BHWC input (verbatim logic)."""
    one_dim = torch.linspace(-1, 1, grid_resolution)
    x, y, z = torch.meshgrid(one_dim, one_dim, one_dim, indexing='ij')
    grid_points = torch.stack((x, y, z), dim=-1)
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    grid_points = torch.matmul(grid_points, rotation_matrix.T).reshape(-1, 3)
    front = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, -2.0],
                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    B = 1
    cam = torch.tensor([camera_angle_x]); dist = torch.tensor([distance]); ms = torch.tensor([mesh_scale])
    gp = grid_points.expand(B, -1, -1)
    gp = gp / ms.unsqueeze(-1).unsqueeze(-1) / 2
    T = front.expand(B, -1, -1).clone()
    T[:, 1, 3] = -dist
    image_points, depth, valid = project_points_to_image_batch(gp, T, cam, image_resolution)
    image_points_norm = (image_points + 0.5) / image_resolution * 2 - 1
    fmap = fmap_bhwc.permute(0, 3, 1, 2)
    out = sample_features(fmap, image_points_norm)   # [B,C,K]
    return out.permute(0, 2, 1)                      # [B,K,C]


# ----------------------------------------------------------------------------- #
def run_case(grid_res, img_res, C, cam, dist, ms, seed):
    g = torch.Generator().manual_seed(seed)
    h = w = img_res // 16  # DINOv3 patch grid (patch 16)
    fmap = torch.randn(1, h, w, C, generator=g, dtype=torch.float32)

    t_out = torch_proj_grid_forward(fmap, cam, dist, ms, grid_res, img_res)[0].numpy()
    n_out = ref.proj_grid_forward(fmap[0].numpy(), cam, dist, ms, grid_res, img_res)

    assert t_out.shape == n_out.shape, (t_out.shape, n_out.shape)
    diff = np.abs(t_out - n_out)
    denom = np.abs(t_out) + 1e-6
    maxabs = float(diff.max())
    maxrel = float((diff / denom).max())
    ok = maxabs < 1e-4
    print(f"  grid{grid_res} img{img_res} C{C} cam{cam:.3f} d{dist:.2f} s{ms:.2f}: "
          f"maxabs={maxabs:.2e} maxrel={maxrel:.2e} {'PASS' if ok else 'FAIL'}")
    return ok


def dump_golden(dump_dir):
    """Dump one case (SS grid16) as npy goldens for the C++ mirror to validate against."""
    import json
    os.makedirs(dump_dir, exist_ok=True)
    grid_res, img_res, C, cam, dist, ms, seed = (16, 512, 64, 0.8575560450553894, 2.0, 1.0, 100)
    g = torch.Generator().manual_seed(seed)
    h = w = img_res // 16
    fmap = torch.randn(1, h, w, C, generator=g, dtype=torch.float32)  # [1,h,w,C]
    t_out = torch_proj_grid_forward(fmap, cam, dist, ms, grid_res, img_res)[0].numpy()  # [R^3,C]
    np.save(os.path.join(dump_dir, 'fmap_hwc.npy'),
            np.ascontiguousarray(fmap[0].numpy(), dtype=np.float32))
    np.save(os.path.join(dump_dir, 'expected.npy'),
            np.ascontiguousarray(t_out, dtype=np.float32))
    json.dump(dict(grid_resolution=grid_res, image_resolution=img_res, C=C,
                   camera_angle_x=cam, distance=dist, mesh_scale=ms, H=h, W=w),
              open(os.path.join(dump_dir, 'meta.json'), 'w'), indent=2)
    print(f"[test_proj_cond] dumped golden -> {dump_dir} (fmap[{h},{w},{C}], expected[{grid_res**3},{C}])")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--dump':
        dump_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'golden_ref', 'proj_case0')
        dump_golden(dump_dir)
        return
    print(f"[test_proj_cond] torch {torch.__version__}, cuda devices={torch.cuda.device_count()} (expect 0)")
    cases = [
        # (grid_res, img_res, C, camera_angle_x, distance, mesh_scale, seed)
        (16, 512, 1024, 0.8575560450553894, 2.0, 1.0, 0),   # SS stage (real default)
        (16, 512, 1024, 0.50, 1.8, 1.0, 1),
        (16, 512, 1024, 1.20, 2.5, 0.8, 2),
        (32, 512, 1024, 0.8575560450553894, 2.0, 1.0, 3),   # shape_512 grid
        (32, 512, 16,   0.70, 2.2, 1.1, 4),
        (64, 1024, 1024, 0.8575560450553894, 2.0, 1.0, 5),  # shape_1024 grid
        (64, 1024, 16,  0.30, 3.0, 0.9, 6),
        (16, 512, 8,    0.95, 1.5, 1.3, 7),
    ]
    allok = True
    for c in cases:
        allok &= run_case(*c)
    print("[test_proj_cond]", "ALL PASS" if allok else "SOME FAILED")
    sys.exit(0 if allok else 1)


if __name__ == '__main__':
    main()
