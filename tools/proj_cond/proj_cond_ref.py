"""
Numpy reference for the Pixal3D "proj" image-conditioning geometry — the net-new
op #2/#3 (camera unproject + 2D bilinear grid_sample).

This is the GPU-FREE correctness oracle (mirrors sparse_conv_ref.py's role). The
camera-unproject is pure host arithmetic; only the bilinear sample needs a GPU op in
production, and we reference it here on CPU. Validated bit/tol-exact against the actual
torch ProjGrid in test_proj_cond.py (CPU, CUDA hidden).

Layout matches `pixal3d/trainers/flow_matching/mixins/image_conditioned_proj.py`
(`project_points_to_image_batch`, `sample_features`, `ProjGrid`). Everything fp32 to
mirror torch fp32 (linalg.inv etc.). The ONLY runtime inputs are 3 scalars
(camera_angle_x, distance, mesh_scale) + the DINOv3 feature map; the grid is constant.
"""
import numpy as np


# Blender-frame rotation applied to the canonical [-1,1]^3 grid (constant).
_ROT = np.array([[1.0, 0.0, 0.0],
                 [0.0, 0.0, -1.0],
                 [0.0, 1.0, 0.0]], dtype=np.float32)


def build_grid_points(grid_resolution: int) -> np.ndarray:
    """meshgrid(linspace(-1,1,R), indexing='ij') stacked (x,y,z) -> [R^3,3], then
    rotated to the Blender frame (gp @ ROT.T). Token order = ((i*R)+j)*R+k. Constant."""
    one = np.linspace(-1.0, 1.0, grid_resolution, dtype=np.float32)
    x, y, z = np.meshgrid(one, one, one, indexing='ij')
    gp = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    return gp @ _ROT.T


def front_view_transform(distance: float) -> np.ndarray:
    """Front-view camera-to-world 4x4 with translation row [1,3] = -distance."""
    T = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0, -float(distance)],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return T


def project_points_to_image(points_3d: np.ndarray, transform_matrix: np.ndarray,
                            camera_angle_x: float, resolution: int):
    """Port of project_points_to_image_batch (B=1). points_3d [N,3].
    Returns points_2d [N,2] (x_pixel,y_pixel), depth [N], valid_mask [N]."""
    N = points_3d.shape[0]
    ones = np.ones((N, 1), dtype=np.float32)
    ph = np.concatenate([points_3d, ones], axis=-1)            # [N,4]
    world_to_camera = np.linalg.inv(transform_matrix.astype(np.float32)).astype(np.float32)
    pc = (ph @ world_to_camera.T)[:, :3]                       # [N,3]
    x_cam, y_cam, z_cam = pc[:, 0], pc[:, 1], pc[:, 2]
    depth = -z_cam
    focal = 16.0 / np.tan(np.float32(camera_angle_x) / 2.0)
    focal_px = focal * resolution / 32.0                       # sensor_width=32mm
    x_ndc = focal_px * x_cam / (-z_cam + 1e-8)
    y_ndc = focal_px * y_cam / (-z_cam + 1e-8)
    x_pixel = x_ndc + resolution / 2.0
    y_pixel = -y_ndc + resolution / 2.0
    valid = ((x_pixel >= 0) & (x_pixel < resolution) &
             (y_pixel >= 0) & (y_pixel < resolution) & (depth > 0))
    pts = np.stack([x_pixel, y_pixel], axis=-1).astype(np.float32)
    return pts, depth.astype(np.float32), valid


def grid_sample_bilinear_border(fmap: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Port of F.grid_sample(mode=bilinear, align_corners=False, padding_mode='border').
    fmap [C,H,W], grid [K,2] in [-1,1] (gx->W, gy->H). Returns [C,K].

    align_corners=False:  src = (g+1)/2 * size - 0.5
    border padding:       clamp the 4 corner *indices* to [0,size-1]; weights from the
                          unclamped fractional position (== torch semantics)."""
    C, H, W = fmap.shape
    gx = grid[:, 0]
    gy = grid[:, 1]
    ix = (gx + 1.0) * 0.5 * W - 0.5
    iy = (gy + 1.0) * 0.5 * H - 0.5
    x0 = np.floor(ix).astype(np.int64)
    y0 = np.floor(iy).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1
    wx1 = (ix - x0).astype(np.float32)
    wy1 = (iy - y0).astype(np.float32)
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    x0c = np.clip(x0, 0, W - 1); x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1); y1c = np.clip(y1, 0, H - 1)
    # gather [C,K] for each corner
    f00 = fmap[:, y0c, x0c]
    f01 = fmap[:, y0c, x1c]
    f10 = fmap[:, y1c, x0c]
    f11 = fmap[:, y1c, x1c]
    out = (f00 * (wy0 * wx0) + f01 * (wy0 * wx1)
           + f10 * (wy1 * wx0) + f11 * (wy1 * wx1))
    return out.astype(np.float32)


def proj_grid_forward(fmap_bhwc: np.ndarray, camera_angle_x: float, distance: float,
                      mesh_scale: float, grid_resolution: int, image_resolution: int):
    """Port of ProjGrid.forward (B=1, BHWC input). fmap_bhwc [H,W,C].
    Returns z_proj [R^3, C]."""
    gp = build_grid_points(grid_resolution)
    gp = gp / np.float32(mesh_scale) / 2.0
    T = front_view_transform(distance)
    pts, depth, valid = project_points_to_image(gp, T, camera_angle_x, image_resolution)
    grid_norm = (pts + 0.5) / image_resolution * 2.0 - 1.0   # [R^3,2]
    fmap_chw = np.transpose(fmap_bhwc, (2, 0, 1))            # [C,H,W]
    sampled = grid_sample_bilinear_border(fmap_chw, grid_norm.astype(np.float32))  # [C,K]
    return sampled.T.astype(np.float32)                      # [K,C]
