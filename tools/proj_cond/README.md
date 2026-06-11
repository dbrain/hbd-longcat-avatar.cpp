# proj_cond — Pixal3D "proj" image-conditioning geometry (net-new op #2/#3)

GPU-free correctness oracle for the camera-unproject + 2D bilinear grid_sample that
turns a DINOv3 feature map into per-voxel `z_proj` features. Mirrors the sparse-conv
spike methodology (numpy ref → C++ ref, validated bit/tol-exact vs the production math).

**Key insight:** the unproject is pure host arithmetic — the 3D grid is constant and the
camera is 3 scalars `(camera_angle_x, distance, mesh_scale)`, so the grid_sample
coordinates are computed ONCE per stage on the host. Only the bilinear gather needs a
GPU kernel in production.

## Files
- `proj_cond_ref.py` — numpy reference (unproject + grid_sample). The golden generator.
- `test_proj_cond.py` — validates `proj_cond_ref` vs the **real torch `ProjGrid`** (copied
  verbatim from `image_conditioned_proj.py`) on CPU. Also `--dump` writes a C++ golden.
- `proj_cond.cpp` — C++ mirror (4x4 inv + CPU grid_sample), host production candidate.
- `golden_ref/proj_case0/` — dumped torch golden (fmap[32,32,64] + expected[4096,64]).

## Run (CPU only — CUDA hidden)
```
VENV=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
CUDA_VISIBLE_DEVICES="" $VENV test_proj_cond.py            # numpy vs torch, all configs
CUDA_VISIBLE_DEVICES="" $VENV test_proj_cond.py --dump     # regenerate golden_ref/proj_case0
g++ -O2 -std=c++17 proj_cond.cpp -o proj_cond_test && ./proj_cond_test golden_ref/proj_case0
```
Status (2026-06-11): numpy-vs-torch ALL PASS (maxabs ~1e-5); C++-vs-golden maxabs 1.53e-5.

## grid_sample semantics to replicate in CUDA (exact)
`F.grid_sample(mode=bilinear, align_corners=False, padding_mode='border')`:
`src = (g+1)/2*size - 0.5`; bilinear with the 4 corner **indices clamped** to
`[0,size-1]` (border) but weights from the **unclamped** fractional position. fp32.

## Next (GPU): validate a CUDA grid_sample kernel vs `golden_ref/proj_case0/`, then vs the
real DINOv3 map captured in M0 `golden_stages/stage1_cond/` (z_proj end-to-end).
