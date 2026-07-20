#!/usr/bin/env python3
"""Capture golden output of o_voxel.convert.mesh_to_flexible_dual_grid for the texture pipeline.

Replicates pixal3d Trellis2TexturingPipeline.encode_shape_slat EXACTLY:
  mesh -> preprocess_mesh (center/scale 0.99999/extent + y/z swap to [-0.5,0.5]) ->
  o_voxel.convert.mesh_to_flexible_dual_grid(grid_size=1024, aabb=+-0.5, fw=1, bw=0.2, reg=1e-2)
Dumps coords[N,3] int32, dual_vertices[N,3] f32, intersected[N,3] int8 to OUT dir.

Also (optionally) runs the SAME downsample the encoder coords pass through (shape via S2C
floor-div /16) so we can confirm the voxelizer output reduces to the golden_69k @grid64 coords
WITHOUT needing the GPU encoder. (The encoder only changes feats, not the coord lattice.)
"""
import os, sys, numpy as np, torch, trimesh
# Import the voxelizer module DIRECTLY (the o_voxel package __init__ pulls in triton/flex_gemm
# which needs a live CUDA driver; the _C cpu solver does not).
import importlib.util, types
_OV = '/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/lib/python3.10/site-packages/o_voxel'
# stub package so the relative `from .. import _C` resolves
pkg = types.ModuleType('o_voxel'); pkg.__path__ = [_OV]; sys.modules['o_voxel'] = pkg
_spec_c = importlib.util.spec_from_file_location(
    'o_voxel._C', _OV + '/_C.cpython-310-x86_64-linux-gnu.so')
_C = importlib.util.module_from_spec(_spec_c); sys.modules['o_voxel._C'] = _C
_spec_c.loader.exec_module(_C)
conv_pkg = types.ModuleType('o_voxel.convert'); conv_pkg.__path__ = [_OV + '/convert']
sys.modules['o_voxel.convert'] = conv_pkg
_spec = importlib.util.spec_from_file_location(
    'o_voxel.convert.flexible_dual_grid', _OV + '/convert/flexible_dual_grid.py')
fdg = importlib.util.module_from_spec(_spec); sys.modules['o_voxel.convert.flexible_dual_grid'] = fdg
_spec.loader.exec_module(fdg)


class _ConvNS:
    mesh_to_flexible_dual_grid = staticmethod(fdg.mesh_to_flexible_dual_grid)


class _OVNS:
    convert = _ConvNS()


o_voxel = _OVNS()

GLB = sys.argv[1] if len(sys.argv) > 1 else \
    '/home/dbrain/dev/longcat-sparse-spike/tools/m1_ref/cpp_port/us_native_69k.glb'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/mnt/hdd/pixal3d_tex/golden_voxel'
DO_PREPROCESS = os.environ.get('PREPROCESS', '1') == '1'
RES = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
BW = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
REG = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-2
os.makedirs(OUT, exist_ok=True)


def preprocess_mesh(vertices):
    """Exact copy of Trellis2TexturingPipeline.preprocess_mesh."""
    vertices = vertices.copy()
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()
    vertices = (vertices - center) * scale
    tmp = vertices[:, 1].copy()
    vertices[:, 1] = -vertices[:, 2]
    vertices[:, 2] = tmp
    assert np.all(vertices >= -0.5) and np.all(vertices <= 0.5), 'vertices out of range'
    return vertices


def main():
    m = trimesh.load(GLB, force='mesh', process=False)
    verts = np.asarray(m.vertices, dtype=np.float64)
    faces = np.asarray(m.faces, dtype=np.int64)
    print(f'loaded {GLB}: V={len(verts)} F={len(faces)}')
    print(f'  raw vmin={verts.min(0)} vmax={verts.max(0)}')

    if DO_PREPROCESS:
        verts = preprocess_mesh(verts)
        print(f'  after preprocess vmin={verts.min(0)} vmax={verts.max(0)}')
    else:
        print('  PREPROCESS=0 -> using verts as-is')

    vt = torch.from_numpy(verts).float().cpu()
    ft = torch.from_numpy(faces).long().cpu()

    voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
        vt, ft,
        grid_size=RES,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=BW,
        regularization_weight=REG,
        timing=False,
    )
    coords = voxel_indices.cpu().numpy()
    dual = dual_vertices.cpu().numpy()
    inter = intersected.cpu().numpy()
    print(f'\n=== RAW VOXELIZER OUTPUT @grid{RES} ===')
    print('coords', coords.shape, coords.dtype, 'min', coords.min(0), 'max', coords.max(0))
    print('dual  ', dual.shape, dual.dtype, 'min', dual.min(0), 'max', dual.max(0))
    print('inter ', inter.shape, inter.dtype, 'uniq', np.unique(inter), 'sum/axis', inter.reshape(-1,inter.shape[-1]).sum(0) if inter.ndim>1 else inter.sum())

    coords = coords.astype(np.int32)
    dual = dual.astype(np.float32)
    inter = inter.astype(np.int8)
    np.save(os.path.join(OUT, 'coords.npy'), coords)
    np.save(os.path.join(OUT, 'dual_vertices.npy'), dual)
    np.save(os.path.join(OUT, 'intersected.npy'), inter)
    # also save the preprocessed mesh so the C++ test reads IDENTICAL vertices
    np.save(os.path.join(OUT, 'verts_pre.npy'), verts.astype(np.float32))
    np.save(os.path.join(OUT, 'faces.npy'), faces.astype(np.int32))
    print(f'\nsaved coords/dual_vertices/intersected/verts_pre/faces to {OUT}')

    # encoder feats6 (for reference): [dual*1024 - coords - 0.5, inter - 0.5]
    feats6 = np.concatenate([dual * RES - coords - 0.5, inter.astype(np.float32) - 0.5], axis=1)
    np.save(os.path.join(OUT, 'feats6.npy'), feats6.astype(np.float32))

    # SANITY: downsample coords 1024->64 (floor-div by 16) -> unique -> compare set to golden_69k @grid64
    gpath = '/mnt/hdd/pixal3d_tex/golden_69k_1024/shape_slat_coords.npy'
    if os.path.exists(gpath):
        gc = np.load(gpath)  # [M,4] (b,x,y,z) @grid64
        gset = set(map(tuple, gc[:, 1:].tolist()))
        ds = coords // 16  # 1024 -> 64 (4 S2C halvings)
        dsset = set(map(tuple, np.unique(ds, axis=0).tolist()))
        inter_ = len(gset & dsset)
        print(f'\n=== DOWNSAMPLE 1024->64 vs golden_69k coord SET ===')
        print(f'  golden cells {len(gset)}, our-downsampled cells {len(dsset)}, intersection {inter_}')
        iou = inter_ / len(gset | dsset)
        print(f'  IoU {iou:.4f}  (recall {inter_/len(gset):.4f})')


if __name__ == '__main__':
    main()
