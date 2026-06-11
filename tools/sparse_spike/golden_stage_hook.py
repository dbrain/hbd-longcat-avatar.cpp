"""
M0 — Staged golden dumper for the Pixal3D / TRELLIS.2 image->3D pipeline.

Generalizes golden_hook.py (which only captured submanifold-conv sub-layers) to
dump EVERY *stage boundary* of the geometry pipeline + per-stage peak VRAM, so the
C++/ggml port can be validated stage-by-stage (milestones M1..M5) against real
Python output. GPU-time, run-once. Authoring is GPU-free.

WHAT IT CAPTURES (monkeypatches Pixal3DImageTo3DPipeline class methods + a couple
of model forwards; each capture is wrapped in try/except so one failure never
aborts the decode):

  pre/         preprocess_image   -> preprocessed square RGB (png + uint8 npy)
  cam.json     camera_params (camera_angle_x, distance, mesh_scale)  [from run()]
  stage1_cond/ get_proj_cond_ss   -> z_global [B,5,1024], z_proj [B,16^3,Cproj]
  stage1_ssdec/sparse_structure_decoder.forward -> z_s (decoder INPUT [1,8,16,16,16])
                                                  + ss_logits (OUTPUT [1,1,64,64,64])
  stage1_out/  sample_sparse_structure -> coords [N,4] int32  (occupied @ res 32)
  stage2_cond/ get_proj_cond_shape #0 (shape_512, grid 32) -> global + proj(sparse feats,coords)
  stage2_out/  sample_shape_slat  -> lr_slat (denorm SparseTensor feats[N,32], coords)
  stage3a_up/  shape_slat_decoder.upsample -> hr_coords [N',4]
  stage3b_cond/get_proj_cond_shape #1 (shape_1024, grid actual_hr//16) -> global + proj
  stage3b_out/ decode_shape_slat INPUT -> shape_slat (final geom latent feats[M,32],coords)
  stage4_cond/ get_proj_cond_shape #2 (tex_1024) -> global + proj         [PHASE-2]
  stage4_out/  sample_tex_slat    -> tex_slat (denorm SparseTensor feats[M,32])  [PHASE-2]
  stage5_mesh/ decode_shape_slat OUTPUT meshes -> verts[V,3], faces[T,3] (pre fill_holes)
                                              + subs (per-up-block subdivision coords)
  stage5_final/decode_latent      -> final mesh verts/faces (post fill_holes)
  vram.json    per-stage peak allocated/reserved MiB (low_vram sequential offload)
  pipeline.json (copied)          + resolved_config.json (channel widths etc.)

USAGE — see golden_stage_runner.py. Quick:
  cd <Pixal3D>; source .venv/bin/activate
  ATTN_BACKEND=sdpa python <spike>/tools/sparse_spike/golden_stage_runner.py \
      --image <img> --out <spike>/tools/sparse_spike/golden_stages

NOTE on precision: flow-model torsos run bf16 + fp16 VAEs (per ckpt configs), so
these goldens carry ~bf16/fp16 noise. The C++ port stays fp32 (correctness-first);
expect validation tol vs these goldens ~1e-2 rel on flow outputs, tighter (~1e-3)
on the fp16 VAE/decoder paths. Judge by E2E mesh agreement, not a fixed tol.
"""
import json
import os
import shutil
import numpy as np


_STATE = {
    'out_dir': 'golden_stages',
    'installed': False,
    'shape_cond_calls': 0,   # get_proj_cond_shape is called 3x (512 / 1024 / tex)
    'vram': {},              # stage_name -> {alloc_mib, reserved_mib}
    'manifest': {},          # stage_name -> small dict of shapes/dtypes
}

_SHAPE_COND_NAMES = ['stage2_cond', 'stage3b_cond', 'stage4_cond']


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _np(t):
    import torch
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        return t
    return t.detach().to('cpu', dtype=torch.float32).numpy()


def _np_raw(t):
    """Preserve integer dtype (coords) instead of forcing float32."""
    import torch
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        return t
    return t.detach().cpu().numpy()


def _save(stage, name, arr):
    if arr is None:
        return None
    d = os.path.join(_STATE['out_dir'], stage)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name + '.npy')
    np.save(p, arr)
    return list(arr.shape), str(arr.dtype)


def _record(stage, **shapes):
    _STATE['manifest'][stage] = shapes
    d = os.path.join(_STATE['out_dir'], stage)
    os.makedirs(d, exist_ok=True)
    json.dump(shapes, open(os.path.join(d, 'manifest.json'), 'w'), indent=2)
    print(f"[stage_hook] {stage}: {shapes}", flush=True)


def _is_sparse(x):
    return hasattr(x, 'feats') and hasattr(x, 'coords')


def _dump_sparse(stage, prefix, st):
    """Dump a SparseTensor's feats + coords (+ spatial metadata)."""
    info = {}
    f = _save(stage, prefix + '_feats', _np(st.feats))
    c = _save(stage, prefix + '_coords', _np_raw(st.coords).astype(np.int32))
    if f:
        info[prefix + '_feats'] = f
    if c:
        info[prefix + '_coords'] = c
    try:
        info[prefix + '_shape'] = list(st.shape)
        info[prefix + '_spatial_shape'] = list(st.spatial_shape)
    except Exception:
        pass
    return info


class _Vram:
    """Context manager: reset + capture peak CUDA memory for a stage."""
    def __init__(self, stage):
        self.stage = stage

    def __enter__(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
        return self

    def __exit__(self, *a):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                self_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
                self_res = torch.cuda.max_memory_reserved() / (1024 ** 2)
                _STATE['vram'][self.stage] = {
                    'peak_alloc_mib': round(self_alloc, 1),
                    'peak_reserved_mib': round(self_res, 1),
                }
                print(f"[stage_hook] VRAM {self.stage}: "
                      f"alloc={self_alloc:.1f} MiB reserved={self_res:.1f} MiB", flush=True)
        except Exception:
            pass
        return False  # never swallow exceptions


# --------------------------------------------------------------------------- #
# install
# --------------------------------------------------------------------------- #
def install(out_dir='golden_stages'):
    if _STATE['installed']:
        return
    _STATE['out_dir'] = out_dir
    os.makedirs(out_dir, exist_ok=True)

    from pixal3d.pipelines.pixal3d_image_to_3d import Pixal3DImageTo3DPipeline as P

    # ---- copy the config files for ground-truth port reference ----
    _copy_configs(out_dir)

    # ---- PRE: preprocess_image ----
    _orig_pre = P.preprocess_image

    def preprocess_image(self, input, bg_color=(0, 0, 0)):
        with _Vram('pre'):
            out = _orig_pre(self, input, bg_color)
        try:
            arr = np.array(out)
            _save('pre', 'preprocessed_rgb', arr.astype(np.uint8))
            try:
                out.save(os.path.join(out_dir, 'pre', 'preprocessed.png'))
            except Exception:
                pass
            _record('pre', preprocessed_rgb=[list(arr.shape), str(arr.dtype)])
        except Exception as e:
            print(f"[stage_hook] pre capture skipped: {e}")
        return out

    P.preprocess_image = preprocess_image

    # ---- STAGE 1a: get_proj_cond_ss ----
    _orig_cond_ss = P.get_proj_cond_ss

    def get_proj_cond_ss(self, image, **kw):
        with _Vram('stage1_cond'):
            out = _orig_cond_ss(self, image, **kw)
        try:
            cam = {k: float(v) for k, v in kw.items()
                   if k in ('camera_angle_x', 'distance', 'mesh_scale')}
            json.dump(cam, open(os.path.join(out_dir, 'cam.json'), 'w'), indent=2)
            c = out['cond']
            info = {}
            info.update({'global': _save('stage1_cond', 'global', _np(c['global']))})
            # SS proj cond is a DENSE tensor [B, 16^3, Cproj]
            if _is_sparse(c['proj']):
                info.update(_dump_sparse('stage1_cond', 'proj', c['proj']))
            else:
                info['proj'] = _save('stage1_cond', 'proj', _np(c['proj']))
            _record('stage1_cond', **info)
        except Exception as e:
            print(f"[stage_hook] stage1_cond capture skipped: {e}")
        return out

    P.get_proj_cond_ss = get_proj_cond_ss

    # ---- STAGE 1b: sparse_structure_decoder.forward (grab z_s + logits) ----
    # Patch lazily inside sample_sparse_structure so the module exists / is on GPU.
    _orig_sample_ss = P.sample_sparse_structure

    def sample_sparse_structure(self, cond, resolution, num_samples=1, sampler_params={}):
        dec = self.models.get('sparse_structure_decoder')
        if dec is not None and not getattr(dec, '_stage_hooked', False):
            _orig_fwd = dec.forward

            def dec_fwd(x, _of=_orig_fwd):
                try:
                    _save('stage1_ssdec', 'z_s', _np(x))
                except Exception as e:
                    print(f"[stage_hook] z_s capture skipped: {e}")
                out = _of(x)
                try:
                    sh = _save('stage1_ssdec', 'ss_logits', _np(out))
                    _record('stage1_ssdec', z_s=[list(x.shape), str(x.dtype)], ss_logits=sh)
                except Exception as e:
                    print(f"[stage_hook] ss_logits capture skipped: {e}")
                return out

            dec.forward = dec_fwd
            dec._stage_hooked = True

        with _Vram('stage1'):
            coords = _orig_sample_ss(self, cond, resolution, num_samples, sampler_params)
        try:
            sh = _save('stage1_out', 'coords', _np_raw(coords).astype(np.int32))
            _record('stage1_out', coords=sh, n_voxels=int(coords.shape[0]),
                    resolution=int(resolution))
        except Exception as e:
            print(f"[stage_hook] stage1_out capture skipped: {e}")
        return coords

    P.sample_sparse_structure = sample_sparse_structure

    # ---- STAGE 2a / 3b / 4: get_proj_cond_shape (3 calls) ----
    _orig_cond_shape = P.get_proj_cond_shape

    def get_proj_cond_shape(self, image_cond_model, image, coords, **kw):
        idx = _STATE['shape_cond_calls']
        name = _SHAPE_COND_NAMES[idx] if idx < len(_SHAPE_COND_NAMES) else f'shapecond{idx}'
        _STATE['shape_cond_calls'] += 1
        with _Vram(name):
            out = _orig_cond_shape(self, image_cond_model, image, coords, **kw)
        try:
            c = out['cond']
            info = {'grid_resolution': int(getattr(image_cond_model, 'grid_resolution', -1)),
                    'global': _save(name, 'global', _np(c['global']))}
            if _is_sparse(c['proj']):
                info.update(_dump_sparse(name, 'proj', c['proj']))
            else:
                info['proj'] = _save(name, 'proj', _np(c['proj']))
            _record(name, **info)
        except Exception as e:
            print(f"[stage_hook] {name} capture skipped: {e}")
        return out

    P.get_proj_cond_shape = get_proj_cond_shape

    # ---- STAGE 2b: sample_shape_slat (LR slat) ----
    _orig_sample_shape = P.sample_shape_slat

    def sample_shape_slat(self, cond, flow_model, coords, sampler_params={}):
        _patch_upsample(self)  # Stage 3a upsample fires right after this in run()
        with _Vram('stage2'):
            slat = _orig_sample_shape(self, cond, flow_model, coords, sampler_params)
        try:
            info = _dump_sparse('stage2_out', 'lr_slat', slat)
            _record('stage2_out', **info)
        except Exception as e:
            print(f"[stage_hook] stage2_out capture skipped: {e}")
        return slat

    P.sample_shape_slat = sample_shape_slat

    # ---- STAGE 3a: shape_slat_decoder.upsample (hr_coords) ----
    # Patch lazily: the decoder module exists after pipeline load.
    def _patch_upsample(self):
        dec = self.models.get('shape_slat_decoder')
        if dec is not None and not getattr(dec, '_stage_up_hooked', False):
            _orig_up = dec.upsample

            def up(slat, upsample_times=4, _ou=_orig_up):
                with _Vram('stage3a'):
                    hr = _ou(slat, upsample_times=upsample_times)
                try:
                    sh = _save('stage3a_up', 'hr_coords', _np_raw(hr).astype(np.int32))
                    _record('stage3a_up', hr_coords=sh, n=int(hr.shape[0]),
                            upsample_times=int(upsample_times))
                except Exception as e:
                    print(f"[stage_hook] stage3a capture skipped: {e}")
                return hr

            dec.upsample = up
            dec._stage_up_hooked = True

    # ---- STAGE 3b out + STAGE 5 mesh: decode_shape_slat ----
    _orig_decode_shape = P.decode_shape_slat

    def decode_shape_slat(self, slat, resolution):
        try:
            info = _dump_sparse('stage3b_out', 'shape_slat', slat)
            info['resolution'] = int(resolution)
            _record('stage3b_out', **info)
        except Exception as e:
            print(f"[stage_hook] stage3b_out capture skipped: {e}")
        with _Vram('stage5_decode'):
            ret = _orig_decode_shape(self, slat, resolution)
        try:
            meshes, subs = ret
            m = meshes[0]
            v = _save('stage5_mesh', 'vertices', _np(m.vertices))
            f = _save('stage5_mesh', 'faces', _np_raw(m.faces).astype(np.int64))
            sub_info = {}
            for i, s in enumerate(subs):
                if _is_sparse(s):
                    sub_info.update(_dump_sparse('stage5_mesh', f'sub{i}', s))
            _record('stage5_mesh', vertices=v, faces=f,
                    n_subs=len(subs), **sub_info)
        except Exception as e:
            print(f"[stage_hook] stage5_mesh capture skipped: {e}")
        return ret

    P.decode_shape_slat = decode_shape_slat

    # ---- STAGE 4 out: sample_tex_slat (PHASE-2, captured anyway) ----
    _orig_sample_tex = P.sample_tex_slat

    def sample_tex_slat(self, cond, flow_model, shape_slat, sampler_params={}):
        with _Vram('stage4'):
            slat = _orig_sample_tex(self, cond, flow_model, shape_slat, sampler_params)
        try:
            info = _dump_sparse('stage4_out', 'tex_slat', slat)
            _record('stage4_out', **info)
        except Exception as e:
            print(f"[stage_hook] stage4_out capture skipped: {e}")
        return slat

    P.sample_tex_slat = sample_tex_slat

    # ---- STAGE 5 final: decode_latent (post fill_holes) ----
    _orig_decode_latent = P.decode_latent

    def decode_latent(self, shape_slat, tex_slat, resolution):
        _patch_upsample(self)  # ensure upsample hook is in place (called earlier in run())
        out_mesh = _orig_decode_latent(self, shape_slat, tex_slat, resolution)
        try:
            m = out_mesh[0]
            v = _save('stage5_final', 'vertices', _np(m.vertices))
            f = _save('stage5_final', 'faces', _np_raw(m.faces).astype(np.int64))
            extra = {}
            if hasattr(m, 'coords') and m.coords is not None:
                extra['voxel_coords'] = _save('stage5_final', 'voxel_coords',
                                              _np_raw(m.coords).astype(np.int32))
            if hasattr(m, 'attrs') and m.attrs is not None:
                extra['attrs'] = _save('stage5_final', 'attrs', _np(m.attrs))
            _record('stage5_final', vertices=v, faces=f, resolution=int(resolution), **extra)
        except Exception as e:
            print(f"[stage_hook] stage5_final capture skipped: {e}")
        return out_mesh

    P.decode_latent = decode_latent

    import atexit

    def _dump_meta():
        json.dump(_STATE['vram'], open(os.path.join(out_dir, 'vram.json'), 'w'), indent=2)
        json.dump(_STATE['manifest'], open(os.path.join(out_dir, 'stages_manifest.json'), 'w'), indent=2)
        print(f"[stage_hook] wrote vram.json + stages_manifest.json -> {out_dir}", flush=True)

    atexit.register(_dump_meta)

    _STATE['installed'] = True
    print(f"[stage_hook] installed -> {out_dir}", flush=True)


def _copy_configs(out_dir):
    """Copy pipeline.json + per-model ckpt configs from the HF cache for reference,
    and emit a resolved_config.json summary of the geometry-path channel widths."""
    cfg_out = os.path.join(out_dir, 'configs')
    os.makedirs(cfg_out, exist_ok=True)
    snap = ('/home/dbrain/.cache/huggingface/hub/models--TencentARC--Pixal3D/'
            'snapshots/0b31f9160aa400719af409098bff7936a932f726')
    try:
        pj = os.path.join(snap, 'pipeline.json')
        if os.path.exists(pj):
            shutil.copy(pj, os.path.join(cfg_out, 'pipeline.json'))
        ckpts = os.path.join(snap, 'ckpts')
        if os.path.isdir(ckpts):
            for f in os.listdir(ckpts):
                if f.endswith('.json'):
                    shutil.copy(os.path.join(ckpts, f), os.path.join(cfg_out, f))
        print(f"[stage_hook] copied configs -> {cfg_out}", flush=True)
    except Exception as e:
        print(f"[stage_hook] config copy skipped: {e}")
