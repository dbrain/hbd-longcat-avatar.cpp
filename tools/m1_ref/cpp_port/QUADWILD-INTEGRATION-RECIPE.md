# quadwild-bimdf quad-retopology — build, run, and integration recipe

RUNG 2 of the image→rigged-3D C++ pipeline: add a **quad retopology stage** between the
UltraShape refine and the QEM decimate. DEFAULT tool = **quadwild-bimdf** (Gurobi-free, Bi-MDF /
libSatsuma solver). Pure C++, CPU-only, zero VRAM. Validated on our real toy-robot mesh 2026-07-16.

Result on `refined.glb` (19,140 v tri): **100% quad, watertight+manifold, ~9 s wall, ~62 MB RSS,
surface deviation 0.10 % of bbox-diagonal (mean).**

---

## 0. TL;DR — what exists on disk now

| Thing | Path |
|---|---|
| Source + built binaries (PERSISTENT) | `~/dev/quadwild-bimdf/` |
| Binaries | `~/dev/quadwild-bimdf/build/Build/bin/{quadwild,quad_from_patches,cli_trace,viz_mesh_results}` |
| Pinned CMake 3.x (needed to rebuild — see §1.2) | `~/dev/quadwild-bimdf/toolchain-cmake-3.31.6/bin/cmake` |
| Test I/O + quad results | `/mnt/hdd/3d/avatar-shootout/_shootout_out/quadwild_test/` |
| The quad mesh (canonical result) | `.../quadwild_test/refined_rem_p0_123_quadrangulation_smooth.obj` |

Repo commit built: `cbda68e5` (github.com/cgg-bern/quadwild-bimdf, GPL-3).

---

## 1. Build (Gurobi-free) — exact commands that worked

Box: CachyOS (Arch), gcc 16.1.1, system CMake 4.3.3, 12 cores. Two toolchain landmines had to be
worked around (both documented below). Deps that were auto-satisfied: **eigen, libigl, satsuma,
lemon, OpenMesh, CoMISo, vcglib, lpsolve** are all vendored git submodules; **gmm** is
FetchContent-downloaded at configure time (needs network once).

### 1.1 Clone + system deps
```bash
cd ~/dev
git clone --recursive https://github.com/cgg-bern/quadwild-bimdf
# system deps (Arch names). boost headers were the only missing one; lapack/blas/cblas already present.
sudo pacman -S --needed --noconfirm boost      # -> boost 1.91
# ninja: any ninja on PATH works (used ~/.pixi/bin/ninja here)
```
Docker/Debian equivalent (from the repo's Dockerfile, for reference): `build-essential cmake ninja-build
libboost-filesystem-dev libboost-system-dev libboost-regex-dev libgmm++-dev liblapack-dev
libopenblas64-serial-dev`.

### 1.2 LANDMINE A — CMake 4.x rejects the vendored submodules
`libs/lemon/CMakeLists.txt` does `cmake_policy(SET CMP0048 OLD)`, which CMake ≥4.0 refuses outright
(several other submodules declare `cmake_minimum_required(VERSION <3.5)`). `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`
is **not** enough (lemon's explicit OLD-policy set still errors). Fix: build with a **CMake 3.x**.
A standalone CMake 3.31.6 was fetched and pinned into the repo so future rebuilds don't re-download:
```bash
CM=~/dev/quadwild-bimdf/toolchain-cmake-3.31.6/bin/cmake   # cmake 3.31.6
# (originally: wget https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz )
```

### 1.3 LANDMINE B — gcc 16 `-Wtemplate-body` rejects vcglib
gcc ≥14 eagerly checks template bodies; vcglib's `import_ply.h` has a never-instantiated `LoadCamera`
that references `this->pi` / `this->camera` (locals, not members) → hard error. Standard legacy-code
fix: `-fpermissive -Wno-template-body`. (The Docker path avoids this by using gcc-12; on this box we
downgrade instead.)

### 1.4 Configure + build (the working invocation)
```bash
cd ~/dev/quadwild-bimdf
export PATH="$HOME/.pixi/bin:$PATH"                       # any ninja on PATH
CM=~/dev/quadwild-bimdf/toolchain-cmake-3.31.6/bin/cmake
rm -rf build && mkdir build
$CM -G Ninja -B build \
  -D QUADRETOPOLOGY_WITH_GUROBI=0 \
  -D SATSUMA_ENABLE_BLOSSOM5=0 \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CXX_FLAGS="-march=native -fpermissive -Wno-template-body" \
  -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  .
$CM --build build -j 12
```
Builds clean (0 errors, ~1 min). Binaries land in `build/Build/bin/`. `osqp not found` is a harmless
warning (CoMISo optional dep); no Gurobi/CPLEX/MOSEK/GLPK needed.

---

## 2. Run — the exact tri→quad CLI (documented two-step flow)

quadwild is **CLI-only** (reads OBJ/PLY files, writes OBJ). The flow config uses **relative**
`config/...` paths, so **CWD must be the repo root**; all outputs are written next to the *input* file.

Our meshes are GLB → convert to OBJ first (CPU trimesh):
```python
# /mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
import trimesh
m = trimesh.load("refined.glb", force='mesh', process=False)
m.export("refined.obj", include_texture=False, include_normals=False)
```

Two steps:
```bash
cd ~/dev/quadwild-bimdf
BIN=./build/Build/bin ; OUT=/mnt/hdd/3d/avatar-shootout/_shootout_out/quadwild_test

# STEP 1+2 : remesh + cross-field + patch tracing.  '2' = stop after tracing.
#            (optional 4th arg = a .sharp feature file, see §4)
$BIN/quadwild $OUT/refined.obj 2 config/prep_config/basic_setup.txt
#   -> writes  refined_rem.obj  refined_rem.rosy  refined_rem.sharp
#              refined_rem_p0.obj  .patch  .corners  .feature  .c_feature

# STEP 3 : Bi-MDF quantization + quad extraction.  <num>=any int (goes into output name).
$BIN/quad_from_patches $OUT/refined_rem_p0.obj 123 \
    config/main_config/flow_noalign_lemon.txt $OUT/refined_stats.json
#   -> writes  refined_rem_p0_123_quadrangulation.obj         (raw quads)
#              refined_rem_p0_123_quadrangulation_smooth.obj   <-- USE THIS (smoothed quads)
```
`flow_noalign_lemon.txt` is the fully-open (Gurobi-free) config: `useFlowSolver 1`, and it points at
`config/main_config/flow_virtual_simple.json` + `config/satsuma/lemon.json` (all relative → CWD=repo root).

Single-binary alternative: `quadwild mesh.obj 3 <config>` runs all three steps internally
(`remeshAndField`→`trace`→`quadrangulate`), but the two-step form is what the README documents and what
was validated here.

---

## 3. Quality on our real mesh (CPU eval, no rendering)

Input `refined.glb`: 19,140 v / 38,276 tri, watertight. quadwild internally remeshes to 5,999 v / 11,994 tri.

| metric | value |
|---|---|
| **quad output** | 6,448 v / **6,446 faces, 100 % quads** (ngon-hist `{4: 6446}`) |
| topology | boundary-edges 0, non-manifold-edges 0 → **watertight + 2-manifold** |
| silhouette (bbox extent) | 0.997×0.886×0.309 vs input 0.998×0.888×0.311 → within **0.3 %** |
| surface deviation (sym. Chamfer, 200k pts/side) | mean **0.10 %** of bbox-diag, p95 0.18 %, p99 0.24 %, max 0.50 % |
| runtime | step1+2 = **7.2 s**, step3 = **1.65 s** → ~9 s total |
| peak RSS | ~62 MB |

**`coarse.glb` (974,354 v) — DOES NOT COMPLETE.** Remesh + field finish (~90 s) but the patch-**tracing**
stage stalls (>9.5 min, killed, no output). quadwild is built for moderate-density tri meshes and
remeshes to a target edge length internally regardless of input density — so a 974k-v input just makes
the front stages pathologically slow for zero quality gain. **Feed quadwild the ~19k-v UltraShape
`refined` mesh, never the dense `coarse` mesh.** If a denser input must be handled, pre-decimate to
~15–40k tris first, or raise `scaleFact` in `basic_setup.txt` to force a coarser remesh.

Config tuning levers (`config/prep_config/basic_setup.txt`):
`do_remesh 1`, `sharp_feature_thr 35` (dihedral angle for auto sharp detection),
`alpha 0.01` (field-vs-smoothness), `scaleFact 1` (↑ = coarser/fewer quads, ↓ = denser).
Organic vs mechanical presets exist (`basic_setup_Organic.txt` / `_Mechanical.txt`).

---

## 4. Feeding P3-SAM part boundaries as feature lines (the loop lever)

quadwild takes user feature edges via a `.sharp` file (extra arg to STEP 1). When present, it **replaces**
the auto dihedral-angle detection (`BPar.UpdateSharp = !hasFeature`) and the remesher preserves/splits
along those edges. Format (parsed by `FieldTriMesh::LoadSharpFeatures`,
`components/field_computation/triangle_mesh_type.h:355`):
```
<N>                       # number of feature half-edges
<Type>,<FaceIdx>,<EdgeIdx>   # Type: 0=concave 1=convex ; EdgeIdx in {0,1,2} ; FaceIdx into the INPUT tri mesh
...
```
**Indices are into the input tri mesh passed to STEP 1** (before remeshing). To drive it from P3-SAM:
1. P3-SAM already yields a per-face part id on the refined mesh (`fids`, see `capture_p3sam_faces.py` /
   `per_part_decimate`).
2. A boundary edge = an edge whose two incident faces have different part ids. For each, find the owning
   `(faceIdx, edgeIdx∈0..2)` and emit `1,faceIdx,edgeIdx` (use `1`/convex as a neutral default).
3. Write the `.sharp`, then: `quadwild refined.obj 2 refined.sharp config/prep_config/basic_setup.txt`.

This is the single lever to nudge quad edge-flow to follow semantic parts (hands, head, joints).

---

## 5. Wiring it into `image_to_rig.cpp`

Insertion point: **`image_to_rig.cpp:332`**, right after
`refined = usr::refine(mesh.verts, mesh.faces, image, rc, use_cuda);` returns the `svae::Mesh`
(`{vector<float> verts; vector<int64_t> faces; int N,F}`), and **before** the decimate/bake
(`ppd::per_part_decimate`, line 357). New stage: refined-tri → quadwild → quad mesh → (feed QEM LOD ladder).

### 5a. First cut — shell out to the two CLIs (RECOMMENDED to start)
Pragmatic, low-risk, matches what was validated here.
```
svae::Mesh refined = usr::refine(...);              // in RAM (line 332)
// 1. write refined.verts/faces -> <tmp>/qw_in.obj   (trivial OBJ writer: 'v x y z' + 'f a b c' 1-based)
// 2. (optional) build <tmp>/qw_in.sharp from P3-SAM fids (§4)
// 3. system(): cd <quadwild_repo> &&
//      build/Build/bin/quadwild        <tmp>/qw_in.obj 2 [<tmp>/qw_in.sharp] config/prep_config/basic_setup.txt
//      build/Build/bin/quad_from_patches <tmp>/qw_in_rem_p0.obj 0 config/main_config/flow_noalign_lemon.txt <tmp>/stats.json
// 4. read <tmp>/qw_in_rem_p0_0_quadrangulation_smooth.obj back:
//      verts -> vector<float>; quad faces 'f a b c d' -> keep as quads, or triangulate (a,b,c),(a,c,d)
```
Caveats: (a) CWD must be the quadwild repo for the relative `config/...` paths — either `chdir` there,
or pre-rewrite the two json paths in `flow_noalign_lemon.txt` to absolute and pass an absolute config.
(b) Needs a scratch dir + temp-file cleanup. (c) `glb_writer.hpp` writes **triangles** only, so to emit a
quad GLB either triangulate on write (loses quad topology in the GLB but keeps the clean field-aligned
edge flow) or carry the quad list in a side channel through the QEM/rig stages. (d) Keep the built repo
+ `basic_setup.txt` + `flow_noalign_lemon.txt` on the deploy image.

### 5b. Proper cut — link the library in-process (later)
The pipeline is exposed as three C++ functions over VCG mesh types (`quadwild/functions.h`):
`remeshAndField(FieldTriMesh&, Parameters, meshFile, sharpFile, fieldFile)` →
trace (`trace.h`) → `quadrangulate(..., PolyMesh& quadmesh, ...)`. To link:
1. Add a small CMake lib target wrapping `quadwild/functions.cpp` + `trace.cpp` (currently compiled
   straight into the `quadwild` executable, not a standalone lib) and link `quadretopology`, `vcglib`,
   `libigl`, `CoMISo`, `satsuma`, `OpenMesh`.
2. Convert `svae::Mesh` → VCG `FieldTriMesh`/`TriangleMesh` (copy verts, faces) and `PolyMesh` → `svae::Mesh`
   (quads → keep or triangulate).
3. Note the current step functions still round-trip **intermediate files** (`.rosy` field, `.patch`,
   `.corners`, `.sharp`); an in-process port would want those refactored to in-RAM buffers, or just let
   them write to a scratch dir. Because of that file-coupling, **5a (shell-out) is the correct first cut**;
   promote to 5b only once the stage earns its keep. Same build-toolchain caveats (CMake 3.x, `-fpermissive`,
   `-Wno-template-body`) apply when linking against these submodules from our tree, which uses a newer gcc.

### 5c. Downstream
Quad output → QEM LOD ladder (quad-aware decimation preserves the edge loops) → SkinTokens rig.
Texture is re-baked onto the new mesh's fresh UV atlas exactly like the existing dense→lowpoly
`tex_reproject` (RP_CANON_TO_DENSE) — topology-independent, so the quad mesh is just another bake target.

---

## 6. Reproduce the eval numbers
```bash
PY=/mnt/hdd/3d/avatar-shootout/Pixal3D/.venv/bin/python
OUT=/mnt/hdd/3d/avatar-shootout/_shootout_out/quadwild_test
# quad fraction / manifold / bbox: parse OBJ face-line arities directly (trimesh triangulates quads on load)
# surface deviation: symmetric KDTree Chamfer on 200k surface samples/side (scipy.spatial.cKDTree; rtree not installed)
```
(eval scripts used are in the session scratchpad; the logic is one screen of numpy/scipy — see the
counts/topology/Chamfer approach above.)
