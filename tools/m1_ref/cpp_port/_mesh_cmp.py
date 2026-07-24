#!/usr/bin/env python3
# Symmetric surface (point-cloud) chamfer between two GLB/PLY meshes — quality A/B for precision levers.
#   _mesh_cmp.py <a.glb> <b.glb>
# Samples points on each surface, kNN nearest-distance to the other's samples (scipy cKDTree, no rtree),
# reports as a fraction of the bbox diagonal (scale-free). ~0 => geometrically identical.
import os, sys, numpy as np
try:
    from scipy.spatial import cKDTree
except ModuleNotFoundError:
    cKDTree=None

def asmesh(p):
    # The server deliberately does not carry the heavyweight trimesh runtime.
    # Accept OBJ directly so a runbook candidate can be measured after the
    # native glb2obj conversion, without installing a renderer or touching a
    # GPU. Keep the old broad-format route when trimesh is available locally.
    if p.lower().endswith('.obj'):
        verts=[]; faces=[]
        with open(p, encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.startswith('v '):
                    xyz=line.split()[1:4]
                    if len(xyz)==3: verts.append([float(x) for x in xyz])
                elif line.startswith('f '):
                    ids=[int(x.split('/')[0])-1 for x in line.split()[1:] if x.split('/')[0]]
                    for i in range(1,len(ids)-1): faces.append([ids[0],ids[i],ids[i+1]])
        return np.asarray(verts,np.float32), np.asarray(faces,np.int64)
    import trimesh
    m=trimesh.load(p, force='mesh')
    return np.asarray(m.vertices,np.float32), np.asarray(m.faces,np.int64)

def sample_surface(mesh, count, rng):
    if not isinstance(mesh, tuple):
        return mesh.sample(count)
    verts, faces=mesh
    tri=verts[faces]
    area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*0.5
    prob=area/area.sum()
    fi=rng.choice(len(faces),size=count,p=prob)
    r=rng.random((count,2)); swap=(r[:,0]+r[:,1])>1.0
    r[swap]=1.0-r[swap]
    t=tri[fi]
    return t[:,0]+r[:,0,None]*(t[:,1]-t[:,0])+r[:,1,None]*(t[:,2]-t[:,0])

def vertices(mesh):
    return mesh[0] if isinstance(mesh,tuple) else mesh.vertices
def face_count(mesh):
    return len(mesh[1]) if isinstance(mesh,tuple) else len(mesh.faces)

def nearest_distances(query, reference, diag):
    if cKDTree is not None:
        return cKDTree(reference).query(query)[0], 'scipy-cKDTree exact'
    # Dependency-free fallback for this lightweight server. A compact spatial
    # hash makes the audit usable without pulling a Python rendering stack; it
    # is explicitly labelled approximate rather than pretending to be a full
    # mesh-to-mesh Hausdorff calculation.
    lo=np.minimum(query.min(0),reference.min(0))-1e-6
    cell=max(diag/128.0,1e-7)
    keys=np.floor((reference-lo)/cell).astype(np.int32)
    bins={}
    for i,k in enumerate(keys): bins.setdefault((int(k[0]),int(k[1]),int(k[2])),[]).append(i)
    out=np.empty(len(query),np.float32)
    for i,p in enumerate(query):
        k=np.floor((p-lo)/cell).astype(np.int32); best=np.inf
        # Most surface neighbours resolve in the local or immediately-adjacent
        # cells. The expanding radius keeps thin features honest without an
        # O(N²) fallback.
        for radius in range(5):
            candidates=[]
            for dz in range(-radius,radius+1):
                for dy in range(-radius,radius+1):
                    for dx in range(-radius,radius+1):
                        candidates.extend(bins.get((int(k[0]+dx),int(k[1]+dy),int(k[2]+dz)),()))
            if candidates:
                d=reference[np.asarray(candidates)]-p
                best=float(np.sqrt((d*d).sum(1)).min())
                # A point closer than the just-searched cell radius cannot be
                # improved by a farther cell.
                if best <= cell*max(1,radius): break
        if not np.isfinite(best):
            # Extremely separated inputs are a failed alignment diagnostic;
            # retain a finite, conservative estimate rather than silently
            # dropping their contribution.
            d=reference-p
            best=float(np.sqrt((d*d).sum(1)).min())
        out[i]=best
    return out, 'dependency-free spatial-hash approximate'

paths=[x for x in sys.argv[1:] if not x.startswith('--')]
if len(paths)!=2:
    raise SystemExit('usage: _mesh_cmp.py <a.glb|obj> <b.glb|obj> [--flip-z-a]')
a = asmesh(paths[0]); b = asmesh(paths[1])
# Python's retained Pixal reference and the native delivery GLB can use
# opposite viewer-forward conventions. This is an explicit audit alignment,
# never an implicit production mesh mutation. Flip around A's own bbox centre
# so the score distinguishes a coordinate convention from actual shape error.
flip_z_a='--flip-z-a' in sys.argv or os.environ.get('MESH_CMP_FLIP_Z_A')=='1'
if flip_z_a:
    av0=vertices(a)
    av0[:,2]=av0[:,2].min()+av0[:,2].max()-av0[:,2]
N = int(os.environ.get('MESH_CMP_SAMPLES', 300000 if cKDTree is not None else 40000))
if N < 1000:
    raise SystemExit('MESH_CMP_SAMPLES must be at least 1000')
av=vertices(a); bv=vertices(b)
diag = float(np.linalg.norm(av.max(0) - av.min(0)))
rng=np.random.default_rng(20260722)
pa = sample_surface(a,N,rng); pb = sample_surface(b,N,rng)
dab, method = nearest_distances(pa,pb,diag)   # a-samples -> nearest b-sample
dba, _ = nearest_distances(pb,pa,diag)        # b-samples -> nearest a-sample
cham = (dab.mean() + dba.mean()) / 2.0
print(f"A: {len(av)} v / {face_count(a)} f   B: {len(bv)} v / {face_count(b)} f")
print(f"samples={N} nearest={method}")
print(f"alignment={'A z-reflected about its bbox centre' if flip_z_a else 'raw export frames'}")
print(f"bbox diag (A) = {diag:.5f}")
print(f"mean A->B = {dab.mean():.6f} ({100*dab.mean()/diag:.4f}% diag)  p95 {np.percentile(dab,95):.6f}  max {dab.max():.6f}")
print(f"mean B->A = {dba.mean():.6f} ({100*dba.mean()/diag:.4f}% diag)  p95 {np.percentile(dba,95):.6f}  max {dba.max():.6f}")
print(f"symmetric chamfer = {cham:.6f}  ({100*cham/diag:.4f}% of bbox diag)")
