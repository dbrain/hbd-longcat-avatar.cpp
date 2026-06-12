#!/usr/bin/env python
# Bake a web-ready GLB from the geometry .ply: vertex normals (smooth shading) +
# a double-sided PBR material (the O-Voxel dual-grid mesh has inconsistent winding,
# so backface culling would punch holes). Optionally a decimated variant for fast web load.
import sys, numpy as np, trimesh

src = sys.argv[1] if len(sys.argv) > 1 else "miku_geometry_e2e.ply"
out = sys.argv[2] if len(sys.argv) > 2 else "miku_web.glb"

m = trimesh.load(src, process=False)
print("loaded", src, len(m.vertices), "verts", len(m.faces), "faces")
# trimesh computes smooth (area-weighted) vertex normals on demand
_ = m.vertex_normals
m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(
    name="geom", baseColorFactor=[190, 196, 205, 255],
    metallicFactor=0.05, roughnessFactor=0.75, doubleSided=True))
m.export(out)
print("wrote", out)

# decimated web variant (faster load) if requested
if len(sys.argv) > 3:
    target = int(sys.argv[3])
    d = m.simplify_quadric_decimation(target) if hasattr(m, "simplify_quadric_decimation") else m
    d.visual = m.visual
    dout = out.replace(".glb", f"_{target}.glb")
    d.export(dout)
    print("wrote", dout, len(d.faces), "faces")
