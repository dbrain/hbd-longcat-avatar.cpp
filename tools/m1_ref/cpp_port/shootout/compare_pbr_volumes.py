#!/usr/bin/env python3
"""Coordinate-keyed raw PBR comparison; ordering never affects the metrics."""
import argparse, json
from pathlib import Path
import numpy as np

def keyed(a):
    a=np.ascontiguousarray(a.astype("<i4",copy=False)); return a.view(np.dtype((np.void,16))).reshape(-1)

ap=argparse.ArgumentParser()
ap.add_argument("--native",required=True,type=Path); ap.add_argument("--python",required=True,type=Path); ap.add_argument("--out",required=True,type=Path)
a=ap.parse_args()
nc=np.load(a.native/"native_pbr_coords.npy").astype(np.int32); nf=np.load(a.native/"native_pbr_feats.npy").astype(np.float32)
pc=np.load(a.python/"native_pbr_coords.npy").astype(np.int32); pf=np.load(a.python/"native_pbr_feats.npy").astype(np.float32)
_,ni,pi=np.intersect1d(keyed(nc),keyed(pc),return_indices=True)
x=nf[ni].astype(np.float64); y=pf[pi].astype(np.float64); d=x-y
report={"native_voxels":len(nc),"python_voxels":len(pc),"common_voxels":len(ni),"native_coverage":len(ni)/len(nc),"python_coverage":len(ni)/len(pc)}
if len(ni): report.update({"mae":float(np.mean(np.abs(d))),"rmse":float(np.sqrt(np.mean(d*d))),"max_abs":float(np.max(np.abs(d))),"cosine":float(np.sum(x*y)/(np.linalg.norm(x)*np.linalg.norm(y)+1e-30)),"per_channel_mae":[float(v) for v in np.mean(np.abs(d),axis=0)]})
a.out.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n"); print(json.dumps(report,sort_keys=True))
