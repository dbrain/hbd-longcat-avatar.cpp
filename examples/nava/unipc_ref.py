#!/usr/bin/env python3
# Generate a fixed-seed PyTorch reference trajectory for FlowUniPCMultistepScheduler
# and dump raw float32 .bin files for the C++ validation harness.
import sys
import os

sys.path.insert(0, '/home/dbrain/dev/NAVA')

import numpy as np
import torch

from nava_src.models.nava.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/unipc_ref'
os.makedirs(OUT, exist_ok=True)

N = 10
D = 64

sched = FlowUniPCMultistepScheduler(
    num_train_timesteps=1000,
    shift=5.0,
    use_dynamic_shifting=False,
)
sched.set_timesteps(N)

print("sigmas:", sched.sigmas.numpy().tolist())
print("timesteps:", sched.timesteps.numpy().tolist())

# Fixed seeded inputs
init = np.random.RandomState(0).randn(D).astype(np.float32).reshape(1, D)
sample = torch.from_numpy(init.copy())

model_outputs = np.zeros((N, D), dtype=np.float32)
for k in range(N):
    model_outputs[k] = np.random.RandomState(100 + k).randn(D).astype(np.float32)

traj = np.zeros((N, D), dtype=np.float32)
for k, t in enumerate(sched.timesteps):
    mo = torch.from_numpy(model_outputs[k].copy()).reshape(1, D)
    out = sched.step(mo, t, sample, return_dict=False)[0]
    sample = out
    traj[k] = sample.numpy().reshape(D)

# Save raw float32 .bin
init.astype(np.float32).tofile(os.path.join(OUT, 'init.bin'))
model_outputs.astype(np.float32).tofile(os.path.join(OUT, 'model_outputs.bin'))
traj.astype(np.float32).tofile(os.path.join(OUT, 'traj.bin'))

with open(os.path.join(OUT, 'meta.txt'), 'w') as f:
    f.write(f"N={N}\nD={D}\n")

print("final step head:", traj[-1][:8].tolist())
print("any nan:", bool(np.isnan(traj).any()), "any inf:", bool(np.isinf(traj).any()))
print("wrote to", OUT)
