"""Exit 3 if the GPU this job sees has less than MIN_FREE_GB free (another process - e.g. a stale one on the node - sits on it)."""
import os
import socket
import sys

import torch

need = float(os.environ.get("MIN_FREE_GB", "50"))
if not torch.cuda.is_available():
    print("GPU GUARD: no CUDA device visible", flush=True)
    sys.exit(3)
free, total = (x / 2 ** 30 for x in torch.cuda.mem_get_info(0))
host = socket.gethostname()
print(f"GPU GUARD: {host} cuda:0 {torch.cuda.get_device_name(0)} free {free:.1f} of {total:.1f} GB (need >= {need:.0f})", flush=True)
if free < need:
    print(f"GPU GUARD FAILED: only {free:.1f} GB free on {host}.  Resubmit with  EXCLUDE=<nodes>,{host}  bash slurm/submit_rudra.sh ...", flush=True)
    sys.exit(3)
