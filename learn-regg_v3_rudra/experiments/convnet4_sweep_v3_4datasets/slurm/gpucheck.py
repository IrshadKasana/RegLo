"""Full-width timing + peak GPU memory of the most expensive trial kind (M5 with GAM) -> choose PROCS per GPU."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

import common as C

assert torch.cuda.is_available(), "no CUDA device visible"
props = torch.cuda.get_device_properties(0)
total = props.total_memory / 2 ** 30
cfg = C.Cfg()
t = time.time()
data = C.load_data("cifar10", ood_names=["svhn", "mnist", "fmnist"])
print(f"data loaded in {time.time() - t:.0f}s on {props.name} ({total:.0f} GB)", flush=True)
torch.cuda.reset_peak_memory_stats()
t = time.time()
lo, info = C.meta_train("gam", 0.01, 0.1, 0, cfg, data, Path("/tmp/gpucheck_lo.pt"), meta_lr=1e-3, rho=0.04)
t_meta = time.time() - t
t = time.time()
out = C.run_lo(lo, 50000, cfg, data)
t_eval = time.time() - t
peak = torch.cuda.max_memory_allocated() / 2 ** 30
res = torch.cuda.max_memory_reserved() / 2 ** 30
print(f"M5-GAM trial: meta-training {t_meta:.0f}s, one eval init {t_eval:.0f}s (x3 inits), val_acc={out['val_acc']:.3f} "
      f"id_auroc={out['id_auroc']:.3f}", flush=True)
print(f"peak allocated {peak:.1f} GB, peak reserved {res:.1f} GB of {total:.0f} GB  ->  "
      f"PROCS <= {max(int(total * 0.9 // res), 1)} worker processes per GPU", flush=True)
