"""Full-width timing + peak GPU memory of the two most expensive trial kinds on ResNet-18 / ImageNet-32:
M2 with GAM (double backward through ResNet-18) and M5 with GAM (proxy meta-training + coordinate-wise LSTM on 11.7 M coordinates)
-> choose PROCS per GPU and estimate the wall time (see README_RESNET18.md)."""
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
n_params = sum(p.numel() for p in C.make_model(cfg).parameters())
t = time.time()
data = C.load_data("imagenet", ood_names=["svhn", "cifar10", "mnist", "fmnist"])
print(f"data loaded in {time.time() - t:.0f}s on {props.name} ({total:.0f} GB); ResNet-18 has {n_params:,} parameters; "
      f"train/val/test = {data.Xb.shape[0]}/{data.Xv.shape[0]}/{data.Xt.shape[0]}", flush=True)
res = {}
for label, fn in (("M1 AdamW   ", lambda: C.run_adamw(50000, cfg, data, 1e-3)),
                  ("M2 GAM     ", lambda: C.run_adamw(50000, cfg, data, 1e-3, "gam", 0.05, 0.05))):
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    out = fn()
    res[label] = (time.time() - t, torch.cuda.max_memory_reserved() / 2 ** 30)
    print(f"{label} one init: {res[label][0]:.0f}s (train loop {out['seconds']:.0f}s)  val_acc={out['val_acc']:.4f} "
          f"id_auroc={out['id_auroc']:.3f}  peak reserved {res[label][1]:.1f} GB", flush=True)
torch.cuda.reset_peak_memory_stats()
t = time.time()
lo, info = C.meta_train("gam", 0.01, 0.1, 0, cfg, data, Path("/tmp/gpucheck_lo.pt"), meta_lr=1e-3, rho=0.04)
t_meta = time.time() - t
t = time.time()
out = C.run_lo(lo, 50000, cfg, data)
t_eval = time.time() - t
peak = torch.cuda.max_memory_reserved() / 2 ** 30
print(f"M5 GAM trial: proxy meta-training {t_meta:.0f}s, ResNet-18 eval of ONE init {t_eval:.0f}s (x{cfg.lo_inits} inits), "
      f"val_acc={out['val_acc']:.4f} id_auroc={out['id_auroc']:.3f}; peak reserved {peak:.1f} GB", flush=True)
worst = max(peak, *(v[1] for v in res.values()))
print(f"peak reserved {worst:.1f} GB of {total:.0f} GB  ->  PROCS <= {max(int(total * 0.9 // worst), 1)} worker processes per GPU", flush=True)
