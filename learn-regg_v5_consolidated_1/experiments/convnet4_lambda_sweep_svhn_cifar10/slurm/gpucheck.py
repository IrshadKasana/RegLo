"""GPU sanity + timing check (run via slurm/gpu_check.sbatch).  Prints GPUCHECK_OK at the end."""
import sys, time
sys.path.insert(0, ".")
import torch, common as C
assert torch.cuda.is_available(), "CUDA not visible - wrong module/env? (see slurm/setup_env.sh)"
print("GPU:", torch.cuda.get_device_name(0), "| mem GB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
      "| torch", torch.__version__, "| device", C.device, flush=True)
cfg = C.Cfg(eval_steps=100)
d = C.load_data("svhn")
print("svhn tensors on", d.Xa.device, flush=True)
sync = torch.cuda.synchronize
for kind, lam in [("none", 0), ("l1", 0.001), ("l2", 0.01), ("sam", 0.1), ("gsam", 0.1), ("gam", 0.1)]:
    sync(); t = time.time(); r = C.run_adamw(1, cfg, d, 1e-3, kind, lam); sync(); dt = time.time() - t
    print(f"M1/M2 {kind:5s} w=64: {dt/100*1000:6.1f} ms/step | 1000 steps = {dt*10/60:5.2f} min | val={r['val_acc']:.3f}", flush=True)
cfg = C.Cfg(stages=(40,), train_period=1, eval_steps=100)
for kind, ls, lr in [("none", 0, 0), ("none", 0.1, 0), ("l1", 0, 0.001), ("l2", 0, 0.01), ("sam", 0, 0.1), ("gsam", 0, 0.1), ("gam", 0, 0.1)]:
    sync(); t = time.time()
    torch.cuda.reset_peak_memory_stats()
    lo, _ = C.meta_train(kind, ls, lr, 1, cfg, d, "/tmp/gpucheck_lo.pt", log=lambda s: None)
    sync(); dt = time.time() - t
    print(f"M3/M4 {kind:5s} ls={ls} lr={lr}: {dt/40*1000:6.0f} ms per unrolled step (incl. 100-step ckpt eval) "
          f"| peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
print("GPUCHECK_OK")
