"""Measure meta-training speed and peak GPU memory of the heaviest units of each case, with the exact
production settings (the case arguments of rudra_jobs.py), and project the meta-training hours.

    python experiments/pipeline/rudra/gpu_check.py [case ...]      (run inside a GPU job: gpu_check.sbatch)

Projection: a unit meta-trains 20 optimizees per outer iteration; the curriculum spends 2 outer iterations
at unroll 100, 200 and 500 and then K outer iterations at unroll 1000 until the eval loss stops improving.
K is unknown in advance; the table shows K = 5 / 10 / 20 (inner steps 132k / 232k / 432k).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import rudra_jobs  # noqa: E402
from learning_regularizers.pipeline.cli import parse_args  # noqa: E402
from learning_regularizers.pipeline.config import load_pipeline_config  # noqa: E402
from learning_regularizers.pipeline.meta_train import MetaTrainer  # noqa: E402
from learning_regularizers.pipeline.methods import RunKey  # noqa: E402

UNITS = [RunKey(3, None, None), RunKey(4, None, None), RunKey(5, None, "gam"), RunKey(6, None, "sam"),
         RunKey(6, None, "gam")]
U = 100


def cfg_for(case: str):
    a = parse_args(rudra_jobs.case_args(case))
    sets = [f"meta_training.optimizee={a.meta_optimizee}", f"meta_training.dataset={a.meta_dataset}"] + list(a.set)
    sets = [s if not s.startswith("experiment.name=") else f"experiment.name=gpucheck_{case}" for s in sets]
    sets += ["meta_training.overrides.train_period=1", "meta_training.overrides.eval_epochs=1"]
    return load_pipeline_config(sets=sets)


def main(cases):
    dev = torch.device("cuda:0")
    print(f"GPU {torch.cuda.get_device_name(0)}  torch {torch.__version__}", flush=True)
    rows = []
    for case in cases:
        cfg = cfg_for(case)
        for key in UNITS:
            torch.cuda.empty_cache()
            tr = MetaTrainer(cfg, key, dev, log=lambda m: print("   ", m, flush=True))
            tr.train_period(2 * tr.args.truncated_bptt_step)       # warm-up
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            tr.train_period(U)
            torch.cuda.synchronize()
            ms = (time.time() - t) / U * 1e3
            t = time.time()
            tr.evaluate(U)
            torch.cuda.synchronize()
            ms_eval = (time.time() - t) / U * 1e3
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            est = rudra_jobs.gpu_gb(case, key.run_id)
            hours = [(steps * ms + steps / 4 * ms_eval) / 3.6e6 for steps in (132e3, 232e3, 432e3)]
            rows.append((case, key.run_id, ms, ms_eval, peak, est, hours))
            print(f"{case:18s} {key.run_id:12s} train {ms:7.1f} ms/step  eval {ms_eval:6.1f} ms/step  "
                  f"peak {peak:5.1f} GB (pool estimate {est:4.1f})  meta-training h for K=5/10/20: "
                  + " / ".join(f"{h:.0f}" for h in hours), flush=True)
            del tr
    print("\nSUMMARY (single process on an otherwise idle A100; several processes per GPU run somewhat slower)")
    for case, rid, ms, mse, peak, est, hours in rows:
        flag = "  <-- raise MEM_SCALE" if peak + 1.5 > est else ""
        print(f"  {case:18s} {rid:12s} {ms:7.1f} ms/step  peak {peak:5.1f} GB vs estimate {est:5.1f}{flag}   "
              f"hours K=5/10/20: {hours[0]:.0f}/{hours[1]:.0f}/{hours[2]:.0f}")
    print("GPUCHECK_OK")


if __name__ == "__main__":
    main(sys.argv[1:] or list(rudra_jobs.CASES))
