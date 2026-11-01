"""compare_runs.py — reproducibility check between two full runs (default: run1 vs fresh).
    python compare_runs.py [results/run1_full] [results/full]"""
import json, sys
from pathlib import Path
a = Path(sys.argv[1] if len(sys.argv) > 1 else "results/run1_full")
b = Path(sys.argv[2] if len(sys.argv) > 2 else "results/full")
ra, rb = json.load(open(a / "results.json")), json.load(open(b / "results.json"))
print(f"{'arm':12s} {'run A test acc (per seed)':34s} {'run B test acc (per seed)':34s} mean A / mean B")
for k in ra["arms"]:
    pa = [x * 100 for x in ra["arms"][k]["test_acc"]["per_seed"]]
    pb = [x * 100 for x in rb["arms"][k]["test_acc"]["per_seed"]]
    fmt = lambda p: ", ".join(f"{v:.2f}" for v in p)
    print(f"{k:12s} {fmt(pa):34s} {fmt(pb):34s} {sum(pa)/len(pa):.2f} / {sum(pb)/len(pb):.2f}")
print("\nselected hyper-parameters:")
for k in ra["selection"]:
    if k.endswith("_grid"): continue
    print(f"  {k:11s} A={ra['selection'][k].get('lr', '-')}/{ra['selection'][k].get('lam')}   "
          f"B={rb['selection'][k].get('lr', '-')}/{rb['selection'][k].get('lam')}")
