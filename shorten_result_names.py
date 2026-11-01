#!/usr/bin/env python3
"""
shorten_result_names.py — migrate legacy over-long result filenames.

WHY: Windows' MAX_PATH is 260 chars for the WHOLE path. Old result files
were named like

  evaluation_learned_optimizer_eps0005_meta_optim_mlp_mnist_from-mnist_
  SWEEP_sw_sam_sl0p5_rho0p005_lr0p1_mnist_mlp_sw_sam_sl0p5_rho0p005_lr0p1_
  20260714_051833.json                                     (~150 chars)

which, under a OneDrive/Desktop prefix, blows past 260 and makes Explorer
fail with "Error 0x80010135: Path too long" when extracting the zip.

The NAME GENERATOR is already fixed (optimizee_train._short_run_name ->
'ev_lo_SAMLO_exp2_sam_mnist_mlp_260715-172419.json'); this tool retro-fits
files produced BEFORE that fix.

SAFE: every consumer (generate_paper_plots, pick_best_lambdas,
generate_experiment_report) selects runs by METADATA — experiment_name,
eval_tag, lo_regularizer — never by filename. Renaming cannot break them.

Usage:
    python shorten_result_names.py --dry-run     # show what would change
    python shorten_result_names.py               # apply
    python shorten_result_names.py --limit 80    # stricter budget
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCAN_DIRS = [ROOT / "artifacts", ROOT / "results"]
TS_RE = re.compile(r"(\d{6,8}[-_]\d{6})")


def short_name(kind, ident, tag, dataset, arch, limit=80):
    """Same compact scheme as optimizee_train._short_run_name."""
    ident, tag = ident or "", tag or ""
    if tag and tag in ident:                       # SWEEP_<tag>_<ds>_<arch>
        ident = ""
    suffix = f"_{dataset}_{arch}"
    if ident.endswith(suffix):                     # SAMLO_mnist_mlp -> SAMLO
        ident = ident[: -len(suffix)]
    base = "_".join(b for b in ("ev", kind, ident, tag, dataset, arch) if b)
    if len(base) > limit:
        base = base[: limit - 8] + "_" + hashlib.md5(base.encode()).hexdigest()[:6]
    return base


def new_name_for(path, limit):
    try:
        meta = (json.loads(path.read_text()) or {}).get("metadata") or {}
    except Exception:
        return None
    kind = "bench" if meta.get("if_normal_optim") else "lo"
    ident = meta.get("saved_model") or meta.get("model_name") or ""
    tag = meta.get("eval_tag") or ""
    ds = meta.get("test_dataset") or meta.get("dataset") or ""
    arch = meta.get("test_architecture") or meta.get("optimizee_model") or ""
    m = TS_RE.search(path.stem)
    ts = m.group(1) if m else ""
    if len(ts) == 15:                              # 20260714_051833 -> 260714-051833
        ts = ts[2:8] + "-" + ts[9:]
    base = short_name(kind, ident, tag, ds, arch, limit=limit)
    return f"{base}{('_' + ts) if ts else ''}.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=80,
                    help="max characters for the descriptive part")
    ap.add_argument("--threshold", type=int, default=95,
                    help="only rename files whose current name exceeds this")
    cli = ap.parse_args()

    renamed = skipped = 0
    for d in SCAN_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.rglob("*.json")):
            if len(f.name) <= cli.threshold:
                continue
            new = new_name_for(f, cli.limit)
            if not new or new == f.name:
                skipped += 1
                continue
            target = f.with_name(new)
            n = 1
            while target.exists() and target != f:
                target = f.with_name(new.replace(".json", f"_{n}.json"))
                n += 1
            print(f"{len(f.name):4d} -> {len(target.name):3d}  {f.name}\n"
                  f"                {target.name}")
            if not cli.dry_run:
                f.rename(target)
            renamed += 1

    longest = max((len(str(p.relative_to(ROOT)))
                   for d in SCAN_DIRS if d.exists()
                   for p in d.rglob("*") if p.is_file()), default=0)
    print(f"\n{'would rename' if cli.dry_run else 'renamed'}: {renamed} "
          f"| already fine: {skipped}")
    print(f"longest remaining relative path: {longest} chars "
          f"(Windows budget: 260 minus your folder prefix)")


if __name__ == "__main__":
    main()
