#!/usr/bin/env python
"""Sanity-check a sweep trial_table/ directory: did every trial run its full steps, inits and curve points?
usage: python check_trial_table.py <trial_table_dir> [<other_trial_table_dir_to_compare>]"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)


def load(d):
    d = Path(d)
    return (pd.read_csv(d / "trials_full.csv"), pd.read_csv(d / "trials_per_init.csv"),
            pd.read_csv(d / "trials_curves.csv"))


def check(name, d):
    full, per_init, curves = load(d)
    print(f"\n==================== {name}: {d}")
    print(f"trials={len(full)}  per-init rows={len(per_init)}  curve points={len(curves)}")
    print("\n-- status counts\n", full.status.value_counts().to_string())

    ok = full[full.status.isin(["PASSED", "PASSED_AFTER_RETRY"])]
    print(f"\n-- passed trials: {len(ok)}")
    print("optimizee_steps values :", ok.optimizee_steps.value_counts().to_dict())
    print("n_inits values         :", ok.n_inits.value_counts().to_dict())
    print("diverged_inits values  :", ok.diverged_inits.value_counts().to_dict())

    # per-init rows: every passed trial has n_inits rows
    cnt = per_init.groupby("trial").size()
    bad = ok[~ok.trial.isin(cnt[cnt == 3].index)]
    print(f"trials without exactly 3 per-init rows: {len(bad)}")

    # curves: every trial x init has steps 0,100,...,1000
    want = list(range(0, 1001, 100))
    g = curves.groupby(["trial", "init"]).step.apply(lambda s: sorted(s.tolist()))
    short = g[g.apply(lambda s: s != want)]
    print(f"trial x init curves that are NOT steps 0..1000 by 100: {len(short)} of {len(g)}")
    if len(short):
        print(short.head(10).to_string())
    miss = set(ok.trial) - set(curves.trial)
    print(f"passed trials with no curve at all: {len(miss)}")

    # last curve point must be step 1000 and must have finite val metrics
    last = curves[curves.step == 1000]
    print("curve step=1000 rows with NaN val_loss/val_acc:", int(last[["val_loss", "val_acc"]].isna().any(axis=1).sum()))
    # loss must actually go down from step 0 to 1000
    s0 = curves[curves.step == 0].set_index(["trial", "init"]).val_loss
    s1 = last.set_index(["trial", "init"]).val_loss
    drop = (s0 - s1).dropna()
    print(f"val_loss improved step0->1000 in {(drop > 0).mean() * 100:.1f}% of trial x init curves (median drop {drop.median():.3f})")

    # headline metrics finite
    cols = ["train_loss", "val_acc", "test_acc", "id_auroc"]
    print("passed trials with NaN in", cols, ":", int(ok[cols].isna().any(axis=1).sum()))
    print("weights_exists False  :", int((ok.weights_exists.astype(str) != "True").sum()))

    # run time per phase (short is OK for M1/M2: 1000 AdamW steps; M3-M5 include meta-training)
    t = ok.groupby(["phase", "method"]).agg(n=("trial", "size"), run_s_median=("run_seconds", "median"),
                                             wall_s_median=("wall_seconds", "median"),
                                             wall_s_min=("wall_seconds", "min"), wall_s_max=("wall_seconds", "max"))
    print("\n-- seconds per trial by phase\n", t.round(1).to_string())

    lo = ok[ok.method.isin(["M3", "M4a", "M5"])]
    if len(lo):
        print("\n-- learned-optimizer trials (meta-training done?)")
        print("trials:", len(lo))
        print("meta_train_stages value counts:", lo.meta_train_stages.value_counts(dropna=False).to_dict())
        print("meta_unroll_steps_total  min/median/max:", lo.meta_unroll_steps_total.min(),
              lo.meta_unroll_steps_total.median(), lo.meta_unroll_steps_total.max())
        print("meta_train_seconds       min/median/max:", lo.meta_train_seconds.min().round(1),
              lo.meta_train_seconds.median().round(1), lo.meta_train_seconds.max().round(1))
        print("checkpoint_exists False :", int((lo.checkpoint_exists.astype(str) != "True").sum()))
    return full


def compare(a, b):
    m = a.merge(b, on="trial", suffixes=("_a", "_b"))
    print(f"\n==================== COMPARE: {len(m)} trials in both (a={len(a)}, b={len(b)})")
    for c in ["val_acc", "test_acc", "id_auroc", "run_seconds", "meta_unroll_steps_total"]:
        if c + "_a" in m:
            d = (m[c + "_a"] - m[c + "_b"]).abs().dropna()
            print(f"{c:26s} median |diff| = {d.median():.4g}   mean(a)={m[c + '_a'].mean():.4g}  mean(b)={m[c + '_b'].mean():.4g}")


if __name__ == "__main__":
    fa = check("A", sys.argv[1])
    if len(sys.argv) > 2:
        fb = check("B", sys.argv[2])
        compare(fa, fb)
