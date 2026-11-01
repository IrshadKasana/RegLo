"""
Aggregate sweep results into the selection report and the final table.

Selection rule:
    1. group runs by (arm, hyperparameters), averaging over seeds
    2. pick the setting with the highest MEAN VALIDATION accuracy
    3. report that setting's TEST accuracy, mean +/- std over seeds
    4. paired t-test against the control arm, pairing on seed

Step 3 is where papers in this area most often cheat, by quietly reporting the
best test number instead of the test number of the val-selected setting.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

HP_KEYS = ["lr", "rho", "alpha", "gamma", "lam", "decouple", "adaptive",
           "grad_rho", "grad_norm_rho", "grad_gamma", "m_sharpness",
           "weight_decay"]


# --------------------------------------------------------------------------- #
def load(patterns) -> list:
    """
    Load JSONL results. Patterns may contain wildcards.

    Globs are expanded here rather than by the shell, because PowerShell does
    not expand wildcards for external commands.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    paths = []
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            paths.extend(hits)
        elif os.path.exists(pat):
            paths.append(pat)
        else:
            print(f"[warn] no files matched {pat!r}")
    if not paths:
        raise SystemExit("no result files matched")
    print(f"[analyze] reading {len(paths)} file(s)")

    rows, bad = [], 0
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    bad += 1        # truncated last line after a killed job
    if bad:
        print(f"[analyze] skipped {bad} malformed line(s)")
    return rows


def hp_id(c: dict):
    return tuple((k, c.get(k)) for k in HP_KEYS if c.get(k) is not None)


def mean_std(xs):
    xs = [x for x in xs if x == x]           # drop NaN
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


# ---- Student t, no scipy dependency (verified to 5 dp vs tables) ----------- #
def _betacf(a, b, x, itmax=200, eps=3e-14):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = c if abs(c) > 1e-30 else 1e-30
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        c = 1.0 + aa / c
        d = 1.0 / (d if abs(d) > 1e-30 else 1e-30)
        c = c if abs(c) > 1e-30 else 1e-30
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_pvalue(t: float, df: int) -> float:
    """Two-sided p-value for Student's t."""
    return _betainc(df / 2.0, 0.5, df / (df + t * t))


def paired_t(a: dict, b: dict):
    """a, b: seed -> value. Returns (mean_diff, p_two_sided, n_pairs)."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    if n < 2:
        return (d[0] if d else float("nan")), float("nan"), n
    m, sd = mean_std(d)
    if sd == 0:
        return m, (0.0 if m != 0 else 1.0), n
    return m, t_pvalue(m / (sd / math.sqrt(n)), n - 1), n


def t_ppf(q: float, df: int) -> float:
    """Inverse Student-t CDF by bisection on the two-sided p-value."""
    if q <= 0.5:
        return -t_ppf(1 - q, df)
    target = 2 * (1 - q)
    lo, hi = 0.0, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_pvalue(mid, df) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def min_detectable_effect(a: dict, b: dict, alpha: float = 0.05,
                          power: float = 0.80):
    """
    Smallest paired difference this experiment could have detected.

    MDE = (t_{1-alpha/2,n-1} + t_{power,n-1}) * sd_diff / sqrt(n)

    THIS IS THE NUMBER THAT MAKES A NULL RESULT INTERPRETABLE. "p = 0.36" on
    its own says nothing; "p = 0.36, and we could only have detected effects
    above 0.23pp" says the experiment was underpowered for the effect size in
    question. Report both.
    """
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    if n < 2:
        return float("nan"), n
    _, sd = mean_std(d)
    df = n - 1
    return (t_ppf(1 - alpha / 2, df) + t_ppf(power, df)) * sd / math.sqrt(n), n


def _rank_avg(vals):
    """Ranks of |values| with ties averaged."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _wilcoxon_exact_p(w_plus: float, n: int) -> float:
    """Exact two-sided p for the signed-rank statistic (no ties, n <= 25)."""
    max_s = n * (n + 1) // 2
    counts = [0] * (max_s + 1)
    counts[0] = 1
    for r in range(1, n + 1):
        for s in range(max_s, r - 1, -1):
            counts[s] += counts[s - r]
    total = float(2 ** n)
    w = min(w_plus, max_s - w_plus)
    tail = sum(counts[: int(w) + 1]) / total
    return min(1.0, 2 * tail)


def wilcoxon(a: dict, b: dict):
    """
    Wilcoxon signed-rank test on the paired differences.

    Non-parametric complement to the paired t-test. With 10 seeds the t-test's
    normality assumption is not really checkable, so reporting both and noting
    they agree is a cheap way to close off that line of criticism. Returns
    (p_two_sided, n_nonzero).
    """
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    d = [x for x in d if x != 0]
    n = len(d)
    if n < 3:
        return float("nan"), n

    absd = [abs(x) for x in d]
    ranks = _rank_avg(absd)
    w_plus = sum(r for x, r in zip(d, ranks) if x > 0)

    has_ties = len(set(absd)) != len(absd)
    if not has_ties and n <= 25:
        return _wilcoxon_exact_p(w_plus, n), n

    # normal approximation with continuity + tie correction
    mu = n * (n + 1) / 4.0
    tie_term = 0.0
    seen = {}
    for v in absd:
        seen[v] = seen.get(v, 0) + 1
    for c in seen.values():
        if c > 1:
            tie_term += c ** 3 - c
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term / 48.0
    if var <= 0:
        return float("nan"), n
    z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
    return math.erfc(z / math.sqrt(2)), n


def cohens_dz(a: dict, b: dict) -> float:
    """Paired effect size. |dz| >= 0.8 is 'large'."""
    seeds = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in seeds]
    m, sd = mean_std(d)
    return m / sd if sd else float("inf") if m else 0.0


# --------------------------------------------------------------------------- #
def aggregate(rows, select_on="best_val_acc", report="test_acc_at_best_val"):
    cells = defaultdict(dict)
    for r in rows:
        c = r["config"]
        cells[(c.get("arm", c["regularizer"]), hp_id(c))][c["seed"]] = r

    agg = {}
    for (arm, hp), byseed in cells.items():
        vm, vs = mean_std([v[select_on] for v in byseed.values()])
        tm, ts = mean_std([v[report] for v in byseed.values()])
        agg[(arm, hp)] = dict(
            n=len(byseed), val_mean=vm, val_std=vs, test_mean=tm, test_std=ts,
            byseed_test={s: v[report] for s, v in byseed.items()},
            byseed_val={s: v[select_on] for s, v in byseed.items()},
            gap=mean_std([v["generalization_gap"] for v in byseed.values()])[0],
            gradnorm=mean_std([v.get("final_grad_norm_train", float("nan"))
                               for v in byseed.values()])[0],
            samgap=mean_std([v.get("sam_gap_rho0.05", float("nan"))
                             for v in byseed.values()])[0],
            passes=mean_std([v["fwd_bwd_passes"] for v in byseed.values()])[0],
            wall=mean_std([v["wall_seconds"] for v in byseed.values()])[0],
        )
    return agg


def pick_winners(agg):
    winners = {}
    for arm in sorted({a for a, _ in agg}):
        cand = [(hp, d) for (a, hp), d in agg.items() if a == arm]
        cand.sort(key=lambda kv: -kv[1]["val_mean"])
        winners[arm] = cand
    return winners


def boundary_warnings(agg, grid_cfg=None):
    """
    Flag any arm whose selected hyperparameter sits at the edge of its grid.

    A baseline tuned to a grid boundary means the true optimum is outside the
    search range, and the baseline is therefore under-tuned. Reviewers check
    for this, and it is trivially fixable by widening the grid.
    """
    msgs = []
    per_arm_vals = defaultdict(lambda: defaultdict(set))
    for (arm, hp), _ in agg.items():
        for k, v in hp:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                per_arm_vals[arm][k].add(v)

    for arm, cand in pick_winners(agg).items():
        best_hp = dict(cand[0][0])
        for k, v in best_hp.items():
            vals = sorted(per_arm_vals[arm].get(k, []))
            if len(vals) < 2 or not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            if v == vals[0]:
                msgs.append(f"  [{arm}] best {k}={v} is the SMALLEST value "
                            f"in the grid {vals} - extend the grid downward")
            elif v == vals[-1]:
                msgs.append(f"  [{arm}] best {k}={v} is the LARGEST value "
                            f"in the grid {vals} - extend the grid upward")
    return msgs



def _byseed_metric(rows, arm, hp, metric):
    """{seed: metric} for the runs matching an arm + hyperparameter dict."""
    out = {}
    for r in rows:
        c = r["config"]
        if c.get("arm", c["regularizer"]) != arm:
            continue
        if any(c.get(k) != v for k, v in hp.items()):
            continue
        v = r.get(metric)
        if v is not None and v == v:
            out[c["seed"]] = v
    return out


def diverged_report(rows, chance: float = 0.15, report_metric="test_acc_at_best_val"):
    """
    Flag configurations where at least one seed collapsed to chance level.

    A collapsed run is not noise -- it means that hyperparameter combination is
    unstable, and pooling it with healthy runs is what produces nonsense error
    bars (the classic symptom is an accuracy error bar whose upper end exceeds
    100%). Listing them explicitly means you can (a) report the instability as
    a finding and (b) know it is not a silent data-quality problem.
    """
    bad = defaultdict(list)
    for r in rows:
        if r[report_metric] < chance:
            c = r["config"]
            arm = c.get("arm", c["regularizer"])
            hp = tuple((k, c.get(k)) for k in HP_KEYS if c.get(k) is not None)
            bad[(arm, hp)].append((c["seed"], r[report_metric] * 100))

    if not bad:
        print(f"\nDIVERGENCE CHECK: no run collapsed below {chance*100:.0f}% "
              f"accuracy. Clean.")
        return {}

    print("\n" + "!" * 72)
    print(f"DIVERGED RUNS (test accuracy < {chance*100:.0f}%, i.e. at or near "
          f"chance)")
    print("These configurations are UNSTABLE, not noisy. Do not pool them with")
    print("healthy runs -- report them as a stability boundary instead.")
    print("!" * 72)
    for (arm, hp), seeds in sorted(bad.items()):
        hs = "  ".join(f"{k}={v}" for k, v in hp)
        ss = ", ".join(f"seed {s}: {a:.1f}%" for s, a in sorted(seeds))
        print(f"  [{arm}] {hs}")
        print(f"         {len(seeds)} run(s) collapsed -- {ss}")
    return bad


def _pick_metric(rows, requested):
    """
    Fall back to validation accuracy when the sweep never touched the test set.

    A tuning study should not read the test split at all, so `eval_test=False`
    leaves `test_acc_at_best_val` as NaN. Reporting would silently produce NaN
    tables; instead we detect that and report validation accuracy, stating so.
    """
    if requested != "test_acc_at_best_val":
        return requested, False
    vals = [r.get(requested) for r in rows]
    have = [v for v in vals if v is not None and v == v]
    if have:
        return requested, False
    print("[analyze] test set was never evaluated (eval_test=False) -- "
          "reporting VALIDATION accuracy instead.")
    return "best_val_acc", True


def report(patterns, select_on="best_val_acc", report_metric="test_acc_at_best_val",
           control="adamw", top_k=3, emit_winners=None, latex=None, csv=None):
    rows = load(patterns)
    report_metric, _val_only = _pick_metric(rows, report_metric)
    diverged_report(rows, report_metric=report_metric)
    agg = aggregate(rows, select_on, report_metric)
    ranked = pick_winners(agg)
    arms = sorted(ranked)

    for arm in arms:
        cand = ranked[arm]
        print(f"\n=== {arm} - top {top_k} by mean validation accuracy "
              f"({cand[0][1]['n']} seeds) ===")
        for hp, d in cand[:top_k]:
            hs = "  ".join(f"{k}={v}" for k, v in hp)
            print(f"  val {d['val_mean']*100:6.2f}+-{d['val_std']*100:4.2f}   "
                  f"test {d['test_mean']*100:6.2f}+-{d['test_std']*100:4.2f}   {hs}")

    warn = boundary_warnings(agg)
    if warn:
        print("\n" + "!" * 72)
        print("GRID BOUNDARY WARNINGS - the search range is too narrow:")
        print("\n".join(warn))
        print("!" * 72)

    ctrl_arm = control if control in ranked else arms[0]
    ctrl = ranked[ctrl_arm][0][1]["byseed_test"]

    print("\n" + "=" * 104)
    _hdr = "val acc (%)" if report_metric == "best_val_acc" else "test acc (%)"
    print(f"{'arm':8s} {_hdr:>16s} {'delta':>8s} {'p':>9s} {'dz':>7s} "
          f"{'gen gap':>9s} {'|grad|':>9s} {'SAM gap':>9s} {'passes':>11s}")
    print("=" * 104)
    table, npairs = [], 0
    for arm in arms:
        hp, d = ranked[arm][0]
        diff, p, npairs = paired_t(d["byseed_test"], ctrl)
        dz = cohens_dz(d["byseed_test"], ctrl)
        star = "" if arm == ctrl_arm else ("***" if p < 0.001 else "**" if p < 0.01
                                           else "*" if p < 0.05 else "")
        print(f"{arm:8s} {d['test_mean']*100:9.2f} +-{d['test_std']*100:5.2f} "
              f"{diff*100:+8.2f} {p:9.4f}{star:<3s} {dz:6.2f} "
              f"{d['gap']*100:9.2f} {d['gradnorm']:9.4f} {d['samgap']:9.4f} "
              f"{d['passes']:11.0f}")
        table.append((arm, hp, d, diff, p, dz))
    print("=" * 104)
    print(f"paired over {npairs} seeds vs '{ctrl_arm}';  * p<.05  ** p<.01  *** p<.001")
    print(f"dz = Cohen's d for paired samples; |dz| > 0.8 is a large effect")

    # ---- power analysis: what a null result actually means ---------------- #
    print("\n" + "-" * 104)
    print("POWER: smallest paired difference this experiment could detect "
          "(alpha=.05, 80% power)")
    print("-" * 104)
    print(f"{'arm':8s} {'observed':>10s} {'MDE':>10s} {'Wilcoxon p':>12s} "
          f"{'verdict':>34s}")
    for arm in arms:
        if arm == ctrl_arm:
            continue
        d = ranked[arm][0][1]
        diff, p, _ = paired_t(d["byseed_test"], ctrl)
        mde, n = min_detectable_effect(d["byseed_test"], ctrl)
        wp, _ = wilcoxon(d["byseed_test"], ctrl)
        if p < 0.05:
            verdict = "significant"
        elif abs(diff) < mde:
            verdict = "UNDERPOWERED - cannot conclude"
        else:
            verdict = "no effect at this sample size"
        print(f"{arm:8s} {diff*100:+9.2f}pp {mde*100:9.2f}pp {wp:12.4f} "
              f"{verdict:>34s}")
    print("\n  'UNDERPOWERED' means the observed difference is smaller than what")
    print("  n seeds could resolve. It is NOT evidence of no effect -- report it")
    print("  as inconclusive, or add seeds until MDE drops below the effect.")

    # ---- secondary metrics: often where the real result is ---------------- #
    print("\n" + "=" * 104)
    print("SECONDARY METRICS (paired vs control) - report these when accuracy "
          "is null")
    print("=" * 104)
    print(f"{'arm':8s} {'gen gap (pp)':>22s} {'p':>9s} "
          f"{'SAM gap':>20s} {'ratio':>8s} {'p':>9s}")
    for arm in arms:
        d = ranked[arm][0][1]
        gap_by = {s: r for s, r in _byseed_metric(rows, arm,
                  dict(ranked[arm][0][0]), "generalization_gap").items()}
        shp_by = _byseed_metric(rows, arm, dict(ranked[arm][0][0]),
                                "sam_gap_rho0.05")
        cg = _byseed_metric(rows, ctrl_arm, dict(ranked[ctrl_arm][0][0]),
                            "generalization_gap")
        cs = _byseed_metric(rows, ctrl_arm, dict(ranked[ctrl_arm][0][0]),
                            "sam_gap_rho0.05")
        gm, gs = mean_std(list(gap_by.values()))
        sm, ss = mean_std(list(shp_by.values()))
        base_s = mean_std(list(cs.values()))[0]
        if arm == ctrl_arm:
            print(f"{arm:8s} {gm*100:14.2f} +-{gs*100:5.2f} {'--':>9s} "
                  f"{sm:14.5f} +-{ss:5.5f} {'1.0x':>8s} {'--':>9s}")
            continue
        _, pg, _ = paired_t(gap_by, cg)
        _, ps, _ = paired_t(shp_by, cs)
        star_g = "***" if pg < .001 else "**" if pg < .01 else "*" if pg < .05 else ""
        star_s = "***" if ps < .001 else "**" if ps < .01 else "*" if ps < .05 else ""
        ratio = base_s / sm if sm else float("nan")
        print(f"{arm:8s} {gm*100:14.2f} +-{gs*100:5.2f} {pg:9.4f}{star_g:<3s}"
              f"{sm:12.5f} +-{ss:5.5f} {ratio:7.1f}x {ps:9.4f}{star_s}")
    print("=" * 104)

    print("\nSelected hyperparameters (chosen on VALIDATION accuracy):")
    for arm in arms:
        print(f"  {arm:8s} {dict(ranked[arm][0][0])}")

    if emit_winners:
        import yaml
        payload = {"winners": [dict(list(ranked[a][0][0]) + [("arm", a)])
                               for a in arms]}
        os.makedirs(os.path.dirname(emit_winners) or ".", exist_ok=True)
        with open(emit_winners, "w") as f:
            yaml.safe_dump(payload, f, sort_keys=False)
        print(f"\nwrote {emit_winners}")

    if csv:
        os.makedirs(os.path.dirname(csv) or ".", exist_ok=True)
        with open(csv, "w") as f:
            f.write("arm,hyperparameters,n_seeds,val_mean,val_std,test_mean,"
                    "test_std,delta,p_value,cohens_dz,gen_gap,grad_norm,"
                    "sam_gap,passes\n")
            for arm, hp, d, diff, p, dz in table:
                hs = ";".join(f"{k}={v}" for k, v in hp)
                f.write(f"{arm},\"{hs}\",{d['n']},{d['val_mean']:.6f},"
                        f"{d['val_std']:.6f},{d['test_mean']:.6f},"
                        f"{d['test_std']:.6f},{diff:.6f},{p:.6f},{dz:.4f},"
                        f"{d['gap']:.6f},{d['gradnorm']:.6f},"
                        f"{d['samgap']:.6f},{d['passes']:.0f}\n")
        print(f"wrote {csv}")

    if latex:
        os.makedirs(os.path.dirname(latex) or ".", exist_ok=True)
        lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
                 r"Optimizer & Test acc.\ (\%) & $\Delta$ & Gen.\ gap (\%) "
                 r"& SAM gap & Passes \\", r"\midrule"]
        for arm, hp, d, diff, p, dz in table:
            sig = "" if arm == ctrl_arm else (r"$^{***}$" if p < 0.001 else
                                              r"$^{**}$" if p < 0.01 else
                                              r"$^{*}$" if p < 0.05 else "")
            lines.append(
                f"{arm.upper()} & ${d['test_mean']*100:.2f} \\pm "
                f"{d['test_std']*100:.2f}${sig} & ${diff*100:+.2f}$ & "
                f"${d['gap']*100:.2f}$ & ${d['samgap']:.4f}$ & "
                f"${d['passes']/1000:.0f}$k \\\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(latex, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"wrote {latex}")

    return agg, ranked
