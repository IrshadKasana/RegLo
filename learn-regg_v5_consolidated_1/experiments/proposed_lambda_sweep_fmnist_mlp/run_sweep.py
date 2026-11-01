#!/usr/bin/env python
"""
run_sweep.py — sweep of (lambda_smooth, lambda_reg) of the PROPOSED method on
FashionMNIST with an MLP optimizee, over the user-specified grid

    G = {0, 0.001, 0.002, 0.004, 0.006, 0.007, 0.01, 0.02, 0.03, 0.04, 0.05,
         0.07, 0.1, 0.2, 0.3, 0.4, 0.5}      (same G for both lambdas; 0 = term off)

    optimizee (MLP 784-20-10) + learned optimizer (LSTM LO)
    meta-loss = L_task + lambda_smooth * L_smooth + lambda_reg * L_SAM

The full G x G factorial is 289 cells x several meta-seeds, which is ~30 h on this
CPU-only box, so it is searched in stages that use exactly the values of G:

  stage 'axes'  : lambda_smooth over G with lambda_reg = 0, and lambda_reg over G
                  with lambda_smooth = 0   (33 cells; each term in isolation;
                  (0,0) = plain 'optimizee + optimizer' is shared)
  stage 'local' : the full 2-D neighbourhood (+-radius grid steps in G, non-zero
                  values) around the best value found on each axis
  stage 'full'  : every G x G cell (optional, resumable; finished cells are kept)

Each cell is meta-TRAINED on several meta-seeds, then EVALUATED (a fresh MLP is
trained 1000 steps with the frozen LO) on the held-out validation split.
Selection is on validation accuracy; test accuracy is stored, never used.

    python run_sweep.py all [--stage both|axes|local|full] [--smoke]
    python run_sweep.py report
Resumable: a job whose result JSON exists is skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PY = sys.executable
EVAL_SEED0 = 50_000


def g(x):
    return f"{x:g}"


def dirs(smoke):
    root = HERE / "results" / ("smoke" if smoke else "full")
    d = {k: root / k for k in ("lo", "ckpt", "logs", "ref")}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    d["root"] = root
    return d


def make_cfg(smoke, a=None):
    import common as C
    cfg = C.Cfg(**C.SMOKE) if smoke else C.Cfg()
    if a is not None and getattr(a, "train_period", None):
        cfg.train_period = a.train_period
    return cfg


# ---------------------------------------------------------------- jobs ------
def job_ref(a):
    import common as C
    cfg, D = make_cfg(a.smoke, a), dirs(a.smoke)
    data = C.load_fmnist()
    rows = []
    for lr in cfg.adamw_lrs:
        ev = [C.run_adamw(EVAL_SEED0 + i, cfg, data, lr) for i in range(cfg.adamw_inits)]
        rows.append(dict(lr=lr, wd=cfg.adamw_wd, evals=ev))
        print(f"adamw lr={lr:g} val={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)
    json.dump(dict(rows=rows), open(D["ref"] / "adamw.json", "w"))


def job_lo(a):
    import common as C
    cfg, D = make_cfg(a.smoke, a), dirs(a.smoke)
    data = C.load_fmnist()
    tag = f"ls{g(a.ls)}_lr{g(a.lr)}_seed{a.seed}"
    lo, info = C.meta_train(a.ls, a.lr, a.seed, cfg, data, D["ckpt"] / f"{tag}.pt",
                            log=lambda s: print(s, flush=True))
    ev = [C.run_lo(lo, EVAL_SEED0 + i, cfg, data) for i in range(cfg.lo_inits)]
    json.dump(dict(ls=a.ls, lr=a.lr, seed=a.seed, evals=ev, meta_train=info,
                   cfg=asdict(cfg), device=str(C.device)),
              open(D["lo"] / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: val_acc={np.mean([e['val_acc'] for e in ev]):.4f}", flush=True)


# ------------------------------------------------------- orchestration ------
def run_jobs(jobs, workers, logdir, threads):
    todo = [j for j in jobs if not Path(j[2]).exists()]
    print(f"[orchestrator] {len(jobs) - len(todo)} done, {len(todo)} to run, {workers} workers", flush=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))

    def go(j):
        name, argv, _ = j
        t = time.time()
        with open(logdir / f"{name}.log", "w") as fh:
            rc = subprocess.call([PY, str(HERE / "run_sweep.py"), *argv],
                                 stdout=fh, stderr=subprocess.STDOUT, env=env)
        print(f"[orchestrator] {'ok  ' if rc == 0 else 'FAIL'} {name} ({time.time() - t:.0f}s)", flush=True)
        return rc

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rcs = list(ex.map(go, todo))
    if any(rcs):
        print("[orchestrator] some jobs failed - see results/*/logs", flush=True)


def cells_axes(G):
    return [(x, 0.0) for x in G] + [(0.0, x) for x in G if x != 0.0]


def cells_local(G, center, radius):
    nz = [x for x in G if x != 0.0]
    def rng(c):
        i = nz.index(min(nz, key=lambda v: abs(v - c)))
        return nz[max(0, i - radius): i + radius + 1]
    return [(a_, b_) for a_ in rng(center[0]) for b_ in rng(center[1])]


def cells_full(G):
    return [(a_, b_) for a_ in G for b_ in G]


def build_jobs(cells, seeds, D, sm, tp, ref=False):
    jobs = []
    if ref:
        jobs.append(("ref_adamw", ["job-ref", *sm, *tp], D["ref"] / "adamw.json"))
    for s in seeds:                       # seed-major: usable partial results early
        for ls, lr in cells:
            tag = f"ls{g(ls)}_lr{g(lr)}_seed{s}"
            jobs.append((tag, ["job-lo", "--ls", str(ls), "--lr", str(lr), "--seed", str(s), *sm, *tp],
                         D["lo"] / f"{tag}.json"))
    return jobs


def cmd_all(a):
    cfg, D = make_cfg(a.smoke, a), dirs(a.smoke)
    G = list(cfg.grid)
    sm = ["--smoke"] if a.smoke else []
    tp = ["--train-period", str(a.train_period)] if a.train_period else []
    t0 = time.time()
    center = tuple(a.center) if a.center else None

    if a.stage in ("both", "axes"):
        run_jobs(build_jobs(cells_axes(G), a.meta_seeds, D, sm, tp, ref=True), a.workers, D["logs"], a.threads)
        out = cmd_report(a)
        center = center or (out["axis_best"]["lambda_smooth"], out["axis_best"]["lambda_reg"])
        print(f"[orchestrator] axis winners -> centre of stage 'local': {center}", flush=True)
    if a.stage in ("both", "local"):
        if center is None:
            ctr = json.load(open(D["root"] / "best_lambdas.json"))["axis_best"]
            center = (ctr["lambda_smooth"], ctr["lambda_reg"])
        run_jobs(build_jobs(cells_local(G, center, a.radius), a.meta_seeds, D, sm, tp), a.workers, D["logs"], a.threads)
        cmd_report(a)
    if a.stage == "full":
        run_jobs(build_jobs(cells_full(G), a.meta_seeds, D, sm, tp, ref=True), a.workers, D["logs"], a.threads)
        cmd_report(a)
    print(f"[orchestrator] total {(time.time() - t0) / 60:.1f} min", flush=True)


# --------------------------------------------------------------- report -----
def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def _rss(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, float(r @ r), r, int(np.linalg.matrix_rank(X))


def axis_analysis(vals, seeds, G, seed_boot=2000):
    """vals: {lambda: {seed: val_acc}} along one axis (lambda=0 is the ablation level).
    Returns per-level stats, F-test that lambda matters (levels as factor, seed blocks),
    and a quadratic-in-log10(lambda) fit over the NON-zero levels (optimum + bootstrap CI)."""
    from scipy import stats
    rng = np.random.default_rng(0)
    seeds = [s for s in seeds if all(s in vals[l] for l in vals)]
    levels = sorted(vals)
    out = dict(seeds=seeds, levels={}, n_levels=len(levels))
    for l in levels:
        v = np.array([vals[l][s] for s in seeds])
        out["levels"][l] = dict(mean=float(v.mean()), se=se(v), per_seed=v.tolist())
    if len(seeds) < 2 or len(levels) < 3:
        return out
    y = np.array([vals[l][s] for l in levels for s in seeds], float)
    li = np.array([i for i, _ in enumerate(levels) for _ in seeds])
    si = np.array([j for _ in levels for j, _ in enumerate(seeds)])
    n = len(y)
    one = np.ones((n, 1))
    S = np.stack([(si == k).astype(float) for k in range(1, len(seeds))], 1)
    L = np.stack([(li == k).astype(float) for k in range(1, len(levels))], 1)
    _, r0, _, _ = _rss(np.hstack([one, S]), y)
    _, r1, _, p1 = _rss(np.hstack([one, S, L]), y)
    df = n - p1
    F = ((r0 - r1) / (len(levels) - 1)) / (r1 / df)
    out["F"], out["p_any"], out["df"] = float(F), float(1 - stats.f.cdf(F, len(levels) - 1, df)), (len(levels) - 1, df)
    out["resid_sd"] = float(np.sqrt(r1 / df))
    # quadratic over non-zero levels
    nz = [l for l in levels if l > 0]
    if len(nz) >= 4:
        xs = np.log10(np.array([l for l in nz for _ in seeds]))
        yy = np.array([vals[l][s] for l in nz for s in seeds], float)
        gi = np.array([j for _ in nz for j, _ in enumerate(seeds)])
        m = len(yy)
        xc = (xs - xs.mean()) / (xs.max() - xs.min()) * 2
        Sg = np.stack([(gi == k).astype(float) for k in range(len(seeds))], 1)
        X2, X1 = np.c_[Sg, xc, xc ** 2], np.c_[Sg, xc]
        b2, rr2, res2, p2 = _rss(X2, yy)
        _, rr1, _, _ = _rss(X1, yy)
        d2 = m - p2
        Fc = (rr1 - rr2) / (rr2 / d2)
        out["p_curv"] = float(1 - stats.f.cdf(Fc, 1, d2))
        grid = np.linspace(-2, 2, 801)

        def opt(beta):
            f = beta[-2] * grid + beta[-1] * grid ** 2
            i = int(np.argmax(f))
            ok = beta[-1] < 0 and 0 < i < len(grid) - 1
            return grid[i] / 2 * (xs.max() - xs.min()) + xs.mean(), ok
        lx, ok = opt(b2)
        fitted = X2 @ b2
        bs, inter = [], 0
        for _ in range(seed_boot):
            yb = fitted + rng.choice(res2, size=m, replace=True) * np.sqrt(m / d2)
            bb = _rss(X2, yb)[0]
            l2, ok2 = opt(bb)
            bs.append(l2)
            inter += ok2
        bs = np.array(bs)
        out["fit"] = dict(opt=float(10 ** lx), lo=float(10 ** np.percentile(bs, 5)),
                          hi=float(10 ** np.percentile(bs, 95)), interior=bool(ok),
                          p_interior=inter / seed_boot)
    return out


def pick_center(ax):
    """Best non-zero level on an axis: fitted optimum snapped to the grid when the
    curve is significant and interior; otherwise the raw non-zero arg-max."""
    nz = {l: v["mean"] for l, v in ax["levels"].items() if l > 0}
    raw = max(nz, key=nz.get)
    f = ax.get("fit")
    if f and f["interior"] and ax.get("p_curv", 1) < 0.10:
        return min(nz, key=lambda l: abs(np.log10(l) - np.log10(f["opt"]))), "fitted optimum"
    return raw, "raw arg-max"


def cmd_report(a):
    D = dirs(a.smoke)
    cfg = make_cfg(a.smoke, a)
    G = list(cfg.grid)
    val, nll, test, seeds_seen = {}, {}, {}, set()
    for f in sorted(D["lo"].glob("ls*_lr*_seed*.json")):
        j = json.load(open(f))
        k = (j["ls"], j["lr"])
        mean = lambda key: float(np.mean([e[key] for e in j["evals"]]))
        val.setdefault(k, {})[j["seed"]] = 100 * mean("val_acc")
        nll.setdefault(k, {})[j["seed"]] = mean("val_loss")
        test.setdefault(k, {})[j["seed"]] = 100 * mean("test_acc")
        seeds_seen.add(j["seed"])
    if not val:
        print("no results yet")
        return {}
    seeds = sorted(seeds_seen)
    cell = {k: dict(mean=float(np.mean(list(v.values()))), se=se(list(v.values())), n=len(v)) for k, v in val.items()}
    ref = val.get((0.0, 0.0), {})
    out = dict(seeds=seeds)
    md = ["# Proposed method on FMNIST + MLP: λ_smooth × λ_reg sweep\n"]
    md.append("Meta-loss = `L_task + λ_smooth·L_smooth + λ_reg·L_SAM` (repo's consolidated Alg. 1, penalty mode, "
              f"ρ={cfg.rho:g}, ε_smooth={cfg.smooth_eps:g}); 0 switches a term off, so (0, 0) is the plain "
              "'optimizee + optimizer' reference.\n")
    md.append(f"Grid for both λ's: {', '.join(g(x) for x in G)}. Each cell: LO meta-trained on meta-seeds {seeds} "
              "(3 optimizee inits each, averaged), then a fresh MLP is trained 1000 steps with the frozen LO. "
              "**Validation accuracy** (3 000 held-out images) selects; `±` = SE across meta-seeds. "
              "`Δ` = paired difference to the (0, 0) reference, by meta-seed.\n")

    def paired(k):
        v = val.get(k, {})
        c = sorted(set(v) & set(ref))
        if k == (0.0, 0.0) or len(c) < 2:
            return None
        d = np.array([v[s] - ref[s] for s in c])
        return float(d.mean()), se(d), int((d > 0).sum()), len(d)

    # ---- stage 1: the two axes -------------------------------------------------
    axes_out = {}
    for name, key, cells in (("λ_smooth (λ_reg = 0)", "lambda_smooth", [(x, 0.0) for x in G]),
                             ("λ_reg (λ_smooth = 0)", "lambda_reg", [(0.0, x) for x in G])):
        have = [(c, x) for c, x in zip(cells, G) if c in val]
        if len(have) < 3:
            continue
        idx = 0 if key == "lambda_smooth" else 1
        vals = {x: val[c] for c, x in have}
        ax = axis_analysis(vals, seeds, G)
        center, how = pick_center(ax) if len(ax["seeds"]) >= 2 and any(l > 0 for l in ax["levels"]) else (None, "")
        axes_out[key] = dict(ax, center=center, how=how)
        md.append(f"## Axis sweep — {name}\n")
        md.append("| λ | val acc % | ± SE | Δ vs (0,0) pp | val NLL | test acc % |")
        md.append("|---|---|---|---|---|---|")
        bestk = max((c for c, _ in have), key=lambda c: cell[c]["mean"])
        for c, x in have:
            p = paired(c)
            dtxt = f"{p[0]:+.2f} ± {p[1]:.2f} ({p[2]}/{p[3]})" if p else "ref"
            row = f"{cell[c]['mean']:.2f} | {cell[c]['se']:.2f} | {dtxt} | {np.mean(list(nll[c].values())):.4f} | {np.mean(list(test[c].values())):.2f}"
            md.append(f"| {'**' + g(x) + '**' if c == bestk else g(x)} | {row} |")
        md.append("")
        if "F" in ax:
            md.append(f"- λ matters? one-way F({ax['df'][0]}, {ax['df'][1]}) = {ax['F']:.2f}, **p = {ax['p_any']:.3g}** "
                      f"(seed-blocked; residual SD {ax['resid_sd']:.2f} pp)")
        if "fit" in ax:
            f = ax["fit"]
            md.append(f"- quadratic fit over the non-zero λ's: optimum λ* = {f['opt']:.3g} "
                      f"(90 % interval {f['lo']:.3g} – {f['hi']:.3g}), P(interior max) = {f['p_interior']:.2f}, "
                      f"curvature p = {ax['p_curv']:.3g}")
        if center is not None:
            md.append(f"- **best non-zero λ on this axis: {g(center)}** ({how}); raw arg-max over all levels: "
                      f"{g(max((x for _, x in have), key=lambda x: cell[(x, 0.0) if idx == 0 else (0.0, x)]['mean']))}")
        md.append("")
    out["axes"] = {k: dict(center=v["center"], how=v["how"], p_any=v.get("p_any"), fit=v.get("fit"),
                           levels={g(l): s for l, s in v["levels"].items()}) for k, v in axes_out.items()}
    if len(axes_out) == 2:
        out["axis_best"] = dict(lambda_smooth=axes_out["lambda_smooth"]["center"],
                                lambda_reg=axes_out["lambda_reg"]["center"])

    # ---- stage 2: the 2-D neighbourhood ---------------------------------------
    two = {k: v for k, v in cell.items() if k[0] > 0 and k[1] > 0}
    if two:
        rows = sorted({k[0] for k in two})
        cols = sorted({k[1] for k in two})
        full_rect = all((r, c) in two for r in rows for c in cols)
        md.append("## 2-D neighbourhood (both terms on)\n")
        md.append("| λ_smooth \\ λ_reg | " + " | ".join(g(c) for c in cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        bestk = max(two, key=lambda k: cell[k]["mean"])
        for r in rows:
            md.append(f"| {g(r)} | " + " | ".join(
                (("**" + f"{two[(r, c)]['mean']:.2f} ± {two[(r, c)]['se']:.2f}" + "**") if (r, c) == bestk
                 else f"{two[(r, c)]['mean']:.2f} ± {two[(r, c)]['se']:.2f}") if (r, c) in two else "–" for c in cols) + " |")
        md.append("")
        md.append("Paired Δ vs (0, 0), pp:\n")
        md.append("| λ_smooth \\ λ_reg | " + " | ".join(g(c) for c in cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for r in rows:
            cells_ = []
            for c in cols:
                p = paired((r, c)) if (r, c) in two else None
                cells_.append(f"{p[0]:+.2f} ± {p[1]:.2f} ({p[2]}/{p[3]})" if p else "–")
            md.append(f"| {g(r)} | " + " | ".join(cells_) + " |")
        md.append("")
        if full_rect and len(rows) > 1 and len(cols) > 1:
            sb = [s for s in seeds if all(s in two[k] for k in two)] or seeds
            from scipy import stats
            ys, ri, ci, si = [], [], [], []
            for r in rows:
                for c in cols:
                    for s in sb:
                        if s in val[(r, c)]:
                            ys.append(val[(r, c)][s]); ri.append(rows.index(r)); ci.append(cols.index(c)); si.append(sb.index(s))
            y, ri, ci, si = np.array(ys), np.array(ri), np.array(ci), np.array(si)
            n = len(y)
            d = lambda v, k: np.stack([(v == i).astype(float) for i in range(1, k)], 1) if k > 1 else np.zeros((n, 0))
            B, A, C_ = d(si, len(sb)), d(ri, len(rows)), d(ci, len(cols))
            AC = np.stack([A[:, i] * C_[:, j] for i in range(A.shape[1]) for j in range(C_.shape[1])], 1)
            one = np.ones((n, 1))
            _, rf, _, pf = _rss(np.hstack([one, B, A, C_, AC]), y)
            dfr = n - pf
            md.append(f"Two-way ANOVA on this {len(rows)}×{len(cols)} block (meta-seeds {sb}):\n")
            md.append("| Effect | F | df | p |")
            md.append("|---|---|---|---|")
            parts = {"λ_smooth": A, "λ_reg": C_, "interaction": AC}
            an = {}
            for nm in parts:
                X = np.hstack([one, B] + [m for k, m in parts.items() if k != nm])
                _, r0_, _, p0 = _rss(X, y)
                dd = pf - p0
                F = ((r0_ - rf) / dd) / (rf / dfr) if dd > 0 and dfr > 0 else float("nan")
                pv = float(1 - stats.f.cdf(F, dd, dfr)) if F == F else float("nan")
                an[nm] = dict(F=float(F), df=(int(dd), int(dfr)), p=pv)
                md.append(f"| {nm} | {F:.2f} | ({dd}, {dfr}) | {pv:.3g} |")
            md.append("")
            out["anova_2d"] = an

    # ---- overall winner ----------------------------------------------------------
    best = max(cell, key=lambda k: (cell[k]["mean"], -np.mean(list(nll[k].values()))))
    c = cell[best]
    thr = c["mean"] - (c["se"] if c["se"] == c["se"] else 0.0)
    plateau = sorted(k for k in cell if cell[k]["mean"] >= thr)
    md.append("## Best λ's (selected on validation accuracy over every cell run)\n")
    md.append(f"- **λ_smooth = {g(best[0])}, λ_reg = {g(best[1])}** — val {c['mean']:.2f} ± {c['se']:.2f} % "
              f"(n={c['n']} meta-seeds), test {np.mean(list(test[best].values())):.2f} %")
    if (0.0, 0.0) in cell:
        md.append(f"- reference (0, 0): val {cell[(0.0, 0.0)]['mean']:.2f} ± {cell[(0.0, 0.0)]['se']:.2f} %")
    p = paired(best)
    if p:
        md.append(f"- paired gain of the winner over the reference: {p[0]:+.2f} ± {p[1]:.2f} pp ({p[2]}/{p[3]} meta-seeds better)")
    md.append(f"- {len(plateau)} cells within 1 SE of the winner: " + ", ".join(f"({g(x)}, {g(y)})" for x, y in plateau[:12])
              + (" …" if len(plateau) > 12 else ""))
    if two and (best[0] == 0.0 or best[1] == 0.0):
        bb = max(two, key=lambda k: cell[k]["mean"])
        pb = paired(bb)
        md.append(f"- best cell with **both** terms on: λ_smooth = {g(bb[0])}, λ_reg = {g(bb[1])} — "
                  f"val {cell[bb]['mean']:.2f} ± {cell[bb]['se']:.2f} %"
                  + (f", Δ vs (0,0) {pb[0]:+.2f} ± {pb[1]:.2f} pp ({pb[2]}/{pb[3]})" if pb else ""))
        out["best_both_on"] = dict(lambda_smooth=bb[0], lambda_reg=bb[1], val=cell[bb]["mean"], se=cell[bb]["se"])
    md.append(f"- cells run: {len(cell)} of {len(G) ** 2} possible; meta-seeds per cell: "
              f"{sorted({v['n'] for v in cell.values()})}")
    md.append("")

    rf_ = D["ref"] / "adamw.json"
    if rf_.exists():
        rows_ = json.load(open(rf_))["rows"]
        b = max(rows_, key=lambda r: np.mean([e["val_acc"] for e in r["evals"]]))
        va = np.array([100 * e["val_acc"] for e in b["evals"]])
        te = np.array([100 * e["test_acc"] for e in b["evals"]])
        md.append("## AdamW reference on FMNIST + MLP (not part of the sweep)\n")
        md.append(f"lr={b['lr']:g}, wd={b['wd']:g}: val {va.mean():.2f} ± {se(va):.2f} %, "
                  f"test {te.mean():.2f} ± {se(te):.2f} % ({len(va)} optimizee inits)\n")
        out["adamw_ref"] = dict(lr=b["lr"], val=float(va.mean()), val_se=se(va), test=float(te.mean()))

    out["best_cell"] = dict(lambda_smooth=best[0], lambda_reg=best[1], val=c["mean"], se=c["se"])
    out["cells"] = {f"{g(k[0])},{g(k[1])}": dict(val=cell[k]["mean"], se=cell[k]["se"], n=cell[k]["n"],
                                                   val_nll=float(np.mean(list(nll[k].values()))),
                                                   test=float(np.mean(list(test[k].values()))),
                                                   per_seed=val[k]) for k in cell}
    (D["root"] / "RESULTS.md").write_text("\n".join(md))
    json.dump(out, open(D["root"] / "best_lambdas.json", "w"), indent=1)
    try:
        _plot(D, G, val, cell, axes_out, best)
    except Exception as e:
        print("plot failed:", repr(e))
    print("\n".join(md))
    return out


BLUE, ORANGE, INK, INK2, SURF, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


def _plot(D, G, val, cell, axes_out, best):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    ncol = 2 if any(k[0] > 0 and k[1] > 0 for k in cell) else 1
    n_ax = sum(1 for k in ("lambda_smooth", "lambda_reg") if k in axes_out)
    if n_ax == 0:
        return
    fig, axs = plt.subplots(1, n_ax + (ncol - 1), figsize=(5.6 * (n_ax + ncol - 1), 4.4), facecolor=SURF)
    axs = np.atleast_1d(axs)
    ref = cell.get((0.0, 0.0))
    for ax, (key, label, color) in zip(axs, (("lambda_smooth", "λ_smooth  (λ_reg = 0)", BLUE),
                                              ("lambda_reg", "λ_reg  (λ_smooth = 0)", ORANGE))):
        if key not in axes_out:
            continue
        ax.set_facecolor(SURF)
        lv = axes_out[key]["levels"]
        xs = [i for i, x in enumerate(G) if x in lv]
        ys = [lv[G[i]]["mean"] for i in xs]
        es = [lv[G[i]]["se"] if lv[G[i]]["se"] == lv[G[i]]["se"] else 0 for i in xs]
        for i in xs:
            ax.scatter([i] * len(lv[G[i]]["per_seed"]), lv[G[i]]["per_seed"], s=10, color=color, alpha=.35, zorder=2)
        ax.errorbar(xs, ys, yerr=es, color=color, lw=2, marker="o", ms=6, mfc=color, mec=SURF, mew=1.5,
                    capsize=3, zorder=3)
        if ref:
            ax.axhline(ref["mean"], color=INK2, lw=1.2, ls=(0, (5, 3)), zorder=1)
            ax.text(len(G) - 0.5, ref["mean"], " (0,0) ref", va="bottom", ha="right", fontsize=8, color=INK2)
        c = axes_out[key]["center"]
        if c is not None and c in G:
            ax.axvline(G.index(c), color=INK2, lw=1, alpha=.5)
            ax.annotate(f"best λ={c:g}", (G.index(c), ax.get_ylim()[0]), xytext=(4, 6), textcoords="offset points",
                        fontsize=9, color=INK)
        ax.set_xticks(range(len(G)), [f"{x:g}" for x in G], rotation=60, fontsize=7, color=INK2)
        ax.set_xlabel(label, color=INK2)
        ax.set_ylabel("validation accuracy (%)", color=INK2)
        ax.grid(axis="y", color=GRID, lw=.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2)
    two = {k: v for k, v in cell.items() if k[0] > 0 and k[1] > 0}
    if two and ncol == 2:
        ax = axs[-1]
        rows, cols = sorted({k[0] for k in two}), sorted({k[1] for k in two})
        M = np.full((len(rows), len(cols)), np.nan)
        for i, r in enumerate(rows):
            for j, c in enumerate(cols):
                if (r, c) in two:
                    M[i, j] = two[(r, c)]["mean"]
        cmap = LinearSegmentedColormap.from_list("seq", ["#eaf2fc", "#a9c9f2", "#2a78d6", "#12407f"])
        lo_, hi_ = np.nanmin(M), np.nanmax(M)
        im = ax.imshow(M, cmap=cmap, vmin=lo_, vmax=hi_, aspect="auto")
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isnan(M[i, j]):
                    dark = (M[i, j] - lo_) / max(hi_ - lo_, 1e-9) > .55
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color=SURF if dark else INK)
        if best in two:
            ax.add_patch(plt.Rectangle((cols.index(best[1]) - .5, rows.index(best[0]) - .5), 1, 1,
                                       fill=False, ec=ORANGE, lw=3))
        ax.set_xticks(range(len(cols)), [f"{x:g}" for x in cols], color=INK2, fontsize=8)
        ax.set_yticks(range(len(rows)), [f"{x:g}" for x in rows], color=INK2, fontsize=8)
        ax.set_xlabel("λ_reg", color=INK2)
        ax.set_ylabel("λ_smooth", color=INK2)
        ax.set_title("both terms on: val acc (%)", color=INK, loc="left", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(D["root"] / "lambda_sweep.png", dpi=150, facecolor=SURF)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["all", "report", "job-ref", "job-lo"])
    p.add_argument("--stage", choices=["both", "axes", "local", "full"], default="both")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--workers", type=int, default=14)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--meta-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--train-period", type=int, default=None)
    p.add_argument("--radius", type=int, default=2, help="grid steps on each side of the centre (stage local)")
    p.add_argument("--center", type=float, nargs=2, default=None, metavar=("LS", "LR"))
    p.add_argument("--ls", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sys.path.insert(0, str(HERE))
    {"all": cmd_all, "report": cmd_report, "job-ref": job_ref, "job-lo": job_lo}[a.cmd](a)
