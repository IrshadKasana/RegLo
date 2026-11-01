#!/usr/bin/env python
"""
curve_fit.py — pooled smooth-curve analysis of the λ sweep.

Ranking 8 individually noisy λ points picks noise when the run-to-run (meta-seed)
spread is as large as the λ effect.  Instead, for each (method, regularizer) fit

        val_acc[λ, seed] = α_seed + b·x + c·x²         x = log10 λ (centred)

with a free intercept per seed (absorbs the "this meta-seed was a good LO" shift),
and report
  * F-test: does λ matter at all?   (quadratic vs. λ-independent, same seed effects)
  * F-test: is there curvature?     (quadratic vs. linear)
  * the fitted optimum x* = -b/2c, with a residual-bootstrap 90 % interval
  * the fitted accuracy at x*, and the fitted spread of the curve across the grid

Method 2 (AdamW, best lr/wd per λ) is fitted the same way, with the optimizee
init as the "seed", as a sharply-peaked contrast.  Because that curve is strongly
asymmetric (flat, then a steep fall) a quadratic is only fitted LOCALLY, within
+-1 decade of the empirical argmax.

    python curve_fit.py [--seeds stage1|all]
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R = HERE / "results" / "full"
STAGE1 = (0, 1)
rng = np.random.default_rng(0)


def load_lo(seeds_mode):
    d = {}
    for f in (R / "lo").glob("lo_*_seed*.json"):
        j = json.load(open(f))
        if seeds_mode == "stage1" and j["seed"] not in STAGE1:
            continue
        d[(j["kind"], j["lam"], j["seed"])] = 100 * np.mean([e["val_acc"] for e in j["evals"]])
    return d


def load_adamw(reg):
    rows = []
    for f in (R / "adamw" / f"adamw_{reg}.json", R / "adamw" / f"adamw_{reg}_x.json"):
        if f.exists():
            rows += json.load(open(f))["rows"]
    prof = {}
    for r in rows:
        v = np.array([100 * e["val_acc"] for e in r["evals"]])
        if r["lam"] not in prof or v.mean() > prof[r["lam"]].mean():
            prof[r["lam"]] = v
    return prof        # lam -> per-init val acc (paired across lambdas by init index)


def design(lam, grp, order):
    x = np.log10(np.asarray(lam, float))
    xc = (x - x.mean()) / (x.max() - x.min() + 1e-12) * 2          # ~[-1, 1]
    g = np.asarray(grp)
    G = np.stack([(g == k).astype(float) for k in order], 1)         # one intercept per seed
    return x, xc, G


def rss(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    return beta, float(res @ res), res


def analyse(lam, grp, y):
    lam, y = np.asarray(lam, float), np.asarray(y, float)
    order = sorted(set(grp))
    x, xc, G = design(lam, grp, order)
    X0, X1, X2 = G, np.c_[G, xc], np.c_[G, xc, xc ** 2]
    _, r0, _ = rss(X0, y)
    _, r1, _ = rss(X1, y)
    b2, r2, res2 = rss(X2, y)
    n, p2 = len(y), X2.shape[1]
    df2 = n - p2
    F_any = ((r0 - r2) / 2) / (r2 / df2)
    p_any = 1 - stats.f.cdf(F_any, 2, df2)
    F_curv = (r1 - r2) / (r2 / df2)
    p_curv = 1 - stats.f.cdf(F_curv, 1, df2)

    ul = np.array(sorted(set(lam)))
    ux = np.log10(ul)
    uxc = (ux - x.mean()) / (x.max() - x.min() + 1e-12) * 2

    def opt(beta):
        b, c = beta[-2], beta[-1]
        grid = np.linspace(uxc.min(), uxc.max(), 400)
        f = b * grid + c * grid ** 2
        i = int(np.argmax(f))
        interior = c < 0 and 0 < i < len(grid) - 1
        xs = grid[i]
        # back-transform to log10 lambda
        lx = xs / 2 * (x.max() - x.min()) + x.mean()
        return lx, f[i], interior, f

    lx, fpk, interior, f_curve = opt(b2)
    a_mean = b2[:len(order)].mean()
    fit_grid = a_mean + (b2[-2] * uxc + b2[-1] * uxc ** 2)

    # residual bootstrap (parametric noise = fitted residual pool)
    fitted = X2 @ b2
    boots, inter = [], 0
    for _ in range(3000):
        yb = fitted + rng.choice(res2, size=n, replace=True) * np.sqrt(n / df2)
        bb, *_ = rss(X2, yb)
        l, _, it, _ = opt(bb)
        boots.append(l)
        inter += it
    boots = np.array(boots)
    return dict(n=n, df=df2, resid_sd=float(np.sqrt(r2 / df2)),
                F_any=float(F_any), p_any=float(p_any), F_curv=float(F_curv), p_curv=float(p_curv),
                opt_lam=float(10 ** lx), opt_lo=float(10 ** np.percentile(boots, 5)),
                opt_hi=float(10 ** np.percentile(boots, 95)), interior=bool(interior),
                p_interior=inter / len(boots), peak=float(a_mean + fpk),
                curve_range=float(fit_grid.max() - fit_grid.min()), lams=ul.tolist(),
                fit=fit_grid.tolist(), x_all=ux.tolist())


def verdict(a):
    if a["p_any"] > 0.10:
        return "λ has **no detectable effect** (p>0.10): any λ in the grid is statistically equivalent"
    if a["p_curv"] < 0.10 and a["interior"]:
        return "λ matters and the curve has an **interior optimum**"
    return "λ matters but the curve is **monotone/edge-driven** (no interior optimum identified)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", choices=["stage1", "all"], default="stage1")
    a = ap.parse_args()
    lo = load_lo(a.seeds)
    out, md = {}, []
    md.append("# Pooled curve fit of the λ sweep\n")
    md.append(f"Model: `val_acc = α_seed + b·log10λ + c·(log10λ)²`  (per-seed intercepts). "
              f"Seeds used for method 4: **{'0,1 (stage 1, all 8 λ, balanced)' if a.seeds == 'stage1' else 'all available'}**. "
              f"Bootstrap = 3000 residual resamples; 90 % interval.\n")
    md.append("Method 2 is fitted in a ±1-decade window around its empirical best (5 λ's); method 4 uses all 8 λ.\n")
    md.append("| Method | Reg. | n | resid. SD (pp) | p(λ matters) | p(curvature) | fitted optimum λ* | 90 % interval | P(interior max) | fitted peak val % | curve range over grid (pp) | reading |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for reg in ("l1", "l2"):
        # method 2
        prof = load_adamw(reg)
        l_best = max(prof, key=lambda l: prof[l].mean())            # empirical argmax
        prof = {l: v for l, v in prof.items() if abs(np.log10(l / l_best)) <= 1.0 + 1e-9}   # local window (+-1 decade)
        lam, grp, y = [], [], []
        for l, v in prof.items():
            for i, vv in enumerate(v):
                lam.append(l); grp.append(i); y.append(vv)
        r2 = analyse(lam, grp, y)
        # method 4
        lam, grp, y = [], [], []
        for (k, l, s), v in lo.items():
            if k == reg:
                lam.append(l); grp.append(s); y.append(v)
        r4 = analyse(lam, grp, y)
        out[reg] = dict(m2=r2, m4=r4)
        for name, r in (("2. AdamW + R", r2), ("4. LO + R", r4)):
            md.append(f"| {name} | {reg.upper()} | {r['n']} | {r['resid_sd']:.2f} | {r['p_any']:.3g} | {r['p_curv']:.3g} | "
                      f"{r['opt_lam']:.2g} | {r['opt_lo']:.2g} – {r['opt_hi']:.2g} | {r['p_interior']:.2f} | "
                      f"{r['peak']:.2f} | {r['curve_range']:.2f} | {verdict(r)} |")
    md.append("")
    md.append("How to read it: `p(λ matters)` is an F-test of the quadratic-in-log λ model against a λ-independent model with the same "
              "per-seed intercepts; `P(interior max)` is the fraction of bootstrap curves whose maximum lies strictly inside the grid; "
              "`fitted peak` is averaged over seeds/inits. The residual SD includes both meta-training noise and any "
              "lack of fit of a quadratic.\n")
    (R / "CURVE_FIT.md").write_text("\n".join(md))
    json.dump(out, open(R / "curve_fit.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
