"""
Figures.

Seven plots, each answering a specific question a reader or reviewer will ask:

  1. rho_sensitivity      How was rho chosen, and is the choice robust?
  2. val_vs_test          Did validation selection actually pick the test
                          optimum, or did we get lucky / unlucky?
  3. training_curves      Do the arms differ in optimisation or only in
                          generalisation?
  4. sharpness_vs_rho     Does the regularizer do the thing it claims to do?
  5. accuracy_vs_sharpness  Is flatness actually associated with generalisation
                          here, or is that assumption unsupported?
  6. gsam_heatmap         The (rho, alpha) interaction, which a 1-D plot hides.
  7. summary_bars         The headline comparison with error bars.

All plots are written to `figures/` as both PNG (300 dpi) and PDF (vector,
for LaTeX \\includegraphics).
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .analyze import load, aggregate, pick_winners, mean_std

# Colour-blind-safe palette (Okabe-Ito). Consistent arm colours across all
# figures so a reader can track an arm between panels.
PALETTE = {
    "adamw": "#000000", "sam": "#0072B2", "asam": "#56B4E9",
    "gam": "#D55E00", "gsam": "#009E73", "none": "#000000",
    "wsam": "#CC79A7",
}
MARKERS = {"adamw": "o", "sam": "s", "asam": "^", "gam": "D",
           "gsam": "v", "wsam": "P"}


def _style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.25,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.autolayout": False,
    })


def _save(fig, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  figures/{name}.png  +  .pdf")


def _color(arm):
    return PALETTE.get(arm, "#999999")


def _radius_key(arm: str) -> str:
    """
    Which config field is the swept continuum for this arm.
    For WSAM it is gamma (i.e. lambda), not rho -- gamma=0.5 is exactly SAM,
    so sweeping it answers "is SAM's implicit weight of 1 the right weight?"
    """
    if arm == "wsam":
        return "gamma"
    return "grad_rho" if arm == "gam" else "rho"


# --------------------------------------------------------------------------- #
def plot_rho_sensitivity(rows, outdir, report="test_acc_at_best_val",
                         tol_pp: float = 0.10):
    """
    Accuracy vs the regularization radius -- a PROFILE plot.

    THE BUG THIS FIXES
    ------------------
    The previous version pooled every run sharing a given rho, marginalising
    over learning rate AND over the arm's other knobs (alpha for GSAM,
    grad_gamma for GAM). At a large rho some of those combinations collapse to
    ~10% (chance level on MNIST) while others train fine, so the pooled mean
    was dragged down and the pooled s.d. exploded. Concretely, 36 runs at 95.3%
    pooled with 12 at 10.2% gives 74.0 +- 37.2, i.e. an error bar reaching
    111% accuracy -- which is what the old figure showed. It was never a
    diverged seed; it was averaging over configurations that should never have
    been averaged.

    THE FIX
    -------
    For each rho we select the BEST setting of all other hyperparameters on
    mean validation accuracy, and plot that. This is the standard "profile"
    view and it answers the question the figure is actually asking: at its own
    best setting, how good is each radius? Error bars are then across SEEDS
    only, which is what "mean +- s.d. over seeds" claims.

    ALSO FIXED
    ----------
    * y-axis is clipped at 100% (accuracy cannot exceed it)
    * x ticks are placed at the swept values, not matplotlib's log defaults
    * the annotation says "best on validation", since that is what it is
    * ties are broken toward the SMALLER radius: if two radii are within
      `tol_pp` percentage points on validation, prefer the smaller one --
      same performance for a smaller perturbation is the better choice, and
      it stops the marker jumping around on noise.
    """
    # (arm, rho) -> {other_hp_tuple: {"val": [...], "test": [...]}}
    cells = defaultdict(lambda: defaultdict(lambda: {"val": [], "test": []}))
    OTHER = ["lr", "alpha", "grad_gamma", "grad_norm_rho", "gamma", "lam",
             "weight_decay", "adaptive", "m_sharpness"]
    for r in rows:
        c = r["config"]
        arm = c.get("arm", c["regularizer"])
        if arm in ("adamw", "none"):
            continue
        rk = _radius_key(arm)
        if c.get(rk) is None:
            continue
        other = tuple((k, c.get(k)) for k in OTHER if c.get(k) is not None)
        cells[(arm, float(c[rk]))][other]["val"].append(r["best_val_acc"])
        cells[(arm, float(c[rk]))][other]["test"].append(r[report])

    if not cells:
        print("  [skip] rho_sensitivity: no radius values found")
        return

    # profile: keep only the best-on-validation setting per (arm, rho)
    prof = {}
    for (arm, rho), by_other in cells.items():
        best = None
        for other, d in by_other.items():
            vm = mean_std(d["val"])[0]
            if best is None or vm > best[0]:
                best = (vm, other, d)
        prof[(arm, rho)] = best

    arms = sorted({a for a, _ in prof})
    fig, axes = plt.subplots(1, len(arms), figsize=(4.6 * len(arms), 4.0),
                             squeeze=False)

    for ax, arm in zip(axes[0], arms):
        rhos = sorted(r for a, r in prof if a == arm)
        vm, vs, tm, ts, ns = [], [], [], [], []
        for r in rhos:
            _, _, d = prof[(arm, r)]
            a, b = mean_std(d["val"]);  vm.append(a * 100); vs.append(b * 100)
            a, b = mean_std(d["test"]); tm.append(a * 100); ts.append(b * 100)
            ns.append(len(d["test"]))

        ax.errorbar(rhos, vm, yerr=vs, marker="o", ms=4, lw=1.3, capsize=3,
                    color=_color(arm), label="Validation accuracy",
                    alpha=0.55, ls="--")
        ax.errorbar(rhos, tm, yerr=ts, marker=MARKERS.get(arm, "s"), ms=6,
                    lw=1.9, capsize=3, color=_color(arm),
                    label=("Test accuracy" if report != "best_val_acc" else "Validation accuracy (selected epoch)"))

        # ---- selection marker: ARGMAX on validation --------------------
        # This must be the plain argmax so the figure agrees with what
        # `analyze` reports. The previous version applied a "tie-break
        # toward the smaller radius" with a FIXED 0.10 pp tolerance. Because
        # the validation spread between radii here is only 0.01-0.20 pp,
        # nearly every point qualified as a tie and the rule degenerated to
        # "annotate the smallest value in the grid" -- e.g. GAM was labelled
        # 0.02 when the argmax was 0.08, GSAM 0.01 when it was 0.05. Two
        # different numbers for the same quantity in one report.
        best_i = max(range(len(vm)), key=lambda i: vm[i])

        # ---- plateau: statistically indistinguishable from the best -----
        # Tolerance is now the STANDARD ERROR of the best point, not a
        # hard-coded constant, so it scales with the actual noise.
        n_best = ns[best_i]
        se = (vs[best_i] / math.sqrt(n_best)) if n_best > 1 else 0.0
        plateau = [i for i, v in enumerate(vm) if v >= vm[best_i] - se]

        ax.axvline(rhos[best_i], color=_color(arm), ls=":", lw=1.4, alpha=0.9)
        label = f"best on validation\n$\\rho^*$ = {rhos[best_i]:g}"
        if len(plateau) > 1:
            lo, hi = rhos[min(plateau)], rhos[max(plateau)]
            ax.axvspan(lo, hi, color=_color(arm), alpha=0.10, lw=0)
            label += f"\nplateau [{lo:g}, {hi:g}]"
        ax.annotate(label,
                    xy=(rhos[best_i], tm[best_i]), xytext=(8, -34),
                    textcoords="offset points", fontsize=8,
                    color=_color(arm),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white",
                              ec=_color(arm), alpha=0.8, lw=0.6))

        # explicit ticks at the swept values -- matplotlib's log defaults are
        # unreadable over a narrow decade range
        if len(rhos) > 2 and min(rhos) > 0 and max(rhos) / min(rhos) >= 8:
            ax.set_xscale("log")
        ax.set_xticks(rhos)
        ax.set_xticklabels([f"{r:g}" for r in rhos], fontsize=8)
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.tick_params(axis="x", which="minor", bottom=False)

        # accuracy cannot exceed 100%
        lo = min(min(v - s for v, s in zip(vm, vs)),
                 min(t - s for t, s in zip(tm, ts)))
        hi = max(max(v + s for v, s in zip(vm, vs)),
                 max(t + s for t, s in zip(tm, ts)))
        pad = max((hi - lo) * 0.08, 0.05)
        ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + pad))

        label = r"$\rho$" + ("  (grad_rho)" if arm == "gam" else "")
        ax.set_xlabel(label)
        ax.set_ylabel("accuracy (%)")
        ax.set_title(f"{arm.upper()}   (n = {min(ns)}-{max(ns)} seeds/point)",
                     fontsize=10)
        ax.legend(frameon=False, loc="lower left", fontsize=8)

    fig.suptitle("Sensitivity to the regularization radius\n"
                 "each point at ITS OWN best other hyperparameters "
                 "(chosen on validation); bars are s.d. over seeds", y=1.10)
    _save(fig, outdir, "01_rho_sensitivity")


# --------------------------------------------------------------------------- #
def plot_val_vs_test(rows, outdir, report="test_acc_at_best_val"):
    """
    Every (arm, hyperparameter) configuration as one point: mean val accuracy
    on x, mean test accuracy on y.

    A tight monotone cloud means validation is a reliable selector here. A
    scattered cloud means selection is noisy and the reported winner may be an
    artefact of seed variance -- which is an argument for more seeds, and a
    thing worth knowing before you write conclusions.
    """
    agg = aggregate(rows, "best_val_acc", report)
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    for arm in sorted({a for a, _ in agg}):
        pts = [(d["val_mean"] * 100, d["test_mean"] * 100)
               for (a, _), d in agg.items() if a == arm]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=34, alpha=0.75, color=_color(arm),
                   marker=MARKERS.get(arm, "o"), label=arm.upper(),
                   edgecolors="white", linewidths=0.5)

    lo = min(min(ax.get_xlim()), min(ax.get_ylim()))
    hi = max(max(ax.get_xlim()), max(ax.get_ylim()))
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.9, ls="--", alpha=0.6,
            label="y = x")
    ax.set_xlabel("validation accuracy (%)")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Is validation a reliable selector?\n"
                 "one point per hyperparameter configuration")
    ax.legend(frameon=False)
    _save(fig, outdir, "02_val_vs_test")


# --------------------------------------------------------------------------- #
def plot_training_curves(rows, outdir):
    """
    Learning curves for the validation-selected configuration of each arm,
    averaged over seeds with a shaded +/- 1 s.d. band.

    Separates two very different stories: an arm that optimises worse but
    generalises better (the expected sharpness-aware pattern) versus one that
    simply optimises better.
    """
    agg = aggregate(rows)
    ranked = pick_winners(agg)
    want = {arm: dict(cand[0][0]) for arm, cand in ranked.items()}

    curves = defaultdict(list)
    for r in rows:
        c = r["config"]
        arm = c.get("arm", c["regularizer"])
        if arm not in want or not r.get("history"):
            continue
        if any(c.get(k) != v for k, v in want[arm].items()):
            continue
        curves[arm].append(r["history"])

    curves = {k: v for k, v in curves.items() if v}
    if not curves:
        print("  [skip] training_curves: no history saved "
              "(set keep_history: true in the config)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))
    panels = [("train_loss", "training loss", True),
              ("val_acc", "validation accuracy (%)", False),
              ("test_acc", "test accuracy (%)", False)]

    for ax, (field, label, is_loss) in zip(axes, panels):
        for arm in sorted(curves):
            runs = curves[arm]
            n = min(len(h) for h in runs)
            xs = list(range(n))
            series = [[h[i][field] * (1 if is_loss else 100) for i in range(n)]
                      for h in runs]
            mu = [mean_std([s[i] for s in series])[0] for i in xs]
            sd = [mean_std([s[i] for s in series])[1] for i in xs]
            ax.plot(xs, mu, lw=1.7, color=_color(arm), label=arm.upper())
            ax.fill_between(xs, [m - s for m, s in zip(mu, sd)],
                            [m + s for m, s in zip(mu, sd)],
                            color=_color(arm), alpha=0.15, linewidth=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel(label)
        if is_loss:
            ax.set_yscale("log")
    axes[0].legend(frameon=False)
    fig.suptitle("Learning curves for the validation-selected configuration "
                 "(mean $\\pm$ s.d. over seeds)", y=1.02)
    _save(fig, outdir, "03_training_curves")


# --------------------------------------------------------------------------- #
def plot_sharpness_vs_rho(rows, outdir):
    """
    Empirical sharpness (SAM gap) and gradient norm against rho.

    This is the mechanism check. If increasing rho does not reduce measured
    sharpness, the regularizer is not doing what the method claims and any
    accuracy difference has some other cause.
    """
    by_arm = defaultdict(lambda: defaultdict(lambda: {"gap": [], "gn": []}))
    for r in rows:
        c = r["config"]
        arm = c.get("arm", c["regularizer"])
        rk = _radius_key(arm)
        rho = c.get(rk)
        if arm in ("adamw", "none"):
            rho = 0.0
        if rho is None or "sam_gap_rho0.05" not in r:
            continue
        by_arm[arm][float(rho)]["gap"].append(r["sam_gap_rho0.05"])
        by_arm[arm][float(rho)]["gn"].append(r.get("final_grad_norm_train",
                                                   float("nan")))

    if not by_arm:
        print("  [skip] sharpness_vs_rho: no sharpness measurements found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    panels = [(axes[0], "gap", "SAM gap at $\\rho=0.05$"),
              (axes[1], "gn", r"$\|\nabla L(w)\|_2$ on train")]

    for ax, key, lab in panels:
        for arm in sorted(by_arm):
            rhos = sorted(by_arm[arm])
            mu = [mean_std(by_arm[arm][r][key])[0] for r in rhos]
            sd = [mean_std(by_arm[arm][r][key])[1] for r in rhos]
            if arm in ("adamw", "none"):
                ax.axhline(mu[0], color="black", ls="--", lw=1.3,
                           label="AdamW (no reg.)")
                continue
            ax.errorbar(rhos, mu, yerr=sd, marker=MARKERS.get(arm, "o"), ms=5,
                        lw=1.6, capsize=3, color=_color(arm), label=arm.upper())
        # Set outside the arm loop: the control arm's `continue` above would
        # otherwise skip these whenever it sorts last.
        ax.set_ylabel(lab)
        ax.set_xlabel(r"$\rho$   (grad_rho for GAM)")
        ax.set_yscale("log")
        rho_vals = [r for a in by_arm for r in by_arm[a] if r > 0]
        if rho_vals and max(rho_vals) / min(rho_vals) > 8:
            ax.set_xscale("log")
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Mechanism check: does a larger radius actually flatten "
                 "the minimum?", y=1.02)
    _save(fig, outdir, "04_sharpness_vs_rho")


# --------------------------------------------------------------------------- #
def plot_accuracy_vs_sharpness(rows, outdir, report="test_acc_at_best_val",
                               min_acc: float = 90.0):
    """
    Test accuracy against measured sharpness, one point per run.

    TWO PANELS, DELIBERATELY.

    A run whose optimization diverged collapses to predicting one class. Its
    loss surface is trivially flat (near-zero SAM gap) AND its accuracy is
    terrible. A handful of such runs will dominate a Pearson correlation and
    produce a large POSITIVE r between sharpness and accuracy -- i.e. the
    literal opposite of the flat-minima hypothesis -- purely as an artifact.

    So: left panel is every run (the artifact-prone view), right panel drops
    runs below `min_acc` (the view you can actually interpret). Report the
    right one, and state the exclusion rule in the caption.
    """
    recs = []
    for r in rows:
        if "sam_gap_rho0.05" not in r:
            continue
        arm = r["config"].get("arm", r["config"]["regularizer"])
        g = r["sam_gap_rho0.05"]
        acc = r[report] * 100
        if g <= 0 or g != g or acc != acc:
            continue
        recs.append((g, acc, arm))

    if len(recs) < 5:
        print("  [skip] accuracy_vs_sharpness: too few points")
        return

    def pearson_log(sub):
        if len(sub) < 3:
            return float("nan")
        lx = [math.log10(a) for a, _, _ in sub]
        ys = [b for _, b, _ in sub]
        mx, sx = mean_std(lx)
        my, sy = mean_std(ys)
        if not sx or not sy:
            return float("nan")
        return (sum((a - mx) * (b - my) for a, b in zip(lx, ys))
                / ((len(lx) - 1) * sx * sy))

    healthy = [t for t in recs if t[1] >= min_acc]
    dropped = len(recs) - len(healthy)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, sub, title in [
            (axes[0], recs, f"All runs (n = {len(recs)})"),
            (axes[1], healthy, f"Converged runs only, acc $\\geq$ {min_acc:g}% "
                               f"(n = {len(healthy)})")]:
        if not sub:
            continue
        ax.scatter([a for a, _, _ in sub], [b for _, b, _ in sub],
                   c=[_color(c) for _, _, c in sub], s=26, alpha=0.6,
                   edgecolors="white", linewidths=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("empirical sharpness (SAM gap at $\\rho=0.05$)")
        ax.set_ylabel("test accuracy (%)")
        r_val = pearson_log(sub)
        ax.set_title(f"{title}\nPearson r = {r_val:.3f} (log sharpness)")

    if dropped:
        axes[0].annotate(
            f"{dropped} diverged run(s):\ntrivially flat AND inaccurate.\n"
            f"These alone flip the sign of r.",
            xy=(0.03, 0.10), xycoords="axes fraction", fontsize=8,
            color="#B00020",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#B00020",
                      alpha=0.85))

    seen = {}
    for _, _, a in recs:
        seen[a] = _color(a)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=a.upper())
               for a, c in sorted(seen.items())]
    axes[1].legend(handles=handles, frameon=False, fontsize=8)
    fig.suptitle("Flatness vs generalisation "
                 "(the flat-minima hypothesis predicts NEGATIVE r)", y=1.03)
    _save(fig, outdir, "05_accuracy_vs_sharpness")


# --------------------------------------------------------------------------- #
def plot_gsam_heatmap(rows, outdir, report="test_acc_at_best_val"):
    """
    GSAM has two interacting hyperparameters. A 1-D rho sweep at fixed alpha
    can easily miss the joint optimum, so show the grid.
    """
    cell = defaultdict(list)
    for r in rows:
        c = r["config"]
        if c.get("arm") != "gsam":
            continue
        if c.get("rho") is None or c.get("alpha") is None:
            continue
        cell[(float(c["rho"]), float(c["alpha"]))].append(r[report])

    if len(cell) < 4:
        print("  [skip] gsam_heatmap: needs a 2-D rho x alpha grid")
        return

    rhos = sorted({k[0] for k in cell})
    alphas = sorted({k[1] for k in cell})
    Z = [[mean_std(cell.get((r, a), [float("nan")]))[0] * 100 for r in rhos]
         for a in alphas]

    fig, ax = plt.subplots(figsize=(1.15 * len(rhos) + 2.4,
                                    0.85 * len(alphas) + 2.4))
    im = ax.imshow(Z, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(rhos)), [f"{r:g}" for r in rhos])
    ax.set_yticks(range(len(alphas)), [f"{a:g}" for a in alphas])
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title("GSAM: test accuracy (%) over the joint grid")
    ax.grid(False)

    best = max(((i, j) for j in range(len(alphas)) for i in range(len(rhos))
                if Z[j][i] == Z[j][i]), key=lambda ij: Z[ij[1]][ij[0]])
    for j in range(len(alphas)):
        for i in range(len(rhos)):
            v = Z[j][i]
            if v != v:
                continue
            ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v < (max(map(max, Z)) + min(map(min, Z))) / 2
                    else "black",
                    fontweight="bold" if (i, j) == best else "normal")
    fig.colorbar(im, ax=ax, label="test accuracy (%)")
    _save(fig, outdir, "06_gsam_heatmap")


# --------------------------------------------------------------------------- #
def plot_summary_bars(rows, outdir, control="adamw",
                      report="test_acc_at_best_val"):
    """
    The headline figure: selected configuration per arm, individual seeds
    overlaid as dots so the reader can see the actual spread rather than
    trusting an error bar.
    """
    from .analyze import paired_t
    agg = aggregate(rows, "best_val_acc", report)
    ranked = pick_winners(agg)
    arms = sorted(ranked)
    if control not in ranked:
        control = arms[0]
    ctrl = ranked[control][0][1]["byseed_test"]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    ax = axes[0]
    means = [ranked[a][0][1]["test_mean"] * 100 for a in arms]
    stds = [ranked[a][0][1]["test_std"] * 100 for a in arms]
    ax.bar(range(len(arms)), means, yerr=stds, capsize=5,
           color=[_color(a) for a in arms], alpha=0.75, edgecolor="black",
           linewidth=0.7)
    for i, a in enumerate(arms):
        pts = list(ranked[a][0][1]["byseed_test"].values())
        ax.scatter([i] * len(pts), [p * 100 for p in pts], color="black",
                   s=11, zorder=3, alpha=0.65)
        if a != control:
            _, p, _ = paired_t(ranked[a][0][1]["byseed_test"], ctrl)
            star = ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.05 else "n.s.")
            ax.text(i, means[i] + stds[i] + 0.04, star, ha="center", fontsize=9)
    ax.set_xticks(range(len(arms)), [a.upper() for a in arms])
    ax.set_ylabel("test accuracy (%)")
    lo = min(m - s for m, s in zip(means, stds))
    hi = max(m + s for m, s in zip(means, stds))
    ax.set_ylim(lo - 0.35, hi + 0.35)
    ax.set_title(f"Test accuracy (paired vs {control.upper()})")

    ax = axes[1]
    deltas, errs = [], []
    for a in arms:
        d, _, _ = paired_t(ranked[a][0][1]["byseed_test"], ctrl)
        seeds = sorted(set(ranked[a][0][1]["byseed_test"]) & set(ctrl))
        diffs = [ranked[a][0][1]["byseed_test"][s] - ctrl[s] for s in seeds]
        _, sd = mean_std(diffs)
        deltas.append(d * 100)
        errs.append(1.96 * sd * 100 / math.sqrt(max(len(diffs), 1)))
    ax.barh(range(len(arms)), deltas, xerr=errs, capsize=5,
            color=[_color(a) for a in arms], alpha=0.75, edgecolor="black",
            linewidth=0.7)
    ax.axvline(0, color="black", lw=1.0)
    ax.set_yticks(range(len(arms)), [a.upper() for a in arms])
    ax.set_xlabel(f"test accuracy change vs {control.upper()} (pp)")
    ax.set_title("Paired difference, 95% CI\n"
                 "(bars crossing zero are not significant)")

    _save(fig, outdir, "07_summary")


# --------------------------------------------------------------------------- #
def plot_secondary_effects(rows, outdir, control="adamw",
                           report="test_acc_at_best_val"):
    """
    The figure to lead with when the accuracy comparison comes back null.

    Three paired panels for the validation-selected configuration of each arm:
    test accuracy, generalisation gap, and empirical sharpness. Accuracy is
    often the least sensitive of the three -- on an underparameterised model
    near a dataset's accuracy ceiling it can be flat while the gap and the
    sharpness move by factors of two to four.

    That pattern is a real, reportable finding: the regularizers demonstrably
    do what they claim to the loss surface, and the effect simply does not
    convert into accuracy in this regime. Saying that plainly is far better
    than straining a 0.16pp accuracy difference past what the seeds support.
    """
    from .analyze import paired_t, aggregate, pick_winners

    agg = aggregate(rows, "best_val_acc", report)
    ranked = pick_winners(agg)
    arms = sorted(ranked)
    if control not in ranked:
        control = arms[0]

    def by_seed(arm, metric):
        hp = dict(ranked[arm][0][0])
        out = {}
        for r in rows:
            c = r["config"]
            if c.get("arm", c["regularizer"]) != arm:
                continue
            if any(c.get(k) != v for k, v in hp.items()):
                continue
            v = r.get(metric) if metric != report else r[report]
            if v is not None and v == v:
                out[c["seed"]] = v
        return out

    panels = [
        (report, "test accuracy (%)", 100.0, False,
         "Accuracy\n(often the least sensitive metric)"),
        ("generalization_gap", "train $-$ test accuracy (pp)", 100.0, False,
         "Generalisation gap\n(lower is better)"),
        ("sam_gap_rho0.05", "SAM gap at $\\rho=0.05$", 1.0, True,
         "Empirical sharpness\n(lower is flatter)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    for ax, (metric, ylab, scale, logy, title) in zip(axes, panels):
        ctrl = by_seed(control, metric)
        means, stds, labels, colors = [], [], [], []
        for arm in arms:
            vals = by_seed(arm, metric)
            if not vals:
                continue
            m, s = mean_std([v * scale for v in vals.values()])
            means.append(m)
            stds.append(s)
            labels.append(arm.upper())
            colors.append(_color(arm))
            i = len(means) - 1
            ax.scatter([i] * len(vals), [v * scale for v in vals.values()],
                       color="black", s=10, zorder=3, alpha=0.6)
            if arm != control and ctrl:
                _, p, _ = paired_t(vals, ctrl)
                star = ("***" if p < .001 else "**" if p < .01
                        else "*" if p < .05 else "n.s.")
                ax.annotate(star, xy=(i, m + s), xytext=(0, 6),
                            textcoords="offset points", ha="center", fontsize=9)
        ax.bar(range(len(means)), means, yerr=stds, capsize=4, color=colors,
               alpha=0.75, edgecolor="black", linewidth=0.7)
        ax.set_xticks(range(len(labels)), labels, rotation=0)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        if logy:
            ax.set_yscale("log")
        else:
            lo = min(m - s for m, s in zip(means, stds))
            hi = max(m + s for m, s in zip(means, stds))
            pad = (hi - lo) * 0.25 or 0.1
            ax.set_ylim(lo - pad, hi + pad * 1.6)

    fig.suptitle("What the regularizers actually changed "
                 f"(paired vs {control.upper()}, seed-matched)", y=1.02)
    _save(fig, outdir, "08_secondary_effects")


# --------------------------------------------------------------------------- #
def _resolve_report(rows, report):
    """Use validation accuracy when the sweep never evaluated the test set."""
    if report != "test_acc_at_best_val":
        return report, "test"
    have = [r.get(report) for r in rows]
    have = [v for v in have if v is not None and v == v]
    if have:
        return report, "test"
    print("  [plots] test never evaluated -- plotting VALIDATION accuracy")
    return "best_val_acc", "validation"


def make_all(patterns, outdir="figures", control="adamw",
             report="test_acc_at_best_val", min_acc: float = 90.0):
    _style()
    rows = load(patterns)
    report, _kind = _resolve_report(rows, report)
    print(f"[plots] {len(rows)} runs -> {outdir}/")
    plot_rho_sensitivity(rows, outdir, report)
    plot_val_vs_test(rows, outdir, report)
    plot_training_curves(rows, outdir)
    plot_sharpness_vs_rho(rows, outdir)
    plot_accuracy_vs_sharpness(rows, outdir, report, min_acc=min_acc)
    plot_gsam_heatmap(rows, outdir, report)
    plot_summary_bars(rows, outdir, control, report)
    plot_secondary_effects(rows, outdir, control, report)
    print("[plots] done")
