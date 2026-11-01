#!/usr/bin/env python
"""
make_table.py — the full λ-sweep table (all four methods, L1 and L2) from whatever
results exist under results/full.  Re-run any time; writes
    results/full/FULL_TABLE.md   and   results/full/full_table.csv

Conventions
  * Method 1/2 (AdamW): per λ, the (lr, wd) that maximises validation accuracy;
    ± is the SE over 6 optimizee inits.
  * Method 3/4 (LO): every number is the mean over META-seeds (each meta-seed is
    an independent LO; its 3 inits are averaged first); ± is the SE across
    meta-seeds.  'stage 1' = meta-seeds 0,1 (all λ); 'fresh' = meta-seeds >= 2
    (finalists only, used to decide the best λ of method 4).
  * Selection is on validation accuracy (tie-break: validation NLL).  Test
    accuracy is shown for information; it is never used to choose anything.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R = HERE / "results" / "full"
STAGE1 = (0, 1)


def se(x):
    x = np.asarray(x, float)
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float("nan")


def fmt(m, s, nd=2):
    return f"{100 * m:.{nd}f} ± {100 * s:.{nd}f}" if s == s else f"{100 * m:.{nd}f}"


def stats(evals):
    g = lambda k: np.array([e[k] for e in evals], float)
    return dict(n=len(evals), val=g("val_acc").mean(), val_se=se(g("val_acc")),
                nll=g("val_loss").mean(), test=g("test_acc").mean(), test_se=se(g("test_acc")),
                train=g("train_acc").mean(), l1=g("l1_norm").mean())


def meta_stats(per_seed):
    """per_seed: {seed: [evals of that LO]} -> stats across meta-seeds."""
    if not per_seed:
        return None
    ks = sorted(per_seed)
    m = {k: dict(val=np.mean([e["val_acc"] for e in per_seed[k]]), nll=np.mean([e["val_loss"] for e in per_seed[k]]),
                 test=np.mean([e["test_acc"] for e in per_seed[k]]), train=np.mean([e["train_acc"] for e in per_seed[k]]),
                 l1=np.mean([e["l1_norm"] for e in per_seed[k]])) for k in ks}
    a = lambda f: np.array([m[k][f] for k in ks])
    return dict(n=len(ks), seeds=ks, val=a("val").mean(), val_se=se(a("val")), nll=a("nll").mean(),
                test=a("test").mean(), test_se=se(a("test")), train=a("train").mean(), l1=a("l1").mean(),
                per_seed_val=a("val").tolist())


def load():
    ad = {}
    for reg in ("none", "l1", "l2"):
        rows = []
        for f in (R / "adamw" / f"adamw_{reg}.json", R / "adamw" / f"adamw_{reg}_x.json"):
            if f.exists():
                rows += json.load(open(f))["rows"]
        ad[reg] = rows
    lo = {}
    for f in sorted((R / "lo").glob("lo_*_seed*.json")):
        j = json.load(open(f))
        lo.setdefault((j["kind"], j["lam"]), {})[j["seed"]] = j["evals"]
    return ad, lo


def key(s):
    return (s["val"], -s["nll"])


def main():
    ad, lo = load()
    out_rows, md = [], []

    def add(method, reg, lam, st, lr="", wd="", basis="", flag=""):
        out_rows.append(dict(method=method, reg=reg, lam=lam, lr=lr, wd=wd, basis=basis, n=st["n"],
                             val_acc=st["val"], val_se=st["val_se"], val_nll=st["nll"], test_acc=st["test"],
                             test_se=st["test_se"], train_acc=st["train"], l1_norm=st["l1"], selected=flag))

    # ---- λ = 0 references ---------------------------------------------------
    best_ad0 = max(({**r, **stats(r["evals"])} for r in ad["none"]), key=key)
    add("1. AdamW", "-", 0.0, best_ad0, best_ad0["lr"], best_ad0["wd"], "6 inits")
    ref3 = meta_stats({s: e for s, e in lo.get(("none", 0.0), {}).items()})
    ref3f = meta_stats({s: e for s, e in lo.get(("none", 0.0), {}).items() if s not in STAGE1})
    refs = meta_stats({s: e for s, e in lo.get(("smooth", 0.0), {}).items()})
    refsf = meta_stats({s: e for s, e in lo.get(("smooth", 0.0), {}).items() if s not in STAGE1})
    if ref3:
        add("3. LO", "-", 0.0, ref3, basis=f"{ref3['n']} meta-seeds")
    if refs:
        add("(diag) LO + smoothing only", "-", 0.0, refs, basis=f"{refs['n']} meta-seeds")

    md.append("# Full λ-sweep table — MNIST, MLP 784-20-10\n")
    md.append("Validation accuracy decides everything (3 000 held-out images); test accuracy (10 000) is "
              "shown for information only. Bold = best λ of that method on validation accuracy.\n")
    md.append("AdamW arms: ± = SE over 6 optimizee inits; best (lr, wd) chosen per λ on validation. "
              "LO arms: ± = SE across meta-training seeds (each seed = an independent LO).\n")

    md.append("## λ = 0 references\n")
    md.append("| Method | config | seeds | val acc % | val NLL | test acc % |")
    md.append("|---|---|---|---|---|---|")
    md.append(f"| 1. AdamW | lr={best_ad0['lr']:g}, wd={best_ad0['wd']:g} | 6 inits | {fmt(best_ad0['val'], best_ad0['val_se'])} | "
              f"{best_ad0['nll']:.4f} | {fmt(best_ad0['test'], best_ad0['test_se'])} |")
    if ref3:
        md.append(f"| 3. LO | meta-lr 1e-3 | {ref3['n']} meta-seeds | {fmt(ref3['val'], ref3['val_se'])} | "
                  f"{ref3['nll']:.4f} | {fmt(ref3['test'], ref3['test_se'])} |")
    if refs:
        md.append(f"| (diag) LO + smoothing only | meta-lr 1e-3 | {refs['n']} meta-seeds | {fmt(refs['val'], refs['val_se'])} | "
                  f"{refs['nll']:.4f} | {fmt(refs['test'], refs['test_se'])} |")
    md.append("")

    summary = {}
    for reg in ("l1", "l2"):
        Rn = reg.upper()
        # method 2
        prof = {}
        for r in ad[reg]:
            st = {**r, **stats(r["evals"])}
            if r["lam"] not in prof or key(st) > key(prof[r["lam"]]):
                prof[r["lam"]] = st
        lams = sorted(prof)
        best2 = max(lams, key=lambda l: key(prof[l]))
        # method 4
        lams4 = sorted(l for (k, l) in lo if k == reg)
        s1 = {l: meta_stats({s: e for s, e in lo[(reg, l)].items() if s in STAGE1}) for l in lams4}
        fr = {l: meta_stats({s: e for s, e in lo[(reg, l)].items() if s not in STAGE1}) for l in lams4}
        fr = {l: v for l, v in fr.items() if v}
        allm = {l: meta_stats(lo[(reg, l)]) for l in lams4}
        if fr:
            best4, basis4 = max(fr, key=lambda l: key(fr[l])), "fresh meta-seeds"
        else:
            best4, basis4 = max(lams4, key=lambda l: key(s1[l])), "stage-1 only — NOT confirmed"
        summary[reg] = dict(best2=best2, best4=best4, basis4=basis4)

        md.append(f"## {Rn} regularizer\n")
        md.append(f"| λ | **M2** AdamW+{Rn}: lr | wd | val acc % | val NLL | test acc % | "
                  f"**M4** LO+{Rn}: val acc % (stage 1, n=2) | val acc % (fresh) | n fresh | val NLL (stage 1) | test acc % (stage 1) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for l in sorted(set(lams) | set(lams4)):
            p, s, f = prof.get(l), s1.get(l), fr.get(l)
            b2 = f"**{l:g}**" if l == best2 else f"{l:g}"
            c2 = (f"{p['lr']:g} | {p['wd']:g} | {fmt(p['val'], p['val_se'])} | {p['nll']:.4f} | {fmt(p['test'], p['test_se'])}"
                  if p else "– | – | – | – | –")
            if s:
                v1 = fmt(s["val"], s["val_se"])
                vf = fmt(f["val"], f["val_se"]) if f else "–"
                if f and l == best4:
                    vf = f"**{vf}**"
                if not f and l == best4 and not fr:
                    v1 = f"**{v1}**"
                c4 = f"{v1} | {vf} | {f['n'] if f else '–'} | {s['nll']:.4f} | {fmt(s['test'], s['test_se'])}"
            else:
                c4 = "– | – | – | – | –"
            md.append(f"| {b2} | {c2} | {c4} |")
            if p:
                add(f"2. AdamW + {Rn}", reg, l, p, p["lr"], p["wd"], "6 inits", "best" if l == best2 else "")
            if s:
                add(f"4. LO + {Rn}", reg, l, s, basis="stage-1 seeds 0,1")
            if f:
                add(f"4. LO + {Rn}", reg, l, f, basis=f"fresh seeds {f['seeds']}", flag="best" if l == best4 else "")
        md.append("")
        p2, = [prof[best2]]
        md.append(f"**Best λ — method 2: {best2:g}** (lr={p2['lr']:g}, wd={p2['wd']:g}), val {fmt(p2['val'], p2['val_se'])} %. "
                  f"**Method 4: {best4:g}** ({basis4}).\n")

    # ---- summary: best of each method -----------------------------------------
    md.append("## Summary — best configuration per method (selected on validation)\n")
    md.append("| Method | regularizer | best λ | other hyper-params | val acc % | test acc % | basis |")
    md.append("|---|---|---|---|---|---|---|")
    md.append(f"| 1. AdamW | – | – | lr={best_ad0['lr']:g}, wd={best_ad0['wd']:g} | {fmt(best_ad0['val'], best_ad0['val_se'])} | "
              f"{fmt(best_ad0['test'], best_ad0['test_se'])} | 6 inits |")
    for reg in ("l1", "l2"):
        Rn = reg.upper()
        r = next(x for x in out_rows if x["reg"] == reg and x["method"].startswith("2.") and x["selected"] == "best")
        md.append(f"| 2. AdamW + {Rn} | {Rn} | **{r['lam']:g}** | lr={r['lr']:g}, wd={r['wd']:g} | "
                  f"{fmt(r['val_acc'], r['val_se'])} | {fmt(r['test_acc'], r['test_se'])} | 6 inits |")
    if ref3:
        md.append(f"| 3. LO | – | – | meta-lr 1e-3 | {fmt(ref3['val'], ref3['val_se'])} | "
                  f"{fmt(ref3['test'], ref3['test_se'])} | {ref3['n']} meta-seeds |")
    for reg in ("l1", "l2"):
        Rn = reg.upper()
        b = summary[reg]["best4"]
        st = (fr_ := meta_stats({s: e for s, e in lo[(reg, b)].items() if s not in STAGE1})) or \
            meta_stats({s: e for s, e in lo[(reg, b)].items() if s in STAGE1})
        md.append(f"| 4. LO + {Rn} | {Rn} | **{b:g}** | + smoothing | {fmt(st['val'], st['val_se'])} | "
                  f"{fmt(st['test'], st['test_se'])} | {summary[reg]['basis4']} (n={st['n']}) |")
    md.append("")
    md.append("*Method 4 contains the smoothing term (as the proposed model does); compare it with the "
              "'LO + smoothing only' row to isolate the regularizer's effect.*")
    (R / "FULL_TABLE.md").write_text("\n".join(md) + "\n")
    with open(R / "full_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
