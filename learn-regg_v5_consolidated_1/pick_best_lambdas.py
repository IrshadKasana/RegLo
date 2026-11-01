#!/usr/bin/env python3
"""
pick_best_lambdas.py — Phase-1 decision step (poster: 'Select Best lambda').

Reads every sweep-tagged eval JSON (eval_tag = sw_<reg>_sl<..>_rho<..>[_lr<..>]),
ranks each (lambda_smooth, rho, lambda_reg) cell by final VALIDATION accuracy
(fallback: test accuracy), and prints the winner per regularizer plus the exact
--set line to freeze into the configs. Every combination stays in results/ and
in figure 11/17; ONLY the winner is carried into the generalization
experiments (exp8-exp11), per the final design.
"""
import argparse
import json, re, sys
from datetime import datetime
from pathlib import Path
import numpy as np

STRENGTH_KEY = {"sam": "rho", "gam": "grad_rho_max", "gsam": "rho_max"}

JSON_DIR = Path(__file__).parent / "artifacts/dmo/results/json"
# BUGFIX (2026-07-31): alternation must be LONGEST-FIRST. With
# (?:rho|grad_rho|rho_max), a GAM tag '..._grad_rho_max0p005_...' matched
# 'grad_rho' and then required [0-9pm]+ at '_max...' -> NO MATCH: every GAM
# sweep cell was silently invisible to the picker, so GAM never got frozen
# winners and downstream GAM-LO runs kept the YAML defaults (the same
# failure family as the earlier frozen-grad_gamma bug).
PAT = re.compile(r"sw_(sam|gam|gsam)_sl(?P<sl>[0-9pm]+)_(?:grad_rho_max|rho_max|rho)"
                 r"(?P<st>[0-9pm]+)(?:_lr(?P<lr>[0-9pm]+))?(?:_se(?P<se>[0-9pm]+))?")
dec = lambda tok: float(tok.replace("p", ".").replace("m", "-"))

def final_val_acc(d):
    for key in ("instances_acc_val", "instances_acc_ts"):
        rows = d.get(key) or []
        fin = [r[-1] for r in rows if r and r[-1] == r[-1]]
        if fin:
            return float(np.mean(fin)), key
    return None, None

best = {}
for f in sorted(JSON_DIR.glob("*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    tag = (d.get("metadata") or {}).get("eval_tag") or ""
    m = PAT.search(tag)
    if not m:
        continue
    acc, src = final_val_acc(d)
    if acc is None:
        continue
    reg = m.group(1)
    cell = dict(sl=dec(m.group("sl")), strength=dec(m.group("st")),
                lr=dec(m.group("lr")) if m.group("lr") else None,
                se=dec(m.group("se")) if m.group("se") else None,
                acc=acc, metric=src, tag=tag, file=f.name)
    if reg not in best or acc > best[reg]["acc"]:
        best[reg] = cell

if not best:
    sys.exit(f"No sweep JSONs found under {JSON_DIR} (run exp7 / run_sweep first).")

ap = argparse.ArgumentParser()
ap.add_argument("--apply", action="store_true",
                help="FREEZE the winners: write the best values into "
                     "configs/dmo/regularizers/<reg>.yaml (marked lines) and "
                     "record provenance in configs/dmo/best_lambdas.json — "
                     "the generalization experiments then use ONLY the best "
                     "combination, automatically.")
cli = ap.parse_args()

CFG = Path(__file__).parent / "configs/dmo/regularizers"


def _set_yaml_key(text, key, value, tag):
    line = f"  {key}: {value}   # FROZEN from sweep {tag} (pick_best_lambdas --apply)"
    pat = re.compile(rf"^([ \t]*{key}[ \t]*:).*$", re.M)
    if pat.search(text):
        return pat.sub(line, text, count=1)
    return text.rstrip() + "\n" + line + "\n"


prov = {"applied": datetime.now().isoformat(), "winners": {}}
for reg, c in best.items():
    print(f"\n[{reg.upper()}] best cell: {c['tag']}  ({c['metric']}={c['acc']:.4f})")
    lrp = f" lambda_reg={c['lr']}" if c["lr"] is not None else ""
    sep = f" smooth_eps={c['se']}" if c["se"] is not None else ""
    skey = STRENGTH_KEY[reg]
    print(f"  freeze with:  --set smooth_lamb={c['sl']} {skey}={c['strength']}{lrp}{sep} reg_mode=penalty")
    prov["winners"][reg] = {"smooth_lamb": c["sl"], skey: c["strength"],
                            **({"lambda_reg": c["lr"]} if c["lr"] is not None else {}),
                            **({"smooth_eps": c["se"]} if c["se"] is not None else {}),
                            "selected_by": c["metric"], "score": c["acc"],
                            "tag": c["tag"], "json": c["file"]}
    if cli.apply:
        yml = CFG / f"{reg}.yaml"
        t = yml.read_text()
        t = _set_yaml_key(t, "smooth_lamb", c["sl"], c["tag"])
        t = _set_yaml_key(t, skey, c["strength"], c["tag"])
        if c["lr"] is not None:
            t = _set_yaml_key(t, "lambda_reg", c["lr"], c["tag"])
        if c["se"] is not None:
            t = _set_yaml_key(t, "smooth_eps", c["se"], c["tag"])
        yml.write_text(t)
        print(f"  APPLIED -> {yml}")

if cli.apply:
    out = CFG.parent / "best_lambdas.json"
    out.write_text(json.dumps(prov, indent=2))
    print(f"\nprovenance -> {out}")
    print("Generalization runs (exp8-exp11) now inherit ONLY the best "
          "combination via the regularizer configs; run_all_experiments "
          "--use-best-lambdas additionally stamps the frozen values onto "
          "every LO training command for audit.")
