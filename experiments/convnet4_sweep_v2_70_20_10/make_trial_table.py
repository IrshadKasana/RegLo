#!/usr/bin/env python
"""
make_trial_table.py -- ONE ROW PER PLANNED TRIAL of the five-method sweep, with its outcome.

Reads only files the sweep already wrote (nothing is re-run):
    results/<preset>/m1/*.json, m2/*.json, lo/*.json   per-trial results (loss, accuracy, curves, meta-training history)
    results/<preset>/ckpt/*.pt                          learned-optimizer checkpoints (M3/M4a/M5 only)
    results/<preset>/logs/node*_p*.log                  worker logs: "[worker] ok|FAIL <trial> (<s>s)" + OOM retry lines
    results/<preset>/best_config.json                   the selected ("best pick") configuration of every method
and the *planned* trial list, regenerated from sweep_config.py through run_sweep.phase_jobs, so that trials
that never produced a result show up as MISSING / FAILED instead of silently not being there.

Outputs (results/<preset>/trial_table/):
    trials_full.csv      one row per trial: id, method, phase, hyper-parameters, status, attempts, loss, accuracy, OOD,
                         optimizee steps, meta-training steps, checkpoint, best-pick flag, ...
    trials_curves.csv    one row per trial x init x 100-step point (train loss, val loss, val acc)  -> every iteration logged
    trials_meta.csv      one row per learned-optimizer trial x meta-training stage (checkpoint score, saved?, seconds)
    trials_full.html     the same main table, filterable in a browser
    trial_status.md      pass / fail / missing counts per method and the list of every non-clean trial

    python make_trial_table.py --preset full
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_sweep as RS
import sweep_config as SC

LOG_RE = re.compile(r"^\[worker\] (ok|FAIL)\b.* (\S+) \((\d+)s\)\s*$")
OOM_RE = re.compile(r"^\[worker\] CUDA OOM, retry")


def parse_logs(D):
    """{trial_name: dict(ok=n, fail=n, oom_retries=n, last_error=str)} from every worker log (logs are appended to)."""
    out = defaultdict(lambda: dict(ok=0, fail=0, oom_retries=0, last_error=""))
    for f in sorted(D["logs"].glob("node*_p*.log")):
        pending_oom = 0
        for line in open(f, errors="replace"):
            if OOM_RE.match(line):
                pending_oom += 1
                continue
            m = LOG_RE.match(line.rstrip("\n"))
            if not m:
                continue
            status, name = m.group(1), m.group(2)
            rec = out[name]
            rec["oom_retries"] += pending_oom
            pending_oom = 0
            if status == "ok":
                rec["ok"] += 1
            else:
                rec["fail"] += 1
                rec["last_error"] = re.sub(r"\s+", " ", line[line.find("(") + 1: line.rfind(")")].split("Tried")[0]).strip()
    return out


def mean(vs):
    vs = [v for v in vs if v is not None]
    return float(np.mean(vs)) if vs else float("nan")


def eval_cols(evals, targets):
    row = dict(n_inits=len(evals),
               train_loss=mean([e["train_loss"] for e in evals]), train_acc=mean([e["train_acc"] for e in evals]),
               val_loss=mean([e["val_loss"] for e in evals]), val_acc=mean([e["val_acc"] for e in evals]),
               test_loss=mean([e["test_loss"] for e in evals]), test_acc=mean([e["test_acc"] for e in evals]),
               diverged_inits=sum(bool(e.get("diverged")) for e in evals),
               optimizee_steps=(evals[0]["curve"][-1]["step"] if evals and evals[0].get("curve") else ""),
               run_seconds=sum(e.get("seconds", 0.0) for e in evals))
    for i, e in enumerate(evals):
        row[f"val_acc_init{i}"] = e["val_acc"]
        row[f"train_loss_init{i}"] = e["train_loss"]
    for t in targets:
        row[f"ood_{t}_acc"] = mean([e["ood"][t]["acc"] for e in evals])
        row[f"ood_{t}_loss"] = mean([e["ood"][t]["loss"] for e in evals])
        row[f"ood_{t}_auroc"] = mean([e["ood"][t]["auroc_msp"] for e in evals])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    a0 = ap.parse_args()
    a = RS.build_parser().parse_args(["report", "--preset", a0.preset])
    D = RS.dirs(a.preset)
    cfg = RS.make_cfg(a.preset)
    out = D["root"] / "trial_table"
    out.mkdir(exist_ok=True)
    logs = parse_logs(D)
    best = json.load(open(D["root"] / "best_config.json"))["results"]

    # ------------------------------------------------ best-pick lookup
    pick_m1 = {d: best[d]["M1"]["config"]["adamw_lr"] for d in best if "M1" in best[d]}
    pick_m2 = {(d, k.split(":")[1]): (e["config"]["lam"], e["config"]["rho"]) for d in best
               for k, e in best[d].items() if k.startswith("M2:")}
    pick_lo = {}                                       # json file name -> label
    for d in best:
        for k, e in best[d].items():
            for f in e.get("files", []):
                pick_lo[Path(f).name] = k

    # ------------------------------------------------ planned trials (every phase)
    planned = []                                       # (trial_id, phase, kind, meta)
    for ph in ("m1", "m2", "m3", "m4a", "axes", "local", "final"):
        for name, argv, path in RS.phase_jobs(a, ph, D):
            planned.append((name, ph, Path(path)))
    seen, trials = set(), []
    for name, ph, path in planned:
        if path in seen:
            continue
        seen.add(path)
        trials.append((name, ph, path))

    rows, curves, metas = [], [], []
    targets_of = lambda d: [t for t in SC.DATASETS if t != d]

    def base(trial, phase, method, d, **hp):
        return dict(trial=trial, phase=phase, method=method, source=d, reg=hp.get("reg", "-"),
                    adamw_lr=hp.get("adamw_lr", ""), lam=hp.get("lam", ""), rho=hp.get("rho", ""),
                    ls=hp.get("ls", ""), lam_reg=hp.get("lam_reg", ""), meta_lr=hp.get("meta_lr", ""),
                    meta_seed=hp.get("meta_seed", "-"))

    def finish(row, lg, evals, res_path, ckpt=None, meta=None, expected_inits=None):
        row["attempts_ok"], row["attempts_failed"], row["oom_retries"] = lg["ok"], lg["fail"], lg["oom_retries"]
        row["last_error"] = lg["last_error"]
        if evals and len(evals) == expected_inits:
            row["status"] = "PASSED_AFTER_RETRY" if (lg["fail"] or lg["oom_retries"]) else "PASSED"
        elif lg["fail"]:
            row["status"] = "FAILED"
        else:
            row["status"] = "MISSING"
        row["result_json"] = str(res_path.relative_to(D["root"])) if res_path and res_path.exists() else ""
        if ckpt is not None:
            row["checkpoint"] = str(ckpt.relative_to(D["root"])) if ckpt.exists() else ""
            row["checkpoint_exists"] = ckpt.exists()
        else:
            row["checkpoint"], row["checkpoint_exists"] = "(none: AdamW cells keep no checkpoint)", ""
        if meta:
            row["meta_train_stages"] = len(meta["history"])
            row["meta_ckpt_saves"] = sum(bool(h["saved"]) for h in meta["history"])
            row["best_ckpt_score"] = meta["best_ckpt_score"]
            row["meta_train_seconds"] = meta["meta_train_seconds"]
        return row

    for name, ph, path in trials:
        if ph == "m1":
            d = name.split("_", 1)[1]
            res = json.load(open(path)) if path.exists() else {"rows": []}
            done = {r["lr"]: r for r in res["rows"]}
            for lr in cfg.adamw_lrs:
                tid = f"m1_{d}_lr{RS.g(lr)}"
                row = base(tid, ph, "M1", d, adamw_lr=lr)
                ev = done[lr]["evals"] if lr in done else []
                row.update(eval_cols(ev, targets_of(d)) if ev else {})
                finish(row, logs[name], ev, path, expected_inits=cfg.adamw_inits)
                row["best_pick"] = "M1" if pick_m1.get(d) == lr else ""
                rows.append(row)
                for i, e in enumerate(ev):
                    for c in e.get("curve", []):
                        curves.append(dict(trial=tid, init=i, step=c["step"], train_loss=c["train_loss"],
                                           val_loss=c["val_loss"], val_acc=c["val_acc"]))
        elif ph == "m2":
            j = json.load(open(path)) if path.exists() else None
            d, reg = name[3:].split("_")[0], name[3:].split("_")[1]
            m = re.search(r"_lam([0-9.e-]+?)(?:_rho([0-9.e-]+))?$", name[3:])
            lam, rho = float(m.group(1)), (float(m.group(2)) if m.group(2) else "")
            row = base(name, ph, "M2", d, reg=reg, adamw_lr=j["adamw_lr"] if j else "", lam=lam, rho=rho)
            ev = j["evals"] if j else []
            row.update(eval_cols(ev, targets_of(d)) if ev else {})
            finish(row, logs[name], ev, path, expected_inits=cfg.adamw_inits)
            pk = pick_m2.get((d, reg))
            row["best_pick"] = f"M2:{reg}" if pk and abs(pk[0] - lam) < 1e-12 and (
                pk[1] in (None, "") or rho == "" or abs(pk[1] - rho) < 1e-12) else ""
            rows.append(row)
            for i, e in enumerate(ev):
                for c in e.get("curve", []):
                    curves.append(dict(trial=name, init=i, step=c["step"], train_loss=c["train_loss"],
                                       val_loss=c["val_loss"], val_acc=c["val_acc"]))
        else:                                           # learned-optimizer trial
            j = json.load(open(path)) if path.exists() else None
            m = re.match(r"(\w+?)_(none|l1|l2|sam|gsam|gam)_ls([0-9.e-]+)_lr([0-9.e-]+)_mlr([0-9.e-]+)_seed(\d+)$", name)
            d, reg, ls, lr, mlr, seed = m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6))
            method = "M3" if (ls == 0 and lr == 0) else "M4a" if lr == 0 else "M5"
            row = base(name, ph, method, d, reg=reg if lr else "-", ls=ls, lam_reg=lr, meta_lr=mlr, meta_seed=seed)
            ev = j["evals"] if j else []
            row.update(eval_cols(ev, targets_of(d)) if ev else {})
            ck = D["ckpt"] / f"{name}.pt"
            finish(row, logs[name], ev, path, ckpt=ck, meta=j["meta_train"] if j else None, expected_inits=cfg.lo_inits)
            row["best_pick"] = pick_lo.get(f"{name}.json", "")
            rows.append(row)
            for i, e in enumerate(ev):
                for c in e.get("curve", []):
                    curves.append(dict(trial=name, init=i, step=c["step"], train_loss=c["train_loss"],
                                       val_loss=c["val_loss"], val_acc=c["val_acc"]))
            if j:
                for h in j["meta_train"]["history"]:
                    metas.append(dict(trial=name, stage=h["stage"], block=h["block"], ckpt_score=h["ckpt_score"],
                                      saved=h["saved"], elapsed_s=h["elapsed"]))

    # ------------------------------------------------ result files that are NOT in the current plan
    planned_files = {p for _, _, p in trials}
    for sub, method in (("m2", "M2"), ("lo", "LO")):
        for f in sorted(D[sub].glob("*.json")):
            if f in planned_files:
                continue
            j = json.load(open(f))
            d = j["dataset"]
            row = base(("m2_" if sub == "m2" else "") + f.stem, "(old plan)", method if sub == "m2" else
                       ("M3" if j["ls"] == 0 and j["lr"] == 0 else "M4a" if j["lr"] == 0 else "M5"), d,
                       reg=j["reg"] if j["reg"] != "none" else "-", adamw_lr=j.get("adamw_lr", ""), lam=j.get("lam", ""),
                       rho=j.get("rho") or "", ls=j.get("ls", ""), lam_reg=j.get("lr", ""), meta_lr=j.get("mlr", ""),
                       meta_seed=j.get("seed", "-"))
            ev = j["evals"]
            row.update(eval_cols(ev, targets_of(d)))
            lg = logs[f.stem if sub == "lo" else f"m2_{f.stem}"]
            finish(row, lg, ev, f, ckpt=(D["ckpt"] / f"{f.stem}.pt") if sub == "lo" else None,
                   meta=j.get("meta_train"), expected_inits=len(ev))
            row["status"] = "EXTRA_NOT_IN_CURRENT_GRID"
            row["best_pick"] = ""
            rows.append(row)

    # ------------------------------------------------ write
    key = ["trial", "phase", "method", "source", "reg", "adamw_lr", "lam", "rho", "ls", "lam_reg", "meta_lr", "meta_seed",
           "status", "best_pick", "attempts_ok", "attempts_failed", "oom_retries", "last_error",
           "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc"]
    cols = key + [c for c in dict.fromkeys(k for r in rows for k in r) if c not in key]
    with open(out / "trials_full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(rows)
    with open(out / "trials_curves.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "init", "step", "train_loss", "val_loss", "val_acc"])
        w.writeheader()
        w.writerows(curves)
    with open(out / "trials_meta.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trial", "stage", "block", "ckpt_score", "saved", "elapsed_s"])
        w.writeheader()
        w.writerows(metas)

    # ---- html (filterable)
    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) else html.escape(str(v))
    shown = [c for c in cols if not c.startswith(("val_acc_init", "train_loss_init", "ood_"))] + \
            [c for c in cols if c.startswith("ood_") and c.endswith("auroc")]
    colour = {"PASSED": "#e6f4ea", "PASSED_AFTER_RETRY": "#fff4d6", "FAILED": "#fde0dd", "MISSING": "#fde0dd",
              "EXTRA_NOT_IN_CURRENT_GRID": "#eceff1"}
    body = "".join(
        f"<tr style='background:{colour.get(r['status'], '#fff')}{';font-weight:700' if r.get('best_pick') else ''}'>"
        + "".join(f"<td>{fmt(r.get(c, ''))}</td>" for c in shown) + "</tr>" for r in rows)
    (out / "trials_full.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Trial table</title><style>body{font:12px sans-serif;margin:8px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 5px;white-space:nowrap}"
        "th{position:sticky;top:0;background:#f0efec}input{width:100%}</style>"
        "<p>Filter any column (substring). Bold = best pick of its method. Green PASSED, yellow PASSED_AFTER_RETRY "
        "(OOM then retried OK), red FAILED/MISSING, grey = result from an earlier plan, not in the current grid.</p>"
        "<table id=t><thead><tr>" + "".join(f"<th>{c}</th>" for c in shown) + "</tr><tr>"
        + "".join("<th><input oninput='f()'></th>" for _ in shown) + "</tr></thead><tbody>" + body + "</tbody></table>"
        "<script>function f(){const q=[...document.querySelectorAll('thead input')].map(i=>i.value.toLowerCase());"
        "document.querySelectorAll('#t tbody tr').forEach(r=>{const c=r.children;let ok=true;"
        "q.forEach((v,i)=>{if(v&&!c[i].textContent.toLowerCase().includes(v))ok=false});r.style.display=ok?'':'none'})}"
        "</script>")

    # ---- status summary
    cnt = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cnt[(r["source"], r["method"])][r["status"]] += 1
    sts = ["PASSED", "PASSED_AFTER_RETRY", "FAILED", "MISSING", "EXTRA_NOT_IN_CURRENT_GRID"]
    md = [f"# Trial status — preset `{a.preset}`\n", f"{len(rows)} rows in trials_full.csv "
          f"({sum(r['status'] != 'EXTRA_NOT_IN_CURRENT_GRID' for r in rows)} planned trials, "
          f"{sum(r['status'] == 'EXTRA_NOT_IN_CURRENT_GRID' for r in rows)} older results outside the current grid).\n",
          "| source | method | planned | " + " | ".join(sts) + " |", "|---|---|---|" + "---|" * len(sts)]
    for (d, m), c in sorted(cnt.items()):
        md.append(f"| {d} | {m} | {sum(v for k, v in c.items() if k != 'EXTRA_NOT_IN_CURRENT_GRID')} | "
                  + " | ".join(str(c.get(s, 0)) for s in sts) + " |")
    bad = [r for r in rows if r["status"] in ("FAILED", "MISSING")]
    retry = [r for r in rows if r["status"] == "PASSED_AFTER_RETRY"]
    md += ["", f"## FAILED or MISSING trials: {len(bad)}"] + [f"- {r['trial']} ({r['status']}) {r['last_error']}" for r in bad]
    known = {r["trial"] for r in rows} | {f"m1_{d}" for d in SC.DATASETS}
    orphan = {n: l for n, l in logs.items() if l["fail"] and n not in known}
    md += ["", f"## Failed in a worker log, never completed, and no longer part of the plan: {len(orphan)}",
           "(cells of an earlier local block whose centre moved after the axis was completed; nothing is missing from the current grid)"]
    md += [f"- {n}: failed attempts={l['fail']} ({l['last_error']})" for n, l in sorted(orphan.items())]
    md += ["", f"## Passed only after a failed attempt / OOM retry: {len(retry)}"] + \
          [f"- {r['trial']}: failed attempts={r['attempts_failed']}, oom_retries={r['oom_retries']}" for r in retry]
    (out / "trial_status.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:30]))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
