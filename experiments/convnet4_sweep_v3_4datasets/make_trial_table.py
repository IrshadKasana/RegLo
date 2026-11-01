#!/usr/bin/env python
"""
make_trial_table.py -- the EVIDENCE RECORD of the sweep: one row per planned trial, with its outcome.

Reads only files the sweep already wrote (nothing is re-run):
    results/<preset>/m1|m2|lo/*.json     per-trial results (every init: losses, accuracies, ID AUROC, OOD, curves, provenance)
    results/<preset>/ckpt/*.pt           learned-optimizer (LSTM) checkpoints
    results/<preset>/weights/**/*.pt     final optimizee weights of every trial (all inits)
    results/<preset>/logs/node*_p*.log   worker logs: "[worker] ok|FAIL <trial> (<s>s)" + OOM retry lines
    results/<preset>/best_config.json    the selected ("best pick") configuration of every method
and the *planned* trial list, regenerated from sweep_config.py through run_sweep.phase_jobs, so that trials that never
produced a result show up as MISSING / FAILED instead of silently not being there.

Outputs (results/<preset>/trial_table/):
    trials_full.csv        one row per trial (ids, hyper-parameters, status, attempts, losses, accuracies, ID AUROC, OOD, steps,
                           provenance, weights / checkpoint files, best-pick flag)
    trials_per_init.csv    one row per trial x init (init seed, all losses / accuracies / AUROCs / OOD, divergence, seconds)
    trials_ood.csv         one row per trial x OOD target (accuracy, loss, OOD AUROC; mean over inits)
    trials_curves.csv      one row per trial x init x curve point (step 0, 100, ..., 1000)
    trials_meta.csv        one row per learned-optimizer trial x meta-training stage (checkpoint score, meta-loss, saved?)
    coverage.csv           planned / ran / passed / failed per (source, method, regulariser) against the FULL Cartesian grid
    selection_trace.csv    how every best pick was chosen (candidates, runner-up, margin in SE units, validation-only flag)
    trial_status.md        pass / fail / missing counts and every non-clean trial
    trials_full.html       the main table, filterable in a browser
    DATA_DICTIONARY.md     every column, meaning, units

    python make_trial_table.py --preset full
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

import run_sweep as RS
import sweep_config as SC

LOG_RE = re.compile(r"^\[worker\] (ok|FAIL)\b.* (\S+) \((\d+)s\)\s*$")
OOM_RE = re.compile(r"^\[worker\] CUDA OOM, retry")


def parse_logs(D):
    """{log job name: dict(ok, fail, oom_retries, last_error)} from every worker log (logs are appended to)."""
    out = defaultdict(lambda: dict(ok=0, fail=0, oom_retries=0, last_error=""))
    for f in sorted(D["logs"].glob("node*_p*.log")):
        pending = 0
        for line in open(f, errors="replace"):
            if OOM_RE.match(line):
                pending += 1
                continue
            m = LOG_RE.match(line.rstrip("\n"))
            if not m:
                continue
            rec = out[m.group(2)]
            rec["oom_retries"] += pending
            pending = 0
            if m.group(1) == "ok":
                rec["ok"] += 1
            else:
                rec["fail"] += 1
                rec["last_error"] = re.sub(r"\s+", " ", line[line.find("(") + 1: line.rfind(")")].split("Tried")[0]).strip()
    return out


def mean(vs):
    vs = [v for v in vs if v is not None]
    return float(np.mean(vs)) if vs else float("nan")


def size(p: Path):
    return p.stat().st_size if p.exists() else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--datasets", nargs="+", default=list(SC.DATASETS))
    ap.add_argument("--regs", nargs="+", default=list(SC.REGS))
    a0 = ap.parse_args()
    a = RS.build_parser().parse_args(["report", "--preset", a0.preset, "--datasets", *a0.datasets, "--regs", *a0.regs])
    D = RS.dirs(a.preset)
    cfg = RS.make_cfg(a.preset)
    out = D["root"] / "trial_table"
    out.mkdir(exist_ok=True)
    logs = parse_logs(D)
    best = json.load(open(D["root"] / "best_config.json"))["results"]
    n_val = {}
    for d in a.datasets:
        p = D["splits"] / f"{d}.npz"
        if p.exists():
            n_val[d] = int(len(np.load(p)["val"]))

    # ------------------------------------------------ best-pick lookup (file stems of the selected trials)
    pick = {}
    for d in best:
        for k, e in best[d].items():
            for f in e.get("files", []):
                pick[Path(f).stem] = k
    m1_pick = {(d, best[d]["M1"]["config"]["adamw_lr"]) for d in best if "M1" in best[d]}
    m2_pick = {(d, k.split(":")[1], e["config"]["adamw_lr"], e["config"]["lam"], e["config"]["rho"])
               for d in best for k, e in best[d].items() if k.startswith("M2:")}

    # ------------------------------------------------ planned trials (every phase)
    trials, seen = [], set()
    for ph in RS.ORDER:
        for name, argv, path in RS.phase_jobs(a, ph, D):
            if path not in seen:
                seen.add(path)
                trials.append((name, ph, Path(path)))

    rows, per_init, ood_rows, curves, metas = [], [], [], [], []
    all_targets = list(SC.DATASETS)

    def trial_row(name, ph, path, planned=True):
        """-> row dict for one result file (or a stub when the file is missing)."""
        J = json.load(open(path)) if path.exists() else None
        stem = path.stem
        kind = path.parent.name                                   # m1 | m2 | lo
        if kind == "m1":
            tid, logname = f"m1_{stem}", f"m1_{stem}"
            d = stem.split("_")[0]
            lr = float(re.search(r"_lr([0-9.e-]+)$", stem).group(1))
            row = dict(method="M1", source=d, reg="-", adamw_lr=lr)
        elif kind == "m2":
            tid, logname = f"m2_{stem}", f"m2_{stem}"
            m = re.match(r"(\w+?)_(l1|l2|sam|gsam|gam)_lr([0-9.e-]+)_lam([0-9.e-]+)(?:_rho([0-9.e-]+))?$", stem)
            d, reg, lr, lam = m.group(1), m.group(2), float(m.group(3)), float(m.group(4))
            row = dict(method="M2", source=d, reg=reg, adamw_lr=lr, lam=lam, rho=float(m.group(5)) if m.group(5) else "")
        else:
            tid, logname = f"lo_{stem}", stem
            m = re.match(r"(\w+?)_(none|l1|l2|sam|gsam|gam)_ls([0-9.e-]+)_lr([0-9.e-]+)_mlr([0-9.e-]+)(?:_rho([0-9.e-]+))?_seed(\d+)$", stem)
            d, reg, ls, lr, mlr, rho, seed = m.group(1), m.group(2), float(m.group(3)), float(m.group(4)), float(m.group(5)), m.group(6), int(m.group(7))
            method = "M3" if (ls == 0 and lr == 0) else "M4a" if lr == 0 else "M5"
            row = dict(method=method, source=d, reg=reg if lr else "-", ls=ls, lam_reg=lr, meta_lr=mlr,
                       rho=float(rho) if rho else "", meta_seed=seed)
        row.update(trial=tid, phase=ph)
        lg = logs[logname]
        row.update(attempts_ok=lg["ok"], attempts_failed=lg["fail"], oom_retries=lg["oom_retries"], last_error=lg["last_error"])
        expected = cfg.adamw_inits if kind in ("m1", "m2") else cfg.lo_inits
        ev = J["evals"] if J else []
        if ev and len(ev) == expected:
            row["status"] = "PASSED_AFTER_RETRY" if (lg["fail"] or lg["oom_retries"]) else "PASSED"
        else:
            row["status"] = "FAILED" if lg["fail"] else "MISSING"
        if not planned and J:
            row["status"] = "EXTRA_NOT_IN_CURRENT_GRID"
        row["result_json"] = str(path.relative_to(D["root"])) if J else ""
        # best pick label
        if kind == "m1":
            row["best_pick"] = "M1" if (row["source"], row["adamw_lr"]) in m1_pick else ""
        elif kind == "m2":
            rho = row.get("rho") or None
            row["best_pick"] = f"M2:{row['reg']}" if (row["source"], row["reg"], row["adamw_lr"], row["lam"], rho) in m2_pick else ""
        else:
            row["best_pick"] = pick.get(stem, "")
        if not J:
            return row
        prov = J.get("prov", {})
        for k in ("start_utc", "end_utc", "host", "pbs_jobid", "gpu", "code_sha1", "wall_seconds"):
            row[k] = prov.get(k, "")
        targets = list(ev[0]["ood"].keys())
        for k in ("train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "id_auroc", "seconds"):
            row[k if k != "seconds" else "run_seconds"] = (sum(e[k] for e in ev) if k == "seconds" else mean([e[k] for e in ev]))
        row.update(n_inits=len(ev), diverged_inits=sum(bool(e.get("diverged")) for e in ev),
                   optimizee_steps=ev[0]["curve"][-1]["step"] if ev[0].get("curve") else "",
                   init_seeds=" ".join(str(e.get("init_seed", "")) for e in ev))
        for t in all_targets:
            if t in targets:
                row[f"ood_auroc_{t}"] = mean([e["ood"][t]["auroc_msp"] for e in ev])
                row[f"ood_acc_{t}"] = mean([e["ood"][t]["acc"] for e in ev])
        wp = D["root"] / J["weights"] if J.get("weights") else None
        row["weights_file"] = J.get("weights", "")
        row["weights_exists"] = bool(wp and wp.exists())
        row["weights_bytes"] = size(wp) if wp else ""
        if kind == "lo":
            ck = D["ckpt"] / f"{stem}.pt"
            row.update(checkpoint=str(ck.relative_to(D["root"])), checkpoint_exists=ck.exists(), checkpoint_bytes=size(ck))
            mt = J["meta_train"]
            hist = mt["history"]
            row.update(meta_train_stages=len(hist), meta_ckpt_saves=sum(bool(h["saved"]) for h in hist),
                       meta_ckpt_stage_used=max((h["stage"] for h in hist if h["saved"]), default=""),
                       best_ckpt_score=mt["best_ckpt_score"], meta_train_seconds=mt["meta_train_seconds"],
                       meta_unroll_steps_total=sum(h.get("unroll_steps", h["stage"]) for h in hist),
                       meta_loss_last_stage=hist[-1].get("meta_loss_mean", ""))
            for h in hist:
                metas.append(dict(trial=tid, stage=h["stage"], block=h["block"], unroll_steps=h.get("unroll_steps", h["stage"]),
                                  ckpt_score=h["ckpt_score"], saved=h["saved"], elapsed_s=h["elapsed"],
                                  meta_loss_mean=h.get("meta_loss_mean", ""), meta_loss_last=h.get("meta_loss_last", ""),
                                  n_windows=h.get("n_windows", "")))
        else:
            row.update(checkpoint="(none: AdamW cells have no learned parameters)", checkpoint_exists="")
        for i, e in enumerate(ev):
            pi = dict(trial=tid, init=i, init_seed=e.get("init_seed", ""), train_loss=e["train_loss"], train_acc=e["train_acc"],
                      val_loss=e["val_loss"], val_acc=e["val_acc"], test_loss=e["test_loss"], test_acc=e["test_acc"],
                      id_auroc=e["id_auroc"], l1_norm=e["l1_norm"], l2_sq=e["l2_sq"], diverged=e.get("diverged"),
                      seconds=e["seconds"])
            for t in targets:
                pi[f"ood_auroc_{t}"] = e["ood"][t]["auroc_msp"]
                pi[f"ood_acc_{t}"] = e["ood"][t]["acc"]
                pi[f"ood_loss_{t}"] = e["ood"][t]["loss"]
            per_init.append(pi)
            for c in e.get("curve", []):
                curves.append(dict(trial=tid, init=i, step=c["step"], train_loss=c["train_loss"],
                                   train_subset_loss=c.get("train_subset_loss", ""), train_subset_acc=c.get("train_subset_acc", ""),
                                   val_loss=c["val_loss"], val_acc=c["val_acc"]))
        for t in targets:
            ood_rows.append(dict(trial=tid, source=row["source"], target=t,
                                 ood_acc=mean([e["ood"][t]["acc"] for e in ev]), ood_loss=mean([e["ood"][t]["loss"] for e in ev]),
                                 ood_auroc=mean([e["ood"][t]["auroc_msp"] for e in ev])))
        return row

    for name, ph, path in trials:
        rows.append(trial_row(name, ph, path))
    planned_files = {p for _, _, p in trials}
    for sub in ("m1", "m2", "lo"):
        for f in sorted(D[sub].glob("*.json")):
            if f not in planned_files:
                rows.append(trial_row(f.stem, "(old plan)", f, planned=False))

    # ------------------------------------------------ write main tables
    def write(name, recs, key=()):
        cols = list(key) + [c for c in dict.fromkeys(k for r in recs for k in r) if c not in key]
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(recs)
        return cols

    key = ["trial", "phase", "method", "source", "reg", "adamw_lr", "lam", "rho", "ls", "lam_reg", "meta_lr", "meta_seed",
           "status", "best_pick", "attempts_ok", "attempts_failed", "oom_retries", "last_error", "n_inits", "optimizee_steps",
           "train_loss", "train_acc", "val_loss", "val_acc", "test_loss", "test_acc", "id_auroc"]
    cols = write("trials_full.csv", rows, key)
    write("trials_per_init.csv", per_init, ["trial", "init", "init_seed"])
    write("trials_ood.csv", ood_rows, ["trial", "source", "target"])
    write("trials_curves.csv", curves, ["trial", "init", "step"])
    write("trials_meta.csv", metas, ["trial", "stage"])

    # ------------------------------------------------ coverage against the full Cartesian grid
    G, L, M, R = len(cfg.grid), len(cfg.adamw_lrs), len(cfg.meta_lrs), {k: len(v) for k, v in RS.rho_grid(cfg).items()}
    cov = []
    for d in a.datasets:
        groups = [("M1", "-"), ("M3", "-"), ("M4a", "-")] + [("M2", r) for r in a.regs] + [("M5", r) for r in a.regs]
        for m, r in groups:
            rs = [x for x in rows if x["source"] == d and x["method"] == m and x["reg"] == r
                  and x["status"] != "EXTRA_NOT_IN_CURRENT_GRID" and x.get("meta_seed", "-") in ("-", 0)]
            rho_n = R.get(r, 1)
            full = {"M1": L, "M3": M, "M4a": M * G, "M2": L * G * rho_n, "M5": M * G * G * rho_n}[m]
            note = {"M1": "full grid", "M3": "full grid", "M4a": "full meta_lr x lambda_smooth grid",
                    "M2": ("full lr x lambda grid at the default rho, then rho x lambda at the best lr" if r in RS.RHO_REGS
                           else "full lr x lambda grid"),
                    "M5": (f"lambda_smooth x lambda_reg {'full grid' if cfg.m5_mode == 'full' else 'axis + block'} at M4a's meta_lr, then "
                           "meta_lr, then rho at the best cell")}[m]
            st = defaultdict(int)
            for x in rs:
                st[x["status"]] += 1
            ran = st["PASSED"] + st["PASSED_AFTER_RETRY"] + st["FAILED"]
            cov.append(dict(source=d, method=m, reg=r, full_cartesian_cells=full, planned=len(rs), ran=ran,
                            passed=st["PASSED"] + st["PASSED_AFTER_RETRY"], passed_after_retry=st["PASSED_AFTER_RETRY"],
                            failed=st["FAILED"], missing=st["MISSING"],
                            planned_pct_of_full=round(100.0 * len(rs) / full, 2) if full else "", sweep_design=note))
    write("coverage.csv", cov, ["source", "method", "reg"])

    # ------------------------------------------------ selection trace
    trace = []
    for d in a.datasets:
        for key_, e in best.get(d, {}).items():
            m, _, r = key_.partition(":")
            cand = [x for x in rows if x["source"] == d and x["method"] == m and x["status"].startswith("PASSED")
                    and x.get("meta_seed", "-") in ("-", 0) and (m in ("M1", "M3", "M4a") or x["reg"] == r)]
            if not cand:
                continue
            cand.sort(key=lambda x: -x["val_acc"])
            top = cand[0]
            se = math.sqrt(top["val_acc"] * (1 - top["val_acc"]) / n_val[d]) if d in n_val else float("nan")
            run = cand[1] if len(cand) > 1 else None
            by_test = sorted(cand, key=lambda x: -x["test_acc"])
            trace.append(dict(source=d, method=m, reg=r or "-", candidates=len(cand), selected_trial=top["trial"],
                              selected_val_acc=top["val_acc"], selected_test_acc=top["test_acc"],
                              runner_up_trial=run["trial"] if run else "", runner_up_val_acc=run["val_acc"] if run else "",
                              margin_val_acc=(top["val_acc"] - run["val_acc"]) if run else "",
                              margin_in_val_SE=((top["val_acc"] - run["val_acc"]) / se) if run and se else "",
                              val_SE=se, rank_of_selected_by_test_acc=1 + [x["trial"] for x in by_test].index(top["trial"]),
                              best_test_acc_among_candidates=by_test[0]["test_acc"],
                              selection_criterion="seed-0 mean validation accuracy of the source dataset only",
                              matches_best_config_json=(top["best_pick"] == key_)))
    write("selection_trace.csv", trace, ["source", "method", "reg"])

    # ------------------------------------------------ status summary + html
    cnt = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cnt[(r["source"], r["method"])][r["status"]] += 1
    sts = ["PASSED", "PASSED_AFTER_RETRY", "FAILED", "MISSING", "EXTRA_NOT_IN_CURRENT_GRID"]
    npl = sum(r["status"] != "EXTRA_NOT_IN_CURRENT_GRID" for r in rows)
    md = [f"# Trial status — preset `{a.preset}`\n", f"{len(rows)} rows in trials_full.csv ({npl} planned trials, "
          f"{len(rows) - npl} older results outside the current grid).\n",
          "| source | method | planned | " + " | ".join(sts) + " |", "|---|---|---|" + "---|" * len(sts)]
    for (d, m), c in sorted(cnt.items()):
        md.append(f"| {d} | {m} | {sum(v for k, v in c.items() if k != 'EXTRA_NOT_IN_CURRENT_GRID')} | "
                  + " | ".join(str(c.get(s, 0)) for s in sts) + " |")
    bad = [r for r in rows if r["status"] in ("FAILED", "MISSING")]
    retry = [r for r in rows if r["status"] == "PASSED_AFTER_RETRY"]
    known = {r["trial"] for r in rows}
    orphan = {n: l for n, l in logs.items() if l["fail"] and (n if n.startswith(("m1_", "m2_")) else "lo_" + n) not in known}
    noweights = [r for r in rows if r["status"].startswith("PASSED") and not r.get("weights_exists")]
    md += ["", f"## FAILED or MISSING trials: {len(bad)}"] + [f"- {r['trial']} ({r['status']}) {r['last_error']}" for r in bad]
    md += ["", f"## Failed in a worker log, never completed, no longer part of the plan: {len(orphan)}"] + \
          [f"- {n}: failed attempts={l['fail']} ({l['last_error']})" for n, l in sorted(orphan.items())]
    md += ["", f"## Passed only after a failed attempt / OOD retry: {len(retry)}"] + \
          [f"- {r['trial']}: failed attempts={r['attempts_failed']}, oom_retries={r['oom_retries']}" for r in retry[:200]]
    md += ["", f"## Passed trials without a weights file: {len(noweights)}"]
    (out / "trial_status.md").write_text("\n".join(md) + "\n")

    shown = ["trial", "method", "source", "reg", "adamw_lr", "lam", "rho", "ls", "lam_reg", "meta_lr", "meta_seed", "status",
             "best_pick", "attempts_failed", "val_acc", "train_acc", "test_acc", "id_auroc", "optimizee_steps", "checkpoint_exists",
             "weights_exists"]
    colour = {"PASSED": "#e6f4ea", "PASSED_AFTER_RETRY": "#fff4d6", "FAILED": "#fde0dd", "MISSING": "#fde0dd",
              "EXTRA_NOT_IN_CURRENT_GRID": "#eceff1"}
    f4 = lambda v: f"{v:.4f}" if isinstance(v, float) else html.escape(str(v))
    body = "".join(f"<tr style='background:{colour.get(r['status'], '#fff')}{';font-weight:700' if r.get('best_pick') else ''}'>"
                   + "".join(f"<td>{f4(r.get(c, ''))}</td>" for c in shown) + "</tr>" for r in rows)
    (out / "trials_full.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Trial table</title><style>body{font:12px sans-serif;margin:8px}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 5px;white-space:nowrap}"
        "th{position:sticky;top:0;background:#f0efec}input{width:100%}</style>"
        "<p>Filter any column (substring). Bold = best pick of its method. Green PASSED, yellow PASSED_AFTER_RETRY, red FAILED/MISSING, "
        "grey = result of an earlier plan. Accuracies are fractions (0-1). Full column set: trials_full.csv.</p>"
        "<table id=t><thead><tr>" + "".join(f"<th>{c}</th>" for c in shown) + "</tr><tr>"
        + "".join("<th><input oninput='f()'></th>" for _ in shown) + "</tr></thead><tbody>" + body + "</tbody></table>"
        "<script>function f(){const q=[...document.querySelectorAll('thead input')].map(i=>i.value.toLowerCase());"
        "document.querySelectorAll('#t tbody tr').forEach(r=>{const c=r.children;let ok=true;"
        "q.forEach((v,i)=>{if(v&&!c[i].textContent.toLowerCase().includes(v))ok=false});r.style.display=ok?'':'none'})}</script>")

    (out / "DATA_DICTIONARY.md").write_text(DICT)
    print("\n".join(md[:40]))
    print(f"\nwrote {out}  ({len(rows)} rows, {len(per_init)} per-init rows, {len(curves)} curve points)")


DICT = """# Data dictionary (all accuracies / AUROCs are fractions 0-1; losses are mean NLL; times are seconds)

| column | meaning |
|---|---|
| trial | unique id: `m1_<ds>_lr<..>`, `m2_<ds>_<reg>_lr<..>_lam<..>[_rho<..>]`, `lo_<ds>_<reg>_ls<..>_lr<..>_mlr<..>[_rho<..>]_seed<..>` (= result file name with a method prefix) |
| phase | sweep phase that planned it: m1, m3, m4a, m5, m5blk, m5mlr, m5rho, m2, m2rho, final, (old plan) |
| method | M1 AdamW, M2 AdamW+lambda*R, M3 learned optimizer, M4a LO+smooth, M5 LO+smooth+R |
| source | dataset the model was trained / selected on (the ID dataset) |
| reg | regulariser R (l1, l2, sam, gsam, gam); `-` when none |
| adamw_lr | AdamW learning rate (M1/M2) |
| lam | M2 regularisation weight lambda |
| rho | perturbation radius of SAM / GSAM / GAM (M2, M5) |
| ls | lambda_smooth (M4a / M5); 0 for M3 |
| lam_reg | lambda_reg of M5 (the `lr` field of the lo files); 0 for M3 / M4a |
| meta_lr | Adam learning rate of the learned optimizer's own weights |
| meta_seed | meta-training seed of a learned-optimizer trial (0 = sweep, 1 / 2 = final seeds of the selected cells) |
| status | PASSED; PASSED_AFTER_RETRY (a failed / OOM attempt, then OK); FAILED; MISSING (never ran); EXTRA_NOT_IN_CURRENT_GRID |
| best_pick | method(:regulariser) this trial was selected for (validation accuracy of the source, seed 0) |
| attempts_ok / attempts_failed / oom_retries | counts parsed from the worker logs |
| n_inits, init_seeds | optimizee initialisations run (3) and their seeds |
| optimizee_steps | training steps of every optimizee (1000) |
| train_loss, train_acc | on the first 10 000 training images, final weights, mean over inits |
| val_loss, val_acc | validation split (20 %) of the source; the ONLY quantity selection uses |
| test_loss, test_acc | ID test split (10 %) of the source |
| id_auroc | macro one-vs-rest multiclass AUROC of the softmax output on the ID test split |
| ood_auroc_<t>, ood_acc_<t> | OOD AUROC (ID test vs target t's 20 % partition by max-softmax confidence) and accuracy under t's label ids |
| run_seconds | summed optimizee-run time of all inits |
| start_utc, end_utc, host, pbs_jobid, gpu, code_sha1, wall_seconds | provenance of the trial |
| weights_file / weights_exists / weights_bytes | final optimizee weights of all inits (`torch.load` -> init_seeds, states) |
| checkpoint / checkpoint_exists / checkpoint_bytes | learned-optimizer (LSTM) checkpoint of the best meta-training stage |
| meta_train_stages, meta_ckpt_saves, meta_ckpt_stage_used | curriculum stages run, stages that improved the checkpoint score, last stage saved |
| best_ckpt_score, meta_train_seconds, meta_unroll_steps_total, meta_loss_last_stage | meta-training summary |

Other tables: trials_per_init.csv (per init), trials_ood.csv (per OOD target), trials_curves.csv (per 100-step point incl. step 0),
trials_meta.csv (per meta-training stage), coverage.csv (planned vs the full Cartesian grid), selection_trace.csv (how each pick was chosen).
"""

if __name__ == "__main__":
    main()
