"""Stage 4 reporting: collection, seed statistics, round views, the result cubes, the Stage-1
benchmark, the cross-evaluation of the rounds, the per-pair tables, figures.

A fake study "main" is written exactly as the scripts would write it. On ConvNet-4 / SVHN —
round A's source — every variant with three seeds: M1, M2(r) trained with r (Stage 1), and M3 /
M4(r) with the optimizers of both rounds (cross-evaluation). On the other pairs (ResNet-18 /
SVHN, ConvNet-4 / FMNIST, MLP / FMNIST — round B's source, with its own M2 runs — and ResNet-18 /
CIFAR-10) L1 and SAM with two seeds. Plus audits, weight histograms, per-epoch metrics, the
Stage-1 arms of rounds A and B and a compute profile. Settings come from the real experiment
files and configs/rounds.yaml.
"""

import csv
import math
import random
from pathlib import Path

import pytest

from reglo.reporting import build_report
from reglo.reporting.collect import (
    Stat,
    aggregate,
    collect_meta,
    collect_runs,
    discover_meta_tags,
    discover_studies,
    round_view,
)
from reglo.reporting.tables import (
    benchmark_table,
    comparison_table,
    cube_table,
    fmt_number,
    meta_table,
    overview_table,
    settings_columns,
    transfer_summary,
    transfer_table,
    welch_p,
)
from reglo.rounds import load_rounds
from reglo.study import METRICS, Pair, study_settings
from reglo.utils.logging import MetricsWriter, write_json

REGS = ["l1", "l2", "sam", "gam", "gsam"]
SEEDS = [42, 43, 44]
TOP1 = {"m1_adamw": 0.90, "m2_adamw_reg": 0.91, "m3_lstm": 0.89, "m4_lstm_reg": 0.915}
EDGES = [x / 2 for x in range(-20, 3)]  # log10|w| from 1e-10 to 10, half decades
SOURCES = {"A": ("convnet4", "svhn"), "B": ("mlp", "fmnist")}
OTHER_PAIRS = [("resnet18", "svhn"), ("convnet4", "fmnist"), ("mlp", "fmnist"), ("resnet18", "cifar10")]
ACCURACY = next(m for m in METRICS if m.key == "accuracy")
GAP = next(m for m in METRICS if m.key == "gen_gap")
FLOPS = next(m for m in METRICS if m.key == "flops")
LABEL = {"m1_adamw": "M1", "m2_adamw_reg": "M2", "m3_lstm": "M3", "m4_lstm_reg": "M4"}


def _variants(regs=REGS, m2=True):
    """(method, regularizer, round) of one pair: M1, M2(r) if trained there, M3/M4(r) of both rounds."""
    yield "m1_adamw", "none", None
    if m2:
        for r in regs:
            yield "m2_adamw_reg", r, None
    for round_name in ("A", "B"):
        yield "m3_lstm", "none", round_name
        for r in regs:
            yield "m4_lstm_reg", r, round_name


def write_run(root: Path, method: str, reg: str, seed: int, rng: random.Random, status: str = "completed",
              study: str = "main", pair=("convnet4", "svhn"), epochs: int = 5, round_name=None) -> Path:
    tag = (method if reg == "none" else f"{method}-{reg}") + (f"@{round_name}" if round_name else "")
    run = root / "runs" / study / f"{pair[0]}-{pair[1]}" / tag / f"seed{seed}"
    passes = 2 if method == "m2_adamw_reg" and reg in ("sam", "gam", "gsam") else 1
    flops = 1.0e9 * passes + (0.2e9 if method in ("m3_lstm", "m4_lstm_reg") else 0.0)
    seen = round_name is not None and SOURCES[round_name] == tuple(pair)
    top1 = TOP1[method] + (0.01 if seen else 0.0) + rng.gauss(0, 0.002)  # the optimizers do best where they were trained
    sparse = 0.4 if reg == "l1" else 0.05
    final = {
        "status": status, "study": study, "optimizee": pair[0], "dataset": pair[1], "method": method,
        "label": LABEL[method], "regularizer": reg, "round": round_name, "seed": seed, "epochs_completed": epochs,
        "steps": 10000, "steps_per_epoch": 200, "test_loss": 0.35, "test_top1": top1 if status == "completed" else 0.1,
        "test_top5": 0.99, "train_loss": 0.05, "train_top1": 0.99, "gen_gap": 0.30, "gen_gap_top1": 0.99 - top1,
        "wall_clock_s": 600.0 * passes, "seconds_per_epoch": 12.0, "passes_per_step": passes, "flops_per_step": flops,
        "total_gflops": flops * 1e4 / 1e9, "converged_step": 4000,
        "weights": {"num_params": 1000, "num_penalised": 900, "l1": 100.0 - 40 * (reg == "l1"),
                    "l2": 5.0 - 2 * (reg == "l2"), "l1_all": 110.0, "l2_all": 5.5, "sparsity": sparse},
        "gpu": "Tesla V100-PCIE-32GB", "hostname": "gpu01",
    }
    write_json(run / "final.json", final)
    with MetricsWriter(run / "metrics.jsonl") as log:
        for epoch in range(1, epochs + 1):
            log.write(kind="epoch", epoch=epoch, train_loss_running=2.3 * math.exp(-epoch / (2 + passes)),
                      test_top1=top1 * epoch / epochs)
    counts = [0] * (len(EDGES) - 1)
    for _ in range(900):
        value = rng.gauss(-2.0 - 2.5 * (reg == "l1"), 0.6)
        counts[min(max(int((value - EDGES[0]) * 2), 0), len(counts) - 1)] += 1
    write_json(run / "weights.json", {"histogram": {"log10_edges": EDGES, "counts": counts}})
    flat = 1.0 - 0.3 * (method in ("m2_adamw_reg", "m4_lstm_reg") and reg in ("sam", "gam", "gsam"))
    ball = {"0.01": {"loss": 0.05, "max_loss": 0.05 + 0.2 * flat, "surrogate_gap": 0.2 * flat,
                     "grad_norm": 0.1, "max_grad_norm": 0.5 * flat}}
    diag = {"status": "completed", "study": study, "method": method, "regularizer": reg, "seed": seed,
            "hessian": {"final": {"lambda_max": 50.0 * flat, "trace": 900.0 * flat}},
            "sharpness": {"converged": ball, "final": ball},
            "weights": {snap: {"l1": final["weights"]["l1"], "l2": final["weights"]["l2"], "sparsity": sparse}
                        for snap in ("converged", "final")}}
    write_json(root / "diagnostics" / study / f"{pair[0]}-{pair[1]}" / tag / f"seed{seed}" / "diagnostics.json", diag)
    return run


def write_meta(root: Path, tag: str = "A", trajectories: int = 30, truncations: int = 4) -> None:
    rng = random.Random(1)
    for entry, reg in [("m3_lstm", "none")] + [("m4_lstm_reg", r) for r in REGS]:
        arm = entry if reg == "none" else f"{entry}-{reg}"
        folder = root / "meta_training" / tag / arm
        step = 0
        with MetricsWriter(folder / "meta_log.jsonl") as log:
            for t in range(trajectories):
                for k in range(truncations):
                    step += 1
                    l_meta = 2.3 * math.exp(-step / 60) + 0.2 + rng.gauss(0, 0.03)
                    log.write(kind="meta_step", trajectory=t, truncation=k, meta_step=step, l_meta=l_meta,
                              l_smooth=0.0 if reg == "none" else 1e-3 * rng.random(),
                              l_reg=0.0 if reg == "none" else 0.1 + 0.01 * rng.random(),
                              grad_norm=0.5 * rng.random() + 0.05, skipped=False, diverged=False)
                if (t + 1) % 5 == 0:
                    log.write(kind="validation", trajectory=t + 1, val_loss=1.0 / (t + 2), val_acc=0.5, improved=True)
        write_json(folder / "summary.json", {
            "status": "completed", "meta_training": entry, "regularizer": reg, "smooth": reg != "none", "round": tag,
            "trajectories": trajectories, "meta_steps": step, "best_trajectory": trajectories,
            "best_val_loss": 0.03, "best_val_acc": 0.5, "wall_clock_s": 7200.0})


def write_profile(root: Path, pair=("convnet4", "svhn")) -> None:
    rows = []
    for method, reg, round_name in _variants():
        if round_name == "B":
            continue  # timed once: the cost does not depend on whose φ
        passes = 2 if method == "m2_adamw_reg" and reg in ("sam", "gam", "gsam") else 1
        lstm = method in ("m3_lstm", "m4_lstm_reg")
        rows.append({"label": LABEL[method], "method": method, "regularizer": reg, "passes": passes,
                     "phi": "trained" if lstm else None, "flops_per_step": 1e9 * passes + 0.2e9 * lstm,
                     "seconds_per_step": 0.01 * passes + 0.003 * lstm, "steps_per_epoch": 200,
                     "flops_vs_m1": passes + 0.2 * lstm, "time_vs_m1": (0.01 * passes + 0.003 * lstm) / 0.01})
    write_json(root / "diagnostics" / "compute" / f"{pair[0]}-{pair[1]}.json", {
        "optimizee": pair[0], "dataset": pair[1], "batch_size": 128, "num_params": 1000,
        "lstm_flops_per_parameter": 26944, "device": "cuda:0", "gpu": "Tesla V100-PCIE-32GB",
        "warmup_steps": 20, "timed_steps": 100, "rows": rows})


@pytest.fixture(autouse=True)
def _production(monkeypatch):
    """The fake runs below have the production grid's three seeds."""
    monkeypatch.setenv("REGLO_PROFILE", "production")


@pytest.fixture
def tree(tmp_path):
    rng = random.Random(0)
    for method, reg, round_name in _variants():
        for seed in SEEDS:
            if (method, reg, round_name, seed) == ("m4_lstm_reg", "gsam", "A", 44):
                continue  # a missing run
            status = "diverged" if (method, round_name, seed) == ("m3_lstm", "A", 43) else "completed"
            write_run(tmp_path, method, reg, seed, rng, status=status, round_name=round_name)
    for pair in OTHER_PAIRS:  # the rest of the core design: L1 and SAM, two seeds (M2 with r on every pair)
        for method, reg, round_name in _variants(["l1", "sam"]):
            for seed in SEEDS[:2]:
                write_run(tmp_path, method, reg, seed, rng, pair=pair, round_name=round_name)
    write_meta(tmp_path, "A")
    write_meta(tmp_path, "B")
    write_profile(tmp_path)
    return tmp_path


RECORDS_ON_SOURCE_A = (1 + 5 + 2 * 6) * 3 - 1        # every variant, 3 seeds, one missing
RECORDS_ELSEWHERE = 4 * (1 + 2 + 2 * 3) * 2          # M1, M2 (L1, SAM), M3/M4 (L1, SAM) of both rounds


def test_collect_reads_every_run_with_its_round_and_audit(tree):
    assert discover_studies(tree) == ["main"]
    records = collect_runs(tree)
    assert len(records) == RECORDS_ON_SOURCE_A + RECORDS_ELSEWHERE
    rec = next(r for r in records if r["method"] == "m2_adamw_reg" and r["regularizer"] == "sam" and r["seed"] == 42
               and r["optimizee"] == "convnet4" and r["dataset"] == "svhn")
    assert rec["study"] == "main" and rec["audited"] and rec["lambda_max"] == pytest.approx(35.0) and rec["round"] == ""
    assert rec["surrogate_gap_converged"] == pytest.approx(0.14) and rec["gflops_per_step"] == pytest.approx(2.0)
    assert rec["ms_per_step"] == pytest.approx(1e3 * 1200 / 10000)
    assert {r["round"] for r in records if r["method"] == "m4_lstm_reg"} == {"A", "B"}


def test_aggregate_keeps_the_rounds_apart(tree):
    groups = aggregate(collect_runs(tree))
    assert len(groups) == (1 + 5 + 12) + 4 * (1 + 2 + 6)
    m3 = groups[("main", "convnet4", "svhn", "m3_lstm", "none", "A")]
    assert (m3.completed, m3.diverged) == (2, 1) and m3.get("test_top1").n == 2 and len(m3.runs) == 2
    assert groups[("main", "convnet4", "svhn", "m3_lstm", "none", "B")].completed == 3
    assert groups[("main", "convnet4", "svhn", "m4_lstm_reg", "gsam", "A")].completed == 2


def test_a_round_view_has_m2_with_r_on_every_pair_and_its_own_learned_optimizers(tree):
    groups = aggregate(collect_runs(tree))
    view = round_view(groups, "A")
    for pair in (("convnet4", "svhn"), ("resnet18", "svhn")):  # the source pair (Stage 1's runs) and any other
        m2 = view[("main", *pair, "m2_adamw_reg", "sam")]
        assert m2 is groups[("main", *pair, "m2_adamw_reg", "sam", "")]
        assert m2 is not view[("main", *pair, "m1_adamw", "none")] and m2.get("passes_per_step").mean == 2
    assert ("main", "resnet18", "svhn", "m2_adamw_reg", "l2") not in view  # no M2-L2 runs there
    assert view[("main", "convnet4", "svhn", "m4_lstm_reg", "sam")] is groups[
        ("main", "convnet4", "svhn", "m4_lstm_reg", "sam", "A")]


def test_stat_uses_the_sample_standard_deviation():
    s = Stat([1.0, 2.0, 3.0])
    assert s.mean == 2.0 and s.std == pytest.approx(1.0) and s.n == 3
    assert Stat([4.0]).std is None and Stat().mean is None  # one seed: no spread to report


# ───────────────────────────────────────────── Stage 4: the cubes, per round

def test_the_settings_of_round_a_are_the_figure_x_axis():
    settings = study_settings("main")
    assert list(settings) == ["A", "B"]
    assert list(settings["A"]) == ["seen_optimizee", "unseen_optimizee", "seen_dataset", "unseen_dataset"]
    core = settings_columns(settings["A"], extended=False)
    assert [(c.setting.key, c.pair.tag) for c in core] == [
        ("seen_optimizee", "convnet4-svhn"), ("unseen_optimizee", "mlp-svhn"), ("unseen_optimizee", "resnet18-svhn"),
        ("seen_dataset", "convnet4-svhn"), ("unseen_dataset", "convnet4-fmnist"), ("unseen_dataset", "convnet4-cifar10"),
        ("unseen_dataset", "mlp-fmnist"), ("unseen_dataset", "resnet18-fmnist"), ("unseen_dataset", "mlp-cifar10"),
        ("unseen_dataset", "resnet18-cifar10")]
    assert {c.setting.stage for c in core} == {2, 3}
    assert all(c.extended for c in settings_columns(settings["A"], extended=True) if c not in core)
    assert [c.pair.tag for c in settings_columns(settings["B"], False)][:2] == ["mlp-fmnist", "convnet4-fmnist"]


def test_cube_rows_are_regularizer_blocks_with_m2_trained_with_r(tree):
    groups = round_view(aggregate(collect_runs(tree)), "A")
    columns = settings_columns(study_settings("main")["A"], extended=False)
    table = cube_table(groups, "main", ACCURACY, columns, rnd=load_rounds()["A"])
    assert len(table.rows) == 5 * 4  # five regularizers × M1 · M2(r) · M3 · M4(r)
    assert [r[1] for r in table.rows[:4]] == ["M1 AdamW", "M2 AdamW + L1", "M3 LSTM", "M4 LSTM-L1 (ours)"]
    assert [r[0] for r in table.rows[::4]] == ["**L1**", "**L2**", "**SAM**", "**GAM**", "**GSAM**"]
    seen, unseen_opt = 2, 4  # cell columns after Regularizer, Method (MLP / SVHN comes first among the unseen)
    m1_cell, m2_cell = (float(table.rows[i][unseen_opt].strip("*").split()[0]) for i in (0, 1))
    assert m2_cell == pytest.approx(91.0, abs=0.5) and m1_cell == pytest.approx(90.0, abs=0.5)  # M2's own runs
    assert table.rows[3][seen].startswith("**")  # M4 best on the seen optimizee for L1
    assert table.rows[4 + 1][unseen_opt] == "–"  # no L2 runs on ResNet-18 / SVHN in this tree
    assert table.headers[2] == "Seen optimizee: ConvNet-4 / SVHN"
    assert "Unseen dataset: ResNet-18 / CIFAR-10 ‡" in table.headers and "Round A" in table.notes[0]
    tex = table.latex()
    assert "\\multicolumn{6}{c}{Unseen dataset}" in tex and "\\cmidrule(lr){7-12}" in tex
    assert tex.count("\\midrule") == 1 + 4  # under the header, then between the five blocks


def test_flops_cube_shows_ratios_to_m1_without_seed_spread(tree):
    groups = round_view(aggregate(collect_runs(tree)), "A")
    table = cube_table(groups, "main", FLOPS, settings_columns(study_settings("main")["A"], extended=False))
    m2 = [row for row in table.rows if row[1] == "M2 AdamW + SAM"][0]
    assert m2[2] == "2.000 (2.00×)" and m2[4] == "2.000 (2.00×)"  # r on: two passes, on every pair
    m4 = [row for row in table.rows if row[1] == "M4 LSTM-SAM (ours)"][0]
    assert m4[2] == "1.200 (1.20×)" and "±" not in m4[2]


# ───────────────────────────────────────────── Stage 1 benchmark and cross-evaluation

def test_the_benchmark_compares_m2_with_r_against_m1_on_the_source(tree):
    trained = round_view(aggregate(collect_runs(tree)), "A")
    table = benchmark_table(trained, "main", load_rounds()["A"])
    labels = [row[0] for row in table.rows]
    assert labels[:2] == ["M1 AdamW", "M2 AdamW + L1"] and labels[6] == "M3 LSTM" and len(labels) == 12
    flops = table.headers.index("GFLOPs/step")
    assert table.rows[labels.index("M2 AdamW + SAM")][flops] == "2.000"  # r on: two passes per step
    delta = table.headers.index("Δ top-1 vs M1 (pp)")
    assert float(table.rows[1][delta].split()[0]) == pytest.approx(1.0, abs=0.5) and table.rows[0][delta] == "–"
    assert benchmark_table(round_view(aggregate(collect_runs(tree)), "B"), "main",
                           load_rounds()["B"]).rows[1][0] == "M2 AdamW + L1"


def test_the_cross_evaluation_tables(tree):
    groups = aggregate(collect_runs(tree))
    rounds = list(load_rounds().values())
    pairs = [Pair("convnet4", "svhn"), Pair("mlp", "fmnist"), Pair("resnet18", "cifar10"), Pair("mlp", "cifar10")]
    table = transfer_table(groups, "main", rounds, pairs, ACCURACY)
    assert table.headers[:3] == ["Optimizee / dataset", "Role for A", "Role for B"]
    assert table.headers[3:6] == ["M1", "M3·A", "M3·B"]
    rows = {row[0]: row for row in table.rows}
    assert set(rows) == {"ConvNet-4 / SVHN", "MLP / FMNIST", "ResNet-18 / CIFAR-10"}  # no runs on MLP / CIFAR-10
    assert rows["ConvNet-4 / SVHN"][1:3] == ["seen", "unseen optimizee + dataset"]
    summary = transfer_summary(groups, "main", rounds, pairs, ACCURACY)
    by = {(row[0], row[1]): row for row in summary.rows}
    own = summary.headers.index("Source pair: own − other rounds")
    for round_name in ("A", "B"):  # each round's optimizers do best on the pair they were meta-trained on
        assert float(by[("M4-SAM", round_name)][own]) == pytest.approx(1.0, abs=0.5)
    seen = summary.headers.index("seen (vs M1)")
    assert by[("M3", "B")][seen].startswith("–") is False and "(n=1)" in by[("M3", "B")][seen]
    assert transfer_table(groups, "main", rounds[:1], pairs, ACCURACY) is None  # one round: nothing to cross


def test_build_report_writes_rounds_benchmark_transfer_and_flags_missing_runs(tree):
    written = build_report(tree, figures=True)
    names = {p.relative_to(tree).as_posix() for p in written["tables"] + written["figures"]}
    for round_name in ("A", "B"):
        for metric in ("accuracy", "gen_gap", "flops"):
            for ext in ("csv", "md", "tex"):
                assert f"tables/main/round_{round_name}/cube_{metric}.{ext}" in names
            assert f"figures/main/round_{round_name}/cube_{metric}.png" in names
        assert f"tables/main/round_{round_name}/benchmark.md" in names
        assert f"tables/main/round_{round_name}/cube_long.csv" in names
        assert f"tables/meta_training/{round_name}.md" in names
        assert f"figures/meta_training/{round_name}/convergence.png" in names
    assert not any("extended" in n for n in names)
    for reg in REGS:
        for ext in ("csv", "md", "tex"):
            assert f"tables/main/round_A/convnet4-svhn/{reg}.{ext}" in names
    for expected in ("tables/main/summary.md", "tables/main/runs.csv", "tables/overview.tex",
                     "tables/main/transfer_accuracy.md", "tables/main/transfer_summary_gen_gap.csv",
                     "tables/compute/convnet4-svhn.md",
                     "figures/main/round_A/convnet4-svhn/training_curves.png",
                     "figures/main/round_B/resnet18-cifar10/training_curves.png",
                     "figures/main/round_A/convnet4-svhn/pareto_flops.png", "figures/main/round_A/convnet4-svhn/pareto_time.pdf",
                     "figures/main/round_A/convnet4-svhn/poc.png", "figures/main/round_A/convnet4-svhn/weights.png",
                     "figures/meta_training/A/terms.png"):
        assert expected in names, expected
    assert all(p.stat().st_size > 1000 for p in written["figures"])
    summary = (tree / "tables" / "main" / "summary.md").read_text()
    for heading in ("## Stage 1 ·", "### Round A — source ConvNet-4 / SVHN", "### Round B — source MLP / FMNIST",
                    "## Stage 2 · Optimizer evaluation", "## Stage 3 · Optimizee evaluation",
                    "## Stage 4 · Evaluation metrics", "## Cross-evaluation of the rounds"):
        assert heading in summary, heading
    # M1 (2 of 3 seeds) + M2 (L1, SAM: 4 of 15) + round A's M3/M4 (L1, SAM: 6 of 18)
    assert "| A | Unseen optimizee | ResNet-18 / SVHN | unseen optimizee | 12 / 36 | stage2_optimizer_evaluation |" in summary
    assert "missing: convnet4-svhn/m4_lstm_reg-gsam@A/seed44" in summary and "diverged:" in summary
    assert "missing: mlp-svhn/m1_adamw/seed42" in summary  # declared by the grids, not run in this tree
    with open(tree / "tables" / "main" / "round_A" / "cube_long.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["metric"] for r in rows} == {"accuracy", "gen_gap", "flops"} and {r["round"] for r in rows} == {"A"}
    assert {r["setting"] for r in rows} == {"seen_optimizee", "unseen_optimizee", "seen_dataset", "unseen_dataset"}
    assert "same_runs_as" not in rows[0] and {r["method"] for r in rows} >= {"m2_adamw_reg"}


def test_extended_pairs_get_their_own_cubes_once_they_have_results(tree):
    for method, reg, round_name in _variants(["sam"], m2=False):
        if round_name != "B":
            write_run(tree, method, reg, 42, random.Random(9), pair=("wrn28", "cifar100"), round_name=round_name)
    names = {p.relative_to(tree).as_posix() for p in build_report(tree, figures=False)["tables"]}
    assert "tables/main/round_A/cube_accuracy_extended.md" in names
    assert "tables/main/round_B/cube_accuracy_extended.md" not in names  # the extended grid is round A's only
    text = (tree / "tables" / "main" / "round_A" / "cube_accuracy_extended.md").read_text()
    assert "WRN-28 / CIFAR-100 ‡ †" in text and "ViT-Lite" not in text  # declared but without results: left out


# ───────────────────────────────────────────── per-pair detail

def test_comparison_table_rows_bold_and_formats(tree, tmp_path):
    groups = round_view(aggregate(collect_runs(tree)), "A")
    table = comparison_table(groups, "main", "convnet4", "svhn", "l1", round_name="A")
    assert [row[0] for row in table.rows] == ["M1 AdamW", "M2 AdamW + L1", "M3 LSTM", "M4 LSTM-L1 (ours)"]
    assert table.title.endswith("round A")
    top1 = table.headers.index("Top-1 (%)")
    assert table.rows[3][top1].startswith("**") and " ± " in table.rows[3][top1]  # M4 best here, 3 seeds
    assert table.rows[2][1] == "2 (+1 div.)"
    sparsity = table.headers.index("Sparsity (%)")
    assert table.rows[1][sparsity].startswith("**40.0")  # L1-targeted rows are sparser
    tex = table.latex()
    assert "\\toprule" in tex and "\\textbf{" in tex and "$\\pm$" in tex and "±" not in tex
    paths = table.write(tmp_path / "out" / "l1")
    with open(paths[0], newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["row"] for r in rows] == ["M1", "M2", "M3", "M4"] and rows[0]["round"] == "A"
    assert float(rows[0]["test_top1_mean"]) == pytest.approx(0.90, abs=0.01)


def test_away_from_the_source_m2_has_runs_of_its_own(tree):
    groups = round_view(aggregate(collect_runs(tree)), "A")
    table = comparison_table(groups, "main", "resnet18", "svhn", "sam", round_name="A")
    assert table.rows[1][1] == "2" and not any("only on the round's source pair" in n for n in table.notes)


def test_property_columns_follow_the_regularizer(tree):
    groups = round_view(aggregate(collect_runs(tree)), "A")
    headers = {r: comparison_table(groups, "main", "convnet4", "svhn", r).headers for r in REGS}
    assert "Sparsity (%)" in headers["l1"] and "‖θ‖₂" in headers["l2"]
    assert "Max loss (compl.)" in headers["sam"] and "Tr(H)" in headers["gam"]
    assert "Surrogate gap (conv.)" in headers["gsam"]
    assert comparison_table(groups, "main", "convnet4", "cifar100", "sam") is None


def test_overview_and_welch(tree):
    groups = aggregate(collect_runs(tree))
    table = overview_table(groups, ["main"])
    assert len(table.rows) == 2 * 5  # two rounds × five pairs with M4 runs
    first = table.rows[0]
    assert first[:3] == ["main", "A", "ConvNet-4 / SVHN"] and first[3].startswith("+")  # M4 above M1 for L1
    assert table.headers[3:6] == ["L1: vs M1", "L1: vs M2", "L1: vs M3"] and first[4].startswith("+")
    assert welch_p([1.0, 1.1, 0.9], [2.0, 2.1, 1.9]) < 0.01
    assert welch_p([1.0], [2.0, 2.1]) is None


def test_fmt_number():
    assert fmt_number(None) == "–" and fmt_number(float("nan")) == "–"
    assert fmt_number(12345.6) == "12,346" and fmt_number(0.123456, 3) == "0.123"
    assert fmt_number(0.00001, 2) == "1.0e-05" and fmt_number(1.5, 2, signed=True) == "+1.50"


# ───────────────────────────────────────────── Stage 1 meta-training

def test_meta_collection_and_table(tree):
    assert discover_meta_tags(tree) == ["A", "B"]
    arms = collect_meta(tree, "A")
    assert [a["arm"] for a in arms] == ["m3_lstm"] + [f"m4_lstm_reg-{r}" for r in REGS]
    table = meta_table(arms, "A", "ConvNet-4 / SVHN")
    assert table.title == "Stage 1 meta-training, round A (ConvNet-4 / SVHN)"
    smooth, reg = table.headers.index("L_smooth (end)"), table.headers.index("L_reg (end)")
    assert table.rows[0][0] == "M3 · φ" and table.rows[0][smooth] == table.rows[0][reg] == "–"  # M3 uses neither
    assert table.rows[1][smooth] != "–" and table.rows[1][reg] != "–"
    assert table.rows[3][0] == "M4 · φ*_SAM" and table.rows[3][2] == "30"


def test_resumed_meta_logs_count_each_meta_step_once(tmp_path):
    folder = tmp_path / "meta_training" / "A" / "m3_lstm"
    with MetricsWriter(folder / "meta_log.jsonl") as log:
        for t, k, step in [(0, 0, 1), (0, 1, 2), (1, 0, 3), (1, 1, 4), (2, 0, 5), (2, 1, 6),  # killed in trajectory 2
                           (2, 0, 5), (2, 1, 6), (3, 0, 7), (3, 1, 8)]:                     # resumed after 2
            log.write(kind="meta_step", trajectory=t, truncation=k, meta_step=step, l_meta=1.0, grad_norm=0.1,
                      skipped=False, diverged=False)
        for t in (1, 2, 2, 3):
            log.write(kind="validation", trajectory=t, val_loss=1.0)
    arm = collect_meta(tmp_path, "A")[0]
    assert [r["meta_step"] for r in arm["steps"]] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [r["trajectory"] for r in arm["validation"]] == [1, 2, 3]


# ───────────────────────────────────────────── selection and edge cases

def test_build_report_without_figures_and_on_an_empty_tree(tree, tmp_path):
    written = build_report(tree, figures=False)
    assert written["figures"] == [] and written["tables"]
    empty = build_report(tmp_path / "nothing")
    assert empty == {"tables": [], "figures": []}


def test_hand_started_runs_stay_out_of_the_default_report(tree):
    write_run(tree, "m1_adamw", "none", 42, random.Random(3), study="adhoc", pair=("convnet4", "svhn"))
    write_meta(tree, tag="debug", trajectories=2, truncations=2)
    names = {p.relative_to(tree).as_posix() for p in build_report(tree, figures=False)["tables"]}
    assert not any("adhoc" in n or "debug" in n for n in names)
    names = {p.relative_to(tree).as_posix() for p in build_report(tree, ["adhoc"], ["debug"], figures=False)["tables"]}
    assert "tables/adhoc/summary.md" in names and "tables/meta_training/debug.md" in names


def test_a_smoke_report_says_it_is_not_a_result(tree, monkeypatch):
    summary = tree / "tables" / "main" / "summary.md"
    build_report(tree, figures=False)
    assert "(production profile)" in summary.read_text() and "a check that every stage runs" not in summary.read_text()
    monkeypatch.setenv("REGLO_PROFILE", "smoke")
    build_report(tree, figures=False)
    assert "(smoke profile)" in summary.read_text() and "a check that every stage runs" in summary.read_text()


def test_a_grid_without_regularizers_gets_an_m1_vs_m3_table(tmp_path):
    rng = random.Random(5)
    write_run(tmp_path, "m1_adamw", "none", 42, rng, study="pilot")
    write_run(tmp_path, "m3_lstm", "none", 42, rng, study="pilot", round_name="A")
    written = build_report(tmp_path, figures=False)
    names = {p.relative_to(tmp_path).as_posix() for p in written["tables"]}
    assert "tables/pilot/round_A/convnet4-svhn/none.md" in names and not any("cube" in n for n in names)
    summary = (tmp_path / "tables" / "pilot" / "summary.md").read_text()
    assert "M1 against M3" in summary and "M3 − M1:" in summary


def test_cells_show_partial_seed_counts_and_do_not_rank_one_seed_against_three(tree):
    import os

    for seed in (43, 44):  # leave only seed 42 of M2-SAM audited
        os.remove(tree / "diagnostics" / "main" / "convnet4-svhn" / "m2_adamw_reg-sam" / f"seed{seed}" / "diagnostics.json")
    groups = round_view(aggregate(collect_runs(tree)), "A")
    table = comparison_table(groups, "main", "convnet4", "svhn", "sam")
    lam = table.headers.index("λ_max")
    assert table.rows[1][lam] == "35.00 (n=1)"
    assert not any(row[lam].startswith("**") for row in table.rows)
    with open(table.write(tree / "check" / "sam")[0], newline="") as fh:
        row = list(csv.DictReader(fh))[1]
    assert row["lambda_max_n"] == "1" and row["lambda_max_std"] == ""


def test_table_files_keep_dotted_names(tree, tmp_path):
    groups = round_view(aggregate(collect_runs(tree)), "A")
    table = comparison_table(groups, "main", "convnet4", "svhn", "l2")
    a = table.write(tmp_path / "out" / "lam0.1")
    b = table.write(tmp_path / "out" / "lam0.5")
    assert {p.name for p in a + b} == {f"lam0.{x}.{e}" for x in (1, 5) for e in ("csv", "md", "tex")}


def test_latex_escapes_study_names_and_comparison_signs(tree):
    groups = aggregate(collect_runs(tree))
    renamed = {("main_rerun", *k[1:]): g for k, g in groups.items()}
    tex = overview_table(renamed, ["main_rerun"]).latex()
    assert "main\\_rerun" in tex and "p $<$ 0.05" in tex


def test_training_curves_keep_the_last_attempt_after_a_resume(tmp_path):
    from reglo.reporting.collect import epoch_curves

    run = tmp_path / "run"
    with MetricsWriter(run / "metrics.jsonl") as log:
        for epoch, loss in [(1, 2.0), (2, 1.5), (3, 1.4), (2, 1.2), (3, 1.0), (4, 0.8)]:  # killed after 3, resumed from 1
            log.write(kind="epoch", epoch=epoch, train_loss_running=loss)
    epochs, losses = epoch_curves([str(run)])
    assert epochs == [1.0, 2.0, 3.0, 4.0] and losses == [2.0, 1.2, 1.0, 0.8]
