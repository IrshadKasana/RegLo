"""The Excel workbook of the experiment plan (scripts/make_workbook.py): structure, formulas, filling."""

import runpy

import pytest

openpyxl = pytest.importorskip("openpyxl")

from reglo.config import project_root  # noqa: E402
from reglo.reporting.workbook import SHEETS, build_workbook, fill_workbook, plan_study_workbook  # noqa: E402
from reglo.utils.logging import MetricsWriter, write_json  # noqa: E402


@pytest.fixture(scope="module")
def plan():
    return plan_study_workbook("production")


@pytest.fixture(scope="module")
def book(plan, tmp_path_factory):
    path = tmp_path_factory.mktemp("wb") / "plan.xlsx"
    build_workbook(path, "production", plan)
    return path


def test_the_plan_has_every_run_once(plan):
    by_file = {}
    for run in plan.runs:
        by_file[run.experiment] = by_file.get(run.experiment, 0) + 1
    assert by_file == {"stage1_meta_training": 30, "stage2_optimizer_evaluation": 147,
                       "stage3_optimizee_evaluation": 309, "extended_evaluation": 252, "pilot": 2,
                       "imagenet_evaluation": 54}
    assert len(plan.runs) == 740 + 54  # the core design, extended and pilot; plus ResNet-18 / ImageNet
    assert len({run.key for run in plan.runs}) == len(plan.runs) and plan.runs[0].run_id == "R0001"
    assert len(plan.arms) == 12 and {a.round for a in plan.arms} == {"A", "B"}
    assert plan.seeds == [42, 43, 44]


def test_the_experiment_rows_follow_the_design(plan):
    stage1 = [e for e in plan.exps if e.stage == "1"]
    assert len(stage1) == 2 * (6 + 1 + 5)  # per round: six LSTM optimizers, M1's reference, M2 with each r
    core = [e for e in plan.exps if e.stage in ("2", "3")]
    # per round: 3 Stage-2 pairs, 7 Stage-3 pairs (the seen pair is in both), 12 method rows each
    assert len(core) == 2 * (3 + 7) * 12
    m2 = [e for e in core if e.method == "m2_adamw_reg"]
    assert m2 and all(e.runs and all(r.method == "m2_adamw_reg" for r in e.runs) for e in m2)  # its own runs, with r
    on_source = [e for e in m2 if e.role == "seen"]  # the source pairs: Stage 1's runs, not trained again
    assert on_source and all(r.experiment == "stage1_meta_training" for e in on_source for r in e.runs)
    assert all(e.same_as.startswith("E1.") for e in on_source if e.setting_key == "seen_optimizee")
    sources = {(r.optimizee, r.dataset) for r in plan.rounds}
    elsewhere = [e for e in m2 if (e.pair.optimizee, e.pair.dataset) not in sources and not e.extended]
    assert {r.experiment for e in elsewhere for r in e.runs} == {"stage2_optimizer_evaluation",
                                                                "stage3_optimizee_evaluation"}
    m4b = next(e for e in core if e.round == "B" and e.method == "m4_lstm_reg" and e.reg == "sam"
               and e.pair.tag == "convnet4-svhn")
    assert m4b.role == "unseen_both" and m4b.setting_key == "unseen_dataset"
    assert [r.folder for r in m4b.runs] == [f"runs/main/convnet4-svhn/m4_lstm_reg-sam@B/seed{s}" for s in (42, 43, 44)]


def test_the_sheets_and_their_formulas(book, plan):
    wb = openpyxl.load_workbook(book)
    assert wb.sheetnames == list(SHEETS)
    runs = wb["Runs"]
    assert runs.cell(row=5, column=1).value == "Run ID" and runs.max_row == 5 + len(plan.runs)
    gap = runs.cell(row=6, column=20).value
    assert gap.startswith("=IF(AND(ISNUMBER(R6),ISNUMBER(S6))") and gap.endswith('S6-R6,"")')
    t4 = wb["T4 Experiments"]
    exp = next(e for e in plan.exps if e.stage == "2" and e.method == "m4_lstm_reg" and e.reg == "l1")
    row = [t4.cell(row=exp.row, column=c).value for c in range(1, 26)]
    first, last = exp.runs[0].row, exp.runs[-1].row
    assert row[0] == exp.exp_id and row[16] == f"=IFERROR(AVERAGE('Runs'!$R${first}:$R${last}),\"\")"
    assert row[15] == f"=COUNTIF('Runs'!$Q${first}:$Q${last},\"done\")&\" / 3\""
    assert row[21].startswith("=IFERROR(Q") and f"-Q{exp.m1.row}" in row[21]
    cubes = wb["Cubes"]
    references = [c.value for line in cubes.iter_rows() for c in line if isinstance(c.value, str)
                  and c.value.startswith("='T4 Experiments'!")]
    assert len(references) > 500
    cross = wb["Cross-evaluation"]
    texts = [c.value for line in cross.iter_rows() for c in line if isinstance(c.value, str)]
    assert any(t.startswith("=IFERROR(AVERAGEIFS(") for t in texts) and "M4-SAM·B" in texts
    meta = wb["Stage 1 meta-training"]
    assert meta.cell(row=6, column=11).value == "meta_training/A/m3_lstm/phi_best.pt"
    assert wb["T1 Optimizees & Optimizers"].cell(row=5, column=3).value == "MLP"
    assert all(cell.font.name == "Arial" for line in t4.iter_rows(max_row=40) for cell in line if cell.value is not None)


def test_filling_writes_results_and_keeps_the_rest(book, tmp_path):
    root = tmp_path / "artifacts" / "production"
    run = root / "runs" / "main" / "convnet4-svhn" / "m4_lstm_reg-sam@B" / "seed42"
    write_json(run / "final.json", {
        "status": "completed", "study": "main", "optimizee": "convnet4", "dataset": "svhn", "method": "m4_lstm_reg",
        "label": "M4", "regularizer": "sam", "round": "B", "seed": 42, "test_top1": 0.9123, "train_top1": 0.98,
        "test_loss": 0.31, "train_loss": 0.06, "flops_per_step": 1.25e9, "wall_clock_s": 1800.0, "steps": 100})
    running = root / "runs" / "main" / "mlp-svhn" / "m1_adamw" / "seed43"
    with MetricsWriter(running / "metrics.jsonl") as log:
        log.write(kind="epoch", epoch=1, train_loss_running=1.0)
    arm = root / "meta_training" / "B" / "m4_lstm_reg-sam"
    with MetricsWriter(arm / "meta_log.jsonl") as log:
        log.write(kind="meta_step", trajectory=0, truncation=0, meta_step=1, l_meta=2.0, l_smooth=0.01, l_reg=0.3)
    write_json(arm / "summary.json", {"status": "completed", "meta_training": "m4_lstm_reg", "regularizer": "sam",
                                      "trajectories": 10000, "meta_steps": 123, "best_val_loss": 0.5,
                                      "best_val_acc": 0.81, "best_trajectory": 9000, "wall_clock_s": 7200.0})
    copy = tmp_path / "mine.xlsx"
    copy.write_bytes(book.read_bytes())
    wb = openpyxl.load_workbook(copy)
    wb["Runs"].cell(row=6, column=26).value = "my note"
    wb.save(copy)

    counts = fill_workbook(copy, root)
    assert counts == {"runs": 1, "meta": 1, "running": 1}
    wb = openpyxl.load_workbook(copy)
    runs = wb["Runs"]
    folders = {runs.cell(row=r, column=16).value: r for r in range(6, runs.max_row + 1)}
    r = folders["runs/main/convnet4-svhn/m4_lstm_reg-sam@B/seed42"]
    assert [runs.cell(row=r, column=c).value for c in (17, 18, 19, 21, 22, 24, 25)] == [
        "done", pytest.approx(91.23), pytest.approx(98.0), 0.31, 0.06, pytest.approx(1.25), pytest.approx(0.5)]
    assert runs.cell(row=folders["runs/main/mlp-svhn/m1_adamw/seed43"], column=17).value == "running"
    assert runs.cell(row=6, column=26).value == "my note"  # the user's cells survive
    meta = wb["Stage 1 meta-training"]
    row = next(r for r in range(6, meta.max_row + 1)
               if meta.cell(row=r, column=11).value == "meta_training/B/m4_lstm_reg-sam/phi_best.pt")
    assert [meta.cell(row=row, column=c).value for c in (12, 13, 15, 16)] == ["done", 10000, 0.5, pytest.approx(81.0)]


def test_the_script(tmp_path, capsys):
    main = runpy.run_path(str(project_root() / "scripts" / "make_workbook.py"))["main"]
    out = tmp_path / "plan.xlsx"
    assert main(["--profile", "smoke", "--out", str(out), "--artifacts", str(tmp_path / "a")]) == 0
    assert out.is_file() and "smoke plan" in capsys.readouterr().out
    assert main(["--fill", str(tmp_path / "missing.xlsx"), "--artifacts", str(tmp_path / "a")]) == 2
    assert main(["--fill", str(out), "--profile", "smoke", "--artifacts", str(tmp_path / "a")]) == 0
    assert "0 finished runs" in capsys.readouterr().out
