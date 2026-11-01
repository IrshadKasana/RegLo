"""scripts/run_study.py: the plan of a study, its dependencies, and the scheduler."""

import runpy

import pytest

from reglo.config import project_root
from reglo.runner import Runner, Task, plan_study


def _value(line, key):
    return next((a.split("=", 1)[1] for a in line if a.startswith(key + "=")), None)


def test_the_plan_follows_the_study_design(tmp_path):
    tasks = plan_study("main", profile="production", artifacts_root=tmp_path, write=False)
    stages = [t.stage for t in tasks]
    assert stages.count("1") == 12 + 30 and stages.count("2") == 147 and stages.count("3") == 309
    assert stages.count("4") == 9
    arms = {(_value(t.line, "meta_training"), _value(t.line, "regularizer") or "none", _value(t.line, "meta_tag")): t.key
            for t in tasks if t.line[0] == "meta_train"}
    assert len(arms) == 12
    for task in tasks:
        method = _value(task.line, "method")
        if method == "m3_lstm":  # the arm of its own round
            assert task.needs == {arms[("m3_lstm", "none", _value(task.line, "meta_tag"))]}
        elif method == "m4_lstm_reg":
            assert task.needs == {arms[("m4_lstm_reg", _value(task.line, "regularizer"), _value(task.line, "meta_tag"))]}
        elif method in ("m1_adamw", "m2_adamw_reg"):
            assert not task.needs  # AdamW needs no LSTM: these start at once
        elif task.stage == "4":
            assert task.needs == set(arms.values())  # timing uses the trained optimizers
    assert all(t.priority[0] == 0 for t in tasks if t.stage == "1")  # Stage 1 is picked first


def test_the_plan_can_be_narrowed(tmp_path):
    tasks = plan_study("main", stages=[1, 2], regularizers=["sam"], profile="smoke", artifacts_root=tmp_path, write=False)
    assert {t.stage for t in tasks} == {"1", "2"}
    assert {_value(t.line, "regularizer") or "none" for t in tasks} == {"none", "sam"}  # the SAM block, with M1 and M3
    assert len([t for t in tasks if t.line[0] == "meta_train"]) == 2 * 2  # M3 and M4-SAM, both rounds
    assert len([t for t in tasks if t.stage == "1" and t.line[0] == "train_optimizee"]) == 2  # M2-SAM on each source
    round_a = plan_study("main", stages=[1, 2, 3], only={"round": "A"}, profile="smoke", artifacts_root=tmp_path,
                         write=False)
    assert {_value(t.line, "meta_tag") for t in round_a} == {None, "A"}
    assert {(_value(t.line, "optimizee"), _value(t.line, "dataset")) for t in round_a if t.stage == "1"} == {
        ("convnet4", "svhn")}
    extended = plan_study("main", stages=[2, 3], extended=True, pilot=True, profile="smoke", artifacts_root=tmp_path,
                          write=False)
    assert {t.stage for t in extended} == {"2", "3", "2-3", "pilot"}
    assert not any(t.needs for t in extended)  # no Stage 1 in the plan: the optimizers must exist already


def _tasks(*lines, needs=None):
    tasks = [Task(f"t{i}", line, "x", "test", priority=(1, i)) for i, line in enumerate(lines)]
    for i, wanted in (needs or {}).items():
        tasks[i].needs = {f"t{j}" for j in wanted}
    return tasks


GOOD = ["run_diagnostics", "--experiment", "stage4_analysis", "--list"]
BAD = ["profile_compute", "--bogus-flag"]


def test_a_failed_prerequisite_skips_its_dependents_and_nothing_else(tmp_path):
    tasks = _tasks(BAD, GOOD, GOOD, needs={1: [0], 2: []})
    Runner(tasks, log_dir=tmp_path).run()
    assert [t.status for t in tasks] == ["failed", "skipped", "done"]
    assert tasks[0].code == 2 and "t0" in tasks[1].note


@pytest.mark.parametrize("slots", [["cpu"], ["cpu", "cpu"]])
def test_child_processes_run_in_dependency_order_with_logs(tmp_path, slots):
    tasks = _tasks(GOOD, GOOD, BAD, GOOD, needs={1: [0], 3: [2]})
    Runner(tasks, slots, log_dir=tmp_path, poll_seconds=0.1, threads_per_job=1).run()
    assert [t.status for t in tasks] == ["done", "done", "failed", "skipped"]
    assert all(t.log is not None and t.log.is_file() for t in tasks[:3])
    assert "unrecognized arguments" in tasks[2].log.read_text()


def test_unknown_dependencies_are_refused():
    with pytest.raises(ValueError, match="unknown task"):
        Runner(_tasks(GOOD, needs={0: [5]}))


def test_run_study_dry_run(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("REGLO_ARTIFACTS", str(tmp_path))
    main = runpy.run_path(str(project_root() / "scripts" / "run_study.py"))["main"]
    assert main(["--profile", "smoke", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "stage 1: 22, stage 2: 49, stage 3: 103, stage 4: 9" in out and "waits for" in out
    assert not (tmp_path / "smoke" / "manifests").exists()  # a dry run writes nothing
    assert main(["--profile", "nope", "--dry-run"]) == 2


def test_run_study_refuses_production_without_a_gpu(tmp_path, monkeypatch):
    import torch

    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    monkeypatch.setenv("REGLO_ARTIFACTS", str(tmp_path))
    main = runpy.run_path(str(project_root() / "scripts" / "run_study.py"))["main"]
    with pytest.raises(SystemExit, match="runs on CUDA GPUs"):
        main(["--profile", "production", "--stages", "1"])
