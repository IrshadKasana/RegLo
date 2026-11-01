"""The study design in code (reglo.study, reglo.rounds) and the stage-to-stage connectivity check (reglo.checks)."""

import runpy

import pytest

from reglo.checks import check_study
from reglo.config import ConfigError, project_root
from reglo.rounds import load_rounds
from reglo.study import (
    METRICS,
    SETTINGS,
    STAGES,
    Pair,
    pair_overrides,
    study_experiments,
    study_rounds,
    study_settings,
    training_pairs,
)


def test_the_vocabulary_matches_the_figure():
    assert [STAGES[i].split(":")[0].split(" (")[0] for i in (1, 2, 3, 4)] == [
        "Training", "Optimizer evaluation", "Optimizee evaluation", "Evaluation metrics"]
    assert [s.label for s in SETTINGS] == ["Seen optimizee", "Unseen optimizee", "Seen dataset", "Unseen dataset"]
    assert [(m.key, m.better) for m in METRICS] == [("accuracy", "higher"), ("gen_gap", "lower"), ("flops", "lower")]


def test_the_rounds_and_what_a_pair_is_to_them():
    rounds = load_rounds()
    assert list(rounds) == ["A", "B"]
    a, b = rounds["A"], rounds["B"]
    assert (a.optimizee, a.dataset, b.optimizee, b.dataset) == ("convnet4", "svhn", "mlp", "fmnist")
    assert a.role("convnet4", "svhn") == "seen" and b.role("convnet4", "svhn") == "unseen_both"
    assert a.role("resnet18", "svhn") == "unseen_optimizee" and a.role("convnet4", "fmnist") == "unseen_dataset"
    assert b.role("convnet4", "fmnist") == "unseen_optimizee" and b.role("mlp", "svhn") == "unseen_dataset"


def test_training_pairs_accept_every_form():
    assert training_pairs({"train": {"pairs": [["convnet4", "svhn"], ["resnet18", "cifar10"]]}}) == [
        Pair("convnet4", "svhn"), Pair("resnet18", "cifar10")]
    assert training_pairs({"train": {"optimizees": ["convnet4", "resnet18"], "datasets": ["svhn"]}}) == [
        Pair("convnet4", "svhn"), Pair("resnet18", "svhn")]
    assert training_pairs({"name": "x", "rounds": ["B", "A"], "train": {"pairs": "sources"}}) == [
        Pair("mlp", "fmnist"), Pair("convnet4", "svhn")]
    with pytest.raises(ConfigError, match="unknown round 'C'"):
        training_pairs({"name": "x", "rounds": ["C"], "train": {"pairs": "sources"}})
    with pytest.raises(ConfigError, match="needs a rounds list"):
        training_pairs({"name": "x", "train": {"pairs": "sources"}})


def test_pair_overrides_by_optimizee_then_by_pair():
    block = {"pair_overrides": {"resnet18": ["optimizee.width=4"], "resnet18-cifar10": ["training.epochs=100"]}}
    assert pair_overrides(block, Pair("resnet18", "cifar10")) == ["optimizee.width=4", "training.epochs=100"]
    assert pair_overrides(block, Pair("convnet4", "svhn")) == []


def test_study_main_and_its_settings_per_round():
    names = [s["name"] for s in study_experiments("main")]
    assert names[:4] == ["stage1_meta_training", "stage2_optimizer_evaluation", "stage3_optimizee_evaluation",
                         "stage4_analysis"]
    assert names[-1] == "extended_evaluation"  # extended files come last
    assert [r.name for r in study_rounds("main")] == ["A", "B"]
    settings = study_settings("main")

    def pairs(round_name, key, extended=False):
        return [(sp.pair.optimizee, sp.pair.dataset) for sp in settings[round_name].get(key, [])
                if sp.extended == extended]

    # round A: the thesis figure, plus the MLP and the full 3 × 3 grid
    assert pairs("A", "seen_optimizee") == pairs("A", "seen_dataset") == [("convnet4", "svhn")]
    assert pairs("A", "unseen_optimizee") == [("mlp", "svhn"), ("resnet18", "svhn")]
    assert pairs("A", "unseen_optimizee", extended=True) == [("wrn28", "svhn"), ("vit_lite7", "svhn")]
    assert pairs("A", "unseen_dataset")[:2] == [("convnet4", "fmnist"), ("convnet4", "cifar10")]  # same network first
    assert set(pairs("A", "unseen_dataset")) == {(o, d) for o in ("convnet4", "mlp", "resnet18")
                                                 for d in ("fmnist", "cifar10")}
    # round B: the roles turn around
    assert pairs("B", "seen_optimizee") == pairs("B", "seen_dataset") == [("mlp", "fmnist")]
    assert pairs("B", "unseen_optimizee") == [("convnet4", "fmnist"), ("resnet18", "fmnist")]
    assert ("convnet4", "svhn") in pairs("B", "unseen_dataset") and not pairs("B", "unseen_optimizee", extended=True)
    roles = {(sp.pair.optimizee, sp.pair.dataset): sp.role for sp in settings["B"]["unseen_dataset"]}
    assert roles[("mlp", "svhn")] == "unseen_dataset" and roles[("convnet4", "svhn")] == "unseen_both"


@pytest.mark.parametrize("profile", ["production", "smoke", "smoke_offline"])
def test_the_real_study_connects_under_every_profile(profile, tmp_path):
    report = check_study("main", tmp_path / "artifacts", profile)
    assert report.ok, report.text()
    texts = " | ".join(c.text for c in report.checks)
    assert "reads a φ that Stage 1 writes" in texts and "round(s) A, B" in texts
    assert "Stage 1 trains M2 (AdamW + r) on the source pair of every round" in texts
    assert "split fmnist, svhn identically" in texts
    assert "Stage 3 · Optimizee evaluation" in report.text() and f"profile {profile}" in report.text()
    assert "round B — source MLP / FMNIST" in report.text()
    assert report.folder.endswith(f"artifacts/{profile}")


def test_the_pilot_connects(tmp_path):
    report = check_study("pilot", tmp_path / "artifacts", "production")
    assert report.ok, report.text()


@pytest.fixture
def configs_copy(tmp_path, monkeypatch):
    import shutil

    target = tmp_path / "configs"
    shutil.copytree(project_root() / "configs", target)
    monkeypatch.setenv("REGLO_CONFIGS", str(target))
    return target


def test_a_missing_stage1_arm_breaks_the_chain(configs_copy, tmp_path):
    stage1 = configs_copy / "experiments" / "stage1_meta_training.yaml"
    stage1.write_text(stage1.read_text().replace("m4_lstm_reg: [l1, l2, sam, gam, gsam]", "m4_lstm_reg: [l1, l2, sam, gam]"))
    report = check_study("main", tmp_path / "artifacts")
    assert not report.ok
    assert any(not c.ok and "A/m4_lstm_reg-gsam" in c.text and "B/m4_lstm_reg-gsam" in c.text for c in report.checks)


def test_a_round_without_its_stage1_breaks_the_chain(configs_copy, tmp_path):
    stage1 = configs_copy / "experiments" / "stage1_meta_training.yaml"
    stage1.write_text(stage1.read_text().replace("rounds: [A, B]", "rounds: [A]"))
    report = check_study("main", tmp_path / "artifacts")
    failures = " | ".join(c.text for c in report.checks if not c.ok)
    assert "B/m3_lstm" in failures and "A/m3_lstm" not in failures


def test_an_unevaluated_source_and_a_double_trained_pair_are_reported(configs_copy, tmp_path):
    stage3 = configs_copy / "experiments" / "stage3_optimizee_evaluation.yaml"
    stage3.write_text(stage3.read_text().replace("    - [mlp, fmnist]\n", "    - [convnet4, svhn]\n"))
    report = check_study("main", tmp_path / "artifacts")
    failures = " | ".join(c.text for c in report.checks if not c.ok)
    assert "not: B (MLP / FMNIST)" in failures and "twice: ConvNet-4 / SVHN" in failures


def test_m2_reuses_stage1_on_the_source_pairs_only():
    from reglo.config import load_experiment
    from reglo.manifest import expand

    def m2_pairs(name):
        lines = expand(load_experiment(name, profile="production"), ["train"])
        return {(a["optimizee"], a["dataset"]) for a in (dict(x.split("=", 1) for x in line[1:] if "=" in x)
                                                          for line in lines) if a["method"] == "m2_adamw_reg"}

    assert m2_pairs("stage2_optimizer_evaluation") == {("mlp", "svhn"), ("resnet18", "svhn")}  # not ConvNet-4 / SVHN
    assert ("mlp", "fmnist") not in m2_pairs("stage3_optimizee_evaluation")                  # round B's source
    assert len(m2_pairs("stage3_optimizee_evaluation")) == 5 and len(m2_pairs("extended_evaluation")) == 7


def test_the_production_plan_counts():
    from reglo.config import load_experiment
    from reglo.manifest import expand

    def count(name, method=None):
        lines = expand(load_experiment(name, profile="production"), ["train"])
        return sum(1 for line in lines if method is None or f"method={method}" in line)

    assert count("stage1_meta_training") == 30
    assert [count("stage2_optimizer_evaluation", m) for m in ("m1_adamw", "m2_adamw_reg", "m3_lstm", "m4_lstm_reg")] \
        == [9, 30, 18, 90]
    assert [count("stage3_optimizee_evaluation", m) for m in ("m1_adamw", "m2_adamw_reg", "m3_lstm", "m4_lstm_reg")] \
        == [18, 75, 36, 180]
    assert [count("extended_evaluation", m) for m in ("m1_adamw", "m2_adamw_reg", "m3_lstm", "m4_lstm_reg")] \
        == [21, 105, 21, 105]
    assert 30 + count("stage2_optimizer_evaluation") + count("stage3_optimizee_evaluation") \
        + count("extended_evaluation") + count("pilot") == 740


def test_m2_without_from_stage1_would_train_the_source_pair_twice(configs_copy, tmp_path):
    stage2 = configs_copy / "experiments" / "stage2_optimizer_evaluation.yaml"
    stage2.write_text(stage2.read_text().replace("  from_stage1: [m2_adamw_reg]", "  from_stage1: []"))
    report = check_study("main", tmp_path / "artifacts")
    assert any(not c.ok and "appear twice" in c.text for c in report.checks)


def test_an_evaluation_grid_without_m2_is_reported(configs_copy, tmp_path):
    stage3 = configs_copy / "experiments" / "stage3_optimizee_evaluation.yaml"
    stage3.write_text(stage3.read_text().replace("    m2_adamw_reg: [l1, l2, sam, gam, gsam]\n", ""))
    report = check_study("main", tmp_path / "artifacts")
    failed = [c for c in report.checks if not c.ok and "every evaluation pair" in c.text]
    assert failed and "ConvNet-4 / FMNIST" in failed[0].text and "MLP / FMNIST" not in failed[0].text


def test_a_reused_stage1_run_must_match_the_evaluation_grid(configs_copy, tmp_path):
    stage2 = configs_copy / "experiments" / "stage2_optimizer_evaluation.yaml"
    stage2.write_text(stage2.read_text().replace("  overrides: []", "  overrides: [optimizer.lr=0.002]", 1))
    report = check_study("main", tmp_path / "artifacts")
    failed = [c for c in report.checks if not c.ok and "reused from Stage 1" in c.text]
    assert failed and "ConvNet-4 / SVHN" in failed[0].text


def test_the_stage1_benchmark_must_cover_every_source(configs_copy, tmp_path):
    stage1 = configs_copy / "experiments" / "stage1_meta_training.yaml"
    stage1.write_text(stage1.read_text().replace("  pairs: sources", "  pairs: [[convnet4, svhn]]"))
    report = check_study("main", tmp_path / "artifacts")
    assert any(not c.ok and "Stage 1 trains M2" in c.text for c in report.checks)


def test_a_study_grid_may_not_evaluate_on_a_validation_split(configs_copy, tmp_path):
    stage3 = configs_copy / "experiments" / "stage3_optimizee_evaluation.yaml"
    stage3.write_text(stage3.read_text().replace("  overrides: []", "  overrides: [training.validation_fraction=0.1]", 1))
    report = check_study("main", tmp_path / "artifacts")
    assert any(not c.ok and "validation split" in c.text and "stage3_optimizee_evaluation" in c.text
               for c in report.checks)


def test_a_settings_block_is_refused(configs_copy):
    stage2 = configs_copy / "experiments" / "stage2_optimizer_evaluation.yaml"
    stage2.write_text(stage2.read_text() + "settings:\n  seen_optimizee: [[convnet4, svhn]]\n")
    with pytest.raises(ConfigError, match="derived from configs/rounds.yaml"):
        study_settings("main")


def test_check_study_script_exit_codes(tmp_path, capsys):
    main = runpy.run_path(str(project_root() / "scripts" / "check_study.py"))["main"]
    assert main(["--profile", "smoke", "--artifacts", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "all checks passed" in out and "profile smoke" in out
    assert main(["no_such_study", "--artifacts", str(tmp_path)]) == 1
    assert main(["--profile", "nano", "--artifacts", str(tmp_path)]) == 2


def test_a_meta_split_mismatch_between_stage1_and_the_evaluation_runs_is_reported(configs_copy, tmp_path):
    stage2 = configs_copy / "experiments" / "stage2_optimizer_evaluation.yaml"
    stage2.write_text(stage2.read_text().replace("  overrides: []", "  overrides: [dataset.meta_split.seed=1]", 1))
    report = check_study("main", tmp_path / "artifacts")
    assert any(not c.ok and "meta split differs" in c.text for c in report.checks)


def test_a_round_on_a_dataset_without_meta_split_is_refused(configs_copy):
    from reglo.config import compose_meta

    rounds = configs_copy / "rounds.yaml"
    rounds.write_text(rounds.read_text() + "C:\n  optimizee: convnet4\n  dataset: cifar10\n")
    with pytest.raises(ConfigError, match="needs its meta split"):
        compose_meta(["meta_training=m3_lstm", "meta_tag=C", "profile=smoke"])


def test_the_rounds_file_is_validated(configs_copy):
    rounds = configs_copy / "rounds.yaml"
    original = rounds.read_text()
    rounds.write_text(original + "C:\n  optimizee: lenet\n  dataset: svhn\n")
    with pytest.raises(ConfigError, match="unknown optimizee 'lenet'"):
        load_rounds()
    rounds.write_text(original + "C:\n  optimizee: convnet4\n  dataset: svhn\n")
    with pytest.raises(ConfigError, match="the same source pair"):
        load_rounds()
    rounds.write_text(original + "round-c:\n  optimizee: mlp\n  dataset: svhn\n")
    with pytest.raises(ConfigError, match="must be a letter"):
        load_rounds()
