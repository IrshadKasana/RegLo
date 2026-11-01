"""Experiment grids → manifests: expansion, filters, validation, round trips, cluster dispatch."""

import runpy
import shlex
from pathlib import Path

import pytest

from reglo.config import ConfigError, load_experiment, project_root
from reglo.manifest import (
    ManifestError,
    expand,
    manifest_path,
    parse_only,
    read_manifest,
    run_line,
    validate,
    write_manifest,
)
from reglo.utils.logging import write_json

REGS = ["l1", "l2", "sam", "gam", "gsam"]


def _kinds(lines):
    counts = {}
    for line in lines:
        counts[line[0]] = counts.get(line[0], 0) + 1
    return counts


def _value(line, key):
    return next((a.split("=", 1)[1] for a in line if a.startswith(key + "=")), None)


@pytest.mark.parametrize("profile, sizes", [
    # Stage 1: 2 rounds × 6 arms + M2 × 5 r on 2 source pairs × 3 seeds; Stages 2–3: pairs × (M1 + M2 × 5 +
    # 2 rounds × 6) × 3, less M2's 15 runs on the source pair the file contains (Stage 1's); extended: round A only
    ("production", {"stage1_meta_training": 12 + 30, "stage2_optimizer_evaluation": 3 * 54 - 15,
                    "stage3_optimizee_evaluation": 6 * 54 - 15, "extended_evaluation": 7 * 12 * 3, "pilot": 2,
                    "stage4_analysis": 9}),
    ("smoke", {"stage1_meta_training": 12 + 10, "stage2_optimizer_evaluation": 3 * 18 - 5,
               "stage3_optimizee_evaluation": 6 * 18 - 5, "extended_evaluation": 7 * 12, "pilot": 2,
               "stage4_analysis": 9}),  # smoke: one seed, the same grid
])
def test_stage_grids_have_the_design_sizes(tmp_path, profile, sizes):
    for name, size in sizes.items():  # stage 4: compute lines (no finished runs yet, so no audits)
        lines = expand(load_experiment(name, profile=profile), artifacts_root=tmp_path)
        assert len(lines) == size, name
        assert len({tuple(line) for line in lines}) == size, f"{name}: duplicate runs"
        assert all(f"profile={profile}" in line or ["--profile", profile] == line[3:5] for line in lines)


def test_evaluation_grid_is_m1_m2r_plus_m3_m4r_of_every_round_times_seeds_per_pair(tmp_path):
    lines = expand(load_experiment("stage2_optimizer_evaluation", profile="production"), artifacts_root=tmp_path)
    assert all(line[0] == "train_optimizee" and _value(line, "study") == "main" for line in lines)
    for optimizee in ("convnet4", "mlp", "resnet18"):
        combos = {(_value(l, "method"), _value(l, "regularizer") or "none", _value(l, "meta_tag"), _value(l, "seed"))
                  for l in lines if _value(l, "optimizee") == optimizee and _value(l, "dataset") == "svhn"}
        source = optimizee == "convnet4"  # round A's source pair: M2's runs are Stage 1's
        assert len(combos) == 3 + (0 if source else 5 * 3) + 2 * 6 * 3
        for seed in ("42", "43", "44"):
            assert ("m1_adamw", "none", None, seed) in combos  # M1 once: it uses no learned optimizer
            assert all((("m2_adamw_reg", r, None, seed) in combos) != source for r in REGS)  # M2 with r, no round
            for round_name in ("A", "B"):
                assert ("m3_lstm", "none", round_name, seed) in combos
                assert all(("m4_lstm_reg", r, round_name, seed) in combos for r in REGS)


def test_stage1_trains_m2_with_r_on_each_source_pair(tmp_path):
    lines = expand(load_experiment("stage1_meta_training", profile="production"), kinds=["train"], artifacts_root=tmp_path)
    assert {(_value(l, "method"), _value(l, "optimizee"), _value(l, "dataset")) for l in lines} == {
        ("m2_adamw_reg", "convnet4", "svhn"), ("m2_adamw_reg", "mlp", "fmnist")}
    assert {(_value(l, "regularizer"), _value(l, "seed")) for l in lines} == {(r, s) for r in REGS for s in ("42", "43", "44")}
    only_b = expand(load_experiment("stage1_meta_training", profile="production"), only={"round": "B"},
                    artifacts_root=tmp_path)
    assert {(_value(l, "optimizee"), _value(l, "dataset")) for l in only_b} == {("mlp", "fmnist")} and len(only_b) == 6 + 15


def test_stage3_trains_only_the_pairs_stage2_does_not(tmp_path):
    stage2 = {(_value(l, "optimizee"), _value(l, "dataset"))
              for l in expand(load_experiment("stage2_optimizer_evaluation"), artifacts_root=tmp_path)}
    stage3 = {(_value(l, "optimizee"), _value(l, "dataset"))
              for l in expand(load_experiment("stage3_optimizee_evaluation"), artifacts_root=tmp_path)}
    assert stage2 == {(o, "svhn") for o in ("convnet4", "mlp", "resnet18")}
    assert stage3 == {(o, d) for o in ("convnet4", "mlp", "resnet18") for d in ("fmnist", "cifar10")}
    assert not stage2 & stage3


def test_scale_numbers_come_from_the_profile_not_the_lines(tmp_path):
    """The lines carry the profile; epochs, subsets and widths come from it when the run composes."""
    from reglo.config import compose_run

    for profile, epochs in (("production", 100), ("smoke", 2)):
        line = next(l for l in expand(load_experiment("stage3_optimizee_evaluation", profile=profile),
                                      artifacts_root=tmp_path)
                    if (_value(l, "optimizee"), _value(l, "dataset")) == ("resnet18", "cifar10"))
        assert not any(a.startswith(("training.", "dataset.", "optimizee.")) for a in line)
        cfg = compose_run(line[1:])
        assert cfg.training.epochs == epochs and cfg.profile == profile
        assert cfg.optimizee.width == (64 if profile == "production" else 8)


def test_stage1_lines_and_their_sameness_check(tmp_path):
    lines = expand(load_experiment("stage1_meta_training"), kinds=["meta_train"], artifacts_root=tmp_path)
    assert [_value(l, "meta_training") for l in lines] == (["m3_lstm"] + ["m4_lstm_reg"] * 5) * 2
    assert [_value(l, "regularizer") for l in lines[1:6]] == REGS
    assert [(_value(l, "meta_tag"), _value(l, "optimizee"), _value(l, "dataset")) for l in (lines[0], lines[6])] == [
        ("A", "convnet4", "svhn"), ("B", "mlp", "fmnist")]
    validate(lines)  # the real Stage-1 arms differ only in their objective, the rounds only in their source
    tampered = [list(l) for l in lines[:2]]
    tampered[1].append("meta_training.outer.lr=1.0e-3")
    with pytest.raises(ManifestError, match="except for the objective"):
        validate(tampered)
    across = [list(lines[0]), list(lines[6])]  # M3 of round A and of round B
    across[1].append("meta_training.outer.lr=1.0e-3")
    with pytest.raises(ManifestError, match="rounds must meta-train identically"):
        validate(across)


def test_only_filters_and_ignores_keys_a_line_does_not_have(tmp_path):
    spec = load_experiment("extended_evaluation", profile="production")  # round A only
    assert len(expand(spec, only=parse_only(["optimizee=wrn28"]), artifacts_root=tmp_path)) == 3 * 12 * 3
    sam = expand(spec, only=parse_only(["regularizer=sam", "dataset=svhn"]), artifacts_root=tmp_path)
    assert len(sam) == 2 * 2 * 3 and {_value(l, "method") for l in sam} == {"m2_adamw_reg", "m4_lstm_reg"}
    one = expand(spec, only=parse_only(["optimizee=wrn28", "dataset=cifar10", "method=m3_lstm", "seed=43"]),
                 artifacts_root=tmp_path)
    assert len(one) == 1
    stage2 = load_experiment("stage2_optimizer_evaluation", profile="production")
    round_b = expand(stage2, only=parse_only(["round=B"]), artifacts_root=tmp_path)
    assert len(round_b) == 3 * (3 + 6 * 3) + 2 * 5 * 3  # M1, M2 (no round: kept) and round B's M3/M4
    assert {_value(l, "meta_tag") for l in round_b} == {None, "B"}
    assert len(expand(stage2, only=parse_only(["round=B", "method=m3_lstm"]), artifacts_root=tmp_path)) == 3 * 3


def test_parse_only_rejects_typos_and_malformed_filters():
    with pytest.raises(ManifestError, match="not one of"):
        parse_only(["optimiser=resnet18"])  # would otherwise silently launch the whole grid
    with pytest.raises(ManifestError, match="key=value"):
        parse_only(["resnet18"])


def test_per_optimizer_overrides_reach_only_adamw_runs_of_that_optimizee(tmp_path):
    lines = expand(load_experiment("extended_evaluation"), only={"optimizee": "vit_lite7"}, artifacts_root=tmp_path)
    for line in lines:
        adamw = _value(line, "method") in ("m1_adamw", "m2_adamw_reg")
        assert ("optimizer.lr=5.0e-4" in line) == adamw
    others = expand(load_experiment("extended_evaluation"), only={"optimizee": "resnet18"}, artifacts_root=tmp_path)
    assert not any("optimizer.lr=5.0e-4" in line for line in others)


def test_validate_rejects_invalid_runs():
    validate([["train_optimizee", "method=m1_adamw", "optimizee=convnet4", "dataset=svhn"]])
    with pytest.raises(ManifestError, match="line 1"):
        validate([["train_optimizee", "method=m1_adamw", "regularizer=sam", "optimizee=convnet4", "dataset=svhn"]])
    with pytest.raises(ManifestError, match="unknown script"):
        validate([["rm", "-rf", "/"]])
    with pytest.raises(ManifestError, match="Unknown profile"):
        validate([["train_optimizee", "method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=smok"]])


def test_manifest_round_trip_keeps_arguments_exact(tmp_path):
    lines = [["train_optimizee", "method=m1_adamw", "optimizee.channels=[8,8,16]", "+notes=two words", "x='quoted'"],
             ["profile_compute", "--experiment", "stage4_analysis", "--profile", "smoke", "--index", "0"],
             ["run_diagnostics", "--run", "C:/Users/Mohd Irshad/reglo/artifacts/smoke/runs/main/a-b/m1_adamw/seed42"]]
    path = write_manifest(lines, tmp_path / "m.txt")
    assert read_manifest(path) == lines  # paths with spaces (Windows user folders) survive
    assert path.read_text().splitlines()[0] == shlex.join(lines[0])


def test_manifest_paths_name_profile_kind_and_filters(tmp_path):
    name = "stage2_optimizer_evaluation"
    assert manifest_path(name, artifacts_root=tmp_path, profile="smoke") == tmp_path / "smoke" / "manifests" / f"{name}.txt"
    path = manifest_path(name, ["train"], {"seed": "42", "optimizee": "resnet18"}, artifacts_root=tmp_path)
    assert path.parent == tmp_path / "smoke_offline" / "manifests"  # the active profile
    assert path.name == f"{name}__train__optimizee-resnet18__seed-42.txt"


def test_diagnostics_lines_list_finished_runs_of_the_profile_only(tmp_path):
    runs = tmp_path / "smoke" / "runs" / "main" / "convnet4-svhn"
    for tag, status in (("m1_adamw", "completed"), ("m4_lstm_reg-sam@A", "completed"), ("m4_lstm_reg-sam@B", "completed")):
        write_json(runs / tag / "seed42" / "final.json", {"status": status})
    (runs / "m3_lstm@A" / "seed42").mkdir(parents=True)  # still running: no final.json
    write_json(tmp_path / "production" / "runs" / "main" / "convnet4-svhn" / "m1_adamw" / "seed42" / "final.json", {})
    spec = load_experiment("stage4_analysis", profile="smoke")
    lines = expand(spec, kinds=["diagnostics"], artifacts_root=tmp_path)
    # the smoke profile audits one run per pair and method (per_method: 1) …
    assert [line[-1].rsplit("/", 2)[-2] for line in lines] == ["m1_adamw", "m4_lstm_reg-sam@A"]
    assert all(line[:5] == ["run_diagnostics", "--experiment", "stage4_analysis", "--profile", "smoke"] for line in lines)
    only = expand(spec, kinds=["diagnostics"], only={"regularizer": "sam", "round": "B"}, artifacts_root=tmp_path)
    assert len(only) == 1 and only[0][-1].endswith("m4_lstm_reg-sam@B/seed42")  # … after the filters
    for tag in ("m1_adamw", "m4_lstm_reg-sam@A", "m4_lstm_reg-sam@B"):  # production audits every run
        write_json(tmp_path / "production" / "runs" / "main" / "convnet4-svhn" / tag / "seed43" / "final.json", {})
    production = expand(load_experiment("stage4_analysis", profile="production"), kinds=["diagnostics"],
                        artifacts_root=tmp_path)
    assert len(production) == 4


def test_unknown_kind_is_an_error(tmp_path):
    with pytest.raises(ManifestError, match="Unknown kind"):
        expand(load_experiment("stage4_analysis"), kinds=["evaluate"], artifacts_root=tmp_path)


def test_run_line_refuses_unknown_scripts():
    with pytest.raises(ManifestError):
        run_line(["make_report"])


def test_run_manifest_rejects_an_index_past_the_end(tmp_path, capsys):
    path = write_manifest([["profile_compute", "--experiment", "stage4_analysis", "--index", "0"]], tmp_path / "m.txt")
    main = runpy.run_path(str(project_root() / "scripts" / "run_manifest.py"))["main"]
    assert main([str(path), "3"]) == 2
    assert "outside 0..0" in capsys.readouterr().err


def test_make_manifest_writes_and_reports(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("REGLO_ARTIFACTS", str(tmp_path))
    main = runpy.run_path(str(project_root() / "scripts" / "make_manifest.py"))["main"]
    assert main(["pilot", "--print-path"]) == 0
    path = Path(capsys.readouterr().out.strip())
    assert path == tmp_path / "smoke_offline" / "manifests" / "pilot.txt" and len(read_manifest(path)) == 2
    assert main(["pilot", "--profile", "production", "--print-path"]) == 0
    path = Path(capsys.readouterr().out.strip())
    assert path == tmp_path / "production" / "manifests" / "pilot.txt"
    assert all("training.epochs=10" in line for line in read_manifest(path))
    assert main(["pilot", "--only", "method=m9"]) == 2  # nothing matches
    assert main(["pilot", "--only", "optimiser=x"]) == 2
    assert "not one of" in capsys.readouterr().err
    assert main(["pilot", "--profile", "huge"]) == 2
    assert "Unknown profile" in capsys.readouterr().err


@pytest.mark.parametrize("folder, names, marker", [
    ("cluster", ["env.sh"], "REGLO_PYTHON"),
    ("cluster/pbs", ["run_array.pbs", "submit.sh", "submit_pipeline.sh"], "#PBS -q gpu-ext"),
    ("cluster/slurm", ["run_array.slurm", "submit.sh", "submit_pipeline.sh"], "#SBATCH --gres=gpu:1"),
])
def test_cluster_scripts_are_lf_and_parse(folder, names, marker):
    import shutil
    import subprocess

    root = project_root() / folder
    for name in names:
        text = (root / name).read_bytes()
        assert b"\r\n" not in text, f"{name} has Windows line endings"
        if shutil.which("bash"):
            subprocess.run(["bash", "-n", str(root / name)], check=True)
    assert any(marker in (root / name).read_text() for name in names)


# ───────────────────────────────────────────── regressions found in review


@pytest.fixture
def configs_copy(tmp_path, monkeypatch):
    """A private copy of configs/ (REGLO_CONFIGS) that a test may edit."""
    import shutil

    target = tmp_path / "configs"
    shutil.copytree(project_root() / "configs", target)
    monkeypatch.setenv("REGLO_CONFIGS", str(target))
    return target


def test_stage1_check_covers_the_whole_grid_even_for_a_subset(configs_copy, tmp_path):
    m4 = configs_copy / "meta_training" / "m4_lstm_reg.yaml"
    m4.write_text(m4.read_text() + "\ninner:\n  unroll_steps: 10\n")
    spec = load_experiment("stage1_meta_training")
    subset = expand(spec, kinds=["meta_train"], only={"meta_training": "m4_lstm_reg", "round": "A"},
                    artifacts_root=tmp_path)
    assert len(subset) == 5
    validate(subset)  # the five M4 arms agree among themselves …
    with pytest.raises(ManifestError, match="inner"):
        validate(subset, spec)  # … but not with M3, which the grid also holds


def test_stage1_check_compares_the_whole_config_not_only_the_meta_block(configs_copy):
    lstm = (configs_copy / "optimizers" / "lstm.yaml").read_text()
    (configs_copy / "optimizers" / "lstm_big.yaml").write_text(
        lstm.replace("hidden_size: 20", "hidden_size: 64").replace("name: lstm", "name: lstm_big"))
    m4 = configs_copy / "meta_training" / "m4_lstm_reg.yaml"
    text = m4.read_text()
    assert "defaults:" in text
    m4.write_text(text.replace("defaults:", "defaults:\n  optimizer: lstm_big", 1))
    spec = load_experiment("stage1_meta_training")
    with pytest.raises(ManifestError, match="optimizer"):
        validate(expand(spec), spec)


def test_only_with_a_field_no_run_has_is_an_error(tmp_path):
    with pytest.raises(ManifestError, match="no run of 'stage2_optimizer_evaluation'"):
        expand(load_experiment("stage2_optimizer_evaluation"), only={"meta_training": "m3_lstm"}, artifacts_root=tmp_path)
    with pytest.raises(ManifestError, match="no run of 'stage1_meta_training'"):
        expand(load_experiment("stage1_meta_training"), kinds=["meta_train"], only={"method": "m3_lstm"},
               artifacts_root=tmp_path)
    with pytest.raises(ManifestError, match="no run of 'stage4_analysis'"):
        expand(load_experiment("stage4_analysis"), kinds=["compute"], only={"round": "A"}, artifacts_root=tmp_path)
    with pytest.raises(ManifestError, match="twice"):
        parse_only(["seed=42", "seed=43"])


def test_compute_lines_are_validated_with_all_twelve_variants(configs_copy, tmp_path):
    exp = configs_copy / "experiments" / "stage4_analysis.yaml"
    exp.write_text(exp.read_text().replace("- [convnet4, fmnist]", "- [convnet4, fmnits]"))
    lines = expand(load_experiment("stage4_analysis"), kinds=["compute"], artifacts_root=tmp_path)
    with pytest.raises(ManifestError, match="fmnits"):
        validate(lines)
    with pytest.raises(ManifestError, match="outside"):
        validate([["profile_compute", "--experiment", "stage4_analysis", "--index", "9"]])


def test_duplicate_lines_are_refused():
    line = ["train_optimizee", "method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "seed=42"]
    with pytest.raises(ManifestError, match="duplicate"):
        validate([line, list(line)])


def test_experiment_name_must_match_its_file(configs_copy):
    (configs_copy / "experiments" / "stage2_copy.yaml").write_text(
        (configs_copy / "experiments" / "stage2_optimizer_evaluation.yaml").read_text())
    with pytest.raises(ConfigError, match="must equal the file name"):
        load_experiment("stage2_copy")


def test_experiment_files_may_not_pin_scale_numbers(configs_copy):
    """An epoch count in an experiment file would also run in the smoke test: refused, with the fix named."""
    exp = configs_copy / "experiments" / "stage2_optimizer_evaluation.yaml"
    exp.write_text(exp.read_text().replace("  overrides: []", "  overrides: [training.epochs=50]", 1))
    with pytest.raises(ConfigError, match="profile_overrides"):
        load_experiment("stage2_optimizer_evaluation")


def test_a_crashing_line_does_not_stop_the_others(tmp_path, capsys):
    broken = tmp_path / "smoke_offline" / "runs" / "main" / "convnet4-svhn" / "m1_adamw" / "seed42"
    broken.mkdir(parents=True)
    (broken / "final.json").write_text("{")  # a truncated file: the audit crashes
    path = write_manifest([["run_diagnostics", "--run", broken.as_posix()],
                           ["profile_compute", "--bogus-flag"],
                           ["run_diagnostics", "--run", str(tmp_path / "no_such_run")],
                           ["run_diagnostics", "--experiment", "stage4_analysis", "--list"]], tmp_path / "m.txt")
    lines = read_manifest(path)
    assert run_line(lines[0]) == 1  # an exception inside the script → exit code 1
    assert run_line(lines[1]) == 2  # argparse error → its exit code
    assert run_line(lines[2]) == 2  # not a finished run: a clear error, not a crash
    main = runpy.run_path(str(project_root() / "scripts" / "run_manifest.py"))["main"]
    assert main([str(path), "--all"]) == 1
    err = capsys.readouterr().err
    assert "JSONDecodeError" in err and "not a finished run" in err
