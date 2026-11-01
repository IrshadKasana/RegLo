"""Profiles: smoke and production numbers in two mirrored files, applied to runs and grids."""

import shutil

import pytest
from omegaconf import OmegaConf

from reglo.config import ConfigError, compose_meta, compose_run, config_dir, load_experiment, project_root
from reglo.profiles import (
    active_profile,
    available_profiles,
    controlled_keys,
    leaf_keys,
    load_profile,
    mirror_gaps,
    set_active_profile,
)


def test_the_profiles():
    assert available_profiles() == ["production", "smoke", "smoke_offline"]
    assert load_profile("smoke_offline").runs.dataset.source == "synthetic"  # extends smoke …
    assert load_profile("smoke_offline").runs.training.epochs == load_profile("smoke").runs.training.epochs  # … and keeps it
    with pytest.raises(ConfigError, match="Unknown profile 'huge'"):
        load_profile("huge")


def test_smoke_sets_every_number_production_sets():
    """The mirror: no production number can be forgotten in the smoke test (it would run at full size)."""
    production, smoke = load_profile("production"), load_profile("smoke")
    assert mirror_gaps(production, smoke) == []
    assert "training.epochs" in controlled_keys(production) and "meta_training.schedule.num_trajectories" in controlled_keys(smoke)


def _holes(node, prefix=""):
    """Dotted paths whose value is ??? (left to the profiles)."""
    for key, value in node.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from _holes(value, path + ".")
        elif value == "???":
            yield path


def test_the_group_files_leave_the_profile_numbers_open():
    """??? in the group files ⇔ set by every profile: a number has one home per profile."""
    holes = set()
    for path, prefix in [(config_dir() / "default.yaml", ""), (config_dir() / "meta_training" / "base.yaml", "meta_training."),
                         *((p, "dataset.") for p in sorted((config_dir() / "datasets").glob("*.yaml")))]:
        node = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        node.pop("defaults", None)  # group choices, not numbers
        node.pop("name", None)
        holes |= set(_holes(node, prefix))
    assert {"training.epochs", "device", "dataset.train_subset", "meta_training.schedule.num_trajectories"} <= holes
    for name in available_profiles():
        assert holes <= controlled_keys(load_profile(name)), (name, sorted(holes - controlled_keys(load_profile(name))))


def test_production_and_smoke_numbers_reach_the_runs():
    production = compose_meta(["meta_training=m4_lstm_reg", "regularizer=sam", "profile=production"])
    smoke = compose_meta(["meta_training=m4_lstm_reg", "regularizer=sam", "profile=smoke"])
    assert production.meta_training.schedule.num_trajectories == 10000 and smoke.meta_training.schedule.num_trajectories == 4
    assert production.meta_training.schedule.curriculum.trajectory_steps == [100, 200, 500, 1000]
    assert (production.meta_training.validation.every, production.meta_training.validation.optimizees) == (20, 5)
    assert production.device == "cuda" and smoke.device == "auto"
    assert production.optimizee.channels == [32, 64, 128] and smoke.optimizee.channels == [8, 16, 32]
    # everything that is not a scale number is identical: the smoke test runs the real recipe
    same = ("inner", "outer", "objective")
    assert all(production.meta_training[k] == smoke.meta_training[k] for k in same)
    assert production.smooth == smoke.smooth and production.regularizer == smoke.regularizer


def test_per_dataset_budgets():
    for dataset, epochs in (("svhn", 50), ("fmnist", 50), ("cifar10", 100), ("cifar100", 100)):
        cfg = compose_run(["method=m1_adamw", "optimizee=resnet18", f"dataset={dataset}", "profile=production"])
        assert cfg.training.epochs == epochs, dataset
        assert compose_run(["method=m1_adamw", "optimizee=resnet18", f"dataset={dataset}", "profile=smoke"]).training.epochs == 2


def test_explicit_overrides_beat_the_profile():
    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=smoke", "training.epochs=7",
                       "optimizee.channels=[4,4,4]"])
    assert cfg.training.epochs == 7 and cfg.optimizee.channels == [4, 4, 4]


def test_the_active_profile(monkeypatch):
    monkeypatch.delenv("REGLO_PROFILE")
    assert active_profile() == "production"
    assert compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn"]).profile == "production"
    assert set_active_profile("smoke") == "smoke" and active_profile() == "smoke"
    with pytest.raises(ConfigError, match="Unknown profile"):
        set_active_profile("nope")
    assert active_profile() == "smoke"  # unchanged by the failed call


def test_grids_are_scaled_by_the_profile():
    production = load_experiment("stage4_analysis", profile="production")
    smoke = load_experiment("stage4_analysis", profile="smoke")
    assert production["diagnostics"]["hessian"]["power_iterations"] == 30 and smoke["diagnostics"]["hessian"]["power_iterations"] == 2
    assert production["compute"]["timed_steps"] == 100 and smoke["compute"]["timed_steps"] == 2
    radii, few = production["diagnostics"]["sharpness"]["radii"], smoke["diagnostics"]["sharpness"]["radii"]
    assert set(few) < set(radii) and 0.01 in few  # smoke audits two of the radii, including the report's default
    assert production["profile"] == "production" and smoke["profile"] == "smoke"


@pytest.fixture
def configs_copy(tmp_path, monkeypatch):
    target = tmp_path / "configs"
    shutil.copytree(project_root() / "configs", target)
    monkeypatch.setenv("REGLO_CONFIGS", str(target))
    return target


def test_a_typo_in_a_profile_is_reported(configs_copy):
    smoke = configs_copy / "profiles" / "smoke.yaml"
    smoke.write_text(smoke.read_text().replace("    epochs: 2", "    epoch: 2"))
    with pytest.raises(ConfigError, match="training.epoch is not a key"):
        compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=smoke"])
    smoke.write_text(smoke.read_text().replace("runs:", "runz:"))
    with pytest.raises(ConfigError, match="unknown section"):
        compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=smoke"])


def test_a_profile_missing_a_number_cannot_run(configs_copy):
    smoke = configs_copy / "profiles" / "smoke.yaml"
    smoke.write_text(smoke.read_text().replace("    train_eval_every: 1\n", ""))
    with pytest.raises(ConfigError, match="training.train_eval_every"):
        compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=smoke"])
    assert mirror_gaps(load_profile("production"), load_profile("smoke")) == ["training.train_eval_every"]


def test_trajectory_lengths_must_be_whole_truncations():
    with pytest.raises(ConfigError, match="multiple of inner.unroll_steps=30"):
        compose_meta(["meta_training=m3_lstm", "profile=production", "meta_training.inner.unroll_steps=30"])
    with pytest.raises(ConfigError, match="must grow"):
        compose_meta(["meta_training=m3_lstm", "profile=production",
                      "meta_training.schedule.curriculum.trajectory_steps=[200,100]"])
