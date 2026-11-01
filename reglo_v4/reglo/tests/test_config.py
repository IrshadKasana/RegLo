"""Configuration system: every YAML file, every method, every guard."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from reglo.config import (
    ConfigError,
    available,
    compose_meta,
    compose_run,
    config_dir,
    experiment_names,
    load_experiment,
    load_saved_config,
    project_root,
    save_config,
)
from reglo.utils.paths import meta_run_dir, method_tag, run_dir

REGULARIZERS = ["l1", "l2", "sam", "gam", "gsam"]
METHODS = ["m1_adamw", "m2_adamw_reg", "m3_lstm", "m4_lstm_reg"]


def run(*overrides):
    return compose_run(list(overrides))


def meta(*overrides):
    return compose_meta(list(overrides))


# --------------------------------------------------------------------------- the files themselves


def test_every_yaml_file_parses():
    files = sorted(config_dir().rglob("*.yaml"))
    assert len(files) >= 30
    for path in files:
        assert OmegaConf.load(path) is not None, path


def test_available_choices():
    assert available("method") == METHODS
    assert available("optimizer") == ["adamw", "lstm"]
    assert available("optimizee") == ["convnet4", "mlp", "resnet18", "vit_lite7", "wrn28"]
    assert available("dataset") == ["cifar10", "cifar100", "fmnist", "svhn", "synthetic"]
    assert set(available("regularizer")) == set(REGULARIZERS) | {"none"}  # L_smooth is not a choice of r
    assert available("smooth") == ["smooth"]
    assert available("meta_training") == ["m3_lstm", "m4_lstm_reg"]  # base.yaml is not a choice


@pytest.mark.parametrize("reg", REGULARIZERS)
def test_regularizer_files_have_both_roles(reg):
    """Each regularizer serves M2 (explicit, every step) and M4 (meta-objective term)."""
    node = OmegaConf.load(config_dir() / "regularizers" / f"{reg}.yaml")
    assert node.name == reg
    assert "explicit" in node and "meta" in node
    assert node.meta.coef == 0.1  # λ_reg, one value for every r
    assert node.meta.space == "phi"  # M4: every r acts on the learned optimizer's weights φ (Algorithm 1)
    if reg in ("sam", "gam", "gsam"):
        assert node.rho == 0.05
    else:  # M4's L1/L2 term is M2's own penalty (its λ), on φ
        assert node.meta.normalize == "explicit"


def test_smooth_file_holds_the_reference_values():
    node = OmegaConf.load(config_dir() / "regularizers" / "smooth.yaml")
    assert (node.coef, node.epsilon, node.pga_steps, node.reduction) == (1.0, 0.01, 10, "sum")
    assert node.step_ratio == pytest.approx(0.2)


def test_lstm_is_the_dm_optimizer():
    node = OmegaConf.load(config_dir() / "optimizers" / "lstm.yaml")
    assert (node.kind, node.num_layers, node.hidden_size, node.head_layers) == ("learned", 2, 20, 2)


def test_adamw_is_unregularised_by_default():
    node = OmegaConf.load(config_dir() / "optimizers" / "adamw.yaml")
    assert node.kind == "conventional" and node.weight_decay == 0.0


# --------------------------------------------------------------------------- M1-M4


def test_m1_adamw():
    cfg = run("method=m1_adamw", "optimizee=convnet4", "dataset=svhn")
    assert cfg.optimizer.name == "adamw" and cfg.regularizer.name == "none"
    assert cfg.method.regularizer_applied == "never"
    assert (cfg.optimizee.num_classes, cfg.optimizee.in_channels) == (10, 3)


@pytest.mark.parametrize("reg", REGULARIZERS)
def test_m2_adamw_with_each_regularizer(reg):
    cfg = run("method=m2_adamw_reg", f"regularizer={reg}", "optimizee=resnet18", "dataset=cifar10")
    assert cfg.optimizer.name == "adamw" and cfg.regularizer.name == reg
    assert cfg.method.regularizer_applied == "every_step"
    assert method_tag(cfg) == f"m2_adamw_reg-{reg}"


def test_m3_lstm_loads_vanilla_checkpoint():
    cfg = run("method=m3_lstm", "optimizee=resnet18", "dataset=cifar10")
    assert cfg.optimizer.name == "lstm" and cfg.regularizer.name == "none"
    assert cfg.method.checkpoint.endswith("meta_training/A/m3_lstm/phi_best.pt")  # round A by default
    assert method_tag(cfg) == "m3_lstm@A"


@pytest.mark.parametrize("reg", REGULARIZERS)
def test_m4_lstm_loads_the_checkpoint_of_its_regularizer_and_round(reg):
    cfg = run("method=m4_lstm_reg", f"regularizer={reg}", "optimizee=wrn28", "dataset=cifar100")
    assert cfg.optimizer.name == "lstm"
    assert cfg.method.regularizer_applied == "meta_training_only"
    assert cfg.method.checkpoint.endswith(f"meta_training/A/m4_lstm_reg-{reg}/phi_best.pt")
    other = run("method=m4_lstm_reg", f"regularizer={reg}", "optimizee=wrn28", "dataset=cifar100", "meta_tag=B")
    assert other.method.checkpoint.endswith(f"meta_training/B/m4_lstm_reg-{reg}/phi_best.pt")
    assert method_tag(other) == f"m4_lstm_reg-{reg}@B"  # the same pair, trained with another round's φ


def test_meta_training_takes_the_source_pair_of_its_round():
    a, b = meta("meta_training=m3_lstm"), meta("meta_training=m4_lstm_reg", "regularizer=sam", "meta_tag=B")
    assert (a.meta_tag, a.selected.optimizee, a.selected.dataset) == ("A", "convnet4", "svhn")
    assert (b.meta_tag, b.selected.optimizee, b.selected.dataset) == ("B", "mlp", "fmnist")
    assert b.dataset.meta_split.enabled and b.optimizee.hidden == [16, 16]  # smoke_offline's narrow MLP
    with pytest.raises(ConfigError, match="round B of configs/rounds.yaml"):
        meta("meta_training=m3_lstm", "meta_tag=B", "dataset=svhn")
    adhoc = meta("meta_training=m3_lstm", "meta_tag=probe", "optimizee=mlp", "dataset=svhn")  # not a round: any pair
    assert (adhoc.selected.optimizee, adhoc.selected.dataset) == ("mlp", "svhn")
    with pytest.raises(ConfigError, match="letters, digits"):
        run("method=m3_lstm", "optimizee=convnet4", "dataset=svhn", "meta_tag=A@B")


@pytest.mark.parametrize("method", ["m2_adamw_reg", "m4_lstm_reg"])
def test_methods_with_a_regularizer_require_one(method):
    with pytest.raises(ConfigError, match=r"regularizer=<l1\|l2\|sam\|gam\|gsam>"):
        run(f"method={method}", "optimizee=convnet4", "dataset=svhn")


@pytest.mark.parametrize("method", ["m1_adamw", "m3_lstm"])
def test_methods_without_a_regularizer_reject_one(method):
    with pytest.raises(ConfigError, match="takes regularizer=<none>"):
        run(f"method={method}", "regularizer=sam", "optimizee=convnet4", "dataset=svhn")


def test_smooth_is_not_a_choice_of_r():
    with pytest.raises(ConfigError, match="Unknown regularizer 'smooth'"):
        run("method=m2_adamw_reg", "regularizer=smooth", "optimizee=convnet4", "dataset=svhn")
    with pytest.raises(ConfigError, match="Unknown smooth 'sam'"):
        meta("meta_training=m4_lstm_reg", "regularizer=sam", "smooth=sam")


@pytest.mark.parametrize("method, optimizer", [("m1_adamw", "lstm"), ("m3_lstm", "adamw")])
def test_optimizer_must_match_the_method(method, optimizer):
    with pytest.raises(ConfigError, match="optimizer"):
        run(f"method={method}", f"optimizer={optimizer}", "optimizee=convnet4", "dataset=svhn")


# --------------------------------------------------------------------------- Stage 1 (Algorithm 1)


def test_meta_m3_objective_is_l_meta_only():
    cfg = meta("meta_training=m3_lstm")
    assert (cfg.selected.optimizee, cfg.selected.dataset, cfg.selected.optimizer) == ("convnet4", "svhn", "lstm")
    assert cfg.regularizer.name == "none" and "smooth" not in cfg
    assert not cfg.meta_training.objective.smooth and not cfg.meta_training.objective.regularizer


@pytest.mark.parametrize("reg", REGULARIZERS)
def test_meta_m4_objective_has_smooth_and_reg(reg):
    cfg = meta("meta_training=m4_lstm_reg", f"regularizer={reg}")
    obj = cfg.meta_training.objective
    assert obj.smooth and obj.regularizer
    assert cfg.smooth.coef == 1.0 and cfg.regularizer.meta.coef == 0.1
    assert cfg.meta_training.inner.unroll_steps == 20 and cfg.meta_training.outer.lr == pytest.approx(3e-4)


def test_meta_m4_requires_a_regularizer():
    with pytest.raises(ConfigError, match=r"needs: regularizer=<l1\|l2\|sam\|gam\|gsam>"):
        meta("meta_training=m4_lstm_reg")


def test_meta_m3_rejects_a_regularizer():
    with pytest.raises(ConfigError, match="takes regularizer=<none>"):
        meta("meta_training=m3_lstm", "regularizer=sam")


def test_m3_and_m4_meta_training_share_every_setting_but_the_objective():
    """Guards against the v5 'meta-lr confound': whole composed configs (LSTM, optimizee, data,
    schedule …) may differ only in the objective — the regularizer r, L_smooth and bookkeeping."""
    from reglo.manifest import meta_differences

    m3 = OmegaConf.to_container(meta("meta_training=m3_lstm"), resolve=True)
    for reg in REGULARIZERS:
        m4 = OmegaConf.to_container(meta("meta_training=m4_lstm_reg", f"regularizer={reg}"), resolve=True)
        assert meta_differences(m3, m4) == []
        assert m4["meta_training"]["objective"] != m3["meta_training"]["objective"]


def test_meta_training_needs_a_dataset_with_a_meta_split():
    with pytest.raises(ConfigError, match="meta split"):  # an ad-hoc family (not a round) on CIFAR-10
        meta("meta_training=m3_lstm", "dataset=cifar10", "meta_tag=probe")


def test_method_checkpoint_matches_meta_training_output():
    for reg in REGULARIZERS:
        trained = meta_run_dir(meta("meta_training=m4_lstm_reg", f"regularizer={reg}")) / "phi_best.pt"
        used = run("method=m4_lstm_reg", f"regularizer={reg}", "optimizee=resnet18", "dataset=cifar10")
        assert Path(used.method.checkpoint) == trained
    trained = meta_run_dir(meta("meta_training=m3_lstm")) / "phi_best.pt"
    assert Path(run("method=m3_lstm", "optimizee=convnet4", "dataset=svhn").method.checkpoint) == trained


def test_a_run_has_one_entry_file():
    with pytest.raises(ConfigError, match="one entry file"):
        run("method=m1_adamw", "meta_training=m3_lstm", "optimizee=convnet4", "dataset=svhn")


# --------------------------------------------------------------------------- overrides and guards


def test_missing_optimizee_and_dataset_are_reported():
    with pytest.raises(ConfigError, match="dataset=.*optimizee="):
        run("method=m1_adamw")


def test_value_overrides_are_typed():
    cfg = run("method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "training.epochs=7", "optimizer.lr=5e-4")
    assert cfg.training.epochs == 7 and isinstance(cfg.training.epochs, int)
    assert cfg.optimizer.lr == pytest.approx(5e-4) and isinstance(cfg.optimizer.lr, float)


def test_typos_are_rejected():
    with pytest.raises(ConfigError, match="training.epcohs"):
        run("method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "training.epcohs=3")


def test_keys_of_unused_groups_are_rejected():
    with pytest.raises(ConfigError, match="smooth"):
        meta("meta_training=m3_lstm", "smooth.coef=0.2")


def test_plus_adds_a_key_on_purpose():
    cfg = run("method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "+notes=first_try")
    assert cfg.notes == "first_try"


def test_selected_is_read_only():
    with pytest.raises(ConfigError, match="read-only"):
        run("method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "selected.regularizer=sam")


def test_unknown_file_lists_the_choices():
    with pytest.raises(ConfigError, match="convnet4, mlp, resnet18"):
        run("method=m1_adamw", "optimizee=resnet50", "dataset=svhn")


def test_base_file_cannot_be_chosen():
    with pytest.raises(ConfigError, match="shared base"):
        meta("meta_training=base")


def test_malformed_override():
    with pytest.raises(ConfigError, match="key=value"):
        run("method=m1_adamw", "training.epochs")


def test_optimizee_follows_the_dataset():
    assert run("method=m1_adamw", "optimizee=resnet18", "dataset=cifar100").optimizee.num_classes == 100
    fm = run("method=m1_adamw", "optimizee=convnet4", "dataset=fmnist").optimizee
    assert (fm.in_channels, fm.image_size) == (1, 28)


# --------------------------------------------------------------------------- paths


def test_paths_default_to_the_repository_one_folder_per_profile():
    cfg = run("method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=production")
    assert Path(cfg.paths.root) == project_root()
    assert Path(cfg.paths.artifacts) == project_root() / "artifacts" / "production"
    smoke = run("method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=smoke")
    assert Path(smoke.paths.artifacts) == project_root() / "artifacts" / "smoke"
    assert smoke.paths.data == cfg.paths.data  # the datasets are shared


def test_paths_follow_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("REGLO_ARTIFACTS", str(tmp_path / "out"))
    monkeypatch.setenv("REGLO_DATA", str(tmp_path / "data"))
    cfg = run("method=m4_lstm_reg", "regularizer=l1", "optimizee=convnet4", "dataset=svhn", "seed=43", "study=main")
    assert Path(cfg.paths.data) == tmp_path / "data"
    assert run_dir(cfg) == tmp_path / "out/smoke_offline/runs/main/convnet4-svhn/m4_lstm_reg-l1@A/seed43"
    assert Path(cfg.method.checkpoint) == tmp_path / "out/smoke_offline/meta_training/A/m4_lstm_reg-l1/phi_best.pt"


def test_meta_tag_names_a_family_of_lstm_optimizers():
    """One knob, used both where Stage 1 writes φ and where M3/M4 read it (ablations, smoke runs)."""
    written = meta_run_dir(meta("meta_training=m4_lstm_reg", "regularizer=sam", "meta_tag=lam0.1"))
    assert written.parts[-2:] == ("lam0.1", "m4_lstm_reg-sam")
    read = run("method=m4_lstm_reg", "regularizer=sam", "optimizee=convnet4", "dataset=svhn", "meta_tag=lam0.1")
    assert Path(read.method.checkpoint) == written / "phi_best.pt"


def test_save_and_reload(tmp_path):
    cfg = run("method=m2_adamw_reg", "regularizer=gsam", "optimizee=vit_lite7", "dataset=cifar10")
    path = save_config(cfg, tmp_path / "run" / "config.yaml")
    assert OmegaConf.to_container(load_saved_config(path)) == OmegaConf.to_container(cfg)


# --------------------------------------------------------------------------- experiment grids (the study's stages)


def test_experiment_names():
    assert experiment_names() == [
        "extended_evaluation",
        "pilot",
        "stage1_meta_training",
        "stage2_optimizer_evaluation",
        "stage3_optimizee_evaluation",
        "stage4_analysis",
    ]


def test_stages_hold_the_figure_design():
    stage2, stage3 = load_experiment("stage2_optimizer_evaluation"), load_experiment("stage3_optimizee_evaluation")
    assert (stage2["stage"], stage3["stage"], stage2["study"], stage3["study"]) == (2, 3, "main", "main")
    assert stage2["rounds"] == stage3["rounds"] == ["A", "B"]  # every pair with every round's optimizers
    pool = stage2["train"]["pairs"] + stage3["train"]["pairs"]
    assert sorted(map(tuple, pool)) == sorted((o, d) for o in ("convnet4", "mlp", "resnet18")
                                              for d in ("svhn", "fmnist", "cifar10"))  # the 3 × 3 grid, once each
    # the thesis figure's pairs are all there
    for pair in (["convnet4", "svhn"], ["resnet18", "svhn"], ["convnet4", "fmnist"], ["resnet18", "cifar10"]):
        assert pair in pool


@pytest.mark.parametrize("name", ["stage2_optimizer_evaluation", "stage3_optimizee_evaluation", "extended_evaluation"])
def test_evaluation_grids_hold_all_four_methods(name):
    grid = load_experiment(name, profile="production")["train"]
    assert sum(len(regs) for regs in grid["methods"].values()) == 12 and grid["seeds"] == [42, 43, 44]
    assert set(grid["methods"]["m2_adamw_reg"]) == set(grid["methods"]["m4_lstm_reg"]) == set(REGULARIZERS)
    assert list(grid["from_stage1"]) == ["m2_adamw_reg"]  # on a source pair M2's runs are Stage 1's
    assert load_experiment(name, profile="smoke")["train"]["seeds"] == [42]  # the smoke test: the first seed


def test_stage1_trains_six_optimizers_per_round_and_m2_on_the_sources():
    spec = load_experiment("stage1_meta_training", profile="production")
    assert spec["rounds"] == ["A", "B"]
    assert sum(len(regs) for regs in spec["meta_train"]["runs"].values()) == 6
    assert spec["train"]["pairs"] == "sources" and list(spec["train"]["methods"]) == ["m2_adamw_reg"]
    assert set(spec["train"]["methods"]["m2_adamw_reg"]) == set(REGULARIZERS) and spec["train"]["seeds"] == [42, 43, 44]


@pytest.mark.parametrize("profile", ["production", "smoke", "smoke_offline"])
@pytest.mark.parametrize("name", ["stage1_meta_training", "stage2_optimizer_evaluation", "stage3_optimizee_evaluation",
                                  "extended_evaluation", "pilot"])
def test_every_grid_cell_composes(name, profile):
    """Each (pair, method, regularizer, seed) of a grid is a valid run under every profile, in the study's namespace."""
    from reglo.manifest import expand

    spec = load_experiment(name, profile=profile)
    variants = {}
    for line in expand(spec, kinds=["train"]):
        key = tuple(a for a in line if not a.startswith("seed="))
        if key not in variants:
            variants[key] = compose_run(line[1:])
        assert variants[key].study == spec["study"] and variants[key].profile == profile


@pytest.mark.parametrize("profile", ["production", "smoke", "smoke_offline"])
def test_every_meta_run_composes(profile):
    from reglo.manifest import expand

    spec = load_experiment("stage1_meta_training", profile=profile)
    sources = set()
    for line in expand(spec, kinds=["meta_train"]):
        cfg = compose_meta(line[1:])
        sources.add((str(cfg.meta_tag), str(cfg.selected.optimizee), str(cfg.selected.dataset)))
    assert sources == {("A", "convnet4", "svhn"), ("B", "mlp", "fmnist")}


def test_the_pilot_keeps_its_own_budget_only_in_production():
    production = load_experiment("pilot", profile="production")["train"]["overrides"]
    assert production == ["training.epochs=10"]
    assert load_experiment("pilot", profile="smoke")["train"]["overrides"] == []


def test_vit_gets_its_adamw_settings_only_in_adamw_runs():
    grid = load_experiment("extended_evaluation")["train"]
    vit = grid["optimizer_overrides"]["adamw"]["vit_lite7"]
    cfg = compose_run(["method=m2_adamw_reg", "regularizer=sam", "optimizee=vit_lite7", "dataset=cifar10", *vit])
    assert cfg.optimizer.lr == pytest.approx(5e-4) and cfg.optimizer.scheduler.warmup_epochs == 5
    with pytest.raises(ConfigError):  # the LSTM has no learning rate to override
        compose_run(["method=m3_lstm", "optimizee=vit_lite7", "dataset=cifar10", *vit])
