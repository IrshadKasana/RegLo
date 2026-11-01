"""Stages 2–3: M1–M4 training runs, their records, compute accounting and resumption."""

import json
import runpy
from pathlib import Path

import pytest
import torch

from reglo.config import compose_run
from reglo.models.optimizers import build_lstm_optimizer, save_learned_optimizer
from reglo.training import CheckpointMismatch, OptimizeeTrainer, StepResult
from reglo.training.methods import Method
from reglo.utils.logging import read_jsonl
from reglo.utils.paths import run_dir

TINY = [
    "optimizee=convnet4",
    "dataset=synthetic",
    "dataset.train_size=256",
    "dataset.test_size=64",
    "dataset.image_size=16",
    "optimizee.channels=[4,4,4]",
    "training.epochs=2",
    "training.batch_size=32",
    "training.eval_batch_size=64",
    "device=cpu",
    "study=test",
]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("REGLO_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("REGLO_ARTIFACTS", str(tmp_path / "artifacts"))
    return tmp_path


def make_phi(workdir, tag, entry, regularizer):
    """A stand-in Stage-1 result: a fresh LSTM saved where M3/M4 look for it."""
    cfg = compose_run(["method=m3_lstm", *TINY])
    lstm, features = build_lstm_optimizer(cfg.optimizer)
    path = workdir / "artifacts" / "smoke_offline" / "meta_training" / "A" / tag / "phi_best.pt"  # round A: the default
    save_learned_optimizer(path, lstm, features, meta_training=entry, regularizer=regularizer)
    return path


def train(*overrides):
    cfg = compose_run(TINY + list(overrides))  # the test's own overrides win
    return OptimizeeTrainer(cfg).run(), run_dir(cfg)


VARIANTS = [
    (["method=m1_adamw"], 1),
    (["method=m2_adamw_reg", "regularizer=l1"], 1),
    (["method=m2_adamw_reg", "regularizer=l1", "regularizer.explicit.mode=proximal"], 1),
    (["method=m2_adamw_reg", "regularizer=l2"], 1),
    (["method=m2_adamw_reg", "regularizer=l2", "regularizer.explicit.mode=decoupled"], 1),
    (["method=m2_adamw_reg", "regularizer=sam"], 2),
    (["method=m2_adamw_reg", "regularizer=gam"], 2),
    (["method=m2_adamw_reg", "regularizer=gsam"], 2),
    (["method=m3_lstm"], 1),
    (["method=m4_lstm_reg", "regularizer=sam"], 1),
]


@pytest.mark.parametrize("overrides, passes", VARIANTS)
def test_every_method_variant_trains(workdir, overrides, passes):
    make_phi(workdir, "m3_lstm", "m3_lstm", "none")
    make_phi(workdir, "m4_lstm_reg-sam", "m4_lstm_reg", "sam")
    final, _ = train(*overrides, "training.epochs=1")
    assert final["status"] == "completed" and final["passes_per_step"] == passes
    assert 0.0 <= final["test_top1"] <= 1.0 and final["test_loss"] == final["test_loss"]


def test_a_run_records_everything_the_tables_need(workdir):
    final, out = train("method=m1_adamw")
    for name in ("config.yaml", "environment.json", "metrics.jsonl", "final.json", "weights.json", "model_final.pt"):
        assert (out / name).is_file(), name
    assert not (out / "last.pt").exists()  # removed once the run is complete
    assert final["gen_gap"] == pytest.approx(final["test_loss"] - final["train_loss"])
    assert final["steps"] == 2 * final["steps_per_epoch"] == 8  # 128 optimizee-half images / batch 32 × 2 epochs
    assert final["weights"]["num_params"] > final["weights"]["num_penalised"] > 0
    assert final["flops_per_step"] > 0 and final["total_gflops"] == pytest.approx(final["flops_per_step"] * 8 / 1e9)
    epochs = [r for r in read_jsonl(out / "metrics.jsonl") if r["kind"] == "epoch"]
    assert [r["epoch"] for r in epochs] == [1, 2] and all("test_top1" in r for r in epochs)
    assert out.parts[-4:] == ("test", "convnet4-synthetic", "m1_adamw", "seed42")
    assert train("method=m1_adamw")[0] == final  # a finished run is not redone


def test_compute_accounting_separates_the_methods(workdir):
    make_phi(workdir, "m3_lstm", "m3_lstm", "none")
    flops = {}
    for name, overrides in {"m1": ["method=m1_adamw"], "sam": ["method=m2_adamw_reg", "regularizer=sam"],
                            "m3": ["method=m3_lstm"]}.items():
        flops[name] = train(*overrides, "training.epochs=1")[0]["flops_per_step"]
    assert 1.9 < flops["sam"] / flops["m1"] < 2.1  # SAM: two forward–backward passes
    assert flops["m3"] > flops["m1"]  # the LSTM's own step is counted


def test_m4_refuses_an_lstm_trained_for_another_regularizer(workdir):
    make_phi(workdir, "m4_lstm_reg-sam", "m4_lstm_reg", "l1")
    with pytest.raises(CheckpointMismatch, match="regularizer=l1"):
        train("method=m4_lstm_reg", "regularizer=sam")
    make_phi(workdir, "m3_lstm", "m4_lstm_reg", "sam")
    with pytest.raises(CheckpointMismatch, match="m4_lstm_reg"):
        train("method=m3_lstm")


def test_a_round_uses_only_optimizers_meta_trained_on_its_source(workdir):
    cfg = compose_run(["method=m3_lstm", *TINY])
    lstm, features = build_lstm_optimizer(cfg.optimizer)
    path = workdir / "artifacts" / "smoke_offline" / "meta_training" / "B" / "m3_lstm" / "phi_best.pt"
    save_learned_optimizer(path, lstm, features, meta_training="m3_lstm", regularizer="none",
                           config={"selected": {"optimizee": "convnet4", "dataset": "svhn"}})  # round A's pair
    with pytest.raises(CheckpointMismatch, match="round B is meta-trained on mlp / fmnist"):
        train("method=m3_lstm", "meta_tag=B")
    save_learned_optimizer(path, lstm, features, meta_training="m3_lstm", regularizer="none",
                           config={"selected": {"optimizee": "mlp", "dataset": "fmnist"}})
    final, out = train("method=m3_lstm", "meta_tag=B", "training.epochs=1")
    assert out.parts[-2] == "m3_lstm@B"  # the same pair, trained with each round's φ, side by side
    assert (final["round"], final["phi_source"], final["role"]) == ("B", "mlp-fmnist", "unseen_both")
    m1, _ = train("method=m1_adamw", "meta_tag=B", "training.epochs=1")
    assert m1["round"] is None and m1["regularizer_applied"] == "never"


def test_m3_without_stage_1_explains_what_to_run(workdir):
    with pytest.raises(FileNotFoundError, match="Stage 1"):
        train("method=m3_lstm")


def test_same_seed_gives_every_method_the_same_starting_weights(workdir):
    a = OptimizeeTrainer(compose_run(["method=m1_adamw", *TINY])).build_model()
    b = OptimizeeTrainer(compose_run(["method=m2_adamw_reg", "regularizer=sam", *TINY])).build_model()
    c = OptimizeeTrainer(compose_run(["method=m1_adamw", "seed=43", *TINY])).build_model()
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters()))


def test_an_interrupted_run_resumes_exactly(workdir, monkeypatch):
    overrides = ["method=m2_adamw_reg", "regularizer=sam", "training.epochs=3", "training.checkpoint_every=1"]
    reference, ref_dir = train(*overrides, "seed=7", "study=reference")

    original = OptimizeeTrainer._train_epoch

    def killed_in_epoch_3(self, method, data, tracker, state, cap, metrics):
        if state["epoch"] == 2:
            raise KeyboardInterrupt("walltime")
        return original(self, method, data, tracker, state, cap, metrics)

    monkeypatch.setattr(OptimizeeTrainer, "_train_epoch", killed_in_epoch_3)
    with pytest.raises(KeyboardInterrupt):
        train(*overrides, "seed=7")
    monkeypatch.setattr(OptimizeeTrainer, "_train_epoch", original)
    resumed, out = train(*overrides, "seed=7")

    epochs = [r["epoch"] for r in read_jsonl(out / "metrics.jsonl") if r["kind"] == "epoch"]
    assert epochs == [1, 2, 3] and resumed["epochs_completed"] == 3
    a = torch.load(ref_dir / "model_final.pt")
    b = torch.load(out / "model_final.pt")
    assert all(torch.allclose(a[k].float(), b[k].float()) for k in a)
    assert resumed["test_loss"] == pytest.approx(reference["test_loss"])


def test_divergence_is_recorded_not_crashed(workdir, monkeypatch):
    original = Method.step
    calls = {"n": 0}

    def explode(self, x, y):
        calls["n"] += 1
        return StepResult(float("nan")) if calls["n"] > 3 else original(self, x, y)

    monkeypatch.setattr(Method, "step", explode)
    final, _ = train("method=m1_adamw", "training.count_flops=false")
    assert final["status"] == "diverged" and final["steps"] == 3


def test_at_convergence_snapshot(workdir):
    final, out = train("method=m1_adamw", "optimizer.lr=0.0", "training.convergence_patience=3")
    assert final["converged_step"] is not None and (out / "model_converged.pt").is_file()


def test_lstm_methods_compute_no_regularizer(workdir):
    make_phi(workdir, "m4_lstm_reg-gsam", "m4_lstm_reg", "gsam")
    cfg = compose_run(["method=m4_lstm_reg", "regularizer=gsam", *TINY])
    trainer = OptimizeeTrainer(cfg)
    from reglo.training import build_method

    method = build_method(cfg, trainer.build_model(), 4, 8, torch.device("cpu"))
    assert method.regularizer is None and method.passes == 1 and method.lr is None


def test_train_script(workdir, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_optimizee.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["method=m1_adamw", *TINY, "training.epochs=1"]) == 0
    assert main(["method=m2_adamw_reg", *TINY]) == 2
    assert "regularizer=<l1|l2|sam|gam|gsam>" in capsys.readouterr().err
    assert main(["method=m3_lstm", *TINY]) == 3  # no Stage-1 checkpoint yet
    final = json.loads((workdir / "artifacts/smoke_offline/runs/test/convnet4-synthetic/m1_adamw/seed42/final.json")
                       .read_text())
    assert final["status"] == "completed"


def test_max_hours_stops_after_an_epoch_and_the_run_continues(workdir):
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_optimizee.py"
    main = runpy.run_path(str(script))["main"]
    args = TINY + ["method=m1_adamw", "training.epochs=3", "training.checkpoint_every=10"]
    assert main(args + ["max_hours=1e-7"]) == 75
    out = run_dir(compose_run(args))
    assert (out / "last.pt").is_file() and not (out / "final.json").exists()
    assert main(args) == 0
    final = json.loads((out / "final.json").read_text())
    assert final["status"] == "completed" and final["epochs_completed"] == 3 and final["profile"] == "smoke_offline"
    epochs = [r["epoch"] for r in read_jsonl(out / "metrics.jsonl") if r.get("kind") == "epoch"]
    assert epochs == [1, 2, 3]


def test_the_training_set_is_evaluated_on_its_own_cadence(workdir):
    _, out = train("method=m1_adamw", "training.epochs=4", "training.train_eval_every=3")
    rows = [r for r in read_jsonl(out / "metrics.jsonl") if r.get("kind") == "epoch"]
    assert [("train_top1" in r) for r in rows] == [False, False, True, True]  # epoch 3, and always the last
    assert all("test_top1" in r for r in rows)
    assert "gen_gap_top1" in rows[-1]


def test_rerun_does_not_resume_an_old_attempt(workdir):
    final, out = train("method=m1_adamw", "training.epochs=1")
    (out / "last.pt").write_bytes(b"stale")  # what an interrupted earlier attempt would leave
    final, _ = train("method=m1_adamw", "training.epochs=1", "training.rerun=true")
    assert final["status"] == "completed" and not (out / "last.pt").exists()


def test_a_production_run_without_a_gpu_stops_with_a_clear_message(workdir, capsys):
    if torch.cuda.is_available():
        pytest.skip("this machine has a GPU")
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_optimizee.py"
    main = runpy.run_path(str(script))["main"]
    assert main(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn", "profile=production"]) == 3
