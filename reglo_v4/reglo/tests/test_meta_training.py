"""Stage 1 (Algorithm 1): the unroll, the meta-gradient, the objective, and the meta-trainer."""

import json
import runpy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from reglo.config import compose_meta, compose_run
from reglo.meta_training import Curriculum, InnerState, MetaTrainer, build_objective, loss_weights, unroll
from reglo.models import FunctionalModel
from reglo.models.optimizers import GradientFeatures, LSTMOptimizer, load_learned_optimizer
from reglo.regularizers import MetaPenalty
from reglo.utils.logging import read_jsonl
from reglo.utils.paths import meta_run_dir

TINY = [
    "meta_tag=unit",  # an ad-hoc family: round A (the default) would insist on its own source, ConvNet-4 / SVHN
    "dataset=synthetic",
    "dataset.train_size=256",
    "dataset.test_size=64",
    "dataset.image_size=16",
    "optimizee.channels=[4,4,4]",
    "meta_training.inner.unroll_steps=5",
    "meta_training.inner.batch_size=16",
    "meta_training.schedule.num_trajectories=2",
    "meta_training.schedule.trajectory_steps=10",
    "meta_training.schedule.curriculum.enabled=false",
    "meta_training.validation.every=1",
    "meta_training.validation.steps=5",
    "training.eval_batch_size=64",
    "device=cpu",
]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("REGLO_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("REGLO_ARTIFACTS", str(tmp_path / "artifacts"))
    return tmp_path


def tiny(*extra):
    return compose_meta(TINY + list(extra))


M4_SAM = ["meta_training=m4_lstm_reg", "regularizer=sam", "smooth.pga_steps=2"]


# --------------------------------------------------------------------------- the meta-gradient


def _toy_problem():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 5), nn.Tanh(), nn.Linear(5, 3)).double()
    lstm = LSTMOptimizer(num_inputs=1, num_layers=1, hidden_size=4, head_hidden=4, output_scale=0.1).double()
    gen = torch.Generator().manual_seed(1)
    batches = [(torch.randn(8, 4, generator=gen, dtype=torch.float64), torch.randint(0, 3, (8,), generator=gen)) for _ in range(4)]
    initial = [p.detach().clone() for p in model.parameters()]
    return FunctionalModel(model), lstm, GradientFeatures(["grad"], "rms"), batches, initial


def _meta_loss(fmodel, lstm, features, batches, initial, first_order):
    params = [p.clone().requires_grad_(True) for p in initial]
    h, c = lstm.init_state(sum(p.numel() for p in params), dtype=torch.float64)
    out = unroll(fmodel, lstm, features, InnerState(params, h, c), iter(batches), 3, first_order=first_order)
    return torch.stack(out.losses).mean()


def test_meta_gradient_matches_finite_differences():
    """With second-order terms kept, ∇_φ L_meta from the unroll is the exact derivative."""
    fmodel, lstm, features, batches, initial = _toy_problem()
    _meta_loss(fmodel, lstm, features, batches, initial, first_order=False).backward()
    checks = [(lstm.head[-1].weight, (0, 1)), (lstm.cells[0].ih.weight, (2, 0)), (lstm.cells[0].hh.weight, (5, 3))]
    for param, index in checks:
        analytic, h = param.grad[index].item(), 1e-6
        with torch.no_grad():
            param[index] += h
        up = _meta_loss(fmodel, lstm, features, batches, initial, first_order=False).item()
        with torch.no_grad():
            param[index] -= 2 * h
        down = _meta_loss(fmodel, lstm, features, batches, initial, first_order=False).item()
        with torch.no_grad():
            param[index] += h
        assert analytic == pytest.approx((up - down) / (2 * h), rel=1e-5, abs=1e-10)


def test_first_order_meta_gradient_reaches_every_lstm_weight():
    fmodel, lstm, features, batches, initial = _toy_problem()
    _meta_loss(fmodel, lstm, features, batches, initial, first_order=True).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lstm.parameters())
    assert sum(p.grad.abs().sum() for p in lstm.parameters()) > 0


def test_unroll_uses_t_plus_one_batches_and_tracks_steps():
    fmodel, lstm, features, batches, initial = _toy_problem()
    params = [p.clone().requires_grad_(True) for p in initial]
    h, c = lstm.init_state(sum(p.numel() for p in params), dtype=torch.float64)
    out = unroll(fmodel, lstm, features, InnerState(params, h, c, {}, 7), iter(batches), 3)
    assert len(out.losses) == 3 and len(out.batches) == 4 and len(out.train_losses) == 4
    assert out.final.step == 10 and not out.final.is_detached and out.final.detached().is_detached


# --------------------------------------------------------------------------- objective and schedule


def test_objective_combines_the_three_terms():
    losses = [torch.tensor(1.0), torch.tensor(2.0)]
    objective = build_objective(
        losses, loss_weights("uniform", 2), [torch.tensor(0.5), torch.tensor(1.5)], 0.1,
        MetaPenalty(torch.tensor(3.0), 0.7), 0.5,
    )
    assert objective.total.item() == pytest.approx(1.5 + 0.1 * 1.0 + 0.5 * 3.0)
    assert objective.parts == pytest.approx({"l_meta": 1.5, "l_smooth": 1.0, "l_reg": 0.7, "objective": 1.5 + 0.1 + 0.35})
    plain = build_objective(losses, loss_weights("uniform", 2), [], 0.1, None, 0.5)
    assert plain.total.item() == pytest.approx(1.5) and plain.parts["l_smooth"] == plain.parts["l_reg"] == 0.0


def test_loss_weights():
    for kind in ("uniform", "linear", "exponential"):
        w = loss_weights(kind, 50)
        assert w.sum().item() == pytest.approx(1.0) and len(w) == 50
    assert torch.all(loss_weights("linear", 5).diff() > 0)
    with pytest.raises(ValueError):
        loss_weights("random", 5)


def test_curriculum_moves_on_after_patience():
    cur = Curriculum([100, 200, 400], patience=2, enabled=True, default_length=999)
    assert cur.length == 100
    cur.report(1.0)
    assert not cur.report(1.5) and cur.report(1.2) and cur.length == 200  # two non-improvements
    cur.report(0.5)
    cur.report(0.6)
    cur.report(0.7)
    assert cur.length == 400 and cur.finished
    assert not cur.report(9.0) and not cur.report(9.0) and cur.length == 400  # stays at the last length
    assert Curriculum([1, 2], 1, enabled=False, default_length=999).length == 999


# --------------------------------------------------------------------------- the meta-trainer


def test_truncations_never_share_a_graph(workdir):
    trainer = MetaTrainer(tiny(*M4_SAM))
    fmodel, state = trainer.new_optimizee(0)
    for _ in range(3):  # each meta-step would fail if the previous graph were still attached
        state, row = trainer.meta_step(fmodel, state, 5)
        assert state.is_detached and all(p.requires_grad and p.is_leaf for p in state.params)
        assert row["l_reg"] != 0 and row["l_smooth"] > 0 and not row["skipped"]
    assert state.step == 15


def test_m4_meta_training_end_to_end(workdir):
    cfg = tiny(*M4_SAM)
    summary = MetaTrainer(cfg).fit()
    out = meta_run_dir(cfg)
    assert summary["status"] == "completed" and summary["trajectories"] == 2 and summary["meta_steps"] == 4
    for name in ("config.yaml", "environment.json", "meta_log.jsonl", "phi_best.pt", "phi_last.pt", "summary.json"):
        assert (out / name).is_file(), name
    rows = read_jsonl(out / "meta_log.jsonl")
    steps = [r for r in rows if r["kind"] == "meta_step"]
    assert len(steps) == 4 and all(r["l_smooth"] > 0 and r["l_reg"] != 0 for r in steps)
    assert len([r for r in rows if r["kind"] == "validation"]) == 2
    lstm, features, meta = load_learned_optimizer(out / "phi_best.pt")
    assert meta["regularizer"] == "sam" and meta["meta_training"] == "m4_lstm_reg"
    # the M4 run with r = sam loads exactly this file
    run = compose_run(["method=m4_lstm_reg", "regularizer=sam", "optimizee=convnet4", "dataset=synthetic", "meta_tag=unit"])
    assert Path(run.method.checkpoint) == out / "phi_best.pt"
    again = MetaTrainer(cfg).fit()  # finished runs are not redone
    assert again == summary


def test_m3_meta_training_has_no_extra_terms(workdir):
    cfg = tiny("meta_training=m3_lstm", "meta_training.schedule.num_trajectories=1")
    MetaTrainer(cfg).fit()
    steps = [r for r in read_jsonl(meta_run_dir(cfg) / "meta_log.jsonl") if r["kind"] == "meta_step"]
    assert steps and all(r["l_smooth"] == 0 and r["l_reg"] == 0 for r in steps)
    assert all(r["objective"] == pytest.approx(r["l_meta"]) for r in steps)


def test_resume_after_a_killed_job(workdir):
    first = tiny("meta_training=m3_lstm", "meta_training.schedule.num_trajectories=1")
    MetaTrainer(first).fit()
    out = meta_run_dir(first)
    (out / "summary.json").unlink()  # as if the walltime killed the job before it finished
    summary = MetaTrainer(tiny("meta_training=m3_lstm")).fit()
    trajectories = [r["trajectory"] for r in read_jsonl(out / "meta_log.jsonl") if r["kind"] == "meta_step"]
    assert summary["trajectories"] == 2 and trajectories == [0, 0, 1, 1]  # trajectory 0 was not repeated


@pytest.mark.parametrize("reg", ["l1", "l2", "sam", "gsam", "gam"])
def test_phi_space_regularizers_run(workdir, reg):
    """Algorithm 1, lines 21–22: every r acts on the LSTM's weights φ (the configs' default)."""
    trainer = MetaTrainer(tiny("meta_training=m4_lstm_reg", f"regularizer={reg}", "smooth.pga_steps=1"))
    assert trainer.reg_space == "phi" and trainer.regularizer.replays == (reg in ("sam", "gam", "gsam"))
    fmodel, state = trainer.new_optimizee(0)
    before = [p.detach().clone() for p in trainer.lstm.parameters()]
    _, row = trainer.meta_step(fmodel, state, 5)
    assert not row["skipped"] and row["l_reg"] == row["l_reg"]  # finite (not NaN)
    # Adam's first step moves each weight by at most lr; a φ-perturbation left behind
    # (‖ε̂‖ = ρ = 0.05 for SAM/GSAM) would push some weights further.
    largest = max((p.detach() - b).abs().max().item() for p, b in zip(trainer.lstm.parameters(), before))
    assert 0 < largest <= 3e-4 * 1.0001


def test_meta_train_script(workdir, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts" / "meta_train.py"
    main = runpy.run_path(str(script))["main"]
    assert main(TINY + ["meta_training=m3_lstm", "meta_training.schedule.num_trajectories=1"]) == 0
    assert main(["meta_training=m4_lstm_reg"]) == 2  # missing regularizer → clear error, exit code 2
    assert "regularizer=<l1|l2|sam|gam|gsam>" in capsys.readouterr().err
    folder = workdir / "artifacts" / "smoke_offline" / "meta_training" / "unit" / "m3_lstm"
    summary = json.loads((folder / "summary.json").read_text())
    assert summary["status"] == "completed"


# ───────────────────────────────────────────── validation, walltime, reruns


def test_validation_averages_several_fresh_optimizees(workdir):
    trainer = MetaTrainer(tiny("meta_training=m3_lstm", "meta_training.validation.optimizees=3"))
    result = trainer.validate()
    assert len(result["val_losses"]) == 3 and len(set(result["val_losses"])) == 3  # three different optimizees
    assert result["val_loss"] == pytest.approx(sum(result["val_losses"]) / 3)
    assert trainer.validate()["val_losses"] == result["val_losses"]  # the same three every time


def test_max_hours_stops_cleanly_and_the_same_command_continues(workdir):
    script = Path(__file__).resolve().parents[1] / "scripts" / "meta_train.py"
    main = runpy.run_path(str(script))["main"]
    args = TINY + ["meta_training=m3_lstm", "meta_training.checkpoint_every=5"]
    assert main(args + ["max_hours=1e-7"]) == 75  # EX_TEMPFAIL: stopped at max_hours
    out = meta_run_dir(tiny("meta_training=m3_lstm"))
    assert (out / "phi_last.pt").is_file() and not (out / "summary.json").exists()
    assert main(args) == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["status"] == "completed" and summary["trajectories"] == 2
    trajectories = [r["trajectory"] for r in read_jsonl(out / "meta_log.jsonl") if r["kind"] == "meta_step"]
    assert trajectories == [0, 0, 1, 1]  # the first trajectory was not repeated


def test_rerun_starts_from_scratch(workdir):
    cfg = tiny("meta_training=m3_lstm", "meta_training.schedule.num_trajectories=1")
    MetaTrainer(cfg).fit()
    out = meta_run_dir(cfg)
    MetaTrainer(tiny("meta_training=m3_lstm", "meta_training.schedule.num_trajectories=1", "meta_training.rerun=true")).fit()
    steps = [r for r in read_jsonl(out / "meta_log.jsonl") if r["kind"] == "meta_step"]
    assert [r["trajectory"] for r in steps] == [0, 0]  # the old log is gone, nothing was resumed


def test_a_run_whose_validations_never_finish_still_leaves_an_optimizer(workdir, monkeypatch):
    trainer = MetaTrainer(tiny("meta_training=m3_lstm", "meta_training.schedule.num_trajectories=1"))
    monkeypatch.setattr(trainer, "validate", lambda: {"val_loss": float("inf"), "val_acc": 0.0, "diverged": True})
    summary = trainer.fit()
    assert summary["selected"] == "last (no finite validation)" and (trainer.out_dir / "phi_best.pt").is_file()
