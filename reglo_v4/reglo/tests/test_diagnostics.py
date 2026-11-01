"""Stage 4: Hessian probes, the paper's PoC tests, weight statistics, run audits and compute profiling."""

import json
import runpy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from reglo.config import compose_run, load_experiment
from reglo.diagnostics import (
    ConvergenceTracker,
    hessian_trace,
    lstm_flops_per_coordinate,
    max_grad_norm_in_ball,
    max_loss_in_ball,
    top_eigenvalue,
    weight_statistics,
)
from reglo.diagnostics.audit import run_audit
from reglo.diagnostics.compute import profile_pair
from reglo.models.optimizees import ConvNet4
from reglo.models.optimizers import LSTMOptimizer
from reglo.training import OptimizeeTrainer
from reglo.utils.paths import diagnostics_dir_for, find_runs, run_dir


class Quadratic(nn.Module):
    """L(θ) = ½ θᵀAθ: every diagnostic has a closed-form answer."""

    def __init__(self, a: torch.Tensor, theta0=None):
        super().__init__()
        self.register_buffer("a", a)
        self.theta = nn.Parameter(torch.zeros(a.shape[0]) if theta0 is None else theta0.clone())

    def terms(self):
        return [(lambda: 0.5 * self.theta @ self.a @ self.theta, 1.0)]


def random_spd(eigenvalues, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(len(eigenvalues), len(eigenvalues), generator=gen, dtype=torch.float64))
    return (q @ torch.diag(torch.tensor(eigenvalues, dtype=torch.float64)) @ q.T).float()


# --------------------------------------------------------------------------- Hessian


def test_power_iteration_finds_the_top_eigenvalue():
    model = Quadratic(random_spd([5.0, 2.0, 1.0, 0.5, 0.1]), theta0=torch.randn(5))
    result = top_eigenvalue(model.terms(), [model.theta], iterations=200, tol=1e-7)
    assert result["lambda_max"] == pytest.approx(5.0, rel=1e-4)


def test_hutchinson_is_exact_for_a_diagonal_hessian_and_unbiased_otherwise():
    diagonal = Quadratic(torch.diag(torch.tensor([3.0, 1.0, 0.5, 2.0])))
    assert hessian_trace(diagonal.terms(), [diagonal.theta], samples=3)["trace"] == pytest.approx(6.5)
    full = Quadratic(random_spd([4.0, 3.0, 2.0, 1.0, 0.5, 0.5]))
    result = hessian_trace(full.terms(), [full.theta], samples=2000)
    assert result["trace"] == pytest.approx(11.0, abs=4 * result["trace_stderr"])


# --------------------------------------------------------------------------- the paper's PoC tests


def test_max_loss_in_the_ball_reaches_the_worst_corner():
    a = torch.tensor([2.0, 1.0, 0.5])
    model = Quadratic(torch.diag(a))
    result = max_loss_in_ball(model.terms(), [model.theta], radius=0.1, steps=10)
    assert result["loss"] == 0.0
    assert result["max_loss"] == pytest.approx(0.5 * 0.01 * a.sum().item(), rel=1e-5)  # ½ r² Σ aᵢ
    assert result["surrogate_gap"] == pytest.approx(result["max_loss"])
    assert torch.equal(model.theta.detach(), torch.zeros(3))  # weights restored


def test_max_grad_norm_in_the_ball():
    a = torch.tensor([2.0, 1.0, 0.5])
    model = Quadratic(torch.diag(a))
    result = max_grad_norm_in_ball(model.terms(), [model.theta], radius=0.1, steps=10)
    assert result["grad_norm"] == pytest.approx(0.0, abs=1e-12)
    assert result["max_grad_norm"] == pytest.approx(0.1 * a.norm().item(), rel=1e-5)  # ‖A·r·1‖
    assert torch.equal(model.theta.detach(), torch.zeros(3))


def test_relative_ball_scales_with_the_weights():
    model = Quadratic(torch.eye(2), theta0=torch.tensor([1.0, 0.0]))
    result = max_loss_in_ball(model.terms(), [model.theta], radius=0.1, steps=10, relative=True)
    assert result["max_loss"] == pytest.approx(0.5 * 1.1**2, rel=1e-5)  # the zero weight cannot move


# --------------------------------------------------------------------------- weights, convergence, FLOPs


def test_weight_statistics():
    model = ConvNet4(channels=(4, 4, 4))
    with torch.no_grad():
        model.features[0].weight.zero_()
    stats = weight_statistics(model, threshold=1e-3)
    conv = 3 * 4 * 9
    assert stats["sparsity"] >= conv / stats["num_penalised"]
    assert stats["num_params"] > stats["num_penalised"]
    assert stats["l2"] == pytest.approx(torch.cat([model.features[4].weight.flatten(), model.features[8].weight.flatten(),
                                                   model.classifier.weight.flatten()]).norm().item(), rel=1e-5)
    assert sum(stats["histogram"]["counts"]) == stats["num_penalised"]


def test_convergence_tracker_fires_once():
    tracker = ConvergenceTracker(patience=3, smoothing=0.0)
    fired = [tracker.update(step, loss) for step, loss in enumerate([3, 2, 1, 1, 1, 1, 0.5, 1, 1, 1], start=1)]
    assert fired.count(True) == 1 and tracker.converged_step == 6


LSTM_FLOPS = 2 * (4 * 20 * (1 + 20) + 4 * 20 * (20 + 20) + 20 * 20 + 20)  # 10,600


def test_lstm_flops_per_parameter_matches_the_architecture():
    # 2 layers of 20: 4·20·(1+20) + 4·20·(20+20) MACs; head 20·20 + 20; ×2 for FLOPs
    assert lstm_flops_per_coordinate(LSTMOptimizer(num_inputs=1)) == LSTM_FLOPS == 10_600


# --------------------------------------------------------------------------- audits of finished runs

TINY = [
    "optimizee=convnet4", "dataset=synthetic", "dataset.train_size=256", "dataset.test_size=64",
    "dataset.image_size=16", "optimizee.channels=[4,4,4]", "training.epochs=2", "training.batch_size=32",
    "training.eval_batch_size=64", "device=cpu", "study=main", "training.convergence_patience=2",
]


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.setenv("REGLO_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("REGLO_ARTIFACTS", str(tmp_path / "artifacts"))
    return tmp_path


def small_spec():
    spec = load_experiment("stage4_analysis")["diagnostics"]
    spec["hessian"].update(power_iterations=5, hutchinson_samples=3, num_batches=2)
    spec["sharpness"].update(radii=[0.001, 0.01], pga_steps=2, num_batches=1)
    return spec


def test_audit_of_a_finished_run(workdir):
    cfg = compose_run(["method=m1_adamw", *TINY])
    OptimizeeTrainer(cfg).run()
    run = run_dir(cfg)
    assert find_runs(workdir / "artifacts" / "smoke_offline", "main") == [run]
    result = run_audit(run, small_spec(), torch.device("cpu"))
    target = diagnostics_dir_for(run) / "diagnostics.json"
    assert target.is_file() and target.parts[-6:-1] == ("diagnostics", "main", "convnet4-synthetic", "m1_adamw", "seed42")
    assert result["status"] == "completed" and result["label"] == "M1"
    assert set(result["hessian"]) == {"final"} and result["hessian"]["final"]["lambda_max"] > 0
    assert set(result["sharpness"]) == {"final", "converged"}  # at completion and at convergence
    for snapshot in result["sharpness"].values():
        assert set(snapshot) == {"0.001", "0.01"}
        assert snapshot["0.01"]["max_loss"] >= snapshot["0.01"]["loss"]
        assert snapshot["0.01"]["max_grad_norm"] >= snapshot["0.01"]["grad_norm"] * 0.999
    assert set(result["weights"]) == {"final", "converged"}


def test_diverged_runs_are_skipped_with_a_reason(workdir):
    cfg = compose_run(["method=m1_adamw", *TINY])
    run = run_dir(cfg)
    run.mkdir(parents=True)
    (run / "final.json").write_text(json.dumps({"status": "diverged", "label": "M1"}))
    assert run_audit(run, small_spec())["status"] == "skipped"


def test_compute_profile_ranks_the_methods(workdir):
    variants = [("m1_adamw", "none"), ("m2_adamw_reg", "sam"), ("m3_lstm", "none")]
    result = profile_pair(TINY, variants, warmup=1, timed=2, device=torch.device("cpu"))
    rows = {r["method"]: r for r in result["rows"]}
    assert 1.9 < rows["m2_adamw_reg"]["flops_vs_m1"] < 2.1 and rows["m2_adamw_reg"]["passes"] == 2
    assert rows["m3_lstm"]["phi"] == "untrained" and rows["m3_lstm"]["flops_vs_m1"] > 1
    assert all(r["seconds_per_step"] > 0 for r in result["rows"])
    assert result["lstm_flops_per_parameter"] == LSTM_FLOPS
    assert all(r["estimated_hours_per_run"] > 0 for r in result["rows"]) and result["profile"] == "smoke_offline"


def test_diagnostics_scripts(workdir, capsys):
    OptimizeeTrainer(compose_run(["method=m1_adamw", *TINY])).run()
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    diag = runpy.run_path(str(scripts / "run_diagnostics.py"))["main"]
    assert diag(["--list"]) == 0
    listed = capsys.readouterr().out.split()
    assert len(listed) == 1 and listed[0].endswith("m1_adamw/seed42")
    assert diag(["--index", "3"]) == 2  # out of range
    profile = runpy.run_path(str(scripts / "profile_compute.py"))["main"]
    assert profile([*TINY, "--warmup", "1", "--timed", "1", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "M2 sam" in out and "M4 gsam" in out
    assert (workdir / "artifacts/smoke_offline/diagnostics/compute/convnet4-synthetic.json").is_file()
