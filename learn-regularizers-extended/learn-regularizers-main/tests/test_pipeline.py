"""Regression tests for the hierarchical pipeline (CPU, synthetic data, no downloads).

    python -m pytest tests/test_pipeline.py -q        or        python tests/test_pipeline.py
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from learning_regularizers.pipeline import DMO_DIR  # noqa: E402,F401
from learning_regularizers.pipeline.config import load_pipeline_config, meta_training_args  # noqa: E402
from learning_regularizers.pipeline.learned_optimizer import LSTMLearnedOptimizer, smooth_penalty  # noqa: E402
from learning_regularizers.pipeline.methods import METHODS, RunKey, run_grid  # noqa: E402

SYN = ["experiment.synthetic_data=true", "experiment.name=pytest"]


def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def test_base_meta_update_has_no_gradient_path():
    """Documents the base-code bug this pipeline fixes: the LSTM never receives meta-gradients."""
    import torch.nn.functional as F
    mo = _quiet(__import__, "meta_optimizer")
    mo.device = torch.device("cpu")   # base module picks cuda when available; these tests run on CPU
    md = _quiet(__import__, "model")
    meta_model = md.initialize_model("mnist", "mlp")
    lo = mo.MetaOptimizerLSTM(mo.MetaModel(meta_model), 2, 20)
    model = md.initialize_model("mnist", "mlp")
    lo.reset_lstm(keep_states=False, model=model)
    x, y = torch.randn(8, 1, 28, 28), torch.randint(0, 10, (8,))
    for _ in range(3):
        loss = F.nll_loss(model(x), y)
        model.zero_grad()
        loss.backward()
        out = lo.meta_update(model)
    F.nll_loss(out(x), y).backward()
    assert all(p.grad is None for p in lo.parameters())


def test_lstm_matches_base_and_checkpoints_interchange():
    mo = _quiet(__import__, "meta_optimizer")
    mo.device = torch.device("cpu")   # base module picks cuda when available; these tests run on CPU
    ours = LSTMLearnedOptimizer(2, 20)
    base = mo.MetaOptimizerLSTM(mo.MetaModel(torch.nn.Linear(2, 2)), 2, 20)
    base.load_state_dict(ours.state_dict())
    ours.load_state_dict(base.state_dict())
    g = torch.randn(500) * torch.logspace(-7, 0, 500)
    state = ours.init_state(500)
    base.h0, base.c0 = state[0].clone(), state[1].clone()
    for _ in range(3):
        u, state = ours(g, state)
        ub = base(g.view(-1, 1).unsqueeze(1))
    assert torch.allclose(u, ub.detach(), atol=1e-6)
    st2 = ours.init_state(500)
    for _ in range(3):
        uc = ours.forward_chunked(g, st2, 77)
    assert torch.allclose(u.detach(), uc, atol=1e-6)


def test_meta_gradient_matches_finite_differences():
    from learning_regularizers.pipeline.meta_train import MetaTrainer
    torch.set_default_dtype(torch.float64)
    try:
        cfg = load_pipeline_config(sets=SYN + [
            "meta_training.overrides={truncated_bptt_step: 1, optimizee_steps: 1, batch_size: 16, if_cl: false}",
            "models.convnet4.width=4"])
        mt = _quiet(MetaTrainer, cfg, RunKey(5, None, "l2"), torch.device("cpu"))
        mt.lo.double()
        model, names, params = mt._new_optimizee(seed=3)
        model.double()
        params = [p.double() for p in params]
        x, y = mt.stream.next()
        batches = [(x.double(), y)]
        state = mt.lo.init_state(sum(p.numel() for p in params), dtype=torch.float64)
        snap = [b.clone() for b in model.buffers()]

        def meta_loss():
            for b, s in zip(model.buffers(), snap):
                b.copy_(s)
            return mt._unroll(model, names, params, state, batches)[0]

        mt.lo.zero_grad()
        meta_loss().backward()
        assert all(p.grad is not None for p in mt.lo.parameters())
        torch.manual_seed(0)
        d = [torch.randn_like(p) for p in mt.lo.parameters()]
        eps = 1e-6
        with torch.no_grad():
            for p, v in zip(mt.lo.parameters(), d):
                p.add_(eps * v)
            lp = float(meta_loss())
            for p, v in zip(mt.lo.parameters(), d):
                p.sub_(2 * eps * v)
            lm = float(meta_loss())
        fd = (lp - lm) / (2 * eps)
        an = float(sum((p.grad * v).sum() for p, v in zip(mt.lo.parameters(), d)))
        assert abs(an - fd) <= 1e-6 * max(1.0, abs(fd)), (an, fd)
    finally:
        torch.set_default_dtype(torch.float32)


def test_smooth_penalty_has_gradient():
    lo = LSTMLearnedOptimizer(2, 20)
    g = torch.randn(300) * 1e-3
    state = lo.init_state(300)
    _, state = lo(g, state)              # non-zero hidden state
    u, _ = lo(g, state)
    r = smooth_penalty(lo, g, state, u, eps=1.0, num_steps=5)
    r.backward()
    assert r.item() >= 0 and all(p.grad is not None and torch.isfinite(p.grad).all() for p in lo.parameters())


def test_run_grid_and_regularizer_routing():
    cfg = load_pipeline_config(sets=SYN)
    keys = list(run_grid(cfg.methods, cfg.base_optimizers, cfg.regularizers))
    assert len(keys) == 2 + 10 + 1 + 1 + 5 + 5
    # training: R only where the method has R; smooth only for 4 and 6
    for m in (3, 4, 5, 6):
        a = meta_training_args(cfg, METHODS[m], "gam")
        assert a.if_smooth == (m in (4, 6))
        assert a.regularizer == ("gam" if m in (5, 6) else "none")
    # evaluation: regularizers stripped
    from learning_regularizers.pipeline.phases import _descriptor
    assert _descriptor(cfg, RunKey(2, "sgd", "sam"), phase=1).regularizer == "sam"
    assert _descriptor(cfg, RunKey(2, "sgd", "sam"), phase=2).regularizer is None
    assert _descriptor(cfg, RunKey(5, None, "sam"), phase=2).kind == "lo"


def test_base_yaml_files_untouched():
    import hashlib
    expected = {  # sha1 of the original files shipped with the base project
        "base/meta_train.yaml": "d2f9dc81efd8f62f8a1263f79ed0c77f4989ec2c",
        "base/optimizee_eval.yaml": "6fc0869127b7a50971461314239db040de8c4ce0",
        "regularizers/no_reg.yaml": "6ac57123ebaa82aa128e9784e4d2878f2883e17f",
        "regularizers/sam.yaml": "264b419dcad78240bb5de1f1aacaba3edc4bb827",
        "regularizers/sam_opzee.yaml": "de4b35ad8fa42629d97a7c1d6efa0ed92af1f03f",
        "regularizers/gam.yaml": "d6546bde7e0418053e7034c4a2f7bddab689df00",
        "regularizers/gsam.yaml": "80849183079207f2ee3a29a29a6e68d733cf3020",
        "regularizers/smooth.yaml": "ed22e28dad0795b0c5deefeafddd81cb329c7578",
    }
    for rel, sha in expected.items():
        digest = hashlib.sha1((ROOT / "configs" / "dmo" / rel).read_bytes()).hexdigest()
        assert digest == sha, f"{rel} was modified"


def test_task_selection_follows_meta_training():
    """The meta-training task defines "seen" everywhere; defaults reproduce the original setup."""
    cfg = load_pipeline_config(sets=SYN)
    assert cfg.optimizer_evaluation.dataset == "svhn" and cfg.optimizer_evaluation.seen_optimizee == "convnet4"
    assert cfg.optimizee_evaluation.optimizees == ["convnet4"] and cfg.optimizee_evaluation.seen_dataset == "svhn"
    assert cfg.optimizee_evaluation.unseen_datasets == ["mnist", "fmnist", "cifar10", "cifar100", "imagenet"]

    cfg = load_pipeline_config(sets=SYN + ["meta_training.optimizee=resnet18", "meta_training.dataset=imagenet",
                                           "meta_training.models={resnet18: {width: 16}}"])
    oe, ev = cfg.optimizer_evaluation, cfg.optimizee_evaluation
    assert (oe.dataset, oe.seen_optimizee, ev.seen_dataset) == ("imagenet", "resnet18", "imagenet")
    assert ev.optimizees == ["resnet18"] and "imagenet" not in ev.unseen_datasets and "svhn" in ev.unseen_datasets
    a = meta_training_args(cfg, METHODS[3], None)
    assert (a.dataset, a.optimizee_model_name, a.models) == ("imagenet", "resnet18", {"width": 16})

    cfg = load_pipeline_config(sets=SYN + ["meta_training.optimizee=mlp",
                                           "optimizer_evaluation.optimizees=[resnet18]"])
    assert cfg.optimizer_evaluation.optimizees == ["mlp", "resnet18"]       # seen optimizee added

    try:   # an explicit value that contradicts the meta-training task is rejected
        load_pipeline_config(sets=SYN + ["meta_training.dataset=cifar10", "optimizee_evaluation.seen_dataset=svhn"])
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_phase1_retrains_when_meta_task_changes():
    from learning_regularizers.pipeline.meta_train import meta_fingerprint
    key = RunKey(3, None, None)
    base = meta_fingerprint(load_pipeline_config(sets=SYN), key)
    assert base == meta_fingerprint(load_pipeline_config(sets=SYN), key)
    assert base != meta_fingerprint(load_pipeline_config(sets=SYN + ["meta_training.optimizee=resnet18"]), key)
    assert base != meta_fingerprint(load_pipeline_config(sets=SYN + ["meta_training.models={convnet4: {width: 16}}"]), key)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
