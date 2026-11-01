"""Optimizees, AdamW, and the LSTM optimizer F_φ."""

import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from reglo.config import available, compose_run, config_dir
from reglo.models import FunctionalModel, build_optimizee, count_parameters, penalty_mask
from reglo.models.optimizees import MLP, ConvNet4, ViTLite
from reglo.models.optimizers import (
    FrozenLSTMOptimizer,
    GradientFeatures,
    LSTMOptimizer,
    build_adamw,
    build_lstm_optimizer,
    build_scheduler,
    load_learned_optimizer,
    save_learned_optimizer,
)
from reglo.registry import OPTIMIZEES, OPTIMIZERS


def optimizee_cfg(name, dataset="cifar10", *extra):
    """The architecture as the real study builds it (the smoke profiles narrow the networks)."""
    return compose_run(["method=m1_adamw", f"optimizee={name}", f"dataset={dataset}", "profile=production",
                        *extra]).optimizee


# --------------------------------------------------------------------------- optimizees


@pytest.mark.parametrize(
    "name, dataset, expected",
    [
        ("convnet4", "cifar10", 113_962),
        ("mlp", "fmnist", 118_282),       # round B's source: 784-128-128-10
        ("mlp", "svhn", 411_146),         # 3072-128-128-10
        ("resnet18", "cifar10", 11_173_962),
        ("wrn28", "cifar10", 5_849_050),
        ("vit_lite7", "cifar10", 3_722_250),
    ],
)
def test_parameter_counts(name, dataset, expected):
    assert count_parameters(build_optimizee(optimizee_cfg(name, dataset))) == expected


def test_wrn28_10_matches_the_published_size():
    model = build_optimizee(optimizee_cfg("wrn28", "cifar10", "optimizee.widen_factor=10"))
    assert count_parameters(model) == 36_479_194


@pytest.mark.parametrize("name", ["convnet4", "mlp", "resnet18", "wrn28", "vit_lite7"])
@pytest.mark.parametrize("dataset", ["cifar100", "fmnist"])
def test_every_optimizee_fits_every_dataset(name, dataset):
    cfg = compose_run(["method=m1_adamw", f"optimizee={name}", f"dataset={dataset}"])
    model = build_optimizee(cfg.optimizee)
    x = torch.randn(2, cfg.dataset.channels, cfg.dataset.image_size, cfg.dataset.image_size)
    assert model(x).shape == (2, cfg.dataset.num_classes)


def test_activations_follow_the_config():
    assert any(isinstance(m, nn.GELU) for m in build_optimizee(optimizee_cfg("vit_lite7")).modules())
    for name in ("convnet4", "mlp", "resnet18", "wrn28"):
        model = build_optimizee(optimizee_cfg(name))
        assert any(isinstance(m, nn.ReLU) for m in model.modules())
        assert not any(isinstance(m, nn.GELU) for m in model.modules())


@pytest.mark.parametrize("model", [ConvNet4(channels=(4, 8, 8)), ViTLite(embed_dim=16, depth=1, num_heads=2),
                                   MLP(in_channels=3, image_size=32, hidden=(16, 16))])
def test_double_backprop_works(model):
    """GAM's meta-penalty and the Hessian diagnostics differentiate through gradients."""
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    params = list(model.parameters())
    grads = torch.autograd.grad(nn.functional.cross_entropy(model(x), y), params, create_graph=True)
    norm = torch.sqrt(sum((g**2).sum() for g in grads))
    second = torch.autograd.grad(norm, params, allow_unused=True)
    assert any(s is not None and s.abs().sum() > 0 for s in second)


def test_functional_model_matches_the_module():
    torch.manual_seed(0)
    model = ConvNet4(channels=(4, 8, 8)).eval()
    fmodel = FunctionalModel(model)
    x = torch.randn(3, 3, 32, 32)
    assert torch.allclose(fmodel.logits(fmodel.initial_params(), x), model(x))
    shifted = [p + 0.1 for p in fmodel.initial_params()]
    fmodel.load(shifted)
    assert torch.allclose(model(x), fmodel.logits(shifted, x))


def test_penalty_mask_skips_biases_and_normalisation_layers():
    model = ConvNet4(channels=(4, 8, 8))
    mask = dict(zip([n for n, _ in model.named_parameters()], penalty_mask(model)))
    assert mask["features.0.weight"] and mask["classifier.weight"]
    assert not mask["features.1.weight"] and not mask["features.1.bias"] and not mask["classifier.bias"]
    vit = ViTLite(embed_dim=16, depth=1, num_heads=2)
    vmask = dict(zip([n for n, _ in vit.named_parameters()], penalty_mask(vit, ("bias", "norm", "embedding"))))
    assert not vmask["pos_embed"] and not vmask["cls_token"] and vmask["blocks.0.attn.qkv.weight"]
    with pytest.raises(ValueError):
        penalty_mask(model, ("weights",))


def test_every_config_name_is_registered():
    import reglo.models.optimizers  # noqa: F401  (registers adamw and lstm)

    for group, registry in (("optimizee", OPTIMIZEES), ("optimizer", OPTIMIZERS)):
        for name in available(group):
            assert OmegaConf.load(config_dir() / f"{group}s" / f"{name}.yaml").name in registry


# --------------------------------------------------------------------------- AdamW


def test_adamw_is_unregularised_unless_l2_is_decoupled():
    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn"]).optimizer
    model = ConvNet4(channels=(4, 8, 8))
    plain = build_adamw(cfg, model)
    assert [g["weight_decay"] for g in plain.param_groups] == [0.0]
    decayed = build_adamw(cfg, model, decoupled_l2=0.05)
    decays = sorted(g["weight_decay"] for g in decayed.param_groups)
    assert decays == [0.0, 0.05]
    assert sum(len(g["params"]) for g in decayed.param_groups) == len(list(model.parameters()))


def test_cosine_schedule_with_warmup():
    cfg = compose_run(["method=m1_adamw", "optimizee=convnet4", "dataset=svhn",
                       "optimizer.scheduler.warmup_epochs=1"]).optimizer
    opt = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=float(cfg.lr))
    sched = build_scheduler(opt, cfg, total_steps=110, steps_per_epoch=10)
    lrs = []
    for _ in range(110):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-4) and lrs[9] == pytest.approx(1e-3)  # warm-up over 10 steps
    assert lrs[10] == pytest.approx(1e-3) and lrs[60] == pytest.approx(5e-4, rel=0.02) and lrs[-1] < 1e-5


# --------------------------------------------------------------------------- LSTM optimizer F_φ


def lstm_from_config(*extra):
    return build_lstm_optimizer(compose_run(["method=m3_lstm", "optimizee=convnet4", "dataset=svhn", *extra]).optimizer)


def test_lstm_is_the_configured_architecture():
    lstm, features = lstm_from_config()
    assert len(lstm.cells) == 2 and lstm.hidden_size == 20 and features.num_inputs == 1
    linears = [m for m in lstm.head if isinstance(m, nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(20, 20), (20, 1)]


def test_lstm_is_coordinate_wise():
    torch.manual_seed(0)
    lstm = LSTMOptimizer(num_inputs=2)
    x = torch.randn(50, 2)
    h, c = lstm.init_state(50)
    raw, h1, c1 = lstm(x, h, c)
    perm = torch.randperm(50)
    raw_p, _, _ = lstm(x[perm], h[:, perm], c[:, perm])
    assert torch.allclose(raw_p, raw[perm], atol=1e-6)
    assert h1.shape == (2, 50, 20) and c1.shape == (2, 50, 20)


def test_chunked_inference_equals_one_pass():
    torch.manual_seed(0)
    lstm = LSTMOptimizer(num_inputs=1)
    x = torch.randn(1000, 1)
    h, c = lstm.init_state(1000)
    raw, h_ref, c_ref = lstm(x, h, c)
    h2, c2 = lstm.init_state(1000)
    update = lstm.step_inplace(x, h2, c2, chunk_size=128)
    assert torch.allclose(update, raw * lstm.output_scale, atol=1e-7)
    assert torch.allclose(h2, h_ref, atol=1e-6) and torch.allclose(c2, c_ref, atol=1e-6)


def test_initial_updates_are_small():
    torch.manual_seed(0)
    lstm, features = lstm_from_config()
    h, c = lstm.init_state(1000)
    raw, _, _ = lstm(torch.randn(1000, 1), h, c)
    assert (raw * lstm.output_scale).abs().max() < 1e-3


def test_features_normalisations():
    g = [torch.randn(10, 3) * 100, torch.randn(5) * 1e-4]
    rms = GradientFeatures(["grad"], "rms")(g, g, {})[0][:, 0]
    assert rms[:30].pow(2).mean().sqrt() == pytest.approx(1.0, rel=1e-4)
    assert rms[30:].pow(2).mean().sqrt() == pytest.approx(1.0, rel=1e-3)
    glob = GradientFeatures(["grad"], "global_rms")(g, g, {})[0]
    assert glob.pow(2).mean().sqrt() == pytest.approx(1.0, rel=1e-3)
    log_sign = GradientFeatures(["grad"], "log_sign")
    out = log_sign([torch.tensor([1.0, -1e-3, 1e-6, 0.0])], [], {})[0]
    assert out.shape == (4, 2) and log_sign.num_inputs == 2
    assert out[0].tolist() == pytest.approx([0.0, 1.0])
    assert out[1].tolist() == pytest.approx([-0.69078, -1.0], rel=1e-4)  # (log|x|/10, sign x)
    assert out[2].tolist() == pytest.approx([-1.0, 0.0220265], rel=1e-4)  # below e^-10: (−1, e^10·x)
    assert out[3].tolist() == [-1.0, 0.0]
    with pytest.raises(ValueError):
        GradientFeatures(["hessian"])


def test_momentum_feature_is_an_ema():
    feats = GradientFeatures(["grad", "momentum"], "none", momentum_beta=0.5)
    params = [torch.zeros(3)]
    state = feats.init_state(params)
    _, state = feats([torch.ones(3)], params, state)
    inputs, state = feats([torch.ones(3)], params, state)
    assert inputs.shape == (3, 2) and torch.allclose(inputs[:, 1], torch.full((3,), 0.75))


def test_frozen_optimizer_applies_exactly_the_lstm_update():
    torch.manual_seed(0)
    model = ConvNet4(channels=(4, 8, 8))
    lstm, features = LSTMOptimizer(num_inputs=1, output_scale=0.1), GradientFeatures(["grad"], "rms")
    before = [p.detach().clone() for p in model.parameters()]
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))
    nn.functional.cross_entropy(model(x), y).backward()
    grads = [p.grad.clone() for p in model.parameters()]

    opt = FrozenLSTMOptimizer(model.parameters(), lstm, features, chunk_size=1000)
    opt.step()
    inputs, _ = features(grads, before, {})
    h, c = lstm.init_state(inputs.shape[0])
    raw, _, _ = lstm(inputs, h, c)
    expected = torch.cat([b.reshape(-1) for b in before]) + 0.1 * raw.detach()
    after = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    assert torch.allclose(after, expected, atol=1e-6)
    assert not any(p.requires_grad for p in lstm.parameters())


def test_frozen_optimizer_state_round_trip():
    torch.manual_seed(0)
    lstm, features = LSTMOptimizer(num_inputs=2), GradientFeatures(["grad", "momentum"], "rms")
    x, y = torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))

    def run(model, opt, steps):
        for _ in range(steps):
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()

    torch.manual_seed(1)
    a = ConvNet4(channels=(4, 8, 8))
    opt_a = FrozenLSTMOptimizer(a.parameters(), lstm, features)
    run(a, opt_a, 2)
    # state_dict() returns live tensors, like torch.optim (torch.save it right away, or copy it)
    saved_model, saved_opt = copy.deepcopy(a.state_dict()), copy.deepcopy(opt_a.state_dict())
    run(a, opt_a, 2)

    b = ConvNet4(channels=(4, 8, 8))
    b.load_state_dict(saved_model)
    opt_b = FrozenLSTMOptimizer(b.parameters(), lstm, features)
    opt_b.load_state_dict(saved_opt)
    run(b, opt_b, 2)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.allclose(pa, pb, atol=1e-6)


def test_learned_optimizer_checkpoint_round_trip(tmp_path):
    lstm, features = lstm_from_config("optimizer.features=[grad,momentum]")
    path = save_learned_optimizer(tmp_path / "phi.pt", lstm, features, regularizer="sam", trajectory=7)
    loaded, loaded_features, meta = load_learned_optimizer(path)
    assert meta == {"regularizer": "sam", "trajectory": 7}
    assert loaded_features.features == ("grad", "momentum") and loaded.num_inputs == 2
    for a, b in zip(lstm.state_dict().values(), loaded.state_dict().values()):
        assert torch.equal(a, b)
    with pytest.raises(FileNotFoundError, match="Stage 1"):
        load_learned_optimizer(tmp_path / "missing.pt")
