"""Regularizers: the explicit M2 step and the M4 meta-penalty, checked against exact autograd."""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from reglo.config import available, compose_run, load_group_file
from reglo.models import FunctionalModel
from reglo.models.optimizees import ConvNet4
from reglo.models.optimizers import LSTMOptimizer
from reglo.registry import REGULARIZERS
from reglo.regularizers import (
    GAM,
    GSAM,
    L1,
    L2,
    SAM,
    SmoothnessPenalty,
    build_regularizer,
    global_norm,
    gsam_direction,
)

@pytest.fixture(autouse=True, scope="module")
def _float64():
    """Tight gradient checks: float64 for this file's tests only."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def mlp(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))


def data(seed=1, n=16):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(n, 6, generator=gen), torch.randint(0, 3, (n,), generator=gen)


def cfg(name, **over):
    return OmegaConf.merge(load_group_file("regularizer", name), over)


def grads(model):
    return [p.grad.clone() for p in model.parameters()]


def exact_grads(model, objective):
    params = list(model.parameters())
    return torch.autograd.grad(objective(params), params)


def loss_of(model, x, y):
    fmodel = FunctionalModel(model)
    return lambda params: fmodel.loss(params, x, y)


def close(a, b, **kw):
    return all(torch.allclose(u, v, **kw) for u, v in zip(a, b))


# --------------------------------------------------------------------------- M2: explicit steps


def test_none_is_plain_backpropagation():
    model, (x, y) = mlp(), data()
    build_regularizer(cfg("none"), model).explicit_step(model, x, y)
    assert close(grads(model), exact_grads(model, loss_of(model, x, y)))


def test_l1_penalty_gradient_skips_biases():
    model, (x, y) = mlp(), data()
    reg = L1(cfg("l1", explicit={"coef": 0.1}), model)
    reg.explicit_step(model, x, y)
    loss = loss_of(model, x, y)
    expected = exact_grads(model, lambda p: loss(p) + 0.1 * (p[0].abs().sum() + p[2].abs().sum()))
    assert close(grads(model), expected)


def test_l1_proximal_mode_makes_exact_zeros():
    model = mlp()
    with torch.no_grad():
        model[0].weight.copy_(torch.linspace(-0.3, 0.3, 48).view(8, 6))
        bias_before = model[0].bias.clone()
    reg = L1(cfg("l1", explicit={"coef": 0.1, "mode": "proximal"}), model)
    reg.after_step(model, torch.optim.SGD(model.parameters(), lr=1.0))
    w = model[0].weight.detach()
    assert (w == 0).sum() > 0 and w.abs().max().item() == pytest.approx(0.2)
    assert torch.equal(model[0].bias, bias_before)


def test_l2_penalty_gradient_and_decoupled_mode():
    model, (x, y) = mlp(), data()
    L2(cfg("l2", explicit={"coef": 0.3}), model).explicit_step(model, x, y)
    loss = loss_of(model, x, y)
    expected = exact_grads(model, lambda p: loss(p) + 0.15 * (p[0].pow(2).sum() + p[2].pow(2).sum()))
    assert close(grads(model), expected)
    model.zero_grad()
    L2(cfg("l2", explicit={"mode": "decoupled"}), model).explicit_step(model, x, y)
    assert close(grads(model), exact_grads(model, loss))


def test_sam_step_uses_the_gradient_at_the_perturbed_point():
    model, (x, y) = mlp(), data()
    before = [p.detach().clone() for p in model.parameters()]
    loss = loss_of(model, x, y)
    g = exact_grads(model, loss)
    eps = [0.05 * gi / global_norm(g) for gi in g]
    shifted = [b + e for b, e in zip(before, eps)]
    expected = torch.autograd.grad(loss([s.requires_grad_() for s in shifted]), shifted)

    result = SAM(cfg("sam"), model).explicit_step(model, x, y)
    assert close(grads(model), expected)
    assert close(list(model.parameters()), before, atol=0, rtol=0)  # weights restored exactly
    assert result.penalty > 0  # loss rises inside the ρ-ball


def test_flatness_steps_update_batchnorm_statistics_once():
    torch.set_default_dtype(torch.float32)
    try:
        for reg_cls, name in ((SAM, "sam"), (GSAM, "gsam"), (GAM, "gam")):
            torch.manual_seed(0)
            model, reference = ConvNet4(channels=(4, 4, 4)), ConvNet4(channels=(4, 4, 4))
            reference.load_state_dict(model.state_dict())
            x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))
            reg_cls(cfg(name), model).explicit_step(model, x, y)
            reference(x)  # one ordinary training-mode forward
            assert torch.allclose(model.features[1].running_mean, reference.features[1].running_mean)
    finally:
        torch.set_default_dtype(torch.float64)


def test_gsam_direction():
    g, gp = [torch.tensor([1.0, 2.0]), torch.tensor([0.5])], [torch.tensor([2.0, 0.0]), torch.tensor([1.0])]
    assert close(gsam_direction(g, gp, 0.0), gp)
    d = gsam_direction(g, gp, 0.3)
    orth = [(gpi - di) / 0.3 for gpi, di in zip(gp, d)]  # α·∇⊥L recovered from d
    assert sum((o * q).sum() for o, q in zip(orth, gp)).abs() < 1e-12  # ⊥ to ∇L_p


def test_gsam_step_with_alpha_zero_equals_sam():
    model_a, model_b, (x, y) = mlp(), mlp(), data()
    SAM(cfg("sam"), model_a).explicit_step(model_a, x, y)
    GSAM(cfg("gsam", alpha=0.0), model_b).explicit_step(model_b, x, y)
    assert close(grads(model_a), grads(model_b))


def test_gam_step_matches_the_exact_gradient_of_the_penalised_loss():
    model, (x, y) = mlp(), data()
    loss = loss_of(model, x, y)

    def penalised(p):
        g = torch.autograd.grad(loss(p), p, create_graph=True)
        return loss(p) + 0.05 * torch.sqrt(sum(gi.pow(2).sum() for gi in g))

    expected = exact_grads(model, penalised)
    reg = GAM(cfg("gam", explicit={"fd_radius": 1e-5}), model)
    assert reg.passes == 2
    reg.explicit_step(model, x, y)
    assert close(grads(model), expected, rtol=1e-4, atol=1e-7)


def test_gam_max_form_takes_four_passes():
    model, (x, y) = mlp(), data()
    reg = GAM(cfg("gam", form="max"), model)
    assert reg.passes == 4
    before = [p.detach().clone() for p in model.parameters()]
    result = reg.explicit_step(model, x, y)
    assert result.penalty > 0 and all(torch.isfinite(g).all() for g in grads(model))
    assert close(list(model.parameters()), before, atol=1e-12)


# --------------------------------------------------------------------------- M4: meta-penalties


def through_phi(model):
    """θ_T = θ₀ + φ_i·v_i: a stand-in for 'θ_T depends on the LSTM weights φ'."""
    torch.manual_seed(3)
    base = [p.detach().clone() for p in model.parameters()]
    dirs = [torch.randn_like(p) for p in base]
    phi = torch.zeros(len(base), requires_grad=True)
    return phi, lambda f: [b + f[i] * d for i, (b, d) in enumerate(zip(base, dirs))]


@pytest.mark.parametrize("name, over", [("l1", {}), ("l2", {}), ("l1", {"meta": {"normalize": "sum"}}), ("gam", {})])
def test_meta_penalty_gradient_reaches_phi_exactly(name, over):
    model, (x, y) = mlp(), data()
    reg = build_regularizer(cfg(name, **over), model)
    fmodel = FunctionalModel(model)
    phi, theta = through_phi(model)
    (analytic,) = torch.autograd.grad(reg.meta_penalty(fmodel.loss, theta(phi), x, y).term, phi)
    h = 1e-6
    for i in range(len(phi)):
        e = torch.zeros_like(phi)
        e[i] = h
        up = reg.meta_penalty(fmodel.loss, theta(phi + e), x, y).term.item()
        down = reg.meta_penalty(fmodel.loss, theta(phi - e), x, y).term.item()
        assert analytic[i].item() == pytest.approx((up - down) / (2 * h), rel=1e-4, abs=1e-8)


def test_meta_sam_backpropagates_the_gradient_at_the_perturbed_point():
    model, (x, y) = mlp(), data()
    fmodel = FunctionalModel(model)
    theta = [p.detach().clone().requires_grad_() for p in model.parameters()]
    out = SAM(cfg("sam"), model).meta_penalty(fmodel.loss, theta, x, y)
    got = torch.autograd.grad(out.term, theta)
    g = torch.autograd.grad(fmodel.loss(theta, x, y), theta)
    shifted = [(t + 0.05 * gi / global_norm(g)).detach().requires_grad_() for t, gi in zip(theta, g)]
    assert close(got, torch.autograd.grad(fmodel.loss(shifted, x, y), shifted))
    assert out.value > 0


def test_meta_gsam_backpropagates_the_gsam_direction():
    model, (x, y) = mlp(), data()
    fmodel = FunctionalModel(model)
    theta = [p.detach().clone().requires_grad_() for p in model.parameters()]
    out = GSAM(cfg("gsam", alpha=0.2), model).meta_penalty(fmodel.loss, theta, x, y)
    got = torch.autograd.grad(out.term, theta)
    g = torch.autograd.grad(fmodel.loss(theta, x, y), theta)
    shifted = [(t + 0.05 * gi / global_norm(g)).detach().requires_grad_() for t, gi in zip(theta, g)]
    gp = torch.autograd.grad(fmodel.loss(shifted, x, y), shifted)
    assert close(got, gsam_direction(g, gp, 0.2))


def test_meta_gam_max_form_is_differentiable():
    model, (x, y) = mlp(), data()
    phi, theta = through_phi(model)
    out = GAM(cfg("gam", form="max"), model).meta_penalty(FunctionalModel(model).loss, theta(phi), x, y)
    (g,) = torch.autograd.grad(out.term, phi)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_phi_space_pieces():
    g, gp = [torch.tensor([3.0, 4.0])], [torch.tensor([1.0, 1.0])]
    sam = SAM(cfg("sam"))
    assert global_norm(sam.phi_perturbation(g)).item() == pytest.approx(0.05)
    assert close(sam.phi_gradient(g, gp), gp)
    assert close(GSAM(cfg("gsam")).phi_gradient(g, gp), gsam_direction(g, gp, 0.1))
    gam = GAM(cfg("gam"))
    assert global_norm(gam.phi_perturbation(g)).item() == pytest.approx(0.01)
    assert close(gam.phi_gradient(g, gp), [0.05 * (gp[0] - g[0]) / 0.01])
    with pytest.raises(NotImplementedError, match="phi"):
        L1(cfg("l1")).phi_perturbation(g)
    with pytest.raises(NotImplementedError, match="phi"):
        sam.phi_penalty([])


def test_l1_l2_penalise_the_lstm_weights_without_biases():
    named = [("cells.0.ih.weight", torch.tensor([[1.0, -2.0]], requires_grad=True)),
             ("cells.0.ih.bias", torch.tensor([10.0], requires_grad=True))]
    l1 = L1(cfg("l1")).phi_penalty(named)
    assert l1.value == pytest.approx(1e-5 * 3.0)  # M2's λ · ‖φ‖₁, the bias left out
    l1.term.backward()
    assert close([named[0][1].grad], [torch.tensor([[1e-5, -1e-5]])]) and named[1][1].grad is None
    l2 = L2(cfg("l2")).phi_penalty(named)
    assert l2.value == pytest.approx(5e-4 * 0.5 * 5.0)  # ½λ‖φ‖₂²
    assert not L1(cfg("l1")).replays and SAM(cfg("sam")).replays and GAM(cfg("gam")).replays


# --------------------------------------------------------------------------- L_smooth


def smooth(**over):
    return SmoothnessPenalty(OmegaConf.merge(load_group_file("smooth", "smooth"), over))


def lstm_and_state(n=200, seed=0):
    torch.manual_seed(seed)
    lstm = LSTMOptimizer(num_inputs=1)
    h, c = lstm.init_state(n, dtype=torch.float64)
    _, h, c = lstm(torch.randn(n, 1), h, c)  # a non-trivial hidden state
    return lstm, torch.randn(n, 1), h.detach(), c.detach()


def test_smooth_is_zero_inside_a_zero_radius_ball():
    lstm, s, h, c = lstm_and_state()
    assert smooth(epsilon=0.0)(lstm, s, h, c).item() == pytest.approx(0.0, abs=1e-20)


def test_smooth_ascent_finds_a_larger_change_than_noise():
    lstm, s, h, c = lstm_and_state()
    noise_only = smooth(pga_steps=0)(lstm, s, h, c, torch.Generator().manual_seed(0)).item()
    ascended = smooth(pga_steps=10)(lstm, s, h, c, torch.Generator().manual_seed(0)).item()
    assert ascended > 10 * noise_only > 0
    assert smooth(epsilon=0.005)(lstm, s, h, c).item() < ascended  # smaller ball, smaller worst case
    assert smooth(reduction="mean")(lstm, s, h, c).item() == pytest.approx(ascended / len(s))


def test_smooth_gradient_reaches_only_phi():
    lstm, s, h, c = lstm_and_state()
    s.requires_grad_(True)
    value = smooth()(lstm, s, h, c)
    value.backward()
    assert s.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lstm.parameters())


# --------------------------------------------------------------------------- wiring


def test_every_regularizer_file_is_registered_and_buildable():
    import reglo.regularizers  # noqa: F401

    for name in available("regularizer"):
        assert name in REGULARIZERS
    expected_passes = {"l1": 1, "l2": 1, "sam": 2, "gam": 2, "gsam": 2}
    for name, passes in expected_passes.items():
        run = compose_run(["method=m2_adamw_reg", f"regularizer={name}", "optimizee=convnet4", "dataset=svhn"])
        reg = build_regularizer(run.regularizer, ConvNet4(channels=(4, 4, 4)))
        assert reg.name == name and reg.passes == passes and reg.meta_coef == 0.1
