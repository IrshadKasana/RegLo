"""
test_equivalence.py — the fast flat-parameter LO path in common.py must equal the
repo's MetaOptimizerLSTM.meta_update path (same maths, fewer host syncs).

Checks, for one TBPTT window with identical LO weights / init / batches:
  (a) optimizee parameters after 20 LO updates
  (b) meta-gradient w.r.t. every LO parameter for  L_task  and  L_task + lam*R
  (c) the smoothing term L_smooth (same RNG state)
Run:  python test_equivalence.py
"""
import torch
import torch.nn.functional as F
import common as C
from common import (MetaOptimizerLSTM, MetaModel, FastLO, initialize_model, device,
                    smooth_regularize, penalty, task_grad, mlp_logp, reset_state)

torch.manual_seed(0)
ref = MetaOptimizerLSTM(MetaModel(initialize_model("mnist", "mlp").to(device)), 2, 20,
                        preproc_factor=10.0, preproc=True).to(device)
ref.set_double_backward(False)
fast = FastLO(MetaModel(initialize_model("mnist", "mlp")), 2, 20, preproc_factor=10.0,
              preproc=True).to(device)
fast.load_state_dict(ref.state_dict())
fast.set_double_backward(False)

data = C.load_mnist()
st = C.Stream(data.Xa, data.ya, 128, 3)
batches = [st.next() for _ in range(20)]
model = initialize_model("mnist", "mlp").to(device)
theta0 = torch.cat([p.detach().reshape(-1) for p in model.parameters()])

worst = 0.0
for kind, lam in (("none", 0.0), ("l1", 1e-3), ("l2", 1e-2)):
    # ---- reference: repo code path ----
    m = initialize_model("mnist", "mlp").to(device)
    m.load_state_dict(model.state_dict())
    ref.reset_lstm(keep_states=False, model=m)
    for x, y in batches:
        loss = F.nll_loss(m(x), y)
        m.zero_grad()
        loss.backward()
        mt = ref.meta_update(m)
    x, y = batches[-1]
    L = F.nll_loss(mt(x), y)
    th_ref = ref.meta_model.get_flat_params()
    if kind != "none":
        L = L + lam * penalty(th_ref, kind)
    g_ref = torch.autograd.grad(L, list(ref.parameters()))
    th_ref = th_ref.detach().clone()
    flat_grads_ref = torch.cat([p.grad.reshape(-1) for p in m.parameters()])

    # ---- fast path ----
    reset_state(fast, keep=False)
    theta = theta0.clone()
    for x, y in batches:
        _, g = task_grad(theta, x, y)
        theta = theta + fast(g.view(-1, 1, 1))
    L2 = F.nll_loss(mlp_logp(theta, x), y)
    if kind != "none":
        L2 = L2 + lam * penalty(theta, kind)
    g_fast = torch.autograd.grad(L2, list(fast.parameters()))

    d_theta = (theta.detach() - th_ref).abs().max().item()
    d_loss = abs(L.item() - L2.item())
    d_grad = max(((a - b).abs().max() / (a.abs().max() + 1e-12)).item()
                 for a, b in zip(g_ref, g_fast))
    d_in = (g - flat_grads_ref).abs().max().item()
    print(f"{kind:5s} |dtheta|={d_theta:.2e} |dL|={d_loss:.2e} "
          f"max rel dgrad(LO params)={d_grad:.2e} |dg_in|={d_in:.2e}")
    worst = max(worst, d_theta, d_loss, d_grad * 1e-1, d_in)

# (c) smoothing term
for n_, lo in (("ref", ref), ("fast", fast)):
    sm = smooth_regularize(1.0, 0.01, 10, lo)
    torch.manual_seed(5)
    gin = flat_grads_ref.detach()
    lo.h0, lo.c0 = C.initialize_rnn_hidden_state(20, 2, C.N_PARAMS)
    val = sm.run(gin)
    print(f"L_smooth[{n_}] = {val.item():.6e}")
    if n_ == "ref":
        v_ref = val.item()
    else:
        rel = abs(v_ref - val.item()) / (abs(v_ref) + 1e-12)
        print(f"smoothing rel diff = {rel:.2e}")
        worst = max(worst, rel * 1e-1)
assert worst < 1e-3, worst
print("EQUIVALENCE OK")
