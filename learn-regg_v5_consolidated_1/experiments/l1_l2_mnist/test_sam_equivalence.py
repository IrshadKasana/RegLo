"""test_sam_equivalence.py — my flat LO-SAM window must give the same meta-gradient as the
repo's utils.consolidated_reg_train_opt_func(reg='sam') for one TBPTT window (smoothing off:
its equivalence is covered by test_equivalence.py). Repo run uses SGD(lr=1, no momentum), so
(old_param - new_param) == meta-gradient."""
import copy
from argparse import Namespace
import torch
import common as C
import sam_extension as S
from common import MetaOptimizerLSTM, MetaModel, FastLO, initialize_model, device, reset_state
from utils import consolidated_reg_train_opt_func           # repo implementation

torch.manual_seed(0)
ref = MetaOptimizerLSTM(MetaModel(initialize_model("mnist", "mlp").to(device)), 2, 20,
                        preproc_factor=10.0, preproc=True).to(device)
fast = FastLO(MetaModel(initialize_model("mnist", "mlp")), 2, 20, preproc_factor=10.0,
              preproc=True).to(device)
fast.load_state_dict(ref.state_dict()); fast.set_double_backward(False)

data = C.load_mnist()
st = C.Stream(data.Xa, data.ya, 128, 3)
batches = [st.next() for _ in range(20)]
smooth_ref = C.smooth_regularize(1.0, 0.01, 10, ref)
lam_r, rho = 0.7, 0.05

args = Namespace(base_optimizer="sgd", lr=1.0, momentum=0.0, weight_decay=0.0, smooth_lamb=1.0,
                 if_smooth=False, lambda_reg=lam_r, xi=1e-12, rho=rho, regularizer="sam",
                 train_period=1, truncated_bptt_step=20, dataset="mnist",
                 optimizee_model_name="mlp")
before = [p.detach().clone() for p in ref.parameters()]
torch.manual_seed(11)
theta0 = torch.cat([p.detach().reshape(-1) for p in initialize_model("mnist", "mlp").parameters()]).to(device)
torch.manual_seed(11)                                          # repo draws the same first model
consolidated_reg_train_opt_func(20, [b for b in batches], device, ref, smooth_ref, args)
g_ref = [b - p.detach() for b, p in zip(before, ref.parameters())]

reset_state(fast, keep=False)
opt = torch.optim.SGD(fast.parameters(), lr=0.0)
S.sam_window(fast, theta0, batches, lam_r, rho, 1e-12, C.smooth_regularize(1.0, 0.01, 10, fast), 0.0, opt)
g_fast = [p.grad for p in fast.parameters()]

for (n, _), a, b in zip(fast.named_parameters(), g_ref, g_fast):
    rel = ((a - b).abs().max() / (a.abs().max() + 1e-12)).item()
    print(f"{n:28s} |g|max={a.abs().max().item():.3e}  rel diff={rel:.2e}")
    assert rel < 1e-4, (n, rel)
assert max(a.abs().max().item() for a in g_ref) > 0
print("SAM EQUIVALENCE OK")
