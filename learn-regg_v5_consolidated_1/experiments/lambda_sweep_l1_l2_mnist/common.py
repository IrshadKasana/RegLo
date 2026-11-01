"""
common.py — building blocks of the stand-alone lambda sweep (L1 / L2, MNIST).

Four methods, MLP optimizee 784-20-10 from src/learning_regularizers/dmo:

    1. MLP + AdamW
    2. MLP + AdamW + lambda * R(theta)                      R in {L1, L2}
    3. MLP + learned optimizer (LSTM LO)
    4. MLP + learned optimizer meta-trained with lambda * R(theta) (+ the LO
       smoothing term, exactly as in the proposed model; SAM/GAM/GSAM -> R)

This experiment only answers ONE question: which lambda is best?  Hyper-
parameters are therefore scored on a validation split and never on test.

DATA PROTOCOL (same split as experiments/l1_l2_mnist and dataloaders.py)
    A        30000  meta-training stream for the LO
    B_train  27000  optimizee training pool
    B_val     3000  lambda / lr / wd selection -- never trained on
    test     10000  official test set -- stored, never used for selection

Nothing under src/ is modified.
"""
from __future__ import annotations

import contextlib as _ctx
import io as _io
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
DMO = REPO / "src" / "learning_regularizers" / "dmo"
sys.path.insert(0, str(DMO))
sys.path.insert(0, str(REPO))

with _ctx.redirect_stdout(_io.StringIO()):          # meta_optimizer prints a device banner
    from meta_optimizer import MetaOptimizerLSTM, MetaModel, device   # noqa: E402
from model import initialize_model                                    # noqa: E402
from regularizers.optimizers import smooth_regularize                 # noqa: E402
from project_paths import datasets_dir                                # noqa: E402


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    # --- lambda grids: half-decade log grids, chosen so the optimum is interior.
    # sum|theta| ~ 1.5e3 and sum theta^2 ~ 3e2..1.5e3 for this MLP, so
    # lambda*R is ~1e-3..1e-1 of the task loss at the low end and >>1 at the top.
    lams_l1: tuple = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
    lams_l2: tuple = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)
    # --- AdamW arms (methods 1 & 2): lr and weight decay interact with lambda
    adamw_lrs: tuple = (1e-3, 3e-3, 1e-2, 3e-2)
    adamw_wds: tuple = (0.0, 0.01, 0.1)          # 0.01 = repo baseline
    adamw_inits: int = 6                         # optimizee inits per config
    # --- optimizee evaluation (matches optimizee_eval.yaml / l1_l2_mnist)
    eval_steps: int = 1000
    eval_batch: int = 512
    lo_inits: int = 3                            # optimizee inits per LO
    # --- learned optimizer (repo defaults)
    hidden_size: int = 20
    num_layers: int = 2
    meta_lr: float = 1e-3
    meta_batch: int = 128
    tbptt: int = 20
    # reduced curriculum relative to l1_l2_mnist (100,200,500,1000 x 5 blocks x 20
    # inits): same stages, 2 blocks x 10 inits.  Same budget for every lambda.
    stages: tuple = (100, 200, 500, 1000)
    blocks_per_stage: int = 2
    train_period: int = 10
    ckpt_epochs: int = 2                         # trajectories for LO checkpoint selection
    # --- smoothing term (smooth.yaml defaults)
    smooth_lamb: float = 1.0
    smooth_eps: float = 0.01
    smooth_pga: int = 10

    def lams(self, reg):
        return self.lams_l1 if reg == "l1" else self.lams_l2


SMOKE = dict(lams_l1=(1e-5, 1e-4), lams_l2=(1e-4, 1e-3), adamw_lrs=(1e-2,),
             adamw_wds=(0.01,), adamw_inits=2, eval_steps=60, lo_inits=1,
             stages=(20, 40), blocks_per_stage=1, train_period=2, ckpt_epochs=1)


# ----------------------------------------------------------------------------
# data (whole MNIST on the GPU)
# ----------------------------------------------------------------------------
MEAN, STD = 0.1307, 0.3081


@dataclass
class Data:
    Xa: torch.Tensor; ya: torch.Tensor
    Xb: torch.Tensor; yb: torch.Tensor
    Xv: torch.Tensor; yv: torch.Tensor
    Xt: torch.Tensor; yt: torch.Tensor


def load_mnist(split_seed: int = 42, val_fraction: float = 0.1) -> Data:
    from torchvision import datasets
    root = str(datasets_dir(__file__))
    tr = datasets.MNIST(root, train=True, download=False)
    te = datasets.MNIST(root, train=False, download=False)

    def prep(ds):
        x = ds.data.float().div(255.0).sub(MEAN).div(STD).unsqueeze(1)
        return x.to(device), ds.targets.long().to(device)

    X, y = prep(tr)
    Xt, yt = prep(te)
    idx = list(range(len(X)))
    np.random.RandomState(split_seed).shuffle(idx)
    half = len(idx) // 2
    A, pool = idx[:half], idx[half:]
    n_val = int(len(pool) * val_fraction)
    val, trn = pool[:n_val], pool[n_val:]
    t = lambda ix: torch.as_tensor(ix, device=device)
    return Data(X[t(A)], y[t(A)], X[t(trn)], y[t(trn)], X[t(val)], y[t(val)], Xt, yt)


class Stream:
    """Endless shuffled mini-batch stream over GPU tensors."""

    def __init__(self, X, y, bs, seed):
        self.X, self.y, self.bs, self.n = X, y, bs, X.shape[0]
        self.g = torch.Generator().manual_seed(int(seed))
        self._reshuffle()

    def _reshuffle(self):
        self.perm = torch.randperm(self.n, generator=self.g).to(self.X.device)
        self.pos = 0

    def next(self):
        if self.pos + self.bs > self.n:
            self._reshuffle()
        ix = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return self.X[ix], self.y[ix]


# ----------------------------------------------------------------------------
# regularizers: ONE definition of R shared by methods 2 and 4
# ----------------------------------------------------------------------------
def penalty(flat: torch.Tensor, kind: str) -> torch.Tensor:
    """R(theta) over all optimizee parameters.  l1: sum|theta|   l2: sum theta^2."""
    if kind == "l1":
        return flat.abs().sum()
    if kind == "l2":
        return (flat ** 2).sum()
    raise ValueError(kind)


def flat_params(model):
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def flat_grads(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


@torch.no_grad()
def evaluate(model, X, y, chunk=10000):
    loss, correct = 0.0, 0
    for i in range(0, X.shape[0], chunk):
        out = model(X[i:i + chunk])
        loss += F.nll_loss(out, y[i:i + chunk], reduction="sum").item()
        correct += (out.argmax(1) == y[i:i + chunk]).sum().item()
    return loss / X.shape[0], correct / X.shape[0]


def report(model, data: Data) -> dict:
    tr_l, tr_a = evaluate(model, data.Xb, data.yb)
    va_l, va_a = evaluate(model, data.Xv, data.yv)
    te_l, te_a = evaluate(model, data.Xt, data.yt)
    f = flat_params(model).detach()
    return dict(train_loss=tr_l, train_acc=tr_a, val_loss=va_l, val_acc=va_a,
                test_loss=te_l, test_acc=te_a,
                l1_norm=f.abs().sum().item(), l2_sq=(f ** 2).sum().item(),
                frac_small=(f.abs() < 1e-3).float().mean().item())


# ----------------------------------------------------------------------------
# methods 1 & 2
# ----------------------------------------------------------------------------
def run_adamw(seed, cfg: Cfg, data: Data, lr, wd, reg="none", lam=0.0) -> dict:
    torch.manual_seed(seed)
    model = initialize_model("mnist", "mlp").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    stream = Stream(data.Xb, data.yb, cfg.eval_batch, seed)
    for _ in range(cfg.eval_steps):
        x, y = stream.next()
        loss = F.nll_loss(model(x), y)
        if reg != "none":
            loss = loss + lam * penalty(flat_params(model), reg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return report(model, data)


# ----------------------------------------------------------------------------
# methods 3 & 4
# ----------------------------------------------------------------------------
def make_lo(cfg: Cfg):
    meta_model = initialize_model("mnist", "mlp").to(device)
    lo = MetaOptimizerLSTM(MetaModel(meta_model), cfg.num_layers, cfg.hidden_size,
                           preproc_factor=10.0, preproc=True).to(device)
    lo.set_double_backward(False)
    return lo


def run_lo(lo, seed, cfg: Cfg, data: Data, with_report=True):
    """Meta-test: train a fresh MLP with the frozen LO (it only sees the task gradient)."""
    lo.eval()
    torch.manual_seed(seed)
    model = initialize_model("mnist", "mlp").to(device)
    lo.reset_lstm(keep_states=False, model=model)
    stream = Stream(data.Xb, data.yb, cfg.eval_batch, seed)
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = stream.next()
        loss = F.nll_loss(model(x), y)
        model.zero_grad()
        loss.backward()
        losses.append(loss.detach())
        with torch.no_grad():
            lo.meta_update(model)
    lo.train()
    return report(model, data) if with_report else torch.stack(losses)


def ckpt_score(lo, cfg: Cfg, data: Data) -> float:
    """Checkpoint selection: mean task loss over the last 10% of fresh trajectories
    (fixed seeds 9000+ -> comparable across blocks; disjoint from tuning seeds)."""
    tail = max(cfg.eval_steps // 10, 1)
    out = []
    with torch.random.fork_rng():
        for e in range(cfg.ckpt_epochs):
            out.append(run_lo(lo, 9000 + e, cfg, data, with_report=False)[-tail:].mean().item())
    return float(np.mean(out))


def meta_train_lo(kind, lam, seed, cfg: Cfg, data: Data, ckpt_path, log=print):
    """kind='none' -> L_task (method 3); 'smooth' -> L_task + ls*L_smooth (lambda=0
    reference for method 4); 'l1'|'l2' -> L_task + ls*L_smooth + lam*R(theta_T) (method 4).
    TBPTT loop and curriculum follow train_opt_func / consolidated_reg_train_opt_func."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    reg = kind if kind in ("l1", "l2") else "none"
    use_smooth = kind != "none"
    lo = make_lo(cfg)
    opt = torch.optim.Adam(lo.parameters(), lr=cfg.meta_lr)
    smooth = smooth_regularize(cfg.smooth_lamb, cfg.smooth_eps, cfg.smooth_pga, lo)
    stream = Stream(data.Xa, data.ya, cfg.meta_batch, seed)

    best, history, t0 = float("inf"), [], time.time()
    torch.save(lo.state_dict(), ckpt_path)
    for T in cfg.stages:
        tb = min(cfg.tbptt, T)
        for b in range(cfg.blocks_per_stage):
            task_acc = pen_acc = n = 0.0
            for _ in range(cfg.train_period):
                model = initialize_model("mnist", "mlp").to(device)
                for k in range(T // tb):
                    lo.reset_lstm(keep_states=k > 0, model=model)
                    for _j in range(tb):
                        x, y = stream.next()
                        loss = F.nll_loss(model(x), y)
                        model.zero_grad()
                        loss.backward()
                        model_temp = lo.meta_update(model)
                    L_task = F.nll_loss(model_temp(x), y)
                    L = L_task
                    if reg != "none":
                        R = penalty(lo.meta_model.get_flat_params(), reg)
                        L = L + lam * R
                        pen_acc += R.item()
                    if use_smooth:
                        L = L + cfg.smooth_lamb * smooth.run(flat_grads(model))
                    opt.zero_grad()
                    L.backward()
                    opt.step()
                    task_acc += L_task.item()
                    n += 1
            score = ckpt_score(lo, cfg, data)
            improved = score < best
            if improved:
                best = score
                torch.save(lo.state_dict(), ckpt_path)
            history.append(dict(stage=T, block=b, train_task_loss=task_acc / n,
                                mean_R=pen_acc / n, ckpt_score=score, saved=improved,
                                elapsed=time.time() - t0))
            log(f"[{kind} lam={lam:g} seed={seed}] T={T:4d} blk {b + 1}/{cfg.blocks_per_stage} "
                f"task={task_acc / n:.4f} R={pen_acc / n:.1f} ckpt={score:.4f}"
                f"{' *' if improved else ''} ({time.time() - t0:.0f}s)")
    lo.load_state_dict(torch.load(ckpt_path, map_location=device))
    return lo, dict(best_ckpt_score=best, history=history, meta_train_seconds=time.time() - t0)
