"""
common.py — shared pieces of the stand-alone L1 / L2 comparison experiment.

Four methods are compared on MNIST with the repo's MLP optimizee (784-20-10):

    1. MLP + AdamW
    2. MLP + AdamW + explicit regularizer R(theta)       (R = L1 or L2)
    3. MLP + learned optimizer (LSTM LO, no regularizer in the meta-loss)
    4. MLP + learned optimizer + R(theta) in the meta-loss (+ the LO smoothing
       term, exactly like the proposed model, with SAM/GAM/GSAM swapped for R)

Nothing under src/ is modified: the LSTM learned optimizer, the MLP and the
smoothing term are imported from src/learning_regularizers/dmo; everything
that is new (L1/L2 penalties, the meta-training loop, the data splits, the
evaluation protocol) lives in this folder.

DATA PROTOCOL (mirrors dataloaders.py: split_seed=42, 50/50 split, 10% val)
    A        30000  meta-training stream for the LO
    B_train  27000  optimizee training pool   (also LO checkpoint selection)
    B_val     3000  hyper-parameter selection (lambda, lr)  -- never trained on
    test     10000  official MNIST test set   -- reported only, never tuned on
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
DMO = REPO / "src" / "learning_regularizers" / "dmo"
sys.path.insert(0, str(DMO))
sys.path.insert(0, str(REPO))

import io as _io
import contextlib as _ctx

with _ctx.redirect_stdout(_io.StringIO()):       # meta_optimizer prints device banner
    from meta_optimizer import (MetaOptimizerLSTM, MetaModel, device,    # noqa: E402
                                initialize_rnn_hidden_state)
from model import initialize_model                                    # noqa: E402
from regularizers.optimizers import smooth_regularize                 # noqa: E402
from project_paths import datasets_dir                                # noqa: E402


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    # --- meta-training of the learned optimizer (repo defaults) ---
    hidden_size: int = 20
    num_layers: int = 2
    meta_lr: float = 1e-3            # "EQUALIZED" meta-lr in configs/dmo/regularizers/*.yaml
    meta_batch: int = 128            # configs/dmo/base/meta_train.yaml
    tbptt: int = 20                  # truncated_bptt_step
    stages: tuple = (100, 200, 500, 1000)   # curriculum_steps of production_train.yaml
    blocks_per_stage: int = 3        # meta-iterations (train_period optimizee inits each) per stage
    train_period: int = 20           # optimizee initialisations per block
    eval_epochs: int = 5             # trajectories per held-out evaluation
    # smoothing term (smooth.yaml defaults); used by methods 4 and the diagnostic row
    smooth_lamb: float = 1.0
    smooth_eps: float = 0.01
    smooth_pga: int = 10
    # --- optimizee evaluation (optimizee_eval.yaml / production_eval.yaml) ---
    eval_steps: int = 1000
    eval_batch: int = 512
    n_inits: int = 3                 # optimizee initialisations per (method, seed)
    adamw_wd: float = 0.01           # repo AdamW baseline weight decay
    # --- hyper-parameter grids (selected on B_val) ---
    adamw_lrs: tuple = (1e-3, 3e-3, 1e-2, 3e-2)
    lams_l1: tuple = (1e-5, 1e-4, 1e-3, 1e-2)
    lams_l2: tuple = (1e-4, 1e-3, 1e-2, 1e-1)

    def lams(self, reg):
        return self.lams_l1 if reg == "l1" else self.lams_l2


SMOKE = dict(stages=(20, 40), blocks_per_stage=1, train_period=2, eval_epochs=2,
             eval_steps=60, n_inits=1, adamw_lrs=(1e-2,),
             lams_l1=(1e-4,), lams_l2=(1e-3,))


# ----------------------------------------------------------------------------
# data: whole MNIST lives on the GPU as normalised tensors
# ----------------------------------------------------------------------------
MEAN, STD = 0.1307, 0.3081          # dataloaders._NORM['mnist']


@dataclass
class Data:
    Xa: torch.Tensor; ya: torch.Tensor          # meta-train stream (A)
    Xb: torch.Tensor; yb: torch.Tensor          # optimizee train pool (B_train)
    Xv: torch.Tensor; yv: torch.Tensor          # validation (B_val)
    Xt: torch.Tensor; yt: torch.Tensor          # test


def load_mnist(split_seed: int = 42, val_fraction: float = 0.1) -> Data:
    from torchvision import datasets
    root = str(datasets_dir(__file__))
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)

    def prep(ds):
        x = ds.data.float().div(255.0).sub(MEAN).div(STD).unsqueeze(1)
        return x.to(device), ds.targets.long().to(device)

    X, y = prep(tr)
    Xt, yt = prep(te)
    idx = list(range(len(X)))
    np.random.RandomState(split_seed).shuffle(idx)          # same as dataloaders.py
    half = len(idx) // 2
    A, pool = idx[:half], idx[half:]
    n_val = int(len(pool) * val_fraction)
    val, trn = pool[:n_val], pool[n_val:]
    t = lambda ix: torch.as_tensor(ix, device=device)
    return Data(X[t(A)], y[t(A)], X[t(trn)], y[t(trn)], X[t(val)], y[t(val)], Xt, yt)


class Stream:
    """Endless shuffled mini-batch stream over GPU tensors (drops last partial)."""

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
# regularizers  (these are the "new" pieces: L1 did not exist in the repo and
# the L2 meta-trainer was retired; both are defined once here and shared by the
# AdamW arm and the learned-optimizer arm so they are the SAME function R)
# ----------------------------------------------------------------------------
def penalty(flat: torch.Tensor, kind: str) -> torch.Tensor:
    """R(theta) over ALL optimizee parameters (weights + biases).
    l1: sum |theta|      l2: sum theta^2   (same convention as L2_reg_train_opt_func)"""
    if kind == "l1":
        return flat.abs().sum()
    if kind == "l2":
        return (flat ** 2).sum()
    raise ValueError(f"unknown regularizer '{kind}'")


def flat_params(model) -> torch.Tensor:
    return torch.cat([p.reshape(-1) for p in model.parameters()])


def flat_grads(model) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


# ----------------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------------
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
    flat = flat_params(model).detach()
    return dict(train_loss=tr_l, train_acc=tr_a, val_loss=va_l, val_acc=va_a,
                test_loss=te_l, test_acc=te_a,
                l1_norm=flat.abs().sum().item(), l2_sq=(flat ** 2).sum().item(),
                frac_small=(flat.abs() < 1e-3).float().mean().item())


# ----------------------------------------------------------------------------
# methods 1 & 2: AdamW (+ explicit penalty)
# ----------------------------------------------------------------------------
def run_adamw(seed, cfg: Cfg, data: Data, lr, reg="none", lam=0.0) -> dict:
    torch.manual_seed(seed)
    model = initialize_model("mnist", "mlp").to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.adamw_wd)
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
# methods 3 & 4: learned optimizer
#
# SPEED: the repo's meta_update() walks nn.Module trees, copies tensors back and
# forth and uses boolean-mask indexing in _preprocess (host syncs).  These jobs
# are launch-latency bound, so here the optimizee is a FLAT parameter vector
# with a functional forward, and _preprocess is the same function written with
# torch.where.  The maths is identical to MetaOptimizerLSTM.meta_update — this
# is verified numerically by test_equivalence.py (trajectory AND meta-gradient).
# ----------------------------------------------------------------------------
N_PARAMS = 784 * 20 + 20 + 20 * 10 + 10      # 15910, order: w1, b1, w2, b2 (== get_flat_params)


class FastLO(MetaOptimizerLSTM):
    """MetaOptimizerLSTM with a branch-free (sync-free) _preprocess; same weights,
    same state_dict, same forward()."""

    def _preprocess(self, x):
        if not self.preproc:
            return x
        x = x.reshape(-1, 1)
        ax = x.abs()
        keep = ax >= self.preproc_threshold
        c0 = torch.where(keep, torch.log(ax + 1e-8) / self.preproc_factor,
                         torch.full_like(x, -1.0))
        c1 = torch.where(keep, torch.sign(x), float(math.exp(self.preproc_factor)) * x)
        return torch.cat([c0, c1], dim=1).unsqueeze(1)


def mlp_logp(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """784-20-10 sigmoid MLP (model.Model.forward) on a flat parameter vector."""
    w1 = theta[:15680].view(20, 784)
    b1 = theta[15680:15700]
    w2 = theta[15700:15900].view(10, 20)
    b2 = theta[15900:15910]
    h = torch.sigmoid(F.linear(x.view(-1, 784), w1, b1))
    return F.log_softmax(F.linear(h, w2, b2), dim=-1)


def init_theta() -> torch.Tensor:
    """Fresh optimizee init drawn by the repo's own initialiser (global RNG)."""
    m = initialize_model("mnist", "mlp")
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()]).to(device)


def theta_to_model(theta: torch.Tensor):
    m = initialize_model("mnist", "mlp").to(device)
    off = 0
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(theta[off:off + p.numel()].view_as(p))
            off += p.numel()
    return m


def make_lo(cfg: Cfg) -> FastLO:
    dummy = MetaModel(initialize_model("mnist", "mlp"))      # unused by FastLO paths
    lo = FastLO(dummy, cfg.num_layers, cfg.hidden_size, preproc_factor=10.0,
                preproc=True).to(device)
    lo.set_double_backward(False)       # no grad-of-grad (no GAM/GSAM) -> keep cuDNN
    return lo


def reset_state(lo: FastLO, keep: bool):
    if keep:
        lo.h0, lo.c0 = lo.h0.detach(), lo.c0.detach()
    else:
        lo.h0, lo.c0 = initialize_rnn_hidden_state(lo.hidden_size, lo.num_layers, N_PARAMS)


def task_grad(theta, x, y):
    """(loss, detached gradient of the task loss at theta) — the LO input."""
    th = theta.detach().requires_grad_(True)
    loss = F.nll_loss(mlp_logp(th, x), y)
    g, = torch.autograd.grad(loss, th)
    return loss.detach(), g


def run_lo(lo, seed, cfg: Cfg, data: Data, steps=None, with_report=True):
    """Meta-test: train a fresh MLP with the frozen learned optimizer.
    The LO only ever sees the task gradient (no penalty at test time)."""
    steps = steps or cfg.eval_steps
    torch.manual_seed(seed)
    theta = init_theta()
    reset_state(lo, keep=False)
    stream = Stream(data.Xb, data.yb, cfg.eval_batch, seed)
    losses = []
    for _ in range(steps):
        x, y = stream.next()
        with torch.enable_grad():
            loss, g = task_grad(theta, x, y)
        losses.append(loss)
        with torch.no_grad():
            theta = theta + lo(g.view(-1, 1, 1))
    if not with_report:
        return torch.stack(losses)
    return report(theta_to_model(theta), data)


def heldout_score(lo, cfg: Cfg, data: Data) -> float:
    """Checkpoint-selection metric: mean batch loss over the last 10% of a
    cfg.eval_steps trajectory on the held-out stream, averaged over
    cfg.eval_epochs fresh optimizees (fixed seeds -> comparable across blocks)."""
    tail = max(cfg.eval_steps // 10, 1)
    with torch.random.fork_rng():
        s = [run_lo(lo, 9000 + e, cfg, data, with_report=False)[-tail:].mean()
             for e in range(cfg.eval_epochs)]
    return float(torch.stack(s).mean())


def meta_train_lo(kind, lam, seed, cfg: Cfg, data: Data, ckpt_path, log=print):
    """Meta-train an LSTM learned optimizer.

    kind = 'none'   : meta-loss = L_task                       (method 3)
           'smooth' : meta-loss = L_task + ls*L_smooth         (diagnostic)
           'l1'|'l2': meta-loss = L_task + ls*L_smooth + lam*R(theta_T)   (method 4)

    The loop is the repo's TBPTT protocol (train_opt_func /
    consolidated_reg_train_opt_func): unroll `tbptt` steps, meta-loss on the
    post-update optimizee at the last step, one Adam step on the LO per window,
    curriculum over `stages`.  Best checkpoint (held-out loss) is written to
    ckpt_path and returned.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    reg = kind if kind in ("l1", "l2") else "none"
    use_smooth = kind != "none"
    lo = make_lo(cfg)
    opt = torch.optim.Adam(lo.parameters(), lr=cfg.meta_lr)
    smooth = smooth_regularize(cfg.smooth_lamb, cfg.smooth_eps, cfg.smooth_pga, lo)
    stream = Stream(data.Xa, data.ya, cfg.meta_batch, seed)

    best, history, t0 = float("inf"), [], time.time()
    torch.save(lo.state_dict(), ckpt_path)           # guarantee a checkpoint exists
    for T in cfg.stages:
        tb = min(cfg.tbptt, T)
        for b in range(cfg.blocks_per_stage):
            task_acc = torch.zeros((), device=device)
            pen_acc = torch.zeros((), device=device)
            n = 0
            for _ in range(cfg.train_period):
                theta = init_theta()
                for k in range(T // tb):
                    reset_state(lo, keep=k > 0)
                    theta = theta.detach()
                    for _j in range(tb):
                        x, y = stream.next()
                        _, g = task_grad(theta, x, y)
                        theta = theta + lo(g.view(-1, 1, 1))     # graph-connected update
                    L_task = F.nll_loss(mlp_logp(theta, x), y)  # same-batch, as in the repo
                    L = L_task
                    if reg != "none":
                        R = penalty(theta, reg)
                        L = L + lam * R
                        pen_acc += R.detach()
                    if use_smooth:
                        L = L + cfg.smooth_lamb * smooth.run(g)
                    opt.zero_grad(set_to_none=True)
                    L.backward()
                    opt.step()
                    task_acc += L_task.detach()
                    n += 1
            score = heldout_score(lo, cfg, data)
            improved = score < best
            if improved:
                best = score
                torch.save(lo.state_dict(), ckpt_path)
            history.append(dict(stage=T, block=b, train_task_loss=task_acc.item() / n,
                                mean_R=pen_acc.item() / n, heldout=score, saved=improved,
                                elapsed=time.time() - t0))
            log(f"[{kind} lam={lam:g} seed={seed}] stage T={T:4d} block {b + 1}/"
                f"{cfg.blocks_per_stage} task={task_acc.item() / n:.4f} "
                f"R={pen_acc.item() / n:.1f} heldout={score:.4f}"
                f"{' *' if improved else ''}  ({time.time() - t0:.0f}s)")
    lo.load_state_dict(torch.load(ckpt_path, map_location=device))
    return lo, dict(best_heldout=best, history=history,
                    meta_train_seconds=time.time() - t0)
