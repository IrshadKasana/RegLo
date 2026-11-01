"""
common.py — building blocks of the stand-alone sweep of the PROPOSED method on
FashionMNIST with an MLP optimizee.

    meta-loss = L_task + lambda_smooth * L_smooth + lambda_reg * L_SAM

(optimizee + learned optimizer + smoothing + SAM-on-optimizer).  The meta-training
step is the repo's own `utils.consolidated_reg_train_opt_func` (Alg. 1, "penalty"
mode), called unchanged; only the data feed, the curriculum/checkpoint loop and
the evaluation protocol live here.  Nothing under src/ is modified.

DATA PROTOCOL (same as dataloaders.py / the MNIST experiments; FMNIST norm 0.5/0.5)
    A        30000  meta-training stream for the LO
    B_train  27000  optimizee training pool
    B_val     3000  hyper-parameter selection -- never trained on
    test     10000  official test set -- stored, never used for selection
"""
from __future__ import annotations

import contextlib as _ctx
import io as _io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
DMO = REPO / "src" / "learning_regularizers" / "dmo"
sys.path.insert(0, str(DMO))
sys.path.insert(0, str(REPO))

with _ctx.redirect_stdout(_io.StringIO()):          # banners
    from meta_optimizer import MetaOptimizerLSTM, MetaModel, device   # noqa: E402
    from model import initialize_model                                # noqa: E402
    from regularizers.optimizers import smooth_regularize             # noqa: E402
    from utils import consolidated_reg_train_opt_func                 # noqa: E402
from project_paths import datasets_dir                                # noqa: E402

DATASET, MODEL = "fmnist", "mlp"
MEAN, STD = 0.5, 0.5                                 # dataloaders._NORM['fmnist']


@dataclass
class Cfg:
    # --- grids (project's run_sweep.py presets; 0.0 switches a term off) ---
    # user-specified grid, used for BOTH lambda_smooth and lambda_reg (0 = term off)
    grid: tuple = (0.0, 0.001, 0.002, 0.004, 0.006, 0.007, 0.01, 0.02, 0.03,
                   0.04, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5)
    rho: float = 0.005                 # run_sweep.py: SAM ascent radius of the sweep
    smooth_eps: float = 0.01           # smooth.yaml
    smooth_pga: int = 10
    # --- learned optimizer / meta-training ---
    hidden_size: int = 20
    num_layers: int = 2
    meta_opt: str = "adam"             # repo SAM config uses sgd@1e-3; adam used here (see README)
    meta_lr: float = 1e-3
    meta_batch: int = 128
    tbptt: int = 20
    stages: tuple = (100, 200, 500, 1000)
    blocks_per_stage: int = 1
    train_period: int = 6              # optimizee initialisations per call
    ckpt_epochs: int = 1
    # --- optimizee evaluation ---
    eval_steps: int = 1000
    eval_batch: int = 512
    lo_inits: int = 3
    adamw_lrs: tuple = (1e-3, 3e-3, 1e-2, 3e-2)
    adamw_wd: float = 0.01
    adamw_inits: int = 6


SMOKE = dict(grid=(0.0, 0.05, 0.5), stages=(20, 40), train_period=2,
             eval_steps=60, lo_inits=1, adamw_lrs=(1e-2,), adamw_inits=2)


# ----------------------------------------------------------------------------
@dataclass
class Data:
    Xa: torch.Tensor; ya: torch.Tensor
    Xb: torch.Tensor; yb: torch.Tensor
    Xv: torch.Tensor; yv: torch.Tensor
    Xt: torch.Tensor; yt: torch.Tensor


def load_fmnist(split_seed: int = 42, val_fraction: float = 0.1) -> Data:
    from torchvision import datasets
    root = str(datasets_dir(__file__))
    tr = datasets.FashionMNIST(root, train=True, download=False)
    te = datasets.FashionMNIST(root, train=False, download=False)

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


class Loader:
    """Endless GPU-tensor 'DataLoader' accepted by consolidated_reg_train_opt_func."""

    def __init__(self, stream):
        self.s = stream

    def __iter__(self):
        return self

    def __next__(self):
        return self.s.next()


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
    return dict(train_loss=tr_l, train_acc=tr_a, val_loss=va_l, val_acc=va_a,
                test_loss=te_l, test_acc=te_a)


def run_adamw(seed, cfg: Cfg, data: Data, lr) -> dict:
    torch.manual_seed(seed)
    model = initialize_model(DATASET, MODEL).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.adamw_wd)
    st = Stream(data.Xb, data.yb, cfg.eval_batch, seed)
    for _ in range(cfg.eval_steps):
        x, y = st.next()
        loss = F.nll_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return report(model, data)


def make_lo(cfg: Cfg):
    m = initialize_model(DATASET, MODEL).to(device)
    lo = MetaOptimizerLSTM(MetaModel(m), cfg.num_layers, cfg.hidden_size,
                           preproc_factor=10.0, preproc=True).to(device)
    return lo


def run_lo(lo, seed, cfg: Cfg, data: Data, with_report=True):
    lo.eval()
    torch.manual_seed(seed)
    model = initialize_model(DATASET, MODEL).to(device)
    lo.reset_lstm(keep_states=False, model=model)
    st = Stream(data.Xb, data.yb, cfg.eval_batch, seed)
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = st.next()
        loss = F.nll_loss(model(x), y)
        model.zero_grad()
        loss.backward()
        losses.append(loss.detach())
        with torch.no_grad():
            lo.meta_update(model)
    lo.train()
    return report(model, data) if with_report else torch.stack(losses)


def ckpt_score(lo, cfg: Cfg, data: Data) -> float:
    tail = max(cfg.eval_steps // 10, 1)
    out = []
    with torch.random.fork_rng():
        for e in range(cfg.ckpt_epochs):
            out.append(run_lo(lo, 9000 + e, cfg, data, with_report=False)[-tail:].mean().item())
    return float(np.mean(out))


def meta_train(lam_s, lam_r, seed, cfg: Cfg, data: Data, ckpt_path, log=print):
    """Meta-train the LO with  L_task + lam_s*L_smooth + lam_r*L_SAM  using the repo's
    consolidated objective; curriculum over cfg.stages; best checkpoint (tail task
    loss on fixed held-out trajectories) is kept."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    lo = make_lo(cfg)
    smooth = smooth_regularize(lam_s, cfg.smooth_eps, cfg.smooth_pga, lo)
    loader = Loader(Stream(data.Xa, data.ya, cfg.meta_batch, seed))
    args = SimpleNamespace(
        base_optimizer=cfg.meta_opt, lr=cfg.meta_lr, momentum=0.0, weight_decay=0.0,
        if_smooth=lam_s > 0.0, smooth_lamb=lam_s, lambda_reg=lam_r, xi=1e-12,
        regularizer="sam", rho=cfg.rho, train_period=cfg.train_period,
        dataset=DATASET, optimizee_model_name=MODEL, truncated_bptt_step=cfg.tbptt)
    best, history, t0 = float("inf"), [], time.time()
    torch.save(lo.state_dict(), ckpt_path)
    for T in cfg.stages:
        for b in range(cfg.blocks_per_stage):
            with _ctx.redirect_stdout(_io.StringIO()):     # the repo function is chatty
                consolidated_reg_train_opt_func(T, loader, device, lo, smooth, args)
            score = ckpt_score(lo, cfg, data)
            improved = score < best
            if improved:
                best = score
                torch.save(lo.state_dict(), ckpt_path)
            history.append(dict(stage=T, block=b, ckpt_score=score, saved=improved,
                                elapsed=time.time() - t0))
            log(f"[ls={lam_s:g} lr={lam_r:g} seed={seed}] T={T:4d} blk {b + 1}/{cfg.blocks_per_stage} "
                f"ckpt={score:.4f}{' *' if improved else ''} ({time.time() - t0:.0f}s)")
    lo.load_state_dict(torch.load(ckpt_path, map_location=device))
    return lo, dict(best_ckpt_score=best, history=history, meta_train_seconds=time.time() - t0)
