"""
common.py — building blocks of the five-method ConvNet-4 sweep on SVHN and CIFAR-10 (70:20:10, ID + cross-dataset OOD).

Five methods (ConvNet-4 optimizee, 4 x [conv3x3-BN-ReLU-maxpool] + linear):

    M1  optimizee + AdamW
    M2  optimizee + AdamW + lambda * R(theta)                     R in {l1, l2, sam, gsam, gam}
    M3  optimizee + learned optimizer (LSTM LO)                   sweep: meta-lr of the LO's own training
    M4a optimizee + learned optimizer + lambda_smooth * L_smooth
    M5  optimizee + learned optimizer + lambda_smooth * L_smooth + lambda_reg * R   (same R set)

What is NEW relative to the repo (did not exist for ConvNet-4 / SVHN): ConvNet-4, the SVHN pipeline, L1/L2 for the
LO meta-objective, the lambda-weighted AdamW+R trainer.  REUSED unchanged from src/learning_regularizers/dmo:
MetaOptimizerLSTM / MetaModel, smooth_regularize, the Alg.-1 penalty-mode structure of consolidated_reg_train_opt_func.
Nothing under src/ is modified.

DATA PROTOCOL (see sweep_config.py)
    every dataset: official train+test pooled, fixed-seed split 70 % train / 20 % validation / 10 % test
        train (70 %)  optimizee training stream AND the LO's meta-training stream
        val   (20 %)  the ONLY split used to pick hyper-parameters (source dataset)
        test  (10 %)  ID evaluation of the selected configuration; never used for selection
    OOD (source -> target): the selected model is evaluated on the TARGET's 20 % partition, with the SOURCE's
        normalisation statistics, without retraining or tuning.
"""
from __future__ import annotations

import contextlib as _ctx
import gc
import io as _io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
DMO = REPO / "src" / "learning_regularizers" / "dmo"
sys.path.insert(0, str(DMO))
sys.path.insert(0, str(REPO))

with _ctx.redirect_stdout(_io.StringIO()):          # meta_optimizer prints a device banner
    from meta_optimizer import MetaOptimizerLSTM, MetaModel, device   # noqa: E402
    from regularizers.optimizers import smooth_regularize             # noqa: E402
from project_paths import datasets_dir                                # noqa: E402

import sweep_config as SC                                              # noqa: E402  (all sweep values live there)
REGS = SC.REGS
GRID = SC.GRID

NORM = {   # mean, std  (cifar10: dataloaders._NORM; svhn: standard torchvision-community stats)
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2430, 0.2610)),
    "svhn":    ((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
}


# ----------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------
@dataclass
class Cfg:
    width: int = 64                    # ConvNet-4 channels (standard Conv-4 = 64)
    # --- explicit-regularizer strengths that are NOT swept inside M5 (M2 sweeps rho, see rho_grid) ---
    rho_sam: float = SC.RHO_DEFAULT["sam"]      # perturbation radius rho used by M5 (on the LO parameters phi)
    rho_gam: float = SC.RHO_DEFAULT["gam"]      # and as the default of M2 (on theta)
    rho_gsam: float = SC.RHO_DEFAULT["gsam"]
    smooth_eps: float = 0.01           # smooth.yaml
    smooth_pga: int = 10
    # --- learned optimizer / meta-training ---
    hidden_size: int = 20
    num_layers: int = 2
    meta_opt: str = "adam"
    meta_lr: float = SC.META_LR_DEFAULT
    meta_batch: int = 128
    tbptt: int = 20
    stages: tuple = (100, 200, 500, 1000)
    blocks_per_stage: int = 1
    train_period: int = 2              # optimizee initialisations per meta-training call
    ckpt_epochs: int = 1
    # --- optimizee evaluation (identical budget for every method) ---
    eval_steps: int = SC.EVAL_STEPS
    eval_batch: int = SC.EVAL_BATCH
    lo_inits: int = SC.LO_INITS
    adamw_lrs: tuple = SC.ADAMW_LRS
    adamw_wd: float = SC.ADAMW_WD
    adamw_inits: int = SC.ADAMW_INITS
    curve_every: int = SC.CURVE_EVERY
    # --- sweep grids (centralised in sweep_config.py) ---
    grid: tuple = SC.GRID
    meta_lrs: tuple = SC.META_LRS
    rho_grid: tuple = tuple(sorted((k, tuple(v)) for k, v in SC.RHO_GRID.items()))
    radius: int = SC.RADIUS


def rho_of(cfg: Cfg, kind: str) -> float:
    return {"sam": cfg.rho_sam, "gam": cfg.rho_gam, "gsam": cfg.rho_gsam}[kind]


SMOKE = dict(width=16, stages=(20, 40), train_period=1, eval_steps=40, eval_batch=64,
             lo_inits=1, adamw_lrs=SC.SMOKE_ADAMW_LRS, adamw_inits=1, tbptt=10, meta_batch=64, curve_every=20,
             grid=SC.SMOKE_GRID, meta_lrs=SC.SMOKE_META_LRS, radius=1,
             rho_grid=tuple(sorted((k, tuple(v)) for k, v in SC.SMOKE_RHO_GRID.items())))


# ----------------------------------------------------------------------------
# ConvNet-4
# ----------------------------------------------------------------------------
class ConvNet4(nn.Module):
    """Standard Conv-4 (Vinyals/Finn): 4 x [conv3x3(width) - BN - ReLU - maxpool2] + linear.
    Only weight/bias parameters -> compatible with MetaModel's flat-param traversal."""

    def __init__(self, num_classes: int = 10, width: int = 64, in_ch: int = 3):
        super().__init__()
        layers, c = [], in_ch
        for _ in range(4):
            layers += [nn.Conv2d(c, width, 3, padding=1), nn.BatchNorm2d(width),
                       nn.ReLU(), nn.MaxPool2d(2)]
            c = width
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(width * 2 * 2, num_classes)   # 32 -> 2 after 4 pools

    def forward(self, x):
        return F.log_softmax(self.classifier(self.features(x).flatten(1)), dim=1)


def make_model(cfg: Cfg) -> nn.Module:
    return ConvNet4(10, cfg.width, 3).to(device)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
@dataclass
class Data:
    """One SOURCE dataset (70/20/10 split) plus the OOD evaluation sets of every other dataset.

    Xa/ya  meta-training stream of the LO   (= the 70 % train split)
    Xb/yb  optimizee training pool          (= the 70 % train split; same tensors as Xa)
    Xv/yv  20 % validation split            -> hyper-parameter selection ONLY
    Xt/yt  10 % test split                  -> ID evaluation of the selected configuration
    ood    {target name: (X uint8, y)}      -> target's 20 % partition, OOD evaluation ONLY
    """
    name: str
    mean: torch.Tensor; std: torch.Tensor
    Xa: torch.Tensor; ya: torch.Tensor
    Xb: torch.Tensor; yb: torch.Tensor
    Xv: torch.Tensor; yv: torch.Tensor
    Xt: torch.Tensor; yt: torch.Tensor
    ood: dict

    def prep(self, xu8):
        return (xu8.float().div_(255.0) - self.mean) / self.std


def _raw_pool(name, root):
    """Official train + test images of `name` pooled -> (uint8 NCHW tensor, long labels)."""
    from torchvision import datasets
    if name == "cifar10":
        parts = [datasets.CIFAR10(root, train=t, download=False) for t in (True, False)]
        xs = [torch.as_tensor(d.data).permute(0, 3, 1, 2).contiguous() for d in parts]
        ys = [torch.as_tensor(d.targets).long() for d in parts]
    elif name == "svhn":
        parts = [datasets.SVHN(root, split=sp, download=False) for sp in ("train", "test")]
        xs = [torch.as_tensor(d.data).contiguous() for d in parts]
        ys = [torch.as_tensor(d.labels).long() for d in parts]
    else:
        raise ValueError(f"dataset must be one of {sorted(NORM)}, got {name!r}")
    return torch.cat(xs), torch.cat(ys)


def split_indices(n: int, split=SC.SPLIT, seed: int = SC.SPLIT_SEED):
    """Deterministic 70/20/10 split of range(n) -> (train, val, test) index arrays. The SAME call is made
    wherever a dataset appears (as source or as OOD target), so the roles can never overlap."""
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    n_tr = int(round(split[0] * n))
    n_va = int(round(split[1] * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def load_data(name: str, ood_names=(), split=SC.SPLIT, split_seed: int = SC.SPLIT_SEED) -> Data:
    root = str(datasets_dir(__file__))
    X, y = _raw_pool(name, root)
    tr, va, te = split_indices(len(X), split, split_seed)
    t = torch.as_tensor
    m, s = NORM[name]
    dv = lambda z: z.to(device)
    ood = {}
    for o in ood_names:
        if o == name:
            continue
        Xo, yo = _raw_pool(o, root)
        _, vo, _ = split_indices(len(Xo), split, split_seed)       # the target's own 20 % partition
        ood[o] = (dv(Xo[t(vo)]), dv(yo[t(vo)]))
        del Xo, yo
    Xtr, ytr = dv(X[t(tr)]), dv(y[t(tr)])
    return Data(name, dv(torch.tensor(m).view(1, 3, 1, 1)), dv(torch.tensor(s).view(1, 3, 1, 1)),
                Xtr, ytr, Xtr, ytr, dv(X[t(va)]), dv(y[t(va)]), dv(X[t(te)]), dv(y[t(te)]), ood)


class Stream:
    """Endless shuffled mini-batch iterator; also usable as the 'DataLoader' of meta_train."""

    def __init__(self, data: Data, X, y, bs, seed):
        self.d, self.X, self.y, self.bs, self.n = data, X, y, bs, X.shape[0]
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
        return self.d.prep(self.X[ix]), self.y[ix]

    def __iter__(self):
        return self

    __next__ = next


@torch.no_grad()
def evaluate(model, data: Data, X, y, chunk=2000, return_msp=False):
    """-> (nll, accuracy[, max-softmax-probability per sample]); inputs are normalised with data's (source) stats."""
    was = model.training
    model.eval()
    loss, correct, msp = 0.0, 0, []
    for i in range(0, X.shape[0], chunk):
        xb = data.prep(X[i:i + chunk])
        yb = y[i:i + chunk]
        out = model(xb)
        loss += F.nll_loss(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
        if return_msp:
            msp.append(out.exp().max(1).values.float().cpu())
    model.train(was)
    if return_msp:
        return loss / X.shape[0], correct / X.shape[0], torch.cat(msp).numpy()
    return loss / X.shape[0], correct / X.shape[0]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(score_pos > score_neg) (ties count 1/2), via average ranks.  pos = ID samples, neg = OOD samples."""
    sc = np.concatenate([pos, neg])
    try:
        from scipy.stats import rankdata
        ranks = rankdata(sc)
    except Exception:           # noqa: BLE001 - scipy missing: ordinal ranks (ties are rare for float scores)
        order = sc.argsort(kind="mergesort")
        ranks = np.empty(len(sc))
        ranks[order] = np.arange(1, len(sc) + 1)
    n_p, n_n = len(pos), len(neg)
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def report(model, data: Data, n_train_eval: int = 10000) -> dict:
    """Metrics of one trained optimizee on every split.  Selection code reads ONLY val_*; train_*, test_* and the
    ood block are stored for reporting.  ood[target] = accuracy / NLL on the target's 20 % partition and the
    max-softmax-probability AUROC of ID-test vs OOD (a label-space-free measure of the shift)."""
    tr_l, tr_a = evaluate(model, data, data.Xb[:n_train_eval], data.yb[:n_train_eval])
    va_l, va_a = evaluate(model, data, data.Xv, data.yv)
    te_l, te_a, te_msp = evaluate(model, data, data.Xt, data.yt, return_msp=True)
    ood = {}
    for name, (Xo, yo) in data.ood.items():
        o_l, o_a, o_msp = evaluate(model, data, Xo, yo, return_msp=True)
        ood[name] = dict(loss=o_l, acc=o_a, auroc_msp=auroc(te_msp, o_msp))
    flat = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    return dict(train_loss=tr_l, train_acc=tr_a, val_loss=va_l, val_acc=va_a,
                test_loss=te_l, test_acc=te_a, ood=ood,
                l1_norm=flat.abs().sum().item(), l2_sq=(flat ** 2).sum().item(),
                diverged=bool(not np.isfinite(va_l)))


# ----------------------------------------------------------------------------
# regularizers R(theta)  (explicit, optimizee side — Method 2)
# ----------------------------------------------------------------------------
def penalty(params, kind: str) -> torch.Tensor:
    """l1: sum|theta|   l2: sum theta^2  (sum convention, as L2_reg_train_opt_func / l1_l2_mnist)."""
    if kind == "l1":
        return sum(p.abs().sum() for p in params)
    if kind == "l2":
        return sum((p ** 2).sum() for p in params)
    raise ValueError(kind)


@_ctx.contextmanager
def _frozen_bn(model):
    """Second (perturbed) pass must not touch BN running statistics."""
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    moms = [m.momentum for m in bns]
    for m in bns:
        m.momentum = 0.0
    try:
        yield
    finally:
        for m, v in zip(bns, moms):
            m.momentum = v


def fill_grads(model, x, y, kind: str, lam: float, rho: float) -> float:
    """Set p.grad = grad of  L(theta) + lam * R(theta);  returns L(theta).

        l1 / l2 : R = sum|theta| / sum theta^2
        sam     : R = L(theta + eps),                eps = rho * g/||g||  (eps constant)
        gsam    : R = L(theta + eps) - L(theta)      (surrogate gap, NOT detached)
        gam     : R = rho * ||grad L(theta + eps)||  (Hessian-vector product)
    lam == 0 or kind == 'none' is plain AdamW's gradient."""
    params = [p for p in model.parameters()]
    loss = F.nll_loss(model(x), y)
    if kind == "none" or lam == 0.0:
        model.zero_grad(set_to_none=True)
        loss.backward()
        return loss.item()
    if kind in ("l1", "l2"):
        model.zero_grad(set_to_none=True)
        (loss + lam * penalty(params, kind)).backward()
        return loss.item()

    g = torch.autograd.grad(loss, params)
    gn = torch.sqrt(sum((gi ** 2).sum() for gi in g))
    eps = [rho * gi / (gn + 1e-12) for gi in g]
    with torch.no_grad():
        for p, e in zip(params, eps):
            p.add_(e)
    try:
        with _frozen_bn(model):
            loss_adv = F.nll_loss(model(x), y)
            if kind in ("sam", "gsam"):
                g_adv = torch.autograd.grad(loss_adv, params)
            elif kind == "gam":
                ga = torch.autograd.grad(loss_adv, params, create_graph=True)
                r = rho * torch.sqrt(sum((gi ** 2).sum() for gi in ga) + 1e-24)
                g_adv = torch.autograd.grad(r, params)
            else:
                raise ValueError(kind)
    finally:
        with torch.no_grad():
            for p, e in zip(params, eps):
                p.sub_(e)
    for p, gi, ga_ in zip(params, g, g_adv):
        p.grad = gi * (1.0 - lam) + lam * ga_ if kind == "gsam" else gi + lam * ga_
    return loss.item()


# ----------------------------------------------------------------------------
# M1 / M2 : AdamW (+ lambda R)
# ----------------------------------------------------------------------------
def _curve_point(model, data: Data, step, tr_loss):
    va_l, va_a = evaluate(model, data, data.Xv, data.yv)
    return dict(step=step, train_loss=tr_loss, val_loss=va_l, val_acc=va_a)


def run_adamw(seed, cfg: Cfg, data: Data, lr: float, kind="none", lam=0.0, rho=None) -> dict:
    """M1 (kind='none') / M2.  rho=None -> the default radius of `kind`.  Records a train-loss / val-acc curve."""
    torch.manual_seed(seed)
    model = make_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.adamw_wd)
    st = Stream(data, data.Xb, data.yb, cfg.eval_batch, seed)
    model.train()
    if rho is None and kind in ("sam", "gsam", "gam"):
        rho = rho_of(cfg, kind)
    t0, win, curve = time.time(), [], []
    for step in range(1, cfg.eval_steps + 1):
        x, y = st.next()
        win.append(fill_grads(model, x, y, kind, lam, rho if kind in ("sam", "gsam", "gam") else 0.0))
        opt.step()
        if step % cfg.curve_every == 0:
            curve.append(_curve_point(model, data, step, float(np.mean(win))))
            win = []
    out = report(model, data)
    out["seconds"] = time.time() - t0
    out["curve"] = curve
    return out


# ----------------------------------------------------------------------------
# M3 / M4 : learned optimizer
# ----------------------------------------------------------------------------
def make_lo(cfg: Cfg):
    lo = MetaOptimizerLSTM(MetaModel(make_model(cfg)), cfg.num_layers, cfg.hidden_size,
                           preproc_factor=10.0, preproc=True).to(device)
    return lo


def run_lo(lo, seed, cfg: Cfg, data: Data, with_report=True):
    lo.eval()
    torch.manual_seed(seed)
    model = make_model(cfg)
    model.train()
    lo.reset_lstm(keep_states=False, model=model)
    st = Stream(data, data.Xb, data.yb, cfg.eval_batch, seed)
    losses, curve, t0 = [], [], time.time()
    for step in range(1, cfg.eval_steps + 1):
        x, y = st.next()
        loss = F.nll_loss(model(x), y)
        model.zero_grad()
        loss.backward()
        losses.append(loss.detach())
        with torch.no_grad():
            lo.meta_update(model)
        if with_report and step % cfg.curve_every == 0:
            curve.append(_curve_point(model, data, step, float(torch.stack(losses[-cfg.curve_every:]).mean())))
    lo.train()
    if not with_report:
        return torch.stack(losses)
    out = report(model, data)
    out["seconds"] = time.time() - t0
    out["curve"] = curve
    return out


def ckpt_score(lo, cfg: Cfg, data: Data) -> float:
    tail = max(cfg.eval_steps // 10, 1)
    out = []
    with torch.random.fork_rng():
        for e in range(cfg.ckpt_epochs):
            out.append(run_lo(lo, 9000 + e, cfg, data, with_report=False)[-tail:].mean().item())
    return float(np.mean(out))


def _flat_grads(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).requires_grad_()


def meta_unroll(num_unroll, loader, lo, smooth, opt, kind, lam_s, lam_r, cfg: Cfg):
    """One meta-training call. Port of utils.consolidated_reg_train_opt_func (Alg. 1,
    penalty mode) with a ConvNet-4 factory and two extra regularizers:

        total = L_task + lam_s * L_smooth + lam_r * L_reg
        sam : L_reg = L_task(phi_adv)
        gsam: L_reg = L_task(phi_adv) - L_task(phi)    (true surrogate gap, not detached; see README)
        gam : L_reg = rho * ||grad_phi L_task(phi_adv)||
        l1/l2: L_reg = sum|theta_T| / sum theta_T^2    (final unrolled optimizee params)
    """
    phi = [p for p in lo.parameters()]
    for _ in range(cfg.train_period):
        it = iter(loader)
        model, model2 = make_model(cfg), make_model(cfg)
        model.train(); model2.train()
        for k in range(num_unroll // cfg.tbptt):
            lo.reset_lstm(keep_states=k > 0, model=model)
            window_state = lo.snapshot_state()
            window_batches = []
            for a, b in zip(model2.parameters(), model.parameters()):
                a.data.copy_(b.data)

            for _j in range(cfg.tbptt):                              # pass 1 under phi
                x, y = next(it)
                window_batches.append((x, y))
                loss = F.nll_loss(model(x), y)
                model.zero_grad()
                loss.backward()
                model_temp = lo.meta_update(model)

            x, y = window_batches[-1]
            L_task = F.nll_loss(model_temp(x), y)
            L_smooth = torch.zeros(1, device=device)
            if lam_s > 0.0:
                L_smooth = smooth.run(_flat_grads(model))
            state_after_pass1 = lo.snapshot_state()

            if lam_r > 0.0 and kind in ("l1", "l2"):
                R = penalty([p for p in lo.meta_model.model.parameters()], kind)
                opt.zero_grad()
                (L_task + lam_s * L_smooth + lam_r * R).backward()
            else:
                g_task = torch.autograd.grad(L_task, phi, retain_graph=True, allow_unused=True)
                g_task = [torch.zeros_like(p) if g is None else g for p, g in zip(phi, g_task)]
                g_norm = torch.sqrt(sum((g ** 2).sum() for g in g_task))
                opt.zero_grad()
                # gsam: total = L + lam*(L_adv - L)  =>  (1-lam)*L at phi, lam*L_adv at phi_adv
                w_task = (1.0 - lam_r) if (kind == "gsam" and lam_r > 0.0) else 1.0
                (w_task * L_task + lam_s * L_smooth).backward()

                if lam_r > 0.0:
                    scale = rho_of(cfg, kind) / (float(g_norm) + 1e-12)
                    with torch.no_grad():
                        for p, g in zip(phi, g_task):
                            p.add_(g, alpha=scale)
                    lo.reset_lstm(keep_states=False, model=model2)
                    lo.restore_state(window_state)
                    for j in range(cfg.tbptt):                       # pass 2 under phi_adv
                        x, y = window_batches[j]
                        l2_ = F.nll_loss(model2(x), y)
                        model2.zero_grad()
                        l2_.backward()
                        model2_temp = lo.meta_update(model2)
                    x, y = window_batches[-1]
                    L_adv = F.nll_loss(model2_temp(x), y)
                    if kind == "sam":
                        L_reg = L_adv
                    elif kind == "gsam":
                        L_reg = L_adv          # the -L_task half of the gap was applied above
                    elif kind == "gam":
                        ga = torch.autograd.grad(L_adv, phi, create_graph=True, allow_unused=True)
                        ga = [torch.zeros_like(p) if g is None else g for p, g in zip(phi, ga)]
                        L_reg = rho_of(cfg, kind) * torch.sqrt(sum((g ** 2).sum() for g in ga) + 1e-24)
                    else:
                        raise ValueError(kind)
                    (lam_r * L_reg).backward()
                    with torch.no_grad():
                        for p, g in zip(phi, g_task):
                            p.sub_(g, alpha=scale)
                    lo.restore_state(state_after_pass1)
                    lo.meta_model.copy_params_from(model)
            opt.step()
        gc.collect()


def meta_train(kind, lam_s, lam_r, seed, cfg: Cfg, data: Data, ckpt_path, meta_lr=None, log=print):
    """Curriculum meta-training; the best checkpoint (tail task loss on a fixed held-out
    trajectory) is kept.  (lam_s, lam_r) = (0, 0) -> M3; lam_r == 0 -> M4a; both > 0 -> M5 (lam_s == 0 and
    lam_r > 0 is the lambda_reg axis used to place the M5 block).  meta_lr = Adam rate of the LO's weights."""
    meta_lr = cfg.meta_lr if meta_lr is None else meta_lr
    torch.manual_seed(seed)
    np.random.seed(seed)
    lo = make_lo(cfg)
    smooth = smooth_regularize(lam_s, cfg.smooth_eps, cfg.smooth_pga, lo)
    opt = (torch.optim.AdamW if cfg.meta_opt == "adamw" else torch.optim.Adam)(
        lo.parameters(), lr=meta_lr)
    loader = Stream(data, data.Xa, data.ya, cfg.meta_batch, seed)
    best, history, t0 = float("inf"), [], time.time()
    torch.save(lo.state_dict(), ckpt_path)
    for T in cfg.stages:
        for b in range(cfg.blocks_per_stage):
            with _ctx.redirect_stdout(_io.StringIO()):
                meta_unroll(T, loader, lo, smooth, opt, kind, lam_s, lam_r, cfg)
            score = ckpt_score(lo, cfg, data)
            improved = np.isfinite(score) and score < best
            if improved:
                best = score
                torch.save(lo.state_dict(), ckpt_path)
            history.append(dict(stage=T, block=b, ckpt_score=score, saved=bool(improved),
                                elapsed=time.time() - t0))
            log(f"[{data.name} {kind} ls={lam_s:g} lr={lam_r:g} mlr={meta_lr:g} seed={seed}] T={T:4d} "
                f"ckpt={score:.4f}{' *' if improved else ''} ({time.time() - t0:.0f}s)")
    lo.load_state_dict(torch.load(ckpt_path, map_location=device))
    return lo, dict(best_ckpt_score=best, history=history, meta_lr=meta_lr, meta_train_seconds=time.time() - t0)
