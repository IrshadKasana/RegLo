"""
sam_extension.py — SAM as the regularizer R in the 4-method comparison
(stand-alone; imports common.py, does not modify it).

    2. MLP + AdamW + SAM        SAM applied to the optimizee at every step
                                (repo baseline adamw_sam_baseline.yaml: adaptive SAM, AdamW base)
    4. MLP + LO + SAM           the PROPOSED model's objective, penalty mode
                                (consolidated Algorithm 1 / Eq. 7 in utils.py):
                                L = L_task + ls * L_smooth + lam_r * L_SAM
                                L_SAM = L_task evaluated after re-unrolling the SAME window with the
                                learned optimizer's parameters phi moved to phi + rho*g/||g||
                                (SAM on the LO parameters, epsilon-hat treated as constant).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from common import (Cfg, Data, Stream, device, evaluate, report, init_theta, task_grad,
                    mlp_logp, make_lo, reset_state, heldout_score, smooth_regularize,
                    initialize_model)


# ----------------------------------------------------------------------------
# method 2: AdamW + SAM on the optimizee (mirrors regularizers.optimizers.SAM)
# ----------------------------------------------------------------------------
def run_adamw_sam(seed, cfg: Cfg, data: Data, lr, rho, adaptive=True) -> dict:
    torch.manual_seed(seed)
    model = initialize_model("mnist", "mlp").to(device)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.adamw_wd)
    stream = Stream(data.Xb, data.yb, cfg.eval_batch, seed)
    for _ in range(cfg.eval_steps):
        x, y = stream.next()
        opt.zero_grad(set_to_none=True)
        F.nll_loss(model(x), y).backward()
        with torch.no_grad():                                   # first_step
            gn = torch.norm(torch.stack([((p.abs() if adaptive else 1.0) * p.grad).norm(2)
                                         for p in params]), 2)
            scale = rho / (gn + 1e-12)
            old = [p.detach().clone() for p in params]
            for p in params:
                p.add_((p ** 2 if adaptive else 1.0) * p.grad * scale)
        opt.zero_grad(set_to_none=True)
        F.nll_loss(model(x), y).backward()                      # gradient at the perturbed point
        with torch.no_grad():                                   # second_step
            for p, o in zip(params, old):
                p.copy_(o)
        opt.step()
    return report(model, data)


# ----------------------------------------------------------------------------
# method 4: LO meta-trained with the SAM penalty (Algorithm 1, penalty mode)
# ----------------------------------------------------------------------------
def sam_window(lo, theta0, batches, lam_r, rho, xi, smooth, ls, opt):
    """One TBPTT window. Leaves the meta-gradient of
         L_task + ls*L_smooth + lam_r*L_adv
    in lo.parameters().grad (no optimizer step) and returns (theta_end, L_task, L_adv).
    Requires lo.h0/c0 to be the window-start state (reset_state called by the caller)."""
    phi = list(lo.parameters())
    window_state = lo.snapshot_state()
    theta = theta0.detach()
    for x, y in batches:                                          # pass 1 under phi
        _, g = task_grad(theta, x, y)
        theta = theta + lo(g.view(-1, 1, 1))
    x, y = batches[-1]
    L_task = F.nll_loss(mlp_logp(theta, x), y)
    L_smooth = smooth.run(g) if ls > 0 else torch.zeros((), device=device)
    state_after = lo.snapshot_state()

    g_task = torch.autograd.grad(L_task, phi, retain_graph=True, allow_unused=True)
    g_task = [torch.zeros_like(p) if q is None else q for p, q in zip(phi, g_task)]
    g_norm = torch.sqrt(sum((q ** 2).sum() for q in g_task))

    opt.zero_grad(set_to_none=True)
    (L_task + ls * L_smooth).backward()

    scale = float(rho) / (float(g_norm) + xi)                     # shared ascent step on phi
    with torch.no_grad():
        for p, q in zip(phi, g_task):
            p.add_(q, alpha=scale)
    lo.restore_state(window_state)                                # replay SAME window / batches
    th2 = theta0.detach()
    for xb, yb in batches:
        _, g2 = task_grad(th2, xb, yb)
        th2 = th2 + lo(g2.view(-1, 1, 1))
    L_adv = F.nll_loss(mlp_logp(th2, x), y)
    (lam_r * L_adv).backward()                                    # accumulate at phi_adv
    with torch.no_grad():
        for p, q in zip(phi, g_task):
            p.sub_(q, alpha=scale)
    lo.restore_state(state_after)
    return theta.detach(), L_task.detach(), L_adv.detach()


def meta_train_lo_sam(lam_r, seed, cfg: Cfg, data: Data, ckpt_path, rho=0.05, xi=1e-12, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    lo = make_lo(cfg)
    opt = torch.optim.Adam(lo.parameters(), lr=cfg.meta_lr)
    smooth = smooth_regularize(cfg.smooth_lamb, cfg.smooth_eps, cfg.smooth_pga, lo)
    stream = Stream(data.Xa, data.ya, cfg.meta_batch, seed)
    best, history, t0 = float("inf"), [], time.time()
    torch.save(lo.state_dict(), ckpt_path)
    for T in cfg.stages:
        tb = min(cfg.tbptt, T)
        for b in range(cfg.blocks_per_stage):
            lt = la = torch.zeros((), device=device)
            n = 0
            for _ in range(cfg.train_period):
                theta = init_theta()
                for k in range(T // tb):
                    reset_state(lo, keep=k > 0)
                    batches = [stream.next() for _ in range(tb)]
                    theta, L_task, L_adv = sam_window(lo, theta, batches, lam_r, rho, xi,
                                                      smooth, cfg.smooth_lamb, opt)
                    opt.step()
                    lt = lt + L_task
                    la = la + L_adv
                    n += 1
            score = heldout_score(lo, cfg, data)
            improved = score < best
            if improved:
                best = score
                torch.save(lo.state_dict(), ckpt_path)
            history.append(dict(stage=T, block=b, train_task_loss=lt.item() / n,
                                mean_L_adv=la.item() / n, heldout=score, saved=improved,
                                elapsed=time.time() - t0))
            log(f"[sam lam_r={lam_r:g} rho={rho:g} seed={seed}] stage T={T:4d} block {b + 1}/"
                f"{cfg.blocks_per_stage} task={lt.item() / n:.4f} L_adv={la.item() / n:.4f} "
                f"heldout={score:.4f}{' *' if improved else ''}  ({time.time() - t0:.0f}s)")
    lo.load_state_dict(torch.load(ckpt_path, map_location=device))
    return lo, dict(best_heldout=best, history=history, meta_train_seconds=time.time() - t0)
