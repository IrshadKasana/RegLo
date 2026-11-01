"""
Sharpness-aware regularizers, implemented as wrappers around a base optimizer.

CONCEPTUAL NOTE
---------------
None of these is an additive loss penalty of the form `loss + lambda * term`.
They are gradient-modification procedures: the base optimizer never sees a
modified loss, only a modified gradient. `rho` is the RADIUS of a ball in
weight space, not a mixing coefficient.

    SAM   min_w max_{||e||<=rho} L(w+e)
          -> one hyperparameter: rho
    GSAM  descends the SAM objective and the surrogate gap L(w+e)-L(w)
          -> two hyperparameters: rho, alpha
    GAM   min_w L(w) + alpha * rho * max_{||e||<=rho} ||grad L(w+e)||
          -> genuinely additive; radius rho and weight alpha

Uniform API:

    opt = build_optimizer("sam", model.parameters(), hp)
    loss = opt.step(closure)      # closure() does fwd+bwd, returns loss

Cost in forward+backward passes per optimizer step:
    none 1  |  sam 2  |  gsam 2  |  gam 4
"""

from __future__ import annotations

import torch

__all__ = ["SAM", "GSAM", "GAM", "WSAM", "NoReg", "build_optimizer",
           "PASSES_PER_STEP"]

PASSES_PER_STEP = {"none": 1, "sam": 2, "gsam": 2, "wsam": 2, "gam": 4}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _grad_norm(params, adaptive: bool = False) -> torch.Tensor:
    terms = []
    for p in params:
        if p.grad is None:
            continue
        g = torch.abs(p) * p.grad if adaptive else p.grad
        terms.append(g.norm(p=2))
    if not terms:
        return torch.zeros((), device=params[0].device)
    return torch.norm(torch.stack(terms), p=2)


def _clone_grads(params):
    return [None if p.grad is None else p.grad.detach().clone() for p in params]


def _dot(a, b):
    tot = None
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        v = (x * y).sum()
        tot = v if tot is None else tot + v
    return tot


def _list_norm(a):
    terms = [x.norm(p=2) for x in a if x is not None]
    if not terms:
        return None
    return torch.norm(torch.stack(terms), p=2)


class _Wrapper(torch.optim.Optimizer):
    """Shared plumbing: own the base optimizer, share its param_groups."""

    def __init__(self, params, base_optimizer, defaults, base_kwargs):
        super().__init__(params, {**defaults, **base_kwargs})
        self.base_optimizer = base_optimizer(self.param_groups, **base_kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def all_params(self):
        return [p for g in self.param_groups for p in g["params"]]

    @torch.no_grad()
    def _perturb(self, group, radius, adaptive, key="e"):
        gn = _grad_norm(group["params"], adaptive)
        scale = radius / (gn + 1e-12)
        for p in group["params"]:
            if p.grad is None:
                continue
            e = (torch.pow(p, 2) if adaptive else 1.0) * p.grad * scale
            p.add_(e)
            self.state[p][key] = e

    @torch.no_grad()
    def _restore(self, key="e"):
        for group in self.param_groups:
            for p in group["params"]:
                e = self.state[p].pop(key, None)
                if e is not None:
                    p.sub_(e)


# --------------------------------------------------------------------------- #
class NoReg(_Wrapper):
    """Plain AdamW. The control arm."""

    def __init__(self, params, base_optimizer=torch.optim.AdamW, **base_kwargs):
        super().__init__(params, base_optimizer, {}, base_kwargs)

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        self.zero_grad(set_to_none=True)
        loss = closure()
        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)
        return loss


# --------------------------------------------------------------------------- #
class SAM(_Wrapper):
    r"""
    Sharpness-Aware Minimization (Foret et al., ICLR 2021).
    `adaptive=True` gives ASAM (Kwon et al., ICML 2021).

        g   = grad L(w)
        e   = rho * g / ||g||           (adaptive: e_i = rho * w_i^2 g_i / ||T_w g||)
        g~  = grad L(w + e)
        w  <- AdamW(w, g~)

    WARNING ON `adaptive`
    ---------------------
    ASAM rescales the perturbation elementwise by |w|. For a network with
    weights of order 0.01-0.1 the effective radius shrinks by one to two
    orders of magnitude, so ASAM needs rho roughly 10-50x LARGER than vanilla
    SAM. Vanilla lives near rho in [0.01, 0.5]; ASAM near [0.1, 4.0].
    Running `adaptive=True, rho=0.05` is very nearly plain AdamW with double
    the compute, and is the single most common misconfiguration of this method.

    m-SHARPNESS (`m`)
    -----------------
    SAM's reported benefit depends on the batch size used to compute the
    ascent step, not just on rho. With `m` set, the batch is split into
    chunks of `m` examples, a separate perturbation is computed per chunk,
    and the perturbed gradients are averaged. Smaller m generally regularizes
    more strongly. `m=None` (default) perturbs using the whole batch, which is
    what most implementations do. If you report SAM results, state your m --
    it is a genuine hyperparameter that is very often left unreported.
    """

    def __init__(self, params, base_optimizer=torch.optim.AdamW,
                 rho: float = 0.05, adaptive: bool = False, m: int | None = None,
                 **base_kwargs):
        if rho < 0:
            raise ValueError("rho must be non-negative")
        super().__init__(params, base_optimizer,
                         dict(rho=rho, adaptive=adaptive), base_kwargs)
        self.m = m

    def step(self, closure):
        closure = torch.enable_grad()(closure)

        if self.m is None:
            self.zero_grad(set_to_none=True)
            loss = closure()
            for g in self.param_groups:
                self._perturb(g, g["rho"], g["adaptive"])
            self.zero_grad(set_to_none=True)
            closure()
            self._restore()
            self.base_optimizer.step()
            self.zero_grad(set_to_none=True)
            return loss

        # ---- m-sharpness path ------------------------------------------- #
        params = self.all_params()
        accum = [torch.zeros_like(p) for p in params]
        n_chunks, total_loss = 0, 0.0
        for sub in closure(microbatch=self.m):
            self.zero_grad(set_to_none=True)
            loss = sub()
            for g in self.param_groups:
                self._perturb(g, g["rho"], g["adaptive"])
            self.zero_grad(set_to_none=True)
            sub()
            self._restore()
            with torch.no_grad():
                for a, p in zip(accum, params):
                    if p.grad is not None:
                        a.add_(p.grad)
            total_loss += float(loss.detach())
            n_chunks += 1

        with torch.no_grad():
            for a, p in zip(accum, params):
                p.grad = a / max(n_chunks, 1)
        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)
        return torch.tensor(total_loss / max(n_chunks, 1))


# --------------------------------------------------------------------------- #
class GSAM(_Wrapper):
    r"""
    Surrogate Gap Guided SAM (Zhuang et al., ICLR 2022).

        g_par  = (<g, g_adv> / ||g_adv||^2) * g_adv
        g_perp = g - g_par
        g_gsam = g_adv - alpha * g_perp

    Descends the SAM objective while also shrinking the surrogate gap
    h(w) = L(w+e) - L(w), which is what actually tracks flatness.

    Two hyperparameters: `rho` (radius) and `alpha` (gap coefficient).
    `rho_scheduler(step) -> rho` optionally ties rho to the LR schedule as in
    the paper; leave it None for a constant radius, but then do not also set
    rho_min/rho_max anywhere, or one of the two will be silently ignored.
    """

    def __init__(self, params, base_optimizer=torch.optim.AdamW,
                 rho: float = 0.05, alpha: float = 0.1, adaptive: bool = False,
                 rho_scheduler=None, **base_kwargs):
        super().__init__(params, base_optimizer,
                         dict(rho=rho, alpha=alpha, adaptive=adaptive), base_kwargs)
        self.rho_scheduler = rho_scheduler
        self._step = 0

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        params = self.all_params()
        rho = (self.rho_scheduler(self._step) if self.rho_scheduler
               else self.param_groups[0]["rho"])
        alpha = self.param_groups[0]["alpha"]

        self.zero_grad(set_to_none=True)
        loss = closure()
        g_clean = _clone_grads(params)

        for g in self.param_groups:
            self._perturb(g, rho, g["adaptive"])
        self.zero_grad(set_to_none=True)
        closure()
        self._restore()
        g_adv = _clone_grads(params)

        with torch.no_grad():
            denom = _dot(g_adv, g_adv)
            coef = _dot(g_clean, g_adv) / (denom + 1e-12) if denom is not None else 0.0
            for p, gc, ga in zip(params, g_clean, g_adv):
                if gc is None or ga is None:
                    continue
                p.grad = ga - alpha * (gc - coef * ga)

        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)
        self._step += 1
        return loss


# --------------------------------------------------------------------------- #
class WSAM(_Wrapper):
    r"""
    Weighted SAM (Yue et al., KDD 2023) -- sharpness as an EXPLICIT
    regularization term with its own weight. This is the method that actually
    matches the objective "AdamW + lambda * SAM".

        L_WSAM(w) = L(w) + lambda * L_sharp(w),   lambda = gamma / (1 - gamma)
        L_sharp(w) = max_{||d|| <= rho} L(w + d) - L(w)

    Vanilla SAM is the single point lambda = 1 on this continuum:

        gamma = 0     -> lambda = 0    -> plain AdamW, no sharpness
        gamma = 0.5   -> lambda = 1    -> EXACTLY SAM
        gamma = 0.8   -> lambda = 4
        gamma = 0.88  -> lambda = 7.33 -> paper's best on ResNet18/CIFAR-10
        gamma -> 1    -> lambda -> inf -> pure sharpness, ignores the loss

    Accepts either `gamma` or `lam` (converted via gamma = lam/(1+lam)), so you
    can think in whichever parameterisation is natural.

    WEIGHT DECOUPLE (`decouple=True`, the paper's recommended variant)
    The sharpness term bypasses the base optimizer's preconditioner:

        w <- AdamW_step(w, g_clean) - lr * lambda * (g_adv - g_clean)

    Same idea as decoupled weight decay in AdamW. It matters: feeding the
    sharpness term through Adam's second-moment estimate makes the effective
    weight depend on gradient history rather than on gamma.

    Cost: 2 forward-backward passes per step, same as SAM.
    """

    def __init__(self, params, base_optimizer=torch.optim.AdamW,
                 rho: float = 0.05, gamma=None, lam=None,
                 adaptive: bool = False, decouple: bool = True, **base_kwargs):
        if gamma is None and lam is None:
            gamma = 0.5
        if gamma is None:
            if lam < 0:
                raise ValueError("lam must be non-negative")
            gamma = lam / (1.0 + lam)
        if not (0.0 <= gamma < 1.0):
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        super().__init__(params, base_optimizer,
                         dict(rho=rho, gamma=gamma, adaptive=adaptive),
                         base_kwargs)
        self.decouple = bool(decouple)

    @property
    def lam(self) -> float:
        g = self.param_groups[0]["gamma"]
        return g / (1.0 - g)

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        params = self.all_params()

        self.zero_grad(set_to_none=True)
        loss = closure()
        g_clean = _clone_grads(params)

        for group in self.param_groups:
            self._perturb(group, group["rho"], group["adaptive"])
        self.zero_grad(set_to_none=True)
        closure()
        self._restore()
        g_adv = _clone_grads(params)

        if self.decouple:
            with torch.no_grad():
                for p, gc in zip(params, g_clean):
                    if gc is not None:
                        p.grad = gc.clone()
            self.base_optimizer.step()
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    lam = group["gamma"] / (1.0 - group["gamma"])
                    lr = group["lr"]
                    for _ in group["params"]:
                        p, gc, ga = params[i], g_clean[i], g_adv[i]
                        i += 1
                        if gc is None or ga is None:
                            continue
                        p.add_(ga - gc, alpha=-lr * lam)
        else:
            with torch.no_grad():
                i = 0
                for group in self.param_groups:
                    g = group["gamma"]
                    c_adv = g / (1.0 - g)
                    c_cln = (1.0 - 2.0 * g) / (1.0 - g)
                    for _ in group["params"]:
                        p, gc, ga = params[i], g_clean[i], g_adv[i]
                        i += 1
                        if gc is None or ga is None:
                            continue
                        p.grad = c_adv * ga + c_cln * gc
            self.base_optimizer.step()

        self.zero_grad(set_to_none=True)
        return loss


# --------------------------------------------------------------------------- #
class GAM(_Wrapper):
    r"""
    Gradient-norm Aware Minimization (Zhang et al., CVPR 2023).

        min_w  L(w) + alpha * R(w),
        R(w) = rho * max_{||e||<=rho} ||grad L(w+e)||_2

    grad R needs a Hessian-vector product, approximated by finite differences,
    hence four forward-backward passes per step:

        g0 = grad L(w)
        g1 = grad L(w + r * g0/||g0||)                r = grad_norm_rho
        d  = g1 - g0                                  ~ ascent dir of ||grad||
        g2 = grad L(w + grad_rho * d/||d||)
        g3 = grad L(w' + r * g2/||g2||)

        grad <- b0*g0 + gamma*(b1*g2 + b2*g1 + b3*g3)

    `grad_beta_0..3` are the published coefficients that DEFINE the method's
    gradient combination. They are not tuning knobs -- sweeping them means you
    are no longer comparing against GAM. Leave them at (1, 1, -1, 1).

    IMPLEMENTATION CAVEAT: this follows the published algorithm, but has not
    been diffed line-by-line against the authors' released code. If you have
    access to a reference implementation, verify against it before publishing
    GAM numbers.
    """

    def __init__(self, params, base_optimizer=torch.optim.AdamW,
                 grad_rho: float = 0.04, grad_norm_rho: float = 0.2,
                 grad_beta_0: float = 1.0, grad_beta_1: float = 1.0,
                 grad_beta_2: float = -1.0, grad_beta_3: float = 1.0,
                 grad_gamma: float = 0.03, adaptive: bool = True, **base_kwargs):
        super().__init__(params, base_optimizer,
                         dict(grad_rho=grad_rho, grad_norm_rho=grad_norm_rho,
                              grad_beta_0=grad_beta_0, grad_beta_1=grad_beta_1,
                              grad_beta_2=grad_beta_2, grad_beta_3=grad_beta_3,
                              grad_gamma=grad_gamma, adaptive=adaptive),
                         base_kwargs)

    @torch.no_grad()
    def _move(self, params, direction, radius, key):
        dn = _list_norm(direction)
        if dn is None:
            return
        scale = radius / (dn + 1e-12)
        for p, d in zip(params, direction):
            if d is None:
                continue
            e = d * scale
            p.add_(e)
            self.state[p][key] = e

    @torch.no_grad()
    def _undo(self, params, key):
        for p in params:
            e = self.state[p].pop(key, None)
            if e is not None:
                p.sub_(e)

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        params = self.all_params()
        g = self.param_groups[0]

        self.zero_grad(set_to_none=True)
        loss = closure()
        g0 = _clone_grads(params)

        self._move(params, g0, g["grad_norm_rho"], "e_gn")
        self.zero_grad(set_to_none=True)
        closure()
        g1 = _clone_grads(params)
        self._undo(params, "e_gn")

        d = [None if (a is None or b is None) else (b - a) for a, b in zip(g0, g1)]

        self._move(params, d, g["grad_rho"], "e_adv")
        self.zero_grad(set_to_none=True)
        closure()
        g2 = _clone_grads(params)

        self._move(params, g2, g["grad_norm_rho"], "e_gn2")
        self.zero_grad(set_to_none=True)
        closure()
        g3 = _clone_grads(params)
        self._undo(params, "e_gn2")
        self._undo(params, "e_adv")

        b0, b1, b2, b3 = (g["grad_beta_0"], g["grad_beta_1"],
                          g["grad_beta_2"], g["grad_beta_3"])
        gamma = g["grad_gamma"]
        with torch.no_grad():
            for p, a0, a1, a2, a3 in zip(params, g0, g1, g2, g3):
                if a0 is None:
                    continue
                p.grad = b0 * a0 + gamma * (b1 * a2 + b2 * a1 + b3 * a3)

        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)
        return loss


# --------------------------------------------------------------------------- #
VALID_REGULARIZERS = frozenset(PASSES_PER_STEP)


# Keys that look meaningful but are NOT read by any optimizer here. Silently
# ignoring them is how a config ends up claiming a schedule it never ran --
# e.g. a GSAM yaml carrying BOTH `rho: 0.05` and `rho_min/rho_max: 0.5/2.0`,
# where only one can possibly apply. Fail loudly instead.
DEAD_KEYS = {
    "rho_min": "GSAM rho scheduling is not implemented; use a constant `rho`",
    "rho_max": "GSAM rho scheduling is not implemented; use a constant `rho`",
    "grad_rho_min": "GAM radius scheduling is not implemented; use `grad_rho`",
    "grad_rho_max": "GAM radius scheduling is not implemented; use `grad_rho`",
    "grad_norm_rho_min": "not implemented; use `grad_norm_rho`",
    "grad_norm_rho_max": "not implemented; use `grad_norm_rho`",
    "momentum": "AdamW has no momentum arg; use `betas`",
    "optzee_wd": "not read by this codebase; use `weight_decay`",
    "SAMonOptimizer": "not read; the wrapper always steps the base optimizer",
}


def check_dead_keys(hp: dict, strict: bool = True):
    """Raise on config keys that would be silently ignored."""
    found = [(k, why) for k, why in DEAD_KEYS.items() if hp.get(k) is not None]
    if not found:
        return
    msg = "config contains keys that this code does NOT read:\n" + "\n".join(
        f"    {k!r}: {why}" for k, why in found)
    if strict:
        raise ValueError(msg + "\n  Remove them, or the config is claiming "
                               "behaviour that never runs.")
    import warnings
    warnings.warn(msg)


def build_optimizer(name: str, params, hp: dict):
    """
    Factory. Raises loudly on an unknown name -- a typo must never silently
    fall through to the unregularized branch.
    """
    check_dead_keys(hp)
    key = str(name).strip().lower()
    if key not in VALID_REGULARIZERS:
        raise ValueError(
            f"unknown regularizer {name!r}; expected one of "
            f"{sorted(VALID_REGULARIZERS)}"
        )

    base = dict(lr=hp["lr"], weight_decay=hp.get("weight_decay", 0.01),
                betas=tuple(hp.get("betas", (0.9, 0.999))),
                eps=hp.get("eps", 1e-8))

    if key == "none":
        return NoReg(params, **base)
    if key == "sam":
        return SAM(params, rho=hp["rho"], adaptive=hp.get("adaptive", False),
                   m=hp.get("m_sharpness"), **base)
    if key == "gsam":
        return GSAM(params, rho=hp["rho"], alpha=hp["alpha"],
                    adaptive=hp.get("adaptive", False), **base)
    if key == "wsam":
        return WSAM(params, rho=hp["rho"], gamma=hp.get("gamma"),
                    lam=hp.get("lam"), adaptive=hp.get("adaptive", False),
                    decouple=(True if hp.get("decouple") is None
                              else hp["decouple"]), **base)
    return GAM(params, grad_rho=hp["grad_rho"], grad_norm_rho=hp["grad_norm_rho"],
               grad_beta_0=hp.get("grad_beta_0", 1.0),
               grad_beta_1=hp.get("grad_beta_1", 1.0),
               grad_beta_2=hp.get("grad_beta_2", -1.0),
               grad_beta_3=hp.get("grad_beta_3", 1.0),
               grad_gamma=hp.get("grad_gamma", 0.03),
               adaptive=hp.get("adaptive", True), **base)
