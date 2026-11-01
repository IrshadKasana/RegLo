"""
flops_utils.py — single source of truth for per-step FLOPs accounting.

Used by BOTH optimizee_train.py (so every evaluation JSON carries FLOPs in
its efficiency dict — 'FLOPs in the experiment') and benchmark_compute.py
(the standalone timing table), so the two can never disagree.

Accounting model (per optimization step):

  optimizee cost  = (fwd_passes + 2 * bwd_passes) * fwd_flops(model, batch)
                    [the standard bwd ~= 2x fwd convention]
  LO extra cost   = n_params * per-coordinate LSTM cell cost
                    (2-layer LSTM, hidden H=20, input 2 after log-preproc,
                     +1 output head)  — the term the old table omitted
                    (Change.MD item 58).

Pass counts per method:
  adamw+sam / adamw+gsam : 2F + 2B          learned optimizer : 1F + 1B + LSTM
  adamw+gam              : 4F + 4B (upper bound; wall-clock is measured)
  plain adam/sgd         : 1F + 1B
"""
import torch

LSTM_LAYERS, LSTM_HIDDEN, LSTM_IN = 2, 20, 2


def lstm_flops_per_step(n_params, hidden=LSTM_HIDDEN, layers=LSTM_LAYERS,
                        n_in=LSTM_IN):
    per_coord = 8 * hidden * (n_in + hidden + 1)               # layer 1
    per_coord += (layers - 1) * 8 * hidden * (2 * hidden + 1)  # deeper layers
    per_coord += 2 * hidden                                    # output head
    return int(n_params) * per_coord


def optimizee_fwd_flops(model, batch):
    """fvcore if installed; analytic 2*B*params dense fallback; None on error."""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        with torch.no_grad():
            f = FlopCountAnalysis(model, batch)
            f.unsupported_ops_warnings(False)
            return int(f.total())
    except Exception:
        try:
            return int(2 * batch.shape[0] *
                       sum(p.numel() for p in model.parameters()))
        except Exception:
            return None


def passes_for_method(if_normal_optim, regularizer):
    """(fwd, bwd, uses_lstm) per optimization step."""
    if not if_normal_optim:
        return 1, 1, True
    r = (regularizer or "").lower()
    if "gam" in r and "gsam" not in r:
        return 4, 4, False
    if "sam" in r:                      # sam and gsam
        return 2, 2, False
    return 1, 1, False                  # plain adam / sgd


def step_flops(model, batch, if_normal_optim, regularizer):
    """Full per-step accounting dict for the efficiency payload."""
    fwd, bwd, lstm = passes_for_method(if_normal_optim, regularizer)
    f1 = optimizee_fwd_flops(model, batch)
    n_params = sum(p.numel() for p in model.parameters())
    opt = (fwd + 2 * bwd) * f1 if f1 else None
    lo = lstm_flops_per_step(n_params) if lstm else 0
    return {
        "passes_per_step": f"{fwd}F+{bwd}B" + ("+LSTM" if lstm else ""),
        "optimizee_fwd_flops": f1,
        "optimizee_flops_per_step": opt,
        "lstm_flops_per_step": lo,
        "total_flops_per_step": (opt + lo) if opt is not None else None,
        "n_params": int(n_params),
    }


# ======================================================================
# META-TRAINING cost (paper metric: "FLOPs — Meta-train FLOPs & Total").
#
# The LO pays a large ONE-TIME upfront cost that the AdamW baselines do
# not; reporting only meta-test FLOPs would flatter the LO. Accounting:
#
#   meta_update() is invoked exactly ONCE per optimizee step inside the
#   unroll — in BOTH passes of the penalty objective (the unroll under
#   phi AND the same-batch re-unroll under phi_adv), so simply counting
#   those calls captures the doubled cost of Alg. 1 automatically.
#
#   per meta_update  = 1 optimizee fwd + 1 optimizee bwd (= 2x fwd)
#                      + 1 coordinate-wise LSTM forward
#   whole-graph BPTT = BPTT_BWD_FACTOR x (accumulated forward cost)
#
# The counter is a MEASURED count; the per-pass costs are analytic (same
# math as step_flops), and the run's meta_updates count is persisted
# across walltime resumes so a split run reports the true total.
# ======================================================================
COUNTERS = {"meta_updates": 0, "eval_updates": 0}
BPTT_BWD_FACTOR = 2.0          # backward through the unrolled graph ~= 2x fwd

DATASET_SHAPES = {"mnist": (1, 28, 28), "fmnist": (1, 28, 28),
                  "cifar10": (3, 32, 32), "cifar100": (3, 32, 32),
                  "pacs": (3, 224, 224)}   # exp6 (added 2026-07-31)


def bump_meta_update(n=1, bptt=True):
    """bptt=True  -> unroll step that participates in the meta-backward.
    bptt=False -> in-training validation unroll (no_grad): forward only,
    so it must NOT be charged the BPTT backward factor."""
    COUNTERS["meta_updates" if bptt else "eval_updates"] += n


def reset_counters(meta_updates=0, eval_updates=0):
    COUNTERS["meta_updates"] = int(meta_updates or 0)
    COUNTERS["eval_updates"] = int(eval_updates or 0)


def dummy_batch(dataset, batch_size, device="cpu"):
    shape = DATASET_SHAPES.get(str(dataset).lower(), (1, 28, 28))
    return torch.randn(int(batch_size), *shape, device=device)


def meta_train_flops(meta_updates, model, batch, eval_updates=0):
    """Total FLOPs spent meta-training an LO (BPTT unrolls charged the
    backward factor; in-training no_grad validation unrolls charged
    forward-only). Returns None if optimizee FLOPs can't be estimated.
    Baselines do no meta-training and report 0."""
    f1 = optimizee_fwd_flops(model, batch)
    if f1 is None:
        return None
    n_params = sum(p.numel() for p in model.parameters())
    per_update = 3 * f1 + lstm_flops_per_step(n_params)   # 1F + 2F(bwd) + LSTM
    train_part = int(meta_updates) * per_update * (1.0 + BPTT_BWD_FACTOR)
    eval_part = int(eval_updates) * per_update
    return {
        "meta_updates": int(meta_updates),
        "eval_updates": int(eval_updates),
        "flops_per_meta_update": int(per_update),
        "bptt_bwd_factor": BPTT_BWD_FACTOR,
        "meta_train_flops": int(train_part + eval_part),
    }
