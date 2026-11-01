"""Measurements of trained optimizees (Stage 4) and the per-run numbers in final.json.

    convergence.py   the paper's "at convergence" point
    norms.py         ‖θ‖₁, ‖θ‖₂, layerwise norms, sparsity, |w| histograms     (Exp 4.2)
    hessian.py       λ_max by power iteration, Tr(H) by Hutchinson             (Exp 4.1)
    sharpness.py     the paper's point-of-convergence tests in ℓ∞ balls
    flops.py         FLOPs per step and GPU-synchronised timing
    audit.py         all of the above for one finished run → diagnostics.json
    compute.py       all 12 variants timed back-to-back on one GPU            (Stage 4)
"""

from reglo.diagnostics.convergence import ConvergenceTracker
from reglo.diagnostics.flops import count_flops, lstm_flops_per_coordinate, synchronize, time_steps
from reglo.diagnostics.hessian import hessian_trace, hvp, loss_terms, top_eigenvalue
from reglo.diagnostics.norms import weight_statistics
from reglo.diagnostics.sharpness import max_grad_norm_in_ball, max_loss_in_ball

__all__ = [
    "ConvergenceTracker",
    "count_flops",
    "hessian_trace",
    "hvp",
    "loss_terms",
    "lstm_flops_per_coordinate",
    "max_grad_norm_in_ball",
    "max_loss_in_ball",
    "synchronize",
    "time_steps",
    "top_eigenvalue",
    "weight_statistics",
]
