"""Stage 1 — meta-training the LSTM optimizer (Algorithm 1).

    unroll.py        lines 5–15: the T-step differentiable inner loop
    objective.py     lines 16–23: L_meta + λ_smooth·L_smooth + λ_reg·L_reg(θ_T)
    curriculum.py    trajectory-length curriculum (Chen et al. 2020)
    meta_trainer.py  line 24 and the outer loop: TBPTT, validation, checkpoints
"""

from reglo.meta_training.curriculum import Curriculum
from reglo.meta_training.meta_trainer import MetaTrainer
from reglo.meta_training.objective import MetaObjective, build_objective, loss_weights
from reglo.meta_training.unroll import InnerState, UnrollOutput, unroll

__all__ = [
    "Curriculum",
    "InnerState",
    "MetaObjective",
    "MetaTrainer",
    "UnrollOutput",
    "build_objective",
    "loss_weights",
    "unroll",
]
