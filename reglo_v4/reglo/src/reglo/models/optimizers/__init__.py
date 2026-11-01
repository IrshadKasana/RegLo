"""What trains the optimizee.

    AdamW            conventional optimizer — M1, M2                    (adamw.py)
    LSTMOptimizer    learned optimizer F_φ — M3, M4                     (lstm_optimizer.py)
    GradientFeatures what F_φ sees per coordinate                       (features.py)
    FrozenLSTMOptimizer  a meta-trained F_φ used as a torch optimizer   (frozen_lstm.py)
"""

from reglo.models.optimizers.adamw import build_adamw, build_scheduler
from reglo.models.optimizers.features import FEATURES, NORMALIZATIONS, GradientFeatures
from reglo.models.optimizers.frozen_lstm import FrozenLSTMOptimizer
from reglo.models.optimizers.lstm_optimizer import (
    LSTMOptimizer,
    build_lstm_optimizer,
    load_learned_optimizer,
    save_learned_optimizer,
)
from reglo.registry import OPTIMIZERS

OPTIMIZERS.register("adamw")(build_adamw)
OPTIMIZERS.register("lstm")(build_lstm_optimizer)

__all__ = [
    "FEATURES",
    "NORMALIZATIONS",
    "FrozenLSTMOptimizer",
    "GradientFeatures",
    "LSTMOptimizer",
    "build_adamw",
    "build_lstm_optimizer",
    "build_scheduler",
    "load_learned_optimizer",
    "save_learned_optimizer",
]
