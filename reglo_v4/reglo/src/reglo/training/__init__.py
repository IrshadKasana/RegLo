"""Stages 2–3: training one optimizee with M1, M2, M3 or M4.

    methods.py    M1–M4: one training step each (AdamW ± regularizer, or the frozen LSTM)
    trainer.py    the epoch loop, evaluation, checkpoints and final.json
    evaluate.py   loss and top-k accuracy over a batch stream
"""

from reglo.training.evaluate import evaluate
from reglo.training.methods import CheckpointMismatch, Method, StepResult, build_method
from reglo.training.trainer import OptimizeeTrainer

__all__ = ["CheckpointMismatch", "Method", "OptimizeeTrainer", "StepResult", "build_method", "evaluate"]
