"""Hierarchical meta-training / optimizer-evaluation / optimizee-evaluation pipeline.

Extends the base DMO code (``learning_regularizers/dmo``) without modifying it. The base
``regularizers`` package (SAM, GAM, GSAM, schedulers) uses flat imports, so the DMO source
directory is put on ``sys.path`` here and those classes are reused as-is.
"""
from __future__ import annotations

import sys
from pathlib import Path

DMO_DIR = Path(__file__).resolve().parents[1] / "dmo"
if str(DMO_DIR) not in sys.path:
    sys.path.append(str(DMO_DIR))

__all__ = ["DMO_DIR"]
