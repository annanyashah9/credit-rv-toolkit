"""Spread transform for the fair-value model.

G-spread is right-skewed with occasional negatives (bonds richer than Treasuries), so
plain log is unusable. asinh(x/s) = log(x/s + sqrt((x/s)^2 + 1)) is ~linear near zero,
logarithmic in the tails, and defined for negatives — it tames the skew the Phase-1
residual exhibited while preserving sign.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_model_space(spread_bp: pd.Series | np.ndarray, scale: float) -> np.ndarray:
    return np.arcsinh(np.asarray(spread_bp, dtype=float) / scale)


def from_model_space(y: pd.Series | np.ndarray, scale: float) -> np.ndarray:
    return np.sinh(np.asarray(y, dtype=float)) * scale
