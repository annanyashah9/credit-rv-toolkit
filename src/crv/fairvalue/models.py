"""Pooled fair-value models behind a shared interface, for the walk-forward
ML-vs-linear comparison. Both consume the SAME design matrix and asinh target, so any
performance gap is functional form, not features.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


class FairValueModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


class RidgeFairValue:
    """L2-penalized linear baseline (the matched linear arm)."""

    def __init__(self, alpha: float = 1.0):
        self._m = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)


class GBMFairValue:
    """Gradient-boosted trees (the ML arm)."""

    def __init__(self, **params):
        self._m = HistGradientBoostingRegressor(**params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._m.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._m.predict(X)


def make_model_factory(kind: str, cfg):
    """Return a zero-arg factory that builds a fresh model per refit."""
    if kind == "ridge_wf":
        alpha = cfg.model.ridge_alpha
        return lambda: RidgeFairValue(alpha=alpha)
    if kind == "gbm_wf":
        params = dict(cfg.model.gbm)
        return lambda: GBMFairValue(**params)
    raise ValueError(f"not a walk-forward model kind: {kind!r}")
