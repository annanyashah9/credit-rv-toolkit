"""Unit tests for walk-forward windows and the pooled fair-value models."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.windows import walk_forward_windows
from crv.fairvalue.models import GBMFairValue, RidgeFairValue


def _dates(n):
    return pd.date_range("2010-01-31", periods=n, freq="ME")


def test_windows_train_strictly_before_predict():
    wins = walk_forward_windows(_dates(60), train_window_months=24, min_train_months=12,
                                refit_every_months=3, scheme="rolling")
    assert wins
    for train, predict in wins:
        assert train.max() < predict.min()          # leakage guarantee
        assert len(train) <= 24                       # rolling window bound
        assert len(train) >= 12                       # min train enforced


def test_windows_refit_cadence_and_coverage():
    wins = walk_forward_windows(_dates(40), 24, 12, 3, "rolling")
    # predict blocks are contiguous, length <= refit cadence, and disjoint.
    preds = [p for _, p in wins]
    for p in preds:
        assert len(p) <= 3
    flat = pd.DatetimeIndex(np.concatenate([p.values for p in preds]))
    assert flat.is_unique


def test_expanding_grows_train():
    wins = walk_forward_windows(_dates(40), train_window_months=24, min_train_months=12,
                                refit_every_months=6, scheme="expanding")
    lengths = [len(t) for t, _ in wins]
    assert lengths == sorted(lengths) and lengths[0] >= 12


def test_gbm_beats_ridge_on_nonlinear_truth():
    # y is a clearly nonlinear (sinusoidal) function of one feature; GBM should fit it
    # much better than a linear Ridge in-sample -- sanity that the ML arm has capacity.
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(800, 1))
    y = np.sin(X[:, 0]) + 0.05 * rng.normal(size=800)
    ridge = RidgeFairValue(alpha=1.0)
    ridge.fit(X, y)
    gbm = GBMFairValue(max_depth=3, max_iter=200, learning_rate=0.1)
    gbm.fit(X, y)
    rmse_r = np.sqrt(np.mean((ridge.predict(X) - y) ** 2))
    rmse_g = np.sqrt(np.mean((gbm.predict(X) - y) ** 2))
    assert rmse_g < rmse_r * 0.5
