"""Unit tests for the naive fair-value residual and universe trade-frequency logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.fairvalue.linear import fit_predict_fair_spread
from crv.fairvalue.residual import standardized_residual


def test_residual_centered_and_cheap_is_positive():
    # A clean cross-section where one name is clearly cheap (wide vs fair).
    resid = pd.Series([-10.0, -5.0, 0.0, 5.0, 10.0, 80.0])
    observed = pd.Series([100, 105, 110, 115, 120, 200], dtype=float)
    fair = observed - resid
    out = standardized_residual(observed, fair, robust=True)
    assert np.isclose(out["z"].median(), 0.0, atol=1e-9)   # median-centered
    assert out["z"].iloc[-1] > 0                            # the cheap name scores positive
    assert out["resid_bp"].iloc[-1] == 80.0


def test_fair_value_recovers_linear_term_structure():
    # If gspread is an exact linear function of term within one sector, OLS fits it
    # and residuals are ~0.
    n = 50
    term = np.linspace(1, 10, n)
    gs = 50 + 12 * term  # perfectly linear
    df = pd.DataFrame({"ttm": term, "sector_ff30": "A", "gspread_bp": gs})
    fair = fit_predict_fair_spread(df)
    assert np.allclose(fair.to_numpy(), gs, atol=1e-6)


def test_fair_value_too_few_names_returns_nan():
    df = pd.DataFrame({"ttm": [1.0, 2.0], "sector_ff30": ["A", "B"], "gspread_bp": [10.0, 20.0]})
    fair = fit_predict_fair_spread(df)
    assert fair.isna().all()
