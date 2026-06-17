"""Naive cross-sectional OLS fair-value model (Phase 1).

Fits G-spread on term + sector WITHIN a single date's cross-section. Point-in-time by
construction: each date's fit uses only that date's bonds, so there is no look-ahead
in the fair value or residual. (Walk-forward across dates is a Phase-3 concern for the
backtest; the signal itself is contemporaneous.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.fairvalue.features import build_design_matrix


def fit_predict_fair_spread(df: pd.DataFrame, y_col: str = "gspread_bp") -> pd.Series:
    """OLS-predicted fair spread (bps) for one cross-section.

    Solves least squares on the design matrix; returns fitted values aligned to df.
    Rows with NaN in X or y are excluded from the fit but get a prediction where X is
    complete (NaN otherwise).
    """
    X = build_design_matrix(df)
    y = pd.to_numeric(df[y_col], errors="coerce")

    fit_mask = X.notna().all(axis=1) & y.notna()
    if fit_mask.sum() < X.shape[1] + 1:
        return pd.Series(np.nan, index=df.index, name="fair_bp")

    Xm = X[fit_mask].to_numpy()
    ym = y[fit_mask].to_numpy()
    beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)

    pred_mask = X.notna().all(axis=1)
    fair = pd.Series(np.nan, index=df.index, name="fair_bp")
    fair[pred_mask] = X[pred_mask].to_numpy() @ beta
    return fair
