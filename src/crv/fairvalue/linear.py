"""Cross-sectional linear fair-value models.

Both fit WITHIN a single date's cross-section, so they are point-in-time by
construction (no temporal look-ahead). 'naive' (Phase 1) = OLS on term + sector in bp
space. 'peer_shrunk' (Phase 2a) = OLS on term + sector + liquidity in asinh space,
plus an empirical-Bayes issuer adjustment shrinking each issuer toward its sector peers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.fairvalue.features import build_design_matrix
from crv.fairvalue.shrink import eb_issuer_shrinkage


def _ols_fitted(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """OLS fitted values aligned to X.index; NaN where X/y incomplete or rank-deficient."""
    fit_mask = X.notna().all(axis=1) & y.notna()
    fitted = pd.Series(np.nan, index=X.index)
    if fit_mask.sum() < X.shape[1] + 1:
        return fitted
    beta, *_ = np.linalg.lstsq(X[fit_mask].to_numpy(), y[fit_mask].to_numpy(), rcond=None)
    pred_mask = X.notna().all(axis=1)
    fitted[pred_mask] = X[pred_mask].to_numpy() @ beta
    return fitted


def fit_predict_fair_spread(df: pd.DataFrame, y_col: str = "gspread_bp") -> pd.Series:
    """Naive OLS fair spread (bp) for one cross-section (Phase 1)."""
    X = build_design_matrix(df, feature_set="naive")
    y = pd.to_numeric(df[y_col], errors="coerce")
    return _ols_fitted(X, y).rename("fair_bp")


def peer_shrunk_fair_spread(
    df: pd.DataFrame, y_col: str = "y_model", issuer_col: str = "issuer",
    k: float | None = None,
) -> pd.Series:
    """Peer-shrunk fair value (in the model/asinh space) for one cross-section.

    Stage 1: OLS of y on term + sector + liquidity (the sector/peer curve).
    Stage 2: empirical-Bayes issuer adjustment on the stage-1 residual.
    fair = base_fit + shrunk_issuer_effect.
    """
    X = build_design_matrix(df, feature_set="full")
    y = pd.to_numeric(df[y_col], errors="coerce")
    base = _ols_fitted(X, y)

    resid = y - base
    valid = resid.notna()
    shrunk = pd.Series(0.0, index=df.index)
    if valid.any():
        shrunk.loc[valid] = eb_issuer_shrinkage(resid[valid], df.loc[valid, issuer_col], k=k)
    return (base + shrunk).rename("fair_model")
