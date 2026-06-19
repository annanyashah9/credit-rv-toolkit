"""Empirical-Bayes shrinkage of issuer effects (the peer-shrinkage core).

After the base cross-sectional regression (term + sector + liquidity), each issuer has
a mean residual m_j over its n_j bonds. We shrink it toward 0 (which, since residuals
are already net of the sector curve, means toward the sector peer level):

    m_j_shrunk = m_j * n_j / (n_j + k)

k = within-issuer variance / between-issuer variance (James-Stein / random-effects
form). Large, consistent issuers keep their curve; single-bond issuers collapse to the
sector. Deterministic, closed-form — the "ridge 2-stage" decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_k(resid: pd.Series, issuer: pd.Series) -> float:
    """Empirical-Bayes k = within-var / between-var, estimated from this cross-section.

    Falls back to a large k (=> heavy shrinkage) if between-issuer variance is ~0 or
    undefined (e.g. every issuer has one bond)."""
    g = resid.groupby(issuer)
    counts = g.size()
    means = g.mean()
    within = g.var(ddof=1)
    within_var = float(np.nanmean(within[counts > 1])) if (counts > 1).any() else np.nan
    between_var = float(means.var(ddof=1)) if len(means) > 1 else np.nan
    if not np.isfinite(between_var) or between_var <= 0:
        return 1e6
    if not np.isfinite(within_var):
        within_var = float(resid.var(ddof=1))
    return max(within_var / between_var, 0.0)


def eb_issuer_shrinkage(
    resid: pd.Series, issuer: pd.Series, k: float | None = None
) -> pd.Series:
    """Return the shrunk issuer effect per row (aligned to resid.index)."""
    if k is None:
        k = estimate_k(resid, issuer)
    g = resid.groupby(issuer)
    m = g.transform("mean")
    n = g.transform("size")
    return m * (n / (n + k))
