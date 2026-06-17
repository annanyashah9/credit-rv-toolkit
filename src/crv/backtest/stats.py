"""HAC (Newey-West) statistics for overlapping-window time series.

The IC time series from overlapping forward windows is autocorrelated, so the naive
SE of its mean understates uncertainty. We use a Newey-West HAC estimator of the
variance of the sample mean (equivalent to regressing the series on a constant with
HAC errors). Default lag = horizon-1 (the overlap length).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def newey_west_mean(series: pd.Series | np.ndarray, lags: int) -> tuple[float, float, float]:
    """Return (mean, hac_se, t_stat) for the mean of `series`.

    lags=0 reduces to the ordinary SE of the mean. Uses Bartlett weights.
    """
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(x)
    if n < 2:
        return (float(np.mean(x)) if n else float("nan"), float("nan"), float("nan"))

    mean = float(x.mean())
    e = x - mean
    lags = max(0, min(int(lags), n - 1))

    # Long-run variance of e via Bartlett-weighted autocovariances.
    gamma0 = float(e @ e) / n
    lrv = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1)
        gamma_k = float(e[k:] @ e[:-k]) / n
        lrv += 2.0 * w * gamma_k

    lrv = max(lrv, 0.0)                 # guard tiny negatives from finite samples
    var_mean = lrv / n
    se = float(np.sqrt(var_mean)) if var_mean > 0 else float("nan")
    tstat = mean / se if se and np.isfinite(se) and se > 0 else float("nan")
    return mean, se, tstat
