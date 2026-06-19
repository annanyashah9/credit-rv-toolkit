"""Assemble the liquidity-control feature table on the (cusip, rebalance_date) grid.

Combines the daily-derived measures (Bao gamma, Amihud) with the membership-derived
ones (trade_freq, age, issue_size). Features are winsorized and asinh-scaled so they
behave as well-conditioned regressors in the fair-value model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.config import Config
from crv.liquidity.illiquidity import daily_illiquidity, sample_asof

LIQ_FEATURES = ["bao_gamma", "amihud", "trade_freq", "log_age", "log_issue_size"]


def _winsorize(s: pd.Series, pct: float) -> pd.Series:
    lo, hi = s.quantile(pct), s.quantile(1 - pct)
    return s.clip(lo, hi)


def build_liquidity_features(
    panel: pd.DataFrame, universe: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Return [cusip, rebalance_date, <LIQ_FEATURES>] aligned to universe membership.

    bao_gamma/amihud come from the daily panel (trailing window, as-of sampled);
    trade_freq/age/issue_size are reused from the membership table.
    """
    lc = cfg.liquidity
    daily = daily_illiquidity(panel, window=lc.window_days, min_obs=lc.min_obs)
    asof = sample_asof(daily, universe["rebalance_date"], universe["cusip"])

    feat = universe.merge(asof, on=["cusip", "rebalance_date"], how="left")
    feat["log_age"] = np.log1p(feat["age"].clip(lower=0))
    feat["log_issue_size"] = np.log(feat["issue_size"].clip(lower=1))

    # Winsorize the noisy raw measures, then asinh to compress heavy tails.
    for col in ("bao_gamma", "amihud"):
        feat[col] = np.arcsinh(_winsorize(feat[col], lc.winsor_pct))

    keep = ["cusip", "rebalance_date", *LIQ_FEATURES]
    return feat[keep]
