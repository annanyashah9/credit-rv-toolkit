"""Unit tests for liquidity measures: Bao gamma sign, Amihud monotonicity, as-of join."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.liquidity.illiquidity import daily_illiquidity, sample_asof


def _bouncing_panel(n=300, seed=0):
    """One cusip with a bid-ask 'bounce' (alternating price reversals) -> illiquid,
    so Bao gamma should be positive."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    mid = 100 + np.cumsum(rng.normal(scale=0.05, size=n))
    bounce = 0.5 * ((-1.0) ** np.arange(n))      # strong transitory reversals
    price = mid + bounce
    return pd.DataFrame({
        "cusip": "X", "date": dates, "price": price,
        "dvolume": rng.uniform(1e6, 2e6, n),
    })


def test_bao_gamma_positive_for_bouncing_price():
    df = _bouncing_panel()
    out = daily_illiquidity(df, window=60, min_obs=30)
    assert out["bao_gamma"].dropna().mean() > 0


def test_amihud_rises_when_volume_falls():
    df = _bouncing_panel()
    low_vol = df.copy()
    low_vol["dvolume"] = df["dvolume"] / 100.0
    a_hi = daily_illiquidity(df, 60, 30)["amihud"].dropna().mean()
    a_lo = daily_illiquidity(low_vol, 60, 30)["amihud"].dropna().mean()
    assert a_lo > a_hi


def test_sample_asof_is_backward_only():
    df = _bouncing_panel(n=120)
    daily = daily_illiquidity(df, 30, 20)
    # Ask for a rebalance date well inside the series; the matched measure must come
    # from a date <= the rebalance date.
    rebal = pd.Timestamp("2019-04-01")
    out = sample_asof(daily, [rebal], pd.Series(["X"]))
    assert len(out) == 1
    # A rebalance date before any history yields NaN (nothing to look back to).
    early = sample_asof(daily, [pd.Timestamp("2018-01-01")], pd.Series(["X"]))
    assert early["bao_gamma"].isna().all()
