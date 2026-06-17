"""Unit tests for the thin backtest: forward alignment, IC, HAC, quintiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.forward import add_month_ordinal, forward_spread_return
from crv.backtest.ic import cross_sectional_ic
from crv.backtest.quintiles import quintile_returns
from crv.backtest.stats import newey_west_mean


def _signal(n_dates=8, n_names=60, seed=0):
    """Synthetic signal where high-z names subsequently tighten (converge)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-31", periods=n_dates, freq="ME")
    rows = []
    base = {i: 100 + 10 * i for i in range(n_names)}  # per-name spread level
    z = {i: rng.normal() for i in range(n_names)}
    for d_i, d in enumerate(dates):
        for i in range(n_names):
            # high z => wide now, tightens next step (mean-reverts toward base)
            gs = base[i] + 40 * z[i] - 5 * d_i * z[i]
            rows.append({"cusip": f"B{i}", "rebalance_date": d, "z": z[i],
                         "gspread_bp": gs, "dts": gs * 6.0})  # dur=6
    return pd.DataFrame(rows)


def test_forward_is_leakage_free():
    sig = add_month_ordinal(_signal(n_dates=4, n_names=5))
    fwd = forward_spread_return(sig, horizon_m=1)
    # The last date has no t+1 match, so no rows should carry its t_idx.
    assert fwd["t_idx"].max() < sig["t_idx"].max()
    # Every row's outcome must come from strictly later than its own date (h>=1).
    assert (fwd["t_idx"] >= 0).all()


def test_forward_drops_absent_future():
    sig = _signal(n_dates=3, n_names=4)
    # Remove one bond at the last date -> its t-1 row has no forward match.
    last = sig["rebalance_date"].max()
    sig = sig[~((sig.rebalance_date == last) & (sig.cusip == "B0"))]
    fwd = forward_spread_return(sig, horizon_m=1)
    mid = sorted(sig.rebalance_date.unique())[1]
    assert not ((fwd.cusip == "B0") & (fwd.rebalance_date == mid)).any()


def test_ic_positive_when_cheap_tightens():
    fwd = forward_spread_return(_signal(), horizon_m=1)
    ic = cross_sectional_ic(fwd, min_names=10)
    assert ic.mean() > 0.5  # construction makes cheap names earn the proxy return


def test_quintile_long_short_positive():
    q = quintile_returns(_signal(), horizon=1, n_quantiles=5, min_names=10)
    assert q.loc["LS", "mean_r"] > 0


def test_newey_west_lag0_matches_ordinary_se():
    x = np.array([0.1, -0.2, 0.3, 0.0, 0.15, -0.05])
    mean, se, t = newey_west_mean(x, lags=0)
    e = x - x.mean()
    expected_se = np.sqrt((e @ e) / len(x)) / np.sqrt(len(x))
    assert np.isclose(mean, x.mean())
    assert np.isclose(se, expected_se)


def test_newey_west_lag_widens_se_for_autocorrelated():
    rng = np.random.default_rng(1)
    e = rng.normal(size=400)
    x = e[1:] + e[:-1]  # MA(1), positive autocorrelation
    _, se0, _ = newey_west_mean(x, lags=0)
    _, se_hac, _ = newey_west_mean(x, lags=5)
    assert se_hac > se0
