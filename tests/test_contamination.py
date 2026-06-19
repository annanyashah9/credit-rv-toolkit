"""Unit tests for Phase 3b: forward real returns, no-trade bands, contamination."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.contamination import default_odds_by_quantile
from crv.backtest.defaults import DEFAULTED
from crv.backtest.portfolio import banded_ls_weights, long_short_weights
from crv.backtest.returns import forward_excess_return
from crv.config import Config


def _r1(cusips, dates, val=0.01):
    rows = [{"cusip": c, "rebalance_date": d, "r1": val} for c in cusips for d in dates]
    return pd.DataFrame(rows)


def test_forward_excess_return_sums_h_months_leakage_free():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    r1 = _r1(["A"], dates, 0.01)
    fwd = forward_excess_return(r1, horizon=3).set_index("rebalance_date")
    # First date: sum of 3 future months = 0.03; only dates with a full window survive.
    assert np.isclose(fwd.loc[dates[0], "r_fwd"], 0.03)
    assert len(fwd) == 2  # only first two formation dates have 3 forward months


def test_forward_excess_return_drops_truncated_window():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    fwd = forward_excess_return(_r1(["A"], dates), horizon=3)
    assert fwd.empty  # no date has 3 forward months


def test_band_reduces_turnover_vs_rerank():
    # A name oscillating around the quintile boundary churns under plain ranking but
    # is held under the no-trade band.
    n = 20
    idx = [f"B{i}" for i in range(n)]
    zn = pd.Series(np.linspace(-1, 1, n), index=idx)
    # Unbanded: top quintile = top 4 names.
    w_plain = long_short_weights(zn, 5)
    # Banded entering from empty: same top quintile initially.
    w_band, lng, sht = banded_ls_weights(zn, 5, enter_q=4, exit_q=3, prev_long=set(),
                                         prev_short=set())
    assert (w_plain[w_plain > 0].index == w_band[w_band > 0].index).all()
    # Next period a top name slips one quintile: band still holds it, plain drops it.
    zn2 = zn.copy()
    top_name = zn.index[-1]
    zn2[top_name] = zn.quantile(0.65)  # now in quintile 3, not 4
    w_band2, _, _ = banded_ls_weights(zn2, 5, 4, 3, lng, sht)
    assert top_name in w_band2[w_band2 > 0].index  # held by hysteresis


def test_default_odds_higher_for_cheap_when_constructed():
    dates = pd.date_range("2020-01-31", periods=1, freq="ME")
    n = 100
    z = np.linspace(-2, 2, n)  # high z = cheap
    sig = pd.DataFrame({"cusip": [f"B{i}" for i in range(n)],
                        "rebalance_date": dates[0], "z": z})
    # Make the cheapest 20 default soon after.
    cheap = sig.sort_values("z").tail(20)["cusip"]
    exits = pd.DataFrame({"cusip": sig["cusip"],
                          "exit_type": np.where(sig["cusip"].isin(cheap), DEFAULTED, "ACTIVE"),
                          "default_date": np.where(sig["cusip"].isin(cheap),
                                                   dates[0] + pd.Timedelta(days=30), pd.NaT)})
    exits["default_date"] = pd.to_datetime(exits["default_date"])
    cfg = Config()
    odds = default_odds_by_quantile(sig, exits, cfg)
    assert odds["default_rate"].iloc[-1] > odds["default_rate"].iloc[0]  # cheapest default more
