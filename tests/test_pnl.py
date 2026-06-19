"""Unit tests for Phase 3a: excess returns, default override, costs, neutralization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.defaults import DEFAULTED, detect_exits
from crv.backtest.portfolio import long_short_weights, neutralize_signal
from crv.backtest.returns import monthly_excess_return
from crv.config import Config


def _cfg(**bt):
    c = Config()
    for k, v in bt.items():
        setattr(c.backtest, k, v)
    return c


def _spreads(cusips, dates, spread_bp, dur=5.0, price=100.0):
    rows = []
    for c in cusips:
        for d in dates:
            rows.append({"cusip": c, "rebalance_date": d, "gspread_bp": spread_bp,
                         "mod_duration": dur, "price": price})
    return pd.DataFrame(rows)


def test_excess_return_carry_and_spread_change():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    # Spread constant at 200bp, dur 5 -> r1 = 0.02/12 - 5*0 = 0.001667 carry only.
    sp = _spreads(["A"], dates, 200.0)
    exits = pd.DataFrame({"cusip": ["A"], "exit_type": ["ACTIVE"], "default_date": [pd.NaT]})
    r = monthly_excess_return(sp, exits, _cfg())
    assert np.isclose(r["r1"].iloc[0], 0.02 / 12, atol=1e-6)


def test_default_override_realizes_recovery_loss():
    dates = pd.date_range("2020-01-31", periods=3, freq="ME")
    sp = _spreads(["A"], dates, 300.0, price=80.0)
    # Default in the month after the first rebalance date.
    ddate = dates[0] + pd.Timedelta(days=10)
    exits = pd.DataFrame({"cusip": ["A"], "exit_type": [DEFAULTED], "default_date": [ddate]})
    r = monthly_excess_return(sp, exits, _cfg(recovery=0.4)).set_index("rebalance_date")
    # loss = (0.4*100 - 80)/80 = -0.5
    assert np.isclose(r.loc[dates[0], "r1"], -0.5, atol=1e-9)
    # No position after default.
    assert dates[2] not in r.index


def test_detect_exits_classifies_default_vs_matured():
    cfg = _cfg(distress_floor=55.0)
    end = pd.Timestamp("2024-12-31")
    daily = []
    # Defaulter: price collapses to 30 and exits early (2021) with ttm still > 0.5.
    for d in pd.bdate_range("2020-01-01", "2021-06-30"):
        daily.append({"cusip": "D", "date": d, "price": 30.0, "maturity": 5.0})
    # Matured: trades to near-maturity at panel end.
    for d in pd.bdate_range("2020-01-01", end):
        daily.append({"cusip": "M", "date": d, "price": 100.0, "maturity": 0.05})
    res = detect_exits(pd.DataFrame(daily), cfg).set_index("cusip")
    assert res.loc["D", "exit_type"] == DEFAULTED
    assert res.loc["M", "exit_type"] == "MATURED"


def test_neutralization_removes_duration_exposure():
    rng = np.random.default_rng(0)
    n = 300
    dur = rng.uniform(2, 12, n)
    # z is strongly driven by duration; neutralized z should be ~uncorrelated with it.
    z = 0.8 * dur + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"z": z, "mod_duration": dur, "dts": dur * 100,
                       "sector_ff30": rng.choice(list("ABC"), n)})
    zn = neutralize_signal(df, ["sector", "duration", "dts"])
    assert abs(np.corrcoef(zn, dur)[0, 1]) < 0.1


def test_long_short_weights_dollar_neutral():
    zn = pd.Series(np.arange(100.0))
    w = long_short_weights(zn, n_quantiles=5)
    assert np.isclose(w[w > 0].sum(), 1.0)
    assert np.isclose(w[w < 0].sum(), -1.0)
    assert np.isclose(w.sum(), 0.0, atol=1e-12)
