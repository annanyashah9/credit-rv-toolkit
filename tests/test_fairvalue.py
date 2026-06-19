"""Unit tests for the Phase-2a fair-value pieces: asinh transform, EB shrinkage,
peer-shrunk fit, and residual orthogonality to liquidity controls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.fairvalue.linear import peer_shrunk_fair_spread
from crv.fairvalue.shrink import eb_issuer_shrinkage
from crv.fairvalue.transform import from_model_space, to_model_space


def test_asinh_round_trip_and_sign():
    x = np.array([-50.0, 0.0, 100.0, 500.0])
    y = to_model_space(x, scale=100.0)
    assert np.allclose(from_model_space(y, 100.0), x)
    assert y[0] < 0 < y[2]            # sign preserved
    assert (np.diff(y) > 0).all()     # monotone increasing


def test_eb_shrinks_singletons_more_than_big_issuers():
    # Issuer A: 50 bonds with mean residual 10; issuer B: 1 bond with residual 10.
    resid = pd.Series([10.0] * 50 + [10.0])
    issuer = pd.Series(["A"] * 50 + ["B"])
    shrunk = eb_issuer_shrinkage(resid, issuer)
    a_eff = shrunk.iloc[0]
    b_eff = shrunk.iloc[-1]
    assert a_eff > b_eff               # big issuer keeps more of its effect
    assert b_eff < 10.0                # singleton is pulled toward 0


def test_peer_shrunk_residual_orthogonal_to_liquidity():
    # Build a cross-section where spread depends on term, sector, and a liquidity
    # control; the peer-shrunk residual should be ~uncorrelated with that control.
    rng = np.random.default_rng(0)
    n = 400
    term = rng.uniform(1, 10, n)
    sector = rng.choice(["A", "B", "C"], n)
    bao = rng.normal(size=n)
    sec_eff = pd.Series(sector).map({"A": 0.0, "B": 0.3, "C": -0.2}).to_numpy()
    y = 0.5 + 0.1 * term + sec_eff + 0.4 * bao + rng.normal(scale=0.05, size=n)
    df = pd.DataFrame({
        "ttm": term, "sector_ff30": sector, "issuer": rng.integers(0, 60, n),
        "bao_gamma": bao, "amihud": rng.normal(size=n), "trade_freq": rng.uniform(.5, 1, n),
        "log_age": rng.uniform(0, 2, n), "log_issue_size": rng.uniform(12, 14, n),
        "y_model": y,
    })
    fair = peer_shrunk_fair_spread(df, y_col="y_model")
    resid = df["y_model"] - fair
    corr = np.corrcoef(resid, df["bao_gamma"])[0, 1]
    assert abs(corr) < 0.1             # liquidity priced into fair value, not the residual
