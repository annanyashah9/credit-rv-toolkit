"""Unit tests for curve interpolation, G-spread, and DTS."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.spreads.discount import interp_curve_yield
from crv.spreads.dts import dts
from crv.spreads.gspread import g_spread


def _curve():
    d = pd.Timestamp("2020-01-02")
    return pd.DataFrame({1.0: [0.01], 5.0: [0.02], 10.0: [0.03]}, index=[d])


def test_interp_exact_and_midpoint():
    curve = _curve()
    d = curve.index[0]
    out = interp_curve_yield(curve, pd.Series([d, d, d]), np.array([1.0, 3.0, 10.0]))
    # exact tenors, plus linear midpoint between 1y(0.01) and 5y(0.02) at 3y -> 0.015
    assert np.isclose(out[0], 0.01)
    assert np.isclose(out[1], 0.015)
    assert np.isclose(out[2], 0.03)


def test_interp_clips_beyond_grid():
    curve = _curve()
    d = curve.index[0]
    out = interp_curve_yield(curve, pd.Series([d, d]), np.array([0.1, 50.0]))
    assert np.isclose(out[0], 0.01)   # below shortest tenor -> flat
    assert np.isclose(out[1], 0.03)   # beyond longest tenor -> flat


def test_interp_unknown_date_is_nan():
    curve = _curve()
    out = interp_curve_yield(curve, pd.Series([pd.Timestamp("1990-01-01")]), np.array([5.0]))
    assert np.isnan(out[0])


def test_g_spread_units_and_sign():
    curve = _curve()
    d = curve.index[0]
    panel = pd.DataFrame({"date": [d], "ytm": [0.04], "maturity": [5.0]})
    # (0.04 - 0.02) * 1e4 = 200 bp
    gs = g_spread(panel, curve)
    assert np.isclose(gs.iloc[0], 200.0)


def test_dts_product():
    out = dts(pd.Series([200.0, 100.0]), pd.Series([5.0, 7.0]))
    assert out.tolist() == [1000.0, 700.0]
