"""Duration Times Spread (DTS): the credit-beta exposure measure.

DTS = spread (bp) x spread-duration. For fixed-rate bullets, spread duration is well
approximated by modified duration. DTS is both a backtest neutralization axis and a
liquidity/term context feature.
"""

from __future__ import annotations

import pandas as pd


def dts(spread_bp: pd.Series, spread_duration: pd.Series) -> pd.Series:
    """DTS in bp-years: spread_bp * spread_duration (elementwise)."""
    return (spread_bp * spread_duration).rename("dts")
