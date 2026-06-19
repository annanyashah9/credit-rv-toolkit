"""Summarize a monthly portfolio P&L series with HAC inference.

Overlapping holding periods autocorrelate the monthly returns, so the mean return's
t-stat uses Newey-West with lag = holding_months - 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crv.backtest.stats import newey_west_mean


def summarize(book: pd.DataFrame, cfg) -> dict:
    """Performance dict for a run_portfolio output (gross/net/turnover/exposure series)."""
    af = cfg.backtest.ann_factor
    lag = max(cfg.backtest.holding_months - 1, 0)
    net, gross = book["net"], book["gross"]

    mean_net, se_net, t_net = newey_west_mean(net, lags=lag)
    vol = float(net.std(ddof=1))
    return {
        "n_months": int(len(net)),
        "gross_ann": float(gross.mean() * af),
        "net_ann": float(mean_net * af),
        "net_vol_ann": vol * np.sqrt(af),
        "net_sharpe": (mean_net * af) / (vol * np.sqrt(af)) if vol > 0 else float("nan"),
        "net_hac_t": t_net,
        "avg_turnover": float(book["turnover"].mean()),
        "avg_breadth": float(book["breadth"].mean()),
        "cost_drag_ann": float(book["cost"].mean() * af),
        "avg_net_dur": float(book["net_dur"].mean()),
        "avg_net_dts": float(book["net_dts"].mean()),
    }
