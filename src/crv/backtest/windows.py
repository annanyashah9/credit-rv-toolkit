"""Walk-forward window generation for out-of-sample model training.

Produces (train_dates, predict_dates) splits over the sorted unique rebalance dates.
A model is refit at the start of each block on the trailing window, then used to
predict every rebalance date in the block until the next refit. Train dates are always
strictly before their predict dates — the leakage guarantee.
"""

from __future__ import annotations

import pandas as pd


def walk_forward_windows(
    dates,
    train_window_months: int,
    min_train_months: int,
    refit_every_months: int,
    scheme: str = "rolling",
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Return a list of (train_dates, predict_dates).

    dates are month-end rebalance dates; "months" are counted in rebalance steps
    (one step per element of the sorted unique dates). For each refit anchor i with
    at least `min_train_months` of history:
      - train = dates in [i - train_window_months, i)  (rolling) or [0, i) (expanding)
      - predict = dates [i, i + refit_every_months)
    """
    d = pd.DatetimeIndex(sorted(pd.to_datetime(pd.Index(dates)).unique()))
    n = len(d)
    out = []
    i = min_train_months
    while i < n:
        lo = 0 if scheme == "expanding" else max(0, i - train_window_months)
        train = d[lo:i]
        predict = d[i: min(i + refit_every_months, n)]
        if len(train) >= min_train_months and len(predict) > 0:
            out.append((train, predict))
        i += refit_every_months
    return out
