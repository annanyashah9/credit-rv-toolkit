"""Matplotlib figures for the thin backtest (Agg backend; saves PNGs, no display)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_ic_decay(ic_table: pd.DataFrame, out_png: str | Path) -> Path:
    """Mean IC vs horizon with +/-1 HAC-SE error bars."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(
        ic_table["horizon_m"], ic_table["mean_ic"], yerr=ic_table["hac_se"],
        marker="o", capsize=4,
    )
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("horizon (months)")
    ax.set_ylabel("mean rank IC")
    ax.set_title("IC decay (naive signal; HAC-SE error bars)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_equity_curves(books: dict, out_png: str | Path) -> Path:
    """Cumulative net (and gross) return curves for each portfolio style."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    for style, book in books.items():
        ax.plot(book.index, book["net"].cumsum(), lw=1.5, label=f"{style} net")
        ax.plot(book.index, book["gross"].cumsum(), lw=0.8, ls="--", alpha=0.6,
                label=f"{style} gross")
    ax.axhline(0, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("cumulative excess return")
    ax.set_title("Phase 3a — cumulative net-of-cost P&L (excess-return proxy)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def plot_ic_timeseries(ic: pd.Series, horizon: int, out_png: str | Path) -> Path:
    """IC over time for one horizon, with a cumulative-mean overlay."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(ic.index, ic.values, width=20, alpha=0.5, label="per-date IC")
    ax.plot(ic.index, ic.expanding().mean(), color="black", lw=1.5, label="cumulative mean")
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("rank IC")
    ax.set_title(f"{horizon}m-horizon IC over time")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png
