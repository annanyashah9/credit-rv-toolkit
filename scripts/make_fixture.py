"""Generate a tiny synthetic OBAP-like daily panel so the pipeline is runnable and
testable without the real (large) dataset.

Column names intentionally differ from our canonical names (e.g. `trd_exctn_dt`,
`prc_clean`, `gspread_bp`) so the Phase 0 role classifier gets exercised realistically.
Not representative data -- plumbing only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "synth_panel.parquet"

SECTORS = ["INDUSTRIAL", "FINANCIAL", "UTILITY"]
RATINGS = ["AA", "A", "BBB"]


def main() -> None:
    rng = np.random.default_rng(42)
    n_bonds = 40
    dates = pd.bdate_range("2018-01-01", "2024-12-31")  # ~7 years of business days

    bonds = []
    for i in range(n_bonds):
        sector = SECTORS[i % len(SECTORS)]
        rating = RATINGS[i % len(RATINGS)]
        base_spread = {"AA": 50, "A": 90, "BBB": 160}[rating] + rng.normal(0, 15)
        bonds.append(
            {
                "CUSIP": f"SYN{i:05d}AA{i % 9}",
                "issuer_ticker": f"ISSU{i % 12:02d}",
                "sector": sector,
                "rating": rating,
                "coupon": round(rng.uniform(2.0, 6.0), 3),
                "maturity_dt": pd.Timestamp("2018-01-01")
                + pd.Timedelta(days=int(rng.integers(3, 30) * 365)),
                "issue_size_mm": float(rng.choice([300, 500, 750, 1000, 1500])),
                "_base_spread": base_spread,
            }
        )

    rows = []
    for b in bonds:
        # AR(1)-ish spread path around the bond's base level.
        n = len(dates)
        shocks = rng.normal(0, 3, n)
        path = np.zeros(n)
        path[0] = b["_base_spread"]
        for t in range(1, n):
            path[t] = 0.98 * path[t - 1] + 0.02 * b["_base_spread"] + shocks[t]
        ttm = (b["maturity_dt"] - dates).days / 365.25
        mod_dur = np.clip(ttm * 0.9, 0.5, 15)
        ust = 2.5  # crude flat base yield, bps in pct terms below
        yld = ust + path / 100.0
        prc_clean = 100 - (yld - b["coupon"]) * mod_dur  # rough price proxy
        vol = rng.lognormal(mean=1.5, sigma=1.0, size=n)
        for t in range(n):
            rows.append(
                {
                    "CUSIP": b["CUSIP"],
                    "issuer_ticker": b["issuer_ticker"],
                    "trd_exctn_dt": dates[t],
                    "prc_clean": round(float(prc_clean[t]), 4),
                    "yield": round(float(yld[t]), 4),
                    "gspread_bp": round(float(path[t]), 2),
                    "mod_dur": round(float(mod_dur[t]), 4),
                    "par_volume": round(float(vol[t]), 2),
                    "rating": b["rating"],
                    "sector": b["sector"],
                    "coupon": b["coupon"],
                    "maturity_dt": b["maturity_dt"],
                    "issue_size_mm": b["issue_size_mm"],
                }
            )

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Wrote {len(df):,} rows x {df.shape[1]} cols -> {OUT}")


if __name__ == "__main__":
    main()
