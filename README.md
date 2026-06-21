# credit-rv-toolkit

A cross-sectional **credit relative-value engine** for US corporate bonds: fit a fair-value
spread model to a curated universe, flag idiosyncratic cheap/rich names as standardized
residuals, and **rigorously backtest** whether those flags predict convergence — net of
transaction costs, neutral to rates and credit beta, and tested for liquidity-premium and
impending-default contamination.

The backtest is the centerpiece. The fair-value engine is run two ways — a transparent
linear/peer-shrunk model and a gradient-boosted model — and the backtest adjudicates which
produces the better out-of-sample, cost-aware, risk-neutralized signal.

- Plan & architecture: [docs/PLAN.md](docs/PLAN.md)
- Data-truth gate (run this first): [docs/phase0-gate.md](docs/phase0-gate.md)

## Data sources (all free)
- **Open Source Bond Asset Pricing** panel — cleaned prices/yields/spreads/duration.
- **FRED** — SOFR and Treasury curves.
- **FINRA Fixed Income Data Center** — per-bond reference (calibration only, not a pipeline stage).
- **Hand-collected CSV** — ratings & call schedules for the curated universe
  (`data/reference/ratings_calls.csv`, PIT-ready schema).

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Phase 0 — verify your data before building anything
1. Download the OBAP panel into `data/raw/` (csv or parquet).
2. Set `paths.obap_panel` in `configs/base.yaml`.
3. Run the inventory:
   ```bash
   crv phase0 --config configs/base.yaml
   ```
   This writes `docs/phase0-findings.md`: detected frequency, column roles, coverage, and a
   suggested PASS/BRANCH for each gate check. Resolve the MANUAL items by hand, confirm the
   column roles into `ingest.obap_column_map`, then freeze the configs. Only then does Phase 1
   start. See [docs/phase0-gate.md](docs/phase0-gate.md).

## Phase 1 — universe, spreads, naive signal
After the gate clears, build the bottom of the pipeline:
```bash
crv fred       # cache Treasury curve (keyless FRED)
crv phase1     # universe -> spreads (G-spread + DTS) -> naive fair-value residual
```
Outputs land in `data/interim/` (`universe.parquet`, `spreads.parquet`, `signal.parquet`). The
`z` column is the naive cheap/rich signal. G-spread is validated against the panel's own
`credit_spread` (r ≈ 0.9997).

## One command

```bash
make all          # universe -> spreads -> liquidity -> signal -> backtest -> report
                  # writes docs/REPORT.md + reports/run_manifest.json
make all ARGS=    # add --with-gbm via: crv all --with-gbm   (heavy ML comparison)
```

The consolidated writeup is **[docs/REPORT.md](docs/REPORT.md)**; per-phase detail lives in the
other `docs/*-results.md` files, and `reports/run_manifest.json` records the config and input hashes
for reproducibility.

## Headline result
The relative-value signal has strong, persistent predictive content on realised excess returns
(rank IC ≈ 0.14–0.17, highly significant), and the edge is robust to defaults and to liquidity — but
it is **not profitable net of realistic credit transaction costs at any turnover**. A rigorous
backtest separating "is there signal?" from "is it tradeable?" is the point of the project.

## Status
All phases complete (0 through 4). Remaining optional work: hand-collect the ratings/call-schedule
CSV (`data/reference/ratings_calls.csv`) to activate the strictly bullet, rating-screened curated
universe — the pipeline already supports it as a drop-in.
