# Credit Relative-Value Toolkit — Implementation Plan (v1, Refined ML-RV scope)

## One-liner
A cross-sectional credit relative-value engine: fit a fair-value spread model to a curated US
corporate-bond universe, flag idiosyncratic cheap/rich names as standardized residuals, and
**rigorously backtest** whether those flags predict convergence — net of costs, neutral to rates and
credit beta, and tested for liquidity-premium and impending-default contamination. The backtest is
the centerpiece. The fair-value engine is run **two ways — a transparent linear/peer-shrunk model
and a gradient-boosted model — and the backtest adjudicates which produces the better
out-of-sample, cost-aware, risk-neutralized signal.**

## Scope decisions (locked)
- **Fair value = cross-sectional regression**, not per-issuer term-structure fitting. Spread is
  modeled on term/duration, sector, rating, seniority, and liquidity controls. More robust than NSS
  given few bonds per issuer, and fully explainable.
- **Two fair-value engines, shared interface:** linear (ridge / peer-shrunk) and gradient-boosted.
  The comparison is the cutting-edge hook.
- **Spread substrate = G-spread.** Z-spread optional/feature-flagged. NSS/Svensson and Hull-White
  OAS are **out of v1** (not phased).
- **Peer shrinkage** = ridge / random-effects toward sector(×rating) means — closed-form/sklearn,
  no MCMC.
- **Ratings:** static-now, PIT-ready CSV schema (effective_date/thru_date columns), documented
  look-ahead caveat.
- **Liquidity control = Bao–Pan–Wang gamma** (daily closes) + trade-freq + age + issue-size;
  Amihud only if volume exists. **Cost model is a separate bucketed half-spread schedule** so it
  can't launder alpha.
- **Defaults:** heuristic exit classifier (matured/called/defaulted) using the CSV schedule; defaults
  marked to recovery and **carried**, never dropped.

## Pipeline order
ingest → spreads (G-spread, DTS) → fair value (linear + GBM) → signal (residual, liquidity-neutral)
→ backtest → reporting. Config-driven, reproducible from `make all`.

## Repo structure (revised)
```
credit-rv-toolkit/
├── pyproject.toml            # console_scripts: `crv`
├── Makefile                  # `make all` = full reproducible run
├── configs/                  # base / universe / fairvalue / liquidity / backtest .yaml
├── data/{raw,reference,interim,processed}/
│   └── reference/ratings_calls.csv     # hand-collected, PIT-ready, version-controlled
├── docs/{PLAN.md,phase0-gate.md,phase0-findings.md}
├── src/crv/
│   ├── config.py             # pydantic typed config
│   ├── io.py                 # parquet, caching, run manifest (config+input hashes+seed)
│   ├── ingest/{obap,fred,finra,reference}.py
│   ├── universe.py           # point-in-time inclusion engine
│   ├── spreads/{discount,gspread,zspread,dts}.py   # zspread feature-flagged
│   ├── fairvalue/            # <-- replaces curve/
│   │   ├── features.py       # design matrix: term, sector, rating, seniority, liquidity
│   │   ├── linear.py         # cross-sectional regression + ridge/peer shrinkage
│   │   ├── gbm.py            # gradient-boosted fair value (sklearn)
│   │   └── residual.py       # fair spread + standardized residual (shared across engines)
│   ├── liquidity/{bidask,amihud,controls}.py        # bidask.py => Bao gamma + fallbacks
│   ├── signal.py             # fair value + liquidity-neutralization -> final signal
│   ├── backtest/             # THE CENTERPIECE
│   │   ├── windows.py        # walk-forward / expanding splits (leakage-free)
│   │   ├── ic.py             # IC at horizons, decay, HAC errors
│   │   ├── portfolios.py     # quintiles; duration/sector(/DTS) neutralization
│   │   ├── costs.py          # bucketed half-spread schedule (distinct from liquidity control)
│   │   ├── defaults.py       # exit classifier + recovery carry-through
│   │   ├── contamination.py  # impending-default & liquidity-premium tests
│   │   └── stats.py          # Newey-West HAC, breadth, turnover
│   ├── report/{figures,dashboard}.py
│   └── cli.py                # crv ingest|spreads|fairvalue|signal|backtest|report|all
├── tests/                    # unit + golden-path integration test on a tiny fixture
└── notebooks/                # exploratory only (incl. phase0_inventory), not in repro path
```

## Key interfaces (design, not implementation)
```python
# fairvalue/features.py
def build_design_matrix(panel, ref, liq, asof) -> pd.DataFrame: ...   # term, sector, rating, seniority, liq

# shared engine protocol — both models implement fit/predict_fair_spread
class FairValueModel(Protocol):
    def fit(self, X_train, y_train_spread) -> None: ...
    def predict_fair_spread(self, X) -> np.ndarray: ...

# fairvalue/linear.py
class PeerShrunkLinear(FairValueModel): ...     # ridge / random-effects toward sector(xrating)
# fairvalue/gbm.py
class GBMFairValue(FairValueModel): ...         # sklearn HistGradientBoostingRegressor

# fairvalue/residual.py
def standardized_residual(observed, fair, cross_section_scale) -> pd.Series: ...

# signal.py
def liquidity_neutralize(raw_signal, liq_features) -> pd.Series: ...   # control BEFORE flagging
def make_signal(panel, model: FairValueModel, liq, asof) -> pd.Series: ...

# backtest/ (centerpiece)
def walk_forward_windows(dates, train, step, scheme) -> Iterator[Window]: ...
def information_coefficient(signal, fwd_returns, horizon) -> ICResult:   # carries HAC SE
def build_portfolios(signal, exposures, n=5, neutralize=("duration","sector")) -> Portfolios: ...
def apply_costs(trades, cost_table) -> pd.Series: ...
def carry_defaults(positions, exits, recovery_by_seniority) -> pd.DataFrame: ...
def impending_default_test(signal, exits, horizon) -> ContaminationReport: ...
```

## Dependencies
- Core: numpy, pandas, scipy, pyarrow.
- Models: **scikit-learn** (ridge, HistGradientBoostingRegressor), statsmodels (Newey-West HAC,
  random-effects). No pymc/numpyro, no QuantLib in v1.
- Config/validation: pydantic, pyyaml. Data: fredapi/pandas-datareader; requests+cache for FINRA
  (calibration only). Reporting: matplotlib; streamlit (optional Phase 4). Dev: pytest, ruff, mypy.

## Phasing
- **Phase 0 — Data-truth gate.** See `docs/phase0-gate.md`. Verify frequency/columns/defaults;
  freeze configs. Hard go/no-go.
- **Phase 1 — Spreads + linear fair value + naive residual.** ingest → G-spread → DTS →
  cross-sectional linear fair value → standardized residual. Plumbing.
- **Phase 1.5 — Thin backtest skeleton.** Walk-forward + IC@horizons + HAC on the naive residual.
  Gate: does even the crude signal show predictive content?
- **Phase 2 — Real signal + ML comparison.** Liquidity controls + peer shrinkage; add the
  **GBM fair-value engine** behind the shared interface; produce the proper liquidity-neutral signal
  for both engines.
- **Phase 3 — Full backtest harness.** IC-decay, neutralized quintile portfolios, measured costs,
  default/recovery carry-through, impending-default & liquidity contamination tests,
  breadth/turnover/net-of-cost. Run for **both engines** → the adjudication.
- **Phase 4 — Reporting.** Linear-vs-GBM comparison writeup + dashboard.

## Inherent limits to disclose (not bugs)
- Overlapping-window effective-N is small; HAC fixes SEs, not information content.
- Thin sector×rating cells make shrinkage/effects noisy — measured, not eliminated.
- Static ratings inject look-ahead unless PIT backfilled; severity depends on observed migration.
