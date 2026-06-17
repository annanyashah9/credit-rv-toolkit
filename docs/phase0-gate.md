# Phase 0 — Data-Truth Go/No-Go Gate

**Purpose:** convert the conditional assumptions in the plan into verified facts *before* any
spread/curve/backtest code is built. Every check below has a **PASS criterion** and a **branch**
(what the plan changes if it fails). Nothing downstream starts until each check is resolved to a
known state (PASS or a recorded branch).

**How to run:** a throwaway exploration script (`notebooks/phase0_inventory.*`, not in the
reproducible path) that loads the raw OBAP panel + FRED pulls and emits a one-page
`docs/phase0-findings.md` filling in every "FINDING:" slot below. That findings file becomes the
record that unblocks Phase 1 and freezes the config defaults.

---

## A. Data existence & shape

### A1. Panel frequency  *(Risk #1 — dominant)*
- **Check:** distinct observation dates per CUSIP; modal gap between dates.
- **PASS:** median gap ≈ 1 trading day → daily, as assumed.
- **BRANCH (monthly):** horizons 1m/3m/6m become 1/3/6 obs; Amihud/Bao gamma need a separate
  daily feed or are dropped; HAC lag recomputed in months; IC-decay plot x-axis in months. This is
  a **plan-altering** outcome — re-open horizon and liquidity decisions before proceeding.
- **FINDING:** _____

### A2. Column inventory
- **Check:** full column list + dtypes + non-null %. Tag each as price / yield / spread / duration /
  volume / bid-ask / identifier / static-ref.
- **PASS:** price, yield, some spread, and a duration measure all present with >90% coverage.
- **BRANCH:** any missing core field re-routes the dependent stage (see B/C below).
- **FINDING:** _____

### A3. Identifiers & join keys
- **Check:** which of CUSIP / ISIN / issuer-id are present and stable; can the hand-collected
  ratings/calls CSV join cleanly; can FRED curves join on date.
- **PASS:** a stable per-bond key joins to the CSV with ≥95% match on the intended universe.
- **BRANCH:** poor match → universe shrinks to the matchable subset; document the attrition.
- **FINDING:** _____

### A4. History length & coverage
- **Check:** first/last date; count of bonds alive per date; gaps.
- **PASS:** enough history for ≥1 training window + multiple non-overlapping 6m eval blocks
  (rule of thumb: ≥5 years).
- **BRANCH (short history):** expanding-window only (no rolling); fewer horizons; flag low power.
- **FINDING:** _____

### A5. Price convention
- **Check:** clean vs dirty price; par scaling (100 vs 1.0); presence of accrued.
- **PASS:** convention identified unambiguously.
- **BRANCH:** sets the **default-classifier distress floor** (D3) and spread inputs; cannot set those
  configs until this is known.
- **FINDING:** _____

### A6. Missingness pattern in price series
- **Check:** per-bond run-length of gaps; fraction of zero-return days.
- **PASS:** gaps short enough that Bao gamma / returns are computable on most names.
- **BRANCH:** heavy gaps → minimum-observations filter in the universe rule; some names excluded.
- **FINDING:** _____

---

## B. Spread feasibility

### B1. Volume present?  *(Risk #2)*
- **Check:** is there a daily par/volume column with usable coverage.
- **PASS:** yes → **Amihud enabled** as a liquidity control alongside Bao gamma.
- **BRANCH (no/sparse):** Amihud dropped; Bao gamma + trade-freq + age + issue-size carry the
  liquidity vector. (Already the designed fallback — this just confirms which branch.)
- **FINDING:** _____

### B2. Bid-ask / high-low present?  *(Risk #2/#3)*
- **Check:** explicit bid-ask columns? daily high/low?
- **PASS (explicit bid-ask):** use measured spread for the cost model directly.
- **BRANCH (high/low only):** Corwin–Schultz estimator for costs.
- **BRANCH (close only — expected):** cost model = bucketed half-spread schedule (literature priors,
  optional FINRA spot-calibration); Bao gamma stays the liquidity *control*. Keep them distinct.
- **FINDING:** _____

### B3. Cashflow fields for Z-spread (optional in refined scope)  *(Risk #4)*
- **Check:** coupon, coupon frequency, maturity date, day-count present (panel or CSV)?
- **PASS:** all present → Z-spread feature-flag *available* as a nice-to-have.
- **BRANCH (missing — expected, and fine):** Z-spread stays OFF; **G-spread is the sole signal
  substrate** by design. Not a blocker — pipeline runs end-to-end on G-spread alone.
- **FINDING:** _____

### B4. FRED curve coverage
- **Check:** SOFR + the UST tenors needed to interpolate the maturity range of the universe.
- **PASS:** tenor grid spans the universe's maturities for the full history.
- **BRANCH (gaps):** interpolation/extrapolation rule documented; or restrict universe maturity range.
- **FINDING:** _____

---

## C. Signal & model feasibility

### C1. Universe size after point-in-time inclusion rules  *(Risk #8)*
- **Check:** apply size floor + min-trade-freq + rating exclusions AS OF several sample dates; count
  survivors per date.
- **PASS:** 200–400 names sustained across the window.
- **BRANCH (too few):** loosen a rule (documented), or accept lower breadth and flag power loss.
- **FINDING:** _____

### C2. Peer-cell population  *(Risk #7)*
- **Check:** count bonds per (sector × rating × seniority) cell at sample dates.
- **PASS:** most cells meet the min-cell-size; collapse ladder rarely needed.
- **BRANCH (thin cells):** confirm collapse order (sector×rating → sector → all) and min-cell-size
  config; expect noisier shrinkage — to be measured, not eliminated.
- **FINDING:** _____

### C3. Categorical coverage for the cross-sectional regression
- **Check:** at sample dates, count bonds per regression category (sector, rating, seniority) and the
  spread of term/duration values within each — i.e., is each effect identifiable.
- **PASS:** every category level has enough bonds and term-spread to estimate its effect without
  collinearity; the GBM has enough rows to train without overfitting per window.
- **BRANCH (sparse levels):** collapse rare categories (e.g., merge thin rating notches into tiers),
  or regularize harder (raise ridge λ / limit GBM depth) and document.
- **FINDING:** _____

### C4. Rating migration extent  *(ratings PIT)*
- **Check:** how many universe names changed rating tier over the window (from whatever rating
  history is obtainable, even coarse).
- **PASS (low migration):** static-now ratings are an acceptable v1 approximation; caveat logged.
- **BRANCH (high migration):** static ratings inject material look-ahead → prioritize backfilling the
  PIT effective-date columns before trusting rating-keyed results.
- **FINDING:** _____

---

## D. Backtest feasibility

### D1. Effective sample size under overlap  *(Risk #6 — inherent)*
- **Check:** for each horizon, compute nominal obs and *non-overlapping* block count.
- **PASS:** ≥ ~20 independent blocks at 6m (enough to estimate an IC with HAC).
- **BRANCH (few blocks):** report IC with wide HAC bands and an explicit underpowered warning; do
  not over-interpret 6m. This is disclosure, not a fix.
- **FINDING:** _____

### D2. Exit-event distinguishability  *(Risk #5)*
- **Check:** can panel-exits be split into matured (≈ scheduled maturity), called (≈ call date/price),
  and early-exit, using the CSV schedule?
- **PASS:** exit dates reconcile against CSV maturity/call within tolerance for most names.
- **BRANCH (no schedule reconciliation):** default classifier loses its disambiguator → fall back to a
  pure price-collapse rule and flag higher false-positive risk on the default label.
- **FINDING:** _____

### D3. Distress-floor calibration  *(Risk #5, depends on A5)*
- **Check:** distribution of prices near known/early exits; pick the distress floor + "well-before-
  maturity" margin that separate defaults from matured/called.
- **PASS:** a threshold cleanly separates the two clusters.
- **BRANCH (no clean separation):** widen the ACTIVE-but-distressed tag, hand-review the handful of
  ambiguous exits, record them in `data/reference/`.
- **FINDING:** _____

### D4. Default base rate
- **Check:** how many default events fall inside the window for the universe.
- **PASS:** enough events to run the impending-default contamination test (survivors-only vs full IC).
- **BRANCH (≈0 defaults):** contamination test becomes vacuous → document that v1 cannot test
  default-contamination empirically; recovery-carry logic still built but exercised only on synthetic
  cases / future data.
- **FINDING:** _____

---

## Gate decision

Phase 1 starts only after every check above is PASS or has a recorded BRANCH in
`docs/phase0-findings.md`. The findings file then **freezes** these configs:
`fairvalue.yaml` (ridge λ, peer-shrink target & collapse order, GBM depth/regularization),
`liquidity.yaml` (Amihud on/off, gamma window), `backtest.yaml` (horizons, HAC lag, distress floor,
recovery, neutralization axes), `spreads` Z-spread flag, and the universe inclusion thresholds.

**Plan-altering branches that, if triggered, require revisiting the plan before Phase 1 (not just a
config tweak):** A1 (monthly), A4 (too-short history), C1 (universe too small), D4 (no defaults to
test). Anything else is a config branch handled in stride.
