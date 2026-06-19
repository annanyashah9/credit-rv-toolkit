# Phase 2a — Real Linear Signal: Results & Naive Comparison

Phase 2a replaced the naive level-OLS residual with a **liquidity-controlled, asinh-transformed,
peer-shrunk** residual. Same monthly grid, same thin backtest (`crv backtest`), same costless
spread-return proxy — so the two are directly comparable. Per-model detail:
`docs/phase1_5-results.md` (naive) and `docs/backtest-peer_shrunk.md`.

## What changed in the model
1. **asinh(spread)** target instead of the raw level — tames the right-skew the naive residual had.
2. **Liquidity controls in the fair-value regression** (Bao–Pan–Wang gamma, Amihud, trade
   frequency, log age, log issue size) — the illiquidity premium is now *priced into fair value*,
   so the residual is liquidity-neutral by construction (verified: resid ⟂ liquidity in tests).
3. **Empirical-Bayes peer shrinkage** — each issuer's curve shrinks toward its sector peers;
   single-bond issuers collapse to the sector, large issuers keep their own curve.

## Head-to-head (3m horizon unless noted)

| Metric | Naive (Phase 1) | Peer-shrunk (Phase 2a) | Change |
|---|---|---|---|
| Rank IC 1m | 0.070 (t=4.1) | **0.107 (t=10.4)** | sharper |
| Rank IC 3m | 0.103 (t=3.7) | **0.112 (t=7.0)** | sharper |
| Rank IC 6m | 0.112 (t=2.6) | **0.108 (t=4.4)** | sharper |
| 1m hit rate | 62% | **76%** | +14pp |
| **Quintile LS — MEAN** | **−0.0051 (loses)** | **+0.0023 (positive)** | **sign flip** |
| Quintile LS — median | +0.0022 | +0.0042 | +0.0020 |
| Cheapest-bucket tail (<−5%) | 12.2% | **8.3%** | −3.9pp |

## Read
The headline Phase-1.5 problem — a positive rank IC but a **negative** equal-weight long-short,
because the cheap bucket was contaminated by a distressed/illiquid tail — is **fixed**: the mean
long-short flips positive, the blow-up tail shrinks from 12.2% to 8.3%, and the IC becomes far more
consistent (HAC t roughly doubles, hit-rate +14pp). Controlling for liquidity and shrinking toward
peers stripped much of the contamination while keeping the genuine convergence content.

## Honest caveats (unchanged from 1.5, still binding)
- Outcome is a **costless spread-change proxy off the same price series**; bid-ask bounce inflates
  mean-reversion, so the IC magnitudes (esp. t≈10 at 1m) are optimistic. The real test is Phase 3
  with independent returns, transaction costs, neutralization, and default carry-through.
- Bao gamma is derived from the same price series as the spread, so it is a control, not a clean
  instrument.
- Liquidity controls can over-absorb genuine cheapness (the illiquidity premium is partly the
  alpha); Phase 3's contamination test quantifies that trade-off rather than assuming it.

## Verdict
**Phase 2a succeeded on its own terms** and strengthens the interview narrative: the modeling
choices were motivated by a measured failure (the distressed-tail contamination) and demonstrably
fixed it. GBM engine + ML-vs-linear comparison is Phase 2b; full cost/neutralized/default-aware
evaluation is Phase 3.
