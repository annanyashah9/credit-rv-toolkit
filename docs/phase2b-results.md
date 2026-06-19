# Phase 2b — ML-vs-Linear Comparison (walk-forward)

> All arms scored on the same thin spread-return proxy (costless; bounce-inflated). The
> GAP between models is more trustworthy than absolute levels. WF-GBM and WF-Ridge share
> identical features + rolling-60m training, so their gap isolates functional form;
> peer_shrunk is the Phase-2a cross-sectional reference.

## Summary

| model       | IC_1m          | IC_3m         | IC_6m         |   LS_mean |   LS_median | tail<-5%   |
|:------------|:---------------|:--------------|:--------------|----------:|------------:|:-----------|
| peer_shrunk | 0.107 (t=10.4) | 0.112 (t=7.0) | 0.108 (t=4.4) |    0.0023 |      0.0042 | 8.3%       |
| ridge_wf    | 0.075 (t=4.1)  | 0.098 (t=3.3) | 0.093 (t=2.0) |   -0.0097 |     -0.0003 | 12.2%      |
| gbm_wf      | 0.079 (t=5.1)  | 0.096 (t=3.6) | 0.090 (t=2.2) |   -0.0077 |      0.0006 | 11.1%      |

## Verdict (3m)

**ML ≈ linear.** WF-GBM (0.096) and WF-Ridge (0.098) are within ~10% at 3m — the relationship here is largely linear once liquidity is controlled. A defensible, honest finding: prefer the simpler linear model unless GBM wins clearly after costs.

**Bigger story — model FORM, not complexity.** The cross-sectional peer-shrunk model beats both pooled walk-forward arms at 3m (IC 0.112 vs 0.098/0.096) and is the only arm with a positive equal-weight long-short (LS_mean +0.0023 vs negative for both WF arms). Its edge is the per-date issuer shrinkage toward sector peers — which the pooled models structurally lack (2,988 issuers can't be dummies) — not temporal pooling or tree flexibility. The win came from the right cross-sectional structure, not from a more complex learner.

Per-model detail: `docs/backtest-<kind>.md`.