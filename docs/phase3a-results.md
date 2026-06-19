# Phase 3a — Net-of-Cost Neutralized Backtest

> Real excess returns (carry, spread-change, default loss), measured bid/ask costs,
> default/recovery carry-through, duration/DTS/sector-neutral. Signal: peer_shrunk.
> Holding 3m overlapping; HAC lag 2.

## Performance (annualized, excess-return units)

| style     |   gross_ann |   net_ann |   net_sharpe |   net_HAC_t |   ann_cost |   turnover |   breadth |   net_dur |   net_dts |
|:----------|------------:|----------:|-------------:|------------:|-----------:|-----------:|----------:|----------:|----------:|
| ls        |      0.0146 |   -0.0084 |        -0.76 |       -3.4  |     0.023  |       0.54 |      1625 |      0.3  |     -16.9 |
| long_only |      0.0073 |   -0.0056 |        -0.55 |       -2.61 |     0.0129 |       0.31 |      3041 |      0.59 |     157.8 |

Cost coverage: 92% measured bid/ask, 8% bucketed fallback.

## Read

The duration/DTS/sector-neutral long-short is **gross-positive** (+0.0146/yr) but does NOT survive transaction costs: a 0.0230/yr cost drag at 0.54 monthly turnover flips it to net -0.0084/yr (HAC t=-3.40, Sharpe -0.76). Realized net duration +0.30 and net DTS -16.9. 4771 defaults carried through at 40% recovery (real recovery losses, heavier than the proxy). This is the honest centerpiece result: the signal predicts (gross-positive), but at monthly turnover, wide credit bid/ask makes it untradeable as-is. The lever is turnover — Phase 3b tests longer holding / no-trade bands, plus the impending-default and liquidity contamination tests.

Figure: `reports/equity_curve.png`. Phase 3b adds contamination tests + IC-decay on real returns.