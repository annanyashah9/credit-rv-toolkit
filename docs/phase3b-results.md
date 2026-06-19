# Phase 3b — Turnover Frontier, Contamination, Real-Return Decay

> Closes the Phase-3 centerpiece. All P&L net-of-cost on real excess returns with default
> carry-through. Contamination tests are diagnostics (some use look-ahead).

## Net-of-cost vs turnover frontier

| variant     |   net_ann |   gross_ann |   turnover |   sharpe |   hac_t |   breakeven_hs |
|:------------|----------:|------------:|-----------:|---------:|--------:|---------------:|
| hold3m      |   -0.0084 |      0.0146 |     0.5396 |  -0.7556 | -3.3951 |         0.0023 |
| hold6m      |   -0.0063 |      0.0082 |     0.3438 |  -0.5883 | -2.7397 |         0.002  |
| hold12m     |   -0.0051 |      0.0044 |     0.2265 |  -0.5261 | -2.7052 |         0.0016 |
| hold3m+band |   -0.0074 |      0.0096 |     0.3975 |  -0.783  | -3.5793 |         0.002  |

Best variant: **hold12m** at net -0.0051/yr (turnover 0.23). **No variant is net-positive** — costs dominate at every feasible turnover.

## Real-return IC-decay (rank IC of z vs forward excess return, HAC)

|   horizon_m |   mean_ic |   hac_t |   n_periods |
|------------:|----------:|--------:|------------:|
|           1 |    0.1365 | 13.1844 |         271 |
|           3 |    0.1585 | 10.0524 |         269 |
|           6 |    0.1661 |  7.0103 |         266 |
|          12 |    0.1636 |  5.2796 |         260 |

## Contamination tests

- **Impending default**: full-sample IC 0.166 vs survivors-only 0.167 (0.4% default within 6m). The edge largely survives removing eventual-defaulters — not just a default bet.
- **Default odds by z-quintile** (cheap = top quintile):

|   q |   default_rate |
|----:|---------------:|
|   0 |         0.0025 |
|   1 |         0.002  |
|   2 |         0.0021 |
|   3 |         0.0044 |
|   4 |         0.0073 |
- **Liquidity premium** (bao_gamma): IC liquid 0.181 vs illiquid 0.156. Present in liquid names too — not purely a liquidity premium.

## Verdict

Net of realistic costs the strategy is **not tradeable as-is** at any tested turnover; the signal carries genuine, default-and-liquidity-robust convergence information, but the implementable edge is dominated by credit transaction costs. This is the honest conclusion of the centerpiece backtest.

Figures: `reports/turnover_frontier.png`, `reports/real_ic_decay.png`.