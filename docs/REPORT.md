# A Credit Relative-Value Engine: Construction and Out-of-Sample Evaluation

## Abstract

This project builds a cross-sectional relative-value model for US corporate bonds and subjects its
signal to a deliberately demanding backtest. A fair-value spread curve is fit across a curated
universe; the standardised residual—how cheap or rich a bond looks relative to comparable peers—is
the trading signal. The central question is not whether the signal *correlates* with subsequent
returns, but whether trading it would have made money once transaction costs, default losses, and
exposure neutrality are taken seriously. The answer is instructive: the signal carries strong,
persistent, and economically interpretable predictive content, yet it is not profitable net of
realistic credit transaction costs at any feasible turnover. The contribution is less a strategy
than an honest measurement of the gap between predictive power and tradeability.

## Data and universe

The analysis uses the Open Source Bond Asset Pricing daily panel (29,776,137 bond-day
observations spanning 2002-07-01 to 2025-03-31), supplemented by the FRED Treasury
curve. A point-in-time universe is rebuilt at each month-end from panel-derivable inclusion
rules—an issue-size floor, a minimum trade-frequency threshold, and a maturity band—yielding on
the order of 3,172 bonds per cross-section across 272 monthly
rebalances. Inclusion is evaluated as of each date, so the universe is free of survivorship bias;
defaulting names are detected and carried through at an assumed recovery rather than dropped
(4,771 default events were identified over the sample).

## Signal construction

The fair-value spread is modelled cross-sectionally on term, sector, and explicit liquidity
controls, with the spread first transformed by an inverse hyperbolic sine to tame the pronounced
right skew of credit spreads. Liquidity—Bao–Pan–Wang gamma and Amihud illiquidity, alongside trade
frequency, age, and issue size—is priced *into* the fair value, so that the residual is
liquidity-neutral by construction and the resulting cheapness measure is not merely a repackaged
illiquidity premium. A second stage applies empirical-Bayes shrinkage to each issuer's residual
curve toward its sector peers, so that bonds from sparsely-represented issuers are pulled toward the
sector level while large, well-observed issuers retain their own curves. The standardised residual
from this procedure is the signal.

## Does machine learning help?

Estimated identically—same features, same rolling-window training—the walk-forward ridge and gradient-boosted models post 3-month information coefficients of 0.210 and 0.196 on realised excess returns. The gradient-boosted model does not improve on the linear one, so its added flexibility buys nothing here: once liquidity is controlled, the cheapness-to-return relationship is essentially linear. On realised returns the walk-forward specifications outscore the cross-sectional peer-shrunk model (0.159); under the costless spread-change proxy used for model selection in the earlier comparison the ranking reverses, a reminder that the evaluation target matters. The peer-shrunk model is retained for the P&L study below for its point-in-time interpretability, and, as the next sections show, the tradeability verdict is the same whichever engine is used—the differences are small against the cost hurdle.

## Predictive content on realised returns

Measured against realised excess returns—carry net of duration-scaled spread changes, with default
losses included—the signal's information coefficient is large and decays gracefully with horizon:

|   horizon_m |   mean_ic |   hac_t |
|------------:|----------:|--------:|
|           1 |     0.137 |    13.2 |
|           3 |     0.159 |    10.1 |
|           6 |     0.166 |     7   |
|          12 |     0.164 |     5.3 |

The Newey–West t-statistics (which correct for the overlap induced by multi-month holding periods)
leave little doubt that the relationship is real rather than a small-sample artefact.

## Net-of-cost performance

Predictive content does not survive contact with trading frictions. The table below traces the
duration-, DTS-, and sector-neutralised long-short portfolio across holding periods; longer holding
lowers turnover and therefore cost, but the gross signal decays faster than the costs fall.

| variant   |   net_ann |   gross_ann |   turnover |   net_sharpe |   hac_t |
|:----------|----------:|------------:|-----------:|-------------:|--------:|
| hold3m    |   -0.0084 |      0.0146 |       0.54 |        -0.76 |   -3.4  |
| hold6m    |   -0.0063 |      0.0082 |       0.34 |        -0.59 |   -2.74 |
| hold12m   |   -0.0051 |      0.0044 |       0.23 |        -0.53 |   -2.71 |

The best variant (hold12m) still returns -0.0051 per year net of costs.
No configuration is net-positive: realised credit bid-ask spreads exceed the gross edge per unit of
turnover at every point on the frontier.

## Is the edge genuine, or an artefact?

Two contamination tests address the obvious objections. Excluding names that default within the
following window barely moves the information coefficient (0.159 on the full sample
versus 0.160 among survivors), so the signal is not simply a disguised bet on
impending default. Splitting each cross-section by liquidity, the signal is, if anything, slightly
stronger among the more liquid names (0.175) than the less liquid
(0.149), so the result is not an illiquidity premium harvested in names that
cannot actually be traded.

## Conclusion

The exercise cleanly separates two questions that a cheap/rich screen usually conflates. Is there
signal? Yes—strong, persistent, robust to defaults and to liquidity. Is it tradeable? No: at any
turnover the duration- and credit-beta-neutral long-short is unprofitable once realistic transaction
costs are charged. The honest conclusion is that the mispricings the model identifies are real but
smaller than the cost of harvesting them, which is precisely the sort of finding a rigorous backtest
exists to surface. Natural extensions—activating the hand-collected ratings and call schedule to
enforce a strictly bullet, investment-grade universe, and seeking lower-cost implementation than
monthly long-short rebalancing—are the obvious next steps.

## Reproducibility

The entire pipeline runs from a single command (`make all`); each stage reads and writes versioned
parquet, and a run manifest records the configuration and input hashes. Per-phase detail is in the
companion documents under `docs/`.
