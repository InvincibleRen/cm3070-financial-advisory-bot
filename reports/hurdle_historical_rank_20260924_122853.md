# Hurdle Selection on a Risk-Adjusted Excess-Return Target

Generated at: 2026-09-24 12:28:53

## Configuration

- Universe: point-in-time S&P 500 (each month's actual members)
- Membership coverage: 742 tickers were members at some point; 118 have no price data. Share of each month's members with prices: median 91%, lowest 76% (2015-01), highest 100%
- History: 2015-01-01 to latest, 140 monthly rebalances
- Training window: 12 rebalances rolling
- Models: XGBoost, Random forest, Ridge (auxiliary), xgb classifier (median label, top-N)
- Hurdle: predicted target > 0.0
- Max names: 5 (equal weight among those that qualify)
- Fallbacks: spy, bil, trend
- Transaction cost: 10 bps per unit turnover (fallback switches included)
- Feature normalisation: rank
- Features: technical,fundamental (16 columns: mom_1m, mom_3m, mom_6m, mom_12m, reversal_5d, mom_risk_adj, high_52w, volatility, macd, adx, pe, pb, roe, earnings_growth, net_margin, debt_to_equity)
- Training target: (r - beta x r_SPY) / vol, winsorised per month
- Walk-forward starts: 2015-01-01
- Feature coverage (non-NaN cells, before normalisation):
    - technical: mom_1m 99%, mom_3m 98%, mom_6m 96%, mom_12m 92%, reversal_5d 100%, mom_risk_adj 92%, high_52w 100%, volatility 99%, macd 100%, adx 99%
    - fundamental: pe 25%, pb 30%, roe 26%, earnings_growth 16%, net_margin 32%, debt_to_equity 32%
- Target rows: 60745, mean -0.021, share above zero 48.6%

**Target.** `y = (r_stock - beta * r_SPY) / vol` over the next month, where beta (Blume-adjusted, 252 trading days) and vol (63 days, monthly scale) use only data up to the rebalance date, winsorised at the 1st/99th percentile of each month. Subtracting beta x SPY means a stock cannot score well just by being high-beta in a rising market; dividing by vol puts the payoff in units of risk taken.

**Rule.** Buy the names whose predicted target is above the hurdle, best first, up to the cap, equal-weighted among themselves. If none qualify, hold the fallback. The hurdle is fixed before the run, not tuned on it.

## 1. The picks: win rate, payoff ratio, expectancy

Each pick is judged over the month it was held, two ways. **Raw** = its return minus SPY's, what a trader sees. **Beta-adjusted** = its return minus beta x SPY's (beta estimated before the pick), which a high-beta stock cannot win just because the market rose. The universe column is every stock in the same months, weighted by how many picks were made that month.

Two random benchmarks replace each month's picks with the same number of names from that month. **`p (random)` is the primary test**: stand-ins are drawn from the whole cross-section, so it answers the question a user asks — did these picks earn more than picking at random? `p (vol-matched)` draws each stand-in from the 20 names nearest the pick in volatility, which by construction removes any return earned by holding volatile names; it therefore answers the narrower question of whether anything remains *beyond* beta and volatility, and a selector that works by taking risk is not failed for losing it. p is the share of 2,000 draws that did at least as well; below 0.05 is the usual bar. The two single-factor rows rank on one measure with nothing fitted.

**Raw, minus SPY (primary)**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 660 | 51% | 45.9% | 48.3% | 1.07 | 1.01 | -0.34% | -0.16% | 0.03% | 0.724 | 0.871 |
| Random forest | 660 | 47% | 46.4% | 48.3% | 1.20 | 1.01 | 0.12% | -0.16% | -0.16% | 0.165 | 0.193 |
| Ridge (auxiliary) | 660 | 57% | 45.6% | 48.3% | 1.07 | 1.01 | -0.37% | -0.16% | -0.13% | 0.760 | 0.780 |
| xgb classifier (median label, top-N) | 665 | 58% | 44.1% | 48.4% | 1.03 | 1.01 | -0.73% | -0.16% | -0.09% | 0.977 | 0.979 |
| Rank on volatility alone (single factor) | 665 | 100% | 47.2% | 48.4% | 1.20 | 1.01 | 0.46% | -0.16% | 0.39% | 0.026 | 0.446 |
| Rank on beta alone (single factor) | 665 | 95% | 51.3% | 48.4% | 1.37 | 1.01 | 2.09% | -0.16% | 0.51% | 0.000 | 0.000 |

**Beta-adjusted**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 652 | 51% | 45.9% | 48.6% | 1.01 | 1.00 | -0.50% | -0.16% | -0.09% | 0.866 | 0.903 |
| Random forest | 659 | 47% | 46.9% | 48.6% | 1.18 | 1.00 | 0.13% | -0.16% | -0.15% | 0.151 | 0.196 |
| Ridge (auxiliary) | 658 | 57% | 46.2% | 48.7% | 1.01 | 1.00 | -0.46% | -0.16% | -0.17% | 0.848 | 0.812 |
| xgb classifier (median label, top-N) | 645 | 58% | 43.3% | 48.7% | 1.01 | 1.00 | -0.88% | -0.15% | -0.21% | 0.993 | 0.976 |
| Rank on volatility alone (single factor) | 665 | 100% | 46.8% | 48.7% | 1.13 | 1.00 | -0.05% | -0.15% | -0.08% | 0.373 | 0.477 |
| Rank on beta alone (single factor) | 665 | 95% | 48.9% | 48.7% | 1.27 | 1.00 | 1.08% | -0.15% | 0.04% | 0.000 | 0.016 |

*Payoff ratio = average winning excess return / average losing excess return (the realised risk-reward, no stop-loss or take-profit). Expectancy = mean excess return per pick. Vol percentile of picks = where the picks sat in that month's volatility ranking (50% = no tilt, 100% = always the most volatile). Vol-matched random = the average expectancy of random stand-ins drawn from the 20 names nearest each pick in volatility, i.e. what the picks' risk profile alone would have earned.*

### 1b. Why per-pick excess and compounded return can disagree

The expectancy above is an **arithmetic** mean per pick. An investor earns the **compounded** return, and for small returns the two differ by about half the variance, so a concentrated book gives more back to that term than the index does. The table separates the two; a small residual means the approximation accounts for the gap. All figures annualised, against SPY, on the `spy` fallback variant.

| Model | Arithmetic excess | Variance drag | Residual | Compounded excess | Book vol | SPY vol |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | -6.20% | −1.89% | -0.57% | -8.67% | 24.7% | 15.2% |
| Random forest | -0.74% | −2.95% | -0.23% | -3.92% | 28.6% | 15.2% |
| Ridge (auxiliary) | -6.51% | −1.75% | -0.46% | -8.71% | 24.1% | 15.2% |
| xgb classifier (median label, top-N) | -10.89% | −1.67% | -0.83% | -13.39% | 23.8% | 15.2% |
| Rank on volatility alone (single factor) | 4.61% | −10.86% | 0.12% | -6.13% | 49.0% | 15.2% |
| Rank on beta alone (single factor) | 24.57% | −9.47% | 4.45% | 19.54% | 46.1% | 15.2% |

*Arithmetic excess − variance drag + residual = compounded excess. The drag is not a fault in the selector: it is the arithmetic of compounding a more volatile series, and it shrinks if the book is held wider.*

## 2. Does stricter selection do better? (beta-adjusted)

Top-k by score each month, ignoring the hurdle. If the score orders stocks usefully, the pick columns improve as k shrinks — the best-scored few should earn more than the best-scored many.

**XGBoost**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 130 | 61% | 45.4% | 48.8% | 0.87 | 1.00 | -1.38% | -0.14% | 0.968 | 0.920 |
| 3 | 392 | 54% | 46.4% | 48.7% | 0.96 | 1.00 | -0.63% | -0.15% | 0.898 | 0.956 |
| 5 | 657 | 51% | 46.0% | 48.7% | 1.01 | 1.00 | -0.49% | -0.15% | 0.900 | 0.894 |
| 10 | 1316 | 50% | 45.4% | 48.7% | 0.99 | 1.00 | -0.60% | -0.15% | 0.988 | 0.950 |
| 20 | 2643 | 49% | 45.4% | 48.7% | 1.01 | 1.00 | -0.53% | -0.15% | 0.996 | 0.984 |
| 50 | 6618 | 48% | 46.8% | 48.7% | 0.97 | 1.00 | -0.46% | -0.15% | 1.000 | 0.998 |

**Random forest**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 132 | 50% | 52.3% | 48.6% | 1.46 | 1.00 | 1.64% | -0.16% | 0.008 | 0.028 |
| 3 | 398 | 48% | 46.2% | 48.7% | 1.29 | 1.00 | 0.34% | -0.16% | 0.100 | 0.146 |
| 5 | 664 | 47% | 47.0% | 48.7% | 1.18 | 1.00 | 0.14% | -0.15% | 0.158 | 0.228 |
| 10 | 1328 | 47% | 46.7% | 48.7% | 1.07 | 1.00 | -0.19% | -0.15% | 0.551 | 0.527 |
| 20 | 2654 | 47% | 46.4% | 48.7% | 1.01 | 1.00 | -0.40% | -0.15% | 0.956 | 0.902 |
| 50 | 6639 | 46% | 46.6% | 48.7% | 0.98 | 1.00 | -0.43% | -0.15% | 1.000 | 0.992 |

**Ridge (auxiliary)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 132 | 60% | 47.7% | 48.7% | 0.80 | 1.00 | -0.96% | -0.15% | 0.886 | 0.942 |
| 3 | 397 | 57% | 47.4% | 48.7% | 0.82 | 1.00 | -0.95% | -0.15% | 0.972 | 0.984 |
| 5 | 663 | 57% | 46.3% | 48.7% | 1.01 | 1.00 | -0.46% | -0.15% | 0.846 | 0.830 |
| 10 | 1324 | 55% | 47.2% | 48.7% | 1.02 | 1.00 | -0.28% | -0.15% | 0.750 | 0.687 |
| 20 | 2648 | 52% | 49.1% | 48.7% | 1.02 | 1.00 | -0.04% | -0.15% | 0.263 | 0.349 |
| 50 | 6631 | 49% | 48.2% | 48.7% | 1.07 | 1.00 | -0.02% | -0.15% | 0.062 | 0.102 |

**xgb classifier (median label, top-N)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 120 | 62% | 40.8% | 48.7% | 1.15 | 1.00 | -0.83% | -0.15% | 0.850 | 0.916 |
| 3 | 381 | 60% | 42.5% | 48.6% | 1.07 | 1.00 | -0.82% | -0.17% | 0.930 | 0.960 |
| 5 | 645 | 58% | 43.3% | 48.7% | 1.01 | 1.00 | -0.88% | -0.15% | 0.988 | 0.986 |
| 10 | 1308 | 56% | 44.2% | 48.7% | 1.03 | 1.00 | -0.67% | -0.15% | 0.994 | 0.982 |
| 20 | 2637 | 54% | 46.0% | 48.7% | 1.04 | 1.00 | -0.37% | -0.15% | 0.930 | 0.856 |
| 50 | 6617 | 51% | 47.5% | 48.7% | 1.00 | 1.00 | -0.28% | -0.15% | 0.932 | 0.820 |

**Rank on volatility alone (single factor)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 133 | 100% | 45.1% | 48.7% | 1.20 | 1.00 | -0.09% | -0.15% | 0.445 | 0.479 |
| 3 | 399 | 100% | 46.4% | 48.7% | 1.15 | 1.00 | -0.04% | -0.15% | 0.363 | 0.457 |
| 5 | 665 | 100% | 46.8% | 48.7% | 1.13 | 1.00 | -0.05% | -0.15% | 0.343 | 0.467 |
| 10 | 1330 | 99% | 47.5% | 48.7% | 1.12 | 1.00 | 0.06% | -0.15% | 0.162 | 0.317 |
| 20 | 2660 | 98% | 47.6% | 48.7% | 1.08 | 1.00 | -0.09% | -0.15% | 0.333 | 0.407 |
| 50 | 6650 | 95% | 48.1% | 48.7% | 1.07 | 1.00 | -0.03% | -0.15% | 0.094 | 0.417 |

**Rank on beta alone (single factor)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 133 | 97% | 40.6% | 48.7% | 1.17 | 1.00 | -1.19% | -0.15% | 0.942 | 0.830 |
| 3 | 399 | 96% | 47.6% | 48.7% | 1.22 | 1.00 | 0.59% | -0.15% | 0.020 | 0.178 |
| 5 | 665 | 95% | 48.9% | 48.7% | 1.27 | 1.00 | 1.08% | -0.15% | 0.002 | 0.018 |
| 10 | 1330 | 94% | 47.7% | 48.7% | 1.19 | 1.00 | 0.45% | -0.15% | 0.008 | 0.086 |
| 20 | 2660 | 90% | 48.5% | 48.7% | 1.16 | 1.00 | 0.42% | -0.15% | 0.002 | 0.008 |
| 50 | 6650 | 84% | 48.1% | 48.7% | 1.09 | 1.00 | 0.03% | -0.15% | 0.016 | 0.150 |

## 3. Ranking signal (information coefficient)

| Model | Mean IC | IC t-stat | Months |
|---|---:|---:|---:|
| XGBoost | -0.014 | -1.48 | 132 |
| Random forest | -0.016 | -1.40 | 132 |
| Ridge (auxiliary) | -0.005 | -0.33 | 132 |
| xgb classifier (median label, top-N) | -0.015 | -1.60 | 133 |
| Rank on volatility alone (single factor) | -0.007 | -0.36 | 133 |
| Rank on beta alone (single factor) | -0.008 | -0.37 | 133 |

*Spearman correlation between the predicted and the realised target, per month.*

## 4. Portfolio (monthly, net of costs)

| Model | Fallback | CAGR | Sharpe | Max DD | Months invested | Avg names when invested | Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | 5.57% | 0.34 | -42.59% | 99% | 5.0 | 1.84 |
| XGBoost | bil | 6.17% | 0.36 | -42.59% | 99% | 5.0 | 1.84 |
| XGBoost | trend | 5.57% | 0.34 | -42.59% | 99% | 5.0 | 1.84 |
| Random forest | spy | 10.31% | 0.48 | -34.37% | 99% | 5.0 | 1.86 |
| Random forest | bil | 10.94% | 0.50 | -34.37% | 99% | 5.0 | 1.86 |
| Random forest | trend | 10.31% | 0.48 | -34.37% | 99% | 5.0 | 1.86 |
| Ridge (auxiliary) | spy | 5.52% | 0.33 | -33.35% | 99% | 5.0 | 1.74 |
| Ridge (auxiliary) | bil | 6.12% | 0.36 | -33.35% | 99% | 5.0 | 1.74 |
| Ridge (auxiliary) | trend | 5.52% | 0.33 | -33.35% | 99% | 5.0 | 1.74 |
| xgb classifier (median label, top-N) | spy | 0.85% | 0.15 | -42.34% | 100% | 5.0 | 1.80 |
| xgb classifier (median label, top-N) | bil | 0.85% | 0.15 | -42.34% | 100% | 5.0 | 1.80 |
| xgb classifier (median label, top-N) | trend | 0.85% | 0.15 | -42.34% | 100% | 5.0 | 1.80 |
| Rank on volatility alone (single factor) | spy | 8.10% | 0.39 | -70.90% | 100% | 5.0 | 0.72 |
| Rank on volatility alone (single factor) | bil | 8.10% | 0.39 | -70.90% | 100% | 5.0 | 0.72 |
| Rank on volatility alone (single factor) | trend | 8.10% | 0.39 | -70.90% | 100% | 5.0 | 0.72 |
| Rank on beta alone (single factor) | spy | 33.78% | 0.85 | -39.16% | 100% | 5.0 | 0.39 |
| Rank on beta alone (single factor) | bil | 33.78% | 0.85 | -39.16% | 100% | 5.0 | 0.39 |
| Rank on beta alone (single factor) | trend | 33.78% | 0.85 | -39.16% | 100% | 5.0 | 0.39 |
| SPY (buy & hold) | - | 14.23% | 0.96 | -23.93% | - | - | - |
| BIL (buy & hold) | - | 2.05% | 3.59 | -0.16% | - | - | - |

*`spy`: hold SPY in months with no pick, so those months add zero active return and the result isolates the stock picks. `bil`: hold Treasury bills instead. `trend`: SPY if its month-end close is above its 10-month average, else bills; this is a separate timing rule. Sharpe uses a zero risk-free rate, as in the other reports.*

## 5. Risk-adjusted: CAPM net of the risk-free rate

`r - rf = alpha + beta * (r_SPY - rf)`, monthly, Newey-West t-stats (3 lags), rf = BIL. |t| above about 2 is the usual bar.

| Model | Fallback | Alpha / yr | t | Beta | t (beta vs 1) | R² | Alpha 1st half (t) | Alpha 2nd half (t) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | -5.33% | -0.77 | 0.93 | -0.69 | 0.33 | -8.48% (-1.40) | -2.21% (-0.18) |
| XGBoost | bil | -4.57% | -0.66 | 0.91 | -0.85 | 0.32 | -6.94% (-1.12) | -2.21% (-0.18) |
| XGBoost | trend | -5.33% | -0.77 | 0.93 | -0.69 | 0.33 | -8.48% (-1.40) | -2.21% (-0.18) |
| Random forest | spy | -1.25% | -0.16 | 1.04 | 0.33 | 0.31 | -5.94% (-0.98) | 3.36% (0.23) |
| Random forest | bil | -0.49% | -0.06 | 1.02 | 0.19 | 0.30 | -4.40% (-0.71) | 3.36% (0.23) |
| Random forest | trend | -1.25% | -0.16 | 1.04 | 0.33 | 0.31 | -5.94% (-0.98) | 3.36% (0.23) |
| Ridge (auxiliary) | spy | -2.86% | -0.41 | 0.71 | -3.69 | 0.20 | -6.09% (-1.18) | 0.31% (0.02) |
| Ridge (auxiliary) | bil | -2.10% | -0.31 | 0.69 | -4.05 | 0.19 | -4.55% (-0.95) | 0.31% (0.02) |
| Ridge (auxiliary) | trend | -2.86% | -0.41 | 0.71 | -3.69 | 0.20 | -6.09% (-1.18) | 0.31% (0.02) |
| xgb classifier (median label, top-N) | spy | -11.86% | -2.37 | 1.08 | 0.64 | 0.48 | -17.50% (-3.07) | -6.30% (-0.80) |
| xgb classifier (median label, top-N) | bil | -11.86% | -2.37 | 1.08 | 0.64 | 0.48 | -17.50% (-3.07) | -6.30% (-0.80) |
| xgb classifier (median label, top-N) | trend | -11.86% | -2.37 | 1.08 | 0.64 | 0.48 | -17.50% (-3.07) | -6.30% (-0.80) |
| Rank on volatility alone (single factor) | spy | -8.90% | -0.79 | 2.08 | 3.12 | 0.42 | -4.20% (-0.35) | -13.47% (-0.77) |
| Rank on volatility alone (single factor) | bil | -8.90% | -0.79 | 2.08 | 3.12 | 0.42 | -4.20% (-0.35) | -13.47% (-0.77) |
| Rank on volatility alone (single factor) | trend | -8.90% | -0.79 | 2.08 | 3.12 | 0.42 | -4.20% (-0.35) | -13.47% (-0.77) |
| Rank on beta alone (single factor) | spy | 8.62% | 0.98 | 2.28 | 7.48 | 0.56 | 6.15% (0.48) | 11.05% (0.93) |
| Rank on beta alone (single factor) | bil | 8.62% | 0.98 | 2.28 | 7.48 | 0.56 | 6.15% (0.48) | 11.05% (0.93) |
| Rank on beta alone (single factor) | trend | 8.62% | 0.98 | 2.28 | 7.48 | 0.56 | 6.15% (0.48) | 11.05% (0.93) |

## 6. Year by year (fallback = spy)

| Year | SPY | XGBoost | Random forest | Ridge (auxiliary) | xgb classifier (median label, top-N) | Rank on volatility alone (single factor) | Rank on beta alone (single factor) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2015 | -6.9% | -6.0% | -14.3% | -12.8% | -24.3% | -29.5% | -17.2% |
| 2016 | 20.0% | 25.6% | 28.5% | 21.8% | 21.5% | 137.6% | 105.5% |
| 2017 | 26.3% | 11.2% | 10.0% | 0.2% | 3.3% | -4.9% | 18.8% |
| 2018 | -2.4% | -21.6% | 6.1% | 2.8% | -14.2% | -10.0% | -25.3% |
| 2019 | 21.4% | 26.7% | 18.2% | 14.3% | 7.6% | 40.2% | 44.9% |
| 2020 | 17.2% | -6.7% | -8.0% | -10.9% | -14.9% | 12.6% | 95.1% |
| 2021 | 23.2% | -0.9% | 7.4% | -7.6% | -1.6% | -18.1% | 45.1% |
| 2022 | -8.2% | -26.3% | -19.4% | 8.4% | -8.2% | 19.3% | 8.5% |
| 2023 | 20.6% | -11.1% | 0.3% | -11.0% | -10.2% | -12.8% | 37.7% |
| 2024 | 26.2% | 10.5% | 30.0% | 21.1% | 13.9% | -48.8% | 3.2% |
| 2025 | 16.3% | 53.1% | 58.8% | 50.1% | 19.9% | 45.8% | 86.1% |
| 2026 | 11.4% | 36.3% | 18.5% | 0.5% | 32.8% | 64.7% | 41.6% |

## 7. Latest decision

- **XGBoost** (2026-08-31): PCG, ZTS, ACGL, IQV, VZ
- **Random forest** (2026-08-31): CI, ACGL, VICI, WRB, HIG
- **Ridge (auxiliary)** (2026-08-31): INTC, EXE, CHTR, FOX, FOXA
- **xgb classifier (median label, top-N)** (2026-08-31): MRVL, ZTS, PANW, ADBE, DELL
- **Rank on volatility alone (single factor)** (2026-08-31): MRNA, SNDK, MRVL, SMCI, COHR
- **Rank on beta alone (single factor)** (2026-08-31): SNDK, SMCI, MU, COHR, LRCX

## Verdict

- **XGBoost**: picks earn -0.34% per pick against -0.16% for the same-month universe, winning 45.9% of the time vs 48.6%. Against random picks from the same months p = 0.866, so this selector did not earn more than picking at random.
- **Random forest**: picks earn 0.12% per pick against -0.16% for the same-month universe, winning 46.9% of the time vs 48.6%. Against random picks from the same months p = 0.151, so this selector did not earn more than picking at random.
- **Ridge (auxiliary)**: picks earn -0.37% per pick against -0.16% for the same-month universe, winning 46.2% of the time vs 48.7%. Against random picks from the same months p = 0.848, so this selector did not earn more than picking at random.
- **xgb classifier (median label, top-N)**: picks earn -0.73% per pick against -0.15% for the same-month universe, winning 43.3% of the time vs 48.7%. Against random picks from the same months p = 0.993, so this selector did not earn more than picking at random.
- **Rank on volatility alone (single factor)**: picks earn 0.46% per pick against -0.15% for the same-month universe, winning 46.8% of the time vs 48.7%. Against random picks from the same months p = 0.373, so this selector did not earn more than picking at random.
- **Rank on beta alone (single factor)**: picks earn 2.09% per pick against -0.15% for the same-month universe, winning 48.9% of the time vs 48.7%, beating random picks from the same months at p = 0.000. It also beats volatility-matched stand-ins (p = 0.016), so the advantage is not only the risk it takes.

The universe is the point-in-time S&P 500: each month only the stocks that were in the index on that date. Members whose prices yfinance no longer has (mostly companies later acquired or bankrupt) are missing; see the membership coverage above. Acquired names tend to have risen and bankrupt ones to have fallen, so the remaining bias is smaller but its sign is not certain. Out-of-sample research evaluation only; not financial advice.
