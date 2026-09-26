# Hurdle Selection on a Risk-Adjusted Excess-Return Target

Generated at: 2026-09-24 12:26:26

## Configuration

- Universe: today's S&P 500 list (503 tickers) for every month
- Membership coverage: n/a (fixed universe)
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
    - technical: mom_1m 99%, mom_3m 98%, mom_6m 96%, mom_12m 91%, reversal_5d 100%, mom_risk_adj 91%, high_52w 100%, volatility 99%, macd 100%, adx 99%
    - fundamental: pe 23%, pb 29%, roe 25%, earnings_growth 15%, net_margin 31%, debt_to_equity 30%
- Target rows: 64169, mean 0.012, share above zero 50.0%

**Target.** `y = (r_stock - beta * r_SPY) / vol` over the next month, where beta (Blume-adjusted, 252 trading days) and vol (63 days, monthly scale) use only data up to the rebalance date, winsorised at the 1st/99th percentile of each month. Subtracting beta x SPY means a stock cannot score well just by being high-beta in a rising market; dividing by vol puts the payoff in units of risk taken.

**Rule.** Buy the names whose predicted target is above the hurdle, best first, up to the cap, equal-weighted among themselves. If none qualify, hold the fallback. The hurdle is fixed before the run, not tuned on it.

## 1. The picks: win rate, payoff ratio, expectancy

Each pick is judged over the month it was held, two ways. **Raw** = its return minus SPY's, what a trader sees. **Beta-adjusted** = its return minus beta x SPY's (beta estimated before the pick), which a high-beta stock cannot win just because the market rose. The universe column is every stock in the same months, weighted by how many picks were made that month.

Two random benchmarks replace each month's picks with the same number of names from that month. **`p (random)` is the primary test**: stand-ins are drawn from the whole cross-section, so it answers the question a user asks — did these picks earn more than picking at random? `p (vol-matched)` draws each stand-in from the 20 names nearest the pick in volatility, which by construction removes any return earned by holding volatile names; it therefore answers the narrower question of whether anything remains *beyond* beta and volatility, and a selector that works by taking risk is not failed for losing it. p is the share of 2,000 draws that did at least as well; below 0.05 is the usual bar. The two single-factor rows rank on one measure with nothing fitted.

**Raw, minus SPY (primary)**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 659 | 64% | 49.2% | 49.6% | 1.49 | 1.09 | 1.56% | 0.19% | 1.03% | 0.000 | 0.085 |
| Random forest | 660 | 59% | 49.1% | 49.6% | 1.55 | 1.09 | 1.67% | 0.19% | 0.78% | 0.000 | 0.017 |
| Ridge (auxiliary) | 660 | 66% | 53.2% | 49.6% | 1.34 | 1.09 | 1.68% | 0.19% | 0.76% | 0.000 | 0.010 |
| xgb classifier (median label, top-N) | 665 | 70% | 46.9% | 49.6% | 1.32 | 1.08 | 0.63% | 0.19% | 0.70% | 0.073 | 0.565 |
| Rank on volatility alone (single factor) | 665 | 100% | 52.2% | 49.6% | 1.59 | 1.08 | 3.95% | 0.19% | 2.52% | 0.000 | 0.006 |
| Rank on beta alone (single factor) | 665 | 96% | 54.7% | 49.6% | 1.46 | 1.08 | 3.88% | 0.19% | 2.16% | 0.000 | 0.000 |

**Beta-adjusted**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 622 | 64% | 49.0% | 49.9% | 1.37 | 1.06 | 1.12% | 0.17% | 0.46% | 0.001 | 0.055 |
| Random forest | 658 | 59% | 48.9% | 50.1% | 1.53 | 1.07 | 1.54% | 0.19% | 0.63% | 0.000 | 0.013 |
| Ridge (auxiliary) | 656 | 66% | 52.3% | 50.1% | 1.29 | 1.07 | 1.35% | 0.19% | 0.53% | 0.001 | 0.019 |
| xgb classifier (median label, top-N) | 636 | 70% | 46.4% | 50.2% | 1.28 | 1.07 | 0.40% | 0.19% | 0.53% | 0.258 | 0.641 |
| Rank on volatility alone (single factor) | 665 | 100% | 52.2% | 50.1% | 1.45 | 1.07 | 3.20% | 0.19% | 1.99% | 0.000 | 0.016 |
| Rank on beta alone (single factor) | 665 | 96% | 52.8% | 50.1% | 1.36 | 1.07 | 2.75% | 0.19% | 1.65% | 0.000 | 0.023 |

*Payoff ratio = average winning excess return / average losing excess return (the realised risk-reward, no stop-loss or take-profit). Expectancy = mean excess return per pick. Vol percentile of picks = where the picks sat in that month's volatility ranking (50% = no tilt, 100% = always the most volatile). Vol-matched random = the average expectancy of random stand-ins drawn from the 20 names nearest each pick in volatility, i.e. what the picks' risk profile alone would have earned.*

### 1b. Why per-pick excess and compounded return can disagree

The expectancy above is an **arithmetic** mean per pick. An investor earns the **compounded** return, and for small returns the two differ by about half the variance, so a concentrated book gives more back to that term than the index does. The table separates the two; a small residual means the approximation accounts for the gap. All figures annualised, against SPY, on the `spy` fallback variant.

| Model | Arithmetic excess | Variance drag | Residual | Compounded excess | Book vol | SPY vol |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 16.51% | −5.10% | 3.18% | 14.59% | 35.4% | 15.2% |
| Random forest | 17.83% | −5.51% | 3.19% | 15.51% | 36.5% | 15.2% |
| Ridge (auxiliary) | 17.99% | −3.33% | 3.58% | 18.24% | 29.9% | 15.2% |
| xgb classifier (median label, top-N) | 5.47% | −3.69% | 0.39% | 2.17% | 31.1% | 15.2% |
| Rank on volatility alone (single factor) | 46.63% | −14.24% | 12.84% | 45.24% | 55.5% | 15.2% |
| Rank on beta alone (single factor) | 46.19% | −13.91% | 12.61% | 44.89% | 54.9% | 15.2% |

*Arithmetic excess − variance drag + residual = compounded excess. The drag is not a fault in the selector: it is the arithmetic of compounding a more volatile series, and it shrinks if the book is held wider.*

## 2. Does stricter selection do better? (beta-adjusted)

Top-k by score each month, ignoring the hurdle. If the score orders stocks usefully, the pick columns improve as k shrinks — the best-scored few should earn more than the best-scored many.

**XGBoost**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 124 | 67% | 52.4% | 50.2% | 1.17 | 1.06 | 0.99% | 0.17% | 0.104 | 0.206 |
| 3 | 369 | 65% | 51.8% | 50.0% | 1.40 | 1.06 | 1.68% | 0.17% | 0.002 | 0.010 |
| 5 | 627 | 63% | 49.0% | 50.0% | 1.37 | 1.06 | 1.09% | 0.17% | 0.002 | 0.070 |
| 10 | 1278 | 58% | 47.7% | 50.1% | 1.32 | 1.06 | 0.68% | 0.18% | 0.012 | 0.124 |
| 20 | 2593 | 56% | 46.8% | 50.1% | 1.20 | 1.07 | 0.19% | 0.19% | 0.503 | 0.709 |
| 50 | 6559 | 53% | 48.2% | 50.1% | 1.15 | 1.07 | 0.20% | 0.19% | 0.449 | 0.547 |

**Random forest**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 133 | 59% | 52.6% | 50.1% | 1.39 | 1.07 | 1.97% | 0.19% | 0.010 | 0.088 |
| 3 | 397 | 58% | 48.1% | 50.1% | 1.47 | 1.07 | 1.25% | 0.19% | 0.008 | 0.162 |
| 5 | 663 | 59% | 48.9% | 50.1% | 1.52 | 1.07 | 1.51% | 0.19% | 0.002 | 0.018 |
| 10 | 1326 | 58% | 47.8% | 50.1% | 1.34 | 1.07 | 0.80% | 0.19% | 0.014 | 0.190 |
| 20 | 2651 | 56% | 48.8% | 50.1% | 1.21 | 1.07 | 0.51% | 0.19% | 0.016 | 0.251 |
| 50 | 6626 | 52% | 48.0% | 50.1% | 1.14 | 1.07 | 0.17% | 0.19% | 0.617 | 0.707 |

**Ridge (auxiliary)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 132 | 68% | 45.5% | 50.1% | 1.29 | 1.07 | 0.30% | 0.19% | 0.429 | 0.687 |
| 3 | 397 | 66% | 49.6% | 50.1% | 1.48 | 1.07 | 1.57% | 0.19% | 0.002 | 0.044 |
| 5 | 661 | 65% | 52.2% | 50.1% | 1.29 | 1.07 | 1.32% | 0.19% | 0.002 | 0.022 |
| 10 | 1322 | 63% | 52.0% | 50.1% | 1.22 | 1.07 | 1.03% | 0.19% | 0.002 | 0.008 |
| 20 | 2646 | 60% | 52.1% | 50.1% | 1.19 | 1.07 | 0.87% | 0.19% | 0.002 | 0.010 |
| 50 | 6622 | 55% | 50.4% | 50.1% | 1.18 | 1.07 | 0.57% | 0.19% | 0.002 | 0.002 |

**xgb classifier (median label, top-N)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 122 | 75% | 52.5% | 50.3% | 1.24 | 1.06 | 1.28% | 0.21% | 0.068 | 0.188 |
| 3 | 377 | 72% | 44.3% | 50.2% | 1.38 | 1.06 | 0.38% | 0.19% | 0.287 | 0.557 |
| 5 | 636 | 70% | 46.4% | 50.2% | 1.28 | 1.07 | 0.40% | 0.19% | 0.238 | 0.691 |
| 10 | 1291 | 64% | 46.6% | 50.1% | 1.19 | 1.07 | 0.15% | 0.19% | 0.517 | 0.784 |
| 20 | 2613 | 61% | 47.1% | 50.1% | 1.16 | 1.07 | 0.13% | 0.19% | 0.665 | 0.856 |
| 50 | 6581 | 56% | 48.4% | 50.1% | 1.11 | 1.07 | 0.13% | 0.19% | 0.762 | 0.870 |

**Rank on volatility alone (single factor)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 133 | 100% | 52.6% | 50.1% | 1.56 | 1.07 | 5.16% | 0.19% | 0.002 | 0.008 |
| 3 | 399 | 100% | 52.1% | 50.1% | 1.51 | 1.07 | 3.87% | 0.19% | 0.002 | 0.012 |
| 5 | 665 | 100% | 52.2% | 50.1% | 1.45 | 1.07 | 3.20% | 0.19% | 0.002 | 0.022 |
| 10 | 1330 | 99% | 52.0% | 50.1% | 1.36 | 1.07 | 2.41% | 0.19% | 0.002 | 0.178 |
| 20 | 2660 | 98% | 51.6% | 50.1% | 1.33 | 1.07 | 1.99% | 0.19% | 0.002 | 0.200 |
| 50 | 6650 | 95% | 51.4% | 50.1% | 1.24 | 1.07 | 1.27% | 0.19% | 0.002 | 0.248 |

**Rank on beta alone (single factor)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 133 | 97% | 48.1% | 50.1% | 1.28 | 1.07 | 1.41% | 0.19% | 0.034 | 0.527 |
| 3 | 399 | 97% | 52.6% | 50.1% | 1.34 | 1.07 | 2.81% | 0.19% | 0.002 | 0.052 |
| 5 | 665 | 96% | 52.8% | 50.1% | 1.36 | 1.07 | 2.75% | 0.19% | 0.002 | 0.030 |
| 10 | 1330 | 95% | 51.4% | 50.1% | 1.37 | 1.07 | 2.17% | 0.19% | 0.002 | 0.014 |
| 20 | 2660 | 92% | 51.1% | 50.1% | 1.30 | 1.07 | 1.59% | 0.19% | 0.002 | 0.020 |
| 50 | 6650 | 86% | 50.6% | 50.1% | 1.21 | 1.07 | 0.92% | 0.19% | 0.002 | 0.160 |

## 3. Ranking signal (information coefficient)

| Model | Mean IC | IC t-stat | Months |
|---|---:|---:|---:|
| XGBoost | -0.013 | -1.36 | 132 |
| Random forest | -0.014 | -1.27 | 132 |
| Ridge (auxiliary) | -0.005 | -0.42 | 132 |
| xgb classifier (median label, top-N) | -0.016 | -1.84 | 133 |
| Rank on volatility alone (single factor) | 0.010 | 0.53 | 133 |
| Rank on beta alone (single factor) | 0.003 | 0.13 | 133 |

*Spearman correlation between the predicted and the realised target, per month.*

## 4. Portfolio (monthly, net of costs)

| Model | Fallback | CAGR | Sharpe | Max DD | Months invested | Avg names when invested | Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | 28.82% | 0.88 | -33.50% | 99% | 5.0 | 1.72 |
| XGBoost | bil | 29.56% | 0.90 | -33.50% | 99% | 5.0 | 1.72 |
| XGBoost | trend | 28.82% | 0.88 | -33.50% | 99% | 5.0 | 1.72 |
| Random forest | spy | 29.74% | 0.89 | -37.10% | 99% | 5.0 | 1.71 |
| Random forest | bil | 30.48% | 0.90 | -37.10% | 99% | 5.0 | 1.71 |
| Random forest | trend | 29.74% | 0.89 | -37.10% | 99% | 5.0 | 1.71 |
| Ridge (auxiliary) | spy | 32.47% | 1.09 | -23.75% | 99% | 5.0 | 1.69 |
| Ridge (auxiliary) | bil | 33.22% | 1.11 | -23.75% | 99% | 5.0 | 1.69 |
| Ridge (auxiliary) | trend | 32.47% | 1.09 | -23.75% | 99% | 5.0 | 1.69 |
| xgb classifier (median label, top-N) | spy | 16.40% | 0.64 | -44.15% | 100% | 5.0 | 1.76 |
| xgb classifier (median label, top-N) | bil | 16.40% | 0.64 | -44.15% | 100% | 5.0 | 1.76 |
| xgb classifier (median label, top-N) | trend | 16.40% | 0.64 | -44.15% | 100% | 5.0 | 1.76 |
| Rank on volatility alone (single factor) | spy | 59.47% | 1.10 | -75.88% | 100% | 5.0 | 0.63 |
| Rank on volatility alone (single factor) | bil | 59.47% | 1.10 | -75.88% | 100% | 5.0 | 0.63 |
| Rank on volatility alone (single factor) | trend | 59.47% | 1.10 | -75.88% | 100% | 5.0 | 0.63 |
| Rank on beta alone (single factor) | spy | 59.12% | 1.11 | -71.96% | 100% | 5.0 | 0.35 |
| Rank on beta alone (single factor) | bil | 59.12% | 1.11 | -71.96% | 100% | 5.0 | 0.35 |
| Rank on beta alone (single factor) | trend | 59.12% | 1.11 | -71.96% | 100% | 5.0 | 0.35 |
| SPY (buy & hold) | - | 14.23% | 0.96 | -23.93% | - | - | - |
| BIL (buy & hold) | - | 2.05% | 3.59 | -0.16% | - | - | - |

*`spy`: hold SPY in months with no pick, so those months add zero active return and the result isolates the stock picks. `bil`: hold Treasury bills instead. `trend`: SPY if its month-end close is above its 10-month average, else bills; this is a separate timing rule. Sharpe uses a zero risk-free rate, as in the other reports.*

## 5. Risk-adjusted: CAPM net of the risk-free rate

`r - rf = alpha + beta * (r_SPY - rf)`, monthly, Newey-West t-stats (3 lags), rf = BIL. |t| above about 2 is the usual bar.

| Model | Fallback | Alpha / yr | t | Beta | t (beta vs 1) | R² | Alpha 1st half (t) | Alpha 2nd half (t) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | 13.58% | 1.50 | 1.23 | 1.19 | 0.28 | 11.24% (1.65) | 15.91% (0.96) |
| XGBoost | bil | 14.34% | 1.58 | 1.22 | 1.10 | 0.28 | 12.77% (1.78) | 15.91% (0.96) |
| XGBoost | trend | 13.58% | 1.50 | 1.23 | 1.19 | 0.28 | 11.24% (1.65) | 15.91% (0.96) |
| Random forest | spy | 13.65% | 1.35 | 1.33 | 1.74 | 0.31 | 5.87% (1.06) | 21.28% (1.11) |
| Random forest | bil | 14.42% | 1.42 | 1.32 | 1.65 | 0.30 | 7.40% (1.24) | 21.28% (1.11) |
| Random forest | trend | 13.65% | 1.35 | 1.33 | 1.74 | 0.31 | 5.87% (1.06) | 21.28% (1.11) |
| Ridge (auxiliary) | spy | 17.52% | 2.07 | 1.04 | 0.27 | 0.28 | 13.23% (1.81) | 21.75% (1.45) |
| Ridge (auxiliary) | bil | 18.29% | 2.16 | 1.02 | 0.14 | 0.27 | 14.77% (2.03) | 21.75% (1.45) |
| Ridge (auxiliary) | trend | 17.52% | 2.07 | 1.04 | 0.27 | 0.28 | 13.23% (1.81) | 21.75% (1.45) |
| xgb classifier (median label, top-N) | spy | 1.41% | 0.22 | 1.32 | 2.10 | 0.42 | -1.03% (-0.12) | 3.81% (0.41) |
| xgb classifier (median label, top-N) | bil | 1.41% | 0.22 | 1.32 | 2.10 | 0.42 | -1.03% (-0.12) | 3.81% (0.41) |
| xgb classifier (median label, top-N) | trend | 1.41% | 0.22 | 1.32 | 2.10 | 0.42 | -1.03% (-0.12) | 3.81% (0.41) |
| Rank on volatility alone (single factor) | spy | 28.88% | 2.32 | 2.42 | 5.29 | 0.44 | 38.32% (2.43) | 19.57% (1.04) |
| Rank on volatility alone (single factor) | bil | 28.88% | 2.32 | 2.42 | 5.29 | 0.44 | 38.32% (2.43) | 19.57% (1.04) |
| Rank on volatility alone (single factor) | trend | 28.88% | 2.32 | 2.42 | 5.29 | 0.44 | 38.32% (2.43) | 19.57% (1.04) |
| Rank on beta alone (single factor) | spy | 27.66% | 2.47 | 2.48 | 6.37 | 0.47 | 17.32% (1.29) | 37.83% (2.16) |
| Rank on beta alone (single factor) | bil | 27.66% | 2.47 | 2.48 | 6.37 | 0.47 | 17.32% (1.29) | 37.83% (2.16) |
| Rank on beta alone (single factor) | trend | 27.66% | 2.47 | 2.48 | 6.37 | 0.47 | 17.32% (1.29) | 37.83% (2.16) |

## 6. Year by year (fallback = spy)

| Year | SPY | XGBoost | Random forest | Ridge (auxiliary) | xgb classifier (median label, top-N) | Rank on volatility alone (single factor) | Rank on beta alone (single factor) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2015 | -6.9% | -7.0% | -7.3% | -9.5% | -21.7% | -34.3% | -26.9% |
| 2016 | 20.0% | 46.0% | 28.4% | 35.7% | 38.5% | 172.2% | 143.9% |
| 2017 | 26.3% | 29.9% | 20.3% | 20.9% | 42.9% | 75.8% | 32.6% |
| 2018 | -2.4% | -7.6% | -7.7% | 13.4% | -8.1% | 67.8% | -22.2% |
| 2019 | 21.4% | 49.0% | 29.4% | 15.2% | 9.6% | 70.1% | 73.8% |
| 2020 | 17.2% | 56.5% | 52.9% | 101.3% | 21.7% | 141.9% | 153.4% |
| 2021 | 23.2% | -6.4% | 0.2% | 5.1% | 20.0% | 5.0% | 68.5% |
| 2022 | -8.2% | -10.7% | -26.2% | 19.6% | -15.2% | -48.6% | -48.7% |
| 2023 | 20.6% | 29.1% | 28.4% | 5.3% | 37.4% | 99.2% | 137.3% |
| 2024 | 26.2% | 62.2% | 73.3% | 81.1% | 31.5% | 91.1% | 100.2% |
| 2025 | 16.3% | 120.6% | 180.7% | 185.7% | 46.4% | 89.9% | 197.2% |
| 2026 | 11.4% | 12.6% | 48.6% | -15.7% | 5.3% | 108.2% | 74.1% |

## 7. Latest decision

- **XGBoost** (2026-08-31): PANW, F, AVB, PCG, SNPS
- **Random forest** (2026-08-31): MRNA, ACGL, PANW, INTC, CI
- **Ridge (auxiliary)** (2026-08-31): INTC, F, EXE, FOXA, FOX
- **xgb classifier (median label, top-N)** (2026-08-31): MRNA, AVB, INTU, CPRT, PANW
- **Rank on volatility alone (single factor)** (2026-08-31): MRNA, SNDK, SMCI, COHR, LITE
- **Rank on beta alone (single factor)** (2026-08-31): SNDK, SMCI, MU, COHR, LRCX

## Verdict

- **XGBoost**: picks earn 1.56% per pick against 0.17% for the same-month universe, winning 49.0% of the time vs 49.9%, beating random picks from the same months at p = 0.001. Against volatility-matched stand-ins p = 0.055, so most of the advantage comes through beta and volatility.
- **Random forest**: picks earn 1.67% per pick against 0.19% for the same-month universe, winning 48.9% of the time vs 50.1%, beating random picks from the same months at p = 0.000. It also beats volatility-matched stand-ins (p = 0.013), so the advantage is not only the risk it takes.
- **Ridge (auxiliary)**: picks earn 1.68% per pick against 0.19% for the same-month universe, winning 52.3% of the time vs 50.1%, beating random picks from the same months at p = 0.001. It also beats volatility-matched stand-ins (p = 0.019), so the advantage is not only the risk it takes.
- **xgb classifier (median label, top-N)**: picks earn 0.63% per pick against 0.19% for the same-month universe, winning 46.4% of the time vs 50.2%. Against random picks from the same months p = 0.258, so this selector did not earn more than picking at random.
- **Rank on volatility alone (single factor)**: picks earn 3.95% per pick against 0.19% for the same-month universe, winning 52.2% of the time vs 50.1%, beating random picks from the same months at p = 0.000. It also beats volatility-matched stand-ins (p = 0.016), so the advantage is not only the risk it takes.
- **Rank on beta alone (single factor)**: picks earn 3.88% per pick against 0.19% for the same-month universe, winning 52.8% of the time vs 50.1%, beating random picks from the same months at p = 0.000. It also beats volatility-matched stand-ins (p = 0.023), so the advantage is not only the risk it takes.

The universe is today's S&P 500 constituent list, so names that were later dropped are missing from the history (survivorship bias); every return above is likely flattered by that. Out-of-sample research evaluation only; not financial advice.
