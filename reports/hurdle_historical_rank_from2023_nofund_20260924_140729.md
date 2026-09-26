# Hurdle Selection on a Risk-Adjusted Excess-Return Target

Generated at: 2026-09-24 14:07:29

## Configuration

- Universe: point-in-time S&P 500 (each month's actual members)
- Membership coverage: 742 tickers were members at some point; 118 have no price data. Share of each month's members with prices: median 91%, lowest 76% (2015-01), highest 100%
- History: 2015-01-01 to latest, 44 monthly rebalances
- Training window: 12 rebalances rolling
- Models: XGBoost, Random forest, Ridge (auxiliary)
- Hurdle: predicted target > 0.0
- Max names: 5 (equal weight among those that qualify)
- Fallbacks: spy, bil, trend
- Transaction cost: 10 bps per unit turnover (fallback switches included)
- Feature normalisation: rank
- Features: technical,fundamental (10 columns: mom_1m, mom_3m, mom_6m, mom_12m, reversal_5d, mom_risk_adj, high_52w, volatility, macd, adx)
- Training target: (r - beta x r_SPY) / vol, winsorised per month
- Walk-forward starts: 2023-01-01 (prices from 2015-01-01)
- Feature coverage (non-NaN cells, before normalisation):
    - technical: mom_1m 100%, mom_3m 100%, mom_6m 100%, mom_12m 100%, reversal_5d 100%, mom_risk_adj 100%, high_52w 100%, volatility 100%, macd 100%, adx 100%
    - fundamental: 
- Target rows: 20997, mean -0.053, share above zero 46.1%

**Target.** `y = (r_stock - beta * r_SPY) / vol` over the next month, where beta (Blume-adjusted, 252 trading days) and vol (63 days, monthly scale) use only data up to the rebalance date, winsorised at the 1st/99th percentile of each month. Subtracting beta x SPY means a stock cannot score well just by being high-beta in a rising market; dividing by vol puts the payoff in units of risk taken.

**Rule.** Buy the names whose predicted target is above the hurdle, best first, up to the cap, equal-weighted among themselves. If none qualify, hold the fallback. The hurdle is fixed before the run, not tuned on it.

## 1. The picks: win rate, payoff ratio, expectancy

Each pick is judged over the month it was held, two ways. **Raw** = its return minus SPY's, what a trader sees. **Beta-adjusted** = its return minus beta x SPY's (beta estimated before the pick), which a high-beta stock cannot win just because the market rose. The universe column is every stock in the same months, weighted by how many picks were made that month.

Two random benchmarks replace each month's picks with the same number of names from that month. **`p (random)` is the primary test**: stand-ins are drawn from the whole cross-section, so it answers the question a user asks — did these picks earn more than picking at random? `p (vol-matched)` draws each stand-in from the 20 names nearest the pick in volatility, which by construction removes any return earned by holding volatile names; it therefore answers the narrower question of whether anything remains *beyond* beta and volatility, and a selector that works by taking risk is not failed for losing it. p is the share of 2,000 draws that did at least as well; below 0.05 is the usual bar. The two single-factor rows rank on one measure with nothing fitted.

**Raw, minus SPY (primary)**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 184 | 51% | 50.5% | 45.8% | 1.03 | 1.03 | 0.21% | -0.44% | 0.23% | 0.163 | 0.503 |
| Random forest | 185 | 52% | 49.7% | 45.8% | 1.25 | 1.03 | 0.92% | -0.44% | 0.31% | 0.024 | 0.230 |
| Ridge (auxiliary) | 185 | 65% | 51.4% | 45.8% | 1.03 | 1.03 | 0.37% | -0.44% | -0.31% | 0.110 | 0.218 |
| Rank on volatility alone (single factor) | 185 | 100% | 46.5% | 45.8% | 1.12 | 1.03 | -0.21% | -0.44% | 0.17% | 0.361 | 0.625 |
| Rank on beta alone (single factor) | 185 | 96% | 46.5% | 45.8% | 1.43 | 1.03 | 1.44% | -0.44% | 0.65% | 0.004 | 0.227 |

**Beta-adjusted**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 176 | 51% | 51.1% | 46.8% | 0.95 | 1.01 | -0.03% | -0.36% | 0.01% | 0.294 | 0.501 |
| Random forest | 185 | 52% | 49.2% | 47.0% | 1.22 | 1.02 | 0.69% | -0.31% | 0.15% | 0.056 | 0.256 |
| Ridge (auxiliary) | 184 | 65% | 51.6% | 47.0% | 0.94 | 1.02 | 0.04% | -0.30% | -0.51% | 0.303 | 0.261 |
| Rank on volatility alone (single factor) | 185 | 100% | 45.4% | 47.0% | 1.04 | 1.02 | -1.05% | -0.31% | -0.56% | 0.890 | 0.664 |
| Rank on beta alone (single factor) | 185 | 96% | 42.7% | 47.0% | 1.33 | 1.02 | -0.08% | -0.31% | -0.09% | 0.353 | 0.489 |

*Payoff ratio = average winning excess return / average losing excess return (the realised risk-reward, no stop-loss or take-profit). Expectancy = mean excess return per pick. Vol percentile of picks = where the picks sat in that month's volatility ranking (50% = no tilt, 100% = always the most volatile). Vol-matched random = the average expectancy of random stand-ins drawn from the 20 names nearest each pick in volatility, i.e. what the picks' risk profile alone would have earned.*

### 1b. Why per-pick excess and compounded return can disagree

The expectancy above is an **arithmetic** mean per pick. An investor earns the **compounded** return, and for small returns the two differ by about half the variance, so a concentrated book gives more back to that term than the index does. The table separates the two; a small residual means the approximation accounts for the gap. All figures annualised, against SPY, on the `spy` fallback variant.

| Model | Arithmetic excess | Variance drag | Residual | Compounded excess | Book vol | SPY vol |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.48% | −5.48% | 0.11% | -4.88% | 35.5% | 12.9% |
| Random forest | 8.89% | −8.34% | 1.69% | 2.23% | 42.8% | 12.9% |
| Ridge (auxiliary) | 2.28% | −6.64% | -0.69% | -5.05% | 38.7% | 12.9% |
| Rank on volatility alone (single factor) | -3.41% | −11.86% | -0.49% | -15.76% | 50.4% | 12.9% |
| Rank on beta alone (single factor) | 16.75% | −10.70% | 2.53% | 8.58% | 48.0% | 12.9% |

*Arithmetic excess − variance drag + residual = compounded excess. The drag is not a fault in the selector: it is the arithmetic of compounding a more volatile series, and it shrinks if the book is held wider.*

## 2. Does stricter selection do better? (beta-adjusted)

Top-k by score each month, ignoring the hurdle. If the score orders stocks usefully, the pick columns improve as k shrinks — the best-scored few should earn more than the best-scored many.

**XGBoost**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 44% | 59.4% | 45.6% | 0.94 | 0.97 | 0.99% | -0.61% | 0.140 | 0.267 |
| 3 | 103 | 47% | 54.4% | 46.8% | 0.75 | 1.00 | -0.47% | -0.38% | 0.571 | 0.679 |
| 5 | 176 | 51% | 51.1% | 46.8% | 0.95 | 1.01 | -0.03% | -0.36% | 0.281 | 0.507 |
| 10 | 359 | 52% | 49.0% | 46.9% | 0.85 | 1.02 | -0.77% | -0.34% | 0.822 | 0.892 |
| 20 | 727 | 53% | 47.7% | 46.9% | 1.02 | 1.02 | -0.25% | -0.33% | 0.381 | 0.571 |
| 50 | 1836 | 52% | 47.9% | 46.9% | 1.08 | 1.02 | -0.04% | -0.32% | 0.062 | 0.188 |

**Random forest**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37 | 57% | 45.9% | 47.0% | 1.42 | 1.02 | 0.83% | -0.31% | 0.198 | 0.279 |
| 3 | 111 | 53% | 53.2% | 47.0% | 1.15 | 1.02 | 1.03% | -0.31% | 0.050 | 0.192 |
| 5 | 185 | 52% | 49.2% | 47.0% | 1.22 | 1.02 | 0.69% | -0.31% | 0.048 | 0.287 |
| 10 | 370 | 52% | 49.7% | 47.0% | 1.20 | 1.02 | 0.69% | -0.31% | 0.020 | 0.084 |
| 20 | 740 | 53% | 48.5% | 47.0% | 1.10 | 1.02 | 0.14% | -0.31% | 0.072 | 0.263 |
| 50 | 1848 | 51% | 46.8% | 47.0% | 1.08 | 1.02 | -0.18% | -0.31% | 0.251 | 0.341 |

**Ridge (auxiliary)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37 | 67% | 45.9% | 47.0% | 0.78 | 1.02 | -2.12% | -0.31% | 0.910 | 0.828 |
| 3 | 110 | 67% | 50.9% | 47.0% | 0.96 | 1.02 | -0.01% | -0.30% | 0.387 | 0.313 |
| 5 | 184 | 65% | 51.6% | 47.0% | 0.94 | 1.02 | 0.04% | -0.30% | 0.287 | 0.250 |
| 10 | 368 | 62% | 48.6% | 47.0% | 1.02 | 1.02 | -0.14% | -0.30% | 0.373 | 0.521 |
| 20 | 737 | 60% | 48.7% | 47.0% | 1.03 | 1.02 | -0.08% | -0.31% | 0.250 | 0.319 |
| 50 | 1844 | 57% | 46.5% | 47.0% | 1.08 | 1.02 | -0.24% | -0.31% | 0.331 | 0.497 |

**Rank on volatility alone (single factor)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37 | 100% | 40.5% | 47.0% | 1.26 | 1.02 | -1.75% | -0.31% | 0.872 | 0.669 |
| 3 | 111 | 100% | 44.1% | 47.0% | 0.96 | 1.02 | -2.26% | -0.31% | 0.994 | 0.900 |
| 5 | 185 | 100% | 45.4% | 47.0% | 1.04 | 1.02 | -1.05% | -0.31% | 0.894 | 0.685 |
| 10 | 370 | 99% | 47.8% | 47.0% | 0.99 | 1.02 | -0.63% | -0.31% | 0.772 | 0.545 |
| 20 | 740 | 98% | 47.3% | 47.0% | 1.02 | 1.02 | -0.54% | -0.31% | 0.780 | 0.523 |
| 50 | 1850 | 95% | 47.4% | 47.0% | 1.11 | 1.02 | 0.02% | -0.31% | 0.030 | 0.447 |

**Rank on beta alone (single factor)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37 | 98% | 27.0% | 47.0% | 1.81 | 1.02 | -2.91% | -0.31% | 0.972 | 0.820 |
| 3 | 111 | 97% | 38.7% | 47.0% | 1.41 | 1.02 | -0.73% | -0.31% | 0.687 | 0.675 |
| 5 | 185 | 96% | 42.7% | 47.0% | 1.33 | 1.02 | -0.08% | -0.31% | 0.321 | 0.473 |
| 10 | 370 | 95% | 44.1% | 47.0% | 1.21 | 1.02 | -0.29% | -0.31% | 0.481 | 0.601 |
| 20 | 740 | 93% | 46.8% | 47.0% | 1.15 | 1.02 | 0.06% | -0.31% | 0.130 | 0.261 |
| 50 | 1850 | 87% | 47.1% | 47.0% | 1.10 | 1.02 | -0.09% | -0.31% | 0.110 | 0.325 |

## 3. Ranking signal (information coefficient)

| Model | Mean IC | IC t-stat | Months |
|---|---:|---:|---:|
| XGBoost | -0.005 | -0.28 | 37 |
| Random forest | -0.013 | -0.62 | 37 |
| Ridge (auxiliary) | -0.013 | -0.56 | 37 |
| Rank on volatility alone (single factor) | 0.016 | 0.44 | 37 |
| Rank on beta alone (single factor) | 0.001 | 0.01 | 37 |

*Spearman correlation between the predicted and the realised target, per month.*

## 4. Portfolio (monthly, net of costs)

| Model | Fallback | CAGR | Sharpe | Max DD | Months invested | Avg names when invested | Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | 14.78% | 0.55 | -27.26% | 100% | 5.0 | 1.79 |
| XGBoost | bil | 14.78% | 0.55 | -27.26% | 100% | 5.0 | 1.79 |
| XGBoost | trend | 14.78% | 0.55 | -27.26% | 100% | 5.0 | 1.79 |
| Random forest | spy | 21.90% | 0.65 | -27.49% | 100% | 5.0 | 1.82 |
| Random forest | bil | 21.90% | 0.65 | -27.49% | 100% | 5.0 | 1.82 |
| Random forest | trend | 21.90% | 0.65 | -27.49% | 100% | 5.0 | 1.82 |
| Ridge (auxiliary) | spy | 14.61% | 0.55 | -35.58% | 100% | 5.0 | 1.76 |
| Ridge (auxiliary) | bil | 14.61% | 0.55 | -35.58% | 100% | 5.0 | 1.76 |
| Ridge (auxiliary) | trend | 14.61% | 0.55 | -35.58% | 100% | 5.0 | 1.76 |
| Rank on volatility alone (single factor) | spy | 3.90% | 0.31 | -64.04% | 100% | 5.0 | 0.76 |
| Rank on volatility alone (single factor) | bil | 3.90% | 0.31 | -64.04% | 100% | 5.0 | 0.76 |
| Rank on volatility alone (single factor) | trend | 3.90% | 0.31 | -64.04% | 100% | 5.0 | 0.76 |
| Rank on beta alone (single factor) | spy | 28.24% | 0.74 | -27.02% | 100% | 5.0 | 0.42 |
| Rank on beta alone (single factor) | bil | 28.24% | 0.74 | -27.02% | 100% | 5.0 | 0.42 |
| Rank on beta alone (single factor) | trend | 28.24% | 0.74 | -27.02% | 100% | 5.0 | 0.42 |
| SPY (buy & hold) | - | 19.66% | 1.46 | -7.58% | - | - | - |
| BIL (buy & hold) | - | 4.54% | 20.30 | 0.00% | - | - | - |

*`spy`: hold SPY in months with no pick, so those months add zero active return and the result isolates the stock picks. `bil`: hold Treasury bills instead. `trend`: SPY if its month-end close is above its 10-month average, else bills; this is a separate timing rule. Sharpe uses a zero risk-free rate, as in the other reports.*

## 5. Risk-adjusted: CAPM net of the risk-free rate

`r - rf = alpha + beta * (r_SPY - rf)`, monthly, Newey-West t-stats (3 lags), rf = BIL. |t| above about 2 is the usual bar.

| Model | Fallback | Alpha / yr | t | Beta | t (beta vs 1) | R² | Alpha 1st half (t) | Alpha 2nd half (t) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | -1.79% | -0.10 | 1.16 | 0.30 | 0.18 | -37.31% (-2.81) | 32.01% (1.25) |
| XGBoost | bil | -1.79% | -0.10 | 1.16 | 0.30 | 0.18 | -37.31% (-2.81) | 32.01% (1.25) |
| XGBoost | trend | -1.79% | -0.10 | 1.16 | 0.30 | 0.18 | -37.31% (-2.81) | 32.01% (1.25) |
| Random forest | spy | -0.34% | -0.02 | 1.64 | 0.95 | 0.24 | -20.72% (-1.90) | 20.16% (0.63) |
| Random forest | bil | -0.34% | -0.02 | 1.64 | 0.95 | 0.24 | -20.72% (-1.90) | 20.16% (0.63) |
| Random forest | trend | -0.34% | -0.02 | 1.64 | 0.95 | 0.24 | -20.72% (-1.90) | 20.16% (0.63) |
| Ridge (auxiliary) | spy | -2.19% | -0.15 | 1.31 | 1.17 | 0.19 | -21.41% (-2.75) | 15.85% (0.65) |
| Ridge (auxiliary) | bil | -2.19% | -0.15 | 1.31 | 1.17 | 0.19 | -21.41% (-2.75) | 15.85% (0.65) |
| Ridge (auxiliary) | trend | -2.19% | -0.15 | 1.31 | 1.17 | 0.19 | -21.41% (-2.75) | 15.85% (0.65) |
| Rank on volatility alone (single factor) | spy | -16.29% | -0.59 | 1.89 | 1.71 | 0.23 | -65.00% (-3.12) | 31.34% (0.79) |
| Rank on volatility alone (single factor) | bil | -16.29% | -0.59 | 1.89 | 1.71 | 0.23 | -65.00% (-3.12) | 31.34% (0.79) |
| Rank on volatility alone (single factor) | trend | -16.29% | -0.59 | 1.89 | 1.71 | 0.23 | -65.00% (-3.12) | 31.34% (0.79) |
| Rank on beta alone (single factor) | spy | -6.33% | -0.32 | 2.60 | 4.21 | 0.48 | -43.74% (-2.16) | 29.86% (1.15) |
| Rank on beta alone (single factor) | bil | -6.33% | -0.32 | 2.60 | 4.21 | 0.48 | -43.74% (-2.16) | 29.86% (1.15) |
| Rank on beta alone (single factor) | trend | -6.33% | -0.32 | 2.60 | 4.21 | 0.48 | -43.74% (-2.16) | 29.86% (1.15) |

## 6. Year by year (fallback = spy)

| Year | SPY | XGBoost | Random forest | Ridge (auxiliary) | Rank on volatility alone (single factor) | Rank on beta alone (single factor) |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 6.3% | -23.6% | 1.1% | -9.6% | -8.6% | -20.8% |
| 2024 | 26.2% | -5.1% | -4.6% | 12.7% | -48.8% | 3.2% |
| 2025 | 16.3% | 16.3% | 33.1% | 16.4% | 45.8% | 86.1% |
| 2026 | 11.4% | 81.5% | 43.3% | 28.4% | 64.7% | 41.6% |

## 7. Latest decision

- **XGBoost** (2026-08-31): FDXF, SNDK, MU, HPE, CPRT
- **Random forest** (2026-08-31): MRVL, HPE, AES, SNDK, EVRG
- **Ridge (auxiliary)** (2026-08-31): SNDK, MU, LITE, DELL, CIEN
- **Rank on volatility alone (single factor)** (2026-08-31): MRNA, SNDK, MRVL, SMCI, COHR
- **Rank on beta alone (single factor)** (2026-08-31): SNDK, SMCI, MU, COHR, LRCX

## Verdict

- **XGBoost**: picks earn 0.21% per pick against -0.36% for the same-month universe, winning 51.1% of the time vs 46.8%. Against random picks from the same months p = 0.294, so this selector did not earn more than picking at random.
- **Random forest**: picks earn 0.92% per pick against -0.31% for the same-month universe, winning 49.2% of the time vs 47.0%. Against random picks from the same months p = 0.056, so this selector did not earn more than picking at random.
- **Ridge (auxiliary)**: picks earn 0.37% per pick against -0.30% for the same-month universe, winning 51.6% of the time vs 47.0%. Against random picks from the same months p = 0.303, so this selector did not earn more than picking at random.
- **Rank on volatility alone (single factor)**: picks earn -0.21% per pick against -0.31% for the same-month universe, winning 45.4% of the time vs 47.0%. Against random picks from the same months p = 0.890, so this selector did not earn more than picking at random.
- **Rank on beta alone (single factor)**: picks earn 1.44% per pick against -0.31% for the same-month universe, winning 42.7% of the time vs 47.0%. Against random picks from the same months p = 0.353, so this selector did not earn more than picking at random.

The universe is the point-in-time S&P 500: each month only the stocks that were in the index on that date. Members whose prices yfinance no longer has (mostly companies later acquired or bankrupt) are missing; see the membership coverage above. Acquired names tend to have risen and bankrupt ones to have fallen, so the remaining bias is smaller but its sign is not certain. Out-of-sample research evaluation only; not financial advice.
