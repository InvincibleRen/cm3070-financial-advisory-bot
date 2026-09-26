# Hurdle Selection on a Risk-Adjusted Excess-Return Target

Generated at: 2026-09-24 14:09:04

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
- Features: technical,fundamental (16 columns: mom_1m, mom_3m, mom_6m, mom_12m, reversal_5d, mom_risk_adj, high_52w, volatility, macd, adx, pe, pb, roe, earnings_growth, net_margin, debt_to_equity)
- Training target: (r - beta x r_SPY) / vol, winsorised per month
- Walk-forward starts: 2023-01-01 (prices from 2015-01-01)
- Feature coverage (non-NaN cells, before normalisation):
    - technical: mom_1m 100%, mom_3m 100%, mom_6m 100%, mom_12m 100%, reversal_5d 100%, mom_risk_adj 100%, high_52w 100%, volatility 100%, macd 100%, adx 100%
    - fundamental: pe 72%, pb 87%, roe 77%, earnings_growth 46%, net_margin 94%, debt_to_equity 93%
- Target rows: 20997, mean -0.053, share above zero 46.1%

**Target.** `y = (r_stock - beta * r_SPY) / vol` over the next month, where beta (Blume-adjusted, 252 trading days) and vol (63 days, monthly scale) use only data up to the rebalance date, winsorised at the 1st/99th percentile of each month. Subtracting beta x SPY means a stock cannot score well just by being high-beta in a rising market; dividing by vol puts the payoff in units of risk taken.

**Rule.** Buy the names whose predicted target is above the hurdle, best first, up to the cap, equal-weighted among themselves. If none qualify, hold the fallback. The hurdle is fixed before the run, not tuned on it.

## 1. The picks: win rate, payoff ratio, expectancy

Each pick is judged over the month it was held, two ways. **Raw** = its return minus SPY's, what a trader sees. **Beta-adjusted** = its return minus beta x SPY's (beta estimated before the pick), which a high-beta stock cannot win just because the market rose. The universe column is every stock in the same months, weighted by how many picks were made that month.

Two random benchmarks replace each month's picks with the same number of names from that month. **`p (random)` is the primary test**: stand-ins are drawn from the whole cross-section, so it answers the question a user asks — did these picks earn more than picking at random? `p (vol-matched)` draws each stand-in from the 20 names nearest the pick in volatility, which by construction removes any return earned by holding volatile names; it therefore answers the narrower question of whether anything remains *beyond* beta and volatility, and a selector that works by taking risk is not failed for losing it. p is the share of 2,000 draws that did at least as well; below 0.05 is the usual bar. The two single-factor rows rank on one measure with nothing fitted.

**Raw, minus SPY (primary)**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 185 | 55% | 46.5% | 45.8% | 1.52 | 1.03 | 1.22% | -0.44% | 0.36% | 0.012 | 0.135 |
| Random forest | 185 | 52% | 47.6% | 45.8% | 1.60 | 1.03 | 1.72% | -0.44% | 0.04% | 0.003 | 0.022 |
| Ridge (auxiliary) | 185 | 62% | 47.0% | 45.8% | 1.19 | 1.03 | 0.25% | -0.44% | -0.26% | 0.145 | 0.257 |
| Rank on volatility alone (single factor) | 185 | 100% | 46.5% | 45.8% | 1.12 | 1.03 | -0.21% | -0.44% | 0.17% | 0.361 | 0.625 |
| Rank on beta alone (single factor) | 185 | 96% | 46.5% | 45.8% | 1.43 | 1.03 | 1.44% | -0.44% | 0.65% | 0.004 | 0.227 |

**Beta-adjusted**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 180 | 55% | 45.6% | 46.8% | 1.47 | 1.02 | 0.86% | -0.34% | 0.09% | 0.034 | 0.161 |
| Random forest | 184 | 52% | 47.3% | 46.9% | 1.65 | 1.02 | 1.68% | -0.32% | 0.14% | 0.002 | 0.032 |
| Ridge (auxiliary) | 184 | 62% | 47.8% | 47.0% | 1.14 | 1.02 | 0.22% | -0.30% | -0.19% | 0.220 | 0.294 |
| Rank on volatility alone (single factor) | 185 | 100% | 45.4% | 47.0% | 1.04 | 1.02 | -1.05% | -0.31% | -0.56% | 0.890 | 0.664 |
| Rank on beta alone (single factor) | 185 | 96% | 42.7% | 47.0% | 1.33 | 1.02 | -0.08% | -0.31% | -0.09% | 0.353 | 0.489 |

*Payoff ratio = average winning excess return / average losing excess return (the realised risk-reward, no stop-loss or take-profit). Expectancy = mean excess return per pick. Vol percentile of picks = where the picks sat in that month's volatility ranking (50% = no tilt, 100% = always the most volatile). Vol-matched random = the average expectancy of random stand-ins drawn from the 20 names nearest each pick in volatility, i.e. what the picks' risk profile alone would have earned.*

### 1b. Why per-pick excess and compounded return can disagree

The expectancy above is an **arithmetic** mean per pick. An investor earns the **compounded** return, and for small returns the two differ by about half the variance, so a concentrated book gives more back to that term than the index does. The table separates the two; a small residual means the approximation accounts for the gap. All figures annualised, against SPY, on the `spy` fallback variant.

| Model | Arithmetic excess | Variance drag | Residual | Compounded excess | Book vol | SPY vol |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 12.62% | −4.14% | 2.67% | 11.15% | 31.5% | 12.9% |
| Random forest | 18.55% | −8.35% | 3.41% | 13.61% | 42.8% | 12.9% |
| Ridge (auxiliary) | 1.06% | −5.21% | 0.41% | -3.73% | 34.7% | 12.9% |
| Rank on volatility alone (single factor) | -3.41% | −11.86% | -0.49% | -15.76% | 50.4% | 12.9% |
| Rank on beta alone (single factor) | 16.75% | −10.70% | 2.53% | 8.58% | 48.0% | 12.9% |

*Arithmetic excess − variance drag + residual = compounded excess. The drag is not a fault in the selector: it is the arithmetic of compounding a more volatile series, and it shrinks if the book is held wider.*

## 2. Does stricter selection do better? (beta-adjusted)

Top-k by score each month, ignoring the hurdle. If the score orders stocks usefully, the pick columns improve as k shrinks — the best-scored few should earn more than the best-scored many.

**XGBoost**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 65% | 55.6% | 46.6% | 1.28 | 1.02 | 1.76% | -0.37% | 0.072 | 0.168 |
| 3 | 107 | 58% | 48.6% | 46.8% | 1.44 | 1.02 | 1.21% | -0.34% | 0.040 | 0.218 |
| 5 | 180 | 55% | 45.6% | 46.8% | 1.47 | 1.02 | 0.86% | -0.34% | 0.052 | 0.154 |
| 10 | 362 | 55% | 43.9% | 46.8% | 1.36 | 1.02 | 0.24% | -0.34% | 0.104 | 0.168 |
| 20 | 731 | 54% | 43.5% | 46.9% | 1.27 | 1.02 | -0.09% | -0.33% | 0.244 | 0.283 |
| 50 | 1838 | 52% | 44.7% | 46.9% | 1.05 | 1.02 | -0.56% | -0.32% | 0.890 | 0.890 |

**Random forest**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 36 | 52% | 50.0% | 46.7% | 2.97 | 1.02 | 5.38% | -0.34% | 0.002 | 0.012 |
| 3 | 110 | 51% | 45.5% | 46.9% | 2.03 | 1.02 | 2.30% | -0.32% | 0.002 | 0.022 |
| 5 | 184 | 52% | 47.3% | 46.9% | 1.65 | 1.02 | 1.68% | -0.32% | 0.002 | 0.030 |
| 10 | 369 | 53% | 43.4% | 46.9% | 1.40 | 1.02 | 0.26% | -0.31% | 0.094 | 0.166 |
| 20 | 737 | 51% | 43.4% | 46.9% | 1.20 | 1.02 | -0.30% | -0.31% | 0.533 | 0.409 |
| 50 | 1846 | 49% | 42.6% | 47.0% | 1.06 | 1.02 | -0.79% | -0.31% | 0.998 | 0.928 |

**Ridge (auxiliary)**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37 | 65% | 51.4% | 47.0% | 0.45 | 1.02 | -3.00% | -0.31% | 0.978 | 0.994 |
| 3 | 110 | 60% | 47.3% | 47.0% | 0.65 | 1.02 | -2.11% | -0.30% | 0.992 | 0.988 |
| 5 | 184 | 62% | 47.8% | 47.0% | 1.14 | 1.02 | 0.22% | -0.30% | 0.194 | 0.275 |
| 10 | 365 | 60% | 46.8% | 47.0% | 1.23 | 1.02 | 0.30% | -0.31% | 0.094 | 0.198 |
| 20 | 732 | 58% | 47.4% | 46.9% | 1.17 | 1.02 | 0.20% | -0.31% | 0.050 | 0.309 |
| 50 | 1839 | 55% | 45.8% | 47.0% | 1.22 | 1.02 | 0.12% | -0.31% | 0.022 | 0.110 |

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
| XGBoost | -0.012 | -0.85 | 37 |
| Random forest | -0.025 | -1.49 | 37 |
| Ridge (auxiliary) | -0.005 | -0.24 | 37 |
| Rank on volatility alone (single factor) | 0.016 | 0.44 | 37 |
| Rank on beta alone (single factor) | 0.001 | 0.01 | 37 |

*Spearman correlation between the predicted and the realised target, per month.*

## 4. Portfolio (monthly, net of costs)

| Model | Fallback | CAGR | Sharpe | Max DD | Months invested | Avg names when invested | Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | 30.81% | 1.00 | -16.63% | 100% | 5.0 | 1.66 |
| XGBoost | bil | 30.81% | 1.00 | -16.63% | 100% | 5.0 | 1.66 |
| XGBoost | trend | 30.81% | 1.00 | -16.63% | 100% | 5.0 | 1.66 |
| Random forest | spy | 33.28% | 0.87 | -32.35% | 100% | 5.0 | 1.69 |
| Random forest | bil | 33.28% | 0.87 | -32.35% | 100% | 5.0 | 1.69 |
| Random forest | trend | 33.28% | 0.87 | -32.35% | 100% | 5.0 | 1.69 |
| Ridge (auxiliary) | spy | 15.93% | 0.57 | -16.69% | 100% | 5.0 | 1.65 |
| Ridge (auxiliary) | bil | 15.93% | 0.57 | -16.69% | 100% | 5.0 | 1.65 |
| Ridge (auxiliary) | trend | 15.93% | 0.57 | -16.69% | 100% | 5.0 | 1.65 |
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
| XGBoost | spy | 14.27% | 0.78 | 0.89 | -0.38 | 0.13 | -8.66% (-1.16) | 36.11% (1.12) |
| XGBoost | bil | 14.27% | 0.78 | 0.89 | -0.38 | 0.13 | -8.66% (-1.16) | 36.11% (1.12) |
| XGBoost | trend | 14.27% | 0.78 | 0.89 | -0.38 | 0.13 | -8.66% (-1.16) | 36.11% (1.12) |
| Random forest | spy | 13.56% | 0.57 | 1.35 | 0.81 | 0.16 | 0.28% (0.02) | 26.99% (0.62) |
| Random forest | bil | 13.56% | 0.57 | 1.35 | 0.81 | 0.16 | 0.28% (0.02) | 26.99% (0.62) |
| Random forest | trend | 13.56% | 0.57 | 1.35 | 0.81 | 0.16 | 0.28% (0.02) | 26.99% (0.62) |
| Ridge (auxiliary) | spy | 2.97% | 0.14 | 0.87 | -0.74 | 0.10 | -17.46% (-1.03) | 21.43% (0.62) |
| Ridge (auxiliary) | bil | 2.97% | 0.14 | 0.87 | -0.74 | 0.10 | -17.46% (-1.03) | 21.43% (0.62) |
| Ridge (auxiliary) | trend | 2.97% | 0.14 | 0.87 | -0.74 | 0.10 | -17.46% (-1.03) | 21.43% (0.62) |
| Rank on volatility alone (single factor) | spy | -16.29% | -0.59 | 1.89 | 1.71 | 0.23 | -65.00% (-3.12) | 31.34% (0.79) |
| Rank on volatility alone (single factor) | bil | -16.29% | -0.59 | 1.89 | 1.71 | 0.23 | -65.00% (-3.12) | 31.34% (0.79) |
| Rank on volatility alone (single factor) | trend | -16.29% | -0.59 | 1.89 | 1.71 | 0.23 | -65.00% (-3.12) | 31.34% (0.79) |
| Rank on beta alone (single factor) | spy | -6.33% | -0.32 | 2.60 | 4.21 | 0.48 | -43.74% (-2.16) | 29.86% (1.15) |
| Rank on beta alone (single factor) | bil | -6.33% | -0.32 | 2.60 | 4.21 | 0.48 | -43.74% (-2.16) | 29.86% (1.15) |
| Rank on beta alone (single factor) | trend | -6.33% | -0.32 | 2.60 | 4.21 | 0.48 | -43.74% (-2.16) | 29.86% (1.15) |

## 6. Year by year (fallback = spy)

| Year | SPY | XGBoost | Random forest | Ridge (auxiliary) | Rank on volatility alone (single factor) | Rank on beta alone (single factor) |
|---|---:|---:|---:|---:|---:|---:|
| 2023 | 6.3% | -0.8% | -0.9% | -13.7% | -8.6% | -20.8% |
| 2024 | 26.2% | 10.6% | 30.0% | 21.1% | -48.8% | 3.2% |
| 2025 | 16.3% | 53.1% | 58.8% | 50.1% | 45.8% | 86.1% |
| 2026 | 11.4% | 36.3% | 18.5% | 0.5% | 64.7% | 41.6% |

## 7. Latest decision

- **XGBoost** (2026-08-31): PCG, ZTS, ACGL, IQV, VZ
- **Random forest** (2026-08-31): CI, ACGL, VICI, WRB, HIG
- **Ridge (auxiliary)** (2026-08-31): INTC, EXE, CHTR, FOX, FOXA
- **Rank on volatility alone (single factor)** (2026-08-31): MRNA, SNDK, MRVL, SMCI, COHR
- **Rank on beta alone (single factor)** (2026-08-31): SNDK, SMCI, MU, COHR, LRCX

## Verdict

- **XGBoost**: picks earn 1.22% per pick against -0.34% for the same-month universe, winning 45.6% of the time vs 46.8%, beating random picks from the same months at p = 0.034. Against volatility-matched stand-ins p = 0.161, so most of the advantage comes through beta and volatility.
- **Random forest**: picks earn 1.72% per pick against -0.32% for the same-month universe, winning 47.3% of the time vs 46.9%, beating random picks from the same months at p = 0.002. It also beats volatility-matched stand-ins (p = 0.032), so the advantage is not only the risk it takes.
- **Ridge (auxiliary)**: picks earn 0.25% per pick against -0.30% for the same-month universe, winning 47.8% of the time vs 47.0%. Against random picks from the same months p = 0.220, so this selector did not earn more than picking at random.
- **Rank on volatility alone (single factor)**: picks earn -0.21% per pick against -0.31% for the same-month universe, winning 45.4% of the time vs 47.0%. Against random picks from the same months p = 0.890, so this selector did not earn more than picking at random.
- **Rank on beta alone (single factor)**: picks earn 1.44% per pick against -0.31% for the same-month universe, winning 42.7% of the time vs 47.0%. Against random picks from the same months p = 0.353, so this selector did not earn more than picking at random.

The universe is the point-in-time S&P 500: each month only the stocks that were in the index on that date. Members whose prices yfinance no longer has (mostly companies later acquired or bankrupt) are missing; see the membership coverage above. Acquired names tend to have risen and bankrupt ones to have fallen, so the remaining bias is smaller but its sign is not certain. Out-of-sample research evaluation only; not financial advice.
