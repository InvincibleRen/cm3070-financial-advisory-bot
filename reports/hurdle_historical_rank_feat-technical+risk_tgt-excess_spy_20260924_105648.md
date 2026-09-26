# Hurdle Selection on a Risk-Adjusted Excess-Return Target

Generated at: 2026-09-24 10:56:48

## Configuration

- Universe: point-in-time S&P 500 (each month's actual members)
- Membership coverage: 742 tickers were members at some point; 118 have no price data. Share of each month's members with prices: median 91%, lowest 76% (2015-01), highest 100%
- History: 2015-01-01 to latest, 140 monthly rebalances
- Training window: 12 rebalances rolling
- Models: XGBoost
- Hurdle: predicted target > 0.0
- Max names: 5 (equal weight among those that qualify)
- Fallbacks: spy
- Transaction cost: 10 bps per unit turnover (fallback switches included)
- Feature normalisation: rank
- Features: technical,risk (11 columns: mom_1m, mom_3m, mom_6m, mom_12m, reversal_5d, mom_risk_adj, high_52w, volatility, macd, adx, beta)
- Training target: r - r_SPY, winsorised per month
- Walk-forward starts: 2015-01-01
- Feature coverage (non-NaN cells, before normalisation):
    - technical: mom_1m 99%, mom_3m 98%, mom_6m 96%, mom_12m 92%, reversal_5d 100%, mom_risk_adj 92%, high_52w 100%, volatility 99%, macd 100%, adx 99%
    - fundamental: 
- Target rows: 60745, mean -0.002, share above zero 48.2%

**Target.** `y = (r_stock - beta * r_SPY) / vol` over the next month, where beta (Blume-adjusted, 252 trading days) and vol (63 days, monthly scale) use only data up to the rebalance date, winsorised at the 1st/99th percentile of each month. Subtracting beta x SPY means a stock cannot score well just by being high-beta in a rising market; dividing by vol puts the payoff in units of risk taken.

**Rule.** Buy the names whose predicted target is above the hurdle, best first, up to the cap, equal-weighted among themselves. If none qualify, hold the fallback. The hurdle is fixed before the run, not tuned on it.

## 1. The picks: win rate, payoff ratio, expectancy

Each pick is judged over the month it was held, two ways. **Raw** = its return minus SPY's, what a trader sees. **Beta-adjusted** = its return minus beta x SPY's (beta estimated before the pick), which a high-beta stock cannot win just because the market rose. The universe column is every stock in the same months, weighted by how many picks were made that month.

Two random benchmarks replace each month's picks with the same number of names from that month. **`p (random)` is the primary test**: stand-ins are drawn from the whole cross-section, so it answers the question a user asks — did these picks earn more than picking at random? `p (vol-matched)` draws each stand-in from the 20 names nearest the pick in volatility, which by construction removes any return earned by holding volatile names; it therefore answers the narrower question of whether anything remains *beyond* beta and volatility, and a selector that works by taking risk is not failed for losing it. p is the share of 2,000 draws that did at least as well; below 0.05 is the usual bar. The two single-factor rows rank on one measure with nothing fitted.

**Raw, minus SPY (primary)**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 659 | 85% | 47.8% | 48.3% | 0.99 | 1.01 | -0.45% | -0.16% | 0.14% | 0.812 | 0.917 |
| Rank on volatility alone (single factor) | 665 | 100% | 47.2% | 48.4% | 1.20 | 1.01 | 0.46% | -0.16% | 0.39% | 0.026 | 0.446 |
| Rank on beta alone (single factor) | 665 | 95% | 51.3% | 48.4% | 1.37 | 1.01 | 2.09% | -0.16% | 0.51% | 0.000 | 0.000 |

**Beta-adjusted**

| Model | Picks | Vol percentile of picks | Win rate | Universe | Payoff ratio | Universe | Expectancy / pick | Universe | Vol-matched random | p (random) | p (vol-matched) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | 642 | 85% | 45.8% | 48.7% | 0.96 | 0.99 | -0.92% | -0.15% | -0.15% | 0.995 | 0.967 |
| Rank on volatility alone (single factor) | 665 | 100% | 46.8% | 48.7% | 1.13 | 1.00 | -0.05% | -0.15% | -0.08% | 0.373 | 0.477 |
| Rank on beta alone (single factor) | 665 | 95% | 48.9% | 48.7% | 1.27 | 1.00 | 1.08% | -0.15% | 0.04% | 0.000 | 0.016 |

*Payoff ratio = average winning excess return / average losing excess return (the realised risk-reward, no stop-loss or take-profit). Expectancy = mean excess return per pick. Vol percentile of picks = where the picks sat in that month's volatility ranking (50% = no tilt, 100% = always the most volatile). Vol-matched random = the average expectancy of random stand-ins drawn from the 20 names nearest each pick in volatility, i.e. what the picks' risk profile alone would have earned.*

### 1b. Why per-pick excess and compounded return can disagree

The expectancy above is an **arithmetic** mean per pick. An investor earns the **compounded** return, and for small returns the two differ by about half the variance, so a concentrated book gives more back to that term than the index does. The table separates the two; a small residual means the approximation accounts for the gap. All figures annualised, against SPY, on the `spy` fallback variant.

| Model | Arithmetic excess | Variance drag | Residual | Compounded excess | Book vol | SPY vol |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | -7.30% | −4.87% | -0.63% | -12.80% | 34.7% | 15.2% |
| Rank on volatility alone (single factor) | 4.61% | −10.86% | 0.12% | -6.13% | 49.0% | 15.2% |
| Rank on beta alone (single factor) | 24.57% | −9.47% | 4.45% | 19.54% | 46.1% | 15.2% |

*Arithmetic excess − variance drag + residual = compounded excess. The drag is not a fault in the selector: it is the arithmetic of compounding a more volatile series, and it shrinks if the book is held wider.*

## 2. Does stricter selection do better? (beta-adjusted)

Top-k by score each month, ignoring the hurdle. If the score orders stocks usefully, the pick columns improve as k shrinks — the best-scored few should earn more than the best-scored many.

**XGBoost**

| Top k | Picks | Vol pct | Win rate | Universe | Payoff | Universe | Expectancy | Universe | p (random) | p (vol-matched) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 127 | 87% | 46.5% | 48.9% | 0.71 | 1.00 | -2.35% | -0.13% | 1.000 | 0.978 |
| 3 | 384 | 86% | 45.3% | 48.8% | 0.97 | 0.99 | -0.99% | -0.15% | 0.996 | 0.966 |
| 5 | 647 | 85% | 45.9% | 48.8% | 0.96 | 0.99 | -0.91% | -0.15% | 0.992 | 0.966 |
| 10 | 1305 | 79% | 46.0% | 48.8% | 1.00 | 1.00 | -0.68% | -0.15% | 0.998 | 0.958 |
| 20 | 2627 | 73% | 46.4% | 48.7% | 1.01 | 1.00 | -0.51% | -0.15% | 0.996 | 0.960 |
| 50 | 6605 | 64% | 46.9% | 48.7% | 1.01 | 1.00 | -0.39% | -0.15% | 0.996 | 0.986 |

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
| XGBoost | -0.007 | -0.55 | 132 |
| Rank on volatility alone (single factor) | 0.004 | 0.18 | 133 |
| Rank on beta alone (single factor) | 0.022 | 0.86 | 133 |

*Spearman correlation between the predicted and the realised target, per month.*

## 4. Portfolio (monthly, net of costs)

| Model | Fallback | CAGR | Sharpe | Max DD | Months invested | Avg names when invested | Turnover |
|---|---|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | 1.44% | 0.21 | -50.17% | 99% | 5.0 | 1.71 |
| Rank on volatility alone (single factor) | spy | 8.10% | 0.39 | -70.90% | 100% | 5.0 | 0.72 |
| Rank on beta alone (single factor) | spy | 33.78% | 0.85 | -39.16% | 100% | 5.0 | 0.39 |
| SPY (buy & hold) | - | 14.23% | 0.96 | -23.93% | - | - | - |
| BIL (buy & hold) | - | 2.05% | 3.59 | -0.16% | - | - | - |

*`spy`: hold SPY in months with no pick, so those months add zero active return and the result isolates the stock picks. `bil`: hold Treasury bills instead. `trend`: SPY if its month-end close is above its 10-month average, else bills; this is a separate timing rule. Sharpe uses a zero risk-free rate, as in the other reports.*

## 5. Risk-adjusted: CAPM net of the risk-free rate

`r - rf = alpha + beta * (r_SPY - rf)`, monthly, Newey-West t-stats (3 lags), rf = BIL. |t| above about 2 is the usual bar.

| Model | Fallback | Alpha / yr | t | Beta | t (beta vs 1) | R² | Alpha 1st half (t) | Alpha 2nd half (t) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| XGBoost | spy | -16.34% | -2.54 | 1.72 | 4.37 | 0.57 | -16.69% (-2.14) | -16.01% (-1.52) |
| Rank on volatility alone (single factor) | spy | -8.90% | -0.79 | 2.08 | 3.12 | 0.42 | -4.20% (-0.35) | -13.47% (-0.77) |
| Rank on beta alone (single factor) | spy | 8.62% | 0.98 | 2.28 | 7.48 | 0.56 | 6.15% (0.48) | 11.05% (0.93) |

## 6. Year by year (fallback = spy)

| Year | SPY | XGBoost | Rank on volatility alone (single factor) | Rank on beta alone (single factor) |
|---|---:|---:|---:|---:|
| 2015 | -6.9% | -17.7% | -29.5% | -17.2% |
| 2016 | 20.0% | 42.9% | 137.6% | 105.5% |
| 2017 | 26.3% | -2.6% | -4.9% | 18.8% |
| 2018 | -2.4% | -25.7% | -10.0% | -25.3% |
| 2019 | 21.4% | 44.4% | 40.2% | 44.9% |
| 2020 | 17.2% | -18.1% | 12.6% | 95.1% |
| 2021 | 23.2% | 3.9% | -18.1% | 45.1% |
| 2022 | -8.2% | -18.5% | 19.3% | 8.5% |
| 2023 | 20.6% | -1.3% | -12.8% | 37.7% |
| 2024 | 26.2% | -16.5% | -48.8% | 3.2% |
| 2025 | 16.3% | 0.9% | 45.8% | 86.1% |
| 2026 | 11.4% | 64.8% | 64.7% | 41.6% |

## 7. Latest decision

- **XGBoost** (2026-08-31): MRVL, MU, DELL, SNDK, LRCX
- **Rank on volatility alone (single factor)** (2026-08-31): MRNA, SNDK, MRVL, SMCI, COHR
- **Rank on beta alone (single factor)** (2026-08-31): SNDK, SMCI, MU, COHR, LRCX

## Verdict

- **XGBoost**: picks earn -0.45% per pick against -0.15% for the same-month universe, winning 45.8% of the time vs 48.7%. Against random picks from the same months p = 0.995, so this selector did not earn more than picking at random.
- **Rank on volatility alone (single factor)**: picks earn 0.46% per pick against -0.15% for the same-month universe, winning 46.8% of the time vs 48.7%. Against random picks from the same months p = 0.373, so this selector did not earn more than picking at random.
- **Rank on beta alone (single factor)**: picks earn 2.09% per pick against -0.15% for the same-month universe, winning 48.9% of the time vs 48.7%, beating random picks from the same months at p = 0.000. It also beats volatility-matched stand-ins (p = 0.016), so the advantage is not only the risk it takes.

The universe is the point-in-time S&P 500: each month only the stocks that were in the index on that date. Members whose prices yfinance no longer has (mostly companies later acquired or bankrupt) are missing; see the membership coverage above. Acquired names tend to have risen and bankrupt ones to have fallen, so the remaining bias is smaller but its sign is not certain. Out-of-sample research evaluation only; not financial advice.
