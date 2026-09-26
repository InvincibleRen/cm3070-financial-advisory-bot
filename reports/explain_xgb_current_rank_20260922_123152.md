# What the XGBoost Return Model Learned

Generated at: 2026-09-22 12:31:52

## Setup

- Model: XGBoost regressor on the risk-adjusted excess-return target
- Universe: today's S&P 500 list (survivorship-biased)
- Features: rank per month
- Months explained: 133 (2015-07 to 2026-07)
- Predictions explained: 64626
- SHAP check: contributions sum to the predictions (max error 1.1e-06)

Every number below is out of sample: each month's model (trained only on the previous 12 months) explains the predictions it made for that month's stocks. **Reliance** is the average absolute SHAP contribution, i.e. how far the feature moves a prediction; the shares add up to 100%. **Direction** is the rank correlation between a feature's value and its contribution. **Own IC** is how well the feature alone ranks the realised target each month (mean Spearman correlation, t-stat across months). A feature the model leans on without its own IC behind it is being used to fit noise.

## 1. Reliance, direction and whether the data backs it

| # | Feature | Meaning | Reliance | Direction learned | Own IC (t) | Backed by own IC? | Coverage |
|---:|---|---|---:|---|---:|---|---:|
| 1 | `high_52w` | price / 52-week high | 9.6% | higher -> lower prediction (-0.12) | -0.008 (-0.50) | no support (own IC not significant) | 100% |
| 2 | `volatility` | 21-day volatility | 9.3% | higher -> lower prediction (-0.12) | +0.007 (+0.41) | no support (own IC not significant) | 100% |
| 3 | `mom_6m` | 6-month return | 9.3% | higher -> higher prediction (+0.19) | +0.005 (+0.32) | no support (own IC not significant) | 100% |
| 4 | `mom_1m` | 1-month return | 9.2% | higher -> higher prediction (+0.17) | -0.002 (-0.17) | no support (own IC not significant) | 100% |
| 5 | `mom_3m` | 3-month return | 8.8% | higher -> lower prediction (-0.21) | -0.008 (-0.55) | no support (own IC not significant) | 100% |
| 6 | `mom_12m` | 12-1 month return (skips the last month) | 8.8% | no clear direction (-0.06) | -0.004 (-0.25) | n/a | 95% |
| 7 | `reversal_5d` | last 5 days' return | 8.6% | higher -> lower prediction (-0.20) | -0.010 (-0.82) | no support (own IC not significant) | 100% |
| 8 | `mom_risk_adj` | 12-1 momentum / volatility | 7.8% | higher -> higher prediction (+0.10) | -0.002 (-0.10) | no support (own IC not significant) | 95% |
| 9 | `macd` | MACD | 7.4% | higher -> lower prediction (-0.26) | -0.008 (-0.67) | no support (own IC not significant) | 100% |
| 10 | `adx` | ADX trend strength | 7.1% | no clear direction (-0.09) | +0.001 (+0.17) | n/a | 100% |
| 11 | `earnings_growth` | earnings growth (YoY) | 2.9% | higher -> higher prediction (+0.42) | +0.016 (+0.46) | no support (own IC not significant) | 15% |
| 12 | `pe` | price / earnings | 2.5% | higher -> lower prediction (-0.42) | -0.028 (-1.29) | no support (own IC not significant) | 24% |
| 13 | `debt_to_equity` | debt / equity | 2.4% | no clear direction (+0.09) | -0.009 (-0.61) | n/a | 31% |
| 14 | `roe` | return on equity | 2.4% | higher -> lower prediction (-0.30) | +0.004 (+0.18) | no support (own IC not significant) | 26% |
| 15 | `pb` | price / book | 2.1% | no clear direction (+0.02) | -0.012 (-0.48) | n/a | 29% |
| 16 | `net_margin` | net margin | 1.9% | higher -> lower prediction (-0.23) | -0.003 (-0.21) | no support (own IC not significant) | 31% |

## 2. Shape of the top 6 features

Average contribution to the predicted target by the feature's quintile within each month (Q1 = lowest fifth of stocks, Q5 = highest). Positive = pushes the prediction up.

| Feature | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| `high_52w` | +0.0274 | -0.0063 | -0.0067 | -0.0000 | -0.0103 |
| `volatility` | +0.0177 | -0.0008 | -0.0037 | -0.0070 | +0.0008 |
| `mom_6m` | -0.0213 | -0.0093 | +0.0012 | +0.0058 | +0.0241 |
| `mom_1m` | -0.0210 | +0.0006 | +0.0042 | +0.0031 | +0.0080 |
| `mom_3m` | +0.0229 | +0.0066 | -0.0044 | -0.0095 | -0.0137 |
| `mom_12m` | +0.0015 | +0.0070 | -0.0018 | -0.0085 | -0.0003 |

## 3. Does the model lean on the same features over time?

| Feature | 2015-2017 | 2018-2020 | 2021-2023 | 2024-2026 |
|---|---:|---:|---:|---:|
| `high_52w` | 9.7% | 13.1% | 10.6% | 5.4% |
| `volatility` | 11.5% | 8.8% | 9.9% | 7.3% |
| `mom_6m` | 12.0% | 10.3% | 8.7% | 6.8% |
| `mom_1m` | 9.2% | 12.0% | 8.9% | 7.0% |
| `mom_3m` | 13.5% | 8.6% | 7.3% | 6.9% |
| `mom_12m` | 8.2% | 11.1% | 9.9% | 6.0% |
| `reversal_5d` | 11.9% | 10.3% | 7.6% | 5.5% |
| `mom_risk_adj` | 8.2% | 7.8% | 9.0% | 6.2% |

## Reading guide

- Reliance says what the model *uses*, not whether it is right; the IC column says whether the data supports it. Only rows marked 'supported' reflect a relationship that held out of sample.
- Fundamental features are mostly missing (see coverage); the booster routes missing values down a default branch, so their reliance mostly reflects whether a value exists.
- Out-of-sample research only; not financial advice.
