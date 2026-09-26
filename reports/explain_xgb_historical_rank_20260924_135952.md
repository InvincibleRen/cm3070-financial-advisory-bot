# What the XGBoost Return Model Learned

Generated at: 2026-09-24 13:59:52

## Setup

- Model: XGBoost regressor on the risk-adjusted excess-return target
- Universe: point-in-time S&P 500
- Features: rank per month
- Months explained: 134 (2015-07 to 2026-08)
- Predictions explained: 61379
- SHAP check: contributions sum to the predictions (max error 1.7e-06)

Every number below is out of sample: each month's model (trained only on the previous 12 months) explains the predictions it made for that month's stocks. **Reliance** is the average absolute SHAP contribution, i.e. how far the feature moves a prediction; the shares add up to 100%. **Direction** is the rank correlation between a feature's value and its contribution. **Own IC** is how well the feature alone ranks the realised target each month (mean Spearman correlation, t-stat across months). A feature the model leans on without its own IC behind it is being used to fit noise.

## 1. Reliance, direction and whether the data backs it

| # | Feature | Meaning | Reliance | Direction learned | Own IC (t) | Backed by own IC? | Coverage |
|---:|---|---|---:|---|---:|---|---:|
| 1 | `mom_12m` | 12-1 month return (skips the last month) | 9.6% | higher -> lower prediction (-0.19) | -0.014 (-0.83) | no support (own IC not significant) | 96% |
| 2 | `mom_1m` | 1-month return | 9.3% | higher -> higher prediction (+0.15) | -0.004 (-0.31) | no support (own IC not significant) | 100% |
| 3 | `mom_6m` | 6-month return | 9.2% | higher -> higher prediction (+0.17) | -0.000 (-0.01) | no support (own IC not significant) | 100% |
| 4 | `high_52w` | price / 52-week high | 9.2% | no clear direction (-0.05) | -0.005 (-0.28) | n/a | 100% |
| 5 | `mom_3m` | 3-month return | 8.9% | higher -> lower prediction (-0.22) | -0.011 (-0.71) | no support (own IC not significant) | 100% |
| 6 | `volatility` | 21-day volatility | 8.7% | higher -> lower prediction (-0.18) | -0.006 (-0.35) | no support (own IC not significant) | 100% |
| 7 | `reversal_5d` | last 5 days' return | 8.4% | higher -> lower prediction (-0.17) | -0.011 (-0.82) | no support (own IC not significant) | 100% |
| 8 | `macd` | MACD | 7.7% | higher -> lower prediction (-0.25) | -0.008 (-0.59) | no support (own IC not significant) | 100% |
| 9 | `mom_risk_adj` | 12-1 momentum / volatility | 7.5% | higher -> higher prediction (+0.21) | -0.008 (-0.48) | no support (own IC not significant) | 96% |
| 10 | `adx` | ADX trend strength | 7.2% | no clear direction (-0.06) | +0.000 (+0.03) | n/a | 100% |
| 11 | `earnings_growth` | earnings growth (YoY) | 3.1% | higher -> higher prediction (+0.26) | +0.010 (+0.30) | no support (own IC not significant) | 16% |
| 12 | `pe` | price / earnings | 2.4% | higher -> lower prediction (-0.28) | -0.030 (-1.23) | no support (own IC not significant) | 26% |
| 13 | `roe` | return on equity | 2.4% | higher -> lower prediction (-0.29) | +0.006 (+0.32) | no support (own IC not significant) | 27% |
| 14 | `debt_to_equity` | debt / equity | 2.4% | higher -> higher prediction (+0.27) | -0.003 (-0.19) | no support (own IC not significant) | 33% |
| 15 | `pb` | price / book | 2.2% | higher -> lower prediction (-0.28) | -0.026 (-1.15) | no support (own IC not significant) | 31% |
| 16 | `net_margin` | net margin | 1.9% | higher -> higher prediction (+0.17) | +0.017 (+0.93) | no support (own IC not significant) | 33% |

## 2. Shape of the top 6 features

Average contribution to the predicted target by the feature's quintile within each month (Q1 = lowest fifth of stocks, Q5 = highest). Positive = pushes the prediction up.

| Feature | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|
| `mom_12m` | +0.0203 | +0.0177 | +0.0010 | -0.0156 | -0.0180 |
| `mom_1m` | -0.0222 | +0.0038 | +0.0031 | +0.0044 | +0.0073 |
| `mom_6m` | -0.0264 | -0.0033 | +0.0062 | +0.0061 | +0.0198 |
| `high_52w` | +0.0070 | +0.0006 | -0.0053 | +0.0018 | -0.0056 |
| `mom_3m` | +0.0215 | +0.0087 | -0.0022 | -0.0081 | -0.0149 |
| `volatility` | +0.0286 | -0.0002 | -0.0065 | -0.0103 | -0.0071 |

## 3. Does the model lean on the same features over time?

| Feature | 2015-2017 | 2018-2020 | 2021-2023 | 2024-2026 |
|---|---:|---:|---:|---:|
| `mom_12m` | 11.6% | 11.3% | 10.7% | 5.5% |
| `mom_1m` | 9.9% | 12.1% | 8.9% | 7.0% |
| `mom_6m` | 13.0% | 10.1% | 7.9% | 6.7% |
| `high_52w` | 7.6% | 13.1% | 10.9% | 5.3% |
| `mom_3m` | 13.0% | 8.7% | 8.2% | 6.5% |
| `volatility` | 10.5% | 8.4% | 9.3% | 6.8% |
| `reversal_5d` | 11.6% | 10.1% | 7.8% | 5.2% |
| `macd` | 8.5% | 7.9% | 10.0% | 4.6% |

## Reading guide

- Reliance says what the model *uses*, not whether it is right; the IC column says whether the data supports it. Only rows marked 'supported' reflect a relationship that held out of sample.
- Fundamental features are mostly missing (see coverage); the booster routes missing values down a default branch, so their reliance mostly reflects whether a value exists.
- Out-of-sample research only; not financial advice.
