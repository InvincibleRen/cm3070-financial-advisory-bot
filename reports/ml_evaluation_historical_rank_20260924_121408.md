# Machine-Learning Evaluation of the Prediction Task

Generated at: 2026-09-24 12:14:08

## Setup

- Task: regression on the risk_adjusted target
- Universe: point-in-time S&P 500
- Features: 16 (technical,fundamental), rank per month
- Rebalances: 140 (2015-01 to 2026-08)
- Training window: 12 rebalances rolling
- First rebalance: 2015-01-01
- Baseline: Training-window mean (constant)

Every figure is out of sample. Each month's model is trained only on earlier rebalances and scores that month's cross-section; the metrics pool the rows of all months, so a wide month counts for more than a thin one. The baseline throughout is the mean of the model's **own training window** — the only constant a forecaster could have used without seeing the future (Campbell and Thompson, 2008). R² above zero means the model predicts better than that constant; below zero means worse.

## 1. Does the model beat a constant? (out-of-sample R²)

| Model | R² out of sample | R² in sample | Gap | RMSE | Baseline RMSE | Mean IC (t) | Folds | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| XGBoost | -0.0841 | +0.2026 | +0.2866 | 0.8989 | 0.8696 | -0.014 (-1.48) | 132 | worse than the constant |
| Random forest | -0.0389 | +0.0921 | +0.1310 | 0.8834 | 0.8696 | -0.016 (-1.40) | 132 | worse than the constant |
| Ridge (auxiliary) | -0.0292 | +0.0177 | +0.0469 | 0.8797 | 0.8696 | -0.005 (-0.33) | 132 | worse than the constant |

**Gap** is in-sample R² minus out-of-sample R². A large positive gap means the model explains the window it was fitted on and not the month it then scores, which is memorisation rather than learning.

## 2. Does more training data help? (learning curve)

| Training window | R² out of sample | R² in sample | Gap | Mean IC (t) | Folds |
|---|---:|---:|---:|---:|---:|
| 3m | -0.1782 | +0.4973 | +0.6756 | -0.017 (-1.89) | 132 |
| 6m | -0.1222 | +0.3227 | +0.4449 | -0.012 (-1.20) | 132 |
| 12m | -0.0841 | +0.2026 | +0.2866 | -0.014 (-1.48) | 132 |
| 24m | -0.0589 | +0.1305 | +0.1894 | -0.015 (-1.76) | 132 |
| 36m | -0.0450 | +0.1061 | +0.1511 | -0.008 (-1.02) | 132 |
| expanding | -0.0373 | +0.0807 | +0.1180 | -0.004 (-0.53) | 132 |

A model learning a real relationship improves, or at least stops degrading, as the window lengthens. One fitting noise mostly trades one kind of variance for another. The pipeline's fixed twelve-month window is one row of this table, and this is the first test of whether it was the right choice.

## 3. Is the ordering stable month to month?

- Mean rank correlation between this month's and last month's predictions for the same stocks: **+0.215** (132 adjacent pairs)
- The same statistic for the realised target: **-0.004**

The second line bounds the first: no honest model can be more persistent than the thing it predicts. A prediction correlation far above the target's means the model is repeating itself rather than tracking anything.

## 4. Where do the errors sit?

### By year

| Year | R² out of sample | IC | Rows |
|---|---:|---:|---:|
| 2015 | -0.1722 | -0.054 | 1953 |
| 2016 | -0.0478 | +0.036 | 4892 |
| 2017 | -0.0472 | -0.011 | 5103 |
| 2018 | -0.0459 | -0.009 | 5250 |
| 2019 | -0.0226 | +0.061 | 5379 |
| 2020 | -0.0943 | -0.061 | 5507 |
| 2021 | -0.0780 | -0.071 | 5592 |
| 2022 | -0.0982 | -0.055 | 5684 |
| 2023 | -0.0912 | -0.101 | 5776 |
| 2024 | -0.0705 | -0.056 | 5829 |
| 2025 | -0.0772 | -0.053 | 5910 |
| 2026 | -0.0516 | -0.025 | 3482 |

### By ex-ante volatility bucket (0 = calmest fifth, 4 = most volatile)

| Bucket | R² out of sample | IC | Rows |
|---|---:|---:|---:|
| 0 | -0.0665 | -0.036 | 12123 |
| 1 | -0.0625 | -0.037 | 12044 |
| 2 | -0.0556 | -0.023 | 12049 |
| 3 | -0.0598 | -0.017 | 12044 |
| 4 | -0.0952 | -0.013 | 12097 |

The top bucket matters most: those are the names a hurdle selector tends to buy, so an error concentrated there is an error in the part of the cross-section the product actually acts on.

## Reading guide

- These are model-quality results, not investment results. A model can predict poorly and still sit inside a portfolio that rises, because the market rises; the portfolio evidence is reported separately.
- Hyperparameters are fixed at the values in `model.ReturnModel` and were not tuned against any of these numbers. No search was run, so nothing here is a best-of-many result, and no held-out tuning split is needed.
- Out-of-sample research only; not financial advice.
