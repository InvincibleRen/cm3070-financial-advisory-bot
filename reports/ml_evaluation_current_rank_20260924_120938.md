# Machine-Learning Evaluation of the Prediction Task

Generated at: 2026-09-24 12:09:38

## Setup

- Task: regression on the risk_adjusted target
- Universe: today's S&P 500 list (survivorship-biased)
- Features: 16 (technical,fundamental), rank per month
- Rebalances: 140 (2015-01 to 2026-08)
- Training window: 12 rebalances rolling
- First rebalance: 2015-01-01
- Baseline: Training-window mean (constant)

Every figure is out of sample. Each month's model is trained only on earlier rebalances and scores that month's cross-section; the metrics pool the rows of all months, so a wide month counts for more than a thin one. The baseline throughout is the mean of the model's **own training window** — the only constant a forecaster could have used without seeing the future (Campbell and Thompson, 2008). R² above zero means the model predicts better than that constant; below zero means worse.

## 1. Does the model beat a constant? (out-of-sample R²)

| Model | R² out of sample | R² in sample | Gap | RMSE | Baseline RMSE | Mean IC (t) | Folds | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| XGBoost | -0.0747 | +0.1919 | +0.2665 | 0.8960 | 0.8692 | -0.013 (-1.36) | 132 | worse than the constant |
| Random forest | -0.0321 | +0.0845 | +0.1167 | 0.8810 | 0.8692 | -0.014 (-1.27) | 132 | worse than the constant |
| Ridge (auxiliary) | -0.0251 | +0.0163 | +0.0413 | 0.8780 | 0.8692 | -0.005 (-0.42) | 132 | worse than the constant |

**Gap** is in-sample R² minus out-of-sample R². A large positive gap means the model explains the window it was fitted on and not the month it then scores, which is memorisation rather than learning.

## 2. Does more training data help? (learning curve)

| Training window | R² out of sample | R² in sample | Gap | Mean IC (t) | Folds |
|---|---:|---:|---:|---:|---:|
| 3m | -0.1702 | +0.4821 | +0.6523 | -0.020 (-2.26) | 132 |
| 6m | -0.1148 | +0.3090 | +0.4238 | -0.015 (-1.67) | 132 |
| 12m | -0.0747 | +0.1919 | +0.2665 | -0.013 (-1.36) | 132 |
| 24m | -0.0495 | +0.1230 | +0.1725 | -0.007 (-0.89) | 132 |
| 36m | -0.0403 | +0.0998 | +0.1401 | -0.004 (-0.50) | 132 |
| expanding | -0.0345 | +0.0751 | +0.1097 | -0.005 (-0.64) | 132 |

A model learning a real relationship improves, or at least stops degrading, as the window lengthens. One fitting noise mostly trades one kind of variance for another. The pipeline's fixed twelve-month window is one row of this table, and this is the first test of whether it was the right choice.

## 3. Is the ordering stable month to month?

- Mean rank correlation between this month's and last month's predictions for the same stocks: **+0.227** (132 adjacent pairs)
- The same statistic for the realised target: **+0.000**

The second line bounds the first: no honest model can be more persistent than the thing it predicts. A prediction correlation far above the target's means the model is repeating itself rather than tracking anything.

## 4. Where do the errors sit?

### By year

| Year | R² out of sample | IC | Rows |
|---|---:|---:|---:|
| 2015 | -0.1599 | -0.075 | 2294 |
| 2016 | -0.0508 | +0.012 | 5561 |
| 2017 | -0.0409 | -0.029 | 5628 |
| 2018 | -0.0471 | -0.011 | 5670 |
| 2019 | -0.0267 | +0.041 | 5711 |
| 2020 | -0.0616 | -0.028 | 5790 |
| 2021 | -0.0703 | -0.068 | 5842 |
| 2022 | -0.0878 | -0.046 | 5898 |
| 2023 | -0.0833 | -0.068 | 5913 |
| 2024 | -0.0590 | -0.038 | 5945 |
| 2025 | -0.0813 | -0.056 | 5969 |
| 2026 | -0.0540 | -0.019 | 3490 |

### By ex-ante volatility bucket (0 = calmest fifth, 4 = most volatile)

| Bucket | R² out of sample | IC | Rows |
|---|---:|---:|---:|
| 0 | -0.0610 | -0.022 | 12794 |
| 1 | -0.0520 | -0.017 | 12714 |
| 2 | -0.0571 | -0.027 | 12706 |
| 3 | -0.0565 | -0.019 | 12714 |
| 4 | -0.0843 | -0.022 | 12783 |

The top bucket matters most: those are the names a hurdle selector tends to buy, so an error concentrated there is an error in the part of the cross-section the product actually acts on.

## Reading guide

- These are model-quality results, not investment results. A model can predict poorly and still sit inside a portfolio that rises, because the market rises; the portfolio evidence is reported separately.
- Hyperparameters are fixed at the values in `model.ReturnModel` and were not tuned against any of these numbers. No search was run, so nothing here is a best-of-many result, and no held-out tuning split is needed.
- Out-of-sample research only; not financial advice.
