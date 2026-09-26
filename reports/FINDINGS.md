# Findings: Does Technical Analysis Earn Its Keep? An Evidence Check of the Advisor Bot

This file collects the final results behind the report's evaluation chapter and points
to the generated report each number comes from. All results are out of sample, from
walk-forward runs, on dividend-adjusted prices.

## Summary

Two answers to two different questions. **As a forecaster the model fails**: out-of-sample
R² is negative for every model tried (XGBoost −0.084, random forest −0.039, ridge −0.029),
i.e. worse than predicting the training-window mean, while in-sample R² stays positive —
memorisation, not learning. Per-stock month-ahead prediction is not something this model
can do. **As an instrument for locating factors it works**: the feature-group ablation
(§3.8) runs one framework with the inputs swapped and shows that beta and volatility carry
the selection while the ten technical indicators do not. That finding is confirmed directly —
on a survivorship-free universe, ranking on the beta factor picked stocks that beat the average
same-month constituent by 2.09% per holding month (51.3% win rate vs 48.4%, p < 0.001) and
compounded to 33.8% a year against SPY's 14.2%; ranking on volatility is marginally significant
(p = 0.026). **The fitted ML models' own picks, by contrast, do not beat random selection**
(p from 0.165 to 0.977) and every one finished below the index. So the value is in the factor the
model points to, not in the model's picks; the advisor's own rule-based technical signals show no
advantage at all. Results that looked strong on today's constituent list (42% a
year with a significant alpha) came from survivorship bias and disappear under the
survivorship-free test. Comparing feature sets inside one model, **beta and volatility are
the more important factors**: two risk measures guided the selector better (14.1% a year)
than the ten technical indicators (12.6%) or all seventeen features (12.1%), and adding the
indicators on top of them made it worse (1.4%). The natural reading is that the stocks
earning the higher returns were the higher-risk stocks, in a predominantly rising market.
The only positive sign,
fundamentals helping the selector in 2023–2026, rests on a short, concentrated sample
with known data issues and is reported as preliminary. The bot therefore presents each
signal alongside its own track record and uses a low-cost index fund as the reference.

## 1. Research questions

0. Does the model predict its own training target out of sample at all — better than a
   constant — and does it generalise or memorise?
1. Can the ML selectors (XGBoost, random forest; ridge / logistic as baselines) pick
   S&P 500 stocks that beat the index over the next month, net of costs?
2. Do apparent gains survive a survivorship-free universe and a permutation test?
3. Do the advisor's technical Buy / Sell signals predict what a stock does next?
4. Do stocks rise or fall around S&P 500 additions and removals?
5. Do fundamental features add value where the data exists (2023 onwards)?

## 2. Method (what makes the tests honest)

| Threat | Control |
|---|---|
| Survivorship bias | Point-in-time S&P 500 membership (github.com/fja05680/sp500), 30 verified ticker-change aliases; per-month coverage reported (median 92%, lowest 76% in 2015) |
| Look-ahead | Features use data up to the rebalance date; fundamentals only after period end + 60 days; walk-forward trains strictly on earlier months |
| No baseline for the prediction task | Out-of-sample R2 against the mean of the model's own training window (Campbell-Thompson); in-sample R2 reported beside it so memorisation is visible |
| Counting correlated events as independent | Event studies average each date first; overlapping holding periods use Newey-West t-stats |
| Reading one lucky window | Walk-forward over 140 monthly rebalances; per-year and per-window breakdowns; label-permutation test |

Targets compared: (a) beat the cross-sectional median (binary, original), (b) next-month
return minus beta x SPY, divided by volatility (continuous, the current default), and
(c) plain next-month return minus SPY, used where a risk feature has to be able to compete
(target (b) defines beta away). Selection: top 5 by score, or names above a pre-set
hurdle, SPY held when none qualify.

**How results are judged, and where risk belongs.** The headline judgement is on
**return** — annualised return, win rate, expectancy, IC, out-of-sample R2. Sharpe ratios,
betas and drawdowns are reported as *descriptive* columns, not as pass/fail tests: an
investor willing to carry more risk is entitled to want the higher-returning book, so a
lower Sharpe is not by itself a failure. Risk enters only at the interpretation stage, and
in one specific form: the feature comparison (§3.8) finds that **beta and volatility are
the more important factors** — they guided the selector better than the ten technical
indicators — so the stocks that earned the high returns in this sample were most likely
the high-risk stocks. That is a statement about which factors carry the result and about
what the returns mean, not a claim that the model "only picks risk".

## 3. Results

### 3.1 The model does not predict its own target (`ml_evaluation_historical_rank_20260924_000023.md`)

Before asking what a portfolio of the picks earned, ask whether the model predicts the
thing it is trained on. Baseline throughout: the mean of the model's **own training
window**, the only constant available at forecast time (Campbell–Thompson). R² > 0 means
the model beats that constant; R² < 0 means it is worse than a constant.

| Model | R² out of sample | R² in sample | Gap | Mean IC (t) |
|---|---:|---:|---:|---:|
| XGBoost | **−0.0841** | +0.2026 | +0.2866 | −0.014 (−1.48) |
| Random forest | **−0.0389** | +0.0921 | +0.1310 | −0.016 (−1.40) |
| Ridge (auxiliary) | **−0.0292** | +0.0177 | +0.0469 | −0.005 (−0.33) |

All three are worse than a constant out of sample, and the ranking is the giveaway: the
more capacity a model has, the better it fits its training window and the worse it
predicts. That is memorisation, measured directly rather than inferred from attribution.

**Learning curve** (same 132 test months at every window):

| Training window | 3m | 6m | 12m (pipeline default) | 24m | 36m | expanding |
|---|---:|---:|---:|---:|---:|---:|
| R² out of sample | −0.1782 | −0.1222 | −0.0841 | −0.0589 | −0.0450 | −0.0373 |
| Train−test gap | +0.6756 | +0.4449 | +0.2866 | +0.1894 | +0.1511 | +0.1180 |

More data monotonically reduces the gap and never lifts R² above zero. Two readings: the
12-month window the pipeline fixed was not the best choice (an expanding window halves the
gap), and even the best window has nothing to learn — extra data only buys variance
reduction on a signal that is not there.

**Prediction stability**: the rank correlation between this month's and last month's
predictions for the same stocks is **+0.215**, while the same statistic for the realised
target is **−0.004**. The model is roughly fifty times more persistent than the thing it
predicts, i.e. it repeats its own ordering (slow-moving features such as volatility and
distance from the 52-week high) rather than tracking anything that recurs in the target.

**Error analysis**: R² is negative in all twelve calendar years (−0.023 in the best,
2019; −0.172 in 2015), so this is not one bad regime. By ex-ante volatility bucket it is
worst in the most volatile fifth (−0.095 against −0.056 to −0.067 elsewhere) — which is
exactly the part of the cross-section a hurdle selector buys from.

**The survivorship defect does not touch this layer**
(`ml_evaluation_current_rank_20260924_002929.md`). Re-running the identical diagnostics on
today's constituent list gives R² of −0.0747 / −0.0321 / −0.0251 and a gap of +0.2665 /
+0.1167 / +0.0413 — indistinguishable from the point-in-time figures above, while the same
change lifted the portfolio from 9.0% to 42.3% a year (§3.2). The biased sample therefore
bought no predictive improvement whatsoever: the 42.3% came from which stocks were in the
list, not from the model forecasting them better. Part A measures the model; §3.2 measures
the sample.

### 3.2 Survivorship bias manufactures skill

Same models, features, folds and costs; only the universe differs. Months with no pick
hold SPY; CAPM alpha is net of the risk-free rate (`hurdle_current_rank_20260922_122342.md`
vs `hurdle_historical_rank_20260922_122643.md`).

| | Today's list: CAGR / alpha (t) | Point-in-time: CAGR / alpha (t) |
|---|---|---|
| XGBoost regressor | 42.3% / +24.3% (2.56) | 9.0% / −0.9% (−0.11) |
| Random forest regressor | 26.8% / +13.6% (1.34) | 13.3% / +4.5% (0.49) |
| Ridge (auxiliary) | 33.9% / +20.0% (2.48) | 6.3% / −1.6% (−0.22) |
| XGBoost classifier, median label (the app's selector) | 25.2% / +9.7% (1.52) | 3.2% / −5.2% (−1.06) |
| No model: buy the 5 most volatile | 58.7% / +30.7% (2.43) | 7.0% / −7.1% (−0.62) |
| SPY | 14.2% | 14.2% |

On today's list a rule with no model at all beats every model, and its alpha stays
"significant" after removing a beta of 2.4: the list only contains the volatile companies
that went on to succeed. On the point-in-time universe the same rule earns 7% with a −71%
drawdown.

### 3.3 Survivorship-free selectors, 2015–2026

Point-in-time universe, 140 monthly rebalances, 10 bps per unit turnover, on the **frozen
2026-09-24 price snapshot** (`hurdle_historical_rank_20260924_041758.md`). **Primary test:
`p (random)`** — the share of 2,000 random books drawn from the whole cross-section of the
same months that did at least as well. Expectancy is per pick, minus SPY; the universe
averaged −0.16%.

| Selector | Expectancy / pick | Win rate | Payoff | **p (random)** | CAGR | Beta |
|---|---:|---:|---:|---:|---:|---:|
| Rank on beta alone | **+2.09%** | 51.3% | 1.37 | **0.000** | **33.8%** | 2.28 |
| Rank on volatility alone | **+0.46%** | 47.2% | 1.20 | **0.026** | 8.1% | 2.08 |
| Random forest regressor | +0.12% | 46.4% | 1.20 | 0.165 | 10.3% | 1.04 |
| XGBoost regressor | −0.34% | 45.9% | 1.07 | 0.724 | 5.6% | 0.93 |
| Ridge (auxiliary) | −0.37% | 45.6% | 1.07 | 0.760 | 5.5% | 0.71 |
| XGBoost classifier, median label | −0.73% | 44.1% | 1.03 | 0.977 | 0.9% | 1.08 |
| SPY | | | | | 14.2% | 1.00 |

**None of the four fitted ML models beats random selection** (p from 0.165 to 0.977; three of
four have negative per-pick expectancy), and on the portfolio side every fitted model finished
below the index. Only the single-factor rules beat random: ranking on beta strongly (+2.09%,
p<0.001, 33.8% a year) and ranking on volatility marginally (p=0.026). The earlier 0922 run had
shown the random forest significant (p=0.001); that did not survive a data revision, which is
itself the point — the fitted models' edge is within noise, while the deterministic factor rules
are stable. Positive per-pick expectancy can still compound below the index (see the volatility
rule, and §1b): expectancy is arithmetic, an investor earns the geometric return, and a five-name
book at ~46% volatility gives far more back to the variance term than an index at ~15%.

Secondary, and a different question: once each pick's random stand-in is drawn from its 20
nearest volatility neighbours instead of the whole cross-section, only the beta ranking stays
significant (p = 0.016 beta-adjusted). That comparison is built to strip
out return earned by holding volatile names, so it does not test whether the selection was
worth making — it locates where it comes from, and the answer agrees with §3.8.

Raw (un-ranked) features change little. The fixed hurdle rarely binds (99% of months
invested), because five of ~500 names almost always clear it.

### 3.4 Why: what the model learned (SHAP)

TreeSHAP on every out-of-sample prediction of the XGBoost regressor; contributions sum to
the predictions to within 1e-6 (`explain_xgb_historical_rank_20260922_123447.md`,
`explain_xgb_current_rank_20260922_123152.md`).

- Reliance is spread almost evenly: each of the ten technical features carries 7–10%, each
  fundamental 2–3% (fundamentals only exist from 2023).
- None of the 16 features has a significant out-of-sample IC on its own (all |t| < 1.3 on
  the point-in-time universe).
- Adjacent momentum horizons are learned with alternating signs (1-month +, 3-month −,
  6-month +, 12-1-month −), which no real momentum or reversal effect produces.
- The learned rules change with the universe: on today's list the model favours the stocks
  furthest below their 52-week high; on the point-in-time universe that preference almost
  vanishes.
- A pre-fix hint that profitability (ROE, net margin) carried signal disappeared once the
  fundamentals were read correctly (ROE own-IC t from 1.59 to 0.27).

Together: the model fits noise; on the biased universe that noise happened to line up with
survivorship.

### 3.5 The advisor's technical signals (`signal_study_20260924_135513.md`)

596 stocks, 1.25 million member stock-days, 2015–2026. "vs universe" = minus the average
S&P 500 member over the same days (the cleanest test of stock picking).

| Signal, held 20 days | Events | Beat same-day universe | Mean vs universe (t) | Mean vs SPY, 60 days (t) |
|---|---:|---:|---:|---:|
| Typical member stock-day (reference) | 1,248,457 | 49.4% | 0 by construction | −0.44% (−1.34) |
| New Buy signal | 33,103 | 49.7% | −0.08% (−1.23) | −0.77% (−2.10) |
| Any Buy day | 312,898 | 49.1% | −0.09% (−1.87) | −0.80% (−2.63) |
| New Sell signal | 24,840 | 49.3% | −0.01% (0.39) | −0.27% (−1.01) |
| Any Sell day | 202,404 | 49.8% | +0.06% (0.37) | +0.07% (−0.92) |

Buy signals were no better than a typical stock and trailed SPY over 60 days. New Sell
signals were followed by a slight one-week *out*performance (+0.06%, t = 2.35, mostly
before 2020), a short-term reversal opposite to the signal's meaning. All differences are
below 0.1% a week, under the cost of a round trip.

### 3.6 Index additions and removals (`index_events_20260922_101307.md`)

| Window (trading days, 0 = effective date) | Added (199): beat SPY / mean (t) | Demoted (100): beat SPY / mean (t) |
|---|---|---|
| −60 to −1 | 76% / +15.2% (6.48) | 20% / −13.5% (−5.33) |
| −5 to −1 | 47% / +1.2% (2.95) | 42% / −1.3% (−1.07) |
| 0 to +4 | 51% / +0.0% (0.48) | 43% / +0.7% (0.99) |
| 0 to +59 | 52% / −0.1% (−0.07) | 39% / +0.8% (0.22) |
| 0 to +249 | 39% / −4.4% (−1.85) | 41% / −1.7% (−0.43) |

The moves happen before the change (the index adds stocks that rose and removes ones that
fell); afterwards there is no reliable pattern, and no dependable post-removal rebound.
Announcement dates are not available, so the tradable part of the −5 to −1 window cannot
be separated from the announcement jump.

### 3.7 Fundamentals, 2023–2026 (`hurdle_historical_rank_from2023_*.md`)

Fundamentals only exist from 2023 (yfinance keeps about four years of statements), so this
is the only window where they can be tested. Same universe, same folds, with and without
the six fundamental features (37 out-of-sample months):

| XGBoost | With fundamentals | Technical only |
|---|---:|---:|
| CAGR / Sharpe / max drawdown | 30.8% / 1.00 / −16.6% | 14.8% / 0.55 / −27.3% |
| CAPM alpha (t), beta | +14.3% (0.78), 0.89 | −1.8% (−0.10), 1.16 |
| Beta-adjusted expectancy vs vol-matched random, p | +0.86% vs +0.09%, p = 0.161 | — |
| Pick vs random (whole cross-section), p | +1.22%, p = 0.012 | — |
| SPY | 19.7% / 1.46 / −7.6% | same |

Preliminary only: the alpha splits into −8.7% (t −1.16) in the first half and +36.1% (t 1.12) in
the second, neither significant; IC is about zero; a 46% hit rate with a 1.5 payoff ratio means a
few big winners; Sharpe (1.00) is below SPY's (1.46); against volatility-matched random the pick
edge is not significant (p = 0.16); and the fundamentals carry possible look-ahead (current share
count used for past PE / PB, a fixed 60-day disclosure lag, restated figures).

### 3.8 Which information guided the choice (`hurdle_historical_rank_feat-*_tgt-excess_spy_*.md`)

Which inputs actually drive the selection? Same XGBoost, same folds, universe, costs and
rules on the **frozen 2026-09-24 snapshot**; only the feature set changes. The default
risk-adjusted target divides by each stock's own volatility, which defines a risk feature
away before it can compete, so these runs use the plain benchmark-relative target
(`--target excess_spy`: forward total return minus SPY's, winsorised 1%/99%). Judgement is
on **return**; Sharpe, beta and drawdown are descriptive.

| Features | CAGR | Sharpe | Max DD | Beta | Alpha (t) |
|---|---:|---:|---:|---:|---:|
| risk (beta, volatility) | 14.08% | 0.59 | −35.8% | 1.37 | −1.88% (−0.33) |
| technical (10) | 12.58% | 0.52 | −43.2% | 1.53 | −4.29% (−0.63) |
| technical + risk | 1.44% | 0.21 | −50.2% | 1.72 | −16.34% (−2.54) |
| all 17 | 12.07% | 0.52 | −39.6% | 1.53 | −4.77% (−0.73) |
| SPY | 14.23% | 0.96 | −23.9% | 1.00 | — |

**Beta and volatility are the more important factors.** Given only those two, the model
returned 14.08% a year; given only the ten technical indicators it returned 12.58%; given
all seventeen, 12.07%. Two risk measures beat ten technical indicators, and adding the
technical indicators on top of them made things worse rather than better (1.44%). Whatever
the selector is doing well, it is doing through beta and volatility, not through the
moving averages, RSI and MACD the advisor's own signals are built on.

**Interpretation.** The natural reading is that the stocks earning the high returns in
this sample were the high-risk stocks: the better-returning configurations also carried
higher betas (1.37–1.53 against the index's 1.00) and deeper drawdowns. That is a caveat
about *what kind of return this is* — it is compensation for risk borne, in a
predominantly rising market, and it would look different in a falling one. It is not a
finding that the model is defective for choosing such stocks: an investor who wants that
exposure is entitled to want it, and the ranking above is a real statement about which
inputs carry the result.

What the comparison does not support is a claim of *selection* beyond those two factors.
Once each pick's random stand-in is drawn from its 20 nearest volatility neighbours, the
beta-adjusted pick p-values are 0.384 (risk), 0.358 (technical), 0.917 (technical+risk) and
0.385 (all): with volatility held constant, no feature set picks better than random.

## 4. Threats to validity

- Members with no yfinance prices (118 of 742, mostly acquired or bankrupt) are missing.
- Fundamentals: sparse before 2023, current share count, fixed reporting lag, restated values;
  some annual rows lost when an empty quarterly row shared the same period end.
- Many configurations were tested; isolated |t| slightly above 2 is weak evidence.
- One market regime dominated 2015–2026 (a long bull market led by large technology stocks).
- The live advisor scores golden / death crosses by strength (+1 to +3) where the replay
  uses +2, and adds a small intraday nudge.

## 5. What this means for the bot

Three things follow for the product.

1. **The technical signals the bot shows are the part with no support.** The 1.25M
   stock-day study (§3.5) finds no advantage after a Buy, and the feature comparison
   (§3.8) finds the ten technical indicators add nothing to two risk measures and
   subtract when combined with them. Each signal is therefore displayed with its own
   survivorship-free track record ("How Reliable Is This Signal?"), worded from the
   evidence file rather than hard-coded (`src/ui/evidence.py`,
   `data/evidence/evidence.json`).
2. **The model's useful role here is locating factors, not forecasting prices.** Returns
   are produced by factors; a model is an instrument for finding out which ones matter.
   Used that way — the feature-group ablation of §3.8, one framework with the inputs
   swapped — it gave a clear answer: beta and volatility carry the selection, the ten
   technical indicators do not. Ranking on beta then earned p < 0.001 and 33.8% a year,
   which confirms what the instrument pointed at. What the model cannot do (§3.1) is
   forecast a given stock's next month — and its own top-five picks do not beat random
   selection (§3.3) — so the product should present the *factor* finding and must not
   present per-stock scores as predictions.
3. **Returns of that kind come from carrying more market risk.** The high-return
   selectors ran betas above 2, in a predominantly rising market; the same exposure
   behaves very differently in a falling one. That belongs in the copy as a plain
   statement of what the return is, not as a warning that the selection is invalid.

All wording is general historical information, not personal advice.

## 6. Reproduction

```bash
python -m src.cli.hurdle_cli --universe historical --normalize rank --compare-classifier
python -m src.cli.hurdle_cli --universe historical --normalize none --compare-classifier
python -m src.cli.hurdle_cli --universe current --normalize rank --compare-classifier
python -m src.cli.hurdle_cli --first-rebalance 2023-01-01
python -m src.cli.hurdle_cli --first-rebalance 2023-01-01 --no-fundamentals
python -m src.cli.hurdle_cli --universe historical --features risk --target excess_spy
python -m src.cli.hurdle_cli --universe historical --features technical --target excess_spy
python -m src.cli.hurdle_cli --universe historical --features technical,risk --target excess_spy
python -m src.cli.hurdle_cli --universe historical --features technical,risk,fundamental --target excess_spy
python -m src.cli.ml_eval_cli --universe historical --normalize rank
python -m src.cli.ml_eval_cli --universe current --normalize rank
python -m src.cli.explain_cli --universe current --normalize rank
python -m src.cli.explain_cli --universe historical --normalize rank
python -m src.cli.signal_study_cli
python -m src.cli.index_events_cli
python -m src.cli.evidence_cli
```
