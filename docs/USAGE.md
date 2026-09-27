# CLI usage reference

The web app (`./start.sh`, see the main [README](../README.md)) needs none of this. This page is for the command-line tools underneath it.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt    # core ML pipeline + backend + test deps, in one command
```

Later sessions, just re-activate:

```bash
source venv/bin/activate           # Windows: venv\Scripts\activate
```

All commands below run from the project root with the virtual environment active. The CLI tools live under `src.cli.*` (code is organised by function; see *Project layout* below).

## 1. Get a live recommendation

```bash
python -m src.cli.cli AAPL
```

`--basic` gives the scope-faithful prototype mode (SMA + crossover only, no RSI/MACD):

```bash
python -m src.cli.cli AAPL --basic
```

## 2. Single-split backtest

Backtests the rule-based strategy against buy-and-hold for one ticker; writes a markdown report to `reports/`.

```bash
python -m src.cli.backtest_cli AAPL --period 5y --cash 10000
```

| Option | Meaning | Default |
|---|---|---|
| `--period` | History to download (e.g. `2y`, `5y`, `10y`) | `5y` |
| `--cash` | Starting cash for the simulation | `10000` |

## 3. Multi-stock backtest

```bash
python -m src.cli.multi_backtest_cli AAPL MSFT NVDA TSLA SPY --period 5y
```

## 4. Walk-forward backtest (recommended evaluation)

Evaluates the strategy on a sequence of unseen windows: cumulative & annualised return, Sharpe, max drawdown, hit rate, per fold and aggregated, net of transaction costs. Writes a markdown report to `reports/`.

```bash
# One ticker, default settings (train 252 days, test 63 days, 10 bps costs)
python -m src.cli.walkforward_cli AAPL

# Custom windows, longer history, several tickers
python -m src.cli.walkforward_cli AAPL SPY --period 8y --train 252 --test 63 --cost 0.001
```

| Option | Meaning | Default |
|---|---|---|
| `--period` | History to download | `8y` |
| `--train` | Training window, in trading days | `252` (~1 year) |
| `--test` | Test window, in trading days | `63` (~1 quarter) |
| `--step` | Days to advance between folds | test size (contiguous) |
| `--cost` | Transaction cost per side, as a fraction (`0.001` = 10 bps) | `0.001` |
| `--rf` | Annual risk-free rate used for the Sharpe ratio | `0.0` |
| `--no-save` | Print results only; do not write a report | off |

## 5. Cross-sectional ML stock selection (the core)

Loads the universe, builds the leakage-safe feature matrix and forward-return labels, then walk-forward selects the top-N each month and evaluates, net of transaction costs, against an equal-weight universe benchmark and SPY. Reports both the gradient-boosting ranker and the logistic baseline, plus prediction-quality metrics (precision@N, mean AUC vs the ~0.5 base rate) and a per-year consistency breakdown (strategy vs benchmark, monthly win-rate). Writes a markdown report to `reports/`.

```bash
# Default universe (built-in ~27 large-caps), top-3, both rankers
python -m src.cli.select_cli

# Custom universe / settings
python -m src.cli.select_cli AAPL MSFT NVDA AMZN GOOGL META --top-n 5 --start 2015-01-01
```

| Option | Meaning | Default |
|---|---|---|
| `tickers` | Universe to select from (positional) | built-in `DEFAULT_UNIVERSE` |
| `--start` / `--end` | History date range | `2015-01-01` / today |
| `--top-n` | Stocks held each rebalance | `3` |
| `--model` | `gbm`, `logistic`, or `both` | `both` |
| `--cost` | Per-turnover transaction cost (`0.001` = 10 bps; `0` for gross) | `0.001` |
| `--min-train` | Prior rebalances required before a fold is scored | `6` |
| `--normalize` | Leakage-safe cross-sectional feature standardisation per rebalance date: `rank`, `zscore`, or `none` | `rank` |
| `--no-save` | Print only; do not write a report | off |

**Cross-sectional normalisation (`--normalize`).** Selection is a *relative* problem, so each feature is standardised within each rebalance date's own cross-section (default `rank` = within-date percentile). This only looks at one date's cross-section, so it's leakage-safe by construction, and it lifts the ranker's out-of-sample AUC/precision (gbm mean AUC 0.511 → 0.524). `--normalize none` gives the older raw-level behaviour.

**Pipeline internals.** `universe.py`, `features.py`, `labels.py` are library modules (no CLI of their own): they build the cached price/fundamentals panel, the leakage-safe feature matrix, and the forward-return labels that the ranker (`model.py`) consumes via the walk-forward selector (`select.py`). All covered by offline tests.

### 5b. Significance of the ranker (permutation + bootstrap)

Tests whether the selector's out-of-sample skill is real rather than luck: shuffles outcome labels within each rebalance date's cross-section (100× by default) for a "no-skill" null and a permutation p-value on mean AUC / precision@N; bootstraps per-fold AUCs for a 95% CI. Leakage-free (only labels are permuted; no re-training). Writes a markdown report to `reports/`.

```bash
python -m src.cli.significance_cli                        # both models, 100 shuffles
python -m src.cli.significance_cli --model gbm --permutations 200
```

### 5c. Robustness / self-honesty (Deflated Sharpe + sensitivity + bootstrap)

Checks the selector's edge isn't (a) a lucky best-of-many-trials, (b) a cherry-picked setting, or (c) carried by a few months: Deflated Sharpe Ratio (discounts for configs tried), a top-N × cost sensitivity grid, bootstrap 95% CIs on annualised return / Sharpe. Writes a markdown report to `reports/`.

```bash
python -m src.cli.robustness_cli                         # gbm, deflate for 20 trials
python -m src.cli.robustness_cli --model logistic --n-trials 30
```

## 6. News sentiment (extension, optional)

FinBERT scores recent headlines into a sentiment feature that joins the core feature matrix; its value is proved by an ablation (selector with vs without the sentiment column). Removable: deleting `src/sentiment/` leaves the core selector fully working.

Heavy dependencies are kept out of the core install:

```bash
# Apple Silicon (arm64) / Linux / Windows - default torch backend
pip install -r requirements/extension-a.txt

# x86 macOS / Intel Mac - torch-free backend
pip install -r requirements/extension-a-onnx-local.txt
```

| Backend | Flag | Runs on | Notes |
|---|---|---|---|
| `torch` (default) | `--backend torch` | Apple Silicon / Linux / Windows | Normal `pip install torch`; safetensors avoids the `torch.load` CVE gate. |
| `onnx-local` | `--backend onnx-local` | **x86 macOS / Intel Mac**, + everywhere | Torch-free: loads a pre-exported `model.onnx` on ONNX Runtime + `tokenizers`. |
| `onnx` (optimum) | `--backend onnx` | Where torch ≥ 2.4 exists | Exports via `optimum` (needs torch to trace); not for Intel Mac. |

**macOS note:** PyTorch's last x86-macOS (Intel Mac) build is 2.2.2, past what the modern ML stack needs, so neither `torch` nor optimum `onnx` installs there; use `--backend onnx-local`, which never imports torch. On Apple Silicon, create the venv from an **arm64** Python (e.g. `/opt/homebrew/bin/python3`), not x86/Rosetta, and the default `torch` backend works with Metal (MPS).

Live "sentiment right now" demo (yfinance only exposes recent news, so this is a demonstration, not a historical backtest):

```bash
python -m src.cli.sentiment_cli                       # full DEFAULT_UNIVERSE
python -m src.cli.sentiment_cli AAPL MSFT NVDA        # specific tickers
python -m src.cli.sentiment_cli --lookback-days 30    # wider window = more headlines

# x86 macOS / Intel Mac - add the torch-free backend flag:
python -m src.cli.sentiment_cli AAPL MSFT NVDA --backend onnx-local
```

Prints a sentiment score per ticker plus a "market mood" average across tickers with news. A longer `--lookback-days` (e.g. 30) gives each score more headlines to average.

To use sentiment inside the selector, pass the sentiment frame into the feature matrix and run the ablation:

```python
from src.sentiment.finbert import FinBERTScorer
from src.sentiment.news_source import YFinanceNewsSource
from src.selection.features import build_feature_matrix
from src.selection.select import run_ablation

sent = FinBERTScorer().build_sentiment_feature(YFinanceNewsSource(), tickers, rebalance_dates)
fm = build_feature_matrix(prices, fundamentals, rebalance_dates, sentiment=sent)
results = run_ablation(fm, labels, prices, rebalance_dates)   # {"with_sentiment", "without_sentiment"}
```

## 7. Per-stock direction forecast (extension, optional)

A short-term up/down confirmation signal for the stocks the core selected, run only on the top-N. Framework-light (same scikit-learn stack as the core, no torch/TF; runs on any machine including Intel Macs) and evaluated honestly: out-of-sample directional accuracy against the naive base rate, plus AUC. Removable: deleting `src/direction/` leaves the core selector fully working.

```bash
python -m src.cli.direction_cli AAPL MSFT NVDA          # confirm the core's picks
python -m src.cli.direction_cli AAPL --model logistic --horizon 10 --period 8y
python -m src.cli.direction_cli AAPL --model baseline   # momentum-persistence floor
python -m src.cli.direction_cli AAPL --no-calibrate     # raw (uncalibrated) P(up)
```

Prints, per ticker: out-of-sample accuracy, base rate, edge (accuracy − base rate), AUC, Brier score, calibration error (ECE), and current P(up) over the next `--horizon` days; writes a markdown report to `reports/`. Single-stock direction is close to a coin flip, so a small or negative edge is an honest, acceptable outcome.

**Pooled cross-stock path (`src/direction/pooled.py`).** The per-stock model relearns from only ~250 rows each fold, which overfits noise. The pooled path trains one model per rebalance date across a whole basket of stocks (tens of thousands of rows), with the same leakage guarantees (causal features, forward labels, a horizon-day embargo, a trailing window). Pooling plus calibration lifts accuracy to the base rate and shrinks calibration error; selective prediction (acting only on the most confident calls) recovers a small positive edge. Reads the local price cache; fully offline and reproducible:

```bash
python -m src.cli.pooled_direction_cli                    # default 30-name basket, horizon 5
python -m src.cli.pooled_direction_cli AAPL MSFT NVDA     # a custom basket
python -m src.cli.pooled_direction_cli --no-calibrate     # ablate the calibration step
```

Prints the pooled out-of-sample metrics and a selective-prediction table (accuracy when only the most confident X% of calls are kept); writes a `direction_pooled_*.md` report. AUC stays near 0.5, which honestly locates the ceiling: single-stock short-horizon direction carries little ranking information; the pooled path's value is trustworthy probabilities and a usable confidence ranking, not a large predictive edge.

**Probability calibration (`--calibrate`, on by default).** The displayed P(up) is calibrated to read as a true frequency (a "63%" day really rises ~63% of the time). Fit only on held-out CV folds of each training window (leakage-safe); cuts calibration error sharply (AAPL: ECE 0.17 → 0.06) without costing discrimination. `--no-calibrate` for the raw score.

## 8. Web interface (React + FastAPI)

A React single-page front end (`frontend/`) talks to a FastAPI back end (`src/api/main.py`) over the UI-framework-free logic layer in `src/ui/data_access.py`. Reuses the core scikit-learn stack (no torch), so it runs on any machine including Intel Macs. Data is fetched on demand per ticker and cached within the session, so re-runs are instant.

```bash
./start.sh   # first run auto-installs deps, then starts FastAPI (:8000) + Vite (:5173)
```

Optional FinBERT news-sentiment panel: `source venv/bin/activate && pip install -r requirements/extension-a-onnx-local.txt`.

See `src/ui/README.md` for the view-by-view breakdown.

## 9. Run the tests

Everything the tests need is already in `requirements.txt` (including `httpx`, used by the API tests). The full suite runs offline except a few live-data checks, and takes a few minutes.

```bash
# Whole suite
python -m pytest

# Just the ML-core selection pipeline (Phases 0-2)
python -m pytest tests/test_universe.py tests/test_features.py tests/test_labels.py \
                 tests/test_model.py tests/test_select.py tests/test_select_cli.py

# Extension A (sentiment), fully offline (no model download)
python -m pytest tests/test_sentiment.py tests/test_sentiment_onnx.py

# Extension B (direction forecast), fully offline
python -m pytest tests/test_direction.py

# Web UI logic layer, fully offline (no web server needed)
python -m pytest tests/test_ui.py
```

## 10. Audit the explanations

The advisor's central claim is that every sentence it shows is emitted by the condition that produced it, so it can't offer a justification it didn't use. This script measures that claim: it generates recommendations across the universe at many historical cut-offs and, for each sentence, re-derives the claim from the indicator snapshot and checks that it holds. The check is written against the indicator values rather than the advisor's control flow, so it audits the output instead of restating the code.

```bash
python scripts/run_faithfulness.py
```

Runs entirely off the committed CSV cache, no network needed. Result written to `reports/faithfulness_results.json`, which is committed: 11,975 recommendations, 59,952 sentences, complete coverage and accuracy, no violation recorded. Section 5.6 of the final report quotes these figures.

## Project layout

Code is organised **by function**, matching the project's core-plus-extensions design (main function ~70%, two removable extensions ~15% each). Each package has its own `README.md`.

```
src/
  common/                      # shared infrastructure (used by core + all extensions)
    data/yfinance_provider.py  #   market data (quotes, intraday, historical)
    indicators.py              #   SMA, MACD, ADX (trend strength), crossover detection
    metrics.py                 #   finance metrics (Sharpe, drawdown, hit rate, costs)
    walkforward.py             #   model-agnostic walk-forward evaluation engine
    backtester.py              #   single-split rule-based backtest vs buy-and-hold
  selection/                   # CORE - cross-sectional ML stock selection (~70%)
    universe.py                #   fixed universe + multi-ticker/fundamentals loader
    features.py                #   leakage-safe feature matrix
    labels.py                  #   forward-return labels vs universe median
    model.py                   #   ranker (gradient boosting) + logistic baseline
    select.py                  #   top-N selection + walk-forward eval + ablation
  sentiment/                   # Extension A - FinBERT news sentiment (~15%)
    news_source.py             #   leakage-safe headline feed (in-memory + yfinance)
    finbert.py                 #   FinBERT scoring -> per-(date,ticker) sentiment feature
  direction/                   # Extension B - per-stock direction forecast (~15%)
    forecaster.py              #   DirectionForecaster (gbm/logistic/baseline) + features/labels
    evaluate.py                #   walk-forward directional evaluation (accuracy vs base rate)
  explanation/advisor.py       # rule-based signal + plain-language explanation
  ui/                          # UI-framework-free logic layer for the web app
    data_access.py             #   provider-injectable logic layer (single-stock, selection, sentiment)
    evidence.py                #   study results -> plain sentences shown beside each signal/pick
    profile.py                 #   risk/horizon profile -> view settings
  api/main.py                  # FastAPI backend serving the React frontend (frontend/)
  cli/                         # command-line entry points (cli, backtest, multi, walkforward, select, sentiment, direction)
scripts/                       # evaluation drivers (run_faithfulness, run_latency, run_methodology) -> write to reports/
tests/                         # pytest suite
reports/                       # generated backtest reports + committed evidence (faithfulness/methodology/latency results)
requirements/                  # base.txt (core) + api.txt, extension-a.txt, extension-a-onnx-local.txt (optional)
```

The ML core (Phases 0-2) is complete: universe loader, feature matrix, labels, the ranker + baseline, and the walk-forward top-N selector. Extension A (`sentiment/`) and Extension B (`direction/`) are both implemented and offline-tested, and the React + FastAPI web app (`frontend/` + `src/api/`, over the `ui/` logic layer) ties everything together in a single-stock investigator. All planned components are in place.

## Project direction

Built as a main function plus two removable extensions:

1. **Core (~70%): cross-sectional stock selection.** From a universe of stocks with many indicators (PE, sector, earnings, momentum…), rank and select the few most likely to be profitable, then tune the model.
2. **Extension B (~15%): per-stock direction forecast.** Time-series model that feeds financial factors for one asset and predicts up/down, run only on the selected top-N as a confirmation signal.
3. **Extension A (~15%): NLP news/sentiment.** FinBERT scores market news and sentiment as an added feature, proved via an ablation study.
