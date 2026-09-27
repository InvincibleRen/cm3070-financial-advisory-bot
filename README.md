# Financial Advisor Bot

An explainable stock-market advisor bot for retail investors (CM3070 project). For a ticker, it shows technical signals, an ML stock-selection read, and news sentiment — each in plain language, and each checked against its own out-of-sample track record instead of just asserted.

## Quick start

Needs **Python 3.10+** and **Node.js** on PATH.

```bash
./start.sh
```

First run installs everything (Python venv + npm) and starts the backend (`:8000`) and frontend (`:5173`). Open **http://localhost:5173**. `Ctrl+C` stops both.

> **macOS:** run this from a normal local folder (e.g. `~/Projects/`), not iCloud Drive — iCloud evicts files inside `node_modules` and breaks the Vite build (`operation timed out`). Fix: `rm -rf frontend/node_modules && ./start.sh`.

## What it does

- **Technical signals** — SMA/MACD crossover, with ADX as a trend-strength amplifier, explained in plain language.
- **ML stock selection** (core) — ranks a universe of stocks cross-sectionally each month, walk-forward evaluated net of costs, with significance and robustness checks against luck and overfitting.
- **Direction forecast** (extension) — a per-stock up/down confirmation signal for the ML pick, scored against the naive base rate.
- **News sentiment** (extension) — a FinBERT score from recent headlines, with its value proven by an ablation study.
- **Web app** — a single-stock investigator that combines all of the above with an honest comparison against a low-cost S&P 500 index fund.
- **Faithfulness audit** — every explanation sentence is checked against the underlying indicator values across ~12,000 historical recommendations, so the bot can't show a reason it didn't actually use.

Command-line tools, all options, and the project layout → [docs/USAGE.md](docs/USAGE.md).

## Requirements

- Python 3.10+, Node.js
- Internet connection (live prices via `yfinance`)

## Disclaimer

Educational / decision-support only. Not financial advice — do not use as the sole basis for trading decisions.
