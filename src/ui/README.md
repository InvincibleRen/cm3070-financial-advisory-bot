# `ui/` - web-UI logic layer

The **UI-framework-free** logic layer behind the web app. It reuses the existing
Python modules (rule-based advisor, cross-sectional selector, FinBERT sentiment,
direction forecaster) and exposes them as plain, provider-injectable functions.
The React front end (`frontend/`) talks to these through the FastAPI back end
(`src/api/main.py`); nothing here imports a UI framework.

| File | Responsibility | Status |
|---|---|---|
| `data_access.py` | **UI-framework-free** logic layer: provider-injectable functions (single-stock analysis, ticker evaluation, sentiment scoring, universe selection). Unit-tested offline. | done |
| `evidence.py` | Turns the survivorship-free study results (`data/evidence/evidence.json`) into the plain sentences shown beside each signal and pick. | done |
| `profile.py` | Maps the three-question risk/horizon profile to view settings (top-N, index-first). | done |

## Consumers

- **FastAPI** (`src/api/main.py`) - serialises the dataclass results into JSON.
- **React SPA** (`frontend/`) - renders the single-stock investigator (technical
  analysis, the ML read beside its track record, news sentiment, the verdict, the
  risk factor, and the stock against a low-cost index fund).
- **Tests** (`tests/test_ui.py`) - exercise `data_access.py` headlessly with a fake
  provider (no network, no web server, no model download).

## Design note

**Logic vs presentation split (Rule 1).** All application logic lives in
`data_access.py`, which imports no UI framework and takes an injectable data
provider - so every number shown comes from a unit-tested function, and the
presentation layer (React) contains only rendering.
