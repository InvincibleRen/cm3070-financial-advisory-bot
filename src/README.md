# `src/` - package layout (function-based, mapped to weightage)

The code is organised by *function*, matching the project's core-plus-extensions design.

| Package | Role | Weight | Status |
|---|---|---|---|
| `common/` | Shared infrastructure: data access, indicators, metrics, walk-forward engine, legacy rule-based backtester. Used by the core and every extension. | - | Built |
| `selection/` | **CORE - Direction 1.** Cross-sectional ML stock selection: universe → features → labels → ranker → top-N. | ~70% | Scaffold |
| `sentiment/` | **Extension A - Direction 3.** FinBERT news sentiment as an added feature (proved via ablation). | ~15% | Scaffold |
| `direction/` | **Extension B - Direction 2.** Per-stock time-series up/down forecast, run only on the selected top-N as a confirmation layer. | ~15% | Scaffold |
| `explanation/` | Plain-language explanation layer (template-based, shared UX). | - | Built (rule-based) |
| `ui/` | UI-framework-free logic layer for the web app (consumed by `src/api/` + the React `frontend/`). | - | done |
| `cli/` | Existing command-line entry points (to be superseded by the UI). | - | Built |

Each extension lives in its own package and is *additive and removable*: deleting `sentiment/` or `direction/` must not break `selection/`. Build order: `common` → `selection` → `sentiment` → `ui` → `direction`.
