---
name: refactoring
description: Extract reusable, domain-agnostic functions when refactoring duplicated logic. Apply PEP 8, DRY, and single-responsibility practices. Use when refactoring code, extracting helpers, deduplicating notebooks or app logic, or creating shared modules under src/.
---

# Refactoring

## Principles

1. **Domain-agnostic helpers** — Shared functions should not encode dataset, model, or product names. Pass domain specifics as parameters (column names, titles, category orders, label maps). Keep domain logic in the module that owns the data; keep reusable mechanics in neutral packages (e.g. `src/sentiment_ltr/viz/`).
2. **DRY** — If the same transform or chart pattern appears in two places, extract it once and call it from both. Prefer calling an existing domain helper (e.g. `phrasebank_probability_chart_frame()`) over re-implementing its steps in a notebook or `app.py`.
3. **PEP 8** — Follow standard Python style: `snake_case` functions, keyword-only optional args after `*`, type hints on public APIs, module docstrings, one responsibility per function, imports grouped (stdlib → third party → local).
4. **Thin call sites** — Notebooks and UI layers wire data + labels + titles; they should not contain melt/groupby/plot boilerplate that already lives in a helper.
5. **Minimal scope** — Extract only what is repeated or clearly reusable. Do not over-abstract one-off code.

## Extraction checklist

Before finishing a refactor:

```
- [ ] Shared function names and parameters are domain-neutral (no "phrasebank", "ravenpack", ticker names in viz/utils)
- [ ] Domain constants and business rules stay in domain modules; callers pass them in
- [ ] No copy-pasted blocks left in app.py, notebooks, or sibling modules
- [ ] Public helpers exported from package `__init__.py` when intended for reuse
- [ ] ReadLints clean; syntax/import smoke test on changed Python
- [ ] Call sites updated to import and use the new helpers (re-run affected notebook cells or app tab)
```

## Layering (this repo)

| Layer | Responsibility | Example |
| --- | --- | --- |
| **Generic** (`src/sentiment_ltr/viz/`, small utils) | Data shape + presentation mechanics | `melt_wide_metrics`, `split_series_distribution_figures` |
| **Domain** (`models/`, `data/`) | Load data, run models, build domain-specific frames | `phrasebank_probability_chart_frame` |
| **UI / notebook** | Titles, layout, caching, user-facing copy | Streamlit tab, Jupyter display |

## Anti-patterns

- Hard-coding `"negative"`, `"train"`, or dataset paths inside a generic viz function.
- Re-melting probability columns in a notebook when `phrasebank_probability_chart_frame()` already exists.
- Three nearly identical Plotly blocks differing only in column names — extract one builder with parameters.
- Generic `utils.py` dumping ground with mixed concerns; prefer focused subpackages (`viz/`, `data/`).

## Reference

Plotting-specific hover and Plotly defaults: [plotting-preferences](../plotting-preferences/SKILL.md).

Project example: `src/sentiment_ltr/viz/plotly_charts.py` (generic) + `phrasebank_sentiment.phrasebank_probability_chart_frame()` (domain).
