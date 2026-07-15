# Streamlit → FastAPI Migration Plan

## Why

`app.py` has grown to ~6,800 lines / 137 functions / 780 `st.*` calls. We keep
hitting Streamlit architectural limits (HTML sanitization stripping
`onclick`, full-script reruns on every widget interaction, awkward anchor
scrolling, limited layout/JS control). Given this app will be used daily for
research for at least another year, it's worth migrating to a stack that
gives full control over routing, DOM, and interactivity: **FastAPI + Jinja2 +
HTMX + a JS charting lib (Plotly.js/ECharts)**.

## Strategy: strangler-fig, not a rewrite

- `app.py` (Streamlit) stays **untouched and fully working** for the entire
  migration. Never break it.
- New `webapp/` FastAPI app is built **side by side**, reusing all business
  logic from `src/sentiment_ltr/` (no logic duplication — only the
  presentation layer is reimplemented).
- Tabs are migrated **one at a time**, easiest/most self-contained first.
- Only after every tab below is ✅ **and** the new app has been used for real
  research work for a while do we deprecate `app.py`.
- This plan doc is the single source of truth for migration status — update
  the checkboxes/status column as work lands.

## Target architecture

```
app.py                        # existing Streamlit app — untouched
webapp/
  main.py                     # FastAPI app, routing, startup
  api/                        # JSON endpoints wrapping src/sentiment_ltr/ logic
    __init__.py
    data_explorer.py
    batch_pipeline.py
    phrasebank_baseline.py
    ravenpack_eval.py
    ravenpack_finetune.py
    sentiment_lab.py
    paper_validation.py
  templates/                  # Jinja2 templates, server-rendered + HTMX
    base.html
    nav.html
    tabs/
      data_explorer.html
      batch_pipeline.html
      ...
  static/
    css/
    js/
src/sentiment_ltr/             # unchanged — both UIs call into this
```

## Migration status

Legend: ⬜ Not started · 🟨 In progress · ✅ Done · 🚫 Deferred/won't port

| # | Tab | Status | Sections | Notes |
|---|-----|--------|----------|-------|
| 1 | **Data Explorer** | ✅ | 1A API status & ticker form · 1B Overview pane · 1C Prices pane · 1D News pane · 1E Sentiment pane · 1F Raw data pane | Ported at `/data-explorer`: cache-first loading, optional live refresh, provider status, Plotly price/news/sentiment charts, and raw tables. |
| 2 | **Batch Pipeline (Top-1K)** | ⬜ | 2A Runner controls & live progress · 2B Cached data snapshot · 2C Failure reasons by provider · 2D Delisting reasons (CRSP) · 2E Cash-merger exits | Background job control + polling; needs a clean way to stream/poll progress (HTMX polling or SSE/WebSocket). |
| 3 | **PhraseBank HF Baseline** | 🟨 | 3A Model & training ✅ · 3B Reproduction recipe · 3C Performance metrics ✅ · 3D Dataset dashboard ✅ · 3F W&B experiment tracking | Ported at `/phrasebank` (`webapp/api/phrasebank_baseline.py`): dataset dashboard, training metrics, on-demand train/val/test split evaluation via `evaluate_checkpoint_on_split()`, probability charts — same Plotly figures as Streamlit. 3B and 3F not yet ported. |
| 4 | **RavenPack Baseline Eval** | ⬜ | 4C Class-level metrics · 4D Label distribution shift · 4E Run evaluation | Zero-shot eval of PhraseBank checkpoint on RavenPack headlines. |
| 5 | **RavenPack Fine-Tuning ⭐** | 🟨 | 5·1 Train/val/test split · 5·2 Tokenization & padding · 5·3 Macro-F1 before/after · 5·4 Per-class F1 · 5·5 Label prevalence · 5·6 Sample headlines · 5·7 Hyperparameters & provenance · **5·8 Train (1/5/N tickers) ✅ ported** | Main experiment tab — most complex. **5·8 is the first ported section** (`webapp/templates/finetune.html` + `webapp/api/ravenpack_finetune.py`): ticker multi-select, HTMX-updated coverage table, background training job via `webapp/jobs.py`, live-polled status. Sections 5·1–5·7 (charts/tables) not yet ported. |
| 6 | **Sentiment Lab** | ⬜ | 6A News data coverage · 6B Compute device · 6C Financial PhraseBank dataset · 6D RavenPack articles browser · 6E Live inference (score a headline) | Interactive version of `liquidAI_prep.ipynb`; live-inference form is a good HTMX exercise. |
| 7 | **Paper Validation (2003–2014)** | ⬜ | 7A Universe summary · 7B Top 20 by volume · 7C Monthly volume over time · 7D Monthly prices | Sanity-check charts/tables over the CRSP candidate universe. Simplest, mostly static — **recommended first proof-of-concept**. |

### Cross-cutting infra (do once, early)

| Item | Status | Notes |
|------|--------|-------|
| FastAPI project scaffold (`webapp/`, deps, run script) | ✅ | `webapp/main.py`, `webapp/api/`, `webapp/templates/`, `webapp/static/`. Run with `conda run -n sentiment-ltr-paper uvicorn webapp.main:app --reload --port 8001`. |
| Shared base layout + top nav (mirrors the 7-tab structure) | ✅ | `webapp/templates/base.html` — disabled/greyed-out nav items for unmigrated tabs. |
| In-tab anchor/section navigation (real `<a href="#...">` + native scroll, no sanitizer fighting) | ✅ | Implemented for Data Explorer sections 1A–1F. |
| Data access layer reused from `src/sentiment_ltr/` (no duplication) | ✅ | `webapp/api/ravenpack_finetune.py` wraps `sentiment_ltr.models.ravenpack_sentiment` directly — same `train_ravenpack()`, `load_ravenpack_labeled_frame()`, etc. as `app.py`. |
| Charting approach decided (Plotly.js vs ECharts vs server-rendered images) | ✅ | Existing Plotly figures are embedded as HTML, matching the approach used by PhraseBank. |
| Background job polling pattern (batch pipeline tab) decided | ✅ (for training jobs) | In-process `webapp/jobs.py` `JobManager` (thread + polling), used by 5·8's fine-tune button. HTMX polls `/finetune/train/{job_id}/status` every 2s. Same pattern should work for Tab 2's batch pipeline runner. |
| Dev task/run config (`tasks.json` or `uvicorn --reload`) | 🟨 | Documented run command above; no VS Code task added yet. |
| Auth/session parity if `app.py` has any (check) | ⬜ | `app.py` has no auth — likely N/A, not yet explicitly verified. |

## Section 5·8 proof-of-concept — what was built

- `webapp/api/ravenpack_finetune.py` — thin wrapper: `available_tickers()`,
  `pilot_default_tickers()`, `deps_status()`, `coverage_summary()`,
  `run_training()`. All delegate to `sentiment_ltr.models.ravenpack_sentiment`
  (no logic duplication).
- `webapp/jobs.py` — minimal in-memory `JobManager` running `train_ravenpack()`
  in a background thread (HF `Trainer.train()` is blocking); exposes
  pending/running/done/error status for polling.
- `webapp/main.py` — routes: `GET /finetune` (page), `POST /finetune/coverage`
  (HTMX partial — updates the coverage table when the ticker selection
  changes), `POST /finetune/train` (kicks off a background job),
  `GET /finetune/train/{job_id}/status` (polled every 2s while running).
- `webapp/templates/finetune.html` + `partials/coverage_table.html` +
  `partials/train_status.html` — Jinja2 + HTMX, ported from the Streamlit
  `render_ravenpack_finetuning_tab()` section 5·8 (ticker multi-select →
  coverage table → train button → status).
- **Known caveat carried over from Streamlit:** only **AAPL** currently has a
  "rich" RavenPack export with a `headline` column
  (`data/raw/news/ravenpack/`); the `data_explorer_top1k` batch-pipeline
  exports for other tickers lack `headline` and raise in
  `load_ravenpack_labeled_frame`. `pilot_default_tickers()` defaults to AAPL
  only and `coverage_summary()` now catches and surfaces this as a friendly
  error instead of a 500, matching the Streamlit tab's own `TICKER = "AAPL"
  # only ticker with a rich RavenPack export` comment.
- **Verified:** app boots (`GET /` and `GET /finetune` return 200), AAPL
  coverage table renders real row counts (68,722 labeled / 37,832 train /
  18,154 test). **Not yet verified:** a full end-to-end training run
  completing successfully through the UI (kicked off but not confirmed to
  finish) — do this before marking 5·8 ✅.


## Working agreement

1. Update this file's status table whenever a tab/section moves state.
2. When a tab is marked ✅, it must be verified against the Streamlit version
   for output parity on real data before moving on.
3. Don't remove or modify `app.py` functionality while this migration is in
   progress — it's the fallback until FastAPI reaches parity.
4. Prefer small, mergeable increments (one section at a time within a tab)
   over big-bang tab rewrites.
