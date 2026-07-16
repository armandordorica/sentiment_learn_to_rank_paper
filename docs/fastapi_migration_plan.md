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
| 2 | **Batch Pipeline (Top-1K)** | ✅ | 2A Runner controls & live progress · 2B Cached data snapshot · 2C Failure reasons by provider · 2D Delisting reasons (CRSP) · 2E Cash-merger exits | Ported at `/batch` (`webapp/api/batch_pipeline.py`): same `scripts/run_batch_pipeline.py` subprocess (pid/status/log files), HTMX 5s status polling, manifests table with emoji provider/delisting/exit cells, delisting + cash-merger WRDS fetch, CSV export, combined parquets. Verified side by side against the running Streamlit tab — all metrics identical (see "Tab 2 port — what was built & verified" below). Caveat: launch/stop builds the identical runner CLI but a live batch hasn't been started from the new UI yet. |
| 3 | **PhraseBank HF Baseline** | 🟨 | 3A Model & training ✅ · 3B Reproduction recipe · 3C Performance metrics ✅ · 3D Dataset dashboard ✅ · 3F W&B experiment tracking | Ported at `/phrasebank` (`webapp/api/phrasebank_baseline.py`): dataset dashboard, training metrics, on-demand train/val/test split evaluation via `evaluate_checkpoint_on_split()`, probability charts — same Plotly figures as Streamlit. 3B and 3F not yet ported. |
| 4 | **RavenPack Baseline Eval** | ✅ | 4C Class-level metrics · 4D Label distribution shift · 4E Run evaluation | Ported at `/raven-eval` (`webapp/api/ravenpack_eval.py`): same `evaluate_phrasebank_baseline_on_ravenpack()` scoring, 4C/4D lazy-load via HTMX (first run scores the split, memoized after), provenance snapshot, on-demand 4E eval with split/row-cap controls. Covered by `tests/test_ravenpack_eval.py` (15 tests) and verified in-browser on real AAPL data — see "Tab 4 port" below. |
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
| Background job polling pattern (batch pipeline tab) decided | ✅ | Two patterns in use: in-process `webapp/jobs.py` `JobManager` (thread + polling) for training jobs (5·8, HTMX polls every 2s), and detached-subprocess + file polling for Tab 2's batch runner (`batch.pid` / `batch_status.json` written by `scripts/run_batch_pipeline.py`, HTMX polls `/batch/status` every 5s — survives webapp restarts). |
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


## Tab 2 port — what was built & verified

Ported 2026-07-15 (commits `fd98260`, `2000f27`).

- `webapp/api/batch_pipeline.py` — adapter porting the Tab 2 helpers from
  `app.py`. Key design choice: the batch runner stays the **same detached
  subprocess** (`scripts/run_batch_pipeline.py`), which writes its own
  `batch.pid` / `batch_status.json` / `batch_runner.log` under
  `data/raw/data_explorer_top1k/`. Both UIs (and the runner itself) share those
  files, so either app can monitor or stop a batch the other started, and a
  run survives webapp restarts. Manifest loading is cached on a
  `count:mtime` token (same invalidation rule as the Streamlit
  `_manifest_cache_token()`), and all emoji cell formatters
  (status/provider/delisting/exit) were ported verbatim.
- Routes in `webapp/main.py`: `GET /batch` (page), `GET /batch/status`
  (HTMX partial, self-polls every 5s while a run is live), `POST
  /batch/launch`, `POST /batch/stop`, `POST /batch/tickers` (filtered
  per-ticker table), `GET /batch/ticker` (detail panel), `GET /batch/log`,
  `POST /batch/delisting/fetch`, `POST /batch/cash-merger/fetch`,
  `GET /batch/cash-merger.csv` (download), `POST /batch/combined`.
- Templates: `batch_pipeline.html` + partials `batch_status`,
  `batch_ticker_table`, `batch_ticker_detail`, `batch_delisting`,
  `batch_cash_merger`, `batch_log`, `batch_combined_result`.
- **Verified (two passes):**
  1. Output parity against an independent count of the raw
     `by_ticker/rank_*/manifest.json` files: 437 complete / 563 partial;
     provider ok-counts WRDS 1,000 · Yahoo 517 · RavenPack 598 ·
     Refinitiv 483.
  2. Side-by-side against the running Streamlit tab (port 8501): identical
     last-run banner/timestamp, 2B snapshot metrics, delisting metrics
     (561 delisted / 439 active / mean dlret −0.115), cash-merger counts
     (195/195 resolved via CRSP dlret), and "Showing 1,000 of 1,000".
     Also exercised: status/ticker/provider-failure filters, ticker detail
     panel, full-log lazy-load, CSV export — zero browser console errors.
- **Fixed during verification:** no-match filters now say "showing 0 of N"
  (was a misleading "No ticker data cached yet"), and clearing the
  ticker-detail select empties the panel instead of showing "not found".
- **Not yet verified:** launching/stopping a *live* batch run from the new
  UI (identical CLI construction to `app.py`'s `_launch_batch`, but a real
  run takes hours over 1,000 tickers) — exercise on the next intentional
  batch run before fully trusting 2A's launch path.

## Tab 4 port — what was built & verified

Ported 2026-07-15.

- `webapp/api/ravenpack_eval.py` — adapter. The confusion-matrix → metrics
  transforms (`class_metrics_from_confusion`, `summary_from_class_metrics`,
  `label_prevalence_from_confusion`, `prevalence_gap_table`) are verbatim
  ports of the app.py helpers kept as **pure functions** so they're
  unit-testable without loading DistilBERT. Expensive computations
  (`run_eval`, `phrasebank_class_metrics`) are memoized on a checkpoint
  mtime token — the FastAPI equivalent of app.py's
  `@st.cache_data(ttl=3600)` wrappers.
- Routes: `GET /raven-eval` (page — fast, no inference), `GET
  /raven-eval/dataset` (ticker-change partial), `GET /raven-eval/static`
  (4D shift charts + 4C class dashboard; lazy-loaded via HTMX on page load
  because the first call scores the whole split), `POST /raven-eval/run`
  (4E results). Default ticker is AAPL (only rich RavenPack export), not
  the alphabetical first, deliberately improving on Streamlit's index-0
  default which errors on ticker "A".
- Templates: `ravenpack_eval.html` + partials `raven_eval_dataset`,
  `raven_eval_static`, `raven_eval_provenance`, `raven_eval_results`.
- **Tests** — `tests/test_ravenpack_eval.py` (15 passed, ~2s), first test
  suite in the repo (`pytest` + `httpx` added to requirements-webapp.txt):
  hand-computed precision/recall/F1 on a synthetic confusion matrix,
  cross-checked against scikit-learn; prevalence/gap math;
  presentation-context builders with the model layer monkeypatched; route
  wiring + template rendering + error paths via FastAPI `TestClient`.
  Run with `python -m pytest` from the project root.
- **Verified in-browser on real data:** AAPL dataset summary (68,722
  labeled / 37,832 train / 18,154 test — matches the fine-tune tab), 4C
  summary (PhraseBank train 94.7% / validation 83.5% / test 82.1% macro-F1
  vs RavenPack test 27.5% out-of-domain), 4E full test-split eval (18,154
  rows, accuracy 27.3%, macro-F1 27.5%, confusion matrices consistent with
  the prevalence charts), provenance section, graceful error for tickers
  without a rich export (e.g. MSFT) — zero console errors. Numbers come
  from the same `evaluate_phrasebank_baseline_on_ravenpack()` call as
  Streamlit, which is deterministic given the same checkpoint (row caps
  use a fixed sampling seed).
- **Known caveat (inherited from Streamlit):** only AAPL currently has a
  RavenPack export with a `headline` column; other tickers error with
  `'str' object has no attribute 'map'` in `load_ravenpack_labeled_frame`.
  Both UIs surface this as an error box. Fixing the message (or filtering
  the ticker list to rich exports) would improve both.

## Full-app interactive pass — 2026-07-15

After the Tab 4 port, a button-by-button pass over all five ported tabs
found **no defects** (nothing to fix, no code changes):

- **Tab 4:** dataset expander (split counts cross-check against
  confusion-matrix row totals), section anchors, three real 4E evals
  (test full split · validation capped 500 · "all" capped 500, exercising
  the `eval_split=None` path), report + mismatch expanders.
- **Tab 3:** train-split eval button → 94.7% macro-F1, identical to
  Tab 4's 4C row for the same split (shared checkpoint, shared numbers).
- **Tab 1:** quick-ticker + paper-window buttons, cached MSFT load
  (4 providers ok, 4 charts; RavenPack rows 499,136 = Tab 2's cell).
- **Tab 5·8:** ticker multi-select coverage refresh.
- **Tab 2:** status banner, fetch buttons correctly disabled at 0
  missing, CSV link, lazy log, table filters.
- **Cross-cutting:** all 9 GET routes 200, homepage/nav links correct
  (tabs 6/7 disabled), zero console errors, 15/15 tests green.

Still never exercised (long-running/heavy): batch launch, fine-tune
training job, combined-parquets write, Data Explorer live pulls.

## Working agreement

1. Update this file's status table whenever a tab/section moves state.
2. When a tab is marked ✅, it must be verified against the Streamlit version
   for output parity on real data before moving on.
3. Don't remove or modify `app.py` functionality while this migration is in
   progress — it's the fallback until FastAPI reaches parity.
4. Prefer small, mergeable increments (one section at a time within a tab)
   over big-bang tab rewrites.
