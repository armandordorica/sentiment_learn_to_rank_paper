---
title: Sentiment Learn To Rank Paper
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.51.0
app_file: app.py
pinned: false
---

# Reproducing "Stock Portfolio Selection Using Learning-to-Rank Algorithms With News Sentiment"

This repository is intended to recreate the experiments from:

> Song, Q., Liu, A., and Yang, S. Y. (2017). Stock portfolio selection using learning-to-rank algorithms with news sentiment. *Neurocomputing*, 264, 20-28. https://doi.org/10.1016/j.neucom.2017.02.097

The paper builds weekly stock rankings from financial news sentiment and recent market information, then uses learning-to-rank models to select long-only and long-short portfolios. The original study uses Bloomberg market data and Thomson Reuters News Analytics (TRNA) sentiment data from January 2003 through December 2014.

## Reproduction Status

The paper depends on proprietary data:

- Bloomberg: stock prices, trading volume, GICS sectors, SPY/S&P 500 benchmark, and HFRIEMN index.
- Thomson Reuters News Analytics: per-article company-level sentiment probabilities and relevance scores.

To reproduce the exact paper, you need access to those datasets or archived equivalents. If those are unavailable, the same pipeline can be recreated with substitutes such as CRSP/Compustat or Yahoo Finance for market data and RavenPack, Refinitiv, GDELT, Bloomberg News Analytics, or a custom news sentiment model for sentiment data. Results should then be treated as a methodological replication, not an exact numerical replication.

## Current State & Development Log

*Last updated: June 2026*

This section records what has been built so far, what is working end-to-end, and what remains before the paper's learning-to-rank backtest can run.

### Where We Are

| Phase | Status | Summary |
| --- | --- | --- |
| Data infrastructure | **In progress** | Multi-provider pulls, WRDS/CRSP universe, and per-ticker batch caching are in place. |
| Universe (market side) | **Done** | Top-1,000 CRSP candidates by 2003–2014 average volume are committed in `app_data/crsp_top_volume_universe.csv`. |
| Universe (news filter → 512) | **To do** | Paper's ≥1 article/week TRNA filter not yet applied at scale. |
| Feature engineering | **To do** | Weekly sentiment shock/trend, lagged returns, and labels not implemented. |
| RankNet / ListNet | **To do** | Models and rolling 2006–2014 backtest not implemented. |
| Results reproduction | **To do** | Table 3 / cumulative-return targets not yet attempted. |

The project is past the "can we pull data?" stage and into "can we pull it reliably for the full candidate universe, diagnose gaps, and cache it for modeling?"

### What Works Today

**Streamlit app (`app.py`)** — six tabs:

1. **Data Explorer** — single-ticker queries across Refinitiv, WRDS/CRSP, Yahoo, and RavenPack with combined price charts, news coverage, sentiment panes, and raw schema inspection.
2. **Batch Pipeline (Top-1K)** — launch, monitor, and inspect bulk fetches for all 1,000 universe tickers; live status banner, cache snapshot, per-provider failure breakdown, and per-ticker manifest table.
3. **PhraseBank HF Baseline** — Hugging Face DistilBERT benchmark on Financial PhraseBank: dataset dashboard (gold labels, box + p50 probability charts), training metrics, and checkpoint details.
4. **RavenPack Baseline Eval** — score the PhraseBank checkpoint on cached RavenPack headlines (out-of-domain): accuracy, macro-F1, confusion matrices, and mismatch samples.
5. **Sentiment Lab** — fine-tuning workflow for the news-sentiment model (TRNA substitute): RavenPack article browser, PhraseBank/RavenPack training, headline scoring, and compute-device report.
6. **Paper Validation (2003-2014)** — bundled universe CSV and validation charts without a live WRDS connection.

**Batch caching** — `scripts/run_batch_pipeline.py` writes one directory per ticker under:

```text
data/raw/data_explorer_top1k/by_ticker/rank_XXXX_TICKER/
├── manifest.json
├── provider_status.parquet
├── wrds_prices.parquet
├── yahoo_prices.parquet      (when ok)
├── ravenpack_articles.parquet (when ok)
└── refinitiv_prices.parquet   (when ok)
```

Runs are resumable: completed tickers are skipped; **smart partial retry** re-fetches only providers that failed while preserving already-ok parquet files.

**Identifier handling** — pulls are keyed on **CRSP PERMNO** from the universe file, not just the displayed ticker. Renamed symbols (e.g. FB→META) resolve via CRSP name history; RavenPack entity lookup uses PERMNO where available.

**Price alignment** — WRDS/CRSP prices use split adjustment (`cfacpr`) so CRSP and Yahoo series are comparable in the Data Explorer.

**Provider diagnostics** — `src/sentiment_ltr/data/provider_reason_codes.py` assigns stable `fail_reason` codes (e.g. `delisted_no_vendor_history`, `delisted_ric_retired`, `ticker_recycled_wrong_entity`) saved in manifests and rolled up in the Batch Pipeline UI.

### Known Data-Quality Patterns (Not Bugs)

Many top-volume names from 2003–2014 later **delisted, merged, or went private** (Lucent, Sun, Dell, Fannie Mae, etc.). The CRSP universe **includes** these securities (`last_trade_date` in the universe CSV) — this is correct and avoids survivorship bias at universe construction.

| Pattern | WRDS/CRSP | Yahoo | Refinitiv | RavenPack |
| --- | --- | --- | --- | --- |
| Delisted / merged | Usually **ok** through last trade | Often **fails** (no vendor history) | Often **RIC retired** | May have **no articles** after exit |
| Ticker recycled (e.g. LU) | Ok via PERMNO | Wrong entity risk | RIC confusion | Wrong-entity risk if keyed on current ticker |
| Still trading | Ok | Ok | Usually ok | Usually ok |

For replication, **WRDS/CRSP is the authoritative price source**; Yahoo and Refinitiv gaps on delisted names are expected. A ticker with WRDS ok + Yahoo/Refinitiv fail is often **`partial`**, not a data-collection failure. The paper used Bloomberg/TRNA vendor masters that largely hid these issues.

**Still to decide and document:** delisting-return handling in the backtest (`msedelist` / `dlret`), point-in-time vs static 512-stock universe, and how to chain RavenPack entities across mergers.

### Journal (what we did, decisions, and why)

This is the running journal of the project. Each entry records **what** changed,
the **decision** behind it, and **why** — so the rationale survives even after the
code does. Keep it updated as part of every commit (see
`.cursor/skills/validate-before-commit/SKILL.md`).

| Date | What / Decision / Why |
| --- | --- |
| 2026-06-19 | **CRSP top-1k universe** — built `notebooks/build_top1k_volume_universe.ipynb` + committed `app_data/crsp_top_volume_universe.csv`. *Why:* a static, version-controlled candidate set keeps the universe reproducible and avoids survivorship bias at construction. |
| 2026-06-21 | **Unified data pulls** — shared `live_data.py` for Streamlit and notebooks; multi-API dashboard. *Why:* one code path for fetching prevents app/notebook drift. |
| 2026-06-21 | **Batch pipeline v1** — `run_batch_pipeline.py` + Batch Pipeline tab; per-ticker manifests and parquets. *Why:* the full universe is too large to fetch interactively; caching per ticker makes runs resumable. |
| 2026-06-21 | **Live batch UX** — real-time `batch_status.json`, in-progress row in ticker table, incremental RavenPack year chunks. *Why:* long batch runs need visible progress and partial persistence. |
| 2026-06-22 | **Data Explorer fixes** — CRSP `cfacpr` adjustment, wider date inputs, PERMNO-based rename fallback (FB→META), RavenPack headline hover fix. *Why:* CRSP vs Yahoo must be comparable, and renamed tickers must resolve via PERMNO, not symbol. |
| 2026-06-22 | **Smart partial retry** — `partial` tickers re-fetch only failed providers; `load_cached_providers()` preserves ok data. *Why:* avoid re-paying for data already cached when only one provider failed. |
| 2026-06-23 | **Provider fail reasons** — machine codes + labels in manifests; cache snapshot and per-API failure tabs in Batch Pipeline UI. *Why:* separate expected delistings from real collection errors at a glance. |
| 2026-06-23 | **Manifest load performance** — fixed UI hang during active batch runs (cache token + session state; no WRDS lookups on page load). *Why:* the page must stay responsive while a batch writes manifests. |
| 2026-06-25 | **Cash-merger exit returns** — `src/sentiment_ltr/data/cash_merger_exits.py`, batch integration, and an Exit column / expander in the app for CRSP `dlstcd` 232/233. *Why:* cash mergers end a security's price history; modeling needs the realized exit return, not a silent gap. Dropped `exchcd` from the `crsp.dsedelist` query because that column isn't present there. |
| 2026-06-28 | **Fine-tuning prep** — `docs/news_sentiment_finetuning_plan.md` (kanban + progress log), `notebooks/liquidAI_prep.ipynb`, and split env files (`requirements-finetuning.txt`, `environment.yml`/`.lock.yml`). Baseline: 1-epoch DistilBERT on Financial PhraseBank (~80.6% test). *Why:* a custom sentiment model is the TRNA substitute. Used the `atrost/financial_phrasebank` Parquet mirror because the canonical script-based dataset breaks on `datasets` v5. |
| 2026-06-28 | **Sentiment Lab tab + module** — `src/sentiment_ltr/models/phrasebank_sentiment.py` (load/train/predict, `device_report`, `benchmark_matmul`) surfaced as a 4th app tab. *Why:* make the fine-tuning workflow runnable and demoable from the app, with a device report so GPU/MPS/CPU is explicit. Narrowed `.gitignore` from `models/` to `data/models/` so source under `src/sentiment_ltr/models/` is tracked while trained weights stay local. |
| 2026-06-28 | **Demo URL sharing** — `share.sh` Cloudflare quick tunnel (default port 8501) + README docs. *Why:* a one-command temporary public URL for sharing the app without deploying; URLs are ephemeral and regenerate each run. |
| 2026-06-28 | **README Journal + commit skill** — renamed Development Log to a Journal section (what / decision / why), backfilled recent milestones, updated tab list to four tabs, and required Journal updates in `validate-before-commit` skill. *Why:* decisions and rationale were scattered across chat/commits; a structured journal keeps the README as the single source of project history. |
| 2026-06-28 | **PhraseBank dataset explainer** — added an "ℹ️ What is Financial PhraseBank?" expander in the Sentiment Lab tab (what / who / why / schema sample / label construction) plus a full `docs/financial_phrasebank.md` reference, linked via "Read more" to the doc, HF dataset card, and the Malo et al. paper. *Why:* viewers should understand the training data's provenance and labeling without leaving the app, with deep detail one click away. |
| 2026-06-28 | **Cache-aware Data Explorer** — the Unified Ticker Data Explorer now has **Load data** (prefers the local batch cache, instant, no API login; falls back to live only when nothing is cached) and **Re-pull live (ignore cache)**; added `load_cached_dashboard_result()` that rebuilds the dashboard result shape from cached parquets with date filtering, plus a cache-status banner and a source (cache/live) indicator. *Why:* live pulls for high-volume names (e.g. Citigroup: ~865k RavenPack rows over 12 years) hang on sequential, no-timeout queries and require Refinitiv/WRDS auth; most tickers are already cached, so reading the cache is the fast, offline default. |
| 2026-06-28 | **Refinitiv story panel fix** — unified dashboard News tab on one story session state, removed duplicate story renderers (coverage drill-down vs full headline table), and replaced sticky `st.text_area` with read-only `st.write` display. *Why:* clicking a new headline kept showing the first loaded story (widget state) or a story from the other panel. |
| 2026-06-29 | **Iteration 2 fine-tuning** — macro-F1 + accuracy in `compute_metrics`, 3-epoch training with `load_best_model_at_end` on val F1 (`phrasebank_distilbert_best/`); test **F1 82.1%**, **acc 83.9%** vs 1-epoch baseline 80.6% acc. Documented atrost pre-defined splits (no re-split). *Why:* neutral class dominates PhraseBank; accuracy alone is misleading, and multi-epoch + best checkpoint beats the 1-epoch smoke test. |
| 2026-07-03 | **PhraseBank HF Baseline tab + dataset dashboard** — new top-level tab documenting the Hugging Face DistilBERT benchmark (dataset provenance, metrics, gold-label charts, box + p50 predicted-probability charts). *Why:* separate reference UI from interactive Sentiment Lab training/inference. |
| 2026-07-03 | **RavenPack fine-tuning path** — `ravenpack_sentiment.py`, Sentiment Lab RavenPack article browser + train UI, `finetune_on_ravenpack.ipynb`, news data coverage section, optional RavenPack `include_text` live pull. *Why:* adapt the PhraseBank checkpoint to RavenPack headline labels (TRNA substitute) for Iteration 4. |
| 2026-07-03 | **Reusable viz module + refactoring skill** — extracted domain-agnostic Plotly helpers to `src/sentiment_ltr/viz/` (`melt_wide_metrics`, `split_series_distribution_figures`, bar/box builders); PhraseBank HF Baseline tab and `finetune_on_ravenpack.ipynb` now call shared helpers; added `.cursor/skills/refactoring/SKILL.md` (DRY, PEP 8, domain-agnostic layering). *Decision:* generic chart mechanics in `viz/`, dataset framing in callers; notebook uses `%autoreload` and derives split order from chart data to avoid stale kernel imports. *Why:* one hover/layout implementation across app and notebooks; easier reuse for RavenPack probability charts later. |
| 2026-07-03 | **PhraseBank baseline reproduction recipe** — PhraseBank HF Baseline tab now documents training/tokenization hyperparameters plus GitHub code pointers (including inner `tokenizer(...)` calls for train vs inference); `phrasebank_baseline_recipe()` / `phrasebank_tokenization_code_pointers()` in `phrasebank_sentiment.py`; `finetune_on_ravenpack.ipynb` cell documents tokenizer contract. *Decision:* `inspect`-based GitHub blob links stay accurate as code moves; reload `phrasebank_sentiment` on app startup to avoid stale Streamlit imports. *Why:* readers can recreate the baseline without hunting through notebooks. |
| 2026-07-03 | **RavenPack config + label schema in Sentiment Lab** — `ravenpack_finetune_config_recipe()`, `ravenpack_label_schema_table()`, and GitHub code pointers in `ravenpack_sentiment.py`; Sentiment Lab expander documents parquet inputs/outputs, HF `DatasetDict` splits, and `id2label` alignment with PhraseBank; `phrasebank_checkpoint_label_maps()` for checkpoint verification; notebook cells for AAPL walkthrough and label-schema check. *Decision:* mirror the PhraseBank reproduction-recipe pattern so both fine-tune paths share the same verify-in-code UX. *Why:* RavenPack training reuses the PhraseBank head only if label ids match (0=negative, 1=neutral, 2=positive). |
| 2026-07-03 | **RavenPack Baseline Eval tab** — standalone top tab scores the PhraseBank checkpoint on RavenPack headlines (accuracy, macro-F1, Plotly heatmap, pandas `.style.bar` confusion matrices); `evaluate_phrasebank_baseline_on_ravenpack()` in `ravenpack_sentiment.py`; `importlib.reload` for `ravenpack_sentiment` on app startup; notebook inference cell for AAPL pred vs actual. *Decision:* keep out-of-domain eval out of Sentiment Lab so fine-tuning and zero-shot baseline are separate workflows. *Why:* ~27% test accuracy on AAPL before RavenPack adapt is a key baseline readers need to find quickly. |
| 2026-07-04 | **W&B experiment tracking for sentiment models** — installed/authenticated Weights & Biases in the `sentiment-ltr-paper` conda env, added `wandb>=0.28,<1`, and changed both Hugging Face `Trainer` paths (`train_baseline()` and `train_ravenpack()`) from `report_to="none"` to `report_to="wandb"` with descriptive run names encoding dataset/init, epochs, learning rate, batch size, and seed. *Decision:* keep local `metrics.json` for app reproducibility while using W&B for live run comparison and experiment history. *Why:* model performance and version lineage should survive beyond local checkpoint folders and notebook output. |
| 2026-07-04 | **Offline PhraseBank metrics backfilled to W&B + app links** — created `scripts/import_phrasebank_metrics_to_wandb.py`, imported the saved 3-epoch and 1-epoch PhraseBank snapshots into the `sentiment-ltr-transformers` W&B project, ignored local `wandb/` logs, and added W&B project/run buttons to the PhraseBank HF Baseline and Sentiment Lab tabs. *Decision:* historical runs are logged as `offline-metrics` imports rather than retrained, with `metrics.json` / `provenance.json` uploaded as lightweight W&B artifacts. *Why:* the dashboard now reflects both past baselines and future Trainer runs in one place, and the Streamlit app links readers directly to the experiment record. |
| 2026-07-04 | **Pushed W&B tracking work to GitHub** — committed and pushed `a6a5f0f` (`Add wandb tracking for sentiment models`) to `origin/main`, including W&B Trainer reporting, the metrics import script, dashboard links in `app.py`, `.gitignore` updates, and current notebook/app provenance changes. *Decision:* pushed directly to `main` after scanning the staged diff for private key/API-token patterns and confirming `.env`, `lseg-data.config.json`, `data/`, and local `wandb/` output remained ignored. *Why:* the public repo should capture today's experiment-tracking setup without leaking local credentials or generated model/data artifacts. |
| 2026-07-05 | **RavenPack baseline diagnostic charts** — extended `notebooks/finetune_on_ravenpack.ipynb` and the RavenPack Baseline Eval tab with static Plotly diagnostics for PhraseBank train/validation/test vs RavenPack out-of-domain: class-level F1, precision vs recall, observed-vs-predicted label prevalence, and prediction-prevalence gaps. *Decision:* compute the app dashboard automatically from the cached baseline evaluation and confusion matrix, so readers do not need to click **Run evaluation** to see the domain-shift diagnosis. *Why:* the baseline failure mode is not just lower macro-F1; it is visible in label prevalence shift and in whether precision or recall is driving each class-level F1 drop. |

### Immediate Next Steps

1. **Finish or validate the full 1k batch cache** — confirm WRDS + RavenPack coverage counts; use failure breakdown to separate expected delistings from real errors.
2. **Apply the paper's news filter** — average ≥1 article/week over 2003–2014 (TRNA in paper; RavenPack or Refinitiv substitute here) to move from 1,000 → ~512 candidates.
3. **Document corporate-events policy** — static vs point-in-time universe, delisting returns, when `partial` is acceptable for modeling.
4. **Build the weekly feature panel** — sentiment aggregation, shock/trend, lagged returns, quartile labels.
5. **Implement RankNet/ListNet and rolling backtest** — target paper Table 3 metrics.

## Target Results From The Paper

The full backtest period is 2006-2014. The paper reports these annualized results:

| Strategy | Return | Volatility | Sharpe | Max Drawdown |
| --- | ---: | ---: | ---: | ---: |
| ListNet long-only | 15.07% | 25.37% | 0.59 | 52.90% |
| RankNet long-only | 12.78% | 25.61% | 0.50 | 57.10% |
| ListNet long-short | 9.56% | 6.36% | 1.50 | 10.42% |
| RankNet long-short | 7.99% | 7.49% | 1.07 | 9.10% |
| Benchmark (SPY) | 7.25% | 21.27% | 0.34 | 55.19% |

Use these as sanity-check targets after implementing the pipeline.

## Environment Setup

Use the Conda environment to keep Python and package versions reproducible:

```bash
conda env create -f environment.yml
conda activate sentiment-ltr-paper
python -m ipykernel install --user --name sentiment-ltr-paper --display-name "Python (sentiment-ltr-paper)"
```

For the exact package versions solved on the original development machine, use:

```bash
conda env create -f environment.lock.yml
```

For a lighter pip-only setup, install the base requirements in your active environment:

```bash
pip install -r requirements.txt
```

The recommended environment uses Python 3.11 because the optional legacy Alpaca client may not resolve cleanly on Python 3.13.

### Optional extras (install only what you need)

The base environment runs the data pipeline and Streamlit app. Heavier or
vendor-specific stacks are split into separate requirements files so a fresh
machine only installs what it needs:

| Extra | File | For |
| --- | --- | --- |
| **Fine-tuning** | `requirements-finetuning.txt` | Hugging Face + PyTorch news-sentiment model (`notebooks/liquidAI_prep.ipynb`, the TRNA substitute). Included automatically by `environment.yml`. |
| Refinitiv/LSEG | `requirements-refinitiv.txt` | Refinitiv prices/news in the Data Explorer. |
| Alpaca (legacy) | `requirements-alpaca.txt` | Optional legacy market-data client. |

`conda env create -f environment.yml` already pulls in the fine-tuning stack. For a
pip-only setup, add it explicitly:

```bash
pip install -r requirements-finetuning.txt
```

This installs `torch`, `transformers`, `datasets`, `evaluate`, and `accelerate`.
Verified working on Python 3.11 with numpy 2.4 / pandas 2.2 / pyarrow 21 /
scikit-learn 1.9 (clean `pip check`): torch 2.12, transformers 5.12, datasets 5.0,
evaluate 0.4, accelerate 1.14. Runs on CUDA, Apple MPS, or CPU (auto-detected in the
notebook).

> **Dataset note:** the canonical `financial_phrasebank` ships as a loading script,
> which `datasets` v4/v5 no longer supports. Load a Parquet mirror instead (e.g.
> `atrost/financial_phrasebank`); `notebooks/liquidAI_prep.ipynb` already does this.

For WRDS/CRSP access, add your WRDS credentials to a local `.env` file:

```bash
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
```

Do not commit `.env` to the repository. Start with `wrds_connection.ipynb` to verify authentication and CRSP table access.

## Streamlit Web App

The repository includes a Streamlit app (`app.py`) for interactively exploring the data sources used in the replication. It is useful for smoke-testing API credentials, inspecting ticker-level data coverage, comparing price providers, and browsing news/sentiment records before building batch datasets.

Run it locally from the project root:

```bash
conda activate sentiment-ltr-paper
python -m streamlit run app.py
```

Then open the local URL printed by Streamlit, usually `http://localhost:8501`.

### Sharing a demo URL

To give someone (e.g. an advisor) temporary access to the locally-running app, use the
`share.sh` helper, which opens a [Cloudflare quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/)
(no Cloudflare account required). Start the app first, then in a second terminal:

```bash
./share.sh          # tunnels http://localhost:8501 (the Streamlit app)
./share.sh 7860     # tunnels a different port
```

The script prints a public `https://<random>.trycloudflare.com` URL. Keep that terminal
open and press `Ctrl-C` to stop sharing. Requires `cloudflared`
(`brew install cloudflared`).

Things to know:

- The URL is **temporary** and **changes every run** — share the fresh one each time.
- Your **laptop must stay awake** and the **webapp must keep running** for the URL to
  work; if either stops, the link goes dead.
- The URL is **public** — anyone who has it can access your app while the tunnel is up,
  so only share it deliberately and stop the tunnel when the demo is over.

#### How the public URL works (what `share.sh` actually does)

The app only listens on `localhost`, which is not reachable from the internet (or even
from your phone on the same Wi-Fi, in general). A **Cloudflare quick tunnel** bridges
that gap without any firewall changes, port forwarding, or DNS setup:

1. **Local server.** Streamlit serves the app on `http://localhost:8501` on your machine
   only.
2. **`cloudflared` makes an outbound connection.** `share.sh` runs
   `cloudflared tunnel --url http://localhost:8501`. The `cloudflared` client opens an
   *outbound* connection from your laptop to Cloudflare's edge network. Because the
   connection originates from your machine, no inbound ports need to be opened — this is
   why it works behind home routers, NAT, and most firewalls.
3. **Cloudflare assigns a public hostname.** Cloudflare hands back a random
   `https://<random>.trycloudflare.com` address and terminates HTTPS for you (the public
   link is encrypted even though your local app is plain HTTP).
4. **Traffic is proxied back down the tunnel.** When your advisor opens the URL,
   Cloudflare forwards each request through the established tunnel to `cloudflared` on
   your laptop, which hands it to `localhost:8501` and relays the response back. To the
   visitor it looks like a normal website; under the hood every request round-trips to
   your machine.

What `share.sh` adds on top of raw `cloudflared`:

- Defaults to port **8501** (this repo's Streamlit app); accepts any port as `$1`.
- Pre-flight check with `lsof` that **something is actually listening** on the port, so
  you don't publish a dead tunnel.
- Captures `cloudflared`'s log, **extracts the `*.trycloudflare.com` URL**, and prints it
  in a highlighted box so it's easy to copy.
- Installs a `trap` so pressing **Ctrl-C** cleanly shuts down `cloudflared` and tears the
  tunnel down.

Because the tunnel is just a proxy to your local process, the moment you stop `share.sh`
(or your laptop sleeps / the app stops), the public URL stops resolving. This is ideal
for a quick live demo and intentionally *not* a way to host the app permanently — for
that you would deploy to a server or a platform like Streamlit Community Cloud or
Hugging Face Spaces.

### What You Can Do

The app has three top-level tabs:

- **Data Explorer**: one ticker/date-range form that can query Refinitiv, WRDS/CRSP, Yahoo Finance, and RavenPack together.
- **Batch Pipeline (Top-1K)**: launch and monitor bulk fetches for the full CRSP top-volume universe; inspect cache coverage, per-provider failure reasons, and per-ticker manifests.
- **Paper Validation (2003-2014)**: bundled validation charts for the CRSP top-volume universe saved under `app_data/`.

In **Data Explorer**, enter a ticker such as `AAPL`, choose a start and end date, select the data sources to query, and click **Retrieve Dashboard Data**. Results are split into panes:

- **Overview**: combined price chart and a sentiment snapshot when RavenPack data is available.
- **Prices**: provider-specific price charts and tables for Refinitiv, WRDS/CRSP, and Yahoo Finance.
- **News**: Refinitiv daily headline counts, coverage metrics, and returned headline rows.
- **Sentiment**: RavenPack article-level sentiment scatter plus daily and weekly average sentiment charts.
- **Raw Data**: expandable raw data frames for debugging, export, or schema inspection.

The Plotly charts use point-level hover behavior so you can inspect the exact date, provider, article count, sentiment score, relevance, and headline metadata behind each point.

## FastAPI Web App (migration in progress)

The Streamlit app above is being ported, one tab at a time, to a FastAPI + Jinja2 +
HTMX stack under `webapp/`. Both apps currently run side by side against the same
underlying code in `src/sentiment_ltr/`, so numbers match exactly between the two UIs.
See `docs/fastapi_migration_plan.md` for the tab-by-tab migration status.

**Ported so far:**

- **Tab 1 — Data Explorer** → `/data-explorer` (cache-first ticker/date query,
  provider status, price/news/sentiment charts, and raw provider tables; optional
  live refresh across Refinitiv, WRDS/CRSP, Yahoo, and RavenPack).
- **Tab 2 — Batch Pipeline (Top-1K)** → `/batch` (runner controls with live
  progress polling, cached-data snapshot, failure reasons by provider, CRSP
  delisting reasons, cash-merger exits, filterable per-ticker status table).
- **Tab 3 — PhraseBank HF Baseline** → `/phrasebank` (dataset dashboard, training
  metrics, live train/val/test evaluation, probability charts).
- **Tab 5·8 — RavenPack Fine-Tuning** → `/finetune` (ticker multi-select, coverage
  table, background training job with live HTMX status polling).

Not yet ported: RavenPack Baseline Eval, Sentiment Lab, Paper Validation.

### Run it locally

Install the extra webapp dependencies (FastAPI, Jinja2, Uvicorn, python-multipart) on
top of the base/fine-tuning requirements:

```bash
conda activate sentiment-ltr-paper
pip install -r requirements-webapp.txt
```

Start the dev server with auto-reload from the project root:

```bash
uvicorn webapp.main:app --reload --port 8001
```

Then open <http://localhost:8001> (or <http://localhost:8001/phrasebank> /
`/finetune` directly). You can keep the Streamlit app running at the same time on its
own port (`8501`) to compare the two side by side.

### Expose a public link

The same `share.sh` Cloudflare quick-tunnel helper used for the Streamlit app works for
the FastAPI app — just point it at the FastAPI port instead of the default:

```bash
uvicorn webapp.main:app --reload --port 8001   # terminal 1
./share.sh 8001                                 # terminal 2
```

`share.sh` prints a temporary public `https://<random>.trycloudflare.com` URL that
proxies to your local FastAPI server. The same caveats as the Streamlit tunnel apply:
the URL is ephemeral, regenerates every run, requires your laptop and the `uvicorn`
process to stay up, and is publicly reachable while the tunnel is open — see
[Sharing a demo URL](#sharing-a-demo-url) above for how the tunnel mechanism works and
what `share.sh` adds on top of raw `cloudflared`.

### Data Sources And Requirements

| Source | Used For | Required Setup |
| --- | --- | --- |
| Refinitiv/LSEG | Daily prices, news coverage, headline rows, optional story text | Install `requirements-refinitiv.txt`; configure `lseg-data.config.json` for local Workspace or `LSEG_APP_KEY`, `LSEG_USERNAME`, and `LSEG_PASSWORD` for cloud/API usage. |
| WRDS/CRSP | CRSP daily prices and name history | Set `WRDS_USERNAME` and `WRDS_PASSWORD` in `.env` or Streamlit secrets. |
| RavenPack via WRDS | Article-level sentiment, relevance, event text, and taxonomy fields | Same WRDS credentials plus RavenPack table access on WRDS. |
| Yahoo Finance | Public price cross-checks | `yfinance` from `requirements.txt`; outbound internet access. |

WRDS connections are opened non-interactively in the app. If credentials are missing or invalid, the app shows a credential error instead of waiting for terminal input.

Yahoo Finance is best-effort. Some hosted or sandboxed networks block Yahoo's HTTPS requests; if that happens, uncheck Yahoo and use Refinitiv or WRDS/CRSP as the primary price source.

### Typical Workflows

1. **Check whether a ticker has usable price coverage**: select Refinitiv, WRDS/CRSP, and Yahoo; compare the combined price chart and provider tables in the **Prices** pane.
2. **Inspect Refinitiv news coverage**: select Refinitiv and enable news coverage; use the **News** pane to check daily counts, headline rows, and the average articles-per-week paper filter.
3. **Inspect RavenPack sentiment**: select RavenPack sentiment; use the **Sentiment** pane to review article-level sentiment scores, daily/weekly averages, event text, and classification fields.
4. **Debug provider schemas**: use **Raw Data** to inspect returned columns and row counts before writing a notebook or batch pipeline.
5. **Validate bundled paper-window artifacts**: open **Paper Validation (2003-2014)** to review the committed CRSP top-volume universe and top-20 validation plots without querying WRDS.
6. **Bulk-cache the full universe**: open **Batch Pipeline (Top-1K)**, configure providers and date range (2003-01-01 to 2014-12-31), launch the batch, and use the cache snapshot and failure-reason tabs to track progress across all 1,000 tickers.

### Batch Pipeline

Launch a background batch from the UI or CLI:

```bash
python scripts/run_batch_pipeline.py --start 2003-01-01 --end 2014-12-31 --rerun-partial
```

Useful flags:

| Flag | Purpose |
| --- | --- |
| `--rerun-partial` | Smart retry: re-fetch only failed providers on `partial` tickers (default in UI). |
| `--rerun-failed` | Retry tickers whose overall status is `failed` or `error`. |
| `--force-rerun` | Ignore cache and re-fetch everything. |
| `--no-refinitiv` / `--no-yahoo` | Skip optional price cross-check providers. |
| `--max-tickers N` | Smoke-test on the first N tickers from the start rank. |

Each completed ticker writes `manifest.json` with per-provider `status`, `rows`, `fail_reason`, and `fail_reason_label`. The UI reads these manifests to show universe-wide coverage without re-querying WRDS.

## Data Needed

### 1. Market Data

Collect daily data from January 2003 through December 2014 for a broad US equity universe:

- Ticker or Reuters Instrument Code (RIC)
- Adjusted close price
- Close price used for portfolio rebalancing
- Daily return
- Trading volume
- GICS sector
- Corporate action adjustments, if available

Also collect:

- SPY or S&P 500 benchmark returns
- HFRIEMN index returns, if comparing against hedge fund market neutral performance

This repository includes a market-data helper adapted from `armandordorica/Portfolio_Optimization_2023`. It tries Alpaca first when `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are configured, then falls back to Yahoo Finance.

```python
from sentiment_ltr.data import MarketDataClient, MarketDataConfig

config = MarketDataConfig(start="2003-01-01", end="2014-12-31")
client = MarketDataClient(config)

spy = client.fetch("SPY")
prices = client.fetch_many(["AAPL", "MSFT", "SPY"])
```

If you are not using the Conda environment, install the base dependencies with:

```bash
pip install -r requirements.txt
```

If you want to use Alpaca, configure `.env` from `.env.example` and install the optional client:

```bash
pip install -r requirements-alpaca.txt
```

Start with `notebooks/market_data_smoke_test.ipynb` to verify local Yahoo Finance access and timing. For WRDS/CRSP candidate-universe pulls, validation commands, and output schemas, see `docs/data_pull_validation.md`.

### 2. News Sentiment Data

The original paper uses TRNA fields:

- `datetime`: news timestamp
- `price`: RIC or company identifier for the stock mentioned
- `sentiment`: predominant sentiment, where positive = 1, neutral = 0, negative = -1
- `pos`: positive sentiment probability
- `obj`: neutral/objective sentiment probability
- `neg`: negative sentiment probability
- `relevance`: relevance of the article to the company, from 0 to 1

Compute article-level sentiment as:

```text
S_sentiment = relevance * (pos - neg)
```

Aggregate to weekly stock-level sentiment by averaging all article-level sentiment scores for each stock within each calendar week.

For a starter Refinitiv/LSEG Workspace API workflow, install the optional client and run `notebooks/refinitiv_news_smoke_test.ipynb` while Workspace is open and signed in:

```bash
pip install -r requirements-refinitiv.txt
cp lseg-data.config.example.json lseg-data.config.json
```

Generate an App Key in Workspace with the App Key Generator (`APPKEY`), paste it into the ignored local `lseg-data.config.json`, and keep Workspace running in the background. The helper loads that key and applies it with `get_config().set_param(...)` before opening the session. Test the desktop session with:

```bash
python scripts/test_refinitiv_connection.py
```

The notebook pulls sample headlines and story text, checks whether returned fields include TRNA-style sentiment/relevance columns, and exports raw samples under `data/raw/news/refinitiv/`.

## Stock Universe Construction

Follow the paper's two filters:

1. Select the top 1000 stocks by average trading volume.
2. Exclude stocks with fewer than one news article per week on average.

The paper obtains 512 stocks across the 10 GICS sectors. Treat this number as a target check, but expect differences if using a different data vendor, survivorship rules, or corporate action handling.

Important implementation choice: avoid look-ahead bias. Ideally, construct the universe using only information available at each point in time. The paper describes a fixed filtered universe, but a production-quality replication should document whether the universe is static or point-in-time.

### Market-Side Candidate Universe Artifact

**What it is:** `app_data/crsp_top_volume_universe.csv` — the top 1,000 US common stocks ranked by average daily share volume over the full paper window (2003-01-01 to 2014-12-31). This is the market-side candidate pool before the news-coverage filter is applied.

**How it was constructed:** `notebooks/build_top1k_volume_universe.ipynb` queries WRDS CRSP using a server-side SQL aggregation over `crsp.dsf` joined to `crsp.msenames`:

- Date range: 2003-01-01 to 2014-12-31
- Share codes: `shrcd IN (10, 11)` — ordinary common shares only (no ETFs, ADRs, REITs, preferred)
- Exchanges: `exchcd IN (1, 2, 3)` — NYSE (1), AMEX/ARCA (2), NASDAQ (3)
- Non-null daily volume observations only
- Ranked by `AVG(vol)` (average daily share volume) descending; top 1,000 taken
- Each PERMNO is joined to its most recent name record from `crsp.msenames` for ticker, company name, share code, and exchange code

The result is written to `data/raw/market/crsp_top_volume_universe.csv` (gitignored) and synced to the git-tracked copy at `app_data/crsp_top_volume_universe.csv`.

**Schema:**

| Column | Description |
| --- | --- |
| `volume_rank` | Integer rank 1–1000, 1 = highest average volume |
| `permno` | CRSP permanent security identifier |
| `permco` | CRSP permanent company identifier |
| `ticker` | Most recent ticker symbol during the window |
| `comnam` | Most recent company name during the window |
| `shrcd` | CRSP share code (10 or 11) |
| `exchcd` | CRSP exchange code (1 = NYSE, 2 = AMEX/ARCA, 3 = NASDAQ) |
| `trading_days` | Number of days with non-null volume in the window |
| `first_trade_date` | First trading day observed for the security |
| `last_trade_date` | Last trading day observed for the security |
| `avg_volume` | Average daily share volume over all eligible trading days |
| `avg_dollar_volume` | Average daily dollar volume (shares × price) |
| `avg_abs_price` | Average absolute daily closing price |
| `avg_shares_outstanding` | Average daily shares outstanding (thousands) |
| `latest_name_start` | Start date of the most recent name record |
| `latest_name_end` | End date of the most recent name record |
| `avg_volume_millions` | `avg_volume / 1,000,000` |
| `avg_dollar_volume_billions` | `avg_dollar_volume / 1,000,000,000` |

**How to load in another notebook:**

```python
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent

universe = pd.read_csv(
    PROJECT_ROOT / "app_data" / "crsp_top_volume_universe.csv",
    parse_dates=["first_trade_date", "last_trade_date"],
)
```

No WRDS connection is required — the file is committed to the repository. To regenerate it from WRDS, run `notebooks/build_top1k_volume_universe.ipynb` with valid `WRDS_USERNAME` and `WRDS_PASSWORD` credentials in `.env`.

## Weekly Feature Engineering

Each week is one learning-to-rank query. Each stock in the universe is an item in that query.

For each stock-week, compute six features:

1. Sentiment shock score
2. Sentiment trend score
3. Previous 1-week return
4. Previous 1-month return
5. Previous 1-week average sentiment
6. Previous 1-month average sentiment

The paper describes these as "leading" features, but the trading workflow implies they are historical inputs available before ranking the following week's returns. Implement them as lagged historical features to avoid look-ahead bias.

### Sentiment Shock

For stock `i` at week `t`, with sector-specific lookback window `N`:

```text
S_shock(t) = (S_sentiment(t) - mean(S_sentiment[t-N, t-1])) / std(S_sentiment[t-N, t-1])
```

If the rolling standard deviation is zero or unavailable, mark the feature missing and handle it consistently during model training.

### Sentiment Trend

For stock `i` at week `t`, with sector-specific lookback window `N`:

```text
S_trend(t) = sum(delta_S_sentiment(k), for k = t-N to t-1)
delta_S_sentiment(k) = S_sentiment(k) - S_sentiment(k-1)
```

### Sector-Specific Lookback Windows

The paper optimizes sentiment lookback windows using Spearman rank correlation over 2003-2006 training data. To match the paper, use:

| GICS Sector | Shock Window | Trend Window |
| --- | ---: | ---: |
| Consumer Discretionary | 15 | 14 |
| Information Technology | 11 | 30 |
| Consumer Staples | 18 | 19 |
| Materials | 15 | 16 |
| Industrials | 21 | 18 |
| Utilities | 16 | 28 |
| Health Care | 10 | 15 |
| Energy | 25 | 20 |
| Financials | 11 | 25 |
| Telecommunication Services | 19 | 24 |

If rebuilding the optimization, maximize Spearman rank correlation between each sentiment indicator and the following 1-week stock return within each sector.

## Labels

For each weekly query:

1. Compute each stock's following 1-week return.
2. Rank stocks cross-sectionally by that forward return.
3. Assign four relevance labels by return quartile:
   - Label 4: top 25% future returns
   - Label 3: second quartile
   - Label 2: third quartile
   - Label 1: bottom 25% future returns

For the long-short strategy, train a separate "short book" ranking target by reversing the labels so that the worst future returns receive the highest label. This lets NDCG emphasize accurate identification of the worst performers.

## Models

The paper trains two neural learning-to-rank algorithms:

- RankNet: pairwise ranking with cross-entropy loss on pairwise order probabilities.
- ListNet: listwise ranking with cross-entropy loss on ranked-list probabilities.

Model settings reported in the paper:

- Hidden layers: 1
- Hidden nodes: 10
- Learning rate: 0.00005
- RankNet training iterations: 150
- ListNet training iterations: 1500

The iteration counts are selected using NDCG on the first three backtest years, 2006-2009, with a 70/30 train/validation split.

Suggested modern Python libraries:

- `pandas`, `numpy`, and `pyarrow` for data processing
- `scipy` for Spearman correlation
- `scikit-learn` for preprocessing and metrics
- `pytorch` for RankNet/ListNet implementation
- `matplotlib` or `plotly` for result plots
- `pytest` for pipeline tests

## Rolling Backtest

The paper uses a three-year rolling training window and a one-year test window.

Example schedule:

| Training Window | Test Window |
| --- | --- |
| 2003-2005 | 2006 |
| 2004-2006 | 2007 |
| 2005-2007 | 2008 |
| ... | ... |
| 2011-2013 | 2014 |

For each test year:

1. Train RankNet and ListNet on the prior three years of weekly queries.
2. At the start of each calendar week, score all stocks using the latest available features.
3. Rank stocks by predicted score.
4. Rebalance at the close of the first trading day of the week.
5. Hold until the next weekly rebalance.
6. Compute daily portfolio returns from daily closing prices.

## Portfolio Rules

### Long-Only Strategy

For each weekly rebalance:

1. Select the top 25% of stocks by predicted rank score.
2. Equal-weight all selected stocks.
3. Hold until the next weekly rebalance.

The paper's universe has 512 stocks, so this selects 128 stocks.

### Long-Short Strategy

For each weekly rebalance:

1. Long the top 25% of stocks according to the long-book ranking model.
2. Short the top 25% of stocks according to the short-book ranking model, where labels have been reversed to identify likely underperformers.
3. Equal-weight the long book and equal-weight the short book.
4. Hold until the next weekly rebalance.

Document the leverage convention explicitly. A common market-neutral implementation is 100% long and 100% short, with daily portfolio return:

```text
portfolio_return = average(long_stock_returns) - average(short_stock_returns)
```

If using 50% long and 50% short, returns and volatility will scale differently from the paper.

## Performance Evaluation

Compute:

- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Cumulative return curve

The paper also splits the 2006-2014 backtest into volatility regimes using six-month realized market volatility. The high-volatility regime is October 2008 through May 2009, with a threshold of 36.93%, defined as two standard deviations above average realized market volatility.

## Suggested Repository Structure

```text
.
├── README.md
├── data/
│   ├── raw/
│   │   ├── market/
│   │   └── news/
│   ├── interim/
│   └── processed/
├── notebooks/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── backtest/
│   └── evaluation/
├── tests/
└── reports/
    └── figures/
```

Recommended first implementation milestones:

1. Build a reproducible data loader for market and news sentiment data.
2. Aggregate article sentiment to weekly stock sentiment.
3. Recreate the 512-stock universe or document the replicated universe.
4. Generate weekly features and labels.
5. Implement RankNet and ListNet training.
6. Implement the rolling yearly backtest.
7. Reproduce Table 3 and the cumulative return figure.

## Project Todo List

Use these statuses while building the replication:

- To Do: not started yet.
- In Progress: actively being worked on.
- Pending Review: implemented and awaiting verification, comparison, or cleanup.
- Blocked: cannot move forward without external data, access, or a methodological decision.
- Done: completed and checked.

### Project Setup And Reproducibility

| Task | Status | Notes |
| --- | --- | --- |
| Define data schema and file formats | Pending Review | Per-ticker manifest + parquet layout under `data/raw/data_explorer_top1k/`; see **Current State** above. |
| Maintain reproducible environment | Pending Review | Conda environment, lock file, and notebook kernel are in place; update when dependencies change. |
| Document replication limitations | In Progress | Journal and delisting/survivorship notes in README; corporate-events backtest policy still TBD. |

### Market Data And Universe

| Task | Status | Notes |
| --- | --- | --- |
| Choose primary market data source | Done | Use WRDS/CRSP as the primary market data source; keep Yahoo Finance for quick public smoke tests and fallback checks. |
| Verify WRDS connection and CRSP access | Done | `wrds_connection.ipynb` connected to WRDS, confirmed CRSP access, and returned a tiny CRSP sample query. |
| Define market data universe candidate list | Done | Added a WRDS/CRSP script that builds the top 1,000 common-stock candidates by average daily volume for 2003-2014. |
| Validate CRSP candidate universe | Pending Review | `notebooks/crsp_universe_validation.ipynb` checks row counts and filters, then displays the top 20 stocks by average daily volume. |
| Build raw market data loader | In Progress | `live_data.py` + batch pipeline pull WRDS/CRSP daily prices; Yahoo/Refinitiv as optional cross-checks. |
| Pull daily OHLCV data for 2003-2014 | In Progress | Batch pipeline caches per-ticker WRDS parquets for the full top-1k universe; delisting returns not yet pulled. |
| Pull benchmark market data | To Do | Download SPY or S&P 500 benchmark data for the full 2003-2014 window. |
| Pull or approximate GICS sectors | To Do | Needed for sector-specific sentiment lookback windows and stock-universe diagnostics. |
| Map identifiers across data sources | In Progress | Universe and batch pulls use PERMNO; RavenPack entity resolution uses PERMNO; RIC mapping via Refinitiv where available. |
| Validate market data coverage | In Progress | Batch Pipeline UI: cache snapshot, per-provider fail-reason rollups, and per-ticker manifest inspection. |
| Store raw market data locally | In Progress | Per-ticker cache under `data/raw/data_explorer_top1k/by_ticker/` (gitignored). |
| Create market data manifest | Done | Each ticker directory has `manifest.json` + `provider_status.parquet` with status, row counts, and fail reasons. |
| Build processed daily market panel | To Do | Produce a clean daily table ready for weekly aggregation and feature generation. |

### News Sentiment Data

| Task | Status | Notes |
| --- | --- | --- |
| Obtain company-level news sentiment data | Blocked | Exact replication requires TRNA; using RavenPack via WRDS as substitute in batch pipeline. |
| Build raw news sentiment loader | In Progress | RavenPack article pulls in `live_data.py` and batch pipeline; Refinitiv headlines available in Data Explorer. |
| Aggregate weekly stock sentiment | To Do | Compute `S_sentiment = relevance * (pos - neg)` and average by stock-week. |

### Feature Dataset

| Task | Status | Notes |
| --- | --- | --- |
| Construct final stock universe | To Do | Filter top 1000 stocks by average volume, then remove stocks with fewer than one news article per week. |
| Implement sentiment shock and trend features | To Do | Use the sector-specific lookback windows from the paper. |
| Generate lagged return and sentiment features | To Do | Add previous 1-week return, previous 1-month return, previous 1-week sentiment, and previous 1-month sentiment. |
| Generate weekly ranking labels | To Do | Rank following 1-week returns and assign quartile labels from 1 to 4. |

### Learning-To-Rank Models

| Task | Status | Notes |
| --- | --- | --- |
| Implement RankNet model | To Do | One hidden layer, 10 hidden nodes, learning rate `0.00005`, 150 iterations. |
| Implement ListNet model | To Do | One hidden layer, 10 hidden nodes, learning rate `0.00005`, 1500 iterations. |
| Implement NDCG validation workflow | To Do | Recreate the 2006-2009 70/30 train-validation selection process. |

### Backtesting And Evaluation

| Task | Status | Notes |
| --- | --- | --- |
| Implement rolling annual backtest | To Do | Train on three years and test on the following year from 2006 through 2014. |
| Implement long-only portfolio strategy | To Do | Equal-weight the top 25% of ranked stocks at weekly rebalances. |
| Implement long-short portfolio strategy | To Do | Long top 25% and short predicted bottom 25%, documenting leverage convention. |
| Compute performance metrics | To Do | Annualized return, volatility, Sharpe ratio, maximum drawdown, and cumulative return. |
| Reproduce paper result tables and figures | To Do | Compare against Table 3, Table 4, and cumulative return plots. |

### Quality And Bias Checks

| Task | Status | Notes |
| --- | --- | --- |
| Add tests for no look-ahead bias | To Do | Verify features use only information available before each rebalance. |
| Validate survivorship-bias handling | In Progress | Universe includes delisted names via CRSP `last_trade_date`; backtest exit policy and `msedelist` returns still TBD. |
| Validate portfolio accounting assumptions | To Do | Check leverage convention, transaction-cost assumptions, shorting assumptions, and benchmark alignment. |

## Reproducibility Checks

Before trusting results, verify:

- Weekly sentiment uses only news available before the rebalance.
- Forward returns are used only for labels, never as model features.
- Prices are adjusted for splits and dividends when calculating returns.
- Delisted stocks and corporate actions are handled consistently.
- Universe construction does not introduce survivorship bias, or the limitation is documented.
- Portfolio returns reflect the intended long-short leverage convention.
- Transaction costs and borrow costs are either excluded to match the paper or included and reported separately.

## Known Ambiguities In The Paper

- The exact constituent list of the 512-stock universe is not provided.
- Transaction costs, short borrow costs, and financing assumptions are not specified.
- The paper does not fully specify missing-data handling.
- It is unclear whether the stock universe is static or point-in-time.
- The phrase "leading return" appears in the feature list, but the backtest design requires historical lagged returns as inputs.

Document any choices made for these points so the replication can be audited and improved later.
