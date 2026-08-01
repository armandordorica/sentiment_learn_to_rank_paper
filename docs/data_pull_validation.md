# Data Pull And Validation Guide

This guide documents how to pull and validate the current market-side candidate universe for the paper replication.

At this stage, the goal is not to apply the TRNA news-coverage filter. The goal is to build a sufficiently large CRSP candidate universe that approximates the paper's first universe step: selecting liquid stocks by trading volume.

## Paper Requirements vs. What We Have

Song et al. (2017) need TRNA-style per-article fields and weekly aggregates built from them. We use RavenPack + a custom DistilBERT classifier as substitutes. Status below is the inventory for replication — not an exact numerical match to TRNA.

**Webapps (local):**

| App | Start | Base URL |
| --- | --- | --- |
| Streamlit (`app.py`) | `python -m streamlit run app.py` | <http://localhost:8501> |
| FastAPI (`webapp/`) | `uvicorn webapp.main:app --reload --port 8001` | <http://localhost:8001> |

Streamlit tabs are numbered in the UI (`1 · Data Explorer`, …). FastAPI routes mirror those tabs once migrated.

### Universe & market data

| Paper need | Status | Local path(s) | Webapp |
| --- | --- | --- | --- |
| Top 1,000 stocks by average trading volume (2003–2014) | **Have** | Tracked: `app_data/crsp_top_volume_universe.csv`. Raw (gitignored): `data/raw/market/crsp_top_volume_universe.csv` + `…_manifest.json`. Build: `scripts/build_crsp_market_universe.py`, `notebooks/build_top1k_volume_universe.ipynb`. | Streamlit **7 · Paper Validation**; FastAPI [`/paper-validation`](http://localhost:8001/paper-validation) |
| Exclude stocks with &lt;1 news article/week → ~512 names | **Partial** | News side is largely cached: Batch Pipeline snapshot shows **1,000/1,000** manifests, **437 complete** / **563 partial**, RavenPack **598 ok** / 322 failed / 80 empty (~60% of universe). Helper for the paper threshold lives in `src/sentiment_ltr/data/news_coverage.py` (`avg_articles_per_week >= 1`). Still missing: run that filter over RavenPack-ok tickers and write the final ~512-name universe artifact. | Streamlit **2 · Batch Pipeline** (section **2B · Cached data snapshot**); FastAPI [`/batch`](http://localhost:8001/batch) |
| Daily OHLCV / returns per universe stock | **Partial** | Per ticker under `data/raw/data_explorer_top1k/by_ticker/rank_XXXX_TICKER/wrds_prices.parquet` (WRDS/CRSP; ~1,000 tickers). Optional cross-checks: `yahoo_prices.parquet`, `refinitiv_prices.parquet`. | Streamlit **1 · Data Explorer**, **2 · Batch Pipeline**; FastAPI [`/data-explorer`](http://localhost:8001/data-explorer), [`/batch`](http://localhost:8001/batch) |
| Top-20 monthly volume / price validation charts | **Have** | `app_data/top20_monthly_volume.csv`, `app_data/top20_monthly_prices.csv` (also under `data/processed/validation/`). | Streamlit **7 · Paper Validation**; FastAPI [`/paper-validation`](http://localhost:8001/paper-validation) |
| Benchmark (SPY / S&P 500) over 2003–2014 | **Missing** | — | — |
| GICS sector membership (lookback windows) | **Missing** | — | — |

### Per-article news sentiment (TRNA fields → our substitutes)

Paper formula: `S_sentiment = relevance × (pos − neg)`.

| Paper (TRNA) field | Status | What we have / path(s) | Webapp |
| --- | --- | --- | --- |
| `datetime` (article timestamp) | **Have** | RavenPack `article_time` / `timestamp_utc`. Batch: `data/raw/data_explorer_top1k/by_ticker/rank_XXXX_TICKER/ravenpack_articles.parquet` (~598 tickers with RavenPack). Richer AAPL-style exports: `data/raw/news/ravenpack/{ticker}_articles_2003_2014.parquet`. | Streamlit **1 · Data Explorer** (Sentiment pane), **6 · Sentiment Lab**; FastAPI [`/data-explorer`](http://localhost:8001/data-explorer), [`/sentiment-lab`](http://localhost:8001/sentiment-lab) |
| Company / stock identifier | **Have** | `ticker` on article rows; batch keyed by CRSP `permno` via `manifest.json` in each `rank_XXXX_TICKER/` dir. | Same as above; Batch Pipeline ticker table |
| `relevance` ∈ [0, 1] | **Have** | `relevance_score` (= RavenPack `relevance` / 100). Same parquet paths as above. | Data Explorer Sentiment pane shows relevance metrics/columns |
| Predominant `sentiment` ∈ {−1, 0, +1} | **Partial** | Derived from RavenPack `event_sentiment_score` thresholds for labeling; not TRNA’s native field. Fine-tune labels live in RavenPack training path (`src/sentiment_ltr/models/ravenpack_sentiment.py`). | Streamlit **5 · RavenPack Fine-Tuning**, **6 · Sentiment Lab**; FastAPI [`/finetune`](http://localhost:8001/finetune), [`/sentiment-lab`](http://localhost:8001/sentiment-lab) |
| `pos`, `obj`, `neg` (class probabilities) | **Partial — model only** | Checkpoints that emit `p(positive)`, `p(neutral)`, `p(negative)`: `data/models/phrasebank_distilbert_best/`, `data/models/ravenpack_distilbert_best/`, `data/models/ravenpack_distilbert_5stock/`. **Not** written back onto the full article corpus yet (Iteration 4.2). RavenPack vendor rows do **not** include these three probs. | Live / eval scoring: Streamlit **3 · PhraseBank HF Baseline**, **4 · RavenPack Baseline Eval**, **5 · RavenPack Fine-Tuning**, **6 · Sentiment Lab**; FastAPI [`/phrasebank`](http://localhost:8001/phrasebank), [`/raven-eval`](http://localhost:8001/raven-eval), [`/finetune`](http://localhost:8001/finetune), [`/sentiment-lab`](http://localhost:8001/sentiment-lab) |
| `S_sentiment = relevance × (pos − neg)` | **Proxy only** | Vendor proxy (Eq. 8 in fetch notebook): `sentiment_score = relevance_score × event_sentiment_score` on the RavenPack article parquets above. True TRNA-style `relevance × (p_pos − p_neg)` needs batch model scores. | Data Explorer / Sentiment Lab article tables |
| Headline / story text (for custom model) | **Partial** | RavenPack WRDS pulls **always include `headline`**, and by default also **`event_text`** when RavenPack has one (often null for tabular rows). Full wire bodies remain Refinitiv-only. Legacy batch parquets without titles are enriched from rich exports under `data/raw/news/ravenpack/` when present. | Sentiment Lab article browser; Data Explorer 1D/1E |



### Weekly aggregates & LTR features

| Paper need | Status | Local path(s) | Webapp |
| --- | --- | --- | --- |
| Weekly stock sentiment = mean of article `S_sentiment` | **Pilot only** | Example: `data/raw/news/ravenpack/aapl_weekly_sentiment_2003_2014.parquet` (vendor proxy scores, not model `pos`/`neg`). Universe-wide weekly panel **not** built. | Charts for single-ticker RavenPack averages in Data Explorer Sentiment pane |
| Sentiment shock & trend (sector lookbacks) | **Missing** | — | — |
| Lagged 1-week / 1-month return & sentiment features | **Missing** | — | — |
| Quartile relevance labels (1–4) from next-week returns | **Missing** | — | — |
| RankNet / ListNet + 2006–2014 rolling backtest | **Missing** | — | — |

### Quick read on the TRNA gap

| Piece | Done? |
| --- | --- |
| Relevance per article | Yes — RavenPack `relevance_score` |
| `P(pos)`, `P(neutral)`, `P(neg)` per article at scale | No — model can produce them; batch write to corpus (plan **4.2**) still open |
| Paper `S_sentiment` and weekly shock/trend features | No — only RavenPack proxy `sentiment_score` + AAPL weekly pilot |

## Prerequisites

Create and activate the project environment:

```bash
conda env create -f environment.yml
conda activate sentiment-ltr-paper
```

Configure WRDS credentials in a local `.env` file:

```bash
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
```

Do not commit `.env`. It is ignored by Git.

For the Hugging Face Streamlit app's live ticker lookup, add the same values as Space secrets named `WRDS_USERNAME` and `WRDS_PASSWORD`. The app does not print or store these credentials; it only uses them to open a WRDS connection when the lookup button is clicked.

Before pulling data, verify WRDS and CRSP access:

```bash
conda run -n sentiment-ltr-paper jupyter nbconvert \
  --to notebook \
  --execute wrds_connection.ipynb \
  --inplace \
  --ExecutePreprocessor.kernel_name=sentiment-ltr-paper
```

Expected checks:

- WRDS connection succeeds.
- `crsp` appears in the available libraries.
- A tiny `crsp.dsf` sample query returns rows.

## Pull The CRSP Candidate Universe

Run:

```bash
conda run -n sentiment-ltr-paper python scripts/build_crsp_market_universe.py
```

The script queries WRDS/CRSP using:

- Daily stock file: `crsp.dsf`
- Name/security history: `crsp.msenames`
- Date range: `2003-01-01` to `2014-12-31`
- Share-code filter: `shrcd in (10, 11)` for common stocks
- Exchange-code filter: `exchcd in (1, 2, 3)` for NYSE, AMEX, and Nasdaq
- Ranking rule: top 1,000 securities by average daily CRSP share volume

The script writes local raw outputs:

```text
data/raw/market/crsp_top_volume_universe.csv
data/raw/market/crsp_top_volume_universe_manifest.json
```

These files are intentionally ignored by Git. They should be reproduced from WRDS rather than committed.

The Streamlit app uses tracked aggregated copies under `app_data/`:

```text
app_data/crsp_top_volume_universe.csv
app_data/top20_monthly_volume.csv
app_data/top20_monthly_prices.csv
```

These are small validation artifacts used only to render the hosted charts. They are not the full raw CRSP daily panel.

To refresh the top-20 monthly validation artifacts, run:

```bash
conda run -n sentiment-ltr-paper python scripts/export_top20_monthly_volume.py
conda run -n sentiment-ltr-paper python scripts/export_top20_monthly_prices.py
cp data/processed/validation/top20_monthly_volume.csv app_data/top20_monthly_volume.csv
cp data/processed/validation/top20_monthly_prices.csv app_data/top20_monthly_prices.csv
```

## Validate The Candidate Universe

Run:

```bash
conda run -n sentiment-ltr-paper jupyter nbconvert \
  --to notebook \
  --execute notebooks/crsp_universe_validation.ipynb \
  --inplace \
  --ExecutePreprocessor.kernel_name=sentiment-ltr-paper
```

The validation notebook checks:

- The candidate universe has the expected 1,000 rows.
- `volume_rank` is unique.
- `permno` is unique.
- `avg_volume` is sorted in descending order.
- `shrcd` values are within the configured common-stock filters.
- `exchcd` values are within the configured exchange filters.
- The top 20 stocks by average daily share volume are displayed in a table.
- The top 20 stocks are plotted as an interactive Plotly horizontal bar chart.
- CRSP daily volume for the top 20 stocks is queried over 2003-2014.
- Daily volume is aggregated to monthly average daily volume and plotted over time with an interactive Plotly line chart.
- CRSP daily `openprc` and `prc` for the top 20 stocks are aggregated into monthly open, close, and average price series for the Streamlit validation app.
- The Streamlit app can optionally query WRDS live for an arbitrary ticker and selected date range, returning CRSP name-history matches and daily rows from `crsp.dsf`.

Current validation result:

- Rows: 1,000
- Expected rows: 1,000
- Top 3 by average daily volume: `C`, `BAC`, `MSFT`
- Spot checks found: `MSFT`, `GE`, `AAPL`, `XOM`
- `SPY` is not included, as expected, because ETFs are excluded by the common-stock filter.
- Top-20 over-time volume query returned 49,121 daily CRSP rows.
- Top-20 monthly price export produced 2,344 monthly stock rows.

## Output Schema

### `crsp_top_volume_universe.csv`

| Column | Description |
| --- | --- |
| `volume_rank` | Rank by average daily CRSP share volume over the configured date range. Rank 1 is highest volume. |
| `permno` | CRSP permanent security identifier. Primary security-level identifier for future CRSP joins. |
| `permco` | CRSP permanent company identifier. Useful for grouping multiple securities issued by the same company. |
| `ticker` | Latest ticker found in `crsp.msenames` overlapping the configured date range. |
| `comnam` | Latest company name found in `crsp.msenames` overlapping the configured date range. |
| `shrcd` | CRSP share code from the selected name record. Current filters keep common shares: 10 and 11. |
| `exchcd` | CRSP exchange code from the selected name record. Current filters keep NYSE, AMEX, and Nasdaq: 1, 2, and 3. |
| `trading_days` | Number of daily CRSP observations with non-missing volume in the date range. |
| `first_trade_date` | First observed CRSP daily row for the security within the date range. |
| `last_trade_date` | Last observed CRSP daily row for the security within the date range. |
| `avg_volume` | Average daily CRSP share volume over the date range. Used for ranking candidates. |
| `avg_dollar_volume` | Average daily dollar volume, computed as `abs(prc) * vol`. Useful for alternative liquidity checks. |
| `avg_abs_price` | Average absolute CRSP price over the date range. CRSP prices can be negative when they are bid/ask averages, so the script uses `abs(prc)`. |
| `avg_shares_outstanding` | Average CRSP shares outstanding over the date range. |
| `latest_name_start` | Start date for the selected latest overlapping CRSP name record. |
| `latest_name_end` | End date for the selected latest overlapping CRSP name record. |

### `crsp_top_volume_universe_manifest.json`

| Field | Description |
| --- | --- |
| `created_at` | UTC timestamp when the local output was generated. |
| `source` | WRDS/CRSP tables used by the script. |
| `start` | Start date used for the CRSP query. |
| `end` | End date used for the CRSP query. |
| `candidate_count` | Number of top-volume candidates requested. |
| `share_codes` | CRSP share codes used in the filter. |
| `exchange_codes` | CRSP exchange codes used in the filter. |
| `rows` | Number of rows written to the candidate CSV. |
| `columns` | Ordered list of columns in the candidate CSV. |
| `output_file` | Relative path to the candidate CSV. |
| `ranking_rule` | Human-readable description of the ranking rule. |
| `note` | Important limitation: this is only the market-side candidate universe and does not apply the TRNA news filter. |

### `top20_monthly_prices.csv`

| Column | Description |
| --- | --- |
| `month` | Month bucket for the CRSP daily observations. |
| `ticker` | CRSP ticker for the selected top-20 security. |
| `comnam` | Company name for the selected top-20 security. |
| `open_price` | First available absolute CRSP `openprc` value in the month. |
| `close_price` | Last available absolute CRSP `prc` value in the month. |
| `avg_price` | Average absolute CRSP closing price, computed from daily `prc`, during the month. |
| `trading_days` | Number of daily CRSP observations used for that monthly stock row. |

### Live Streamlit Ticker Lookup

The app's arbitrary ticker lookup queries `crsp.msenames` first to find matching CRSP securities whose name records overlap the selected date range. It then joins those PERMNO values to `crsp.dsf` and returns the daily rows WRDS provides for that ticker/date range.

Returned daily fields include:

| Column | Description |
| --- | --- |
| `permno` | CRSP permanent security identifier. |
| `permco` | CRSP permanent company identifier from the matching name record. |
| `date` | CRSP daily observation date. |
| `ticker` | Ticker from the matching CRSP name-history record. |
| `comnam` | Company name from the matching CRSP name-history record. |
| `shrcd` | CRSP share code. |
| `exchcd` | CRSP exchange code. |
| `openprc` | CRSP daily opening price, when available. |
| `prc` | CRSP daily closing price. Negative values can indicate bid/ask averages, so the app also computes absolute-price columns. |
| `ret` | CRSP daily return including distributions. |
| `retx` | CRSP daily return excluding distributions. |
| `vol` | CRSP daily trading volume. |
| `shrout` | CRSP shares outstanding. |
| `cfacpr` | CRSP cumulative price adjustment factor. |
| `cfacshr` | CRSP cumulative share adjustment factor. |
| `bidlo` | CRSP daily low/bid-low field. |
| `askhi` | CRSP daily high/ask-high field. |
| `abs_openprc`, `abs_prc`, `abs_bidlo`, `abs_askhi` | Absolute-value helper columns for easier plotting and interpretation. |

## Interpretation Notes

This output is a market-side candidate universe. It is suitable for the next market-data step: pulling daily OHLCV and return data for the selected `permno` values.

It does not yet match the paper's final 512-stock universe: RavenPack articles are cached for ~598 tickers (Batch Pipeline **2B**), and `news_coverage.py` can compute the ≥1 article/week threshold, but that filter has not been applied to produce a committed ~512-name universe file.

The current ranking uses average daily share volume because the paper says it selected the top 1,000 stocks by average trading volume. The manifest also includes average dollar volume so we can compare or switch liquidity definitions if needed.
