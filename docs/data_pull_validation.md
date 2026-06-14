# Data Pull And Validation Guide

This guide documents how to pull and validate the current market-side candidate universe for the paper replication.

At this stage, the goal is not to apply the TRNA news-coverage filter. The goal is to build a sufficiently large CRSP candidate universe that approximates the paper's first universe step: selecting liquid stocks by trading volume.

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

## Interpretation Notes

This output is a market-side candidate universe. It is suitable for the next market-data step: pulling daily OHLCV and return data for the selected `permno` values.

It does not yet match the paper's final 512-stock universe because the paper also excludes stocks with fewer than one news article per week in TRNA. That filter will be applied later after the news sentiment source is available.

The current ranking uses average daily share volume because the paper says it selected the top 1,000 stocks by average trading volume. The manifest also includes average dollar volume so we can compare or switch liquidity definitions if needed.
