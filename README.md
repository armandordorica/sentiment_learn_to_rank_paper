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

For WRDS/CRSP access, add your WRDS credentials to a local `.env` file:

```bash
WRDS_USERNAME=your_wrds_username
WRDS_PASSWORD=your_wrds_password
```

Do not commit `.env` to the repository. Start with `wrds_connection.ipynb` to verify authentication and CRSP table access.

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
| Define data schema and file formats | To Do | Specify columns, date conventions, identifiers, and storage paths under `data/`. |
| Maintain reproducible environment | Pending Review | Conda environment, lock file, and notebook kernel are in place; update when dependencies change. |
| Document replication limitations | To Do | Record data substitutions, missing assumptions, and deviations from the paper. |

### Market Data And Universe

| Task | Status | Notes |
| --- | --- | --- |
| Choose primary market data source | Done | Use WRDS/CRSP as the primary market data source; keep Yahoo Finance for quick public smoke tests and fallback checks. |
| Verify WRDS connection and CRSP access | Done | `wrds_connection.ipynb` connected to WRDS, confirmed CRSP access, and returned a tiny CRSP sample query. |
| Define market data universe candidate list | Done | Added a WRDS/CRSP script that builds the top 1,000 common-stock candidates by average daily volume for 2003-2014. |
| Validate CRSP candidate universe | Pending Review | `notebooks/crsp_universe_validation.ipynb` checks row counts and filters, then displays the top 20 stocks by average daily volume. |
| Build raw market data loader | Pending Review | Adapted the Yahoo Finance and Alpaca retrieval helper from `Portfolio_Optimization_2023`; WRDS/CRSP loader still needs to be implemented. |
| Pull daily OHLCV data for 2003-2014 | To Do | Next task: download adjusted close, close, open, high, low, volume, returns, shares outstanding, and delisting data for the CRSP candidate securities. |
| Pull benchmark market data | To Do | Download SPY or S&P 500 benchmark data for the full 2003-2014 window. |
| Pull or approximate GICS sectors | To Do | Needed for sector-specific sentiment lookback windows and stock-universe diagnostics. |
| Map identifiers across data sources | To Do | Maintain PERMNO, PERMCO, ticker, RIC, company name, exchange, and sector identifiers so market data can join to news sentiment. |
| Validate market data coverage | To Do | Check missing prices, stale symbols, delistings, split/dividend adjustment, and date coverage. |
| Store raw market data locally | To Do | Save reproducible raw pulls under `data/raw/market/`, which is intentionally ignored by Git. |
| Create market data manifest | To Do | Track source, pull date, symbols, date range, row counts, failures, and schema version. |
| Build processed daily market panel | To Do | Produce a clean daily table ready for weekly aggregation and feature generation. |

### News Sentiment Data

| Task | Status | Notes |
| --- | --- | --- |
| Obtain company-level news sentiment data | Blocked | Exact replication requires TRNA or an equivalent historical sentiment dataset. |
| Build raw news sentiment loader | To Do | Load article timestamps, stock identifiers, `pos`, `obj`, `neg`, and `relevance`. |
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
| Validate survivorship-bias handling | To Do | Confirm CRSP delisted securities and name history are handled correctly. |
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
