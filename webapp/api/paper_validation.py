"""FastAPI presentation adapter for Streamlit Tab 7: Paper Validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

from webapp.api import batch_pipeline as bp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATHS = [PROJECT_ROOT / "app_data" / "crsp_top_volume_universe.csv",
                  PROJECT_ROOT / "data" / "raw" / "market" / "crsp_top_volume_universe.csv"]
VOLUME_PATH = PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_volume.csv"
PRICE_PATH = PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_prices.csv"
TOP1K_BY_TICKER_DIR = PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "by_ticker"
PAPER_WEEKS = ((pd.Timestamp("2014-12-31") - pd.Timestamp("2003-01-01")).days + 1) / 7.0
PAPER_WEEKLY_ARTICLE_MEAN = 5.18
PAPER_WEEKLY_ARTICLE_MEDIAN = 2.0


def _chart_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@lru_cache(maxsize=1)
def load_universe() -> pd.DataFrame:
    path = next((p for p in UNIVERSE_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError("CRSP candidate-universe CSV was not found.")
    data = pd.read_csv(path)
    for col in ("first_trade_date", "last_trade_date", "latest_name_start", "latest_name_end"):
        if col in data:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    if "avg_volume_millions" not in data:
        data["avg_volume_millions"] = data["avg_volume"] / 1_000_000
    if "avg_dollar_volume_billions" not in data:
        data["avg_dollar_volume_billions"] = data["avg_dollar_volume"] / 1_000_000_000
    return data.sort_values("volume_rank").reset_index(drop=True)


def validation_checks(universe: pd.DataFrame) -> list[dict[str, Any]]:
    checks = {
        "Exactly 1,000 candidate rows": len(universe) == 1000,
        "Volume rank is unique": universe["volume_rank"].is_unique,
        "PERMNO is unique": universe["permno"].is_unique,
        "Average volume is descending": universe["avg_volume"].is_monotonic_decreasing,
        "Only CRSP common-share codes 10/11": set(universe["shrcd"].dropna().astype(int)).issubset({10, 11}),
        "Only NYSE/AMEX/Nasdaq exchange codes": set(universe["exchcd"].dropna().astype(int)).issubset({1, 2, 3}),
    }
    return [{"check": name, "passed": bool(passed)} for name, passed in checks.items()]


def _parquet_row_count(path: Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.read_metadata(path).num_rows)
    except Exception:
        try:
            return int(len(pd.read_parquet(path, columns=[])))
        except Exception:
            return 0


@lru_cache(maxsize=1)
def _ravenpack_news_filter_stats() -> dict[str, Any]:
    """Fast per-ticker RavenPack volume vs the paper's ≥1 article/week filter.

    Uses parquet row counts / paper-window weeks (batch caches are already
    clipped to 2003–2014). Not a full timestamp scan.
    """
    if not TOP1K_BY_TICKER_DIR.exists():
        return {
            "tickers_with_file": 0,
            "pass_ge_1_per_week": 0,
            "median_avg_per_week": None,
            "mean_avg_per_week": None,
            "paper_mean": PAPER_WEEKLY_ARTICLE_MEAN,
            "paper_median": PAPER_WEEKLY_ARTICLE_MEDIAN,
        }
    avgs: list[float] = []
    for path in TOP1K_BY_TICKER_DIR.glob("rank_*/ravenpack_articles.parquet"):
        n = _parquet_row_count(path)
        if n <= 0:
            continue
        avgs.append(n / PAPER_WEEKS)
    if not avgs:
        return {
            "tickers_with_file": 0,
            "pass_ge_1_per_week": 0,
            "median_avg_per_week": None,
            "mean_avg_per_week": None,
            "paper_mean": PAPER_WEEKLY_ARTICLE_MEAN,
            "paper_median": PAPER_WEEKLY_ARTICLE_MEDIAN,
        }
    series = pd.Series(avgs)
    return {
        "tickers_with_file": int(len(avgs)),
        "pass_ge_1_per_week": int((series >= 1.0).sum()),
        "median_avg_per_week": float(series.median()),
        "mean_avg_per_week": float(series.mean()),
        "paper_mean": PAPER_WEEKLY_ARTICLE_MEAN,
        "paper_median": PAPER_WEEKLY_ARTICLE_MEDIAN,
    }


def _provider_ok(snap: dict[str, Any], name: str) -> int | None:
    for row in snap.get("coverage") or []:
        if str(row.get("provider", "")).upper() == name.upper():
            return int(row.get("ok") or 0)
    return None


def replication_inputs(*, universe_rows: int | None = None) -> list[dict[str, Any]]:
    """Side-by-side paper data needs vs what this repo already fulfills."""
    try:
        mdf = bp.load_manifests()
    except Exception:
        mdf = pd.DataFrame()
    snap = bp.snapshot(mdf) if not mdf.empty else {"empty": True}
    news = _ravenpack_news_filter_stats()

    n_complete = snap.get("n_complete")
    n_partial = snap.get("n_partial")
    n_cached = snap.get("n_cached")
    wrds_ok = _provider_ok(snap, "WRDS")
    rp_ok = _provider_ok(snap, "RAVENPACK")
    yahoo_ok = _provider_ok(snap, "YAHOO")
    ref_ok = _provider_ok(snap, "REFINITIV")

    def _batch_line() -> str:
        if snap.get("empty"):
            return (
                "Batch cache not found under "
                "`data/raw/data_explorer_top1k/by_ticker/`. "
                "Run Tab 2 · Many stocks — retrieve & store first."
            )
        return (
            f"{n_cached:,} / {snap.get('universe_size', 1000):,} tickers cached · "
            f"{n_complete:,} complete · {n_partial:,} partial "
            f"({snap.get('pct_complete', 0):.1f}% fully done). "
            f"Provider ok — WRDS {wrds_ok}, RavenPack {rp_ok}, "
            f"Yahoo {yahoo_ok}, Refinitiv {ref_ok}."
        )

    universe_n = universe_rows
    if universe_n is None:
        try:
            universe_n = len(load_universe())
        except Exception:
            universe_n = None

    if news["tickers_with_file"]:
        news_ours = (
            f"{_batch_line()} "
            f"Among RavenPack files, {news['pass_ge_1_per_week']:,} / "
            f"{news['tickers_with_file']:,} tickers average ≥1 article/week "
            f"(row-count estimate over ~{PAPER_WEEKS:.0f} weeks). "
            f"Our median avg articles/week ≈ {news['median_avg_per_week']:.1f} "
            f"vs paper Table 1 median {news['paper_median']:.0f} "
            f"(RavenPack ≫ TRNA density). "
            "Committed proxy list: `app_data/ravenpack_news_threshold_universe.csv` "
            "(rebuild via `scripts/build_news_threshold_universe.py`). "
            "Partials are mostly delisted / Yahoo-Refinitiv gaps with WRDS still ok."
        )
    else:
        news_ours = _batch_line()

    phrasebank = PROJECT_ROOT / "data" / "models" / "phrasebank_distilbert_best"
    raven_best = PROJECT_ROOT / "data" / "models" / "ravenpack_distilbert_best"
    model_bits = []
    if phrasebank.exists():
        model_bits.append("`data/models/phrasebank_distilbert_best/`")
    if raven_best.exists():
        model_bits.append("`data/models/ravenpack_distilbert_best/`")
    model_line = (
        "Checkpoints that emit `p(positive)` / `p(neutral)` / `p(negative)`: "
        + ", ".join(model_bits)
        if model_bits
        else "No local DistilBERT checkpoints found yet."
    )

    return [
        {
            "id": "top1k",
            "status": "have",
            "status_label": "Have",
            "paper_title": "Top 1,000 stocks by average trading volume (2003–2014)",
            "paper_detail": (
                "Paper step 1: liquid-name candidate pool before the news filter. "
                "Common shares on NYSE / AMEX / Nasdaq."
            ),
            "ours_title": (
                f"{universe_n:,} CRSP candidates cached"
                if universe_n is not None
                else "CRSP top-1k universe"
            ),
            "ours_detail": (
                "Built from WRDS `crsp.dsf` × `crsp.msenames`, ranked by average "
                "daily share volume. Tracked copy: `app_data/crsp_top_volume_universe.csv`."
            ),
            "code_pointers": [
                "scripts/build_crsp_market_universe.py",
                "notebooks/build_top1k_volume_universe.ipynb",
                "docs/data_pull_validation.md",
            ],
            "webapp_pointers": [
                {"label": "7 · Paper Validation (this tab)", "href": "/paper-validation"},
                {"label": "Streamlit tab 7 · Paper Validation", "href": "http://localhost:8501"},
            ],
        },
        {
            "id": "news_filter",
            "status": "partial",
            "status_label": "Partial",
            "paper_title": "Filter to ~512 stocks with ≥1 news article per week",
            "paper_detail": (
                "Paper step 2 (TRNA): drop names with fewer than one article/week on "
                "average → 512 stocks across 10 GICS sectors. Table 1: mean 5.18 / "
                "median 2 articles per stock-week."
            ),
            "ours_title": "RavenPack coverage + batch complete/partial split",
            "ours_detail": news_ours,
            "code_pointers": [
                "src/sentiment_ltr/data/news_coverage.py",
                "src/sentiment_ltr/data/batch_universe.py",
                "scripts/build_news_threshold_universe.py",
                "scripts/backfill_refinitiv_headlines.py",
                "app_data/ravenpack_news_threshold_universe.csv",
                "data/raw/data_explorer_top1k/by_ticker/rank_XXXX_TICKER/",
            ],
            "webapp_pointers": [
                {"label": "2 · Many stocks — retrieve & store → 2B cache", "href": "/batch"},
                {"label": "1 · One stock — inspect & retrieve (sentiment)", "href": "/data-explorer"},
            ],
        },
        {
            "id": "daily_prices",
            "status": "partial" if (wrds_ok or 0) < 1000 else "have",
            "status_label": "Partial" if (wrds_ok or 0) < 1000 else "Have",
            "paper_title": "Daily market data (OHLCV / returns) for the universe",
            "paper_detail": (
                "Bloomberg in the paper; we use WRDS/CRSP as the authoritative "
                "price source, with Yahoo / Refinitiv as cross-checks."
            ),
            "ours_title": (
                f"WRDS prices for {wrds_ok:,} tickers"
                if wrds_ok is not None
                else "Per-ticker WRDS price caches"
            ),
            "ours_detail": (
                f"{_batch_line()} "
                "Files: `…/rank_XXXX_TICKER/wrds_prices.parquet` "
                "(plus optional `yahoo_prices.parquet`, `refinitiv_prices.parquet`)."
            ),
            "code_pointers": [
                "src/sentiment_ltr/data/live_data.py",
                "scripts/run_batch_pipeline.py",
            ],
            "webapp_pointers": [
                {"label": "1 · One stock — inspect & retrieve", "href": "/data-explorer"},
                {"label": "2 · Many stocks — retrieve & store", "href": "/batch"},
            ],
        },
        {
            "id": "trna_fields",
            "status": "partial",
            "status_label": "Partial",
            "paper_title": "Per-article relevance + P(pos) / P(neutral) / P(neg)",
            "paper_detail": (
                "TRNA fields → `S_sentiment = relevance × (pos − neg)`. "
                "Exact TRNA is not licensed; RavenPack + DistilBERT are the substitutes. "
                "Field map below."
            ),
            "ours_title": "RavenPack relevance + model probabilities (not batched yet)",
            "ours_detail": (
                f"Relevance on {rp_ok or news['tickers_with_file']:,} RavenPack tickers. "
                f"{model_line} "
                "Batch-scoring the corpus with class probs (plan 4.2) is still open."
            ),
            "field_map": [
                {
                    "field": "datetime",
                    "paper_source": "TRNA article timestamp",
                    "our_substitute": "RavenPack `article_time` / `timestamp_utc`",
                    "status": "Have",
                },
                {
                    "field": "price (RIC / company id)",
                    "paper_source": "TRNA RIC linked to the mention",
                    "our_substitute": "CRSP `permno` + `ticker` on batch rows / manifests",
                    "status": "Have",
                },
                {
                    "field": "relevance ∈ [0, 1]",
                    "paper_source": "TRNA `relevance`",
                    "our_substitute": "RavenPack `relevance_score` (= `relevance` / 100)",
                    "status": "Have",
                },
                {
                    "field": "sentiment ∈ {−1, 0, +1}",
                    "paper_source": "TRNA predominant `sentiment`",
                    "our_substitute": (
                        "Derived from RavenPack `event_sentiment_score` "
                        "(label thresholds for fine-tuning)"
                    ),
                    "status": "Partial",
                },
                {
                    "field": "pos",
                    "paper_source": "TRNA P(positive)",
                    "our_substitute": (
                        "DistilBERT `p(positive)` "
                        "(PhraseBank / RavenPack checkpoints; not written to corpus yet)"
                    ),
                    "status": "Partial",
                },
                {
                    "field": "obj",
                    "paper_source": "TRNA P(neutral / objective)",
                    "our_substitute": "DistilBERT `p(neutral)` (same checkpoints)",
                    "status": "Partial",
                },
                {
                    "field": "neg",
                    "paper_source": "TRNA P(negative)",
                    "our_substitute": "DistilBERT `p(negative)` (same checkpoints)",
                    "status": "Partial",
                },
                {
                    "field": "S_sentiment",
                    "paper_source": "`relevance × (pos − neg)`",
                    "our_substitute": (
                        "Proxy now: `relevance_score × event_sentiment_score`. "
                        "Target: `relevance_score × (p(pos) − p(neg))` after batch score"
                    ),
                    "status": "Proxy",
                },
            ],
            "code_pointers": [
                "src/sentiment_ltr/models/ravenpack_sentiment.py",
                "src/sentiment_ltr/models/phrasebank_sentiment.py",
                "docs/news_sentiment_finetuning_plan.md",
            ],
            "webapp_pointers": [
                {"label": "5 · RavenPack Fine-Tuning", "href": "/finetune"},
                {"label": "6 · Sentiment Lab", "href": "/sentiment-lab"},
                {"label": "4 · RavenPack Baseline Eval", "href": "/raven-eval"},
            ],
        },
        {
            "id": "weekly_features",
            "status": "missing",
            "status_label": "Missing",
            "paper_title": "Weekly sentiment shock / trend + lagged return features",
            "paper_detail": (
                "Six LTR features per stock-week, sector-specific lookbacks, "
                "quartile labels 1–4 from next-week returns."
            ),
            "ours_title": "AAPL pilot only",
            "ours_detail": (
                "Example weekly proxy file: "
                "`data/raw/news/ravenpack/aapl_weekly_sentiment_2003_2014.parquet`. "
                "Universe-wide shock/trend panel and RankNet/ListNet backtest not built."
            ),
            "code_pointers": [
                "notebooks/fetch_news_articles.ipynb",
                "README.md → Weekly Feature Engineering",
            ],
            "webapp_pointers": [
                {"label": "1 · One stock — weekly RavenPack avg", "href": "/data-explorer"},
            ],
        },
        {
            "id": "benchmark_gics",
            "status": "missing",
            "status_label": "Missing",
            "paper_title": "SPY / S&P 500 benchmark + GICS sectors",
            "paper_detail": (
                "Benchmark for performance charts; GICS drives sector lookback windows "
                "in Table 2 of the paper."
            ),
            "ours_title": "Not pulled yet",
            "ours_detail": (
                "No committed SPY panel or GICS map under `app_data/` / batch cache."
            ),
            "code_pointers": ["README.md → Reproduction Status"],
            "webapp_pointers": [],
        },
    ]


def _top20_chart(top20: pd.DataFrame):
    data = top20.sort_values("avg_volume_millions").copy()
    data["label"] = data["ticker"] + " — " + data["comnam"].str.title()
    fig = px.bar(data, x="avg_volume_millions", y="label", orientation="h",
                 title="Top 20 CRSP common stocks by average daily share volume, 2003–2014",
                 labels={"avg_volume_millions": "Average daily volume, millions of shares", "label": ""},
                 hover_data={"ticker": True, "permno": True, "trading_days": ":,",
                             "avg_volume_millions": ":,.2f"}, color_discrete_sequence=["#4C78A8"])
    fig.update_layout(height=700, hovermode="closest")
    return fig


@lru_cache(maxsize=1)
def page_context() -> dict[str, Any]:
    universe = load_universe()
    top20 = universe.head(20).copy()
    volume = pd.read_csv(VOLUME_PATH, parse_dates=["month"])
    volume_fig = px.line(volume.sort_values(["ticker", "month"]), x="month",
                         y="avg_daily_volume_millions", color="ticker",
                         title="Monthly average daily trading volume for top 20 CRSP candidates, 2003–2014",
                         labels={"month": "Month", "avg_daily_volume_millions": "Average daily volume, millions of shares"},
                         hover_data={"comnam": True, "trading_days": True})
    volume_fig.update_traces(mode="lines+markers", line={"width": 1.8}, marker={"size": 4})
    volume_fig.update_layout(height=750, hovermode="closest")
    display_cols = ["volume_rank", "permno", "ticker", "comnam", "trading_days",
                    "avg_volume_millions", "avg_dollar_volume_billions",
                    "first_trade_date", "last_trade_date"]
    rows = top20[display_cols].copy()
    for col in ("first_trade_date", "last_trade_date"):
        rows[col] = rows[col].dt.strftime("%Y-%m-%d")
    return {
        "summary": {"rows": len(universe), "unique_permnos": int(universe["permno"].nunique()),
                    "top20_rows": len(top20), "date_range": "2003–2014"},
        "checks": validation_checks(universe), "top20_rows": rows.to_dict(orient="records"),
        "top20_columns": display_cols, "top20_chart": _chart_html(_top20_chart(top20)),
        "volume_chart": _chart_html(volume_fig),
        "tickers": top20["ticker"].dropna().tolist(),
    }


@lru_cache(maxsize=24)
def price_chart(ticker: str) -> dict[str, str]:
    ticker = ticker.strip().upper()
    prices = pd.read_csv(PRICE_PATH, parse_dates=["month"])
    stock = prices[prices["ticker"].str.upper() == ticker].copy()
    if stock.empty:
        raise ValueError(f"No monthly price data found for {ticker}.")
    company = stock["comnam"].iloc[0]
    long = stock.melt(id_vars=["month", "ticker", "comnam", "trading_days"],
                      value_vars=["open_price", "close_price", "avg_price"],
                      var_name="price_type", value_name="price")
    long["price_type"] = long["price_type"].map(
        {"open_price": "Open price", "close_price": "Close price", "avg_price": "Average price"})
    fig = px.line(long, x="month", y="price", color="price_type",
                  title=f"Monthly open, close, and average price for {ticker} — {company.title()}",
                  labels={"month": "Month", "price": "Price, USD", "price_type": "Series"},
                  hover_data={"trading_days": True})
    fig.update_traces(mode="lines+markers", line={"width": 2}, marker={"size": 5})
    fig.update_layout(height=650, hovermode="closest")
    return {"ticker": ticker, "chart": _chart_html(fig)}
