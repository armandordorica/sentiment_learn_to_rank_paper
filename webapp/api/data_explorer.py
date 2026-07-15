"""FastAPI adapter for Streamlit Tab 1: Unified Ticker Data Explorer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
load_dotenv(PROJECT_ROOT / ".env")

from sentiment_ltr.data import live_data  # noqa: E402

DEFAULT_START = "2003-01-01"
DEFAULT_END = "2014-12-31"
QUICK_TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "TSLA"]
TOP1K_BY_TICKER_DIR = PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "by_ticker"


def _refinitiv_ready() -> bool:
    try:
        from sentiment_ltr.data.refinitiv_queries import refinitiv_configured

        return bool(refinitiv_configured(PROJECT_ROOT))
    except Exception:
        return False


def page_defaults() -> dict[str, Any]:
    wrds_ready = live_data.wrds_credentials_available()
    return {
        "ticker": "AAPL",
        "start_date": DEFAULT_START,
        "end_date": DEFAULT_END,
        "today": pd.Timestamp.today().strftime("%Y-%m-%d"),
        "quick_tickers": QUICK_TICKERS,
        "status": {
            "refinitiv": "Ready" if _refinitiv_ready() else "Not configured",
            "wrds": "Ready" if wrds_ready else "Not configured",
            "yahoo": "Ready",
            "ravenpack": "Ready" if wrds_ready else "Not configured",
        },
        "defaults": {
            "refinitiv": _refinitiv_ready(),
            "wrds": wrds_ready,
            "yahoo": True,
            "ravenpack": wrds_ready,
        },
    }


def _cache_dir(ticker: str) -> Path | None:
    slug = "".join(ch if ch.isalnum() else "_" for ch in ticker.upper().strip())
    if not slug or not TOP1K_BY_TICKER_DIR.exists():
        return None
    matches = sorted(TOP1K_BY_TICKER_DIR.glob(f"rank_*_{slug}"))
    for directory in matches:
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if str(manifest.get("ticker", "")).upper() == ticker.upper().strip():
                return directory
        except Exception:
            continue
    return matches[0] if matches else None


def cache_info(ticker: str) -> dict[str, Any] | None:
    directory = _cache_dir(ticker)
    if directory is None:
        return None
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    return {
        "company_name": manifest.get("company_name") or ticker.upper(),
        "volume_rank": manifest.get("volume_rank"),
        "created_at": str(manifest.get("created_at") or "")[:10],
        "start_date": manifest.get("start_date"),
        "end_date": manifest.get("end_date"),
    }


def _filter(df: pd.DataFrame, column: str, start: str, end: str) -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return df
    dates = pd.to_datetime(df[column], utc=True, errors="coerce").dt.tz_localize(None)
    return df[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end) + pd.Timedelta(days=1))].copy()


def load_cached(ticker: str, start: str, end: str) -> dict[str, Any] | None:
    directory = _cache_dir(ticker)
    if directory is None:
        return None

    def read(name: str) -> pd.DataFrame:
        path = directory / name
        try:
            return pd.read_parquet(path) if path.exists() else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    frames = {
        "refinitiv_prices": _filter(read("refinitiv_prices.parquet"), "date", start, end),
        "refinitiv_news": _filter(read("refinitiv_news.parquet"), "date", start, end),
        "refinitiv_daily": _filter(read("refinitiv_news_daily_counts.parquet"), "date", start, end),
        "wrds_prices": _filter(read("wrds_prices.parquet"), "date", start, end),
        "wrds_names": read("wrds_names.parquet"),
        "yahoo_prices": _filter(read("yahoo_prices.parquet"), "date", start, end),
        "ravenpack": _filter(read("ravenpack_articles.parquet"), "timestamp_utc", start, end),
    }

    def provider(frame: pd.DataFrame, **extra: Any) -> dict[str, Any]:
        return {"status": "ok" if not frame.empty else "empty", "error": None, "prices": frame, **extra}

    info = cache_info(ticker) or {}
    return {
        "ticker": ticker.upper(), "start_date": start, "end_date": end, "source": "cache",
        "cache_created_at": info.get("created_at"),
        "providers": {
            "refinitiv": provider(frames["refinitiv_prices"], news=frames["refinitiv_news"], news_daily_counts=frames["refinitiv_daily"]),
            "wrds": provider(frames["wrds_prices"], names=frames["wrds_names"]),
            "yahoo": provider(frames["yahoo_prices"]),
            "ravenpack": {"status": "ok" if not frames["ravenpack"].empty else "empty", "error": None, "articles": frames["ravenpack"]},
        },
    }


def query(ticker: str, start: str, end: str, *, force_live: bool, refinitiv: bool,
          wrds: bool, yahoo: bool, ravenpack: bool, include_news: bool) -> dict[str, Any]:
    ticker = live_data.clean_ticker(ticker)
    if not ticker:
        raise ValueError("Enter a valid ticker.")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Start date must be on or before end date.")
    if not force_live:
        cached = load_cached(ticker, start, end)
        if cached is not None:
            return cached
    if not any((refinitiv, wrds, yahoo, ravenpack)):
        raise ValueError("Select at least one data source for a live pull.")
    result = live_data.run_ticker_data_query(
        PROJECT_ROOT, ticker, start, end, query_refinitiv=refinitiv, query_wrds=wrds,
        query_yahoo=yahoo, query_ravenpack=ravenpack,
        news_count=1 if include_news else 0, wrds_limit=10_000,
    )
    result["source"] = "live"
    return result


def _html(fig: Any) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _records(df: pd.DataFrame, limit: int = 250) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    display = df.head(limit).copy()
    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].astype(str)
    display = display.where(pd.notna(display), None)
    return {"columns": list(display.columns), "rows": display.to_dict(orient="records"), "total": len(df)}


def present(result: dict[str, Any]) -> dict[str, Any]:
    providers = result["providers"]
    ticker = str(result["ticker"])
    price_frames = {
        name: block.get("prices") for name, block in providers.items()
        if isinstance(block.get("prices"), pd.DataFrame) and not block["prices"].empty
    }
    charts: dict[str, str] = {}
    if price_frames:
        parts = []
        adjusted = {k: v for k, v in price_frames.items() if k != "refinitiv"} or price_frames
        for name, frame in adjusted.items():
            part = frame[["date", "close_price"]].copy()
            part["provider"] = name.title()
            parts.append(part)
        combined = pd.concat(parts, ignore_index=True)
        fig = px.line(combined, x="date", y="close_price", color="provider",
                      title=f"Split-adjusted close price — {ticker}",
                      labels={"close_price": "Close price (USD)", "date": "Date"})
        fig.update_layout(height=480, hovermode="x unified")
        charts["price_overview"] = _html(fig)
        charts["prices"] = _html(fig)

    news = providers.get("refinitiv", {}).get("news", pd.DataFrame())
    daily = providers.get("refinitiv", {}).get("news_daily_counts", pd.DataFrame())
    if isinstance(daily, pd.DataFrame) and not daily.empty and "article_count" in daily:
        fig = px.bar(daily[daily["article_count"] > 0], x="date", y="article_count",
                     title=f"{ticker} Refinitiv articles per day")
        charts["news"] = _html(fig)

    articles = providers.get("ravenpack", {}).get("articles", pd.DataFrame())
    if isinstance(articles, pd.DataFrame) and not articles.empty and "sentiment_score" in articles:
        work = articles.dropna(subset=["sentiment_score"]).copy()
        time_col = "article_time" if "article_time" in work else "timestamp_utc"
        work["article_time"] = pd.to_datetime(work[time_col], utc=True)
        fig = px.scatter(work, x="article_time", y="sentiment_score", color="sentiment_score",
                         hover_data=[c for c in ["headline", "relevance_score"] if c in work],
                         title=f"{ticker} RavenPack article sentiment")
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        charts["sentiment"] = _html(fig)

    statuses = {}
    raw = []
    for name, block in providers.items():
        frame = block.get("articles") if name == "ravenpack" else block.get("prices")
        rows = len(frame) if isinstance(frame, pd.DataFrame) else 0
        statuses[name] = {"status": block.get("status", "unknown"), "rows": rows, "error": block.get("error")}
        table = _records(frame.sort_values(frame.columns[0], ascending=False) if isinstance(frame, pd.DataFrame) and not frame.empty else frame)
        raw.append({"label": f"{name.title()} {'articles' if name == 'ravenpack' else 'prices'}", "table": table, "message": block.get("error") or block.get("status")})

    sentiment_table = _records(articles[[c for c in ["article_time", "headline", "event_text", "relevance_score", "event_sentiment_score", "sentiment_score", "topic", "news_type"] if c in articles.columns]] if isinstance(articles, pd.DataFrame) and not articles.empty else pd.DataFrame())
    return {
        "ticker": ticker, "start_date": result["start_date"], "end_date": result["end_date"],
        "source": result.get("source", "live"), "cache_created_at": result.get("cache_created_at"),
        "statuses": statuses, "charts": charts, "news": _records(news),
        "sentiment": sentiment_table, "raw": raw,
    }
