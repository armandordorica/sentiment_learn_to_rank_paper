"""FastAPI adapter for Streamlit Tab 1: Unified Ticker Data Explorer."""

from __future__ import annotations

import json
import hashlib
import re
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
from sentiment_ltr.data.ravenpack_match import (  # noqa: E402
    find_best_ravenpack_match,
    resolve_ravenpack_for_story,
)
from sentiment_ltr.data.refinitiv_queries import fetch_refinitiv_story  # noqa: E402

DEFAULT_START = "2003-01-01"
DEFAULT_END = "2014-12-31"
QUICK_TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "TSLA"]
TOP1K_BY_TICKER_DIR = PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "by_ticker"
FULL_STORY_DIR = PROJECT_ROOT / "data" / "raw" / "data_explorer_full_stories"
RAVENPACK_DAY_DIR = PROJECT_ROOT / "data" / "raw" / "data_explorer_ravenpack_days"
RP_LIST_LIMIT_CHOICES = (10, 25, 50, 100, 250)
DEFAULT_RP_LIST_LIMIT = 25
SOFT_MATCH_LIMIT_CHOICES = (25, 50, 100, 250)
DEFAULT_SOFT_MATCH_LIMIT = 50
DEFAULT_SOFT_MATCH_MIN_SCORE = 0.45
DEFAULT_SOFT_MATCH_WINDOW_HOURS = 36.0


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

    ravenpack = load_cached_ravenpack(ticker)
    if ravenpack.empty:
        ravenpack = read("ravenpack_articles.parquet")
    time_col = "timestamp_utc" if "timestamp_utc" in ravenpack.columns else "article_time"
    frames = {
        "refinitiv_prices": _filter(read("refinitiv_prices.parquet"), "date", start, end),
        "refinitiv_news": _filter(read("refinitiv_news.parquet"), "date", start, end),
        "refinitiv_daily": _filter(read("refinitiv_news_daily_counts.parquet"), "date", start, end),
        "wrds_prices": _filter(read("wrds_prices.parquet"), "date", start, end),
        "wrds_names": read("wrds_names.parquet"),
        "yahoo_prices": _filter(read("yahoo_prices.parquet"), "date", start, end),
        "ravenpack": _filter(ravenpack, time_col, start, end) if time_col in ravenpack.columns else ravenpack,
    }


    def provider(frame: pd.DataFrame, **extra: Any) -> dict[str, Any]:
        return {"status": "ok" if not frame.empty else "empty", "error": None, "prices": frame, **extra}

    info = cache_info(ticker) or {}
    return {
        "ticker": ticker.upper(), "start_date": start, "end_date": end, "source": "cache",
        "cache_created_at": info.get("created_at"),
        "cache_dir": str(directory.resolve()),
        "data_paths": {
            "refinitiv_news": str((directory / "refinitiv_news.parquet").resolve()),
        },
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
    # Plotly is loaded once by base.html. Loading it inside an HTMX fragment can
    # race the inline newPlot call and leave an otherwise valid chart blank.
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _records(df: pd.DataFrame, limit: int = 250) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    display = df.head(limit).copy()
    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].astype(str)
    display = display.where(pd.notna(display), None)
    return {"columns": list(display.columns), "rows": display.to_dict(orient="records"), "total": len(df)}


HEADLINE_LIMIT_CHOICES = (10, 25, 50, 100, 250)
DEFAULT_HEADLINE_LIMIT = 25
HEADLINE_SORT_CHOICES = {
    "date_desc": ("date", False),
    "date_asc": ("date", True),
    "headline_asc": ("headline", True),
    "headline_desc": ("headline", False),
    "source_asc": ("sourceCode", True),
    "source_desc": ("sourceCode", False),
}


def refinitiv_headline_list(
    news: pd.DataFrame,
    limit: int = DEFAULT_HEADLINE_LIMIT,
    *,
    query: str = "",
    sort: str = "date_desc",
) -> dict[str, Any] | None:
    """Presentation rows for the selectable Refinitiv full-story list.

    Filters the full frame first, then sorts, then takes the top ``limit`` rows
    so search is not limited to the currently displayed page.
    """
    if not isinstance(news, pd.DataFrame) or news.empty:
        return None
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DEFAULT_HEADLINE_LIMIT
    limit = max(1, min(limit, max(HEADLINE_LIMIT_CHOICES)))
    sort = sort if sort in HEADLINE_SORT_CHOICES else "date_desc"
    columns = [c for c in ("date", "headline", "sourceCode", "storyId") if c in news]
    display = news[columns].copy()
    total = int(len(display))

    q = str(query or "").strip().lower()
    if q:
        mask = pd.Series(False, index=display.index)
        for col in ("headline", "sourceCode", "storyId"):
            if col in display.columns:
                mask = mask | display[col].astype(str).str.lower().str.contains(
                    q, na=False, regex=False
                )
        display = display.loc[mask]
    matched = int(len(display))

    sort_col, ascending = HEADLINE_SORT_CHOICES[sort]
    if sort_col in display.columns:
        if sort_col == "date":
            keys = pd.to_datetime(display["date"], errors="coerce")
            display = display.assign(_sort_key=keys).sort_values(
                "_sort_key", ascending=ascending, kind="stable"
            ).drop(columns="_sort_key")
        else:
            display = display.sort_values(
                sort_col, ascending=ascending, kind="stable", na_position="last"
            )
    elif "date" in display.columns:
        keys = pd.to_datetime(display["date"], errors="coerce")
        display = display.assign(_sort_key=keys).sort_values(
            "_sort_key", ascending=False, kind="stable"
        ).drop(columns="_sort_key")

    if "date" in display.columns:
        display["date"] = pd.to_datetime(display["date"], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M"
        )
    display = display.head(limit)
    display = display.where(pd.notna(display), None)
    return {
        "rows": display.to_dict(orient="records"),
        "total": total,
        "matched": matched,
        "shown": int(len(display)),
        "limit": limit,
        "query": str(query or "").strip(),
        "sort": sort,
        "limit_choices": list(HEADLINE_LIMIT_CHOICES),
    }


def headlines_from_cache(
    ticker: str,
    start: str,
    end: str,
    *,
    limit: int = DEFAULT_HEADLINE_LIMIT,
    query: str = "",
    sort: str = "date_desc",
) -> dict[str, Any]:
    """Reload Refinitiv headlines from the ticker cache for HTMX filter/sort/top-N."""
    cached = load_cached(ticker, start, end)
    if cached is None:
        raise ValueError(
            "No local cache for this ticker/window. Re-run Load data, or use a cached ticker."
        )
    news = cached["providers"].get("refinitiv", {}).get("news", pd.DataFrame())
    headlines = refinitiv_headline_list(news, limit=limit, query=query, sort=sort)
    if headlines is None:
        raise ValueError("No Refinitiv headlines in cache for this ticker/window.")
    headlines["ticker"] = live_data.clean_ticker(ticker) or ticker.upper()
    headlines["start_date"] = start
    headlines["end_date"] = end
    return headlines


def _story_path(story_id: str, headline: str, ticker: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "-", headline.lower()).strip("-")[:80] or "refinitiv-story"
    digest = hashlib.sha256(story_id.encode("utf-8")).hexdigest()[:12]
    clean_ticker = live_data.clean_ticker(ticker) or "UNKNOWN"
    return FULL_STORY_DIR / clean_ticker / f"{slug}--{digest}.txt"


def _query_ravenpack_day(ticker: str, start: str, end: str) -> pd.DataFrame:
    """WRDS day pull with headlines for story↔relevance matching."""
    return live_data.query_ravenpack_articles(ticker, start, end, include_text=True)


def load_cached_ravenpack(ticker: str) -> pd.DataFrame:
    """RavenPack rows for a ticker, preferring frames that include ``headline``.

    Order:
    1. Rich export under ``data/raw/news/ravenpack/{ticker}_articles_*.parquet``
    2. Batch Data Explorer cache, enriched with rich headlines when possible
    """
    rich = _load_rich_ravenpack_export(ticker)
    if not rich.empty and "headline" in rich.columns:
        return rich

    directory = _cache_dir(ticker)
    if directory is None:
        return pd.DataFrame()
    path = directory / "ravenpack_articles.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        batch = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if batch.empty:
        return batch
    if "headline" in batch.columns and batch["headline"].notna().any():
        return batch
    # Legacy batch caches omitted titles — attach from rich export by story id.
    if not rich.empty:
        return _attach_headlines_from_rich(batch, rich)
    return batch


def _load_rich_ravenpack_export(ticker: str) -> pd.DataFrame:
    """Load the headline-bearing RavenPack training export when present."""
    try:
        from sentiment_ltr.models.ravenpack_sentiment import discover_ravenpack_article_files
    except Exception:
        return pd.DataFrame()
    paths = discover_ravenpack_article_files([live_data.clean_ticker(ticker) or ticker])
    for path in paths:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        if frame.empty:
            continue
        work = frame.copy()
        if "rp_story_id" not in work.columns and "story_id" in work.columns:
            work["rp_story_id"] = work["story_id"]
        if "timestamp_utc" not in work.columns and "article_time" in work.columns:
            work["timestamp_utc"] = work["article_time"]
        if "relevance" not in work.columns and "relevance_score" in work.columns:
            # rich export already stores 0–1 relevance_score
            work["relevance"] = pd.to_numeric(work["relevance_score"], errors="coerce") * 100.0
        if "relevance_score" not in work.columns and "relevance" in work.columns:
            work["relevance_score"] = pd.to_numeric(work["relevance"], errors="coerce") / 100.0
        if "ticker" not in work.columns:
            work["ticker"] = live_data.clean_ticker(ticker) or ticker.upper()
        return work
    return pd.DataFrame()


def _attach_headlines_from_rich(batch: pd.DataFrame, rich: pd.DataFrame) -> pd.DataFrame:
    """Left-join rich headlines onto a legacy batch frame by story id."""
    if batch.empty or rich.empty:
        return batch
    left = batch.copy()
    right = rich.copy()
    left_id = "rp_story_id" if "rp_story_id" in left.columns else None
    right_id = "rp_story_id" if "rp_story_id" in right.columns else (
        "story_id" if "story_id" in right.columns else None
    )
    if not left_id or not right_id:
        return left
    cols = [right_id]
    for c in ("headline", "event_text"):
        if c in right.columns:
            cols.append(c)
    merge_right = right[cols].drop_duplicates(subset=[right_id], keep="first")
    if right_id != left_id:
        merge_right = merge_right.rename(columns={right_id: left_id})
    already = [c for c in ("headline", "event_text") if c in left.columns]
    if already:
        left = left.drop(columns=already)
    out = left.merge(merge_right, on=left_id, how="left")
    return out


def ravenpack_article_list(
    articles: pd.DataFrame,
    *,
    ticker: str,
    limit: int = DEFAULT_RP_LIST_LIMIT,
    query: str = "",
    sort: str = "date_desc",
) -> dict[str, Any] | None:
    """Selectable RavenPack rows for 1E inspect (headline search when present)."""
    if not isinstance(articles, pd.DataFrame) or articles.empty:
        return None
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DEFAULT_RP_LIST_LIMIT
    limit = max(1, min(limit, max(RP_LIST_LIMIT_CHOICES)))
    sort = sort if sort in {"date_desc", "date_asc", "headline_asc", "headline_desc",
                            "relevance_desc", "relevance_asc"} else "date_desc"

    work = articles.copy()
    if "relevance_score" not in work.columns and "relevance" in work.columns:
        work["relevance_score"] = pd.to_numeric(work["relevance"], errors="coerce") / 100.0
    time_col = "article_time" if "article_time" in work.columns else "timestamp_utc"
    total = int(len(work))
    has_headline = "headline" in work.columns

    q = str(query or "").strip().lower()
    if q:
        mask = pd.Series(False, index=work.index)
        for col in ("headline", "event_text", "topic", "type", "rp_story_id", "news_type"):
            if col in work.columns:
                mask = mask | work[col].astype(str).str.lower().str.contains(
                    q, na=False, regex=False
                )
        work = work.loc[mask]
    matched = int(len(work))

    if sort.startswith("date") and time_col in work.columns:
        keys = pd.to_datetime(work[time_col], utc=True, errors="coerce")
        work = work.assign(_sort_key=keys).sort_values(
            "_sort_key", ascending=sort == "date_asc", kind="stable"
        ).drop(columns="_sort_key")
    elif sort.startswith("headline") and has_headline:
        work = work.sort_values(
            "headline", ascending=sort == "headline_asc", kind="stable", na_position="last"
        )
    elif sort.startswith("relevance") and "relevance_score" in work.columns:
        work = work.sort_values(
            "relevance_score",
            ascending=sort == "relevance_asc",
            kind="stable",
            na_position="last",
        )
    elif time_col in work.columns:
        keys = pd.to_datetime(work[time_col], utc=True, errors="coerce")
        work = work.assign(_sort_key=keys).sort_values(
            "_sort_key", ascending=False, kind="stable"
        ).drop(columns="_sort_key")

    cols = [
        c for c in [
            time_col, "headline", "event_text", "relevance_score",
            "event_sentiment_score", "sentiment_score", "topic", "group", "type",
            "sub_type", "news_type", "source_name", "rp_story_id", "css", "nip",
        ]
        if c in work.columns
    ]
    display = work[cols].head(limit).copy()
    if time_col in display.columns:
        display[time_col] = pd.to_datetime(display[time_col], utc=True, errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M"
        )
        display = display.rename(columns={time_col: "article_time"})
    display = display.where(pd.notna(display), None)
    return {
        "ticker": ticker,
        "total": total,
        "matched": matched,
        "shown": int(len(display)),
        "limit": limit,
        "query": str(query or "").strip(),
        "sort": sort,
        "limit_choices": list(RP_LIST_LIMIT_CHOICES),
        "has_headline": has_headline,
        "rows": display.to_dict(orient="records"),
    }


def ravenpack_list_from_cache(
    ticker: str,
    start: str,
    end: str,
    *,
    limit: int = DEFAULT_RP_LIST_LIMIT,
    query: str = "",
    sort: str = "date_desc",
) -> dict[str, Any]:
    """Reload RavenPack rows from the ticker cache for HTMX filter/sort/top-N."""
    cached = load_cached(ticker, start, end)
    if cached is None:
        raise ValueError(
            "No local cache for this ticker/window. Re-run Load data, or use a cached ticker."
        )
    articles = cached["providers"].get("ravenpack", {}).get("articles", pd.DataFrame())
    listing = ravenpack_article_list(
        articles, ticker=live_data.clean_ticker(ticker) or ticker.upper(),
        limit=limit, query=query, sort=sort,
    )
    if listing is None:
        raise ValueError("No RavenPack articles in cache for this ticker/window.")
    listing["start_date"] = start
    listing["end_date"] = end
    return listing


def soft_match_comparison(
    news: pd.DataFrame,
    articles: pd.DataFrame,
    *,
    ticker: str,
    limit: int = DEFAULT_SOFT_MATCH_LIMIT,
    query: str = "",
    min_score: float = DEFAULT_SOFT_MATCH_MIN_SCORE,
    window_hours: float = DEFAULT_SOFT_MATCH_WINDOW_HOURS,
    matched_only: bool = False,
) -> dict[str, Any] | None:
    """Side-by-side Refinitiv ↔ RavenPack soft-match table for inspection."""
    if not isinstance(news, pd.DataFrame) or news.empty or "headline" not in news.columns:
        return None
    if not isinstance(articles, pd.DataFrame) or articles.empty or "headline" not in articles.columns:
        return {
            "ticker": ticker,
            "rows": [],
            "total_refinitiv": int(len(news)),
            "matched": 0,
            "shown": 0,
            "limit": limit,
            "query": str(query or "").strip(),
            "min_score": float(min_score),
            "window_hours": float(window_hours),
            "matched_only": matched_only,
            "limit_choices": list(SOFT_MATCH_LIMIT_CHOICES),
            "note": "RavenPack cache has no headlines to soft-match against.",
        }

    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = DEFAULT_SOFT_MATCH_LIMIT
    limit = max(1, min(limit, max(SOFT_MATCH_LIMIT_CHOICES)))
    try:
        min_score = float(min_score)
    except (TypeError, ValueError):
        min_score = DEFAULT_SOFT_MATCH_MIN_SCORE
    min_score = max(0.0, min(min_score, 1.0))

    ref = news.copy()
    date_col = "date" if "date" in ref.columns else None
    total_ref = int(len(ref))
    q = str(query or "").strip().lower()
    if q:
        mask = ref["headline"].astype(str).str.lower().str.contains(q, na=False, regex=False)
        if "storyId" in ref.columns:
            mask = mask | ref["storyId"].astype(str).str.lower().str.contains(q, na=False, regex=False)
        ref = ref.loc[mask]
    if date_col:
        keys = pd.to_datetime(ref[date_col], utc=True, errors="coerce")
        ref = ref.assign(_sort_key=keys).sort_values(
            "_sort_key", ascending=False, kind="stable"
        ).drop(columns="_sort_key")
    ref = ref.head(limit)

    rows: list[dict[str, Any]] = []
    n_matched = 0
    for _, item in ref.iterrows():
        headline = str(item.get("headline") or "")
        story_time = pd.to_datetime(item.get(date_col), utc=True, errors="coerce") if date_col else pd.NaT
        if not pd.isna(story_time):
            story_time = story_time.tz_localize(None) if getattr(story_time, "tzinfo", None) else story_time
        hit = find_best_ravenpack_match(
            articles,
            headline=headline,
            story_time=None if pd.isna(story_time) else story_time,
            window_hours=window_hours,
            min_score=min_score,
            fallback_outside_window=False,
        )
        date_ref = (
            story_time.strftime("%Y-%m-%d %H:%M") if not pd.isna(story_time) else None
        )
        if hit is None:
            if matched_only:
                continue
            rows.append({
                "date_refinitiv": date_ref,
                "date_ravenpack": None,
                "headline_refinitiv": headline or None,
                "headline_ravenpack": None,
                "match_score": None,
                "relevance_score": None,
                "matched": False,
                "story_id": item.get("storyId"),
                "rp_story_id": None,
            })
            continue
        n_matched += 1
        rp_time = hit.get("article_time")
        if rp_time:
            parsed = pd.to_datetime(rp_time, utc=True, errors="coerce")
            if not pd.isna(parsed):
                parsed = parsed.tz_localize(None) if getattr(parsed, "tzinfo", None) else parsed
                rp_time = parsed.strftime("%Y-%m-%d %H:%M")
        rows.append({
            "date_refinitiv": date_ref,
            "date_ravenpack": rp_time,
            "headline_refinitiv": headline or None,
            "headline_ravenpack": hit.get("headline"),
            "match_score": hit.get("match_score"),
            "relevance_score": hit.get("relevance_score"),
            "matched": True,
            "story_id": item.get("storyId"),
            "rp_story_id": hit.get("rp_story_id"),
        })

    return {
        "ticker": ticker,
        "rows": rows,
        "total_refinitiv": total_ref,
        "candidates": int(len(ref)),
        "matched": n_matched,
        "shown": len(rows),
        "limit": limit,
        "query": str(query or "").strip(),
        "min_score": float(min_score),
        "window_hours": float(window_hours),
        "matched_only": matched_only,
        "limit_choices": list(SOFT_MATCH_LIMIT_CHOICES),
        "note": None,
    }


def soft_match_from_cache(
    ticker: str,
    start: str,
    end: str,
    *,
    limit: int = DEFAULT_SOFT_MATCH_LIMIT,
    query: str = "",
    min_score: float = DEFAULT_SOFT_MATCH_MIN_SCORE,
    matched_only: bool = False,
) -> dict[str, Any]:
    """Reload soft-match comparison from cache for HTMX controls."""
    cached = load_cached(ticker, start, end)
    if cached is None:
        raise ValueError(
            "No local cache for this ticker/window. Re-run Load data, or use a cached ticker."
        )
    news = cached["providers"].get("refinitiv", {}).get("news", pd.DataFrame())
    articles = cached["providers"].get("ravenpack", {}).get("articles", pd.DataFrame())
    table = soft_match_comparison(
        news,
        articles,
        ticker=live_data.clean_ticker(ticker) or ticker.upper(),
        limit=limit,
        query=query,
        min_score=min_score,
        matched_only=matched_only,
    )
    if table is None:
        raise ValueError("No Refinitiv headlines in cache for this ticker/window.")
    table["start_date"] = start
    table["end_date"] = end
    return table


def _ravenpack_inspect_note(
    *,
    headline: object,
    event_text: object,
    news_type: object,
) -> str | None:
    """Explain empty RavenPack fields — FULL-ARTICLE ≠ full wire body in WRDS."""
    if not headline:
        return (
            "Batch RavenPack cache often omits headline/event_text; "
            "relevance and taxonomy still apply to this ticker."
        )
    if event_text:
        return None
    nt = news_type or "—"
    return (
        "RavenPack WRDS has no full wire body for this row — only the headline "
        f"(taxonomy/scores may also be empty). `news_type={nt}` labels the source "
        "item type; it is not downloadable article text. Use Refinitiv "
        "'Read story' for the Reuters body when a soft-matched headline exists nearby."
    )


def ravenpack_article_detail(
    ticker: str,
    *,
    rp_story_id: str = "",
    article_time: str = "",
    headline: str = "",
) -> dict[str, Any]:
    """1E inspect panel for one RavenPack row from the ticker cache."""
    articles = load_cached_ravenpack(ticker)
    if articles.empty:
        raise ValueError(f"No cached RavenPack articles for {ticker}.")
    work = articles.copy()
    if "relevance_score" not in work.columns and "relevance" in work.columns:
        work["relevance_score"] = pd.to_numeric(work["relevance"], errors="coerce") / 100.0
    hit = None
    if rp_story_id and "rp_story_id" in work.columns:
        matched = work[work["rp_story_id"].astype(str) == str(rp_story_id)]
        if not matched.empty:
            hit = matched.iloc[0]
    if hit is None and headline and "headline" in work.columns:
        matched = work[work["headline"].astype(str) == str(headline)]
        if not matched.empty:
            hit = matched.iloc[0]
    if hit is None and article_time:
        time_col = "article_time" if "article_time" in work.columns else "timestamp_utc"
        times = pd.to_datetime(work[time_col], utc=True, errors="coerce").dt.tz_localize(None)
        target = pd.to_datetime(article_time, utc=True, errors="coerce")
        if not pd.isna(target):
            target = target.tz_localize(None) if getattr(target, "tzinfo", None) else target
            idx = (times - target).abs().idxmin()
            hit = work.loc[idx]
    if hit is None:
        raise ValueError("Could not locate that RavenPack article in cache.")
    def _val(key: str):
        v = hit.get(key) if hasattr(hit, "get") else hit[key] if key in hit.index else None
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if hasattr(v, "isoformat"):
            return str(v)
        return v if not isinstance(v, (pd.Timestamp,)) else str(v)

    return {
        "ticker": ticker,
        "headline": _val("headline") or "(no headline in batch cache)",
        "event_text": _val("event_text"),
        "relevance_score": float(_val("relevance_score")) if _val("relevance_score") is not None else None,
        "event_sentiment_score": float(_val("event_sentiment_score")) if _val("event_sentiment_score") is not None else None,
        "sentiment_score": float(_val("sentiment_score")) if _val("sentiment_score") is not None else None,
        "topic": _val("topic"),
        "group": _val("group"),
        "type": _val("type"),
        "sub_type": _val("sub_type"),
        "news_type": _val("news_type"),
        "source_name": _val("source_name"),
        "rp_story_id": _val("rp_story_id"),
        "article_time": _val("article_time") or _val("timestamp_utc"),
        "css": _val("css"),
        "nip": _val("nip"),
        "note": _ravenpack_inspect_note(
            headline=_val("headline"),
            event_text=_val("event_text"),
            news_type=_val("news_type"),
        ),
    }


def load_story(
    story_id: str,
    headline: str | None = None,
    ticker: str = "UNKNOWN",
    story_date: str | None = None,
    *,
    include_ravenpack: bool = True,
) -> dict[str, Any]:
    """Fetch and persist one Refinitiv story, with RavenPack relevance when possible."""
    story_id = str(story_id or "").strip()
    if not story_id:
        raise ValueError("Select a Refinitiv headline with a story ID.")
    headline = str(headline or "Selected headline")
    clean_ticker = live_data.clean_ticker(ticker) or "UNKNOWN"
    text = fetch_refinitiv_story(PROJECT_ROOT, story_id)
    path = _story_path(story_id, headline, clean_ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"Headline: {headline}\nStory ID: {story_id}\nTicker: {clean_ticker}\n\n{text}\n",
        encoding="utf-8",
    )
    relative_path = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
    ravenpack: dict[str, Any] | None = None
    if include_ravenpack and clean_ticker != "UNKNOWN":
        query_fn = _query_ravenpack_day if live_data.wrds_credentials_available() else None
        ravenpack = resolve_ravenpack_for_story(
            ticker=clean_ticker,
            headline=headline,
            story_time=story_date,
            cached_articles=load_cached_ravenpack(clean_ticker),
            day_cache_dir=RAVENPACK_DAY_DIR if query_fn else None,
            query_day_fn=query_fn,
        )
    return {
        "story_id": story_id,
        "headline": headline,
        "ticker": clean_ticker,
        "story_date": story_date or "",
        "text": text,
        "path": str(path.resolve()),
        "relative_path": relative_path,
        "ravenpack": ravenpack,
    }


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
        charts["prices"] = _html(fig)

    news = providers.get("refinitiv", {}).get("news", pd.DataFrame())
    news_path = result.get("data_paths", {}).get("refinitiv_news")
    news_storage = {
        "saved": bool(news_path),
        "path": news_path,
        "relative_path": (
            str(Path(news_path).relative_to(PROJECT_ROOT))
            if news_path and Path(news_path).is_relative_to(PROJECT_ROOT)
            else news_path
        ),
    }
    daily = providers.get("refinitiv", {}).get("news_daily_counts", pd.DataFrame())
    if isinstance(daily, pd.DataFrame) and not daily.empty and "article_count" in daily:
        daily_nonzero = daily[daily["article_count"] > 0]
        fig = px.bar(
            daily_nonzero,
            x="date",
            y="article_count",
            title=f"{ticker} Refinitiv articles per day",
            color_discrete_sequence=["#dc4f52"],
            labels={"article_count": "Articles", "date": "Publication date"},
        )
        fig.update_traces(
            opacity=1,
            marker_line_color="#9f2528",
            marker_line_width=0.35,
            hovertemplate="%{x|%b %d, %Y}<br><b>%{y} articles</b><extra></extra>",
        )
        fig.update_layout(
            height=440,
            bargap=0.08,
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            yaxis={"rangemode": "tozero", "gridcolor": "#d8dee8"},
            xaxis={"gridcolor": "#eef1f5"},
        )
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
    headlines = refinitiv_headline_list(news, limit=DEFAULT_HEADLINE_LIMIT)
    if headlines is not None:
        headlines["ticker"] = ticker
        headlines["start_date"] = result["start_date"]
        headlines["end_date"] = result["end_date"]
    rp_list = ravenpack_article_list(articles, ticker=ticker, limit=DEFAULT_RP_LIST_LIMIT) if isinstance(articles, pd.DataFrame) else None
    if rp_list is not None:
        rp_list["start_date"] = result["start_date"]
        rp_list["end_date"] = result["end_date"]
    # Soft-match table is built on demand via HTMX (can be multi-second on dense tickers).
    soft_matches = None
    if isinstance(news, pd.DataFrame) and not news.empty and "headline" in news.columns:
        soft_matches = {
            "ticker": ticker,
            "start_date": result["start_date"],
            "end_date": result["end_date"],
            "rows": [],
            "total_refinitiv": int(len(news)),
            "candidates": 0,
            "matched": 0,
            "shown": 0,
            "limit": DEFAULT_SOFT_MATCH_LIMIT,
            "query": "",
            "min_score": DEFAULT_SOFT_MATCH_MIN_SCORE,
            "window_hours": DEFAULT_SOFT_MATCH_WINDOW_HOURS,
            "matched_only": False,
            "limit_choices": list(SOFT_MATCH_LIMIT_CHOICES),
            "note": (
                "Click Apply (or type a filter such as sony) to build the soft-match table. "
                "Deferred on Load data so dense tickers stay responsive."
            ),
        }
    return {
        "ticker": ticker, "start_date": result["start_date"], "end_date": result["end_date"],
        "source": result.get("source", "live"), "cache_created_at": result.get("cache_created_at"),
        "statuses": statuses, "charts": charts,
        "news": _records(news),
        "refinitiv_headlines": headlines,
        "news_storage": news_storage,
        "sentiment": sentiment_table,
        "ravenpack_articles": rp_list,
        "soft_matches": soft_matches,
        "raw": raw,
    }
