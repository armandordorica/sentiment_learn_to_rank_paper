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
from sentiment_ltr.data.refinitiv_story_cache import (  # noqa: E402
    digests_on_disk,
    read_progress as read_story_pull_progress,
    story_path as cached_story_path,
    write_story_file,
)
from webapp.api import data_explorer_inventory as inventory  # noqa: E402


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
        "pull_products": inventory.PULL_PRODUCTS,
        "default_selected": inventory.DEFAULT_SELECTED,
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
    return cached_story_path(PROJECT_ROOT, ticker, story_id, headline)


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
    only_event_text: bool = False,
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
    with_event_text_total = 0
    if "event_text" in work.columns:
        et_all = _nonempty_mask(work["event_text"])
        with_event_text_total = int(et_all.sum())

    if only_event_text and "event_text" in work.columns:
        work = work.loc[_nonempty_mask(work["event_text"])]
    elif only_event_text:
        work = work.iloc[0:0].copy()

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
        "with_event_text": with_event_text_total,
        "matched": matched,
        "shown": int(len(display)),
        "limit": limit,
        "query": str(query or "").strip(),
        "sort": sort,
        "only_event_text": bool(only_event_text),
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
    only_event_text: bool = False,
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
        limit=limit, query=query, sort=sort, only_event_text=only_event_text,
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
    force_fetch: bool = False,
) -> dict[str, Any]:
    """Fetch and persist one Refinitiv story, with RavenPack relevance when possible."""
    story_id = str(story_id or "").strip()
    if not story_id:
        raise ValueError("Select a Refinitiv headline with a story ID.")
    headline = str(headline or "Selected headline")
    clean_ticker = live_data.clean_ticker(ticker) or "UNKNOWN"
    path = _story_path(story_id, headline, clean_ticker)
    if path.exists() and not force_fetch:
        raw = path.read_text(encoding="utf-8")
        marker = "\n\n"
        text = raw.split(marker, 1)[1].rstrip("\n") if marker in raw else raw
    else:
        text = fetch_refinitiv_story(PROJECT_ROOT, story_id)
        write_story_file(
            path,
            story_id=story_id,
            headline=headline,
            ticker=clean_ticker,
            text=text,
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



def _nonempty_mask(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    return series.notna() & text.ne("") & ~text.isin({"None", "nan", "NaT", "<NA>"})


def _field_row(frame: pd.DataFrame, column: str, *, note: str = "") -> dict[str, Any] | None:
    if column not in frame.columns:
        return None
    total = int(len(frame))
    filled = int(_nonempty_mask(frame[column]).sum())
    return {
        "field": column,
        "filled": filled,
        "missing": total - filled,
        "pct": round(100.0 * filled / total, 1) if total else 0.0,
        "note": note,
    }


def _refinitiv_story_body_coverage(news: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """How many headline rows have a full body saved under FULL_STORY_DIR."""
    total = int(len(news))
    # Prefer the module-level FULL_STORY_DIR so tests can monkeypatch it.
    story_dir = FULL_STORY_DIR / (live_data.clean_ticker(ticker) or ticker.upper())
    digests: set[str] = set()
    if story_dir.is_dir():
        for path in story_dir.glob("*.txt"):
            stem = path.stem
            digests.add(stem.rsplit("--", 1)[-1] if "--" in stem else stem)
    with_body = 0
    if "storyId" in news.columns and digests:
        for story_id in news["storyId"].astype(str):
            digest = hashlib.sha256(story_id.encode("utf-8")).hexdigest()[:12]
            if digest in digests:
                with_body += 1
    pull = read_story_pull_progress(PROJECT_ROOT, ticker) or {}
    note = (
        f"On-demand wire body · {len(digests)} file(s) on disk"
        + (f" · last pull {pull.get('status')} ({pull.get('fetched', 0)} fetched)" if pull else "")
    )
    return {
        "field": "full_story_body (on disk)",
        "filled": with_body,
        "missing": total - with_body,
        "pct": round(100.0 * with_body / total, 1) if total else 0.0,
        "note": note,
        "files_on_disk": len(digests),
        "pull_progress": pull or None,
    }


def field_coverage(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news: pd.DataFrame,
    articles: pd.DataFrame,
    price_blocks: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    """Per-field coverage for the selected ticker and date window."""
    ref_fields: list[dict[str, Any]] = []
    if isinstance(news, pd.DataFrame) and not news.empty:
        for col, note in (
            ("date", "Headline timestamp"),
            ("headline", "Wire headline text"),
            ("storyId", "Needed to fetch full body"),
            ("sourceCode", "e.g. NS:RTRS"),
        ):
            row = _field_row(news, col, note=note)
            if row:
                ref_fields.append(row)
        ref_fields.append(_refinitiv_story_body_coverage(news, ticker))

    rp_fields: list[dict[str, Any]] = []
    if isinstance(articles, pd.DataFrame) and not articles.empty:
        work = articles.copy()
        if "relevance_score" not in work.columns and "relevance" in work.columns:
            work["relevance_score"] = pd.to_numeric(work["relevance"], errors="coerce") / 100.0
        for col, note in (
            ("headline", "Always present in rich exports"),
            ("event_text", "Short ≤400-char snippet when RavenPack tagged an event"),
            ("relevance_score", "Entity relevance to this ticker"),
            ("event_sentiment_score", "Only when an event was tagged"),
            ("sentiment_score", "relevance × event sentiment"),
            ("topic", "Event taxonomy"),
            ("group", "Event taxonomy"),
            ("type", "Event taxonomy"),
            ("news_type", "Source item class (not a body)"),
            ("rp_story_id", "Story key"),
        ):
            row = _field_row(work, col, note=note)
            if row:
                rp_fields.append(row)

    prices = []
    for name, frame in price_blocks.items():
        n = int(len(frame)) if isinstance(frame, pd.DataFrame) else 0
        prices.append({"provider": name, "rows": n})

    return {
        "ticker": ticker,
        "start_date": start_date,
        "end_date": end_date,
        "refinitiv": {
            "rows": int(len(news)) if isinstance(news, pd.DataFrame) else 0,
            "fields": ref_fields,
        },
        "ravenpack": {
            "rows": int(len(articles)) if isinstance(articles, pd.DataFrame) else 0,
            "fields": rp_fields,
        },
        "prices": prices,
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
    all_price_blocks = {
        name: block.get("prices", pd.DataFrame())
        for name, block in providers.items()
        if name != "ravenpack"
    }
    coverage = field_coverage(
        ticker=ticker,
        start_date=result["start_date"],
        end_date=result["end_date"],
        news=news if isinstance(news, pd.DataFrame) else pd.DataFrame(),
        articles=articles if isinstance(articles, pd.DataFrame) else pd.DataFrame(),
        price_blocks=all_price_blocks,
    )

    return {
        "ticker": ticker, "start_date": result["start_date"], "end_date": result["end_date"],
        "source": result.get("source", "live"), "cache_created_at": result.get("cache_created_at"),
        "statuses": statuses, "charts": charts,
        "coverage": coverage,
        "news": _records(news),
        "refinitiv_headlines": headlines,
        "news_storage": news_storage,
        "sentiment": sentiment_table,
        "ravenpack_articles": rp_list,
        "soft_matches": soft_matches,
        "raw": raw,
    }



def build_inventory(ticker: str, start: str, end: str) -> dict[str, Any]:
    """Fast local inventory for the selective Load form."""
    return inventory.inspect_inventory(
        ticker=ticker,
        start=start,
        end=end,
        cache_dir=_cache_dir(ticker),
        project_root=PROJECT_ROOT,
        load_cached_ravenpack=load_cached_ravenpack,
        full_story_dir=FULL_STORY_DIR,
    )


def selective_load(
    ticker: str,
    start: str,
    end: str,
    *,
    selected_ids: list[str],
    action: str = "load",
) -> dict[str, Any]:
    """Load selected products; skip live pulls when cache already covers them.

    ``action``:
      - ``check``: inventory only (no present/charts)
      - ``load``: use cache; live-pull only missing tabular products; queue overnight stories
      - ``live``: force live for selected tabular products; still skip existing story bodies
    """
    ticker = live_data.clean_ticker(ticker)
    if not ticker:
        raise ValueError("Enter a valid ticker.")
    if pd.Timestamp(start) > pd.Timestamp(end):
        raise ValueError("Start date must be on or before end date.")
    selected_ids = [str(x) for x in selected_ids if str(x).strip()]
    if not selected_ids:
        raise ValueError("Select at least one service / field bundle.")

    inv = build_inventory(ticker, start, end)
    by_id = {p["id"]: p for p in inv["products"]}
    flags = inventory.selected_provider_flags(selected_ids)
    messages: list[dict[str, str]] = []
    force_live = action == "live"

    # Overnight full-story handling
    story_job = None
    if flags["full_stories"]:
        story = by_id.get("refinitiv_full_stories", {})
        status = story.get("status")
        if status == "ready" and not force_live:
            messages.append({"level": "success", "text": story["message"]})
        elif status == "blocked":
            messages.append({"level": "error", "text": story["message"]})
        else:
            if action == "check":
                messages.append({"level": "info", "text": story.get("message") or "Full stories not fully cached."})
            else:
                # Only queue missing bodies (script skips existing digests).
                story_job = start_full_story_cache_job(ticker, start, end)
                messages.append({
                    "level": "info",
                    "text": (
                        f"Started overnight Refinitiv full-story pull for {ticker} "
                        f"{start} → {end} (pid {story_job['pid']}). "
                        f"{story.get('detail') or story.get('message') or ''} "
                        "Existing bodies are skipped."
                    ).strip(),
                })

    # Tabular products
    for pid in selected_ids:
        if pid == "refinitiv_full_stories":
            continue
        prod = by_id.get(pid)
        if not prod:
            continue
        if prod["status"] == "ready" and not force_live:
            messages.append({"level": "success", "text": prod["message"]})
        elif action == "check":
            level = "success" if prod["status"] == "ready" else ("warn" if prod["status"] == "partial" else "info")
            messages.append({"level": level, "text": prod["message"]})
        elif prod["status"] != "ready" or force_live:
            messages.append({
                "level": "info",
                "text": (
                    f"Will live-pull {prod['service']} {prod['label'].lower()} for "
                    f"{ticker} {start} → {end}."
                    if force_live or prod["status"] == "missing"
                    else prod["message"] + " Live pull may enrich missing fields."
                ),
            })

    if action == "check":
        return {
            "mode": "check",
            "inventory": inv,
            "messages": messages,
            "story_job": story_job,
            "result": None,
        }

    # Decide live providers
    need_refinitiv = False
    need_news = False
    need_wrds = False
    need_yahoo = False
    need_ravenpack = False
    for pid in selected_ids:
        if pid == "refinitiv_full_stories":
            continue
        prod = by_id.get(pid, {})
        stale = force_live or prod.get("status") != "ready"
        if not stale:
            continue
        if pid == "refinitiv_prices":
            need_refinitiv = True
        elif pid == "refinitiv_headlines":
            need_refinitiv = True
            need_news = True
        elif pid == "wrds_prices":
            need_wrds = True
        elif pid == "yahoo_prices":
            need_yahoo = True
        elif pid == "ravenpack_articles":
            need_ravenpack = True

    raw = None
    if any((need_refinitiv, need_wrds, need_yahoo, need_ravenpack)):
        raw = live_data.run_ticker_data_query(
            PROJECT_ROOT, ticker, start, end,
            query_refinitiv=need_refinitiv,
            query_wrds=need_wrds,
            query_yahoo=need_yahoo,
            query_ravenpack=need_ravenpack,
            news_count=1 if need_news else 0,
            wrds_limit=10_000,
        )
        raw["source"] = "live"
        messages.append({
            "level": "info",
            "text": (
                f"Live pull finished for {ticker} {start} → {end} "
                f"(providers: "
                + ", ".join(
                    name for name, flag in [
                        ("refinitiv", need_refinitiv),
                        ("wrds", need_wrds),
                        ("yahoo", need_yahoo),
                        ("ravenpack", need_ravenpack),
                    ] if flag
                )
                + ")."
            ),
        })
    else:
        raw = load_cached(ticker, start, end)
        if raw is None:
            # Still allow presentation shell when only overnight stories selected.
            raw = {
                "ticker": ticker,
                "start_date": start,
                "end_date": end,
                "source": "cache",
                "providers": {
                    "refinitiv": {"status": "empty", "error": None, "prices": pd.DataFrame(), "news": pd.DataFrame(), "news_daily_counts": pd.DataFrame()},
                    "wrds": {"status": "empty", "error": None, "prices": pd.DataFrame()},
                    "yahoo": {"status": "empty", "error": None, "prices": pd.DataFrame()},
                    "ravenpack": {"status": "empty", "error": None, "articles": pd.DataFrame()},
                },
            }
        else:
            messages.append({
                "level": "success",
                "text": (
                    f"Loaded {ticker} {start} → {end} from local cache "
                    "(no live API calls for the selected tabular fields)."
                ),
            })

    # Prefer full cache merge so unselected-but-present series still chart if loaded
    cached = load_cached(ticker, start, end)
    if cached is not None and raw.get("source") == "live":
        # Merge: keep live for pulled providers, cache for others
        for name, block in cached["providers"].items():
            live_block = raw["providers"].get(name, {})
            live_status = live_block.get("status")
            if live_status in {None, "skipped", "empty", "unavailable", "failed"}:
                raw["providers"][name] = block
            elif name == "refinitiv":
                # Keep live prices/news when present; fill missing news from cache
                if isinstance(live_block.get("news"), pd.DataFrame) and live_block["news"].empty:
                    live_block["news"] = block.get("news", pd.DataFrame())
                    live_block["news_daily_counts"] = block.get("news_daily_counts", pd.DataFrame())
                if isinstance(live_block.get("prices"), pd.DataFrame) and live_block["prices"].empty:
                    live_block["prices"] = block.get("prices", pd.DataFrame())
                raw["providers"][name] = live_block

    presented = present(raw)
    presented["pull_report"] = {
        "action": action,
        "messages": messages,
        "inventory": build_inventory(ticker, start, end),
        "selected_ids": selected_ids,
        "story_job": story_job,
    }
    return {
        "mode": action,
        "inventory": presented["pull_report"]["inventory"],
        "messages": messages,
        "story_job": story_job,
        "result": presented,
    }


def start_full_story_cache_job(
    ticker: str,
    start: str,
    end: str,
    *,
    limit: int | None = None,
    sleep_s: float = 0.25,
) -> dict[str, Any]:
    """Launch ``scripts/cache_refinitiv_full_stories.py`` as a detached process."""
    import subprocess

    ticker = live_data.clean_ticker(ticker) or ticker.upper()
    script = PROJECT_ROOT / "scripts" / "cache_refinitiv_full_stories.py"
    cmd = [
        sys.executable,
        str(script),
        "--ticker", ticker,
        "--start", start,
        "--end", end,
        "--sleep", str(sleep_s),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(int(limit))])
    log_path = FULL_STORY_DIR / ticker / "_pull.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return {
        "ticker": ticker,
        "pid": proc.pid,
        "cmd": cmd,
        "log_path": str(log_path),
        "progress_path": str(FULL_STORY_DIR / ticker / "_pull_progress.json"),
    }


def full_story_cache_status(ticker: str) -> dict[str, Any]:
    ticker = live_data.clean_ticker(ticker) or ticker.upper()
    progress = read_story_pull_progress(PROJECT_ROOT, ticker) or {}
    n_files = len(digests_on_disk(PROJECT_ROOT, ticker))
    return {
        "ticker": ticker,
        "files_on_disk": n_files,
        "progress": progress,
        "log_path": str(FULL_STORY_DIR / ticker / "_pull.log"),
    }
