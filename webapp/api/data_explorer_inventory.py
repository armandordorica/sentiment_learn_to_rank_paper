"""Selective cache inventory for Data Explorer one-stock pulls.

Reports which service/field bundles already exist for a ticker + window so Load
can skip live calls and overnight jobs when nothing is missing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_ltr.data import live_data
from sentiment_ltr.data.refinitiv_story_cache import digests_on_disk

# Catalog of pullable products. ``provider`` maps to live_data / cache keys.
PULL_PRODUCTS: list[dict[str, Any]] = [
    {
        "id": "refinitiv_prices",
        "service": "Refinitiv",
        "label": "Prices",
        "fields": ["date", "close_price", "volume"],
        "provider": "refinitiv",
        "kind": "prices",
        "cache_file": "refinitiv_prices.parquet",
        "time_col": "date",
    },
    {
        "id": "refinitiv_headlines",
        "service": "Refinitiv",
        "label": "Headlines",
        "fields": ["date", "headline", "storyId", "sourceCode"],
        "provider": "refinitiv",
        "kind": "news",
        "cache_file": "refinitiv_news.parquet",
        "time_col": "date",
    },
    {
        "id": "refinitiv_full_stories",
        "service": "Refinitiv",
        "label": "Full story bodies",
        "fields": ["full_story_body"],
        "provider": "refinitiv",
        "kind": "story_bodies",
        "cache_file": "refinitiv_news.parquet",
        "time_col": "date",
        "overnight": True,
    },
    {
        "id": "wrds_prices",
        "service": "WRDS/CRSP",
        "label": "Prices",
        "fields": ["date", "close_price", "vol", "permno"],
        "provider": "wrds",
        "kind": "prices",
        "cache_file": "wrds_prices.parquet",
        "time_col": "date",
    },
    {
        "id": "yahoo_prices",
        "service": "Yahoo",
        "label": "Prices",
        "fields": ["date", "close_price", "volume"],
        "provider": "yahoo",
        "kind": "prices",
        "cache_file": "yahoo_prices.parquet",
        "time_col": "date",
    },
    {
        "id": "ravenpack_articles",
        "service": "RavenPack",
        "label": "Entity articles",
        "fields": [
            "headline",
            "event_text",
            "relevance_score",
            "event_sentiment_score",
            "topic",
            "news_type",
            "rp_story_id",
        ],
        "provider": "ravenpack",
        "kind": "articles",
        "cache_file": "ravenpack_articles.parquet",
        "time_col": "article_time",
        "alt_time_col": "timestamp_utc",
        "rich_export": True,
    },
]

DEFAULT_SELECTED = [
    "refinitiv_prices",
    "refinitiv_headlines",
    "wrds_prices",
    "yahoo_prices",
    "ravenpack_articles",
]


def _filter_window(df: pd.DataFrame, time_col: str, start: str, end: str) -> pd.DataFrame:
    if df.empty or time_col not in df.columns:
        return df.iloc[0:0].copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    dates = pd.to_datetime(df[time_col], utc=True, errors="coerce").dt.tz_localize(None)
    return df[(dates >= pd.Timestamp(start)) & (dates < pd.Timestamp(end) + pd.Timedelta(days=1))].copy()


def _read_parquet(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _present_fields(frame: pd.DataFrame, fields: list[str]) -> list[str]:
    present = []
    for field in fields:
        if field == "full_story_body":
            continue
        if field == "relevance_score" and field not in frame.columns and "relevance" in frame.columns:
            present.append(field)
            continue
        if field == "rp_story_id" and field not in frame.columns and "story_id" in frame.columns:
            present.append(field)
            continue
        if field in frame.columns:
            present.append(field)
    return present


def inspect_inventory(
    *,
    ticker: str,
    start: str,
    end: str,
    cache_dir: Path | None,
    project_root: Path,
    load_cached_ravenpack,
    full_story_dir: Path,
) -> dict[str, Any]:
    """Return per-product availability for ``ticker`` in ``[start, end]``."""
    ticker = live_data.clean_ticker(ticker) or ticker.upper()
    products: list[dict[str, Any]] = []

    for spec in PULL_PRODUCTS:
        item: dict[str, Any] = {
            "id": spec["id"],
            "service": spec["service"],
            "label": spec["label"],
            "fields": list(spec["fields"]),
            "overnight": bool(spec.get("overnight")),
            "status": "missing",
            "rows": 0,
            "fields_present": [],
            "fields_missing": list(spec["fields"]),
            "message": "",
            "detail": "",
        }

        if spec["kind"] == "story_bodies":
            news = pd.DataFrame()
            if cache_dir is not None:
                news = _filter_window(
                    _read_parquet(cache_dir / "refinitiv_news.parquet"),
                    "date",
                    start,
                    end,
                )
            n_headlines = int(len(news))
            if n_headlines == 0:
                item["status"] = "blocked"
                item["message"] = (
                    f"{ticker} Refinitiv full story bodies need headlines first for "
                    f"{start} → {end} (none in cache)."
                )
            else:
                digests = digests_on_disk(project_root, ticker)
                have = 0
                if "storyId" in news.columns and digests:
                    for story_id in news["storyId"].astype(str):
                        digest = hashlib.sha256(story_id.encode("utf-8")).hexdigest()[:12]
                        if digest in digests:
                            have += 1
                missing = n_headlines - have
                item["rows"] = have
                item["fields_present"] = ["full_story_body"] if have else []
                item["fields_missing"] = [] if missing == 0 else ["full_story_body"]
                item["detail"] = f"{have:,}/{n_headlines:,} bodies on disk"
                if missing == 0:
                    item["status"] = "ready"
                    item["message"] = (
                        f"{ticker} Refinitiv full story bodies already exist for "
                        f"{start} → {end} ({have:,}/{n_headlines:,} cached)."
                    )
                elif have == 0:
                    item["status"] = "missing"
                    item["message"] = (
                        f"{ticker} Refinitiv full story bodies not cached for "
                        f"{start} → {end} (0/{n_headlines:,}); overnight pull required."
                    )
                else:
                    item["status"] = "partial"
                    item["message"] = (
                        f"{ticker} Refinitiv full story bodies partially cached for "
                        f"{start} → {end} ({have:,}/{n_headlines:,}); "
                        f"{missing:,} still to fetch overnight."
                    )
            products.append(item)
            continue

        frame = pd.DataFrame()
        if spec.get("rich_export"):
            frame = load_cached_ravenpack(ticker)
            time_col = "article_time" if "article_time" in frame.columns else spec.get("alt_time_col", "timestamp_utc")
            if not frame.empty and time_col in frame.columns:
                frame = _filter_window(frame, time_col, start, end)
        elif cache_dir is not None:
            frame = _read_parquet(cache_dir / spec["cache_file"])
            time_col = spec["time_col"]
            if spec.get("alt_time_col") and time_col not in frame.columns:
                time_col = spec["alt_time_col"]
            frame = _filter_window(frame, time_col, start, end)

        rows = int(len(frame))
        present = _present_fields(frame, spec["fields"])
        missing_fields = [f for f in spec["fields"] if f not in present]
        item["rows"] = rows
        item["fields_present"] = present
        item["fields_missing"] = missing_fields

        if rows == 0:
            item["status"] = "missing"
            item["message"] = (
                f"{ticker} {spec['service']} {spec['label'].lower()} not in local cache "
                f"for {start} → {end}."
            )
        elif missing_fields:
            item["status"] = "partial"
            item["message"] = (
                f"{ticker} {spec['service']} {spec['label'].lower()} partially available "
                f"for {start} → {end} ({rows:,} rows); "
                f"have {', '.join(present) or '—'}; missing {', '.join(missing_fields)}."
            )
        else:
            item["status"] = "ready"
            item["message"] = (
                f"{ticker} {spec['service']} fields {', '.join(present)} already exist "
                f"for {start} → {end} ({rows:,} rows)."
            )
        products.append(item)

    ready = sum(1 for p in products if p["status"] == "ready")
    return {
        "ticker": ticker,
        "start_date": start,
        "end_date": end,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "products": products,
        "ready_count": ready,
        "total_count": len(products),
        "summary": (
            f"{ticker} · {start} → {end}: {ready}/{len(products)} pull products fully cached."
        ),
    }


def selected_provider_flags(selected_ids: list[str]) -> dict[str, bool]:
    """Map selected product ids to live query provider booleans."""
    selected = set(selected_ids)
    return {
        "refinitiv": bool(selected & {"refinitiv_prices", "refinitiv_headlines"}),
        "include_news": "refinitiv_headlines" in selected,
        "wrds": "wrds_prices" in selected,
        "yahoo": "yahoo_prices" in selected,
        "ravenpack": "ravenpack_articles" in selected,
        "full_stories": "refinitiv_full_stories" in selected,
    }
