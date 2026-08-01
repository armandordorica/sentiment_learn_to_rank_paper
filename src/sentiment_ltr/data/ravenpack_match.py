"""Match a Refinitiv headline to RavenPack entity rows for the same ticker.

RavenPack carries `relevance` / `relevance_score` for the mapped entity; Refinitiv
wire stories do not. This module finds the best RavenPack row near a story's
timestamp so the Data Explorer can show both full story text and stock relevance.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import pandas as pd

_UPDATE_PREFIX = re.compile(
    r"^(?:update\s*\d+\s*[-–—]\s*|media\s*[-–—]\s*|press\s+digest\s*[-–—]\s*)",
    re.IGNORECASE,
)


def soft_headline(value: object) -> str:
    """Lowercased headline with common wire prefixes stripped."""
    text = str(value or "").strip()
    text = _UPDATE_PREFIX.sub("", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def headline_key(value: object) -> str:
    return hashlib.sha256(soft_headline(value).encode("utf-8")).hexdigest()


def token_set(value: object) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", soft_headline(value)))


def headline_match_score(left: object, right: object) -> float:
    """1.0 exact soft-key match; otherwise token Jaccard in [0, 1)."""
    if not soft_headline(left) or not soft_headline(right):
        return 0.0
    if headline_key(left) == headline_key(right):
        return 1.0
    a, b = token_set(left), token_set(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _ensure_relevance_score(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    if "relevance_score" not in work.columns and "relevance" in work.columns:
        work["relevance_score"] = pd.to_numeric(work["relevance"], errors="coerce") / 100.0
    elif "relevance_score" in work.columns:
        work["relevance_score"] = pd.to_numeric(work["relevance_score"], errors="coerce")
    return work


def _article_time(frame: pd.DataFrame) -> pd.Series:
    if "article_time" in frame.columns:
        return pd.to_datetime(frame["article_time"], utc=True, errors="coerce").dt.tz_localize(None)
    if "timestamp_utc" in frame.columns:
        return pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce").dt.tz_localize(None)
    return pd.Series(pd.NaT, index=frame.index)


def find_best_ravenpack_match(
    articles: pd.DataFrame,
    *,
    headline: str,
    story_time: pd.Timestamp | None,
    window_hours: float = 36.0,
    min_score: float = 0.45,
) -> dict[str, Any] | None:
    """Return the best RavenPack row for ``headline`` near ``story_time``.

    Requires a ``headline`` column on ``articles``. Returns ``None`` when no row
    clears ``min_score``.
    """
    if not isinstance(articles, pd.DataFrame) or articles.empty or "headline" not in articles.columns:
        return None
    work = _ensure_relevance_score(articles)
    work = work[work["headline"].notna() & (work["headline"].astype(str).str.strip() != "")].copy()
    if work.empty:
        return None

    times = _article_time(work)
    if story_time is not None and not pd.isna(story_time):
        center = pd.Timestamp(story_time).tz_localize(None) if getattr(story_time, "tzinfo", None) else pd.Timestamp(story_time)
        delta = pd.Timedelta(hours=window_hours)
        in_window = times.notna() & (times >= center - delta) & (times <= center + delta)
        candidates = work.loc[in_window] if in_window.any() else work
        cand_times = times.loc[candidates.index]
    else:
        candidates = work
        cand_times = times

    best_idx = None
    best_score = -1.0
    for idx, row in candidates.iterrows():
        score = headline_match_score(headline, row.get("headline"))
        if score > best_score:
            best_score = score
            best_idx = idx
    if best_idx is None or best_score < min_score:
        return None

    row = candidates.loc[best_idx]
    published = cand_times.loc[best_idx] if best_idx in cand_times.index else None
    event_text = row.get("event_text")
    event_text_s = None if event_text is None or str(event_text) in {"", "None", "nan"} else str(event_text)
    return {
        "matched": True,
        "match_score": round(float(best_score), 3),
        "headline": str(row.get("headline") or ""),
        "event_text": event_text_s,
        "relevance_score": _float_or_none(row.get("relevance_score")),
        "event_sentiment_score": _float_or_none(row.get("event_sentiment_score")),
        "sentiment_score": _float_or_none(row.get("sentiment_score")),
        "topic": _str_or_none(row.get("topic")),
        "group": _str_or_none(row.get("group")),
        "type": _str_or_none(row.get("type")),
        "rp_story_id": _str_or_none(row.get("rp_story_id")),
        "article_time": None if published is None or pd.isna(published) else str(published),
        "source": "headline_match",
    }


def nearby_ravenpack_summary(
    articles: pd.DataFrame,
    *,
    story_time: pd.Timestamp | None,
    window_hours: float = 36.0,
    limit: int = 5,
) -> dict[str, Any] | None:
    """Fallback when RavenPack rows lack headlines (batch cache).

    Surfaces the highest-relevance entity events near the story time so the UI
    can still show stock relevance context.
    """
    if not isinstance(articles, pd.DataFrame) or articles.empty or story_time is None or pd.isna(story_time):
        return None
    work = _ensure_relevance_score(articles)
    if "relevance_score" not in work.columns:
        return None
    times = _article_time(work)
    center = pd.Timestamp(story_time).tz_localize(None) if getattr(pd.Timestamp(story_time), "tzinfo", None) else pd.Timestamp(story_time)
    delta = pd.Timedelta(hours=window_hours)
    mask = times.notna() & (times >= center - delta) & (times <= center + delta)
    window = work.loc[mask].copy()
    if window.empty:
        return None
    window = window.sort_values("relevance_score", ascending=False).head(limit)
    times_w = _article_time(window)
    rows = []
    for idx, row in window.iterrows():
        t = times_w.loc[idx]
        rows.append({
            "relevance_score": _float_or_none(row.get("relevance_score")),
            "event_sentiment_score": _float_or_none(row.get("event_sentiment_score")),
            "sentiment_score": _float_or_none(row.get("sentiment_score")),
            "topic": _str_or_none(row.get("topic")),
            "group": _str_or_none(row.get("group")),
            "type": _str_or_none(row.get("type")),
            "article_time": None if pd.isna(t) else str(t),
            "headline": _str_or_none(row.get("headline")),
        })
    top = rows[0] if rows else None
    return {
        "matched": False,
        "match_score": None,
        "source": "nearby_events",
        "note": (
            "No RavenPack headline match in this window. Showing the highest-relevance "
            "RavenPack events for this ticker near the story time "
            "(batch cache often omits headlines)."
        ),
        "relevance_score": top.get("relevance_score") if top else None,
        "event_sentiment_score": top.get("event_sentiment_score") if top else None,
        "sentiment_score": top.get("sentiment_score") if top else None,
        "topic": top.get("topic") if top else None,
        "group": top.get("group") if top else None,
        "type": top.get("type") if top else None,
        "headline": top.get("headline") if top else None,
        "event_text": None,
        "rp_story_id": None,
        "article_time": top.get("article_time") if top else None,
        "nearby": rows,
    }


def load_ravenpack_day_with_text(
    ticker: str,
    day: str,
    *,
    cache_dir: Path,
    query_fn,
) -> pd.DataFrame:
    """Load one calendar day of RavenPack rows including headline/event_text.

    Caches under ``cache_dir / TICKER / YYYY-MM-DD.parquet``. ``query_fn`` should
    wrap ``live_data.query_ravenpack_articles(..., include_text=True)``.
    """
    clean = re.sub(r"[^A-Za-z0-9._-]+", "", str(ticker or "").upper()) or "UNKNOWN"
    day_ts = pd.Timestamp(day).strftime("%Y-%m-%d")
    path = cache_dir / clean / f"{day_ts}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            pass
    start = day_ts
    end = day_ts
    frame = query_fn(clean, start, end)
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            frame.to_parquet(path, index=False)
        except Exception:
            pass
        return frame
    return frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()


def resolve_ravenpack_for_story(
    *,
    ticker: str,
    headline: str,
    story_time: str | pd.Timestamp | None,
    cached_articles: pd.DataFrame | None = None,
    day_cache_dir: Path | None = None,
    query_day_fn=None,
) -> dict[str, Any]:
    """Best-effort RavenPack relevance context for a Refinitiv story click."""
    center = pd.to_datetime(story_time, utc=True, errors="coerce")
    if pd.isna(center):
        center = None
    else:
        center = center.tz_localize(None) if getattr(center, "tzinfo", None) else center

    # 1) Prefer a text-bearing day pull (live or day-cache).
    if center is not None and day_cache_dir is not None and query_day_fn is not None:
        try:
            day_frame = load_ravenpack_day_with_text(
                ticker,
                center.strftime("%Y-%m-%d"),
                cache_dir=day_cache_dir,
                query_fn=query_day_fn,
            )
            hit = find_best_ravenpack_match(
                day_frame, headline=headline, story_time=center
            )
            if hit:
                hit["source"] = "day_headline_match"
                return hit
            # Also try ±1 calendar day if the story is near midnight.
            for offset in (-1, 1):
                other = (center + pd.Timedelta(days=offset)).strftime("%Y-%m-%d")
                other_frame = load_ravenpack_day_with_text(
                    ticker, other, cache_dir=day_cache_dir, query_fn=query_day_fn
                )
                hit = find_best_ravenpack_match(
                    other_frame, headline=headline, story_time=center
                )
                if hit:
                    hit["source"] = "day_headline_match"
                    return hit
        except Exception as exc:  # noqa: BLE001 — UI must still show the story
            day_error = str(exc)[:200]
        else:
            day_error = None
    else:
        day_error = None

    # 2) Try headline match against whatever the explorer already loaded.
    if isinstance(cached_articles, pd.DataFrame) and not cached_articles.empty:
        hit = find_best_ravenpack_match(
            cached_articles, headline=headline, story_time=center
        )
        if hit:
            return hit
        nearby = nearby_ravenpack_summary(cached_articles, story_time=center)
        if nearby:
            if day_error:
                nearby["note"] = f"{nearby['note']} Day text pull failed: {day_error}"
            return nearby

    return {
        "matched": False,
        "match_score": None,
        "source": "none",
        "note": (
            day_error
            or "No RavenPack relevance found near this story. "
            "Enable RavenPack for the ticker load, or check WRDS credentials for a day text pull."
        ),
        "relevance_score": None,
        "event_sentiment_score": None,
        "sentiment_score": None,
        "headline": None,
        "event_text": None,
        "topic": None,
        "group": None,
        "type": None,
        "rp_story_id": None,
        "article_time": None,
        "nearby": [],
    }


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return None if text in {"", "None", "nan"} else text
