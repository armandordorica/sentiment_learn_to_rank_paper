"""Refinitiv news coverage counting helpers for universe validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_ltr.data.refinitiv_queries import ticker_to_ric_candidates
from sentiment_ltr.data.refinitiv_session import open_refinitiv_session


@dataclass(frozen=True)
class NewsCoverageSummary:
    """Summary statistics for one stock over a date range."""

    ticker: str
    ric: str
    start_date: str
    end_date: str
    total_articles: int
    trading_days_with_news: int
    calendar_days_with_news: int
    calendar_days_in_range: int
    avg_articles_per_calendar_day: float
    avg_articles_per_week: float
    weeks_in_range: int
    weeks_with_zero_articles: int
    passes_paper_weekly_threshold: bool


def build_news_query(ric: str, *, language: str = "LEN") -> str:
    """Build a Refinitiv headline query for one RIC."""
    return f"R:{ric} AND Language:{language}"


def _headlines_to_frame(headlines: Any) -> pd.DataFrame:
    """Convert a Refinitiv headlines response to a normalized dataframe."""
    if headlines is None:
        return pd.DataFrame()

    data = headlines.copy() if isinstance(headlines, pd.DataFrame) else headlines
    if not isinstance(data, pd.DataFrame):
        return pd.DataFrame()

    if isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()
        if "index" in data.columns and "versionCreated" not in data.columns:
            data = data.rename(columns={"index": "versionCreated"})

    if "versionCreated" in data.columns:
        data["article_time"] = pd.to_datetime(data["versionCreated"], utc=True, errors="coerce", format="mixed")
    else:
        for column in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data = data.rename(columns={column: "article_time"})
                data["article_time"] = pd.to_datetime(data["article_time"], utc=True, errors="coerce", format="mixed")
                break

    if "storyId" not in data.columns:
        for column in data.columns:
            if str(column).lower() == "storyid":
                data = data.rename(columns={column: "storyId"})
                break

    keep_cols = [col for col in ["article_time", "headline", "storyId", "sourceCode"] if col in data.columns]
    if not keep_cols or "article_time" not in keep_cols:
        return pd.DataFrame()

    result = data[keep_cols].dropna(subset=["article_time"]).copy()
    result["article_date"] = result["article_time"].dt.tz_convert(None).dt.normalize()
    return result.sort_values("article_time")


def month_chunks(start_date: str, end_date: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a closed date range into calendar-month chunks."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start.replace(day=1)
    while cursor <= end:
        month_end = (cursor + pd.offsets.MonthEnd(0)).normalize()
        chunk_start = max(cursor, start)
        chunk_end = min(month_end, end)
        chunks.append((chunk_start, chunk_end))
        cursor = (cursor + pd.offsets.MonthBegin(1)).normalize()
    return chunks


def fetch_headlines_for_window(
    ld_module: Any,
    query: str,
    start_date: str,
    end_date: str,
    *,
    max_headlines: int = 10_000,
) -> pd.DataFrame:
    """Fetch headlines for one query and date window, following API pagination."""
    start_ts = pd.Timestamp(start_date).to_pydatetime()
    end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).to_pydatetime()
    headlines = ld_module.news.get_headlines(
        query,
        start=start_ts,
        end=end_exclusive,
        count=int(max_headlines),
    )
    return _headlines_to_frame(headlines)


def fetch_ticker_headlines(
    project_root: Path,
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    max_headlines_per_chunk: int = 10_000,
    ld_module: Any | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch deduplicated headlines for a ticker across a long date range."""
    ld = ld_module
    opened_here = False
    if ld is None:
        import lseg.data as ld  # type: ignore

        open_refinitiv_session(project_root, ld)
        opened_here = True

    errors: list[str] = []
    frames: list[pd.DataFrame] = []
    selected_ric: str | None = None

    try:
        for ric in ticker_to_ric_candidates(ticker):
            query = build_news_query(ric)
            chunk_frames: list[pd.DataFrame] = []
            try:
                for chunk_start, chunk_end in month_chunks(start_date, end_date):
                    frame = fetch_headlines_for_window(
                        ld,
                        query,
                        chunk_start.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d"),
                        max_headlines=max_headlines_per_chunk,
                    )
                    if not frame.empty:
                        chunk_frames.append(frame)
            except Exception as exc:
                errors.append(f"{ric}: {exc}")
                continue

            if not chunk_frames:
                errors.append(f"{ric}: no headlines returned")
                continue

            combined = pd.concat(chunk_frames, ignore_index=True)
            if "storyId" in combined.columns:
                combined = combined.drop_duplicates(subset=["storyId"], keep="first")
            else:
                combined = combined.drop_duplicates(
                    subset=["article_time", "headline"],
                    keep="first",
                )

            if combined.empty:
                errors.append(f"{ric}: headlines were empty after deduplication")
                continue

            frames = [combined]
            selected_ric = ric
            break
    finally:
        if opened_here:
            ld.close_session()

    if not frames or selected_ric is None:
        detail = errors[-1] if errors else "No Refinitiv headlines returned."
        raise ValueError(detail)

    return frames[0].sort_values("article_time"), selected_ric


def headlines_to_app_frame(headlines: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized headlines to the Streamlit app's headline schema."""
    if headlines.empty:
        return pd.DataFrame(columns=["date", "headline", "storyId", "sourceCode"])

    result = headlines.copy()
    result["date"] = result["article_time"]
    keep_cols = [col for col in ["date", "headline", "storyId", "sourceCode"] if col in result.columns]
    return result[keep_cols].sort_values("date", ascending=False)


def filter_headlines_by_date(headlines: pd.DataFrame, selected_date: pd.Timestamp) -> pd.DataFrame:
    """Return headline rows whose calendar date matches the selected day."""
    if headlines.empty or "date" not in headlines.columns:
        return pd.DataFrame(columns=["date", "headline", "storyId", "sourceCode"])

    target = pd.Timestamp(selected_date).normalize()
    dates = pd.to_datetime(headlines["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    return headlines.loc[dates == target].copy()


def build_news_coverage_result(
    project_root: Path,
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    max_headlines_per_chunk: int = 10_000,
    ld_module: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, NewsCoverageSummary, str]:
    """Fetch headlines and return app-ready daily counts plus summary metrics."""
    headlines, ric = fetch_ticker_headlines(
        project_root,
        ticker,
        start_date,
        end_date,
        max_headlines_per_chunk=max_headlines_per_chunk,
        ld_module=ld_module,
    )
    daily = daily_article_counts(headlines, start_date, end_date)
    summary = summarize_news_coverage(ticker, ric, headlines, start_date, end_date)
    app_news = headlines_to_app_frame(headlines)
    return app_news, daily, summary, ric


def daily_article_counts(headlines: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Aggregate headline rows to one count per calendar day."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    full_index = pd.date_range(start, end, freq="D")

    if headlines.empty:
        daily = pd.DataFrame({"date": full_index, "article_count": 0})
        return daily

    grouped = (
        headlines.groupby("article_date", as_index=False)
        .size()
        .rename(columns={"size": "article_count", "article_date": "date"})
    )
    daily = (
        grouped.set_index("date")
        .reindex(full_index, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )
    daily["article_count"] = daily["article_count"].astype(int)
    return daily


def weekly_article_counts(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily counts to ISO calendar weeks ending on Sunday."""
    weekly_input = daily.copy()
    weekly_input["week_start"] = weekly_input["date"].dt.to_period("W-SUN").apply(lambda period: period.start_time)
    weekly = (
        weekly_input.groupby("week_start", as_index=False)["article_count"]
        .sum()
        .rename(columns={"article_count": "article_count"})
        .sort_values("week_start")
    )
    return weekly


def summarize_news_coverage(
    ticker: str,
    ric: str,
    headlines: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> NewsCoverageSummary:
    """Compute daily/weekly coverage metrics used by the paper's news filter."""
    daily = daily_article_counts(headlines, start_date, end_date)
    weekly = weekly_article_counts(daily)

    total_articles = int(headlines["storyId"].nunique() if "storyId" in headlines.columns else len(headlines))
    calendar_days_in_range = len(daily)
    weeks_in_range = len(weekly)
    weeks_with_zero = int((weekly["article_count"] == 0).sum())
    avg_articles_per_week = total_articles / weeks_in_range if weeks_in_range else 0.0

    return NewsCoverageSummary(
        ticker=ticker.upper(),
        ric=ric,
        start_date=start_date,
        end_date=end_date,
        total_articles=total_articles,
        trading_days_with_news=int((daily["article_count"] > 0).sum()),
        calendar_days_with_news=int((daily["article_count"] > 0).sum()),
        calendar_days_in_range=calendar_days_in_range,
        avg_articles_per_calendar_day=total_articles / calendar_days_in_range if calendar_days_in_range else 0.0,
        avg_articles_per_week=avg_articles_per_week,
        weeks_in_range=weeks_in_range,
        weeks_with_zero_articles=weeks_with_zero,
        passes_paper_weekly_threshold=avg_articles_per_week >= 1.0,
    )
