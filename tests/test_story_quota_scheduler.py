from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from sentiment_ltr.data.story_quota_scheduler import (
    load_ticker_queue,
    next_ticker_needing_bodies,
    quota_snapshot_from_times,
)


def test_load_ticker_queue_skips_comments_and_dupes(tmp_path):
    path = tmp_path / "queue.txt"
    path.write_text("# header\nMSFT\nmsft\nINTC  # chip\n\n", encoding="utf-8")
    assert load_ticker_queue(path) == ["MSFT", "INTC"]


def test_quota_snapshot_uses_tighter_of_calendar_and_rolling():
    tz = timezone(timedelta(hours=-4))
    now = datetime(2026, 8, 5, 1, 0, tzinfo=tz)
    times = [
        now - timedelta(hours=2),
        now - timedelta(minutes=30),
        now - timedelta(minutes=10),
    ]
    snap = quota_snapshot_from_times(times, max_per_day=5, now=now)
    assert snap.calendar_used == 2
    assert snap.rolling_used == 3
    assert snap.remaining == 2


def test_quota_snapshot_rolling_wait_ages_out_oldest():
    now = datetime(2026, 8, 5, 8, 0, tzinfo=timezone(timedelta(hours=-4)))
    times = [now - timedelta(hours=23, minutes=50) + timedelta(seconds=i) for i in range(5)]
    snap = quota_snapshot_from_times(times, max_per_day=3, now=now, rolling_hours=24)
    assert snap.remaining == 0
    assert snap.blocking == "rolling_24h"
    assert snap.rolling_used == 5
    # Need to drop 2 fetches (5-3+1=3? wait remaining>=1 needs rolling<=2, drop=3)
    # rem_roll = 0 when rolling_used=5, cap=3
    # drop = 5 - 3 + 1 = 3 to get remaining >= 1 (rolling <= 2)
    # idx = 2, ready = times[2] + 24h
    expected = times[2] + timedelta(hours=24, seconds=5)
    assert abs(snap.wait_s - (expected - now).total_seconds()) < 2


def test_quota_snapshot_calendar_blocks_when_rolling_has_room():
    tz = timezone(timedelta(hours=-4))
    now = datetime(2026, 8, 5, 18, 0, tzinfo=tz)
    times = [datetime(2026, 8, 5, 10, 0, tzinfo=tz) + timedelta(seconds=i) for i in range(3)]
    snap = quota_snapshot_from_times(times, max_per_day=2, now=now, rolling_hours=1)
    assert snap.remaining == 0
    assert snap.blocking == "calendar_day"
    assert snap.calendar_used == 3
    assert snap.rolling_used == 0


def test_next_ticker_skips_complete_and_missing_news(tmp_path):
    from sentiment_ltr.data import refinitiv_story_cache as cache

    news_msft = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01"]),
        "headline": ["M"],
        "storyId": ["m1"],
    })
    news_intc = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01"]),
        "headline": ["I"],
        "storyId": ["i1"],
    })
    cache.write_story_file(
        cache.story_path(tmp_path, "MSFT", "m1", "M", story_time=news_msft.loc[0, "date"]),
        story_id="m1",
        headline="M",
        ticker="MSFT",
        text="body",
        story_time=news_msft.loc[0, "date"],
    )

    def loader(ticker: str):
        if ticker == "AAPL":
            return None
        if ticker == "MSFT":
            return news_msft
        if ticker == "INTC":
            return news_intc
        return None

    assert next_ticker_needing_bodies(
        ["AAPL", "MSFT", "INTC"],
        news_loader=loader,
        project_root=tmp_path,
    ) == "INTC"
