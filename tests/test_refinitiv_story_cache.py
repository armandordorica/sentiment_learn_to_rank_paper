from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sentiment_ltr.data import refinitiv_story_cache as cache


def test_story_filename_includes_timestamp():
    name = cache.story_filename(
        "id-1",
        "Apple seen unveiling video iPod at Macworld Expo",
        story_time="2003-01-07 15:30:00",
    )
    assert name.startswith("2003-01-07_153000_")
    assert name.endswith(f"--{cache.story_digest('id-1')}.txt")
    assert "apple-seen-unveiling" in name


def test_headlines_needing_bodies_skips_existing(tmp_path: Path):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01 10:00:00", "2014-01-02 11:00:00"]),
        "headline": ["First story", "Second story"],
        "storyId": ["id-1", "id-2"],
    })
    path = cache.story_path(
        tmp_path, "AAPL", "id-1", "First story", story_time=news.loc[0, "date"]
    )
    cache.write_story_file(
        path,
        story_id="id-1",
        headline="First story",
        ticker="AAPL",
        text="body",
        story_time=news.loc[0, "date"],
    )
    pending = cache.headlines_needing_bodies(news, tmp_path, "AAPL")
    assert list(pending["storyId"]) == ["id-2"]


def test_find_cached_story_path_finds_legacy_names(tmp_path: Path):
    legacy = (
        cache.full_story_root(tmp_path)
        / "AAPL"
        / f"first-story--{cache.story_digest('id-1')}.txt"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_text("body", encoding="utf-8")
    found = cache.find_cached_story_path(tmp_path, "AAPL", "id-1")
    assert found == legacy


def test_rename_legacy_story_files(tmp_path: Path):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2003-01-07 15:30:00"]),
        "headline": ["Apple seen unveiling video iPod"],
        "storyId": ["id-1"],
    })
    digest = cache.story_digest("id-1")
    legacy = cache.full_story_root(tmp_path) / "AAPL" / f"apple-seen--{digest}.txt"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("body", encoding="utf-8")
    result = cache.rename_legacy_story_files(tmp_path, "AAPL", news)
    assert result["renamed"] == 1
    assert not legacy.exists()
    renamed = list((cache.full_story_root(tmp_path) / "AAPL").glob("*.txt"))
    assert len(renamed) == 1
    assert renamed[0].name.startswith("2003-01-07_153000_")


def test_cache_refinitiv_stories_uses_injected_ld(tmp_path: Path, monkeypatch):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01 09:00:00", "2014-01-02 10:00:00"]),
        "headline": ["Alpha", "Beta"],
        "storyId": ["s1", "s2"],
    })

    def fake_fetch(project_root, story_id, *, as_text=True, ld_module=None):
        return f"BODY::{story_id}"

    monkeypatch.setattr(cache, "fetch_refinitiv_story", fake_fetch)
    summary = cache.cache_refinitiv_stories(
        tmp_path,
        "AAPL",
        news,
        sleep_s=0.0,
        min_sleep_s=0.0,
        adaptive=False,
        ld_module=object(),  # skip opening a real session
    )
    assert summary["fetched"] == 2
    assert summary["failed"] == 0
    assert summary["pct_run"] == 100.0
    assert summary["bytes_on_disk"] > 0
    path = cache.story_path(
        tmp_path, "AAPL", "s1", "Alpha", story_time=news.loc[0, "date"]
    )
    assert path.exists()
    assert path.name.startswith("2014-01-01_090000_")
    assert "BODY::s1" in path.read_text()

    # Second run skips both.
    summary2 = cache.cache_refinitiv_stories(
        tmp_path,
        "AAPL",
        news,
        sleep_s=0.0,
        min_sleep_s=0.0,
        adaptive=False,
        ld_module=object(),
    )
    assert summary2["fetched"] == 0
    assert summary2["already_cached"] == 2


def test_pull_pauses_on_control_file(tmp_path: Path, monkeypatch):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01", "2014-01-02", "2014-01-03"]),
        "headline": ["A", "B", "C"],
        "storyId": ["s1", "s2", "s3"],
    })
    calls = {"n": 0}

    def fake_fetch(project_root, story_id, *, as_text=True, ld_module=None):
        calls["n"] += 1
        if calls["n"] == 1:
            cache.request_pull_control(tmp_path, "AAPL", "pause")
        return f"BODY::{story_id}"

    monkeypatch.setattr(cache, "fetch_refinitiv_story", fake_fetch)
    summary = cache.cache_refinitiv_stories(
        tmp_path,
        "AAPL",
        news,
        sleep_s=0.0,
        min_sleep_s=0.0,
        adaptive=False,
        ld_module=object(),
    )
    assert summary["status"] == "paused"
    assert summary["fetched"] == 1
    assert summary["processed"] == 1
    assert calls["n"] == 1


def test_format_bytes_and_story_dir_stats(tmp_path: Path):
    assert cache.format_bytes(500) == "500 B"
    assert cache.format_bytes(2048).endswith("KB")
    path = cache.story_path(tmp_path, "MSFT", "id", "Hello", story_time="2014-01-01")
    cache.write_story_file(
        path, story_id="id", headline="Hello", ticker="MSFT", text="x" * 100, story_time="2014-01-01"
    )
    stats = cache.story_dir_stats(tmp_path, "MSFT")
    assert stats["files"] == 1
    assert stats["bytes"] > 100
    rows = cache.list_story_cache_stats(tmp_path)
    assert rows and rows[0]["ticker"] == "MSFT"


def test_build_story_day_grid_colors(tmp_path: Path, monkeypatch):
    news = pd.DataFrame({
        "date": pd.to_datetime([
            "2003-01-01 10:00:00",
            "2003-01-01 12:00:00",
            "2003-01-02 09:00:00",
            "2003-01-03 09:00:00",
        ]),
        "headline": ["A", "B", "C", "D"],
        "storyId": ["s1", "s2", "s3", "s4"],
    })
    # Jan 1: both bodies present → green
    for sid, hl, ts in [
        ("s1", "A", news.loc[0, "date"]),
        ("s2", "B", news.loc[1, "date"]),
    ]:
        path = cache.story_path(tmp_path, "AAPL", sid, hl, story_time=ts)
        cache.write_story_file(
            path, story_id=sid, headline=hl, ticker="AAPL", text="ok", story_time=ts
        )
    # Jan 2: failed in manifest, no body → red when pending cleared
    cache._append_manifest(
        tmp_path,
        "AAPL",
        {"story_id": "s3", "status": "failed", "headline": "C"},
    )
    grid = cache.build_story_day_grid(
        tmp_path,
        "AAPL",
        news,
        start="2003-01-01",
        end="2003-01-04",
        current_date="2003-01-03",
        pull_running=True,
    )
    by_date = {d["date"]: d for d in grid["days"]}
    assert by_date["2003-01-01"]["status"] == "green"
    assert by_date["2003-01-01"]["n"] == 2
    assert by_date["2003-01-01"]["shade"] == 2  # 2 articles → shade 2
    assert by_date["2003-01-01"]["tried"] == 2
    assert by_date["2003-01-01"]["have"] == 2
    assert "Retrieved successfully: 2" in by_date["2003-01-01"]["tip"]
    assert by_date["2003-01-02"]["status"] == "red"
    assert by_date["2003-01-02"]["shade"] == 1  # 1 failure → shade 1
    assert by_date["2003-01-02"]["tried"] == 1
    assert by_date["2003-01-02"]["failed"] == 1
    assert by_date["2003-01-03"]["status"] == "yellow"
    assert by_date["2003-01-04"]["status"] == "blue"
    assert "2003-01-04" in by_date["2003-01-04"]["tip"]
    assert grid["counts"]["green"] == 1
    assert grid["counts"]["blue"] == 1
    assert grid["years"][0]["months"][0]["label"] == "Jan"
    assert grid["years"][0]["months"][0]["label_full"] == "January 2003"


def test_summarize_pacer_bandit_marks_preferred_arm():
    summary = cache.summarize_pacer_bandit(
        {
            "policy": "thompson",
            "arms": [1.0, 4.0, 16.0],
            "alphas": [1.0, 5.0, 1.0],
            "betas": [1.0, 1.0, 4.0],
            "rate_limit_hits": 8,
        }
    )
    assert summary is not None
    assert summary["total_wins"] == 4
    assert summary["total_losses"] == 3
    preferred = [r for r in summary["arms"] if r["preferred"]]
    assert len(preferred) == 1
    assert preferred[0]["arm_s"] == 4.0
    assert preferred[0]["wins"] == 4
    assert preferred[0]["losses"] == 0

    pacer = cache.AdaptivePacer(
        sleep_s=0.5, min_sleep_s=0.25, max_sleep_s=60.0, adaptive=True, policy="aimd"
    )
    pacer.on_rate_limit()
    assert pacer.sleep_s >= 1.0
    assert pacer.rate_limit_hits == 1
    before = pacer.sleep_s
    for _ in range(5):
        pacer.on_success()
    assert pacer.sleep_s <= before


def test_thompson_pacer_learns_from_rate_limit_outcomes():
    pacer = cache.AdaptivePacer(
        sleep_s=0.5,
        min_sleep_s=0.25,
        max_sleep_s=8.0,
        adaptive=True,
        policy="thompson",
        n_arms=5,
    )
    pacer._rng.seed(0)
    assert len(pacer.arms) == 5
    pacer.on_rate_limit()
    first_wait = pacer.sleep_s
    assert first_wait in pacer.arms
    assert pacer._pending_arm is not None
    # Failed cooldown → debit arm, pick again.
    before_beta = list(pacer.betas)
    pacer.on_rate_limit()
    assert sum(pacer.betas) > sum(before_beta)
    # Successful recovery → credit pending arm.
    before_alpha = list(pacer.alphas)
    pacer.on_success()
    assert sum(pacer.alphas) > sum(before_alpha)
    assert pacer._pending_arm is None
    snap = pacer.snapshot()
    assert snap["pacer_policy"] == "thompson"
    assert snap["preferred_sleep_s"] >= pacer.min_sleep_s


def test_thompson_pacer_state_roundtrip(tmp_path: Path):
    pacer = cache.AdaptivePacer(
        sleep_s=2.0,
        min_sleep_s=0.5,
        max_sleep_s=16.0,
        policy="thompson",
        n_arms=4,
    )
    pacer._rng.seed(1)
    pacer.on_rate_limit()
    pacer.on_success()
    cache.save_pacer_state(tmp_path, "AAPL", pacer)
    restored = cache.AdaptivePacer.from_state(
        cache.load_pacer_state(tmp_path, "AAPL"),
        sleep_s=0.5,
        min_sleep_s=0.5,
        max_sleep_s=16.0,
        policy="thompson",
        n_arms=4,
    )
    assert restored.alphas == pacer.alphas
    assert restored.betas == pacer.betas
    assert restored.rate_limit_hits == pacer.rate_limit_hits


def test_cache_takes_cooloff_after_rate_limit_burst(tmp_path: Path, monkeypatch):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01 09:00:00"]),
        "headline": ["Alpha"],
        "storyId": ["s1"],
    })
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_fetch(project_root, story_id, *, as_text=True, ld_module=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Error code 429 | Too many requests, please try again later.")
        return f"BODY::{story_id}"

    monkeypatch.setattr(cache, "fetch_refinitiv_story", fake_fetch)
    monkeypatch.setattr(cache.time, "sleep", lambda s: sleeps.append(float(s)))
    summary = cache.cache_refinitiv_stories(
        tmp_path,
        "AAPL",
        news,
        sleep_s=0.0,
        min_sleep_s=0.0,
        max_sleep_s=1.0,
        adaptive=True,
        pacer_policy="thompson",
        rate_limit_retries=5,
        cooloff_after_rl=2,
        cooloff_s=12.0,
        ld_module=object(),
    )
    assert summary["fetched"] == 1
    # Cool-off sleeps in ≤5s chunks; 12s → at least one long pause path.
    assert sum(sleeps) >= 12.0
    assert any(s >= 5.0 for s in sleeps)


def test_cache_retries_rate_limit_then_succeeds(tmp_path: Path, monkeypatch):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01 09:00:00"]),
        "headline": ["Alpha"],
        "storyId": ["s1"],
    })
    calls = {"n": 0}

    def fake_fetch(project_root, story_id, *, as_text=True, ld_module=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("Error code 429 | Too many requests, please try again later.")
        return f"BODY::{story_id}"

    monkeypatch.setattr(cache, "fetch_refinitiv_story", fake_fetch)
    summary = cache.cache_refinitiv_stories(
        tmp_path,
        "AAPL",
        news,
        sleep_s=0.0,
        min_sleep_s=0.0,
        max_sleep_s=1.0,
        adaptive=True,
        pacer_policy="thompson",
        rate_limit_retries=5,
        ld_module=object(),
    )
    assert summary["fetched"] == 1
    assert summary["failed"] == 0
    assert summary["rate_limit_hits"] >= 2
    assert summary["pacer_policy"] == "thompson"
    assert calls["n"] == 3
    assert cache.pacer_state_path(tmp_path, "AAPL").is_file()


def test_cache_stops_early_on_scope_errors(tmp_path: Path, monkeypatch):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01", "2014-01-02", "2014-01-03"]),
        "headline": ["A", "B", "C"],
        "storyId": ["s1", "s2", "s3"],
    })

    def fake_fetch(project_root, story_id, *, as_text=True, ld_module=None):
        raise RuntimeError(
            "Insufficient scope for key=/data/news/v1/stories/{storyId}. "
            "Missing scopes: {'trapi.data.news.read'}"
        )

    monkeypatch.setattr(cache, "fetch_refinitiv_story", fake_fetch)
    summary = cache.cache_refinitiv_stories(
        tmp_path,
        "AAPL",
        news,
        sleep_s=0.0,
        min_sleep_s=0.0,
        adaptive=False,
        max_failures=50,
        max_scope_failures=2,
        ld_module=object(),
    )
    assert summary["status"] == "stopped"
    assert "max_scope_failures" in (summary.get("stop_reason") or "")
    assert summary["failed"] == 2
    assert summary["scope_failures"] == 2


def test_count_story_fetches_on_local_day(tmp_path: Path):
    from datetime import datetime, timezone

    man = cache.full_story_root(tmp_path) / "AAPL" / "_manifest.jsonl"
    man.parent.mkdir(parents=True)
    today = datetime.now().astimezone().isoformat()
    yesterday = datetime(2020, 1, 1, tzinfo=timezone.utc).isoformat()
    rows = [
        {"status": "fetched", "fetched_at": today, "story_id": "a"},
        {"status": "fetched", "fetched_at": today, "story_id": "b"},
        {"status": "failed", "fetched_at": today, "story_id": "c"},
        {"status": "fetched", "fetched_at": yesterday, "story_id": "d"},
    ]
    man.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert cache.count_story_fetches_on_local_day(tmp_path) == 2


def test_daily_quota_waits_then_fetches(tmp_path: Path, monkeypatch):
    from sentiment_ltr.data.story_quota_scheduler import StoryQuotaSnapshot

    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01 09:00:00"]),
        "headline": ["Alpha"],
        "storyId": ["s1"],
    })
    waits: list[float] = []
    blocked = StoryQuotaSnapshot(
        max_per_window=1,
        calendar_used=1,
        rolling_used=1,
        remaining=0,
        remaining_calendar=0,
        remaining_rolling=0,
        blocking="rolling_24h",
        wait_s=1.0,
        wait_until_local="2026-08-05T01:00:00-04:00",
    )
    open_snap = StoryQuotaSnapshot(
        max_per_window=1,
        calendar_used=0,
        rolling_used=0,
        remaining=1,
        remaining_calendar=1,
        remaining_rolling=1,
        blocking=None,
        wait_s=0.0,
        wait_until_local=None,
    )
    snaps = iter([blocked, open_snap, open_snap, open_snap, open_snap])

    monkeypatch.setattr(
        "sentiment_ltr.data.story_quota_scheduler.quota_snapshot_from_times",
        lambda *a, **k: next(snaps, open_snap),
    )
    monkeypatch.setattr(cache.time, "sleep", lambda s: waits.append(float(s)))
    monkeypatch.setattr(cache, "fetch_refinitiv_story", lambda *a, **k: "BODY")
    progress: list[dict] = []

    summary = cache.cache_refinitiv_stories(
        tmp_path,
        "AAPL",
        news,
        sleep_s=0.0,
        min_sleep_s=0.0,
        max_sleep_s=1.0,
        adaptive=False,
        max_requests_per_day=1,
        ld_module=object(),
        progress_callback=progress.append,
    )
    assert summary["fetched"] == 1
    assert waits
    quota_rows = [row for row in progress if row.get("last_status") == "daily_quota"]
    assert quota_rows
    assert quota_rows[0]["quota_blocking"] == "rolling_24h"
    assert quota_rows[0]["quota_wait_until"] == "2026-08-05T01:00:00-04:00"
