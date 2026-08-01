from __future__ import annotations

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
        ld_module=object(),  # skip opening a real session
    )
    assert summary["fetched"] == 2
    assert summary["failed"] == 0
    path = cache.story_path(
        tmp_path, "AAPL", "s1", "Alpha", story_time=news.loc[0, "date"]
    )
    assert path.exists()
    assert path.name.startswith("2014-01-01_090000_")
    assert "BODY::s1" in path.read_text()

    # Second run skips both.
    summary2 = cache.cache_refinitiv_stories(
        tmp_path, "AAPL", news, sleep_s=0.0, ld_module=object()
    )
    assert summary2["fetched"] == 0
    assert summary2["already_cached"] == 2
