from __future__ import annotations

from pathlib import Path

import pandas as pd

from sentiment_ltr.data import refinitiv_story_cache as cache


def test_headlines_needing_bodies_skips_existing(tmp_path: Path):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01", "2014-01-02"]),
        "headline": ["First story", "Second story"],
        "storyId": ["id-1", "id-2"],
    })
    path = cache.story_path(tmp_path, "AAPL", "id-1", "First story")
    cache.write_story_file(path, story_id="id-1", headline="First story", ticker="AAPL", text="body")
    pending = cache.headlines_needing_bodies(news, tmp_path, "AAPL")
    assert list(pending["storyId"]) == ["id-2"]


def test_cache_refinitiv_stories_uses_injected_ld(tmp_path: Path, monkeypatch):
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-01-01", "2014-01-02"]),
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
    assert cache.story_path(tmp_path, "AAPL", "s1", "Alpha").exists()
    assert "BODY::s1" in cache.story_path(tmp_path, "AAPL", "s1", "Alpha").read_text()

    # Second run skips both.
    summary2 = cache.cache_refinitiv_stories(
        tmp_path, "AAPL", news, sleep_s=0.0, ld_module=object()
    )
    assert summary2["fetched"] == 0
    assert summary2["already_cached"] == 2
