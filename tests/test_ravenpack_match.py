from __future__ import annotations

import pandas as pd

from sentiment_ltr.data.ravenpack_match import (
    find_best_ravenpack_match,
    headline_match_score,
    nearby_ravenpack_summary,
    resolve_ravenpack_for_story,
)


def test_headline_match_strips_wire_prefixes():
    assert headline_match_score(
        "MEDIA-Apple granted patent for communicating stylus - CNBC",
        "Apple granted patent for communicating stylus",
    ) >= 0.8


def test_headline_match_rewritten_same_story():
    """Refinitiv and RavenPack often rewrite the same event with different verbs."""
    score = headline_match_score(
        "UPDATE 2-Sony's 'Interview' lands on pay TV and in 580 theaters",
        "Sony Makes 'The Interview' Available on Cable, Satellite Systems",
    )
    assert score >= 0.45


def test_find_best_ravenpack_match_prefers_windowed_headline():
    articles = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([
            "2014-12-30 10:00:00",
            "2014-12-31 21:00:00",
            "2014-12-31 21:30:00",
        ]),
        "headline": [
            "Unrelated bank story",
            "Apple granted patent for communicating stylus",
            "Sony Interview lands on pay TV",
        ],
        "relevance": [99, 88, 10],
        "event_sentiment_score": [0.1, 0.5, -0.2],
        "event_text": ["x", "stylus event", "sony event"],
        "topic": ["a", "business", "media"],
        "type": ["t1", "product", "film"],
        "rp_story_id": ["1", "2", "3"],
    })
    hit = find_best_ravenpack_match(
        articles,
        headline="MEDIA-Apple granted patent for 'communicating stylus' - CNBC",
        story_time=pd.Timestamp("2014-12-31 22:00:00"),
    )
    assert hit is not None
    assert hit["matched"] is True
    assert hit["relevance_score"] == 0.88
    assert hit["event_text"] == "stylus event"


def test_nearby_helper_still_available_but_resolve_does_not_use_it():
    """Timestamp-adjacent scores must not be assigned as this story's relevance."""
    articles = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2014-12-31 20:00:00", "2014-12-31 21:00:00"]),
        "relevance": [40, 95],
        "event_sentiment_score": [0.1, 0.2],
        "topic": ["old", "hot"],
        "type": ["a", "b"],
    })
    summary = nearby_ravenpack_summary(
        articles, story_time=pd.Timestamp("2014-12-31 22:00:00")
    )
    assert summary is not None
    assert summary["relevance_score"] == 0.95

    result = resolve_ravenpack_for_story(
        ticker="AAPL",
        headline="UPDATE 2-Sony's 'Interview' lands on pay TV and in 580 theaters",
        story_time="2014-12-31 22:30:00",
        cached_articles=articles,
        day_cache_dir=None,
        query_day_fn=None,
    )
    assert result["matched"] is False
    assert result["relevance_score"] is None
    assert "headline" in result["note"].lower() or "matched" in result["note"].lower()


def test_resolve_uses_injected_day_frame(tmp_path):
    day = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2014-01-15 12:00:00"]),
        "headline": ["Microsoft announces cloud deal"],
        "event_text": ["Cloud deal details"],
        "relevance": [80],
        "event_sentiment_score": [0.3],
        "rp_story_id": ["rp"],
        "topic": ["business"],
        "type": ["deal"],
    })

    def query_fn(ticker, start, end):
        assert ticker == "MSFT"
        return day

    result = resolve_ravenpack_for_story(
        ticker="MSFT",
        headline="Microsoft announces cloud deal",
        story_time="2014-01-15 12:30:00",
        cached_articles=pd.DataFrame(),
        day_cache_dir=tmp_path,
        query_day_fn=query_fn,
    )
    assert result["matched"] is True
    assert result["relevance_score"] == 0.8
    assert (tmp_path / "MSFT" / "2014-01-15.parquet").exists()
