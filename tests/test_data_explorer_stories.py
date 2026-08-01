from __future__ import annotations

import pandas as pd
import plotly.express as px
import pytest

from webapp.api import data_explorer as de


def test_refinitiv_headline_list_formats_and_limits_rows():
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-12-29T12:00:00Z", "2014-12-30T12:00:00Z"]),
        "headline": ["Older", "Newer"],
        "storyId": ["story-1", "story-2"],
        "sourceCode": ["NS:RTRS", "NS:RTRS"],
    })
    result = de.refinitiv_headline_list(news, limit=1)
    assert result["total"] == 2
    assert result["matched"] == 2
    assert result["shown"] == 1
    assert result["rows"][0]["headline"] == "Newer"
    assert result["rows"][0]["date"] == "2014-12-30 12:00"


def test_refinitiv_headline_list_filters_then_limits():
    news = pd.DataFrame({
        "date": pd.to_datetime([
            "2014-12-29T12:00:00Z",
            "2014-12-30T12:00:00Z",
            "2014-12-31T12:00:00Z",
        ]),
        "headline": ["Apple patent", "Sony film", "Apple earnings"],
        "storyId": ["a", "b", "c"],
        "sourceCode": ["NS:RTRS", "NS:RTRS", "NS:RTRS"],
    })
    result = de.refinitiv_headline_list(news, limit=1, query="Apple", sort="date_asc")
    assert result["matched"] == 2
    assert result["shown"] == 1
    assert result["rows"][0]["headline"] == "Apple patent"
    assert result["query"] == "Apple"
    assert result["sort"] == "date_asc"


def test_load_story_uses_shared_refinitiv_loader(monkeypatch, tmp_path):
    monkeypatch.setattr(de, "fetch_refinitiv_story", lambda root, story_id: "Full body")
    monkeypatch.setattr(de, "FULL_STORY_DIR", tmp_path)
    story = de.load_story(" story-123 ", "BUZZ Headline", "MSFT", include_ravenpack=False)
    assert story["story_id"] == "story-123"
    assert story["headline"] == "BUZZ Headline"
    assert story["text"] == "Full body"
    assert story["ravenpack"] is None
    saved = de.Path(story["path"])
    assert saved.exists()
    assert saved.parent.name == "MSFT"
    assert "Full body" in saved.read_text(encoding="utf-8")


def test_load_story_attaches_ravenpack_match(monkeypatch, tmp_path):
    monkeypatch.setattr(de, "fetch_refinitiv_story", lambda root, story_id: "Body")
    monkeypatch.setattr(de, "FULL_STORY_DIR", tmp_path)
    monkeypatch.setattr(de, "load_cached_ravenpack", lambda ticker: pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2014-12-31 20:00:00"]),
        "headline": ["Apple granted patent for communicating stylus"],
        "event_text": ["Apple wins stylus patent."],
        "relevance": [92],
        "event_sentiment_score": [0.4],
        "rp_story_id": ["rp-1"],
        "topic": ["business"],
        "type": ["product-release"],
    }))
    monkeypatch.setattr(de.live_data, "wrds_credentials_available", lambda: False)
    story = de.load_story(
        "story-1",
        "MEDIA-Apple granted patent for 'communicating stylus' - CNBC",
        "AAPL",
        "2014-12-31 22:00",
    )
    rp = story["ravenpack"]
    assert rp["matched"] is True
    assert rp["relevance_score"] == pytest.approx(0.92)
    assert "stylus" in (rp["headline"] or "").lower()


def test_chart_fragment_uses_plotly_from_page_shell():
    html = de._html(px.line(x=[1, 2], y=[3, 4]))
    assert "Plotly.newPlot" in html
    assert "cdn.plot.ly" not in html


def test_present_exposes_saved_refinitiv_news_path(tmp_path):
    news_path = tmp_path / "refinitiv_news.parquet"
    result = de.present({
        "ticker": "MSFT",
        "start_date": "2014-01-01",
        "end_date": "2014-12-31",
        "source": "cache",
        "data_paths": {"refinitiv_news": str(news_path)},
        "providers": {
            "refinitiv": {
                "status": "ok", "error": None, "prices": pd.DataFrame(),
                "news": pd.DataFrame(), "news_daily_counts": pd.DataFrame(),
            }
        },
    })
    assert result["news_storage"] == {
        "saved": True,
        "path": str(news_path),
        "relative_path": str(news_path),
    }


def test_news_chart_uses_high_contrast_bars():
    result = de.present({
        "ticker": "MSFT", "start_date": "2014-01-01", "end_date": "2014-01-02",
        "providers": {
            "refinitiv": {
                "status": "ok", "error": None, "prices": pd.DataFrame(),
                "news": pd.DataFrame(),
                "news_daily_counts": pd.DataFrame({
                    "date": pd.to_datetime(["2014-01-01", "2014-01-02"]),
                    "article_count": [4, 8],
                }),
            }
        },
    })
    chart = result["charts"]["news"]
    assert "#dc4f52" in chart
    assert '"plot_bgcolor":"#ffffff"' in chart
