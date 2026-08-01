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


def test_ravenpack_article_list_filters_headline():
    articles = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([
            "2014-12-30T10:00:00Z",
            "2014-12-31T19:07:00Z",
            "2014-12-31T20:00:00Z",
        ]),
        "headline": [
            "Apple patent granted",
            "Sony Makes 'The Interview' Available on Cable, Satellite Systems",
            "Microsoft cloud deal",
        ],
        "relevance": [90, 12, 80],
        "event_sentiment_score": [0.1, None, 0.2],
        "topic": ["business", None, "business"],
        "rp_story_id": ["1", "2", "3"],
    })
    result = de.ravenpack_article_list(
        articles, ticker="AAPL", limit=10, query="sony", sort="date_desc"
    )
    assert result["matched"] == 1
    assert result["shown"] == 1
    assert "Interview" in result["rows"][0]["headline"]
    assert result["query"] == "sony"


def test_soft_match_comparison_sony_interview_pair():
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-12-31T22:30:00Z"]),
        "headline": ["UPDATE 2-Sony's 'Interview' lands on pay TV and in 580 theaters"],
        "storyId": ["urn:newsml:nNRA110go5:0"],
        "sourceCode": ["NS:RTRS"],
    })
    articles = pd.DataFrame({
        "timestamp_utc": pd.to_datetime([
            "2014-12-31T19:07:40Z",
            "2014-12-31T21:00:00Z",
        ]),
        "headline": [
            "Sony Makes 'The Interview' Available on Cable, Satellite Systems",
            "Unrelated bank merger announced today",
        ],
        "relevance": [12, 99],
        "event_sentiment_score": [None, 0.1],
        "rp_story_id": ["6BA5", "OTHER"],
    })
    table = de.soft_match_comparison(
        news, articles, ticker="AAPL", limit=25, query="sony", min_score=0.45
    )
    assert table["matched"] == 1
    assert table["shown"] == 1
    row = table["rows"][0]
    assert row["matched"] is True
    assert "Interview" in row["headline_ravenpack"]
    assert row["date_refinitiv"].startswith("2014-12-31")
    assert row["date_ravenpack"].startswith("2014-12-31")
    assert row["relevance_score"] == pytest.approx(0.12)


def test_soft_match_comparison_unmatched_rows_optional():
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-12-31T12:00:00Z"]),
        "headline": ["Completely unique widget recall nobody else covered"],
        "storyId": ["x"],
    })
    articles = pd.DataFrame({
        "timestamp_utc": pd.to_datetime(["2014-12-31T12:30:00Z"]),
        "headline": ["Apple launches new iPhone in China"],
        "relevance": [90],
    })
    all_rows = de.soft_match_comparison(
        news, articles, ticker="AAPL", matched_only=False, min_score=0.45
    )
    assert all_rows["matched"] == 0
    assert all_rows["shown"] == 1
    assert all_rows["rows"][0]["headline_ravenpack"] is None

    only = de.soft_match_comparison(
        news, articles, ticker="AAPL", matched_only=True, min_score=0.45
    )
    assert only["shown"] == 0


def test_field_coverage_refinitiv_and_ravenpack(tmp_path, monkeypatch):
    monkeypatch.setattr(de, "FULL_STORY_DIR", tmp_path)
    news = pd.DataFrame({
        "date": pd.to_datetime(["2014-12-31T22:30:00Z"]),
        "headline": ["Sony Interview lands on pay TV"],
        "storyId": ["urn:story:1"],
        "sourceCode": ["NS:RTRS"],
    })
    digest = __import__("hashlib").sha256(b"urn:story:1").hexdigest()[:12]
    body_dir = tmp_path / "AAPL"
    body_dir.mkdir()
    (body_dir / f"sony--{digest}.txt").write_text("full body", encoding="utf-8")

    articles = pd.DataFrame({
        "headline": ["Sony Makes The Interview Available", "Tabular quote"],
        "event_text": ["Sony Pictures signed deals", None],
        "relevance_score": [0.12, 1.0],
        "event_sentiment_score": [0.1, None],
        "sentiment_score": [0.012, None],
        "topic": ["business", None],
        "group": ["media", None],
        "type": ["contract", None],
        "news_type": ["FULL-ARTICLE", "TABULAR-MATERIAL"],
        "rp_story_id": ["a", "b"],
    })
    cov = de.field_coverage(
        ticker="AAPL",
        start_date="2014-01-01",
        end_date="2014-12-31",
        news=news,
        articles=articles,
        price_blocks={"wrds": pd.DataFrame({"date": [1]})},
    )
    assert cov["refinitiv"]["rows"] == 1
    headline = next(r for r in cov["refinitiv"]["fields"] if r["field"] == "headline")
    assert headline["filled"] == 1 and headline["pct"] == 100.0
    body = next(r for r in cov["refinitiv"]["fields"] if "full_story_body" in r["field"])
    assert body["filled"] == 1
    et = next(r for r in cov["ravenpack"]["fields"] if r["field"] == "event_text")
    assert et["filled"] == 1 and et["pct"] == 50.0
    assert cov["prices"][0]["rows"] == 1


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
    assert "coverage" in result
