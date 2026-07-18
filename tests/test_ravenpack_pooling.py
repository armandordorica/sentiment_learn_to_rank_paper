from __future__ import annotations

import pandas as pd

from sentiment_ltr.models.ravenpack_sentiment import (
    deduplicate_pooled_headlines,
    load_ravenpack_labeled_frame,
    split_leakage_audit,
)


def test_global_headline_dedup_prevents_cross_stock_and_cross_split_leakage():
    frame = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "JPM", "XOM"],
        "article_date": pd.to_datetime(["2011-01-01", "2013-01-01", "2012-06-01", "2014-01-01"]),
        "headline": [
            "Company reports record profit!",
            "company reports record profit",
            "Bank raises its dividend",
            "Oil production increases",
        ],
        "story_id": ["s1", "s2", "s3", "s4"],
    })
    pooled = deduplicate_pooled_headlines(frame)
    assert len(pooled) == 3
    audit = split_leakage_audit(pooled)
    assert audit["passed"] is True
    assert audit["headlines_crossing_splits"] == 0


def test_audit_detects_story_and_content_crossing_splits():
    frame = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "article_date": pd.to_datetime(["2011-01-01", "2013-01-01"]),
        "headline": ["Same story", "Same story"],
        "story_id": ["shared", "shared"],
    })
    audit = split_leakage_audit(frame)
    assert audit["passed"] is False
    assert audit["story_ids_crossing_splits"] == 1
    assert audit["headlines_crossing_splits"] == 1


def test_loader_pools_historical_and_fresh_story_id_columns(tmp_path):
    common = {
        "article_time": pd.to_datetime(["2011-01-01"], utc=True),
        "headline": ["Distinct headline"],
        "event_sentiment_score": [0.8],
    }
    pd.DataFrame({
        **common, "article_date": pd.to_datetime(["2011-01-01"]), "story_id": ["old-1"]
    }).to_parquet(
        tmp_path / "aaa_articles_2003_2014.parquet", index=False
    )
    pd.DataFrame({
        **{**common, "headline": ["Another distinct headline"]},
        "rp_story_id": ["new-1"],
    }).to_parquet(tmp_path / "bbb_articles_2003_2014.parquet", index=False)

    pooled = load_ravenpack_labeled_frame(["AAA", "BBB"], news_dir=tmp_path)
    assert set(pooled["ticker"]) == {"AAA", "BBB"}
    assert set(pooled["story_id"]) == {"old-1", "new-1"}
    assert pooled["article_date"].notna().all()
