from __future__ import annotations

import json

import pandas as pd

from sentiment_ltr.data.batch_universe import (
    headline_counts_by_ticker,
    news_threshold_universe,
    ordered_story_queue_tickers,
    ordered_universe_story_queue,
    render_story_queue,
    scan_batch_ticker_coverage,
    tickers_needing_headline_backfill,
    write_news_threshold_universe,
)


def _write_ticker(root, *, rank: int, ticker: str, permno: int, rp_rows: int, news_rows: int, status="complete"):
    cache = root / f"rank_{rank:04d}_{ticker}"
    cache.mkdir(parents=True)
    manifest = {
        "volume_rank": rank,
        "permno": permno,
        "ticker": ticker,
        "company_name": ticker,
        "status": status,
        "provider_status": [
            {"provider": "wrds", "status": "ok", "rows": 100},
            {"provider": "ravenpack", "status": "ok" if rp_rows else "empty", "rows": rp_rows},
            {"provider": "refinitiv", "status": "ok", "rows": 10},
        ],
    }
    (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if rp_rows:
        pd.DataFrame({"headline": [f"h{i}" for i in range(rp_rows)]}).to_parquet(
            cache / "ravenpack_articles.parquet", index=False
        )
    if news_rows:
        pd.DataFrame({
            "date": pd.date_range("2003-01-01", periods=news_rows, freq="D"),
            "headline": [f"n{i}" for i in range(news_rows)],
            "storyId": [f"s{i}" for i in range(news_rows)],
        }).to_parquet(cache / "refinitiv_news.parquet", index=False)


def test_news_threshold_and_story_queue(tmp_path):
    _write_ticker(tmp_path, rank=1, ticker="C", permno=11, rp_rows=20000, news_rows=500)
    _write_ticker(tmp_path, rank=2, ticker="MSFT", permno=22, rp_rows=15000, news_rows=200)
    _write_ticker(tmp_path, rank=3, ticker="THIN", permno=33, rp_rows=10, news_rows=0)
    _write_ticker(tmp_path, rank=4, ticker="NEED", permno=44, rp_rows=5000, news_rows=3)

    coverage = scan_batch_ticker_coverage(tmp_path)
    universe = news_threshold_universe(coverage)
    assert set(universe["ticker"]) == {"C", "MSFT", "NEED"}
    assert not universe.loc[universe["ticker"] == "NEED", "has_usable_refinitiv_headlines"].iloc[0]

    needed = tickers_needing_headline_backfill(coverage)
    assert list(needed["ticker"]) == ["NEED"]

    tickers = ordered_story_queue_tickers(headline_counts_by_ticker(coverage))
    assert tickers[0] == "MSFT"
    assert tickers[1] == "C"
    assert "NEED" not in tickers

    full = ordered_universe_story_queue(universe, prefer_first=("MSFT",))
    assert full[0] == "MSFT"
    assert set(full) == {"C", "MSFT", "NEED"}
    assert full.index("C") < full.index("NEED")

    text = render_story_queue(full)
    assert "NEED" in text
    assert text.startswith("#")

    dest = tmp_path / "universe.csv"
    written = write_news_threshold_universe(tmp_path, dest)
    assert dest.is_file()
    assert len(written) == 3
