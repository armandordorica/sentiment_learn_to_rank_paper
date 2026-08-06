from __future__ import annotations

import pandas as pd

from webapp.api import batch_pipeline as bp


def test_snapshot_includes_news_threshold(tmp_path, monkeypatch):
    path = tmp_path / "ravenpack_news_threshold_universe.csv"
    pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "NEED"],
        "has_usable_refinitiv_headlines": [True, True, False],
    }).to_csv(path, index=False)
    monkeypatch.setattr(bp, "NEWS_THRESHOLD_PATH", path)
    snap = bp.snapshot(pd.DataFrame({
        "status": ["complete", "partial"],
        "wrds_status": ["ok", "ok"],
        "yahoo_status": ["ok", "failed"],
        "ravenpack_status": ["ok", "ok"],
        "refinitiv_status": ["ok", "ok"],
    }))
    assert snap["news_threshold"]["n_names"] == 3
    assert snap["news_threshold"]["n_with_headlines"] == 2
    assert snap["news_threshold"]["n_missing_headlines"] == 1
