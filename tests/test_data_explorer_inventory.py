from __future__ import annotations

from webapp.api import data_explorer as de


def test_build_inventory_aapl_paper_window_ready_products():
    inv = de.build_inventory("AAPL", "2003-01-01", "2014-12-31")
    by_id = {p["id"]: p for p in inv["products"]}
    assert by_id["refinitiv_headlines"]["status"] == "ready"
    assert by_id["refinitiv_headlines"]["rows"] == 27210
    assert "headline" in by_id["refinitiv_headlines"]["fields_present"]
    assert by_id["ravenpack_articles"]["status"] == "ready"
    assert by_id["ravenpack_articles"]["rows"] > 100000
    # Full bodies are not fully cached yet
    assert by_id["refinitiv_full_stories"]["status"] in {"missing", "partial"}


def test_selective_load_check_reports_existing_without_result():
    out = de.selective_load(
        "AAPL",
        "2003-01-01",
        "2014-12-31",
        selected_ids=["ravenpack_articles", "refinitiv_headlines"],
        action="check",
    )
    assert out["mode"] == "check"
    assert out["result"] is None
    assert any("already exist" in m["text"] for m in out["messages"])
    assert any("RavenPack" in m["text"] for m in out["messages"])


def test_selective_load_uses_cache_when_ready(monkeypatch):
    called = {"live": 0}

    def boom(*args, **kwargs):
        called["live"] += 1
        raise AssertionError("live pull should not run for ready cache")

    monkeypatch.setattr(de.live_data, "run_ticker_data_query", boom)
    out = de.selective_load(
        "AAPL",
        "2003-01-01",
        "2014-12-31",
        selected_ids=["refinitiv_headlines", "ravenpack_articles"],
        action="load",
    )
    assert called["live"] == 0
    assert out["result"] is not None
    assert out["result"]["refinitiv_headlines"]["total"] == 27210
    assert any("local cache" in m["text"].lower() for m in out["messages"])
