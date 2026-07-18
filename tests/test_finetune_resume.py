"""Tests for refresh-safe fine-tune run recovery (webapp/api/ravenpack_finetune.py).

These cover the disk-backed resume logic that lets Section 5 re-attach to a
running (or finished) training job after a browser refresh or webapp restart —
without loading any model.
"""

from __future__ import annotations

import json

import pytest

from webapp.api import ravenpack_finetune as rp


@pytest.fixture()
def run_dir(tmp_path, monkeypatch):
    """Point the module's run-status directory at a temp dir."""
    d = tmp_path / "_finetune_runs"
    d.mkdir()
    monkeypatch.setattr(rp, "_FINETUNE_RUN_DIR", d)
    return d


def _write_status(run_dir, job_id: str, payload: dict) -> None:
    (run_dir / f"{job_id}_status.json").write_text(json.dumps(payload), encoding="utf-8")


def test_latest_run_id_none_when_empty(run_dir):
    assert rp.latest_run_id() is None


def test_five_stock_targets_remain_selectable_before_exports_are_ready(monkeypatch):
    monkeypatch.setattr(rp, "rich_export_tickers", lambda: ["AAPL"])
    assert set(rp.DEFAULT_FIVE_STOCK_TICKERS).issubset(rp.available_tickers())


def test_latest_run_id_picks_newest(run_dir):
    _write_status(run_dir, "old", {"status": "done"})
    newest = run_dir / "new_status.json"
    _write_status(run_dir, "new", {"status": "running"})
    # Force "new" to be the most recently modified.
    import os
    import time

    os.utime(run_dir / "old_status.json", (time.time() - 100, time.time() - 100))
    os.utime(newest, None)
    assert rp.latest_run_id() == "new"


def test_latest_run_id_skips_stale_running_status(run_dir):
    import os
    import time

    _write_status(run_dir, "finished", {"status": "done"})
    stale = run_dir / "stale_status.json"
    _write_status(run_dir, "stale", {"status": "running", "step": 3900})
    os.utime(stale, (time.time() - 600, time.time() - 600))
    assert rp.latest_run_id() == "finished"


def test_latest_run_id_accepts_live_worker_pid(run_dir):
    import os

    _write_status(run_dir, "live", {"status": "running", "worker_pid": os.getpid()})
    assert rp.latest_run_id() == "live"


def test_read_run_state_missing_and_corrupt(run_dir):
    assert rp.read_run_state("nope") is None
    (run_dir / "bad_status.json").write_text("{not json", encoding="utf-8")
    assert rp.read_run_state("bad") is None  # mid-write / corrupt → None, no crash


def test_run_view_running(run_dir):
    _write_status(run_dir, "job1", {
        "status": "running", "message": "Training on Apple GPU (Metal / MPS) — step 100 / 4730",
        "device": "mps", "device_name": "Apple GPU (Metal / MPS)",
        "step": 100, "total_steps": 4730, "pct": 2.1, "epoch": 0.04,
    })
    view = rp.run_view("job1")
    assert view.id == "job1"
    assert view.status == "running"          # template polls in this state
    assert view.result is None
    assert view.error is None
    assert view.progress["device"] == "mps"
    assert view.progress["step"] == 100
    assert "Apple GPU" in view.progress_message


def test_run_view_done_builds_result(run_dir):
    _write_status(run_dir, "job2", {
        "status": "done", "message": "Finished.",
        "device": "mps", "test_f1": 0.49, "test_acc": 0.62,
    })
    view = rp.run_view("job2")
    assert view.status == "done"
    assert view.result["test_f1"] == 0.49
    assert view.result["test_acc"] == 0.62
    assert view.result["device"] == "mps"
    assert view.result["checkpoint_dir"].endswith("ravenpack_distilbert_best")


def test_run_view_error_surfaces_traceback(run_dir):
    _write_status(run_dir, "job3", {
        "status": "error", "message": "Training failed: boom",
        "error": "boom\n\nTraceback (most recent call last): ...",
    })
    view = rp.run_view("job3")
    assert view.status == "error"
    assert "boom" in view.error


def test_run_view_marks_stale_running_record_as_error(run_dir):
    import os
    import time

    path = run_dir / "stale_status.json"
    _write_status(run_dir, "stale", {"status": "running", "message": "Training"})
    os.utime(path, (time.time() - 600, time.time() - 600))
    view = rp.run_view("stale")
    assert view.status == "error"
    assert "no active training worker" in view.error


def test_run_view_missing_returns_none(run_dir):
    assert rp.run_view("ghost") is None


def test_run_view_paused_exposes_resume_checkpoint(run_dir):
    _write_status(run_dir, "paused", {
        "status": "paused", "message": "Paused safely.", "step": 321,
        "total_steps": 1000, "checkpoint_path": "/tmp/checkpoint-321",
    })
    view = rp.run_view("paused")
    assert view.status == "paused"
    assert view.progress["checkpoint_path"] == "/tmp/checkpoint-321"


def test_request_pause_writes_control_signal(run_dir):
    _write_status(run_dir, "live", {"status": "running", "worker_pid": 123, "step": 5})
    view = rp.request_pause("live")
    control = json.loads((run_dir / "live_control.json").read_text())
    assert control == {"action": "pause"}
    assert "Pause requested" in view.progress_message


def test_loss_chart_requires_two_history_points():
    assert rp.loss_chart(None) is None
    assert rp.loss_chart({"loss_history": [{"step": 10, "loss": 0.9}]}) is None


def test_loss_chart_builds_svg_geometry_and_keeps_eval_history():
    chart = rp.loss_chart({
        "total_steps": 100, "epochs": 2,
        "loss_history": [
            {"step": 10, "epoch": 0.2, "loss": 0.9},
            {"step": 50, "epoch": 1.0, "loss": 0.5},
        ],
        "eval_history": [{"step": 50, "epoch": 1.0, "eval_f1": 0.6}],
    })
    assert chart["n_points"] == 2
    assert chart["last_loss"] == 0.5
    assert len(chart["polyline"].split()) == 2
    assert chart["epoch_marks"][0]["label"] == "end e1"
    assert chart["eval_history"][0]["eval_f1"] == 0.6


def test_coverage_summary_reports_actual_split_date_ranges(monkeypatch):
    import pandas as pd

    frame = pd.DataFrame({
        "ticker": ["AAPL"] * 4,
        "article_date": pd.to_datetime(["2010-02-03", "2011-11-30", "2012-06-15", "2013-04-09"]),
        "label_name": ["positive", "neutral", "negative", "positive"],
        "headline": ["one", "two", "three", "four"],
    })
    monkeypatch.setattr(rp, "load_ravenpack_labeled_frame", lambda tickers: frame)
    coverage = rp.coverage_summary(["AAPL"])
    by_split = {row["split"]: row for row in coverage["splits_table"]}
    assert by_split["train"]["start_date"] == "2010-02-03"
    assert by_split["train"]["end_date"] == "2011-11-30"
    assert by_split["validation"]["start_date"] == "2012-06-15"
    assert by_split["test"]["end_date"] == "2013-04-09"


def test_compare_checkpoints_uses_same_test_split(monkeypatch):
    models = [
        {"id": "phrasebank_distilbert_best", "title": "Before", "sha": "aaa", "description": "before", "trained_tickers": []},
        {"id": "ravenpack_distilbert_best", "title": "After", "sha": "bbb", "description": "after", "trained_tickers": ["AAPL"]},
    ]
    monkeypatch.setattr(rp, "comparison_models", lambda: models)
    calls = []

    def fake_eval(tickers, *, model_dir, eval_split):
        calls.append((tickers, model_dir.name, eval_split))
        score = 0.4 if model_dir.name.startswith("phrasebank") else 0.6
        return {"n_rows": 100, "macro_f1": score, "accuracy": score + 0.1}

    monkeypatch.setattr(rp._ravenpack_sentiment, "evaluate_phrasebank_baseline_on_ravenpack", fake_eval)
    result = rp.compare_checkpoints(
        ["AAPL"], ["phrasebank_distilbert_best", "ravenpack_distilbert_best"]
    )
    assert calls == [
        (["AAPL"], "phrasebank_distilbert_best", "test"),
        (["AAPL"], "ravenpack_distilbert_best", "test"),
    ]
    assert result["same_test_rows"] is True
    assert result["best"]["macro_f1"] == pytest.approx(0.6)
    assert result["models"][0]["domain_label"] == "Dataset OOD: no RavenPack training"
    assert result["models"][1]["domain_label"] == "Held-out, in-universe tickers"


def test_three_basket_ood_benchmark_averages_baskets_and_builds_graphs(monkeypatch):
    monkeypatch.setattr(rp, "DEFAULT_OOD_BASKETS", {
        "Basket 1": ["AAA"], "Basket 2": ["BBB"], "Basket 3": ["CCC"],
    })

    def fake_compare(tickers, model_ids, job=None):
        score = {"AAA": 0.3, "BBB": 0.6, "CCC": 0.9}[tickers[0]]
        models = []
        for model_id, bump in (("base", 0.0), ("adapted", 0.1)):
            models.append({
                "id": model_id, "title": model_id.title(), "sha": model_id,
                "description": model_id, "n_rows": 10,
                "macro_f1": score + bump, "accuracy": score + bump,
                "per_ticker": [{"ticker": tickers[0], "n_rows": 10,
                                "macro_f1": score + bump, "accuracy": score + bump}],
            })
        return {"models": models, "tickers": tickers}

    monkeypatch.setattr(rp, "compare_checkpoints", fake_compare)
    result = rp.compare_ood_baskets(["base", "adapted"])
    assert result["averages"][0]["macro_f1"] == pytest.approx(0.6)
    assert result["averages"][1]["macro_f1"] == pytest.approx(0.7)
    assert result["best"]["id"] == "adapted"
    assert "Plotly.newPlot" in result["charts"]["macro_f1"]
    assert "Plotly.newPlot" in result["charts"]["ticker_heatmap"]
