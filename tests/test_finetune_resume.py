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


def test_run_view_missing_returns_none(run_dir):
    assert rp.run_view("ghost") is None


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
