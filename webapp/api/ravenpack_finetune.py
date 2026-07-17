"""RavenPack fine-tuning (Tab 5·8): train/re-train on 1 / 5 / N tickers.

This module wraps ``src/sentiment_ltr/models/ravenpack_sentiment.py`` — no
business logic is duplicated here, only the data shaping needed for the
FastAPI/Jinja2 presentation layer (mirrors Streamlit's ``app.py``
``render_ravenpack_finetuning_tab`` → section 5·8).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_ltr.models import phrasebank_sentiment as _phrasebank_sentiment  # noqa: E402
from sentiment_ltr.models import ravenpack_sentiment as _ravenpack_sentiment  # noqa: E402

model_is_saved = _phrasebank_sentiment.model_is_saved
resolve_model_dir = _phrasebank_sentiment.resolve_model_dir
finetuning_deps_available = _phrasebank_sentiment.finetuning_deps_available

DEFAULT_RAVENPACK_MODEL_DIR = _ravenpack_sentiment.DEFAULT_RAVENPACK_MODEL_DIR
DEFAULT_RAVENPACK_TRAIN_EPOCHS = _ravenpack_sentiment.DEFAULT_RAVENPACK_TRAIN_EPOCHS
discover_ravenpack_article_files = _ravenpack_sentiment.discover_ravenpack_article_files
_ticker_from_article_path = _ravenpack_sentiment._ticker_from_article_path
load_ravenpack_labeled_frame = _ravenpack_sentiment.load_ravenpack_labeled_frame
ravenpack_split_summary = _ravenpack_sentiment.ravenpack_split_summary
ravenpack_model_is_saved = _ravenpack_sentiment.ravenpack_model_is_saved
resolve_ravenpack_model_dir = _ravenpack_sentiment.resolve_ravenpack_model_dir
train_ravenpack = _ravenpack_sentiment.train_ravenpack
assign_time_split = _ravenpack_sentiment.assign_time_split

PILOT_TICKERS = ["AAPL", "MSFT", "JPM", "XOM", "JNJ", "WMT"]

# Only AAPL currently has a "rich" RavenPack export (data/raw/news/ravenpack/)
# with a `headline` column; the data_explorer_top1k batch-pipeline exports for
# other tickers lack `headline` and will raise in load_ravenpack_labeled_frame.
# Mirrors the same caveat called out in app.py's render_ravenpack_finetuning_tab.
RICH_EXPORT_TICKERS = ["AAPL"]


def available_tickers() -> list[str]:
    """All tickers with a discoverable RavenPack export (mirrors Streamlit's picker)."""
    paths = discover_ravenpack_article_files()
    return sorted({_ticker_from_article_path(p) for p in paths})


def pilot_default_tickers(available: list[str]) -> list[str]:
    """Default ticker selection — restricted to tickers with a full headline export."""
    rich = [t for t in RICH_EXPORT_TICKERS if t in available]
    if rich:
        return rich
    return [t for t in PILOT_TICKERS if t in available]


def deps_status() -> dict[str, Any]:
    return {
        "finetuning_deps_available": finetuning_deps_available(),
        "has_phrasebank_checkpoint": model_is_saved(resolve_model_dir()),
        "has_ravenpack_checkpoint": ravenpack_model_is_saved(),
    }


def coverage_summary(tickers: list[str]) -> dict[str, Any]:
    """Row counts + train/val/test split coverage for the selected tickers.

    Mirrors the 5·8 coverage table in Streamlit, including the per-ticker
    breakdown shown when more than one ticker is selected.
    """
    try:
        labeled = load_ravenpack_labeled_frame(tickers)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not load training data: {exc}"}
    splits = ravenpack_split_summary(labeled)

    def _split_rows(split_name: str) -> int:
        rows = splits.loc[splits["split"] == split_name, "rows"]
        return int(rows.iloc[0]) if not rows.empty else 0

    per_ticker: list[dict[str, Any]] = []
    if len(tickers) > 1:
        for t in tickers:
            t_frame = labeled[labeled["ticker"].str.upper() == t]
            t_split = assign_time_split(t_frame["article_date"])
            per_ticker.append({
                "ticker": t,
                "labeled": int(len(t_frame)),
                "train": int((t_split == "train").sum()),
                "val": int((t_split == "validation").sum()),
                "test": int((t_split == "test").sum()),
            })

    return {
        "tickers": tickers,
        "error": None,
        "total_labeled": int(len(labeled)),
        "train_rows": _split_rows("train"),
        "val_rows": _split_rows("validation"),
        "test_rows": _split_rows("test"),
        "splits_table": splits.to_dict(orient="records"),
        "per_ticker": per_ticker,
    }


device_report = _phrasebank_sentiment.device_report

FINETUNE_WORKER = PROJECT_ROOT / "scripts" / "finetune_worker.py"
_FINETUNE_RUN_DIR = PROJECT_ROOT / "data" / "models" / "_finetune_runs"


def run_training(
    tickers: list[str],
    *,
    init_from_phrasebank: bool,
    num_train_epochs: int,
    job: Any | None = None,
) -> dict[str, Any]:
    """Run fine-tuning in a **subprocess** and stream its progress into ``job``.

    Training goes through ``scripts/finetune_worker.py`` (a fresh process, main
    thread) rather than in-thread, because HF ``Trainer`` + ``accelerate``'s
    process-global state crashes when ``trainer.train()`` runs in the server's
    background thread. The worker writes a JSON status file after every step;
    this function polls it and mirrors it into ``job.progress`` /
    ``job.progress_message`` for the existing HTMX status partial.
    """
    import subprocess
    import sys
    import time as _time

    _FINETUNE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_id = job.id if job is not None else f"{int(_time.time())}"
    status_file = _FINETUNE_RUN_DIR / f"{run_id}_status.json"
    metrics_out = _FINETUNE_RUN_DIR / f"{run_id}_metrics.json"
    status_file.unlink(missing_ok=True)
    metrics_out.unlink(missing_ok=True)

    cmd = [
        sys.executable, str(FINETUNE_WORKER),
        "--status-file", str(status_file),
        "--metrics-out", str(metrics_out),
        "--tickers", *tickers,
        "--epochs", str(num_train_epochs),
    ]
    if init_from_phrasebank:
        cmd.append("--init-from-phrasebank")

    if job is not None:
        job.progress_message = "Launching training subprocess…"

    # start_new_session detaches the worker into its own process group so a
    # dev-server auto-reload (or the JobManager thread dying) can't kill an
    # in-flight training run — its status file on disk stays the source of truth.
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), start_new_session=True)

    def _sync_job() -> dict[str, Any] | None:
        if not status_file.exists():
            return None
        try:
            state = json.loads(status_file.read_text(encoding="utf-8"))
        except Exception:
            return None  # mid-write; try again next poll
        if job is not None:
            job.progress = state
            job.progress_message = state.get("message", job.progress_message)
        return state

    # Poll the status file until the worker process exits.
    last: dict[str, Any] | None = None
    while proc.poll() is None:
        last = _sync_job() or last
        _time.sleep(1.0)
    last = _sync_job() or last  # final read after exit

    if proc.returncode != 0:
        err = (last or {}).get("error") or f"Training subprocess exited with code {proc.returncode}."
        raise RuntimeError(err)

    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    return {
        "metrics": metrics,
        "test_f1": metrics.get("test", {}).get("eval_f1"),
        "test_acc": metrics.get("test", {}).get("eval_accuracy"),
        "device": metrics.get("device"),
        "checkpoint_dir": str(DEFAULT_RAVENPACK_MODEL_DIR.relative_to(PROJECT_ROOT)),
    }


# ── Refresh-safe run recovery ─────────────────────────────────────────────────
# Every run writes a status file to _FINETUNE_RUN_DIR (see run_training). Reading
# that file lets the fine-tune page rebuild the live status after a browser
# refresh or even a webapp restart, so a long training run is never "lost".


def latest_run_id() -> str | None:
    """Job id of the most recent run (by status-file mtime), or None."""
    if not _FINETUNE_RUN_DIR.exists():
        return None
    files = sorted(_FINETUNE_RUN_DIR.glob("*_status.json"), key=lambda p: p.stat().st_mtime)
    return files[-1].name[: -len("_status.json")] if files else None


def read_run_state(job_id: str) -> dict[str, Any] | None:
    """The status dict a running/finished worker last wrote for ``job_id``."""
    path = _FINETUNE_RUN_DIR / f"{job_id}_status.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None  # mid-write; caller can retry on the next poll


def run_view(job_id: str) -> SimpleNamespace | None:
    """Reconstruct a job-like object from disk for the status template.

    Used when the in-memory ``Job`` is gone (webapp restarted) or hasn't been
    looked up yet — matches the attributes ``partials/train_status.html`` reads
    (``id`` / ``status`` / ``progress`` / ``progress_message`` / ``result`` /
    ``error``) so the same template renders from either source.
    """
    state = read_run_state(job_id)
    if state is None:
        return None
    disk_status = state.get("status", "running")
    status = disk_status if disk_status in ("done", "error") else "running"
    result = None
    if disk_status == "done":
        result = {
            "test_f1": state.get("test_f1"),
            "test_acc": state.get("test_acc"),
            "device": state.get("device"),
            "checkpoint_dir": str(DEFAULT_RAVENPACK_MODEL_DIR.relative_to(PROJECT_ROOT)),
        }
    return SimpleNamespace(
        id=job_id,
        status=status,
        progress=state,
        progress_message=state.get("message", ""),
        result=result,
        error=state.get("error"),
    )
