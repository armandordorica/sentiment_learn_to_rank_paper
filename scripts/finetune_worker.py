"""Standalone RavenPack fine-tuning worker (run as a subprocess by the webapp).

Why a subprocess: HuggingFace ``Trainer`` relies on ``accelerate``'s
process-global ``AcceleratorState`` singleton, which gets torn down and crashes
(`AcceleratorState has no attribute distributed_type`) when ``trainer.train()``
runs in the FastAPI server's background *thread*. Running training in its own
process — fresh interpreter, main thread — reproduces the notebook/CLI
conditions that work, isolates the global state, and means a training crash
can never take down the web server.

Progress (step / pct / device / loss / per-epoch eval) is written as JSON to
``--status-file`` after every step; the webapp polls that file for the live
status panel. Final metrics go to ``--metrics-out`` on success; the traceback
goes into the status file on error.

Usage (built by webapp/api/ravenpack_finetune.py, not meant to be typed):
    python scripts/finetune_worker.py \
        --status-file <path> --metrics-out <path> \
        --tickers AAPL MSFT --epochs 2 --init-from-phrasebank
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file + rename so readers never see a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    parser = argparse.ArgumentParser(description="RavenPack fine-tuning worker")
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--metrics-out", required=True)
    parser.add_argument("--control-file", required=True)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--init-from-phrasebank", action="store_true")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--wandb-run-id")
    args = parser.parse_args()

    if args.wandb_run_id:
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id
        os.environ["WANDB_RESUME"] = "allow"

    status_file = Path(args.status_file)
    metrics_out = Path(args.metrics_out)
    control_file = Path(args.control_file)

    from sentiment_ltr.models.phrasebank_sentiment import device_report
    from sentiment_ltr.models.ravenpack_sentiment import TrainingPaused, train_ravenpack

    report = device_report()
    state: dict[str, Any] = {
        "status": "running",
        "worker_pid": os.getpid(),
        "phase": "preparing",
        "message": "Preparing dataset (load + tokenize) and model…",
        "device": report["selected"],
        "device_name": report["device_name"],
        "torch_version": report["torch_version"],
        "tickers": args.tickers,
        "epochs": args.epochs,
        "init_from_phrasebank": args.init_from_phrasebank,
        "resumed_from_checkpoint": args.resume_from_checkpoint,
        "step": 0,
        "total_steps": 0,
    }
    started = time.monotonic()
    step0_time: dict[str, float] = {}

    def _write() -> None:
        _atomic_write_json(status_file, state)

    _write()

    # Series for the live convergence chart (small: loss is logged every
    # ~100 steps, eval once per epoch).
    loss_history: list[dict[str, Any]] = []
    eval_history: list[dict[str, Any]] = []

    def _on_progress(update: dict[str, Any]) -> None:
        state.update(update)
        step = int(state.get("step") or 0)
        total = int(state.get("total_steps") or 0)
        if step and total:
            state["pct"] = 100 * step / total
            now = time.monotonic()
            # Measure rate from the first observed step so model-load and
            # tokenization time don't skew the ETA.
            if "t0" not in step0_time:
                step0_time["t0"], step0_time["step0"] = now, step
            elif step > step0_time["step0"] and now > step0_time["t0"]:
                rate = (step - step0_time["step0"]) / (now - step0_time["t0"])
                if rate > 0:
                    eta_s = int((total - step) / rate)
                    state["rate"] = f"{rate:.1f} steps/s"
                    state["eta"] = f"{eta_s // 60}m {eta_s % 60:02d}s"
        if "loss" in update:
            loss_history.append({
                "step": step,
                "epoch": round(float(state.get("epoch") or 0), 3),
                "loss": float(update["loss"]),
            })
            state["loss_history"] = loss_history
        if update.get("phase") == "evaluating":
            em = update.get("eval_metrics") or {}
            eval_history.append({
                "step": step,
                "epoch": round(float(state.get("epoch") or 0), 3),
                **{k: float(v) for k, v in em.items()},
            })
            state["eval_history"] = eval_history
            state["message"] = f"Evaluating checkpoint… (step {step:,} / {total:,})"
        elif step and total:
            state["phase"] = "training"
            state["message"] = f"Training on {state['device_name']} — step {step:,} / {total:,}"
        _write()

    def _pause_requested() -> bool:
        if not control_file.exists():
            return False
        try:
            return json.loads(control_file.read_text(encoding="utf-8")).get("action") == "pause"
        except Exception:
            return False

    try:
        metrics = train_ravenpack(
            tickers=args.tickers,
            init_from_phrasebank=args.init_from_phrasebank,
            num_train_epochs=args.epochs,
            progress_callback=_on_progress,
            pause_requested=_pause_requested,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        _atomic_write_json(metrics_out, metrics)
        state.update({
            "status": "done",
            "phase": "done",
            "message": "Finished.",
            "elapsed_s": round(time.monotonic() - started),
            "test_f1": metrics.get("test", {}).get("eval_f1"),
            "test_acc": metrics.get("test", {}).get("eval_accuracy"),
            "device": metrics.get("device", state["device"]),
            "wandb_project_url": metrics.get("wandb_project_url"),
            "wandb_run_id": metrics.get("wandb_run_id"),
            "wandb_run_url": metrics.get("wandb_run_url"),
        })
        _write()
        return 0
    except TrainingPaused as exc:
        state.update({
            "status": "paused",
            "phase": "paused",
            "message": f"Paused safely at step {exc.step:,}. Progress is saved.",
            "checkpoint_path": exc.checkpoint_path,
            "wandb_run_id": state.get("run_id"),
            "wandb_run_url": state.get("run_url"),
            "elapsed_s": round(time.monotonic() - started),
        })
        _write()
        return 0
    except Exception as exc:  # noqa: BLE001
        state.update({
            "status": "error",
            "phase": "error",
            "message": f"Training failed: {exc}",
            "error": f"{exc}\n\n{traceback.format_exc()}",
        })
        _write()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
