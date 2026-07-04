"""Backfill saved PhraseBank metrics.json snapshots into W&B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import wandb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT = "sentiment-ltr-transformers"

SNAPSHOTS = {
    "phrasebank_distilbert_best": PROJECT_ROOT / "data" / "models" / "phrasebank_distilbert_best",
    "phrasebank_distilbert_1ep": PROJECT_ROOT / "data" / "models" / "phrasebank_distilbert_1ep",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {
        "train/loss": float(metrics["train_loss"]),
        "train/runtime_s": float(metrics["train_runtime_s"]),
        "train/epoch": float(metrics["epochs"]),
    }
    for split in ("validation", "test"):
        split_metrics = metrics.get(split, {})
        for key, value in split_metrics.items():
            if isinstance(value, (int, float)):
                clean_key = key.removeprefix("eval_")
                out[f"{split}/{clean_key}"] = float(value)
    return out


def _config_from_metrics(metrics: dict[str, Any], model_dir: Path) -> dict[str, Any]:
    config: dict[str, Any] = {
        "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
        "model_name": metrics.get("model_name"),
        "dataset": metrics.get("dataset"),
        "split_source": metrics.get("split_source"),
        "epochs": metrics.get("epochs"),
        "learning_rate": metrics.get("learning_rate"),
        "per_device_train_batch_size": metrics.get("per_device_train_batch_size"),
        "max_length": metrics.get("max_length"),
        "metric_for_best_model": metrics.get("metric_for_best_model"),
        "device": metrics.get("device"),
        "saved_at": metrics.get("saved_at"),
        "import_source": "local metrics.json",
    }

    provenance_path = model_dir / "provenance.json"
    if provenance_path.exists():
        provenance = _load_json(provenance_path)
        config["git"] = provenance.get("git")
        config["data"] = provenance.get("data")
        config["weights"] = provenance.get("weights")
        config["tokenizer"] = provenance.get("tokenizer")
    return config


def import_snapshot(project: str, snapshot_name: str, model_dir: Path) -> str:
    metrics_path = model_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json found at {metrics_path}")

    metrics = _load_json(metrics_path)
    run_name = (
        f"{snapshot_name}-offline-metrics"
        f"-epochs{metrics.get('epochs')}"
        f"-lr{metrics.get('learning_rate'):g}"
        f"-batch{metrics.get('per_device_train_batch_size')}"
    )
    run = wandb.init(
        project=project,
        name=run_name,
        job_type="metrics-import",
        tags=["phrasebank", "distilbert", "offline-metrics", snapshot_name],
        config=_config_from_metrics(metrics, model_dir),
    )

    flat_metrics = _flatten_metrics(metrics)
    wandb.log(flat_metrics)
    run.summary.update(flat_metrics)
    run.summary["local_metrics_path"] = str(metrics_path.relative_to(PROJECT_ROOT))

    artifact = wandb.Artifact(
        name=f"{snapshot_name}-metrics",
        type="metrics",
        metadata={
            "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
            "saved_at": metrics.get("saved_at"),
            "epochs": metrics.get("epochs"),
        },
    )
    artifact.add_file(str(metrics_path), name="metrics.json")
    provenance_path = model_dir / "provenance.json"
    if provenance_path.exists():
        artifact.add_file(str(provenance_path), name="provenance.json")
    run.log_artifact(artifact)

    url = run.url
    run.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument(
        "--snapshot",
        choices=[*SNAPSHOTS.keys(), "all"],
        default="all",
    )
    args = parser.parse_args()

    selected = SNAPSHOTS if args.snapshot == "all" else {args.snapshot: SNAPSHOTS[args.snapshot]}
    for snapshot_name, model_dir in selected.items():
        url = import_snapshot(args.project, snapshot_name, model_dir)
        print(f"{snapshot_name}: {url}")


if __name__ == "__main__":
    main()
