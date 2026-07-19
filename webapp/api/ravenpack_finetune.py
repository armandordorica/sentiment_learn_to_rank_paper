"""RavenPack fine-tuning (Tab 5·8): train/re-train on 1 / 5 / N tickers.

This module wraps ``src/sentiment_ltr/models/ravenpack_sentiment.py`` — no
business logic is duplicated here, only the data shaping needed for the
FastAPI/Jinja2 presentation layer (mirrors Streamlit's ``app.py``
``render_ravenpack_finetuning_tab`` → section 5·8).
"""

from __future__ import annotations

import json
import hashlib
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any

import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_ltr.models import phrasebank_sentiment as _phrasebank_sentiment  # noqa: E402
from sentiment_ltr.models import ravenpack_sentiment as _ravenpack_sentiment  # noqa: E402
from sentiment_ltr.wandb_logging import checkpoint_wandb_links  # noqa: E402
from webapp.api import sentiment_lab as _sentiment_lab  # noqa: E402
from webapp.api.sentiment_lab import available_models as _available_models  # noqa: E402

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
split_leakage_audit = _ravenpack_sentiment.split_leakage_audit
DEFAULT_FIVE_STOCK_TICKERS = _ravenpack_sentiment.DEFAULT_FIVE_STOCK_TICKERS
DEFAULT_TWENTY_STOCK_TICKERS = DEFAULT_FIVE_STOCK_TICKERS + [
    "C", "GS", "AIG", "ORCL", "QCOM", "HPQ", "IBM", "T", "WMT", "KO",
    "MCD", "AMZN", "CAT", "BA", "CVX",
]
DEFAULT_OOD_BASKETS = {
    "Basket 1": ["BAC", "INTC", "GE", "F", "PFE"],
    "Basket 2": ["CSCO", "WFC", "GM", "VZ", "CMCSA"],
    "Basket 3": ["BMY", "TSN", "CSX", "EBAY", "MS"],
}
DEFAULT_OOD_TICKERS = [ticker for basket in DEFAULT_OOD_BASKETS.values() for ticker in basket]
OOD_BENCHMARK_CACHE_VERSION = 1

# The pooled pilot set is the single source of truth in ravenpack_sentiment
# (DEFAULT_FIVE_STOCK_TICKERS = AAPL, MSFT, JPM, XOM, JNJ).
PILOT_TICKERS = DEFAULT_FIVE_STOCK_TICKERS

# "Rich" RavenPack exports live in data/raw/news/ravenpack/ as
# `{ticker}_articles_*.parquet` and carry the `headline` column the model
# classifies. The data_explorer_top1k batch exports only have sentiment scores
# (no headline) and would train on zero rows, so only rich exports are
# trainable. This set is discovered from disk, so dropping a new
# `{ticker}_articles_*.parquet` there makes that ticker selectable/trainable
# with no code change — that's how the 5-ticker pilot scales past AAPL.
RAVENPACK_NEWS_DIR = _ravenpack_sentiment.RAVENPACK_NEWS_DIR


def rich_export_tickers() -> list[str]:
    """Tickers with a headline-bearing rich RavenPack export on disk (trainable)."""
    if not RAVENPACK_NEWS_DIR.exists():
        return []
    return sorted({
        p.name.split("_articles_")[0].upper()
        for p in RAVENPACK_NEWS_DIR.glob("*_articles_*.parquet")
    })


def available_tickers() -> list[str]:
    """Offer the five-stock targets even before all rich exports are prepared.

    Keeping targets visible lets the preset select all five and makes readiness
    explicit in the UI. The loader still blocks training on metadata-only files.
    """
    return sorted(set(rich_export_tickers()) | set(DEFAULT_FIVE_STOCK_TICKERS))


def pilot_default_tickers(available: list[str]) -> list[str]:
    """Default selection: the pilot set, restricted to trainable tickers present.

    AAPL alone until the other pilot exports (MSFT/JPM/XOM/JNJ) are added, then
    the full pooled five-stock pilot preselects automatically.
    """
    ready = set(rich_export_tickers())
    return [t for t in PILOT_TICKERS if t in available and t in ready]


def five_stock_readiness() -> list[dict[str, Any]]:
    """Whether each pooled-pilot stock has a headline-bearing rich export."""
    import pyarrow.parquet as pq

    rows = []
    for ticker in DEFAULT_FIVE_STOCK_TICKERS:
        candidates = discover_ravenpack_article_files([ticker])
        rich_path = None
        for path in candidates:
            try:
                if "headline" in pq.read_schema(path).names:
                    rich_path = path
                    break
            except Exception:
                continue
        rows.append({"ticker": ticker, "ready": rich_path is not None,
                     "path": str(rich_path.relative_to(PROJECT_ROOT)) if rich_path else None})
    return rows


def deps_status() -> dict[str, Any]:
    return {
        "finetuning_deps_available": finetuning_deps_available(),
        "has_phrasebank_checkpoint": model_is_saved(resolve_model_dir()),
        "has_ravenpack_checkpoint": ravenpack_model_is_saved(),
    }


def wandb_context() -> dict[str, Any]:
    metrics = {}
    metrics_path = resolve_ravenpack_model_dir() / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return checkpoint_wandb_links("ravenpack_distilbert_best", metrics)


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
    leakage = split_leakage_audit(labeled)
    assigned = assign_time_split(labeled["article_date"])
    date_ranges: dict[str, dict[str, str | None]] = {}
    for split_name in ("train", "validation", "test"):
        dates = labeled.loc[assigned == split_name, "article_date"].dropna()
        date_ranges[split_name] = {
            "start_date": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
            "end_date": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
        }

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
        "splits_table": [
            {**row, **date_ranges[row["split"]]}
            for row in splits.to_dict(orient="records")
        ],
        "per_ticker": per_ticker,
        "leakage": leakage,
    }


@lru_cache(maxsize=1)
def twenty_stock_coverage() -> dict[str, Any]:
    """Cached audited coverage for the explicit 20-stock expansion universe."""
    return coverage_summary(DEFAULT_TWENTY_STOCK_TICKERS)


def comparison_models() -> list[dict[str, Any]]:
    """The research checkpoints evaluated on one shared test universe."""
    candidates = [
        ("phrasebank_distilbert_best", "PhraseBank — out-of-the-box", [],
         "Financial PhraseBank checkpoint with no RavenPack adaptation."),
        ("ravenpack_distilbert_best.bak-20260705-checkpoint", "RavenPack fine-tuned — 1 stock (AAPL)", ["AAPL"],
         "PhraseBank checkpoint fine-tuned only on AAPL RavenPack headlines."),
        ("ravenpack_distilbert_5stock", "RavenPack fine-tuned — 5 stocks", DEFAULT_FIVE_STOCK_TICKERS,
         "PhraseBank checkpoint jointly fine-tuned on AAPL, MSFT, JPM, XOM, and JNJ."),
        ("ravenpack_distilbert_best", "RavenPack fine-tuned — 20 stocks", DEFAULT_TWENTY_STOCK_TICKERS,
         "PhraseBank checkpoint jointly fine-tuned on the pooled 20-stock expansion universe."),
    ]
    discovered = {model["id"]: model for model in _available_models()}
    models = []
    for model_id, title, trained_tickers, description in candidates:
        path = PROJECT_ROOT / "data" / "models" / model_id
        weights = path / "model.safetensors"
        if not model_is_saved(path) or not weights.exists():
            continue
        known = discovered.get(model_id)
        sha = known["weights_sha_short"] if known else _sentiment_lab._checkpoint_fingerprint(
            model_id, weights.stat().st_mtime_ns,
        )[:12]
        models.append({
            "id": model_id, "title": title, "sha": sha,
            "description": description, "trained_tickers": trained_tickers,
        })
    return models


def compare_checkpoints(
    tickers: list[str], model_ids: list[str] | None = None, *, job: Any | None = None,
) -> dict[str, Any]:
    """Evaluate the research checkpoints against identical held-out test rows."""
    allowed = {model["id"]: model for model in comparison_models()}
    selected = list(dict.fromkeys(model_ids or list(allowed)))
    if len(selected) < 2 or any(model_id not in allowed for model_id in selected):
        raise ValueError("Select at least two available saved checkpoints.")
    results: list[dict[str, Any]] = []
    eval_tickers = {str(t).upper() for t in tickers}
    for index, model_id in enumerate(selected, start=1):
        model = allowed[model_id]
        if job is not None:
            job.progress_message = f"Evaluating {index}/{len(selected)}: {model['title']} on the test set…"
        evaluated = _ravenpack_sentiment.evaluate_phrasebank_baseline_on_ravenpack(
            tickers, model_dir=PROJECT_ROOT / "data" / "models" / model_id,
            eval_split="test",
        )
        trained = set(model["trained_tickers"])
        seen = sorted(eval_tickers & trained)
        unseen = sorted(eval_tickers - trained)
        if not trained:
            domain_label = "Dataset OOD: no RavenPack training"
        elif not seen:
            domain_label = "Strict cross-stock OOD"
        elif unseen:
            domain_label = f"Mixed: {len(seen)} seen, {len(unseen)} unseen tickers"
        else:
            domain_label = "Held-out, in-universe tickers"
        results.append({
            **model,
            "n_rows": evaluated["n_rows"],
            "macro_f1": evaluated["macro_f1"],
            "accuracy": evaluated["accuracy"],
            "per_ticker": evaluated.get("per_ticker", []),
            "domain_label": domain_label,
            "seen_tickers": seen,
            "unseen_tickers": unseen,
        })
    baseline = results[0]
    best = max(results, key=lambda row: row["macro_f1"])
    return {
        "models": results,
        "baseline": baseline,
        "best": best,
        "same_test_rows": len({row["n_rows"] for row in results}) == 1,
        "tickers": tickers,
    }


def _comparison_chart(baskets: list[dict[str, Any]], averages: list[dict[str, Any]], metric: str) -> str:
    labels = [basket["name"] for basket in baskets] + ["3-basket average"]
    fig = go.Figure()
    for model_index, average in enumerate(averages):
        values = [basket["models"][model_index][metric] * 100 for basket in baskets]
        values.append(average[metric] * 100)
        fig.add_bar(name=average["title"], x=labels, y=values, text=[f"{v:.1f}%" for v in values], textposition="auto")
    label = "Macro-F1" if metric == "macro_f1" else "Accuracy"
    fig.update_layout(
        barmode="group", height=460, title=f"Strict OOD {label} by basket",
        yaxis={"title": label, "ticksuffix": "%", "range": [0, 100], "gridcolor": "#d8dee8"},
        plot_bgcolor="#fff", paper_bgcolor="#fff", legend={"orientation": "h", "y": -0.2},
        margin={"t": 60, "b": 120},
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _ticker_heatmap(baskets: list[dict[str, Any]], models: list[dict[str, Any]]) -> str:
    tickers = [ticker for basket in baskets for ticker in basket["tickers"]]
    by_model: dict[str, dict[str, float]] = {model["id"]: {} for model in models}
    for basket in baskets:
        for model in basket["models"]:
            by_model[model["id"]].update({row["ticker"]: row["macro_f1"] * 100 for row in model["per_ticker"]})
    z = [[by_model[model["id"]].get(ticker) for ticker in tickers] for model in models]
    fig = go.Figure(go.Heatmap(
        z=z, x=tickers, y=[model["title"] for model in models],
        colorscale="RdYlGn", zmin=0, zmax=100,
        text=[[f"{value:.1f}%" if value is not None else "—" for value in row] for row in z],
        texttemplate="%{text}", hovertemplate="%{y}<br>%{x}: %{z:.1f}% macro-F1<extra></extra>",
        colorbar={"title": "Macro-F1"},
    ))
    fig.update_layout(height=390, title="Per-stock OOD macro-F1", margin={"t": 60, "l": 230})
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _ood_cache_path(model_ids: list[str] | None) -> Path:
    """Return a content-addressed cache path for one benchmark definition.

    Checkpoint SHAs invalidate stale scores after retraining. Source file size and
    modification time invalidate them when any OOD dataset is refreshed.
    """
    available = {model["id"]: model for model in comparison_models()}
    selected_ids = list(dict.fromkeys(model_ids or list(available)))
    model_fingerprints = [
        {"id": model_id, "sha": available.get(model_id, {}).get("sha")}
        for model_id in selected_ids
    ]
    source_fingerprints = []
    for path in discover_ravenpack_article_files(DEFAULT_OOD_TICKERS):
        stat = path.stat()
        source_fingerprints.append({
            "path": str(path.relative_to(PROJECT_ROOT)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        })
    payload = {
        "version": OOD_BENCHMARK_CACHE_VERSION,
        "models": model_fingerprints,
        "baskets": DEFAULT_OOD_BASKETS,
        "sources": source_fingerprints,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:20]
    return PROJECT_ROOT / "data" / "models" / "_benchmark_cache" / f"ood_{digest}.json"


def _read_ood_cache(path: Path) -> dict[str, Any] | None:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    result["cache_hit"] = True
    return result


def _write_ood_cache(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(path)


def compare_ood_baskets(
    model_ids: list[str] | None = None, *, job: Any | None = None, use_cache: bool = True,
) -> dict[str, Any]:
    """Evaluate all checkpoints on three disjoint, never-trained stock baskets."""
    cache_path = _ood_cache_path(model_ids) if use_cache else None
    if cache_path is not None and (cached := _read_ood_cache(cache_path)) is not None:
        if job is not None:
            job.progress_message = "Loaded the matching benchmark from cache."
        return cached

    basket_results = []
    total = len(DEFAULT_OOD_BASKETS)
    for index, (name, tickers) in enumerate(DEFAULT_OOD_BASKETS.items(), start=1):
        if job is not None:
            job.progress_message = f"Evaluating OOD basket {index}/{total}: {', '.join(tickers)}…"
        result = compare_checkpoints(tickers, model_ids, job=None)
        basket_results.append({"name": name, "tickers": tickers, **result})

    averages = []
    for model_index, model in enumerate(basket_results[0]["models"]):
        averages.append({
            **{key: model[key] for key in ("id", "title", "sha", "description")},
            "macro_f1": mean(basket["models"][model_index]["macro_f1"] for basket in basket_results),
            "accuracy": mean(basket["models"][model_index]["accuracy"] for basket in basket_results),
            "total_rows": sum(basket["models"][model_index]["n_rows"] for basket in basket_results),
        })
    baseline = averages[0]
    best = max(averages, key=lambda row: row["macro_f1"])
    result = {
        "baskets": basket_results,
        "averages": averages,
        "baseline": baseline,
        "best": best,
        "charts": {
            "macro_f1": _comparison_chart(basket_results, averages, "macro_f1"),
            "accuracy": _comparison_chart(basket_results, averages, "accuracy"),
            "ticker_heatmap": _ticker_heatmap(basket_results, averages),
        },
        "cache_hit": False,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    if cache_path is not None:
        _write_ood_cache(cache_path, result)
    return result


device_report = _phrasebank_sentiment.device_report

FINETUNE_WORKER = PROJECT_ROOT / "scripts" / "finetune_worker.py"
_FINETUNE_RUN_DIR = PROJECT_ROOT / "data" / "models" / "_finetune_runs"


def run_training(
    tickers: list[str],
    *,
    init_from_phrasebank: bool,
    num_train_epochs: int,
    job: Any | None = None,
    resume_from_checkpoint: str | None = None,
    wandb_run_id: str | None = None,
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
    control_file = _FINETUNE_RUN_DIR / f"{run_id}_control.json"
    log_file = _FINETUNE_RUN_DIR / f"{run_id}.log"
    status_file.unlink(missing_ok=True)
    metrics_out.unlink(missing_ok=True)
    control_file.unlink(missing_ok=True)

    cmd = [
        sys.executable, str(FINETUNE_WORKER),
        "--status-file", str(status_file),
        "--metrics-out", str(metrics_out),
        "--control-file", str(control_file),
        "--tickers", *tickers,
        "--epochs", str(num_train_epochs),
    ]
    if init_from_phrasebank:
        cmd.append("--init-from-phrasebank")
    if resume_from_checkpoint:
        cmd.extend(["--resume-from-checkpoint", resume_from_checkpoint])
    if wandb_run_id:
        cmd.extend(["--wandb-run-id", wandb_run_id])

    if job is not None:
        job.progress_message = "Launching training subprocess…"

    # start_new_session detaches the worker into its own process group so a
    # dev-server auto-reload (or the JobManager thread dying) can't kill an
    # in-flight training run — its status file on disk stays the source of truth.
    # Keep output in a durable file. A server restart closes its terminal pipe;
    # inheriting that pipe previously caused tqdm/W&B BrokenPipeError mid-run.
    log_handle = log_file.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT_ROOT), start_new_session=True,
        stdout=log_handle, stderr=subprocess.STDOUT,
    )

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
    log_handle.close()

    if proc.returncode != 0:
        err = (last or {}).get("error") or f"Training subprocess exited with code {proc.returncode}."
        raise RuntimeError(err)

    if (last or {}).get("status") == "paused":
        return {
            "paused": True,
            "checkpoint_path": last.get("checkpoint_path"),
            "step": last.get("step"),
            "total_steps": last.get("total_steps"),
            "tickers": tickers,
            "epochs": num_train_epochs,
            "init_from_phrasebank": init_from_phrasebank,
            "wandb_run_id": last.get("wandb_run_id"),
        }

    metrics = json.loads(metrics_out.read_text(encoding="utf-8"))
    return {
        "metrics": metrics,
        "test_f1": metrics.get("test", {}).get("eval_f1"),
        "test_acc": metrics.get("test", {}).get("eval_accuracy"),
        "per_ticker_test": _per_ticker_rows(metrics.get("per_ticker_test")),
        "device": metrics.get("device"),
        "checkpoint_dir": str(DEFAULT_RAVENPACK_MODEL_DIR.relative_to(PROJECT_ROOT)),
        "wandb_run_url": metrics.get("wandb_run_url"),
        "wandb_project_url": metrics.get("wandb_project_url"),
    }


def _per_ticker_rows(per_ticker_test: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Flatten metrics.json's ``per_ticker_test`` map into template-friendly rows.

    Sorted worst-F1 first so a ticker suffering negative transfer surfaces at the
    top of the breakdown table. Empty for single-ticker runs (never computed).
    """
    if not per_ticker_test:
        return []
    rows = [
        {
            "ticker": ticker,
            "test_rows": stats.get("test_rows"),
            "macro_f1": stats.get("macro_f1"),
            "accuracy": stats.get("accuracy"),
        }
        for ticker, stats in per_ticker_test.items()
    ]
    rows.sort(key=lambda r: (r["macro_f1"] is None, r["macro_f1"] if r["macro_f1"] is not None else 0.0))
    return rows


# ── Refresh-safe run recovery ─────────────────────────────────────────────────
# Every run writes a status file to _FINETUNE_RUN_DIR (see run_training). Reading
# that file lets the fine-tune page rebuild the live status after a browser
# refresh or even a webapp restart, so a long training run is never "lost".


def latest_run_id() -> str | None:
    """Newest completed/error run or genuinely active worker run."""
    if not _FINETUNE_RUN_DIR.exists():
        return None
    files = sorted(_FINETUNE_RUN_DIR.glob("*_status.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        job_id = path.name[: -len("_status.json")]
        state = read_run_state(job_id)
        if state is None:
            continue
        if state.get("status") == "running" and not _running_state_is_live(state, path):
            continue
        return job_id
    return None


def _running_state_is_live(state: dict[str, Any], path: Path) -> bool:
    """True when a running status belongs to a live worker or was just updated."""
    import os
    import time

    pid = state.get("worker_pid")
    if pid:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, ValueError):
            return False
    # Compatibility for older workers that did not persist a PID: they write
    # every step, so a recent file is active; an old one is stale.
    return time.time() - path.stat().st_mtime < 120


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
    status_path = _FINETUNE_RUN_DIR / f"{job_id}_status.json"
    if disk_status == "running" and not _running_state_is_live(state, status_path):
        disk_status = "error"
        state["message"] = "This run stopped updating and no active training worker was found."
        state["error"] = state["message"]
    status = disk_status if disk_status in ("done", "error", "paused") else "running"
    result = None
    if disk_status == "done":
        wandb = wandb_context()
        # The status file doesn't carry the per-ticker breakdown; read it back
        # from the persisted metrics.json (authoritative for the saved checkpoint)
        # so it survives a browser refresh / webapp restart.
        saved_metrics = _ravenpack_sentiment.load_ravenpack_metrics() or {}
        result = {
            "test_f1": state.get("test_f1"),
            "test_acc": state.get("test_acc"),
            "per_ticker_test": _per_ticker_rows(saved_metrics.get("per_ticker_test")),
            "device": state.get("device"),
            "checkpoint_dir": str(DEFAULT_RAVENPACK_MODEL_DIR.relative_to(PROJECT_ROOT)),
            "wandb_run_url": state.get("wandb_run_url") or wandb.get("run_url"),
            "wandb_project_url": state.get("wandb_project_url") or wandb.get("project_url"),
        }
    return SimpleNamespace(
        id=job_id,
        status=status,
        progress=state,
        progress_message=state.get("message", ""),
        result=result,
        error=state.get("error"),
    )


def request_pause(job_id: str) -> SimpleNamespace | None:
    """Request a graceful checkpoint-and-stop at the end of the current step."""
    state = read_run_state(job_id)
    if state is None or state.get("status") != "running":
        return run_view(job_id)
    control = _FINETUNE_RUN_DIR / f"{job_id}_control.json"
    control.write_text(json.dumps({"action": "pause"}), encoding="utf-8")
    state["message"] = "Pause requested — saving after the current step…"
    return SimpleNamespace(
        id=job_id, status="running", progress=state,
        progress_message=state["message"], result=None, error=None,
    )


def loss_chart(progress: dict[str, Any] | None) -> dict[str, Any] | None:
    """Geometry for the live training-loss chart, drawn as inline SVG.

    Returns pixel coordinates (in a fixed 640×230 viewBox) so the template can
    render a self-contained line chart with no JS/Plotly — it simply redraws on
    each HTMX status poll. ``None`` until at least two loss points exist.
    """
    if not progress:
        return None
    history = progress.get("loss_history") or []
    if len(history) < 2:
        return None

    steps = [int(h["step"]) for h in history]
    losses = [float(h["loss"]) for h in history]
    total = int(progress.get("total_steps") or 0) or max(steps)
    lo, hi = min(losses), max(losses)
    if hi <= lo:
        hi = lo + 1e-6
    pad = (hi - lo) * 0.10
    lo_p, hi_p = lo - pad, hi + pad

    W, H = 640, 230
    L, R, T, B = 52, 14, 12, 26  # margins: L leaves room for y labels, B for x
    pw, ph = W - L - R, H - T - B

    def px(step: float) -> float:
        return L + (step / total) * pw if total else L

    def py(val: float) -> float:
        return T + (1 - (val - lo_p) / (hi_p - lo_p)) * ph

    polyline = " ".join(f"{px(s):.1f},{py(v):.1f}" for s, v in zip(steps, losses))

    # Three horizontal gridlines: top = max loss, bottom = min loss.
    yticks = []
    for frac in (0.0, 0.5, 1.0):
        val = hi_p - frac * (hi_p - lo_p)
        yticks.append({"y": round(py(val), 1), "label": f"{val:.3f}"})

    # Dashed vertical lines at epoch boundaries.
    epochs = int(progress.get("epochs") or 0)
    epoch_marks = []
    if epochs > 1 and total:
        for i in range(1, epochs):
            s = total * i / epochs
            epoch_marks.append({"x": round(px(s), 1), "label": f"end e{i}"})

    return {
        "width": W, "height": H,
        "plot_left": L, "plot_right": W - R, "plot_top": T, "plot_bottom": H - B,
        "polyline": polyline,
        "yticks": yticks,
        "epoch_marks": epoch_marks,
        "x_max_label": f"{total:,}",
        "last_loss": losses[-1],
        "last_step": steps[-1],
        "n_points": len(history),
        "eval_history": progress.get("eval_history") or [],
    }
