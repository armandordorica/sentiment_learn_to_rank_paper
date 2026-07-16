"""Presentation adapter for the interactive Sentiment Lab (Streamlit tab 6)."""

from __future__ import annotations

import sys
import json
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_ltr.models import phrasebank_sentiment as ps  # noqa: E402

RAVENPACK_DIR = PROJECT_ROOT / "data" / "raw" / "news" / "ravenpack"
DEFAULT_START = "2003-01-01"
DEFAULT_END = "2014-12-31"
MODEL_DIR = PROJECT_ROOT / "data" / "models"


@lru_cache(maxsize=12)
def _checkpoint_fingerprint(model_id: str, weights_mtime_ns: int) -> str:
    """Return the full weights SHA-256, preferring the saved provenance record."""
    path = MODEL_DIR / model_id
    provenance_path = path / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        for item in provenance.get("weights", []):
            if item.get("file") == "model.safetensors" and item.get("sha256"):
                return str(item["sha256"])
    digest = hashlib.sha256()
    with (path / "model.safetensors").open("rb") as weights:
        for chunk in iter(lambda: weights.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_context() -> dict[str, Any]:
    deps = ps.finetuning_deps_available()
    has_model = ps.model_is_saved(ps.resolve_model_dir())
    metrics = ps.load_metrics()
    device = ps.device_report() if deps else None
    models = available_models()
    return {"deps_available": deps, "has_model": has_model, "metrics": metrics,
            "device": device, "coverage": coverage_context(), "models": models,
            "default_model_ids": [m["id"] for m in models if m["recommended"]]}


def available_models() -> list[dict[str, Any]]:
    """Discover usable saved checkpoints and attach comparison-friendly metadata."""
    phrasebank_ids = ["phrasebank_distilbert_1ep", "phrasebank_distilbert_best"]
    ravenpack_ids = sorted(
        path.name for path in MODEL_DIR.glob("ravenpack_distilbert*")
        if path.is_dir() and ".bak" not in path.name
    )
    preferred = phrasebank_ids + ravenpack_ids
    models: list[dict[str, Any]] = []
    for model_id in preferred:
        path = MODEL_DIR / model_id
        metrics_path = path / "metrics.json"
        if not (path / "config.json").exists() or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        weights_path = path / "model.safetensors"
        weights_sha = _checkpoint_fingerprint(model_id, weights_path.stat().st_mtime_ns)
        dataset = str(metrics.get("dataset", "unknown"))
        epochs = metrics.get("epochs", "—")
        test = metrics.get("test", {})
        if dataset == "ravenpack_headlines":
            tickers = [str(t).upper() for t in metrics.get("tickers", [])]
            ticker_label = ", ".join(tickers) if tickers else "tickers not recorded"
            stock_count = len(tickers)
            stock_word = "stock" if stock_count == 1 else "stocks"
            title = f"RavenPack fine-tuned — {stock_count} {stock_word} ({ticker_label})"
            row_count = metrics.get("labeled_rows")
            rows_text = f"{int(row_count):,} labeled headlines" if row_count is not None else "labeled headlines"
            description = (f"Started from the best PhraseBank checkpoint, then fine-tuned on "
                           f"{rows_text} for {ticker_label}. This version measures adaptation "
                           f"to a {stock_count}-stock RavenPack training universe.")
        elif model_id.endswith("_1ep"):
            title = "PhraseBank — 1 epoch"
            description = "Early one-epoch baseline trained on Financial PhraseBank."
        else:
            title = "PhraseBank — best checkpoint"
            description = ("Multi-epoch Financial PhraseBank model selected by validation "
                           "macro-F1; the current primary baseline.")
        models.append({
            "id": model_id, "title": title, "description": description,
            "dataset": dataset, "epochs": epochs,
            "test_f1": test.get("eval_f1"), "test_accuracy": test.get("eval_accuracy"),
            "recommended": model_id in {"phrasebank_distilbert_best", "ravenpack_distilbert_best"},
            "tickers": metrics.get("tickers", []),
            "labeled_rows": metrics.get("labeled_rows"),
            "checkpoint_id": model_id,
            "weights_sha256": weights_sha,
            "weights_sha_short": weights_sha[:12],
        })
    return models


def coverage_context() -> dict[str, Any]:
    rich = sorted(RAVENPACK_DIR.glob("*_articles_*.parquet"))
    ref_dir = PROJECT_ROOT / "data" / "raw" / "news" / "refinitiv"
    return {
        "window": f"{DEFAULT_START} → {DEFAULT_END}",
        "ravenpack_files": len(rich),
        "ravenpack_tickers": [p.name.split("_", 1)[0].upper() for p in rich],
        "refinitiv_files": len(list(ref_dir.glob("*.parquet"))) if ref_dir.exists() else 0,
    }


def phrasebank_dataset() -> dict[str, Any]:
    raw = ps.load_phrasebank()
    balance = ps.dataset_class_balance(raw).sort_values("count")
    return {"splits": {k: int(raw[k].num_rows) for k in raw},
            "balance": balance.to_dict(orient="records")}


def cached_articles(ticker: str, start: str, end: str, max_rows: int) -> dict[str, Any]:
    ticker = ticker.strip().upper()
    candidates = [RAVENPACK_DIR / f"{ticker.lower()}_articles_2003_2014.parquet",
                  RAVENPACK_DIR / f"{ticker.lower()}_rp_checkpoint.parquet"]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError(f"No rich RavenPack cache found for {ticker}.")
    df = pd.read_parquet(path)
    date_col = next((c for c in ("article_time", "article_date", "timestamp_utc") if c in df), None)
    if date_col:
        dates = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        df = df[(dates >= pd.Timestamp(start, tz="UTC")) &
                (dates < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))].copy()
        df[date_col] = dates.loc[df.index].astype(str)
        df = df.sort_values(date_col, ascending=False)
    keep = [c for c in (date_col, "headline", "event_text", "news_type", "source_name",
                         "event_sentiment_score", "relevance", "novelty") if c and c in df]
    rows = df[keep].head(max(1, min(max_rows, 500))).fillna("").to_dict(orient="records")
    return {"ticker": ticker, "source": path.name, "rows": rows, "columns": keep}


@lru_cache(maxsize=6)
def _load_model(model_id: str):
    valid = {m["id"] for m in available_models()}
    if model_id not in valid:
        raise ValueError(f"Unknown or unavailable model: {model_id}")
    return ps.load_classifier(MODEL_DIR / model_id)


def score(text: str, model_ids: list[str]) -> dict[str, Any]:
    sentences = [line.strip() for line in text.splitlines() if line.strip()]
    if not sentences:
        raise ValueError("Enter at least one sentence.")
    if not model_ids:
        raise ValueError("Select at least one model version.")
    metadata = {m["id"]: m for m in available_models()}
    results = []
    for model_id in dict.fromkeys(model_ids):
        if model_id not in metadata:
            raise ValueError(f"Unknown or unavailable model: {model_id}")
        tokenizer, model, device = _load_model(model_id)
        rows = ps.predict_sentences(sentences, tokenizer, model, device).to_dict(orient="records")
        results.append({"model": metadata[model_id], "rows": rows})
    comparisons = []
    for row_index, sentence in enumerate(sentences):
        model_scores = []
        for result in results:
            prediction = result["rows"][row_index]
            model_scores.append({
                "model_id": result["model"]["id"],
                "pred": prediction["pred"],
                "negative": prediction["p(negative)"],
                "neutral": prediction["p(neutral)"],
                "positive": prediction["p(positive)"],
            })
        comparisons.append({"sentence": sentence, "scores": model_scores})
    return {"models": [result["model"] for result in results], "comparisons": comparisons}


def train(job: Any) -> dict[str, Any]:
    job.progress_message = "Preparing Financial PhraseBank and DistilBERT…"
    result = ps.train_baseline()
    job.progress_message = "Saving best checkpoint and metrics…"
    return {"metrics": result}
