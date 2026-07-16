"""Presentation adapter for the interactive Sentiment Lab (Streamlit tab 6)."""

from __future__ import annotations

import sys
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


def page_context() -> dict[str, Any]:
    deps = ps.finetuning_deps_available()
    has_model = ps.model_is_saved(ps.resolve_model_dir())
    metrics = ps.load_metrics()
    device = ps.device_report() if deps else None
    return {"deps_available": deps, "has_model": has_model, "metrics": metrics,
            "device": device, "coverage": coverage_context()}


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


def score(text: str) -> list[dict[str, Any]]:
    sentences = [line.strip() for line in text.splitlines() if line.strip()]
    if not sentences:
        raise ValueError("Enter at least one sentence.")
    tokenizer, model, device = ps.load_classifier()
    return ps.predict_sentences(sentences, tokenizer, model, device).to_dict(orient="records")


def train(job: Any) -> dict[str, Any]:
    job.progress_message = "Preparing Financial PhraseBank and DistilBERT…"
    result = ps.train_baseline()
    job.progress_message = "Saving best checkpoint and metrics…"
    return {"metrics": result}
