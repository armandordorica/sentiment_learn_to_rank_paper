"""RavenPack fine-tuning (Tab 5·8): train/re-train on 1 / 5 / N tickers.

This module wraps ``src/sentiment_ltr/models/ravenpack_sentiment.py`` — no
business logic is duplicated here, only the data shaping needed for the
FastAPI/Jinja2 presentation layer (mirrors Streamlit's ``app.py``
``render_ravenpack_finetuning_tab`` → section 5·8).
"""

from __future__ import annotations

import sys
from pathlib import Path
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


def run_training(
    tickers: list[str],
    *,
    init_from_phrasebank: bool,
    num_train_epochs: int,
) -> dict[str, Any]:
    """Blocking call — run inside a background job thread (see ``webapp/jobs.py``)."""
    metrics = train_ravenpack(
        tickers=tickers,
        init_from_phrasebank=init_from_phrasebank,
        num_train_epochs=num_train_epochs,
    )
    test_f1 = metrics.get("test", {}).get("eval_f1")
    test_acc = metrics.get("test", {}).get("eval_accuracy")
    return {
        "metrics": metrics,
        "test_f1": test_f1,
        "test_acc": test_acc,
        "checkpoint_dir": str(DEFAULT_RAVENPACK_MODEL_DIR.relative_to(PROJECT_ROOT)),
    }
