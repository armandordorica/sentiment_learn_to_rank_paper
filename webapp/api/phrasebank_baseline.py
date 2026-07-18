"""PhraseBank HF Baseline (Tab 3): training metrics + on-demand train-split eval.

Wraps ``src/sentiment_ltr/models/phrasebank_sentiment.py`` — reuses
``evaluate_checkpoint_on_split()``, the exact same function the Streamlit tab's
"Performance on the training set" button calls, so both UIs report identical
numbers from one code path.
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
from sentiment_ltr.wandb_logging import checkpoint_wandb_links  # noqa: E402
from sentiment_ltr.viz import (  # noqa: E402
    horizontal_bar_figure,
    split_series_distribution_figures,
    vertical_bar_figure,
)

model_is_saved = _phrasebank_sentiment.model_is_saved
resolve_model_dir = _phrasebank_sentiment.resolve_model_dir
finetuning_deps_available = _phrasebank_sentiment.finetuning_deps_available
load_metrics = _phrasebank_sentiment.load_metrics
evaluate_checkpoint_on_split = _phrasebank_sentiment.evaluate_checkpoint_on_split
load_phrasebank = _phrasebank_sentiment.load_phrasebank
dataset_class_balance = _phrasebank_sentiment.dataset_class_balance
phrasebank_probability_chart_frame = _phrasebank_sentiment.phrasebank_probability_chart_frame
PHRASEBANK_SPLIT_ORDER = _phrasebank_sentiment.PHRASEBANK_SPLIT_ORDER
MODEL_NAME = _phrasebank_sentiment.MODEL_NAME

# Simple process-wide caches — mirror the Streamlit tab's st.cache_data /
# st.cache_resource decorators (small static dataset + checkpoint-mtime token).
_split_eval_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
_dataset_summary_cache: dict[str, Any] | None = None
_probability_chart_cache: dict[str, Any] = {}


def _plotly_to_html_div(fig) -> str:
    """Render a Plotly figure to an embeddable <div> (no full HTML doc, CDN JS)."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def deps_status() -> dict[str, Any]:
    return {
        "finetuning_deps_available": finetuning_deps_available(),
        "has_phrasebank_checkpoint": model_is_saved(resolve_model_dir()),
    }


def _checkpoint_cache_token() -> str:
    model_dir = resolve_model_dir()
    config_path = model_dir / "config.json"
    if config_path.exists():
        return str(config_path.stat().st_mtime)
    return "no-checkpoint"


def training_summary() -> dict[str, Any]:
    """Mirrors Streamlit 3A/3C: headline training config + val/test metrics."""
    metrics = load_metrics()
    return {
        "model_name": str(metrics.get("model_name", MODEL_NAME)).split("/")[-1],
        "epochs": metrics.get("epochs"),
        "learning_rate": metrics.get("learning_rate"),
        "batch_size": metrics.get("per_device_train_batch_size"),
        "device": str(metrics.get("device", "—")).upper(),
        "val_f1": metrics.get("validation", {}).get("eval_f1"),
        "val_acc": metrics.get("validation", {}).get("eval_accuracy"),
        "test_f1": metrics.get("test", {}).get("eval_f1"),
        "test_acc": metrics.get("test", {}).get("eval_accuracy"),
        "train_loss": metrics.get("train_loss"),
        "raw_metrics": metrics,
        "wandb": checkpoint_wandb_links("phrasebank_distilbert_best", metrics),
    }


def train_split_eval(*, use_cache: bool = True) -> dict[str, Any]:
    """Score the saved checkpoint on the PhraseBank **train** split (on demand).

    Same underlying call as the Streamlit "▶ Evaluate on train split" button
    (``evaluate_checkpoint_on_split("train", ...)``) — identical numbers by
    construction, not by coincidence.
    """
    model_dir = resolve_model_dir()
    token = _checkpoint_cache_token()
    cache_key = (str(model_dir), "train", token)

    if use_cache and cache_key in _split_eval_cache:
        return _split_eval_cache[cache_key]

    result = evaluate_checkpoint_on_split("train", model_dir=model_dir)

    # Reshape pandas objects into plain JSON/Jinja2-friendly structures.
    cm = result["confusion_counts"]
    cm_pct = result["confusion_pct"]
    shaped = {
        "model_dir": result["model_dir"],
        "split": result["split"],
        "n_rows": result["n_rows"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "classification_report": result["classification_report"],
        "label_order": result["label_order"],
        "confusion_counts": {
            "labels": result["label_order"],
            "rows": [
                {"actual": actual, **{pred: int(cm.loc[actual, pred]) for pred in result["label_order"]}}
                for actual in result["label_order"]
            ],
        },
        "confusion_pct": {
            "labels": result["label_order"],
            "rows": [
                {"actual": actual, **{pred: round(float(cm_pct.loc[actual, pred]), 1) for pred in result["label_order"]}}
                for actual in result["label_order"]
            ],
        },
        "mismatches_sample": result["mismatches_sample"].to_dict(orient="records"),
    }
    _split_eval_cache[cache_key] = shaped
    return shaped


def dataset_dashboard() -> dict[str, Any]:
    """Mirrors Streamlit 3D "Gold labels & splits": bar chart HTML + split table.

    Uses the exact same ``dataset_class_balance()`` + ``horizontal_bar_figure()``
    / ``vertical_bar_figure()`` calls as ``app.py``'s
    ``render_phrasebank_hf_baseline_tab()``, so the charts are pixel-identical
    (same Plotly figure, just embedded via ``fig.to_html()`` instead of
    ``st.plotly_chart()``).
    """
    global _dataset_summary_cache
    if _dataset_summary_cache is not None:
        return _dataset_summary_cache

    raw = load_phrasebank()
    balance = dataset_class_balance(raw)
    splits = {name: int(raw[name].num_rows) for name in raw}

    import pandas as pd

    split_df = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "rows": [splits.get("train", 0), splits.get("validation", 0), splits.get("test", 0)],
    })

    fig_balance = horizontal_bar_figure(
        balance.sort_values("count"),
        x="count",
        y="label",
        title="Gold label balance (train split)",
        axis_labels={"label": "Class", "count": "Train rows"},
        x_hover_label="Count",
        y_hover_label="Class",
    )
    fig_splits = vertical_bar_figure(
        split_df,
        x="split",
        y="rows",
        title="Dataset size by split",
        color="split",
        axis_labels={"split": "Split", "rows": "Sentences"},
        x_hover_label="Split",
        y_hover_label="Rows",
    )

    result = {
        "train_rows": splits.get("train", 0),
        "val_rows": splits.get("validation", 0),
        "test_rows": splits.get("test", 0),
        "total_rows": sum(splits.values()),
        "balance_chart_html": _plotly_to_html_div(fig_balance),
        "splits_chart_html": _plotly_to_html_div(fig_splits),
    }
    _dataset_summary_cache = result
    return result


def probability_charts() -> dict[str, Any] | None:
    """Mirrors Streamlit 3D "Predicted probabilities on each split": box + median-bar charts.

    Requires a saved checkpoint (scores every PhraseBank sentence). Returns
    ``None`` when no checkpoint is saved, matching the Streamlit tab's guard.
    """
    if not model_is_saved(resolve_model_dir()):
        return None

    token = _checkpoint_cache_token()
    if token in _probability_chart_cache:
        return _probability_chart_cache[token]

    long_probs = phrasebank_probability_chart_frame()
    chart_orders = {
        "split": PHRASEBANK_SPLIT_ORDER,
        "class": ["negative", "neutral", "positive"],
    }
    chart_labels = {
        "split": "PhraseBank split",
        "probability": "Predicted probability",
        "p50": "Median predicted probability",
        "class": "Class",
    }
    fig_box, fig_p50, p50 = split_series_distribution_figures(
        long_probs,
        x="split",
        y="probability",
        series="class",
        category_orders=chart_orders,
        axis_labels=chart_labels,
        box_title="Class probabilities by split (box & whisker)",
        median_title="Median class probability by split (p50)",
        median_col="p50",
        x_hover_label="PhraseBank split",
        series_hover_label="Class",
        y_hover_label="Predicted probability",
        median_y_hover_label="Median predicted probability",
    )

    p50_table = p50.pivot(index="split", columns="class", values="p50").reset_index()
    result = {
        "box_chart_html": _plotly_to_html_div(fig_box),
        "median_chart_html": _plotly_to_html_div(fig_p50),
        "p50_columns": list(p50_table.columns),
        "p50_rows": p50_table.to_dict(orient="records"),
    }
    # Only cache once per checkpoint; clear the whole dict since it's keyed by token.
    _probability_chart_cache.clear()
    _probability_chart_cache[token] = result
    return result
