"""FastAPI adapter for Streamlit Tab 4: RavenPack Baseline Evaluation.

Zero-shot evaluation of the PhraseBank-trained DistilBERT checkpoint on
RavenPack headline labels (no RavenPack fine-tuning). Mirrors
``render_ravenpack_baseline_eval_tab()`` and its helpers in ``app.py`` —
the same ``evaluate_phrasebank_baseline_on_ravenpack()`` call scores the
headlines, so both UIs report identical numbers.

The confusion-matrix → metrics transforms are deliberately pure functions
(``class_metrics_from_confusion``, ``summary_from_class_metrics``,
``label_prevalence_from_confusion``, ``prevalence_gap_table``) so they can be
unit-tested without loading the model — see ``tests/test_ravenpack_eval.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
load_dotenv(PROJECT_ROOT / ".env")

from sentiment_ltr.models import phrasebank_sentiment as pbs  # noqa: E402
from sentiment_ltr.models import ravenpack_sentiment as rps  # noqa: E402

LABEL_NAMES = rps.LABEL_NAMES  # ["negative", "neutral", "positive"]
PHRASEBANK_SPLIT_ORDER = pbs.PHRASEBANK_SPLIT_ORDER
EVAL_SPLITS = {
    "test": "test (≥2013)",
    "validation": "validation (2012)",
    "train": "train (≤2011)",
    "all": "all labeled rows",
}


def deps_status() -> dict[str, Any]:
    return {
        "finetuning_deps_available": pbs.finetuning_deps_available(),
        "has_phrasebank_checkpoint": pbs.model_is_saved(),
        "model_dir": str(pbs.DEFAULT_MODEL_DIR.relative_to(PROJECT_ROOT)),
    }


def available_tickers() -> list[str]:
    return sorted({
        rps._ticker_from_article_path(p) for p in rps.discover_ravenpack_article_files()
    })


def _html(fig: Any) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _records(df: pd.DataFrame, limit: int = 250) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    display = df.head(limit).copy()
    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].astype(str)
    display = display.where(pd.notna(display), None)
    return {"columns": list(display.columns), "rows": display.to_dict(orient="records"), "total": len(df)}


# ── Pure confusion-matrix / metrics transforms (unit-tested) ─────────────────


def class_metrics_from_confusion(
    cm: pd.DataFrame, *, dataset: str, split: str, domain: str,
) -> list[dict[str, Any]]:
    """Per-class precision/recall/F1 derived from a confusion matrix.

    Verbatim port of app.py's ``_class_metrics_from_confusion``.
    """
    rows: list[dict[str, Any]] = []
    labels = [label for label in LABEL_NAMES if label in cm.index and label in cm.columns]
    for label in labels:
        tp = float(cm.loc[label, label])
        pred_total = float(cm[label].sum())
        actual_total = float(cm.loc[label].sum())
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / actual_total if actual_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "dataset": dataset,
            "split": split,
            "domain": domain,
            "evaluation": f"{dataset} {split}",
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(actual_total),
        })
    return rows


def summary_from_class_metrics(class_metrics: pd.DataFrame) -> pd.DataFrame:
    """Macro-F1 rows from class-level metrics (mean of class F1s, sum of support)."""
    return (
        class_metrics.groupby(["dataset", "split", "domain", "evaluation"], as_index=False)
        .agg(macro_f1=("f1", "mean"), n_rows=("support", "sum"))
        .sort_values(["domain", "dataset", "split"])
    )


def label_prevalence_from_confusion(cm: pd.DataFrame) -> pd.DataFrame:
    """Observed vs predicted label prevalence for the evaluated rows."""
    labels = [label for label in LABEL_NAMES if label in cm.index and label in cm.columns]
    observed_counts = cm.loc[labels, labels].sum(axis=1)
    predicted_counts = cm.loc[labels, labels].sum(axis=0)
    total = float(observed_counts.sum())

    observed = pd.DataFrame({
        "label": labels,
        "count": observed_counts.to_numpy(dtype=int),
        "series": "Observed / actual",
    })
    predicted = pd.DataFrame({
        "label": labels,
        "count": predicted_counts.to_numpy(dtype=int),
        "series": "Predicted by checkpoint",
    })
    prevalence = pd.concat([observed, predicted], ignore_index=True)
    prevalence["pct"] = prevalence["count"] / total if total else 0.0
    prevalence["pct_label"] = prevalence.apply(
        lambda r: f"{r['pct']:.1%}<br>n={int(r['count']):,}", axis=1,
    )
    prevalence["label"] = pd.Categorical(prevalence["label"], categories=LABEL_NAMES, ordered=True)
    return prevalence


def prevalence_gap_table(prevalence: pd.DataFrame) -> pd.DataFrame:
    """Predicted-minus-actual share per label, in percentage points."""
    gap = prevalence.pivot(index="label", columns="series", values="pct").reindex(LABEL_NAMES)
    gap["prediction_minus_actual_pp"] = (
        gap["Predicted by checkpoint"] - gap["Observed / actual"]
    ) * 100
    return gap[["prediction_minus_actual_pp"]]


# ── Cached expensive computations ─────────────────────────────────────────────

_eval_cache: dict[tuple, dict[str, Any]] = {}
_pb_class_metrics_cache: tuple[str, pd.DataFrame] | None = None
_labeled_cache: dict[str, pd.DataFrame] = {}


def _checkpoint_token() -> str:
    model_dir = pbs.resolve_model_dir()
    parts: list[str] = []
    for name in ("config.json", "model.safetensors", "pytorch_model.bin", "metrics.json"):
        path = model_dir / name
        if path.exists():
            parts.append(f"{name}:{path.stat().st_mtime:.0f}")
    return "|".join(parts) or "none"


def load_labeled(ticker: str) -> pd.DataFrame:
    """RavenPack labeled frame for one ticker (cached — parquet reads are cheap but repeated)."""
    if ticker not in _labeled_cache:
        _labeled_cache[ticker] = rps.load_ravenpack_labeled_frame([ticker])
    return _labeled_cache[ticker]


def run_eval(ticker: str, eval_split: str, max_rows: int) -> dict[str, Any]:
    """Score RavenPack headlines with the PhraseBank checkpoint (memoized like Streamlit's 1h cache)."""
    key = (ticker, eval_split, max_rows or 0, _checkpoint_token())
    if key not in _eval_cache:
        _eval_cache[key] = rps.evaluate_phrasebank_baseline_on_ravenpack(
            [ticker],
            model_dir=pbs.resolve_model_dir(),
            eval_split=eval_split if eval_split != "all" else None,
            max_rows=max_rows or None,
        )
    return _eval_cache[key]


def phrasebank_class_metrics() -> pd.DataFrame:
    """Class-level metrics for the checkpoint on all PhraseBank splits (memoized)."""
    global _pb_class_metrics_cache
    token = _checkpoint_token()
    if _pb_class_metrics_cache is not None and _pb_class_metrics_cache[0] == token:
        return _pb_class_metrics_cache[1]

    raw = pbs.load_phrasebank()
    _, id2label, _ = pbs.label_maps(raw)
    tokenizer, model, device = pbs.load_classifier(pbs.resolve_model_dir())

    rows: list[dict[str, Any]] = []
    for split_name in PHRASEBANK_SPLIT_ORDER:
        split_df = raw[split_name].to_pandas()
        sentences = split_df["sentence"].tolist()
        pred_chunks: list[pd.DataFrame] = []
        for start in range(0, len(sentences), 64):
            pred_chunks.append(
                pbs.predict_sentences(sentences[start : start + 64], tokenizer, model, device)
            )
        preds = pd.concat(pred_chunks, ignore_index=True)
        cm = pd.crosstab(
            pd.Categorical(split_df["label"].map(id2label), categories=LABEL_NAMES),
            pd.Categorical(preds["pred"], categories=LABEL_NAMES),
            rownames=["actual"], colnames=["pred"], dropna=False,
        )
        rows.extend(
            class_metrics_from_confusion(cm, dataset="PhraseBank", split=split_name, domain="in-domain")
        )
    df = pd.DataFrame(rows)
    _pb_class_metrics_cache = (token, df)
    return df


def phrasebank_per_split_dist() -> pd.DataFrame:
    """Per-split label distribution for PhraseBank (train / validation / test)."""
    raw = pbs.load_phrasebank()
    label_feature = raw["train"].features["label"]
    id2label = {i: label_feature.int2str(i) for i in range(label_feature.num_classes)}
    rows = []
    for split_name in PHRASEBANK_SPLIT_ORDER:
        if split_name not in raw:
            continue
        series = pd.Series(raw[split_name]["label"]).map(id2label)
        total = len(series)
        counts = series.value_counts()
        for cls in LABEL_NAMES:
            n = int(counts.get(cls, 0))
            rows.append({"split": split_name, "label": cls, "count": n, "pct": 100 * n / total})
    return pd.DataFrame(rows)


# ── Section contexts for templates ───────────────────────────────────────────


def dataset_summary(ticker: str) -> dict[str, Any]:
    """Header metrics + splits table + class-balance chart for the selected ticker."""
    rp_labeled = load_labeled(ticker)
    rp_balance = rps.ravenpack_class_balance(rp_labeled)
    rp_splits = rps.ravenpack_split_summary(rp_labeled)
    fig = px.bar(
        rp_balance.sort_values("count"),
        x="count", y="label", orientation="h",
        labels={"label": "Class", "count": "Rows"},
        title=f"RavenPack label balance ({ticker})",
    )
    fig.update_traces(hovertemplate="Class: %{y}<br>Count: %{x}<extra></extra>")
    fig.update_layout(hovermode="closest", showlegend=False, height=220)

    def _split_rows(name: str) -> int:
        sel = rp_splits.loc[rp_splits["split"] == name, "rows"]
        return int(sel.iloc[0]) if len(sel) else 0

    return {
        "ticker": ticker,
        "n_labeled": len(rp_labeled),
        "n_train": _split_rows("train"),
        "n_test": _split_rows("test"),
        "splits": _records(rp_splits),
        "balance_chart": _html(fig),
    }


def distribution_shift_context(ticker: str, eval_result: dict[str, Any] | None) -> dict[str, Any]:
    """4D — the two label-distribution charts (PhraseBank vs RavenPack, per-split)."""
    rp_labeled = load_labeled(ticker)
    pb_dist = phrasebank_per_split_dist()

    pb_total = (
        pb_dist.groupby("label", as_index=False)[["count"]].sum()
        .assign(pct=lambda d: d["count"] / d["count"].sum() * 100)
    ).set_index("label").reindex(LABEL_NAMES)

    rp_actual_vc = rp_labeled["label_name"].value_counts().reindex(LABEL_NAMES, fill_value=0)
    rp_actual = pd.DataFrame({
        "label": LABEL_NAMES,
        "count": rp_actual_vc.values.tolist(),
    }).assign(pct=lambda d: d["count"] / d["count"].sum() * 100).set_index("label").reindex(LABEL_NAMES)

    fig1 = go.Figure()
    traces = [
        ("PhraseBank (all splits)", pb_total),
        ("RavenPack actual (all splits)", rp_actual),
    ]
    if eval_result is not None:
        eval_split = eval_result.get("eval_split") or "all"
        cm: pd.DataFrame = eval_result["confusion_counts"]
        pred_counts = cm.sum(axis=0).reindex(LABEL_NAMES, fill_value=0)
        pred_total = pred_counts.sum()
        rp_pred = pd.DataFrame({
            "label": LABEL_NAMES,
            "count": pred_counts.values.tolist(),
        }).assign(pct=lambda d: d["count"] / pred_total * 100).set_index("label").reindex(LABEL_NAMES)
        traces.append((f"RavenPack predicted ({eval_split})", rp_pred))
    for name, df in traces:
        fig1.add_trace(go.Bar(
            name=name, x=LABEL_NAMES, y=df["pct"].tolist(),
            text=[f"{p:.1f}%<br>n={n:,}" for p, n in zip(df["pct"], df["count"])],
            textposition="outside",
        ))
    fig1.update_layout(
        barmode="group",
        title="Label distribution shift: PhraseBank (train) → RavenPack (out-of-domain)",
        xaxis_title="Sentiment class",
        yaxis=dict(title="% of dataset", range=[0, 105]),
        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02),
        height=500, margin=dict(t=60, r=280),
    )

    fig2 = go.Figure()
    for split_name in PHRASEBANK_SPLIT_ORDER:
        sdf = pb_dist[pb_dist["split"] == split_name].set_index("label").reindex(LABEL_NAMES)
        if sdf.empty:
            continue
        fig2.add_trace(go.Bar(
            name=f"PhraseBank {split_name}", x=LABEL_NAMES, y=sdf["pct"].tolist(),
            text=[f"{p:.1f}%<br>n={n:,}" for p, n in zip(sdf["pct"], sdf["count"])],
            textposition="outside",
        ))
    fig2.update_layout(
        barmode="group",
        title="PhraseBank label distribution by split",
        xaxis_title="Sentiment class",
        yaxis=dict(title="% of split", range=[0, 105]),
        height=430,
    )
    return {"shift_chart": _html(fig1), "per_split_chart": _html(fig2)}


def class_metrics_context(eval_result: dict[str, Any]) -> dict[str, Any]:
    """4C — summary table, prevalence chart + gap, class-F1 and precision/recall charts."""
    pb_metrics = phrasebank_class_metrics()
    rp_metrics = pd.DataFrame(class_metrics_from_confusion(
        eval_result["confusion_counts"],
        dataset="RavenPack",
        split=eval_result.get("eval_split") or "all",
        domain="out-of-domain",
    ))
    class_metrics = pd.concat([pb_metrics, rp_metrics], ignore_index=True)

    summary = summary_from_class_metrics(class_metrics)
    summary_display = summary.copy()
    summary_display["macro_f1"] = summary_display["macro_f1"].map(lambda x: f"{x:.1%}")
    summary_display["n_rows"] = summary_display["n_rows"].map(lambda x: f"{x:,}")

    prevalence = label_prevalence_from_confusion(eval_result["confusion_counts"])
    split = eval_result.get("eval_split") or "all"
    n_rows = int(eval_result.get("n_rows", prevalence["count"].sum() / 2))
    prevalence_fig = px.bar(
        prevalence, x="label", y="pct", color="series", barmode="group", text="pct_label",
        category_orders={"label": LABEL_NAMES, "series": ["Observed / actual", "Predicted by checkpoint"]},
        hover_data={"count": ":,", "pct": ":.2%", "label": False},
        title=f"RavenPack label prevalence: observed vs predicted ({split}, n={n_rows:,})",
        color_discrete_map={"Observed / actual": "#0f766e", "Predicted by checkpoint": "#dc2626"},
    )
    ymax = max(0.05, float(prevalence["pct"].max()) * 1.18)
    prevalence_fig.update_traces(textposition="outside", cliponaxis=False)
    prevalence_fig.update_yaxes(title="Share of out-of-domain rows", tickformat=".0%", range=[0, ymax])
    prevalence_fig.update_xaxes(title="Sentiment label")
    prevalence_fig.update_layout(height=420, legend_title_text="Distribution", margin=dict(t=80, r=30, b=60, l=60))

    gap = prevalence_gap_table(prevalence).reset_index()
    gap["prediction_minus_actual_pp"] = gap["prediction_minus_actual_pp"].map(lambda x: f"{x:+.1f} pp")

    f1_frame = class_metrics.copy()
    f1_frame["f1_label"] = f1_frame["f1"].map(lambda x: f"{x:.1%}")
    f1_fig = px.bar(
        f1_frame, x="label", y="f1", color="evaluation", barmode="group", text="f1_label",
        facet_col="domain",
        hover_data={"support": ":,", "precision": ":.3f", "recall": ":.3f", "f1": ":.3f"},
        title="Class-level F1 by domain and split",
    )
    f1_fig.update_traces(textposition="outside", cliponaxis=False)
    f1_fig.update_yaxes(title="Class F1", tickformat=".0%", range=[0, 1.08])
    f1_fig.update_xaxes(title="Sentiment class")
    f1_fig.update_layout(height=500, legend_title_text="Evaluation", margin=dict(t=80, r=30, b=60, l=60))
    f1_fig.for_each_annotation(lambda a: a.update(text=a.text.replace("domain=", "")))

    pr_frame = class_metrics.melt(
        id_vars=["dataset", "split", "domain", "evaluation", "label", "support"],
        value_vars=["precision", "recall"], var_name="metric", value_name="score",
    )
    pr_frame["metric"] = pr_frame["metric"].str.title()
    pr_frame["score_label"] = pr_frame["score"].map(lambda x: f"{x:.1%}")
    pr_fig = px.bar(
        pr_frame, x="label", y="score", color="metric", barmode="group", text="score_label",
        facet_row="domain", facet_col="evaluation",
        hover_data={"support": ":,", "score": ":.3f"},
        title="Precision vs recall by class, domain, and split",
        color_discrete_map={"Precision": "#7c3aed", "Recall": "#ea580c"},
    )
    pr_fig.update_traces(textposition="outside", cliponaxis=False)
    pr_fig.update_yaxes(title="Score", tickformat=".0%", range=[0, 1.08])
    pr_fig.update_xaxes(title="Sentiment class")
    pr_fig.update_layout(height=720, legend_title_text="Metric", margin=dict(t=90, r=30, b=60, l=60))
    pr_fig.for_each_annotation(
        lambda a: a.update(text=a.text.replace("domain=", "").replace("evaluation=", ""))
    )

    return {
        "summary": _records(summary_display),
        "prevalence_chart": _html(prevalence_fig),
        "gap": _records(gap),
        "f1_chart": _html(f1_fig),
        "pr_chart": _html(pr_fig),
    }


def eval_results_context(ticker: str, eval_split: str, max_rows: int) -> dict[str, Any]:
    """4E — headline metrics, confusion heatmap/tables, report, mismatch sample."""
    eval_result = run_eval(ticker, eval_split, max_rows)
    cm: pd.DataFrame = eval_result["confusion_counts"]
    cm_pct: pd.DataFrame = eval_result["confusion_pct"]

    heatmap = px.imshow(
        cm_pct.values, x=list(cm_pct.index), y=list(cm_pct.index),
        text_auto=".1f", aspect="auto",
        color_continuous_scale=[[0.0, "#d65f5f"], [0.5, "#ffffcc"], [1.0, "#90ee90"]],
        labels={"x": "Predicted", "y": "Actual", "color": "% of actual row"},
    )
    heatmap.update_yaxes(autorange="reversed")
    heatmap.update_traces(
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Row share: %{z:.1f}%<extra></extra>",
    )
    heatmap.update_layout(
        title=f"Row-normalized confusion — PhraseBank on {ticker} RavenPack ({eval_split})",
        hovermode="closest", height=360,
    )

    pb_metrics = pbs.load_metrics()
    pb_test_f1 = pb_metrics.get("test", {}).get("eval_f1")
    pb_test_acc = pb_metrics.get("test", {}).get("eval_accuracy")

    cm_counts_df = cm.reset_index()
    cm_pct_display = cm_pct.round(1).reset_index()

    mismatch_df = eval_result["mismatches_sample"]
    mismatch_cols = [
        "article_date", "headline", "event_sentiment_score", "actual", "pred",
        "p(negative)", "p(neutral)", "p(positive)",
    ]
    mismatches = (
        _records(mismatch_df[[c for c in mismatch_cols if c in mismatch_df.columns]])
        if isinstance(mismatch_df, pd.DataFrame) and not mismatch_df.empty else None
    )

    return {
        "ticker": ticker,
        "eval_split": eval_split,
        "n_rows": eval_result["n_rows"],
        "accuracy": eval_result["accuracy"],
        "macro_f1": eval_result["macro_f1"],
        "pb_test_f1": pb_test_f1,
        "pb_test_acc": pb_test_acc,
        "model_dir": str(pbs.resolve_model_dir().relative_to(PROJECT_ROOT)),
        "heatmap": _html(heatmap),
        "cm_counts": _records(cm_counts_df),
        "cm_pct": _records(cm_pct_display),
        "classification_report": eval_result["classification_report"],
        "mismatches": mismatches,
    }


def provenance_context() -> dict[str, Any] | None:
    """Model provenance & reproducibility snapshot (provenance.json), if present."""
    model_dir = pbs.resolve_model_dir()
    path = model_dir / "provenance.json"
    if not path.exists():
        return None
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    ckpt = provenance.get("checkpoint", {})
    git = provenance.get("git", {})
    cfg = provenance.get("model_config", {})
    tok = provenance.get("tokenizer", {})
    data = provenance.get("data", {})
    split_sizes = data.get("split_sizes", {})
    split_hashes = data.get("split_content_sha256", {})
    splits = (
        _records(pd.DataFrame([
            {"split": name, "rows": rows, "content_sha256": split_hashes.get(name, "—")}
            for name, rows in split_sizes.items()
        ]))
        if split_sizes else None
    )
    weights = provenance.get("weights", [])
    return {
        "generated_at": provenance.get("generated_at", "—"),
        "checkpoint_label": ckpt.get("label", "—"),
        "checkpoint_path": ckpt.get("path", "—"),
        "git_commit": git.get("commit_hash_short", "—"),
        "git_branch": git.get("branch", "—"),
        "git_dirty": bool(git.get("is_dirty")),
        "dirty_files": git.get("dirty_files") or [],
        "weights": _records(pd.DataFrame(weights)) if weights else None,
        "num_labels": cfg.get("num_labels", "—"),
        "model_type": cfg.get("model_type", "—"),
        "architecture": ", ".join(cfg.get("architectures") or []) or "—",
        "id2label": str(cfg.get("id2label", {})),
        "tok_max_length": tok.get("max_length_used", "—"),
        "tok_padding": tok.get("padding_strategy", "—"),
        "tok_truncation": bool(tok.get("truncation")),
        "dataset_repo": data.get("dataset_repo", "—"),
        "dataset_config": data.get("dataset_config", "—"),
        "split_type": data.get("split_type", "—"),
        "training_seed": data.get("training_seed", "—"),
        "splits": splits,
        "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
    }
