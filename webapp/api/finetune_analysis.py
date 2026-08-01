"""Section 5·1–5·7 presentation: AAPL before/after fine-tuning analysis.

Mirrors Streamlit ``render_ravenpack_finetuning_tab`` subsections that were not
yet on FastAPI. Heavy scoring is memoized by checkpoint mtime (same idea as
Tab 4 / Streamlit ``@st.cache_data``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_ltr.models import phrasebank_sentiment as pbs  # noqa: E402
from sentiment_ltr.models import ravenpack_sentiment as rps  # noqa: E402

ANALYSIS_TICKER = "AAPL"
CLASS_ORDER = ["negative", "neutral", "positive"]

_before_after_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
_split_cache: dict[str, Any] | None = None


def _html(fig: Any) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _mtime_token(path: Path) -> str:
    return str(path.stat().st_mtime) if path.exists() else "missing"


def split_overview(ticker: str = ANALYSIS_TICKER) -> dict[str, Any]:
    """5·1 — temporal split metrics + year/class facet chart."""
    global _split_cache
    if _split_cache is not None and _split_cache.get("ticker") == ticker:
        return _split_cache
    labeled = rps.load_ravenpack_labeled_frame([ticker])
    split_df = rps.ravenpack_split_summary(labeled)
    rows = {str(r["split"]): int(r["rows"]) for r in split_df.to_dict(orient="records")}
    work = labeled.copy()
    work["year"] = pd.to_datetime(work["article_date"]).dt.year
    work["split"] = rps.assign_time_split(work["article_date"])
    year_counts = (
        work.groupby(["year", "label_name", "split"]).size().reset_index(name="count")
    )
    fig = px.bar(
        year_counts,
        x="year",
        y="count",
        color="label_name",
        facet_col="split",
        category_orders={
            "label_name": CLASS_ORDER,
            "split": ["train", "validation", "test"],
        },
        title=f"{ticker} labeled headlines by year, class, and split",
        labels={"count": "Headlines", "year": "Year", "label_name": "Class"},
        color_discrete_map={
            "negative": "#ef4444",
            "neutral": "#94a3b8",
            "positive": "#22c55e",
        },
    )
    fig.update_layout(height=380, margin={"t": 60, "r": 20, "b": 50, "l": 60})
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("split=", "")))
    threshold = rps.SENTIMENT_SCORE_THRESHOLD
    result = {
        "ticker": ticker,
        "train_rows": rows.get("train", 0),
        "val_rows": rows.get("validation", 0),
        "test_rows": rows.get("test", 0),
        "splits_table": split_df.to_dict(orient="records"),
        "threshold": threshold,
        "windows": {
            "train": "2003 – 2011",
            "validation": "2012",
            "test": "2013 – 2014",
        },
        "chart_html": _html(fig),
        "error": None,
    }
    _split_cache = result
    return result


def tokenization_context() -> dict[str, Any]:
    """5·2 — shared PhraseBank / RavenPack tokenizer contract."""
    metrics = pbs.load_metrics()
    return {
        "tokenizer_class": "DistilBertTokenizerFast",
        "max_length": metrics.get("max_length", 128),
        "padding": "max_length (fixed)",
        "truncation": True,
        "vocab_size": "30 522 (BERT WordPiece)",
        "base_model": metrics.get("model_name", pbs.MODEL_NAME),
        "rows": [
            {"setting": "tokenizer_class", "value": "DistilBertTokenizerFast"},
            {"setting": "max_length", "value": str(metrics.get("max_length", 128))},
            {
                "setting": "padding_strategy",
                "value": "max_length (all sequences padded to same length)",
            },
            {"setting": "truncation", "value": "True"},
            {"setting": "vocab_size", "value": "30 522 (BERT WordPiece)"},
            {
                "setting": "base_model",
                "value": metrics.get("model_name", pbs.MODEL_NAME),
            },
        ],
    }


def _provenance_for_dir(model_dir: Path) -> dict[str, Any] | None:
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
    return {
        "generated_at": provenance.get("generated_at", "—"),
        "checkpoint_label": ckpt.get("label", "—"),
        "checkpoint_path": ckpt.get("path", "—"),
        "git_commit": git.get("commit_hash_short", "—"),
        "git_branch": git.get("branch", "—"),
        "git_dirty": bool(git.get("is_dirty")),
        "num_labels": cfg.get("num_labels", "—"),
        "model_type": cfg.get("model_type", "—"),
        "architecture": ", ".join(cfg.get("architectures") or []) or "—",
        "tok_max_length": tok.get("max_length_used", "—"),
        "tok_padding": tok.get("padding_strategy", "—"),
        "dataset_repo": data.get("dataset_repo", "—"),
        "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
    }


def hyperparams_context() -> dict[str, Any]:
    """5·7 — RavenPack metrics.json + provenance snapshot."""
    model_dir = rps.resolve_ravenpack_model_dir()
    metrics_path = model_dir / "metrics.json"
    metrics: dict[str, Any] = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            metrics = {}
    init = str(metrics.get("init_checkpoint") or "")
    init_label = "PhraseBank" if init.endswith("phrasebank_distilbert_best") else (init or "—")
    runtime_s = metrics.get("train_runtime_s")
    return {
        "has_metrics": bool(metrics),
        "init_checkpoint": init_label,
        "epochs": metrics.get("epochs"),
        "learning_rate": metrics.get("learning_rate"),
        "batch_size": metrics.get("per_device_train_batch_size"),
        "train_loss": metrics.get("train_loss"),
        "runtime_min": (runtime_s / 60.0) if runtime_s else None,
        "device": str(metrics.get("device", "—")).upper(),
        "tickers": ", ".join(metrics.get("tickers") or [ANALYSIS_TICKER]),
        "metrics_json": metrics,
        "provenance": _provenance_for_dir(model_dir),
        "model_dir": str(model_dir.relative_to(PROJECT_ROOT)),
    }


def _cm_to_series(cm: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    rows_actual, rows_pred = [], []
    for actual_label in cm.index:
        for pred_label in cm.columns:
            count = int(cm.loc[actual_label, pred_label])
            rows_actual.extend([actual_label] * count)
            rows_pred.extend([pred_label] * count)
    return pd.Series(rows_actual), pd.Series(rows_pred)


def _compare_checkpoints(ticker: str, eval_split: str = "test") -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

    pb_dir = pbs.resolve_model_dir()
    rp_dir = rps.resolve_ravenpack_model_dir()
    key = (
        ticker,
        eval_split,
        f"{_mtime_token(pb_dir / 'config.json')}|{_mtime_token(rp_dir / 'config.json')}",
    )
    if key in _before_after_cache:
        return _before_after_cache[key]

    pb_result = rps.evaluate_phrasebank_baseline_on_ravenpack(
        [ticker], model_dir=pb_dir, eval_split=eval_split, max_rows=None
    )
    has_rp = rps.ravenpack_model_is_saved(rp_dir)
    rp_result = None
    if has_rp:
        rp_result = rps.evaluate_phrasebank_baseline_on_ravenpack(
            [ticker], model_dir=rp_dir, eval_split=eval_split, max_rows=None
        )

    payload: dict[str, Any] = {
        "pb_result": pb_result,
        "rp_result": rp_result,
        "n_rows": int(pb_result["n_rows"]),
        "has_rp": has_rp,
    }
    if rp_result is not None:
        y_true, pb_pred = _cm_to_series(pb_result["confusion_counts"])
        _, rp_pred = _cm_to_series(rp_result["confusion_counts"])
        cmp_rows = []
        for ckpt_name, y_pred in [
            ("PhraseBank (no fine-tune)", pb_pred),
            ("RavenPack fine-tuned", rp_pred),
        ]:
            p, r, f, s = precision_recall_fscore_support(
                y_true, y_pred, labels=CLASS_ORDER, zero_division=0
            )
            for label, pi, ri, fi, si in zip(CLASS_ORDER, p, r, f, s):
                cmp_rows.append({
                    "checkpoint": ckpt_name,
                    "label": label,
                    "precision": float(pi),
                    "recall": float(ri),
                    "f1": float(fi),
                    "support": int(si),
                })
            cmp_rows.append({
                "checkpoint": ckpt_name,
                "label": "macro avg",
                "precision": float(p.mean()),
                "recall": float(r.mean()),
                "f1": float(
                    f1_score(
                        y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
                    )
                ),
                "support": int(len(y_true)),
            })
        overall_rows = []
        for ckpt_name, y_pred in [
            ("PhraseBank (no fine-tune)", pb_pred),
            ("RavenPack fine-tuned", rp_pred),
        ]:
            overall_rows.append({
                "checkpoint": ckpt_name,
                "macro_f1": float(
                    f1_score(
                        y_true, y_pred, labels=CLASS_ORDER, average="macro", zero_division=0
                    )
                ),
                "accuracy": float(accuracy_score(y_true, y_pred)),
            })
        payload["ckpt_cmp"] = pd.DataFrame(cmp_rows)
        payload["overall_cmp"] = pd.DataFrame(overall_rows)

    _before_after_cache[key] = payload
    return payload


def before_after_context(ticker: str = ANALYSIS_TICKER) -> dict[str, Any]:
    """5·3–5·6 presentation dict (HTML charts + tables)."""
    pb_metrics = pbs.load_metrics()
    rp_metrics: dict[str, Any] = {}
    rp_path = rps.resolve_ravenpack_model_dir() / "metrics.json"
    if rp_path.exists():
        try:
            rp_metrics = json.loads(rp_path.read_text(encoding="utf-8"))
        except Exception:
            rp_metrics = {}

    pb_in_domain_f1 = pb_metrics.get("test", {}).get("eval_f1")
    pb_in_domain_acc = pb_metrics.get("test", {}).get("eval_accuracy")
    rp_test_f1 = rp_metrics.get("test", {}).get("eval_f1")
    rp_test_acc = rp_metrics.get("test", {}).get("eval_accuracy")

    try:
        cmp_data = _compare_checkpoints(ticker, "test")
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "ticker": ticker}

    pb_ood = cmp_data["pb_result"]
    pb_ood_f1 = float(pb_ood["macro_f1"])
    pb_ood_acc = float(pb_ood["accuracy"])
    has_rp = bool(cmp_data.get("has_rp") and cmp_data.get("rp_result") is not None)

    summary_rows = [
        {
            "checkpoint": "PhraseBank (in-domain)",
            "domain": "in-domain (PhraseBank test)",
            "macro_f1": pb_in_domain_f1,
            "accuracy": pb_in_domain_acc,
        },
        {
            "checkpoint": "PhraseBank — no fine-tune (OOD)",
            "domain": "out-of-domain (RavenPack test 2013–2014)",
            "macro_f1": pb_ood_f1,
            "accuracy": pb_ood_acc,
        },
        {
            "checkpoint": "RavenPack fine-tuned (AAPL)",
            "domain": "out-of-domain (RavenPack test 2013–2014)",
            "macro_f1": rp_test_f1,
            "accuracy": rp_test_acc,
        },
    ]

    chart_rows = []
    for row in summary_rows:
        for metric_name, key in [("Macro-F1", "macro_f1"), ("Accuracy", "accuracy")]:
            val = row[key]
            if val is None:
                continue
            chart_rows.append({
                "checkpoint": row["checkpoint"],
                "metric": metric_name,
                "score": float(val),
                "score_label": f"{float(val):.1%}",
                "domain": row["domain"],
            })
    summary_chart = None
    if chart_rows:
        fig = px.bar(
            pd.DataFrame(chart_rows),
            x="checkpoint",
            y="score",
            color="metric",
            barmode="group",
            text="score_label",
            hover_data={"domain": True, "score": ":.3f"},
            title=f"Macro-F1 & Accuracy: baseline → RavenPack fine-tuned ({ticker} test)",
            color_discrete_map={"Macro-F1": "#2563eb", "Accuracy": "#059669"},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_yaxes(title="Score", tickformat=".0%", range=[0, 1.10])
        fig.update_xaxes(title="")
        fig.update_layout(height=460, legend_title_text="Metric", margin={"t": 70, "r": 30, "b": 120, "l": 60})
        summary_chart = _html(fig)

    # 5·4 per-class
    class_pivot_rows: list[dict[str, Any]] = []
    overall_chart = None
    class_chart = None
    if has_rp and "ckpt_cmp" in cmp_data:
        ckpt_cmp = cmp_data["ckpt_cmp"]
        overall_cmp = cmp_data["overall_cmp"]
        n_test = cmp_data["n_rows"]
        for label in CLASS_ORDER + ["macro avg"]:
            row: dict[str, Any] = {"label": label}
            for ckpt, prefix in [
                ("PhraseBank (no fine-tune)", "pb"),
                ("RavenPack fine-tuned", "rp"),
            ]:
                sub = ckpt_cmp[(ckpt_cmp["checkpoint"] == ckpt) & (ckpt_cmp["label"] == label)]
                if not sub.empty:
                    row[f"{prefix}_precision"] = float(sub.iloc[0]["precision"])
                    row[f"{prefix}_recall"] = float(sub.iloc[0]["recall"])
                    row[f"{prefix}_f1"] = float(sub.iloc[0]["f1"])
            class_pivot_rows.append(row)

        overall_long = overall_cmp.melt(
            id_vars="checkpoint",
            value_vars=["macro_f1", "accuracy"],
            var_name="metric",
            value_name="score",
        )
        overall_long["metric"] = overall_long["metric"].map(
            {"macro_f1": "Macro-F1", "accuracy": "Accuracy"}
        )
        overall_long["score_label"] = overall_long["score"].map(lambda x: f"{x:.1%}")
        fig_o = px.bar(
            overall_long,
            x="checkpoint",
            y="score",
            color="metric",
            barmode="group",
            text="score_label",
            title=f"Overall: PhraseBank vs RavenPack fine-tuned | {ticker} test, n={n_test:,}",
            color_discrete_map={"Macro-F1": "#2563eb", "Accuracy": "#059669"},
        )
        fig_o.update_traces(textposition="outside", cliponaxis=False)
        fig_o.update_yaxes(title="Score", tickformat=".0%", range=[0, 1.10])
        fig_o.update_layout(height=460, margin={"t": 70, "r": 30, "b": 80, "l": 60})
        overall_chart = _html(fig_o)

        class_cmp = ckpt_cmp[ckpt_cmp["label"] != "macro avg"].copy()
        class_cmp["f1_label"] = class_cmp["f1"].map(lambda x: f"{x:.1%}")
        fig_c = px.bar(
            class_cmp,
            x="label",
            y="f1",
            color="checkpoint",
            barmode="group",
            text="f1_label",
            category_orders={"label": CLASS_ORDER},
            title=f"Per-class F1 | {ticker} test, n={n_test:,}",
            color_discrete_map={
                "PhraseBank (no fine-tune)": "#94a3b8",
                "RavenPack fine-tuned": "#2563eb",
            },
        )
        fig_c.update_traces(textposition="outside", cliponaxis=False)
        fig_c.update_yaxes(title="Class F1", tickformat=".0%", range=[0, 1.10])
        fig_c.update_layout(height=430, margin={"t": 70, "r": 30, "b": 60, "l": 60})
        class_chart = _html(fig_c)

    # 5·5 prevalence
    prevalence_chart = None
    prevalence_gaps: list[dict[str, Any]] = []
    cm = pb_ood["confusion_counts"]
    actual_vc = cm.sum(axis=1).reindex(CLASS_ORDER, fill_value=0)
    pb_pred_vc = cm.sum(axis=0).reindex(CLASS_ORDER, fill_value=0)
    n_test = int(pb_ood["n_rows"])
    prev_frames = [
        pd.DataFrame({
            "label": CLASS_ORDER,
            "count": actual_vc.values,
            "series": "Actual (ground truth)",
        }),
        pd.DataFrame({
            "label": CLASS_ORDER,
            "count": pb_pred_vc.values,
            "series": "Predicted — PhraseBank (no fine-tune)",
        }),
    ]
    if has_rp and cmp_data.get("rp_result") is not None:
        rp_cm = cmp_data["rp_result"]["confusion_counts"]
        rp_pred_vc = rp_cm.sum(axis=0).reindex(CLASS_ORDER, fill_value=0)
        prev_frames.append(pd.DataFrame({
            "label": CLASS_ORDER,
            "count": rp_pred_vc.values,
            "series": "Predicted — RavenPack fine-tuned",
        }))
    prev_df = pd.concat(prev_frames, ignore_index=True)
    prev_df["pct"] = prev_df["count"] / max(n_test, 1)
    prev_df["pct_label"] = prev_df.apply(
        lambda r: f"{r['pct']:.1%}<br>n={int(r['count']):,}", axis=1
    )
    color_map = {
        "Actual (ground truth)": "#0f766e",
        "Predicted — PhraseBank (no fine-tune)": "#94a3b8",
        "Predicted — RavenPack fine-tuned": "#2563eb",
    }
    series_order = [s for s in color_map if s in set(prev_df["series"])]
    fig_p = px.bar(
        prev_df,
        x="label",
        y="pct",
        color="series",
        barmode="group",
        text="pct_label",
        category_orders={"label": CLASS_ORDER, "series": series_order},
        title=f"Label prevalence: actual vs predicted | {ticker} 2013–2014, n={n_test:,}",
        color_discrete_map=color_map,
    )
    fig_p.update_traces(textposition="outside", cliponaxis=False)
    fig_p.update_yaxes(
        title="Share of test rows",
        tickformat=".0%",
        range=[0, max(0.05, float(prev_df["pct"].max()) * 1.20)],
    )
    fig_p.update_layout(height=480, margin={"t": 80, "r": 30, "b": 60, "l": 60})
    prevalence_chart = _html(fig_p)
    pivot = prev_df.pivot(index="label", columns="series", values="pct").reindex(CLASS_ORDER)
    for col in series_order:
        if col == "Actual (ground truth)":
            continue
        for label in CLASS_ORDER:
            prevalence_gaps.append({
                "label": label,
                "series": col,
                "delta_pp": float((pivot.loc[label, col] - pivot.loc[label, "Actual (ground truth)"]) * 100),
            })

    # 5·6 sample mismatches (PB wrong)
    samples: list[dict[str, Any]] = []
    mis = pb_ood.get("mismatches_sample")
    if isinstance(mis, pd.DataFrame) and not mis.empty:
        sample = mis.sample(min(20, len(mis)), random_state=99)
        for _, row in sample.iterrows():
            samples.append({
                "article_date": str(row.get("article_date", "")),
                "headline": row.get("headline"),
                "actual": row.get("actual"),
                "pb_pred": row.get("pred"),
                "event_sentiment_score": _float(row.get("event_sentiment_score")),
                "p_negative": _float(row.get("p(negative)")),
                "p_neutral": _float(row.get("p(neutral)")),
                "p_positive": _float(row.get("p(positive)")),
            })

    return {
        "error": None,
        "ticker": ticker,
        "has_rp": has_rp,
        "n_rows": n_test,
        "pb_in_domain_f1": pb_in_domain_f1,
        "pb_ood_f1": pb_ood_f1,
        "rp_test_f1": rp_test_f1,
        "delta_ood_vs_in": (
            pb_ood_f1 - pb_in_domain_f1
            if pb_ood_f1 is not None and pb_in_domain_f1 is not None
            else None
        ),
        "delta_after_vs_before": (
            rp_test_f1 - pb_ood_f1
            if rp_test_f1 is not None and pb_ood_f1 is not None
            else None
        ),
        "summary_rows": summary_rows,
        "summary_chart_html": summary_chart,
        "class_pivot_rows": class_pivot_rows,
        "overall_chart_html": overall_chart,
        "class_chart_html": class_chart,
        "prevalence_chart_html": prevalence_chart,
        "prevalence_gaps": prevalence_gaps,
        "samples": samples,
    }


def static_analysis_context(ticker: str = ANALYSIS_TICKER) -> dict[str, Any]:
    """Fast pieces for GET /finetune (5·1, 5·2, 5·7)."""
    out: dict[str, Any] = {
        "ticker": ticker,
        "tokenization": tokenization_context(),
        "hyperparams": hyperparams_context(),
        "split": None,
        "split_error": None,
    }
    try:
        out["split"] = split_overview(ticker)
    except Exception as exc:  # noqa: BLE001
        out["split_error"] = str(exc)
    return out


def _float(value: object) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
