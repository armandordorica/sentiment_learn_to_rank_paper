"""Tests for the Tab 4 (RavenPack Baseline Eval) FastAPI port.

Two layers:

1. The pure confusion-matrix → metrics transforms in
   ``webapp/api/ravenpack_eval.py``, checked against hand-computed values and
   cross-checked against scikit-learn on the same synthetic matrix.
2. The FastAPI routes, with the expensive model/dataset layer monkeypatched so
   the wiring + template rendering is exercised without loading DistilBERT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from webapp.api import ravenpack_eval as re_
from webapp.main import app

LABELS = ["negative", "neutral", "positive"]


def make_confusion() -> pd.DataFrame:
    """3×3 confusion matrix with known per-class metrics (rows = actual)."""
    return pd.DataFrame(
        [[8, 1, 1], [2, 6, 2], [0, 1, 9]],
        index=pd.Index(LABELS, name="actual"),
        columns=pd.Index(LABELS, name="pred"),
    )


def make_eval_result() -> dict:
    cm = make_confusion()
    return {
        "model_dir": "data/models/phrasebank_distilbert_best",
        "tickers": ["TEST"],
        "eval_split": "test",
        "n_rows": 30,
        "accuracy": 23 / 30,
        "macro_f1": (0.8 + 2 / 3 + 9 / 11) / 3,
        "classification_report": "synthetic report",
        "confusion_counts": cm,
        "confusion_pct": cm.div(cm.sum(axis=1), axis=0).mul(100),
        "label_order": LABELS,
        "mismatches_sample": pd.DataFrame({
            "article_date": ["2013-05-01"],
            "headline": ["Test headline"],
            "event_sentiment_score": [0.4],
            "actual": ["positive"],
            "pred": ["neutral"],
            "p(negative)": [0.1],
            "p(neutral)": [0.6],
            "p(positive)": [0.3],
        }),
    }


# ── Pure metric transforms ────────────────────────────────────────────────────


class TestClassMetricsFromConfusion:
    def test_hand_computed_values(self):
        rows = re_.class_metrics_from_confusion(
            make_confusion(), dataset="RavenPack", split="test", domain="out-of-domain",
        )
        by_label = {r["label"]: r for r in rows}
        assert list(by_label) == LABELS

        # negative: tp=8, predicted total=10, actual total=10
        assert by_label["negative"]["precision"] == pytest.approx(0.8)
        assert by_label["negative"]["recall"] == pytest.approx(0.8)
        assert by_label["negative"]["f1"] == pytest.approx(0.8)
        # neutral: tp=6, predicted total=8, actual total=10
        assert by_label["neutral"]["precision"] == pytest.approx(0.75)
        assert by_label["neutral"]["recall"] == pytest.approx(0.6)
        assert by_label["neutral"]["f1"] == pytest.approx(2 / 3)
        # positive: tp=9, predicted total=12, actual total=10
        assert by_label["positive"]["precision"] == pytest.approx(0.75)
        assert by_label["positive"]["recall"] == pytest.approx(0.9)
        assert by_label["positive"]["f1"] == pytest.approx(9 / 11)

        assert all(r["support"] == 10 for r in rows)
        assert all(r["evaluation"] == "RavenPack test" for r in rows)

    def test_matches_sklearn(self):
        from sklearn.metrics import precision_recall_fscore_support

        cm = make_confusion()
        # Reconstruct per-row labels from the matrix and let sklearn recompute.
        y_true, y_pred = [], []
        for actual in LABELS:
            for pred in LABELS:
                n = int(cm.loc[actual, pred])
                y_true += [actual] * n
                y_pred += [pred] * n
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, zero_division=0,
        )
        rows = re_.class_metrics_from_confusion(
            cm, dataset="RavenPack", split="test", domain="out-of-domain",
        )
        np.testing.assert_allclose([r["precision"] for r in rows], precision)
        np.testing.assert_allclose([r["recall"] for r in rows], recall)
        np.testing.assert_allclose([r["f1"] for r in rows], f1)
        assert [r["support"] for r in rows] == list(support)

    def test_zero_division_safe(self):
        cm = pd.DataFrame(
            [[0, 0, 0], [0, 5, 0], [0, 0, 0]],
            index=pd.Index(LABELS, name="actual"),
            columns=pd.Index(LABELS, name="pred"),
        )
        rows = re_.class_metrics_from_confusion(
            cm, dataset="X", split="s", domain="d",
        )
        by_label = {r["label"]: r for r in rows}
        assert by_label["negative"]["precision"] == 0.0
        assert by_label["negative"]["recall"] == 0.0
        assert by_label["negative"]["f1"] == 0.0
        assert by_label["neutral"]["f1"] == pytest.approx(1.0)

    def test_missing_labels_skipped(self):
        cm = pd.DataFrame(
            [[3, 1], [0, 4]],
            index=pd.Index(["negative", "positive"], name="actual"),
            columns=pd.Index(["negative", "positive"], name="pred"),
        )
        rows = re_.class_metrics_from_confusion(cm, dataset="X", split="s", domain="d")
        assert [r["label"] for r in rows] == ["negative", "positive"]


class TestSummaryFromClassMetrics:
    def test_macro_f1_is_mean_and_rows_sum(self):
        rows = re_.class_metrics_from_confusion(
            make_confusion(), dataset="RavenPack", split="test", domain="out-of-domain",
        )
        summary = re_.summary_from_class_metrics(pd.DataFrame(rows))
        assert len(summary) == 1
        row = summary.iloc[0]
        assert row["macro_f1"] == pytest.approx((0.8 + 2 / 3 + 9 / 11) / 3)
        assert row["n_rows"] == 30
        assert row["evaluation"] == "RavenPack test"


class TestLabelPrevalence:
    def test_observed_and_predicted_counts(self):
        prevalence = re_.label_prevalence_from_confusion(make_confusion())
        observed = prevalence[prevalence["series"] == "Observed / actual"].set_index("label")
        predicted = prevalence[prevalence["series"] == "Predicted by checkpoint"].set_index("label")
        assert observed.loc[LABELS, "count"].tolist() == [10, 10, 10]
        assert predicted.loc[LABELS, "count"].tolist() == [10, 8, 12]
        # Shares are relative to total observed rows (30)
        assert observed.loc["negative", "pct"] == pytest.approx(1 / 3)
        assert predicted.loc["positive", "pct"] == pytest.approx(12 / 30)

    def test_gap_table_in_percentage_points(self):
        prevalence = re_.label_prevalence_from_confusion(make_confusion())
        gap = re_.prevalence_gap_table(prevalence)["prediction_minus_actual_pp"]
        assert gap.loc["negative"] == pytest.approx(0.0)
        assert gap.loc["neutral"] == pytest.approx(-100 * 2 / 30)
        assert gap.loc["positive"] == pytest.approx(100 * 2 / 30)


# ── Presentation contexts (plotly/records, no model) ─────────────────────────


def synthetic_pb_class_metrics() -> pd.DataFrame:
    rows = []
    for split in ["train", "validation", "test"]:
        rows.extend(re_.class_metrics_from_confusion(
            make_confusion(), dataset="PhraseBank", split=split, domain="in-domain",
        ))
    return pd.DataFrame(rows)


def synthetic_pb_dist() -> pd.DataFrame:
    rows = []
    for split in ["train", "validation", "test"]:
        for label, n in zip(LABELS, [10, 20, 70]):
            rows.append({"split": split, "label": label, "count": n, "pct": n})
    return pd.DataFrame(rows)


def synthetic_labeled() -> pd.DataFrame:
    return pd.DataFrame({"label_name": ["negative"] * 5 + ["neutral"] * 3 + ["positive"] * 2})


def test_class_metrics_context(monkeypatch):
    monkeypatch.setattr(re_, "phrasebank_class_metrics", synthetic_pb_class_metrics)
    ctx = re_.class_metrics_context(make_eval_result())
    assert ctx["summary"]["total"] == 4  # 3 PhraseBank splits + 1 RavenPack row
    summary_rows = {r["evaluation"]: r for r in ctx["summary"]["rows"]}
    assert summary_rows["RavenPack test"]["n_rows"] == "30"
    assert summary_rows["RavenPack test"]["macro_f1"] == "76.2%"  # (0.8 + 2/3 + 9/11) / 3
    for key in ("prevalence_chart", "f1_chart", "pr_chart"):
        assert "<div" in ctx[key]
    gap_rows = {r["label"]: r for r in ctx["gap"]["rows"]}
    assert gap_rows["neutral"]["prediction_minus_actual_pp"] == "-6.7 pp"


def test_distribution_shift_context(monkeypatch):
    monkeypatch.setattr(re_, "load_labeled", lambda ticker: synthetic_labeled())
    monkeypatch.setattr(re_, "phrasebank_per_split_dist", synthetic_pb_dist)
    ctx = re_.distribution_shift_context("TEST", make_eval_result())
    assert "<div" in ctx["shift_chart"]
    assert "<div" in ctx["per_split_chart"]
    # The predicted trace is present only when an eval result is passed
    assert "RavenPack predicted (test)" in ctx["shift_chart"]
    ctx_no_eval = re_.distribution_shift_context("TEST", None)
    assert "RavenPack predicted" not in ctx_no_eval["shift_chart"]


def test_eval_results_context(monkeypatch):
    monkeypatch.setattr(re_, "run_eval", lambda t, s, m: make_eval_result())
    monkeypatch.setattr(
        re_.pbs, "load_metrics",
        lambda model_dir=None: {"test": {"eval_f1": 0.9, "eval_accuracy": 0.91}},
    )
    ctx = re_.eval_results_context("TEST", "test", 0)
    assert ctx["n_rows"] == 30
    assert ctx["accuracy"] == pytest.approx(23 / 30)
    assert ctx["pb_test_f1"] == 0.9
    assert "<div" in ctx["heatmap"]
    assert ctx["cm_counts"]["total"] == 3
    assert ctx["mismatches"]["total"] == 1


# ── Route wiring + template rendering ────────────────────────────────────────


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_raven_eval_page(monkeypatch, client):
    monkeypatch.setattr(re_, "deps_status", lambda: {
        "finetuning_deps_available": True,
        "has_phrasebank_checkpoint": True,
        "model_dir": "data/models/phrasebank_distilbert_best",
    })
    monkeypatch.setattr(re_, "available_tickers", lambda: ["TEST"])
    monkeypatch.setattr(re_, "dataset_summary", lambda ticker: {
        "ticker": ticker, "n_labeled": 10, "n_train": 6, "n_test": 2,
        "splits": {"columns": ["split", "rows"], "rows": [{"split": "train", "rows": 6}], "total": 1},
        "balance_chart": "<div>chart</div>",
    })
    monkeypatch.setattr(re_, "provenance_context", lambda: None)
    resp = client.get("/raven-eval")
    assert resp.status_code == 200
    assert "RavenPack Baseline Evaluation" in resp.text
    assert "TEST" in resp.text
    assert "4E · Run evaluation" in resp.text
    assert "No <code>provenance.json</code>" in resp.text


def test_raven_eval_static_partial(monkeypatch, client):
    monkeypatch.setattr(re_, "run_eval", lambda t, s, m: make_eval_result())
    monkeypatch.setattr(re_, "load_labeled", lambda ticker: synthetic_labeled())
    monkeypatch.setattr(re_, "phrasebank_per_split_dist", synthetic_pb_dist)
    monkeypatch.setattr(re_, "phrasebank_class_metrics", synthetic_pb_class_metrics)
    resp = client.get("/raven-eval/static", params={"ticker": "TEST"})
    assert resp.status_code == 200
    assert "4C · Class-level baseline metrics" in resp.text
    assert "Prediction prevalence gap" in resp.text


def test_raven_eval_run_endpoint(monkeypatch, client):
    monkeypatch.setattr(re_, "run_eval", lambda t, s, m: make_eval_result())
    monkeypatch.setattr(
        re_.pbs, "load_metrics",
        lambda model_dir=None: {"test": {"eval_f1": 0.9, "eval_accuracy": 0.91}},
    )
    resp = client.post(
        "/raven-eval/run",
        data={"ticker": "TEST", "eval_split": "test", "max_rows": "0"},
    )
    assert resp.status_code == 200
    assert "Rows scored" in resp.text
    assert "76.7%" in resp.text  # accuracy 23/30
    assert "synthetic report" in resp.text
    assert "Sample mismatches" in resp.text


def test_raven_eval_run_error_path(monkeypatch, client):
    def boom(t, s, m):
        raise ValueError("No RavenPack rows for split='validation'.")

    monkeypatch.setattr(re_, "run_eval", boom)
    resp = client.post(
        "/raven-eval/run",
        data={"ticker": "TEST", "eval_split": "validation", "max_rows": "0"},
    )
    assert resp.status_code == 200
    assert "Baseline evaluation failed" in resp.text
    assert "No RavenPack rows" in resp.text


def test_raven_eval_dataset_error_path(monkeypatch, client):
    def boom(ticker):
        raise KeyError("headline")

    monkeypatch.setattr(re_, "dataset_summary", boom)
    resp = client.get("/raven-eval/dataset", params={"ticker": "BAD"})
    assert resp.status_code == 200
    assert "Could not load RavenPack data for BAD" in resp.text
