"""Financial PhraseBank sentiment classifier — shared by notebook and Streamlit.

Mirrors the workflow in ``notebooks/liquidAI_prep.ipynb``: load a script-free
Parquet mirror of Financial PhraseBank, fine-tune DistilBERT, and run inference.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "phrasebank_distilbert_1ep"
METRICS_FILENAME = "metrics.json"

PRIMARY_DATASET = "atrost/financial_phrasebank"
FALLBACK_DATASET = "warwickai/financial_phrasebank_mirror"
LABEL_NAMES_FALLBACK = ["negative", "neutral", "positive"]

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128

# Recorded from the first successful 1-epoch baseline (MPS, 2026-06-28).
FALLBACK_METRICS: dict[str, Any] = {
    "model_name": MODEL_NAME,
    "dataset": PRIMARY_DATASET,
    "epochs": 1,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "max_length": MAX_LENGTH,
    "train_loss": 0.674,
    "train_runtime_s": 40.8,
    "validation": {"eval_loss": 0.5109, "eval_accuracy": 0.7887},
    "test": {"eval_loss": 0.5214, "eval_accuracy": 0.8062},
    "device": "mps",
    "note": "Fallback metrics from notebooks/liquidAI_prep.ipynb until a saved checkpoint exists.",
}


def finetuning_deps_available() -> bool:
    """Return True when torch/transformers/datasets are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import datasets  # noqa: F401
        return True
    except ImportError:
        return False


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_report() -> dict[str, Any]:
    """Summarize available compute backends and the selected device."""
    import torch

    cuda_ok = torch.cuda.is_available()
    mps_ok = (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    )
    selected = pick_device()
    name = "CPU"
    if selected == "cuda":
        name = torch.cuda.get_device_name(0)
    elif selected == "mps":
        name = "Apple GPU (Metal / MPS)"
    return {
        "cuda_available": cuda_ok,
        "mps_available": mps_ok,
        "selected": selected,
        "device_name": name,
        "torch_version": torch.__version__,
    }


def benchmark_matmul(device: str, n: int = 4096, iters: int = 10) -> float:
    """Time ``iters`` n×n matmuls on ``device``; returns seconds (sync-corrected).

    GPUs run asynchronously, so we synchronize before stopping the timer and run a
    short warm-up first (kernel compile / lazy alloc) for a fair measurement.
    """
    import torch

    a = torch.randn(n, n, device=device)
    b = torch.randn(n, n, device=device)

    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    for _ in range(3):
        _ = a @ b
    _sync()

    start = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    _sync()
    return time.perf_counter() - start


def load_phrasebank():
    """Load Financial PhraseBank (datasets v5 compatible)."""
    from datasets import ClassLabel, load_dataset

    try:
        return load_dataset(PRIMARY_DATASET)
    except Exception as exc:
        print(f"[phrasebank] primary load failed ({type(exc).__name__}); using mirror.")
        raw = load_dataset(FALLBACK_DATASET)
        if not isinstance(raw["train"].features["label"], ClassLabel):
            raw = raw.cast_column("label", ClassLabel(names=LABEL_NAMES_FALLBACK))
        return raw


def label_maps(raw) -> tuple[list[str], dict[int, str], dict[str, int]]:
    names = raw["train"].features["label"].names
    id2label = {i: name for i, name in enumerate(names)}
    label2id = {name: i for i, name in id2label.items()}
    return names, id2label, label2id


def dataset_class_balance(raw) -> pd.DataFrame:
    """Class counts on the train split."""
    _, id2label, _ = label_maps(raw)
    train = raw["train"]
    counts = pd.Series(train["label"]).map(id2label).value_counts()
    return pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "pct": (counts.values / counts.sum() * 100).round(1),
    }).reset_index(drop=True)


def metrics_path(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    return model_dir / METRICS_FILENAME


def load_metrics(model_dir: Path = DEFAULT_MODEL_DIR) -> dict[str, Any]:
    """Load saved training metrics, or return documented fallback."""
    path = metrics_path(model_dir)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return FALLBACK_METRICS.copy()


def model_is_saved(model_dir: Path = DEFAULT_MODEL_DIR) -> bool:
    return (model_dir / "config.json").exists() and (
        (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
    )


def tokenize_dataset(raw, tokenizer):
    """Tokenize all splits for Trainer."""

    def _batch(batch: dict) -> dict:
        return tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    return raw.map(_batch, batched=True, remove_columns=["sentence"])


def train_baseline(
    *,
    output_dir: Path = DEFAULT_MODEL_DIR,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune DistilBERT on PhraseBank and persist model + metrics."""
    import evaluate
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    device = pick_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = load_phrasebank()
    label_names, id2label, label2id = label_maps(raw)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
    )

    tokenized = tokenize_dataset(raw, tokenizer)
    accuracy_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return accuracy_metric.compute(predictions=preds, references=labels)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    val_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"])

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    metrics = {
        "model_name": MODEL_NAME,
        "dataset": PRIMARY_DATASET,
        "epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "max_length": MAX_LENGTH,
        "train_loss": float(train_result.training_loss),
        "train_runtime_s": float(train_result.metrics.get("train_runtime", 0)),
        "validation": {k: float(v) if isinstance(v, (int, float)) else v for k, v in val_metrics.items()},
        "test": {k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()},
        "device": device,
        "saved_at": pd.Timestamp.utcnow().isoformat(),
    }
    metrics_path(output_dir).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def load_classifier(model_dir: Path = DEFAULT_MODEL_DIR):
    """Load a saved tokenizer + classification model."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    if not model_is_saved(model_dir):
        raise FileNotFoundError(
            f"No saved model at {model_dir}. Run training from the Sentiment Lab tab "
            "or notebooks/liquidAI_prep.ipynb first."
        )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    device = pick_device()
    model.to(device)
    model.eval()
    return tokenizer, model, device


def predict_sentences(
    sentences: list[str],
    tokenizer,
    model,
    device: str,
    *,
    id2label: dict[int, str] | None = None,
) -> pd.DataFrame:
    """Return per-sentence label probabilities."""
    import torch

    if not sentences:
        return pd.DataFrame()

    if id2label is None:
        id2label = {int(k): v for k, v in model.config.id2label.items()}

    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    pred_ids = probs.argmax(axis=-1)

    rows = []
    for i, sentence in enumerate(sentences):
        row = {"sentence": sentence, "pred": id2label[int(pred_ids[i])]}
        for j, name in id2label.items():
            row[f"p({name})"] = round(float(probs[i, j]), 4)
        rows.append(row)
    return pd.DataFrame(rows)
