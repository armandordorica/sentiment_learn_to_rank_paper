"""Fine-tune DistilBERT on RavenPack headline labels (TRNA substitute).

Loads cached RavenPack article exports (headline + ``event_sentiment_score``),
maps scores to the same 3-way labels as Financial PhraseBank, and continues
fine-tuning from the PhraseBank checkpoint when available.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sentiment_ltr.models.phrasebank_sentiment import (
    DEFAULT_MODEL_DIR,
    MAX_LENGTH,
    METRICS_FILENAME,
    MODEL_NAME,
    build_compute_metrics,
    metrics_path,
    model_is_saved,
    pick_device,
    tokenize_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAVENPACK_NEWS_DIR = PROJECT_ROOT / "data" / "raw" / "news" / "ravenpack"
DEFAULT_RAVENPACK_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "ravenpack_distilbert_best"

LABEL_NAMES = ["negative", "neutral", "positive"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}

# Match the Sentiment Lab RavenPack polarity display (``app._ravenpack_polarity``).
SENTIMENT_SCORE_THRESHOLD = 0.05

# Time-based split boundaries (inclusive) on ``article_date``.
TRAIN_END = "2011-12-31"
VAL_START = "2012-01-01"
VAL_END = "2012-12-31"
TEST_START = "2013-01-01"

SPLIT_SOURCE = (
    "RavenPack time split: train ≤2011, val 2012, test ≥2013 "
    f"(|score|>{SENTIMENT_SCORE_THRESHOLD} → label; else neutral)"
)

DEFAULT_RAVENPACK_TRAIN_EPOCHS = 2


def score_to_label(score: object) -> str | None:
    """Map RavenPack ``event_sentiment_score`` to negative / neutral / positive."""
    try:
        val = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(val):
        return None
    if val > SENTIMENT_SCORE_THRESHOLD:
        return "positive"
    if val < -SENTIMENT_SCORE_THRESHOLD:
        return "negative"
    return "neutral"


def discover_ravenpack_article_files(
    tickers: list[str] | None = None,
    *,
    news_dir: Path = RAVENPACK_NEWS_DIR,
) -> list[Path]:
    """Return rich RavenPack article parquet paths (``*_articles_*.parquet``)."""
    if not news_dir.exists():
        return []

    paths = sorted(news_dir.glob("*_articles_*.parquet"))
    if tickers:
        wanted = {t.strip().upper() for t in tickers if t.strip()}
        paths = [
            p for p in paths
            if p.name.split("_articles_")[0].upper() in wanted
        ]
    return paths


def _ticker_from_article_path(path: Path) -> str:
    return path.name.split("_articles_")[0].upper()


def load_ravenpack_labeled_frame(
    tickers: list[str] | None = None,
    *,
    news_dir: Path = RAVENPACK_NEWS_DIR,
) -> pd.DataFrame:
    """Load labeled RavenPack rows (headline + score) from local rich exports."""
    paths = discover_ravenpack_article_files(tickers, news_dir=news_dir)
    if not paths:
        tickers_hint = ", ".join(tickers) if tickers else "any"
        raise FileNotFoundError(
            f"No RavenPack article exports found under {news_dir} for tickers: {tickers_hint}. "
            "Run notebooks/fetch_news_articles.ipynb to build "
            "`{{ticker}}_articles_2003_2014.parquet`."
        )

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        ticker = _ticker_from_article_path(path)
        if "ticker" not in df.columns:
            df = df.copy()
            df["ticker"] = ticker
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    if "article_date" not in combined.columns and "article_time" in combined.columns:
        combined["article_date"] = pd.to_datetime(combined["article_time"], errors="coerce").dt.normalize()
    combined["article_date"] = pd.to_datetime(combined["article_date"], errors="coerce")

    combined["headline"] = combined.get("headline", "").map(
        lambda v: re.sub(r"\s+", " ", str(v or "").strip())
    )
    combined["label_name"] = combined["event_sentiment_score"].map(score_to_label)

    labeled = combined[
        combined["label_name"].notna()
        & combined["headline"].astype(str).str.len().gt(0)
        & combined["article_date"].notna()
    ].copy()

    dedupe_col = "story_id" if "story_id" in labeled.columns else "rp_story_id"
    if dedupe_col in labeled.columns:
        labeled = labeled.drop_duplicates(subset=[dedupe_col], keep="first")

    labeled["label"] = labeled["label_name"].map(LABEL2ID).astype(int)
    return labeled.reset_index(drop=True)


def ravenpack_class_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Class counts for a labeled RavenPack frame."""
    counts = frame["label_name"].value_counts()
    return pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "pct": (counts.values / counts.sum() * 100).round(1),
    }).reset_index(drop=True)


def ravenpack_split_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Row counts per time split and class."""
    split = assign_time_split(frame["article_date"])
    rows = []
    for split_name in ("train", "validation", "test"):
        part = frame[split == split_name]
        rows.append({
            "split": split_name,
            "rows": len(part),
            "negative": int((part["label_name"] == "negative").sum()),
            "neutral": int((part["label_name"] == "neutral").sum()),
            "positive": int((part["label_name"] == "positive").sum()),
        })
    return pd.DataFrame(rows)


def assign_time_split(dates: pd.Series) -> pd.Series:
    """Assign train / validation / test from article dates."""
    ts = pd.to_datetime(dates, errors="coerce")
    out = pd.Series("train", index=dates.index, dtype="object")
    out[ts >= pd.Timestamp(TEST_START)] = "test"
    out[(ts >= pd.Timestamp(VAL_START)) & (ts <= pd.Timestamp(VAL_END))] = "validation"
    out[ts <= pd.Timestamp(TRAIN_END)] = "train"
    return out


def ravenpack_to_hf_dataset(frame: pd.DataFrame):
    """Build a Hugging Face ``DatasetDict`` with time-based splits."""
    from datasets import ClassLabel, Dataset, DatasetDict

    working = frame.copy()
    working["sentence"] = working["headline"]
    working["split"] = assign_time_split(working["article_date"])

    feature_label = ClassLabel(names=LABEL_NAMES)
    datasets: dict[str, Dataset] = {}
    for split_name in ("train", "validation", "test"):
        part = working[working["split"] == split_name][["sentence", "label"]]
        if part.empty:
            raise ValueError(f"RavenPack {split_name} split is empty — check date coverage.")
        datasets[split_name] = Dataset.from_pandas(part, preserve_index=False).cast_column(
            "label", feature_label
        )
    return DatasetDict(datasets)


def resolve_ravenpack_model_dir() -> Path:
    """Return the RavenPack fine-tune checkpoint directory if it exists."""
    directory = DEFAULT_RAVENPACK_MODEL_DIR
    if (directory / "config.json").exists() and (
        (directory / "model.safetensors").exists()
        or (directory / "pytorch_model.bin").exists()
    ):
        return directory
    return DEFAULT_RAVENPACK_MODEL_DIR


def ravenpack_model_is_saved(model_dir: Path | None = None) -> bool:
    directory = Path(model_dir) if model_dir is not None else resolve_ravenpack_model_dir()
    return (directory / "config.json").exists() and (
        (directory / "model.safetensors").exists()
        or (directory / "pytorch_model.bin").exists()
    )


def load_ravenpack_metrics(model_dir: Path | None = None) -> dict[str, Any] | None:
    directory = resolve_ravenpack_model_dir() if model_dir is None else Path(model_dir)
    path = metrics_path(directory)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def resolve_init_checkpoint(*, from_phrasebank: bool = True) -> str:
    """Prefer the PhraseBank checkpoint as the starting weights."""
    if from_phrasebank and model_is_saved(DEFAULT_MODEL_DIR):
        return str(DEFAULT_MODEL_DIR)
    return MODEL_NAME


def train_ravenpack(
    *,
    tickers: list[str] | None = None,
    output_dir: Path | None = None,
    init_from_phrasebank: bool = True,
    num_train_epochs: int = DEFAULT_RAVENPACK_TRAIN_EPOCHS,
    learning_rate: float = 2e-5,
    per_device_train_batch_size: int = 16,
    seed: int = 42,
) -> dict[str, Any]:
    """Continue fine-tuning DistilBERT on RavenPack headline labels."""
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    device = pick_device()
    output_dir = Path(output_dir or DEFAULT_RAVENPACK_MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = load_ravenpack_labeled_frame(tickers)
    raw = ravenpack_to_hf_dataset(frame)
    init_checkpoint = resolve_init_checkpoint(from_phrasebank=init_from_phrasebank)

    tokenizer = AutoTokenizer.from_pretrained(init_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        init_checkpoint,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    tokenized = tokenize_dataset(raw, tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=32,
        learning_rate=learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=100,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        compute_metrics=build_compute_metrics(),
    )

    train_result = trainer.train()
    val_metrics = trainer.evaluate(tokenized["validation"])
    test_metrics = trainer.evaluate(tokenized["test"])

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    tickers_used = sorted(frame["ticker"].astype(str).str.upper().unique().tolist())
    metrics = {
        "model_name": MODEL_NAME,
        "dataset": "ravenpack_headlines",
        "dataset_paths": [str(p) for p in discover_ravenpack_article_files(tickers)],
        "tickers": tickers_used,
        "labeled_rows": int(len(frame)),
        "split_source": SPLIT_SOURCE,
        "init_checkpoint": init_checkpoint,
        "score_threshold": SENTIMENT_SCORE_THRESHOLD,
        "epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "max_length": MAX_LENGTH,
        "metric_for_best_model": "f1",
        "train_loss": float(train_result.training_loss),
        "train_runtime_s": float(train_result.metrics.get("train_runtime", 0)),
        "validation": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in val_metrics.items()},
        "test": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in test_metrics.items()},
        "device": device,
        "saved_at": pd.Timestamp.utcnow().isoformat(),
    }
    metrics_path(output_dir).write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics
