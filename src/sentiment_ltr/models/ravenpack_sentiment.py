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
    github_blob_link,
    label_maps,
    load_phrasebank,
    metrics_path,
    model_is_saved,
    phrasebank_checkpoint_label_maps,
    pick_device,
    tokenize_dataset,
    train_baseline,
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


def _ravenpack_module_path() -> str:
    return Path(__file__).relative_to(PROJECT_ROOT).as_posix()


def _phrasebank_module_path() -> str:
    from sentiment_ltr.models import phrasebank_sentiment as pb_mod

    return Path(pb_mod.__file__).relative_to(PROJECT_ROOT).as_posix()


def _rp_fn_github_link(func, label: str | None = None) -> str:
    """Markdown link to a function definition in this module."""
    import inspect

    lines, start = inspect.getsourcelines(func)
    end = start + len(lines) - 1
    name = label or f"{func.__name__}()"
    return github_blob_link(_ravenpack_module_path(), start, end, label=f"`{name}`")


def _pb_fn_github_link(func, label: str | None = None) -> str:
    """Markdown link to a function definition in ``phrasebank_sentiment``."""
    import inspect

    lines, start = inspect.getsourcelines(func)
    end = start + len(lines) - 1
    name = label or f"{func.__name__}()"
    return github_blob_link(_phrasebank_module_path(), start, end, label=f"`{name}`")


def _rp_snippet_github_link(
    func,
    needle: str,
    *,
    label: str,
    span: int = 0,
) -> str:
    """Markdown link to a line range inside ``func`` that contains ``needle``."""
    import inspect

    source_lines, start = inspect.getsourcelines(func)
    for i, line in enumerate(source_lines):
        if needle in line:
            line_start = start + i
            line_end = line_start + span
            return github_blob_link(
                _ravenpack_module_path(),
                line_start,
                line_end,
                label=f"`{label}`",
            )
    return _rp_fn_github_link(func, label=label)


def ravenpack_parquet_summary(
    ticker: str,
    *,
    news_dir: Path = RAVENPACK_NEWS_DIR,
) -> dict[str, Any]:
    """Row count and column names for a ticker’s RavenPack export (metadata only)."""
    paths = discover_ravenpack_article_files([ticker], news_dir=news_dir)
    if not paths:
        return {}
    path = paths[0]
    try:
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        columns = list(parquet_file.schema_arrow.names)
        rows = parquet_file.metadata.num_rows
    except Exception:
        columns = []
        rows = 0
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "rows": int(rows),
        "columns": columns,
        "ticker": ticker.strip().upper(),
    }


def ravenpack_label_schema_table() -> pd.DataFrame:
    """Compare RavenPack, PhraseBank dataset, and checkpoint ``id2label`` maps."""
    rows: list[dict[str, object]] = []
    for label_id, name in ID2LABEL.items():
        rows.append({
            "source": "RavenPack (ID2LABEL)",
            "id": label_id,
            "label": name,
            "matches_ravenpack": True,
        })

    try:
        raw = load_phrasebank()
        _, ds_id2label, _ = label_maps(raw)
        for label_id in sorted(ds_id2label):
            name = ds_id2label[label_id]
            rows.append({
                "source": "PhraseBank HF dataset",
                "id": label_id,
                "label": name,
                "matches_ravenpack": name == ID2LABEL.get(label_id),
            })
    except Exception:
        pass

    if model_is_saved(DEFAULT_MODEL_DIR):
        try:
            ckpt_id2label, _ = phrasebank_checkpoint_label_maps()
            for label_id in sorted(ckpt_id2label):
                name = ckpt_id2label[label_id]
                rows.append({
                    "source": "PhraseBank checkpoint (config.json)",
                    "id": label_id,
                    "label": name,
                    "matches_ravenpack": name == ID2LABEL.get(label_id),
                })
        except Exception:
            pass

    return pd.DataFrame(rows)


def ravenpack_data_code_pointers() -> list[tuple[str, str]]:
    """GitHub links for the RavenPack data pipeline."""
    return [
        ("Discover parquet exports", _rp_fn_github_link(discover_ravenpack_article_files)),
        ("Load labeled frame", _rp_fn_github_link(load_ravenpack_labeled_frame)),
        ("Score → label rule", _rp_fn_github_link(score_to_label)),
        ("Time-based split", _rp_fn_github_link(assign_time_split)),
        ("Build HF DatasetDict", _rp_fn_github_link(ravenpack_to_hf_dataset)),
        ("Class balance", _rp_fn_github_link(ravenpack_class_balance)),
        ("Split summary", _rp_fn_github_link(ravenpack_split_summary)),
        (
            "Notebook: RavenPack fine-tune",
            github_blob_link(
                "notebooks/finetune_on_ravenpack.ipynb",
                label="`notebooks/finetune_on_ravenpack.ipynb`",
            ),
        ),
        (
            "Notebook: fetch raw exports",
            github_blob_link(
                "notebooks/fetch_news_articles.ipynb",
                label="`notebooks/fetch_news_articles.ipynb`",
            ),
        ),
    ]


def ravenpack_training_code_pointers() -> list[tuple[str, str]]:
    """GitHub links for RavenPack training and shared PhraseBank helpers."""
    return [
        ("train_ravenpack()", _rp_fn_github_link(train_ravenpack)),
        ("Resolve init checkpoint", _rp_fn_github_link(resolve_init_checkpoint)),
        (
            "Pass id2label to model head",
            _rp_snippet_github_link(
                train_ravenpack,
                "id2label=ID2LABEL",
                label="AutoModelForSequenceClassification(..., id2label=ID2LABEL)",
                span=3,
            ),
        ),
        ("Shared tokenize_dataset()", _pb_fn_github_link(tokenize_dataset)),
        ("PhraseBank label_maps()", _pb_fn_github_link(label_maps)),
        ("PhraseBank checkpoint id2label", _pb_fn_github_link(phrasebank_checkpoint_label_maps)),
        ("Shared compute_metrics", _pb_fn_github_link(build_compute_metrics)),
        ("PhraseBank train_baseline (init pattern)", _pb_fn_github_link(train_baseline)),
    ]


def ravenpack_finetune_config_recipe(
    ticker: str | None = None,
    metrics: dict[str, Any] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Grouped inputs/outputs, label schema, and training settings for RavenPack fine-tune."""
    m = metrics or {}
    epochs = m.get("epochs", DEFAULT_RAVENPACK_TRAIN_EPOCHS)
    lr = m.get("learning_rate", 2e-5)
    batch = m.get("per_device_train_batch_size", 16)
    max_len = m.get("max_length", MAX_LENGTH)
    init_ckpt = m.get("init_checkpoint", f"{DEFAULT_MODEL_DIR.relative_to(PROJECT_ROOT)} (if saved)")
    threshold = m.get("score_threshold", SENTIMENT_SCORE_THRESHOLD)

    parquet_bits: list[tuple[str, str]] = []
    labeled_bits: list[tuple[str, str]] = []
    if ticker:
        snap = ravenpack_parquet_summary(ticker)
        if snap:
            parquet_bits = [
                ("Ticker", snap["ticker"]),
                ("Parquet path", f"`{snap['path']}`"),
                ("Raw rows", f"{snap['rows']:,}"),
                ("Column count", str(len(snap["columns"]))),
                (
                    "Key input columns",
                    "headline, event_sentiment_score, article_date (or article_time), "
                    "story_id / rp_story_id, ticker, …",
                ),
                (
                    "All columns",
                    ", ".join(f"`{c}`" for c in snap["columns"][:12])
                    + (" …" if len(snap["columns"]) > 12 else ""),
                ),
            ]
            try:
                labeled = load_ravenpack_labeled_frame([ticker])
                splits = ravenpack_split_summary(labeled)
                parquet_bits.append(
                    ("Labeled rows (after filter + dedupe)", f"{len(labeled):,}")
                )
                for _, row in splits.iterrows():
                    labeled_bits.append((
                        f"HF `{row['split']}` split",
                        f"{int(row['rows']):,} rows "
                        f"(neg {int(row['negative']):,} · "
                        f"neu {int(row['neutral']):,} · "
                        f"pos {int(row['positive']):,})",
                    ))
            except Exception:
                pass

    label_rows: list[tuple[str, str]] = [
        ("RavenPack ID2LABEL", ", ".join(f"{i}→{n}" for i, n in ID2LABEL.items())),
        ("RavenPack LABEL2ID", ", ".join(f"{n}→{i}" for n, i in LABEL2ID.items())),
        ("Score → label", f"|event_sentiment_score| > {threshold} → pos/neg; else neutral"),
        ("Code: score_to_label()", _rp_fn_github_link(score_to_label)),
        (
            "Code: ID2LABEL / LABEL2ID",
            github_blob_link(_ravenpack_module_path(), 40, 41, label="`ID2LABEL` / `LABEL2ID`"),
        ),
    ]

    try:
        raw = load_phrasebank()
        _, ds_id2label, _ = label_maps(raw)
        label_rows.append((
            "PhraseBank HF dataset id2label",
            ", ".join(f"{i}→{n}" for i, n in sorted(ds_id2label.items())),
        ))
        label_rows.append(("Code: PhraseBank label_maps()", _pb_fn_github_link(label_maps)))
        aligned_ds = all(ds_id2label.get(i) == ID2LABEL.get(i) for i in range(len(LABEL_NAMES)))
        label_rows.append(("Dataset maps match RavenPack", "Yes" if aligned_ds else "No"))
    except Exception:
        label_rows.append(("PhraseBank HF dataset", "Run with finetuning deps to compare"))

    if model_is_saved(DEFAULT_MODEL_DIR):
        try:
            ckpt_id2label, _ = phrasebank_checkpoint_label_maps()
            label_rows.append((
                "PhraseBank checkpoint id2label",
                ", ".join(f"{i}→{n}" for i, n in sorted(ckpt_id2label.items())),
            ))
            label_rows.append((
                "Code: phrasebank_checkpoint_label_maps()",
                _pb_fn_github_link(phrasebank_checkpoint_label_maps),
            ))
            aligned_ckpt = all(ckpt_id2label.get(i) == ID2LABEL.get(i) for i in range(len(LABEL_NAMES)))
            label_rows.append(("Checkpoint maps match RavenPack", "Yes" if aligned_ckpt else "No"))
        except Exception:
            pass
    else:
        label_rows.append((
            "PhraseBank checkpoint",
            f"Not saved at `{DEFAULT_MODEL_DIR.relative_to(PROJECT_ROOT)}` — "
            "train PhraseBank first to verify checkpoint id2label",
        ))

    train_link = _rp_fn_github_link(train_ravenpack)
    tokenize_link = _pb_fn_github_link(tokenize_dataset)
    train_tokenize_invoke = _rp_snippet_github_link(
        train_ravenpack,
        "tokenize_dataset(raw, tokenizer)",
        label="tokenize_dataset(...) in train_ravenpack",
    )

    return {
        "Code pointers (data pipeline)": ravenpack_data_code_pointers(),
        "Code pointers (training & label schema)": ravenpack_training_code_pointers(),
        "Label schema (id2label)": label_rows,
        "RavenPack raw input (parquet)": parquet_bits or [
            ("Pattern", f"`{{ticker}}_articles_*.parquet` under `{RAVENPACK_NEWS_DIR.relative_to(PROJECT_ROOT)}/`"),
            ("Source notebook", github_blob_link(
                "notebooks/fetch_news_articles.ipynb",
                label="`fetch_news_articles.ipynb`",
            )),
            ("Code", _rp_fn_github_link(discover_ravenpack_article_files)),
        ],
        "Labeled frame → HF DatasetDict": [
            ("Filter", "valid score, non-empty headline, valid article_date"),
            ("Dedupe", "first row per story_id / rp_story_id"),
            ("Derived columns", "label_name (str), label (int 0/1/2), sentence (= headline)"),
            ("HF columns", "sentence, label (ClassLabel: negative / neutral / positive)"),
            ("Time splits", SPLIT_SOURCE),
            *labeled_bits,
            ("Code: load_ravenpack_labeled_frame()", _rp_fn_github_link(load_ravenpack_labeled_frame)),
            ("Code: ravenpack_to_hf_dataset()", _rp_fn_github_link(ravenpack_to_hf_dataset)),
        ],
        "Init checkpoint & model head": [
            ("Default init", f"PhraseBank checkpoint at `{DEFAULT_MODEL_DIR.relative_to(PROJECT_ROOT)}`"),
            ("Fallback init", MODEL_NAME),
            ("num_labels", str(len(LABEL_NAMES))),
            ("id2label / label2id", "RavenPack ID2LABEL / LABEL2ID (must match PhraseBank)"),
            ("init_checkpoint (last run)", str(init_ckpt)),
            ("Code: resolve_init_checkpoint()", _rp_fn_github_link(resolve_init_checkpoint)),
            ("Code: model init", _rp_snippet_github_link(
                train_ravenpack,
                "AutoModelForSequenceClassification.from_pretrained",
                label="AutoModelForSequenceClassification.from_pretrained(...)",
                span=5,
            )),
        ],
        "Tokenization (shared with PhraseBank)": [
            ("Input column", "sentence"),
            ("truncation", "True"),
            ("padding", "max_length"),
            ("max_length", str(max_len)),
            ("Application", "datasets.Dataset.map(batched=True)"),
            ("Code: tokenize_dataset()", tokenize_link),
            ("Code: wired in train_ravenpack()", train_tokenize_invoke),
        ],
        "Hugging Face TrainingArguments": [
            ("num_train_epochs", str(epochs)),
            ("per_device_train_batch_size", str(batch)),
            ("per_device_eval_batch_size", "32"),
            ("learning_rate", str(lr)),
            ("weight_decay", "0.01"),
            ("eval_strategy", "epoch"),
            ("save_strategy", "epoch"),
            ("load_best_model_at_end", "True"),
            ("metric_for_best_model", str(m.get("metric_for_best_model", "f1"))),
            ("greater_is_better", "True"),
            ("save_total_limit", "1"),
            ("logging_steps", "100"),
            ("report_to", "none"),
            ("seed", "42"),
            ("Code", f"{train_link} (`TrainingArguments` block)"),
        ],
        "Trainer & output": [
            ("train_dataset", "tokenized['train']"),
            ("eval_dataset", "tokenized['validation']"),
            ("test split", "evaluated after training; not used to pick the checkpoint"),
            ("compute_metrics", "accuracy + macro-F1 (shared with PhraseBank)"),
            ("Default output_dir", str(DEFAULT_RAVENPACK_MODEL_DIR.relative_to(PROJECT_ROOT))),
            ("Saved artifacts", "model weights, tokenizer, metrics.json"),
            ("Python API", train_link),
            ("Notebook", github_blob_link(
                "notebooks/finetune_on_ravenpack.ipynb",
                label="`finetune_on_ravenpack.ipynb`",
            )),
        ],
    }


def ravenpack_confusion_matrix_stylers(
    cm: pd.DataFrame,
    cm_pct: pd.DataFrame,
) -> tuple[object, object]:
    """Return pandas Styler objects for counts and row-normalized % confusion matrices."""
    counts_styler = (
        cm.style.format("{:,.0f}")
        .bar(align="mid", color=["red", "lightgreen"])
    )
    pct_styler = (
        cm_pct.round(1)
        .style.format("{:.1f}%")
        .bar(align="mid", color=["red", "lightgreen"], vmin=0, vmax=100)
    )
    return counts_styler, pct_styler


def evaluate_phrasebank_baseline_on_ravenpack(
    tickers: list[str] | None,
    *,
    model_dir: Path | None = None,
    eval_split: str | None = "test",
    batch_size: int = 64,
    max_rows: int | None = None,
    mismatch_sample: int = 10,
) -> dict[str, Any]:
    """Score RavenPack headlines with the PhraseBank checkpoint (no RavenPack fine-tune)."""
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    from sentiment_ltr.models.phrasebank_sentiment import load_classifier, predict_sentences

    model_dir = Path(model_dir or DEFAULT_MODEL_DIR)
    labeled = load_ravenpack_labeled_frame(tickers)
    labeled["split"] = assign_time_split(labeled["article_date"])
    if eval_split:
        eval_df = labeled[labeled["split"] == eval_split].reset_index(drop=True)
    else:
        eval_df = labeled.reset_index(drop=True)

    if eval_df.empty:
        split_hint = eval_split or "all"
        raise ValueError(f"No RavenPack rows for split={split_hint!r}.")

    if max_rows is not None and len(eval_df) > max_rows:
        eval_df = eval_df.sample(max_rows, random_state=42).reset_index(drop=True)

    tokenizer, model, device = load_classifier(model_dir)
    headlines = eval_df["headline"].tolist()
    pred_chunks: list[pd.DataFrame] = []
    for start in range(0, len(headlines), batch_size):
        batch = headlines[start : start + batch_size]
        pred_chunks.append(predict_sentences(batch, tokenizer, model, device))
    preds = pd.concat(pred_chunks, ignore_index=True)

    results = eval_df[
        ["split", "article_date", "headline", "event_sentiment_score", "label_name"]
    ].join(preds.drop(columns=["sentence"]))
    results = results.rename(columns={"label_name": "actual"})
    results["match"] = results["actual"] == results["pred"]

    label_order = [ID2LABEL[i] for i in sorted(ID2LABEL)]
    y_true = results["actual"]
    y_pred = results["pred"]

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=label_order),
        index=pd.Index(label_order, name="actual"),
        columns=pd.Index(label_order, name="pred"),
    )
    cm_pct = cm.div(cm.sum(axis=1), axis=0).mul(100)

    tickers_used = sorted(eval_df["ticker"].astype(str).str.upper().unique().tolist())
    mismatches = results.loc[~results["match"]]
    sample_n = min(mismatch_sample, len(mismatches))

    return {
        "model_dir": str(model_dir),
        "tickers": tickers_used,
        "eval_split": eval_split,
        "n_rows": int(len(eval_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_order, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=label_order,
            digits=3,
            zero_division=0,
        ),
        "confusion_counts": cm,
        "confusion_pct": cm_pct,
        "label_order": label_order,
        "mismatches_sample": mismatches.sample(sample_n, random_state=42).reset_index(drop=True)
        if sample_n
        else pd.DataFrame(),
    }
