"""Model and data provenance helpers — reusable across checkpoints and notebooks.

All functions are model-framework-agnostic where possible: they accept plain
Python objects (dicts, Paths) rather than live ``transformers`` objects, so they
can be called from scripts, the Streamlit app, or other notebooks without
pulling in the full training stack.

Typical usage::

    from sentiment_ltr.provenance import build_checkpoint_provenance, save_provenance

    prov = build_checkpoint_provenance(
        checkpoint_label="RavenPack fine-tuned",
        model_dir=model_dir,
        project_root=PROJECT_ROOT,
        config_info=get_model_config_info(model),
        tokenizer_info=get_tokenizer_provenance(tokenizer, metrics),
        data_info=ravenpack_data_provenance(labeled, ticker, split_source),
    )
    save_provenance(prov, model_dir)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sentiment_ltr.utils import get_git_info, hash_file


# ── Weight-file snapshot ──────────────────────────────────────────────────────

def get_weight_files_snapshot(model_dir: Path) -> list[dict[str, Any]]:
    """Return a list of dicts describing every weight file in ``model_dir``.

    Each entry contains ``file`` (filename), ``size_bytes``, and ``sha256``
    (hex digest). Only ``.safetensors`` and ``.bin`` files are included.
    Deterministically sorted by filename.
    """
    weight_files = sorted(
        p for p in model_dir.glob("*")
        if p.suffix in {".safetensors", ".bin"}
    )
    return [
        {
            "file": p.name,
            "size_bytes": p.stat().st_size,
            "sha256": hash_file(p),
        }
        for p in weight_files
    ]


# ── Hugging Face model / tokenizer info ──────────────────────────────────────

def get_model_config_info(model) -> dict[str, Any]:
    """Extract label schema and architecture info from a loaded HF model.

    Parameters
    ----------
    model:
        A loaded ``transformers.PreTrainedModel`` instance.

    Returns
    -------
    dict with keys: ``num_labels``, ``id2label``, ``label2id``,
    ``architectures``, ``model_type``, ``full_config``.
    """
    config_dict = model.config.to_dict()
    return {
        "num_labels": model.config.num_labels,
        "id2label": model.config.id2label,
        "label2id": model.config.label2id,
        "architectures": config_dict.get("architectures"),
        "model_type": config_dict.get("model_type"),
        "full_config": config_dict,
    }


def get_tokenizer_provenance(tokenizer, metrics: dict[str, Any], max_length_key: str = "max_length") -> dict[str, Any]:
    """Extract tokenizer settings used at training time.

    Parameters
    ----------
    tokenizer:
        A loaded ``transformers.PreTrainedTokenizer`` (or Fast) instance.
    metrics:
        The ``metrics.json`` dict saved alongside the checkpoint.
    max_length_key:
        Key inside *metrics* that stores the training ``max_length``.
    """
    from sentiment_ltr.models.phrasebank_sentiment import MAX_LENGTH

    return {
        "tokenizer_class": type(tokenizer).__name__,
        "max_length_used": metrics.get(max_length_key, MAX_LENGTH),
        "padding_strategy": "max_length",
        "truncation": True,
        "model_max_length": tokenizer.model_max_length,
        "vocab_size": tokenizer.vocab_size,
    }


# ── Data provenance ───────────────────────────────────────────────────────────

def build_data_provenance(
    *,
    dataset_repo: str,
    dataset_config: str,
    split_sizes: dict[str, int],
    split_type: str,
    training_seed: int,
    split_content_sha256: dict[str, str],
) -> dict[str, Any]:
    """Assemble a structured data-provenance record.

    All parameters are plain Python values so this function works with any
    labeled dataset, not just PhraseBank or RavenPack.

    Parameters
    ----------
    dataset_repo:
        HF Hub id or data source description (e.g. ``"atrost/financial_phrasebank"``).
    dataset_config:
        Config name or split strategy string.
    split_sizes:
        Mapping of split name → row count, e.g. ``{"train": 3100, ...}``.
    split_type:
        Human-readable description of how splits were created.
    training_seed:
        RNG seed used by the Trainer.
    split_content_sha256:
        Mapping of split name → SHA-256 of ``(text, label)`` pairs, produced
        by :func:`sentiment_ltr.utils.hash_text_label_pairs`.
    """
    return {
        "dataset_repo": dataset_repo,
        "dataset_config": dataset_config,
        "split_sizes": split_sizes,
        "split_type": split_type,
        "training_seed": training_seed,
        "split_content_sha256": split_content_sha256,
    }


# ── Top-level provenance assembly & persistence ───────────────────────────────

def build_checkpoint_provenance(
    *,
    checkpoint_label: str,
    model_dir: Path,
    project_root: Path,
    config_info: dict[str, Any],
    tokenizer_info: dict[str, Any],
    data_info: dict[str, Any],
    repo_path: Path | None = None,
) -> dict[str, Any]:
    """Assemble a full provenance snapshot for a model checkpoint.

    Parameters
    ----------
    checkpoint_label:
        Human-readable label for this checkpoint (e.g. ``"RavenPack fine-tuned"``).
    model_dir:
        Path to the directory containing ``config.json`` and weight files.
    project_root:
        Root of the project; used to compute a relative checkpoint path.
    config_info:
        Output of :func:`get_model_config_info`.
    tokenizer_info:
        Output of :func:`get_tokenizer_provenance`.
    data_info:
        Output of :func:`build_data_provenance`.
    repo_path:
        Directory to use for git introspection; defaults to *project_root*.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": {
            "label": checkpoint_label,
            "path": str(model_dir.relative_to(project_root)),
        },
        "git": get_git_info(repo_path or project_root),
        "weights": get_weight_files_snapshot(model_dir),
        "model_config": config_info,
        "tokenizer": tokenizer_info,
        "data": data_info,
    }


def save_provenance(provenance: dict[str, Any], model_dir: Path) -> Path:
    """Write *provenance* to ``provenance.json`` inside *model_dir*.

    Returns the resolved path of the written file.
    """
    dest = model_dir / "provenance.json"
    dest.write_text(
        json.dumps(provenance, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return dest.resolve()


def print_provenance_summary(provenance: dict[str, Any], project_root: Path) -> None:
    """Print a compact human-readable summary of a provenance snapshot."""
    ckpt = provenance["checkpoint"]
    git = provenance["git"]
    weights = provenance["weights"]
    tok = provenance["tokenizer"]
    data = provenance["data"]

    print(f"  checkpoint      : {ckpt['label']} ({ckpt['path']})")
    print(f"  git commit      : {git['commit_hash_short']} (dirty={git['is_dirty']})")
    print(f"  weight files    : {[w['file'] for w in weights]}")
    print(f"  num_labels      : {provenance['model_config']['num_labels']}")
    print(f"  max_length      : {tok['max_length_used']} (padding={tok['padding_strategy']})")
    print(f"  split sizes     : {data['split_sizes']}")
