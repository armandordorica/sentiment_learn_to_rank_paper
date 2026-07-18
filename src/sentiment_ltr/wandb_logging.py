"""Weights & Biases logging helpers for the sentiment learning-to-rank project.

Separates W&B plumbing from notebook and script logic:
- Generic mechanics (table conversion, run initialisation) live here.
- Domain-specific metric assembly (RavenPack diagnostics) lives here as a
  thin wrapper that callers can invoke without knowing the W&B API.

All public functions accept plain Python / pandas objects and return early with
a descriptive error when ``wandb`` is not installed, so imports in the rest of
the codebase are never gated on the optional W&B dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_ltr.utils import slugify

DEFAULT_WANDB_ENTITY = "armando-ordorica-university-of-toronto"
DEFAULT_WANDB_PROJECT = "sentiment-ltr-transformers"
IMPORTED_CHECKPOINT_RUN_IDS = {
    "phrasebank_distilbert_best": "ri5500fc",
    "phrasebank_distilbert_1ep": "24rkyvrn",
}


def configure_wandb_environment() -> dict[str, str]:
    """Set stable project defaults before Hugging Face initializes W&B."""
    entity = os.environ.setdefault("WANDB_ENTITY", DEFAULT_WANDB_ENTITY)
    project = os.environ.setdefault("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    return {"entity": entity, "project": project,
            "project_url": f"https://wandb.ai/{entity}/{project}"}


def current_wandb_run_metadata() -> dict[str, str | None]:
    """Metadata for the active Trainer-created run, with a project fallback."""
    base = configure_wandb_environment()
    try:
        import wandb
        run = wandb.run
    except Exception:
        run = None
    return {**base, "run_id": getattr(run, "id", None),
            "run_url": getattr(run, "url", None)}


def checkpoint_wandb_links(model_id: str, metrics: dict[str, Any] | None = None) -> dict[str, str | None]:
    """Resolve a checkpoint-specific run URL, falling back to the project."""
    base = configure_wandb_environment()
    metrics = metrics or {}
    run_id = metrics.get("wandb_run_id") or IMPORTED_CHECKPOINT_RUN_IDS.get(model_id)
    run_url = metrics.get("wandb_run_url")
    if not run_url and run_id:
        run_url = f"{base['project_url']}/runs/{run_id}"
    return {**base, "run_id": run_id, "run_url": run_url,
            "url": run_url or base["project_url"]}


# ── Generic helpers ───────────────────────────────────────────────────────────

def df_to_wandb_table(df: pd.DataFrame):
    """Convert a DataFrame to a ``wandb.Table``.

    Categorical columns are coerced to ``str`` because W&B does not natively
    support the pandas ``CategoricalDtype``.

    Parameters
    ----------
    df:
        Any pandas DataFrame.

    Returns
    -------
    ``wandb.Table`` — importable only when ``wandb`` is installed.
    """
    import wandb

    clean = df.copy()
    for col in clean.columns:
        if isinstance(clean[col].dtype, pd.CategoricalDtype):
            clean[col] = clean[col].astype(str)
    return wandb.Table(dataframe=clean)


# ── RavenPack baseline diagnostics ───────────────────────────────────────────

def build_ravenpack_diagnostics_run_config(
    *,
    ticker: str,
    eval_split: str,
    ckpt_label: str,
    model_dir: Path,
    project_root: Path,
    notebook: str = "notebooks/finetune_on_ravenpack.ipynb",
) -> dict[str, Any]:
    """Assemble the W&B run ``config`` dict for a RavenPack diagnostics run.

    All values are plain Python scalars so the dict can be serialised directly
    to JSON or passed to ``wandb.init(config=...)``.
    """
    return {
        "ticker": ticker,
        "eval_split": eval_split,
        "checkpoint_label": ckpt_label,
        "checkpoint_path": (
            str(model_dir.relative_to(project_root))
            if model_dir.is_relative_to(project_root)
            else str(model_dir)
        ),
        "notebook": notebook,
        "diagnostic_source": "macro_f1_precision_recall_prevalence_cells",
    }


def build_ravenpack_diagnostics_metrics(
    *,
    f1_comparison: pd.DataFrame,
    class_f1_comparison: pd.DataFrame,
    prevalence_gap: pd.Series | dict,
) -> dict[str, float | int]:
    """Assemble the scalar summary-metrics dict for a RavenPack diagnostics run.

    Parameters
    ----------
    f1_comparison:
        DataFrame produced by the macro-F1 comparison cell; must contain columns
        ``dataset``, ``split``, ``macro_f1``, ``accuracy``, ``n_rows``.
    class_f1_comparison:
        DataFrame with per-class precision / recall / F1 / support; must contain
        ``evaluation``, ``label``, ``precision``, ``recall``, ``f1``, ``support``.
    prevalence_gap:
        Series or dict mapping sentiment label → prevalence gap in percentage
        points (``prediction_minus_actual_pp`` column from the prevalence cell).
    """
    pb_test = f1_comparison.loc[
        (f1_comparison["dataset"] == "PhraseBank") & (f1_comparison["split"] == "test")
    ].iloc[0]
    rp_eval = f1_comparison.loc[f1_comparison["dataset"] == "RavenPack"].iloc[0]

    gap = dict(prevalence_gap) if not isinstance(prevalence_gap, dict) else prevalence_gap

    metrics: dict[str, float | int] = {
        "diagnostics/phrasebank_test_macro_f1": float(pb_test["macro_f1"]),
        "diagnostics/phrasebank_test_accuracy": float(pb_test["accuracy"]),
        "diagnostics/ravenpack_macro_f1": float(rp_eval["macro_f1"]),
        "diagnostics/ravenpack_accuracy": float(rp_eval["accuracy"]),
        "diagnostics/domain_shift_macro_f1_gap": float(pb_test["macro_f1"] - rp_eval["macro_f1"]),
        "diagnostics/ravenpack_n_rows": int(rp_eval["n_rows"]),
        **{f"diagnostics/prevalence_gap_pp/{lbl}": float(v) for lbl, v in gap.items()},
    }
    for row in class_f1_comparison.itertuples(index=False):
        prefix = f"class_metrics/{slugify(row.evaluation)}/{row.label}"
        metrics[f"{prefix}/precision"] = float(row.precision)
        metrics[f"{prefix}/recall"] = float(row.recall)
        metrics[f"{prefix}/f1"] = float(row.f1)
        metrics[f"{prefix}/support"] = int(row.support)

    return metrics


def log_ravenpack_diagnostics(
    *,
    ticker: str,
    eval_split: str | None,
    ckpt_label: str,
    model_dir: Path,
    project_root: Path,
    f1_comparison: pd.DataFrame,
    class_f1_comparison: pd.DataFrame,
    precision_recall_long: pd.DataFrame,
    prevalence: pd.DataFrame,
    gap: pd.DataFrame,
    overall_metric_fig,
    class_f1_fig,
    precision_recall_fig,
    ood_prevalence_fig,
    project: str | None = None,
    entity: str | None = None,
) -> str:
    """Log a full RavenPack baseline-diagnostics run to Weights & Biases.

    Creates a single W&B run containing scalar summary metrics, four tables,
    and four Plotly figures.  Returns the run URL.

    Parameters
    ----------
    ticker:
        Ticker symbol evaluated (e.g. ``"AAPL"``).
    eval_split:
        Split name evaluated (``"train"``, ``"validation"``, ``"test"``, or
        ``None`` for all rows).
    ckpt_label:
        Human-readable checkpoint label (e.g. ``"PhraseBank (out-of-box)"``).
    model_dir:
        Path to the checkpoint directory.
    project_root:
        Root of the project repository (used for relative path display).
    f1_comparison, class_f1_comparison, precision_recall_long, prevalence, gap:
        DataFrames produced by the macro-F1 and prevalence notebook cells.
    overall_metric_fig, class_f1_fig, precision_recall_fig, ood_prevalence_fig:
        Plotly figures from the same cells.
    project:
        W&B project name; defaults to the ``WANDB_PROJECT`` env var or
        ``"sentiment-ltr-transformers"``.
    entity:
        W&B entity; defaults to the ``WANDB_ENTITY`` env var.
    """
    import wandb

    rp_split = eval_split or "all"
    wandb_project = project or os.getenv("WANDB_PROJECT", "sentiment-ltr-transformers")
    wandb_entity = entity or os.getenv("WANDB_ENTITY") or None

    run_name = (
        f"ravenpack-baseline-diagnostics"
        f"-{ticker.lower()}"
        f"-{rp_split}"
        f"-{slugify(ckpt_label)}"
    )

    run_config = build_ravenpack_diagnostics_run_config(
        ticker=ticker,
        eval_split=rp_split,
        ckpt_label=ckpt_label,
        model_dir=model_dir,
        project_root=project_root,
    )
    summary_metrics = build_ravenpack_diagnostics_metrics(
        f1_comparison=f1_comparison,
        class_f1_comparison=class_f1_comparison,
        prevalence_gap=gap["prediction_minus_actual_pp"].to_dict(),
    )

    with wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        job_type="baseline-diagnostics",
        tags=["ravenpack", "phrasebank-baseline", "diagnostics", "domain-shift"],
        config=run_config,
    ) as run:
        run.log({
            **summary_metrics,
            "tables/f1_comparison": df_to_wandb_table(f1_comparison),
            "tables/class_level_metrics": df_to_wandb_table(class_f1_comparison),
            "tables/precision_recall_by_class": df_to_wandb_table(precision_recall_long),
            "tables/ood_label_prevalence": df_to_wandb_table(prevalence),
            "tables/prevalence_gap": df_to_wandb_table(gap.reset_index()),
            "charts/overall_macro_f1_accuracy": overall_metric_fig,
            "charts/class_level_f1": class_f1_fig,
            "charts/precision_vs_recall": precision_recall_fig,
            "charts/ood_observed_vs_predicted_prevalence": ood_prevalence_fig,
        })
        run.summary.update(summary_metrics)
        url = run.url

    return url
