from sentiment_ltr.wandb_logging import (
    checkpoint_wandb_links,
    configure_wandb_environment,
)
from webapp.api import sentiment_lab as sl


def test_wandb_environment_has_stable_project_defaults(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    links = configure_wandb_environment()
    assert links["project"] == "sentiment-ltr-transformers"
    assert links["project_url"].endswith(
        "/sentiment-ltr-transformers?nw=nwuserarmandoordorica"
    )


def test_imported_phrasebank_checkpoint_resolves_exact_run():
    links = checkpoint_wandb_links("phrasebank_distilbert_best")
    assert links["run_id"] == "ri5500fc"
    assert links["url"].endswith("/runs/ri5500fc?nw=nwuserarmandoordorica")


def test_metrics_run_url_overrides_checkpoint_fallback():
    links = checkpoint_wandb_links(
        "ravenpack_distilbert_best",
        {"wandb_run_id": "abc123", "wandb_run_url": "https://wandb.ai/e/p/runs/abc123"},
    )
    assert links["run_id"] == "abc123"
    assert links["url"] == (
        "https://wandb.ai/e/p/runs/abc123?nw=nwuserarmandoordorica"
    )


def test_every_inference_model_exposes_wandb_link():
    models = sl.available_models()
    assert models
    assert all(model["wandb"]["url"].startswith("https://wandb.ai/") for model in models)


def test_ravenpack_checkpoint_uses_accessible_project_fallback(monkeypatch):
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    links = checkpoint_wandb_links("ravenpack_distilbert_best")
    assert links["run_id"] is None
    assert links["run_url"] is None
    assert links["url"].endswith(
        "/sentiment-ltr-transformers?nw=nwuserarmandoordorica"
    )
