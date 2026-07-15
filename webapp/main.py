"""FastAPI app — Streamlit → FastAPI migration proof-of-concept.

Currently implements Tab 5·8 (RavenPack fine-tuning: train on 1 / 5 / N
tickers) as the first ported section. See
``docs/fastapi_migration_plan.md`` for overall migration status.

Run with:
    uvicorn webapp.main:app --reload --port 8001
(keep the Streamlit app, `streamlit run app.py`, running separately/side by side)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.api import phrasebank_baseline as pb
from webapp.api import ravenpack_finetune as rp
from webapp.api import data_explorer as de
from webapp.jobs import job_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Sentiment LTR — FastAPI (migration POC)")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

NAV_ITEMS = [
    {"num": "1", "label": "Data Explorer", "href": "/data-explorer", "enabled": True},
    {"num": "2", "label": "Batch Pipeline", "href": "#", "enabled": False},
    {"num": "3", "label": "PhraseBank Baseline", "href": "/phrasebank", "enabled": True},
    {"num": "4", "label": "RavenPack Baseline Eval", "href": "#", "enabled": False},
    {"num": "5", "label": "RavenPack Fine-Tuning", "href": "/finetune", "enabled": True},
    {"num": "6", "label": "Sentiment Lab", "href": "#", "enabled": False},
    {"num": "7", "label": "Paper Validation", "href": "#", "enabled": False},
]


def _base_context(active_href: str) -> dict[str, Any]:
    return {"nav_items": NAV_ITEMS, "active_href": active_href}


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        _base_context(active_href="/"),
    )


@app.get("/data-explorer", response_class=HTMLResponse)
def data_explorer_page(request: Request) -> HTMLResponse:
    ctx = _base_context(active_href="/data-explorer")
    ctx.update(de.page_defaults())
    ctx.update({"cache": de.cache_info("AAPL"), "result": None, "error": None})
    return templates.TemplateResponse(request, "data_explorer.html", ctx)


@app.post("/data-explorer/query", response_class=HTMLResponse)
def data_explorer_query(
    request: Request,
    ticker: str = Form(default="AAPL"),
    start_date: str = Form(default=de.DEFAULT_START),
    end_date: str = Form(default=de.DEFAULT_END),
    action: str = Form(default="load"),
    use_refinitiv: bool = Form(default=False),
    use_wrds: bool = Form(default=False),
    use_yahoo: bool = Form(default=False),
    use_ravenpack: bool = Form(default=False),
    include_refinitiv_news: bool = Form(default=False),
) -> HTMLResponse:
    try:
        raw = de.query(
            ticker, start_date, end_date, force_live=action == "live",
            refinitiv=use_refinitiv, wrds=use_wrds, yahoo=use_yahoo,
            ravenpack=use_ravenpack, include_news=include_refinitiv_news,
        )
        result, error = de.present(raw), None
    except Exception as exc:  # noqa: BLE001
        result, error = None, str(exc)
    return templates.TemplateResponse(
        request, "partials/data_explorer_results.html", {"result": result, "error": error}
    )


@app.get("/finetune", response_class=HTMLResponse)
def finetune_page(request: Request) -> HTMLResponse:
    """Tab 5·8 — mirrors ``render_ravenpack_finetuning_tab`` section 5·8 in app.py."""
    tickers = rp.available_tickers()
    default_tickers = rp.pilot_default_tickers(tickers)
    ctx = _base_context(active_href="/finetune")
    ctx.update({
        "deps": rp.deps_status(),
        "tickers": tickers,
        "default_tickers": default_tickers,
        "default_epochs": rp.DEFAULT_RAVENPACK_TRAIN_EPOCHS,
        "coverage": rp.coverage_summary(default_tickers) if default_tickers else None,
        "job": None,
        "error": None,
    })
    return templates.TemplateResponse(request, "finetune.html", ctx)


@app.post("/finetune/coverage", response_class=HTMLResponse)
def finetune_coverage(request: Request, tickers: list[str] = Form(default=[])) -> HTMLResponse:
    """HTMX partial: re-render the coverage table when the ticker selection changes."""
    coverage = rp.coverage_summary(tickers) if tickers else None
    return templates.TemplateResponse(
        request,
        "partials/coverage_table.html",
        {"coverage": coverage},
    )


@app.post("/finetune/train", response_class=HTMLResponse)
def finetune_train(
    request: Request,
    tickers: list[str] = Form(default=[]),
    init_from_phrasebank: bool = Form(default=False),
    num_train_epochs: int = Form(default=rp.DEFAULT_RAVENPACK_TRAIN_EPOCHS),
) -> HTMLResponse:
    """Kick off a background training job and return the polling status partial."""
    if not tickers:
        return templates.TemplateResponse(
            request,
            "partials/train_status.html",
            {"job": None, "error": "Select at least one ticker."},
        )

    job_id = job_manager.start(
        kind="ravenpack_finetune",
        fn=lambda job: rp.run_training(
            tickers,
            init_from_phrasebank=init_from_phrasebank,
            num_train_epochs=num_train_epochs,
        ),
    )
    job = job_manager.get(job_id)
    return templates.TemplateResponse(
        request,
        "partials/train_status.html",
        {"job": job, "error": None},
    )


@app.get("/finetune/train/{job_id}/status", response_class=HTMLResponse)
def finetune_train_status(request: Request, job_id: str) -> HTMLResponse:
    """Polled by HTMX every couple seconds while a training job is running."""
    job = job_manager.get(job_id)
    return templates.TemplateResponse(
        request,
        "partials/train_status.html",
        {"job": job, "error": None if job else "Job not found."},
    )


@app.get("/phrasebank", response_class=HTMLResponse)
def phrasebank_page(request: Request) -> HTMLResponse:
    """Tab 3 — mirrors Streamlit's ``render_phrasebank_hf_baseline_tab`` (3A/3C/3D)."""
    deps = pb.deps_status()
    ctx = _base_context(active_href="/phrasebank")
    ctx.update({
        "deps": deps,
        "training": pb.training_summary() if deps["has_phrasebank_checkpoint"] else None,
        "train_eval": None,
        "dataset": pb.dataset_dashboard() if deps["finetuning_deps_available"] else None,
        "probability_charts": pb.probability_charts() if deps["finetuning_deps_available"] else None,
    })
    return templates.TemplateResponse(request, "phrasebank.html", ctx)


@app.post("/phrasebank/train-eval", response_class=HTMLResponse)
def phrasebank_train_eval(request: Request) -> HTMLResponse:
    """HTMX partial: run (or fetch cached) train-split evaluation on button click.

    Calls the exact same ``evaluate_checkpoint_on_split("train", ...)`` as the
    Streamlit "▶ Evaluate on train split" button, so both UIs report identical
    macro-F1 / accuracy / confusion matrix numbers.
    """
    try:
        train_eval = pb.train_split_eval()
        error = None
    except Exception as exc:  # noqa: BLE001
        train_eval = None
        error = f"Could not evaluate train split: {exc}"
    return templates.TemplateResponse(
        request,
        "partials/phrasebank_train_eval.html",
        {"train_eval": train_eval, "error": error},
    )

