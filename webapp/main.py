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

from webapp.api import ravenpack_finetune as rp
from webapp.jobs import job_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Sentiment LTR — FastAPI (migration POC)")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

NAV_ITEMS = [
    {"num": "1", "label": "Data Explorer", "href": "#", "enabled": False},
    {"num": "2", "label": "Batch Pipeline", "href": "#", "enabled": False},
    {"num": "3", "label": "PhraseBank Baseline", "href": "#", "enabled": False},
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

