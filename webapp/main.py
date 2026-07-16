"""FastAPI app — Streamlit → FastAPI migration proof-of-concept.

Currently implements Tab 5·8 (RavenPack fine-tuning: train on 1 / 5 / N
tickers) as the first ported section. See
``docs/fastapi_migration_plan.md`` for overall migration status.

Run with:
    uvicorn webapp.main:app --reload --port 8001
(keep the Streamlit app, `streamlit run app.py`, running separately/side by side)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from webapp.api import batch_pipeline as bp
from webapp.api import phrasebank_baseline as pb
from webapp.api import ravenpack_eval as re_
from webapp.api import ravenpack_finetune as rp
from webapp.api import data_explorer as de
from webapp.api import sentiment_lab as sl
from webapp.jobs import job_manager

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEBAPP_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Sentiment LTR — FastAPI (migration POC)")
app.mount("/static", StaticFiles(directory=str(WEBAPP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(WEBAPP_DIR / "templates"))

NAV_ITEMS = [
    {"num": "1", "label": "Data Explorer", "href": "/data-explorer", "enabled": True},
    {"num": "2", "label": "Batch Pipeline", "href": "/batch", "enabled": True},
    {"num": "3", "label": "PhraseBank Baseline", "href": "/phrasebank", "enabled": True},
    {"num": "4", "label": "RavenPack Baseline Eval", "href": "/raven-eval", "enabled": True},
    {"num": "5", "label": "RavenPack Fine-Tuning", "href": "/finetune", "enabled": True},
    {"num": "6", "label": "Sentiment Lab", "href": "/sentiment-lab", "enabled": True},
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


@app.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request) -> HTMLResponse:
    """Tab 2 — mirrors ``render_batch_pipeline_tab()`` in app.py."""
    mdf = bp.load_manifests()
    ctx = _base_context(active_href="/batch")
    ctx.update({
        "status_ctx": bp.status_context(),
        "form": bp.form_defaults(),
        "snapshot": bp.snapshot(mdf),
        "fail_sections": bp.fail_reasons(mdf),
        "delisting": bp.delisting_context(mdf),
        "delisting_message": None, "delisting_error": None,
        "cash": bp.cash_merger_context(),
        "cash_message": None, "cash_error": None,
        "ticker_view": bp.ticker_table(mdf),
        "tickers": bp.ticker_options(mdf),
        "storage": bp.storage_context(),
    })
    return templates.TemplateResponse(request, "batch_pipeline.html", ctx)


@app.get("/batch/status", response_class=HTMLResponse)
def batch_status(request: Request) -> HTMLResponse:
    """HTMX partial: live banner + progress + log tail, self-polls while running."""
    return templates.TemplateResponse(
        request, "partials/batch_status.html",
        {"s": bp.status_context(), "message": None},
    )


@app.post("/batch/launch", response_class=HTMLResponse)
def batch_launch(
    request: Request,
    start_date: str = Form(default="2003-01-01"),
    end_date: str = Form(default="2014-12-31"),
    start_rank: int = Form(default=1),
    max_tickers: str = Form(default=""),
    sleep_sec: float = Form(default=0.25),
    stop_after: int = Form(default=25),
    provider_timeout: float = Form(default=300),
    year_timeout: int = Form(default=90),
    use_wrds: bool = Form(default=False),
    use_yahoo: bool = Form(default=False),
    use_ravenpack: bool = Form(default=False),
    use_refinitiv: bool = Form(default=False),
    force_rerun: bool = Form(default=False),
    rerun_failed: bool = Form(default=False),
    rerun_partial: bool = Form(default=False),
    combined_parquets: bool = Form(default=False),
) -> HTMLResponse:
    message = None
    if bp.pid_running() is not None:
        message = "A batch is already running — stop it before launching another."
    else:
        bp.launch(
            start=start_date, end=end_date,
            start_rank=start_rank,
            max_tickers=int(max_tickers) if max_tickers.strip().isdigit() else None,
            force_rerun=force_rerun, rerun_failed=rerun_failed, rerun_partial=rerun_partial,
            sleep_sec=sleep_sec, stop_after=stop_after,
            provider_timeout=provider_timeout, year_timeout=year_timeout,
            use_wrds=use_wrds, use_yahoo=use_yahoo,
            use_ravenpack=use_ravenpack, use_refinitiv=use_refinitiv,
            combined_parquets=combined_parquets,
        )
        time.sleep(1.5)  # give the runner a moment to write batch.pid
        message = "Batch launched."
    return templates.TemplateResponse(
        request, "partials/batch_status.html",
        {"s": bp.status_context(), "message": message},
    )


@app.post("/batch/stop", response_class=HTMLResponse)
def batch_stop(request: Request) -> HTMLResponse:
    message = bp.stop() or "No running batch process found."
    time.sleep(0.5)
    return templates.TemplateResponse(
        request, "partials/batch_status.html",
        {"s": bp.status_context(), "message": message},
    )


@app.post("/batch/tickers", response_class=HTMLResponse)
def batch_tickers(
    request: Request,
    status_filter: list[str] = Form(default=[]),
    ticker_filter: str = Form(default=""),
    only_failed: bool = Form(default=False),
) -> HTMLResponse:
    ticker_view = bp.ticker_table(
        bp.load_manifests(),
        status_filter=status_filter or None,
        ticker_filter=ticker_filter.strip(),
        only_failed=only_failed,
    )
    return templates.TemplateResponse(
        request, "partials/batch_ticker_table.html", {"ticker_view": ticker_view},
    )


@app.get("/batch/ticker", response_class=HTMLResponse)
def batch_ticker_detail(request: Request, ticker: str = "") -> HTMLResponse:
    if not ticker:
        return HTMLResponse("")  # "—" placeholder selected — clear the panel
    return templates.TemplateResponse(
        request, "partials/batch_ticker_detail.html",
        {"detail": bp.ticker_detail(bp.load_manifests(), ticker)},
    )


@app.get("/batch/log", response_class=HTMLResponse)
def batch_log(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/batch_log.html", {"log": bp.log_tail(100)},
    )


@app.post("/batch/delisting/fetch", response_class=HTMLResponse)
def batch_delisting_fetch(request: Request, mode: str = Form(default="missing")) -> HTMLResponse:
    n_new, error = bp.fetch_delisting(refetch_all=mode == "all")
    message = None
    if error is None:
        message = (f"Updated delisting cache — queried {n_new:,} PERMNO(s). "
                   f"Saved to {bp.DELISTING_CACHE_PATH.relative_to(bp.PROJECT_ROOT)}.")
    return templates.TemplateResponse(
        request, "partials/batch_delisting.html",
        {"delisting": bp.delisting_context(bp.load_manifests()),
         "delisting_message": message, "delisting_error": error},
    )


@app.post("/batch/cash-merger/fetch", response_class=HTMLResponse)
def batch_cash_merger_fetch(request: Request, mode: str = Form(default="missing")) -> HTMLResponse:
    n_new, error = bp.fetch_cash_merger(refetch_all=mode == "all")
    message = None
    if error is None:
        message = (f"Updated cash-merger cache — checked {n_new:,} PERMNO(s). "
                   f"Saved to {bp.CASH_MERGER_CACHE_PATH.relative_to(bp.PROJECT_ROOT)}.")
    return templates.TemplateResponse(
        request, "partials/batch_cash_merger.html",
        {"cash": bp.cash_merger_context(),
         "cash_message": message, "cash_error": error},
    )


@app.get("/batch/cash-merger.csv")
def batch_cash_merger_csv() -> Response:
    return Response(
        content=bp.cash_merger_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cash_merger_exits.csv"},
    )


@app.post("/batch/combined", response_class=HTMLResponse)
def batch_write_combined(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/batch_combined_result.html", {"written": bp.write_combined()},
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


@app.get("/raven-eval", response_class=HTMLResponse)
def raven_eval_page(request: Request) -> HTMLResponse:
    """Tab 4 — mirrors ``render_ravenpack_baseline_eval_tab()`` in app.py."""
    deps = re_.deps_status()
    tickers = (
        re_.available_tickers()
        if deps["finetuning_deps_available"] and deps["has_phrasebank_checkpoint"]
        else []
    )
    # Default to AAPL — currently the only ticker with a rich RavenPack export
    # (headline column); same reason rp.pilot_default_tickers() defaults to it.
    ticker = ("AAPL" if "AAPL" in tickers else tickers[0]) if tickers else None
    dataset, dataset_error = None, None
    if ticker:
        try:
            dataset = re_.dataset_summary(ticker)
        except Exception as exc:  # noqa: BLE001
            dataset_error = str(exc)
    ctx = _base_context(active_href="/raven-eval")
    ctx.update({
        "deps": deps,
        "tickers": tickers,
        "ticker": ticker,
        "dataset": dataset,
        "dataset_error": dataset_error,
        "eval_splits": re_.EVAL_SPLITS,
        "provenance": re_.provenance_context(),
    })
    return templates.TemplateResponse(request, "ravenpack_eval.html", ctx)


@app.get("/raven-eval/dataset", response_class=HTMLResponse)
def raven_eval_dataset(request: Request, ticker: str = "") -> HTMLResponse:
    """HTMX partial: dataset summary when the ticker selection changes."""
    dataset, dataset_error = None, None
    try:
        dataset = re_.dataset_summary(ticker)
    except Exception as exc:  # noqa: BLE001
        dataset_error = str(exc)
    return templates.TemplateResponse(
        request, "partials/raven_eval_dataset.html",
        {"ticker": ticker, "dataset": dataset, "dataset_error": dataset_error},
    )


@app.get("/raven-eval/static", response_class=HTMLResponse)
def raven_eval_static(
    request: Request,
    ticker: str = "",
    eval_split: str = "test",
    max_rows: int = 0,
) -> HTMLResponse:
    """HTMX partial: 4D distribution-shift charts + 4C class-metric dashboard.

    Requires scoring the selected split with the checkpoint (slow on first
    call, memoized afterwards) — hence loaded lazily after page render.
    """
    try:
        eval_result = re_.run_eval(ticker, eval_split, max_rows)
        ctx = {
            "shift": re_.distribution_shift_context(ticker, eval_result),
            "metrics": re_.class_metrics_context(eval_result),
            "static_error": None,
        }
    except Exception as exc:  # noqa: BLE001
        ctx = {"shift": None, "metrics": None, "static_error": str(exc)}
    return templates.TemplateResponse(request, "partials/raven_eval_static.html", ctx)


@app.post("/raven-eval/run", response_class=HTMLResponse)
def raven_eval_run(
    request: Request,
    ticker: str = Form(default=""),
    eval_split: str = Form(default="test"),
    max_rows: int = Form(default=0),
) -> HTMLResponse:
    """4E — run (or fetch memoized) baseline evaluation and render results."""
    try:
        result, eval_error = re_.eval_results_context(ticker, eval_split, max_rows), None
    except Exception as exc:  # noqa: BLE001
        result, eval_error = None, str(exc)
    return templates.TemplateResponse(
        request, "partials/raven_eval_results.html",
        {"result": result, "eval_error": eval_error},
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


@app.get("/sentiment-lab", response_class=HTMLResponse)
def sentiment_lab_page(request: Request) -> HTMLResponse:
    ctx = _base_context(active_href="/sentiment-lab")
    ctx.update(sl.page_context())
    try:
        ctx["dataset"], ctx["dataset_error"] = sl.phrasebank_dataset(), None
    except Exception as exc:  # noqa: BLE001
        ctx["dataset"], ctx["dataset_error"] = None, str(exc)
    return templates.TemplateResponse(request, "sentiment_lab.html", ctx)


@app.post("/sentiment-lab/articles", response_class=HTMLResponse)
def sentiment_lab_articles(request: Request, ticker: str = Form("AAPL"),
                           start: str = Form(sl.DEFAULT_START), end: str = Form(sl.DEFAULT_END),
                           max_rows: int = Form(50)) -> HTMLResponse:
    try:
        result, error = sl.cached_articles(ticker, start, end, max_rows), None
    except Exception as exc:  # noqa: BLE001
        result, error = None, str(exc)
    return templates.TemplateResponse(request, "partials/sentiment_articles.html",
                                      {"result": result, "error": error})


@app.post("/sentiment-lab/score", response_class=HTMLResponse)
def sentiment_lab_score(request: Request, text: str = Form(""),
                        model_ids: list[str] = Form(default=[])) -> HTMLResponse:
    try:
        results, error = sl.score(text, model_ids), None
    except Exception as exc:  # noqa: BLE001
        results, error = None, str(exc)
    return templates.TemplateResponse(request, "partials/sentiment_scores.html",
                                      {"results": results, "error": error})


@app.post("/sentiment-lab/train", response_class=HTMLResponse)
def sentiment_lab_train(request: Request) -> HTMLResponse:
    job_id = job_manager.start(kind="phrasebank_train", fn=sl.train)
    return templates.TemplateResponse(request, "partials/sentiment_train_status.html",
                                      {"job": job_manager.get(job_id), "error": None})


@app.get("/sentiment-lab/train/{job_id}/status", response_class=HTMLResponse)
def sentiment_lab_train_status(request: Request, job_id: str) -> HTMLResponse:
    job = job_manager.get(job_id)
    return templates.TemplateResponse(request, "partials/sentiment_train_status.html",
                                      {"job": job, "error": None if job else "Job not found."})
