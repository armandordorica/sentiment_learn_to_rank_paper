"""Streamlit app for CRSP universe validation charts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

try:
    import wrds
except ImportError:  # pragma: no cover - handled in the Streamlit UI
    wrds = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - handled in the Streamlit UI
    yf = None


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_ltr.data import live_data
from sentiment_ltr.data.provider_reason_codes import enrich_provider_status_records, reason_label
from sentiment_ltr.data.crsp_delisting import (
    DELISTING_CACHE_PATH,
    load_delisting_cache,
    update_delisting_cache,
)
from sentiment_ltr.data.cash_merger_exits import (
    CASH_MERGER_CACHE_PATH,
    CASH_MERGER_CODES,
    get_cash_merger_summary,
    load_cash_merger_cache,
    update_cash_merger_cache,
)
from sentiment_ltr.models.phrasebank_sentiment import (
    DEFAULT_MODEL_DIR,
    MODEL_NAME,
    PRIMARY_DATASET,
    SPLIT_SOURCE,
    benchmark_matmul,
    dataset_class_balance,
    device_report,
    finetuning_deps_available,
    load_classifier,
    load_metrics,
    load_phrasebank,
    model_is_saved,
    phrasebank_probability_chart_frame,
    predict_sentences,
    resolve_model_dir,
    train_baseline,
)
from sentiment_ltr.models.ravenpack_sentiment import (
    DEFAULT_RAVENPACK_MODEL_DIR,
    DEFAULT_RAVENPACK_TRAIN_EPOCHS,
    discover_ravenpack_article_files,
    load_ravenpack_labeled_frame,
    load_ravenpack_metrics,
    ravenpack_class_balance,
    ravenpack_model_is_saved,
    ravenpack_split_summary,
    resolve_ravenpack_model_dir,
    train_ravenpack,
)

# 1-epoch baseline numbers for the progress comparison table (Iteration 1).
PHRASEBANK_BASELINE_METRICS: dict[str, object] = {
    "epochs": 1,
    "validation": {"eval_accuracy": 0.7887},
    "test": {"eval_accuracy": 0.8062},
}
DEFAULT_TRAIN_EPOCHS_UI = 3
DEFAULT_RAVENPACK_TRAIN_EPOCHS_UI = DEFAULT_RAVENPACK_TRAIN_EPOCHS
PHRASEBANK_SPLIT_ORDER = ["train", "validation", "test"]

REFINITIV_IMPORT_ERROR: str | None = None


def _bind_refinitiv_helpers() -> bool:
    """Import or reload Refinitiv helpers, avoiding stale Streamlit module cache."""
    global REFINITIV_IMPORT_ERROR
    global query_refinitiv_news, query_refinitiv_prices, fetch_refinitiv_story
    global refinitiv_configured, refinitiv_setup_message, ticker_to_ric_candidates, refinitiv_session_mode
    global open_refinitiv_session, get_last_refinitiv_session_info

    try:
        import importlib

        from sentiment_ltr.data import refinitiv_queries, refinitiv_session

        refinitiv_queries = importlib.reload(refinitiv_queries)
        refinitiv_session = importlib.reload(refinitiv_session)
        query_refinitiv_news = refinitiv_queries.query_refinitiv_news
        query_refinitiv_prices = refinitiv_queries.query_refinitiv_prices
        fetch_refinitiv_story = refinitiv_queries.fetch_refinitiv_story
        refinitiv_configured = refinitiv_queries.refinitiv_configured
        refinitiv_setup_message = refinitiv_queries.refinitiv_setup_message
        ticker_to_ric_candidates = refinitiv_queries.ticker_to_ric_candidates
        refinitiv_session_mode = refinitiv_queries.refinitiv_session_mode
        open_refinitiv_session = refinitiv_session.open_refinitiv_session
        get_last_refinitiv_session_info = refinitiv_session.get_last_refinitiv_session_info
        REFINITIV_IMPORT_ERROR = None
        return True
    except ImportError as exc:
        query_refinitiv_news = None
        query_refinitiv_prices = None
        fetch_refinitiv_story = None
        open_refinitiv_session = None
        get_last_refinitiv_session_info = None
        REFINITIV_IMPORT_ERROR = str(exc)

        def refinitiv_configured(_project_root: Path) -> bool:
            return False

        def refinitiv_setup_message(_project_root: Path) -> str:
            if is_huggingface_space():
                return (
                    "Hosted Refinitiv needs LSEG Data Platform credentials. Add Space secrets "
                    "`LSEG_APP_KEY`, `LSEG_USERNAME`, and `LSEG_PASSWORD`."
                )
            if REFINITIV_IMPORT_ERROR:
                return (
                    "Refinitiv helpers failed to import. Restart Streamlit after "
                    f"`pip install -r requirements-refinitiv.txt`. Details: {REFINITIV_IMPORT_ERROR}"
                )
            return (
                "Install the Refinitiv SDK with "
                "`pip install -r requirements-refinitiv.txt`, then restart the Streamlit app."
            )

        def ticker_to_ric_candidates(_ticker: str) -> list[str]:
            return []

        def refinitiv_session_mode(_project_root: Path) -> str | None:
            return None

        return False


def refinitiv_status_label(project_root: Path) -> str:
    """Return a short Refinitiv readiness label for the status metric."""
    _bind_refinitiv_helpers()
    if not refinitiv_configured(project_root):
        return "Not configured"
    mode = refinitiv_session_mode(project_root)
    if mode == "platform":
        return "Cloud ready"
    if mode == "desktop":
        return "Workspace ready"
    return "Ready"


def is_huggingface_space() -> bool:
    """Return whether the app is running on a Hugging Face Space."""
    return bool(os.environ.get("SPACE_ID")) or os.environ.get("SYSTEM") == "spaces"


def format_refinitiv_news_error(error: str | None, session_info: object | None = None) -> str:
    """Turn verbose LSEG scope errors into a short, actionable message."""
    if not error:
        return "Refinitiv news is unavailable for this session."

    lowered = error.lower()
    used_cloud_fallback = isinstance(session_info, dict) and bool(session_info.get("fallback"))

    if "trapi.data.news.read" in error or "insufficient scope" in lowered:
        if used_cloud_fallback:
            return (
                "News is unavailable on the LSEG cloud fallback session. Your U of T cloud credentials "
                "do not include the `trapi.data.news.read` scope. Prices still work via cloud; news needs "
                "either a working local Workspace desktop session or a scope upgrade from U of T Rotman Library."
            )
        return (
            "News is unavailable on the current LSEG cloud session because the account is missing "
            "the `trapi.data.news.read` scope. Ask U of T Rotman Library to enable Refinitiv news access."
        )

    if used_cloud_fallback and ("session is not opened" in lowered or "application key" in lowered):
        return (
            "News requires a working local Workspace desktop session, but the app fell back to cloud "
            f"after desktop failed. Details: {error}"
        )

    return error


_bind_refinitiv_helpers()

APP_DATA_DIR = PROJECT_ROOT / "app_data"
DEFAULT_UNIVERSE_PATHS = [
    APP_DATA_DIR / "crsp_top_volume_universe.csv",
    PROJECT_ROOT / "data" / "raw" / "market" / "crsp_top_volume_universe.csv",
]
DEFAULT_MONTHLY_VOLUME_PATHS = [
    APP_DATA_DIR / "top20_monthly_volume.csv",
    PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_volume.csv",
]
DEFAULT_MONTHLY_PRICE_PATHS = [
    APP_DATA_DIR / "top20_monthly_prices.csv",
    PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_prices.csv",
]
DEFAULT_LOOKUP_START = "2003-01-01"
DEFAULT_LOOKUP_END = "2014-12-31"
QUICK_TEST_TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "TSLA"]

TOP1K_OUTPUT_DIR    = PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k"
TOP1K_BY_TICKER_DIR = TOP1K_OUTPUT_DIR / "by_ticker"
TOP1K_COMBINED_DIR  = TOP1K_OUTPUT_DIR / "combined"
BATCH_PROGRESS_CSV  = TOP1K_OUTPUT_DIR / "batch_progress.csv"
BATCH_STATUS_FILE   = TOP1K_OUTPUT_DIR / "batch_status.json"
BATCH_PID_FILE      = TOP1K_OUTPUT_DIR / "batch.pid"
BATCH_RUNNER_SCRIPT = PROJECT_ROOT / "scripts" / "run_batch_pipeline.py"
TOP1K_UNIVERSE_PATH = PROJECT_ROOT / "app_data" / "crsp_top_volume_universe.csv"
NEWS_RAVENPACK_DIR = PROJECT_ROOT / "data" / "raw" / "news" / "ravenpack"
NEWS_REFINITIV_DIR = PROJECT_ROOT / "data" / "raw" / "news" / "refinitiv"
DATE_RANGE_PRESETS = {
    "Last 7 calendar days": {"mode": "calendar", "days": 7},
    "Last 30 calendar days": {"mode": "calendar", "days": 30},
    "Last 90 calendar days": {"mode": "calendar", "days": 90},
    "Last 1 calendar year": {"mode": "calendar", "days": 365},
    "Latest 7 CRSP days": {"mode": "crsp", "days": 7},
    "Latest 30 CRSP days": {"mode": "crsp", "days": 30},
    "Latest 90 CRSP days": {"mode": "crsp", "days": 90},
    "Latest 1 CRSP year": {"mode": "crsp", "days": 365},
    "Paper window (2003-2014)": {"mode": "paper"},
}


def load_bundled_csv(fallback_paths: list[Path]) -> pd.DataFrame | None:
    """Load the first bundled CSV available from the fallback paths."""
    for fallback_path in fallback_paths:
        if fallback_path.exists():
            return pd.read_csv(fallback_path)
    return None


def get_secret_or_env(name: str) -> str | None:
    """Read a value from environment variables or Streamlit secrets."""
    env_value = os.environ.get(name)
    if env_value:
        return env_value

    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None
    return secret_value or None


def wrds_credential_status() -> dict[str, bool]:
    """Return non-sensitive WRDS credential presence checks."""
    return live_data.wrds_credential_status()


def wrds_credentials_available() -> bool:
    """Return whether the app has enough configuration for live WRDS queries."""
    return live_data.wrds_credentials_available()


def open_wrds_connection():
    """Open WRDS without allowing the library to fall back to interactive prompts."""
    return live_data.open_wrds_connection()


@st.cache_data(ttl=3600, show_spinner=False)
def query_wrds_ticker_data(ticker: str, start_date: str, end_date: str, row_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query CRSP name history and daily stock data for a ticker."""
    return live_data.query_wrds_ticker_data(ticker, start_date, end_date, row_limit)


def default_api_test_end(latest_crsp_date: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the latest date usable for CRSP smoke tests."""
    today = pd.Timestamp.today().normalize()
    if latest_crsp_date is None:
        return today
    return min(today, pd.Timestamp(latest_crsp_date).normalize())


def default_api_test_start(days: int = 30, latest_crsp_date: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return a recent start date for live API smoke tests."""
    return default_api_test_end(latest_crsp_date) - pd.Timedelta(days=days)


def resolve_api_test_range(
    preset_label: str,
    latest_crsp_date: pd.Timestamp,
    *,
    use_custom_dates: bool = False,
    custom_start: pd.Timestamp | None = None,
    custom_end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None, pd.Timestamp | None]:
    """Resolve requested and WRDS-clamped query windows from a preset or custom picker."""
    preset = DATE_RANGE_PRESETS[preset_label]
    today = pd.Timestamp.today().normalize()
    crsp_end = min(today, pd.Timestamp(latest_crsp_date).normalize())

    if preset["mode"] == "paper":
        start = pd.Timestamp(DEFAULT_LOOKUP_START)
        end = pd.Timestamp(DEFAULT_LOOKUP_END)
        return start, end, start, end

    if use_custom_dates:
        if custom_start is None or custom_end is None:
            raise ValueError("Custom dates are enabled but start/end were not provided.")
        requested_start = pd.Timestamp(custom_start).normalize()
        requested_end = pd.Timestamp(custom_end).normalize()
    elif preset["mode"] == "calendar":
        requested_end = today
        requested_start = today - pd.Timedelta(days=int(preset["days"]))
    else:
        requested_end = crsp_end
        requested_start = requested_end - pd.Timedelta(days=int(preset["days"]))

    if requested_start > requested_end:
        raise ValueError(
            f"Start date {requested_start.date()} is after end date {requested_end.date()}."
        )

    wrds_end = min(requested_end, crsp_end)
    wrds_start = requested_start
    if wrds_start > wrds_end:
        return requested_start, requested_end, None, None

    return requested_start, requested_end, wrds_start, wrds_end


def latest_crsp_fallback_range(
    latest_crsp_date: pd.Timestamp,
    days: int = 30,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the most recent CRSP window available in WRDS."""
    crsp_end = min(pd.Timestamp.today().normalize(), pd.Timestamp(latest_crsp_date).normalize())
    return crsp_end - pd.Timedelta(days=days), crsp_end


def calendar_preset_days(preset_label: str, default: int = 30) -> int:
    """Return the day count associated with a calendar preset."""
    preset = DATE_RANGE_PRESETS.get(preset_label, {})
    if preset.get("mode") != "calendar":
        return default
    return int(preset["days"])


@st.cache_data(ttl=3600, show_spinner=False)
def get_latest_crsp_date() -> pd.Timestamp:
    """Return the latest daily observation date available in WRDS CRSP."""
    return live_data.get_latest_crsp_date()


def google_finance_url(ticker: str) -> str:
    """Build a Google Finance quote URL for manual cross-checks."""
    clean_ticker = ticker.upper().strip()
    return f"https://www.google.com/finance/quote/{clean_ticker}"


@st.cache_data(ttl=600, show_spinner=False)
def load_refinitiv_story_text(story_id: str) -> str:
    """Fetch and cache a Refinitiv news story body by storyId."""
    if fetch_refinitiv_story is None:
        raise RuntimeError("Refinitiv story loading is unavailable in this environment.")
    return fetch_refinitiv_story(PROJECT_ROOT, story_id)


@st.cache_data(ttl=300, show_spinner=False)
def test_wrds_connection() -> dict[str, object]:
    """Run a minimal WRDS/CRSP query to verify credentials and database access."""
    return live_data.test_wrds_connection()


def to_query_date(value: pd.Timestamp | str) -> str:
    """Normalize a date-like value to YYYY-MM-DD for WRDS/Yahoo queries."""
    return live_data.to_query_date(value)


def _standardize_yahoo_daily(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a Yahoo Finance OHLCV frame to the app's daily schema."""
    return live_data._standardize_yahoo_daily(data, ticker)


def _yahoo_rate_limited(exc: Exception) -> bool:
    return live_data._yahoo_rate_limited(exc)


def _yahoo_network_blocked(exc: Exception) -> bool:
    """Return whether Yahoo was blocked by the local/cloud network path."""
    return live_data._yahoo_network_blocked(exc)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily Yahoo Finance prices for a public cross-check."""
    return live_data.fetch_yahoo_daily(ticker, start_date, end_date)


def compare_crsp_with_yahoo(crsp_daily: pd.DataFrame, yahoo_daily: pd.DataFrame) -> pd.DataFrame:
    """Compare CRSP and Yahoo closes on overlapping trading days."""
    crsp = crsp_daily.copy()
    crsp["date"] = pd.to_datetime(crsp["date"]).dt.normalize()
    crsp = crsp.sort_values("date").drop_duplicates("date", keep="last")
    crsp["crsp_close"] = crsp["prc"].abs()

    yahoo = yahoo_daily.copy()
    yahoo["date"] = pd.to_datetime(yahoo["date"]).dt.normalize()
    yahoo = yahoo.sort_values("date").drop_duplicates("date", keep="last")

    merged = crsp.merge(yahoo, on="date", how="inner", suffixes=("_crsp", "_yahoo"))
    if merged.empty:
        return merged

    merged["close_diff_usd"] = merged["crsp_close"] - merged["yahoo_close"]
    merged["close_diff_pct"] = 100 * merged["close_diff_usd"] / merged["yahoo_close"]
    if "vol" in merged.columns and "yahoo_volume" in merged.columns:
        merged["volume_diff_pct"] = 100 * (merged["vol"] - merged["yahoo_volume"]) / merged["yahoo_volume"]

    display_cols = [
        "date",
        "crsp_close",
        "yahoo_close",
        "close_diff_usd",
        "close_diff_pct",
    ]
    if "vol" in merged.columns and "yahoo_volume" in merged.columns:
        display_cols.extend(["vol", "yahoo_volume", "volume_diff_pct"])
    return merged[display_cols].sort_values("date", ascending=False)


def make_lookup_price_chart(daily_data: pd.DataFrame, ticker: str):
    """Build a close-price chart for an arbitrary WRDS ticker lookup."""
    if daily_data.empty:
        raise ValueError("No daily rows available to plot.")
    plot_data = daily_data.sort_values("date").copy()
    plot_data["close_price"] = plot_data["prc"].abs()
    fig = px.line(
        plot_data,
        x="date",
        y="close_price",
        color="permno",
        title=f"CRSP Daily Close Price For {ticker.upper()}",
        labels={"date": "Date", "close_price": "Absolute CRSP close price, USD", "permno": "PERMNO"},
        hover_data={
            "ticker": True,
            "comnam": True,
            "permno": True,
            "date": "|%Y-%m-%d",
            "close_price": ":,.2f",
            "ret": ":.4f",
            "vol": ":,",
        },
    )
    fig.update_traces(mode="lines+markers", marker={"size": 4})
    fig.update_layout(height=550, hovermode="closest")
    return fig


def prepare_universe(universe: pd.DataFrame) -> pd.DataFrame:
    """Prepare candidate-universe data for validation charts."""
    data = universe.copy()
    for column in ["first_trade_date", "last_trade_date", "latest_name_start", "latest_name_end"]:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    if "avg_volume_millions" not in data.columns:
        data["avg_volume_millions"] = data["avg_volume"] / 1_000_000
    if "avg_dollar_volume_billions" not in data.columns and "avg_dollar_volume" in data.columns:
        data["avg_dollar_volume_billions"] = data["avg_dollar_volume"] / 1_000_000_000

    if "volume_rank" in data.columns:
        data = data.sort_values("volume_rank")
    else:
        data = data.sort_values("avg_volume", ascending=False).assign(
            volume_rank=lambda frame: range(1, len(frame) + 1)
        )
    return data


def validation_summary(universe: pd.DataFrame) -> pd.DataFrame:
    """Return basic validation checks for the candidate universe."""
    checks = {
        "has_1000_rows": len(universe) == 1000,
        "volume_rank_is_unique": universe["volume_rank"].is_unique,
        "permno_is_unique": universe["permno"].is_unique,
        "avg_volume_descending": universe["avg_volume"].is_monotonic_decreasing,
        "common_share_codes": set(universe["shrcd"].dropna().astype(int)).issubset({10, 11}),
        "main_exchange_codes": set(universe["exchcd"].dropna().astype(int)).issubset({1, 2, 3}),
    }
    return pd.Series(checks, name="passed").to_frame()


def make_top20_bar_chart(top20: pd.DataFrame):
    """Build the interactive top-20 average-volume bar chart."""
    plot_data = top20.sort_values("avg_volume_millions", ascending=True).copy()
    plot_data["label"] = plot_data["ticker"] + " - " + plot_data["comnam"].str.title()

    fig = px.bar(
        plot_data,
        x="avg_volume_millions",
        y="label",
        orientation="h",
        title="Top 20 CRSP Common Stocks By Average Daily Share Volume, 2003-2014",
        labels={
            "avg_volume_millions": "Average daily volume, millions of shares",
            "label": "",
        },
        hover_data={
            "label": False,
            "ticker": True,
            "comnam": True,
            "permno": True,
            "volume_rank": True,
            "trading_days": ":,",
            "avg_volume_millions": ":,.2f",
            "avg_dollar_volume_billions": ":,.2f",
            "first_trade_date": "|%Y-%m-%d",
            "last_trade_date": "|%Y-%m-%d",
        },
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(
        height=700,
        hovermode="closest",
        yaxis={"categoryorder": "array", "categoryarray": plot_data["label"].tolist()},
    )
    return fig


def prepare_monthly_volume(volume_data: pd.DataFrame, top20: pd.DataFrame) -> pd.DataFrame:
    """Prepare monthly top-20 volume data from monthly or daily volume input."""
    data = volume_data.copy()

    if {"month", "ticker", "avg_daily_volume_millions", "trading_days"}.issubset(data.columns):
        data["month"] = pd.to_datetime(data["month"], errors="coerce")
        return data

    if {"date", "permno", "vol"}.issubset(data.columns):
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        lookup = top20[["permno", "ticker", "comnam"]].copy()
        data = data.merge(lookup, on="permno", how="inner")
        data["month"] = data["date"].dt.to_period("M").dt.to_timestamp()
        data["volume_millions"] = data["vol"] / 1_000_000
        return (
            data.groupby(["month", "ticker", "comnam"], as_index=False)
            .agg(
                avg_daily_volume_millions=("volume_millions", "mean"),
                trading_days=("volume_millions", "size"),
            )
            .sort_values(["ticker", "month"])
        )

    raise ValueError(
        "Monthly volume CSV must include either month/ticker/avg_daily_volume_millions/trading_days "
        "or daily CRSP-style date/permno/vol columns."
    )


def make_monthly_volume_chart(monthly_volume: pd.DataFrame):
    """Build the interactive top-20 monthly volume time-series chart."""
    plot_data = monthly_volume.copy()
    fig = px.line(
        plot_data.sort_values(["ticker", "month"]),
        x="month",
        y="avg_daily_volume_millions",
        color="ticker",
        title="Monthly Average Daily Trading Volume For Top 20 CRSP Candidates, 2003-2014",
        labels={
            "month": "Month",
            "avg_daily_volume_millions": "Average daily volume, millions of shares",
            "ticker": "Ticker",
        },
        hover_data={
            "ticker": True,
            "comnam": True,
            "month": "|%Y-%m",
            "avg_daily_volume_millions": ":,.2f",
            "trading_days": True,
        },
    )
    fig.update_traces(mode="lines+markers", line={"width": 1.8}, marker={"size": 4})
    fig.update_layout(height=750, legend_title_text="Ticker", hovermode="closest")
    return fig


def prepare_monthly_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare monthly top-20 price data for plotting."""
    data = price_data.copy()
    required = {"month", "ticker", "comnam", "open_price", "close_price", "avg_price"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Monthly price CSV is missing required columns: {sorted(missing)}")
    data["month"] = pd.to_datetime(data["month"], errors="coerce")
    return data.sort_values(["ticker", "month"])


def make_monthly_price_chart(monthly_prices: pd.DataFrame, ticker: str):
    """Build an interactive monthly open/close/average price chart for one ticker."""
    stock_prices = monthly_prices[monthly_prices["ticker"] == ticker].copy()
    if stock_prices.empty:
        raise ValueError(f"No monthly price data found for {ticker}.")

    company_name = stock_prices["comnam"].iloc[0]
    long_prices = stock_prices.melt(
        id_vars=["month", "ticker", "comnam", "trading_days"],
        value_vars=["open_price", "close_price", "avg_price"],
        var_name="price_type",
        value_name="price",
    )
    price_labels = {
        "open_price": "Open Price",
        "close_price": "Close Price",
        "avg_price": "Average Price",
    }
    long_prices["price_type"] = long_prices["price_type"].map(price_labels)

    fig = px.line(
        long_prices,
        x="month",
        y="price",
        color="price_type",
        title=f"Monthly Open, Close, and Average Price For {ticker} - {company_name.title()}",
        labels={
            "month": "Month",
            "price": "Price, USD",
            "price_type": "Series",
        },
        hover_data={
            "ticker": True,
            "comnam": True,
            "month": "|%Y-%m",
            "price_type": True,
            "price": ":,.2f",
            "trading_days": True,
        },
    )
    fig.update_traces(mode="lines+markers", line={"width": 2}, marker={"size": 5})
    fig.update_layout(height=650, hovermode="closest")
    return fig


def default_live_api_end() -> pd.Timestamp:
    """Return today's date for the live API tab."""
    return pd.Timestamp.today().normalize()


def default_live_api_start(days: int = 30) -> pd.Timestamp:
    """Return a default start date for the live API tab."""
    return default_live_api_end() - pd.Timedelta(days=days)


def wrds_price_frame(daily_lookup: pd.DataFrame) -> pd.DataFrame:
    """Convert CRSP daily rows to a common price schema."""
    return live_data.wrds_price_frame(daily_lookup)


def yahoo_price_frame(yahoo_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert Yahoo rows to a common price schema."""
    return live_data.yahoo_price_frame(yahoo_daily)


def make_provider_price_chart(price_data: pd.DataFrame, ticker: str, provider: str):
    """Build a close-price chart for any provider-specific price frame."""
    if price_data.empty:
        raise ValueError("No price rows available to plot.")
    plot_data = price_data.sort_values("date").copy()
    fig = px.line(
        plot_data,
        x="date",
        y="close_price",
        title=f"{provider.title()} Daily Close Price For {ticker.upper()}",
        labels={"date": "Date", "close_price": "Close price, USD"},
        hover_data={"date": "|%Y-%m-%d", "close_price": ":,.2f"},
    )
    fig.update_traces(mode="lines+markers", marker={"size": 4})
    fig.update_layout(height=550, hovermode="closest")
    return fig


def make_combined_price_chart(price_frames: dict[str, pd.DataFrame], ticker: str):
    """Overlay split-adjusted close prices from CRSP and Yahoo on one chart.

    Refinitiv is excluded here because it returns unadjusted prices — its
    pre-split values are orders-of-magnitude larger and would compress CRSP/Yahoo
    to invisible flat lines.  Refinitiv's own panel below shows its absolute prices.

    Both CRSP (cfacpr-adjusted) and Yahoo (retroactively split-adjusted) should
    overlay very closely, making this chart a good cross-provider sanity check.
    """
    # Exclude Refinitiv from the combined adjusted-price chart.
    adjusted_providers = {k: v for k, v in price_frames.items() if k != "refinitiv"}
    parts: list[pd.DataFrame] = []
    for provider, frame in adjusted_providers.items():
        if frame.empty:
            continue
        part = frame[["date", "close_price"]].copy().sort_values("date")
        part["provider"] = provider.title()
        parts.append(part)

    if not parts:
        # Fallback: show all providers if adjusted-only set is empty.
        for provider, frame in price_frames.items():
            if frame.empty:
                continue
            part = frame[["date", "close_price"]].copy().sort_values("date")
            part["provider"] = provider.title()
            parts.append(part)

    if not parts:
        raise ValueError("No provider price rows available to plot.")

    plot_data = pd.concat(parts, ignore_index=True).sort_values(["provider", "date"])
    fig = px.line(
        plot_data,
        x="date",
        y="close_price",
        color="provider",
        title=f"CRSP vs Yahoo Split-Adjusted Close Price — {ticker.upper()}",
        labels={"date": "Date", "close_price": "Split-adjusted close price (USD)", "provider": "Provider"},
        hover_data={"date": "|%Y-%m-%d", "close_price": ":,.4f"},
    )
    fig.update_traces(mode="lines", line={"width": 2})
    fig.update_layout(height=500, hovermode="x unified")
    return fig


def build_cross_provider_comparison(price_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge provider close prices onto one date index for side-by-side comparison."""
    merged: pd.DataFrame | None = None
    for provider, frame in price_frames.items():
        if frame.empty:
            continue
        part = frame[["date", "close_price"]].copy()
        part["date"] = pd.to_datetime(part["date"]).dt.normalize()
        part = part.sort_values("date").drop_duplicates("date", keep="last")
        part = part.rename(columns={"close_price": f"{provider}_close"})
        merged = part if merged is None else merged.merge(part, on="date", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame()
    return merged.sort_values("date", ascending=False)


def run_live_api_query(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    query_refinitiv: bool = True,
    query_wrds: bool = True,
    query_yahoo: bool = True,
    query_ravenpack: bool = False,
    news_count: int = 50,
    wrds_limit: int = 500,
    latest_crsp_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Query selected market-data providers in parallel for the same ticker and date range."""
    return live_data.run_ticker_data_query(
        PROJECT_ROOT,
        ticker,
        start_date,
        end_date,
        query_refinitiv=query_refinitiv,
        query_wrds=query_wrds,
        query_yahoo=query_yahoo,
        query_ravenpack=query_ravenpack,
        news_count=news_count,
        wrds_limit=wrds_limit,
        latest_crsp_date=latest_crsp_date,
    )


def _provider_status_label(result: dict[str, object]) -> str:
    """Return a short status label for a provider result block."""
    status = str(result.get("status", "unknown"))
    if status == "ok":
        prices = result.get("prices")
        row_count = len(prices) if isinstance(prices, pd.DataFrame) else 0
        return f"OK ({row_count:,} rows)"
    if status == "empty":
        reason = result.get("fail_reason_label") or result.get("fail_reason")
        return f"No rows — {reason}" if reason else "No rows"
    if status == "unavailable":
        reason = result.get("fail_reason_label") or result.get("fail_reason")
        return f"Unavailable — {reason}" if reason else "Unavailable"
    if status == "skipped":
        return "Skipped"
    reason = result.get("fail_reason_label") or result.get("fail_reason")
    if reason:
        return f"Failed — {reason}"
    return "Failed"


def render_provider_price_column(
    column,
    provider_name: str,
    provider_result: dict[str, object],
    ticker: str,
    *,
    key_prefix: str | None = None,
) -> None:
    """Render one provider's price panel inside a Streamlit column."""
    with column:
        st.markdown(f"#### {provider_name}")
        st.caption(_provider_status_label(provider_result))
        prices = provider_result.get("prices")
        if provider_result.get("status") == "ok" and isinstance(prices, pd.DataFrame) and not prices.empty:
            st.plotly_chart(
                make_provider_price_chart(prices, ticker, provider_name.lower()),
                use_container_width=True,
                key=f"{key_prefix}_{provider_name.lower()}_price_chart" if key_prefix else None,
            )
            st.dataframe(prices.sort_values("date", ascending=False), use_container_width=True, height=250)
        else:
            st.warning(str(provider_result.get("error") or provider_result.get("status")))


def make_news_daily_count_chart(daily: pd.DataFrame, ticker: str):
    """Build a bar chart of non-zero daily Refinitiv article counts."""
    plot_data = daily.copy()
    plot_data["date"] = pd.to_datetime(plot_data["date"])
    plot_data = plot_data[plot_data["article_count"] > 0].sort_values("date")
    if plot_data.empty:
        raise ValueError("No Refinitiv articles were found on any day in the selected range.")

    fig = px.bar(
        plot_data,
        x="date",
        y="article_count",
        title=f"{ticker.upper()} Refinitiv Articles Per Day",
        labels={"date": "Date", "article_count": "Articles"},
        hover_data={"date": "|%Y-%m-%d", "article_count": True},
    )
    fig.update_layout(hovermode="closest", height=420)
    return fig


def _selected_date_from_plotly(selection: object | None) -> pd.Timestamp | None:
    """Parse a selected bar date from a Streamlit Plotly selection event."""
    if selection is None or not hasattr(selection, "selection"):
        return None
    points = getattr(selection.selection, "points", None) or []
    if not points:
        return None
    x_value = points[0].get("x")
    if x_value is None:
        return None
    return pd.Timestamp(x_value).normalize()


def _render_readonly_story_text(story_text: str) -> None:
    """Display story body without a Streamlit input widget (avoids stale widget state)."""
    with st.container(height=420, border=True):
        st.write(story_text)


def render_refinitiv_story_body(
    story_id: str,
    headline: str | None = None,
    *,
    close_key: str,
    story_id_key: str = "refinitiv_open_story_id",
    story_headline_key: str = "refinitiv_open_story_headline",
) -> None:
    """Render the full Refinitiv news story text for one storyId."""
    st.markdown("---")
    st.markdown(f"**{headline or 'Selected headline'}**")
    with st.spinner("Loading full Refinitiv story..."):
        try:
            story_text = load_refinitiv_story_text(story_id)
        except Exception as exc:
            st.error(f"Could not load story: {exc}")
            return

    _render_readonly_story_text(story_text)
    if st.button("Close story", key=close_key):
        st.session_state.pop(story_id_key, None)
        st.session_state.pop(story_headline_key, None)
        st.session_state.pop("dashboard_news_story_text", None)  # legacy widget key
        st.query_params.pop("refinitiv_story", None)
        st.query_params.pop("news_date", None)
        st.rerun()


def render_refinitiv_news_coverage_section(
    news_df: pd.DataFrame,
    daily_counts: pd.DataFrame,
    news_summary: dict[str, object] | None,
    ticker: str,
    *,
    story_id_key: str = "refinitiv_open_story_id",
    story_headline_key: str = "refinitiv_open_story_headline",
    embed_story: bool = True,
) -> None:
    """Render daily news counts with drill-down into headline rows for one day."""
    news_date_param = st.query_params.get("news_date")
    if news_date_param:
        st.session_state.news_coverage_selected_date = pd.Timestamp(news_date_param).normalize()

    st.markdown("#### Refinitiv News Coverage")
    st.caption(
        "Daily counts use deduplicated Refinitiv headline `storyId` values. "
        "Click a bar or daily-count row to inspect the exact headlines included in that day's total."
    )

    if news_summary:
        metric_cols = st.columns(4)
        metric_cols[0].metric("Total articles", f"{int(news_summary.get('total_articles', 0)):,}")
        metric_cols[1].metric(
            "Avg articles / week",
            f"{float(news_summary.get('avg_articles_per_week', 0.0)):.2f}",
        )
        metric_cols[2].metric(
            "Days with news",
            f"{int(news_summary.get('calendar_days_with_news', 0)):,}",
        )
        threshold_label = "Pass" if news_summary.get("passes_paper_weekly_threshold") else "Fail"
        metric_cols[3].metric("Paper weekly rule", threshold_label)

    version = st.session_state.get("refinitiv_news_version", 0)
    nonzero_daily = daily_counts.copy()
    nonzero_daily["date"] = pd.to_datetime(nonzero_daily["date"])
    nonzero_daily = nonzero_daily[nonzero_daily["article_count"] > 0].sort_values("date", ascending=False)

    try:
        fig = make_news_daily_count_chart(daily_counts, ticker)
    except ValueError as exc:
        st.info(str(exc))
        return

    chart_selection = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key=f"refinitiv_news_daily_chart_{version}",
    )
    chart_date = _selected_date_from_plotly(chart_selection)
    if chart_date is not None:
        st.session_state.news_coverage_selected_date = chart_date

    daily_display = nonzero_daily.copy()
    daily_display["date_label"] = daily_display["date"].dt.strftime("%Y-%m-%d")
    daily_table = st.dataframe(
        daily_display[["date_label", "article_count"]].rename(
            columns={"date_label": "date", "article_count": "articles"}
        ),
        use_container_width=True,
        hide_index=True,
        height=220,
        on_select="rerun",
        selection_mode="single-row",
        key=f"refinitiv_news_daily_table_{version}",
    )
    if daily_table is not None and hasattr(daily_table, "selection") and daily_table.selection.rows:
        row_idx = int(daily_table.selection.rows[0])
        if 0 <= row_idx < len(daily_display):
            st.session_state.news_coverage_selected_date = pd.Timestamp(daily_display.iloc[row_idx]["date"]).normalize()

    selected_date = st.session_state.get("news_coverage_selected_date")
    if selected_date is not None:
        selected_date = pd.Timestamp(selected_date).normalize()
        if selected_date < pd.Timestamp(daily_counts["date"].min()) or selected_date > pd.Timestamp(
            daily_counts["date"].max()
        ):
            selected_date = None

    if selected_date is None:
        st.info("Select a day from the chart or the daily counts table to inspect headline rows.")
        return

    from sentiment_ltr.data.news_coverage import filter_headlines_by_date

    day_news = filter_headlines_by_date(news_df, selected_date)
    st.markdown(f"##### Headlines on **{selected_date.strftime('%Y-%m-%d')}**")
    st.caption(f"{len(day_news):,} headline(s) counted on this day. The `#` column runs 1–{len(day_news):,} for verification.")
    if day_news.empty:
        st.warning("No headline rows matched the selected day.")
        return

    render_refinitiv_news_headlines(
        day_news,
        table_key_suffix=f"_{selected_date.strftime('%Y%m%d')}",
        show_section_title=False,
        story_id_key=story_id_key,
        story_headline_key=story_headline_key,
        embed_story=embed_story,
    )


def render_refinitiv_news_headlines(
    news_df: pd.DataFrame,
    *,
    table_key_suffix: str = "",
    show_section_title: bool = True,
    story_id_key: str = "refinitiv_open_story_id",
    story_headline_key: str = "refinitiv_open_story_headline",
    embed_story: bool = True,
) -> None:
    """Render Refinitiv headlines with clickable links to read full stories."""
    if news_df.empty:
        return

    if show_section_title:
        st.markdown("#### Refinitiv News Headlines")
        st.caption(
            "Click a headline link to load the full story from Refinitiv Workspace. "
            "These stories are licensed content, not public web pages."
        )
    elif fetch_refinitiv_story is not None:
        st.caption("Click a headline link below to load the full story text.")

    if "storyId" not in news_df.columns:
        indexed_df = news_df.copy().reset_index(drop=True)
        indexed_df.insert(0, "#", range(1, len(indexed_df) + 1))
        summary_cols = ["#"] + [col for col in ["date", "headline", "sourceCode"] if col in indexed_df.columns]
        st.dataframe(indexed_df[summary_cols], use_container_width=True, hide_index=True)
        st.warning("Headline rows did not include a `storyId`, so full stories cannot be opened.")
        return

    if fetch_refinitiv_story is None:
        st.warning("Refinitiv story loading is unavailable in this environment.")
        return

    display_df = news_df.copy().reset_index(drop=True)
    display_df.insert(0, "#", range(1, len(display_df) + 1))
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d %H:%M")

    header = st.columns([0.5, 1.4, 6.3, 1.2])
    header[0].markdown("**#**")
    header[1].markdown("**Date**")
    header[2].markdown("**Headline**")
    header[3].markdown("**Source**")

    for _, row in display_df.iterrows():
        cols = st.columns([0.5, 1.4, 6.3, 1.2], gap="small")
        cols[0].write(str(row["#"]))
        cols[1].write(str(row["date"]))
        if cols[2].button(
            str(row["headline"]),
            key=f"story_link_{table_key_suffix}_{row['#']}",
            type="tertiary",
            use_container_width=True,
        ):
            st.session_state[story_id_key] = str(row["storyId"])
            st.session_state[story_headline_key] = str(row["headline"])
            st.session_state.pop("dashboard_news_story_text", None)  # legacy widget key
            st.rerun()
        source_code = row.get("sourceCode", "")
        cols[3].write("" if pd.isna(source_code) else str(source_code))

    if embed_story:
        story_id = st.session_state.get(story_id_key) or st.query_params.get("refinitiv_story")
        if story_id:
            match = display_df[display_df["storyId"].astype(str) == str(story_id)]
            headline = st.session_state.get(story_headline_key)
            if headline is None and not match.empty:
                headline = str(match.iloc[0]["headline"])
            render_refinitiv_story_body(
                str(story_id),
                str(headline) if headline is not None else None,
                close_key=f"close_story_{table_key_suffix}_{story_id}",
                story_id_key=story_id_key,
                story_headline_key=story_headline_key,
            )
        else:
            st.caption("Click a headline link to read the full story below.")


def render_live_api_results(query_result: dict[str, object]) -> None:
    """Render parallel multi-provider API query results side by side."""
    ticker = str(query_result["ticker"])
    providers = query_result["providers"]
    price_frames = query_result["price_frames"]
    start_date = query_result["start_date"]
    end_date = query_result["end_date"]

    st.caption(f"Requested window: **{start_date}** to **{end_date}** for **{ticker}**")

    status_cols = st.columns(3)
    status_cols[0].metric("Refinitiv", _provider_status_label(providers["refinitiv"]))
    status_cols[1].metric("WRDS/CRSP", _provider_status_label(providers["wrds"]))
    status_cols[2].metric("Yahoo Finance", _provider_status_label(providers["yahoo"]))

    if providers["refinitiv"].get("status") == "unavailable" and refinitiv_configured(PROJECT_ROOT):
        st.info(
            "Refinitiv looks configured now, but this panel still shows an older unavailable result. "
            "Click **Run parallel query** to refresh."
        )

    refinitiv = providers["refinitiv"]
    session_info = refinitiv.get("session_info")
    if isinstance(session_info, dict) and session_info.get("fallback"):
        st.info(str(session_info.get("message") or "Using LSEG cloud API after desktop Workspace failed."))
    refinitiv_news = refinitiv.get("news")
    news_daily_counts = refinitiv.get("news_daily_counts")
    news_summary = refinitiv.get("news_summary")
    selected = query_result.get("selected_providers", {})
    if selected.get("refinitiv", True) and isinstance(news_summary, dict):
        render_refinitiv_news_coverage_section(
            refinitiv_news if isinstance(refinitiv_news, pd.DataFrame) else pd.DataFrame(),
            news_daily_counts if isinstance(news_daily_counts, pd.DataFrame) else pd.DataFrame(),
            news_summary,
            ticker,
        )
    elif refinitiv.get("error") and refinitiv.get("status") == "ok":
        st.warning(format_refinitiv_news_error(str(refinitiv["error"]), session_info))
        with st.expander("Technical details"):
            st.code(str(refinitiv["error"]))

    if len(price_frames) >= 2:
        st.markdown("#### Combined Close Price Comparison")
        st.plotly_chart(make_combined_price_chart(price_frames, ticker), use_container_width=True)
        comparison = build_cross_provider_comparison(price_frames)
        if not comparison.empty:
            st.dataframe(comparison.head(30), use_container_width=True)

    active_panels = [
        ("Refinitiv", providers["refinitiv"]),
        ("WRDS/CRSP", providers["wrds"]),
        ("Yahoo Finance", providers["yahoo"]),
    ]
    visible_panels = [
        (name, result)
        for (name, result), key in zip(active_panels, ["refinitiv", "wrds", "yahoo"])
        if selected.get(key, True)
    ]
    if visible_panels:
        panel_cols = st.columns(len(visible_panels))
        for column, (name, result) in zip(panel_cols, visible_panels):
            render_provider_price_column(column, name, result, ticker)

    if not price_frames:
        st.error("No price data returned from the selected providers for that ticker and date range.")


def render_live_api_test_tab() -> None:
    """Render the live API smoke-test tab with parallel multi-provider queries."""
    st.subheader("Live API Test")
    st.caption(
        "Query Refinitiv, WRDS/CRSP, and Yahoo Finance in parallel for the same ticker and calendar dates. "
        "Results render side by side so you can compare providers directly."
    )
    st.info(
        "There is no official Google Finance API for programmatic price pulls. "
        "Use the Google Finance link below for manual checks; Yahoo Finance is the free public benchmark in this app."
    )
    st.warning(
        "Refinitiv can use either local LSEG Workspace or LSEG Data Platform cloud credentials. "
        "WRDS credentials should only be enabled where sharing returned CRSP data is permitted under your data-use terms."
    )

    if is_huggingface_space():
        st.caption(
            "Yahoo Finance is best-effort on this hosted Space. Yahoo may rate-limit shared cloud IPs, "
            "so WRDS and Refinitiv are the more reliable live providers here."
        )
        if refinitiv_configured(PROJECT_ROOT):
            st.info(
                "Refinitiv cloud credentials are configured on this Space. "
                "Queries use the LSEG Data Platform API (no local Workspace required)."
            )
        else:
            st.info(
                "To enable Refinitiv on this hosted Space, add secrets "
                "`LSEG_APP_KEY`, `LSEG_USERNAME`, and `LSEG_PASSWORD` from U of T's RDP cloud API access."
            )

    status_cols = st.columns(3)
    status_cols[0].metric("Refinitiv", refinitiv_status_label(PROJECT_ROOT))
    status_cols[1].metric(
        "WRDS",
        "Ready" if wrds_credentials_available() else "Not configured",
    )
    status_cols[2].metric("Yahoo", "Ready")

    if not refinitiv_configured(PROJECT_ROOT):
        st.caption(
            "Local runs need `LSEG_APP_KEY` plus Workspace open. "
            "Hosted runs need `LSEG_APP_KEY`, `LSEG_USERNAME`, and `LSEG_PASSWORD` as Space secrets."
        )

    latest_crsp_date: pd.Timestamp | None = None
    if wrds_credentials_available():
        try:
            latest_crsp_date = get_latest_crsp_date()
            st.session_state.latest_crsp_date = latest_crsp_date
            st.caption(f"WRDS CRSP backup coverage runs through **{latest_crsp_date.date()}**.")
        except Exception:
            latest_crsp_date = (
                pd.Timestamp(st.session_state.latest_crsp_date).normalize()
                if "latest_crsp_date" in st.session_state
                else None
            )

    st.markdown("#### Pull Any Ticker")
    preset_cols = st.columns(len(QUICK_TEST_TICKERS))
    for column, preset_ticker in zip(preset_cols, QUICK_TEST_TICKERS):
        if column.button(preset_ticker, use_container_width=True):
            st.session_state.api_test_ticker = preset_ticker

    quick_date_cols = st.columns(4)
    if quick_date_cols[0].button("Last 7 days"):
        st.session_state.api_start_date = (default_live_api_end() - pd.Timedelta(days=7)).date()
        st.session_state.api_end_date = default_live_api_end().date()
    if quick_date_cols[1].button("Last 30 days"):
        st.session_state.api_start_date = default_live_api_start(30).date()
        st.session_state.api_end_date = default_live_api_end().date()
    if quick_date_cols[2].button("Paper window"):
        st.session_state.api_start_date = pd.Timestamp(DEFAULT_LOOKUP_START).date()
        st.session_state.api_end_date = pd.Timestamp(DEFAULT_LOOKUP_END).date()
    if quick_date_cols[3].button("End → Today"):
        st.session_state.api_end_date = default_live_api_end().date()

    if "api_test_ticker" not in st.session_state:
        st.session_state.api_test_ticker = "AAPL"
    if "api_start_date" not in st.session_state:
        st.session_state.api_start_date = pd.Timestamp(DEFAULT_LOOKUP_START).date()
    if "api_end_date" not in st.session_state:
        st.session_state.api_end_date = pd.Timestamp(DEFAULT_LOOKUP_END).date()

    ric_hint = ", ".join(ticker_to_ric_candidates(st.session_state.api_test_ticker)[:2])

    with st.form("live_api_query", clear_on_submit=False):
        control_cols = st.columns([1, 1, 1])
        lookup_ticker = (
            control_cols[0]
            .text_input("Ticker or RIC", key="api_test_ticker", max_chars=16)
            .strip()
            .upper()
        )
        start_date = control_cols[1].date_input(
            "Start date",
            key="api_start_date",
            min_value=pd.Timestamp("1990-01-01").date(),
            max_value=default_live_api_end().date(),
        )
        end_date = control_cols[2].date_input(
            "End date",
            key="api_end_date",
            min_value=pd.Timestamp("1990-01-01").date(),
            max_value=default_live_api_end().date(),
        )
        st.caption(
            f"Refinitiv will try RIC candidates such as **{ric_hint}**. "
            "News coverage pulls all headlines in the selected date range and may take longer for the paper window."
        )
        include_news = st.checkbox(
            "Include Refinitiv news coverage (daily counts + drill-down headlines)",
            value=refinitiv_configured(PROJECT_ROOT),
        )
        provider_cols = st.columns(3)
        query_refinitiv = provider_cols[0].checkbox(
            "Query Refinitiv",
            value=refinitiv_configured(PROJECT_ROOT),
        )
        query_wrds = provider_cols[1].checkbox(
            "Query WRDS/CRSP",
            value=wrds_credentials_available(),
        )
        query_yahoo = provider_cols[2].checkbox("Query Yahoo Finance", value=True)
        submitted = st.form_submit_button("Query Selected APIs", type="primary")

    link_cols = st.columns(2)
    link_cols[0].markdown(f"[Open {lookup_ticker} on Google Finance]({google_finance_url(lookup_ticker)})")
    link_cols[1].markdown(
        f"[Open {lookup_ticker} on Yahoo Finance](https://finance.yahoo.com/quote/{lookup_ticker}/history/)"
    )

    if submitted:
        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return

        if not any([query_refinitiv, query_wrds, query_yahoo]):
            st.error("Select at least one provider to query.")
            return

        with st.spinner(f"Querying selected APIs in parallel for {lookup_ticker}..."):
            query_result = run_live_api_query(
                lookup_ticker,
                to_query_date(start_date),
                to_query_date(end_date),
                query_refinitiv=query_refinitiv,
                query_wrds=query_wrds,
                query_yahoo=query_yahoo,
                news_count=1 if include_news else 0,
                latest_crsp_date=latest_crsp_date,
            )
        st.session_state.refinitiv_news_version = st.session_state.get("refinitiv_news_version", 0) + 1
        st.session_state.pop("news_coverage_selected_date", None)
        st.session_state.pop("refinitiv_open_story_id", None)
        st.session_state.pop("refinitiv_open_story_headline", None)
        st.session_state.live_api_query_result = query_result

    if "live_api_query_result" in st.session_state:
        render_live_api_results(st.session_state.live_api_query_result)


# ── RavenPack Sentiment helpers ───────────────────────────────────────────────

def _pg_sql(db_conn, sql: str) -> pd.DataFrame:
    """Execute SQL via raw psycopg2, bypassing SQLAlchemy 2.x incompatibility."""
    cur = db_conn.connection.connection.cursor()
    cur.execute(sql)
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    cur.close()
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def query_ravenpack_articles(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch RavenPack sentiment articles for a ticker from WRDS."""
    return live_data.query_ravenpack_articles(ticker, start_date, end_date)


def make_ravenpack_sentiment_chart(articles: pd.DataFrame, ticker: str):
    """Scatter of sentiment_score over time, coloured by sign.

    Each point carries the original `articles` row index in customdata[0] so that
    a Plotly click event can map directly back to the article table row.
    """
    df = articles.dropna(subset=["sentiment_score"]).copy()
    df["_orig_idx"] = df.index          # preserved so click → table row works
    df["polarity"]  = df["sentiment_score"].apply(
        lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
    )
    df["date_str"] = df["article_time"].dt.strftime("%Y-%m-%d %H:%M UTC")
    fig = px.scatter(
        df,
        x="article_time",
        y="sentiment_score",
        color="polarity",
        custom_data=["_orig_idx"],
        color_discrete_map={"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#aec7e8"},
        hover_data={
            "article_time": False,
            "_orig_idx": False,
            "date_str": True,
            "relevance_score": ":.2f",
            "event_sentiment_score": ":.3f",
            "sentiment_score": ":.3f",
            "topic": True,
        },
        title=f"{ticker} — RavenPack Sentiment Score Over Time",
        labels={"article_time": "Date", "sentiment_score": "Sentiment score", "polarity": ""},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_traces(marker={"size": 6, "opacity": 0.75})
    fig.update_layout(height=420, hovermode="closest")
    return fig


def render_ravenpack_article_features(row: pd.Series) -> None:
    """Render the full RavenPack feature panel for a selected article."""
    st.markdown(f"##### {row.get('headline', '(no headline)')}")

    event_text = row.get("event_text")
    if event_text and str(event_text) not in {"None", "nan", ""}:
        st.info(str(event_text))

    m1, m2, m3 = st.columns(3)
    m1.metric("Relevance score",        f"{row.get('relevance_score', float('nan')):.2f}")
    m2.metric("Event sentiment score",  f"{row.get('event_sentiment_score', float('nan')):.3f}")
    m3.metric("Sentiment score (Eq. 8)", f"{row.get('sentiment_score', float('nan')):.3f}")

    st.markdown("**Categorisation**")
    cat_cols = st.columns(4)
    for col, field in zip(cat_cols, ["topic", "group", "type", "sub_type"]):
        val = row.get(field)
        col.markdown(f"**{field}**  \n{val if val and str(val) not in {'None','nan'} else '—'}")

    detail_cols = st.columns(3)
    for col, field in zip(detail_cols, ["news_type", "source_name", "rp_story_id"]):
        val = row.get(field)
        col.markdown(f"**{field}**  \n{val if val and str(val) not in {'None','nan'} else '—'}")

    ts = row.get("article_time")
    if ts is not None:
        st.caption(f"Published: {pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M UTC')}")

    with st.expander("Advanced scores (css, nip)"):
        adv = {k: row.get(k) for k in ["css", "nip"]}
        st.json({k: (None if str(v) in {"None", "nan"} else float(v)) for k, v in adv.items()})


def render_ravenpack_sentiment_tab() -> None:
    """Render the RavenPack Sentiment browser tab."""
    st.subheader("RavenPack Sentiment")
    st.caption(
        "Fetch RavenPack Dow Jones articles from WRDS for any ticker and date range. "
        "Select an article row to inspect all sentiment features."
    )

    if not wrds_credentials_available():
        st.warning("WRDS credentials are not configured. Add `WRDS_USERNAME` and `WRDS_PASSWORD` to `.env`.")
        return

    with st.form("rp_sentiment_form"):
        form_cols = st.columns([1, 1, 1])
        rp_ticker = form_cols[0].text_input("Ticker", value="AAPL", max_chars=12).strip().upper()
        rp_start  = form_cols[1].date_input(
            "Start date",
            value=pd.Timestamp("2007-01-01").date(),
            min_value=pd.Timestamp("2000-01-01").date(),
            max_value=pd.Timestamp("2026-12-31").date(),
        )
        rp_end    = form_cols[2].date_input(
            "End date",
            value=pd.Timestamp("2007-03-31").date(),
            min_value=pd.Timestamp("2000-01-01").date(),
            max_value=pd.Timestamp("2026-12-31").date(),
        )
        rp_submitted = st.form_submit_button("Fetch RavenPack data", type="primary")

    if rp_submitted:
        if rp_start > rp_end:
            st.error("Start date must be before end date.")
            return
        st.session_state.rp_articles      = None
        st.session_state.rp_selected_row  = None
        st.session_state.rp_query_start   = rp_start.strftime("%Y-%m-%d")
        st.session_state.rp_query_end     = rp_end.strftime("%Y-%m-%d")
        with st.spinner(f"Querying RavenPack on WRDS for {rp_ticker}…"):
            try:
                st.session_state.rp_articles = query_ravenpack_articles(
                    rp_ticker,
                    rp_start.strftime("%Y-%m-%d"),
                    rp_end.strftime("%Y-%m-%d"),
                )
                st.session_state.rp_ticker = rp_ticker
            except Exception as exc:
                st.error(str(exc))
                return

    articles: pd.DataFrame | None = st.session_state.get("rp_articles")
    if articles is None:
        st.info("Enter a ticker and date range, then click **Fetch RavenPack data**.")
        return

    rp_ticker     = st.session_state.get("rp_ticker", "")
    query_start   = st.session_state.get("rp_query_start", "")
    query_end     = st.session_state.get("rp_query_end", "")

    if articles.empty:
        st.warning(f"No RavenPack articles found for {rp_ticker} in the selected range.")
        return

    # ── SQL inspector ─────────────────────────────────────────────────────────
    with st.expander("Show SQL query"):
        st.caption(
            "Entity ID is first resolved with: "
            f"`SELECT DISTINCT rp_entity_id FROM ravenpack_common.wrds_rpa_company_mappings "
            f"WHERE ticker = '{rp_ticker}'`"
        )
        sql_parts: list[str] = []
        if query_start and query_end:
            for yr in range(int(query_start[:4]), int(query_end[:4]) + 1):
                yr_s = max(query_start, f"{yr}-01-01")
                yr_e = min(query_end,   f"{yr}-12-31")
                sql_parts.append(
                    f"-- Year {yr}\n"
                    f"SELECT timestamp_utc, rp_story_id, relevance, event_sentiment_score,\n"
                    f"       headline, event_text, source_name, topic, \"group\", \"type\",\n"
                    f"       sub_type, news_type, css, nip\n"
                    f"FROM ravenpack_dj.rpa_djpr_equities_{yr}\n"
                    f"WHERE rp_entity_id = '{{rp_entity_id}}'  -- from mapping table\n"
                    f"  AND rpa_date_utc BETWEEN '{yr_s}' AND '{yr_e}'\n"
                    f"ORDER BY timestamp_utc;"
                )
        st.code("\n\n".join(sql_parts) if sql_parts else "Run a query first.", language="sql")

    # ── Summary metrics ───────────────────────────────────────────────────────
    with_score = articles["sentiment_score"].notna().sum()
    sm_cols = st.columns(4)
    sm_cols[0].metric("Total articles",       f"{len(articles):,}")
    sm_cols[1].metric("With sentiment score", f"{with_score:,}")
    sm_cols[2].metric("Avg sentiment score",
                      f"{articles['sentiment_score'].mean():.3f}" if with_score else "—")
    sm_cols[3].metric("Avg relevance",
                      f"{articles['relevance_score'].mean():.2f}" if articles["relevance_score"].notna().any() else "—")

    # ── Sentiment timeline — clicking a dot selects that article ──────────────
    prev_scatter_idx: int | None = st.session_state.get("rp_prev_scatter_idx")
    scatter_just_changed = False

    if with_score:
        scatter_sel = st.plotly_chart(
            make_ravenpack_sentiment_chart(articles, rp_ticker),
            use_container_width=True,
            on_select="rerun",
            selection_mode="points",
            key="rp_scatter_chart",
        )
        pts = getattr(scatter_sel.selection, "points", []) if hasattr(scatter_sel, "selection") else []
        scatter_idx: int | None = None
        if pts:
            raw_cd = pts[0].get("customdata")
            if raw_cd:
                scatter_idx = int(raw_cd[0])

        # Only treat scatter as "just changed" when the selected article is new —
        # this prevents the table's persistent selection from overriding it.
        scatter_just_changed = (scatter_idx is not None and scatter_idx != prev_scatter_idx)
        st.session_state.rp_prev_scatter_idx = scatter_idx
        if scatter_just_changed:
            st.session_state.rp_selected_row = scatter_idx

    # ── Article table (with ▶ marker showing the active row) ─────────────────
    st.markdown("#### Articles — click a row or a scatter dot to inspect features")
    selected_row_idx: int | None = st.session_state.get("rp_selected_row")

    table_df = articles[[
        "article_time", "headline", "relevance_score",
        "event_sentiment_score", "sentiment_score", "topic", "news_type",
    ]].copy()
    table_df.insert(0, "#", range(1, len(table_df) + 1))
    table_df["article_time"] = table_df["article_time"].dt.strftime("%Y-%m-%d %H:%M")
    # Visual marker so the user can see which row the scatter selected
    table_df.insert(0, " ", [
        "▶" if i == selected_row_idx else "" for i in range(len(table_df))
    ])

    selection = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        height=320,
        on_select="rerun",
        selection_mode="single-row",
        key="rp_article_table",
    )

    # Table click wins only when the scatter didn't just fire
    if not scatter_just_changed and hasattr(selection, "selection") and selection.selection.rows:
        st.session_state.rp_selected_row = int(selection.selection.rows[0])
        selected_row_idx = st.session_state.rp_selected_row

    if selected_row_idx is not None and 0 <= selected_row_idx < len(articles):
        st.markdown("---")
        st.markdown("#### Article Feature Detail")
        render_ravenpack_article_features(articles.iloc[selected_row_idx])
    else:
        st.caption("Click a scatter dot or a table row to see the full feature breakdown below.")


def make_ravenpack_aggregate_sentiment_chart(articles: pd.DataFrame, ticker: str, *, freq: str = "D"):
    """Build an aggregated RavenPack sentiment timeline."""
    if articles.empty or "sentiment_score" not in articles.columns:
        raise ValueError("No RavenPack sentiment rows available to plot.")

    plot_data = articles.dropna(subset=["sentiment_score"]).copy()
    if plot_data.empty:
        raise ValueError("No non-null RavenPack sentiment scores available to plot.")

    plot_data["article_time"] = pd.to_datetime(plot_data["article_time"], utc=True)
    plot_data["period"] = plot_data["article_time"].dt.tz_convert(None).dt.to_period(freq).dt.start_time
    agg = (
        plot_data.groupby("period", as_index=False)
        .agg(
            avg_sentiment=("sentiment_score", "mean"),
            article_count=("sentiment_score", "size"),
            avg_relevance=("relevance_score", "mean"),
        )
        .sort_values("period")
    )

    label = "Daily" if freq == "D" else "Weekly"
    fig = px.line(
        agg,
        x="period",
        y="avg_sentiment",
        title=f"{ticker.upper()} — {label} Average RavenPack Sentiment",
        labels={"period": "Date", "avg_sentiment": "Average sentiment score"},
        hover_data={
            "period": "|%Y-%m-%d",
            "avg_sentiment": ":.4f",
            "article_count": ":,",
            "avg_relevance": ":.2f",
        },
    )
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)
    fig.update_traces(mode="lines+markers", marker={"size": 5})
    fig.update_layout(height=420, hovermode="closest")
    return fig


def render_dashboard_price_pane(live_result: dict[str, object], *, key_prefix: str) -> None:
    """Render dashboard price charts and provider panels."""
    ticker = str(live_result["ticker"])
    providers = live_result["providers"]
    price_frames = live_result["price_frames"]
    selected = live_result.get("selected_providers", {})

    if len(price_frames) >= 2:
        st.plotly_chart(
            make_combined_price_chart(price_frames, ticker),
            use_container_width=True,
            key=f"{key_prefix}_combined_price_chart",
        )
        comparison = build_cross_provider_comparison(price_frames)
        if not comparison.empty:
            st.markdown("##### Cross-provider close-price comparison")
            st.dataframe(comparison, use_container_width=True, height=280)
    elif len(price_frames) == 1:
        provider, prices = next(iter(price_frames.items()))
        st.plotly_chart(
            make_provider_price_chart(prices, ticker, provider),
            use_container_width=True,
            key=f"{key_prefix}_{provider}_single_price_chart",
        )
    else:
        st.warning("No selected price provider returned data for this ticker/date range.")

    visible_panels = [
        (name, providers[key])
        for name, key in [("Refinitiv", "refinitiv"), ("WRDS/CRSP", "wrds"), ("Yahoo Finance", "yahoo")]
        if selected.get(key, True)
    ]
    if visible_panels:
        st.markdown("##### Provider panes")
        panel_cols = st.columns(len(visible_panels))
        for column, (name, result) in zip(panel_cols, visible_panels):
            render_provider_price_column(column, name, result, ticker, key_prefix=key_prefix)


def render_dashboard_news_pane(live_result: dict[str, object]) -> None:
    """Render Refinitiv news coverage in the dashboard."""
    ticker = str(live_result["ticker"])
    refinitiv = live_result["providers"]["refinitiv"]
    session_info = refinitiv.get("session_info")
    news_df = refinitiv.get("news")
    daily_counts = refinitiv.get("news_daily_counts")
    news_summary = refinitiv.get("news_summary")
    story_id_key = "dashboard_news_story_id"
    story_headline_key = "dashboard_news_story_headline"

    if isinstance(news_df, pd.DataFrame) and not news_df.empty:
        if isinstance(daily_counts, pd.DataFrame) and not daily_counts.empty:
            render_refinitiv_news_coverage_section(
                news_df,
                daily_counts,
                news_summary if isinstance(news_summary, dict) else None,
                ticker,
                story_id_key=story_id_key,
                story_headline_key=story_headline_key,
                embed_story=False,
            )

        st.markdown("#### Full Refinitiv Headline List")
        st.caption(
            "Select any headline row below to load its full Refinitiv story text. "
            "The daily drill-down above is useful for coverage checks; this table exposes the complete returned list."
        )
        display = news_df.copy().sort_values("date", ascending=False).reset_index(drop=True)
        display.insert(0, "#", range(1, len(display) + 1))
        if "date" in display.columns:
            display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d %H:%M")
        show_cols = ["#"] + [col for col in ["date", "headline", "sourceCode", "storyId"] if col in display.columns]
        selection = st.dataframe(
            display[show_cols],
            use_container_width=True,
            hide_index=True,
            height=420,
            on_select="rerun",
            selection_mode="single-row",
            key="dashboard_all_refinitiv_headlines",
        )
        if selection is not None and hasattr(selection, "selection") and selection.selection.rows:
            row_idx = int(selection.selection.rows[0])
            if 0 <= row_idx < len(display):
                row = display.iloc[row_idx]
                if "storyId" in row and pd.notna(row["storyId"]):
                    new_story_id = str(row["storyId"])
                    new_headline = str(row.get("headline", "Selected headline"))
                    if (
                        st.session_state.get(story_id_key) != new_story_id
                        or st.session_state.get(story_headline_key) != new_headline
                    ):
                        st.session_state[story_id_key] = new_story_id
                        st.session_state[story_headline_key] = new_headline
                        st.session_state.pop("dashboard_news_story_text", None)  # legacy widget key
                        st.session_state.pop("refinitiv_open_story_id", None)
                        st.session_state.pop("refinitiv_open_story_headline", None)

        story_id = st.session_state.get(story_id_key)
        if story_id:
            headline = st.session_state.get(story_headline_key, "Selected headline")
            render_refinitiv_story_body(
                str(story_id),
                str(headline),
                close_key=f"dashboard_close_story_{story_id}",
                story_id_key=story_id_key,
                story_headline_key=story_headline_key,
            )
    else:
        error = refinitiv.get("error")
        if error:
            st.warning(format_refinitiv_news_error(str(error), session_info))
        else:
            st.info("No Refinitiv headline rows were returned.")


def render_dashboard_sentiment_pane(ravenpack_articles: pd.DataFrame, ticker: str, ravenpack_error: str | None) -> None:
    """Render RavenPack sentiment charts and data tables."""
    if ravenpack_error:
        st.warning(ravenpack_error)
        return
    if ravenpack_articles.empty:
        st.info("No RavenPack articles were returned, or RavenPack was not selected.")
        return

    articles = ravenpack_articles.copy()
    with_score = articles["sentiment_score"].notna().sum()
    metric_cols = st.columns(4)
    metric_cols[0].metric("RavenPack articles", f"{len(articles):,}")
    metric_cols[1].metric("With sentiment", f"{with_score:,}")
    metric_cols[2].metric("Avg sentiment", f"{articles['sentiment_score'].mean():.3f}" if with_score else "—")
    metric_cols[3].metric(
        "Avg relevance",
        f"{articles['relevance_score'].mean():.2f}" if articles["relevance_score"].notna().any() else "—",
    )

    if with_score:
        chart_tabs = st.tabs(["Article Scatter", "Daily Average", "Weekly Average"])
        with chart_tabs[0]:
            st.plotly_chart(
                make_ravenpack_sentiment_chart(articles, ticker),
                use_container_width=True,
                key="dashboard_sentiment_article_scatter",
            )
        with chart_tabs[1]:
            st.plotly_chart(
                make_ravenpack_aggregate_sentiment_chart(articles, ticker, freq="D"),
                use_container_width=True,
                key="dashboard_sentiment_daily_average",
            )
        with chart_tabs[2]:
            st.plotly_chart(
                make_ravenpack_aggregate_sentiment_chart(articles, ticker, freq="W"),
                use_container_width=True,
                key="dashboard_sentiment_weekly_average",
            )

    show_cols = [
        "article_time", "headline", "event_text", "relevance_score",
        "event_sentiment_score", "sentiment_score", "topic", "group", "type", "news_type",
    ]
    display = articles[[col for col in show_cols if col in articles.columns]].copy()
    if "article_time" in display.columns:
        display["article_time"] = pd.to_datetime(display["article_time"], utc=True).dt.strftime("%Y-%m-%d %H:%M")
    st.markdown("##### RavenPack article rows")
    st.dataframe(display, use_container_width=True, height=360)


def render_dashboard_raw_data_pane(live_result: dict[str, object], ravenpack_articles: pd.DataFrame) -> None:
    """Render raw provider frames for export/debugging."""
    providers = live_result["providers"]
    for label, key in [("Refinitiv prices", "refinitiv"), ("WRDS/CRSP prices", "wrds"), ("Yahoo prices", "yahoo")]:
        provider_result = providers[key]
        prices = provider_result.get("prices")
        with st.expander(label, expanded=False):
            if isinstance(prices, pd.DataFrame) and not prices.empty:
                st.dataframe(prices.sort_values("date", ascending=False), use_container_width=True)
            else:
                st.caption(str(provider_result.get("error") or provider_result.get("status")))

    refinitiv_news = providers["refinitiv"].get("news")
    with st.expander("Refinitiv news headlines", expanded=False):
        if isinstance(refinitiv_news, pd.DataFrame) and not refinitiv_news.empty:
            st.dataframe(refinitiv_news.sort_values("date", ascending=False), use_container_width=True)
        else:
            st.caption("No Refinitiv news frame available.")

    with st.expander("RavenPack sentiment articles", expanded=False):
        if not ravenpack_articles.empty:
            st.dataframe(ravenpack_articles, use_container_width=True)
        else:
            st.caption("No RavenPack article frame available.")


def _dashboard_cache_dir(ticker: str) -> Path | None:
    """Find the batch-cache directory for a ticker (rank_XXXX_SLUG), if any."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in ticker.upper().strip())
    if not slug or not TOP1K_BY_TICKER_DIR.exists():
        return None
    matches = sorted(TOP1K_BY_TICKER_DIR.glob(f"rank_*_{slug}"))
    for directory in matches:
        manifest = directory / "manifest.json"
        if not manifest.exists():
            continue
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("ticker", "")).strip().upper() == ticker.strip().upper():
            return directory
    return matches[0] if matches else None


def _dashboard_cache_info(ticker: str) -> dict[str, object] | None:
    """Lightweight cache summary for a ticker (no parquet reads)."""
    directory = _dashboard_cache_dir(ticker)
    if directory is None:
        return None
    manifest_path = directory / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
    ok_providers = [
        str(p.get("provider"))
        for p in manifest.get("provider_status", [])
        if isinstance(p, dict) and p.get("status") == "ok"
    ]
    return {
        "dir": directory,
        "manifest": manifest,
        "ok_providers": ok_providers,
        "created_at": manifest.get("created_at"),
        "company_name": manifest.get("company_name"),
        "volume_rank": manifest.get("volume_rank"),
        "start_date": manifest.get("start_date"),
        "end_date": manifest.get("end_date"),
    }


def _filter_cached_frame(
    df: pd.DataFrame, date_col: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Filter a cached frame to [start_date, end_date] on date_col when present."""
    if not isinstance(df, pd.DataFrame) or df.empty or date_col not in df.columns:
        return df
    try:
        dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        return df[(dates >= start) & (dates < end)]
    except Exception:
        return df


def load_cached_dashboard_result(
    ticker: str, start_date: str, end_date: str
) -> dict[str, object] | None:
    """Build a dashboard result dict from cached parquet files (no network/login).

    Returns the same shape as run_live_api_query so the same renderers work.
    Returns None when no cache directory exists for the ticker.
    """
    directory = _dashboard_cache_dir(ticker)
    if directory is None:
        return None

    ticker_clean = ticker.upper().strip()
    start_s = to_query_date(start_date)
    end_s = to_query_date(end_date)

    manifest: dict[str, object] = {}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    status_by_provider: dict[str, str] = {}
    ps_path = directory / "provider_status.parquet"
    if ps_path.exists():
        try:
            ps = pd.read_parquet(ps_path)
            status_by_provider = {
                str(r["provider"]): str(r.get("status", "")) for _, r in ps.iterrows()
            }
        except Exception:
            status_by_provider = {}

    def _read(name: str) -> pd.DataFrame:
        path = directory / name
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()

    refinitiv_prices = _filter_cached_frame(_read("refinitiv_prices.parquet"), "date", start_s, end_s)
    refinitiv_news = _filter_cached_frame(_read("refinitiv_news.parquet"), "date", start_s, end_s)
    refinitiv_daily = _filter_cached_frame(_read("refinitiv_news_daily_counts.parquet"), "date", start_s, end_s)
    wrds_prices = _filter_cached_frame(_read("wrds_prices.parquet"), "date", start_s, end_s)
    wrds_names = _read("wrds_names.parquet")
    yahoo_prices = _filter_cached_frame(_read("yahoo_prices.parquet"), "date", start_s, end_s)
    ravenpack_articles = _filter_cached_frame(_read("ravenpack_articles.parquet"), "timestamp_utc", start_s, end_s)

    def _status(provider: str, frame: pd.DataFrame) -> str:
        if provider not in status_by_provider and frame.empty:
            return "skipped"
        if frame.empty:
            return "empty"
        return "ok"

    providers: dict[str, dict[str, object]] = {
        "refinitiv": {
            "status": _status("refinitiv", refinitiv_prices),
            "error": None if not refinitiv_prices.empty else "No cached Refinitiv price history.",
            "prices": refinitiv_prices,
            "news": refinitiv_news,
            "news_daily_counts": refinitiv_daily,
            "news_summary": None,
            "ric": None,
            "session_info": None,
        },
        "wrds": {
            "status": _status("wrds", wrds_prices),
            "error": None if not wrds_prices.empty else "No cached CRSP rows.",
            "prices": wrds_prices,
            "names": wrds_names,
        },
        "yahoo": {
            "status": _status("yahoo", yahoo_prices),
            "error": None if not yahoo_prices.empty else "No cached Yahoo rows.",
            "prices": yahoo_prices,
        },
        "ravenpack": {
            "status": _status("ravenpack", ravenpack_articles),
            "error": None if not ravenpack_articles.empty else "No cached RavenPack articles.",
            "articles": ravenpack_articles,
        },
    }

    price_frames = {
        provider: result["prices"]
        for provider, result in providers.items()
        if isinstance(result.get("prices"), pd.DataFrame) and not result["prices"].empty
    }

    selected = manifest.get("selected_providers") if isinstance(manifest.get("selected_providers"), dict) else None

    return {
        "ticker": ticker_clean,
        "start_date": start_s,
        "end_date": end_s,
        "providers": providers,
        "price_frames": price_frames,
        "selected_providers": selected or {p: True for p in providers},
        "source": "cache",
        "cache_dir": str(directory),
        "cache_created_at": manifest.get("created_at"),
        "cache_window": (manifest.get("start_date"), manifest.get("end_date")),
    }


def render_multi_api_dashboard_tab() -> None:
    """Render a ticker/date dashboard that combines all available APIs into panes."""
    st.subheader("Unified Ticker Data Explorer")
    st.caption(
        "Enter one ticker and date range, then retrieve prices, Refinitiv news, and RavenPack sentiment "
        "from the configured sources. This replaces the separate Live API Test and RavenPack Sentiment "
        "workflows with one shared input form."
    )

    status_cols = st.columns(4)
    status_cols[0].metric("Refinitiv", refinitiv_status_label(PROJECT_ROOT))
    status_cols[1].metric("WRDS/CRSP", "Ready" if wrds_credentials_available() else "Not configured")
    status_cols[2].metric("Yahoo", "Ready")
    status_cols[3].metric("RavenPack", "Ready" if wrds_credentials_available() else "Not configured")

    if "dashboard_ticker" not in st.session_state:
        st.session_state.dashboard_ticker = "AAPL"
    if "dashboard_start_date" not in st.session_state:
        st.session_state.dashboard_start_date = pd.Timestamp(DEFAULT_LOOKUP_START).date()
    if "dashboard_end_date" not in st.session_state:
        st.session_state.dashboard_end_date = pd.Timestamp(DEFAULT_LOOKUP_END).date()

    preset_cols = st.columns(len(QUICK_TEST_TICKERS))
    for column, preset_ticker in zip(preset_cols, QUICK_TEST_TICKERS):
        if column.button(preset_ticker, key=f"dashboard_ticker_{preset_ticker}", use_container_width=True):
            st.session_state.dashboard_ticker = preset_ticker

    dash_date_cols = st.columns(3)
    if dash_date_cols[0].button("Paper window", key="dash_paper_window"):
        st.session_state.dashboard_start_date = pd.Timestamp(DEFAULT_LOOKUP_START).date()
        st.session_state.dashboard_end_date = pd.Timestamp(DEFAULT_LOOKUP_END).date()
    if dash_date_cols[1].button("End → Today", key="dash_end_today"):
        st.session_state.dashboard_end_date = default_live_api_end().date()
    if dash_date_cols[2].button("Full history (1990 → today)", key="dash_full_history"):
        st.session_state.dashboard_start_date = pd.Timestamp("1990-01-01").date()
        st.session_state.dashboard_end_date = default_live_api_end().date()

    current_ticker_input = str(st.session_state.get("dashboard_ticker", "")).strip().upper()
    cache_info = _dashboard_cache_info(current_ticker_input) if current_ticker_input else None
    if cache_info:
        rank = cache_info.get("volume_rank")
        company = cache_info.get("company_name") or current_ticker_input
        created = str(cache_info.get("created_at") or "")[:10]
        providers_ok = ", ".join(cache_info.get("ok_providers") or []) or "none"
        win = cache_info.get("start_date"), cache_info.get("end_date")
        st.success(
            f"✅ **Cached data available** for {current_ticker_input} ({company}"
            + (f", rank {rank}" if rank else "")
            + f"). Pulled {created or 'n/a'}; window {win[0]}→{win[1]}; providers: {providers_ok}. "
            "**Load cached data** is instant and needs no API login."
        )
    elif current_ticker_input:
        st.info(
            f"ℹ️ No cached data for {current_ticker_input}. **Load data** will pull live "
            "from the online APIs (requires WRDS / Refinitiv access)."
        )

    with st.form("multi_api_dashboard_query", clear_on_submit=False):
        control_cols = st.columns([1, 1, 1])
        ticker = control_cols[0].text_input("Ticker", key="dashboard_ticker", max_chars=16).strip().upper()
        start_date = control_cols[1].date_input(
            "Start date",
            key="dashboard_start_date",
            min_value=pd.Timestamp("1990-01-01").date(),
            max_value=default_live_api_end().date(),
        )
        end_date = control_cols[2].date_input(
            "End date",
            key="dashboard_end_date",
            min_value=pd.Timestamp("1990-01-01").date(),
            max_value=default_live_api_end().date(),
        )

        provider_cols = st.columns(4)
        use_refinitiv = provider_cols[0].checkbox(
            "Refinitiv prices/news",
            value=refinitiv_configured(PROJECT_ROOT),
            key="dashboard_use_refinitiv",
        )
        use_wrds = provider_cols[1].checkbox(
            "WRDS/CRSP prices",
            value=wrds_credentials_available(),
            key="dashboard_use_wrds",
        )
        use_yahoo = provider_cols[2].checkbox("Yahoo prices", value=True, key="dashboard_use_yahoo")
        use_ravenpack = provider_cols[3].checkbox(
            "RavenPack sentiment",
            value=wrds_credentials_available(),
            key="dashboard_use_ravenpack",
        )
        include_refinitiv_news = st.checkbox(
            "Include Refinitiv news coverage and headline rows",
            value=refinitiv_configured(PROJECT_ROOT),
            key="dashboard_include_refinitiv_news",
        )
        st.caption(
            "**Load data** uses the local cache when available (instant, no login) and "
            "only pulls live when nothing is cached. **Re-pull live** ignores the cache "
            "and re-fetches from the online APIs. Provider checkboxes apply to live pulls."
        )
        button_cols = st.columns(2)
        load_clicked = button_cols[0].form_submit_button("Load data", type="primary")
        relive_clicked = button_cols[1].form_submit_button("Re-pull live (ignore cache)")

    submitted = load_clicked or relive_clicked
    if submitted:
        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return

        # Prefer cache unless the user explicitly asked to re-pull live.
        cached_result: dict[str, object] | None = None
        if not relive_clicked:
            cached_result = load_cached_dashboard_result(
                ticker, to_query_date(start_date), to_query_date(end_date)
            )

        if cached_result is not None:
            st.session_state.dashboard_result = {"live": cached_result, "source": "cache"}
        else:
            if not any([use_refinitiv, use_wrds, use_yahoo, use_ravenpack]):
                st.error("Select at least one data source for a live pull.")
                return
            if not relive_clicked:
                st.info(
                    f"No cached data for {ticker} — pulling live from the selected providers."
                )

            latest_crsp_date: pd.Timestamp | None = None
            if use_wrds and wrds_credentials_available():
                try:
                    latest_crsp_date = get_latest_crsp_date()
                except Exception:
                    latest_crsp_date = None

            with st.spinner(f"Retrieving dashboard data for {ticker}..."):
                # 10 000 rows ≈ 40 years of daily data — enough for any date window.
                live_result = run_live_api_query(
                    ticker,
                    to_query_date(start_date),
                    to_query_date(end_date),
                    query_refinitiv=use_refinitiv,
                    query_wrds=use_wrds,
                    query_yahoo=use_yahoo,
                    query_ravenpack=use_ravenpack,
                    news_count=1 if include_refinitiv_news else 0,
                    latest_crsp_date=latest_crsp_date,
                    wrds_limit=10_000,
                )

            st.session_state.dashboard_result = {"live": live_result, "source": "live"}

    dashboard_result = st.session_state.get("dashboard_result")
    if not dashboard_result:
        st.info("Choose data sources above and click **Retrieve Dashboard Data**.")
        return

    live_result = dashboard_result["live"]
    source = dashboard_result.get("source", "live")
    ravenpack_result = live_result["providers"].get("ravenpack", {})
    ravenpack_articles = ravenpack_result.get("articles", pd.DataFrame())
    if not isinstance(ravenpack_articles, pd.DataFrame):
        ravenpack_articles = pd.DataFrame()
    ravenpack_error = ravenpack_result.get("error")
    ticker = str(live_result["ticker"])

    providers = live_result["providers"]
    result_cols = st.columns(4)
    result_cols[0].metric("Refinitiv", _provider_status_label(providers["refinitiv"]))
    result_cols[1].metric("WRDS/CRSP", _provider_status_label(providers["wrds"]))
    result_cols[2].metric("Yahoo", _provider_status_label(providers["yahoo"]))
    result_cols[3].metric(
        "RavenPack",
        "OK" if not ravenpack_articles.empty else ("Failed" if ravenpack_error else "No rows"),
    )
    if source == "cache":
        created = str(live_result.get("cache_created_at") or "")[:10]
        st.caption(
            f"📦 Source: **local cache** (no live API call){' · pulled ' + created if created else ''}. "
            f"Window **{live_result['start_date']}** to **{live_result['end_date']}** for **{ticker}**. "
            "Use **Re-pull live** above to refresh from the online APIs."
        )
    else:
        st.caption(
            f"🌐 Source: **live API pull**. Window **{live_result['start_date']}** to "
            f"**{live_result['end_date']}** for **{ticker}**"
        )

    pane_overview, pane_prices, pane_news, pane_sentiment, pane_raw = st.tabs([
        "Overview",
        "Prices",
        "News",
        "Sentiment",
        "Raw Data",
    ])

    with pane_overview:
        # Debug: show provider row counts so we can diagnose empty charts.
        with st.expander("🔍 Provider debug", expanded=not bool(live_result["price_frames"])):
            for pname, pres in live_result["providers"].items():
                prices = pres.get("prices")
                n = len(prices) if isinstance(prices, pd.DataFrame) else "—"
                err = pres.get("error") or ""
                reason = pres.get("fail_reason") or ""
                st.caption(
                    f"**{pname}** status={pres.get('status')} rows={n} "
                    f"reason={reason or '—'} err={str(err)[:80]}"
                )
        if live_result["price_frames"]:
            render_dashboard_price_pane(live_result, key_prefix="dashboard_overview")
        else:
            st.warning("No provider returned price data for this ticker/date range. Check the debug expander above.")
        if not ravenpack_articles.empty:
            st.markdown("#### Sentiment snapshot")
            st.plotly_chart(
                make_ravenpack_aggregate_sentiment_chart(ravenpack_articles, ticker, freq="D"),
                use_container_width=True,
                key="dashboard_overview_sentiment_snapshot",
            )

    with pane_prices:
        render_dashboard_price_pane(live_result, key_prefix="dashboard_prices")

    with pane_news:
        render_dashboard_news_pane(live_result)

    with pane_sentiment:
        render_dashboard_sentiment_pane(ravenpack_articles, ticker, str(ravenpack_error) if ravenpack_error else None)

    with pane_raw:
        render_dashboard_raw_data_pane(live_result, ravenpack_articles)


# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sentiment LTR Paper: Data Explorer",
    layout="wide",
)

st.title("Sentiment LTR Data Explorer")
st.caption(
    "Unified ticker/date queries for Refinitiv, WRDS/CRSP, Yahoo, and RavenPack, plus paper-replication validation charts."
)

# ── Batch Pipeline helpers ────────────────────────────────────────────────────


def _batch_pid_running() -> int | None:
    """Return the PID from batch.pid if the process is still alive, else None."""
    if not BATCH_PID_FILE.exists():
        return None
    try:
        pid = int(BATCH_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)  # signal 0 = existence check only
        return pid
    except (ValueError, OSError):
        return None


def _read_batch_status() -> dict:
    if not BATCH_STATUS_FILE.exists():
        return {}
    try:
        return json.loads(BATCH_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_batch_progress() -> pd.DataFrame | None:
    if not BATCH_PROGRESS_CSV.exists():
        return None
    try:
        return pd.read_csv(BATCH_PROGRESS_CSV)
    except Exception:
        return None


def _manifest_cache_token() -> str:
    """Invalidate manifest cache when tickers are added/updated on disk.

    Intentionally excludes batch_status.json — that file updates every second
    during a run and would bust the cache on every auto-refresh.
    """
    if not TOP1K_BY_TICKER_DIR.exists():
        return "0"
    latest = 0.0
    count = 0
    for p in TOP1K_BY_TICKER_DIR.glob("rank_*/manifest.json"):
        count += 1
        latest = max(latest, p.stat().st_mtime)
    return f"{count}:{latest:.0f}"


@st.cache_data(ttl=15, show_spinner=False)
def _load_all_manifests_cached(cache_token: str) -> pd.DataFrame:
    """Cached wrapper — reloads when manifests change (see cache_token)."""
    return _load_all_manifests()


def _get_manifests_df() -> pd.DataFrame:
    """Return manifests DataFrame, using session cache to avoid reload spinners."""
    token = _manifest_cache_token()
    if (
        "_manifests_df" in st.session_state
        and st.session_state.get("_manifests_token") == token
    ):
        return st.session_state["_manifests_df"]
    df = _load_all_manifests_cached(token)
    st.session_state["_manifests_df"] = df
    st.session_state["_manifests_token"] = token
    return df


def _load_all_manifests() -> pd.DataFrame:
    """Walk by_ticker/ dirs and collect manifest data into a DataFrame."""
    rows = []
    if not TOP1K_BY_TICKER_DIR.exists():
        return pd.DataFrame()
    for mfile in sorted(TOP1K_BY_TICKER_DIR.glob("rank_*/manifest.json")):
        try:
            m = json.loads(mfile.read_text(encoding="utf-8"))
            provider_status = list(m.get("provider_status", []))
            needs_enrich = any(
                not ps.get("fail_reason") and str(ps.get("status", "")) not in ("ok", "skipped")
                for ps in provider_status
            )
            if needs_enrich:
                # Fast backfill from error text only — skip reading wrds parquets
                # (reading 1000 parquets on every page load was causing the UI to hang).
                provider_status = enrich_provider_status_records(
                    provider_status,
                    ticker=str(m.get("ticker", "")),
                    permno=m.get("permno"),
                    query_start=m.get("start_date"),
                    query_end=m.get("end_date"),
                    wrds_last_trade_date=None,
                    skip_permno_lookup=True,
                )
            provider_map: dict[str, str] = {}
            provider_rows_map: dict[str, int] = {}
            provider_reason_map: dict[str, str] = {}
            for ps in provider_status:
                pname = ps.get("provider", "")
                provider_map[pname] = ps.get("status", "")
                provider_rows_map[pname] = int(ps.get("rows") or 0)
                provider_reason_map[pname] = ps.get("fail_reason") or ""
            rows.append({
                "rank":       m.get("volume_rank"),
                "permno":     m.get("permno"),
                "ticker":     m.get("ticker"),
                "company":    m.get("company_name", ""),
                "status":     m.get("status"),
                "ok":         m.get("ok_provider_count", 0),
                "fail":       m.get("failed_provider_count", 0),
                "wrds_status":       provider_map.get("wrds", ""),
                "yahoo_status":      provider_map.get("yahoo", ""),
                "ravenpack_status":  provider_map.get("ravenpack", ""),
                "refinitiv_status":  provider_map.get("refinitiv", ""),
                "wrds_rows":         provider_rows_map.get("wrds", 0),
                "yahoo_rows":        provider_rows_map.get("yahoo", 0),
                "ravenpack_rows":    provider_rows_map.get("ravenpack", 0),
                "refinitiv_rows":    provider_rows_map.get("refinitiv", 0),
                "yahoo_fail_reason":      provider_reason_map.get("yahoo", ""),
                "ravenpack_fail_reason":  provider_reason_map.get("ravenpack", ""),
                "refinitiv_fail_reason":  provider_reason_map.get("refinitiv", ""),
                "wrds_fail_reason":       provider_reason_map.get("wrds", ""),
                "created_at": m.get("created_at", ""),
            })
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("rank").reset_index(drop=True)


def _launch_batch(
    start: str, end: str,
    start_rank: int, max_tickers: int | None,
    force_rerun: bool, rerun_failed: bool, rerun_partial: bool, sleep_sec: float,
    stop_after: int, provider_timeout: float, year_timeout: int,
    use_wrds: bool, use_yahoo: bool, use_ravenpack: bool, use_refinitiv: bool,
    combined_parquets: bool,
) -> subprocess.Popen:
    cmd = [sys.executable, str(BATCH_RUNNER_SCRIPT),
           "--start", start, "--end", end,
           "--start-rank", str(start_rank),
           "--sleep", str(sleep_sec),
           "--stop-after-failures", str(stop_after),
           "--provider-timeout", str(provider_timeout),
           "--year-timeout", str(year_timeout)]
    if max_tickers is not None:
        cmd += ["--max-tickers", str(max_tickers)]
    if force_rerun:
        cmd.append("--force-rerun")
    if rerun_failed:
        cmd.append("--rerun-failed")
    if rerun_partial:
        cmd.append("--rerun-partial")
    if not use_wrds:
        cmd.append("--no-wrds")
    if not use_yahoo:
        cmd.append("--no-yahoo")
    if not use_ravenpack:
        cmd.append("--no-ravenpack")
    if not use_refinitiv:
        cmd.append("--no-refinitiv")
    if not combined_parquets:
        cmd.append("--no-combined-parquets")
    log_path = TOP1K_OUTPUT_DIR / "batch_runner.log"
    TOP1K_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "a", encoding="utf-8")  # noqa: SIM115
    log_file.write(f"\n\n=== Run started at {datetime.now(timezone.utc).isoformat()} ===\n")
    log_file.flush()
    return subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)


def _status_color(status: str) -> str:
    return {
        "complete":       "🟢",
        "partial":        "🟡",
        "failed":         "🔴",
        "error":          "🔴",
        "skipped_cached": "⚪",
    }.get(status, "⬜")


def _provider_status_cell(status: str, rows: object, fail_reason: str = "") -> str:
    """One-line provider status for the per-ticker table: icon, rows, reason code."""
    status = str(status or "").strip()
    if not status or status in ("…", "—"):
        return "…"
    try:
        row_n = int(rows or 0)
    except (TypeError, ValueError):
        row_n = 0
    if status == "ok":
        return f"✅ {row_n:,}" if row_n else "✅"
    icon = {"failed": "❌", "empty": "⚠️", "timeout": "⏱", "skipped": "—", "unavailable": "⛔"}.get(status, "•")
    if fail_reason:
        return f"{icon} {fail_reason}"
    return f"{icon} {status}"


def _load_top1k_universe() -> pd.DataFrame:
    """Load the committed top-1k universe (permno, ticker, rank, company)."""
    if not TOP1K_UNIVERSE_PATH.exists():
        return pd.DataFrame()
    u = pd.read_csv(TOP1K_UNIVERSE_PATH)
    u["volume_rank"] = u["volume_rank"].astype(int)
    return u.sort_values("volume_rank").reset_index(drop=True)


def _build_delisting_lookup() -> dict[int, dict]:
    """Map permno → cached CRSP delisting info (empty when nothing cached yet)."""
    try:
        cache = load_delisting_cache()
    except Exception:
        return {}
    if cache.empty or "permno" not in cache.columns:
        return {}
    lookup: dict[int, dict] = {}
    for rec in cache.to_dict(orient="records"):
        try:
            permno = int(rec["permno"])
        except (TypeError, ValueError, KeyError):
            continue
        lookup[permno] = rec
    return lookup


def _delisting_cell(permno: object, lookup: dict[int, dict]) -> str:
    """One-line CRSP delisting summary for the per-ticker table."""
    try:
        rec = lookup[int(permno)]
    except (TypeError, ValueError, KeyError):
        return "…"  # PERMNO not looked up yet
    delisted = bool(rec.get("delisted"))
    code = rec.get("dlstcd")
    label = rec.get("delisting_label") or ""
    if not delisted:
        return "🟢 active"
    dlret = rec.get("dlret")
    try:
        ret_str = f"  ({float(dlret):+.1%})" if pd.notna(dlret) else ""
    except (TypeError, ValueError):
        ret_str = ""
    try:
        code_str = f"{int(code)} " if pd.notna(code) else ""
    except (TypeError, ValueError):
        code_str = ""
    return f"⛔ {code_str}{label}{ret_str}"


_EXIT_SOURCE_ICONS: dict[str, str] = {
    "crsp_dlret": "✅",
    "sdc_pricepersh": "🟡",
    "crsp_dlprc_fallback": "⬜",
}


def _build_cash_merger_lookup() -> dict[int, dict]:
    """Map permno → cached cash-merger exit info (empty when nothing cached yet)."""
    try:
        cache = load_cash_merger_cache()
    except Exception:
        return {}
    if cache.empty or "permno" not in cache.columns:
        return {}
    lookup: dict[int, dict] = {}
    for rec in cache.to_dict(orient="records"):
        try:
            permno = int(rec["permno"])
        except (TypeError, ValueError, KeyError):
            continue
        lookup[permno] = rec
    return lookup


def _exit_cell(permno: object, lookup: dict[int, dict]) -> str:
    """Cash-merger exit summary for the per-ticker table: icon + return + price."""
    try:
        rec = lookup[int(permno)]
    except (TypeError, ValueError, KeyError):
        return "—"  # no cash-merger record (not a merger / not checked)
    source = rec.get("exit_source")
    icon = _EXIT_SOURCE_ICONS.get(str(source))
    if not icon:
        return "—"
    try:
        ret = float(rec.get("exit_return"))
        ret_str = f"{ret:+.1%}"
    except (TypeError, ValueError):
        ret_str = "n/a"
    try:
        price = abs(float(rec.get("dlprc")))
        price_str = f" @ ${price:,.2f}"
    except (TypeError, ValueError):
        price_str = ""
    return f"{icon} {ret_str}{price_str}"


def _render_cache_snapshot(manifests_df: pd.DataFrame, universe_size: int = 1_000) -> None:
    """At-a-glance view of everything cached on disk across the 1k universe."""
    st.markdown("### 📦 Cached data snapshot")

    if manifests_df.empty:
        st.info(f"No tickers cached yet — 0 / {universe_size:,} in the universe.")
        return

    n_cached = len(manifests_df)
    n_complete = int((manifests_df["status"] == "complete").sum())
    n_partial = int((manifests_df["status"] == "partial").sum())
    n_failed = int(manifests_df["status"].isin(["failed", "error"]).sum())
    n_never = max(0, universe_size - n_cached)

    # ── Universe progress ─────────────────────────────────────────────────────
    top_cols = st.columns(6)
    top_cols[0].metric("Cached", f"{n_cached:,} / {universe_size:,}")
    top_cols[1].metric("🟢 Complete", f"{n_complete:,}")
    top_cols[2].metric("🟡 Partial", f"{n_partial:,}")
    top_cols[3].metric("🔴 Failed", f"{n_failed:,}")
    top_cols[4].metric("⬜ Not cached", f"{n_never:,}")
    pct_complete = n_complete / universe_size if universe_size else 0
    top_cols[5].metric("Fully done", f"{pct_complete * 100:.1f}%")

    st.progress(
        n_cached / universe_size,
        text=f"{n_cached:,} tickers have at least one cached manifest  ·  "
             f"{n_complete:,} fully complete ({pct_complete * 100:.1f}% of universe)",
    )

    # ── Per-provider coverage matrix ─────────────────────────────────────────
    st.markdown("**Provider coverage** — across all cached tickers")
    provider_names = ["wrds", "yahoo", "ravenpack", "refinitiv"]
    coverage_rows: list[dict] = []
    for pname in provider_names:
        status_col = f"{pname}_status"
        statuses = manifests_df[status_col] if status_col in manifests_df.columns else pd.Series(dtype=str)
        n_ok = int((statuses == "ok").sum())
        n_fail = int(statuses.isin(["failed", "timeout"]).sum())
        n_empty = int((statuses == "empty").sum())
        n_other = int(n_cached - n_ok - n_fail - n_empty)
        coverage_rows.append({
            "Provider": pname.upper(),
            "✅ ok": n_ok,
            "❌ failed": n_fail,
            "⚠️ empty": n_empty,
            "other": n_other,
            "% ok": f"{100 * n_ok / n_cached:.1f}%" if n_cached else "—",
        })

    cov_df = pd.DataFrame(coverage_rows)
    st.dataframe(cov_df, use_container_width=True, hide_index=True)

    bar_cols = st.columns(4)
    for col, row in zip(bar_cols, coverage_rows):
        ok_n = int(row["✅ ok"])
        with col:
            st.caption(row["Provider"])
            st.progress(ok_n / n_cached if n_cached else 0.0, text=f"{ok_n:,} ok")


_BATCH_PROVIDER_NAMES = ("wrds", "yahoo", "ravenpack", "refinitiv")
_OK_PROVIDER_STATUSES = frozenset({"ok", "skipped", "…", ""})


def _fail_reason_counts_for_provider(manifests_df: pd.DataFrame, pname: str) -> pd.DataFrame:
    """Group non-ok tickers by fail_reason for one provider."""
    reason_col = f"{pname}_fail_reason"
    status_col = f"{pname}_status"
    empty = pd.DataFrame(columns=["Reason", "Label", "Count"])
    if manifests_df.empty or status_col not in manifests_df.columns:
        return empty

    statuses = manifests_df[status_col].fillna("").astype(str)
    not_ok = manifests_df[~statuses.isin(_OK_PROVIDER_STATUSES)]
    if not_ok.empty:
        return empty

    if reason_col in not_ok.columns:
        reasons = not_ok[reason_col].fillna("").astype(str).replace("", "(no reason recorded)")
    else:
        reasons = pd.Series("(no reason recorded)", index=not_ok.index)

    counts = (
        reasons.value_counts()
        .rename_axis("Reason")
        .reset_index(name="Count")
    )
    counts["Label"] = counts["Reason"].apply(
        lambda code: reason_label(code) if code != "(no reason recorded)" else code
    )
    return counts.sort_values("Count", ascending=False).reset_index(drop=True)


def _render_fail_reasons_by_provider(manifests_df: pd.DataFrame) -> None:
    """Per-API tables and charts of failure reason counts across cached tickers."""
    st.markdown("### Failure reasons by provider")
    st.caption(
        "Counts of non-ok tickers among cached manifests, grouped by machine-readable "
        "`fail_reason` code. Hover a bar for the human-readable label."
    )

    if manifests_df.empty:
        st.info("No cached tickers yet — failure breakdown will appear after the first batch run.")
        return

    tab_labels = [p.upper() for p in _BATCH_PROVIDER_NAMES]
    tabs = st.tabs(tab_labels)
    for tab, pname in zip(tabs, _BATCH_PROVIDER_NAMES):
        with tab:
            counts_df = _fail_reason_counts_for_provider(manifests_df, pname)
            if counts_df.empty:
                st.success("No failures recorded for this provider.")
                continue

            total_failures = int(counts_df["Count"].sum())
            st.metric("Non-ok tickers", f"{total_failures:,}")

            display_df = counts_df[["Reason", "Label", "Count"]]
            st.dataframe(display_df, use_container_width=True, hide_index=True)

            chart_df = counts_df.sort_values("Count", ascending=True)
            fig = px.bar(
                chart_df,
                x="Count",
                y="Reason",
                orientation="h",
                custom_data=["Label"],
                labels={"Reason": "Reason code", "Count": "Tickers"},
            )
            fig.update_traces(
                hovertemplate=(
                    "Reason: %{y}<br>Label: %{customdata[0]}<br>Tickers: %{x}<extra></extra>"
                ),
            )
            fig.update_layout(
                hovermode="closest",
                height=max(220, 36 * len(chart_df)),
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(categoryorder="total ascending"),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)


def _render_delisting_section(manifests_df: pd.DataFrame) -> None:
    """CRSP delisting reasons for the full top-1k universe, with on-disk caching."""
    st.markdown("### Delisting reasons (CRSP)")
    st.caption(
        "CRSP `crsp.msedelist` records *why* a stock left the market (`dlstcd`) and the "
        "return on exit (`dlret`). Cached for all **1,000** universe PERMNOs — only missing "
        "names are queried from WRDS."
    )

    universe = _load_top1k_universe()
    if universe.empty:
        st.warning(f"Universe file not found: `{TOP1K_UNIVERSE_PATH}`")
        return

    target_permnos = {int(p) for p in universe["permno"].dropna().tolist()}

    cache = load_delisting_cache()
    cached_set = (
        {int(p) for p in cache["permno"].dropna().tolist()} if not cache.empty else set()
    )
    missing = sorted(target_permnos - cached_set)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Universe tickers", f"{len(target_permnos):,}")
    m2.metric("Delisting info cached", f"{len(target_permnos & cached_set):,}")
    m3.metric("Missing (not yet checked)", f"{len(missing):,}")
    m4.metric("Cache rows total", f"{len(cache):,}")

    creds_ok = wrds_credentials_available()
    btn_cols = st.columns([2, 2, 4])
    fetch_missing = btn_cols[0].button(
        f"⬇️ Fetch missing ({len(missing):,})",
        disabled=(not creds_ok or not missing),
        help=None if creds_ok else "WRDS credentials are required to query CRSP.",
    )
    refetch_all = btn_cols[1].button(
        "🔄 Re-fetch entire universe",
        disabled=not creds_ok,
        help="Ignore the cache and re-query all 1,000 PERMNOs from CRSP.",
    )
    if not creds_ok:
        st.warning("WRDS credentials not configured — set `WRDS_USERNAME` / `WRDS_PASSWORD` to fetch.")

    if fetch_missing or refetch_all:
        permnos_to_pull = target_permnos if refetch_all else set(missing)
        with st.spinner(f"Querying CRSP delisting info for {len(permnos_to_pull):,} PERMNOs…"):
            try:
                cache, n_new = update_delisting_cache(permnos_to_pull, force=refetch_all)
                st.success(f"Updated delisting cache — queried {n_new:,} PERMNO(s). "
                           f"Saved to `{DELISTING_CACHE_PATH.relative_to(PROJECT_ROOT)}`.")
                st.rerun()
            except Exception as exc:
                st.error(f"Delisting lookup failed: {exc}")

    if cache.empty:
        st.info("No delisting info cached yet. Click **Fetch missing** to pull it from CRSP.")
        return

    view = cache[cache["permno"].isin(target_permnos)].copy()
    if view.empty:
        st.caption("No cached delisting rows match the universe.")
        return

    n_delisted = int(view["delisted"].fillna(False).astype(bool).sum())
    n_active = len(view) - n_delisted
    s1, s2, s3 = st.columns(3)
    s1.metric("Delisted (CRSP exit)", f"{n_delisted:,}")
    s2.metric("Still active", f"{n_active:,}")
    delisted_only = view[view["delisted"].fillna(False).astype(bool)]
    avg_dlret = pd.to_numeric(delisted_only["dlret"], errors="coerce").mean()
    s3.metric("Mean delisting return", f"{avg_dlret:.3f}" if pd.notna(avg_dlret) else "—")

    if not delisted_only.empty:
        cat_counts = (
            delisted_only["delisting_category"].fillna("unknown")
            .value_counts().rename_axis("Category").reset_index(name="Count")
        )
        fig = px.bar(
            cat_counts.sort_values("Count"),
            x="Count", y="Category", orientation="h",
            labels={"Category": "Delisting category", "Count": "Tickers"},
        )
        fig.update_traces(
            hovertemplate="Category: %{y}<br>Tickers: %{x}<extra></extra>",
        )
        fig.update_layout(
            hovermode="closest",
            height=max(200, 40 * len(cat_counts)),
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        code_counts = (
            delisted_only[["dlstcd", "delisting_label"]]
            .value_counts().rename_axis(["dlstcd", "delisting_label"]).reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        st.dataframe(code_counts, use_container_width=True, hide_index=True)

    # Per-ticker detail: universe + optional manifest status + delisting fields.
    detail = universe.rename(columns={"volume_rank": "rank", "comnam": "company"}).copy()
    if not manifests_df.empty and "permno" in manifests_df.columns:
        status_map = manifests_df[["permno", "status"]].drop_duplicates("permno")
        detail = detail.merge(status_map, on="permno", how="left")
    detail = detail.merge(
        view[["permno", "delisted", "dlstdt", "dlstcd", "delisting_category",
              "delisting_label", "dlret", "nwperm"]],
        on="permno", how="left",
    )
    detail_cols = [c for c in [
        "rank", "ticker", "company", "status", "delisted", "dlstdt", "dlstcd",
        "delisting_category", "delisting_label", "dlret", "nwperm",
    ] if c in detail.columns]
    with st.expander(f"📋 Per-ticker delisting detail ({len(detail):,} universe tickers)", expanded=False):
        st.dataframe(
            detail[detail_cols].sort_values("rank"),
            use_container_width=True, hide_index=True,
        )


def _render_cash_merger_section(manifests_df: pd.DataFrame) -> None:
    """Cash-merger exit returns (CRSP dlstcd 232/233) for the top-1k universe."""
    with st.expander("💵 Cash Merger Exits", expanded=False):
        st.caption(
            "For stocks that left the market via a **cash merger** (CRSP `dlstcd` 232/233), "
            "the final-week return is recovered from CRSP `dlret`, estimated from the SDC "
            "M&A deal price, or set to 0 as a last-price fallback. Only missing PERMNOs are "
            "queried from WRDS."
        )

        universe = _load_top1k_universe()
        if universe.empty:
            st.warning(f"Universe file not found: `{TOP1K_UNIVERSE_PATH}`")
            return

        # Cash-merger candidates come from the CRSP delisting cache (dlstcd 232/233).
        delist = load_delisting_cache()
        candidate_permnos: set[int] = set()
        if not delist.empty and "dlstcd" in delist.columns:
            mask = delist["dlstcd"].isin(CASH_MERGER_CODES)
            candidate_permnos = {int(p) for p in delist.loc[mask, "permno"].dropna().tolist()}

        exit_cache = load_cash_merger_cache()
        checked_permnos = (
            {int(p) for p in exit_cache["permno"].dropna().tolist()}
            if not exit_cache.empty else set()
        )
        missing = sorted(candidate_permnos - checked_permnos)

        m1, m2, m3 = st.columns(3)
        m1.metric("Cash-merger candidates", f"{len(candidate_permnos):,}")
        m2.metric("Resolved (cached)", f"{len(candidate_permnos & checked_permnos):,}")
        m3.metric("Missing (not yet checked)", f"{len(missing):,}")

        if not candidate_permnos:
            st.info(
                "No cash-merger candidates found yet. Populate the **Delisting reasons (CRSP)** "
                "cache first so dlstcd 232/233 names can be identified."
            )
            return

        creds_ok = wrds_credentials_available()
        btn_cols = st.columns([2, 2, 4])
        fetch_missing = btn_cols[0].button(
            f"⬇️ Resolve missing ({len(missing):,})",
            disabled=(not creds_ok or not missing),
            help=None if creds_ok else "WRDS credentials are required to query CRSP/SDC.",
            key="cash_merger_fetch_missing",
        )
        refetch_all = btn_cols[1].button(
            "🔄 Re-resolve all candidates",
            disabled=not creds_ok,
            help="Ignore the cache and re-query every cash-merger PERMNO.",
            key="cash_merger_refetch_all",
        )
        if not creds_ok:
            st.warning("WRDS credentials not configured — set `WRDS_USERNAME` / `WRDS_PASSWORD` to fetch.")

        if fetch_missing or refetch_all:
            permnos_to_pull = candidate_permnos if refetch_all else set(missing)
            with st.spinner(f"Resolving cash-merger exits for {len(permnos_to_pull):,} PERMNOs…"):
                try:
                    exit_cache, n_new = update_cash_merger_cache(
                        permnos_to_pull, force=refetch_all
                    )
                    st.success(
                        f"Updated cash-merger cache — checked {n_new:,} PERMNO(s). "
                        f"Saved to `{CASH_MERGER_CACHE_PATH.relative_to(PROJECT_ROOT)}`."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Cash-merger resolution failed: {exc}")

        if exit_cache.empty:
            st.info("No cash-merger exits resolved yet. Click **Resolve missing** to compute them.")
            return

        # Merger rows only (exclude the "not_cash_merger" bookkeeping rows).
        resolved = exit_cache[
            exit_cache["exit_source"].isin(list(_EXIT_SOURCE_ICONS.keys()))
        ].copy()
        if resolved.empty:
            st.caption("No resolved cash-merger exits in the cache yet.")
            return

        summary = get_cash_merger_summary(resolved)
        if not summary.empty:
            summary_display = summary.copy()
            summary_display["exit_source"] = summary_display["exit_source"].map(
                lambda s: f"{_EXIT_SOURCE_ICONS.get(s, '')} {s}".strip()
            )
            st.dataframe(summary_display, use_container_width=True, hide_index=True)

        counts = resolved["exit_source"].value_counts()
        n_crsp = int(counts.get("crsp_dlret", 0))
        n_sdc = int(counts.get("sdc_pricepersh", 0))
        n_fallback = int(counts.get("crsp_dlprc_fallback", 0))
        st.markdown(
            f"**{n_crsp} of {len(resolved)}** cash merger tickers resolved via CRSP `dlret` "
            f"| **{n_sdc}** via SDC deal price | **{n_fallback}** fallback (last price)."
        )

        # Attach ticker/company for readability before download.
        export = resolved.merge(
            universe.rename(columns={"comnam": "company"})[["permno", "ticker", "company"]],
            on="permno", how="left",
        )
        st.download_button(
            "⬇️ Download cash-merger exits (CSV)",
            data=export.to_csv(index=False).encode("utf-8"),
            file_name="cash_merger_exits.csv",
            mime="text/csv",
            key="cash_merger_download",
        )


@st.cache_resource(show_spinner=False)
def _cached_sentiment_classifier(model_dir: str):
    """Load the fine-tuned PhraseBank classifier once per Streamlit session."""
    tokenizer, model, device = load_classifier(Path(model_dir))
    return tokenizer, model, device


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_phrasebank_summary():
    """Load PhraseBank class balance (cached — dataset is small and static)."""
    raw = load_phrasebank()
    balance = dataset_class_balance(raw)
    splits = {name: int(raw[name].num_rows) for name in raw}
    return balance, splits


def _clean_news_text(value: object) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "none", "nan"}:
        return ""
    return text


def _normalize_headline_key(value: object) -> str:
    return re.sub(r"\s+", " ", _clean_news_text(value).lower())


def _event_text_matches_headline(headline: str, event_text: str) -> bool:
    """Heuristic: RavenPack event_text often tags the AAPL mention, not the headline subject."""
    if not headline or not event_text:
        return True
    stop = {
        "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at", "by",
        "market", "talk", "press", "release", "update", "dj", "wsj",
    }
    h_tokens = {
        t for t in re.findall(r"[a-z0-9]+", headline.lower())
        if t not in stop and len(t) > 2
    }
    e_tokens = {
        t for t in re.findall(r"[a-z0-9]+", event_text.lower())
        if t not in stop and len(t) > 2
    }
    if not h_tokens or not e_tokens:
        return True
    return bool(h_tokens & e_tokens)


def _meaningful_event_text(headline: str, event_text: str) -> bool:
    """True when RavenPack provides a non-empty summary that isn't just the headline repeated."""
    if not event_text:
        return False
    if event_text == headline:
        return False
    return len(event_text) >= 15


@st.cache_data(ttl=3600, show_spinner=False)
def _load_refinitiv_story_lookup(ticker: str) -> dict[str, str]:
    """Map normalized Refinitiv headline -> cached full story text (when available)."""
    slug = ticker.strip().lower()
    headlines_path = NEWS_REFINITIV_DIR / f"{slug}_headlines_checkpoint.parquet"
    stories_path = NEWS_REFINITIV_DIR / f"{slug}_story_text_checkpoint.parquet"
    if not headlines_path.exists() or not stories_path.exists():
        return {}
    try:
        headlines = pd.read_parquet(headlines_path)
        stories = pd.read_parquet(stories_path)
        story_col = "story_id" if "story_id" in stories.columns else "storyId"
        headline_col = "story_id" if "story_id" in headlines.columns else "storyId"
        text_col = "article_text" if "article_text" in stories.columns else "story_text"
        merged = headlines.merge(stories, left_on=headline_col, right_on=story_col, how="inner")
        lookup: dict[str, str] = {}
        for _, row in merged.iterrows():
            text = _clean_news_text(row.get(text_col))
            if not text:
                continue
            lookup[_normalize_headline_key(row.get("headline"))] = text
        return lookup
    except Exception:
        return {}


def _attach_refinitiv_stories(articles: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Add cached Refinitiv full-story text when the headline matches exactly."""
    if articles.empty:
        return articles
    lookup = _load_refinitiv_story_lookup(ticker)
    if not lookup:
        articles = articles.copy()
        articles["full_story_text"] = None
        articles["refinitiv_story_id"] = None
        return articles

    out = articles.copy()
    out["full_story_text"] = out["headline"].map(
        lambda h: lookup.get(_normalize_headline_key(h))
    )
    out["refinitiv_story_id"] = None
    return out


def _ravenpack_polarity(score: object) -> str:
    """Map RavenPack event_sentiment_score to a coarse polarity label."""
    try:
        val = float(score)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if pd.isna(val):
        return "—"
    if val > 0.05:
        return "Positive"
    if val < -0.05:
        return "Negative"
    return "Neutral"


def _normalize_ravenpack_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Unify RavenPack frames from rich cache, batch cache, or live WRDS."""
    if df.empty:
        return df

    out = df.copy()
    if "article_time" not in out.columns:
        if "timestamp_utc" in out.columns:
            out["article_time"] = pd.to_datetime(out["timestamp_utc"], utc=True, errors="coerce")
        else:
            out["article_time"] = pd.NaT
    else:
        out["article_time"] = pd.to_datetime(out["article_time"], utc=True, errors="coerce")

    if "relevance_score" not in out.columns and "relevance" in out.columns:
        out["relevance_score"] = pd.to_numeric(out["relevance"], errors="coerce") / 100
    else:
        out["relevance_score"] = pd.to_numeric(out.get("relevance_score"), errors="coerce")

    out["event_sentiment_score"] = pd.to_numeric(out.get("event_sentiment_score"), errors="coerce")
    if "sentiment_score" not in out.columns:
        out["sentiment_score"] = out["relevance_score"] * out["event_sentiment_score"]
    else:
        out["sentiment_score"] = pd.to_numeric(out["sentiment_score"], errors="coerce")

    if "headline" in out.columns:
        out["headline"] = out["headline"].map(_clean_news_text)
    else:
        out["headline"] = ""
    if "event_text" in out.columns:
        out["event_text"] = out["event_text"].map(_clean_news_text)
    else:
        out["event_text"] = ""

    out["text"] = out.apply(
        lambda row: row["event_text"] or row["headline"] or "",
        axis=1,
    )
    out["has_event_summary"] = out.apply(
        lambda row: _meaningful_event_text(str(row["headline"]), str(row["event_text"])),
        axis=1,
    )
    out["event_text_matches_headline"] = out.apply(
        lambda row: _event_text_matches_headline(str(row["headline"]), str(row["event_text"])),
        axis=1,
    )
    out["polarity"] = out["event_sentiment_score"].map(_ravenpack_polarity)
    out["date_str"] = out["article_time"].dt.strftime("%Y-%m-%d %H:%M").fillna("—")
    out["text_preview"] = (
        out["text"].astype(str).str.slice(0, 120).str.replace("\n", " ", regex=False)
    )
    return out


def _load_ravenpack_articles_for_display(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    live: bool,
    max_rows: int,
) -> tuple[pd.DataFrame, str]:
    """Load RavenPack articles, preferring rich per-ticker cache with headline/event_text."""
    ticker_clean = ticker.strip().upper()
    if not ticker_clean:
        return pd.DataFrame(), "Enter a ticker."

    start_s = to_query_date(start_date)
    end_s = to_query_date(end_date)
    slug = ticker_clean.lower()
    source_notes: list[str] = []
    articles = pd.DataFrame()

    if not live:
        rich_candidates = [
            NEWS_RAVENPACK_DIR / f"{slug}_articles_2003_2014.parquet",
            NEWS_RAVENPACK_DIR / f"{slug}_rp_checkpoint.parquet",
        ]
        for path in rich_candidates:
            if not path.exists():
                continue
            frame = pd.read_parquet(path)
            frame = _normalize_ravenpack_frame(frame)
            frame = _filter_cached_frame(frame, "article_time", start_s, end_s)
            if not frame.empty:
                articles = frame
                source_notes.append(f"rich cache `{path.name}`")
                break

        if articles.empty:
            cached = load_cached_dashboard_result(ticker_clean, start_s, end_s)
            if cached:
                rp_block = cached.get("providers", {}).get("ravenpack", {})
                batch_frame = rp_block.get("articles")
                if isinstance(batch_frame, pd.DataFrame) and not batch_frame.empty:
                    articles = _normalize_ravenpack_frame(batch_frame)
                    articles = _filter_cached_frame(articles, "article_time", start_s, end_s)
                    if not articles.empty:
                        source_notes.append("batch cache (scores only — no headline/event_text)")

        if not source_notes:
            source_notes.append("no cached RavenPack for this ticker/range")
    else:
        if not wrds_credentials_available():
            return pd.DataFrame(), "WRDS not configured — add credentials to `.env`."
        try:
            live_frame = live_data.query_ravenpack_articles(
                ticker_clean,
                start_s,
                end_s,
                include_text=True,
            )
            articles = _normalize_ravenpack_frame(live_frame)
            articles = _filter_cached_frame(articles, "article_time", start_s, end_s)
            if not articles.empty:
                source_notes.append("live WRDS (with headline/event_text)")
            else:
                source_notes.append("live WRDS returned no rows")
        except Exception as exc:
            return pd.DataFrame(), f"RavenPack live pull failed: {exc}"

    if not articles.empty and max_rows > 0:
        articles = articles.sort_values("article_time", ascending=False).head(int(max_rows))
    articles = articles.reset_index(drop=True)
    return _attach_refinitiv_stories(articles, ticker_clean), "; ".join(source_notes)


def _enrich_ravenpack_with_model(articles: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """Optionally add PhraseBank model labels alongside RavenPack scores."""
    if articles.empty:
        return articles
    scorable = articles["text"].astype(str).str.strip()
    if not scorable.any():
        return articles
    tokenizer, model, device = _cached_sentiment_classifier(str(model_dir))
    preds = predict_sentences(scorable.tolist(), tokenizer, model, device)
    out = articles.copy()
    out["model_label"] = preds["pred"].values
    for col in preds.columns:
        if col.startswith("p("):
            out[col] = preds[col].values
    return out


def _render_ravenpack_article_side_by_side(row: pd.Series) -> None:
    """Two-column layout: article text (left) and RavenPack scores (right)."""
    text_col, label_col = st.columns([3, 1])

    with text_col:
        st.markdown("##### Article")
        headline = _clean_news_text(row.get("headline"))
        body = _clean_news_text(row.get("event_text"))
        news_type = _clean_news_text(row.get("news_type"))
        source_name = _clean_news_text(row.get("source_name"))

        if headline:
            st.markdown(f"**{headline}**")

        if body and body != headline:
            if not _event_text_matches_headline(headline, body):
                st.warning(
                    "RavenPack's `event_text` does **not** describe this headline — it is the "
                    "AAPL-tagged snippet from the same news item (common in *Market Talk* / "
                    "market-wrap columns). This is **not** a full article body."
                )
            st.markdown("**RavenPack `event_text`** *(short entity-tagged snippet, avg ~37 chars)*")
            st.write(body)
            st.caption(
                "This is the richest text RavenPack stores for most rows. It is **not** the full "
                "Reuters/Dow Jones article."
            )
        elif headline:
            st.info(
                "RavenPack provides **headline only** for this row — there is no `event_text` "
                "sentence in the dataset. "
                + (
                    f"This row is tagged `{news_type}` (tabular/market data, not a news article). "
                    if news_type == "TABULAR-MATERIAL"
                    else "Roughly **78%** of RavenPack rows are headline-only; enable "
                    "**Only rows with RP summary** above to browse entries with a snippet."
                )
            )
        else:
            fallback = _clean_news_text(row.get("text"))
            if fallback:
                st.write(fallback)
            else:
                st.warning(
                    "No text available. Batch cache stores scores only — run "
                    "`notebooks/fetch_news_articles.ipynb` or **Re-pull live** for headlines."
                )

        meta_bits = [bit for bit in [source_name, news_type] if bit]
        if meta_bits:
            st.caption("Source · " + " · ".join(meta_bits))

        full_story = _clean_news_text(row.get("full_story_text"))
        if full_story and full_story not in {headline, body}:
            st.markdown("**Full story (Refinitiv cache)**")
            st.write(full_story)
        elif headline and not full_story:
            with st.expander("Need the full article text?", expanded=False):
                st.markdown(
                    "RavenPack does not store full wire stories. Options:\n"
                    "- Check whether this headline exists in the Refinitiv cache "
                    f"(`{NEWS_REFINITIV_DIR.relative_to(PROJECT_ROOT)}/`)\n"
                    "- Pull story text via **Data Explorer → Refinitiv news** (live API)\n"
                    "- Re-run `notebooks/fetch_news_articles.ipynb` to expand story checkpoints"
                )
                ref_story_id = _clean_news_text(row.get("refinitiv_story_id"))
                if ref_story_id and refinitiv_configured(PROJECT_ROOT):
                    if st.button(
                        "Fetch full story from Refinitiv (live)",
                        key=f"rp_fetch_story_{row.name}",
                    ):
                        try:
                            st.write(load_refinitiv_story_text(ref_story_id))
                        except Exception as exc:
                            st.error(f"Could not load story: {exc}")

    with label_col:
        st.markdown("##### TRNA substitute")
        st.caption("RavenPack → paper Eq. 8: `relevance × (pos − neg)`")
        rel = row.get("relevance_score")
        ess = row.get("event_sentiment_score")
        ss = row.get("sentiment_score")
        st.metric("Relevance (0–1)", f"{rel:.2f}" if pd.notna(rel) else "—")
        st.metric("Event sentiment (−1 to +1)", f"{ess:.3f}" if pd.notna(ess) else "—")
        st.metric("Sentiment score", f"{ss:.3f}" if pd.notna(ss) else "—")
        st.metric("Polarity", str(row.get("polarity", "—")))

        if pd.notna(row.get("model_label")):
            st.divider()
            st.markdown("##### PhraseBank model")
            st.metric("Model label", str(row["model_label"]))
            if pd.notna(row.get("p(positive)")):
                st.caption(
                    f"P(pos) {row['p(positive)']:.1%} · "
                    f"P(neu) {row['p(neutral)']:.1%} · "
                    f"P(neg) {row['p(negative)']:.1%}"
                )


def _render_ravenpack_articles_browser(articles: pd.DataFrame) -> None:
    """Browse RavenPack articles with a summary table and side-by-side detail view."""
    with_summary = int(articles.get("has_event_summary", pd.Series(dtype=bool)).sum())
    with_full_story = int(
        articles["full_story_text"].apply(lambda x: bool(_clean_news_text(x))).sum()
        if "full_story_text" in articles.columns
        else 0
    )
    with_scores = int(articles["event_sentiment_score"].notna().sum())
    with_headline = int(articles["headline"].astype(str).str.strip().ne("").sum())
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Articles", f"{len(articles):,}")
    m2.metric("With RP summary", f"{with_summary:,}")
    m3.metric("Full Refinitiv story", f"{with_full_story:,}")
    m4.metric("Headline only", f"{with_headline - with_summary:,}")

    filter_cols = st.columns([2, 1])
    polarity_filter = filter_cols[0].multiselect(
        "Filter by RavenPack polarity",
        options=["Positive", "Neutral", "Negative", "—"],
        default=["Positive", "Neutral", "Negative", "—"],
        key="ravenpack_polarity_filter",
    )
    only_event_text = filter_cols[1].checkbox(
        "Only rows with RP summary",
        value=False,
        help="RavenPack `event_text` is a short entity-tagged snippet (~37 chars on average), "
        "not a full article. This filter hides headline-only rows.",
        key="ravenpack_only_event_text",
    )
    filter_token = (tuple(sorted(polarity_filter)), only_event_text)
    if st.session_state.get("ravenpack_polarity_filter_token") != filter_token:
        st.session_state["ravenpack_polarity_filter_token"] = filter_token
        st.session_state.pop("ravenpack_articles_table_selected_idx", None)
    view = articles[articles["polarity"].isin(polarity_filter)].copy()
    if only_event_text and "has_event_summary" in view.columns:
        view = view[view["has_event_summary"]]
    view = view.reset_index(drop=True)
    if view.empty:
        st.info("No articles match the selected polarity filter.")
        return

    table_cols = [
        c
        for c in [
            "date_str",
            "polarity",
            "relevance_score",
            "event_sentiment_score",
            "sentiment_score",
            "model_label",
            "has_event_summary",
            "event_text_matches_headline",
            "text_preview",
        ]
        if c in view.columns
    ]
    display_df = view[table_cols].rename(
        columns={
            "date_str": "Date",
            "polarity": "Polarity",
            "relevance_score": "Relevance",
            "event_sentiment_score": "Event sent.",
            "sentiment_score": "Sent. score",
            "model_label": "Model label",
            "has_event_summary": "RP summary",
            "event_text_matches_headline": "Summary≈headline",
            "text_preview": "Text preview",
        }
    )
    table_key = "ravenpack_articles_table"
    selection_event = st.dataframe(
        display_df,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
        use_container_width=True,
        hide_index=True,
        height=320,
    )
    selected_rows = (
        selection_event.selection.rows
        if selection_event.selection is not None
        else []
    )
    if not selected_rows:
        prior = st.session_state.get(f"{table_key}_selected_idx")
        selected_idx = int(prior) if isinstance(prior, int) and 0 <= prior < len(view) else 0
    else:
        selected_idx = int(selected_rows[0])
        st.session_state[f"{table_key}_selected_idx"] = selected_idx

    st.caption("Click any row (e.g. **Text preview**) to inspect the full article below.")
    _render_ravenpack_article_side_by_side(view.iloc[selected_idx])


def _news_inventory_cache_token() -> str:
    """Invalidate news-inventory cache when batch manifests or news exports change."""
    parts = [_manifest_cache_token()]
    for directory in (NEWS_RAVENPACK_DIR, NEWS_REFINITIV_DIR):
        if directory.exists():
            parquet_files = list(directory.glob("*.parquet"))
            latest = max((p.stat().st_mtime for p in parquet_files), default=0.0)
            parts.append(f"{len(parquet_files)}:{latest:.0f}")
        else:
            parts.append("0")
    return "|".join(parts)


def _parquet_row_count(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


def _parquet_non_null_count(path: Path, column: str) -> tuple[int, int]:
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    rows = int(parquet_file.metadata.num_rows)
    if column not in parquet_file.schema_arrow.names:
        return rows, 0
    series = parquet_file.read(columns=[column])[column]
    return rows, rows - int(series.null_count)


@st.cache_data(ttl=60, show_spinner=False)
def _scan_news_data_inventory(cache_token: str) -> dict[str, object]:
    """Aggregate local news/sentiment coverage from batch cache and rich exports."""
    del cache_token  # cache-bust token only

    window_start = DEFAULT_LOOKUP_START
    window_end = DEFAULT_LOOKUP_END

    rp_batch = {
        "tickers": 0,
        "rows": 0,
        "with_relevance": 0,
        "with_sentiment": 0,
    }
    rf_batch = {"tickers": 0, "rows": 0}
    wrds_tickers = 0

    if TOP1K_BY_TICKER_DIR.exists():
        for ticker_dir in TOP1K_BY_TICKER_DIR.glob("rank_*"):
            manifest_path = ticker_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    window_start = str(manifest.get("start_date") or window_start)
                    window_end = str(manifest.get("end_date") or window_end)
                    for provider in manifest.get("provider_status", []):
                        if provider.get("provider") == "wrds" and provider.get("status") == "ok":
                            wrds_tickers += 1
                            break
                except Exception:
                    pass

            rp_path = ticker_dir / "ravenpack_articles.parquet"
            if rp_path.exists():
                rows, with_relevance = _parquet_non_null_count(rp_path, "relevance")
                _, with_sentiment = _parquet_non_null_count(rp_path, "event_sentiment_score")
                rp_batch["tickers"] += 1
                rp_batch["rows"] += rows
                rp_batch["with_relevance"] += with_relevance
                rp_batch["with_sentiment"] += with_sentiment

            rf_path = ticker_dir / "refinitiv_news.parquet"
            if rf_path.exists():
                rf_batch["tickers"] += 1
                rf_batch["rows"] += _parquet_row_count(rf_path)

    rp_rich: list[dict[str, object]] = []
    if NEWS_RAVENPACK_DIR.exists():
        for path in sorted(NEWS_RAVENPACK_DIR.glob("*_articles_*.parquet")):
            ticker = path.name.split("_articles_")[0].upper()
            rows, with_headlines = _parquet_non_null_count(path, "headline")
            _, with_sentiment = _parquet_non_null_count(path, "event_sentiment_score")
            _, with_event_text = _parquet_non_null_count(path, "event_text")
            rp_rich.append({
                "ticker": ticker,
                "rows": rows,
                "with_headlines": with_headlines,
                "with_sentiment": with_sentiment,
                "with_event_text": with_event_text,
            })

    rf_checkpoints: list[dict[str, object]] = []
    if NEWS_REFINITIV_DIR.exists():
        for path in sorted(NEWS_REFINITIV_DIR.glob("*_story_text_checkpoint.parquet")):
            ticker = path.name.replace("_story_text_checkpoint.parquet", "").upper()
            stories, with_text = _parquet_non_null_count(path, "article_text")
            headlines_path = NEWS_REFINITIV_DIR / f"{ticker.lower()}_headlines_checkpoint.parquet"
            headline_rows = _parquet_row_count(headlines_path) if headlines_path.exists() else 0
            rf_checkpoints.append({
                "ticker": ticker,
                "stories": stories,
                "with_full_text": with_text,
                "headlines_checkpoint": headline_rows,
            })

    return {
        "window_start": window_start,
        "window_end": window_end,
        "wrds_tickers": wrds_tickers,
        "ravenpack_batch": rp_batch,
        "refinitiv_batch": rf_batch,
        "ravenpack_rich": rp_rich,
        "refinitiv_checkpoints": rf_checkpoints,
    }


def _fmt_count(value: object) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def render_news_data_coverage_section() -> None:
    """Summarize which news/sentiment datasets are cached locally and what they contain."""
    st.markdown("### Local news & sentiment data — coverage")
    st.caption(
        "Snapshot of datasets on **this machine** under `data/raw/`. "
        f"Paper replication window: **{DEFAULT_LOOKUP_START}** → **{DEFAULT_LOOKUP_END}** "
        "(top-1,000 CRSP tickers by volume). "
        "**TRNA** (Thomson Reuters News Analytics) is proprietary and **not** cached here — "
        "RavenPack scores are our TRNA substitute."
    )

    with st.expander("What is each dataset?", expanded=False):
        st.markdown(
            "| Source | Role | Text | Labels |\n"
            "|---|---|---|---|\n"
            "| **Financial PhraseBank** | Fine-tune a 3-way sentiment classifier | Short finance sentences | Human `negative` / `neutral` / `positive` |\n"
            "| **RavenPack (WRDS)** | TRNA substitute — relevance + sentiment scores | Headlines + short `event_text` snippets in rich exports; **batch cache stores scores only** | `relevance` on all rows; `event_sentiment_score` / `sentiment_score` on a subset |\n"
            "| **Refinitiv / LSEG** | Wire headlines + optional full stories | Headlines in batch cache; full story text only when explicitly fetched | **No** TRNA-style sentiment labels |\n"
            "| **WRDS CRSP** | Prices, delistings, universe metadata | Not news | Not sentiment |\n\n"
            "RavenPack `event_text` is **not** a full article — it is typically a short "
            "entity-tagged snippet (~37 characters on average for AAPL), not the wire story body."
        )

    try:
        inventory = _scan_news_data_inventory(_news_inventory_cache_token())
    except Exception as exc:
        st.warning(f"Could not scan local news caches: {exc}")
        return

    window = f"{inventory['window_start']} → {inventory['window_end']}"
    rp_batch = inventory["ravenpack_batch"]
    rf_batch = inventory["refinitiv_batch"]
    rp_rich = inventory["ravenpack_rich"]
    rf_chk = inventory["refinitiv_checkpoints"]

    rich_tickers = ", ".join(str(r["ticker"]) for r in rp_rich) or "—"
    story_tickers = ", ".join(str(r["ticker"]) for r in rf_chk) or "—"
    rich_headlines = sum(int(r["with_headlines"]) for r in rp_rich)
    rich_sentiment = sum(int(r["with_sentiment"]) for r in rp_rich)
    rich_event_text = sum(int(r["with_event_text"]) for r in rp_rich)
    story_full_text = sum(int(r["with_full_text"]) for r in rf_chk)

    phrasebank_total: int | None = None
    try:
        _, splits = _cached_phrasebank_summary()
        phrasebank_total = int(sum(splits.values()))
    except Exception:
        pass

    overview_rows = [
        {
            "Dataset": "Financial PhraseBank",
            "Role": "Classifier training (not ticker news)",
            "Tickers": "—",
            "Date window": "Static benchmark",
            "Rows / articles": _fmt_count(phrasebank_total),
            "Headlines / titles": _fmt_count(phrasebank_total),
            "Sentiment labels": _fmt_count(phrasebank_total),
            "Full article text": _fmt_count(phrasebank_total),
        },
        {
            "Dataset": "RavenPack — batch cache",
            "Role": "TRNA substitute (scores only)",
            "Tickers": _fmt_count(rp_batch["tickers"]),
            "Date window": window,
            "Rows / articles": _fmt_count(rp_batch["rows"]),
            "Headlines / titles": "0 (not stored)",
            "Sentiment labels": (
                f"{_fmt_count(rp_batch['with_relevance'])} relevance · "
                f"{_fmt_count(rp_batch['with_sentiment'])} event_sentiment"
            ),
            "Full article text": "0",
        },
        {
            "Dataset": "RavenPack — rich export (text + scores)",
            "Role": "TRNA substitute + readable text",
            "Tickers": rich_tickers,
            "Date window": window,
            "Rows / articles": _fmt_count(sum(int(r["rows"]) for r in rp_rich) or None),
            "Headlines / titles": _fmt_count(rich_headlines or None),
            "Sentiment labels": _fmt_count(rich_sentiment or None),
            "Full article text": f"{_fmt_count(rich_event_text or None)} RP snippets (not full stories)",
        },
        {
            "Dataset": "Refinitiv — batch headlines",
            "Role": "Wire headlines (no sentiment)",
            "Tickers": _fmt_count(rf_batch["tickers"]),
            "Date window": window,
            "Rows / articles": _fmt_count(rf_batch["rows"]),
            "Headlines / titles": _fmt_count(rf_batch["rows"]),
            "Sentiment labels": "0",
            "Full article text": "0",
        },
        {
            "Dataset": "Refinitiv — story text checkpoints",
            "Role": "Full wire stories (on-demand fetch)",
            "Tickers": story_tickers,
            "Date window": "Partial (per ticker)",
            "Rows / articles": _fmt_count(sum(int(r["stories"]) for r in rf_chk) or None),
            "Headlines / titles": _fmt_count(sum(int(r["headlines_checkpoint"]) for r in rf_chk) or None),
            "Sentiment labels": "0",
            "Full article text": _fmt_count(story_full_text or None),
        },
    ]

    st.dataframe(pd.DataFrame(overview_rows), hide_index=True, use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WRDS universe tickers", _fmt_count(inventory["wrds_tickers"]))
    m2.metric("RavenPack tickers (batch)", _fmt_count(rp_batch["tickers"]))
    m3.metric("Refinitiv tickers (batch)", _fmt_count(rf_batch["tickers"]))
    m4.metric("Rich text exports", f"{len(rp_rich)} ticker(s)")

    if rp_rich:
        st.caption(
            "**RavenPack rich exports:** "
            + " · ".join(
                f"{r['ticker']}: {_fmt_count(r['rows'])} rows, "
                f"{_fmt_count(r['with_headlines'])} headlines, "
                f"{_fmt_count(r['with_sentiment'])} scored, "
                f"{_fmt_count(r['with_event_text'])} with RP snippet"
                for r in rp_rich
            )
        )
    if rf_chk:
        st.caption(
            "**Refinitiv story checkpoints:** "
            + " · ".join(
                f"{r['ticker']}: {_fmt_count(r['headlines_checkpoint'])} headlines cached, "
                f"{_fmt_count(r['with_full_text'])} full stories"
                for r in rf_chk
            )
        )


def _phrasebank_model_cache_token() -> str:
    """Invalidate probability-chart cache when the saved checkpoint changes."""
    model_dir = resolve_model_dir()
    parts: list[str] = []
    for name in ("config.json", "model.safetensors", "pytorch_model.bin", "metrics.json"):
        path = model_dir / name
        if path.exists():
            parts.append(f"{name}:{path.stat().st_mtime:.0f}")
    return "|".join(parts) or "none"


@st.cache_data(ttl=3600, show_spinner="Scoring PhraseBank splits for probability chart…")
def _cached_phrasebank_probability_chart(cache_token: str) -> pd.DataFrame:
    del cache_token  # cache-bust token only
    return phrasebank_probability_chart_frame()


def render_phrasebank_hf_baseline_tab() -> None:
    """Standalone overview of the Hugging Face PhraseBank baseline model."""
    st.header("PhraseBank HF Baseline")
    st.caption(
        "Benchmark classifier: **`distilbert-base-uncased`** fine-tuned on Financial PhraseBank "
        "(Hugging Face). Documents the training dataset, evaluation metrics, and predicted class "
        "probabilities across PhraseBank splits. For live inference and RavenPack fine-tuning, "
        "use the **Sentiment Lab** tab."
    )

    if not finetuning_deps_available():
        st.warning(
            "Fine-tuning dependencies are not installed. Run "
            "`pip install -r requirements-finetuning.txt` to load the dataset and chart."
        )
        return

    metrics = load_metrics()
    model_dir = resolve_model_dir()
    has_model = model_is_saved(model_dir)

    # ── Training summary ────────────────────────────────────────────────────────
    st.markdown("### Model & training")
    epochs = metrics.get("epochs")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Backbone", str(metrics.get("model_name", MODEL_NAME)).split("/")[-1])
    t2.metric("Epochs trained", str(epochs) if epochs is not None else "—")
    t3.metric("Learning rate", str(metrics.get("learning_rate", "—")))
    t4.metric("Batch size", str(metrics.get("per_device_train_batch_size", "—")))

    st.markdown(
        f"| Item | Value |\n"
        f"| --- | --- |\n"
        f"| **Checkpoint** | `{model_dir.relative_to(PROJECT_ROOT)}` "
        f"({'on disk' if has_model else 'not saved — metrics from notebook run'}) |\n"
        f"| **Base weights** | `{metrics.get('model_name', MODEL_NAME)}` (Hugging Face) |\n"
        f"| **Task** | 3-way sequence classification (`negative` / `neutral` / `positive`) |\n"
        f"| **Max tokens** | {metrics.get('max_length', 128)} |\n"
        f"| **Best checkpoint** | validation **macro-F1** (`load_best_model_at_end`) |\n"
        f"| **Training device** | {str(metrics.get('device', '—')).upper()} |\n"
        f"| **Notebook** | `notebooks/liquidAI_prep.ipynb` |\n"
        f"| **RavenPack adapt** | `notebooks/finetune_on_ravenpack.ipynb` (next step) |"
    )

    # ── Performance metrics ─────────────────────────────────────────────────────
    st.markdown("### Performance metrics")
    val_f1 = metrics.get("validation", {}).get("eval_f1")
    test_f1 = metrics.get("test", {}).get("eval_f1")
    val_acc = metrics.get("validation", {}).get("eval_accuracy")
    test_acc = metrics.get("test", {}).get("eval_accuracy")
    val_loss = metrics.get("validation", {}).get("eval_loss")
    test_loss = metrics.get("test", {}).get("eval_loss")
    train_loss = metrics.get("train_loss")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test macro-F1", f"{test_f1:.1%}" if test_f1 is not None else "—")
    m2.metric("Test accuracy", f"{test_acc:.1%}" if test_acc is not None else "—")
    m3.metric("Val macro-F1", f"{val_f1:.1%}" if val_f1 is not None else "—")
    m4.metric("Val accuracy", f"{val_acc:.1%}" if val_acc is not None else "—")
    loss_bits: list[str] = []
    if train_loss is not None:
        loss_bits.append(f"Train loss: {train_loss:.3f}")
    if val_loss is not None and test_loss is not None:
        loss_bits.append(f"Val loss: {val_loss:.3f} · Test loss: {test_loss:.3f}")
    st.caption(
        "**macro-F1** is the primary metric (equal weight per class)."
        + (f" {' · '.join(loss_bits)}" if loss_bits else "")
    )

    with st.expander("Raw training metrics (metrics.json)", expanded=False):
        st.json(metrics)

    st.divider()

    # ── Dataset dashboard ───────────────────────────────────────────────────────
    st.markdown("### Dataset dashboard")
    st.caption(
        f"Financial PhraseBank (`{PRIMARY_DATASET}`): gold-label composition, split sizes, "
        "and how the saved checkpoint scores every sentence in each split."
    )

    with st.expander("Where it comes from & how labels were built", expanded=False):
        st.markdown(
            "**Source** — Malo et al. (Aalto University, 2014); "
            "[original paper](https://arxiv.org/abs/1307.5336). "
            "~4,840 English financial-news sentences, 3 classes.\n\n"
            "**Annotation** — 16 finance-background annotators; 5–8 votes per sentence; "
            "gold label = majority vote. We use the **`sentences_50agree`** subset "
            "(≥50% annotator agreement — most data, noisiest labels).\n\n"
            "**Canonical HF dataset** — `takala/financial_phrasebank` is script-based and "
            "no longer loads on `datasets` v4/v5; this repo uses the Parquet mirror above.\n\n"
            "**Splits** — " + SPLIT_SOURCE + "\n\n"
            "**Read more** — "
            "[`docs/financial_phrasebank.md`](https://github.com/armandordorica/"
            "sentiment_learn_to_rank_paper/blob/main/docs/financial_phrasebank.md) · "
            "[dataset card](https://huggingface.co/datasets/takala/financial_phrasebank)"
        )

    try:
        balance, splits = _cached_phrasebank_summary()
        st.markdown("#### Gold labels & splits")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Train rows", f"{splits.get('train', 0):,}")
        s2.metric("Validation rows", f"{splits.get('validation', 0):,}")
        s3.metric("Test rows", f"{splits.get('test', 0):,}")
        s4.metric("Total", f"{sum(splits.values()):,}")

        split_df = pd.DataFrame(
            {
                "split": ["train", "validation", "test"],
                "rows": [splits.get("train", 0), splits.get("validation", 0), splits.get("test", 0)],
            }
        )
        c1, c2 = st.columns(2)
        with c1:
            fig_bal = px.bar(
                balance.sort_values("count"),
                x="count",
                y="label",
                orientation="h",
                labels={"label": "Class", "count": "Train rows"},
                title="Gold label balance (train split)",
            )
            fig_bal.update_traces(hovertemplate="Class: %{y}<br>Count: %{x}<extra></extra>")
            fig_bal.update_layout(hovermode="closest", showlegend=False, height=240)
            st.plotly_chart(fig_bal, use_container_width=True)
        with c2:
            fig_splits = px.bar(
                split_df,
                x="split",
                y="rows",
                labels={"split": "Split", "rows": "Sentences"},
                title="Dataset size by split",
                color="split",
            )
            fig_splits.update_traces(hovertemplate="Split: %{x}<br>Rows: %{y}<extra></extra>")
            fig_splits.update_layout(hovermode="closest", showlegend=False, height=240)
            st.plotly_chart(fig_splits, use_container_width=True)

        st.dataframe(
            pd.DataFrame(
                {
                    "column": ["sentence", "label"],
                    "type": ["string", "ClassLabel"],
                    "meaning": [
                        "One financial-news sentence",
                        "0=negative, 1=neutral, 2=positive",
                    ],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    except Exception as exc:
        st.error(f"Could not load PhraseBank dataset summary: {exc}")

    st.markdown("#### Predicted probabilities on each split")
    st.caption(
        "Scores from the saved checkpoint on every sentence (not gold-label accuracy). "
        "Whisker plot = full distribution; bar chart = **median (p50)** per class."
    )

    if not has_model:
        st.info(
            "No saved checkpoint — train the model in **Sentiment Lab** to populate the charts below."
        )
    else:
        try:
            long_probs = _cached_phrasebank_probability_chart(_phrasebank_model_cache_token())
            chart_orders = {
                "split": PHRASEBANK_SPLIT_ORDER,
                "class": ["negative", "neutral", "positive"],
            }
            chart_labels = {
                "split": "PhraseBank split",
                "probability": "Predicted probability",
                "p50": "Median predicted probability",
                "class": "Class",
            }

            fig_box = px.box(
                long_probs,
                x="split",
                y="probability",
                color="class",
                category_orders=chart_orders,
                labels=chart_labels,
                title="Class probabilities by split (box & whisker)",
                points="outliers",
            )
            fig_box.update_layout(hovermode="closest", boxmode="group", yaxis_range=[0, 1])
            fig_box.update_traces(
                hovertemplate="Split: %{x}<br>Class: %{fullData.name}<br>Probability: %{y:.3f}<extra></extra>"
            )
            st.plotly_chart(fig_box, use_container_width=True)

            p50 = (
                long_probs.groupby(["split", "class"], as_index=False, observed=True)["probability"]
                .median()
                .rename(columns={"probability": "p50"})
            )
            fig_p50 = px.bar(
                p50,
                x="split",
                y="p50",
                color="class",
                barmode="group",
                category_orders=chart_orders,
                labels=chart_labels,
                title="Median class probability by split (p50)",
                text="p50",
            )
            fig_p50.update_traces(
                texttemplate="%{y:.2f}",
                textposition="outside",
                hovertemplate="Split: %{x}<br>Class: %{fullData.name}<br>p50: %{y:.3f}<extra></extra>",
            )
            fig_p50.update_layout(hovermode="closest", yaxis_range=[0, 1])
            st.plotly_chart(fig_p50, use_container_width=True)

            st.dataframe(
                p50.pivot(index="split", columns="class", values="p50").reset_index(),
                hide_index=True,
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Could not build probability charts: {exc}")


def render_sentiment_lab_tab() -> None:
    """Interactive view of notebooks/liquidAI_prep.ipynb — dataset, metrics, inference."""
    st.header("News Sentiment Lab")
    st.caption(
        "Web version of `notebooks/liquidAI_prep.ipynb`: Financial PhraseBank + "
        "DistilBERT fine-tuning for 3-way finance sentiment (negative / neutral / positive). "
        "This is the TRNA-substitute sentiment model for the paper replication. "
        "Below you can see **what was trained**, the **results** (test macro-F1 / accuracy), "
        "the experiment's **inputs & outputs**, and **try the model live** on your own headlines."
    )

    render_news_data_coverage_section()
    st.divider()

    if not finetuning_deps_available():
        st.warning(
            "Fine-tuning dependencies are not installed. Run:\n\n"
            "`pip install -r requirements-finetuning.txt`\n\n"
            "or recreate the conda env from `environment.yml`."
        )
        return

    # ── Compute device ────────────────────────────────────────────────────────
    st.markdown("### Compute device")
    try:
        dev = device_report()
        d1, d2, d3 = st.columns(3)
        d1.metric("CUDA (NVIDIA)", "✅" if dev["cuda_available"] else "—")
        d2.metric("MPS (Apple GPU)", "✅" if dev["mps_available"] else "—")
        accel = dev["selected"].upper()
        d3.metric("Active device", accel if accel != "CPU" else "CPU (no GPU)")
        st.caption(
            f"Selected **{dev['selected']}** — {dev['device_name']} · torch {dev['torch_version']}. "
            "Training and inference below run on this device."
        )
        with st.expander("Run GPU benchmark (CPU vs active device)", expanded=False):
            st.caption(
                "Times the same 10× (4096×4096) matmul on CPU vs the active device. "
                "GPU calls are synchronized and warmed up first for a fair measurement."
            )
            if st.button("Run benchmark", key="sentiment_lab_benchmark"):
                with st.spinner("Benchmarking…"):
                    try:
                        cpu_t = benchmark_matmul("cpu")
                        if dev["selected"] != "cpu":
                            gpu_t = benchmark_matmul(dev["selected"])
                            b1, b2, b3 = st.columns(3)
                            b1.metric("CPU", f"{cpu_t:.3f}s")
                            b2.metric(accel, f"{gpu_t:.3f}s")
                            b3.metric("Speed-up", f"{cpu_t / gpu_t:.1f}×" if gpu_t else "—")
                        else:
                            st.metric("CPU", f"{cpu_t:.3f}s")
                            st.info("No GPU detected to compare against.")
                    except Exception as exc:
                        st.error(f"Benchmark failed: {exc}")
    except Exception as exc:
        st.warning(f"Could not query compute device: {exc}")

    st.divider()

    metrics = load_metrics()
    model_dir = resolve_model_dir()
    has_model = model_is_saved(model_dir)

    # ── Latest training run ────────────────────────────────────────────────────
    epochs = metrics.get("epochs")
    st.markdown("### Latest training run — results")
    st.caption(
        "Fine-tuned **DistilBERT** on Financial PhraseBank for 3-way sentiment. "
        f"This run: **{epochs} epoch(s)**, learning rate {metrics.get('learning_rate', '—')}, "
        f"batch {metrics.get('per_device_train_batch_size', '—')}, on **{str(metrics.get('device', '—')).upper()}**. "
        + (
            "Multi-epoch with the best **validation macro-F1** checkpoint kept "
            "(`load_best_model_at_end`)."
            if metrics.get("metric_for_best_model")
            else "Single-epoch smoke baseline."
        )
    )

    val_f1 = metrics.get("validation", {}).get("eval_f1")
    test_f1 = metrics.get("test", {}).get("eval_f1")
    val_acc = metrics.get("validation", {}).get("eval_accuracy")
    test_acc = metrics.get("test", {}).get("eval_accuracy")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test macro-F1", f"{test_f1:.1%}" if test_f1 is not None else "—")
    m2.metric("Test accuracy", f"{test_acc:.1%}" if test_acc is not None else "—")
    m3.metric("Val macro-F1", f"{val_f1:.1%}" if val_f1 is not None else "—")
    m4.metric("Val accuracy", f"{val_acc:.1%}" if val_acc is not None else "—")
    st.caption(
        "**macro-F1** is the primary metric (averages all three classes equally, so the "
        "dominant *neutral* class can't hide weak *negative*/*positive* performance); "
        "accuracy is secondary."
    )

    if not has_model:
        st.info(
            "No saved checkpoint yet — metrics above are from the documented notebook run. "
            "Use **Train / refresh model** below to write a checkpoint to disk."
        )

    # ── What was done: inputs → process → outputs ──────────────────────────────
    with st.expander("🧪 What this experiment did (inputs → process → outputs)", expanded=False):
        in_col, out_col = st.columns(2)
        with in_col:
            st.markdown("**Inputs**")
            st.markdown(
                f"- **Base model:** `{metrics.get('model_name', '—')}` (pretrained)\n"
                f"- **Dataset:** `{metrics.get('dataset', '—')}`\n"
                f"- **Splits:** {metrics.get('split_source', 'pre-defined train/val/test')}\n"
                f"- **Max tokens:** {metrics.get('max_length', '—')}\n"
                f"- **Epochs / LR / batch:** {epochs} / {metrics.get('learning_rate', '—')} / "
                f"{metrics.get('per_device_train_batch_size', '—')}"
            )
        with out_col:
            st.markdown("**Outputs**")
            saved_at = str(metrics.get("saved_at", ""))[:19].replace("T", " ")
            try:
                rel_dir = model_dir.relative_to(PROJECT_ROOT)
            except ValueError:
                rel_dir = model_dir
            train_loss = metrics.get("train_loss")
            train_loss_str = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else "—"
            runtime = metrics.get("train_runtime_s")
            runtime_str = f"{runtime:.0f}s" if isinstance(runtime, (int, float)) else "—"
            st.markdown(
                f"- **Saved checkpoint:** `{rel_dir}`\n"
                f"- **Metrics file:** `metrics.json` in that folder\n"
                f"- **Train loss:** {train_loss_str}\n"
                f"- **Train runtime:** {runtime_str} on {str(metrics.get('device', '—')).upper()}\n"
                f"- **Saved at:** {saved_at or '—'} UTC"
            )
        st.markdown("**Process**")
        st.markdown(
            "1. Load PhraseBank → 2. tokenize all splits (`max_length` padding) → "
            "3. fine-tune with Hugging Face `Trainer`, evaluating **accuracy + macro-F1** each "
            "epoch → 4. keep the best validation-F1 checkpoint → 5. report on validation **and** "
            "the held-out **test** split → 6. save model + `metrics.json`."
        )
        st.caption("Mirrors `notebooks/liquidAI_prep.ipynb`; see `docs/news_sentiment_finetuning_plan.md`.")

    # ── Comparison vs the 1-epoch baseline ─────────────────────────────────────
    baseline = PHRASEBANK_BASELINE_METRICS
    if metrics.get("epochs") and metrics.get("epochs") != baseline.get("epochs"):
        with st.expander("📈 Progress vs the 1-epoch baseline", expanded=False):
            comparison = pd.DataFrame(
                [
                    {
                        "run": "1-epoch baseline",
                        "epochs": baseline.get("epochs"),
                        "test macro-F1": baseline.get("test", {}).get("eval_f1"),
                        "test accuracy": baseline.get("test", {}).get("eval_accuracy"),
                        "val accuracy": baseline.get("validation", {}).get("eval_accuracy"),
                    },
                    {
                        "run": f"current ({epochs}-epoch, best val-F1)",
                        "epochs": epochs,
                        "test macro-F1": test_f1,
                        "test accuracy": test_acc,
                        "val accuracy": val_acc,
                    },
                ]
            )
            st.dataframe(
                comparison.style.format(
                    {
                        "test macro-F1": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
                        "test accuracy": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
                        "val accuracy": lambda v: f"{v:.1%}" if pd.notna(v) else "—",
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )
            st.caption(
                "The 1-epoch baseline predates macro-F1 logging (accuracy only); macro-F1 was "
                "added in Iteration 2."
            )

    with st.expander("Raw training metrics (metrics.json)", expanded=False):
        st.json(metrics)

    st.divider()

    # ── Dataset snapshot ──────────────────────────────────────────────────────
    st.markdown("### Dataset — Financial PhraseBank")
    st.caption(
        "Loaded from the script-free Parquet mirror `atrost/financial_phrasebank` "
        "(datasets v5 compatible). Train / validation / test splits are pre-defined."
    )

    with st.expander("ℹ️ What is Financial PhraseBank?", expanded=False):
        st.markdown(
            "**What it is** — ~4,840 English **financial-news sentences**, each labeled "
            "with sentiment *from an investor's view* (would the news move the stock "
            "price?): `negative` / `neutral` / `positive`. Sentence-level, finance-specific.\n\n"
            "**Who & why** — built by Malo, Sinha, Korhonen, Wallenius & Takala "
            "(Aalto University, 2014) as a human-annotated benchmark, because a "
            "sentence's overall sentiment often differs from its individual words "
            "(*\"cost reduction\"* = positive; *\"dividend cut\"* = negative).\n\n"
            "**Schema** — two columns: `sentence` (string) and `label` "
            "(`0=negative`, `1=neutral`, `2=positive`)."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "sentence": [
                        "Pretax profit rose to EUR 0.6 mn from EUR 0.4 mn …",
                        "The total headcount reduction will be 50 persons …",
                        "Investment management and investment advisory …",
                    ],
                    "label": ["positive", "negative", "neutral"],
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.markdown(
            "**How labels were built** — 16 finance-background annotators; each "
            "sentence got 5–8 independent annotations; gold label = **majority vote**. "
            "The `50agree` config we use keeps sentences with ≥50% agreement (most "
            "data, noisiest labels).\n\n"
            "**Read more** — "
            "[detailed notes](https://github.com/armandordorica/sentiment_learn_to_rank_paper/blob/main/docs/financial_phrasebank.md) · "
            "[dataset card](https://huggingface.co/datasets/takala/financial_phrasebank) · "
            "[original paper](https://arxiv.org/abs/1307.5336)"
        )

    try:
        balance, splits = _cached_phrasebank_summary()
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Train", f"{splits.get('train', 0):,}")
        s2.metric("Validation", f"{splits.get('validation', 0):,}")
        s3.metric("Test", f"{splits.get('test', 0):,}")
        s4.metric("Total", f"{sum(splits.values()):,}")

        fig = px.bar(
            balance.sort_values("count"),
            x="count", y="label", orientation="h",
            labels={"label": "Class", "count": "Train rows"},
            title="Class balance (train split)",
        )
        fig.update_traces(hovertemplate="Class: %{y}<br>Count: %{x}<extra></extra>")
        fig.update_layout(hovermode="closest", showlegend=False, height=220)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:
        st.error(f"Could not load PhraseBank: {exc}")

    st.divider()

    # ── RavenPack browser (TRNA substitute) ───────────────────────────────────
    st.markdown("### RavenPack articles — text + sentiment (TRNA substitute)")
    st.caption(
        "RavenPack is our TRNA substitute for **sentiment scores**. Text is usually a **headline**; "
        "~22% of rows also have `event_text` — a **short AAPL-tagged snippet** (avg ~37 characters), "
        "**not** a full news article. For the few headlines with a cached Refinitiv story, the full "
        "wire text appears below the RavenPack snippet."
    )

    with st.form("sentiment_lab_ravenpack_news", clear_on_submit=False):
        rp_cols = st.columns([1, 1, 1, 1])
        rp_ticker = rp_cols[0].text_input(
            "Ticker", value="AAPL", max_chars=16, key="sentiment_lab_rp_ticker"
        ).strip().upper()
        rp_start = rp_cols[1].date_input(
            "Start date",
            value=pd.Timestamp(DEFAULT_LOOKUP_START).date(),
            min_value=pd.Timestamp("1990-01-01").date(),
            max_value=default_live_api_end().date(),
            key="sentiment_lab_rp_start",
        )
        rp_end = rp_cols[2].date_input(
            "End date",
            value=pd.Timestamp(DEFAULT_LOOKUP_END).date(),
            min_value=pd.Timestamp("1990-01-01").date(),
            max_value=default_live_api_end().date(),
            key="sentiment_lab_rp_end",
        )
        rp_max = int(
            rp_cols[3].number_input(
                "Max articles",
                min_value=5,
                max_value=500,
                value=50,
                step=5,
                key="sentiment_lab_rp_max",
            )
        )
        rich_path = NEWS_RAVENPACK_DIR / f"{rp_ticker.lower()}_articles_2003_2014.parquet"
        batch_info = _dashboard_cache_info(rp_ticker) if rp_ticker else None
        if rich_path.exists():
            st.success(f"Rich RavenPack cache found: `{rich_path.relative_to(PROJECT_ROOT)}` (headline + event text).")
        elif batch_info:
            st.info(
                f"Batch cache exists for **{rp_ticker}** but without article text. "
                "Use **Re-pull live** or run `notebooks/fetch_news_articles.ipynb` for full text."
            )
        elif rp_ticker:
            st.info(f"No RavenPack cache for **{rp_ticker}** yet — use **Re-pull live** (WRDS).")

        also_model = st.checkbox(
            "Also score with PhraseBank model (optional)",
            value=False,
            disabled=not has_model,
            key="sentiment_lab_rp_model",
        )
        rp_btn_cols = st.columns(2)
        rp_cache_btn = rp_btn_cols[0].form_submit_button("Load cached", type="primary")
        rp_live_btn = rp_btn_cols[1].form_submit_button("Re-pull live (WRDS)")

    if rp_cache_btn or rp_live_btn:
        if rp_start > rp_end:
            st.error("Start date must be on or before end date.")
        else:
            with st.spinner(f"Loading RavenPack articles for {rp_ticker}…"):
                rp_articles, rp_note = _load_ravenpack_articles_for_display(
                    rp_ticker,
                    to_query_date(rp_start),
                    to_query_date(rp_end),
                    live=rp_live_btn,
                    max_rows=rp_max,
                )
            if rp_articles.empty:
                st.warning(f"No RavenPack rows for **{rp_ticker}** ({rp_note}).")
            else:
                if also_model and has_model:
                    with st.spinner("Scoring with PhraseBank model…"):
                        rp_articles = _enrich_ravenpack_with_model(rp_articles, model_dir)
                st.session_state.sentiment_lab_ravenpack_articles = rp_articles
                st.session_state.sentiment_lab_ravenpack_meta = {
                    "ticker": rp_ticker,
                    "start": str(rp_start),
                    "end": str(rp_end),
                    "source": rp_note,
                }

    if st.session_state.get("sentiment_lab_ravenpack_articles") is not None:
        rp_meta = st.session_state.get("sentiment_lab_ravenpack_meta", {})
        st.caption(
            f"**{rp_meta.get('ticker', '—')}** "
            f"({rp_meta.get('start', '—')} → {rp_meta.get('end', '—')}) · {rp_meta.get('source', '')}"
        )
        _render_ravenpack_articles_browser(st.session_state.sentiment_lab_ravenpack_articles)

    st.divider()

    # ── Inference demo ────────────────────────────────────────────────────────
    st.markdown("### Try it — score a headline")
    default_examples = (
        "The company reported record quarterly profit and raised its dividend.\n"
        "Shares plunged after the firm warned of widening losses and layoffs.\n"
        "The board will meet on Thursday to review the quarterly filing."
    )
    text_in = st.text_area(
        "Enter one sentence per line",
        value=default_examples,
        height=140,
        key="sentiment_lab_input",
    )
    run_pred = st.button("Score sentiment", type="primary", disabled=not has_model)

    if run_pred:
        sentences = [ln.strip() for ln in text_in.splitlines() if ln.strip()]
        if not sentences:
            st.warning("Enter at least one sentence.")
        else:
            try:
                tokenizer, model, device = _cached_sentiment_classifier(str(model_dir))
                preds = predict_sentences(sentences, tokenizer, model, device)
                st.dataframe(preds, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")

    st.divider()

    # ── RavenPack fine-tune (TRNA substitute labels) ──────────────────────────
    st.markdown("### Fine-tune on RavenPack headlines")
    st.caption(
        "Continue from the PhraseBank-trained DistilBERT checkpoint and adapt it to "
        "RavenPack `event_sentiment_score` labels (negative / neutral / positive). "
        "Uses cached rich exports (`data/raw/news/ravenpack/*_articles_*.parquet`) with "
        "a **time-based split**: train ≤2011, validation 2012, test ≥2013."
    )

    rp_export_paths = discover_ravenpack_article_files()
    rp_tickers_available = sorted({
        p.name.split("_articles_")[0].upper() for p in rp_export_paths
    })
    has_ravenpack_model = ravenpack_model_is_saved()
    rp_model_dir = resolve_ravenpack_model_dir()
    rp_metrics = load_ravenpack_metrics(rp_model_dir)

    if not rp_tickers_available:
        st.warning(
            "No RavenPack article exports found. Run `notebooks/fetch_news_articles.ipynb` "
            "to build `{ticker}_articles_2003_2014.parquet` under "
            f"`{NEWS_RAVENPACK_DIR.relative_to(PROJECT_ROOT)}/`."
        )
    else:
        rp_train_ticker = st.selectbox(
            "Ticker to train on",
            options=rp_tickers_available,
            index=0,
            key="sentiment_lab_rp_train_ticker",
        )
        init_from_phrasebank = st.checkbox(
            "Start from PhraseBank checkpoint (recommended)",
            value=has_model,
            disabled=not has_model,
            help=(
                "Loads `phrasebank_distilbert_best/` weights before RavenPack fine-tuning. "
                "If unchecked, trains from `distilbert-base-uncased`."
            ),
            key="sentiment_lab_rp_init_phrasebank",
        )
        if not has_model:
            st.caption(
                "PhraseBank checkpoint not found — RavenPack training will start from "
                "`distilbert-base-uncased` unless you train PhraseBank first."
            )

        try:
            rp_labeled = load_ravenpack_labeled_frame([rp_train_ticker])
            rp_balance = ravenpack_class_balance(rp_labeled)
            rp_splits = ravenpack_split_summary(rp_labeled)
            c1, c2, c3 = st.columns(3)
            c1.metric("Labeled headlines", f"{len(rp_labeled):,}")
            c2.metric("Train rows", f"{int(rp_splits.loc[rp_splits['split'] == 'train', 'rows'].iloc[0]):,}")
            c3.metric("Test rows", f"{int(rp_splits.loc[rp_splits['split'] == 'test', 'rows'].iloc[0]):,}")
            st.dataframe(rp_splits, hide_index=True, use_container_width=True)
            fig_rp = px.bar(
                rp_balance.sort_values("count"),
                x="count",
                y="label",
                orientation="h",
                labels={"label": "Class", "count": "Rows"},
                title=f"RavenPack label balance ({rp_train_ticker})",
            )
            fig_rp.update_traces(hovertemplate="Class: %{y}<br>Count: %{x}<extra></extra>")
            fig_rp.update_layout(hovermode="closest", showlegend=False, height=220)
            st.plotly_chart(fig_rp, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not load RavenPack training data: {exc}")
            rp_labeled = None

        if has_ravenpack_model and rp_metrics:
            rp_test_f1 = rp_metrics.get("test", {}).get("eval_f1")
            rp_test_acc = rp_metrics.get("test", {}).get("eval_accuracy")
            st.caption(
                f"Saved RavenPack model: `{rp_model_dir.relative_to(PROJECT_ROOT)}` · "
                f"test macro-F1 **{rp_test_f1:.1%}** · accuracy **{rp_test_acc:.1%}**"
                if rp_test_f1 is not None and rp_test_acc is not None
                else f"Saved RavenPack model: `{rp_model_dir.relative_to(PROJECT_ROOT)}`"
            )

        if st.button(
            f"Fine-tune on RavenPack ({DEFAULT_RAVENPACK_TRAIN_EPOCHS_UI} epochs)",
            key="sentiment_lab_train_ravenpack",
            disabled=rp_labeled is None,
        ):
            with st.spinner(
                f"Fine-tuning DistilBERT on RavenPack ({rp_train_ticker})… "
                "this may take several minutes."
            ):
                try:
                    new_rp_metrics = train_ravenpack(
                        tickers=[rp_train_ticker],
                        init_from_phrasebank=init_from_phrasebank and has_model,
                        num_train_epochs=DEFAULT_RAVENPACK_TRAIN_EPOCHS_UI,
                    )
                    _cached_sentiment_classifier.clear()
                    test_f1 = new_rp_metrics["test"].get("eval_f1")
                    test_acc = new_rp_metrics["test"].get("eval_accuracy")
                    st.success(
                        f"Done — RavenPack test macro-F1 {test_f1:.1%}, accuracy {test_acc:.1%}. "
                        f"Saved to `{DEFAULT_RAVENPACK_MODEL_DIR.relative_to(PROJECT_ROOT)}`."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"RavenPack training failed: {exc}")

    st.divider()

    # ── Train / refresh (PhraseBank) ──────────────────────────────────────────
    with st.expander(
        f"Train / refresh PhraseBank model (~{DEFAULT_TRAIN_EPOCHS_UI} epochs on Apple Silicon)",
        expanded=not has_model,
    ):
        st.caption(
            f"Runs the Iteration-2 workflow from the notebook: {DEFAULT_TRAIN_EPOCHS_UI} epochs, "
            "macro-F1 + accuracy, `load_best_model_at_end` on validation F1. Saves to "
            f"`{DEFAULT_MODEL_DIR.relative_to(PROJECT_ROOT)}`."
        )
        if st.button("Train PhraseBank baseline now", key="sentiment_lab_train"):
            with st.spinner("Training DistilBERT on Financial PhraseBank…"):
                try:
                    new_metrics = train_baseline()
                    _cached_sentiment_classifier.clear()
                    _cached_phrasebank_summary.clear()
                    test_f1 = new_metrics["test"].get("eval_f1")
                    test_acc = new_metrics["test"].get("eval_accuracy")
                    st.success(
                        f"Done — test macro-F1 {test_f1:.1%}, accuracy {test_acc:.1%}. "
                        "Scroll up to score headlines."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Training failed: {exc}")


def render_batch_pipeline_tab() -> bool:  # noqa: C901 – intentionally long UI function
    # Force Refinitiv off by default so stale session state never auto-enables it
    if "batch_use_refinitiv" not in st.session_state:
        st.session_state["batch_use_refinitiv"] = True

    st.header("Top-1,000 Batch Pipeline")
    st.caption(
        "Pull and cache WRDS/CRSP, Yahoo, and RavenPack data for every ticker in the "
        "CRSP top-volume universe. Each ticker is cached immediately; reruns skip "
        "completed tickers automatically."
    )

    # Load cached manifests once per render (fast path when nothing changed).
    manifests_df = _get_manifests_df()

    # ── Live status banner ────────────────────────────────────────────────────
    pid = _batch_pid_running()
    batch_status = _read_batch_status()
    is_running = pid is not None

    if is_running:
        current  = batch_status.get("current_ticker", "starting…")
        rank     = batch_status.get("current_rank",   "—")
        step     = batch_status.get("current_step",   "")
        done     = batch_status.get("done", 0) or 0
        total    = batch_status.get("total") or 0
        # Compute real elapsed from ticker_started_at rather than the stale
        # elapsed_s field (which is frozen at the time status was last written).
        ticker_started_at = batch_status.get("ticker_started_at")
        if ticker_started_at:
            try:
                from datetime import datetime, timezone as _tz
                started_dt = datetime.fromisoformat(ticker_started_at)
                real_elapsed = round((datetime.now(_tz.utc) - started_dt).total_seconds())
                elapsed_str = f"  |  **{real_elapsed}s** on this ticker"
            except Exception:
                elapsed_str = ""
        else:
            elapsed_str = ""
        step_str    = f"  —  *{step}*" if step else ""

        # Banner + stop button in the same row
        banner_col, stop_col, refresh_col = st.columns([6, 1, 1])
        with banner_col:
            st.success(
                f"**Batch running** (PID {pid}){step_str}\n\n"
                f"Now on: **{current}** [rank {rank}]{elapsed_str}  "
                f"|  **{done}** of **{total}** done"
            )
        with stop_col:
            if st.button("🛑 Stop", type="secondary", use_container_width=True):
                try:
                    import signal as _signal
                    os.kill(pid, _signal.SIGKILL)
                    time.sleep(0.3)
                except OSError:
                    pass
                BATCH_PID_FILE.unlink(missing_ok=True)
                st.warning(f"Killed PID {pid}.")
                time.sleep(0.8)
                st.rerun()
        with refresh_col:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()

        # Progress bar driven purely by batch_status.json
        if total > 0:
            frac = min(1.0, done / total)
            st.progress(frac, text=f"{done:,} / {total:,} tickers  ({frac*100:.1f}%)")
        else:
            st.progress(0.0, text="Starting…")

        # Live log — collapsed so it doesn't push the cache snapshot off-screen.
        log_path = TOP1K_OUTPUT_DIR / "batch_runner.log"
        with st.expander("📜 Live batch log", expanded=False):
            if log_path.exists():
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    st.code("\n".join(lines[-40:]), language=None)
                except Exception:
                    st.caption("Could not read log file.")
            else:
                st.info("Log file not yet created — the batch process is still starting up.")

    elif batch_status:
        final_status = batch_status.get("status", "unknown")
        done  = batch_status.get("done", 0) or 0
        total = batch_status.get("total", 0) or 0
        updated = batch_status.get("updated_at", "")
        if final_status == "complete":
            st.success(f"Last batch **completed** — {done:,} tickers — {updated}")
            if total > 0:
                st.progress(1.0, text=f"{done:,} / {total:,} tickers")
        elif final_status == "stopped_failures":
            st.error(f"Last batch **stopped** (too many consecutive failures) — {batch_status.get('error', '')} — {updated}")
            if total > 0:
                st.progress(min(1.0, done / total), text=f"{done:,} / {total:,} tickers before stop")
        else:
            st.info(f"Batch status: **{final_status}** — {updated}")
    else:
        st.info("No batch has been run yet. Use the configuration below to launch one.")

    # ── Cache snapshot (always visible — what's on disk right now) ───────────
    _render_cache_snapshot(manifests_df)
    st.divider()
    _render_cash_merger_section(manifests_df)
    st.divider()
    _render_fail_reasons_by_provider(manifests_df)
    st.divider()
    _render_delisting_section(manifests_df)

    # ── Refresh / auto-refresh controls ─────────────────────────────────────
    ctrl_cols = st.columns([1, 5])
    if not is_running:
        if ctrl_cols[0].button("🔄  Refresh now"):
            st.rerun()
    auto_refresh = ctrl_cols[1].checkbox(
        "Auto-refresh every 5 s", value=is_running, key="batch_auto_refresh"
    )

    st.divider()

    # ── Per-ticker status (always visible — full cached universe) ─────────────
    st.subheader("Per-Ticker Status")

    # Prepend a live row for the ticker currently being processed so it shows
    # immediately without waiting for the manifest to be written.
    if is_running and batch_status:
        live_rank   = batch_status.get("current_rank")
        live_ticker = batch_status.get("current_ticker", "")
        live_step   = batch_status.get("current_step", "starting…")
        psf         = batch_status.get("providers_so_far", {})  # partial provider results
        if live_ticker and live_rank:
            # Compute real elapsed
            try:
                from datetime import datetime, timezone as _tz2
                _ts = batch_status.get("ticker_started_at", "")
                _live_elapsed = round((datetime.now(_tz2.utc) - datetime.fromisoformat(_ts)).total_seconds()) if _ts else 0
            except Exception:
                _live_elapsed = 0

            def _psf_status(p):
                if p not in psf:
                    return "…"        # not started yet
                return psf[p].get("status", "…")

            def _psf_rows(p):
                if p not in psf:
                    return ""
                return psf[p].get("rows", 0)

            live_row = pd.DataFrame([{
                "rank": live_rank, "ticker": live_ticker, "company": "⚡ in progress",
                "status": f"⚡ {live_step}",
                "wrds_status":      _psf_status("wrds"),
                "wrds_rows":        _psf_rows("wrds"),
                "yahoo_status":     _psf_status("yahoo"),
                "yahoo_rows":       _psf_rows("yahoo"),
                "ravenpack_status": _psf_status("ravenpack"),
                "ravenpack_rows":   _psf_rows("ravenpack"),
                "refinitiv_status": _psf_status("refinitiv"),
                "refinitiv_rows":   _psf_rows("refinitiv"),
                "wrds_fail_reason": "",
                "yahoo_fail_reason": "",
                "ravenpack_fail_reason": "",
                "refinitiv_fail_reason": "",
                "ok": sum(1 for p in psf if psf[p].get("status") == "ok"),
                "fail": sum(1 for p in psf if psf[p].get("status") in ("failed", "timeout")),
                "created_at": f"{_live_elapsed}s elapsed",
            }])
            # Only prepend if this ticker isn't already in the manifests table
            if manifests_df.empty or live_ticker not in manifests_df["ticker"].values:
                manifests_df = pd.concat([live_row, manifests_df], ignore_index=True)

    if not manifests_df.empty:
        # Filter controls
        filter_cols = st.columns([2, 1, 1, 1])
        status_filter = filter_cols[0].multiselect(
            "Filter by status",
            options=["complete", "partial", "failed", "error"],
            default=[],
            key="batch_status_filter",
            placeholder="All statuses",
        )
        show_ticker = filter_cols[1].text_input("Filter ticker", value="", key="batch_ticker_filter").strip().upper()
        show_only_failed_providers = filter_cols[2].checkbox("Only show provider failures", key="batch_prov_fail_filter")

        view_df = manifests_df.copy()
        for col in ("wrds_fail_reason", "yahoo_fail_reason", "ravenpack_fail_reason", "refinitiv_fail_reason"):
            if col not in view_df.columns:
                view_df[col] = ""
        if status_filter:
            view_df = view_df[view_df["status"].isin(status_filter)]
        if show_ticker:
            view_df = view_df[view_df["ticker"].str.upper().str.contains(show_ticker)]
        if show_only_failed_providers:
            view_df = view_df[view_df["fail"] > 0]

        # Build display table — each provider column shows status + fail_reason inline
        keep = [
            "rank", "ticker", "company", "status",
            "wrds_status", "wrds_rows", "wrds_fail_reason",
            "yahoo_status", "yahoo_rows", "yahoo_fail_reason",
            "ravenpack_status", "ravenpack_rows", "ravenpack_fail_reason",
            "refinitiv_status", "refinitiv_rows", "refinitiv_fail_reason",
            "created_at",
        ]
        if "permno" in view_df.columns:
            keep.insert(3, "permno")
        display = view_df[keep].copy()
        delist_lookup = _build_delisting_lookup()
        display["CRSP"] = (
            display["permno"].apply(lambda p: _delisting_cell(p, delist_lookup))
            if "permno" in display.columns
            else "…"
        )
        exit_lookup = _build_cash_merger_lookup()
        display["Exit"] = (
            display["permno"].apply(lambda p: _exit_cell(p, exit_lookup))
            if "permno" in display.columns
            else "—"
        )
        display["Status"] = display["status"].apply(lambda s: f"{_status_color(s)} {s}")
        display["WRDS"] = display.apply(
            lambda r: _provider_status_cell(r["wrds_status"], r["wrds_rows"], r["wrds_fail_reason"]), axis=1,
        )
        display["Yahoo"] = display.apply(
            lambda r: _provider_status_cell(r["yahoo_status"], r["yahoo_rows"], r["yahoo_fail_reason"]), axis=1,
        )
        display["RavenPack"] = display.apply(
            lambda r: _provider_status_cell(
                r["ravenpack_status"], r["ravenpack_rows"], r["ravenpack_fail_reason"],
            ), axis=1,
        )
        display["Refinitiv"] = display.apply(
            lambda r: _provider_status_cell(
                r["refinitiv_status"], r["refinitiv_rows"], r["refinitiv_fail_reason"],
            ), axis=1,
        )
        display = display[[
            "rank", "ticker", "company", "Status",
            "WRDS", "Exit", "Yahoo", "RavenPack", "Refinitiv", "CRSP", "created_at",
        ]]
        display.columns = [
            "Rank", "Ticker", "Company", "Status",
            "WRDS", "Exit", "Yahoo", "RavenPack", "Refinitiv", "CRSP delisting", "Cached at",
        ]

        st.dataframe(
            display,
            use_container_width=True,
            height=480,
            column_config={
                "Exit": st.column_config.TextColumn(
                    "Exit",
                    width="medium",
                    help=(
                        "Cash-merger exit (CRSP dlstcd 232/233): icon + exit return + delisting price.  "
                        "✅ from CRSP dlret · 🟡 estimated from SDC deal price · "
                        "⬜ fallback (last price, return = 0) · — not a cash merger / not checked"
                    ),
                ),
            },
        )
        st.caption(
            f"Showing {len(view_df):,} of {len(manifests_df):,} cached tickers. "
            "Provider cells: ✅ ok + row count · ❌/⚠️ fail_reason code. "
            "Exit (cash mergers): icon + exit return + delisting price — "
            "✅ CRSP dlret · 🟡 SDC deal price · ⬜ fallback · — n/a. "
            "CRSP delisting: 🟢 active · ⛔ code + reason + delisting return (dlret); "
            "`…` = not looked up yet (use the Delisting reasons section to fetch)."
        )

        # ── Ticker detail expander ────────────────────────────────────────────
        st.subheader("Ticker Detail")
        ticker_options = sorted(manifests_df["ticker"].dropna().unique().tolist())
        selected_detail = st.selectbox("Select ticker for detail", options=[""] + ticker_options, key="batch_detail_ticker")
        if selected_detail:
            row = manifests_df[manifests_df["ticker"] == selected_detail]
            if not row.empty:
                r = row.iloc[0]
                rank_dir = None
                slug = "".join(ch if ch.isalnum() else "_" for ch in selected_detail.upper().strip())
                for d in TOP1K_BY_TICKER_DIR.glob(f"rank_{int(r['rank']):04d}_{slug}"):
                    rank_dir = d
                    break

                d_cols = st.columns(4)
                d_cols[0].metric("Rank",    r["rank"])
                d_cols[1].metric("Ticker",  r["ticker"])
                d_cols[2].metric("Status",  r["status"])
                d_cols[3].metric("OK / Fail providers", f"{int(r['ok'])} / {int(r['fail'])}")
                st.caption(f"Company: {r['company']}   |   Cached: {r['created_at']}")

                # Per-provider detail table
                prov_rows = []
                for pname in ["wrds", "yahoo", "ravenpack", "refinitiv"]:
                    reason = r.get(f"{pname}_fail_reason") or ""
                    prov_rows.append({
                        "Provider":  pname,
                        "Status":    r.get(f"{pname}_status") or "—",
                        "Rows":      int(r.get(f"{pname}_rows") or 0),
                        "Reason":    reason,
                        "Reason (label)": reason_label(reason) if reason else "",
                    })
                st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)

                # CRSP delisting detail for this ticker
                st.markdown("**CRSP delisting (crsp.msedelist)**")
                detail_lookup = _build_delisting_lookup()
                rec = detail_lookup.get(int(r["permno"])) if pd.notna(r.get("permno")) else None
                if rec is None:
                    st.caption(
                        "Not looked up yet — use the **Delisting reasons (CRSP)** section "
                        "above to fetch this ticker's delisting code."
                    )
                elif not bool(rec.get("delisted")):
                    st.success("🟢 Still active — no CRSP delisting record (dlstcd 100).")
                else:
                    dl_cols = st.columns(4)
                    dl_cols[0].metric("Delisting code", str(rec.get("dlstcd") or "—"))
                    dlret = rec.get("dlret")
                    dl_cols[1].metric(
                        "Delisting return",
                        f"{float(dlret):+.2%}" if pd.notna(dlret) else "—",
                    )
                    dlstdt = rec.get("dlstdt")
                    dl_cols[2].metric(
                        "Delisting date",
                        pd.Timestamp(dlstdt).strftime("%Y-%m-%d") if pd.notna(dlstdt) else "—",
                    )
                    nwperm = rec.get("nwperm")
                    dl_cols[3].metric(
                        "Successor PERMNO",
                        str(int(nwperm)) if pd.notna(nwperm) else "—",
                    )
                    st.caption(
                        f"**{rec.get('delisting_category', '')}** — {rec.get('delisting_label', '')}"
                    )

                # Show output files
                if rank_dir and (rank_dir / "manifest.json").exists():
                    try:
                        full_manifest = json.loads((rank_dir / "manifest.json").read_text(encoding="utf-8"))
                        outputs = full_manifest.get("outputs", {})
                        if outputs:
                            st.markdown("**Cached files**")
                            for key, path in outputs.items():
                                st.code(path)
                    except Exception:
                        pass
    else:
        st.info("No ticker data cached yet.")

    st.divider()

    # ── Configuration form ────────────────────────────────────────────────────
    with st.expander("⚙️  Configuration", expanded=not is_running):
        with st.form("batch_config_form"):
            date_cols = st.columns(2)
            start_date = date_cols[0].text_input("Start date", value="2003-01-01", key="batch_start")
            end_date   = date_cols[1].text_input("End date",   value="2014-12-31", key="batch_end")

            rank_cols = st.columns(3)
            start_rank = rank_cols[0].number_input("Start rank", min_value=1, max_value=999, value=1, step=1, key="batch_start_rank")
            max_tickers_input = rank_cols[1].text_input(
                "Max tickers (blank = all)", value="", key="batch_max_tickers",
                help="Leave blank to run all tickers from start rank onward. Enter a number for a smoke test."
            )
            sleep_sec = rank_cols[2].number_input("Sleep between tickers (s)", min_value=0.0, max_value=10.0, value=0.25, step=0.05, key="batch_sleep")

            timeout_cols = st.columns(3)
            stop_after = timeout_cols[0].number_input(
                "Stop after N consecutive failures", min_value=1, max_value=200, value=25, step=1, key="batch_stop_after"
            )
            provider_timeout = timeout_cols[1].number_input(
                "Provider timeout (s)", min_value=30, max_value=1800, value=300, step=30,
                key="batch_provider_timeout",
                help="Max seconds to wait for a full provider (safety net). Default 300s = 5 min.",
            )
            year_timeout = timeout_cols[2].number_input(
                "RavenPack per-year timeout (s)", min_value=10, max_value=300, value=90, step=10,
                key="batch_year_timeout",
                help="Server-side timeout per yearly RavenPack query. Timed-out years are skipped; data from other years is still saved. Default 90s.",
            )

            st.markdown("**Providers**")
            prov_cols = st.columns(4)
            use_wrds       = prov_cols[0].checkbox("WRDS/CRSP",      value=wrds_credentials_available(), key="batch_use_wrds")
            use_yahoo      = prov_cols[1].checkbox("Yahoo Finance",   value=True,                         key="batch_use_yahoo")
            use_ravenpack  = prov_cols[2].checkbox("RavenPack",       value=wrds_credentials_available(), key="batch_use_ravenpack")
            use_refinitiv  = prov_cols[3].checkbox(
                "Refinitiv",
                value=True,
                key="batch_use_refinitiv",
            )

            opt_cols = st.columns(4)
            force_rerun       = opt_cols[0].checkbox("Force rerun (ignore cache)",    value=False, key="batch_force_rerun")
            rerun_failed      = opt_cols[1].checkbox("Retry failed tickers",          value=True,  key="batch_rerun_failed")
            rerun_partial     = opt_cols[2].checkbox("Smart retry partial tickers",   value=True,  key="batch_rerun_partial",
                                                     help="Re-fetch only the providers that failed; keeps already-ok data intact.")
            combined_parquets = opt_cols[3].checkbox("Write combined parquets",       value=True,  key="batch_combined")

            submitted = st.form_submit_button(
                "🚀  Launch Batch" if not is_running else "⚠️  Batch already running",
                type="primary",
                disabled=is_running,
            )

        if submitted and not is_running:
            max_tickers = int(max_tickers_input) if max_tickers_input.strip().isdigit() else None
            _launch_batch(
                start=start_date, end=end_date,
                start_rank=int(start_rank), max_tickers=max_tickers,
                force_rerun=force_rerun, rerun_failed=rerun_failed,
                rerun_partial=rerun_partial,
                sleep_sec=float(sleep_sec), stop_after=int(stop_after),
                provider_timeout=float(provider_timeout), year_timeout=int(year_timeout),
                use_wrds=use_wrds, use_yahoo=use_yahoo,
                use_ravenpack=use_ravenpack, use_refinitiv=use_refinitiv,
                combined_parquets=combined_parquets,
            )
            time.sleep(1.5)
            st.rerun()

    # ── Full log (collapsed) ──────────────────────────────────────────────────
    log_path = TOP1K_OUTPUT_DIR / "batch_runner.log"
    if log_path.exists() and not is_running:
        with st.expander("📄  Full batch log"):
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                st.code("\n".join(lines[-100:]), language=None)
            except Exception as exc:
                st.warning(f"Could not read log: {exc}")

    # ── Storage summary ───────────────────────────────────────────────────────
    with st.expander("📁  Storage — where files are saved", expanded=manifests_df.empty and not is_running):
        st.markdown(f"**Root directory:** `{TOP1K_OUTPUT_DIR.relative_to(PROJECT_ROOT)}/`")
        st.code(
            "by_ticker/\n"
            "  rank_0001_C/\n"
            "    wrds_prices.parquet       ← CRSP daily prices\n"
            "    wrds_names.parquet        ← CRSP name history\n"
            "    yahoo_prices.parquet      ← Yahoo Finance daily prices\n"
            "    ravenpack_articles.parquet← RavenPack sentiment articles\n"
            "    refinitiv_prices.parquet  ← Refinitiv daily prices (if enabled)\n"
            "    manifest.json             ← row counts, status, file paths\n"
            "    provider_status.parquet   ← per-provider ok/fail summary\n"
            "  rank_0002_BAC/\n"
            "  rank_0003_MSFT/\n"
            "  ...\n"
            "combined/\n"
            "  wrds_prices.parquet         ← all tickers merged (written at end)\n"
            "  yahoo_prices.parquet\n"
            "  ravenpack_articles.parquet\n"
            "batch_progress.csv            ← one row per ticker processed\n"
            "batch_status.json             ← current run state\n"
            "batch_runner.log              ← full stdout log\n",
            language=None,
        )
        # Live storage stats
        if TOP1K_BY_TICKER_DIR.exists():
            ticker_dirs = list(TOP1K_BY_TICKER_DIR.glob("rank_*"))
            total_bytes = sum(
                f.stat().st_size for d in ticker_dirs for f in d.rglob("*") if f.is_file()
            )
            total_mb = total_bytes / 1_048_576
            s_cols = st.columns(3)
            s_cols[0].metric("Ticker folders", f"{len(ticker_dirs):,}")
            s_cols[1].metric("Total size on disk", f"{total_mb:.1f} MB" if total_mb < 1024 else f"{total_mb/1024:.2f} GB")
            s_cols[2].metric("Avg per ticker", f"{total_mb/len(ticker_dirs):.1f} MB" if ticker_dirs else "—")

    # ── Combined parquets ─────────────────────────────────────────────────────
    with st.expander("📦  Combined parquets"):
        st.markdown(
            "Once the batch is complete (or partially done), click below to merge "
            "all per-ticker parquets into provider-level files under `data/raw/data_explorer_top1k/combined/`."
        )
        if st.button("Write combined parquets now", key="batch_write_combined"):
            combined_keys = ["wrds_prices", "wrds_names", "yahoo_prices", "ravenpack_articles",
                             "refinitiv_prices", "refinitiv_news"]
            TOP1K_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
            written = []
            for key in combined_keys:
                filename = f"{key}.parquet"
                frames = []
                for ticker_dir in sorted(TOP1K_BY_TICKER_DIR.glob("rank_*")):
                    p = ticker_dir / filename
                    if p.exists():
                        try:
                            frames.append(pd.read_parquet(p))
                        except Exception:
                            pass
                if frames:
                    out = TOP1K_COMBINED_DIR / filename
                    pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
                    written.append(f"{filename}  ({len(frames)} tickers, {sum(len(f) for f in frames):,} rows)")
            if written:
                st.success("Written:\n" + "\n".join(f"- `{w}`" for w in written))
            else:
                st.info("No ticker parquets found yet.")

    # ── Auto-refresh trigger ──────────────────────────────────────────────────
    # IMPORTANT: do NOT call st.rerun() here. This function runs before the
    # Sentiment Lab and Paper Validation tabs in Streamlit's single top-to-bottom
    # script pass, so an early rerun would abort the run mid-way and leave those
    # later tabs blank. Instead, signal the caller to schedule the refresh only
    # after every tab has finished rendering.
    return bool(auto_refresh and is_running)


# ─────────────────────────────────────────────────────────────────────────────

st.info(
    "The **Data Explorer** tab uses one ticker/date-range form for prices, Refinitiv news, and RavenPack sentiment. "
    "The **PhraseBank HF Baseline** tab documents the Hugging Face DistilBERT benchmark "
    "(Financial PhraseBank training data, metrics, probability chart). "
    "The **Sentiment Lab** tab hosts interactive fine-tuning and inference from `notebooks/liquidAI_prep.ipynb`. "
    "The **Paper Validation** tab uses bundled 2003-2014 CSVs from `app_data/`."
)

tab_dashboard, tab_batch, tab_phrasebank_baseline, tab_sentiment, tab_validation = st.tabs([
    "Data Explorer",
    "Batch Pipeline (Top-1K)",
    "PhraseBank HF Baseline",
    "Sentiment Lab",
    "Paper Validation (2003-2014)",
])

with tab_dashboard:
    render_multi_api_dashboard_tab()

with tab_batch:
    _batch_auto_refresh = render_batch_pipeline_tab()

with tab_phrasebank_baseline:
    render_phrasebank_hf_baseline_tab()

with tab_sentiment:
    render_sentiment_lab_tab()

with tab_validation:
    universe = load_bundled_csv(DEFAULT_UNIVERSE_PATHS)
    if universe is None:
        st.info(
            "Bundled validation CSVs were not found in `app_data/`. "
            "Generate them locally with `python scripts/build_crsp_market_universe.py`."
        )
    else:
        universe = prepare_universe(universe)
        top20 = universe.head(20).copy()

        st.subheader("Candidate Universe Checks")
        metrics = st.columns(4)
        metrics[0].metric("Rows", f"{len(universe):,}")
        metrics[1].metric("Unique PERMNOs", f"{universe['permno'].nunique():,}")
        metrics[2].metric("Top 20 Rows", f"{len(top20):,}")
        metrics[3].metric("Date Range", "2003-2014")

        st.dataframe(validation_summary(universe), use_container_width=True)

        st.subheader("Top 20 By Average Daily Share Volume")
        st.plotly_chart(make_top20_bar_chart(top20), use_container_width=True)

        display_cols = [
            "volume_rank",
            "permno",
            "ticker",
            "comnam",
            "trading_days",
            "avg_volume_millions",
            "avg_dollar_volume_billions",
            "first_trade_date",
            "last_trade_date",
        ]
        st.dataframe(top20[display_cols], use_container_width=True)

        st.subheader("Top 20 Trading Volume Over Time")
        monthly_volume = load_bundled_csv(DEFAULT_MONTHLY_VOLUME_PATHS)
        if monthly_volume is None:
            st.info(
                "Bundled monthly volume data is unavailable. "
                "The validation notebook shows how to query and aggregate the top-20 CRSP volume series."
            )
        else:
            try:
                monthly_volume = prepare_monthly_volume(monthly_volume, top20)
                st.plotly_chart(make_monthly_volume_chart(monthly_volume), use_container_width=True)
            except ValueError as exc:
                st.error(str(exc))

        st.subheader("Top 20 Monthly Open, Close, And Average Price")
        monthly_prices = load_bundled_csv(DEFAULT_MONTHLY_PRICE_PATHS)
        if monthly_prices is None:
            st.info(
                "Monthly price data is unavailable. Generate it locally with "
                "`python scripts/export_top20_monthly_prices.py`."
            )
        else:
            try:
                monthly_prices = prepare_monthly_prices(monthly_prices)
                ticker_options = top20["ticker"].dropna().tolist()
                selected_ticker = st.selectbox("Select a top-20 ticker", ticker_options)
                st.plotly_chart(
                    make_monthly_price_chart(monthly_prices, selected_ticker),
                    use_container_width=True,
                )
            except ValueError as exc:
                st.error(str(exc))


# ── Deferred batch auto-refresh ───────────────────────────────────────────────
# Run the 5-second auto-refresh ONLY after every tab above has rendered. If this
# lived inside the Batch tab it would st.rerun() before the Sentiment Lab /
# Paper Validation tabs got their turn, blanking them out while a batch runs.
if globals().get("_batch_auto_refresh"):
    time.sleep(5)
    st.rerun()
