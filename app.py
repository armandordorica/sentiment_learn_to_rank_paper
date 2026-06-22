"""Streamlit app for CRSP universe validation charts."""

from __future__ import annotations

import json
import os
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
    """Overlay close prices from multiple providers on one chart."""
    parts: list[pd.DataFrame] = []
    for provider, frame in price_frames.items():
        if frame.empty:
            continue
        part = frame[["date", "close_price"]].copy()
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
        title=f"Provider Close Price Comparison For {ticker.upper()}",
        labels={"date": "Date", "close_price": "Close price, USD", "provider": "Provider"},
        hover_data={"date": "|%Y-%m-%d", "close_price": ":,.2f"},
    )
    fig.update_traces(mode="lines+markers", marker={"size": 4})
    fig.update_layout(height=550, hovermode="closest")
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
        return "No rows"
    if status == "unavailable":
        return "Unavailable"
    if status == "skipped":
        return "Skipped"
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


def render_refinitiv_story_body(story_id: str, headline: str | None = None, *, close_key: str) -> None:
    """Render the full Refinitiv story text for one storyId."""
    st.markdown("---")
    st.markdown(f"**{headline or 'Selected headline'}**")
    with st.spinner("Loading full Refinitiv story..."):
        try:
            story_text = load_refinitiv_story_text(story_id)
        except Exception as exc:
            st.error(f"Could not load story: {exc}")
            return

    st.text_area(
        "Full story",
        value=story_text,
        height=420,
        disabled=True,
        label_visibility="collapsed",
    )
    if st.button("Close story", key=close_key):
        st.session_state.pop("refinitiv_open_story_id", None)
        st.session_state.pop("refinitiv_open_story_headline", None)
        st.query_params.pop("refinitiv_story", None)
        st.query_params.pop("news_date", None)
        st.rerun()


def render_refinitiv_news_coverage_section(
    news_df: pd.DataFrame,
    daily_counts: pd.DataFrame,
    news_summary: dict[str, object] | None,
    ticker: str,
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
    )


def render_refinitiv_news_headlines(
    news_df: pd.DataFrame,
    *,
    table_key_suffix: str = "",
    show_section_title: bool = True,
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
            st.session_state["refinitiv_open_story_id"] = str(row["storyId"])
            st.session_state["refinitiv_open_story_headline"] = str(row["headline"])
            st.rerun()
        source_code = row.get("sourceCode", "")
        cols[3].write("" if pd.isna(source_code) else str(source_code))

    story_id = st.session_state.get("refinitiv_open_story_id") or st.query_params.get("refinitiv_story")
    if story_id:
        match = display_df[display_df["storyId"].astype(str) == str(story_id)]
        headline = st.session_state.get("refinitiv_open_story_headline")
        if headline is None and not match.empty:
            headline = str(match.iloc[0]["headline"])
        render_refinitiv_story_body(
            str(story_id),
            str(headline) if headline is not None else None,
            close_key=f"close_story_{table_key_suffix}_{story_id}",
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

    quick_date_cols = st.columns(3)
    if quick_date_cols[0].button("Last 7 days"):
        st.session_state.api_start_date = (default_live_api_end() - pd.Timedelta(days=7)).date()
        st.session_state.api_end_date = default_live_api_end().date()
    if quick_date_cols[1].button("Last 30 days"):
        st.session_state.api_start_date = default_live_api_start(30).date()
        st.session_state.api_end_date = default_live_api_end().date()
    if quick_date_cols[2].button("Paper window"):
        st.session_state.api_start_date = pd.Timestamp(DEFAULT_LOOKUP_START).date()
        st.session_state.api_end_date = pd.Timestamp(DEFAULT_LOOKUP_END).date()

    if "api_test_ticker" not in st.session_state:
        st.session_state.api_test_ticker = "AAPL"
    if "api_start_date" not in st.session_state:
        st.session_state.api_start_date = default_live_api_start(30).date()
    if "api_end_date" not in st.session_state:
        st.session_state.api_end_date = default_live_api_end().date()

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
            max_value=default_live_api_end().date(),
        )
        end_date = control_cols[2].date_input(
            "End date",
            key="api_end_date",
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
            "headline": True,
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

    if isinstance(news_df, pd.DataFrame) and not news_df.empty:
        if isinstance(daily_counts, pd.DataFrame) and not daily_counts.empty:
            render_refinitiv_news_coverage_section(
                news_df,
                daily_counts,
                news_summary if isinstance(news_summary, dict) else None,
                ticker,
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
                    st.session_state.dashboard_news_story_id = str(row["storyId"])
                    st.session_state.dashboard_news_story_headline = str(row.get("headline", "Selected headline"))

        story_id = st.session_state.get("dashboard_news_story_id")
        if story_id:
            headline = st.session_state.get("dashboard_news_story_headline", "Selected headline")
            st.markdown("---")
            st.markdown(f"**{headline}**")
            with st.spinner("Loading full Refinitiv story..."):
                try:
                    story_text = load_refinitiv_story_text(str(story_id))
                except Exception as exc:
                    st.error(f"Could not load story: {exc}")
                else:
                    st.text_area(
                        "Full story",
                        value=story_text,
                        height=420,
                        disabled=True,
                        label_visibility="collapsed",
                        key="dashboard_news_story_text",
                    )
            if st.button("Close story", key="dashboard_close_news_story"):
                st.session_state.pop("dashboard_news_story_id", None)
                st.session_state.pop("dashboard_news_story_headline", None)
                st.rerun()
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
        st.session_state.dashboard_start_date = default_live_api_start(30).date()
    if "dashboard_end_date" not in st.session_state:
        st.session_state.dashboard_end_date = default_live_api_end().date()

    preset_cols = st.columns(len(QUICK_TEST_TICKERS))
    for column, preset_ticker in zip(preset_cols, QUICK_TEST_TICKERS):
        if column.button(preset_ticker, key=f"dashboard_ticker_{preset_ticker}", use_container_width=True):
            st.session_state.dashboard_ticker = preset_ticker

    with st.form("multi_api_dashboard_query", clear_on_submit=False):
        control_cols = st.columns([1, 1, 1])
        ticker = control_cols[0].text_input("Ticker", key="dashboard_ticker", max_chars=16).strip().upper()
        start_date = control_cols[1].date_input(
            "Start date",
            key="dashboard_start_date",
            max_value=default_live_api_end().date(),
        )
        end_date = control_cols[2].date_input(
            "End date",
            key="dashboard_end_date",
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
        submitted = st.form_submit_button("Retrieve Dashboard Data", type="primary")

    if submitted:
        if start_date > end_date:
            st.error("Start date must be on or before end date.")
            return
        if not any([use_refinitiv, use_wrds, use_yahoo, use_ravenpack]):
            st.error("Select at least one data source.")
            return

        latest_crsp_date: pd.Timestamp | None = None
        if wrds_credentials_available():
            try:
                latest_crsp_date = get_latest_crsp_date()
            except Exception:
                latest_crsp_date = None

        with st.spinner(f"Retrieving dashboard data for {ticker}..."):
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
            )

        st.session_state.dashboard_result = {
            "live": live_result,
        }

    dashboard_result = st.session_state.get("dashboard_result")
    if not dashboard_result:
        st.info("Choose data sources above and click **Retrieve Dashboard Data**.")
        return

    live_result = dashboard_result["live"]
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
    st.caption(f"Dashboard window: **{live_result['start_date']}** to **{live_result['end_date']}** for **{ticker}**")

    pane_overview, pane_prices, pane_news, pane_sentiment, pane_raw = st.tabs([
        "Overview",
        "Prices",
        "News",
        "Sentiment",
        "Raw Data",
    ])

    with pane_overview:
        if live_result["price_frames"]:
            render_dashboard_price_pane(live_result, key_prefix="dashboard_overview")
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


def _load_all_manifests() -> pd.DataFrame:
    """Walk by_ticker/ dirs and collect manifest data into a DataFrame."""
    rows = []
    if not TOP1K_BY_TICKER_DIR.exists():
        return pd.DataFrame()
    for mfile in sorted(TOP1K_BY_TICKER_DIR.glob("rank_*/manifest.json")):
        try:
            m = json.loads(mfile.read_text(encoding="utf-8"))
            provider_map: dict[str, str] = {}
            provider_rows_map: dict[str, int] = {}
            for ps in m.get("provider_status", []):
                pname = ps.get("provider", "")
                provider_map[pname] = ps.get("status", "")
                provider_rows_map[pname] = int(ps.get("rows") or 0)
            rows.append({
                "rank":       m.get("volume_rank"),
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
    force_rerun: bool, rerun_failed: bool, sleep_sec: float,
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
    if not use_wrds:
        cmd.append("--no-wrds")
    if not use_yahoo:
        cmd.append("--no-yahoo")
    if not use_ravenpack:
        cmd.append("--no-ravenpack")
    if use_refinitiv:
        cmd.append("--refinitiv")
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


def render_batch_pipeline_tab() -> None:  # noqa: C901 – intentionally long UI function
    # Force Refinitiv off by default so stale session state never auto-enables it
    if "batch_use_refinitiv" not in st.session_state:
        st.session_state["batch_use_refinitiv"] = False

    st.header("Top-1,000 Batch Pipeline")
    st.caption(
        "Pull and cache WRDS/CRSP, Yahoo, and RavenPack data for every ticker in the "
        "CRSP top-volume universe. Each ticker is cached immediately; reruns skip "
        "completed tickers automatically."
    )

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

        # Live log tail — always visible while running
        log_path = TOP1K_OUTPUT_DIR / "batch_runner.log"
        if log_path.exists():
            try:
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                st.code("\n".join(lines[-25:]), language=None)
            except Exception:
                pass
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
                value=False,  # off by default — news scope may not be available even with a config file
                key="batch_use_refinitiv",
                help="Only enable if your LSEG account has the news/prices scope. Leave off for WRDS+Yahoo+RavenPack only.",
            )

            opt_cols = st.columns(3)
            force_rerun       = opt_cols[0].checkbox("Force rerun (ignore cache)", value=False, key="batch_force_rerun")
            rerun_failed      = opt_cols[1].checkbox("Retry failed tickers",       value=True,  key="batch_rerun_failed")
            combined_parquets = opt_cols[2].checkbox("Write combined parquets",    value=True,  key="batch_combined")

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
                sleep_sec=float(sleep_sec), stop_after=int(stop_after),
                provider_timeout=float(provider_timeout), year_timeout=int(year_timeout),
                use_wrds=use_wrds, use_yahoo=use_yahoo,
                use_ravenpack=use_ravenpack, use_refinitiv=use_refinitiv,
                combined_parquets=combined_parquets,
            )
            time.sleep(1.5)
            st.rerun()

    # ── Refresh / auto-refresh controls (below config) ───────────────────────
    ctrl_cols = st.columns([1, 5])
    if not is_running:
        if ctrl_cols[0].button("🔄  Refresh now"):
            st.rerun()
    auto_refresh = ctrl_cols[1].checkbox(
        "Auto-refresh every 5 s", value=is_running, key="batch_auto_refresh"
    )

    st.divider()

    # ── Read progress data ────────────────────────────────────────────────────
    progress_df = _read_batch_progress()
    manifests_df = _load_all_manifests()

    # ── Overall progress metrics ──────────────────────────────────────────────
    if progress_df is not None and not progress_df.empty:
        total_n    = len(progress_df)
        complete_n = int((progress_df["status"] == "complete").sum())
        partial_n  = int((progress_df["status"] == "partial").sum())
        failed_n   = int(progress_df["status"].isin(["failed", "error"]).sum())
        skipped_n  = int((progress_df["status"] == "skipped_cached").sum())
        pending_n  = max(0, (batch_status.get("total") or 0) - total_n)

        m_cols = st.columns(6)
        m_cols[0].metric("Processed",    f"{total_n:,}")
        m_cols[1].metric("🟢 Complete",  f"{complete_n:,}")
        m_cols[2].metric("🟡 Partial",   f"{partial_n:,}")
        m_cols[3].metric("🔴 Failed",    f"{failed_n:,}")
        m_cols[4].metric("⚪ Skipped",   f"{skipped_n:,}")
        m_cols[5].metric("⏳ Remaining", f"{pending_n:,}" if batch_status.get("total") else "—")
    elif not is_running:
        st.info("No batch progress recorded yet. Configure and launch a batch above.")

    st.divider()

    # ── Per-ticker progress table ─────────────────────────────────────────────
    st.subheader("Per-Ticker Status")

    # Prepend a live row for the ticker currently being processed so it shows
    # immediately without waiting for the manifest to be written.
    if is_running and batch_status:
        live_rank   = batch_status.get("current_rank")
        live_ticker = batch_status.get("current_ticker", "")
        live_step   = batch_status.get("current_step", "starting…")
        if live_ticker and live_rank:
            # Compute real elapsed
            try:
                from datetime import datetime, timezone as _tz2
                _ts = batch_status.get("ticker_started_at", "")
                _live_elapsed = round((datetime.now(_tz2.utc) - datetime.fromisoformat(_ts)).total_seconds()) if _ts else 0
            except Exception:
                _live_elapsed = 0
            live_row = pd.DataFrame([{
                "rank": live_rank, "ticker": live_ticker, "company": "⚡ in progress",
                "status": f"⚡ {live_step}",
                "wrds_status": "…", "wrds_rows": "",
                "yahoo_status": "…", "yahoo_rows": "",
                "ravenpack_status": "…", "ravenpack_rows": "",
                "refinitiv_status": "…", "refinitiv_rows": "",
                "ok": 0, "fail": 0, "created_at": f"{_live_elapsed}s elapsed",
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
        if status_filter:
            view_df = view_df[view_df["status"].isin(status_filter)]
        if show_ticker:
            view_df = view_df[view_df["ticker"].str.upper().str.contains(show_ticker)]
        if show_only_failed_providers:
            view_df = view_df[view_df["fail"] > 0]

        # Build display table
        display = view_df[[
            "rank", "ticker", "company", "status",
            "wrds_status", "wrds_rows",
            "yahoo_status", "yahoo_rows",
            "ravenpack_status", "ravenpack_rows",
            "refinitiv_status", "refinitiv_rows",
            "created_at",
        ]].copy()
        display.columns = [
            "Rank", "Ticker", "Company", "Status",
            "WRDS", "WRDS rows",
            "Yahoo", "Yahoo rows",
            "RavenPack", "RP rows",
            "Refinitiv", "RF rows",
            "Cached at",
        ]
        display["Status"] = display["Status"].apply(lambda s: f"{_status_color(s)} {s}")
        for col in ["WRDS", "Yahoo", "RavenPack", "Refinitiv"]:
            display[col] = display[col].apply(lambda s: ("✅" if s == "ok" else ("❌" if s == "failed" else ("—" if not s else s))))

        st.dataframe(display, use_container_width=True, height=480)
        st.caption(f"Showing {len(view_df):,} of {len(manifests_df):,} cached tickers")

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
                    prov_rows.append({
                        "Provider":  pname,
                        "Status":    r[f"{pname}_status"] or "—",
                        "Rows":      int(r[f"{pname}_rows"]) if r[f"{pname}_rows"] else 0,
                    })
                st.dataframe(pd.DataFrame(prov_rows), use_container_width=True, hide_index=True)

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
    elif progress_df is not None:
        st.info("Progress CSV exists but no manifest files found yet — the batch may still be on the first few tickers.")
    else:
        st.info("No ticker data cached yet.")

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
    if auto_refresh and is_running:
        time.sleep(1)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────

st.info(
    "The **Data Explorer** tab uses one ticker/date-range form for prices, Refinitiv news, and RavenPack sentiment. "
    "The **Paper Validation** tab uses bundled 2003-2014 CSVs from `app_data/`."
)

tab_dashboard, tab_batch, tab_validation = st.tabs([
    "Data Explorer",
    "Batch Pipeline (Top-1K)",
    "Paper Validation (2003-2014)",
])

with tab_dashboard:
    render_multi_api_dashboard_tab()

with tab_batch:
    render_batch_pipeline_tab()

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
