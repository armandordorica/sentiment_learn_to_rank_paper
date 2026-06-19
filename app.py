"""Streamlit app for CRSP universe validation charts."""

from __future__ import annotations

import os
import sys
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

REFINITIV_IMPORT_ERROR: str | None = None


def _bind_refinitiv_helpers() -> bool:
    """Import or reload Refinitiv helpers, avoiding stale Streamlit module cache."""
    global REFINITIV_IMPORT_ERROR
    global query_refinitiv_news, query_refinitiv_prices, fetch_refinitiv_story
    global refinitiv_configured, refinitiv_setup_message, ticker_to_ric_candidates, refinitiv_session_mode

    try:
        import importlib

        from sentiment_ltr.data import refinitiv_queries

        refinitiv_queries = importlib.reload(refinitiv_queries)
        query_refinitiv_news = refinitiv_queries.query_refinitiv_news
        query_refinitiv_prices = refinitiv_queries.query_refinitiv_prices
        fetch_refinitiv_story = refinitiv_queries.fetch_refinitiv_story
        refinitiv_configured = refinitiv_queries.refinitiv_configured
        refinitiv_setup_message = refinitiv_queries.refinitiv_setup_message
        ticker_to_ric_candidates = refinitiv_queries.ticker_to_ric_candidates
        refinitiv_session_mode = refinitiv_queries.refinitiv_session_mode
        REFINITIV_IMPORT_ERROR = None
        return True
    except ImportError as exc:
        query_refinitiv_news = None
        query_refinitiv_prices = None
        fetch_refinitiv_story = None
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
    return {
        "WRDS_USERNAME": bool(get_secret_or_env("WRDS_USERNAME")),
        "WRDS_PASSWORD": bool(get_secret_or_env("WRDS_PASSWORD")),
    }


def wrds_credentials_available() -> bool:
    """Return whether the app has enough configuration for live WRDS queries."""
    status = wrds_credential_status()
    return status["WRDS_USERNAME"] and status["WRDS_PASSWORD"]


@st.cache_data(ttl=3600, show_spinner=False)
def query_wrds_ticker_data(ticker: str, start_date: str, end_date: str, row_limit: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query CRSP name history and daily stock data for a ticker."""
    if wrds is None:
        raise RuntimeError("The `wrds` package is not installed in this environment.")

    wrds_username = get_secret_or_env("WRDS_USERNAME")
    wrds_password = get_secret_or_env("WRDS_PASSWORD")
    if not wrds_username or not wrds_password:
        raise RuntimeError("WRDS credentials are not configured.")

    clean_ticker = "".join(char for char in ticker.upper().strip() if char.isalnum() or char in {".", "-"})
    if not clean_ticker:
        raise ValueError("Enter a valid ticker.")

    db = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    try:
        names_query = f"""
        select
            permno,
            permco,
            namedt,
            nameendt,
            ticker,
            comnam,
            shrcd,
            exchcd
        from crsp.msenames
        where trim(ticker) = '{clean_ticker}'
          and namedt <= '{end_date}'
          and nameendt >= '{start_date}'
        order by namedt, permno
        """
        names = db.raw_sql(names_query, date_cols=["namedt", "nameendt"])

        if names.empty:
            fallback_names_query = f"""
            select
                permno,
                permco,
                namedt,
                nameendt,
                ticker,
                comnam,
                shrcd,
                exchcd
            from crsp.msenames
            where trim(ticker) = '{clean_ticker}'
            order by nameendt desc, namedt desc
            """
            names = db.raw_sql(fallback_names_query, date_cols=["namedt", "nameendt"])

        if names.empty:
            return names, pd.DataFrame()

        permno_sql = ", ".join(str(int(permno)) for permno in sorted(names["permno"].dropna().unique()))
        daily_query = f"""
        select
            d.permno,
            n.permco,
            d.date,
            n.ticker,
            n.comnam,
            n.shrcd,
            n.exchcd,
            d.openprc,
            d.prc,
            d.ret,
            d.retx,
            d.vol,
            d.shrout,
            d.cfacpr,
            d.cfacshr,
            d.bidlo,
            d.askhi
        from crsp.dsf as d
        join crsp.msenames as n
          on d.permno = n.permno
         and d.date between n.namedt and n.nameendt
        where d.date between '{start_date}' and '{end_date}'
          and d.permno in ({permno_sql})
          and trim(n.ticker) = '{clean_ticker}'
        order by d.date desc, d.permno
        limit {int(row_limit)}
        """
        daily = db.raw_sql(daily_query, date_cols=["date"])
    finally:
        db.close()

    for column in ["openprc", "prc", "bidlo", "askhi"]:
        if column in daily.columns:
            daily[f"abs_{column}"] = daily[column].abs()
    return names, daily


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
    connection_info = test_wrds_connection()
    return pd.Timestamp(connection_info["latest_crsp_date"]).normalize()


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
    if wrds is None:
        raise RuntimeError("The `wrds` package is not installed in this environment.")

    wrds_username = get_secret_or_env("WRDS_USERNAME")
    wrds_password = get_secret_or_env("WRDS_PASSWORD")
    if not wrds_username or not wrds_password:
        raise RuntimeError("WRDS credentials are not configured.")

    db = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    try:
        latest = db.raw_sql("select max(date) as latest_crsp_date from crsp.dsf", date_cols=["latest_crsp_date"])
        sample = db.raw_sql(
            """
            select permno, date, prc, vol
            from crsp.dsf
            order by date desc
            limit 5
            """,
            date_cols=["date"],
        )
    finally:
        db.close()

    latest_date = latest["latest_crsp_date"].iloc[0]
    return {
        "latest_crsp_date": latest_date,
        "sample_rows": sample,
    }


def to_query_date(value: pd.Timestamp | str) -> str:
    """Normalize a date-like value to YYYY-MM-DD for WRDS/Yahoo queries."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_yahoo_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily Yahoo Finance prices for a public cross-check."""
    if yf is None:
        raise RuntimeError("The `yfinance` package is not installed in this environment.")

    start_date = to_query_date(start_date)
    end_date = to_query_date(end_date)
    end_exclusive = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(
        ticker.upper().strip(),
        start=start_date,
        end=end_exclusive,
        auto_adjust=False,
        progress=False,
    )
    if data is None or data.empty:
        raise ValueError(f"Yahoo Finance returned no rows for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    result = data.reset_index()
    date_column = "Date" if "Date" in result.columns else result.columns[0]
    result = result.rename(
        columns={
            date_column: "date",
            "Open": "yahoo_open",
            "Close": "yahoo_close",
            "Volume": "yahoo_volume",
        }
    )
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.tz_localize(None).dt.normalize()
    keep_cols = [col for col in ["date", "yahoo_open", "yahoo_close", "yahoo_volume"] if col in result.columns]
    if "yahoo_close" not in keep_cols:
        raise ValueError(f"Yahoo Finance response for {ticker} did not include a Close column.")
    return result[keep_cols].sort_values("date")


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
    if daily_lookup.empty:
        return pd.DataFrame()
    data = daily_lookup.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["close_price"] = data["prc"].abs()
    data["provider"] = "wrds"
    keep_cols = [col for col in ["date", "close_price", "vol", "provider", "ticker", "permno"] if col in data.columns]
    return data[keep_cols].sort_values("date")


def yahoo_price_frame(yahoo_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert Yahoo rows to a common price schema."""
    if yahoo_daily.empty:
        return pd.DataFrame()
    data = yahoo_daily.copy()
    data["close_price"] = data["yahoo_close"]
    data["provider"] = "yahoo"
    if "yahoo_volume" in data.columns:
        data["volume"] = data["yahoo_volume"]
    keep_cols = [col for col in ["date", "close_price", "volume", "provider"] if col in data.columns]
    return data[keep_cols].sort_values("date")


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
    news_count: int = 50,
    wrds_limit: int = 500,
    latest_crsp_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Query selected market-data providers in parallel for the same ticker and date range."""
    clean_ticker = ticker.upper().strip()
    start_s = to_query_date(start_date)
    end_s = to_query_date(end_date)
    providers: dict[str, dict[str, object]] = {
        "refinitiv": {"status": "skipped", "error": None, "prices": pd.DataFrame(), "news": pd.DataFrame(), "ric": None},
        "wrds": {"status": "skipped", "error": None, "prices": pd.DataFrame(), "names": pd.DataFrame()},
        "yahoo": {"status": "skipped", "error": None, "prices": pd.DataFrame()},
    }

    if query_refinitiv:
        _bind_refinitiv_helpers()
        if query_refinitiv_prices is not None and refinitiv_configured(PROJECT_ROOT):
            try:
                refinitiv_prices, ric = query_refinitiv_prices(PROJECT_ROOT, clean_ticker, start_s, end_s)
                refinitiv_news = pd.DataFrame()
                news_error = None
                if query_refinitiv_news is not None and news_count > 0:
                    try:
                        refinitiv_news, _ = query_refinitiv_news(
                            PROJECT_ROOT,
                            clean_ticker,
                            start_s,
                            end_s,
                            count=news_count,
                        )
                    except Exception as exc:
                        news_error = str(exc)
                providers["refinitiv"] = {
                    "status": "ok" if not refinitiv_prices.empty else "empty",
                    "error": news_error if not refinitiv_prices.empty else "No Refinitiv price history returned.",
                    "prices": refinitiv_prices,
                    "news": refinitiv_news,
                    "ric": ric,
                }
            except Exception as exc:
                providers["refinitiv"] = {
                    "status": "failed",
                    "error": str(exc),
                    "prices": pd.DataFrame(),
                    "news": pd.DataFrame(),
                    "ric": None,
                }
        else:
            providers["refinitiv"] = {
                "status": "unavailable",
                "error": refinitiv_setup_message(PROJECT_ROOT),
                "prices": pd.DataFrame(),
                "news": pd.DataFrame(),
                "ric": None,
            }

    if query_wrds:
        if wrds_credentials_available():
            wrds_start = pd.Timestamp(start_s)
            wrds_end = pd.Timestamp(end_s)
            if latest_crsp_date is not None:
                crsp_end = min(pd.Timestamp.today().normalize(), pd.Timestamp(latest_crsp_date).normalize())
                wrds_end = min(wrds_end, crsp_end)
            if wrds_start <= wrds_end:
                try:
                    name_history, daily_lookup = query_wrds_ticker_data(
                        clean_ticker,
                        to_query_date(wrds_start),
                        to_query_date(wrds_end),
                        int(wrds_limit),
                    )
                    wrds_prices = wrds_price_frame(daily_lookup)
                    providers["wrds"] = {
                        "status": "ok" if not wrds_prices.empty else "empty",
                        "error": None if not wrds_prices.empty else "No CRSP rows in the selected date range.",
                        "prices": wrds_prices,
                        "names": name_history,
                        "query_start": wrds_start,
                        "query_end": wrds_end,
                    }
                except Exception as exc:
                    providers["wrds"] = {
                        "status": "failed",
                        "error": str(exc),
                        "prices": pd.DataFrame(),
                        "names": pd.DataFrame(),
                    }
            else:
                providers["wrds"] = {
                    "status": "empty",
                    "error": "Selected range is entirely after the latest CRSP date available in WRDS.",
                    "prices": pd.DataFrame(),
                    "names": pd.DataFrame(),
                }
        else:
            providers["wrds"] = {
                "status": "unavailable",
                "error": "WRDS credentials are not configured.",
                "prices": pd.DataFrame(),
                "names": pd.DataFrame(),
            }

    if query_yahoo:
        try:
            yahoo_daily = fetch_yahoo_daily(clean_ticker, start_s, end_s)
            yahoo_prices = yahoo_price_frame(yahoo_daily)
            providers["yahoo"] = {
                "status": "ok" if not yahoo_prices.empty else "empty",
                "error": None if not yahoo_prices.empty else "Yahoo Finance returned no rows.",
                "prices": yahoo_prices,
            }
        except Exception as exc:
            providers["yahoo"] = {
                "status": "failed",
                "error": str(exc),
                "prices": pd.DataFrame(),
            }

    price_frames = {
        provider: result["prices"]
        for provider, result in providers.items()
        if isinstance(result.get("prices"), pd.DataFrame) and not result["prices"].empty
    }

    return {
        "ticker": clean_ticker,
        "start_date": start_s,
        "end_date": end_s,
        "providers": providers,
        "price_frames": price_frames,
        "selected_providers": {
            "refinitiv": query_refinitiv,
            "wrds": query_wrds,
            "yahoo": query_yahoo,
        },
    }


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
            )
            st.dataframe(prices.sort_values("date", ascending=False), use_container_width=True, height=250)
        else:
            st.warning(str(provider_result.get("error") or provider_result.get("status")))


def render_refinitiv_news_headlines(news_df: pd.DataFrame) -> None:
    """Render Refinitiv headlines with row selection to read full stories."""
    if news_df.empty:
        return

    st.markdown("#### Refinitiv News Headlines")
    st.caption(
        "Select a headline row to load the full story from Refinitiv Workspace. "
        "These stories are not available as public web links."
    )

    if "storyId" not in news_df.columns:
        summary_cols = [col for col in ["date", "headline", "sourceCode"] if col in news_df.columns]
        st.dataframe(news_df[summary_cols], use_container_width=True, hide_index=True)
        st.warning("Headline rows did not include a `storyId`, so full stories cannot be opened.")
        return

    if fetch_refinitiv_story is None:
        st.warning("Refinitiv story loading is unavailable in this environment.")
        return

    display_df = news_df.copy().reset_index(drop=True)
    if "date" in display_df.columns:
        display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d %H:%M")

    story_ids = display_df["storyId"].astype(str).tolist()
    table_columns = [col for col in ["date", "headline", "sourceCode"] if col in display_df.columns]
    table_df = display_df[table_columns]
    table_key = f"refinitiv_news_table_{st.session_state.get('refinitiv_news_version', 0)}"

    selection = st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key=table_key,
    )

    selected_rows: list[int] = []
    if selection is not None and hasattr(selection, "selection") and selection.selection.rows:
        selected_rows = list(selection.selection.rows)

    if not selected_rows:
        st.info("Click a headline row above to read the full story.")
        return

    row_idx = int(selected_rows[0])
    if row_idx < 0 or row_idx >= len(story_ids):
        st.warning("Could not resolve the selected headline.")
        return

    story_id = story_ids[row_idx]
    headline = str(table_df.iloc[row_idx].get("headline", "Selected headline"))

    st.markdown("---")
    st.markdown(f"**Reading:** {headline}")
    with st.spinner("Loading full Refinitiv story..."):
        try:
            story_text = load_refinitiv_story_text(story_id)
        except Exception as exc:
            st.error(f"Could not load story: {exc}")
        else:
            st.text_area(
                "Full story",
                value=story_text,
                height=420,
                label_visibility="collapsed",
            )


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
    refinitiv_news = refinitiv.get("news")
    selected = query_result.get("selected_providers", {})
    if (
        selected.get("refinitiv", True)
        and isinstance(refinitiv_news, pd.DataFrame)
        and not refinitiv_news.empty
    ):
        render_refinitiv_news_headlines(refinitiv_news)
    elif refinitiv.get("error") and refinitiv.get("status") == "ok":
        st.caption(f"Refinitiv news note: {refinitiv['error']}")

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
        control_cols = st.columns([1, 1, 1, 1])
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
        news_count = control_cols[3].number_input("Max news rows", min_value=5, max_value=200, value=25, step=5)
        st.caption(
            f"Refinitiv will try RIC candidates such as **{ric_hint}**. "
            "Use a full RIC like `AAPL.O` when needed."
        )
        include_news = st.checkbox(
            "Include Refinitiv news headlines",
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
                news_count=0 if not include_news else int(news_count),
                latest_crsp_date=latest_crsp_date,
            )
        st.session_state.refinitiv_news_version = st.session_state.get("refinitiv_news_version", 0) + 1
        st.session_state.live_api_query_result = query_result

    if "live_api_query_result" in st.session_state:
        render_live_api_results(st.session_state.live_api_query_result)


st.set_page_config(
    page_title="Sentiment LTR Paper: CRSP Universe Validation",
    layout="wide",
)

st.title("CRSP Universe Validation")
st.caption(
    "Paper-replication validation charts plus a live API test tab for parallel Refinitiv, WRDS, and Yahoo comparisons."
)

st.info(
    "The **Paper Validation** tab uses bundled 2003-2014 CSVs from `app_data/`. "
    "The **Live API Test** tab queries selected providers simultaneously and renders them side by side."
)

tab_validation, tab_live_api = st.tabs(["Paper Validation (2003-2014)", "Live API Test"])

with tab_live_api:
    render_live_api_test_tab()

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
