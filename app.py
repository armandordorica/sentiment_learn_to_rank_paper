"""Streamlit app for CRSP universe validation charts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
APP_DATA_DIR = PROJECT_ROOT / "app_data"
DEFAULT_UNIVERSE_PATHS = [
    APP_DATA_DIR / "crsp_top_volume_universe.csv",
    PROJECT_ROOT / "data" / "raw" / "market" / "crsp_top_volume_universe.csv",
]
DEFAULT_MONTHLY_VOLUME_PATHS = [
    APP_DATA_DIR / "top20_monthly_volume.csv",
    PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_volume.csv",
]


def load_csv(uploaded_file, fallback_paths: list[Path]) -> pd.DataFrame | None:
    """Load an uploaded CSV, or a local fallback if it exists."""
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    for fallback_path in fallback_paths:
        if fallback_path.exists():
            return pd.read_csv(fallback_path)
    return None


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


st.set_page_config(
    page_title="Sentiment LTR Paper: CRSP Universe Validation",
    layout="wide",
)

st.title("CRSP Universe Validation")
st.caption(
    "Interactive validation charts for the market-side candidate universe used in the "
    "sentiment learning-to-rank paper replication."
)

st.info(
    "This app bundles small aggregated validation CSVs so the charts render immediately. "
    "Raw CRSP/WRDS daily data and credentials are not included."
)

with st.sidebar:
    st.header("Data Inputs")
    universe_upload = st.file_uploader(
        "CRSP top-volume universe CSV",
        type=["csv"],
        help="Expected file: data/raw/market/crsp_top_volume_universe.csv",
    )
    monthly_volume_upload = st.file_uploader(
        "Optional top-20 monthly or daily volume CSV",
        type=["csv"],
        help=(
            "Use monthly columns month/ticker/avg_daily_volume_millions/trading_days, "
            "or daily CRSP-style date/permno/vol columns."
        ),
    )

universe = load_csv(universe_upload, DEFAULT_UNIVERSE_PATHS)
if universe is None:
    st.info(
        "Upload `crsp_top_volume_universe.csv` to render the validation charts. "
        "Generate it locally with `python scripts/build_crsp_market_universe.py`."
    )
    st.stop()

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
monthly_volume = load_csv(monthly_volume_upload, DEFAULT_MONTHLY_VOLUME_PATHS)
if monthly_volume is None:
    st.info(
        "Upload a monthly or daily volume CSV to render the over-time chart. "
        "The validation notebook shows how to query and aggregate the top-20 CRSP volume series."
    )
else:
    try:
        monthly_volume = prepare_monthly_volume(monthly_volume, top20)
        st.plotly_chart(make_monthly_volume_chart(monthly_volume), use_container_width=True)
    except ValueError as exc:
        st.error(str(exc))
