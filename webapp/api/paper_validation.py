"""FastAPI presentation adapter for Streamlit Tab 7: Paper Validation."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNIVERSE_PATHS = [PROJECT_ROOT / "app_data" / "crsp_top_volume_universe.csv",
                  PROJECT_ROOT / "data" / "raw" / "market" / "crsp_top_volume_universe.csv"]
VOLUME_PATH = PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_volume.csv"
PRICE_PATH = PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_prices.csv"


def _chart_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


@lru_cache(maxsize=1)
def load_universe() -> pd.DataFrame:
    path = next((p for p in UNIVERSE_PATHS if p.exists()), None)
    if path is None:
        raise FileNotFoundError("CRSP candidate-universe CSV was not found.")
    data = pd.read_csv(path)
    for col in ("first_trade_date", "last_trade_date", "latest_name_start", "latest_name_end"):
        if col in data:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    if "avg_volume_millions" not in data:
        data["avg_volume_millions"] = data["avg_volume"] / 1_000_000
    if "avg_dollar_volume_billions" not in data:
        data["avg_dollar_volume_billions"] = data["avg_dollar_volume"] / 1_000_000_000
    return data.sort_values("volume_rank").reset_index(drop=True)


def validation_checks(universe: pd.DataFrame) -> list[dict[str, Any]]:
    checks = {
        "Exactly 1,000 candidate rows": len(universe) == 1000,
        "Volume rank is unique": universe["volume_rank"].is_unique,
        "PERMNO is unique": universe["permno"].is_unique,
        "Average volume is descending": universe["avg_volume"].is_monotonic_decreasing,
        "Only CRSP common-share codes 10/11": set(universe["shrcd"].dropna().astype(int)).issubset({10, 11}),
        "Only NYSE/AMEX/Nasdaq exchange codes": set(universe["exchcd"].dropna().astype(int)).issubset({1, 2, 3}),
    }
    return [{"check": name, "passed": bool(passed)} for name, passed in checks.items()]


def _top20_chart(top20: pd.DataFrame):
    data = top20.sort_values("avg_volume_millions").copy()
    data["label"] = data["ticker"] + " — " + data["comnam"].str.title()
    fig = px.bar(data, x="avg_volume_millions", y="label", orientation="h",
                 title="Top 20 CRSP common stocks by average daily share volume, 2003–2014",
                 labels={"avg_volume_millions": "Average daily volume, millions of shares", "label": ""},
                 hover_data={"ticker": True, "permno": True, "trading_days": ":,",
                             "avg_volume_millions": ":,.2f"}, color_discrete_sequence=["#4C78A8"])
    fig.update_layout(height=700, hovermode="closest")
    return fig


@lru_cache(maxsize=1)
def page_context() -> dict[str, Any]:
    universe = load_universe()
    top20 = universe.head(20).copy()
    volume = pd.read_csv(VOLUME_PATH, parse_dates=["month"])
    volume_fig = px.line(volume.sort_values(["ticker", "month"]), x="month",
                         y="avg_daily_volume_millions", color="ticker",
                         title="Monthly average daily trading volume for top 20 CRSP candidates, 2003–2014",
                         labels={"month": "Month", "avg_daily_volume_millions": "Average daily volume, millions of shares"},
                         hover_data={"comnam": True, "trading_days": True})
    volume_fig.update_traces(mode="lines+markers", line={"width": 1.8}, marker={"size": 4})
    volume_fig.update_layout(height=750, hovermode="closest")
    display_cols = ["volume_rank", "permno", "ticker", "comnam", "trading_days",
                    "avg_volume_millions", "avg_dollar_volume_billions",
                    "first_trade_date", "last_trade_date"]
    rows = top20[display_cols].copy()
    for col in ("first_trade_date", "last_trade_date"):
        rows[col] = rows[col].dt.strftime("%Y-%m-%d")
    return {
        "summary": {"rows": len(universe), "unique_permnos": int(universe["permno"].nunique()),
                    "top20_rows": len(top20), "date_range": "2003–2014"},
        "checks": validation_checks(universe), "top20_rows": rows.to_dict(orient="records"),
        "top20_columns": display_cols, "top20_chart": _chart_html(_top20_chart(top20)),
        "volume_chart": _chart_html(volume_fig),
        "tickers": top20["ticker"].dropna().tolist(),
    }


@lru_cache(maxsize=24)
def price_chart(ticker: str) -> dict[str, str]:
    ticker = ticker.strip().upper()
    prices = pd.read_csv(PRICE_PATH, parse_dates=["month"])
    stock = prices[prices["ticker"].str.upper() == ticker].copy()
    if stock.empty:
        raise ValueError(f"No monthly price data found for {ticker}.")
    company = stock["comnam"].iloc[0]
    long = stock.melt(id_vars=["month", "ticker", "comnam", "trading_days"],
                      value_vars=["open_price", "close_price", "avg_price"],
                      var_name="price_type", value_name="price")
    long["price_type"] = long["price_type"].map(
        {"open_price": "Open price", "close_price": "Close price", "avg_price": "Average price"})
    fig = px.line(long, x="month", y="price", color="price_type",
                  title=f"Monthly open, close, and average price for {ticker} — {company.title()}",
                  labels={"month": "Month", "price": "Price, USD", "price_type": "Series"},
                  hover_data={"trading_days": True})
    fig.update_traces(mode="lines+markers", line={"width": 2}, marker={"size": 5})
    fig.update_layout(height=650, hovermode="closest")
    return {"ticker": ticker, "chart": _chart_html(fig)}
