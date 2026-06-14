"""Market data retrieval helpers.

This module is adapted from the `Stocks.py` helper in
`armandordorica/Portfolio_Optimization_2023`. It keeps the same basic behavior:
try Alpaca first when credentials are available, then fall back to Yahoo Finance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Literal

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

Provider = Literal["alpaca", "yahoo"]


@dataclass(frozen=True)
class MarketDataConfig:
    """Configuration for market data retrieval."""

    start: str
    end: str
    timeframe: str = "1D"
    prefer_provider: Provider = "alpaca"
    yahoo_suffixes: tuple[str, ...] = (".TO",)


class MarketDataClient:
    """Fetch OHLCV market data from Alpaca with Yahoo Finance fallback."""

    def __init__(
        self,
        config: MarketDataConfig,
        alpaca_api_key: str | None = None,
        alpaca_secret_key: str | None = None,
    ) -> None:
        load_dotenv()
        self.config = config
        self.alpaca_api_key = alpaca_api_key or os.getenv("ALPACA_API_KEY")
        self.alpaca_secret_key = alpaca_secret_key or os.getenv("ALPACA_SECRET_KEY")

    def fetch(self, ticker: str) -> pd.DataFrame:
        """Fetch one ticker and return a daily OHLCV dataframe."""
        if self.config.prefer_provider == "alpaca":
            try:
                return self.fetch_alpaca(ticker)
            except Exception:
                return self.fetch_yahoo(ticker)

        return self.fetch_yahoo(ticker)

    def fetch_many(self, tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
        """Fetch several tickers keyed by their requested ticker symbol."""
        return {ticker: self.fetch(ticker) for ticker in tickers}

    def fetch_alpaca(self, ticker: str) -> pd.DataFrame:
        """Fetch one ticker from Alpaca."""
        if not self.alpaca_api_key or not self.alpaca_secret_key:
            raise ValueError("Alpaca credentials are not configured.")

        import alpaca_trade_api as tradeapi

        api = tradeapi.REST(
            self.alpaca_api_key,
            self.alpaca_secret_key,
            api_version="v2",
        )
        data = api.get_bars(
            ticker,
            self.config.timeframe,
            start=self.config.start,
            end=self.config.end,
        ).df

        if data.empty:
            raise ValueError(f"Alpaca returned no data for {ticker}.")

        data = data.loc[self.config.start : self.config.end]
        data = data[["open", "high", "low", "close", "volume"]]
        data.columns = ["Open", "High", "Low", "Close", "Volume"]
        return self._standardize(data, ticker=ticker, provider="alpaca")

    def fetch_yahoo(self, ticker: str) -> pd.DataFrame:
        """Fetch one ticker from Yahoo Finance."""
        errors: list[Exception] = []
        for yahoo_ticker in self._yahoo_ticker_candidates(ticker):
            try:
                data = yf.download(
                    yahoo_ticker,
                    start=self.config.start,
                    end=self.config.end,
                    auto_adjust=False,
                    progress=False,
                )
            except Exception as exc:
                errors.append(exc)
                continue

            if data is None or data.empty:
                continue

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            columns = [col for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if col in data.columns]
            data = data[columns]
            standardized = self._standardize(data, ticker=ticker, provider="yahoo")
            standardized["source_symbol"] = yahoo_ticker
            return standardized

        if errors:
            raise ValueError(f"Yahoo Finance failed for {ticker}: {errors[-1]}") from errors[-1]
        raise ValueError(f"Yahoo Finance returned no data for {ticker}.")

    def _yahoo_ticker_candidates(self, ticker: str) -> list[str]:
        candidates = [ticker]
        if "." not in ticker:
            candidates.extend(f"{ticker}{suffix}" for suffix in self.config.yahoo_suffixes)
        return candidates

    @staticmethod
    def _standardize(data: pd.DataFrame, ticker: str, provider: Provider) -> pd.DataFrame:
        result = data.copy()
        result.index = pd.to_datetime(result.index).tz_localize(None)
        result.index.name = "date"
        result = result.sort_index()
        result["ticker"] = ticker
        result["provider"] = provider
        if "Adj Close" in result.columns:
            result["Return"] = result["Adj Close"].pct_change()
        else:
            result["Return"] = result["Close"].pct_change()
        return result
