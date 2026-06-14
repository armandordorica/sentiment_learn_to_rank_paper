"""Candidate stock-universe helpers."""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests


@dataclass(frozen=True)
class UniverseConfig:
    """Configuration for candidate universe retrieval."""

    wikipedia_sp500_url: str = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    replace_dot_with_dash: bool = True


def load_sp500_candidates(config: UniverseConfig | None = None) -> pd.DataFrame:
    """Load a reproducible open-data candidate list from Wikipedia's S&P 500 table.

    This is a development universe, not an exact replication of the paper's
    Bloomberg top-1000-by-volume universe.
    """
    config = config or UniverseConfig()
    response = requests.get(
        config.wikipedia_sp500_url,
        headers={"User-Agent": "sentiment-ltr-paper/0.1"},
        timeout=30,
    )
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    constituents = tables[0].copy()
    constituents = constituents.rename(
        columns={
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "Headquarters Location": "headquarters_location",
            "Date added": "date_added",
            "CIK": "cik",
            "Founded": "founded",
        }
    )
    constituents["source_ticker"] = constituents["ticker"]
    if config.replace_dot_with_dash:
        constituents["ticker"] = constituents["ticker"].str.replace(".", "-", regex=False)
    return constituents[
        [
            "ticker",
            "source_ticker",
            "company_name",
            "gics_sector",
            "gics_sub_industry",
            "headquarters_location",
            "date_added",
            "cik",
            "founded",
        ]
    ]
