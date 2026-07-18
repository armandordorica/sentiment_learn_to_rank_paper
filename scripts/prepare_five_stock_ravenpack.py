#!/usr/bin/env python
"""Fetch rich RavenPack headline exports for the default five-stock pool.

This is intentionally separate from model training: all stocks are materialized
first, then concatenated and split once by date. That prevents sequential
fine-tuning/forgetting and makes leakage checks deterministic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

from sentiment_ltr.data.live_data import query_ravenpack_articles
from sentiment_ltr.models.ravenpack_sentiment import (
    DEFAULT_FIVE_STOCK_TICKERS,
    RAVENPACK_NEWS_DIR,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare pooled five-stock RavenPack headlines")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_FIVE_STOCK_TICKERS)
    parser.add_argument("--start", default="2003-01-01")
    parser.add_argument("--end", default="2014-12-31")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    RAVENPACK_NEWS_DIR.mkdir(parents=True, exist_ok=True)

    for ticker in [str(t).upper() for t in args.tickers]:
        path = RAVENPACK_NEWS_DIR / f"{ticker.lower()}_articles_2003_2014.parquet"
        if path.exists() and not args.force:
            existing = pd.read_parquet(path)
            if "headline" in existing and existing["headline"].notna().any():
                print(f"{ticker}: ready ({len(existing):,} rows); skipping", flush=True)
                continue
        print(f"{ticker}: fetching rich RavenPack text…", flush=True)
        frame = query_ravenpack_articles(
            ticker, args.start, args.end, include_text=True,
            year_progress_callback=lambda year, rows, elapsed, error, t=ticker: print(
                f"  {t} {year}: {rows:,} rows in {elapsed:.1f}s" + (f" — {error}" if error else ""),
                flush=True,
            ),
        )
        if frame.empty or "headline" not in frame:
            raise RuntimeError(f"{ticker}: WRDS returned no usable headline text.")
        frame.to_parquet(path, index=False)
        print(f"{ticker}: saved {len(frame):,} rows to {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
