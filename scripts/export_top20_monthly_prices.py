"""Export monthly open, close, and average prices for top-20 CRSP candidates."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import wrds
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = PROJECT_ROOT / "data" / "raw" / "market" / "crsp_top_volume_universe.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_prices.csv"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    wrds_username = os.environ.get("WRDS_USERNAME")
    wrds_password = os.environ.get("WRDS_PASSWORD")
    if not wrds_username or not wrds_password:
        raise SystemExit("Set WRDS_USERNAME and WRDS_PASSWORD in .env before running.")

    universe = pd.read_csv(UNIVERSE_PATH)
    top20 = universe.sort_values("volume_rank").head(20)
    permno_sql = ", ".join(str(int(permno)) for permno in top20["permno"])

    query = f"""
    select
        permno,
        date,
        abs(openprc) as open_price,
        abs(prc) as close_price
    from crsp.dsf
    where date between '2003-01-01' and '2014-12-31'
      and permno in ({permno_sql})
      and prc is not null
    """

    db = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    try:
        daily_prices = db.raw_sql(query, date_cols=["date"])
    finally:
        db.close()

    prices = daily_prices.merge(top20[["permno", "ticker", "comnam"]], on="permno", how="inner")
    prices = prices.sort_values(["permno", "date"])
    prices["month"] = prices["date"].dt.to_period("M").dt.to_timestamp()

    monthly_prices = (
        prices.groupby(["month", "ticker", "comnam"], as_index=False)
        .agg(
            open_price=("open_price", "first"),
            close_price=("close_price", "last"),
            avg_price=("close_price", "mean"),
            trading_days=("close_price", "size"),
        )
        .sort_values(["ticker", "month"])
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly_prices.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(monthly_prices):,} monthly price rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
