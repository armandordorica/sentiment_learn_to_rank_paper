"""Export monthly average daily volume for the top-20 CRSP candidates."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import wrds
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_PATH = PROJECT_ROOT / "data" / "raw" / "market" / "crsp_top_volume_universe.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "validation" / "top20_monthly_volume.csv"


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
        vol
    from crsp.dsf
    where date between '2003-01-01' and '2014-12-31'
      and permno in ({permno_sql})
      and vol is not null
    """

    db = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    try:
        daily_volume = db.raw_sql(query, date_cols=["date"])
    finally:
        db.close()

    volume = daily_volume.merge(top20[["permno", "ticker", "comnam"]], on="permno", how="inner")
    volume["month"] = volume["date"].dt.to_period("M").dt.to_timestamp()
    volume["volume_millions"] = volume["vol"] / 1_000_000
    monthly_volume = (
        volume.groupby(["month", "ticker", "comnam"], as_index=False)
        .agg(
            avg_daily_volume_millions=("volume_millions", "mean"),
            trading_days=("volume_millions", "size"),
        )
        .sort_values(["ticker", "month"])
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly_volume.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(monthly_volume):,} monthly rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
