"""Build a CRSP top-volume candidate universe from WRDS.

This script creates the market-side candidate universe for the paper
replication. It does not apply the TRNA news-coverage filter.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wrds
import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _sql_list(values: list[int]) -> str:
    return ", ".join(str(value) for value in values)


def build_query(
    start: str,
    end: str,
    candidate_count: int,
    share_codes: list[int],
    exchange_codes: list[int],
) -> str:
    """Create a server-side CRSP query for top average-volume securities."""
    return f"""
with eligible_daily as (
    select
        d.permno,
        d.date,
        abs(d.prc) as price,
        d.vol,
        d.ret,
        d.shrout
    from crsp.dsf as d
    join crsp.msenames as n
      on d.permno = n.permno
     and d.date between n.namedt and n.nameendt
    where d.date between '{start}' and '{end}'
      and n.shrcd in ({_sql_list(share_codes)})
      and n.exchcd in ({_sql_list(exchange_codes)})
      and d.vol is not null
),
volume_rank as (
    select
        permno,
        count(*) as trading_days,
        min(date) as first_trade_date,
        max(date) as last_trade_date,
        avg(vol) as avg_volume,
        avg(price * vol) as avg_dollar_volume,
        avg(abs(price)) as avg_abs_price,
        avg(shrout) as avg_shares_outstanding
    from eligible_daily
    group by permno
),
top_candidates as (
    select *
    from volume_rank
    order by avg_volume desc
    limit {candidate_count}
),
latest_names as (
    select distinct on (n.permno)
        n.permno,
        n.permco,
        n.ticker,
        n.comnam,
        n.shrcd,
        n.exchcd,
        n.namedt,
        n.nameendt
    from crsp.msenames as n
    join top_candidates as t
      on n.permno = t.permno
    where n.namedt <= '{end}'
      and n.nameendt >= '{start}'
    order by n.permno, n.nameendt desc, n.namedt desc
)
select
    row_number() over (order by t.avg_volume desc) as volume_rank,
    t.permno,
    n.permco,
    n.ticker,
    n.comnam,
    n.shrcd,
    n.exchcd,
    t.trading_days,
    t.first_trade_date,
    t.last_trade_date,
    t.avg_volume,
    t.avg_dollar_volume,
    t.avg_abs_price,
    t.avg_shares_outstanding,
    n.namedt as latest_name_start,
    n.nameendt as latest_name_end
from top_candidates as t
left join latest_names as n
  on t.permno = n.permno
order by t.avg_volume desc
"""


def main() -> None:
    load_dotenv()
    wrds_username = os.environ.get("WRDS_USERNAME")
    wrds_password = os.environ.get("WRDS_PASSWORD")
    if not wrds_username or not wrds_password:
        raise SystemExit("Set WRDS_USERNAME and WRDS_PASSWORD in a local .env file before running.")

    config_path = PROJECT_ROOT / "config" / "market_data.yml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    market_config = config["market_data"]
    crsp_config = market_config["crsp"]

    query = build_query(
        start=market_config["start"],
        end=market_config["end"],
        candidate_count=crsp_config["candidate_count"],
        share_codes=crsp_config["share_codes"],
        exchange_codes=crsp_config["exchange_codes"],
    )

    db = wrds.Connection(wrds_username=wrds_username, wrds_password=wrds_password)
    try:
        candidates = db.raw_sql(
            query,
            date_cols=[
                "first_trade_date",
                "last_trade_date",
                "latest_name_start",
                "latest_name_end",
            ],
        )
    finally:
        db.close()

    output_dir = PROJECT_ROOT / "data" / "raw" / "market"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = output_dir / "crsp_top_volume_universe.csv"
    manifest_path = output_dir / "crsp_top_volume_universe_manifest.json"

    candidates.to_csv(candidates_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "wrds.crsp.dsf joined to wrds.crsp.msenames",
        "start": market_config["start"],
        "end": market_config["end"],
        "candidate_count": crsp_config["candidate_count"],
        "share_codes": crsp_config["share_codes"],
        "exchange_codes": crsp_config["exchange_codes"],
        "rows": int(len(candidates)),
        "columns": list(candidates.columns),
        "output_file": str(candidates_path.relative_to(PROJECT_ROOT)),
        "ranking_rule": "Top securities by average CRSP daily share volume over the configured date range.",
        "note": "This is the market-side candidate universe only; the paper's final universe also filters by TRNA news coverage.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(candidates):,} CRSP candidates to {candidates_path}")
    print(f"Wrote manifest to {manifest_path}")
    print("\nTop 10 by average volume:")
    display_cols = ["volume_rank", "permno", "ticker", "comnam", "avg_volume", "trading_days"]
    print(candidates[display_cols].head(10).to_string(index=False))

    spot_tickers = ["AAPL", "MSFT", "GE", "XOM", "SPY"]
    spot_checks = candidates[candidates["ticker"].isin(spot_tickers)][display_cols]
    print("\nSpot-check tickers found:")
    if spot_checks.empty:
        print("None of the requested spot-check tickers were in the top-volume candidate list.")
    else:
        print(spot_checks.to_string(index=False))


if __name__ == "__main__":
    main()
