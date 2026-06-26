#!/usr/bin/env python
"""Retrieve CRSP delisting reasons for the Top-1,000 universe.

By default looks up all 1,000 PERMNOs in ``crsp.msedelist`` and caches results
locally (only missing PERMNOs are queried on repeat runs).

Usage:

    python scripts/fetch_delisting_info.py              # all 1,000 (default)
    python scripts/fetch_delisting_info.py --non-complete-only
    python scripts/fetch_delisting_info.py --status partial failed

Output:
    data/raw/data_explorer_top1k/delisting_info.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

from sentiment_ltr.data.crsp_delisting import (
    DELISTING_CACHE_PATH,
    load_delisting_cache,
    update_delisting_cache,
)

UNIVERSE_PATH = PROJECT_ROOT / "app_data" / "crsp_top_volume_universe.csv"
BY_TICKER_DIR = PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "by_ticker"
OUTPUT_CSV = PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "delisting_info.csv"


def _safe_slug(ticker: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(ticker).upper().strip())


def manifest_status(rank: int, ticker: str) -> str:
    """Return the cached manifest status, or 'never_cached' when absent."""
    path = BY_TICKER_DIR / f"rank_{int(rank):04d}_{_safe_slug(ticker)}" / "manifest.json"
    if not path.exists():
        return "never_cached"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status") or "unknown")
    except Exception:
        return "unreadable"


def load_universe_with_status() -> pd.DataFrame:
    universe = pd.read_csv(UNIVERSE_PATH, parse_dates=["first_trade_date", "last_trade_date"])
    universe["volume_rank"] = universe["volume_rank"].astype(int)
    universe = universe.sort_values("volume_rank").reset_index(drop=True)
    universe["manifest_status"] = [
        manifest_status(r.volume_rank, r.ticker) for r in universe.itertuples()
    ]
    return universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status", nargs="*", default=None,
        help="Manifest statuses to include (overrides default universe scope).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Explicitly select all 1,000 universe PERMNOs (default when no filter is set).",
    )
    parser.add_argument(
        "--non-complete-only", action="store_true",
        help="Only tickers whose batch manifest is not 'complete'.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--force", action="store_true",
        help="Re-query CRSP even for PERMNOs already in the local cache.",
    )
    args = parser.parse_args()

    universe = load_universe_with_status()
    total = len(universe)

    if args.status:
        target = universe[universe["manifest_status"].isin(args.status)].copy()
    elif args.non_complete_only:
        target = universe[universe["manifest_status"] != "complete"].copy()
    else:
        target = universe.copy()

    n_complete = int((universe["manifest_status"] == "complete").sum())
    print(f"[delist] Universe: {total} tickers  |  complete: {n_complete}  |  "
          f"selected for delisting lookup: {len(target)}")
    print("[delist] Manifest status breakdown:")
    for status, count in universe["manifest_status"].value_counts().items():
        print(f"         {status:>14}: {count}")

    if target.empty:
        print("[delist] Nothing to look up.")
        return

    already = len(load_delisting_cache())
    print(f"[delist] Cache currently holds {already} checked PERMNOs "
          f"({DELISTING_CACHE_PATH.relative_to(PROJECT_ROOT)}).")
    if args.force:
        print("[delist] --force: re-querying all target PERMNOs from CRSP.")
    print(f"[delist] Updating cache for {target['permno'].nunique()} target PERMNOs "
          "(only missing ones are queried)…")
    cache, n_new = update_delisting_cache(target["permno"].tolist(), force=args.force)
    print(f"[delist] Queried {n_new} new PERMNO(s) from CRSP.")

    merged = target.merge(
        cache[["permno", "delisted", "dlstdt", "dlstcd", "delisting_category",
               "delisting_label", "dlret", "dlretx", "nwperm"]],
        on="permno", how="left",
    )
    merged["delisted"] = merged["delisted"].fillna(False).astype(bool)

    keep_cols = [
        "volume_rank", "permno", "ticker", "comnam", "manifest_status", "delisted",
        "last_trade_date", "dlstdt", "dlstcd", "delisting_category",
        "delisting_label", "dlret", "dlretx", "nwperm",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    report = merged[keep_cols].sort_values("volume_rank").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)

    n_delisted = int(merged["delisted"].fillna(False).astype(bool).sum())
    n_active = len(merged) - n_delisted
    print(f"\n[delist] {n_delisted} of {len(merged)} selected tickers are CRSP-delisted; "
          f"{n_active} are still active (dlstcd 100 or no msedelist row).")

    delisted_only = merged[merged["delisted"].fillna(False).astype(bool)]

    print("\n[delist] Delisting reason breakdown (CRSP-delisted only):")
    if delisted_only.empty:
        print("         (none)")
    else:
        for category, count in delisted_only["delisting_category"].fillna("unknown").value_counts().items():
            print(f"         {category:>12}: {count}")

    print("\n[delist] Top specific delisting codes (CRSP-delisted only):")
    if delisted_only.empty:
        print("         (none)")
    else:
        codes = (
            delisted_only[["dlstcd", "delisting_label"]]
            .value_counts()
            .head(20)
        )
        for (code, label), count in codes.items():
            print(f"         {int(code):>4}  {count:>4}  {label}")

    if n_delisted:
        avg_dlret = pd.to_numeric(delisted_only["dlret"], errors="coerce").mean()
        print(f"\n[delist] Mean delisting return (dlret) across delisted names: {avg_dlret:.4f}")

    print(f"\n[delist] Wrote {len(report)} rows → {args.output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
