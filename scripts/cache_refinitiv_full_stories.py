#!/usr/bin/env python
"""Cache Refinitiv full story bodies for a ticker/window (resumable).

Example (AAPL paper window):

  python scripts/cache_refinitiv_full_stories.py --ticker AAPL \\
      --start 2003-01-01 --end 2014-12-31

Skips stories already under data/raw/data_explorer_full_stories/{TICKER}/.
Use --limit 20 for a smoke test. Progress: .../{TICKER}/_pull_progress.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from sentiment_ltr.data.refinitiv_story_cache import (  # noqa: E402
    cache_refinitiv_stories,
    digests_on_disk,
    headlines_needing_bodies,
    read_progress,
    rename_legacy_story_files,
)
from webapp.api import data_explorer as de  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default=de.DEFAULT_START)
    parser.add_argument("--end", default=de.DEFAULT_END)
    parser.add_argument("--limit", type=int, default=None, help="Max new stories to fetch this run")
    parser.add_argument("--sleep", type=float, default=0.25, help="Seconds between live fetches")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if a body file exists")
    parser.add_argument("--max-failures", type=int, default=50)
    parser.add_argument("--status", action="store_true", help="Print progress JSON and exit")
    parser.add_argument(
        "--rename-legacy",
        action="store_true",
        help="Rename existing slug--digest.txt files to include article timestamps, then exit",
    )
    args = parser.parse_args()

    ticker = str(args.ticker).upper().strip()
    if args.status:
        progress = read_progress(PROJECT_ROOT, ticker)
        print(progress or {"ticker": ticker, "status": "no progress file"})
        return 0

    cached = de.load_cached(ticker, args.start, args.end)
    if cached is None:
        raise SystemExit(
            f"No local Data Explorer cache for {ticker} {args.start}→{args.end}. "
            "Load data in the webapp / batch pipeline first."
        )
    news = cached["providers"].get("refinitiv", {}).get("news")
    if news is None or news.empty:
        raise SystemExit(f"No Refinitiv headlines in cache for {ticker}.")

    if args.rename_legacy:
        result = rename_legacy_story_files(PROJECT_ROOT, ticker, news)
        print(
            f"Renamed {result['renamed']} · skipped {result['skipped']} · "
            f"no headline meta {result['missing_meta']}",
            flush=True,
        )
        return 0

    have = digests_on_disk(PROJECT_ROOT, ticker)
    pending = headlines_needing_bodies(news, PROJECT_ROOT, ticker, force=args.force)
    print(
        f"{ticker}: {len(news):,} headlines · {len(have):,} bodies on disk · "
        f"{len(pending):,} still missing"
        + (f" · fetching up to {args.limit}" if args.limit else ""),
        flush=True,
    )
    if pending.empty and not args.force:
        print("Nothing to fetch.", flush=True)
        return 0

    def on_progress(payload: dict) -> None:
        processed = payload.get("processed", 0)
        pending_n = payload.get("pending_this_run", 0)
        if processed == 0 or processed % 25 == 0 or payload.get("status") != "running":
            print(
                f"  [{payload.get('status')}] "
                f"{payload.get('pct_run', 0)}% run · {payload.get('pct_overall', 0)}% overall · "
                f"fetched={payload.get('fetched')} failed={payload.get('failed')} "
                f"processed={processed}/{pending_n} "
                f"{payload.get('rate_per_min', 0)}/min · ETA {payload.get('eta_human', '—')} · "
                f"{payload.get('bytes_human', '?')} on disk",
                flush=True,
            )

    summary = cache_refinitiv_stories(
        PROJECT_ROOT,
        ticker,
        news,
        force=args.force,
        limit=args.limit,
        sleep_s=args.sleep,
        max_failures=args.max_failures,
        progress_callback=on_progress,
        window_start=args.start,
        window_end=args.end,
    )
    print(
        f"Done: fetched={summary['fetched']} failed={summary['failed']} "
        f"elapsed={summary['elapsed_s']}s → {summary['story_dir']}",
        flush=True,
    )
    return 0 if summary["failed"] == 0 or summary["fetched"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
