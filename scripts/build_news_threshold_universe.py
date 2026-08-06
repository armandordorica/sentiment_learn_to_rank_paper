#!/usr/bin/env python
"""Build the RavenPack ≥1 article/week universe from the 2B top-1k cache.

Writes ``app_data/ravenpack_news_threshold_universe.csv`` and optionally refreshes
``app_data/story_pull_queue.txt`` with the full news-threshold universe (volume-rank
order). Cron skips names that still lack usable Refinitiv headlines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sentiment_ltr.data.batch_universe import (  # noqa: E402
    DEFAULT_STORY_QUEUE,
    DEFAULT_UNIVERSE_CSV,
    headline_counts_by_ticker,
    scan_batch_ticker_coverage,
    sync_story_queue_file,
    write_news_threshold_universe,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--by-ticker-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "by_ticker",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_UNIVERSE_CSV,
    )
    parser.add_argument("--min-avg-per-week", type=float, default=1.0)
    parser.add_argument("--sync-story-queue", action="store_true")
    parser.add_argument(
        "--story-queue",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_STORY_QUEUE,
    )
    args = parser.parse_args()

    universe = write_news_threshold_universe(
        args.by_ticker_dir,
        args.output,
        min_avg_per_week=args.min_avg_per_week,
    )
    coverage = scan_batch_ticker_coverage(args.by_ticker_dir)
    usable = int(universe["has_usable_refinitiv_headlines"].sum()) if not universe.empty else 0
    print(
        f"wrote {len(universe):,} news-threshold names to {args.output} "
        f"({usable:,} already have usable Refinitiv headlines)",
        flush=True,
    )
    if args.sync_story_queue:
        tickers = sync_story_queue_file(
            args.story_queue,
            headline_counts_by_ticker(coverage),
            universe=universe,
        )
        ready = int(universe["has_usable_refinitiv_headlines"].sum()) if not universe.empty else 0
        print(
            f"story queue {args.story_queue}: {len(tickers):,} tickers "
            f"({ready:,} ready with headlines; cron skips the rest until backfilled) "
            f"({', '.join(tickers[:8])}{'…' if len(tickers) > 8 else ''})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
