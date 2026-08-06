#!/usr/bin/env python
"""Backfill Refinitiv headlines for news-threshold tickers missing a usable cache.

Shares the Workspace news quota with ``get_story``. Skips automatically when a
full-story pull is already running unless ``--ignore-live-story`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from sentiment_ltr.data.batch_universe import (  # noqa: E402
    DEFAULT_STORY_QUEUE,
    MIN_USABLE_HEADLINES,
    PAPER_END,
    PAPER_START,
    headline_counts_by_ticker,
    news_threshold_universe,
    scan_batch_ticker_coverage,
    sync_story_queue_file,
    tickers_needing_headline_backfill,
)
from sentiment_ltr.data.news_coverage import build_news_coverage_result  # noqa: E402
from sentiment_ltr.data.story_quota_scheduler import live_story_pull_tickers  # noqa: E402


def _log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def _write_headline_cache(cache_dir: Path, news, daily) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    news.to_parquet(cache_dir / "refinitiv_news.parquet", index=False)
    daily.to_parquet(cache_dir / "refinitiv_news_daily_counts.parquet", index=False)
    note_path = cache_dir / "refinitiv_news_backfill.json"
    note_path.write_text(
        json.dumps(
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "headline_rows": int(len(news)),
                "daily_rows": int(len(daily)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--by-ticker-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "by_ticker",
    )
    parser.add_argument("--start", default=PAPER_START)
    parser.add_argument("--end", default=PAPER_END)
    parser.add_argument("--limit", type=int, default=0, help="Max tickers this run (0 = all)")
    parser.add_argument("--sleep", type=float, default=0.3, help="Pause between month chunks")
    parser.add_argument("--min-headlines", type=int, default=MIN_USABLE_HEADLINES)
    parser.add_argument("--ignore-live-story", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sync-story-queue", action="store_true", default=True)
    parser.add_argument("--no-sync-story-queue", action="store_false", dest="sync_story_queue")
    parser.add_argument(
        "--story-queue",
        type=Path,
        default=PROJECT_ROOT / DEFAULT_STORY_QUEUE,
    )
    args = parser.parse_args()

    live = live_story_pull_tickers(PROJECT_ROOT)
    if live and not args.ignore_live_story:
        _log(f"skip: story pull already running for {', '.join(live)} (shared news quota)")
        return 0

    coverage = scan_batch_ticker_coverage(args.by_ticker_dir)
    needed = tickers_needing_headline_backfill(
        coverage,
        min_headlines=args.min_headlines,
    )
    if args.limit and args.limit > 0:
        needed = needed.head(int(args.limit)).copy()
    _log(
        f"headline backfill candidates={len(needed):,} "
        f"min_headlines={args.min_headlines} dry_run={args.dry_run}"
    )
    if needed.empty:
        return 0

    for idx, row in needed.iterrows():
        ticker = str(row["ticker"]).upper()
        cache_dir = Path(str(row["cache_dir"]))
        _log(
            f"{idx + 1}/{len(needed)} {ticker} rank={row.get('volume_rank')} "
            f"existing_headlines={int(row.get('refinitiv_headline_rows') or 0)}"
        )
        if args.dry_run:
            continue
        try:
            news, daily, summary, ric = build_news_coverage_result(
                PROJECT_ROOT,
                ticker,
                args.start,
                args.end,
                sleep_s=float(args.sleep),
            )
        except Exception as exc:
            _log(f"failed {ticker}: {exc}")
            continue
        _write_headline_cache(cache_dir, news, daily)
        _log(
            f"saved {ticker} ric={ric} headlines={summary.total_articles:,} "
            f"avg/week={summary.avg_articles_per_week:.2f}"
        )
        if args.sync_story_queue:
            updated = scan_batch_ticker_coverage(args.by_ticker_dir)
            sync_story_queue_file(
                args.story_queue,
                headline_counts_by_ticker(updated),
                universe=news_threshold_universe(updated),
            )
        time.sleep(max(0.0, float(args.sleep)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
