#!/usr/bin/env python
"""Spend today's Workspace story quota (default from story-quota settings).

Intended for cron / launchd just after midnight and as an hourly watchdog:

  caffeinate -is python scripts/run_daily_story_quota.py

Skips if a pull is already running or the daily cap is exhausted. Walks
``app_data/story_pull_queue.txt`` in order and fills one ticker at a time
until today's remaining quota is used. Tune ``max_per_day`` / cron under
Many stocks → 2F, or edit ``app_data/story_quota_settings.json``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from sentiment_ltr.data.story_quota_scheduler import (  # noqa: E402
    live_story_pull_tickers,
    load_ticker_queue,
    next_ticker_needing_bodies,
    remaining_daily_quota,
    story_quota_snapshot,
)
from sentiment_ltr.data.story_quota_settings import (  # noqa: E402
    load_story_quota_settings,
)
from webapp.api import data_explorer as de  # noqa: E402

DEFAULT_QUEUE = PROJECT_ROOT / "app_data" / "story_pull_queue.txt"
PYTHON = sys.executable
CACHE_SCRIPT = PROJECT_ROOT / "scripts" / "cache_refinitiv_full_stories.py"


def _log(message: str) -> None:
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def _load_news(ticker: str, start: str, end: str):
    cached = de.load_cached(ticker, start, end)
    if cached is None:
        return None
    return cached.get("providers", {}).get("refinitiv", {}).get("news")


def main() -> int:
    cfg = load_story_quota_settings(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--start", default=de.DEFAULT_START)
    parser.add_argument("--end", default=de.DEFAULT_END)
    parser.add_argument(
        "--max-per-day",
        type=int,
        default=None,
        help=f"Daily get_story budget (default from settings: {cfg.max_per_day})",
    )
    parser.add_argument(
        "--min-sleep",
        type=float,
        default=None,
        help=f"Cruise sleep between calls (default from settings: {cfg.min_sleep_s})",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    max_per_day = int(cfg.max_per_day if args.max_per_day is None else args.max_per_day)
    min_sleep = float(cfg.min_sleep_s if args.min_sleep is None else args.min_sleep)

    os.chdir(PROJECT_ROOT)
    live = live_story_pull_tickers(PROJECT_ROOT)
    if live:
        _log(f"skip: pull already running for {', '.join(live)}")
        return 0

    snap = story_quota_snapshot(PROJECT_ROOT, max_per_day=max_per_day)
    remaining = int(snap.remaining)
    _log(
        f"quota remaining={remaining:,}/{max_per_day:,} "
        f"calendar={snap.calendar_used:,} rolling24h={snap.rolling_used:,} "
        f"blocking={snap.blocking or 'none'} wait_until={snap.wait_until_local or '-'}"
    )
    if remaining <= 0:
        _log("skip: calendar-day or rolling-24h cap is exhausted")
        return 0

    queue = load_ticker_queue(args.queue)
    if not queue:
        _log(f"skip: empty queue at {args.queue}")
        return 0

    def news_loader(ticker: str):
        return _load_news(ticker, args.start, args.end)

    while remaining > 0:
        ticker = next_ticker_needing_bodies(
            queue, news_loader=news_loader, project_root=PROJECT_ROOT
        )
        if ticker is None:
            _log("done: no queued ticker still needs bodies")
            return 0
        cmd = [
            PYTHON,
            str(CACHE_SCRIPT),
            "--ticker", ticker,
            "--start", args.start,
            "--end", args.end,
            "--pacer", "thompson",
            "--min-sleep", str(min_sleep),
            "--sleep", str(min_sleep),
            "--max-sleep", "180",
            "--max-per-day", str(max_per_day),
            "--limit", str(remaining),
            "--cooloff-after-rl", "6",
            "--cooloff-s", "900",
            "--rate-limit-retries", "60",
            "--max-failures", "200",
        ]
        _log(f"start {ticker} limit={remaining:,}: {' '.join(cmd)}")
        if args.dry_run:
            return 0
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        _log(f"finished {ticker} exit={result.returncode}")
        remaining = remaining_daily_quota(PROJECT_ROOT, max_per_day=max_per_day)
        _log(f"remaining quota today: {remaining:,}")
        if result.returncode != 0 and remaining > 0:
            # Avoid a tight crash loop inside one cron tick.
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
