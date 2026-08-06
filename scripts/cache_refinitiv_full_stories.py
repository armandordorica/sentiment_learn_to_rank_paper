#!/usr/bin/env python
"""Cache Refinitiv full story bodies for a ticker/window (resumable).

Example (AAPL paper window):

  python scripts/cache_refinitiv_full_stories.py --ticker AAPL \\
      --start 2003-01-01 --end 2014-12-31

Overnight (cool off on 429 storms, restart sessions until done):

  caffeinate -is python scripts/cache_refinitiv_full_stories.py --ticker AAPL \\
      --start 2003-01-01 --end 2014-12-31 --overnight

Skips stories already under data/raw/data_explorer_full_stories/{TICKER}/.
Use --limit 20 for a smoke test. Progress: .../{TICKER}/_pull_progress.json
"""

from __future__ import annotations

import argparse
import sys
import time
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
from sentiment_ltr.data.story_quota_settings import (  # noqa: E402
    load_story_quota_settings,
)
from webapp.api import data_explorer as de  # noqa: E402


def main() -> int:
    cfg = load_story_quota_settings(PROJECT_ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default=de.DEFAULT_START)
    parser.add_argument("--end", default=de.DEFAULT_END)
    parser.add_argument("--limit", type=int, default=None, help="Max new stories to fetch this run")
    parser.add_argument("--sleep", type=float, default=0.5, help="Initial seconds between live fetches")
    parser.add_argument("--min-sleep", type=float, default=0.25, help="Adaptive floor (seconds)")
    parser.add_argument("--max-sleep", type=float, default=180.0, help="Adaptive ceiling on 429 backoff")
    parser.add_argument(
        "--pacer",
        choices=["thompson", "aimd", "fixed"],
        default="thompson",
        help="Retry wait policy: thompson (bandit RL), aimd, or fixed --sleep",
    )
    parser.add_argument(
        "--no-adaptive",
        action="store_true",
        help="Disable learned sleep adaptation (fixed --sleep only; alias for --pacer fixed)",
    )
    parser.add_argument("--force", action="store_true", help="Re-fetch even if a body file exists")
    parser.add_argument("--max-failures", type=int, default=50, help="Stop after N hard failures (429s excluded)")
    parser.add_argument(
        "--max-scope-failures",
        type=int,
        default=5,
        help="Stop early after N missing-scope errors (auth/entitlement)",
    )
    parser.add_argument(
        "--rate-limit-retries",
        type=int,
        default=40,
        help="Per-story retries when LSEG returns HTTP 429",
    )
    parser.add_argument(
        "--cooloff-after-rl",
        type=int,
        default=6,
        help="After this many consecutive 429s, take a long cool-off pause",
    )
    parser.add_argument(
        "--cooloff-s",
        type=float,
        default=900.0,
        help="Seconds to pause during a cool-off (default 15 min)",
    )
    parser.add_argument(
        "--overnight",
        action="store_true",
        help="Keep restarting until the window is fully cached (or user halt)",
    )
    parser.add_argument(
        "--overnight-pause-s",
        type=float,
        default=1200.0,
        help="Between overnight sessions when still missing bodies (default 20 min)",
    )
    parser.add_argument(
        "--max-per-day",
        type=int,
        default=None,
        help="Pause after this many successful get_story calls today across all tickers "
        f"(default from settings: {cfg.max_per_day}; 0 disables the cap).",
    )
    parser.add_argument("--status", action="store_true", help="Print progress JSON and exit")
    parser.add_argument(
        "--rename-legacy",
        action="store_true",
        help="Rename existing slug--digest.txt files to include article timestamps, then exit",
    )
    args = parser.parse_args()
    max_per_day = int(cfg.max_per_day if args.max_per_day is None else args.max_per_day)

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

    def on_progress(payload: dict) -> None:
        processed = payload.get("processed", 0)
        pending_n = payload.get("pending_this_run", 0)
        last = payload.get("last_status")
        noisy = last in {"rate_limited", "cooling_off", "daily_quota", "overnight_pause"}
        if processed == 0 or processed % 25 == 0 or payload.get("status") != "running" or noisy:
            extra = ""
            if noisy:
                extra = (
                    f" · {last} wait={payload.get('waiting_s', '?')}s"
                    f" attempt={payload.get('rate_limit_attempt', '?')}"
                )
            print(
                f"  [{payload.get('status')}] "
                f"{payload.get('pct_run', 0)}% run · {payload.get('pct_overall', 0)}% overall · "
                f"fetched={payload.get('fetched')} failed={payload.get('failed')} "
                f"processed={processed}/{pending_n} "
                f"{payload.get('rate_per_min', 0)}/min · ETA {payload.get('eta_human', '—')} · "
                f"sleep={payload.get('sleep_s', '?')}s · prefer={payload.get('preferred_sleep_s', '?')}s · "
                f"policy={payload.get('pacer_policy', '?')} · rl_hits={payload.get('rate_limit_hits', 0)}"
                f"{extra} · {payload.get('bytes_human', '?')} on disk",
                flush=True,
            )

    pacer_policy = "fixed" if args.no_adaptive else args.pacer
    session = 0
    last_summary: dict | None = None
    while True:
        session += 1
        have = digests_on_disk(PROJECT_ROOT, ticker)
        pending = headlines_needing_bodies(news, PROJECT_ROOT, ticker, force=args.force)
        print(
            f"{ticker}: session {session} · {len(news):,} headlines · {len(have):,} bodies on disk · "
            f"{len(pending):,} still missing"
            + (f" · fetching up to {args.limit}" if args.limit else ""),
            flush=True,
        )
        if pending.empty and not args.force:
            print("Nothing to fetch.", flush=True)
            return 0

        summary = cache_refinitiv_stories(
            PROJECT_ROOT,
            ticker,
            news,
            force=args.force,
            limit=args.limit,
            sleep_s=args.sleep,
            min_sleep_s=args.min_sleep,
            max_sleep_s=args.max_sleep,
            adaptive=pacer_policy != "fixed",
            pacer_policy=pacer_policy,
            max_failures=args.max_failures,
            max_scope_failures=args.max_scope_failures,
            rate_limit_retries=args.rate_limit_retries,
            cooloff_after_rl=args.cooloff_after_rl,
            cooloff_s=args.cooloff_s,
            max_requests_per_day=None if max_per_day <= 0 else max_per_day,
            progress_callback=on_progress,
            window_start=args.start,
            window_end=args.end,
        )
        last_summary = summary
        print(
            f"Done session {session}: status={summary.get('status')} "
            f"fetched={summary['fetched']} failed={summary['failed']} "
            f"elapsed={summary['elapsed_s']}s → {summary['story_dir']}",
            flush=True,
        )

        if summary.get("status") in {"paused", "halted"}:
            print(f"Stopped by user ({summary.get('stop_reason')}).", flush=True)
            return 0
        if summary.get("status") == "completed" or (
            not args.force and headlines_needing_bodies(news, PROJECT_ROOT, ticker).empty
        ):
            return 0 if (summary["failed"] == 0 or summary["fetched"] > 0) else 1
        if not args.overnight:
            return 0 if summary["failed"] == 0 or summary["fetched"] > 0 else 1

        pause = max(60.0, float(args.overnight_pause_s))
        print(
            f"Overnight: still missing bodies — sleeping {pause:.0f}s before next session "
            "(Halt via UI/control file to stop).",
            flush=True,
        )
        from datetime import datetime, timezone

        from sentiment_ltr.data.refinitiv_story_cache import (
            _write_progress,
            clear_pull_control,
            read_pull_control,
        )

        deadline = time.monotonic() + pause
        while time.monotonic() < deadline:
            prog = read_progress(PROJECT_ROOT, ticker) or {
                "ticker": ticker,
                "window_start": args.start,
                "window_end": args.end,
            }
            prog.update(
                {
                    "status": "running",
                    "last_status": "overnight_pause",
                    "waiting_s": round(max(0.0, deadline - time.monotonic()), 1),
                    "stop_reason": None,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            _write_progress(PROJECT_ROOT, ticker, prog)
            ctrl = read_pull_control(PROJECT_ROOT, ticker) or {}
            action = str(ctrl.get("action") or "").lower()
            if action in {"pause", "halt", "stop"}:
                clear_pull_control(PROJECT_ROOT, ticker)
                print(f"Overnight stopped by user ({action}).", flush=True)
                return 0
            time.sleep(min(10.0, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    raise SystemExit(main())
