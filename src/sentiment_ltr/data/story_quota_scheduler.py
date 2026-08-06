"""Pick the next full-story ticker under shared Workspace quota windows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from sentiment_ltr.data.refinitiv_story_cache import (
    full_story_root,
    headlines_needing_bodies,
    load_story_fetch_times,
    read_pull_pid,
)


from sentiment_ltr.data.story_quota_settings import DEFAULT_MAX_PER_DAY


def load_ticker_queue(path: Path) -> list[str]:
    if not path.is_file():
        return []
    tickers: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip().upper()
        if not line or line in seen:
            continue
        seen.add(line)
        tickers.append(line)
    return tickers


@dataclass(frozen=True)
class StoryQuotaSnapshot:
    max_per_window: int
    calendar_used: int
    rolling_used: int
    remaining: int
    remaining_calendar: int
    remaining_rolling: int
    blocking: str | None
    wait_s: float
    wait_until_local: str | None

    def as_progress(self) -> dict[str, object]:
        return {
            "requests_today": int(self.calendar_used),
            "requests_rolling_24h": int(self.rolling_used),
            "remaining_quota": int(self.remaining),
            "max_requests_per_day": int(self.max_per_window),
            "quota_blocking": self.blocking,
            "waiting_s": round(float(self.wait_s), 1) if self.wait_s else 0.0,
            "quota_wait_until": self.wait_until_local,
        }


def quota_snapshot_from_times(
    times: list[datetime],
    *,
    max_per_day: int = DEFAULT_MAX_PER_DAY,
    now: datetime | None = None,
    rolling_hours: float = 24.0,
) -> StoryQuotaSnapshot:
    """Remaining calls under *both* local calendar-day and rolling 24h caps.

    LSEG documents ~10k requests/day without saying calendar vs rolling. We take
    the tighter of the two so we do not 429-loop after either window is spent.
    """
    current = (now or datetime.now().astimezone()).astimezone()
    cap = max(0, int(max_per_day))
    calendar_used = sum(1 for dt in times if dt.astimezone().date() == current.date())
    cutoff = current - timedelta(hours=float(rolling_hours))
    rolling = sorted(dt.astimezone() for dt in times if dt.astimezone() > cutoff)
    rolling_used = len(rolling)
    rem_cal = max(0, cap - calendar_used)
    rem_roll = max(0, cap - rolling_used)
    remaining = min(rem_cal, rem_roll)
    if remaining > 0 or cap == 0:
        return StoryQuotaSnapshot(
            max_per_window=cap,
            calendar_used=calendar_used,
            rolling_used=rolling_used,
            remaining=remaining if cap else 10**9,
            remaining_calendar=rem_cal if cap else 10**9,
            remaining_rolling=rem_roll if cap else 10**9,
            blocking=None,
            wait_s=0.0,
            wait_until_local=None,
        )

    waits: list[tuple[str, float, datetime]] = []
    if rem_cal <= 0:
        nxt_midnight = (current + timedelta(days=1)).replace(
            hour=0, minute=0, second=5, microsecond=0
        )
        cal_wait = max(1.0, (nxt_midnight - current).total_seconds())
        waits.append(("calendar_day", cal_wait, nxt_midnight))
    if rem_roll <= 0:
        drop = rolling_used - cap + 1
        idx = min(max(0, drop - 1), len(rolling) - 1)
        ready_at = rolling[idx] + timedelta(hours=float(rolling_hours), seconds=5)
        waits.append(
            (
                "rolling_24h",
                max(1.0, (ready_at - current).total_seconds()),
                ready_at,
            )
        )
    binding_name, wait_s, ready_at = max(waits, key=lambda item: item[1])
    return StoryQuotaSnapshot(
        max_per_window=cap,
        calendar_used=calendar_used,
        rolling_used=rolling_used,
        remaining=0,
        remaining_calendar=rem_cal,
        remaining_rolling=rem_roll,
        blocking=binding_name,
        wait_s=float(wait_s),
        wait_until_local=ready_at.astimezone().isoformat(timespec="seconds"),
    )


def story_quota_snapshot(
    project_root: Path,
    *,
    max_per_day: int = DEFAULT_MAX_PER_DAY,
    now: datetime | None = None,
    rolling_hours: float = 24.0,
) -> StoryQuotaSnapshot:
    return quota_snapshot_from_times(
        load_story_fetch_times(project_root),
        max_per_day=max_per_day,
        now=now,
        rolling_hours=rolling_hours,
    )


def remaining_daily_quota(project_root: Path, *, max_per_day: int = DEFAULT_MAX_PER_DAY) -> int:
    return int(story_quota_snapshot(project_root, max_per_day=max_per_day).remaining)


def pid_is_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False
    return True


def live_story_pull_tickers(project_root: Path) -> list[str]:
    root = full_story_root(project_root)
    if not root.is_dir():
        return []
    live: list[str] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        pid = read_pull_pid(project_root, path.name)
        if pid is not None and pid_is_alive(pid):
            live.append(path.name.upper())
    return live


def next_ticker_needing_bodies(
    queue: list[str],
    *,
    news_loader: Callable[[str], object],
    project_root: Path,
) -> str | None:
    """Return the first queued ticker that still has uncached story bodies."""
    import pandas as pd

    for ticker in queue:
        news = news_loader(str(ticker).upper())
        if news is None or not isinstance(news, pd.DataFrame) or news.empty:
            continue
        pending = headlines_needing_bodies(news, project_root, ticker)
        if not pending.empty:
            return str(ticker).upper()
    return None
