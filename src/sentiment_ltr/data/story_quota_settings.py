"""Persisted Workspace story-quota settings (shared by CLI, cron, and webapp)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SETTINGS_RELPATH = Path("app_data") / "story_quota_settings.json"
DEFAULT_MAX_PER_DAY = 9800
DEFAULT_MIN_SLEEP_S = 0.3
DEFAULT_CRON_MINUTE = 5
CRON_MARKER = "run_daily_story_quota.py"
MAX_PER_DAY_MIN = 100
MAX_PER_DAY_MAX = 10000


@dataclass(frozen=True)
class StoryQuotaSettings:
    max_per_day: int = DEFAULT_MAX_PER_DAY
    min_sleep_s: float = DEFAULT_MIN_SLEEP_S
    cron_enabled: bool = True
    cron_minute: int = DEFAULT_CRON_MINUTE
    updated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def settings_path(project_root: Path) -> Path:
    return Path(project_root) / SETTINGS_RELPATH


def _clamp_max_per_day(value: object) -> int:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        n = DEFAULT_MAX_PER_DAY
    return max(MAX_PER_DAY_MIN, min(MAX_PER_DAY_MAX, n))


def _clamp_cron_minute(value: object) -> int:
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        n = DEFAULT_CRON_MINUTE
    return max(0, min(59, n))


def _clamp_min_sleep(value: object) -> float:
    try:
        n = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        n = DEFAULT_MIN_SLEEP_S
    return max(0.05, min(5.0, n))


def normalize_settings(
    *,
    max_per_day: object = DEFAULT_MAX_PER_DAY,
    min_sleep_s: object = DEFAULT_MIN_SLEEP_S,
    cron_enabled: object = True,
    cron_minute: object = DEFAULT_CRON_MINUTE,
    updated_at: str | None = None,
) -> StoryQuotaSettings:
    return StoryQuotaSettings(
        max_per_day=_clamp_max_per_day(max_per_day),
        min_sleep_s=_clamp_min_sleep(min_sleep_s),
        cron_enabled=bool(cron_enabled),
        cron_minute=_clamp_cron_minute(cron_minute),
        updated_at=updated_at,
    )


def load_story_quota_settings(project_root: Path) -> StoryQuotaSettings:
    path = settings_path(project_root)
    if not path.is_file():
        return StoryQuotaSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return StoryQuotaSettings()
    if not isinstance(raw, dict):
        return StoryQuotaSettings()
    return normalize_settings(
        max_per_day=raw.get("max_per_day", DEFAULT_MAX_PER_DAY),
        min_sleep_s=raw.get("min_sleep_s", DEFAULT_MIN_SLEEP_S),
        cron_enabled=raw.get("cron_enabled", True),
        cron_minute=raw.get("cron_minute", DEFAULT_CRON_MINUTE),
        updated_at=str(raw.get("updated_at") or "") or None,
    )


def save_story_quota_settings(
    project_root: Path,
    settings: StoryQuotaSettings,
) -> StoryQuotaSettings:
    stamped = normalize_settings(
        max_per_day=settings.max_per_day,
        min_sleep_s=settings.min_sleep_s,
        cron_enabled=settings.cron_enabled,
        cron_minute=settings.cron_minute,
        updated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    path = settings_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stamped.as_dict(), indent=2) + "\n", encoding="utf-8")
    return stamped


def default_max_per_day(project_root: Path) -> int:
    return int(load_story_quota_settings(project_root).max_per_day)


def default_min_sleep_s(project_root: Path) -> float:
    return float(load_story_quota_settings(project_root).min_sleep_s)


def _python_executable() -> str:
    return os.environ.get("SENTIMENT_LTR_PYTHON") or sys.executable


def cron_command_line(project_root: Path, settings: StoryQuotaSettings | None = None) -> str:
    cfg = settings or load_story_quota_settings(project_root)
    root = Path(project_root).resolve()
    python = _python_executable()
    script = root / "scripts" / "run_daily_story_quota.py"
    log = root / "data" / "raw" / "data_explorer_full_stories" / "_daily_quota.log"
    return (
        f"{int(cfg.cron_minute)} * * * * cd \"{root}\" && "
        f"/usr/bin/caffeinate -is \"{python}\" \"{script}\" "
        f">> \"{log}\" 2>&1"
    )


def read_crontab() -> str:
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        # Empty crontab often exits 1 with "no crontab for user"
        return ""
    return result.stdout or ""


def installed_story_quota_cron_line() -> str | None:
    for line in read_crontab().splitlines():
        if CRON_MARKER in line and not line.strip().startswith("#"):
            return line
    return None


def write_crontab(lines: list[str]) -> None:
    body = "\n".join(line for line in lines if line.strip()) + ("\n" if lines else "")
    subprocess.run(
        ["crontab", "-"],
        input=body,
        check=True,
        text=True,
        capture_output=True,
    )


def apply_story_quota_cron(
    project_root: Path,
    settings: StoryQuotaSettings | None = None,
) -> dict[str, Any]:
    """Install or remove the hourly story-quota cron line to match settings."""
    cfg = settings or load_story_quota_settings(project_root)
    existing = [
        line
        for line in read_crontab().splitlines()
        if CRON_MARKER not in line
    ]
    installed: str | None = None
    if cfg.cron_enabled:
        installed = cron_command_line(project_root, cfg)
        existing.append(installed)
        log_path = (
            Path(project_root).resolve()
            / "data"
            / "raw"
            / "data_explorer_full_stories"
            / "_daily_quota.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
    write_crontab(existing)
    return {
        "cron_enabled": bool(cfg.cron_enabled),
        "cron_line": installed,
        "cron_installed": installed_story_quota_cron_line(),
    }


def parse_cron_minute(line: str | None) -> int | None:
    if not line:
        return None
    match = re.match(r"^\s*(\d+)\s+\*\s+\*\s+\*\s+\*", line)
    if not match:
        return None
    return _clamp_cron_minute(match.group(1))
