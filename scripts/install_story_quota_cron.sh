#!/bin/bash
# Install (or refresh) the hourly watchdog cron entry for story-quota pulls.
# Uses app_data/story_quota_settings.json (editable under Many stocks → 2F).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${SENTIMENT_LTR_PYTHON:-/Users/armandoordoricadelatorre/miniconda/envs/sentiment-ltr-paper/bin/python}"
export SENTIMENT_LTR_PYTHON="$PYTHON"
cd "$ROOT"
"$PYTHON" - <<PY
from pathlib import Path
import sys
ROOT = Path("$ROOT")
sys.path.insert(0, str(ROOT / "src"))
from sentiment_ltr.data.story_quota_settings import (
    apply_story_quota_cron,
    load_story_quota_settings,
    normalize_settings,
    save_story_quota_settings,
)

cfg = load_story_quota_settings(ROOT)
if not cfg.cron_enabled:
    cfg = save_story_quota_settings(
        ROOT,
        normalize_settings(
            max_per_day=cfg.max_per_day,
            min_sleep_s=cfg.min_sleep_s,
            cron_enabled=True,
            cron_minute=cfg.cron_minute,
        ),
    )
info = apply_story_quota_cron(ROOT, cfg)
print("Settings:", cfg.as_dict())
print("Installed cron line:")
print(" ", info.get("cron_line") or info.get("cron_installed") or "(none)")
PY
echo
echo "Current crontab:"
crontab -l
echo
echo "Keep the Mac awake + Workspace signed in. Cron on macOS may need"
echo "Full Disk Access for /usr/sbin/cron in System Settings → Privacy & Security."
