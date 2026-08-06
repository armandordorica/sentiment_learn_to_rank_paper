from __future__ import annotations

import json

from sentiment_ltr.data.story_quota_settings import (
    apply_story_quota_cron,
    cron_command_line,
    load_story_quota_settings,
    normalize_settings,
    parse_cron_minute,
    save_story_quota_settings,
)


def test_normalize_clamps_bounds():
    cfg = normalize_settings(max_per_day=50_000, cron_minute=99, min_sleep_s=0.01)
    assert cfg.max_per_day == 10000
    assert cfg.cron_minute == 59
    assert cfg.min_sleep_s == 0.05


def test_save_and_load_roundtrip(tmp_path):
    saved = save_story_quota_settings(
        tmp_path,
        normalize_settings(max_per_day=9800, min_sleep_s=0.3, cron_enabled=True, cron_minute=7),
    )
    path = tmp_path / "app_data" / "story_quota_settings.json"
    assert path.is_file()
    loaded = load_story_quota_settings(tmp_path)
    assert loaded.max_per_day == 9800
    assert loaded.cron_minute == 7
    assert loaded.cron_enabled is True
    assert saved.updated_at


def test_cron_line_uses_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTIMENT_LTR_PYTHON", "/tmp/fake-python")
    cfg = normalize_settings(max_per_day=9800, cron_minute=12, cron_enabled=True)
    line = cron_command_line(tmp_path, cfg)
    assert line.startswith("12 * * * *")
    assert "run_daily_story_quota.py" in line
    assert "/tmp/fake-python" in line
    assert parse_cron_minute(line) == 12


def test_apply_cron_install_and_remove(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    state = {"lines": ["0 9 * * * echo keep-me"]}

    def fake_run(cmd, **kwargs):
        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        if cmd[:2] == ["crontab", "-l"]:
            Result.stdout = "\n".join(state["lines"]) + "\n"
            return Result()
        if cmd[:2] == ["crontab", "-"]:
            body = kwargs.get("input") or ""
            state["lines"] = [ln for ln in body.splitlines() if ln.strip()]
            calls.append(cmd)
            return Result()
        raise AssertionError(cmd)

    monkeypatch.setattr(
        "sentiment_ltr.data.story_quota_settings.subprocess.run",
        fake_run,
    )
    cfg = normalize_settings(cron_enabled=True, cron_minute=5)
    info = apply_story_quota_cron(tmp_path, cfg)
    assert info["cron_enabled"] is True
    assert any("run_daily_story_quota.py" in ln for ln in state["lines"])
    assert any("echo keep-me" in ln for ln in state["lines"])

    off = normalize_settings(cron_enabled=False, cron_minute=5)
    apply_story_quota_cron(tmp_path, off)
    assert not any("run_daily_story_quota.py" in ln for ln in state["lines"])
    assert any("echo keep-me" in ln for ln in state["lines"])
