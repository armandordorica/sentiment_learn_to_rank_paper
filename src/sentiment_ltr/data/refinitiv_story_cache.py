"""Cache Refinitiv full story bodies on disk for offline inspection.

Headlines live in parquet with ``storyId``; full wire text is only available via
``ld.news.get_story``. Bodies are saved under
``data/raw/data_explorer_full_stories/{TICKER}/`` as
``{YYYY-MM-DD_HHMMSS}_{headline-slug}--{digest}.txt`` so folders sort by article
time for manual window checks. A per-ticker manifest keeps pulls resumable.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from sentiment_ltr.data.refinitiv_queries import fetch_refinitiv_story

PROJECT_ROOT_DEFAULT = Path(__file__).resolve().parents[3]
FULL_STORY_DIR_NAME = Path("data") / "raw" / "data_explorer_full_stories"


def full_story_root(project_root: Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else PROJECT_ROOT_DEFAULT
    return root / FULL_STORY_DIR_NAME


def story_digest(story_id: str) -> str:
    return hashlib.sha256(str(story_id).encode("utf-8")).hexdigest()[:12]


def format_story_timestamp(story_time: Any) -> str:
    """Filesystem-safe stamp for sorting / manual window checks (``YYYY-MM-DD_HHMMSS``)."""
    if story_time is None:
        return "undated"
    try:
        if isinstance(story_time, float) and pd.isna(story_time):
            return "undated"
        ts = pd.Timestamp(story_time)
    except Exception:
        return "undated"
    if pd.isna(ts):
        return "undated"
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.strftime("%Y-%m-%d_%H%M%S")


def story_filename(
    story_id: str,
    headline: str,
    story_time: Any = None,
) -> str:
    """Name on disk: ``{stamp}_{slug}--{digest}.txt`` (stamp sorts by article time)."""
    stamp = format_story_timestamp(story_time)
    slug = re.sub(r"[^a-z0-9]+", "-", str(headline or "").lower()).strip("-")[:80] or "refinitiv-story"
    return f"{stamp}_{slug}--{story_digest(story_id)}.txt"


def story_path(
    project_root: Path,
    ticker: str,
    story_id: str,
    headline: str = "",
    story_time: Any = None,
) -> Path:
    clean = str(ticker or "UNKNOWN").upper().strip()
    return full_story_root(project_root) / clean / story_filename(
        story_id, headline, story_time=story_time
    )


def digest_from_story_filename(name: str) -> str:
    stem = Path(name).stem
    return stem.rsplit("--", 1)[-1] if "--" in stem else stem


def find_cached_story_path(
    project_root: Path,
    ticker: str,
    story_id: str,
) -> Path | None:
    """Locate an existing body by digest (works for legacy and timestamped names)."""
    digest = story_digest(story_id)
    directory = full_story_root(project_root) / str(ticker).upper().strip()
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"*--{digest}.txt"))
    return matches[0] if matches else None


def write_story_file(
    path: Path,
    *,
    story_id: str,
    headline: str,
    ticker: str,
    text: str,
    story_time: Any = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = format_story_timestamp(story_time)
    path.write_text(
        f"Headline: {headline}\nStory ID: {story_id}\nTicker: {ticker}\n"
        f"Date: {stamp}\n\n{text}\n",
        encoding="utf-8",
    )


def digests_on_disk(project_root: Path, ticker: str) -> set[str]:
    directory = full_story_root(project_root) / str(ticker).upper().strip()
    found: set[str] = set()
    if not directory.is_dir():
        return found
    for path in directory.glob("*.txt"):
        found.add(digest_from_story_filename(path.name))
    return found


def manifest_path(project_root: Path, ticker: str) -> Path:
    return full_story_root(project_root) / str(ticker).upper().strip() / "_manifest.jsonl"


def progress_path(project_root: Path, ticker: str) -> Path:
    return full_story_root(project_root) / str(ticker).upper().strip() / "_pull_progress.json"


def control_path(project_root: Path, ticker: str) -> Path:
    return full_story_root(project_root) / str(ticker).upper().strip() / "_pull_control.json"


def pid_path(project_root: Path, ticker: str) -> Path:
    return full_story_root(project_root) / str(ticker).upper().strip() / "_pull.pid"


def format_bytes(n: int | float | None) -> str:
    """Human-readable size (``1.2 MB``)."""
    if n is None:
        return "—"
    value = float(n)
    if value < 0:
        value = 0.0
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while value >= 1024.0 and idx < len(units) - 1:
        value /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(value)} {units[idx]}"
    return f"{value:.1f} {units[idx]}"


def story_dir_stats(project_root: Path, ticker: str) -> dict[str, Any]:
    """File count + byte size for one ticker's full-story folder."""
    ticker = str(ticker).upper().strip()
    directory = full_story_root(project_root) / ticker
    n_files = 0
    n_bytes = 0
    if directory.is_dir():
        for path in directory.glob("*.txt"):
            n_files += 1
            try:
                n_bytes += path.stat().st_size
            except OSError:
                pass
    return {
        "ticker": ticker,
        "files": n_files,
        "bytes": n_bytes,
        "bytes_human": format_bytes(n_bytes),
        "avg_bytes": int(n_bytes / n_files) if n_files else 0,
        "story_dir": str(directory),
    }


def list_story_cache_stats(project_root: Path) -> list[dict[str, Any]]:
    """Per-ticker storage summary under the full-story root (sorted by bytes desc)."""
    root = full_story_root(project_root)
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for path in sorted(root.iterdir()):
        if path.is_dir() and not path.name.startswith("."):
            rows.append(story_dir_stats(project_root, path.name))
    rows.sort(key=lambda r: (-int(r["bytes"]), r["ticker"]))
    return rows


def failed_digests_from_manifest(project_root: Path, ticker: str) -> set[str]:
    """Story digests whose latest manifest status is ``failed`` (and not later success)."""
    path = manifest_path(project_root, ticker)
    latest: dict[str, str] = {}
    if not path.exists():
        return set()
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                sid = str(row.get("story_id") or "").strip()
                if not sid:
                    continue
                latest[story_digest(sid)] = str(row.get("status") or "")
    except OSError:
        return set()
    return {d for d, status in latest.items() if status == "failed"}


def _day_shade_level(count: int) -> int:
    """Map a count to shade 1 (light) … 5 (dark) for green/red day cells."""
    n = max(0, int(count))
    if n <= 0:
        return 1
    if n == 1:
        return 1
    if n <= 3:
        return 2
    if n <= 7:
        return 3
    if n <= 15:
        return 4
    return 5


def build_story_day_grid(
    project_root: Path,
    ticker: str,
    news: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    current_date: str | None = None,
    pull_running: bool = False,
) -> dict[str, Any]:
    """Calendar day squares for a ticker's full-story pull window.

    Status codes:
      - gray: headlines remain without bodies (not finished)
      - red: every headline attempted; at least one failed; some still missing bodies
      - blue: no headlines that calendar day (nothing to load)
      - yellow: day currently being fetched
      - green: all headlines that day have bodies on disk (``n`` = distinct bodies)
    """
    ticker = str(ticker).upper().strip()
    if news is None or news.empty or "date" not in getattr(news, "columns", []):
        work = pd.DataFrame(columns=["date", "storyId"])
    else:
        work = news.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"])

    if start:
        start_ts = pd.Timestamp(start).normalize()
    elif not work.empty:
        start_ts = work["date"].min().normalize()
    else:
        start_ts = pd.Timestamp("2003-01-01")
    if end:
        end_ts = pd.Timestamp(end).normalize()
    elif not work.empty:
        end_ts = work["date"].max().normalize()
    else:
        end_ts = start_ts

    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    have = digests_on_disk(project_root, ticker)
    failed = failed_digests_from_manifest(project_root, ticker) - have

    by_day: dict[str, dict[str, int]] = {}
    if not work.empty and "storyId" in work.columns:
        work = work.copy()
        work["storyId"] = work["storyId"].astype(str).str.strip()
        work = work[work["storyId"].ne("") & work["storyId"].ne("nan")]
        work["day"] = work["date"].dt.strftime("%Y-%m-%d")
        work["digest"] = work["storyId"].map(story_digest)
        for day, group in work.groupby("day", sort=False):
            digests = set(group["digest"].tolist())
            n = len(digests)
            n_have = len(digests & have)
            n_failed = len(digests & failed)
            by_day[str(day)] = {
                "n": n,
                "have": n_have,
                "failed": n_failed,
                "pending": max(0, n - n_have - n_failed),
            }

    current = None
    if current_date:
        try:
            current = pd.Timestamp(current_date).strftime("%Y-%m-%d")
        except Exception:
            current = str(current_date)[:10]

    days: list[dict[str, Any]] = []
    counts = {"gray": 0, "red": 0, "blue": 0, "yellow": 0, "green": 0}
    cursor = start_ts
    one_day = pd.Timedelta(days=1)
    while cursor <= end_ts:
        key = cursor.strftime("%Y-%m-%d")
        info = by_day.get(key, {"n": 0, "have": 0, "failed": 0, "pending": 0})
        n = int(info["n"])
        have_n = int(info["have"])
        failed_n = int(info["failed"])
        pending_n = int(info["pending"])

        if current and key == current and pull_running:
            status = "yellow"
            status_label = "in progress"
        elif n == 0:
            status = "blue"
            status_label = "no articles that day"
        elif pending_n == 0 and failed_n == 0 and have_n >= n and n > 0:
            status = "green"
            status_label = "all articles loaded"
        elif pending_n == 0 and failed_n > 0:
            status = "red"
            status_label = "finished with failures"
        else:
            status = "gray"
            status_label = "not done yet"

        tried_n = have_n + failed_n
        if status == "green":
            shade = _day_shade_level(have_n)
            shade_metric = f"loaded articles: {have_n} → shade {shade}/5"
        elif status == "red":
            shade = _day_shade_level(failed_n)
            shade_metric = f"failed articles: {failed_n} → shade {shade}/5"
        else:
            shade = 0
            shade_metric = None

        weekday = cursor.strftime("%A")
        tip_lines = [
            f"{cursor.strftime('%B %Y')} · day {int(cursor.day)}",
            f"{key} ({weekday})",
            f"Status: {status_label}",
            f"Headlines that day: {n}",
            f"Tried: {tried_n}",
            f"Retrieved successfully: {have_n}",
            f"Failed: {failed_n}",
            f"Still to try: {pending_n}",
        ]
        if shade_metric:
            tip_lines.append(shade_metric)
        title = " · ".join(tip_lines)
        tip = "\n".join(tip_lines)
        # Pre-escape for HTML attributes; keep &#10; so CSS/JS see real line breaks.
        tip_attr = (
            tip.replace("&", "&amp;")
            .replace('"', "&quot;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "&#10;")
        )

        counts[status] = counts.get(status, 0) + 1
        days.append({
            "date": key,
            "year": int(cursor.year),
            "month": int(cursor.month),
            "day": int(cursor.day),
            "month_label": cursor.strftime("%b"),
            "month_full": cursor.strftime("%B %Y"),
            "weekday": weekday,
            "status": status,
            "status_label": status_label,
            "shade": shade,
            "n": have_n if status == "green" else n,
            "headlines": n,
            "tried": tried_n,
            "have": have_n,
            "failed": failed_n,
            "pending": pending_n,
            "title": title,
            "tip": tip,
            "tip_attr": tip_attr,
            "is_month_start": int(cursor.day) == 1,
        })
        cursor += one_day

    years: list[dict[str, Any]] = []
    for day in days:
        year = int(day["year"])
        month = int(day["month"])
        if not years or years[-1]["year"] != year:
            years.append({"year": year, "months": [], "days": []})
        years[-1]["days"].append(day)
        months = years[-1]["months"]
        if not months or months[-1]["month"] != month:
            months.append({
                "month": month,
                "label": day["month_label"],
                "label_full": day["month_full"],
                "days": [],
            })
        months[-1]["days"].append(day)

    return {
        "ticker": ticker,
        "start": start_ts.strftime("%Y-%m-%d"),
        "end": end_ts.strftime("%Y-%m-%d"),
        "n_days": len(days),
        "counts": counts,
        "years": years,
        "days": days,
    }


def request_pull_control(project_root: Path, ticker: str, action: str) -> Path:
    """Ask a running pull to pause or halt after the current story."""
    action = str(action or "").strip().lower()
    if action not in {"pause", "halt", "stop"}:
        raise ValueError("action must be pause, halt, or stop")
    if action == "stop":
        action = "halt"
    ticker = str(ticker).upper().strip()
    path = control_path(project_root, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "action": action,
                "requested_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def read_pull_control(project_root: Path, ticker: str) -> dict[str, Any] | None:
    path = control_path(project_root, ticker)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def clear_pull_control(project_root: Path, ticker: str) -> None:
    path = control_path(project_root, ticker)
    if path.exists():
        path.unlink()


def write_pull_pid(project_root: Path, ticker: str, pid: int) -> None:
    path = pid_path(project_root, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(int(pid)), encoding="utf-8")


def read_pull_pid(project_root: Path, ticker: str) -> int | None:
    path = pid_path(project_root, ticker)
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def clear_pull_pid(project_root: Path, ticker: str) -> None:
    path = pid_path(project_root, ticker)
    if path.exists():
        path.unlink()


@dataclass
class StoryPullResult:
    story_id: str
    headline: str
    status: str  # cached | fetched | failed | skipped
    path: str | None = None
    n_chars: int | None = None
    error: str | None = None
    elapsed_s: float | None = None


def _append_manifest(project_root: Path, ticker: str, row: dict[str, Any]) -> None:
    path = manifest_path(project_root, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_progress(project_root: Path, ticker: str, payload: dict[str, Any]) -> None:
    path = progress_path(project_root, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_progress(project_root: Path, ticker: str) -> dict[str, Any] | None:
    path = progress_path(project_root, ticker)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def headlines_needing_bodies(
    news: pd.DataFrame,
    project_root: Path,
    ticker: str,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Return headline rows whose story body is not yet on disk."""
    if news is None or news.empty or "storyId" not in news.columns:
        return pd.DataFrame(columns=["date", "headline", "storyId", "sourceCode"])
    work = news.copy()
    work["storyId"] = work["storyId"].astype(str).str.strip()
    work = work[work["storyId"].ne("") & work["storyId"].ne("nan")]
    if "headline" not in work.columns:
        work["headline"] = ""
    if force:
        return work.reset_index(drop=True)
    have = digests_on_disk(project_root, ticker)
    if not have:
        return work.reset_index(drop=True)
    digests = work["storyId"].map(story_digest)
    return work.loc[~digests.isin(have)].reset_index(drop=True)


def cache_refinitiv_stories(
    project_root: Path,
    ticker: str,
    news: pd.DataFrame,
    *,
    force: bool = False,
    limit: int | None = None,
    sleep_s: float = 0.25,
    max_failures: int | None = 50,
    ld_module: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    window_start: str | None = None,
    window_end: str | None = None,
) -> dict[str, Any]:
    """Fetch and cache full Refinitiv bodies for ``news`` rows.

    Reuses one LSEG session for the whole run. Already-cached digests are skipped
    unless ``force`` is True. Progress is written to ``_pull_progress.json``.
    Write ``_pull_control.json`` with ``{"action":"pause"}`` or ``{"action":"halt"}``
    to stop after the current story (resume by starting another pull).
    """
    import os

    ticker = str(ticker).upper().strip()
    pending = headlines_needing_bodies(news, project_root, ticker, force=force)
    total_headlines = int(len(news)) if isinstance(news, pd.DataFrame) else 0
    already = total_headlines - int(len(pending))
    if limit is not None:
        pending = pending.head(max(0, int(limit)))

    if window_start is None and isinstance(news, pd.DataFrame) and not news.empty and "date" in news.columns:
        try:
            window_start = str(pd.to_datetime(news["date"]).min().date())
        except Exception:
            window_start = None
    if window_end is None and isinstance(news, pd.DataFrame) and not news.empty and "date" in news.columns:
        try:
            window_end = str(pd.to_datetime(news["date"]).max().date())
        except Exception:
            window_end = None

    opened_here = False
    ld = ld_module
    if ld is None and not pending.empty:
        import lseg.data as ld  # type: ignore
        from sentiment_ltr.data.refinitiv_session import open_refinitiv_session

        open_refinitiv_session(project_root, ld)
        opened_here = True

    results: list[StoryPullResult] = []
    fetched = 0
    failed = 0
    cached_hits = already
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.monotonic()
    disk0 = story_dir_stats(project_root, ticker)
    bytes_at_start = int(disk0["bytes"])
    files_at_start = int(disk0["files"])
    run_pid = os.getpid()
    write_pull_pid(project_root, ticker, run_pid)
    clear_pull_control(project_root, ticker)
    final_status = "completed"
    stop_reason: str | None = None
    current_story_date: str | None = None

    def _metrics(extra: dict[str, Any] | None = None) -> dict[str, Any]:
        processed = fetched + failed
        pending_n = int(len(pending))
        elapsed = max(time.monotonic() - t0, 1e-6)
        disk = story_dir_stats(project_root, ticker)
        bytes_now = int(disk["bytes"])
        bytes_delta = max(0, bytes_now - bytes_at_start)
        rate_per_s = fetched / elapsed if fetched else 0.0
        rate_per_min = rate_per_s * 60.0
        remaining = max(0, pending_n - processed)
        eta_s = int(remaining / rate_per_s) if rate_per_s > 0 else None
        pct_run = round(100.0 * processed / pending_n, 1) if pending_n else 100.0
        overall_done = min(total_headlines, already + fetched)
        pct_overall = (
            round(100.0 * overall_done / total_headlines, 1) if total_headlines else 100.0
        )
        avg_new = int(bytes_delta / fetched) if fetched else int(disk["avg_bytes"] or 0)
        projected_total = bytes_now + avg_new * remaining if avg_new else bytes_now
        payload = {
            "ticker": ticker,
            "status": "running",
            "pid": run_pid,
            "started_at": started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "window_start": window_start,
            "window_end": window_end,
            "current_story_date": current_story_date,
            "total_headlines": total_headlines,
            "already_cached": already,
            "pending_this_run": pending_n,
            "fetched": fetched,
            "failed": failed,
            "processed": processed,
            "remaining": remaining,
            "pct_run": pct_run,
            "pct_overall": pct_overall,
            "overall_done": overall_done,
            "elapsed_s": round(elapsed, 1),
            "rate_per_min": round(rate_per_min, 2),
            "eta_s": eta_s,
            "eta_human": _format_eta(eta_s),
            "files_on_disk": int(disk["files"]),
            "bytes_on_disk": bytes_now,
            "bytes_human": disk["bytes_human"],
            "bytes_added_this_run": bytes_delta,
            "bytes_added_human": format_bytes(bytes_delta),
            "avg_bytes_per_story": avg_new or int(disk["avg_bytes"] or 0),
            "projected_total_bytes": projected_total,
            "projected_total_human": format_bytes(projected_total),
            "files_at_start": files_at_start,
            "story_dir": str(full_story_root(project_root) / ticker),
            **(extra or {}),
        }
        return payload

    def _emit(extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = _metrics(extra)
        _write_progress(project_root, ticker, payload)
        if progress_callback:
            progress_callback(payload)
        return payload

    try:
        _emit()
        for idx, row in pending.iterrows():
            story_id = str(row.get("storyId") or "").strip()
            headline = str(row.get("headline") or "")
            story_time = row.get("date") if "date" in pending.columns else None
            try:
                current_story_date = (
                    pd.Timestamp(story_time).strftime("%Y-%m-%d") if story_time is not None else None
                )
            except Exception:
                current_story_date = None
            existing = None if force else find_cached_story_path(project_root, ticker, story_id)
            path = existing or story_path(
                project_root, ticker, story_id, headline, story_time=story_time
            )
            item_t0 = time.monotonic()
            _emit({"last_story_id": story_id, "last_status": "fetching"})
            try:
                if existing is not None and existing.exists() and not force:
                    text = existing.read_text(encoding="utf-8")
                    status = "cached"
                    cached_hits += 1
                    path = existing
                else:
                    text = fetch_refinitiv_story(project_root, story_id, ld_module=ld)
                    write_story_file(
                        path,
                        story_id=story_id,
                        headline=headline,
                        ticker=ticker,
                        text=text,
                        story_time=story_time,
                    )
                    status = "fetched"
                    fetched += 1
                result = StoryPullResult(
                    story_id=story_id,
                    headline=headline,
                    status=status,
                    path=str(path),
                    n_chars=len(text),
                    elapsed_s=round(time.monotonic() - item_t0, 3),
                )
            except Exception as exc:  # noqa: BLE001
                failed += 1
                result = StoryPullResult(
                    story_id=story_id,
                    headline=headline,
                    status="failed",
                    error=str(exc)[:500],
                    elapsed_s=round(time.monotonic() - item_t0, 3),
                )
            results.append(result)
            _append_manifest(
                project_root,
                ticker,
                {
                    **asdict(result),
                    "fetched_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            # Update often enough for a live progress bar (every story).
            _emit({"last_story_id": story_id, "last_status": result.status})
            if max_failures is not None and failed >= int(max_failures):
                final_status = "stopped"
                stop_reason = f"hit max_failures={max_failures}"
                _emit({"status": final_status, "stop_reason": stop_reason})
                break
            ctrl = read_pull_control(project_root, ticker) or {}
            action = str(ctrl.get("action") or "").lower()
            if action in {"pause", "halt", "stop"}:
                clear_pull_control(project_root, ticker)
                final_status = "paused" if action == "pause" else "halted"
                stop_reason = f"user {action}"
                _emit({"status": final_status, "stop_reason": stop_reason})
                break
            if result.status == "fetched" and sleep_s > 0:
                time.sleep(float(sleep_s))
    finally:
        if opened_here and ld is not None:
            try:
                ld.close_session()
            except Exception:
                pass
        clear_pull_pid(project_root, ticker)

    summary = _metrics(
        {
            "status": final_status,
            "stop_reason": stop_reason,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _write_progress(project_root, ticker, summary)
    if progress_callback:
        progress_callback(summary)
    return summary


def _format_eta(eta_s: int | None) -> str:
    if eta_s is None:
        return "—"
    eta_s = max(0, int(eta_s))
    hours, rem = divmod(eta_s, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"~{hours}h {minutes}m"
    if minutes:
        return f"~{minutes}m {seconds}s"
    return f"~{seconds}s"


_TIMESTAMPED_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}(?:_\d{6})?_")


def rename_legacy_story_files(
    project_root: Path,
    ticker: str,
    news: pd.DataFrame,
) -> dict[str, int]:
    """Rename ``slug--digest.txt`` files to ``{stamp}_{slug}--digest.txt`` using news dates.

    Already-timestamped names are left alone. Returns counts renamed / skipped / missing.
    """
    ticker = str(ticker).upper().strip()
    directory = full_story_root(project_root) / ticker
    renamed = 0
    skipped = 0
    missing_meta = 0
    if not directory.is_dir() or news is None or news.empty or "storyId" not in news.columns:
        return {"renamed": 0, "skipped": 0, "missing_meta": 0}

    meta: dict[str, tuple[str, Any, str]] = {}
    for _, row in news.iterrows():
        sid = str(row.get("storyId") or "").strip()
        if not sid or sid == "nan":
            continue
        digest = story_digest(sid)
        if digest in meta:
            continue
        meta[digest] = (
            sid,
            row.get("date") if "date" in news.columns else None,
            str(row.get("headline") or ""),
        )

    for path in sorted(directory.glob("*.txt")):
        if _TIMESTAMPED_NAME.match(path.name):
            skipped += 1
            continue
        digest = digest_from_story_filename(path.name)
        info = meta.get(digest)
        if info is None:
            missing_meta += 1
            continue
        story_id, story_time, headline = info
        if not headline and "--" in path.stem:
            headline = path.stem.rsplit("--", 1)[0].replace("-", " ")
        target = directory / story_filename(story_id, headline, story_time=story_time)
        if target == path or target.exists():
            skipped += 1
            continue
        path.rename(target)
        renamed += 1
    return {"renamed": renamed, "skipped": skipped, "missing_meta": missing_meta}
