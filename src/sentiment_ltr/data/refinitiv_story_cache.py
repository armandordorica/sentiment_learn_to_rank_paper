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
) -> dict[str, Any]:
    """Fetch and cache full Refinitiv bodies for ``news`` rows.

    Reuses one LSEG session for the whole run. Already-cached digests are skipped
    unless ``force`` is True. Progress is written to ``_pull_progress.json``.
    """
    ticker = str(ticker).upper().strip()
    pending = headlines_needing_bodies(news, project_root, ticker, force=force)
    total_headlines = int(len(news)) if isinstance(news, pd.DataFrame) else 0
    already = total_headlines - int(len(pending))
    if limit is not None:
        pending = pending.head(max(0, int(limit)))

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

    def _emit(extra: dict[str, Any] | None = None) -> None:
        payload = {
            "ticker": ticker,
            "status": "running",
            "started_at": started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_headlines": total_headlines,
            "already_cached": cached_hits,
            "pending_this_run": int(len(pending)),
            "fetched": fetched,
            "failed": failed,
            "processed": fetched + failed,
            "elapsed_s": round(time.monotonic() - t0, 1),
            **(extra or {}),
        }
        _write_progress(project_root, ticker, payload)
        if progress_callback:
            progress_callback(payload)

    try:
        _emit()
        for idx, row in pending.iterrows():
            story_id = str(row.get("storyId") or "").strip()
            headline = str(row.get("headline") or "")
            story_time = row.get("date") if "date" in pending.columns else None
            existing = None if force else find_cached_story_path(project_root, ticker, story_id)
            path = existing or story_path(
                project_root, ticker, story_id, headline, story_time=story_time
            )
            item_t0 = time.monotonic()
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
            if (fetched + failed) % 5 == 0 or result.status == "failed" or (fetched + failed) == 1:
                _emit({"last_story_id": story_id, "last_status": result.status})
            if max_failures is not None and failed >= int(max_failures):
                _emit({"status": "stopped", "stop_reason": f"hit max_failures={max_failures}"})
                break
            if result.status == "fetched" and sleep_s > 0:
                time.sleep(float(sleep_s))
    finally:
        if opened_here and ld is not None:
            try:
                ld.close_session()
            except Exception:
                pass

    summary = {
        "ticker": ticker,
        "status": "completed",
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_headlines": total_headlines,
        "already_cached": already,
        "pending_this_run": int(len(pending)),
        "fetched": fetched,
        "failed": failed,
        "processed": fetched + failed,
        "elapsed_s": round(time.monotonic() - t0, 1),
        "story_dir": str(full_story_root(project_root) / ticker),
    }
    _write_progress(project_root, ticker, summary)
    if progress_callback:
        progress_callback(summary)
    return summary


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
