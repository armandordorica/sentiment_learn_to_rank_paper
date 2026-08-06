"""FastAPI adapter for Streamlit Tab 2: Top-1,000 Batch Pipeline.

Mirrors ``render_batch_pipeline_tab()`` and its helpers in ``app.py``: the
batch runner subprocess is the same ``scripts/run_batch_pipeline.py`` (which
writes its own ``batch.pid`` / ``batch_status.json``), and all summaries are
built from the same per-ticker ``manifest.json`` files, so both UIs report
identical numbers.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
load_dotenv(PROJECT_ROOT / ".env")

from sentiment_ltr.data import live_data  # noqa: E402
from sentiment_ltr.data.provider_reason_codes import (  # noqa: E402
    enrich_provider_status_records,
    reason_label,
)
from sentiment_ltr.data.story_quota_settings import (  # noqa: E402
    DEFAULT_MAX_PER_DAY,
    DEFAULT_MIN_SLEEP_S,
    apply_story_quota_cron,
    installed_story_quota_cron_line,
    load_story_quota_settings,
    normalize_settings,
    parse_cron_minute,
    save_story_quota_settings,
    settings_path,
)
from sentiment_ltr.data.story_quota_scheduler import (  # noqa: E402
    live_story_pull_tickers,
    load_ticker_queue,
    story_quota_snapshot,
)
from sentiment_ltr.data.refinitiv_story_cache import (  # noqa: E402
    full_story_root,
    read_progress,
)
from sentiment_ltr.data.crsp_delisting import (  # noqa: E402
    DELISTING_CACHE_PATH,
    load_delisting_cache,
    update_delisting_cache,
)
from sentiment_ltr.data.cash_merger_exits import (  # noqa: E402
    CASH_MERGER_CACHE_PATH,
    CASH_MERGER_CODES,
    get_cash_merger_summary,
    load_cash_merger_cache,
    update_cash_merger_cache,
)

TOP1K_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k"
TOP1K_BY_TICKER_DIR = TOP1K_OUTPUT_DIR / "by_ticker"
TOP1K_COMBINED_DIR = TOP1K_OUTPUT_DIR / "combined"
BATCH_STATUS_FILE = TOP1K_OUTPUT_DIR / "batch_status.json"
BATCH_PID_FILE = TOP1K_OUTPUT_DIR / "batch.pid"
BATCH_LOG_FILE = TOP1K_OUTPUT_DIR / "batch_runner.log"
BATCH_RUNNER_SCRIPT = PROJECT_ROOT / "scripts" / "run_batch_pipeline.py"
TOP1K_UNIVERSE_PATH = PROJECT_ROOT / "app_data" / "crsp_top_volume_universe.csv"
NEWS_THRESHOLD_PATH = PROJECT_ROOT / "app_data" / "ravenpack_news_threshold_universe.csv"

UNIVERSE_SIZE = 1_000
PROVIDER_NAMES = ("wrds", "yahoo", "ravenpack", "refinitiv")
_OK_PROVIDER_STATUSES = frozenset({"ok", "skipped", "…", ""})

_EXIT_SOURCE_ICONS: dict[str, str] = {
    "crsp_dlret": "✅",
    "sdc_pricepersh": "🟡",
    "crsp_dlprc_fallback": "⬜",
}

_STATUS_ICONS = {
    "complete": "🟢",
    "partial": "🟡",
    "failed": "🔴",
    "error": "🔴",
    "skipped_cached": "⚪",
}


def wrds_credentials_available() -> bool:
    return live_data.wrds_credentials_available()


# ── Batch process state ───────────────────────────────────────────────────────


def pid_running() -> int | None:
    """Return the PID from batch.pid if the process is still alive, else None."""
    if not BATCH_PID_FILE.exists():
        return None
    try:
        pid = int(BATCH_PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)  # signal 0 = existence check only
        return pid
    except (ValueError, OSError):
        return None


def read_status() -> dict:
    if not BATCH_STATUS_FILE.exists():
        return {}
    try:
        return json.loads(BATCH_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def log_tail(n: int = 40) -> str | None:
    if not BATCH_LOG_FILE.exists():
        return None
    try:
        lines = BATCH_LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return None


def launch(
    *, start: str, end: str,
    start_rank: int, max_tickers: int | None,
    force_rerun: bool, rerun_failed: bool, rerun_partial: bool, sleep_sec: float,
    stop_after: int, provider_timeout: float, year_timeout: int,
    use_wrds: bool, use_yahoo: bool, use_ravenpack: bool, use_refinitiv: bool,
    combined_parquets: bool,
) -> None:
    """Start the batch runner subprocess — same CLI flags as app.py's _launch_batch."""
    cmd = [sys.executable, str(BATCH_RUNNER_SCRIPT),
           "--start", start, "--end", end,
           "--start-rank", str(start_rank),
           "--sleep", str(sleep_sec),
           "--stop-after-failures", str(stop_after),
           "--provider-timeout", str(provider_timeout),
           "--year-timeout", str(year_timeout)]
    if max_tickers is not None:
        cmd += ["--max-tickers", str(max_tickers)]
    if force_rerun:
        cmd.append("--force-rerun")
    if rerun_failed:
        cmd.append("--rerun-failed")
    if rerun_partial:
        cmd.append("--rerun-partial")
    if not use_wrds:
        cmd.append("--no-wrds")
    if not use_yahoo:
        cmd.append("--no-yahoo")
    if not use_ravenpack:
        cmd.append("--no-ravenpack")
    if not use_refinitiv:
        cmd.append("--no-refinitiv")
    if not combined_parquets:
        cmd.append("--no-combined-parquets")
    TOP1K_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(BATCH_LOG_FILE, "a", encoding="utf-8")  # noqa: SIM115
    log_file.write(f"\n\n=== Run started at {datetime.now(timezone.utc).isoformat()} ===\n")
    log_file.flush()
    subprocess.Popen(cmd, stdout=log_file, stderr=log_file, start_new_session=True)


def stop() -> str | None:
    """Kill the running batch process; return a message (None when nothing ran)."""
    pid = pid_running()
    if pid is None:
        return None
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    BATCH_PID_FILE.unlink(missing_ok=True)
    return f"Killed PID {pid}."


def status_context() -> dict[str, Any]:
    """Everything the 2A live-status partial needs (banner, progress, log tail)."""
    pid = pid_running()
    status = read_status()
    is_running = pid is not None
    ctx: dict[str, Any] = {
        "is_running": is_running,
        "pid": pid,
        "status": status,
        "done": status.get("done", 0) or 0,
        "total": status.get("total", 0) or 0,
        "log_tail": log_tail(40) if is_running else None,
    }
    if is_running:
        started = status.get("ticker_started_at")
        elapsed = None
        if started:
            try:
                elapsed = round(
                    (datetime.now(timezone.utc) - datetime.fromisoformat(started)).total_seconds()
                )
            except Exception:
                elapsed = None
        ctx["ticker_elapsed"] = elapsed
    if ctx["total"]:
        ctx["progress"] = min(1.0, ctx["done"] / ctx["total"])
    else:
        ctx["progress"] = 0.0
    return ctx


# ── Manifests ─────────────────────────────────────────────────────────────────

_manifest_cache: tuple[str, pd.DataFrame] | None = None


def _manifest_cache_token() -> str:
    """Changes when tickers are added/updated on disk (excludes batch_status.json)."""
    if not TOP1K_BY_TICKER_DIR.exists():
        return "0"
    latest = 0.0
    count = 0
    for p in TOP1K_BY_TICKER_DIR.glob("rank_*/manifest.json"):
        count += 1
        latest = max(latest, p.stat().st_mtime)
    return f"{count}:{latest:.0f}"


def load_manifests() -> pd.DataFrame:
    """Walk by_ticker/ dirs and collect manifest data (cached until files change)."""
    global _manifest_cache
    token = _manifest_cache_token()
    if _manifest_cache is not None and _manifest_cache[0] == token:
        return _manifest_cache[1]

    rows = []
    if TOP1K_BY_TICKER_DIR.exists():
        for mfile in sorted(TOP1K_BY_TICKER_DIR.glob("rank_*/manifest.json")):
            try:
                m = json.loads(mfile.read_text(encoding="utf-8"))
                provider_status = list(m.get("provider_status", []))
                needs_enrich = any(
                    not ps.get("fail_reason") and str(ps.get("status", "")) not in ("ok", "skipped")
                    for ps in provider_status
                )
                if needs_enrich:
                    # Fast backfill from error text only — skip reading wrds parquets.
                    provider_status = enrich_provider_status_records(
                        provider_status,
                        ticker=str(m.get("ticker", "")),
                        permno=m.get("permno"),
                        query_start=m.get("start_date"),
                        query_end=m.get("end_date"),
                        wrds_last_trade_date=None,
                        skip_permno_lookup=True,
                    )
                provider_map: dict[str, str] = {}
                provider_rows_map: dict[str, int] = {}
                provider_reason_map: dict[str, str] = {}
                for ps in provider_status:
                    pname = ps.get("provider", "")
                    provider_map[pname] = ps.get("status", "")
                    provider_rows_map[pname] = int(ps.get("rows") or 0)
                    provider_reason_map[pname] = ps.get("fail_reason") or ""
                rows.append({
                    "rank":       m.get("volume_rank"),
                    "permno":     m.get("permno"),
                    "ticker":     m.get("ticker"),
                    "company":    m.get("company_name", ""),
                    "status":     m.get("status"),
                    "ok":         m.get("ok_provider_count", 0),
                    "fail":       m.get("failed_provider_count", 0),
                    "wrds_status":       provider_map.get("wrds", ""),
                    "yahoo_status":      provider_map.get("yahoo", ""),
                    "ravenpack_status":  provider_map.get("ravenpack", ""),
                    "refinitiv_status":  provider_map.get("refinitiv", ""),
                    "wrds_rows":         provider_rows_map.get("wrds", 0),
                    "yahoo_rows":        provider_rows_map.get("yahoo", 0),
                    "ravenpack_rows":    provider_rows_map.get("ravenpack", 0),
                    "refinitiv_rows":    provider_rows_map.get("refinitiv", 0),
                    "yahoo_fail_reason":      provider_reason_map.get("yahoo", ""),
                    "ravenpack_fail_reason":  provider_reason_map.get("ravenpack", ""),
                    "refinitiv_fail_reason":  provider_reason_map.get("refinitiv", ""),
                    "wrds_fail_reason":       provider_reason_map.get("wrds", ""),
                    "created_at": m.get("created_at", ""),
                })
            except Exception:
                pass
    df = pd.DataFrame(rows).sort_values("rank").reset_index(drop=True) if rows else pd.DataFrame()
    _manifest_cache = (token, df)
    return df


# ── Display-cell formatters (same emoji vocabulary as the Streamlit tab) ─────


def _status_cell(status: str) -> str:
    return f"{_STATUS_ICONS.get(str(status), '⬜')} {status}"


def _provider_cell(status: str, rows: object, fail_reason: str = "") -> str:
    status = str(status or "").strip()
    if not status or status in ("…", "—"):
        return "…"
    try:
        row_n = int(rows or 0)
    except (TypeError, ValueError):
        row_n = 0
    if status == "ok":
        return f"✅ {row_n:,}" if row_n else "✅"
    icon = {"failed": "❌", "empty": "⚠️", "timeout": "⏱", "skipped": "—", "unavailable": "⛔"}.get(status, "•")
    if fail_reason:
        return f"{icon} {fail_reason}"
    return f"{icon} {status}"


def _delisting_cell(permno: object, lookup: dict[int, dict]) -> str:
    try:
        rec = lookup[int(permno)]
    except (TypeError, ValueError, KeyError):
        return "…"  # PERMNO not looked up yet
    if not bool(rec.get("delisted")):
        return "🟢 active"
    code = rec.get("dlstcd")
    label = rec.get("delisting_label") or ""
    dlret = rec.get("dlret")
    try:
        ret_str = f"  ({float(dlret):+.1%})" if pd.notna(dlret) else ""
    except (TypeError, ValueError):
        ret_str = ""
    try:
        code_str = f"{int(code)} " if pd.notna(code) else ""
    except (TypeError, ValueError):
        code_str = ""
    return f"⛔ {code_str}{label}{ret_str}"


def _exit_cell(permno: object, lookup: dict[int, dict]) -> str:
    try:
        rec = lookup[int(permno)]
    except (TypeError, ValueError, KeyError):
        return "—"  # no cash-merger record (not a merger / not checked)
    icon = _EXIT_SOURCE_ICONS.get(str(rec.get("exit_source")))
    if not icon:
        return "—"
    try:
        ret_str = f"{float(rec.get('exit_return')):+.1%}"
    except (TypeError, ValueError):
        ret_str = "n/a"
    try:
        price_str = f" @ ${abs(float(rec.get('dlprc'))):,.2f}"
    except (TypeError, ValueError):
        price_str = ""
    return f"{icon} {ret_str}{price_str}"


def _permno_lookup(cache: pd.DataFrame) -> dict[int, dict]:
    if cache.empty or "permno" not in cache.columns:
        return {}
    lookup: dict[int, dict] = {}
    for rec in cache.to_dict(orient="records"):
        try:
            lookup[int(rec["permno"])] = rec
        except (TypeError, ValueError, KeyError):
            continue
    return lookup


def _delisting_lookup() -> dict[int, dict]:
    try:
        return _permno_lookup(load_delisting_cache())
    except Exception:
        return {}


def _cash_merger_lookup() -> dict[int, dict]:
    try:
        return _permno_lookup(load_cash_merger_cache())
    except Exception:
        return {}


def _html(fig: Any) -> str:
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _records(df: pd.DataFrame, limit: int = 250) -> dict[str, Any] | None:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    display = df.head(limit).copy()
    for col in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[col]):
            display[col] = display[col].astype(str)
    display = display.where(pd.notna(display), None)
    return {"columns": list(display.columns), "rows": display.to_dict(orient="records"), "total": len(df)}


def _load_universe() -> pd.DataFrame:
    if not TOP1K_UNIVERSE_PATH.exists():
        return pd.DataFrame()
    u = pd.read_csv(TOP1K_UNIVERSE_PATH)
    u["volume_rank"] = u["volume_rank"].astype(int)
    return u.sort_values("volume_rank").reset_index(drop=True)


# ── 2B Cached data snapshot ───────────────────────────────────────────────────


def snapshot(mdf: pd.DataFrame) -> dict[str, Any]:
    if mdf.empty:
        return {
            "empty": True,
            "universe_size": UNIVERSE_SIZE,
            "news_threshold": news_threshold_snapshot(),
        }
    n_cached = len(mdf)
    n_complete = int((mdf["status"] == "complete").sum())
    n_partial = int((mdf["status"] == "partial").sum())
    n_failed = int(mdf["status"].isin(["failed", "error"]).sum())
    coverage = []
    for pname in PROVIDER_NAMES:
        statuses = mdf.get(f"{pname}_status", pd.Series(dtype=str))
        n_ok = int((statuses == "ok").sum())
        coverage.append({
            "provider": pname.upper(),
            "ok": n_ok,
            "failed": int(statuses.isin(["failed", "timeout"]).sum()),
            "empty": int((statuses == "empty").sum()),
            "other": int(n_cached - n_ok
                         - int(statuses.isin(["failed", "timeout"]).sum())
                         - int((statuses == "empty").sum())),
            "pct_ok": (100 * n_ok / n_cached) if n_cached else 0.0,
        })
    return {
        "empty": False,
        "universe_size": UNIVERSE_SIZE,
        "n_cached": n_cached,
        "n_complete": n_complete,
        "n_partial": n_partial,
        "n_failed": n_failed,
        "n_never": max(0, UNIVERSE_SIZE - n_cached),
        "pct_complete": 100 * n_complete / UNIVERSE_SIZE,
        "cached_frac": n_cached / UNIVERSE_SIZE,
        "coverage": coverage,
        "news_threshold": news_threshold_snapshot(),
    }


def news_threshold_snapshot() -> dict[str, Any]:
    if not NEWS_THRESHOLD_PATH.is_file():
        return {"empty": True, "path": str(NEWS_THRESHOLD_PATH)}
    try:
        frame = pd.read_csv(NEWS_THRESHOLD_PATH)
    except Exception:
        return {"empty": True, "path": str(NEWS_THRESHOLD_PATH)}
    if frame.empty:
        return {"empty": True, "n_names": 0, "path": str(NEWS_THRESHOLD_PATH)}
    if "has_usable_refinitiv_headlines" in frame.columns:
        flags = frame["has_usable_refinitiv_headlines"]
        if flags.dtype != bool:
            flags = flags.astype(str).str.lower().isin(["true", "1", "yes"])
        n_headlines = int(flags.fillna(False).sum())
    else:
        n_headlines = 0
    return {
        "empty": False,
        "n_names": int(len(frame)),
        "n_with_headlines": n_headlines,
        "n_missing_headlines": int(len(frame) - n_headlines),
        "path": (
            str(NEWS_THRESHOLD_PATH.relative_to(PROJECT_ROOT))
            if NEWS_THRESHOLD_PATH.is_relative_to(PROJECT_ROOT)
            else str(NEWS_THRESHOLD_PATH)
        ),
    }


# ── 2C Failure reasons by provider ───────────────────────────────────────────


def _fail_reason_counts(mdf: pd.DataFrame, pname: str) -> pd.DataFrame:
    reason_col = f"{pname}_fail_reason"
    status_col = f"{pname}_status"
    empty = pd.DataFrame(columns=["Reason", "Label", "Count"])
    if mdf.empty or status_col not in mdf.columns:
        return empty
    statuses = mdf[status_col].fillna("").astype(str)
    not_ok = mdf[~statuses.isin(_OK_PROVIDER_STATUSES)]
    if not_ok.empty:
        return empty
    if reason_col in not_ok.columns:
        reasons = not_ok[reason_col].fillna("").astype(str).replace("", "(no reason recorded)")
    else:
        reasons = pd.Series("(no reason recorded)", index=not_ok.index)
    counts = reasons.value_counts().rename_axis("Reason").reset_index(name="Count")
    counts["Label"] = counts["Reason"].apply(
        lambda code: reason_label(code) if code != "(no reason recorded)" else code
    )
    return counts.sort_values("Count", ascending=False).reset_index(drop=True)


def fail_reasons(mdf: pd.DataFrame) -> list[dict[str, Any]]:
    """One entry per provider: total non-ok tickers, counts table, bar chart."""
    sections: list[dict[str, Any]] = []
    for pname in PROVIDER_NAMES:
        counts = _fail_reason_counts(mdf, pname)
        if counts.empty:
            sections.append({"provider": pname.upper(), "total": 0, "table": None, "chart": None})
            continue
        chart_df = counts.sort_values("Count", ascending=True)
        fig = px.bar(
            chart_df, x="Count", y="Reason", orientation="h",
            custom_data=["Label"],
            labels={"Reason": "Reason code", "Count": "Tickers"},
        )
        fig.update_traces(
            hovertemplate="Reason: %{y}<br>Label: %{customdata[0]}<br>Tickers: %{x}<extra></extra>",
        )
        fig.update_layout(
            hovermode="closest",
            height=max(220, 36 * len(chart_df)),
            margin=dict(l=10, r=10, t=30, b=10),
            yaxis=dict(categoryorder="total ascending"),
            showlegend=False,
        )
        sections.append({
            "provider": pname.upper(),
            "total": int(counts["Count"].sum()),
            "table": _records(counts[["Reason", "Label", "Count"]]),
            "chart": _html(fig),
        })
    return sections


# ── 2D Delisting reasons (CRSP) ───────────────────────────────────────────────


def delisting_context(mdf: pd.DataFrame) -> dict[str, Any]:
    universe = _load_universe()
    if universe.empty:
        return {"universe_missing": True, "universe_path": str(TOP1K_UNIVERSE_PATH)}
    target = {int(p) for p in universe["permno"].dropna().tolist()}
    try:
        cache = load_delisting_cache()
    except Exception:
        cache = pd.DataFrame()
    cached_set = {int(p) for p in cache["permno"].dropna().tolist()} if not cache.empty else set()
    missing = sorted(target - cached_set)

    ctx: dict[str, Any] = {
        "universe_missing": False,
        "n_universe": len(target),
        "n_cached": len(target & cached_set),
        "n_missing": len(missing),
        "n_cache_rows": len(cache),
        "creds_ok": wrds_credentials_available(),
        "cache_path": str(DELISTING_CACHE_PATH.relative_to(PROJECT_ROOT)),
        "cache_empty": cache.empty,
        "category_chart": None,
        "code_counts": None,
        "detail": None,
        "n_delisted": None,
        "n_active": None,
        "mean_dlret": None,
    }
    if cache.empty:
        return ctx

    view = cache[cache["permno"].isin(target)].copy()
    if view.empty:
        return ctx

    delisted_only = view[view["delisted"].fillna(False).astype(bool)]
    ctx["n_delisted"] = len(delisted_only)
    ctx["n_active"] = len(view) - len(delisted_only)
    avg_dlret = pd.to_numeric(delisted_only["dlret"], errors="coerce").mean()
    ctx["mean_dlret"] = f"{avg_dlret:.3f}" if pd.notna(avg_dlret) else "—"

    if not delisted_only.empty:
        cat_counts = (
            delisted_only["delisting_category"].fillna("unknown")
            .value_counts().rename_axis("Category").reset_index(name="Count")
        )
        fig = px.bar(
            cat_counts.sort_values("Count"),
            x="Count", y="Category", orientation="h",
            labels={"Category": "Delisting category", "Count": "Tickers"},
        )
        fig.update_traces(hovertemplate="Category: %{y}<br>Tickers: %{x}<extra></extra>")
        fig.update_layout(
            hovermode="closest",
            height=max(200, 40 * len(cat_counts)),
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=False,
        )
        ctx["category_chart"] = _html(fig)

        code_counts = (
            delisted_only[["dlstcd", "delisting_label"]]
            .value_counts().rename_axis(["dlstcd", "delisting_label"]).reset_index(name="Count")
            .sort_values("Count", ascending=False)
        )
        ctx["code_counts"] = _records(code_counts)

    detail = universe.rename(columns={"volume_rank": "rank", "comnam": "company"}).copy()
    if not mdf.empty and "permno" in mdf.columns:
        status_map = mdf[["permno", "status"]].drop_duplicates("permno")
        detail = detail.merge(status_map, on="permno", how="left")
    detail = detail.merge(
        view[["permno", "delisted", "dlstdt", "dlstcd", "delisting_category",
              "delisting_label", "dlret", "nwperm"]],
        on="permno", how="left",
    )
    detail_cols = [c for c in [
        "rank", "ticker", "company", "status", "delisted", "dlstdt", "dlstcd",
        "delisting_category", "delisting_label", "dlret", "nwperm",
    ] if c in detail.columns]
    ctx["detail"] = _records(detail[detail_cols].sort_values("rank"), limit=1_200)
    return ctx


def fetch_delisting(refetch_all: bool) -> tuple[int, str | None]:
    """Query CRSP for missing (or all) universe PERMNOs. Returns (n_queried, error)."""
    universe = _load_universe()
    if universe.empty:
        return 0, f"Universe file not found: {TOP1K_UNIVERSE_PATH}"
    target = {int(p) for p in universe["permno"].dropna().tolist()}
    if refetch_all:
        permnos = target
    else:
        try:
            cache = load_delisting_cache()
        except Exception:
            cache = pd.DataFrame()
        cached_set = {int(p) for p in cache["permno"].dropna().tolist()} if not cache.empty else set()
        permnos = target - cached_set
    try:
        _, n_new = update_delisting_cache(permnos, force=refetch_all)
        return n_new, None
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


# ── 2E Cash-merger exits ──────────────────────────────────────────────────────


def _cash_merger_candidates() -> set[int]:
    try:
        delist = load_delisting_cache()
    except Exception:
        return set()
    if delist.empty or "dlstcd" not in delist.columns:
        return set()
    mask = delist["dlstcd"].isin(CASH_MERGER_CODES)
    return {int(p) for p in delist.loc[mask, "permno"].dropna().tolist()}


def cash_merger_context() -> dict[str, Any]:
    universe = _load_universe()
    if universe.empty:
        return {"universe_missing": True, "universe_path": str(TOP1K_UNIVERSE_PATH)}
    candidates = _cash_merger_candidates()
    try:
        exit_cache = load_cash_merger_cache()
    except Exception:
        exit_cache = pd.DataFrame()
    checked = (
        {int(p) for p in exit_cache["permno"].dropna().tolist()}
        if not exit_cache.empty else set()
    )
    missing = sorted(candidates - checked)

    ctx: dict[str, Any] = {
        "universe_missing": False,
        "n_candidates": len(candidates),
        "n_resolved": len(candidates & checked),
        "n_missing": len(missing),
        "creds_ok": wrds_credentials_available(),
        "cache_path": str(CASH_MERGER_CACHE_PATH.relative_to(PROJECT_ROOT)),
        "cache_empty": exit_cache.empty,
        "summary": None,
        "source_counts": None,
        "n_resolved_rows": 0,
    }
    if not candidates or exit_cache.empty:
        return ctx

    resolved = exit_cache[exit_cache["exit_source"].isin(list(_EXIT_SOURCE_ICONS.keys()))].copy()
    ctx["n_resolved_rows"] = len(resolved)
    if resolved.empty:
        return ctx

    summary = get_cash_merger_summary(resolved)
    if not summary.empty:
        summary_display = summary.copy()
        summary_display["exit_source"] = summary_display["exit_source"].map(
            lambda s: f"{_EXIT_SOURCE_ICONS.get(s, '')} {s}".strip()
        )
        ctx["summary"] = _records(summary_display)

    counts = resolved["exit_source"].value_counts()
    ctx["source_counts"] = {
        "crsp": int(counts.get("crsp_dlret", 0)),
        "sdc": int(counts.get("sdc_pricepersh", 0)),
        "fallback": int(counts.get("crsp_dlprc_fallback", 0)),
    }
    return ctx


def fetch_cash_merger(refetch_all: bool) -> tuple[int, str | None]:
    """Resolve cash-merger exits for missing (or all) candidates. Returns (n, error)."""
    candidates = _cash_merger_candidates()
    if not candidates:
        return 0, ("No cash-merger candidates found yet. Populate the Delisting reasons (CRSP) "
                   "cache first so dlstcd 232/233 names can be identified.")
    if refetch_all:
        permnos = candidates
    else:
        try:
            exit_cache = load_cash_merger_cache()
        except Exception:
            exit_cache = pd.DataFrame()
        checked = (
            {int(p) for p in exit_cache["permno"].dropna().tolist()}
            if not exit_cache.empty else set()
        )
        permnos = candidates - checked
    try:
        _, n_new = update_cash_merger_cache(permnos, force=refetch_all)
        return n_new, None
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc)


def cash_merger_csv() -> str:
    """Resolved cash-merger exits joined with ticker/company, as CSV text."""
    try:
        exit_cache = load_cash_merger_cache()
    except Exception:
        exit_cache = pd.DataFrame()
    resolved = (
        exit_cache[exit_cache["exit_source"].isin(list(_EXIT_SOURCE_ICONS.keys()))].copy()
        if not exit_cache.empty else pd.DataFrame()
    )
    universe = _load_universe()
    if not resolved.empty and not universe.empty:
        resolved = resolved.merge(
            universe.rename(columns={"comnam": "company"})[["permno", "ticker", "company"]],
            on="permno", how="left",
        )
    return resolved.to_csv(index=False)


# ── Per-ticker status table ───────────────────────────────────────────────────


def _live_row(status: dict) -> pd.DataFrame | None:
    """Row for the ticker currently being processed (before its manifest exists)."""
    live_rank = status.get("current_rank")
    live_ticker = status.get("current_ticker", "")
    live_step = status.get("current_step", "starting…")
    psf = status.get("providers_so_far", {})
    if not live_ticker or not live_rank:
        return None
    try:
        ts = status.get("ticker_started_at", "")
        elapsed = round((datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()) if ts else 0
    except Exception:
        elapsed = 0
    return pd.DataFrame([{
        "rank": live_rank, "ticker": live_ticker, "company": "⚡ in progress",
        "status": f"⚡ {live_step}", "permno": None,
        **{f"{p}_status": (psf.get(p, {}).get("status", "…") if p in psf else "…") for p in PROVIDER_NAMES},
        **{f"{p}_rows": psf.get(p, {}).get("rows", 0) for p in PROVIDER_NAMES},
        **{f"{p}_fail_reason": "" for p in PROVIDER_NAMES},
        "ok": sum(1 for p in psf if psf[p].get("status") == "ok"),
        "fail": sum(1 for p in psf if psf[p].get("status") in ("failed", "timeout")),
        "created_at": f"{elapsed}s elapsed",
    }])


def ticker_table(
    mdf: pd.DataFrame,
    *,
    status_filter: list[str] | None = None,
    ticker_filter: str = "",
    only_failed: bool = False,
) -> dict[str, Any]:
    """Filtered per-ticker display table with emoji provider/delisting/exit cells."""
    status = read_status()
    if pid_running() is not None and status:
        live = _live_row(status)
        if live is not None and (mdf.empty or live.iloc[0]["ticker"] not in mdf.get("ticker", pd.Series(dtype=str)).values):
            mdf = pd.concat([live, mdf], ignore_index=True)

    if mdf.empty:
        return {"table": None, "n_shown": 0, "n_total": 0}

    view = mdf.copy()
    for col in (f"{p}_fail_reason" for p in PROVIDER_NAMES):
        if col not in view.columns:
            view[col] = ""
    if status_filter:
        view = view[view["status"].isin(status_filter)]
    if ticker_filter:
        view = view[view["ticker"].str.upper().str.contains(ticker_filter.upper())]
    if only_failed:
        view = view[view["fail"] > 0]

    delist_lookup = _delisting_lookup()
    exit_lookup = _cash_merger_lookup()
    display = pd.DataFrame({
        "Rank": view["rank"],
        "Ticker": view["ticker"],
        "Company": view["company"],
        "Status": view["status"].apply(_status_cell),
        "WRDS": view.apply(lambda r: _provider_cell(r["wrds_status"], r["wrds_rows"], r["wrds_fail_reason"]), axis=1),
        "Exit": view["permno"].apply(lambda p: _exit_cell(p, exit_lookup)) if "permno" in view.columns else "—",
        "Yahoo": view.apply(lambda r: _provider_cell(r["yahoo_status"], r["yahoo_rows"], r["yahoo_fail_reason"]), axis=1),
        "RavenPack": view.apply(lambda r: _provider_cell(r["ravenpack_status"], r["ravenpack_rows"], r["ravenpack_fail_reason"]), axis=1),
        "Refinitiv": view.apply(lambda r: _provider_cell(r["refinitiv_status"], r["refinitiv_rows"], r["refinitiv_fail_reason"]), axis=1),
        "CRSP delisting": view["permno"].apply(lambda p: _delisting_cell(p, delist_lookup)) if "permno" in view.columns else "…",
        "Cached at": view["created_at"],
    })
    return {
        "table": _records(display, limit=1_200),
        "n_shown": len(view),
        "n_total": len(mdf),
    }


def ticker_options(mdf: pd.DataFrame) -> list[str]:
    if mdf.empty:
        return []
    return sorted(mdf["ticker"].dropna().unique().tolist())


def ticker_detail(mdf: pd.DataFrame, ticker: str) -> dict[str, Any] | None:
    row = mdf[mdf["ticker"] == ticker] if not mdf.empty else pd.DataFrame()
    if row.empty:
        return None
    r = row.iloc[0]

    prov_rows = []
    for pname in PROVIDER_NAMES:
        reason = r.get(f"{pname}_fail_reason") or ""
        prov_rows.append({
            "Provider": pname,
            "Status": r.get(f"{pname}_status") or "—",
            "Rows": int(r.get(f"{pname}_rows") or 0),
            "Reason": reason,
            "Reason (label)": reason_label(reason) if reason else "",
        })

    delist = None
    rec = _delisting_lookup().get(int(r["permno"])) if pd.notna(r.get("permno")) else None
    if rec is not None:
        if not bool(rec.get("delisted")):
            delist = {"active": True}
        else:
            dlret = rec.get("dlret")
            dlstdt = rec.get("dlstdt")
            nwperm = rec.get("nwperm")
            delist = {
                "active": False,
                "dlstcd": str(rec.get("dlstcd") or "—"),
                "dlret": f"{float(dlret):+.2%}" if pd.notna(dlret) else "—",
                "dlstdt": pd.Timestamp(dlstdt).strftime("%Y-%m-%d") if pd.notna(dlstdt) else "—",
                "nwperm": str(int(nwperm)) if pd.notna(nwperm) else "—",
                "category": rec.get("delisting_category", ""),
                "label": rec.get("delisting_label", ""),
            }

    outputs: list[str] = []
    slug = "".join(ch if ch.isalnum() else "_" for ch in ticker.upper().strip())
    try:
        for d in TOP1K_BY_TICKER_DIR.glob(f"rank_{int(r['rank']):04d}_{slug}"):
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                full = json.loads(manifest_path.read_text(encoding="utf-8"))
                outputs = list(full.get("outputs", {}).values())
            break
    except Exception:
        pass

    return {
        "ticker": r["ticker"],
        "rank": r["rank"],
        "status": r["status"],
        "ok": int(r["ok"]),
        "fail": int(r["fail"]),
        "company": r["company"],
        "created_at": r["created_at"],
        "providers": _records(pd.DataFrame(prov_rows)),
        "delisting": delist,
        "outputs": outputs,
    }


# ── Storage / combined parquets ───────────────────────────────────────────────


def storage_context() -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "root": str(TOP1K_OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "n_dirs": 0, "total_mb": None, "avg_mb": None,
    }
    if TOP1K_BY_TICKER_DIR.exists():
        ticker_dirs = list(TOP1K_BY_TICKER_DIR.glob("rank_*"))
        total_bytes = sum(
            f.stat().st_size for d in ticker_dirs for f in d.rglob("*") if f.is_file()
        )
        total_mb = total_bytes / 1_048_576
        ctx["n_dirs"] = len(ticker_dirs)
        ctx["total_mb"] = f"{total_mb:.1f} MB" if total_mb < 1024 else f"{total_mb/1024:.2f} GB"
        ctx["avg_mb"] = f"{total_mb/len(ticker_dirs):.1f} MB" if ticker_dirs else "—"
    return ctx


def write_combined() -> list[str]:
    """Merge per-ticker parquets into provider-level combined files."""
    combined_keys = ["wrds_prices", "wrds_names", "yahoo_prices", "ravenpack_articles",
                     "refinitiv_prices", "refinitiv_news"]
    TOP1K_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for key in combined_keys:
        filename = f"{key}.parquet"
        frames = []
        for ticker_dir in sorted(TOP1K_BY_TICKER_DIR.glob("rank_*")):
            p = ticker_dir / filename
            if p.exists():
                try:
                    frames.append(pd.read_parquet(p))
                except Exception:
                    pass
        if frames:
            out = TOP1K_COMBINED_DIR / filename
            pd.concat(frames, ignore_index=True).to_parquet(out, index=False)
            written.append(f"{filename}  ({len(frames)} tickers, {sum(len(f) for f in frames):,} rows)")
    return written


# ── Launch-form defaults ──────────────────────────────────────────────────────


def form_defaults() -> dict[str, Any]:
    creds = wrds_credentials_available()
    return {
        "start_date": "2003-01-01",
        "end_date": "2014-12-31",
        "start_rank": 1,
        "sleep_sec": 0.25,
        "stop_after": 25,
        "provider_timeout": 300,
        "year_timeout": 90,
        "use_wrds": creds,
        "use_yahoo": True,
        "use_ravenpack": creds,
        "use_refinitiv": True,
        "creds_ok": creds,
    }


# ── 2F Story quota automation ─────────────────────────────────────────────────


STORY_QUEUE_PATH = PROJECT_ROOT / "app_data" / "story_pull_queue.txt"
DAILY_QUOTA_LOG = (
    PROJECT_ROOT / "data" / "raw" / "data_explorer_full_stories" / "_daily_quota.log"
)


def _cron_human_summary(*, enabled: bool, minute: int, installed: bool) -> dict[str, Any]:
    minute = int(minute)
    stamp = f":{minute:02d}"
    if not enabled:
        plain = "Hourly watchdog is off."
        detail = "Automation will not auto-check the queue on the hour."
    else:
        plain = f"Once every hour at {stamp} local Mac time (24 checks/day)."
        detail = (
            "Each check starts the next queued ticker only if no story job is live "
            "and daily quota remains; otherwise it logs and exits."
        )
    if enabled and not installed:
        detail += " Not installed in crontab yet — use Edit → Save & apply."
    elif enabled and installed:
        detail += " Installed in this Mac’s crontab."
    elif (not enabled) and installed:
        detail += " An old cron line is still installed — Edit to remove it."
    return {
        "enabled": enabled,
        "minute": minute,
        "stamp": stamp,
        "plain": plain,
        "detail": detail,
        "expression": f"{minute} * * * *" if enabled else "(disabled)",
        "checks_per_day": 24 if enabled else 0,
    }


def _next_hourly_cron_fire(minute: int, *, now: datetime | None = None) -> datetime:
    """Next local-time fire for ``M * * * *`` (macOS cron uses the system timezone)."""
    current = (now or datetime.now().astimezone()).astimezone()
    minute = max(0, min(59, int(minute)))
    candidate = current.replace(minute=minute, second=0, microsecond=0)
    if candidate <= current:
        candidate = candidate + timedelta(hours=1)
    return candidate


def _format_run_times(when: datetime) -> dict[str, str]:
    utc = when.astimezone(timezone.utc)
    eastern = when.astimezone(ZoneInfo("America/New_York"))
    # EST or EDT depending on date
    et_name = eastern.tzname() or "ET"
    return {
        "local": when.isoformat(timespec="minutes"),
        "local_human": when.strftime("%Y-%m-%d %H:%M %Z"),
        "utc": utc.strftime("%Y-%m-%d %H:%M UTC"),
        "eastern": eastern.strftime(f"%Y-%m-%d %H:%M {et_name}"),
        "eastern_tz": et_name,
    }


def _story_queue_rows(
    queue: list[str],
    *,
    live: list[str],
    universe_path: Path,
) -> dict[str, Any]:
    """Annotate the full pull queue with ready / waiting / done / running status."""
    meta: dict[str, dict[str, Any]] = {}
    if universe_path.is_file():
        try:
            frame = pd.read_csv(universe_path)
        except Exception:
            frame = pd.DataFrame()
        if not frame.empty and "ticker" in frame.columns:
            for _, row in frame.iterrows():
                ticker = str(row.get("ticker") or "").upper().strip()
                if not ticker:
                    continue
                usable = row.get("has_usable_refinitiv_headlines")
                if not isinstance(usable, (bool, int)):
                    usable = str(usable).lower() in {"true", "1", "yes"}
                meta[ticker] = {
                    "volume_rank": row.get("volume_rank"),
                    "headline_rows": int(row.get("refinitiv_headline_rows") or 0),
                    "usable": bool(usable),
                }

    live_set = {str(t).upper() for t in live}
    story_root = full_story_root(PROJECT_ROOT)
    rows: list[dict[str, Any]] = []
    next_ticker: str | None = None
    counts = {"running": 0, "ready": 0, "needs_headlines": 0, "done": 0}

    for position, ticker in enumerate(queue, start=1):
        info = meta.get(ticker, {})
        usable = bool(info.get("usable"))
        headline_rows = int(info.get("headline_rows") or 0)
        progress = read_progress(PROJECT_ROOT, ticker) or {}
        pct = float(progress.get("pct_overall") or 0)
        status_raw = str(progress.get("status") or "")
        bodies = 0
        ticker_dir = story_root / ticker
        if ticker_dir.is_dir():
            bodies = sum(
                1
                for path in ticker_dir.iterdir()
                if path.is_file() and path.suffix == ".txt" and not path.name.startswith("_")
            )

        if ticker in live_set:
            state = "running"
        elif status_raw == "completed" or pct >= 99.9:
            state = "done"
        elif usable and headline_rows > 0 and bodies >= max(1, int(headline_rows * 0.995)):
            state = "done"
        elif usable:
            state = "ready"
        else:
            state = "needs_headlines"

        if state == "ready" and next_ticker is None and ticker not in live_set:
            state = "next"
            next_ticker = ticker

        if state == "running":
            counts["running"] += 1
        elif state == "done":
            counts["done"] += 1
        elif state in {"ready", "next"}:
            counts["ready"] += 1
        else:
            counts["needs_headlines"] += 1

        rows.append(
            {
                "position": position,
                "ticker": ticker,
                "volume_rank": info.get("volume_rank"),
                "headline_rows": headline_rows,
                "bodies": bodies,
                "pct_overall": pct,
                "state": state,
                "state_label": {
                    "running": "Running",
                    "next": "Next up",
                    "ready": "Ready",
                    "needs_headlines": "Needs headlines",
                    "done": "Done",
                }.get(state, state),
            }
        )

    if live and next_ticker is None:
        for row in rows:
            if row["state"] == "ready":
                row["state"] = "next"
                row["state_label"] = "Next up"
                next_ticker = row["ticker"]
                break

    return {
        "rows": rows,
        "next_ticker": next_ticker or (live[0] if live else None),
        "counts": {
            "total": len(rows),
            "ready": counts["ready"],
            "needs_headlines": counts["needs_headlines"],
            "done": counts["done"],
            "running": counts["running"],
        },
    }


def story_quota_context(
    *,
    message: str | None = None,
    error: str | None = None,
    editing: bool = False,
) -> dict[str, Any]:
    cfg = load_story_quota_settings(PROJECT_ROOT)
    snap = story_quota_snapshot(PROJECT_ROOT, max_per_day=cfg.max_per_day)
    queue = load_ticker_queue(STORY_QUEUE_PATH)
    live = live_story_pull_tickers(PROJECT_ROOT)
    installed = installed_story_quota_cron_line()
    installed_minute = parse_cron_minute(installed)
    approx_rps = round(1.0 / float(cfg.min_sleep_s), 2) if cfg.min_sleep_s else None
    log_tail = None
    if DAILY_QUOTA_LOG.is_file():
        try:
            lines = DAILY_QUOTA_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-12:])
        except OSError:
            log_tail = None
    cron_matches = (
        installed is not None
        and cfg.cron_enabled
        and installed_minute == cfg.cron_minute
    )
    cron_saved = _cron_human_summary(
        enabled=bool(cfg.cron_enabled),
        minute=int(cfg.cron_minute),
        installed=cron_matches,
    )
    effective_minute = (
        int(installed_minute)
        if installed_minute is not None
        else int(cfg.cron_minute)
    )
    cron_live = _cron_human_summary(
        enabled=installed is not None,
        minute=effective_minute,
        installed=installed is not None,
    )
    if installed is None:
        cron_live = {
            **_cron_human_summary(enabled=False, minute=cfg.cron_minute, installed=False),
            "plain": "No story-quota cron line is installed on this Mac.",
            "detail": "Use Edit if you want the hourly watchdog installed.",
            "expression": "(not installed)",
            "checks_per_day": 0,
        }

    next_run = None
    if installed is not None:
        fire = _next_hourly_cron_fire(effective_minute)
        next_run = {
            "enabled": True,
            **_format_run_times(fire),
        }
    elif cfg.cron_enabled:
        fire = _next_hourly_cron_fire(int(cfg.cron_minute))
        next_run = {
            "enabled": False,
            "pending_install": True,
            **_format_run_times(fire),
        }

    pull_status = {
        "running": bool(live),
        "tickers": live,
        "label": (
            f"Running now: {', '.join(live)}"
            if live
            else "Idle — no full-story pull is running"
        ),
    }
    queue_plan = _story_queue_rows(
        queue,
        live=live,
        universe_path=NEWS_THRESHOLD_PATH,
    )

    return {
        "settings": cfg.as_dict(),
        "settings_path": str(settings_path(PROJECT_ROOT).relative_to(PROJECT_ROOT)),
        "queue_path": str(STORY_QUEUE_PATH.relative_to(PROJECT_ROOT)),
        "queue": queue,
        "queue_plan": queue_plan,
        "n_queue": len(queue),
        "live_tickers": live,
        "pull_status": pull_status,
        "quota": {
            "remaining": snap.remaining,
            "calendar_used": snap.calendar_used,
            "rolling_used": snap.rolling_used,
            "blocking": snap.blocking,
            "wait_until": snap.wait_until_local,
            "max_per_day": snap.max_per_window,
        },
        "approx_rps": approx_rps,
        "cron_saved": cron_saved,
        "cron_live": cron_live,
        "next_run": next_run,
        "cron_installed_line": installed,
        "cron_installed": installed is not None,
        "cron_matches_settings": cron_matches,
        "defaults": {
            "max_per_day": DEFAULT_MAX_PER_DAY,
            "min_sleep_s": DEFAULT_MIN_SLEEP_S,
        },
        "editing": bool(editing),
        "log_tail": log_tail,
        "message": message,
        "error": error,
    }


def save_story_quota_automation(
    *,
    max_per_day: int,
    min_sleep_s: float,
    cron_enabled: bool,
    cron_minute: int,
    apply_cron: bool = True,
) -> dict[str, Any]:
    settings = normalize_settings(
        max_per_day=max_per_day,
        min_sleep_s=min_sleep_s,
        cron_enabled=cron_enabled,
        cron_minute=cron_minute,
    )
    saved = save_story_quota_settings(PROJECT_ROOT, settings)
    cron_info: dict[str, Any] = {}
    cron_error: str | None = None
    if apply_cron:
        try:
            cron_info = apply_story_quota_cron(PROJECT_ROOT, saved)
        except Exception as exc:  # noqa: BLE001
            cron_error = str(exc)
    ctx = story_quota_context(
        message=(
            f"Saved. Daily budget={saved.max_per_day:,} stories · "
            f"pause={saved.min_sleep_s}s between calls · "
            + (
                f"watchdog once every hour at :{saved.cron_minute:02d} "
                f"(24 checks/day)."
                if saved.cron_enabled
                else "hourly watchdog off."
            )
            + (
                " Live story jobs keep their launch-time budget until restarted."
                if live_story_pull_tickers(PROJECT_ROOT)
                else ""
            )
        ),
        error=cron_error,
        editing=bool(cron_error),
    )
    ctx["cron_apply"] = cron_info
    return ctx

