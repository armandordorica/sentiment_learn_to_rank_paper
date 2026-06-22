#!/usr/bin/env python
"""Standalone batch runner for the Top-1,000 data pipeline.

Designed to be launched as a subprocess from the Streamlit app's Batch Pipeline tab,
or run directly from the terminal:

    python scripts/run_batch_pipeline.py [options]

All output is written to data/raw/data_explorer_top1k/.
Progress is tracked in batch_progress.csv / batch_progress.json (one row per ticker).
Current run state (status, PID, current ticker) is written to batch_status.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import pandas as pd

from sentiment_ltr.data import live_data

TOP1K_UNIVERSE_PATH = PROJECT_ROOT / "app_data" / "crsp_top_volume_universe.csv"
TOP1K_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k"
TOP1K_BY_TICKER_DIR = TOP1K_OUTPUT_DIR / "by_ticker"
TOP1K_COMBINED_DIR = TOP1K_OUTPUT_DIR / "combined"
PROGRESS_CSV = TOP1K_OUTPUT_DIR / "batch_progress.csv"
PROGRESS_JSON = TOP1K_OUTPUT_DIR / "batch_progress.json"
PID_FILE = TOP1K_OUTPUT_DIR / "batch.pid"
STATUS_FILE = TOP1K_OUTPUT_DIR / "batch_status.json"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _safe_slug(ticker: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(ticker).upper().strip())


def ticker_cache_dir(rank: int, ticker: str) -> Path:
    return TOP1K_BY_TICKER_DIR / f"rank_{int(rank):04d}_{_safe_slug(ticker)}"


def read_manifest(rank: int, ticker: str) -> dict | None:
    path = ticker_cache_dir(rank, ticker) / "manifest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def should_skip(rank: int, ticker: str, force_rerun: bool, rerun_failed: bool) -> bool:
    if force_rerun:
        return False
    manifest = read_manifest(rank, ticker)
    if not manifest:
        return False
    status = manifest.get("status")
    if status in ("complete", "partial"):
        return True
    if status == "failed" and not rerun_failed:
        return True
    return False


def write_progress(records: list[dict]) -> None:
    if not records:
        return
    TOP1K_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(PROGRESS_CSV, index=False)
    PROGRESS_JSON.write_text(
        json.dumps(records, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _ts() -> str:
    """Short local timestamp for log lines."""
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    """Print with timestamp and flush immediately so the log file updates in real time."""
    print(f"[{_ts()}] {msg}", flush=True)


def write_status(
    status: str,
    current_rank: int | None = None,
    current_ticker: str | None = None,
    current_step: str | None = None,
    total: int | None = None,
    done: int | None = None,
    ticker_started_at: str | None = None,
    error: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    elapsed_s: float | None = None
    if ticker_started_at:
        try:
            started = datetime.fromisoformat(ticker_started_at)
            elapsed_s = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
        except Exception:
            pass
    obj = {
        "status": status,
        "updated_at": now,
        "current_rank": current_rank,
        "current_ticker": current_ticker,
        "current_step": current_step,
        "ticker_started_at": ticker_started_at,
        "elapsed_s": elapsed_s,
        "total": total,
        "done": done,
        "error": error,
        "pid": os.getpid(),
    }
    TOP1K_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def _save_frame(frame: object, path: Path, saved: dict, key: str) -> None:
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        saved[key] = str(path.relative_to(PROJECT_ROOT))


def save_ticker_result(row: pd.Series, result: dict) -> dict:
    """Persist all provider payloads for one ticker and return the manifest dict."""
    rank = int(row["volume_rank"])
    ticker = str(row["ticker"]).upper().strip()
    out_dir = ticker_cache_dir(rank, ticker)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}
    providers = result["providers"]
    rf = providers.get("refinitiv", {})
    wr = providers.get("wrds", {})
    yh = providers.get("yahoo", {})
    rp = providers.get("ravenpack", {})

    _save_frame(rf.get("prices"),            out_dir / "refinitiv_prices.parquet",            saved, "refinitiv_prices")
    _save_frame(rf.get("news"),              out_dir / "refinitiv_news.parquet",              saved, "refinitiv_news")
    _save_frame(rf.get("news_daily_counts"), out_dir / "refinitiv_news_daily_counts.parquet", saved, "refinitiv_news_daily_counts")
    _save_frame(wr.get("prices"),            out_dir / "wrds_prices.parquet",                 saved, "wrds_prices")
    _save_frame(wr.get("names"),             out_dir / "wrds_names.parquet",                  saved, "wrds_names")
    _save_frame(yh.get("prices"),            out_dir / "yahoo_prices.parquet",                saved, "yahoo_prices")
    _save_frame(rp.get("articles"),          out_dir / "ravenpack_articles.parquet",          saved, "ravenpack_articles")

    status_rows = []
    for pname, payload in providers.items():
        row_count = sum(
            len(v)
            for k, v in payload.items()
            if isinstance(v, pd.DataFrame) and k in ("prices", "news", "news_daily_counts", "names", "articles")
        )
        status_rows.append({
            "provider": pname,
            "status": payload.get("status"),
            "rows": row_count,
            "error": payload.get("error"),
        })
    status_df = pd.DataFrame(status_rows)
    status_df.to_parquet(out_dir / "provider_status.parquet", index=False)
    saved["provider_status"] = str((out_dir / "provider_status.parquet").relative_to(PROJECT_ROOT))

    ok_count = int(status_df["status"].eq("ok").sum()) if not status_df.empty else 0
    fail_count = int(status_df["status"].eq("failed").sum()) if not status_df.empty else 0
    n_selected = sum(bool(v) for v in result.get("selected_providers", {}).values())
    run_status = "complete" if ok_count == n_selected else ("partial" if ok_count else "failed")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": run_status,
        "volume_rank": rank,
        "ticker": ticker,
        "permno": int(row["permno"]),
        "company_name": row.get("comnam"),
        "start_date": result["start_date"],
        "end_date": result["end_date"],
        "selected_providers": result.get("selected_providers", {}),
        "ok_provider_count": ok_count,
        "failed_provider_count": fail_count,
        "provider_status": status_df.to_dict(orient="records"),
        "outputs": saved,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return manifest


def write_combined_parquets(provider_keys: list[str]) -> None:
    """Merge per-ticker parquets into combined/ files for each provider key."""
    TOP1K_COMBINED_DIR.mkdir(parents=True, exist_ok=True)
    for key in provider_keys:
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
            pd.concat(frames, ignore_index=True).to_parquet(
                TOP1K_COMBINED_DIR / filename, index=False
            )
            print(f"[batch] Combined {len(frames)} frames → combined/{filename}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Top-1,000 batch data pipeline")
    parser.add_argument("--start",               default="2003-01-01", help="Query start date")
    parser.add_argument("--end",                 default="2014-12-31", help="Query end date")
    parser.add_argument("--start-rank",          type=int, default=1)
    parser.add_argument("--max-tickers",         type=int, default=None)
    parser.add_argument("--force-rerun",         action="store_true", default=False)
    parser.add_argument("--rerun-failed",        action="store_true", default=True)
    parser.add_argument("--sleep",               type=float, default=0.25)
    parser.add_argument("--stop-after-failures", type=int, default=25)
    parser.add_argument("--wrds",                action="store_true", default=True)
    parser.add_argument("--no-wrds",             dest="wrds", action="store_false")
    parser.add_argument("--yahoo",               action="store_true", default=True)
    parser.add_argument("--no-yahoo",            dest="yahoo", action="store_false")
    parser.add_argument("--ravenpack",           action="store_true", default=True)
    parser.add_argument("--no-ravenpack",        dest="ravenpack", action="store_false")
    parser.add_argument("--refinitiv",           action="store_true", default=False)
    parser.add_argument("--combined-parquets",   action="store_true", default=True)
    parser.add_argument("--provider-timeout",    type=float, default=300.0,
                        help="Max seconds to wait for a single provider query (default 300s / 5 min)")
    parser.add_argument("--year-timeout",        type=int,   default=90,
                        help="Per-year statement_timeout for RavenPack queries in seconds (default 90s)")
    args = parser.parse_args()

    TOP1K_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TOP1K_BY_TICKER_DIR.mkdir(parents=True, exist_ok=True)
    TOP1K_COMBINED_DIR.mkdir(parents=True, exist_ok=True)

    PID_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")

    universe = pd.read_csv(TOP1K_UNIVERSE_PATH)
    universe["volume_rank"] = universe["volume_rank"].astype(int)
    universe = universe.sort_values("volume_rank").reset_index(drop=True)
    selected = universe[universe["volume_rank"] >= args.start_rank].copy()
    if args.max_tickers is not None:
        selected = selected.head(args.max_tickers)

    latest_crsp_date = None
    if args.wrds and live_data.wrds_credentials_available():
        try:
            latest_crsp_date = live_data.get_latest_crsp_date()
            print(f"[batch] Latest CRSP date: {latest_crsp_date.date()}")
        except Exception as exc:
            print(f"[batch] Could not resolve latest CRSP date: {exc}")

    records: list[dict] = []
    consecutive_failures = 0
    total = len(selected)

    providers_active = [p for p, on in [
        ("WRDS", args.wrds), ("Yahoo", args.yahoo),
        ("RavenPack", args.ravenpack), ("Refinitiv", args.refinitiv)
    ] if on]
    providers_label = " + ".join(providers_active) if providers_active else "none"

    write_status("running", total=total, done=0, current_step="starting")
    _log(f"[batch] Starting — {total} tickers  |  ranks {selected['volume_rank'].min()}–{selected['volume_rank'].max()}")
    _log(f"[batch] Providers active: {providers_label}")
    _log(f"[batch] Window: {args.start} → {args.end}")
    _log("-" * 60)

    for i, (_, row) in enumerate(selected.iterrows()):
        ticker = str(row["ticker"]).upper().strip()
        rank = int(row["volume_rank"])
        company = str(row.get("comnam", "")).strip()
        prefix = f"[{i+1}/{total}]  rank {rank:4d}  {ticker:<8s}"

        if should_skip(rank, ticker, args.force_rerun, args.rerun_failed):
            manifest = read_manifest(rank, ticker) or {}
            prior = manifest.get("status", "?")
            records.append({
                "volume_rank": rank, "ticker": ticker, "company": company,
                "status": "skipped_cached", "cache_status": prior,
                "ok_providers": manifest.get("ok_provider_count"),
                "fail_providers": manifest.get("failed_provider_count"),
                "error": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            write_progress(records)
            write_status("running", current_rank=rank, current_ticker=ticker,
                         current_step="skipped", total=total, done=i + 1)
            _log(f"{prefix} — ⏭  skipped (already cached: {prior})")
            continue

        ticker_started_at = datetime.now(timezone.utc).isoformat()
        _log(f"{prefix} — ▶  starting  |  {providers_label}")

        # ── Query each provider individually so we log exactly which one is slow ──
        prov_results: dict[str, dict] = {}

        def _run_with_timeout(fn, timeout_s, label):
            """Run fn() in a thread; return (result, elapsed, error_str).

            We deliberately avoid the `with ThreadPoolExecutor` context manager
            because its __exit__ calls shutdown(wait=True), which blocks until the
            background thread finishes — defeating the timeout entirely when the
            thread is stuck in a blocking network call.  Instead we call
            shutdown(wait=False) so the stuck thread is abandoned as a daemon and
            the main loop can move on immediately.
            """
            t0 = time.monotonic()
            ex = ThreadPoolExecutor(max_workers=1)
            fut = ex.submit(fn)
            try:
                result = fut.result(timeout=timeout_s)
                ex.shutdown(wait=False)
                return result, round(time.monotonic() - t0, 1), None
            except FuturesTimeoutError:
                ex.shutdown(wait=False, cancel_futures=True)
                return None, round(time.monotonic() - t0, 1), f"TIMEOUT after {timeout_s}s"
            except Exception as exc:
                ex.shutdown(wait=False)
                return None, round(time.monotonic() - t0, 1), str(exc)

        try:
            write_status("running", current_rank=rank, current_ticker=ticker,
                         current_step="starting providers",
                         total=total, done=i, ticker_started_at=ticker_started_at)

            # ── WRDS ────────────────────────────────────────────────────────────
            if args.wrds:
                write_status("running", current_rank=rank, current_ticker=ticker,
                             current_step="querying WRDS/CRSP",
                             total=total, done=i, ticker_started_at=ticker_started_at)
                _log(f"{prefix}    ↳ WRDS/CRSP  …")
                wrds_result, wrds_elapsed, wrds_err = _run_with_timeout(
                    lambda: live_data.query_wrds_ticker_data(
                        ticker, args.start, args.end, 10_000
                    ),
                    args.provider_timeout, "WRDS"
                )
                if wrds_err:
                    prov_results["wrds"] = {"status": "failed" if "TIMEOUT" not in wrds_err else "timeout",
                                            "error": wrds_err, "prices": pd.DataFrame(), "names": pd.DataFrame(), "rows": 0}
                    _log(f"{prefix}    ↳ WRDS/CRSP  ❌  [{wrds_elapsed}s]  {wrds_err}")
                else:
                    name_history, daily_lookup = wrds_result
                    prices = live_data.wrds_price_frame(daily_lookup)
                    rows = len(prices) + len(name_history)
                    prov_results["wrds"] = {"status": "ok" if not prices.empty else "empty",
                                            "error": None, "prices": prices, "names": name_history, "rows": rows}
                    _log(f"{prefix}    ↳ WRDS/CRSP  ✓  [{wrds_elapsed}s]  {rows} rows")

            # ── Yahoo ────────────────────────────────────────────────────────────
            if args.yahoo:
                write_status("running", current_rank=rank, current_ticker=ticker,
                             current_step="querying Yahoo Finance",
                             total=total, done=i, ticker_started_at=ticker_started_at)
                _log(f"{prefix}    ↳ Yahoo Finance  …")
                yh_result, yh_elapsed, yh_err = _run_with_timeout(
                    lambda: live_data.fetch_yahoo_daily(ticker, args.start, args.end),
                    args.provider_timeout, "Yahoo"
                )
                if yh_err:
                    prov_results["yahoo"] = {"status": "failed" if "TIMEOUT" not in yh_err else "timeout",
                                             "error": yh_err, "prices": pd.DataFrame(), "rows": 0}
                    _log(f"{prefix}    ↳ Yahoo Finance  ❌  [{yh_elapsed}s]  {yh_err}")
                else:
                    yh_prices = live_data.yahoo_price_frame(yh_result)
                    rows = len(yh_prices)
                    prov_results["yahoo"] = {"status": "ok" if not yh_prices.empty else "empty",
                                             "error": None, "prices": yh_prices, "rows": rows}
                    _log(f"{prefix}    ↳ Yahoo Finance  ✓  [{yh_elapsed}s]  {rows} rows")

            # ── RavenPack ────────────────────────────────────────────────────────
            if args.ravenpack:
                write_status("running", current_rank=rank, current_ticker=ticker,
                             current_step="querying RavenPack",
                             total=total, done=i, ticker_started_at=ticker_started_at)
                _log(f"{prefix}    ↳ RavenPack  … (year-by-year, {args.year_timeout}s/yr)")
                _permno = int(row["permno"]) if "permno" in row.index else None

                # Per-year callback — runs in the background thread but writes to
                # the log file and status JSON which are safe for cross-thread use.
                def _rp_year_cb(yr, n_rows, elapsed, error, _prefix=prefix, _rank=rank,
                                _ticker=ticker, _total=total, _done=i,
                                _started=ticker_started_at):
                    if error:
                        _log(f"{_prefix}    ↳ RavenPack {yr}  ⚠  [{elapsed}s]  {error}")
                    else:
                        mark = "✓" if n_rows > 0 else "—"
                        _log(f"{_prefix}    ↳ RavenPack {yr}  {mark}  [{elapsed}s]  {n_rows} rows")
                    write_status("running", current_rank=_rank, current_ticker=_ticker,
                                 current_step=f"RavenPack {yr}",
                                 total=_total, done=_done, ticker_started_at=_started)

                rp_result, rp_elapsed, rp_err = _run_with_timeout(
                    lambda: live_data.query_ravenpack_articles(
                        ticker, args.start, args.end,
                        permno=_permno,
                        year_progress_callback=_rp_year_cb,
                        year_timeout_s=args.year_timeout,
                    ),
                    args.provider_timeout, "RavenPack"
                )
                if rp_err:
                    prov_results["ravenpack"] = {"status": "failed" if "TIMEOUT" not in rp_err else "timeout",
                                                 "error": rp_err, "articles": pd.DataFrame(), "rows": 0}
                    _log(f"{prefix}    ↳ RavenPack  ❌  [{rp_elapsed}s]  {rp_err}")
                else:
                    rows = len(rp_result) if rp_result is not None else 0
                    prov_results["ravenpack"] = {
                        "status": "ok" if rows > 0 else "empty",
                        "error": None,
                        "articles": rp_result if rp_result is not None else pd.DataFrame(),
                        "rows": rows,
                    }
                    _log(f"{prefix}    ↳ RavenPack  ✓  [{rp_elapsed}s]  {rows} rows total")

            # ── Refinitiv ────────────────────────────────────────────────────────
            if args.refinitiv:
                write_status("running", current_rank=rank, current_ticker=ticker,
                             current_step="querying Refinitiv",
                             total=total, done=i, ticker_started_at=ticker_started_at)
                _log(f"{prefix}    ↳ Refinitiv  …")
                rf_result, rf_elapsed, rf_err = _run_with_timeout(
                    lambda: live_data.run_ticker_data_query(
                        PROJECT_ROOT, ticker, args.start, args.end,
                        query_refinitiv=True, query_wrds=False,
                        query_yahoo=False, query_ravenpack=False,
                        news_count=1, wrds_limit=0,
                    ),
                    args.provider_timeout, "Refinitiv"
                )
                if rf_err:
                    prov_results["refinitiv"] = {"status": "failed" if "TIMEOUT" not in rf_err else "timeout",
                                                  "error": rf_err, "prices": pd.DataFrame(), "news": pd.DataFrame(), "rows": 0}
                    _log(f"{prefix}    ↳ Refinitiv  ❌  [{rf_elapsed}s]  {rf_err}")
                else:
                    rf_prov = rf_result["providers"].get("refinitiv", {})
                    rows = len(rf_prov.get("prices", pd.DataFrame())) + len(rf_prov.get("news", pd.DataFrame()))
                    prov_results["refinitiv"] = {**rf_prov, "rows": rows}
                    _log(f"{prefix}    ↳ Refinitiv  ✓  [{rf_elapsed}s]  {rows} rows")

            # ── Build result dict and save ────────────────────────────────────────
            write_status("running", current_rank=rank, current_ticker=ticker,
                         current_step="saving results",
                         total=total, done=i, ticker_started_at=ticker_started_at)
            _log(f"{prefix}    ↳ saving …")

            # Assemble a result dict compatible with save_ticker_result
            assembled = {
                "ticker": ticker,
                "start_date": args.start,
                "end_date": args.end,
                "providers": prov_results,
                "selected_providers": {
                    "refinitiv": args.refinitiv,
                    "wrds": args.wrds,
                    "yahoo": args.yahoo,
                    "ravenpack": args.ravenpack,
                },
            }
            manifest = save_ticker_result(row, assembled)
            run_status = manifest["status"]
            ok_n = manifest["ok_provider_count"]
            fail_n = manifest["failed_provider_count"]
            error_msg = None
            consecutive_failures = 0 if run_status != "failed" else consecutive_failures + 1

            elapsed = round((datetime.now(timezone.utc) - datetime.fromisoformat(ticker_started_at)).total_seconds(), 1)
            icon = "✅" if run_status == "complete" else ("⚠️ " if run_status == "partial" else "❌")
            _log(f"{prefix} — {icon}  {run_status}  [{elapsed}s total]  ok={ok_n}  fail={fail_n}")

        except Exception as exc:
            run_status = "error"
            ok_n = 0
            fail_n = None
            error_msg = str(exc)
            consecutive_failures += 1
            elapsed = round((datetime.now(timezone.utc) - datetime.fromisoformat(ticker_started_at)).total_seconds(), 1)
            _log(f"{prefix} — ❌  ERROR [{elapsed}s]: {exc}")

        records.append({
            "volume_rank": rank, "ticker": ticker, "company": company,
            "status": run_status, "cache_status": run_status,
            "ok_providers": ok_n, "fail_providers": fail_n,
            "error": error_msg,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        write_progress(records)
        write_status("running", current_rank=rank, current_ticker=ticker,
                     current_step="done", total=total, done=i + 1,
                     ticker_started_at=ticker_started_at)

        if consecutive_failures >= args.stop_after_failures:
            msg = f"Stopped after {consecutive_failures} consecutive failures"
            _log(f"[batch] ⛔  {msg}")
            write_status("stopped_failures", current_rank=rank, current_ticker=ticker,
                         total=total, done=i + 1, error=msg)
            PID_FILE.unlink(missing_ok=True)
            return

        if args.sleep > 0 and i + 1 < total:
            time.sleep(args.sleep)

    if args.combined_parquets:
        _log("[batch] Writing combined parquets …")
        write_combined_parquets([
            "wrds_prices", "wrds_names", "yahoo_prices",
            "ravenpack_articles", "refinitiv_prices", "refinitiv_news",
        ])

    complete_n  = sum(1 for r in records if r["status"] == "complete")
    partial_n   = sum(1 for r in records if r["status"] == "partial")
    failed_n    = sum(1 for r in records if r["status"] in ("failed", "error"))
    skipped_n   = sum(1 for r in records if r["status"] == "skipped_cached")
    _log("-" * 60)
    _log(f"[batch] ✅  ALL DONE — {len(records)} tickers processed")
    _log(f"[batch]    complete={complete_n}  partial={partial_n}  failed={failed_n}  skipped={skipped_n}")
    write_status("complete", total=total, done=len(records))
    PID_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
