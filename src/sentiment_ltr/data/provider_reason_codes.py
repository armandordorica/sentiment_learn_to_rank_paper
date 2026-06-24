"""Classify why a data provider did not return usable rows for a ticker.

Each non-ok provider outcome gets a stable machine code (``fail_reason``) plus a
short human label (``fail_reason_label``).  Codes are inferred from error text,
provider status, and optional CRSP context (last trade date, permno/ticker mismatch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

# Machine code → short UI description
REASON_LABELS: dict[str, str] = {
    "not_requested": "Provider was not selected for this run",
    "credentials_missing": "Credentials or config not available",
    "unavailable": "Provider unavailable in this environment",
    "query_timeout": "Query exceeded the provider timeout",
    "query_error": "Unhandled provider error",
    "no_rows": "Provider returned no rows for the date range",
    "delisted_no_vendor_history": (
        "Company delisted/merged — Yahoo does not serve historical data for this symbol"
    ),
    "ticker_recycled": (
        "Ticker symbol was later reused by a different company; vendor points at the wrong issuer"
    ),
    "rate_limited": "Yahoo Finance rate-limited the request",
    "network_blocked": "Yahoo Finance blocked by network/proxy",
    "ric_unresolved": "Refinitiv could not resolve a live RIC for this symbol",
    "delisted_ric_retired": "Refinitiv RIC retired — company delisted or merged",
    "insufficient_scope": "Refinitiv account lacks the required API scope",
    "no_vendor_history": "Refinitiv resolved the symbol but returned no price history",
    "no_crsp_rows": "No CRSP rows in the selected date range",
    "date_range_after_crsp_cutoff": "Date range extends beyond latest CRSP data in WRDS",
    "no_entity_mapping": "No RavenPack entity mapping for this company (CUSIP/ticker)",
    "ticker_recycled_wrong_entity": (
        "RavenPack matched the current ticker holder, not the historical company"
    ),
    "delisted_no_entity_or_articles": (
        "Delisted/merged company — no RavenPack entity or articles for this issuer"
    ),
    "no_articles_in_window": "RavenPack entity found but no articles in the date window",
}


@dataclass
class ProviderContext:
    ticker: str
    permno: int | None = None
    query_start: str | None = None
    query_end: str | None = None
    wrds_last_trade_date: pd.Timestamp | None = None
    current_ticker: str | None = None
    ticker_permno_mismatch: bool | None = None  # LU→Lufax style recycle


def reason_label(code: str | None) -> str | None:
    if not code:
        return None
    return REASON_LABELS.get(code, code.replace("_", " ").capitalize())


def _err_lower(error: str | None) -> str:
    return (error or "").lower()


def _likely_delisted(ctx: ProviderContext) -> bool:
    """WRDS history ends well before the query end → merger/delisting likely."""
    if ctx.wrds_last_trade_date is None or not ctx.query_end:
        return False
    gap = (pd.Timestamp(ctx.query_end).normalize() - ctx.wrds_last_trade_date.normalize()).days
    return gap > 60


def _ticker_recycled(ctx: ProviderContext) -> bool:
    if ctx.ticker_permno_mismatch is not None:
        return ctx.ticker_permno_mismatch
    if ctx.permno is None or not ctx.ticker:
        return False
    try:
        from sentiment_ltr.data.live_data import _lookup_permno_for_ticker

        looked_up = _lookup_permno_for_ticker(ctx.ticker.upper().strip())
        return looked_up is not None and int(looked_up) != int(ctx.permno)
    except Exception:
        return False


def _provider_row_count(payload: dict[str, Any]) -> int:
    total = 0
    for key, val in payload.items():
        if isinstance(val, pd.DataFrame) and key in (
            "prices", "news", "news_daily_counts", "names", "articles",
        ):
            total += len(val)
    return total


def classify_provider_reason(
    provider: str,
    status: str | None,
    error: str | None,
    ctx: ProviderContext,
    *,
    rows: int | None = None,
) -> str | None:
    """Return a machine reason code, or None when the provider succeeded."""
    status = str(status or "unknown")
    if status == "ok":
        return None

    err = _err_lower(error)
    row_n = rows if rows is not None else 0

    if status == "skipped":
        return "not_requested"
    if status == "unavailable":
        if "credential" in err or "not configured" in err:
            return "credentials_missing"
        return "unavailable"
    if status == "timeout" or "timeout" in err:
        return "query_timeout"

    if provider == "yahoo":
        if "rate" in err and "limit" in err:
            return "rate_limited"
        if "blocked" in err or "403" in err or "tunnel" in err:
            return "network_blocked"
        if _ticker_recycled(ctx):
            return "ticker_recycled"
        if (
            _likely_delisted(ctx)
            or "delisted" in err
            or "possibly delisted" in err
            or (status == "failed" and "no rows" in err)
        ):
            return "delisted_no_vendor_history"
        if status in ("empty", "failed") and ("no rows" in err or row_n == 0):
            return "no_rows"
        return "query_error"

    if provider == "refinitiv":
        if "insufficient scope" in err:
            return "insufficient_scope"
        if "unable to resolve" in err:
            return "delisted_ric_retired" if _likely_delisted(ctx) else "ric_unresolved"
        if status == "empty" or "no refinitiv" in err or "no price" in err:
            return "delisted_ric_retired" if _likely_delisted(ctx) else "no_vendor_history"
        return "query_error"

    if provider == "ravenpack":
        if "no ravenpack entity" in err:
            return "no_entity_mapping"
        if status == "empty" and _ticker_recycled(ctx):
            return "ticker_recycled_wrong_entity"
        if status == "empty" and _likely_delisted(ctx):
            return "delisted_no_entity_or_articles"
        if status == "empty" or (row_n == 0 and "no " in err):
            return "no_articles_in_window"
        return "query_error"

    if provider == "wrds":
        if "after the latest crsp" in err:
            return "date_range_after_crsp_cutoff"
        if status == "empty" or row_n == 0:
            return "no_crsp_rows"
        return "query_error"

    if status in ("empty", "failed") and row_n == 0:
        return "no_rows"
    return "query_error"


def wrds_last_trade_date(providers: dict[str, dict[str, Any]]) -> pd.Timestamp | None:
    wr = providers.get("wrds", {})
    prices = wr.get("prices")
    if not isinstance(prices, pd.DataFrame) or prices.empty or "date" not in prices.columns:
        return None
    return pd.to_datetime(prices["date"]).max().normalize()


def build_provider_context(
    ticker: str,
    permno: int | None,
    query_start: str | None,
    query_end: str | None,
    providers: dict[str, dict[str, Any]],
    *,
    current_ticker: str | None = None,
    skip_permno_lookup: bool = False,
) -> ProviderContext:
    mismatch = False
    if not skip_permno_lookup and permno is not None and ticker:
        try:
            from sentiment_ltr.data.live_data import _lookup_permno_for_ticker

            looked_up = _lookup_permno_for_ticker(ticker.upper().strip())
            mismatch = looked_up is not None and int(looked_up) != int(permno)
        except Exception:
            pass
    return ProviderContext(
        ticker=ticker.upper().strip(),
        permno=permno,
        query_start=query_start,
        query_end=query_end,
        wrds_last_trade_date=wrds_last_trade_date(providers),
        current_ticker=current_ticker,
        ticker_permno_mismatch=mismatch,
    )


def enrich_provider_status_records(
    records: list[dict],
    *,
    ticker: str,
    permno: int | None,
    query_start: str | None,
    query_end: str | None,
    wrds_last_trade_date: pd.Timestamp | None = None,
    skip_permno_lookup: bool = False,
) -> list[dict]:
    """Backfill fail_reason on saved provider_status rows (e.g. older manifests)."""
    providers = {
        str(r.get("provider", "")): {
            "status": r.get("status"),
            "error": r.get("error"),
            "rows": r.get("rows", 0),
        }
        for r in records
    }
    if wrds_last_trade_date is not None:
        providers["wrds"] = {"prices": pd.DataFrame({"date": [wrds_last_trade_date]})}
    ctx = build_provider_context(
        ticker, permno, query_start, query_end, providers,
        skip_permno_lookup=skip_permno_lookup,
    )
    out: list[dict] = []
    for rec in records:
        rec = dict(rec)
        if not rec.get("fail_reason"):
            pname = str(rec.get("provider", ""))
            code = classify_provider_reason(
                pname,
                str(rec.get("status", "")),
                rec.get("error"),
                ctx,
                rows=int(rec.get("rows") or 0),
            )
            rec["fail_reason"] = code
            rec["fail_reason_label"] = reason_label(code)
        out.append(rec)
    return out


def annotate_provider_results(
    providers: dict[str, dict[str, Any]],
    ctx: ProviderContext,
) -> None:
    """Attach fail_reason / fail_reason_label to each provider payload in-place."""
    for name, payload in providers.items():
        if not isinstance(payload, dict):
            continue
        code = classify_provider_reason(
            name,
            str(payload.get("status", "")),
            payload.get("error") if payload.get("error") is not None else None,
            ctx,
            rows=_provider_row_count(payload),
        )
        payload["fail_reason"] = code
        payload["fail_reason_label"] = reason_label(code)
