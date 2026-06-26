"""Cash-merger exit returns for delisted stocks (CRSP dlstcd 232 / 233).

When a stock leaves the market via a cash merger, its final-week return is often
missing from the daily file. CRSP's delisting return (``dlret``) captures the
cash-out when present; when it is missing we estimate it from the SDC M&A deal
price (``pricepersh``); failing that we fall back to the last traded price (a
0% exit return). Each estimate is tagged with its ``exit_source`` so downstream
backtests know how reliable it is.

Delisting codes handled here (cash mergers only):

* ``232`` — merger, shareholders received cash and stock
* ``233`` — merger, shareholders received primarily cash
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

CASH_MERGER_CODES = (232, 233)

EXIT_COLUMNS = ["permno", "dlstdt", "dlprc", "exit_return", "exit_source"]

EXIT_SOURCE_LABELS: dict[str, str] = {
    "crsp_dlret": "Exit return from CRSP dlret",
    "sdc_pricepersh": "Exit return estimated from SDC deal price",
    "crsp_dlprc_fallback": "Fallback: last price used, return = 0",
    "not_cash_merger": "Not a cash merger (no exit return applied)",
}

# On-disk cache (one row per checked PERMNO so non-mergers are not re-queried).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CASH_MERGER_CACHE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "cash_merger_exits.parquet"
)


def get_cash_merger_exit_returns(permnos: list[int], db) -> pd.DataFrame:
    """Return cash-merger exit returns for the given PERMNOs.

    ``db`` is an open WRDS connection (same object used elsewhere in the codebase).
    Output columns: ``permno | dlstdt | dlprc | exit_return | exit_source`` where
    ``exit_source`` ∈ {``crsp_dlret``, ``sdc_pricepersh``, ``crsp_dlprc_fallback``}.
    Only PERMNOs whose delisting code is a cash merger (232/233) appear in the result.
    """
    clean = sorted({int(p) for p in permnos if pd.notna(p)})
    if not clean:
        return pd.DataFrame(columns=EXIT_COLUMNS)

    placeholders = ", ".join(str(p) for p in clean)

    # Step 1 — CRSP daily delisting events for cash mergers.
    query = f"""
        SELECT permno, dlstdt, dlprc, dlret
        FROM crsp.dsedelist
        WHERE permno IN ({placeholders})
        AND dlstcd IN (232, 233)
    """
    delist = db.raw_sql(query, date_cols=["dlstdt"])
    if delist.empty:
        return pd.DataFrame(columns=EXIT_COLUMNS)

    delist["permno"] = delist["permno"].astype(int)
    delist["dlprc_abs"] = pd.to_numeric(delist["dlprc"], errors="coerce").abs()

    rows: list[dict] = []
    needs_sdc: list[dict] = []
    for _, r in delist.iterrows():
        dlret = pd.to_numeric(r.get("dlret"), errors="coerce")
        if pd.notna(dlret):
            rows.append({
                "permno": int(r["permno"]),
                "dlstdt": r.get("dlstdt"),
                "dlprc": r.get("dlprc"),
                "exit_return": float(dlret),
                "exit_source": "crsp_dlret",
            })
        else:
            needs_sdc.append(r)

    # Step 2 — SDC deal price for rows still missing a return.
    if needs_sdc:
        sdc_returns = _sdc_exit_returns(needs_sdc, db)
        for r in needs_sdc:
            permno = int(r["permno"])
            est = sdc_returns.get(permno)
            if est is not None:
                rows.append({
                    "permno": permno,
                    "dlstdt": r.get("dlstdt"),
                    "dlprc": r.get("dlprc"),
                    "exit_return": float(est),
                    "exit_source": "sdc_pricepersh",
                })
            else:
                # Step 3 — fallback: last price, 0% exit return.
                print(f"[cash_merger] WARNING: no dlret or SDC deal price for permno "
                      f"{permno}; using last-price fallback (exit_return = 0.0).")
                rows.append({
                    "permno": permno,
                    "dlstdt": r.get("dlstdt"),
                    "dlprc": r.get("dlprc"),
                    "exit_return": 0.0,
                    "exit_source": "crsp_dlprc_fallback",
                })

    return pd.DataFrame(rows, columns=EXIT_COLUMNS)


def _sdc_exit_returns(delist_rows: list, db) -> dict[int, float]:
    """Estimate exit returns from SDC M&A deal prices, keyed by PERMNO.

    Defensive: SDC may not be entitled on every WRDS account. Any failure
    returns an empty mapping so the caller falls back gracefully.
    """
    permnos = [int(r["permno"]) for r in delist_rows]
    if not permnos:
        return {}

    placeholders = ", ".join(str(p) for p in permnos)
    try:
        names = db.raw_sql(
            f"""
            SELECT permno, ncusip
            FROM crsp.dsenames
            WHERE permno IN ({placeholders})
            AND ncusip IS NOT NULL
            """
        )
    except Exception as exc:  # pragma: no cover - depends on WRDS entitlements
        print(f"[cash_merger] WARNING: could not read crsp.dsenames CUSIPs: {exc}")
        return {}

    if names.empty:
        return {}

    names["permno"] = names["permno"].astype(int)
    names["ncusip"] = names["ncusip"].astype(str).str.strip()
    # SDC target CUSIPs are the 6-digit issuer CUSIP; CRSP ncusip is 8-digit.
    names["cusip6"] = names["ncusip"].str[:6]
    permno_to_cusip6: dict[int, set[str]] = {}
    for _, r in names.iterrows():
        permno_to_cusip6.setdefault(int(r["permno"]), set()).add(r["cusip6"])

    all_cusip6 = sorted({c for cs in permno_to_cusip6.values() for c in cs if c})
    if not all_cusip6:
        return {}

    cusip_in = ", ".join(f"'{c}'" for c in all_cusip6)
    try:
        sdc = db.raw_sql(
            f"""
            SELECT cusip, dteeff, pricepersh, pctcash
            FROM sdc.ma
            WHERE substr(cusip, 1, 6) IN ({cusip_in})
            AND pctcash >= 50
            AND pricepersh IS NOT NULL
            """,
            date_cols=["dteeff"],
        )
    except Exception as exc:  # pragma: no cover - depends on WRDS entitlements
        print(f"[cash_merger] WARNING: SDC M&A lookup unavailable ({exc}); "
              "falling back to last price.")
        return {}

    if sdc.empty:
        return {}

    sdc = sdc.copy()
    sdc["cusip6"] = sdc["cusip"].astype(str).str.strip().str[:6]
    sdc["dteeff"] = pd.to_datetime(sdc["dteeff"], errors="coerce")
    sdc["pricepersh"] = pd.to_numeric(sdc["pricepersh"], errors="coerce")

    out: dict[int, float] = {}
    for r in delist_rows:
        permno = int(r["permno"])
        dlstdt = pd.to_datetime(r.get("dlstdt"), errors="coerce")
        dlprc_abs = abs(pd.to_numeric(r.get("dlprc"), errors="coerce"))
        if pd.isna(dlstdt) or pd.isna(dlprc_abs) or dlprc_abs == 0:
            continue
        cusip6s = permno_to_cusip6.get(permno, set())
        if not cusip6s:
            continue
        window_lo = dlstdt - pd.Timedelta(days=180)
        window_hi = dlstdt + pd.Timedelta(days=30)
        candidates = sdc[
            sdc["cusip6"].isin(cusip6s)
            & sdc["dteeff"].between(window_lo, window_hi)
            & sdc["pricepersh"].notna()
        ]
        if candidates.empty:
            continue
        # Closest effective date to the delisting date.
        closest = candidates.iloc[
            (candidates["dteeff"] - dlstdt).abs().argsort().iloc[0]
        ]
        price_per_share = float(closest["pricepersh"])
        out[permno] = (price_per_share / float(dlprc_abs)) - 1.0

    return out


def get_cash_merger_summary(exit_df: pd.DataFrame) -> pd.DataFrame:
    """Return count and mean exit_return grouped by exit_source."""
    if exit_df is None or exit_df.empty:
        return pd.DataFrame(columns=["exit_source", "count", "mean_exit_return"])
    grouped = (
        exit_df.groupby("exit_source")["exit_return"]
        .agg(count="count", mean_exit_return="mean")
        .reset_index()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


# ── Local cache (only query PERMNOs we have never checked) ─────────────────────

CACHE_COLUMNS = EXIT_COLUMNS + ["checked_at"]


def load_cash_merger_cache(cache_path: Path = CASH_MERGER_CACHE_PATH) -> pd.DataFrame:
    """Return the cached cash-merger exit frame, or an empty frame."""
    if not cache_path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    try:
        cached = pd.read_parquet(cache_path)
    except Exception:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    if "permno" in cached.columns:
        cached["permno"] = pd.to_numeric(cached["permno"], errors="coerce").astype("Int64")
    return cached


def cached_cash_merger_permnos(cache_path: Path = CASH_MERGER_CACHE_PATH) -> set[int]:
    """Return the set of PERMNOs already checked for cash-merger exits."""
    cached = load_cash_merger_cache(cache_path)
    if cached.empty:
        return set()
    return {int(p) for p in cached["permno"].dropna().tolist()}


def update_cash_merger_cache(
    permnos: Iterable[int],
    *,
    db=None,
    cache_path: Path = CASH_MERGER_CACHE_PATH,
    force: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Resolve cash-merger exit returns for PERMNOs not already checked, then persist.

    Returns ``(full_cache_frame, n_newly_checked)``. Non-merger PERMNOs are cached
    as ``exit_source = "not_cash_merger"`` so they are not re-queried.
    """
    requested = sorted({int(p) for p in permnos if pd.notna(p)})
    existing = load_cash_merger_cache(cache_path)
    already = set() if force else cached_cash_merger_permnos(cache_path)
    missing = [p for p in requested if p not in already]
    if not missing:
        return existing, 0

    own_db = False
    if db is None:
        from sentiment_ltr.data.live_data import open_wrds_connection
        db = open_wrds_connection()
        own_db = True
    try:
        exits = get_cash_merger_exit_returns(missing, db)
    finally:
        if own_db:
            try:
                db.close()
            except Exception:
                pass

    now = datetime.now(timezone.utc).isoformat()
    merger_permnos: set[int] = set()
    rows: list[dict] = []
    if not exits.empty:
        for _, r in exits.iterrows():
            permno = int(r["permno"])
            merger_permnos.add(permno)
            rows.append({**{c: r.get(c) for c in EXIT_COLUMNS}, "checked_at": now})

    for permno in missing:
        if permno in merger_permnos:
            continue
        rows.append({
            "permno": permno, "dlstdt": pd.NaT, "dlprc": pd.NA,
            "exit_return": pd.NA, "exit_source": "not_cash_merger", "checked_at": now,
        })

    new_rows = pd.DataFrame(rows, columns=CACHE_COLUMNS)
    if force and not existing.empty:
        existing = existing[~existing["permno"].isin(missing)]
    frames = [f for f in (existing, new_rows) if not f.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else new_rows
    combined["permno"] = pd.to_numeric(combined["permno"], errors="coerce").astype("Int64")
    combined = combined.drop_duplicates("permno", keep="last").reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    return combined, len(missing)
