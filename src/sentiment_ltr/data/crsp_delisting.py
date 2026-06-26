"""CRSP delisting information — the gold standard for why a stock left the market.

CRSP's delisting event table (``crsp.msedelist``) records, for every security
that stopped trading, a delisting date (``dlstdt``), a delisting code
(``dlstcd``) describing *why* it left, and the delisting return (``dlret``) an
investor earned on the way out (e.g. cash from a merger, or a wipe-out in a
bankruptcy).  This module batch-queries that table by PERMNO and maps the
numeric codes to human-readable categories and labels.

Delisting code structure (first digit = broad category):

* ``100``     — still active / trading
* ``2xx``     — merger / acquisition (absorbed into another company)
* ``3xx``     — exchange (moved exchange, or became a different security)
* ``4xx``     — liquidation
* ``5xx``     — dropped / delisted for cause (low price, insufficient capital,
                bankruptcy, filing delinquency, exchange-rule violations, etc.)
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from sentiment_ltr.data.live_data import open_wrds_connection

# Default on-disk cache location (one row per *checked* PERMNO so we never
# re-query a name we have already looked up, even when it never delisted).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELISTING_CACHE_PATH = (
    PROJECT_ROOT / "data" / "raw" / "data_explorer_top1k" / "delisting_info.parquet"
)

# Broad category by the hundreds digit of dlstcd.
DELISTING_CATEGORY_BY_HUNDRED: dict[int, str] = {
    1: "active",
    2: "merger",
    3: "exchange",
    4: "liquidation",
    5: "dropped",
}

# Specific dlstcd → short label for the codes seen most often in US equities.
DELISTING_CODE_LABELS: dict[int, str] = {
    100: "Active — issue still trading",
    200: "Merger / acquisition",
    201: "Merged — became a different listed company",
    202: "Merged — holding company exchange",
    203: "Merged — leveraged buyout",
    204: "Merged — reverse takeover",
    231: "Merger — shareholders received primarily stock",
    232: "Merger — shareholders received stock and cash",
    233: "Merger — shareholders received primarily cash",
    241: "Merger — payment method unknown",
    242: "Merger — additional payment terms",
    300: "Exchange — moved to a different exchange",
    301: "Exchange — became a new issue",
    331: "Exchange — for another class of stock",
    332: "Exchange — for a different security",
    400: "Liquidation",
    450: "Liquidation — company liquidated",
    470: "Liquidation — distribution to shareholders",
    480: "Liquidation — bankruptcy / Chapter proceedings",
    490: "Liquidation — other",
    500: "Dropped — reason unavailable",
    501: "Dropped — insufficient number of shareholders",
    502: "Dropped — insufficient capital, surplus, or equity",
    503: "Dropped — price fell below acceptable level",
    504: "Dropped — insufficient float or assets",
    505: "Dropped — company request (no reason given)",
    510: "Dropped — delisting required by exchange",
    513: "Dropped — delinquent / non-compliance",
    514: "Dropped — registration cancelled by SEC",
    517: "Dropped — corporate governance issues",
    519: "Dropped — does not meet exchange financial guidelines",
    520: "Dropped — delisting required (general)",
    551: "Dropped — insufficient market makers",
    552: "Dropped — insufficient number of shareholders",
    560: "Dropped — in violation of exchange rules",
    561: "Dropped — failure to register / file",
    570: "Dropped — delinquent in financial reports",
    572: "Dropped — bankruptcy declared",
    573: "Dropped — liquidation / receivership",
    574: "Dropped — bankruptcy / insolvency",
    575: "Dropped — went private",
    580: "Dropped — delisted at company request",
    581: "Dropped — non-payment of fees",
    582: "Dropped — failure to meet listing standards",
    584: "Dropped — does not meet exchange financial guidelines",
    587: "Dropped — protection of investors",
}

DELISTING_COLUMNS = [
    "permno",
    "dlstdt",
    "dlstcd",
    "dlret",
    "dlretx",
    "dlamt",
    "dlpdt",
    "nwperm",
    "nwcomp",
    "nextdt",
]


def delisting_category(dlstcd: object) -> str:
    """Map a raw dlstcd to a broad category (active/merger/exchange/...)."""
    try:
        code = int(dlstcd)
    except (TypeError, ValueError):
        return "unknown"
    return DELISTING_CATEGORY_BY_HUNDRED.get(code // 100, "unknown")


def delisting_label(dlstcd: object) -> str:
    """Map a raw dlstcd to a human-readable label, falling back to the category."""
    try:
        code = int(dlstcd)
    except (TypeError, ValueError):
        return "Unknown delisting code"
    if code in DELISTING_CODE_LABELS:
        return DELISTING_CODE_LABELS[code]
    category = delisting_category(code)
    return f"{category.capitalize()} (code {code})"


def is_crsp_delisted(dlstcd: object) -> bool:
    """Return True when dlstcd indicates the security actually left the market.

    CRSP ``msedelist`` includes rows with ``dlstcd = 100`` for issues that are
    still trading; those are *not* delistings.
    """
    try:
        return int(dlstcd) != 100
    except (TypeError, ValueError):
        return False


def query_crsp_delisting(
    permnos: Iterable[int],
    *,
    table: str = "crsp.msedelist",
    chunk_size: int = 500,
) -> pd.DataFrame:
    """Return CRSP delisting rows for the given PERMNOs.

    One row per delisted security: delisting date, code, and returns. PERMNOs
    that never delisted (still active) simply won't appear in the result.
    """
    permno_list = sorted({int(p) for p in permnos if pd.notna(p)})
    if not permno_list:
        return pd.DataFrame(columns=DELISTING_COLUMNS)

    db = open_wrds_connection()
    frames: list[pd.DataFrame] = []
    try:
        for start in range(0, len(permno_list), chunk_size):
            chunk = permno_list[start:start + chunk_size]
            in_clause = ", ".join(str(p) for p in chunk)
            query = f"""
                select {", ".join(DELISTING_COLUMNS)}
                from {table}
                where permno in ({in_clause})
            """
            frame = db.raw_sql(query, date_cols=["dlstdt", "dlpdt", "nextdt"])
            if not frame.empty:
                frames.append(frame)
    finally:
        db.close()

    if not frames:
        return pd.DataFrame(columns=DELISTING_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    result["dlstcd"] = pd.to_numeric(result["dlstcd"], errors="coerce").astype("Int64")
    result["delisting_category"] = result["dlstcd"].apply(delisting_category)
    result["delisting_label"] = result["dlstcd"].apply(delisting_label)
    return result


# ── Local cache (only query PERMNOs we have never checked) ─────────────────────

CACHE_COLUMNS = [
    "permno", "delisted", "dlstdt", "dlstcd", "dlret", "dlretx",
    "nwperm", "delisting_category", "delisting_label", "checked_at",
]

# Fixed dtypes so concatenating active (all-NA) rows never triggers pandas'
# dtype-inference FutureWarning and the cache round-trips cleanly via parquet.
CACHE_DTYPES: dict[str, str] = {
    "permno": "Int64",
    "delisted": "boolean",
    "dlstdt": "datetime64[ns]",
    "dlstcd": "Int64",
    "dlret": "float64",
    "dlretx": "float64",
    "nwperm": "Int64",
    "delisting_category": "object",
    "delisting_label": "object",
    "checked_at": "object",
}


def _coerce_cache_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with the canonical cache columns and dtypes (concat-safe)."""
    out = df.copy()
    for col in CACHE_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[CACHE_COLUMNS]
    for col, dtype in CACHE_DTYPES.items():
        try:
            if dtype == "datetime64[ns]":
                out[col] = pd.to_datetime(out[col], errors="coerce")
            elif dtype in ("Int64", "float64"):
                out[col] = pd.to_numeric(out[col], errors="coerce").astype(dtype)
            else:
                out[col] = out[col].astype(dtype)
        except (TypeError, ValueError):
            pass
    return out


def load_delisting_cache(cache_path: Path = DELISTING_CACHE_PATH) -> pd.DataFrame:
    """Return the cached delisting frame (one row per checked PERMNO), or empty."""
    if not cache_path.exists():
        return pd.DataFrame(columns=CACHE_COLUMNS)
    try:
        cached = pd.read_parquet(cache_path)
    except Exception:
        return pd.DataFrame(columns=CACHE_COLUMNS)
    if "permno" in cached.columns:
        cached["permno"] = pd.to_numeric(cached["permno"], errors="coerce").astype("Int64")
    return cached


def cached_permnos(cache_path: Path = DELISTING_CACHE_PATH) -> set[int]:
    """Return the set of PERMNOs already checked (present in the cache)."""
    cached = load_delisting_cache(cache_path)
    if cached.empty:
        return set()
    return {int(p) for p in cached["permno"].dropna().tolist()}


def _build_checked_rows(missing: list[int], fetched: pd.DataFrame) -> pd.DataFrame:
    """Make one cache row per checked PERMNO: delisted record or 'still active'."""
    now = datetime.now(timezone.utc).isoformat()
    delisted_permnos: set[int] = set()
    rows: list[dict] = []

    if not fetched.empty:
        # Keep the most recent msedelist record per PERMNO.
        fetched = fetched.sort_values("dlstdt").drop_duplicates("permno", keep="last")
        for _, r in fetched.iterrows():
            permno = int(r["permno"])
            delisted_permnos.add(permno)
            actually_delisted = is_crsp_delisted(r.get("dlstcd"))
            rows.append({
                "permno": permno,
                "delisted": actually_delisted,
                "dlstdt": r.get("dlstdt") if actually_delisted else pd.NaT,
                "dlstcd": r.get("dlstcd"),
                "dlret": r.get("dlret") if actually_delisted else pd.NA,
                "dlretx": r.get("dlretx") if actually_delisted else pd.NA,
                "nwperm": r.get("nwperm") if actually_delisted else pd.NA,
                "delisting_category": (
                    r.get("delisting_category") if actually_delisted else "active"
                ),
                "delisting_label": (
                    r.get("delisting_label") if actually_delisted
                    else DELISTING_CODE_LABELS[100]
                ),
                "checked_at": now,
            })

    for permno in missing:
        if int(permno) in delisted_permnos:
            continue
        rows.append({
            "permno": int(permno),
            "delisted": False,
            "dlstdt": pd.NaT,
            "dlstcd": pd.NA,
            "dlret": pd.NA,
            "dlretx": pd.NA,
            "nwperm": pd.NA,
            "delisting_category": "active_or_no_record",
            "delisting_label": "Still active / no CRSP delisting record",
            "checked_at": now,
        })

    return pd.DataFrame(rows, columns=CACHE_COLUMNS)


def update_delisting_cache(
    permnos: Iterable[int],
    *,
    cache_path: Path = DELISTING_CACHE_PATH,
    force: bool = False,
) -> tuple[pd.DataFrame, int]:
    """Look up only the PERMNOs not already in the cache, then persist.

    Returns ``(full_cache_frame, n_newly_queried)``. When ``force`` is True the
    requested PERMNOs are re-queried even if already cached.
    """
    requested = sorted({int(p) for p in permnos if pd.notna(p)})
    existing = load_delisting_cache(cache_path)
    already = set() if force else cached_permnos(cache_path)
    missing = [p for p in requested if p not in already]

    if not missing:
        return existing, 0

    fetched = query_crsp_delisting(missing)
    new_rows = _coerce_cache_schema(_build_checked_rows(missing, fetched))

    if force and not existing.empty:
        existing = existing[~existing["permno"].isin(missing)]
    # Coerce both sides to the same schema so concat never infers dtypes from
    # all-NA columns (which raises a pandas FutureWarning).
    frames = [_coerce_cache_schema(f) for f in (existing, new_rows) if not f.empty]
    combined = pd.concat(frames, ignore_index=True) if frames else new_rows
    combined = combined.drop_duplicates("permno", keep="last").reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)
    return combined, len(missing)
