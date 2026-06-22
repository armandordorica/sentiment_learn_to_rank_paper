"""Shared ticker/date data pulls for the Streamlit app and notebooks."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from sentiment_ltr.data.secrets import get_env_or_secret

# news_coverage and all Refinitiv helpers are imported lazily inside run_ticker_data_query
# so that the LSEG library never auto-initialises when Refinitiv is not being used.

try:
    import wrds
except ImportError:  # pragma: no cover - environment dependent
    wrds = None

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - environment dependent
    yf = None


def to_query_date(value: pd.Timestamp | str) -> str:
    """Normalize a date-like value to YYYY-MM-DD for API/SQL queries."""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def clean_ticker(ticker: str) -> str:
    """Return an uppercase ticker safe for the app's current query patterns."""
    return "".join(char for char in ticker.upper().strip() if char.isalnum() or char in {".", "-"})


def wrds_credential_status() -> dict[str, bool]:
    """Return non-sensitive WRDS credential presence checks."""
    return {
        "WRDS_USERNAME": bool(get_env_or_secret("WRDS_USERNAME")),
        "WRDS_PASSWORD": bool(get_env_or_secret("WRDS_PASSWORD")),
    }


def wrds_credentials_available() -> bool:
    """Return whether enough WRDS configuration exists for live queries."""
    status = wrds_credential_status()
    return status["WRDS_USERNAME"] and status["WRDS_PASSWORD"]


def open_wrds_connection():
    """Open WRDS without allowing the library to fall back to interactive prompts."""
    if wrds is None:
        raise RuntimeError("The `wrds` package is not installed in this environment.")

    wrds_username = get_env_or_secret("WRDS_USERNAME")
    wrds_password = get_env_or_secret("WRDS_PASSWORD")
    if not wrds_username or not wrds_password:
        raise RuntimeError("WRDS credentials are not configured.")

    db = wrds.Connection(
        autoconnect=False,
        wrds_username=str(wrds_username).strip(),
        wrds_password=str(wrds_password),
    )
    try:
        # `wrds.Connection.connect()` prompts with input() after a failed first
        # attempt. In Streamlit/notebooks that can become "EOF when reading a line".
        db._Connection__make_sa_engine_conn(raise_err=True)
    except Exception as exc:
        try:
            db.close()
        except Exception:
            pass
        raise RuntimeError(
            "WRDS login failed non-interactively. Check `WRDS_USERNAME` and "
            "`WRDS_PASSWORD` in `.env` or Streamlit secrets; this code cannot "
            "answer WRDS terminal prompts."
        ) from exc

    if db.engine is None:
        raise RuntimeError("WRDS login failed; no database engine was created.")
    return db


def query_wrds_ticker_data(
    ticker: str,
    start_date: str,
    end_date: str,
    row_limit: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query CRSP name history and daily stock data for a ticker."""
    ticker_clean = clean_ticker(ticker)
    if not ticker_clean:
        raise ValueError("Enter a valid ticker.")

    start_s = to_query_date(start_date)
    end_s = to_query_date(end_date)

    db = open_wrds_connection()
    try:
        names_query = f"""
        select
            permno,
            permco,
            namedt,
            nameendt,
            ticker,
            comnam,
            shrcd,
            exchcd
        from crsp.msenames
        where trim(ticker) = '{ticker_clean}'
          and namedt <= '{end_s}'
          and nameendt >= '{start_s}'
        order by namedt, permno
        """
        names = db.raw_sql(names_query, date_cols=["namedt", "nameendt"])

        if names.empty:
            fallback_names_query = f"""
            select
                permno,
                permco,
                namedt,
                nameendt,
                ticker,
                comnam,
                shrcd,
                exchcd
            from crsp.msenames
            where trim(ticker) = '{ticker_clean}'
            order by nameendt desc, namedt desc
            """
            names = db.raw_sql(fallback_names_query, date_cols=["namedt", "nameendt"])

        if names.empty:
            return names, pd.DataFrame()

        permno_sql = ", ".join(str(int(permno)) for permno in sorted(names["permno"].dropna().unique()))
        daily_query = f"""
        select
            d.permno,
            n.permco,
            d.date,
            n.ticker,
            n.comnam,
            n.shrcd,
            n.exchcd,
            d.openprc,
            d.prc,
            d.ret,
            d.retx,
            d.vol,
            d.shrout,
            d.cfacpr,
            d.cfacshr,
            d.bidlo,
            d.askhi
        from crsp.dsf as d
        join crsp.msenames as n
          on d.permno = n.permno
         and d.date between n.namedt and n.nameendt
        where d.date between '{start_s}' and '{end_s}'
          and d.permno in ({permno_sql})
          and trim(n.ticker) = '{ticker_clean}'
        order by d.date desc, d.permno
        limit {int(row_limit)}
        """
        daily = db.raw_sql(daily_query, date_cols=["date"])
    finally:
        db.close()

    for column in ["openprc", "prc", "bidlo", "askhi"]:
        if column in daily.columns:
            daily[f"abs_{column}"] = daily[column].abs()
    return names, daily


def test_wrds_connection() -> dict[str, object]:
    """Run a minimal WRDS/CRSP query to verify credentials and database access."""
    db = open_wrds_connection()
    try:
        latest = db.raw_sql("select max(date) as latest_crsp_date from crsp.dsf", date_cols=["latest_crsp_date"])
        sample = db.raw_sql(
            """
            select permno, date, prc, vol
            from crsp.dsf
            order by date desc
            limit 5
            """,
            date_cols=["date"],
        )
    finally:
        db.close()

    return {
        "latest_crsp_date": latest["latest_crsp_date"].iloc[0],
        "sample_rows": sample,
    }


def get_latest_crsp_date() -> pd.Timestamp:
    """Return the latest daily observation date available in WRDS CRSP."""
    connection_info = test_wrds_connection()
    return pd.Timestamp(connection_info["latest_crsp_date"]).normalize()


def wrds_price_frame(daily_lookup: pd.DataFrame) -> pd.DataFrame:
    """Convert CRSP daily rows to a common price schema."""
    if daily_lookup.empty:
        return pd.DataFrame()
    data = daily_lookup.copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["close_price"] = data["prc"].abs()
    data["provider"] = "wrds"
    keep_cols = [col for col in ["date", "close_price", "vol", "provider", "ticker", "permno"] if col in data.columns]
    return data[keep_cols].sort_values("date")


def _standardize_yahoo_daily(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a Yahoo Finance OHLCV frame to the app/notebook daily schema."""
    if data is None or data.empty:
        raise ValueError(f"Yahoo Finance returned no rows for {ticker}.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    result = data.reset_index()
    date_column = "Date" if "Date" in result.columns else result.columns[0]
    result = result.rename(
        columns={
            date_column: "date",
            "Open": "yahoo_open",
            "Close": "yahoo_close",
            "Volume": "yahoo_volume",
        }
    )
    result["date"] = pd.to_datetime(result["date"], utc=True).dt.tz_localize(None).dt.normalize()
    keep_cols = [col for col in ["date", "yahoo_open", "yahoo_close", "yahoo_volume"] if col in result.columns]
    if "yahoo_close" not in keep_cols:
        raise ValueError(f"Yahoo Finance response for {ticker} did not include a Close column.")
    return result[keep_cols].sort_values("date")


def _yahoo_rate_limited(exc: Exception) -> bool:
    message = str(exc).lower()
    return "rate limit" in message or "too many requests" in message


def _yahoo_network_blocked(exc: Exception) -> bool:
    """Return whether Yahoo was blocked by the local/cloud network path."""
    message = str(exc).lower()
    blocked_markers = [
        "connect tunnel failed",
        "proxyerror",
        "response 403",
        "curl: (56)",
        "failed to perform",
    ]
    return any(marker in message for marker in blocked_markers)


def fetch_yahoo_daily(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch daily Yahoo Finance prices for a public cross-check."""
    if yf is None:
        raise RuntimeError("The `yfinance` package is not installed in this environment.")

    start_s = to_query_date(start_date)
    end_s = to_query_date(end_date)
    end_exclusive = (pd.Timestamp(end_s) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    symbol = ticker.upper().strip()
    last_exc: Exception | None = None

    for attempt in range(3):
        try:
            from yfinance._http import new_session

            session = new_session()
            data = yf.download(
                symbol,
                start=start_s,
                end=end_exclusive,
                auto_adjust=False,
                progress=False,
                session=session,
            )
            if data is None or data.empty:
                history = yf.Ticker(symbol, session=session).history(
                    start=start_s,
                    end=end_exclusive,
                    auto_adjust=False,
                )
                data = history
            return _standardize_yahoo_daily(data, ticker)
        except Exception as exc:
            last_exc = exc
            if _yahoo_rate_limited(exc) and attempt < 2:
                time.sleep(2**attempt)
                continue
            break

    if last_exc is not None and _yahoo_rate_limited(last_exc):
        raise RuntimeError("Yahoo Finance rate-limited this request; try again later.") from last_exc
    if last_exc is not None and _yahoo_network_blocked(last_exc):
        raise RuntimeError(
            "Yahoo Finance is blocked by the current network/proxy path "
            "(curl CONNECT tunnel returned 403). Use Refinitiv/WRDS as the primary price sources."
        ) from last_exc
    if last_exc is not None:
        raise last_exc

    try:
        from yfinance._http import HAS_CURL_CFFI
    except ImportError:
        HAS_CURL_CFFI = False
    if not HAS_CURL_CFFI:
        raise RuntimeError(
            "Yahoo Finance requests need curl_cffi for browser TLS impersonation. "
            "Install curl_cffi>=0.15 and restart."
        )
    raise ValueError(f"Yahoo Finance returned no rows for {ticker}.")


def yahoo_price_frame(yahoo_daily: pd.DataFrame) -> pd.DataFrame:
    """Convert Yahoo rows to a common price schema."""
    if yahoo_daily.empty:
        return pd.DataFrame()
    data = yahoo_daily.copy()
    data["close_price"] = data["yahoo_close"]
    data["provider"] = "yahoo"
    if "yahoo_volume" in data.columns:
        data["volume"] = data["yahoo_volume"]
    keep_cols = [col for col in ["date", "close_price", "volume", "provider"] if col in data.columns]
    return data[keep_cols].sort_values("date")


def _pg_sql(db_conn: Any, sql: str) -> pd.DataFrame:
    """Execute SQL via raw psycopg2, bypassing SQLAlchemy 2.x incompatibility."""
    cur = db_conn.connection.connection.cursor()
    cur.execute(sql)
    df = pd.DataFrame(cur.fetchall(), columns=[d[0] for d in cur.description])
    cur.close()
    return df


def query_ravenpack_articles(
    ticker: str,
    start_date: str,
    end_date: str,
    permno: int | None = None,
    year_progress_callback: Callable[[int, int, float, "str | None"], None] | None = None,
    year_timeout_s: int = 90,
) -> pd.DataFrame:
    """Fetch RavenPack sentiment articles for a ticker from WRDS.

    Uses PERMNO→CUSIP→rp_entity_id resolution when permno is supplied (preferred),
    falling back to ticker-only matching otherwise.

    Fetches one year at a time so partial results are guaranteed even when a single
    year's table is very large.  Each year query has a server-side statement_timeout
    of `year_timeout_s` seconds; timed-out years are skipped and the connection is
    rolled back so subsequent years can still run.

    year_progress_callback(yr, n_rows, elapsed_s, error_str | None) is called after
    each year so callers can log real-time progress without polling.
    """
    import time as _time
    ticker_clean = ticker.upper().strip()
    start_s = to_query_date(start_date)
    end_s = to_query_date(end_date)

    db = open_wrds_connection()
    try:
        rp_entity_id: str | None = None

        # ── Primary: PERMNO → CUSIP → RavenPack entity (unambiguous) ──────────
        if permno is not None:
            cusip_row = _pg_sql(db, f"""
                SELECT ncusip
                FROM crsp.stocknames
                WHERE permno = {int(permno)}
                  AND ncusip IS NOT NULL
                  AND ncusip <> ''
                ORDER BY namedt DESC
                LIMIT 1
            """)
            if not cusip_row.empty:
                ncusip = str(cusip_row["ncusip"].iloc[0]).strip()
                # RavenPack CUSIPs are 8-char (no check digit); CRSP ncusip is 8-char too
                emap = _pg_sql(db, f"""
                    SELECT DISTINCT rp_entity_id, entity_name
                    FROM ravenpack_common.wrds_rpa_company_mappings
                    WHERE cusip LIKE '{ncusip[:8]}%'
                    LIMIT 1
                """)
                if not emap.empty:
                    rp_entity_id = str(emap["rp_entity_id"].iloc[0])

        # ── Fallback: ticker match — take the row with the most RavenPack data ─
        if rp_entity_id is None:
            mapping = _pg_sql(db, f"""
                SELECT DISTINCT rp_entity_id, entity_name
                FROM ravenpack_common.wrds_rpa_company_mappings
                WHERE ticker = '{ticker_clean}'
                  AND entity_type = 'COMP'
            """)
            if mapping.empty:
                raise ValueError(f"No RavenPack entity found for {ticker_clean}.")
            if len(mapping) == 1:
                rp_entity_id = str(mapping["rp_entity_id"].iloc[0])
            else:
                # Multiple matches — pick by checking which entity actually has data
                # in one representative year near the middle of the window
                best_id = None
                best_count = -1
                for eid in mapping["rp_entity_id"].tolist():
                    try:
                        cnt = _pg_sql(db, f"""
                            SELECT COUNT(*) AS n
                            FROM ravenpack_dj.rpa_djpr_equities_2007
                            WHERE rp_entity_id = '{eid}'
                        """)
                        n = int(cnt["n"].iloc[0]) if not cnt.empty else 0
                        if n > best_count:
                            best_count = n
                            best_id = eid
                    except Exception:
                        pass
                if best_id is None:
                    best_id = str(mapping["rp_entity_id"].iloc[0])
                rp_entity_id = best_id

        # ── Year-by-year fetch with per-query server-side timeout ───────────────
        # headline / event_text are omitted — large text columns that multiply
        # transfer size while the paper models only need the numeric scores.
        cols = ("timestamp_utc, rp_story_id, relevance, event_sentiment_score, "
                "source_name, topic, \"group\", \"type\", "
                "sub_type, news_type, css, nip")

        # Apply statement_timeout once; re-apply after each rollback.
        def _set_timeout():
            try:
                _pg_sql(db, f"SET statement_timeout = {int(year_timeout_s * 1000)}")
            except Exception:
                pass

        _set_timeout()
        frames: list[pd.DataFrame] = []

        for yr in range(int(start_s[:4]), int(end_s[:4]) + 1):
            yr_start = max(start_s, f"{yr}-01-01")
            yr_end   = min(end_s,   f"{yr}-12-31")
            t0 = _time.monotonic()
            try:
                yr_df = _pg_sql(db, f"""
                    SELECT {cols}
                    FROM ravenpack_dj.rpa_djpr_equities_{yr}
                    WHERE rp_entity_id = '{rp_entity_id}'
                      AND rpa_date_utc BETWEEN '{yr_start}' AND '{yr_end}'
                    ORDER BY timestamp_utc
                """)
                elapsed = round(_time.monotonic() - t0, 1)
                if year_progress_callback:
                    year_progress_callback(yr, len(yr_df), elapsed, None)
                if not yr_df.empty:
                    frames.append(yr_df)
            except Exception as exc:
                elapsed = round(_time.monotonic() - t0, 1)
                if year_progress_callback:
                    year_progress_callback(yr, 0, elapsed, str(exc)[:120])
                # Roll back to clear the error state so the next year can proceed.
                try:
                    db.connection.rollback()
                except Exception:
                    pass
                _set_timeout()
                continue

    finally:
        db.close()

    if not frames:
        return pd.DataFrame()

    articles = pd.concat(frames, ignore_index=True)
    articles["article_time"] = pd.to_datetime(articles["timestamp_utc"], utc=True)
    articles["relevance_score"] = pd.to_numeric(articles["relevance"], errors="coerce") / 100
    articles["event_sentiment_score"] = pd.to_numeric(articles["event_sentiment_score"], errors="coerce")
    articles["sentiment_score"] = articles["relevance_score"] * articles["event_sentiment_score"]
    articles["ticker"] = ticker_clean
    return (
        articles
        .drop_duplicates(subset=["rp_story_id"])
        .sort_values("article_time")
        .reset_index(drop=True)
    )


def run_ticker_data_query(
    project_root: Path,
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    query_refinitiv: bool = True,
    query_wrds: bool = True,
    query_yahoo: bool = True,
    query_ravenpack: bool = False,
    news_count: int = 50,
    wrds_limit: int = 500,
    latest_crsp_date: pd.Timestamp | None = None,
) -> dict[str, object]:
    """Query selected market-data/news providers for the same ticker and date range."""
    ticker_clean = ticker.upper().strip()
    start_s = to_query_date(start_date)
    end_s = to_query_date(end_date)
    providers: dict[str, dict[str, object]] = {
        "refinitiv": {"status": "skipped", "error": None, "prices": pd.DataFrame(), "news": pd.DataFrame(), "ric": None},
        "wrds": {"status": "skipped", "error": None, "prices": pd.DataFrame(), "names": pd.DataFrame()},
        "yahoo": {"status": "skipped", "error": None, "prices": pd.DataFrame()},
        "ravenpack": {"status": "skipped", "error": None, "articles": pd.DataFrame()},
    }

    if query_refinitiv:
        from sentiment_ltr.data.news_coverage import build_news_coverage_result  # lazy – avoids LSEG auto-init
        from sentiment_ltr.data.refinitiv_queries import (
            query_refinitiv_prices,
            refinitiv_configured,
            refinitiv_setup_message,
        )
        from sentiment_ltr.data.refinitiv_session import get_last_refinitiv_session_info, open_refinitiv_session
        if refinitiv_configured(project_root):
            try:
                import lseg.data as ld  # type: ignore

                open_refinitiv_session(project_root, ld)
                session_info = get_last_refinitiv_session_info()
                try:
                    refinitiv_prices, ric = query_refinitiv_prices(
                        project_root,
                        ticker_clean,
                        start_s,
                        end_s,
                        ld_module=ld,
                    )
                    refinitiv_news = pd.DataFrame()
                    news_daily_counts = pd.DataFrame()
                    news_summary: dict[str, object] | None = None
                    news_error = None
                    if news_count > 0:
                        try:
                            coverage_news, news_daily_counts, summary_obj, news_ric = build_news_coverage_result(
                                project_root,
                                ticker_clean,
                                start_s,
                                end_s,
                                ld_module=ld,
                            )
                            refinitiv_news = coverage_news
                            news_summary = summary_obj.__dict__
                            if ric is None:
                                ric = news_ric
                        except Exception as exc:
                            news_error = str(exc)
                    providers["refinitiv"] = {
                        "status": "ok" if not refinitiv_prices.empty else "empty",
                        "error": news_error if not refinitiv_prices.empty else "No Refinitiv price history returned.",
                        "prices": refinitiv_prices,
                        "news": refinitiv_news,
                        "news_daily_counts": news_daily_counts,
                        "news_summary": news_summary,
                        "ric": ric,
                        "session_info": session_info,
                    }
                finally:
                    try:
                        ld.close_session()
                    except Exception:
                        pass
            except Exception as exc:
                providers["refinitiv"] = {
                    "status": "failed",
                    "error": str(exc),
                    "prices": pd.DataFrame(),
                    "news": pd.DataFrame(),
                    "news_daily_counts": pd.DataFrame(),
                    "news_summary": None,
                    "ric": None,
                }
        else:
            providers["refinitiv"] = {
                "status": "unavailable",
                "error": refinitiv_setup_message(project_root),
                "prices": pd.DataFrame(),
                "news": pd.DataFrame(),
                "news_daily_counts": pd.DataFrame(),
                "news_summary": None,
                "ric": None,
            }

    if query_wrds:
        if wrds_credentials_available():
            wrds_start = pd.Timestamp(start_s)
            wrds_end = pd.Timestamp(end_s)
            if latest_crsp_date is not None:
                crsp_end = min(pd.Timestamp.today().normalize(), pd.Timestamp(latest_crsp_date).normalize())
                wrds_end = min(wrds_end, crsp_end)
            if wrds_start <= wrds_end:
                try:
                    name_history, daily_lookup = query_wrds_ticker_data(
                        ticker_clean,
                        to_query_date(wrds_start),
                        to_query_date(wrds_end),
                        int(wrds_limit),
                    )
                    wrds_prices = wrds_price_frame(daily_lookup)
                    providers["wrds"] = {
                        "status": "ok" if not wrds_prices.empty else "empty",
                        "error": None if not wrds_prices.empty else "No CRSP rows in the selected date range.",
                        "prices": wrds_prices,
                        "names": name_history,
                        "query_start": wrds_start,
                        "query_end": wrds_end,
                    }
                except Exception as exc:
                    providers["wrds"] = {
                        "status": "failed",
                        "error": str(exc),
                        "prices": pd.DataFrame(),
                        "names": pd.DataFrame(),
                    }
            else:
                providers["wrds"] = {
                    "status": "empty",
                    "error": "Selected range is entirely after the latest CRSP date available in WRDS.",
                    "prices": pd.DataFrame(),
                    "names": pd.DataFrame(),
                }
        else:
            providers["wrds"] = {
                "status": "unavailable",
                "error": "WRDS credentials are not configured.",
                "prices": pd.DataFrame(),
                "names": pd.DataFrame(),
            }

    if query_yahoo:
        try:
            yahoo_daily = fetch_yahoo_daily(ticker_clean, start_s, end_s)
            yahoo_prices = yahoo_price_frame(yahoo_daily)
            providers["yahoo"] = {
                "status": "ok" if not yahoo_prices.empty else "empty",
                "error": None if not yahoo_prices.empty else "Yahoo Finance returned no rows.",
                "prices": yahoo_prices,
            }
        except Exception as exc:
            providers["yahoo"] = {
                "status": "failed",
                "error": str(exc),
                "prices": pd.DataFrame(),
            }

    if query_ravenpack:
        if wrds_credentials_available():
            try:
                ravenpack_articles = query_ravenpack_articles(ticker_clean, start_s, end_s)
                providers["ravenpack"] = {
                    "status": "ok" if not ravenpack_articles.empty else "empty",
                    "error": None if not ravenpack_articles.empty else "No RavenPack articles in the selected date range.",
                    "articles": ravenpack_articles,
                }
            except Exception as exc:
                providers["ravenpack"] = {
                    "status": "failed",
                    "error": str(exc),
                    "articles": pd.DataFrame(),
                }
        else:
            providers["ravenpack"] = {
                "status": "unavailable",
                "error": "WRDS credentials are not configured.",
                "articles": pd.DataFrame(),
            }

    price_frames = {
        provider: result["prices"]
        for provider, result in providers.items()
        if isinstance(result.get("prices"), pd.DataFrame) and not result["prices"].empty
    }

    return {
        "ticker": ticker_clean,
        "start_date": start_s,
        "end_date": end_s,
        "providers": providers,
        "price_frames": price_frames,
        "selected_providers": {
            "refinitiv": query_refinitiv,
            "wrds": query_wrds,
            "yahoo": query_yahoo,
            "ravenpack": query_ravenpack,
        },
    }
