"""Refinitiv/LSEG Workspace query helpers for the Streamlit validation app."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from sentiment_ltr.data.refinitiv_session import load_app_key, open_workspace_session


def is_huggingface_space() -> bool:
    """Return whether the app is running on a Hugging Face Space."""
    return bool(os.environ.get("SPACE_ID")) or os.environ.get("SYSTEM") == "spaces"


def refinitiv_package_available() -> bool:
    """Return whether the lseg-data package is importable."""
    try:
        import lseg.data  # type: ignore  # noqa: F401
    except ImportError:
        return False
    return True


def refinitiv_setup_message(project_root: Path) -> str:
    """Return a user-facing setup message when Refinitiv is unavailable."""
    if is_huggingface_space():
        return (
            "Refinitiv is not available on Hugging Face Spaces. It requires LSEG Workspace "
            "running on your local machine with an App Key in `.env`. "
            "Use WRDS and Yahoo here, or run `streamlit run app.py` locally for Refinitiv."
        )

    if not refinitiv_package_available():
        return (
            "Install the Refinitiv SDK with "
            "`pip install -r requirements-refinitiv.txt`, then restart the Streamlit app."
        )

    try:
        load_app_key(project_root)
    except FileNotFoundError as exc:
        return str(exc)
    except ValueError as exc:
        return str(exc)
    except OSError as exc:
        return f"Could not read the LSEG config file: {exc}"

    return (
        "Refinitiv credentials look configured. Keep LSEG Workspace running and signed in, "
        "then click **Run parallel query** again."
    )


def refinitiv_configured(project_root: Path) -> bool:
    """Return whether Refinitiv can be queried from this environment."""
    if not refinitiv_package_available():
        return False
    try:
        load_app_key(project_root)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


def ticker_to_ric_candidates(ticker: str) -> list[str]:
    """Build likely Refinitiv RIC candidates from a user-entered ticker."""
    clean = ticker.upper().strip()
    if not clean:
        return []
    if "." in clean:
        return [clean]
    return [f"{clean}.O", f"{clean}.N", clean]


def _normalize_history_frame(history: pd.DataFrame, ric: str) -> pd.DataFrame:
    """Standardize Refinitiv daily history output."""
    if history is None or history.empty:
        return pd.DataFrame()

    data = history.copy()
    if isinstance(data.index, pd.MultiIndex):
        data = data.reset_index()
    elif data.index.name in {"Date", "date", "DATE"} or isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index()

    rename_map = {}
    for column in data.columns:
        column_text = str(column).lower()
        if column_text in {"date", "versioncreated"}:
            rename_map[column] = "date"
        elif "price close" in column_text or column_text.endswith("close"):
            rename_map[column] = "close_price"
        elif "volume" in column_text:
            rename_map[column] = "volume"

    data = data.rename(columns=rename_map)
    if "date" not in data.columns:
        for column in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data = data.rename(columns={column: "date"})
                break

    keep_cols = [col for col in ["date", "close_price", "volume"] if col in data.columns]
    if "date" not in keep_cols or "close_price" not in keep_cols:
        return pd.DataFrame()

    result = data[keep_cols].copy()
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    result["ric"] = ric
    result["provider"] = "refinitiv"
    return result.sort_values("date")


def _normalize_headlines_frame(headlines: Any) -> pd.DataFrame:
    """Standardize Refinitiv news headline output."""
    if headlines is None:
        return pd.DataFrame()

    data = headlines.data.df if hasattr(headlines, "data") else headlines
    if data is None or len(data) == 0:
        return pd.DataFrame()

    result = data.copy()
    if isinstance(result.index, pd.DatetimeIndex):
        result = result.reset_index()
    if "versionCreated" in result.columns:
        result = result.rename(columns={"versionCreated": "date"})
    elif "date" not in result.columns:
        for column in result.columns:
            if pd.api.types.is_datetime64_any_dtype(result[column]):
                result = result.rename(columns={column: "date"})
                break

    result["provider"] = "refinitiv"
    if "storyId" not in result.columns:
        for column in result.columns:
            if str(column).lower() == "storyid":
                result = result.rename(columns={column: "storyId"})
                break

    keep_cols = [col for col in ["date", "headline", "storyId", "sourceCode", "provider"] if col in result.columns]
    if keep_cols:
        result = result[keep_cols]
    return result.sort_values("date", ascending=False) if "date" in result.columns else result


def fetch_refinitiv_story(
    project_root: Path,
    story_id: str,
    *,
    as_text: bool = True,
    ld_module: Any | None = None,
) -> str:
    """Fetch the full Refinitiv news story body for a headline storyId."""
    if not story_id or not str(story_id).strip():
        raise ValueError("A Refinitiv storyId is required.")

    ld = ld_module
    opened_here = False
    if ld is None:
        import lseg.data as ld  # type: ignore

        open_workspace_session(project_root, ld)
        opened_here = True

    try:
        story_format = ld.news.Format.TEXT if as_text else ld.news.Format.HTML
        story = ld.news.get_story(str(story_id).strip(), format=story_format)
    finally:
        if opened_here:
            ld.close_session()

    if story is None:
        raise ValueError(f"No story content returned for {story_id}.")
    return str(story)


def query_refinitiv_prices(
    project_root: Path,
    ticker: str,
    start_date: str,
    end_date: str,
    ld_module: Any | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Fetch daily Refinitiv price history for a ticker and date range."""
    ld = ld_module
    opened_here = False
    if ld is None:
        import lseg.data as ld  # type: ignore

        open_workspace_session(project_root, ld)
        opened_here = True

    errors: list[str] = []
    try:
        for ric in ticker_to_ric_candidates(ticker):
            try:
                history = ld.get_history(
                    universe=[ric],
                    fields=["TR.PriceClose", "TR.Volume"],
                    interval="1D",
                    start=start_date,
                    end=end_date,
                )
            except Exception as exc:
                errors.append(f"{ric}: {exc}")
                continue

            normalized = _normalize_history_frame(history, ric)
            if not normalized.empty:
                return normalized, ric
    finally:
        if opened_here:
            ld.close_session()

    detail = errors[-1] if errors else "No Refinitiv price history returned."
    raise ValueError(detail)


def query_refinitiv_news(
    project_root: Path,
    ticker: str,
    start_date: str,
    end_date: str,
    count: int = 50,
    ld_module: Any | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Fetch Refinitiv news headlines for a ticker and date range."""
    ld = ld_module
    opened_here = False
    if ld is None:
        import lseg.data as ld  # type: ignore

        open_workspace_session(project_root, ld)
        opened_here = True

    start_ts = pd.Timestamp(start_date).to_pydatetime()
    end_ts = pd.Timestamp(end_date).to_pydatetime()
    errors: list[str] = []

    try:
        for ric in ticker_to_ric_candidates(ticker):
            query = f"R:{ric} AND Language:LEN"
            try:
                headlines = ld.news.get_headlines(query, start=start_ts, end=end_ts, count=int(count))
            except Exception as exc:
                errors.append(f"{ric}: {exc}")
                continue

            normalized = _normalize_headlines_frame(headlines)
            if not normalized.empty:
                return normalized, ric
    finally:
        if opened_here:
            ld.close_session()

    detail = errors[-1] if errors else "No Refinitiv news headlines returned."
    raise ValueError(detail)
