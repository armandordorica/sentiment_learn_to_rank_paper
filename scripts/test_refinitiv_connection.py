"""Smoke-test a local LSEG/Refinitiv Workspace desktop session."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import lseg.data as ld

from sentiment_ltr.data.refinitiv_session import open_workspace_session, resolve_config_path


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    config_path = resolve_config_path(PROJECT_ROOT)
    print("Opening LSEG desktop session...")
    print(f"Using config: {config_path}")

    open_workspace_session(PROJECT_ROOT, ld)

    try:
        data = ld.get_data(
            universe=["AAPL.O", "MSFT.O"],
            fields=["BID", "ASK", "TR.Revenue"],
        )
        print(data)
    finally:
        ld.close_session()


if __name__ == "__main__":
    main()
