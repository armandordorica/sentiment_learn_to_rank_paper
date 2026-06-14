"""Build the initial market-data candidate universe."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sentiment_ltr.data import UniverseConfig, load_sp500_candidates


def main() -> None:
    config_path = PROJECT_ROOT / "config" / "market_data.yml"
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    universe_config = config["universe"]
    candidates = load_sp500_candidates(
        UniverseConfig(
            wikipedia_sp500_url=universe_config["wikipedia_sp500_url"],
            replace_dot_with_dash=universe_config["yahoo_symbol_normalization"]["replace_dot_with_dash"],
        )
    )

    output_dir = PROJECT_ROOT / "data" / "raw" / "market"
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = output_dir / "candidate_universe_sp500.csv"
    manifest_path = output_dir / "candidate_universe_manifest.json"

    candidates.to_csv(candidates_path, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_source": universe_config["candidate_source"],
        "source_url": universe_config["wikipedia_sp500_url"],
        "rows": int(len(candidates)),
        "columns": list(candidates.columns),
        "output_file": str(candidates_path.relative_to(PROJECT_ROOT)),
        "known_limitation": "Current S&P 500 constituents are survivorship-biased and do not match the paper's Bloomberg top-1000-by-volume universe.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(candidates)} candidates to {candidates_path}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
