"""Count Refinitiv news articles per day for one stock over a date range."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from sentiment_ltr.data.news_coverage import (  # noqa: E402
    daily_article_counts,
    fetch_ticker_headlines,
    summarize_news_coverage,
    weekly_article_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol, e.g. AAPL")
    parser.add_argument("--start", default="2003-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default="2014-12-31", help="End date YYYY-MM-DD")
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "validation"),
        help="Directory for CSV/JSON outputs",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    args = parse_args()

    ticker = args.ticker.upper().strip()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching Refinitiv headlines for {ticker} from {args.start} to {args.end}...")
    print("Keep LSEG Workspace open when using desktop session mode.")

    headlines, ric = fetch_ticker_headlines(
        PROJECT_ROOT,
        ticker,
        args.start,
        args.end,
    )
    daily = daily_article_counts(headlines, args.start, args.end)
    weekly = weekly_article_counts(daily)
    summary = summarize_news_coverage(ticker, ric, headlines, args.start, args.end)

    slug = ticker.lower()
    headlines_path = output_dir / f"{slug}_news_headlines_{args.start}_{args.end}.csv"
    daily_path = output_dir / f"{slug}_news_daily_counts_{args.start}_{args.end}.csv"
    weekly_path = output_dir / f"{slug}_news_weekly_counts_{args.start}_{args.end}.csv"
    summary_path = output_dir / f"{slug}_news_coverage_summary_{args.start}_{args.end}.json"

    export_headlines = headlines.copy()
    export_headlines["article_time"] = export_headlines["article_time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    export_headlines.to_csv(headlines_path, index=False)
    daily.to_csv(daily_path, index=False)
    weekly.to_csv(weekly_path, index=False)
    summary_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                **summary.__dict__,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nRIC used: {ric}")
    print(f"Headlines pulled: {len(headlines):,}")
    print(f"Unique story IDs: {summary.total_articles:,}")
    print(f"Calendar days with news: {summary.calendar_days_with_news:,} / {summary.calendar_days_in_range:,}")
    print(f"Average articles per week: {summary.avg_articles_per_week:.2f}")
    print(f"Weeks with zero articles: {summary.weeks_with_zero_articles:,} / {summary.weeks_in_range:,}")
    print(
        "Paper weekly threshold (>= 1 article/week on average): "
        + ("PASS" if summary.passes_paper_weekly_threshold else "FAIL")
    )
    print("\nTop 10 days by article count:")
    print(
        daily.sort_values("article_count", ascending=False)
        .head(10)
        .assign(date=lambda frame: frame["date"].dt.strftime("%Y-%m-%d"))
        .to_string(index=False)
    )
    print("\nWrote:")
    print(f"- {headlines_path}")
    print(f"- {daily_path}")
    print(f"- {weekly_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
