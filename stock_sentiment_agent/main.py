"""CLI Entry Point & Pipeline Runner for HengSeng Sentiment Price Prediction Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_PARENT = str(Path(__file__).resolve().parent.parent)
if ROOT_PARENT not in sys.path:
    sys.path.insert(0, ROOT_PARENT)

from stock_sentiment_agent.agents import PipelineOrchestrator
from stock_sentiment_agent.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="HengSeng Sentiment Price Prediction Agent")
    parser.add_argument("--ticker", type=str, default=None, help="Run pipeline for a single ticker (e.g. 0700.HK)")
    parser.add_argument("--use-hf", action="store_true", help="Use Hugging Face XLM-RoBERTa transformer model")
    parser.add_argument("--json", action="store_true", help="Print JSON output to stdout")
    args = parser.parse_args()

    settings = get_settings()
    orchestrator = PipelineOrchestrator(use_hf=args.use_hf)

    if args.ticker:
        match = [s for s in settings.stocks if s.ticker == args.ticker]
        if not match:
            print(f"Ticker {args.ticker} not found in configuration.", file=sys.stderr)
            return 1
        out = orchestrator.run_single(match[0])
    else:
        out = orchestrator.run_all()

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        summaries = out.get("summaries", [out] if "intelligence" in out else [])
        print(f"HengSeng Pipeline Complete — {len(summaries)} stock(s) analyzed")
        for s in summaries:
            intel = s["intelligence"]
            pred = s["prediction"]
            print(
                f"  {intel['ticker']:10s} | {intel['exchange']:4s} | Close={intel['latest_price']:8.2f} | "
                f"Pred({pred['forecast_target_date']})={pred['predicted_close']:8.2f} ({pred['predicted_pct_change']:+6.2f}%) | "
                f"Sent={intel['combined_sentiment_score']:+6.3f} | {intel['market_direction']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
