"""Command-line entry point for the Autonomous Financial Research Analyst.

Examples:
    python main.py "Analyze NVIDIA stock and tell me if it's a good investment"
    python main.py --rag "What are Microsoft's AI initiatives, and how is the stock doing?"
    python main.py --verify --signoff "Should I buy AMD?"
    python main.py --score          # backtest logged recommendations vs. real returns
"""

from __future__ import annotations

import argparse
import logging
import sys


def _utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass


def main() -> int:
    _utf8_stdout()

    parser = argparse.ArgumentParser(description="Autonomous Financial Research Analyst")
    parser.add_argument("query", nargs="?", help="The research question to answer")
    parser.add_argument("--rag", action="store_true",
                        help="Enable retrieval over private PDFs in data/reports/")
    parser.add_argument("--reports-dir", default="data/reports",
                        help="Directory of analyst PDFs for --rag")
    parser.add_argument("--verify", action="store_true",
                        help="Run the citation-verification guardrail on the briefing")
    parser.add_argument("--signoff", action="store_true",
                        help="Require human approval before finalizing a Buy/Sell call")
    parser.add_argument("--ticker", help="Ticker to log for backtesting (e.g. NVDA)")
    parser.add_argument("--score", action="store_true",
                        help="Backtest logged recommendations against real returns and exit")
    parser.add_argument("--verbose", action="store_true", help="Show agent reasoning logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )

    # --- Backtest mode -----------------------------------------------------
    if args.score:
        from financial_analyst_agent.backtest import score_recommendations

        r = score_recommendations()
        if r["status"] != "ok":
            print(r.get("message", "Nothing to score yet."))
            return 0
        print(f"Backtest ({r['horizon_days']}-day horizon): "
              f"{r['evaluated']} evaluated, {r['pending']} pending")
        print(f"Directional accuracy: {r['accuracy_pct']}%")
        for d in r["details"]:
            mark = "✅" if d["correct"] else "❌"
            print(f"  {mark} {d['ticker']:5} {d['action']:10} "
                  f"-> {d['forward_return_pct']:+.2f}%")
        return 0

    if not args.query:
        parser.error("a query is required unless --score is used")

    # --- Analysis mode -----------------------------------------------------
    from financial_analyst_agent.agent import analyze
    from financial_analyst_agent.config import CHAT_MODEL

    if args.rag:
        from financial_analyst_agent.rag import build_retriever

        print(f"Indexing analyst reports in '{args.reports_dir}' ...")
        build_retriever(args.reports_dir)

    print(f"Model: {CHAT_MODEL}\nQuery: {args.query}\n{'-' * 70}")
    result = analyze(args.query, with_rag=args.rag)
    answer = result["answer"]
    print(answer)

    # --- Guardrail: citation verification ----------------------------------
    if args.verify:
        from financial_analyst_agent.verify import format_badge, verify_citations

        print("\n" + "-" * 70)
        print(format_badge(verify_citations(answer, result["tool_outputs"])))

    # --- Recommendation logging + human sign-off ---------------------------
    from financial_analyst_agent.backtest import extract_recommendation, log_recommendation

    rec = extract_recommendation(answer, ticker_hint=args.ticker)
    if rec["action"] != "UNKNOWN" and rec["ticker"]:
        log_recommendation(rec["ticker"], rec["action"], args.query,
                           rec["confidence_pct"])

    if args.signoff:
        from financial_analyst_agent.approval import request_signoff_cli, requires_signoff

        if requires_signoff(answer):
            request_signoff_cli(answer, rec.get("ticker") or "?", rec["action"])
        else:
            print("\n(No actionable Buy/Sell call — sign-off not required.)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
