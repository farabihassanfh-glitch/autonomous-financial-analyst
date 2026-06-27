"""Command-line entry point for the Autonomous Financial Research Analyst.

Examples:
    python main.py "Analyze NVIDIA stock and tell me if it's a good investment"
    python main.py --rag "What are Microsoft's AI initiatives, and how is the stock doing?"
"""

from __future__ import annotations

import argparse
import logging
import sys

from financial_analyst_agent.agent import build_agent
from financial_analyst_agent.config import CHAT_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous Financial Research Analyst")
    parser.add_argument("query", help="The research question to answer")
    parser.add_argument("--rag", action="store_true",
                        help="Enable retrieval over private PDFs in data/reports/")
    parser.add_argument("--reports-dir", default="data/reports",
                        help="Directory of analyst PDFs for --rag (default: data/reports)")
    parser.add_argument("--verbose", action="store_true", help="Show agent reasoning logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
    )

    if args.rag:
        from financial_analyst_agent.rag import build_retriever

        print(f"Indexing analyst reports in '{args.reports_dir}' ...")
        build_retriever(args.reports_dir)

    print(f"Model: {CHAT_MODEL}\nQuery: {args.query}\n{'-' * 70}")

    agent = build_agent(with_rag=args.rag, with_memory=True)
    config = {"configurable": {"thread_id": "cli"}, "recursion_limit": 25}
    from langchain_core.messages import HumanMessage

    result = agent.invoke({"messages": [HumanMessage(content=args.query)]}, config=config)
    print(result["messages"][-1].content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
