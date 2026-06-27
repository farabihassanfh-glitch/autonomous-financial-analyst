# Autonomous Financial Research Analyst

An agentic AI system that autonomously researches public companies and produces
evidence-backed investment briefings. Built with **Claude** ([Anthropic](https://www.anthropic.com/))
and **LangGraph**, it combines live market data, web-news search, and
**retrieval-augmented generation (RAG)** over a private document corpus — and
decides on its own which tools to call to satisfy a research goal.

> A portfolio project demonstrating modern agent design: goal-oriented behavior,
> tool use, graceful error handling, and grounding via RAG.

## Why it's interesting

Most "chatbots" answer the question you asked. An **agent** pursues a goal: given
*"Analyze NVIDIA,"* this system independently pulls the current price, multi-year
performance, recent news, and (optionally) proprietary research, then synthesizes
a recommendation — looping through tools until it has what it needs.

## Architecture

```
                 ┌──────────────────────┐
   user query →  │  agent node (Claude) │ ── decides which tools to call
                 └──────────┬───────────┘
                            │ tool calls          ┌───────────────────────┐
                            ▼                      │  get_stock_price       │
                 ┌──────────────────────┐         │  get_stock_history     │
                 │     tool node        │ ───────▶│  search_financial_news │
                 └──────────┬───────────┘         │  query_private_database│ (RAG)
                            │ results             └───────────────────────┘
                            ▼
              loops back to agent until no tool calls remain
                            │
                            ▼
                   final investment briefing
```

The loop is a [LangGraph](https://langchain-ai.github.io/langgraph/) `StateGraph`:
a Claude reasoning node and a tool node connected by a conditional edge that keeps
cycling until the model stops requesting tools. Conversation state accumulates via
the `add_messages` reducer, and a checkpointer gives the agent memory per session.

## Features

- **Goal-oriented agent** driven by a system "charter" (Claude via `langchain-anthropic`)
- **Tools:** real-time price & multi-year history (`yfinance`), web news (Tavily),
  and a RAG tool over private PDFs
- **RAG pipeline:** PDF load → chunk → local embeddings → Chroma vector store →
  retrieval (no embeddings API key required — runs on CPU)
- **Graceful degradation:** every tool returns a structured error instead of
  crashing, so one failed source doesn't stop the analysis
- **Per-session memory** via LangGraph checkpointing

## Tech stack

Claude (Anthropic) · LangGraph · LangChain · yfinance · Tavily · ChromaDB ·
sentence-transformers

## Setup

```bash
git clone <your-repo-url>
cd autonomous-financial-analyst
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                    # then add your ANTHROPIC_API_KEY
```

## Usage

```bash
# Basic research (price + history + news)
python main.py "Analyze NVIDIA stock and tell me if it's a good investment"

# Show the agent's tool-calling decisions
python main.py --verbose "Compare Microsoft and Google as AI investments"

# Enable RAG over your own analyst PDFs (drop them in data/reports/ first)
python main.py --rag "What are Microsoft's AI initiatives, and how is the stock doing?"
```

Or use it as a library:

```python
from financial_analyst_agent.agent import run_query

print(run_query("What are the risks of investing in Tesla right now?"))
```

## Project layout

```
financial_analyst_agent/
├── config.py     # env + model/embedding factories
├── tools.py      # market-data tools (price, history, news)
├── rag.py        # PDF → vector store + private-database tool
└── agent.py      # LangGraph state machine + charter
main.py           # CLI entry point
```

## Notes

- Defaults to `claude-opus-4-8`; set `ANTHROPIC_MODEL=claude-sonnet-4-6` in `.env`
  for a cheaper, faster run.
- News search is optional — set `TAVILY_API_KEY` to enable it; the agent works
  without it and notes the gap.
- This is an educational/portfolio project, not financial advice.

## License

MIT — see [LICENSE](LICENSE).
