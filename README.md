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

## Production guardrails

Beyond the core agent, the project includes the kind of controls a regulated
setting would demand — the part most agent demos skip:

- **Citation verification** (`--verify`) — after the agent answers, a separate
  "compliance reviewer" Claude pass audits every claim, confirming it is both
  supported by a tool output and carries a source citation. Output gets a badge:
  `✅ Citations verified — 12/12 claims sourced` or a list of flagged claims.
- **Human-in-the-loop sign-off** (`--signoff`) — any actionable Buy/Sell call
  pauses for a human analyst to approve or reject; the decision (approver +
  timestamp) is written to an audit log.
- **Backtesting** (`--score`) — every recommendation is logged, then scored
  against the actual forward stock return from yfinance, producing a directional
  accuracy metric over time.

## Tech stack

Claude (Anthropic) · LangGraph · LangChain · yfinance · Tavily · ChromaDB ·
sentence-transformers

## Setup

```bash
git clone <your-repo-url>
cd autonomous-financial-analyst
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # core agent + web UI
pip install -r requirements-rag.txt                     # optional: enables Private RAG (heavy)
cp .env.example .env                                     # then add your ANTHROPIC_API_KEY
```

## Usage

```bash
# Basic research (price + history + news)
python main.py "Analyze NVIDIA stock and tell me if it's a good investment"

# Show the agent's tool-calling decisions
python main.py --verbose "Compare Microsoft and Google as AI investments"

# Enable RAG over your own analyst PDFs (drop them in data/reports/ first)
python main.py --rag "What are Microsoft's AI initiatives, and how is the stock doing?"

# Production guardrails: verify citations + require human sign-off on Buy/Sell
python main.py --verify --signoff --ticker AMD "Should I buy AMD?"

# Backtest logged recommendations against actual returns
python main.py --score
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

## Deploy

The core `requirements.txt` is intentionally slim (no torch/RAG) so the deployed
app builds small and cheap. RAG stays local-only; its toggle auto-disables when
those deps aren't present.

### Streamlit Community Cloud (free, recommended)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub →
   **New app** → pick this repo, branch `main`, main file `app_streamlit.py`.
3. In **Advanced settings → Secrets**, add (TOML format):
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   TAVILY_API_KEY = "tvly-..."   # optional, enables news
   APP_PASSWORD = "choose-a-password"
   ```
4. Deploy → you get a public `https://<name>.streamlit.app` URL.

### Railway (alternative; Hobby plan ~$5/mo)

Ships a `Dockerfile` + `railway.json`. New Project → Deploy from GitHub repo →
add the same three variables under **Variables** → Generate Domain.

**Before going public:** put a small monthly spend cap (or a small prepaid-credit
balance with auto-reload off) on your Anthropic account so the demo's cost is
bounded, and set `APP_PASSWORD` so only people you share it with can run queries.

## Notes

- Defaults to `claude-opus-4-8`; set `ANTHROPIC_MODEL=claude-sonnet-4-6` in `.env`
  for a cheaper, faster run.
- News search is optional — set `TAVILY_API_KEY` to enable it; the agent works
  without it and notes the gap.
- This is an educational/portfolio project, not financial advice.

## License

MIT — see [LICENSE](LICENSE).
